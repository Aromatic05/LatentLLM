"""Real, independently checkable bytes for per-head KV sequence append."""
from __future__ import annotations

import hashlib
import unittest

import torch

from latentllm.transport.codec import decode_cache, encode_cache, merge_cache, slice_cache
from latentllm.transport.schema import model


class DeltaTests(unittest.TestCase):
    def make_case(self, dtype: torch.dtype, *, batch: int = 2, heads: int = 3):
        old = []
        new = []
        for _ in range(3):
            keys = torch.randn(batch, heads, 17, 16).to(dtype)
            values = torch.randn(batch, heads, 17, 16).to(dtype)
            old.append((keys, values))
            new.append((torch.cat((keys, torch.randn(batch, heads, 7, 16).to(dtype)), dim=2),
                        torch.cat((values, torch.randn(batch, heads, 7, 16).to(dtype)), dim=2)))
        old_raw, old_meta, old_count = encode_cache(old)
        new_raw, new_meta, new_count = encode_cache(new)
        prev = model.Header(media=model.KV_CACHE, positions=old_count, tensors=old_meta)
        current = model.Header(media=model.KV_CACHE, positions=new_count, tensors=new_meta,
                               start_position=old_count, end_position=new_count)
        return prev, old_raw, current, new_raw, old, new

    def test_reconstructs_exact_multibatch_multilayer_tensor(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=str(dtype)):
                prev, before, current, after, _, tensors = self.make_case(dtype)
                fragment, descriptors = slice_cache(prev, before, current, after)
                self.assertEqual(len(fragment) * 24, len(after) * 7)
                header = model.Header(positions=current.positions, start_position=prev.positions,
                                      end_position=current.positions, tensors=descriptors)
                restored, full_meta = merge_cache(prev, before, header, fragment)
                self.assertEqual(hashlib.sha256(restored).digest(), hashlib.sha256(after).digest())
                for original, recovered in zip(tensors, decode_cache(restored, full_meta)):
                    for a, b in zip(original, recovered):
                        self.assertTrue(torch.equal(a, b))

    def test_changed_prefix_is_not_an_append(self):
        prev, before, current, after, _, _ = self.make_case(torch.float16)
        changed = bytearray(after)
        changed[1] ^= 0x01
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            slice_cache(prev, before, current, changed)

    def test_missing_or_invalid_positions_are_not_appends(self):
        prev, before, current, after, _, _ = self.make_case(torch.bfloat16)
        current.positions = prev.positions
        with self.assertRaises(ValueError):
            slice_cache(prev, before, current, after)
        current.positions += 7
        fragment, descriptors = slice_cache(prev, before, current, after)
        broken = model.Header(positions=current.positions, start_position=99,
                              end_position=current.positions, tensors=descriptors)
        with self.assertRaises(ValueError):
            merge_cache(prev, before, broken, fragment)


class NetworkTests(unittest.TestCase):
    """Exercise FULL -> APPEND -> APPEND via two actual gRPC server sockets."""

    class Runner:
        ready = True
        name = 'test-model'
        device = 'cpu'
        large = False

        def step(self, request, incoming):
            sizes = {'planner': 8, 'critic': 13, 'refiner': 21, 'judger': 21}
            count = ({'planner': 1024, 'critic': 1064, 'refiner': 1112, 'judger': 1112}
                     if self.large else sizes)[request.role]
            if self.large:
                source = torch.arange(count * 16, dtype=torch.float32).reshape(1, 1, count, 16).repeat(2, 8, 1, 1)
                state = ((source / 512, source / 512 + 1), (source / 512 + 2, source / 512 + 3))
            else:
                state = None
            def patterned(offset):
                return torch.tensor([head * 1024 + point * 4 + feature + offset
                                     for head in range(2) for point in range(count) for feature in range(4)],
                                    dtype=torch.float32).reshape(1, 2, count, 4)
            if state is None:
                state = ((patterned(0), patterned(2000)), (patterned(4000), patterned(6000)))
            if incoming:
                previous, data = incoming
                cache = decode_cache(data, list(previous.tensors))
                for original, restored in zip(state, cache):
                    for target, prefix in zip(original, restored):
                        if not torch.equal(target[:, :, :previous.positions, :], prefix):
                            raise RuntimeError('Receiver reconstructed the wrong KV prefix')
            elif request.role != 'planner':
                raise RuntimeError('Missing incoming KV cache')
            data, descriptors, length = encode_cache(state)
            header = model.Header(run_id=request.run_id, model=self.name, media=model.KV_CACHE,
                                  positions=length, tensors=descriptors)
            result = dict(output='15' if request.role == 'judger' else '', tokens=1,
                          positions=length, duration_ms=0.1)
            return result, header, data

    def test_three_handoffs_use_full_then_two_appends(self):
        self._scenario(expect_delta=True)

    def test_missing_receiver_base_falls_back_to_full(self):
        self._scenario(expect_delta=True, evict_before_critic=True)

    def test_delta_can_be_disabled_for_a_full_baseline(self):
        self._scenario(expect_delta=False)

    def test_megabyte_chunk_boundaries_preserve_delta(self):
        self._scenario(expect_delta=True, large=True)

    def test_modified_prefix_requires_full_snapshot(self):
        self._scenario(expect_delta=True, change_prefix=True)

    def _scenario(self, *, expect_delta: bool, evict_before_critic: bool = False,
                  large: bool = False, change_prefix: bool = False):
        import asyncio
        import socket
        from latentllm.transport.client import Client
        from latentllm.transport.peer import Peer, serve

        def port():
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                return sock.getsockname()[1]

        async def execute():
            first_runner, second_runner = self.Runner(), self.Runner()
            first_runner.large = second_runner.large = large
            first, second = Peer('node-a', first_runner), Peer('node-b', second_runner)
            address_a, address_b = f'127.0.0.1:{port()}', f'127.0.0.1:{port()}'
            sa, sb = await serve(first, address_a), await serve(second, address_b)
            try:
                ca, cb = Client(address_a), Client(address_b)
                out = await ca.step('delta-run', 'planner', '8+7?', model.KV_CACHE)
                self.assertTrue(out.success, out.error)
                first_copy = await ca.forward('delta-run', 'planner', 'critic', address_b)
                self.assertEqual(first_copy.transfer_mode, model.FULL)
                self.assertEqual(first_copy.size, first_copy.full_size)
                if large:
                    self.assertGreater(first_copy.size, 1024 * 1024)

                out = await cb.step('delta-run', 'critic', '8+7?', model.KV_CACHE)
                self.assertTrue(out.success, out.error)
                if evict_before_critic:
                    first.cache.clear()
                if change_prefix:
                    meta, original = second.outgoing['delta-run']
                    changed = bytearray(original)
                    changed[0] ^= 0x01
                    second.outgoing['delta-run'] = (meta, bytes(changed))
                second_copy = await cb.forward('delta-run', 'critic', 'refiner', address_a,
                                               disable_delta=not expect_delta)
                self.assertEqual(second_copy.transfer_mode,
                                 model.FULL if evict_before_critic or change_prefix or not expect_delta else model.APPEND)
                self.assertEqual(second_copy.fallback, evict_before_critic or change_prefix)
                if second_copy.transfer_mode == model.APPEND:
                    self.assertLess(second_copy.size, second_copy.full_size)
                if change_prefix:
                    self.assertEqual(first.incoming[('delta-run', 'refiner')][1], bytes(changed))
                    return

                out = await ca.step('delta-run', 'refiner', '8+7?', model.KV_CACHE)
                self.assertTrue(out.success, out.error)
                third_copy = await ca.forward('delta-run', 'refiner', 'judger', address_b,
                                              disable_delta=not expect_delta)
                self.assertEqual(third_copy.transfer_mode, model.APPEND if expect_delta else model.FULL)
                self.assertLess(third_copy.size, third_copy.full_size) if expect_delta else None
                out = await cb.step('delta-run', 'judger', '8+7?', model.KV_CACHE)
                self.assertTrue(out.success, out.error)
                self.assertEqual(out.output, '15')
            finally:
                await sa.stop(0)
                await sb.stop(0)
        asyncio.run(execute())
