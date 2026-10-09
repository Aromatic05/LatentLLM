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
