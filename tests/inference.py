"""Real Transformer continuation after a gRPC KV handoff, no external model download."""
from __future__ import annotations

import asyncio
import hashlib
import socket
import unittest

import grpc
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from latentllm.transport.codec import decode_cache, encode_cache
from latentllm.transport.peer import Peer, serve
from latentllm.transport.schema import model as wire, service


class InferenceTests(unittest.TestCase):
    def test_received_kv_continues_with_identical_logits(self):
        for family, config_type, model_type in (
            ('qwen', Qwen3Config, Qwen3ForCausalLM),
            ('llama', LlamaConfig, LlamaForCausalLM),
        ):
            with self.subTest(model=family):
                self.check_continuation(family, config_type, model_type)

    def check_continuation(self, family, config_type, model_type):
        config = config_type(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
        )
        torch.manual_seed(42)
        model = model_type(config).eval()
        with torch.no_grad():
            prefix = model(input_ids=torch.tensor([[1, 7, 8, 12, 21]]), use_cache=True)
        raw, metadata, length = encode_cache(prefix.past_key_values)

        async def transmit():
            peer = Peer('node-b')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                address = f'127.0.0.1:{sock.getsockname()[1]}'
            server = await serve(peer, address)
            try:
                async with grpc.aio.insecure_channel(address) as channel:
                    async def chunks():
                        yield wire.Packet(header=wire.Header(
                            run_id='continuation', source='planner', target='critic',
                            model=f'{family}-test', media=wire.KV_CACHE,
                            size=len(raw), sha256=hashlib.sha256(raw).digest(),
                            tensors=metadata, positions=length,
                        ))
                        for i in range(0, len(raw), 1024):
                            yield wire.Packet(chunk=raw[i:i + 1024])
                    ack = await service.NodeStub(channel).Transfer(chunks())
                self.assertTrue(ack.accepted)
                return peer.incoming[('continuation', 'critic')]
            finally:
                await server.stop(grace=0)

        received_header, received_bytes = asyncio.run(transmit())
        recovered = decode_cache(received_bytes, list(received_header.tensors))
        self.assertEqual(received_header.positions, length)
        with torch.no_grad():
            continuation = torch.tensor([[15]])
            attention = torch.ones((1, length + 1), dtype=torch.long)
            original = model(input_ids=continuation, past_key_values=prefix.past_key_values,
                             attention_mask=attention, use_cache=True).logits
            network = model(input_ids=continuation, past_key_values=recovered,
                            attention_mask=attention, use_cache=True).logits
        self.assertTrue(torch.allclose(original, network, atol=1e-6, rtol=1e-6))

    def test_appended_kv_preserves_real_transformer_continuation(self):
        """Compare full continuation logits against a network-reassembled APPEND."""
        from latentllm.transport.codec import slice_cache
        from latentllm.transport.peer import digest
        for family, config_type, model_type in (
            ('qwen', Qwen3Config, Qwen3ForCausalLM),
            ('llama', LlamaConfig, LlamaForCausalLM),
        ):
            with self.subTest(family=family):
                config = config_type(vocab_size=64, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                     head_dim=16, max_position_embeddings=128)
                torch.manual_seed(55)
                transformer = model_type(config).eval()
                with torch.no_grad():
                    first = transformer(input_ids=torch.tensor([[1, 7, 8]]), use_cache=True)
                    base_bytes, base_meta, start = encode_cache(first.past_key_values)
                    next_step = transformer(input_ids=torch.tensor([[12, 21]]),
                                            attention_mask=torch.ones((1, 5), dtype=torch.long),
                                            past_key_values=first.past_key_values, use_cache=True)
                    target_bytes, target_meta, end = encode_cache(next_step.past_key_values)
                previous = wire.Header(run_id='real-delta', source='planner', target='critic',
                                       model=family, media=wire.KV_CACHE, cache_id='first',
                                       positions=start, tensors=base_meta,
                                       cache_sha256=digest(base_bytes))
                target = wire.Header(run_id='real-delta', source='refiner', target='judger',
                                     model=family, media=wire.KV_CACHE, cache_id='second',
                                     base_cache_id='first', base_sha256=digest(base_bytes),
                                     positions=end, tensors=target_meta, cache_sha256=digest(target_bytes))
                fragment, descriptor = slice_cache(previous, base_bytes, target, target_bytes)
                self.assertLess(len(fragment), len(target_bytes))

                async def transmit():
                    peer = Peer('node-b')
                    with socket.socket() as sock:
                        sock.bind(('127.0.0.1', 0))
                        address = f'127.0.0.1:{sock.getsockname()[1]}'
                    server = await serve(peer, address)
                    try:
                        async with grpc.aio.insecure_channel(address) as channel:
                            stub = service.NodeStub(channel)
                            async def packets(header, data):
                                yield wire.Packet(header=header)
                                for offset in range(0, len(data), 97):
                                    yield wire.Packet(chunk=data[offset:offset + 97])
                            original = wire.Header()
                            original.CopyFrom(previous)
                            original.size, original.sha256 = len(base_bytes), digest(base_bytes)
                            received_first = await stub.Transfer(packets(original, base_bytes))
                            self.assertEqual(received_first.bytes, len(base_bytes))
                            appended = wire.Header()
                            appended.CopyFrom(target)
                            appended.transfer_mode = wire.APPEND
                            appended.start_position = start
                            appended.end_position = end
                            appended.size, appended.sha256 = len(fragment), digest(fragment)
                            del appended.tensors[:]
                            appended.tensors.extend(descriptor)
                            received_second = await stub.Transfer(packets(appended, fragment))
                            self.assertEqual(received_second.bytes, len(fragment))
                        return peer.incoming[('real-delta', 'judger')]
                    finally:
                        await server.stop(0)

                complete, assembled = asyncio.run(transmit())
                self.assertEqual(complete.positions, end)
                self.assertEqual(assembled, target_bytes)
                self.assertEqual(complete.cache_sha256, digest(target_bytes))
                with torch.no_grad():
                    next_token = torch.tensor([[15]])
                    attention = torch.ones((1, end + 1), dtype=torch.long)
                    expected = transformer(input_ids=next_token,
                                           past_key_values=decode_cache(target_bytes, target_meta),
                                           attention_mask=attention).logits
                    actual = transformer(input_ids=next_token,
                                         past_key_values=decode_cache(assembled, list(complete.tensors)),
                                         attention_mask=attention).logits
                self.assertTrue(torch.allclose(expected, actual, rtol=1e-6, atol=1e-6))
