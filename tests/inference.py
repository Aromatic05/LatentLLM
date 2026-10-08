"""Real Transformer continuation after a gRPC KV handoff, no external model download."""
from __future__ import annotations

import asyncio
import hashlib
import socket
import unittest

import grpc
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from latentllm.transport.codec import decode_cache, encode_cache
from latentllm.transport.peer import Peer, serve
from latentllm.transport.schema import model as wire, service


class InferenceTests(unittest.TestCase):
    def test_received_kv_continues_with_identical_logits(self):
        config = Qwen3Config(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
        )
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).eval()
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
                            model='Qwen3-test', media=wire.KV_CACHE,
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
