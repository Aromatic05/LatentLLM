"""Behavior-based tests for actual binary payloads and loopback gRPC streaming."""
import asyncio
import hashlib
import socket
import unittest

import grpc
import torch

from latentllm.transport.client import Client
from latentllm.transport.codec import decode_cache, decode_tokens, encode_cache, encode_tokens
from latentllm.transport.peer import Peer, serve
from latentllm.transport.schema import model


class TransportTests(unittest.TestCase):
    def test_token_roundtrip(self):
        original = [1, 23, 65538, 50112]
        self.assertEqual(original, decode_tokens(encode_tokens(original)))

    def test_cache_roundtrip(self):
        original = tuple((torch.randn(1, 2, 3, 8, dtype=torch.float16),
                          torch.randn(1, 2, 3, 8, dtype=torch.float16)) for _ in range(3))
        raw, tensors, positions = encode_cache(original)
        recovered = decode_cache(raw, tensors)
        self.assertEqual(positions, 3)
        for before, after in zip(original, recovered):
            for source, target in zip(before, after):
                self.assertTrue(torch.equal(source, target))

    def test_corrupt_network_payload_is_rejected(self):
        async def execute():
            import socket
            from latentllm.transport.schema import service
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                address = f'127.0.0.1:{sock.getsockname()[1]}'
            server = await serve(Peer('node-b'), address)
            try:
                async with grpc.aio.insecure_channel(address) as channel:
                    stream = service.NodeStub(channel).Transfer
                    async def payload():
                        yield model.Packet(header=model.Header(
                            run_id='corrupted-run', source='planner', target='critic',
                            size=4, sha256=hashlib.sha256(b'good').digest()))
                        yield model.Packet(chunk=b'bad!')
                    with self.assertRaises(grpc.aio.AioRpcError) as caught:
                        await stream(payload())
                    self.assertEqual(caught.exception.code(), grpc.StatusCode.DATA_LOSS)
            finally:
                await server.stop(grace=0)
        asyncio.run(execute())

    def test_loopback_streaming(self):
        async def execute():
            a, b = Peer('node-a'), Peer('node-b')
            def port():
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    return sock.getsockname()[1]
            address_a = f'127.0.0.1:{port()}'
            address_b = f'127.0.0.1:{port()}'
            sa = await serve(a, address_a)
            sb = await serve(b, address_b)
            try:
                raw = encode_tokens([128, 4096, 65537])
                header = model.Header(run_id='run-test', model='test-model', media=model.TOKEN, tokens=3,
                                      size=len(raw), sha256=hashlib.sha256(raw).digest())
                a.outgoing['run-test'] = (header, raw)
                forward = await Client(address_a).forward('run-test', 'planner', 'critic', address_b)
                self.assertTrue(forward.receipt.accepted)
                self.assertEqual(decode_tokens(b.incoming[('run-test', 'critic')][1]), [128, 4096, 65537])
                # Real tensor bytes over the same RPC, separately identified as KV payload.
                state = ((torch.ones(1, 2, 8, 4, dtype=torch.bfloat16), torch.zeros(1, 2, 8, 4, dtype=torch.bfloat16)),)
                data, tensors, positions = encode_cache(state)
                header = model.Header(run_id='kv-test', model='test-model', media=model.KV_CACHE,
                                      tensors=tensors, positions=positions, size=len(data), sha256=hashlib.sha256(data).digest())
                a.outgoing['kv-test'] = (header, data)
                forward = await Client(address_a).forward('kv-test', 'planner', 'critic', address_b)
                self.assertTrue(forward.receipt.accepted)
                received = b.incoming[('kv-test', 'critic')]
                restored = decode_cache(received[1], list(received[0].tensors))
                self.assertTrue(torch.equal(restored[0][0], state[0][0]))
            finally:
                await sa.stop(grace=0)
                await sb.stop(grace=0)
        asyncio.run(execute())
