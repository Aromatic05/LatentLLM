"""Peer-to-peer gRPC transfer, message integrity and explicit step handoff."""
from __future__ import annotations

import asyncio
import hashlib
import socket
import time
import uuid
from typing import Any

import grpc

from .schema import model, service

CHUNK_SIZE = 1024 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024


class Peer(service.NodeServicer):
    def __init__(self, node: str, runner: Any = None):
        self.node = node
        self.runner = runner
        self.incoming: dict[tuple[str, str], tuple[model.Header, bytes]] = {}
        self.outgoing: dict[str, tuple[model.Header, bytes]] = {}
        self.lock = asyncio.Lock()

    async def Health(self, request: model.HealthRequest, context: grpc.aio.ServicerContext) -> model.HealthReply:
        return model.HealthReply(
            id=self.node, host=socket.gethostname(),
            model=getattr(self.runner, 'name', '') or '',
            device=getattr(self.runner, 'device', '') or '',
            ready=bool(getattr(self.runner, 'ready', False)),
        )

    async def Transfer(self, stream: Any, context: grpc.aio.ServicerContext) -> model.Receipt:
        started = time.perf_counter_ns()
        header = None
        buffer = bytearray()
        async for packet in stream:
            if packet.HasField('header'):
                if header is not None or buffer:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Only one initial header is allowed')
                header = packet.header
                if not header.run_id or not header.target or header.target not in ('planner', 'critic', 'refiner', 'judger'):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Invalid recipient or run identifier')
                if header.size > MAX_PAYLOAD:
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'Payload exceeds size limit')
            elif packet.HasField('chunk'):
                if header is None:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Data received before header')
                if len(buffer) + len(packet.chunk) > header.size:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Payload exceeds declared size')
                buffer.extend(packet.chunk)
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Empty transfer packet')
        if header is None or len(buffer) != header.size or hashlib.sha256(buffer).digest() != header.sha256:
            await context.abort(grpc.StatusCode.DATA_LOSS, 'Transfer integrity check failed')
        if self.runner and header.model != self.runner.name:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Sender/receiver model mismatch')
        async with self.lock:
            self.incoming[(header.run_id, header.target)] = (header, bytes(buffer))
        duration = (time.perf_counter_ns() - started) / 1_000_000
        return model.Receipt(accepted=True, bytes=len(buffer), duration_ms=duration)

    async def Step(self, request: model.StepRequest, context: grpc.aio.ServicerContext) -> model.StepReply:
        if not self.runner or not self.runner.ready:
            return model.StepReply(error='Inference model is not loaded')
        incoming = self.incoming.pop((request.run_id, request.role), None)
        if request.role != 'planner' and incoming is None:
            return model.StepReply(error='Missing input from previous agent')
        try:
            result, header, data = await asyncio.to_thread(self.runner.step, request, incoming)
            self.outgoing[request.run_id] = (header, data)
            return model.StepReply(**result, success=True)
        except Exception as exc:
            return model.StepReply(error=f'{type(exc).__name__}: {exc}')

    async def Forward(self, request: model.ForwardRequest, context: grpc.aio.ServicerContext) -> model.ForwardReply:
        if request.run_id not in self.outgoing:
            await context.abort(grpc.StatusCode.NOT_FOUND, 'Run has no output to forward')
        source_header, payload = self.outgoing[request.run_id]
        header = model.Header()
        header.CopyFrom(source_header)
        header.id = str(uuid.uuid4())
        header.source = request.source
        header.target = request.target
        header.size = len(payload)
        header.sha256 = hashlib.sha256(payload).digest()
        started = time.perf_counter_ns()
        async with grpc.aio.insecure_channel(request.address, options=[('grpc.max_receive_message_length', CHUNK_SIZE * 2)]) as channel:
            stub = service.NodeStub(channel)
            async def packets():
                yield model.Packet(header=header)
                for offset in range(0, len(payload), CHUNK_SIZE):
                    yield model.Packet(chunk=payload[offset:offset + CHUNK_SIZE])
            receipt = await stub.Transfer(packets(), timeout=180)
        send_ms = (time.perf_counter_ns() - started) / 1e6
        if not receipt.accepted:
            await context.abort(grpc.StatusCode.DATA_LOSS, receipt.reason or 'Peer rejected transfer')
        return model.ForwardReply(receipt=receipt, media=header.media, size=len(payload), positions=header.positions, tensors=len(header.tensors), send_ms=send_ms)


async def serve(peer: Peer, address: str) -> grpc.aio.Server:
    server = grpc.aio.server(options=[('grpc.max_receive_message_length', CHUNK_SIZE * 2)])
    service.add_NodeServicer_to_server(peer, server)
    server.add_insecure_port(address)
    await server.start()
    return server
