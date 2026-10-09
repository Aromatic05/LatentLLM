"""Peer-to-peer gRPC transport with versioned FULL/APPEND KV handoffs."""
from __future__ import annotations

import asyncio
import hashlib
import socket
import time
import uuid
from typing import Any

import grpc

from .codec import merge_cache, slice_cache
from .schema import model, service

CHUNK_SIZE = 1024 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024
MAX_CACHE_VERSIONS = 4


def digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


class Peer(service.NodeServicer):
    def __init__(self, node: str, runner: Any = None):
        self.node = node
        self.runner = runner
        self.incoming: dict[tuple[str, str], tuple[model.Header, bytes]] = {}
        self.outgoing: dict[str, tuple[model.Header, bytes]] = {}
        # Only retain bounded, verified full snapshots; APPEND packets are never stored as bases.
        self.cache: dict[str, tuple[model.Header, bytes]] = {}
        self.run_id: str | None = None
        self.lock = asyncio.Lock()

    def _activate(self, run_id: str) -> None:
        if self.run_id != run_id:
            self.incoming.clear()
            self.outgoing.clear()
            self.cache.clear()
            self.run_id = run_id

    def _remember(self, header: model.Header, payload: bytes) -> None:
        if header.media != model.KV_CACHE or not header.cache_id:
            return
        snapshot = model.Header()
        snapshot.CopyFrom(header)
        self.cache[header.cache_id] = (snapshot, payload)
        while len(self.cache) > MAX_CACHE_VERSIONS:
            del self.cache[next(iter(self.cache))]

    async def Health(self, request: model.HealthRequest, context: grpc.aio.ServicerContext) -> model.HealthReply:
        return model.HealthReply(
            id=self.node, host=socket.gethostname(),
            model=getattr(self.runner, 'name', '') or '',
            device=getattr(self.runner, 'device', '') or '',
            ready=bool(getattr(self.runner, 'ready', False)),
        )

    async def Transfer(self, stream: Any, context: grpc.aio.ServicerContext) -> model.Receipt:
        started = time.perf_counter_ns()
        header: model.Header | None = None
        buffer = bytearray()
        async for packet in stream:
            if packet.HasField('header'):
                if header is not None:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Only one initial header is allowed')
                header = packet.header
                if not header.run_id or header.target not in ('planner', 'critic', 'refiner', 'judger'):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Invalid recipient or run identifier')
                if header.size > MAX_PAYLOAD:
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'Payload exceeds size limit')
                if header.transfer_mode == model.APPEND:
                    if header.media != model.KV_CACHE or not header.base_cache_id or not header.cache_id:
                        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'APPEND requires versioned KV state')
                    # Reject before streaming chunks when receiver is missing the advertised base.
                    base = self.cache.get(header.base_cache_id)
                    if (base is None or self.run_id != header.run_id or
                            base[0].cache_sha256 != header.base_sha256 or
                            base[0].model != header.model or
                            base[0].positions != header.start_position):
                        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'KV base is not available')
                elif header.transfer_mode != model.FULL:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Unknown transfer mode')
            elif packet.HasField('chunk'):
                if header is None:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Data received before header')
                if len(buffer) + len(packet.chunk) > header.size:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Payload exceeds declared size')
                buffer.extend(packet.chunk)
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Empty transfer packet')
        if header is None or len(buffer) != header.size or digest(buffer) != header.sha256:
            await context.abort(grpc.StatusCode.DATA_LOSS, 'Transfer integrity check failed')
        if self.runner and header.model != self.runner.name:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Sender/receiver model mismatch')

        received = bytes(buffer)
        if header.transfer_mode == model.APPEND:
            async with self.lock:
                base = self.cache.get(header.base_cache_id)
                if base is None or self.run_id != header.run_id or base[0].cache_sha256 != header.base_sha256:
                    await context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'KV base was evicted')
                try:
                    received, descriptors = merge_cache(base[0], base[1], header, received)
                except ValueError as error:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            if len(received) > MAX_PAYLOAD:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'Reconstructed KV exceeds size limit')
            if digest(received) != header.cache_sha256:
                await context.abort(grpc.StatusCode.DATA_LOSS, 'Reconstructed KV checksum mismatch')
            # Normalize accepted input: Agent runtime always receives complete KV metadata and bytes.
            complete = model.Header()
            complete.CopyFrom(header)
            complete.transfer_mode = model.FULL
            complete.start_position = 0
            complete.end_position = complete.positions
            del complete.tensors[:]
            complete.tensors.extend(descriptors)
            complete.size = len(received)
            complete.sha256 = complete.cache_sha256
            header = complete
        elif header.media == model.KV_CACHE:
            if header.cache_sha256 and digest(received) != header.cache_sha256:
                await context.abort(grpc.StatusCode.DATA_LOSS, 'Complete KV checksum mismatch')
            if not header.cache_id:
                header.cache_id = str(uuid.uuid4())
            header.cache_sha256 = digest(received)
            header.end_position = header.positions

        async with self.lock:
            self._activate(header.run_id)
            self.incoming[(header.run_id, header.target)] = (header, received)
            self._remember(header, received)
        duration = (time.perf_counter_ns() - started) / 1_000_000
        return model.Receipt(accepted=True, bytes=len(buffer), duration_ms=duration)

    async def Step(self, request: model.StepRequest, context: grpc.aio.ServicerContext) -> model.StepReply:
        if not self.runner or not self.runner.ready:
            return model.StepReply(error='Inference model is not loaded')
        async with self.lock:
            if request.role == 'planner':
                self._activate(request.run_id)
            incoming = self.incoming.pop((request.run_id, request.role), None)
        if request.role != 'planner' and incoming is None:
            return model.StepReply(error='Missing input from previous agent')
        try:
            result, header, data = await asyncio.to_thread(self.runner.step, request, incoming)
            if request.role != 'judger':
                if request.media == model.KV_CACHE:
                    header.cache_id = str(uuid.uuid4())
                    header.cache_sha256 = digest(data)
                    header.end_position = header.positions
                    if incoming and incoming[0].cache_id:
                        header.base_cache_id = incoming[0].cache_id
                        header.base_sha256 = incoming[0].cache_sha256
                async with self.lock:
                    self.outgoing[request.run_id] = (header, data)
                    self._remember(header, data)
            return model.StepReply(**result, success=True)
        except Exception as exc:
            return model.StepReply(error=f'{type(exc).__name__}: {exc}')

    async def Forward(self, request: model.ForwardRequest, context: grpc.aio.ServicerContext) -> model.ForwardReply:
        async with self.lock:
            current = self.outgoing.get(request.run_id)
            if current is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, 'Run has no output to forward')
            source_header, payload = current
            base = self.cache.get(source_header.base_cache_id)

        full = model.Header()
        full.CopyFrom(source_header)
        full.id = str(uuid.uuid4())
        full.source = request.source
        full.target = request.target
        full.transfer_mode = model.FULL
        full.start_position = 0
        full.end_position = source_header.positions
        full.size = len(payload)
        full.sha256 = digest(payload)
        if source_header.media == model.KV_CACHE:
            full.cache_sha256 = full.sha256
        header, wire_payload = full, payload
        fallback = False

        if (not request.disable_delta and source_header.media == model.KV_CACHE and
                base and base[0].cache_sha256 == source_header.base_sha256):
            try:
                fragment, descriptors = slice_cache(base[0], base[1], full, payload)
                if len(fragment) < len(payload):
                    header = model.Header()
                    header.CopyFrom(full)
                    header.transfer_mode = model.APPEND
                    header.start_position = base[0].positions
                    header.tensors.clear()
                    header.tensors.extend(descriptors)
                    header.size = len(fragment)
                    header.sha256 = digest(fragment)
                    wire_payload = fragment
            except ValueError:
                # Changed existing sequence positions or incompatible geometry: send a snapshot.
                fallback = True

        started = time.perf_counter_ns()
        async with grpc.aio.insecure_channel(request.address, options=[('grpc.max_receive_message_length', CHUNK_SIZE * 2)]) as channel:
            stub = service.NodeStub(channel)

            async def send(metadata: model.Header, data: bytes) -> model.Receipt:
                async def packets():
                    yield model.Packet(header=metadata)
                    for offset in range(0, len(data), CHUNK_SIZE):
                        yield model.Packet(chunk=data[offset:offset + CHUNK_SIZE])
                return await stub.Transfer(packets(), timeout=180)

            try:
                receipt = await send(header, wire_payload)
            except grpc.aio.AioRpcError as error:
                if header.transfer_mode != model.APPEND or error.code() != grpc.StatusCode.FAILED_PRECONDITION:
                    raise
                # Receiver has lost the base: transparently send a complete copy.
                header, wire_payload = full, payload
                fallback = True
                receipt = await send(header, wire_payload)
        send_ms = (time.perf_counter_ns() - started) / 1e6
        if not receipt.accepted:
            await context.abort(grpc.StatusCode.DATA_LOSS, receipt.reason or 'Peer rejected transfer')
        async with self.lock:
            if self.outgoing.get(request.run_id) is current:
                self.outgoing.pop(request.run_id, None)
        return model.ForwardReply(receipt=receipt, media=header.media, size=len(wire_payload),
                                  positions=full.positions, tensors=len(full.tensors), send_ms=send_ms,
                                  transfer_mode=header.transfer_mode, full_size=len(payload), fallback=fallback)


async def serve(peer: Peer, address: str) -> grpc.aio.Server:
    server = grpc.aio.server(options=[('grpc.max_receive_message_length', CHUNK_SIZE * 2)])
    service.add_NodeServicer_to_server(peer, server)
    server.add_insecure_port(address)
    await server.start()
    return server
