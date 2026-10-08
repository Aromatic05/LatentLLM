"""Lightweight client for remote execution and explicit peer handoffs."""
from __future__ import annotations

import grpc

from .schema import model, service


class Client:
    def __init__(self, address: str):
        self.address = address

    async def health(self) -> model.HealthReply:
        async with grpc.aio.insecure_channel(self.address) as channel:
            return await service.NodeStub(channel).Health(model.HealthRequest(), timeout=5)

    async def step(self, run_id: str, role: str, question: str, media: int) -> model.StepReply:
        async with grpc.aio.insecure_channel(self.address) as channel:
            return await service.NodeStub(channel).Step(model.StepRequest(run_id=run_id, role=role, question=question, media=media), timeout=240)

    async def forward(self, run_id: str, source: str, target: str, address: str) -> model.ForwardReply:
        async with grpc.aio.insecure_channel(self.address) as channel:
            return await service.NodeStub(channel).Forward(model.ForwardRequest(run_id=run_id, source=source, target=target, address=address), timeout=180)
