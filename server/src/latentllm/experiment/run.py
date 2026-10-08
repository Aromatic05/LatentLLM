"""Four-role LatentMAS experiment, run state, and truthful live telemetry."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from latentllm.transport.client import Client
from latentllm.transport.schema import model

ROLES = ('planner', 'critic', 'refiner', 'judger')


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Experiment:
    def __init__(self, addresses: tuple[str, str]):
        self.addresses = addresses
        self.nodes: list[dict[str, Any]] = [
            {'id': 'node-a', 'host': addresses[0], 'online': False, 'model': None, 'device': None},
            {'id': 'node-b', 'host': addresses[1], 'online': False, 'model': None, 'device': None},
        ]
        self.model_ready = False
        self.run: dict[str, Any] | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.counter = 0
        self.lock = asyncio.Lock()

    def snapshot(self) -> dict:
        return {'type': 'snapshot', 'status': {'nodes': self.nodes, 'run': self.run, 'modelReady': self.model_ready}}

    def publish(self) -> None:
        for queue in tuple(self.subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(self.snapshot())

    async def refresh(self) -> None:
        async def probe(index: int) -> dict:
            address = self.addresses[index]
            try:
                response = await Client(address).health()
                return {'id': f'node-{"a" if index == 0 else "b"}', 'host': response.host,
                        'online': True, 'model': response.model or None, 'device': response.device or None,
                        '_ready': response.ready}
            except Exception:
                return {'id': f'node-{"a" if index == 0 else "b"}', 'host': address,
                        'online': False, 'model': None, 'device': None, '_ready': False}
        found = await asyncio.gather(probe(0), probe(1))
        ready = all(node.pop('_ready') for node in found)
        if found != self.nodes or ready != self.model_ready:
            self.nodes = found
            self.model_ready = ready
            self.publish()

    async def observe(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(3)

    def emit(self, kind: str, message: str, *, agent: str | None = None, duration: float | None = None) -> None:
        if not self.run:
            return
        self.counter += 1
        event: dict[str, Any] = {'id': self.counter, 'time': utcnow(), 'kind': kind, 'message': message}
        if agent:
            event['agent'] = agent
        if duration is not None:
            event['durationMs'] = duration
        self.run['events'].append(event)
        self.publish()

    async def start(self, prompt: str, media: str) -> str:
        async with self.lock:
            if self.run and self.run['status'] == 'running':
                raise ValueError('Another experiment is running')
            if not self.model_ready:
                raise RuntimeError('Both inference nodes must be connected and model-ready')
            if media not in ('token', 'kv'):
                raise ValueError('Unsupported media')
            run_id = str(uuid.uuid4())
            self.run = {
                'id': run_id, 'prompt': prompt, 'media': media, 'status': 'running',
                'startedAt': utcnow(), 'agents': [
                    {'id': role, 'role': role.title(), 'node': f'node-{"a" if i % 2 == 0 else "b"}', 'status': 'pending'}
                    for i, role in enumerate(ROLES)],
                'transfers': [], 'events': [],
            }
            self.counter = 0
            self.emit('run.started', '实验已开始，等待执行第一个 Agent')
            asyncio.create_task(self.execute(run_id))
            return run_id

    async def execute(self, run_id: str) -> None:
        run = self.run
        if not run or run['id'] != run_id:
            return
        mode = model.TOKEN if run['media'] == 'token' else model.KV_CACHE
        try:
            for index, role in enumerate(ROLES):
                agent = run['agents'][index]
                address = self.addresses[index % 2]
                agent['status'] = 'running'
                self.emit('agent.started', f'{role.title()} 已开始推理', agent=role)
                result = await Client(address).step(run_id, role, run['prompt'], mode)
                if not result.success:
                    raise RuntimeError(f'{role.title()}: {result.error}')
                agent['status'] = 'completed'
                agent['durationMs'] = result.duration_ms
                agent['outputTokens'] = result.tokens
                agent['output'] = result.output
                if run['media'] == 'kv' and role != 'judger':
                    agent['latentSteps'] = max(0, result.positions)
                message = result.output[:900] if result.output else f'{role.title()} 完成潜在推理，缓存序列长度 {result.positions}，无可读中间文本'
                self.emit('agent.completed', message, agent=role, duration=result.duration_ms)
                if index == len(ROLES) - 1:
                    run['answer'] = result.output
                    continue
                target = ROLES[index + 1]
                transfer = {'id': str(uuid.uuid4()), 'source': role, 'target': target,
                            'media': run['media'], 'bytes': 0, 'durationMs': None, 'status': 'sending'}
                run['transfers'].append(transfer)
                self.emit('transfer.started', f'{role.title()} → {target.title()}：开始 {run["media"]} 传输', agent=role)
                start = time.perf_counter_ns()
                receipt = await Client(address).forward(run_id, role, target, self.addresses[(index + 1) % 2])
                elapsed = (time.perf_counter_ns() - start) / 1_000_000
                if not receipt.receipt.accepted:
                    raise RuntimeError(f'Transfer to {target} was rejected')
                transfer.update(bytes=receipt.size, durationMs=elapsed, status='completed', tensors=receipt.tensors, sequence=receipt.positions)
                self.emit('transfer.completed', f'{role.title()} → {target.title()}：完成 {receipt.size} B，{elapsed:.1f} ms', agent=role, duration=elapsed)
            run['status'] = 'completed'
            run['finishedAt'] = utcnow()
            self.emit('run.completed', '全部 Agent 已完成协作任务')
        except Exception as error:
            run['status'] = 'failed'
            run['finishedAt'] = utcnow()
            run['error'] = f'{type(error).__name__}: {error}'
            self.emit('run.failed', run['error'])
            for agent in run['agents']:
                if agent['status'] == 'running':
                    agent['status'] = 'failed'
            self.publish()
