"""Experiment API and replayable live-state stream for the React workbench."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from latentllm.experiment.run import Experiment


class RunInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    media: str


def create_app(experiment: Experiment) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watch = asyncio.create_task(experiment.observe())
        try:
            yield
        finally:
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)

    app = FastAPI(title='LatentLLM', docs_url='/api/docs', lifespan=lifespan)

    @app.get('/api/status')
    async def status():
        return experiment.snapshot()['status']

    @app.post('/api/runs', status_code=202)
    async def launch(request: RunInput):
        try:
            return {'run_id': await experiment.start(request.prompt, request.media)}
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    @app.websocket('/api/events')
    async def events(socket: WebSocket):
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=12)
        experiment.subscribers.add(queue)
        try:
            await socket.send_json(experiment.snapshot())
            while True:
                await socket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            experiment.subscribers.discard(queue)

    return app
