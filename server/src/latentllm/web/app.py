"""Experiment API and replayable live-state stream for the React workbench."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from latentllm.experiment.run import Experiment


class RunInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    media: str
    expected: str | None = Field(default=None, max_length=80)


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

    @app.get('/api/runs')
    async def history():
        return [experiment.summary(run) for run in reversed(experiment.history)]

    @app.get('/api/runs/{run_id}')
    async def retrieve(run_id: str):
        if experiment.run and experiment.run['id'] == run_id:
            return experiment.run
        for run in reversed(experiment.history):
            if run['id'] == run_id:
                return run
        raise HTTPException(404, 'Experiment not found')

    @app.post('/api/runs', status_code=202)
    async def launch(request: RunInput):
        try:
            return {'run_id': await experiment.start(request.prompt, request.media, request.expected)}
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

    # In the lab a prebuilt React bundle can be served from the same port as
    # the API, so Node.js and a separate reverse proxy are unnecessary.
    distribution = Path(__file__).resolve().parents[4] / 'web' / 'dist'
    if (distribution / 'index.html').is_file():
        assets = distribution / 'assets'
        if assets.is_dir():
            app.mount('/assets', StaticFiles(directory=assets), name='assets')

        @app.get('/', include_in_schema=False)
        async def dashboard():
            return FileResponse(distribution / 'index.html')

    return app
