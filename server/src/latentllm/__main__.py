"""Run a compute node, optionally serving the experiment UI API on PC A."""
from __future__ import annotations

import argparse
import asyncio
import os

from latentllm.agent.runtime import Runtime
from latentllm.experiment.run import Experiment
from latentllm.transport.peer import Peer, serve
from latentllm.web.app import create_app


async def start(args: argparse.Namespace) -> None:
    runtime = Runtime(args.model, args.device, latent_steps=args.latent_steps, max_tokens=args.max_tokens, judge_tokens=args.judge_tokens)
    peer = Peer(args.node, runtime)
    grpc_server = await serve(peer, args.bind)
    print(f'LatentLLM peer {args.node} bound to {args.bind}', flush=True)
    async def initialize():
        try:
            await asyncio.to_thread(runtime.load)
            print(f'Model ready: {runtime.name} ({runtime.device})', flush=True)
        except Exception as error:
            print(f'Model loading failed: {type(error).__name__}: {error}', flush=True)
    loading = asyncio.create_task(initialize())
    try:
        if args.api:
            import uvicorn
            experiment = Experiment((args.local, args.peer))
            config = uvicorn.Config(create_app(experiment), host=args.api_host, port=args.api_port, log_level='info')
            await uvicorn.Server(config).serve()
        else:
            await grpc_server.wait_for_termination()
    finally:
        loading.cancel()
        await grpc_server.stop(grace=3)


def main() -> None:
    parser = argparse.ArgumentParser(prog='latentllm')
    parser.add_argument('--node', choices=['node-a', 'node-b'], required=True)
    parser.add_argument('--bind', required=True, help='gRPC listener host:port')
    parser.add_argument('--model', default=os.environ.get('LATENTLLM_MODEL'), help='Qwen model name/path; omit to run in transport-only mode')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--latent-steps', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--judge-tokens', type=int, default=768, help='Generation budget for final answer, independent of intermediate agents')
    parser.add_argument('--api', action='store_true', help='Serve FastAPI on the coordinator node')
    parser.add_argument('--local', default='127.0.0.1:50051', help='PC A gRPC address')
    parser.add_argument('--peer', default='127.0.0.1:50052', help='PC B gRPC address')
    parser.add_argument('--api-host', default='127.0.0.1')
    parser.add_argument('--api-port', type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(start(args))


if __name__ == '__main__':
    main()
