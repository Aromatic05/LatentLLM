# LatentLLM

Real Token / KV-Cache communication between independent LLM processes, with a live experiment dashboard.

The final Judger uses the upstream Qwen3 thinking template by default. You may try `--judge-no-think` on both nodes to shorten final generation, but this experimental switch can change task accuracy (the 0.6B model has produced incorrect answers under that option). The intermediate latent rollout is unchanged.

The inference algorithms are adapted from [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), pinned as a Git submodule. This is a development laboratory, not a secure public multi-tenant service.

## Features

- TextMAS mode: generated text is encoded as Token IDs (uint32) and transferred to the next role.
- LatentMAS mode: real per-layer KV tensors are copied from GPU memory to binary payloads, transferred over gRPC, reconstructed on the receiver, and reused for subsequent inference.
- Planner → Critic → Refiner → Judger across two independently loaded inference processes.
- React dashboard with topology, real agent outputs, real transfer events, staged compute/serialization measurements and same-prompt Token/KV comparisons.
- Completed experiment traces are archived as `var/experiments/run-*.json` (intentionally not committed).

**Scope:** The initial deployment uses two processes on the same host. This verifies serialization, process separation and inference state restoration, but loopback latency must not be reported as physical two-PC network latency. LatentMAS TextMAS and LatentMAS use different inference algorithms, so a direct wall-time comparison is not a measurement of transport alone.

## Setup

Python 3.10–3.14 and CUDA-capable GPUs are supported by the project configuration. Test the exact installed PyTorch/CUDA combination before depending on it. Tested development runtime: Python 3.14, existing CUDA PyTorch 2.14, Transformers 4.57, grpcio 1.84, Qwen3-0.6B. Model weights are downloaded from Hugging Face unless already cached.

```sh
git submodule update --init
uv venv --python 3.12 server/.venv
uv pip install --python server/.venv/bin/python -e ./server
pnpm install --dir web
```

Note: Installing PyTorch into a new environment may download several gigabytes of CUDA dependencies. If your existing system Python has an appropriate GPU-enabled version, `uv venv --system-site-packages` can share that installation during development.

## Run two local processes

In two terminals from the repository root:

```sh
server/.venv/bin/latentllm --node node-a --bind 127.0.0.1:50051 --model Qwen/Qwen3-0.6B --api --local 127.0.0.1:50051 --peer 127.0.0.1:50052
```

```sh
server/.venv/bin/latentllm --node node-b --bind 127.0.0.1:50052 --model Qwen/Qwen3-0.6B
```

Start the UI:

```sh
pnpm --dir web dev
```

Open `http://localhost:5173/`. The UI reports whether two nodes and their models are ready. Select Token or KV-Cache, input a task, and run both modes with the same prompt to compare results. The Python coordinator API is at `http://127.0.0.1:8000/api/status`; Vite proxies `/api` in development.

For two physical PCs, bind the compute gRPC nodes to a reachable IP and provide reachable `--local` / `--peer` addresses. Never expose unauthenticated gRPC inference services to the public internet.

## Timing interpretation

- `restoreMs`: receiving-side bytes → tokens or KV tensors; includes GPU copy for KV.
- `inferenceMs`: prompt assembly + generation / latent rollout, measured after CUDA synchronization.
- `encodeMs`: producer's payload construction, including GPU → CPU KV copy.
- `sendMs`: producer's outbound gRPC handoff call until receiver acknowledgment.
- `receiveMs`: receiver's stream assembly, checksum and in-memory storage.
- `durationMs` for a transfer: coordinator's RPC to the producer, including the outbound handoff. This is **not** pure network time.

## Verification

```sh
PYTHONPATH=server/src server/.venv/bin/python -m unittest discover -s tests -p '*.py' -v
pnpm --dir web build
```

`tests/` checks externally observable Token ID roundtrips, real PyTorch KV tensor roundtrips, and actual gRPC process-facing streaming behavior. Production inference is checked separately by running real Qwen model processes and reviewing actual outputs.
