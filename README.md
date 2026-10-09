# LatentLLM

Real Token / KV-Cache communication between independent LLM processes, with a live experiment dashboard.

Role prompts are model-independent and are rendered through each model’s own chat template. The original LatentMAS repository hardcodes Qwen prompts; this project keeps its inference algorithm but replaces those prompts so the same Token/KV flow works with Qwen and Llama. This changes the prompt benchmark baseline and must be documented when reporting research results. `--judge-no-think` is an experimental Qwen3-specific generation option, not a generic model requirement.

The latent rollout defaults to 10 steps per role, matching the original LatentMAS implementation. The `--latent-steps` option permits controlled experiments, but comparisons must record this setting.

The inference algorithms are adapted from [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), pinned as a Git submodule. This is a development laboratory, not a secure public multi-tenant service.

## Features

- TextMAS mode: generated text is encoded as Token IDs (uint32) and transferred to the next role.
- LatentMAS mode: real per-layer KV tensors are copied from GPU memory to binary payloads, transferred over gRPC, reconstructed on the receiver, and reused for subsequent inference.
- Planner → Critic → Refiner → Judger across two independently loaded inference processes.
- React dashboard with topology, real agent outputs, real transfer events, staged compute/serialization measurements and same-prompt Token/KV comparisons.
- Completed experiment traces are archived as `var/experiments/run-*.json` (intentionally not committed). After restart, the dashboard restores the latest real run. The **实验历史** selector can load earlier Token and KV traces without re-running the models.

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

Open `http://localhost:5173/`. The UI reports whether two nodes and their models are ready. Select Token or KV-Cache, input a task, and run both modes with the same prompt to compare results. Enter an optional expected answer for conservative final-output grading. The evaluator accepts visible `\boxed{...}` answers and unambiguous final-answer statements, not arbitrary numeric values in working steps. It distinguishes **MATCHED**, **MISMATCH** and **UNPARSED** (such as an unterminated `<think>` block). A completed workflow is not automatically a correct answer. The Python coordinator API is at `http://127.0.0.1:8000/api/status`; Vite proxies `/api` in development.

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

`tests/` checks externally observable Token ID roundtrips, real PyTorch KV tensor roundtrips, actual gRPC process-facing streaming behavior, archive grading, and continuation logits of small real Qwen3 and Llama Transformers after network KV handoffs. Production inference is checked separately by running real Qwen model processes and reviewing actual outputs.

## Shared lab deployment (SYM)

The Git repository does **not** contain model weights or environments.
The lab already provides `LLM Model/Llama-3.2-1B-Instruct` and
`LLM Model/Qwen2.5-3B-Instruct`. Use an **absolute path to an existing model**
for `--model` on both nodes. Both processes must use the identical weights.

Build the React site on the development PC (`pnpm --dir web build`) and copy
only `web/dist` (approximately 1–2 MiB) with the Git sources. If `web/dist`
exists, FastAPI automatically serves the dashboard at its own API port. No
Node.js runtime is required on the experiment machine. For a shared GPU
machine, select unused ports and a GPU that will not interfere with existing
training jobs; default port 8000 is likely occupied.

During initial network or library checks the Python services can run without
`--model`. This is transport-only mode: the dashboard remains available,
but **modelReady is false** and the UI must not claim LLM experiments ran.

## Experimental KV APPEND transport

The two-node handoff protocol supports two distinct **wire** modes without
changing the 1 MiB Protobuf `Packet.chunk` streaming mechanism:

- `FULL`: transfer the complete contiguous KV tensor snapshot.
- `APPEND`: when the receiver already holds a verified base cache version,
  send only newly appended KV **sequence positions**, separately for each
  batch/head/layer K and V tensor. Cache metadata includes cache IDs,
  base SHA-256, sequence interval, tensor shapes/dtypes and both full and
  transmitted checksums.

The first Planner → Critic transfer is FULL. With alternating nodes, Critic →
Refiner and Refiner → Judger can reuse the Planner/Critic cache already stored
on the respective receiving node. The receiver assembles the complete byte
layout, verifies its SHA-256 and passes a normal `DynamicCache` to the next
agent. The receiver rejects missing/mismatching base versions and the sender
**automatically falls back to FULL**. An unexpected mutation to an existing
KV prefix also makes the sender choose FULL. Per-node registries retain up to
four immutable full snapshots and are cleared on a new run.

The UI defaults to APPEND for KV runs; uncheck **KV-Cache 增量传输** to run a
FULL-only control with the same prompt. The REST request body also accepts
`"delta": false` under `POST /api/runs`. Transfer details and archived runs
include `transferMode`, `fullBytes`, `bytes`, `savedBytes` and `fallback`.

**Important limits:** APPEND currently optimizes the bytes transmitted over
network, **not** GPU → CPU serialization: each producer still exports a full
CPU snapshot and compares its byte prefixes. Each receiver still reconstructs
a full CPU buffer before copying to GPU. Therefore APPEND can be slower than
FULL on local loopback, especially with fast networks and CPU contention. It
is not compression, RDMA or GPU Direct. Payload metrics do not include gRPC
framing overhead. Same weights/layout are required at both nodes; model-name
matching is not a cryptographic fingerprint of identical model weights.

Example from one local Qwen3-0.6B two-process run, same prompt and 10 latent
steps (recorded under `var/experiments`): FULL sent 72,253,440 bytes (three
snapshots). APPEND sent 37,961,728 bytes (one FULL, two APPEND), saving
34,291,712 bytes / 47.46%. The sum of coordinator RPC handoff times was about
255 ms FULL vs 303 ms APPEND: no latency acceleration was observed locally.
LLM decoding uses sampling, so total wall-time and answer text should not be
interpreted as deterministic transport benchmarks.

The test suite additionally verifies a real Qwen3 and Llama Transformer can
continue from network-APPEND KV with identical logits; >1 MiB packet
boundaries, receiver cache loss, changed prefixes, and FULL-only baseline.
