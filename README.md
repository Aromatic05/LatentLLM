# LatentLLM

Distributed Token / KV-Cache handoff laboratory, based on [LatentMAS](https://github.com/Gen-Verse/LatentMAS).

P0: interactive, factual experiment visualization; real Token and KV-Cache transport between independent LLM runtimes.

- `web/` — React UI, topology, events, and latency timelines
- `server/` — Python inference nodes, transport, experiment API
- `proto/` — transfer wire protocol and generation
- `upstream/LatentMAS/` — pinned original inference implementation
- `tests/` — behavior-level tests
- `var/` — local experiment artifacts (untracked)

Both physical hosts run the Python node service with the **same model weights**. The browser only receives telemetry, never KV payloads. Initial work must not misrepresent placeholder data as observed measurements.
