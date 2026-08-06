# Startup latency as a measured axis (cold launch to first `/health`)

Status: **harness LANDED 2026-08-06; number PENDING.**
Rows: none new — this extends the existing `SERVE-GATE-ONLINE` harness
(`.agents/specs/cuda-online-serving-gate.md`). `benchmark_binding=false`:
a server-lifecycle axis, no kernel, no forward, no throughput denominator.

## Why

The online serving gate protocol has always listed `startup` among the axes an
interleaved repetition must record (`cuda-online-serving-gate.md`,
`qwen35-plain-bf16-direct-load.md:128` — "peak/stable PSS/RSS, VRAM, **startup**,
total/output throughput, request rate, TTFT, TPOT and ITL"). It was never
captured. `scripts/dgx-online-serving.sh` waited for readiness and threw the
duration away; no manifest carried a startup field; `grep` for `startup_s` /
`time_to_ready` over `tools/`, `scripts/` and `docs/` returned nothing.

So the honest answer to "is our startup faster than vLLM's?" was **unknown**,
and it stayed unknown for the whole campaign. This closes the measurement gap.
It does NOT claim a result.

## Definition

**`startup_seconds` = elapsed wall time from immediately before the server
process is spawned to the first successful `GET /health`.**

Both arms use the identical probe against the identical endpoint, and both go
through the same `start_server`, so the launched commands are the timed grid's
verbatim. The metric therefore answers the only startup question a user asks:
*how long until this server can serve.* It deliberately includes everything
each stack really does before it can answer — for vLLM: Python/torch import,
engine init, any flashinfer JIT, CUDA-graph capture; for ours: weight load and
graph capture.

## What the harness had to change

| Change | Why |
|---|---|
| Readiness cadence 5 s -> 0.2 s (`ready_poll_interval`) | The old cadence was itself ~12% of a ~40 s startup. It was the resolution floor, and the reason no number was reportable. |
| `ready_timeout_seconds=1800`, deadline loop replaces `seq 1 360` | Preserves the previous 30-minute budget exactly under the finer cadence. |
| Launch/ready stamps in `start_server` | Launch stamp sits immediately before the spawn so nothing else is attributed to startup. |
| `record-startup` embeds and validates the leg's cache-drop report | The number is dominated by weight paging, so it is meaningless warm. A startup artifact **cannot exist** for a leg whose page cache was not dropped. |
| `--startup-only` mode | A full `--execute` grid costs hours; this axis needs minutes. Skips the token model gate (no token is compared) and the timed client, builds only the `server` target. |
| `summarize-startup` refuses an incomplete series | A missing leg silently biases the median toward whichever arm finished. |

Known contamination, stated rather than hidden: the memory sampler is launched
between the spawn and the readiness wait, so it sits inside the measured window.
It is launched identically on both arms, so it cancels in the ratio; only the
absolute number carries its (sub-100 ms) cost.

## Measurement recipe

```
scripts/dgx-online-serving.sh --dry-run --claim-root <root>          # manifest
scripts/dgx-online-serving.sh --startup-only --model 27 \
  --snapshot <hf-snapshot> --source-corpus <evidence>/corpus/27 \
  --evidence <evidence> --build-dir <build> --configure-log <log>
```

Three repetitions, interleaved ours/vLLM under one `/tmp/gpu` lock, page cache
dropped before every leg, GPU proven idle before and after each. Output:
`<evidence>/startup/<model>/summary.json` with per-arm median/min/max and
`startup_speedup_vs_vllm` (> 1 means ours reaches readiness sooner).

## Expected variance (read before trusting a single run)

GB10 e2e wall-clock swings with reload behavior — this is exactly the effect
`.agents/state.md` records for the Laguna arm, and startup IS that effect rather
than an incidental contaminant of it. Mitigations in the harness: every leg is
cold by construction, arms are interleaved so thermal/temporal drift hits both,
and the summary reports min/max alongside the median. **Do not report a
one-repetition startup number.**

## Tests

`tests/tools/test_online_gate_startup.py` (19 cases, CPU, no GPU):
artifact provenance and every refusal (`overwrite`, non-positive elapsed, coarse
poll interval, warm page cache, unknown engine/model, non-`/health` probe);
summary medians/ratio plus incomplete-series and wrong-model refusals; and the
driver contracts (fine cadence, preserved timeout, stamp ordering, `--startup-only`
actually parses, interleaving, no timed client, server-target-only build,
`shellcheck` and `bash -n` clean).

Upstream test to port: none. vLLM has no startup-latency test; this is a
harness axis, recorded as a deviation in the sense of
`porting-inventory.md` §9 (measurement tooling is ours, not a vLLM mirror).

## Open

- **The number.** Not measured. On 2026-08-06 dgx.casa root was at 100%
  (20 GiB free) with no existing CUDA `server` build, so the ours arm could not
  be built. Reclaiming the untagged 7.37 GB docker image plus the 4.4 GB builder
  cache was authorized; the run is owed after that.
- 35B and `q3mxfp4` arms after 27B.
- Whether startup should become a standing column of the `--execute` grid
  summary. The stamps are already recorded there for free; only the aggregation
  is missing.
