# AsyncLLM serving-path metric wiring (`SERVE-METRICS`, `ROAD-V1-C8`)

Issue: [#277](https://github.com/mudler/vllm.cpp/issues/277).
Rows: `SERVE-METRICS`, `SERVE-RESPONSE-METRICS` (engine-matrix §9 Serving;
feature-matrix §Serving). Punch-list item 7 of
[roadmap-v1-completion.md](roadmap-v1-completion.md), whose `ROAD-V1-C8`
residual names "AsyncLLM serving-path metric wiring" explicitly.

Companion specs: [prometheus-metrics.md](prometheus-metrics.md) (the catalog and
the synchronous step-site wiring), [per-request-response-metrics.md](per-request-response-metrics.md)
(EngineCoreEvents → per-request timing), [async-serving.md](async-serving.md)
(the AsyncLLM frontend, which deferred "stat loggers" at W2).

## Scope

**In.** The `AsyncLLM` step-site metric wiring: the production HTTP server runs
`AsyncLLM`, whose output handler currently folds nothing into any
`PrometheusStatLogger`, so `/metrics` serves a well-formed but permanently
motionless `vllm:*` catalog. Concretely:

1. `AsyncLLM` gains the same non-owning stat-logger attach point `LLMEngine`
   already has (`set_stat_logger`), mirroring upstream's `logger_ref[0]`
   indirection (`async_llm.py:648-652`).
2. `AsyncLLM::RunOutputHandler` builds one `IterationStats` per pulled
   `EngineCoreOutputs` when a logger is attached and outputs are present
   (`async_llm.py:662-665`), threads it through `OutputProcessor::process_outputs`
   (`:676-678`), and folds it plus `EngineCoreOutputs::scheduler_stats` into the
   logger (`:697-702`).
3. `PrometheusStatLogger` becomes safe for one recorder thread concurrent with
   scraping readers. Upstream serializes this with the GIL; `PromRegistry` is
   documented "not thread-safe by itself", and after (2) the recorder is the
   output-handler thread while `Expose()` runs on an HTTP worker thread. Without
   this, wiring the metric is a data race, not a feature.
4. `EngineCore::step_with_batch_queue` stamps `scheduler_stats` + `timestamp` on
   its `EngineCoreOutputs` the way `EngineCore::step` already does. Upstream
   stamps both in the path *both* step functions share
   (`scheduler.py:1938-1951` for `scheduler_stats`;
   `engine/__init__.py:249-251` `__post_init__` for `timestamp`), so the
   async-scheduling serving path is entitled to the same values. Today it gets
   a default-constructed `SchedulerStats` and `timestamp == 0.0`, which after
   (2) would publish zero gauges and TTFT values equal to `-arrival_time`.
5. `vllm-server` attaches the logger it already constructs to the **async**
   engine as well (`server_main.cpp:931-939`), and the stale "AsyncLLM exposes
   no live logger" residual notes in `api_server.{h,cpp}` are corrected.

**Out (a separate residual on the same row, not reopened here).** The
config-gated metric families vLLM does not register for a plain text engine:
spec-decode, kv-connector/external prefix cache, multimodal cache, LoRA. Also
out: `OutputProcessor.update_scheduler_stats` (`async_llm.py:692`), whose only
body upstream is `self.lora_states.update_scheduler_stats(...)` — it belongs to
the LoRA family; `do_log_stats_with_interval` / `LoggingStatLogger` (the
human-readable log line, never registered by our port); DP/multi-engine
aggregation (`AggregateStatLoggerBase`, `engine_idx`); and the
`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` chunking loop, which our synchronous
`process_outputs` does not have on either engine.

**Dispatch behaviour.** Strictly additive and opt-in: with no logger attached
(the default, and every existing test) `RunOutputHandler` takes the same
`process_outputs(outputs)` no-stats call it takes today, so the greedy token
stream and the `VT_TTFT_DUMP` diagnostic are byte-identical.

## Upstream chain

Pinned oracle: vLLM `555967922` (0.26.0.dev0), recorded in
[upstream-sync.md](../upstream-sync.md). All line numbers are that revision.

| Upstream `file:line` | What it defines |
|---|---|
| `vllm/v1/engine/async_llm.py:638-707` | `_run_output_handler`: the whole async step site this row mirrors |
| `vllm/v1/engine/async_llm.py:648-652` | `logger_ref = [self.logger_manager]` — the mutable indirection so the logger can be swapped without a circular ref; our mirror is an atomic pointer |
| `vllm/v1/engine/async_llm.py:661-662` | `outputs = await engine_core.get_output_async()`; `num_outputs = len(outputs.outputs)` |
| `vllm/v1/engine/async_llm.py:664-665` | `iteration_stats = IterationStats() if (log_stats and num_outputs) else None` |
| `vllm/v1/engine/async_llm.py:676-678` | `output_processor.process_outputs(outputs_slice, outputs.timestamp, iteration_stats)` |
| `vllm/v1/engine/async_llm.py:686-690` | abort the reqs finished by stop strings (already ported) |
| `vllm/v1/engine/async_llm.py:697-702` | `if logger_ref[0]: logger_ref[0].record(engine_idx=..., scheduler_stats=outputs.scheduler_stats, iteration_stats=iteration_stats, ...)` — **no `num_outputs` guard on the record itself** |
| `vllm/v1/engine/llm_engine.py:306-332` | the synchronous twin our `LLMEngine::step` already mirrors, for the invariant comparison |
| `vllm/v1/core/sched/scheduler.py:1938-1951` | `update_from_output` attaches `make_stats()` to the `EngineCoreOutputs` — shared by `step` and `step_with_batch_queue` |
| `vllm/v1/engine/__init__.py:237,249-251` | `EngineCoreOutputs.timestamp` defaults to `time.monotonic()` in `__post_init__`, i.e. at construction inside `update_from_output` — again shared by both step paths |
| `vllm/v1/engine/core.py:622-720` | `step_with_batch_queue`; it does **not** stamp stats itself, because `update_from_output` already did |
| `vllm/v1/metrics/loggers.py:1100-1257` | `PrometheusStatLogger.record` — already ported |
| `vllm/v1/metrics/stats.py:186-259,377-475` | `SchedulerStats` / `IterationStats` / `FinishedRequestStats` — already ported |

**Trace plan.** Dispatch here is static (a pointer test in one function), so no
runtime trace is required. The behavioural claim — that the *async* stack
produces the same metric values the sync stack does for the identical workload —
is established by asserting the same invariants `test_llm_engine.cpp` case 6
asserts, on the async engine, over the CPU reference backend.

## Our baseline

| Anchor | State |
|---|---|
| `src/vllm/v1/engine/llm_engine.cpp:187-207` | Sync wiring, 1:1 with `llm_engine.py:306-332`. Builds `IterationStats` under `stat_logger_ != nullptr`, threads the pointer, folds under `outputs>0`. **This is the seam to reuse.** |
| `include/vllm/v1/engine/llm_engine.h:193-215` | `set_stat_logger()` + non-owning `metrics::PrometheusStatLogger* stat_logger_`. |
| `src/vllm/v1/engine/async_llm.cpp:262-294` | `RunOutputHandler`. The **only** `IterationStats` on the async path is at `:281-289`, built solely under `VT_TTFT_DUMP` for the TTFT-split diagnostic, and folded nowhere. |
| `include/vllm/v1/engine/async_llm.h:153-166` | No logger member, no attach point. |
| `src/vllm/v1/engine/core.cpp:86-89` | `step()` stamps `scheduler_stats = scheduler_.make_stats()` and `timestamp = MonotonicSeconds()`. |
| `src/vllm/v1/engine/core.cpp:219-230` | `step_with_batch_queue()` stamps **neither** — `timestamp` only under `VT_TTFT_DUMP`, `scheduler_stats` never. Its own comment names this the `SERVE-RESPONSE-METRICS` residual. |
| `src/vllm/v1/engine/core_proc.cpp:186-194` | Every queued `EngineCoreOutputs` came from a map entry that exists only when `outputs` is non-empty, so the async handler never observes a zero-output frame. |
| `include/vllm/v1/metrics/prometheus.h:38-40` | "`Not thread-safe by itself; the PrometheusStatLogger that owns one records under the engine's step cadence.`" |
| `src/vllm/entrypoints/openai/server_main.cpp:929-939` | Constructs the logger, attaches it to `loaded->engine()` (the **sync** engine, which the server never steps) and to the `/metrics` route. Its own comment concedes "async may under-report until fully wired." |
| `src/vllm/entrypoints/openai/api_server.cpp:1197-1201`, `include/vllm/entrypoints/openai/api_server.h:350-359` | Record the "no live logger on AsyncLLM" residual as the reason `/metrics` is not wired inside `ConfigureUtilityEndpoints`. |
| `tests/vllm/v1/test_llm_engine.cpp:846-931` | Case 6, the sync gate (44 asserts). No async equivalent exists anywhere. |
| `tests/vllm/v1/test_async_llm.cpp` | Drives the real Scheduler/EngineCoreProc/OutputProcessor with a canned one-token-per-step `RunnerStub`. Asserts nothing about metrics. |

**Honest gap.** A production scrape of `vllm-server --enable-metrics` returns
the full catalog with every counter at 0, every gauge at 0 and every histogram
at `_count 0`, indefinitely, under any load.

## Port map

| Upstream | Local | Note |
|---|---|---|
| `async_llm.py:648-652` `logger_ref` | `include/vllm/v1/engine/async_llm.h` — `set_stat_logger()` + `std::atomic<metrics::PrometheusStatLogger*> stat_logger_` | Deviation: an atomic pointer replaces the one-element Python list. Same purpose (swappable without a circular ref), and it is what makes the attach visible to the already-running handler thread. |
| `async_llm.py:661-665` | `src/vllm/v1/engine/async_llm.cpp` `RunOutputHandler` | `IterationStats` built when `logger != nullptr && !outputs.outputs.empty()`; the `VT_TTFT_DUMP` diagnostic keeps its independent trigger so an unset-env, no-logger run is instruction-identical. |
| `async_llm.py:676-678` | same | `process_outputs(outputs, &iteration_stats)`; `outputs.timestamp` is read inside `process_outputs` (`output_processor.cpp:367`) exactly as upstream passes `engine_core_timestamp`. |
| `async_llm.py:697-702` | same | `logger->Record(outputs.scheduler_stats, iteration_stats)` after leaving the output-processor critical section, mirroring the sync site's ordering relative to `abort_requests`. |
| `scheduler.py:1938-1951` + `engine/__init__.py:249-251` | `src/vllm/v1/engine/core.cpp` `step_with_batch_queue` | Stamp `scheduler_stats` + `timestamp` unconditionally, matching `step()`. We keep the stamp at the two `EngineCore` step sites rather than moving it into `Scheduler::update_from_output`: `step()` already stamps there, and relocating it would restructure the synchronous path this row is not touching. Recorded deviation. |
| `prometheus_client` under the GIL | `include/vllm/v1/metrics/loggers.h`, `src/vllm/v1/metrics/loggers.cpp` | `mutable std::mutex mu_` guarding `Record`, `SetCacheConfigInfo` and `Expose`. Written from scratch (no upstream analogue — Python has no such need); recorded in the porting inventory as such via this spec. |
| `api_server.py:238-240` metrics mount | `src/vllm/entrypoints/openai/server_main.cpp` | Attach the same logger instance to `loaded->async_engine()`. |

## Tests to port

vLLM's own metric tests are HTTP-level (`tests/entrypoints/serve/instrumentator/test_metrics.py`)
and its `EXPECTED_METRICS_V1` catalog assertion is already ported by
`tests/vllm/v1/test_prometheus_metrics.cpp`. There is no upstream unit test that
drives `_run_output_handler` against a registry — upstream covers it only
through the server integration test, which needs a real model. So the async gate
is a **local behavioural gate**, and it is written to assert the *same
invariants* the ported sync gate asserts, on the async stack:

| Case | Where | Asserts |
|---|---|---|
| `async_llm: live per-step stats populate the Prometheus registry` | `tests/vllm/v1/test_llm_engine.cpp:1025` | Baseline zero; after driving two requests to completion through `AsyncLLM`: `vllm:num_requests_running`/`_waiting` gauges fall back to 0, `vllm:prompt_tokens_total` and `vllm:generation_tokens_total` equal the exact counts the run produced, `vllm:request_success_total{finished_reason="length"}` counts the finished requests, and the TTFT / ITL / e2e / TPOT / iteration-tokens / generation-tokens histograms carry the exact expected sample counts. Mirrors `test_llm_engine.cpp` case 6. |
| per-request timing on the async path | same case | `vllm:request_{queue,prefill,inference,decode}_time_seconds` `_sum` values are **positive**, and `inference == prefill + decode`. Mirrors `test_llm_engine.cpp` case 7, which is the `SERVE-RESPONSE-METRICS` invariant. |
| `async_llm: with no logger attached the token stream is unchanged` | `tests/vllm/v1/test_llm_engine.cpp:1148` | The inertness claim: identical token ids with and without an attached logger. |
| async-scheduling (batch-queue) step path | `tests/vllm/v1/test_async_llm.cpp:560,618` | The same registry invariants with `max_concurrent_batches = 2` over an `AsyncScheduler`, which is what proves item (4) of Scope. Two cases: a POLL until `vllm:num_requests_running` reads 1 while a delayed stub keeps a request in flight (RED: times out, the gauge never leaves 0 — and it exercises the recorder mutex, since `Expose()` runs on the test thread concurrently with the handler's `Record()`), and exact token counters + positive TTFT/e2e `_sum` after a short run (RED: unstamped `timestamp` makes both observations `-arrival_time`). |

**Harness note (refinement of the W0 plan).** The primary gate lives in
`test_llm_engine.cpp` rather than `test_async_llm.cpp` because that file already
has `AsyncHarness` — the identical model, KV config, tokenizer and sampling
params as sync case 6, but fronted by `AsyncLLM` — plus the `MetricValue` /
`HistogramCount` / `HistogramSum` helpers and the `kL` label suffix. Asserting
"the same invariants" is then literal rather than approximate. Only the depth-2
batch-queue pair, which needs an `AsyncScheduler` and a delay-injecting runner
stub, lives in `test_async_llm.cpp`, where both patterns already exist.

The async gate scrapes **after `shutdown()`** joins the output-handler thread.
That is the quiescence point: upstream's asyncio handler runs to its next
`await` before a woken consumer resumes, so `record()` always precedes the
consumer, but our handler is a real thread and a drained collector says nothing
about whether that step's fold has retired.

Existing gates that must stay byte-identical: `test_prometheus_metrics` (4/4),
`test_llm_engine` cases 6 and 7, `test_async_llm`'s existing cases,
`test_engine_core_proc`, `test_openai_api_server`.

## Gates

CPU reference backend only; no GPU is available to this claim, and none is
needed — the row is frontend wiring over a canned runner.

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-cpu -j 18
./build-cpu/tests/test_async_llm
./build-cpu/tests/test_llm_engine
./build-cpu/tests/test_prometheus_metrics
ctest --test-dir build-cpu -j 6 --output-on-failure
```

RED-first is mandatory and its verbatim output is recorded in Evidence: the new
case is written and run against the unwired tree first, and must fail because
the registry reports zeros — not because it fails to compile.

No performance axis is claimed. The wiring is inert without an attached logger,
and with one attached it costs one `IterationStats` per engine step on the
output-handler thread — the identical cost the synchronous engine already pays.
No throughput/latency/memory measurement is therefore owed, and none is
recorded in `docs/BENCHMARKS.md` beyond the lifecycle note.

Known-not-mine, per [#274](https://github.com/mudler/vllm.cpp/issues/274): main
carries 5 pre-existing ASan/UBSan failures (`test_load_direct_upload`,
`test_llama_embedding_fold`, `test_laguna_nvfp4_loader`, `test_openai_api_server`,
`test_capi`). Known flaky under a loaded shared box: `test_openai_api_server`,
`test_openai_conformance`, `test_async_llm`, `test_engine_core_proc` — any
failure is re-run **serially** and both numbers reported.

## Dependencies

| Dependency | State |
|---|---|
| `SERVE-METRICS` catalog + `PrometheusStatLogger::Record` | DONE (`prometheus-metrics.md`) |
| `SERVE-RESPONSE-METRICS` EngineCoreEvents → per-request timing | DONE (`per-request-response-metrics.md`); consumed unchanged |
| `SERVE-ASYNC-LLM` frontend + `ENG-CORE-BUSY-LOOP` | DONE (`async-serving.md`) |
| `ENG-BATCH-QUEUE` depth-2 async scheduling | ACTIVE; this row stamps its stats, it does not change its scheduling |
| Toolchain | CPU-only CMake/Ninja, C++20. No model, no checkpoint, no GPU, no network. |

## Work breakdown

Single slice — the wiring is one call site and cannot be split without leaving a
data race or a zeroed path in the tree between commits.

| W | Content |
|---|---|
| W0 | This spec, committed alone, before any implementation. Issue linked here, in `roadmap_v1.md`'s open-issue table, and in the PR body. |
| W1 | RED: the new async metric case in `tests/vllm/v1/test_async_llm.cpp`, run against the unwired tree, output captured verbatim. |
| W2 | GREEN: `AsyncLLM::set_stat_logger` + the `RunOutputHandler` fold; the `step_with_batch_queue` stat/timestamp stamp; the `PrometheusStatLogger` mutex; the `server_main.cpp` async attach; the stale residual comments corrected. |
| W3 | Records: `SERVE-METRICS` + `SERVE-RESPONSE-METRICS` matrix rows, roadmap `ROAD-V1-C8` residual, `docs/STATUS.md`, `docs/BENCHMARKS.md`, `.agents/NOW.md`, the coordination claim. |

## Risks/decisions

* **Lock ordering.** `Record` is called *outside* `output_processor_mutex_`, so
  the logger mutex is a leaf and can never participate in a cycle with the
  output-processor lock or the collector condition variables.
* **A scrape must not stall the engine.** `Expose()` under the same mutex means
  a scrape briefly blocks the output-handler thread. The exposition is a few
  kilobytes of string building over a few hundred series, which is orders of
  magnitude below one engine step; the alternative (a snapshot copy) buys
  nothing measurable and doubles the state. Decision: one mutex.
* **`step_with_batch_queue` stamping is a behaviour change on the async
  scheduling path** for anything that reads `EngineCoreOutputs::timestamp`.
  Today the only reader is `process_outputs` under a non-null `IterationStats`,
  which on that path only ever happened under `VT_TTFT_DUMP` — which stamped the
  timestamp anyway. So no existing behaviour changes; the diagnostic's own
  conditional stamp becomes redundant and is folded into the unconditional one.
* **Not a product decision.** Everything here is vLLM-defined; nothing in this
  spec asks the developer to choose a behaviour.

## Evidence

Build: `cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF`, `-j 18`,
1131/1131 clean under `-Werror`.

**RED** (attach point present, nothing folded, nothing stamped):

```text
test_llm_engine -tc="async_llm*"
  test_llm_engine.cpp:1083: CHECK( MetricValue(t, kPrompt) == 2.0 )        -> 0 == 2
  test_llm_engine.cpp:1084: CHECK( MetricValue(t, kGen) == total_gen )     -> 0 == 8
  test_llm_engine.cpp:404:  REQUIRE( p != npos ) series absent:
                            vllm:request_success_total{...,finished_reason="length"}
  test_llm_engine.cpp:1092: time_to_first_token_seconds _count             -> 0 == 2
  test_llm_engine.cpp:1093: inter_token_latency_seconds _count             -> 0 == 6
  test_llm_engine.cpp:1094: e2e_request_latency_seconds _count             -> 0 == 2
  test_llm_engine.cpp:1095: request_time_per_output_token_seconds _count   -> 0 == 2
  test_llm_engine.cpp:1096: iteration_tokens_total _count                  -> 0 >= 1
  test_llm_engine.cpp:1097: request_generation_tokens _count               -> 0 == 2
  test_llm_engine.cpp:1101-1103: request_{queue,prefill,inference}_time _count -> 0 == 2
  test_llm_engine.cpp:1109-1113: {queue,prefill,inference,decode,e2e} _sum > 0 -> 0 > 0
  test_llm_engine.cpp:1119: time_to_first_token_seconds _sum > 0           -> 0 > 0
  [doctest] test cases:  2 |  1 passed |  1 failed | 13 skipped
  [doctest] assertions: 63 | 44 passed | 19 failed |

test_async_llm -tc="*depth-2*"
  test_async_llm.cpp:608: CHECK( saw_running )                             -> false
  test_async_llm.cpp:657: prompt_tokens_total                              -> 0 == 1
  test_async_llm.cpp:658: generation_tokens_total                          -> 0 == 8
  test_async_llm.cpp:660: time_to_first_token_seconds_count                -> 0 == 1
  test_async_llm.cpp:662: e2e_request_latency_seconds_count                -> 0 == 1
  test_async_llm.cpp:666: time_to_first_token_seconds_sum > 0              -> 0 > 0
  test_async_llm.cpp:668: e2e_request_latency_seconds_sum > 0              -> 0 > 0
  [doctest] test cases: 2 | 0 passed | 2 failed | 8 skipped
  [doctest] assertions: 9 | 2 passed | 7 failed |
```

**GREEN** after the wiring:

| Binary | Result |
|---|---|
| `test_llm_engine` | 15 cases / 291 assertions, 0 failed |
| `test_async_llm` | 10 cases / 325 assertions, 0 failed |
| `test_prometheus_metrics` | 4 cases / 81 assertions, 0 failed |
| `ctest --test-dir build-cpu -j 6` (pre-rebase) | **366/366 passed, 0 failed**, 780.19 s |
| `ctest --test-dir build-cpu -j 6` (post-rebase on `848d4a87`) | **368/368 passed, 0 failed**, 1249.55 s |

No serial re-run was owed on that run: none of the four known-flaky binaries
(`test_openai_api_server`, `test_openai_conformance`, `test_async_llm`,
`test_engine_core_proc`) failed in it.

**A gate log is not evidence until it is checked for contamination.** The
post-rebase re-run appeared to report `test_openai_conformance` failing with 13
of 23 cases. It did not. Three internal inconsistencies in the log file said so
before any test was re-run:

* the failing assertions cited
  `/home/mudler/_git/vllm.cpp-lora-w2/tests/vllm/entrypoints/openai/test_conformance.cpp`
  — a **different worktree**, while `strings` on the binary `ctest` actually
  invoked contains 152 references to this worktree and **zero** to that one;
* the file was **48,261 NUL bytes out of 60,238** — a sparse hole, the
  signature of two processes writing one file at independent offsets;
* its own progress lines counted `1/368` while the summary said `out of 369`.

A concurrent session was running `ctest` in `/home/mudler/_git/vllm.cpp-lora-w2`
(pid 218469, confirmed via `/proc/<pid>/cwd`), and its output landed in this
session's log. Our own run was still executing at the moment that "summary"
appeared. The two ctests together drove the box to a **load average of 360 on 20
cores**, and `test_conformance` binds an *ephemeral* port (`:409
bind_to_any_port`), so the `REQUIRE(res)` failures are `httplib::Result` falsy
from connect/read timeouts under starvation — which is precisely why that binary
is on the known-flaky list, and is not a port clash.

Two things follow, and both are cheap. Gate logs are written **inside the
worktree** (`.gatelogs/`, git-excluded) rather than to a shared temp path. And a
failing gate log is validated before it is believed: check the source paths it
cites against the binary, check for NUL runs, and check that its own test counts
agree.

**A re-run on an uncontaminated log then failed `test_async_llm` for real**, in
the pre-existing case `async_llm test_abort and test_multi_abort leave other
requests healthy` (`:325`, `Drain(engine, reused) == 3` observing 4) — not in
any case this row adds. Because that binary is squarely in this row's blast
radius, it was NOT called a flake on reputation. It was measured, paired, on the
same box, by alternating the two builds in one worktree and running 40
concurrent copies of the single case:

| Tree | round 1 | round 2 | round 3 | total |
|---|---:|---:|---:|---:|
| `main` @ `848d4a87` | 11/40 | 9/40 | 11/40 | **31/120 (25.8%)** |
| `main` + this row's wiring | 9/40 | 15/40 | 7/40 | **31/120 (25.8%)** |

Identical rate: the defect is pre-existing on `main` and this wiring neither
causes nor worsens it. It is a real engine-frontend race — `AsyncLLM::abort`
queues the core abort asynchronously while the case's precondition
(`has_unfinished_requests()`) reads only frontend state, so a token frame from
the previous incarnation of a reused request id can reach the new collector —
and it now has its own issue,
[#294](https://github.com/mudler/vllm.cpp/issues/294), rather than a silent fix
here. Serially on an idle box both that binary and `test_openai_conformance` are
green (10/10 and 23/23), and the full gate re-run on the idle box is **368/368,
0 failed** (log verified: 0 NUL bytes, this worktree only) — recorded above.
The case reappeared once more on the final base under load 167, and the split
was checked rather than assumed: this row's own depth-2 cases 8/8, the #294
case 7/8 — its measured rate, unchanged.

## Stop conditions

* Return `NEEDS_DECISION` rather than implement if the sync and async paths turn
  out to be unable to share `PrometheusStatLogger` without restructuring — a
  second, async-only logger is explicitly forbidden by this spec.
* Return `NEEDS_CONTEXT` if the `SERVE-METRICS` row is found already claimed by
  another live session, or if the row's state no longer matches this record.
* Stop and report rather than widen scope if closing the async wiring turns out
  to require any config-gated metric family (spec-decode / kv-connector / mm /
  LoRA); those are the sibling residual.
* Never turn the new gate green by relaxing an assertion.

## Outcome

The `SERVE-METRICS` row keeps its residual (the config-gated families), so it
does not reach `DONE` here; this section records what closing the AsyncLLM
residual actually established.

**What was measured.** The async serving stack produces the same metric values
the synchronous one does for the identical workload: over two `hello` prompts
at `max_tokens=4` on the CPU reference MoE, both stacks report
`prompt_tokens_total == 2`, `generation_tokens_total == 8`,
`request_success_total{finished_reason="length"} == 2`, running/waiting gauges
back at 0, two TTFT observations, six ITL observations, and two each of e2e /
TPOT / generation-tokens — with every per-request timing `_sum` positive and
`inference == prefill + decode`. That is the whole claim of #277.

**What the RED proved beyond "the numbers were zero".** Three separate defects
were latent, and each has its own discriminator:

1. *No fold.* 19 assertions read 0 on a stack that had just generated 8 tokens.
2. *No `scheduler_stats` on the depth-2 path.* The running gauge never left 0
   even with a request in flight for 20 s. A single end-of-run scrape could not
   have caught this — every gauge is legitimately 0 once the batch drains — so
   the gate had to be a poll against a deliberately slowed runner.
3. *No `timestamp` on the depth-2 path.* TTFT and e2e were computed as
   `0 - arrival_time`. Their `_count` was already correct, so only the sign of
   the `_sum` distinguishes it; a count-only assertion would have passed.

**What was rejected.**

* *A snapshot-copy `Expose()`* (clone the registry under a short lock, render
  outside it). Rejected: rendering a few hundred series is orders of magnitude
  below one engine step, so the lock hold is not worth doubling the state and
  the copy cost on every scrape. One leaf mutex.
* *Moving the stamp into `Scheduler::update_from_output`*, which is literally
  where upstream does it. Rejected for this row: `EngineCore::step()` already
  stamps at its own site, so relocating would restructure the synchronous path
  #277 does not touch, for no behavioural difference. Recorded as a deviation
  in the Port map rather than done silently.
* *Recording inside `output_processor_mutex_`.* Rejected: it would make the
  logger mutex an inner lock under the output-processor lock, which is the only
  way to build a cycle here. `Record` is called after the critical section.

**Why the attach is opt-in.** It mirrors upstream — `logger_ref[0]` is `None`
unless a logger manager exists, and `log_stats` gates the `IterationStats`
build — and it keeps the default path instruction-identical, which is what lets
the no-logger token-stream identity case assert byte-equality rather than
approximate equality. The server turns it on with `--enable-metrics`, whose
default is already on.
