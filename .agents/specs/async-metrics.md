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
| `async_llm: live per-step stats populate the Prometheus registry` | `tests/vllm/v1/test_async_llm.cpp` (new) | Baseline zero; after driving N requests to completion through `AsyncLLM`: `vllm:num_requests_running`/`_waiting` gauges fall back to 0, `vllm:prompt_tokens_total` and `vllm:generation_tokens_total` equal the exact counts the run produced, `vllm:request_success_total{finished_reason="length"}` counts the finished requests, and the TTFT / ITL / e2e / TPOT / iteration-tokens / generation-tokens histograms carry the exact expected sample counts. Mirrors `test_llm_engine.cpp` case 6. |
| per-request timing on the async path | same case | `vllm:request_{queue,prefill,inference,decode}_time_seconds` `_sum` values are **positive**, and `inference == prefill + decode`. Mirrors `test_llm_engine.cpp` case 7, which is the `SERVE-RESPONSE-METRICS` invariant. |
| async-scheduling (batch-queue) step path | same file | The same registry invariants hold with `max_concurrent_batches = 2`, which is what proves item (4) of Scope. RED without the `step_with_batch_queue` stamp: gauges stay 0 and TTFT is nonsensical. |

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

Filled in by the implementing commit: RED output verbatim, focused GREEN
counts, the full `ctest` line, any serial re-run of a known-flaky binary with
both numbers, and the landed SHA/PR.

* RED (pre-wiring): recorded in the PR body and the implementation commit
  message.
* GREEN: `test_async_llm`, `test_llm_engine`, `test_prometheus_metrics`, full
  `ctest`.

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

Filled in when the row's residual reaches `DONE`. Reserved for: what the async
gate measured, what was rejected (a snapshot-copy `Expose`, moving the stamp
into `Scheduler::update_from_output`), and why the logger attach is opt-in
rather than always-on.
