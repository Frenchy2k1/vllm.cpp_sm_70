# SERVE-RESPONSE-METRICS — per-request timing via EngineCoreEvents

Row: `SERVE-RESPONSE-METRICS` (engine-matrix §9 Serving; feature-matrix §Serving).
Claim: `CLAIM-ROADMAP-C8-RESPONSE-METRICS`. Pin: vLLM 0.26.0.dev0 `555967922`
(`${VLLM_SOURCE}` = `/home/mudler/_git/vllm/`). CPU-only, behavioural gate.

## Scope

Port the per-request timing surface (queue/prefill/inference timing intervals +
preemption counter) that vLLM derives from **EngineCoreEvents**. `SERVE-METRICS`
landed the always-on Prometheus catalog and the live per-step stat wiring
(`CLAIM-ROADMAP-C8-METRICS-WIRE`) but explicitly deferred these series:
`vllm:request_{queue,prefill,inference}_time_seconds` and
`vllm:num_preemptions_total`. They already exist in the catalog and are already
observed once per finished request by `PrometheusStatLogger::Record`
(`src/vllm/v1/metrics/loggers.cpp:225,254-257`) — but observed with value 0
because nothing produced the underlying intervals. This row adds the event
surface + the consumer that folds it into the timing.

The row also nominally covers the OpenAI **response-body** timing surface
(streaming/non-streaming per-request timing fields + CLI validation); that part
stays a named residual (see Risks/decisions).

## Upstream chain

Whole chain (file:line), pin `555967922`:

- `vllm/v1/engine/__init__.py:150-155` — `EngineCoreEventType(IntEnum)`
  `QUEUED=1, SCHEDULED=2, PREEMPTED=3`.
- `vllm/v1/engine/__init__.py:158-176` — `EngineCoreEvent(msgspec.Struct)`
  `{type, timestamp}` + `new_event(type, timestamp=None)` (default
  `time.monotonic()`).
- `vllm/v1/engine/__init__.py:193` — `EngineCoreOutput.events: list | None`.
- `vllm/v1/request.py:98` — `Request.events: list[EngineCoreEvent]`.
- `vllm/v1/request.py:309-320` — `record_event` / `take_events` (list | None,
  clears).
- `vllm/v1/core/sched/scheduler.py:2135` — QUEUED at add_request (`:1318` for the
  streaming-session path, deferred with streaming).
- `vllm/v1/core/sched/scheduler.py:461` — `scheduled_timestamp = time.monotonic()`
  once per `schedule()`.
- `vllm/v1/core/sched/scheduler.py:1001-1004` — SCHEDULED on admission to the
  running batch, gated on `self.log_stats`.
- `vllm/v1/core/sched/scheduler.py:1203-1224` — `_preempt_request(request,
  timestamp)`; `:1221` records PREEMPTED, gated on `self.log_stats`.
- `vllm/v1/core/sched/scheduler.py:1839,1874` — `events=request.take_events()`.
- `vllm/v1/metrics/stats.py:218-236` — `RequestStateStats`
  (`queued_ts/scheduled_ts/first_token_ts/last_token_ts`).
- `vllm/v1/metrics/stats.py:428-450` — `IterationStats.update_from_events`:
  QUEUED→`queued_ts`; SCHEDULED→ first only sets `scheduled_ts`; PREEMPTED→
  `num_preempted_reqs += 1`.
- `vllm/v1/metrics/stats.py:452-497` — `update_from_finished_request`:
  `queued = scheduled_ts - queued_ts`, `prefill = first_token_ts - scheduled_ts`,
  `decode = last_token_ts - first_token_ts`, `inference = last_token_ts -
  scheduled_ts`.

## Our baseline

- The Prometheus catalog + `PrometheusStatLogger::Record(SchedulerStats,
  IterationStats)` already observe `queued_time`/`prefill_time`/`inference_time`/
  `decode_time` + `num_preempted_reqs` (`loggers.cpp:225,254-257`). `decode_time`
  already computes correctly (`last_token_ts - first_token_ts`); the three others
  and the preemption counter are the gap (`output_processor.cpp` left the intervals
  at 0.0, `IterationStats.num_preempted_reqs` never incremented).
- `FinishedRequestStats.{queued,prefill,inference}_time` + `num_preempted_reqs`
  fields already exist on the ported structs (`stats.h`).
- The scheduler emits no events; `Request` has no `events`; `EngineCoreOutput`
  has no `events`. That is what this row adds.

## Port map

| Upstream | Ours |
|---|---|
| `engine/__init__.py:150-176` | NEW `include/vllm/v1/engine/event.h` (`EngineCoreEventType` + `EngineCoreEvent`) |
| `request.py:98,309-320` | `include/vllm/v1/request.h` (`events` + `record_event` + `take_events`) |
| `engine/__init__.py:193` | `include/vllm/v1/engine/types.h` (`EngineCoreOutput.events`, omit-default) |
| `scheduler.py:461,1003,1221,2135,1839` | `src/vllm/v1/core/sched/scheduler.cpp` (`add_request`/`schedule`/`preempt_request`/`update_from_output`) + `log_stats_` in `scheduler.h` (`preempt_request` gains a `timestamp` param) |
| `stats.py:218-236,428-476` | `include/vllm/v1/engine/output_processor.h` (`RequestState.queued_ts/scheduled_ts`) + `src/vllm/v1/engine/output_processor.cpp` (`update_from_events` fold + interval fill in `process_outputs`) |
| logger already observes | NO change to `loggers.cpp` |

## Tests to port

vLLM covers this through integration/metrics tests, not a dedicated unit; two
behavioural CPU gates express it:

- `tests/vllm/v1/test_scheduler.cpp` "records QUEUED/SCHEDULED/PREEMPTED
  engine-core events" (15 assertions): reuses the KV-exhaustion preemption
  mechanics; asserts QUEUED at add_request, SCHEDULED on admission, PREEMPTED on
  eviction, in order, `num_preemptions == 1`. RED-first: before the wiring
  `Request.events` is always empty.
- `tests/vllm/v1/test_llm_engine.cpp` "per-request queue/prefill/inference timing
  populates" (26 assertions): drives the reference engine to completion; asserts
  the `_sum` of the queue/prefill/inference histograms flips 0→positive and
  `inference == prefill + decode`, `prefill ≤ inference ≤ e2e`. RED-first: with
  the intervals left at 0, 5 assertions fail.

## Gates

- Correctness: token stream unchanged (the 5 `test_llm_engine` determinism cases
  + `test_prometheus_metrics` 4/4 stay green); the two new behavioural gates
  green; full affected CPU ctest green.
- Performance: `benchmark_binding=false` — observational events off the compute
  path, no vLLM throughput A/B applies (NOT APPLICABLE in docs/BENCHMARKS.md).
- Build: clean full `vllm` library CPU `-Werror` rebuild, 0 warnings.
- Hardware: none (CPU-only).

## Dependencies

- `SERVE-METRICS` (the Prometheus catalog + `PrometheusStatLogger` + the live
  per-step wiring `CLAIM-ROADMAP-C8-METRICS-WIRE`, which built `IterationStats`
  and the `RequestState` timing this row extends).
- No new third-party dependency; `MonotonicSeconds` (`stats.h`) supplies the
  shared steady clock used by both the event timestamps and the step timestamp.

## Work breakdown

- W1 event type + Request events (event.h, request.h).
- W2 scheduler emission + drain (scheduler.{h,cpp}).
- W3 consumer + interval fill (output_processor.{h,cpp}).
- W4 gates (test_scheduler, test_llm_engine) + records.

All landed 2026-07-27.

## Risks/decisions

- Decision: `EngineCoreEvent` lives in a NEW small standalone header
  (`engine/event.h`), not inline in `request.h` or `types.h`, mirroring the fact
  that upstream defines it in `engine/__init__.py` and both `request.py` and the
  output types import it — this avoids a circular include (`request.h` ↔
  `types.h`).
- Decision: `log_stats_` defaults true (mirrors upstream `not
  disable_log_stats`); events are inert unless a frontend stat logger consumes
  them, so the no-logger path is byte-identical. Gating record_event on it keeps
  structural parity + a future disable path.
- Clock: `first_token_ts`/`last_token_ts` (the `EngineCoreOutputs.timestamp`) and
  the event timestamps both come from `MonotonicSeconds` (steady_clock), so every
  interval is non-negative and `inference == prefill + decode` exactly.
- Residual (row stays honest; C8 stays PARTIAL): the OpenAI chat/completion
  RESPONSE-BODY timing surface (protocol field + streaming/non-streaming serving
  emission + invalid multi-choice/multi-prompt suppression + CLI validation,
  `vllm/entrypoints/openai/{completion,chat_completion}/serving.py:461-481,765-784`);
  AsyncLLM production-serving metric wiring; config-gated metric families
  (spec-decode/kv-connector/mm/LoRA); the `:1318` streaming-session QUEUED site.
