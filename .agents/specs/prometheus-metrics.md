# Prometheus `/metrics` exposition (`SERVE-METRICS`, ROAD-V1-C8)

Live spec for the `SERVE-METRICS` engine-matrix row: the Prometheus text
exposition served at `GET /metrics`, mirroring vLLM 0.26.0.dev0
(`555967922`). The oldest open T0 serving debt.

## Scope

Wire a self-contained Prometheus registry + text-format-0.0.4 exposition and the
always-on vLLM metric catalog (`PrometheusStatLogger`), and expose it at
`GET /metrics` behind an opt-in backing on the OpenAI `ApiServer`. The gate is
metric-NAME + label-schema parity with vLLM's own scrape spec. Config-gated
metric families (kv-connector, mm-cache, LoRA, spec-decoding, kv-block-lifetime,
corrupted, prompt_tokens_by_source, engine_sleep_state) are OUT of the always-on
core and stay INVENTORIED — vLLM itself only registers them under the matching
config, so their absence is faithful, not a gap.

## Upstream chain

- `vllm/entrypoints/serve/instrumentator/metrics.py:82` — `make_asgi_app(registry)`
  mounts `/metrics`; `PrometheusResponse` sets `text/plain; version=0.0.4;
  charset=utf-8` (`:52-60`).
- `vllm/v1/metrics/loggers.py:480-1060` — `PrometheusStatLogger` registers every
  metric: names, documentation, type (Counter/Gauge/Histogram/Info), histogram
  buckets, and the `["model_name","engine"]` label schema; `build_buckets`
  /`build_1_2_5_buckets` (`:1284-1305`) build the count buckets.
- `vllm/v1/metrics/loggers.py:1100-1257` — `record(SchedulerStats,
  IterationStats)` folds one engine step into the metrics.
- `vllm/v1/metrics/stats.py:186-259` — `SchedulerStats`, `IterationStats`,
  `FinishedRequestStats`.
- The `prometheus_client` package provides the text format: counters export a
  `_total` sample, histograms export `_bucket{le=...}`/`_sum`/`_count`, Info
  exports `{labels} 1.0`. This project has no Python at runtime, so the format
  bytes are reimplemented.

## Our baseline

Before this row: only `include/vllm/v1/metrics/stats.h` existed (the prefix-cache
`PrefixCacheStats`/`CachingMetrics` from `KV-PREFIX-CACHE`); NO Prometheus
registry, NO metric catalog, and `GET /metrics` was ABSENT from
`src/vllm/entrypoints/openai/api_server.cpp` (only completions/chat/models/health/
version were registered).

## Port map

- `include/vllm/v1/metrics/prometheus.h` + `src/vllm/v1/metrics/prometheus.cpp` —
  `PromRegistry`: Counter/Gauge/Histogram/Info families with multi-label series
  and the text-0.0.4 `Expose()` formatter (`kContentTypeLatest`).
- `include/vllm/v1/metrics/stats.h` — added `SchedulerStats`, `IterationStats`,
  `FinishedRequestStats` (the always-on subset of stats.py).
- `include/vllm/v1/metrics/loggers.h` + `src/vllm/v1/metrics/loggers.cpp` —
  `PrometheusStatLogger`: registers the always-on catalog 1:1 (names/help/type/
  buckets/labels), `Build1_2_5Buckets`, `Record()`, `SetCacheConfigInfo()`,
  `Expose()`.
- `src/vllm/entrypoints/openai/api_server.cpp` + `.h` — `handle_metrics()` and the
  opt-in `set_metrics_logger()` + conditional `GET /metrics` route.

## Tests to port

Upstream `tests/entrypoints/serve/instrumentator/test_metrics.py` — its
`EXPECTED_METRICS_V1` list + `test_metrics_exist` (`assert metric in
response.text`, a substring check over the exposition) is the executable spec.
Re-expressed as `tests/vllm/v1/test_prometheus_metrics.cpp`: substring presence
for every `EXPECTED_METRICS_V1` name (RED-first), label-schema `{model_name,
engine}`, TYPE lines, histogram bucket schedules, `build_1_2_5_buckets`
docstring example, `record()` value folding, and cumulative-bucket monotonicity.
Endpoint-level cases in `tests/vllm/entrypoints/openai/test_api_server.cpp`.

## Gates

- Parity: the `/metrics` exposition CONTAINS every `EXPECTED_METRICS_V1` string
  with the vLLM label schema and bucket bounds (CPU, deterministic). RED-first:
  dropping any family fails the substring assertion.
- Inertness: opt-in — a server without `set_metrics_logger()` does not register
  `/metrics` and is byte-identical; the 22 pre-existing api_server cases stay
  green. Clean CPU `-Werror`.

## Dependencies

None new. Reuses `PrefixCacheStats` (already in stats.h). The scheduler already
exposes `prefix_cache_metrics()` and running/waiting counts; wiring the live
engine's real per-step stats into `Record()` at the serving layer is a follow-on
(the values are engine-fed; the NAME/label/format parity is what this row gates).

## Work breakdown

- W1: `PromRegistry` + text exposition + unit test — DONE.
- W2: `PrometheusStatLogger` always-on catalog + `Record()` + cache_config_info —
  DONE.
- W3: `ApiServer` `/metrics` route (opt-in) + endpoint test — DONE.
- W4 (residual): fold the live EngineCore/Scheduler per-step SchedulerStats +
  IterationStats into `Record()` at the running server, plus the config-gated
  families (spec-decode/kv-connector/mm/LoRA) as their configs land — OPEN.

## Risks/decisions

- Decision: mirror `prometheus_client` byte-for-byte for the format — counters
  emit `_total`, histograms emit cumulative `_bucket`/`_sum`/`_count`, whole
  numbers render as `N.0` (Go `floatToGoString`). We deliberately do NOT emit the
  optional `_created` series (prometheus_client can disable them; vLLM's scrape
  spec never asserts them).
- Decision: register only the ALWAYS-ON families; config-gated ones stay out
  until their config exists, matching vLLM exactly (honest, not a shortcut).
- Risk: live-engine value wiring (W4) must not perturb the hot path — it is an
  opt-in read of already-computed stats, so inert by construction.
