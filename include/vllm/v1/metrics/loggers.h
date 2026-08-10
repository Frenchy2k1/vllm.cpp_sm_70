// Ported from: vllm/v1/metrics/loggers.py (PrometheusStatLogger) @ 555967922.
//
// SCOPE (SERVE-METRICS, ROAD-V1-C8): the ALWAYS-ON Prometheus metric catalog
// vLLM registers for a text engine — the exact family names, help strings,
// metric types, histogram buckets and the {model_name, engine} label schema —
// plus the record() path that folds one engine step's SchedulerStats +
// IterationStats into it, and the cache_config_info Info metric. The scrape
// text is the executable spec (tests/entrypoints/serve/instrumentator/
// test_metrics.py::EXPECTED_METRICS_V1, a substring assertion over the
// exposition), so this mirrors those names 1:1.
//
// DEFERRED (config-gated in vLLM, so NOT part of the always-on core): the
// kv-connector prefix-cache counters, mm-cache counters, LoRA info gauge,
// spec-decoding metrics, kv-block-lifetime histograms, corrupted-request
// counter, prompt_tokens_by_source, engine_sleep_state, and per-reason waiting
// breakdown values. vllm:num_requests_waiting_by_reason IS registered (name
// parity) but recorded at the aggregate only.
#ifndef VLLM_V1_METRICS_LOGGERS_H_
#define VLLM_V1_METRICS_LOGGERS_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "vllm/v1/metrics/prometheus.h"
#include "vllm/v1/metrics/stats.h"

namespace vllm::v1::metrics {

// build_buckets([1,2,5], max_value) (loggers.py:1284-1300): increasing powers
// of ten times the mantissa list, up to and including max_value.
std::vector<double> Build1_2_5Buckets(int64_t max_value);

// THREAD SAFETY (SERVE-METRICS, #277): `PromRegistry` is not thread-safe, and
// upstream never needs it to be — `prometheus_client` mutates under the GIL.
// Once AsyncLLM's output-handler THREAD records (async_llm.py:697-702) while an
// HTTP worker thread scrapes, one recorder and N readers genuinely overlap, so
// every public entry point below takes one mutex. It is a LEAF lock: Record is
// called from outside AsyncLLM's output-processor mutex, so no cycle exists.
class PrometheusStatLogger {
 public:
  // `served_model_name` == vLLM served_model_name (the `model_name` label);
  // `max_model_len` drives the 1-2-5 request-token histogram buckets;
  // `engine_index` labels a single engine (`engine` label, default 0).
  PrometheusStatLogger(std::string served_model_name, int64_t max_model_len,
                       int engine_index = 0);

  // record() (loggers.py:1100-1257): fold one step's scheduler + iteration
  // stats into the registry. Either argument may be default-constructed.
  void Record(const SchedulerStats& scheduler_stats,
              const IterationStats& iteration_stats);

  // cache_config_info (loggers.py) — the Info metric carrying static KV-cache
  // configuration. Both labels are asserted non-empty by the scrape spec.
  void SetCacheConfigInfo(int64_t kv_cache_size_tokens,
                          double kv_cache_max_concurrency);

  // The Prometheus text-0.0.4 exposition for GET /metrics. Safe to call from a
  // scraping thread while the engine's output handler records.
  std::string Expose() const {
    std::lock_guard<std::mutex> lock(mu_);
    return registry_.Expose();
  }

  // Direct registry access for tests/introspection. NOT synchronized — callers
  // must not use it while an engine is recording.
  const PromRegistry& registry() const { return registry_; }

 private:
  // Convenience wrappers that bind the {model_name, engine} label values.
  // Callers hold mu_.
  void Inc(const std::string& name, double v);
  void Set(const std::string& name, double v);
  void Obs(const std::string& name, double v);

  mutable std::mutex mu_;
  PromRegistry registry_;
  std::string model_name_;
  std::string engine_str_;
  std::vector<std::string> labelvalues_;  // {model_name, engine}
};

}  // namespace vllm::v1::metrics

#endif  // VLLM_V1_METRICS_LOGGERS_H_
