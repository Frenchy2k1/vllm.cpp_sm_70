// Ported from: the Prometheus text-exposition contract that vLLM produces via
// `prometheus_client` (vllm/v1/metrics/prometheus.py + the metric objects in
// vllm/v1/metrics/loggers.py) and consumes at GET /metrics
// (vllm/entrypoints/serve/instrumentator/metrics.py:82 make_asgi_app).
//
// SCOPE (SERVE-METRICS, ROAD-V1-C8): a SELF-CONTAINED Prometheus registry +
// text-format-0.0.4 exposition. vLLM depends on the `prometheus_client` Python
// package; this project has no Python at runtime, so this is a small header +
// .cpp reimplementation of the exact exposition BYTES that a prometheus scrape
// expects (`# HELP`/`# TYPE` lines, counter `_total`, histogram
// `_bucket{le=...}`/`_sum`/`_count`, Info `{labels} 1.0`).
//
// This is a pure transport/format dependency (like cpp-httplib), NOT a
// compute/ML dependency, so it is consistent with the no-pytorch/no-ggml rule.
//
// DEVIATIONS from prometheus_client: we do NOT emit the optional `_created`
// gauge series (prometheus_client can disable them and vLLM's own scrape spec
// `tests/entrypoints/serve/instrumentator/test_metrics.py` never asserts them);
// series ordering is registration order then insertion order (stable, but not
// required by the text format).
#ifndef VLLM_V1_METRICS_PROMETHEUS_H_
#define VLLM_V1_METRICS_PROMETHEUS_H_

#include <deque>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm::v1::metrics {

// The prometheus content type for the text exposition (format version 0.0.4).
// Mirrors prometheus_client.CONTENT_TYPE_LATEST, which vLLM returns via
// PrometheusResponse (instrumentator/metrics.py:60).
extern const char* const kContentTypeLatest;

enum class MetricType { kCounter, kGauge, kHistogram, kInfo };

// A minimal multi-label Prometheus registry. Not thread-safe by itself; the
// PrometheusStatLogger that owns one records under the engine's step cadence.
class PromRegistry {
 public:
  // Register a metric family. `labelnames` is the ordered label key list; every
  // series is addressed by a matching-arity ordered label-value vector.
  void RegisterCounter(std::string name, std::string help,
                       std::vector<std::string> labelnames);
  void RegisterGauge(std::string name, std::string help,
                     std::vector<std::string> labelnames);
  // Histogram buckets are the UPPER bounds (le); a synthetic +Inf bucket is
  // always emitted. Buckets must be ascending.
  void RegisterHistogram(std::string name, std::string help,
                         std::vector<std::string> labelnames,
                         std::vector<double> buckets);
  // An Info metric (prometheus_client Info): a single gauge-valued series fixed
  // at 1.0 whose labels carry the info key/values.
  void RegisterInfo(std::string name, std::string help,
                    std::vector<std::string> labelnames);

  // Instantiate a zero-valued series for `labelvalues` so the family is present
  // in the exposition even before the first observation (mirrors
  // prometheus_client, whose child metrics exist from construction).
  void Prime(const std::string& name,
             const std::vector<std::string>& labelvalues);

  void IncCounter(const std::string& name,
                  const std::vector<std::string>& labelvalues, double v = 1.0);
  void SetGauge(const std::string& name,
                const std::vector<std::string>& labelvalues, double v);
  void Observe(const std::string& name,
               const std::vector<std::string>& labelvalues, double v);
  // Set an Info series' label values to 1.0. `labelvalues` matches labelnames.
  void SetInfo(const std::string& name,
               const std::vector<std::string>& labelvalues);

  // Render the whole registry in Prometheus text format 0.0.4.
  std::string Expose() const;

  // True if a family with this name is registered (test/introspection helper).
  bool HasFamily(const std::string& name) const;

 private:
  struct Series {
    std::vector<std::string> labelvalues;
    double value = 0.0;              // counter/gauge/info
    std::vector<uint64_t> bucket_counts;  // histogram (== buckets.size())
    double sum = 0.0;               // histogram
    uint64_t count = 0;             // histogram
  };
  struct Family {
    std::string name;
    std::string help;
    MetricType type = MetricType::kCounter;
    std::vector<std::string> labelnames;
    std::vector<double> buckets;  // histogram upper bounds (ascending)
    // #330: a DEQUE, not a vector. `SeriesFor` hands out `Series&` and
    // `Find`/`Register*` hand out `Family*`, and callers hold them across later
    // registrations. vector::push_back reallocates and invalidates every one of
    // those, which is the use-after-free TSan reported as a data race inside
    // `operator delete` at prometheus.cpp:134. deque::push_back never
    // invalidates references to existing elements, and it keeps the documented
    // insertion order. Same reason for `families_` below.
    std::deque<Series> series;   // insertion-ordered; refs must stay stable
  };

  Family* Find(const std::string& name);
  const Family* Find(const std::string& name) const;
  Series& SeriesFor(Family& fam, const std::vector<std::string>& labelvalues);

  std::deque<Family> families_;  // see the note on Family::series
};

}  // namespace vllm::v1::metrics

#endif  // VLLM_V1_METRICS_PROMETHEUS_H_
