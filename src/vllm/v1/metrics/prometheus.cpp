// Ported from: the prometheus_client text-exposition contract (see
// include/vllm/v1/metrics/prometheus.h). Text format 0.0.4.
#include "vllm/v1/metrics/prometheus.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace vllm::v1::metrics {

const char* const kContentTypeLatest =
    "text/plain; version=0.0.4; charset=utf-8";

namespace {

// prometheus text format: escape backslash, double-quote and newline in a
// label VALUE (label names and metric names are already safe here).
std::string EscapeLabelValue(const std::string& v) {
  std::string out;
  out.reserve(v.size());
  for (char c : v) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += c;
    }
  }
  return out;
}

// prometheus_client renders floats via Go-style repr; a plain shortest-round
// decimal (or integer when whole) is byte-compatible for a scrape parser.
std::string FormatDouble(double v) {
  if (std::isinf(v)) return v > 0 ? "+Inf" : "-Inf";
  if (std::isnan(v)) return "NaN";
  if (v == std::floor(v) && std::abs(v) < 1e15) {
    // whole number → "N.0" is what prometheus_client emits for counts/sums.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.12g", v);
  return buf;
}

std::string RenderLabels(const std::vector<std::string>& names,
                         const std::vector<std::string>& values,
                         const std::string& extra_name = "",
                         const std::string& extra_value = "") {
  if (names.empty() && extra_name.empty()) return "";
  std::string out = "{";
  bool first = true;
  for (size_t i = 0; i < names.size(); ++i) {
    if (!first) out += ",";
    first = false;
    out += names[i];
    out += "=\"";
    out += EscapeLabelValue(i < values.size() ? values[i] : "");
    out += "\"";
  }
  if (!extra_name.empty()) {
    if (!first) out += ",";
    out += extra_name;
    out += "=\"";
    out += EscapeLabelValue(extra_value);
    out += "\"";
  }
  out += "}";
  return out;
}

}  // namespace

PromRegistry::Family* PromRegistry::Find(const std::string& name) {
  for (auto& f : families_) {
    if (f.name == name) return &f;
  }
  return nullptr;
}

const PromRegistry::Family* PromRegistry::Find(const std::string& name) const {
  for (const auto& f : families_) {
    if (f.name == name) return &f;
  }
  return nullptr;
}

bool PromRegistry::HasFamily(const std::string& name) const {
  return Find(name) != nullptr;
}

void PromRegistry::RegisterCounter(std::string name, std::string help,
                                   std::vector<std::string> labelnames) {
  families_.push_back(Family{std::move(name), std::move(help),
                             MetricType::kCounter, std::move(labelnames),
                             {}, {}});
}

void PromRegistry::RegisterGauge(std::string name, std::string help,
                                 std::vector<std::string> labelnames) {
  families_.push_back(Family{std::move(name), std::move(help),
                             MetricType::kGauge, std::move(labelnames),
                             {}, {}});
}

void PromRegistry::RegisterHistogram(std::string name, std::string help,
                                     std::vector<std::string> labelnames,
                                     std::vector<double> buckets) {
  families_.push_back(Family{std::move(name), std::move(help),
                             MetricType::kHistogram, std::move(labelnames),
                             std::move(buckets), {}});
}

void PromRegistry::RegisterInfo(std::string name, std::string help,
                                std::vector<std::string> labelnames) {
  families_.push_back(Family{std::move(name), std::move(help),
                             MetricType::kInfo, std::move(labelnames), {}, {}});
}

PromRegistry::Series& PromRegistry::SeriesFor(
    Family& fam, const std::vector<std::string>& labelvalues) {
  for (auto& s : fam.series) {
    if (s.labelvalues == labelvalues) return s;
  }
  Series s;
  s.labelvalues = labelvalues;
  if (fam.type == MetricType::kHistogram) {
    s.bucket_counts.assign(fam.buckets.size(), 0);
  }
  fam.series.push_back(std::move(s));
  return fam.series.back();
}

void PromRegistry::Prime(const std::string& name,
                         const std::vector<std::string>& labelvalues) {
  Family* f = Find(name);
  if (f == nullptr) throw std::invalid_argument("unknown metric: " + name);
  SeriesFor(*f, labelvalues);
}

void PromRegistry::IncCounter(const std::string& name,
                              const std::vector<std::string>& labelvalues,
                              double v) {
  Family* f = Find(name);
  if (f == nullptr) throw std::invalid_argument("unknown metric: " + name);
  SeriesFor(*f, labelvalues).value += v;
}

void PromRegistry::SetGauge(const std::string& name,
                            const std::vector<std::string>& labelvalues,
                            double v) {
  Family* f = Find(name);
  if (f == nullptr) throw std::invalid_argument("unknown metric: " + name);
  SeriesFor(*f, labelvalues).value = v;
}

void PromRegistry::Observe(const std::string& name,
                           const std::vector<std::string>& labelvalues,
                           double v) {
  Family* f = Find(name);
  if (f == nullptr) throw std::invalid_argument("unknown metric: " + name);
  Series& s = SeriesFor(*f, labelvalues);
  s.sum += v;
  s.count += 1;
  // Cumulative buckets: increment every bucket whose upper bound >= v.
  for (size_t i = 0; i < f->buckets.size(); ++i) {
    if (v <= f->buckets[i]) s.bucket_counts[i] += 1;
  }
}

void PromRegistry::SetInfo(const std::string& name,
                           const std::vector<std::string>& labelvalues) {
  Family* f = Find(name);
  if (f == nullptr) throw std::invalid_argument("unknown metric: " + name);
  SeriesFor(*f, labelvalues).value = 1.0;
}

std::string PromRegistry::Expose() const {
  std::ostringstream os;
  for (const auto& f : families_) {
    os << "# HELP " << f.name << " " << f.help << "\n";
    const char* type_str = "counter";
    switch (f.type) {
      case MetricType::kCounter:
        type_str = "counter";
        break;
      case MetricType::kGauge:
        type_str = "gauge";
        break;
      case MetricType::kHistogram:
        type_str = "histogram";
        break;
      case MetricType::kInfo:
        type_str = "gauge";  // prometheus_client Info exposes TYPE gauge
        break;
    }
    os << "# TYPE " << f.name << " " << type_str << "\n";

    for (const auto& s : f.series) {
      const std::string labels = RenderLabels(f.labelnames, s.labelvalues);
      switch (f.type) {
        case MetricType::kCounter:
          os << f.name << "_total" << labels << " " << FormatDouble(s.value)
             << "\n";
          break;
        case MetricType::kGauge:
        case MetricType::kInfo:
          os << f.name << labels << " " << FormatDouble(s.value) << "\n";
          break;
        case MetricType::kHistogram: {
          for (size_t i = 0; i < f.buckets.size(); ++i) {
            os << f.name << "_bucket"
               << RenderLabels(f.labelnames, s.labelvalues, "le",
                               FormatDouble(f.buckets[i]))
               << " " << FormatDouble(static_cast<double>(s.bucket_counts[i]))
               << "\n";
          }
          os << f.name << "_bucket"
             << RenderLabels(f.labelnames, s.labelvalues, "le", "+Inf") << " "
             << FormatDouble(static_cast<double>(s.count)) << "\n";
          os << f.name << "_sum" << labels << " " << FormatDouble(s.sum)
             << "\n";
          os << f.name << "_count" << labels << " "
             << FormatDouble(static_cast<double>(s.count)) << "\n";
          break;
        }
      }
    }
  }
  return os.str();
}

}  // namespace vllm::v1::metrics
