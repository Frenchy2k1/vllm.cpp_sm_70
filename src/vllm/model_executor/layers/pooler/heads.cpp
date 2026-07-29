// Ported from:
//   vllm/model_executor/layers/pooler/seqwise/heads.py:59-196
// @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/model_executor/layers/pooler/heads.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace vllm {

namespace {

// Extract row r of a packed [rows, cols] PooledData into a std::vector.
std::vector<float> Row(const PooledData& d, int64_t r) {
  return std::vector<float>(d.data.begin() + r * d.cols,
                            d.data.begin() + (r + 1) * d.cols);
}

// Apply a PoolerActivation to a single row in place (a 1xN PooledData view).
void ApplyActivationRow(const PoolerActivation& act, std::vector<float>& row) {
  PooledData one{row, 1, static_cast<int64_t>(row.size())};
  act.Apply(one);
  row = std::move(one.data);
}

void CheckLen(const PooledData& pooled, const PoolingMetadata& metadata) {
  // heads.py:66 — pooled_data and pooling_params must align 1:1.
  if (static_cast<size_t>(pooled.rows) != metadata.pooling_params.size()) {
    throw std::runtime_error(
        "pooler head: pooled_data length does not match pooling_params length");
  }
}

}  // namespace

// heads.py:59 EmbeddingPoolerHead.forward — projector -> matryoshka -> activation
PoolerOutput EmbeddingPoolerHead::Forward(
    const PooledData& pooled_data, const PoolingMetadata& metadata) const {
  CheckLen(pooled_data, metadata);

  // heads.py:78 — ST projector over the whole batch (optional).
  std::unique_ptr<PooledData> projected;
  const PooledData* emb = &pooled_data;
  if (projector_) {
    projected = std::make_unique<PooledData>(projector_(pooled_data));
    emb = projected.get();
  }

  PoolerOutput out;
  out.reserve(static_cast<size_t>(emb->rows));
  for (int64_t i = 0; i < emb->rows; ++i) {
    std::vector<float> vec = Row(*emb, i);
    // heads.py:86 — matryoshka: truncate to `dimensions` if set for this row.
    const std::optional<int64_t>& dims = metadata.pooling_params[i].dimensions;
    if (dims.has_value()) {
      const int64_t d = std::min(*dims, static_cast<int64_t>(vec.size()));
      vec.resize(static_cast<size_t>(d));
    }
    // heads.py:104 — activation (L2 normalize) when the per-request flag is set.
    if (activation_ && metadata.pooling_params[i].activation_enabled()) {
      ApplyActivationRow(*activation_, vec);
    }
    out.push_back(std::move(vec));
  }
  return out;
}

// heads.py:153 ClassifierPoolerHead.forward — classifier -> affine -> activation
PoolerOutput ClassifierPoolerHead::Forward(
    const PooledData& pooled_data, const PoolingMetadata& metadata) const {
  CheckLen(pooled_data, metadata);

  // heads.py:170 — classifier over the whole batch (optional).
  std::unique_ptr<PooledData> classified;
  const PooledData* logits = &pooled_data;
  if (classifier_) {
    classified = std::make_unique<PooledData>(classifier_(pooled_data));
    logits = classified.get();
  }

  PoolerOutput out;
  out.reserve(static_cast<size_t>(logits->rows));
  for (int64_t i = 0; i < logits->rows; ++i) {
    std::vector<float> vec = Row(*logits, i);
    // heads.py:180 — affine score calibration (logit - mean) / sigma.
    if (logit_mean_.has_value()) {
      const float m = static_cast<float>(*logit_mean_);
      for (float& v : vec) v -= m;
    }
    if (logit_sigma_.has_value()) {
      const float s = static_cast<float>(*logit_sigma_);
      for (float& v : vec) v /= s;
    }
    // heads.py:185 — activation when the per-request flag is set.
    if (activation_ && metadata.pooling_params[i].activation_enabled()) {
      ApplyActivationRow(*activation_, vec);
    }
    out.push_back(std::move(vec));
  }
  return out;
}

}  // namespace vllm
