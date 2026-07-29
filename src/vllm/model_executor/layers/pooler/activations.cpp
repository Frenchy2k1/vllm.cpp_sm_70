// Ported from:
//   vllm/model_executor/layers/pooler/activations.py:106-158
// @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/model_executor/layers/pooler/activations.h"

#include <algorithm>
#include <cmath>

namespace vllm {

namespace {

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// F.softmax(dim=-1) with the standard max-subtraction for stability.
void SoftmaxRow(float* row, int64_t n) {
  float max_v = row[0];
  for (int64_t i = 1; i < n; ++i) max_v = std::max(max_v, row[i]);
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    row[i] = std::exp(row[i] - max_v);
    sum += row[i];
  }
  for (int64_t i = 0; i < n; ++i) row[i] /= sum;
}

}  // namespace

// activations.py:106 PoolerIdentity
void PoolerIdentity::Apply(PooledData&) const {}

// activations.py:111 PoolerNormalize — F.normalize(p=2, dim=-1),
// denom = max(||x||_2, 1e-12).
void PoolerNormalize::Apply(PooledData& data) const {
  constexpr float kEps = 1e-12f;
  for (int64_t r = 0; r < data.rows; ++r) {
    float* row = data.data.data() + r * data.cols;
    double sq = 0.0;
    for (int64_t c = 0; c < data.cols; ++c) sq += static_cast<double>(row[c]) * row[c];
    const float denom = std::max(static_cast<float>(std::sqrt(sq)), kEps);
    for (int64_t c = 0; c < data.cols; ++c) row[c] /= denom;
  }
}

// activations.py:116 PoolerMultiLabelClassify — F.sigmoid elementwise.
void PoolerMultiLabelClassify::Apply(PooledData& data) const {
  for (float& v : data.data) v = Sigmoid(v);
}

// activations.py:121 PoolerClassify — sigmoid if num_labels < 2, else softmax.
void PoolerClassify::Apply(PooledData& data) const {
  const int64_t num_labels = num_labels_.has_value() ? *num_labels_ : data.cols;
  if (num_labels < 2) {
    for (float& v : data.data) v = Sigmoid(v);
    return;
  }
  for (int64_t r = 0; r < data.rows; ++r) {
    SoftmaxRow(data.data.data() + r * data.cols, data.cols);
  }
}

}  // namespace vllm
