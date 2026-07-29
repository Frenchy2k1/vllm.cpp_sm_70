// Ported from:
//   vllm/model_executor/layers/pooler/activations.py:84-158
//     (PoolerActivation, PoolerIdentity, PoolerNormalize,
//      PoolerMultiLabelClassify, PoolerClassify)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W1 brick (CLAIM-POOLING): the pooler activation heads. `PoolerNormalize`
// is the L2 normalize that the embedding head applies by default; the classify
// activations (sigmoid / softmax) are the cross-encoder / classification heads.
// Each operates row-wise (dim=-1) on a [batch, dim] PooledData buffer, in place.
//
// Deferred to W2+: the config-driven factory `get_act_fn` /
// `resolve_classifier_act_fn` (activations.py:19-73) that selects the head from
// `problem_type` / sentence-transformers config, and `LambdaPoolerActivation`
// wrapping an arbitrary torch module — both need the model/pooler config that
// lands with the runner brick.
#pragma once

#include <cstdint>
#include <optional>

#include "vllm/model_executor/layers/pooler/methods.h"

namespace vllm {

// activations.py:84 PoolerActivation — row-wise activation over [batch, dim].
class PoolerActivation {
 public:
  virtual ~PoolerActivation() = default;
  virtual void Apply(PooledData& data) const = 0;
};

// activations.py:106 PoolerIdentity — returns the input unchanged.
class PoolerIdentity : public PoolerActivation {
 public:
  void Apply(PooledData& data) const override;
};

// activations.py:111 PoolerNormalize — F.normalize(x, p=2, dim=-1). Denominator
// is max(||x||_2, eps) with torch's default eps=1e-12.
class PoolerNormalize : public PoolerActivation {
 public:
  void Apply(PooledData& data) const override;
};

// activations.py:116 PoolerMultiLabelClassify — F.sigmoid, elementwise.
class PoolerMultiLabelClassify : public PoolerActivation {
 public:
  void Apply(PooledData& data) const override;
};

// activations.py:121 PoolerClassify — sigmoid when num_labels < 2 (or the row
// width when num_labels is unset), softmax(dim=-1) otherwise. Mirrors the
// transformers ForSequenceClassificationLoss alignment.
class PoolerClassify : public PoolerActivation {
 public:
  explicit PoolerClassify(std::optional<int64_t> num_labels = std::nullopt)
      : num_labels_(num_labels) {}

  std::optional<int64_t> num_labels() const { return num_labels_; }
  void Apply(PooledData& data) const override;

 private:
  std::optional<int64_t> num_labels_;
};

}  // namespace vllm
