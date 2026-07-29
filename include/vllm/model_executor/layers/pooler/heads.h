// Ported from:
//   vllm/model_executor/layers/pooler/seqwise/heads.py:19-196
//     (SequencePoolerHead, EmbeddingPoolerHead, ClassifierPoolerHead)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING): the sequence pooler HEADS — the postprocess that
// turns a pooled [batch, hidden] buffer into the final embedding / logit rows.
// `EmbeddingPoolerHead` applies the optional ST projector, the matryoshka
// dimension slice, then the optional (L2-normalize) activation.
// `ClassifierPoolerHead` applies the optional classifier, the affine
// `(logit-mean)/sigma` calibration, then the optional activation.
//
// RETURN-SHAPE DEVIATION: upstream returns a stacked `torch.Tensor` when all
// per-request flags (dimensions / use_activation) agree and a `list[Tensor]`
// when they differ. We always return the per-sequence list form (PoolerOutput);
// the uniform-vs-list split is a torch return-type nuance with no numeric effect
// on the pooled values, and every value assertion in the ported test holds.
//
// Deferred (documented): `head_dtype` is a no-op on our float32 CPU path
// (upstream casts the pooled buffer to the head dtype); the ST projector /
// classifier weights are the caller-supplied ProjectorFn / ClassifierFn seam
// (the `_load_st_projector` loader is the W3/W4 model-loader residual).
#pragma once

#include <memory>
#include <set>

#include "vllm/model_executor/layers/pooler/activations.h"
#include "vllm/model_executor/layers/pooler/common.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"

namespace vllm {

// heads.py:19 SequencePoolerHead.
class SequencePoolerHead {
 public:
  virtual ~SequencePoolerHead() = default;
  virtual std::set<PoolingTask> GetSupportedTasks() const = 0;
  virtual PoolerOutput Forward(const PooledData& pooled_data,
                               const PoolingMetadata& metadata) const = 0;
};

// heads.py:32 EmbeddingPoolerHead — projector -> matryoshka -> activation.
class EmbeddingPoolerHead : public SequencePoolerHead {
 public:
  EmbeddingPoolerHead() = default;
  EmbeddingPoolerHead(ProjectorFn projector,
                      std::shared_ptr<PoolerActivation> activation)
      : projector_(std::move(projector)),
        activation_(std::move(activation)) {}

  // heads.py:57 get_supported_tasks -> {"embed"}.
  std::set<PoolingTask> GetSupportedTasks() const override {
    return {PoolingTask::kEmbed};
  }

  PoolerOutput Forward(const PooledData& pooled_data,
                       const PoolingMetadata& metadata) const override;

 private:
  ProjectorFn projector_;                        // heads.py:41 (optional)
  std::shared_ptr<PoolerActivation> activation_;  // heads.py:43 (optional)
};

// heads.py:123 ClassifierPoolerHead — classifier -> affine calibration ->
// activation.
class ClassifierPoolerHead : public SequencePoolerHead {
 public:
  ClassifierPoolerHead() = default;
  ClassifierPoolerHead(ClassifierFn classifier, std::optional<double> logit_mean,
                       std::optional<double> logit_sigma,
                       std::shared_ptr<PoolerActivation> activation)
      : classifier_(std::move(classifier)),
        logit_mean_(logit_mean),
        logit_sigma_(logit_sigma),
        activation_(std::move(activation)) {}

  // heads.py:172 get_supported_tasks -> {"classify"}.
  std::set<PoolingTask> GetSupportedTasks() const override {
    return {PoolingTask::kClassify};
  }

  PoolerOutput Forward(const PooledData& pooled_data,
                       const PoolingMetadata& metadata) const override;

 private:
  ClassifierFn classifier_;                       // heads.py:133 (optional)
  std::optional<double> logit_mean_;              // heads.py:134
  std::optional<double> logit_sigma_;             // heads.py:135
  std::shared_ptr<PoolerActivation> activation_;  // heads.py:137 (optional)
};

}  // namespace vllm
