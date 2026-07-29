// Ported from:
//   vllm/model_executor/layers/pooler/abstract.py:16-36 (Pooler interface)
//   vllm/model_executor/layers/pooler/seqwise/poolers.py:41-138
//     (SequencePooler, pooler_for_embed, pooler_for_classify)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING): the `Pooler` interface and the `SequencePooler`
// composite — extract a pooled vector with a `SequencePoolingMethod`, then
// postprocess it with a `SequencePoolerHead`. `PoolerForEmbed` /
// `PoolerForClassify` mirror the factories that wire the model's pooler from a
// `PoolerConfig`.
//
// Deferred (documented): `pooler_for_embed` upstream also threads the ST
// projector via `_load_st_projector(model_config)` and `head_dtype`; both are
// the W3/W4 model-loader residual (the projector is the caller-supplied
// ProjectorFn seam here). `resolve_classifier_act_fn`'s config-driven
// `get_act_fn` factory (problem_type / sentence-transformers) needs the HF
// config and is the endpoint-brick residual — the classifier activation is
// caller-supplied (default PoolerClassify).
#pragma once

#include <memory>
#include <set>

#include "vllm/model_executor/layers/pooler/common.h"
#include "vllm/model_executor/layers/pooler/heads.h"
#include "vllm/model_executor/layers/pooler/methods.h"
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vt/tensor.h"

namespace vllm {

// abstract.py:16 Pooler — the interface every model-level pooler implements.
class Pooler {
 public:
  virtual ~Pooler() = default;
  virtual std::set<PoolingTask> GetSupportedTasks() const = 0;
  virtual PoolingParamsUpdate GetPoolingUpdates(PoolingTask task) const {
    (void)task;
    return PoolingParamsUpdate{};
  }
  virtual PoolerOutput Forward(const vt::Tensor& hidden_states,
                               const PoolingMetadata& metadata) const = 0;
};

// poolers.py:41 SequencePooler — pooling method then head.
class SequencePooler : public Pooler {
 public:
  SequencePooler(std::unique_ptr<SequencePoolingMethod> pooling,
                 std::unique_ptr<SequencePoolerHead> head)
      : pooling_(std::move(pooling)), head_(std::move(head)) {}

  // poolers.py:72 — POOLING_TASKS ∩ method.supported ∩ head.supported.
  std::set<PoolingTask> GetSupportedTasks() const override;

  // poolers.py:82 — delegate to the pooling method.
  PoolingParamsUpdate GetPoolingUpdates(PoolingTask task) const override {
    return pooling_->GetPoolingUpdates(task);
  }

  // poolers.py:89 — pooled = pooling(hidden); return head(pooled).
  PoolerOutput Forward(const vt::Tensor& hidden_states,
                       const PoolingMetadata& metadata) const override {
    PooledData pooled = pooling_->Forward(hidden_states, metadata);
    return head_->Forward(pooled, metadata);
  }

 private:
  std::unique_ptr<SequencePoolingMethod> pooling_;
  std::unique_ptr<SequencePoolerHead> head_;
};

// poolers.py:99 pooler_for_embed — LAST/CLS/MEAN + EmbeddingPoolerHead(normalize).
// `default_pooling` is the model's architectural default when the config leaves
// `seq_pooling_type` unset (upstream reads it from the config; the model passes
// its default). `projector` is the optional ST projector seam.
std::unique_ptr<SequencePooler> PoolerForEmbed(
    const PoolerConfig& config,
    SequencePoolingType default_pooling = SequencePoolingType::kLast,
    ProjectorFn projector = nullptr);

// poolers.py:113 pooler_for_classify — pooling + ClassifierPoolerHead(classifier,
// logit_mean/sigma, activation). `classifier` and `act_fn` are the caller seams
// (the config-driven get_act_fn factory is the endpoint-brick residual).
std::unique_ptr<SequencePooler> PoolerForClassify(
    const PoolerConfig& config,
    SequencePoolingType default_pooling = SequencePoolingType::kLast,
    ClassifierFn classifier = nullptr,
    std::shared_ptr<PoolerActivation> act_fn = nullptr);

}  // namespace vllm
