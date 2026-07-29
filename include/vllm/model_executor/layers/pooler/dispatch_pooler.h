// Ported from:
//   vllm/model_executor/layers/pooler/special.py:23-140 (DispatchPooler)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING): the task-routing pooler. A pooling model owns a
// single `DispatchPooler` that holds one sub-pooler per task; `forward` groups
// the batch by the per-sequence task and hands each contiguous group to its
// sub-pooler. `ForEmbedding` / `ForSeqCls` wire the embedding / sequence-classify
// task maps.
//
// SLICING DEVIATION (recorded): upstream slices `hidden_states` per task group
// and shifts the cursor's first/last token indices by the running token offset.
// Our sub-poolers index the full packed [num_tokens, hidden] buffer with the
// ORIGINAL absolute token indices, so we pass the full buffer plus a
// seq-subranged PoolingMetadata (`PoolingMetadata::Slice`) — value-identical,
// no index shift needed.
//
// Deferred (documented): upstream `ForEmbedding` also registers `token_embed`
// and `ForSeqCls` also registers `token_classify` (the tokwise poolers) — those
// are the W5 residual, so our task maps carry only the seqwise `embed` /
// `classify` entries.
#pragma once

#include <map>
#include <memory>
#include <set>

#include "vllm/model_executor/layers/pooler/common.h"
#include "vllm/model_executor/layers/pooler/methods.h"
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/layers/pooler/poolers.h"
#include "vt/tensor.h"

namespace vllm {

// special.py:23 DispatchPooler.
class DispatchPooler : public Pooler {
 public:
  explicit DispatchPooler(
      std::map<PoolingTask, std::unique_ptr<Pooler>> poolers_by_task);

  // special.py:28 for_embedding — {embed: pooler_for_embed}. (token_embed is the
  // W5 tokwise residual.)
  static std::unique_ptr<DispatchPooler> ForEmbedding(
      const PoolerConfig& config,
      SequencePoolingType default_pooling = SequencePoolingType::kLast,
      ProjectorFn projector = nullptr);

  // special.py:37 for_seq_cls — {classify: pooler_for_classify}. (token_classify
  // is the W5 tokwise residual.)
  static std::unique_ptr<DispatchPooler> ForSeqCls(
      const PoolerConfig& config,
      SequencePoolingType default_pooling = SequencePoolingType::kLast,
      ClassifierFn classifier = nullptr,
      std::shared_ptr<PoolerActivation> act_fn = nullptr);

  // special.py:71 get_supported_tasks — the registered task keys.
  std::set<PoolingTask> GetSupportedTasks() const override;

  // special.py:74 get_pooling_updates — delegate to the task's sub-pooler.
  PoolingParamsUpdate GetPoolingUpdates(PoolingTask task) const override;

  // special.py:77 forward — groupby task, route each group to its sub-pooler,
  // concatenate the per-sequence outputs in batch order.
  PoolerOutput Forward(const vt::Tensor& hidden_states,
                       const PoolingMetadata& metadata) const override;

 private:
  std::map<PoolingTask, std::unique_ptr<Pooler>> poolers_by_task_;
};

}  // namespace vllm
