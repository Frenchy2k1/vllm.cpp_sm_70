// Ported from:
//   vllm/model_executor/layers/pooler/seqwise/poolers.py:70-138
// @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/model_executor/layers/pooler/poolers.h"

#include <algorithm>
#include <iterator>

namespace vllm {

// poolers.py:70 SequencePooler.get_supported_tasks — the intersection of
// POOLING_TASKS, the method's tasks, and the head's tasks.
std::set<PoolingTask> SequencePooler::GetSupportedTasks() const {
  // POOLING_TASKS (tasks.py:18) — the full pooling task universe.
  std::set<PoolingTask> tasks = {
      PoolingTask::kEmbed,      PoolingTask::kClassify,
      PoolingTask::kTokenEmbed, PoolingTask::kTokenClassify,
      PoolingTask::kPlugin,     PoolingTask::kEmbedAndTokenClassify};

  auto intersect = [&tasks](const std::set<PoolingTask>& other) {
    std::set<PoolingTask> out;
    std::set_intersection(tasks.begin(), tasks.end(), other.begin(),
                          other.end(), std::inserter(out, out.begin()));
    tasks = std::move(out);
  };
  intersect(pooling_->GetSupportedTasks());
  intersect(head_->GetSupportedTasks());
  return tasks;
}

// poolers.py:99 pooler_for_embed.
std::unique_ptr<SequencePooler> PoolerForEmbed(
    const PoolerConfig& config, SequencePoolingType default_pooling,
    ProjectorFn projector) {
  auto pooling = GetSeqPoolingMethod(config.GetSeqPoolingType(default_pooling));
  auto head = std::make_unique<EmbeddingPoolerHead>(
      std::move(projector), std::make_shared<PoolerNormalize>());
  return std::make_unique<SequencePooler>(std::move(pooling), std::move(head));
}

// poolers.py:113 pooler_for_classify.
std::unique_ptr<SequencePooler> PoolerForClassify(
    const PoolerConfig& config, SequencePoolingType default_pooling,
    ClassifierFn classifier, std::shared_ptr<PoolerActivation> act_fn) {
  auto pooling = GetSeqPoolingMethod(config.GetSeqPoolingType(default_pooling));
  // resolve_classifier_act_fn: caller override, else the default classify head
  // (sigmoid if num_labels<2 else softmax). The config-driven get_act_fn factory
  // (problem_type / sentence-transformers) is the endpoint-brick residual.
  std::shared_ptr<PoolerActivation> activation =
      act_fn ? std::move(act_fn) : std::make_shared<PoolerClassify>();
  auto head = std::make_unique<ClassifierPoolerHead>(
      std::move(classifier), config.logit_mean, config.logit_sigma,
      std::move(activation));
  return std::make_unique<SequencePooler>(std::move(pooling), std::move(head));
}

}  // namespace vllm
