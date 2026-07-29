// Ported from:
//   vllm/model_executor/layers/pooler/special.py:57-140
// @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"

#include <stdexcept>
#include <utility>

namespace vllm {

// special.py:57 __init__ — validate that each sub-pooler supports its task.
DispatchPooler::DispatchPooler(
    std::map<PoolingTask, std::unique_ptr<Pooler>> poolers_by_task)
    : poolers_by_task_(std::move(poolers_by_task)) {
  for (const auto& [task, pooler] : poolers_by_task_) {
    const std::set<PoolingTask> supported = pooler->GetSupportedTasks();
    if (supported.find(task) == supported.end()) {
      throw std::runtime_error("DispatchPooler: sub-pooler does not support task '" +
                               ToString(task) + "'");
    }
  }
}

std::unique_ptr<DispatchPooler> DispatchPooler::ForEmbedding(
    const PoolerConfig& config, SequencePoolingType default_pooling,
    ProjectorFn projector) {
  std::map<PoolingTask, std::unique_ptr<Pooler>> poolers;
  poolers.emplace(PoolingTask::kEmbed,
                  PoolerForEmbed(config, default_pooling, std::move(projector)));
  return std::make_unique<DispatchPooler>(std::move(poolers));
}

std::unique_ptr<DispatchPooler> DispatchPooler::ForSeqCls(
    const PoolerConfig& config, SequencePoolingType default_pooling,
    ClassifierFn classifier, std::shared_ptr<PoolerActivation> act_fn) {
  std::map<PoolingTask, std::unique_ptr<Pooler>> poolers;
  poolers.emplace(PoolingTask::kClassify,
                  PoolerForClassify(config, default_pooling,
                                    std::move(classifier), std::move(act_fn)));
  return std::make_unique<DispatchPooler>(std::move(poolers));
}

// special.py:71 get_supported_tasks.
std::set<PoolingTask> DispatchPooler::GetSupportedTasks() const {
  std::set<PoolingTask> tasks;
  for (const auto& [task, pooler] : poolers_by_task_) {
    (void)pooler;
    tasks.insert(task);
  }
  return tasks;
}

// special.py:74 get_pooling_updates.
PoolingParamsUpdate DispatchPooler::GetPoolingUpdates(PoolingTask task) const {
  auto it = poolers_by_task_.find(task);
  if (it == poolers_by_task_.end()) {
    throw std::runtime_error("DispatchPooler: unsupported task '" +
                             ToString(task) + "'");
  }
  return it->second->GetPoolingUpdates(task);
}

// special.py:77 forward — groupby the per-sequence task; run each contiguous
// group through its sub-pooler over the same full hidden-state buffer (absolute
// cursor indices); concatenate outputs in batch order.
PoolerOutput DispatchPooler::Forward(const vt::Tensor& hidden_states,
                                     const PoolingMetadata& metadata) const {
  const std::vector<PoolingTask>& tasks = metadata.tasks;
  PoolerOutput out;
  int64_t offset = 0;
  const int64_t n = static_cast<int64_t>(tasks.size());
  while (offset < n) {
    const PoolingTask task = tasks[offset];
    int64_t count = 1;
    while (offset + count < n && tasks[offset + count] == task) ++count;

    auto it = poolers_by_task_.find(task);
    if (it == poolers_by_task_.end()) {
      throw std::runtime_error("DispatchPooler: unsupported task '" +
                               ToString(task) + "'");
    }
    PoolingMetadata group = metadata.Slice(offset, count);
    PoolerOutput group_out = it->second->Forward(hidden_states, group);
    for (auto& row : group_out) out.push_back(std::move(row));
    offset += count;
  }
  return out;
}

}  // namespace vllm
