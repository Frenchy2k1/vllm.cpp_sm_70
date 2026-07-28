// Ported from: vllm/v1/engine/parallel_sampling.py @ 555967922 (ParentRequest).
// See include/vllm/v1/engine/parallel_sampling.h for scope, wiring and deviations.
#include "vllm/v1/engine/parallel_sampling.h"

#include <utility>

namespace vllm::v1 {

ParentRequest::ParentRequest(const EngineCoreRequest& request)
    : request_id_(request.request_id),
      // external_req_id == request_id at T0 (see output_processor.h); upstream
      // asserts request.external_req_id is not None (parallel_sampling.py:37).
      external_req_id_(request.request_id),
      sampling_params_(request.sampling_params) {
  // parallel_sampling.py:44-48: the FINAL_ONLY aggregator is pre-sized to n (each
  // slot filled at the child's index); streaming leaves it empty.
  if (sampling_params_.output_kind == RequestOutputKind::kFinalOnly) {
    output_aggregator_.resize(static_cast<size_t>(sampling_params_.n));
  }
}

SamplingParams ParentRequest::GetChildSamplingParams(int index) {
  // parallel_sampling.py:52-81.
  const std::optional<int64_t> seed = sampling_params_.seed;
  if (cached_child_sampling_params_.has_value()) {
    // Reuse the cached (unseeded) child params.
    return *cached_child_sampling_params_;
  }
  // Build child sampling_params: a copy with n forced to 1.
  SamplingParams child = sampling_params_;
  child.n = 1;
  if (!seed.has_value()) {
    // Cache the shared child params for later reuse (all children identical).
    cached_child_sampling_params_ = child;
  } else {
    // Each child gets a unique seed (seed + index) so seeded runs diverge.
    child.seed = *seed + index;
  }
  return child;
}

std::pair<std::string, SamplingParams> ParentRequest::get_child_info(int index) {
  // parallel_sampling.py:83-94: child id is "{index}_{parent_request_id}".
  std::string child_req_id =
      std::to_string(index) + "_" + request_id_;
  child_requests_.insert(child_req_id);
  return {std::move(child_req_id), GetChildSamplingParams(index)};
}

std::pair<std::vector<CompletionOutput>, bool> ParentRequest::get_outputs(
    const std::string& child_request_id, CompletionOutput completion_output) {
  // parallel_sampling.py:100-126.
  bool already_finished_and_returned = false;
  if (completion_output.Finished()) {
    auto it = child_requests_.find(child_request_id);
    if (it != child_requests_.end()) {
      child_requests_.erase(it);
    } else {
      // The child had finished in a previous step and was already returned to
      // the client; do not emit it twice.
      already_finished_and_returned = true;
    }
  }

  std::vector<CompletionOutput> outputs;
  if (sampling_params_.output_kind != RequestOutputKind::kFinalOnly) {
    // Streaming: pass the current output through (unless a finished child was
    // already returned).
    if (!already_finished_and_returned) {
      outputs.push_back(std::move(completion_output));
    }
  } else {
    // Not streaming: stash at the child's index and only emit once every child
    // has finished (child_requests_ drained).
    const size_t idx = static_cast<size_t>(completion_output.index);
    output_aggregator_[idx] = std::move(completion_output);
    if (child_requests_.empty()) {
      outputs.reserve(output_aggregator_.size());
      for (std::optional<CompletionOutput>& slot : output_aggregator_) {
        // Every slot is filled by the time the last child finishes.
        outputs.push_back(std::move(*slot));
      }
    }
  }

  const bool finished = child_requests_.empty();
  return {std::move(outputs), finished};
}

}  // namespace vllm::v1
