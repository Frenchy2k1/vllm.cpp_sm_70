// Ported from: vllm/v1/engine/parallel_sampling.py @ 555967922 (ParentRequest).
//
// Scope (ROAD-V1-C7 `SAMPLE-N`): the parallel-sampling fan-out for a request with
// `n > 1`. A single frontend request with n>1 is expanded by the LLMEngine into n
// CHILD requests that share the same prompt tokens (and, via the existing prefix
// cache, its prefill KV), each with its own decode state + RNG offset. ParentRequest
// holds the shared parent state and (1) hands out per-child request ids + n==1
// child sampling params, and (2) aggregates the n child CompletionOutputs back into
// one RequestOutput carrying n outputs. Mirrors upstream ParentRequest 1:1.
//
// n == 1 (the default) NEVER constructs a ParentRequest — the single-sequence path
// is byte-identical (LLMEngine::add_request keeps its original n==1 branch), exactly
// as upstream llm_engine.py:270-276 (`if n == 1: ... return`).
//
// DEVIATIONS vs upstream (recorded, use OUR names):
//   - __init__ takes the EngineCoreRequest by const ref (upstream reads
//     request.request_id / .external_req_id / .params off it). At T0
//     external_req_id == request_id (see v1/engine/output_processor.h), so we seed
//     external_req_id from request_id when the message carries none.
//   - CompletionOutput has no logprobs-index shuffle here; get_outputs mirrors the
//     upstream aggregation branch-for-branch (streaming pass-through vs FINAL_ONLY
//     collate).
//   - The IterationStats observation helpers (observe_num_generation_tokens /
//     observe_finished_request) are deferred with the rest of parallel-sampling
//     stats — n_params_iter / max_num_generation_tokens_iter are not yet gated.
#ifndef VLLM_V1_ENGINE_PARALLEL_SAMPLING_H_
#define VLLM_V1_ENGINE_PARALLEL_SAMPLING_H_

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/types.h"

namespace vllm::v1 {

// ParentRequest (parallel_sampling.py:13): info, state & processing for a parallel
// sampling request. Stores the parent request id + sampling params, produces child
// request sampling params, and aggregates child completions.
class ParentRequest {
 public:
  // __init__ (parallel_sampling.py:36-50).
  explicit ParentRequest(const EngineCoreRequest& request);

  // get_child_info (parallel_sampling.py:83-94): child request id + n==1 sampling
  // params for the given index within the n children. Registers the child id in
  // child_requests (the set get_outputs drains as children finish).
  std::pair<std::string, SamplingParams> get_child_info(int index);

  // n (parallel_sampling.py:96-98): the requested number of output sequences.
  int n() const { return sampling_params_.n; }

  const std::string& request_id() const { return request_id_; }
  const std::string& external_req_id() const { return external_req_id_; }

  // child_requests is exposed for the OutputProcessor's parent-cleanup check
  // (output_processor.py:720-722: pop the parent once no child remains).
  bool has_children() const { return !child_requests_.empty(); }

  // get_outputs (parallel_sampling.py:100-126): fold one finished/streaming child
  // CompletionOutput into the parent. Returns {outputs, finished}:
  //   - streaming (output_kind != FINAL_ONLY): pass the child output straight
  //     through (empty if a finished child already returned), finished == all
  //     children done.
  //   - FINAL_ONLY: stash the child output at its index; emit the full n-output
  //     list only once every child has finished (else an empty list => the
  //     OutputProcessor suppresses this step's RequestOutput).
  std::pair<std::vector<CompletionOutput>, bool> get_outputs(
      const std::string& child_request_id, CompletionOutput completion_output);

 private:
  // _get_child_sampling_params (parallel_sampling.py:52-81): n==1 clone of the
  // parent params; a unique seed (seed + index) per child when the parent is
  // seeded, else a single cached shared clone.
  SamplingParams GetChildSamplingParams(int index);

  std::string request_id_;
  std::string external_req_id_;
  SamplingParams sampling_params_;

  // To track the completion of child requests (parallel_sampling.py:25).
  std::set<std::string> child_requests_;
  // To aggregate child completions when not streaming (parallel_sampling.py:28).
  // Sized n and index-addressed under FINAL_ONLY; empty under streaming.
  std::vector<std::optional<CompletionOutput>> output_aggregator_;
  // Cached shared child params for the unseeded case (parallel_sampling.py:34).
  std::optional<SamplingParams> cached_child_sampling_params_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ENGINE_PARALLEL_SAMPLING_H_
