// Ported from: vllm/v1/outputs.py @ e24d1b24
// See include/vllm/v1/outputs.h for scope + the flat-vector layout.

#include "vllm/v1/outputs.h"

#include <cstddef>

namespace vllm::v1 {

LogprobsTensors LogprobsTensors::empty_cpu(int num_positions,
                                           int num_tokens_per_position) {
  LogprobsTensors out;
  out.num_positions = num_positions;
  out.num_tokens_per_position = num_tokens_per_position;
  const size_t area = static_cast<size_t>(num_positions) *
                      static_cast<size_t>(num_tokens_per_position);
  out.logprob_token_ids.resize(area);
  out.logprobs.resize(area);
  out.selected_token_ranks.resize(static_cast<size_t>(num_positions));
  return out;
}

LogprobsTensors LogprobsTensors::slice_request(int req_idx,
                                               int request_num_positions) const {
  LogprobsTensors out;
  out.num_positions = request_num_positions;
  out.num_tokens_per_position = num_tokens_per_position;
  const size_t w = static_cast<size_t>(num_tokens_per_position);
  const size_t begin = static_cast<size_t>(req_idx);
  const size_t end = begin + static_cast<size_t>(request_num_positions);
  out.logprob_token_ids.assign(logprob_token_ids.begin() + static_cast<std::ptrdiff_t>(begin * w),
                               logprob_token_ids.begin() + static_cast<std::ptrdiff_t>(end * w));
  out.logprobs.assign(logprobs.begin() + static_cast<std::ptrdiff_t>(begin * w),
                      logprobs.begin() + static_cast<std::ptrdiff_t>(end * w));
  out.selected_token_ranks.assign(
      selected_token_ranks.begin() + static_cast<std::ptrdiff_t>(begin),
      selected_token_ranks.begin() + static_cast<std::ptrdiff_t>(end));
  return out;
}

}  // namespace vllm::v1
