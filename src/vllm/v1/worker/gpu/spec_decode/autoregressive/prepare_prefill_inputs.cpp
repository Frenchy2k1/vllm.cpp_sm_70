// Ported from: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
// `_prepare_prefill_inputs_kernel` (:469-549) + `prepare_prefill_inputs`
// (:552-588) @ e24d1b24. See the header for the full shift-splice rule and the
// device-neutrality note; this is the CPU reference the DGX runner leaf (I5d)
// ports 1:1 to the cited Triton kernel over the GPU-resident draft buffers.
#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vllm::v1 {

SpecPrefillInputs prepare_prefill_inputs(
    const std::vector<int32_t>& target_input_ids,
    const std::vector<int64_t>& target_positions,
    const std::vector<int32_t>& target_query_start_loc,
    const std::vector<int32_t>& target_seq_lens,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled,
    const std::vector<int32_t>& next_prefill_tokens,
    const std::vector<int32_t>& num_sampled,
    const std::vector<int32_t>& num_rejected, int max_num_reqs) {
  assert(!target_query_start_loc.empty() && "query_start_loc has num_reqs+1 entries");
  const int num_reqs = static_cast<int>(target_query_start_loc.size()) - 1;
  assert(num_reqs >= 0);
  assert(max_num_reqs >= num_reqs && "CUDA-graph pad width must cover num_reqs");
  const int32_t num_tokens = target_query_start_loc.back();
  assert(static_cast<int32_t>(target_input_ids.size()) == num_tokens);
  assert(static_cast<int32_t>(target_positions.size()) == num_tokens);
  assert(static_cast<int>(target_seq_lens.size()) == num_reqs);
  assert(static_cast<int>(idx_mapping.size()) == num_reqs);
  assert(static_cast<int>(num_sampled.size()) == num_reqs);
  assert(static_cast<int>(num_rejected.size()) == num_reqs);

  SpecPrefillInputs out;
  // input_ids / positions keep the target batch's total size (rejected positions
  // are padded, not compacted — speculator.py NOTE :162-167). Initialize with the
  // target's values so the untouched rejected tail is a deterministic don't-care;
  // positions are then copied verbatim below (never shifted).
  out.input_ids = target_input_ids;
  out.positions = target_positions;
  out.query_start_loc.assign(static_cast<size_t>(max_num_reqs) + 1, num_tokens);
  out.seq_lens.assign(static_cast<size_t>(max_num_reqs), 0);
  out.last_token_indices.assign(static_cast<size_t>(max_num_reqs), 0);
  out.current_draft_step = 0;  // reset (speculator.py:534)

  for (int req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int req_state_idx = idx_mapping[req_idx];
    const int32_t query_start = target_query_start_loc[req_idx];
    const int32_t query_end = target_query_start_loc[req_idx + 1];
    // True query length after accounting for rejected tokens (:495,:500).
    int32_t query_len = (query_end - query_start) - num_rejected[req_idx];
    assert(query_len >= 1 && "a sampling/prefill row keeps at least one token");

    // Next token spliced into the freed last slot (:502-508).
    int32_t next_token;
    if (num_sampled[req_idx] > 0) {
      next_token = last_sampled[static_cast<size_t>(req_state_idx)];
    } else {
      // Chunked prefilling — take the next prefill token.
      next_token = next_prefill_tokens[static_cast<size_t>(req_state_idx)];
    }

    // Shift target_input_ids left by one within [query_start, query_start+query_len)
    // (:511-515). draft[query_start + i - 1] = target[query_start + i].
    for (int32_t i = 1; i < query_len; ++i) {
      out.input_ids[static_cast<size_t>(query_start) + i - 1] =
          target_input_ids[static_cast<size_t>(query_start) + i];
    }
    const int64_t last_token_index = static_cast<int64_t>(query_start) + query_len - 1;
    out.last_token_indices[static_cast<size_t>(req_idx)] = last_token_index;
    out.input_ids[static_cast<size_t>(last_token_index)] = next_token;  // :519

    // Positions are copied UNCHANGED over [query_start, query_start+query_len)
    // (:522-526). out.positions was seeded from target_positions, so this range
    // already holds the correct values; the copy is explicit for fidelity and to
    // make the don't-care semantics of the tail unmistakable.
    for (int32_t i = 0; i < query_len; ++i) {
      out.positions[static_cast<size_t>(query_start) + i] =
          target_positions[static_cast<size_t>(query_start) + i];
    }

    out.query_start_loc[static_cast<size_t>(req_idx)] = query_start;  // :529
    out.seq_lens[static_cast<size_t>(req_idx)] = target_seq_lens[req_idx];  // :531
  }

  // CUDA-graph request-count padding (:532-549): query_start_loc past num_reqs is
  // the total token count (query_end of the last request); seq_lens and
  // last_token_indices past num_reqs stay 0 (already assigned). query_start_loc[
  // num_reqs] must be the total too (it is num_tokens by the assign above).
  // (out.query_start_loc / seq_lens / last_token_indices were assigned to their
  //  padded defaults up front, so no extra work is needed here — documented for
  //  the 1:1 line correspondence.)
  return out;
}

}  // namespace vllm::v1
