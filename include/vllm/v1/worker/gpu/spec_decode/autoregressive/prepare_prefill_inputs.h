// Ported from: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
// (`_prepare_prefill_inputs_kernel` :469-549 + `prepare_prefill_inputs` wrapper
// :552-588) @ e24d1b24, driven from `AutoRegressiveSpeculator.propose`
// (:184-195) and the runner spec input path `vllm/v1/worker/gpu/model_runner.py`
// (the `if not draft_tokens:` / else split at :866-898, landed for the VERIFY
// half by SPEC-MTP I3).
//
// Scope (SPEC-MTP increment I5b, row SPEC-MTP): the DRAFTER input-prep for the
// prefill (k=1 / first draft) step. Given the just-completed VERIFY step's
// target batch (the flattened input_ids / positions / query_start_loc / seq_lens
// our prepare_inputs already builds) plus the per-request rejection outcome
// (num_sampled / num_rejected from I3's RejectionSampler, and the request's
// just-sampled next token), produce the draft model's input for the next
// forward: each request's input_ids shifted LEFT by one within its query span
// with the just-sampled next token spliced into the freed last slot, the query
// length reduced by num_rejected, and the last-token index / query_start_loc /
// seq_lens the draft forward + sampler consume.
//
// ─── THE SHIFT-SPLICE, EXACTLY (per request r) ─────────────────────────────
// Let the target verify span be [query_start, query_end) with
// query_len0 = query_end - query_start (== 1 + k_r for a decoding request), the
// verify tokens t_0..t_{query_len0-1} at positions p_0..p_{query_len0-1}, and
// nr = num_rejected[r] in [0, k_r]. Mirrors speculator.py:493-531:
//
//   query_len = query_len0 - nr                              # :495,:500
//   next_token = num_sampled[r] > 0 ? last_sampled[idx_mapping[r]]   # :502-504
//                                   : next_prefill_tokens[idx_mapping[r]]  # :508
//   for i in [1, query_len):                                 # :511-515 (shift L1)
//     draft_input_ids[query_start + i - 1] = target_input_ids[query_start + i]
//   last_token_index = query_start + query_len - 1           # :517
//   draft_input_ids[last_token_index] = next_token           # :519
//   for i in [0, query_len):                                 # :522-526 (pos copy)
//     draft_positions[query_start + i] = target_positions[query_start + i]
//   draft_query_start_loc[r] = query_start                   # :529
//   draft_seq_lens[r]        = seq_len (target's, UNCHANGED) # :531
//
// So the draft span [query_start, query_start + query_len) becomes
// [t_1, t_2, ..., t_{query_len-1}, next_token]: the first verify token is
// dropped, the rest shift left one, and the request's just-sampled next token is
// spliced into the freed last slot — the drafter consumes [prev tokens shifted,
// last sampled] to predict the next. POSITIONS are copied UNCHANGED (p_0..) —
// only the token IDs shift — so the draft consumes token t_{i+1} at position
// p_i, the MTP shift the head expects.
//
// KEY INVARIANT (speculator.py NOTE :162-167): input_ids / positions keep the
// TARGET batch's total size — rejected positions are PADDED, not compacted — so
// the draft reuses the target's attention metadata (query_start_loc, seq_lens,
// slot mappings) unchanged. query_start_loc is therefore the target's
// (draft_query_start_loc[r] = target query_start), NOT reduced by nr; only the
// per-request shift range, last_token_index, and the sampled logits row shrink
// by nr. The [query_start + query_len, query_end) tail is DON'T-CARE padding —
// the draft forward attends over it but its rows are never sampled (only
// last_token_indices[r] is), and the next step overwrites it. We fill that tail
// deterministically with the target token/position so the routine is a pure
// function of its inputs (upstream leaves stale buffer content there).
//
// CUDA-graph request-count padding (:532-549, run once for the last request):
// current_draft_step is reset to 0; draft_query_start_loc is padded to the total
// token count for [num_reqs, max_num_reqs+1); draft_seq_lens and
// last_token_indices are zero-padded for [num_reqs, max_num_reqs).
//
// k=1 EARLY-EXIT (speculator.py:236-238): for num_speculative_tokens == 1 the
// drafter runs exactly this one prefill forward and returns draft_tokens[:, :1]
// — no prepare_decode_inputs / multi-step decode. This routine is the whole
// input-prep for that (the tested) path; k>1 adds prepare_decode_inputs
// (:591-665), DEFERRED to M-mtp-3.
//
// DEVICE-NEUTRAL (recorded, exactly as prepare_inputs / combine_sampled_and_
// draft_tokens): these are host std::vectors here — the shift/splice is pure
// integer index logic over the input batch. The DGX runner leaf (I5d) ports this
// same loop to the cited Triton kernel over the GPU-resident draft input_buffers
// so no id round-trips the host; the arithmetic is identical, so the CPU routine
// is the reference. No new CUDA kernel is added by I5b (nothing calls this yet;
// I5d wires the runner loop).
//
// DEFAULT-OFF / INERT: nothing on the production path calls this. With no
// SpeculativeConfig the scheduler never populates scheduled_spec_decode_tokens,
// the speculator is never constructed, and this TU's object code is unreachable.
#ifndef VLLM_V1_WORKER_GPU_SPEC_DECODE_AUTOREGRESSIVE_PREPARE_PREFILL_INPUTS_H_
#define VLLM_V1_WORKER_GPU_SPEC_DECODE_AUTOREGRESSIVE_PREPARE_PREFILL_INPUTS_H_

#include <cstdint>
#include <vector>

namespace vllm::v1 {

// The draft model's prefill inputs, mirroring the buffers
// `_prepare_prefill_inputs_kernel` writes (speculator.py:472-476 outputs +
// last_token_indices + current_draft_step). Sizes: input_ids / positions are the
// TARGET batch's token count (num_tokens); the request-indexed arrays are padded
// to max_num_reqs for CUDA-graph replay. Consumed by the I5d runner loop, which
// feeds input_ids / positions to the draft forward, query_start_loc / seq_lens
// to the draft attention metadata builder, and last_token_indices to the draft
// sampler (speculator.py:342-343 gathers positions / hidden at last_token_indices).
struct SpecPrefillInputs {
  // [num_tokens] draft input ids: each request's span shifted left one with the
  // sampled next token spliced at the last (query_len-1) slot; the rejected tail
  // holds the target token (don't-care padding).
  std::vector<int32_t> input_ids;
  // [num_tokens] draft positions: a copy of the target positions (unchanged).
  std::vector<int64_t> positions;
  // [max_num_reqs + 1] draft_query_start_loc[r] = target query_start_loc[r];
  // padded with the total token count for [num_reqs, max_num_reqs + 1).
  std::vector<int32_t> query_start_loc;
  // [max_num_reqs] draft_seq_lens[r] = target seq_lens[r]; zero-padded past
  // num_reqs.
  std::vector<int32_t> seq_lens;
  // [max_num_reqs] the last (sampled) row of each request's draft span =
  // query_start + query_len - 1; zero-padded past num_reqs.
  std::vector<int64_t> last_token_indices;
  // Reset to 0 at the start of the draft prefill (speculator.py:534).
  int64_t current_draft_step = 0;
};

// Build the drafter's prefill inputs. Pure function of the target verify batch +
// the per-request rejection outcome; see the file header for the exact rule.
//
//   target_input_ids       [num_tokens]     i32 — the verify step's flat inputs
//   target_positions       [num_tokens]     i64 — the verify step's flat positions
//   target_query_start_loc [num_reqs + 1]   i32 — cumulative token offsets
//   target_seq_lens        [num_reqs]       i32 — per-request sequence lengths
//   idx_mapping            [num_reqs]       i32 — batch_idx -> req_state slot
//   last_sampled           [>= max req_state] i32 — per req_state just-sampled id
//   next_prefill_tokens    [>= max req_state] i32 — per req_state next prefill id
//   num_sampled            [num_reqs]       i32 — I3 RejectionSampler output
//   num_rejected           [num_reqs]       i32 — I3 RejectionSampler output
//   max_num_reqs                            — CUDA-graph request-count padding
//
// num_reqs is target_query_start_loc.size() - 1; num_tokens is
// target_query_start_loc.back() (== target_input_ids.size()). Requires
// max_num_reqs >= num_reqs.
SpecPrefillInputs prepare_prefill_inputs(
    const std::vector<int32_t>& target_input_ids,
    const std::vector<int64_t>& target_positions,
    const std::vector<int32_t>& target_query_start_loc,
    const std::vector<int32_t>& target_seq_lens,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled,
    const std::vector<int32_t>& next_prefill_tokens,
    const std::vector<int32_t>& num_sampled,
    const std::vector<int32_t>& num_rejected, int max_num_reqs);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_SPEC_DECODE_AUTOREGRESSIVE_PREPARE_PREFILL_INPUTS_H_
