// Ported from: vllm/v1/worker/gpu/spec_decode/mtp/speculator.py (MTPSpeculator)
// + vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (propose :126-271,
//   _prefill :332-370, sample_draft/_greedy_sample_draft :255-259) @ e24d1b24.
//
// Scope (SPEC-MTP increment I5c, row SPEC-MTP): the k=1 greedy MTP propose — the
// paged draft-prefill forward + argmax draft-token pick — assembled from the
// bricks the earlier increments landed:
//   * prepare_prefill_inputs (I5b) shifts/splices the target verify batch into the
//     draft input_ids + last_token_indices (autoregressive/speculator.py:185-195);
//   * Qwen3_5MTPModel::ForwardPaged (I5c) runs the head + one paged full-attn
//     decoder layer over the DRAFT KV layer using the target's slot mapping
//     (qwen3_5_mtp.py:129-165 over the paged backend, speculator.py:346 _run_model);
//   * the shared lm_head + a per-request argmax over the last_token_indices rows
//     picks the drafted token (_greedy_sample_draft :255-259), and k=1 EARLY-EXITS
//     after this one forward (speculator.py:236-238) — no multi-step decode.
//
// This is a CALLABLE, tested propose brick. It is NOT wired into the runner STEP
// loop (that is I5d): nothing on the production path constructs an MtpProposer or
// calls propose(), so with no SpeculativeConfig this TU's object code is
// unreachable and the engine is byte-identical. The draft KV layer it consumes is
// allocated only when speculative decoding is on (MakeQwen3_5KVCacheSpec num_spec).
#ifndef VLLM_V1_WORKER_GPU_SPEC_DECODE_MTP_SPECULATOR_H_
#define VLLM_V1_WORKER_GPU_SPEC_DECODE_MTP_SPECULATOR_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"      // PagedKvCache
#include "vllm/model_executor/models/qwen3_5_mtp.h"   // Qwen3_5MTPModel
#include "vllm/v1/attention/backend.h"                // CommonAttentionMetadata
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// The k=1 greedy MTP propose (autoregressive/speculator.py:126-271, k=1 branch).
// Runs exactly one paged draft-prefill forward and returns the drafted token id
// per request.
//
//   draft                the MTP draft model (shares the target embed/lm_head)
//   target_attn_meta     the just-completed VERIFY step's full-attn metadata
//                        (query_start_loc / seq_lens / block_table / slot_mapping).
//                        The draft REUSES it unchanged — identical batch shape and
//                        KV layout (speculator.py NOTE :162-167, :222-234).
//   draft_kv             the MTP head's OWN paged K/V layer (index num_hidden_layers)
//   target_hidden        [T,H] bf16 device — the target model's post-final-norm
//                        hidden tap (ForwardDeviceTap output; the drafter's
//                        `hidden_states` input, qwen3_5_mtp.py:129-140)
//   target_input_ids     [T] i32 — the verify step's flat input ids
//   target_positions     [T] i64 — the verify step's flat positions
//   idx_mapping          [num_reqs] i32 — batch_idx -> req_state slot
//   last_sampled         [>= max req_state] i32 — per req_state just-sampled id
//   next_prefill_tokens  [>= max req_state] i32 — per req_state next prefill id
//   num_sampled          [num_reqs] i32 — I3 RejectionSampler output
//   num_rejected         [num_reqs] i32 — I3 RejectionSampler output
//   max_num_reqs         CUDA-graph request-count padding bound (>= num_reqs)
//
// num_reqs is target_attn_meta.num_reqs; T is target_input_ids.size(). Returns
// draft_tokens [num_reqs] (draft_tokens[:num_reqs, :1] flattened for k=1).
std::vector<int32_t> MtpProposePrefill(
    const vllm::Qwen3_5MTPModel& draft,
    const CommonAttentionMetadata& target_attn_meta,
    vllm::PagedKvCache& draft_kv, const vt::Tensor& target_hidden,
    const std::vector<int32_t>& target_input_ids,
    const std::vector<int64_t>& target_positions,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled,
    const std::vector<int32_t>& next_prefill_tokens,
    const std::vector<int32_t>& num_sampled,
    const std::vector<int32_t>& num_rejected, int max_num_reqs,
    vt::Queue& queue);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_SPEC_DECODE_MTP_SPECULATOR_H_
