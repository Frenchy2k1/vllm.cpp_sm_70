// Ported from vllm/v1/worker/gpu/spec_decode/{mtp,autoregressive}/speculator.py
// @ e24d1b24 — the k=1 greedy MTP propose. See the header for scope + the exact
// upstream anchors. SPEC-MTP increment I5c.
#include "vllm/v1/worker/gpu/spec_decode/mtp/speculator.h"

#include <cstddef>

#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm::v1 {

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
    vt::Queue& queue) {
  const int64_t num_reqs = target_attn_meta.num_reqs;
  const int64_t T = static_cast<int64_t>(target_input_ids.size());
  VT_CHECK(num_reqs > 0, "MtpProposePrefill: empty batch");
  VT_CHECK(static_cast<int64_t>(target_positions.size()) == T,
           "MtpProposePrefill: positions length must equal token count");
  VT_CHECK(target_attn_meta.num_actual_tokens == T,
           "MtpProposePrefill: attn metadata token count must equal T");

  // ── Draft input-prep (I5b): shift-splice the verify batch into the drafter's
  // input_ids + last_token_indices; the draft query_start_loc / seq_lens equal the
  // target's (rejected positions padded, not compacted — speculator.py:185-195). ──
  const SpecPrefillInputs spi = prepare_prefill_inputs(
      target_input_ids, target_positions, target_attn_meta.query_start_loc,
      target_attn_meta.seq_lens, idx_mapping, last_sampled, next_prefill_tokens,
      num_sampled, num_rejected, max_num_reqs);

  // The draft REUSES the target's attention metadata + slot mappings unchanged
  // (identical batch shape + KV layout, speculator.py:222-234). ForwardPaged takes
  // i32 positions (as the target/standalone forwards do); the drafter positions are
  // a copy of the target's, so narrow the I5b i64 positions.
  std::vector<int32_t> positions32(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    positions32[static_cast<size_t>(t)] =
        static_cast<int32_t>(spi.positions[static_cast<size_t>(t)]);

  // ── The one paged draft forward (I5c) + shared lm_head. ──────────────────────
  vllm::Qwen3_5MTPHiddenStates hidden = draft.ForwardPaged(
      spi.input_ids, positions32, target_hidden, target_attn_meta, draft_kv, queue);
  vllm::ForwardLogits logits = draft.ComputeLogits(hidden.tensor, queue);
  VT_CHECK(logits.on_device() && logits.rows == T,
           "MtpProposePrefill: unexpected draft logits shape");

  const int64_t vocab = logits.vocab;
  std::vector<float> host(static_cast<size_t>(T) * static_cast<size_t>(vocab));
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  backend.Copy(queue, host.data(), logits.device_tensor.data,
               host.size() * sizeof(float));
  backend.Synchronize(queue);

  // ── Greedy draft pick (_greedy_sample_draft :255-259): argmax over each request's
  // last (sampled) row. Lowest-index tie-break, matching our sampler's argmax. ──
  std::vector<int32_t> draft_tokens(static_cast<size_t>(num_reqs), 0);
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t row = spi.last_token_indices[static_cast<size_t>(r)];
    VT_CHECK(row >= 0 && row < T,
             "MtpProposePrefill: last_token_index out of range");
    const float* logit_row = host.data() + static_cast<size_t>(row) * vocab;
    int32_t best_idx = 0;
    float best_val = logit_row[0];
    for (int64_t v = 1; v < vocab; ++v) {
      if (logit_row[static_cast<size_t>(v)] > best_val) {
        best_val = logit_row[static_cast<size_t>(v)];
        best_idx = static_cast<int32_t>(v);
      }
    }
    draft_tokens[static_cast<size_t>(r)] = best_idx;
  }
  return draft_tokens;
}

}  // namespace vllm::v1
