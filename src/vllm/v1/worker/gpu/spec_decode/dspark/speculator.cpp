// Ported from vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
// (_sample_sequential :100-149, _generate_draft :151-169) @ 555967922.
// See the header for scope and the exact upstream anchors. SPEC-DSPARK W4.
#include "vllm/v1/worker/gpu/spec_decode/dspark/speculator.h"

#include <cstddef>

#include "vt/backend.h"

namespace vllm::v1 {

std::vector<std::vector<int32_t>> SampleDsparkBlockDrafts(
    const std::vector<float>& block_logits, const std::vector<int32_t>& anchor_ids,
    const DsparkBlockLayout& layout, const Qwen3DSparkWeights& weights,
    vt::Queue& queue) {
  const int num_reqs = static_cast<int>(anchor_ids.size());
  const int n_spec = layout.num_speculative_steps;
  const int nqpr = layout.num_query_per_req();
  const int sample_off = layout.first_sample_offset();
  const int64_t draft_vocab = weights.draft_vocab_size;
  VT_CHECK(num_reqs > 0 && n_spec > 0 && draft_vocab > 0,
           "SampleDsparkBlockDrafts: bad shape");
  const int64_t rows = static_cast<int64_t>(num_reqs) * nqpr;
  VT_CHECK(static_cast<int64_t>(block_logits.size()) == rows * draft_vocab,
           "SampleDsparkBlockDrafts: block_logits must be "
           "[num_reqs*num_query_per_req, draft_vocab]");
  for (int32_t id : anchor_ids) {
    VT_CHECK(id >= 0 && static_cast<int64_t>(id) < weights.vocab_size,
             "SampleDsparkBlockDrafts: anchor token id outside the target vocab");
  }

  std::vector<std::vector<int32_t>> drafts(static_cast<size_t>(num_reqs));
  for (auto& row : drafts) row.assign(static_cast<size_t>(n_spec), 0);

  // `prev` starts at the anchor (bonus/verified) token of each request and is
  // then the token this loop just sampled — the sequential dependency DSpark
  // adds on top of the parallel block draft (speculator.py:120-121,148).
  std::vector<int32_t> prev = anchor_ids;

  for (int i = 0; i < n_spec; ++i) {
    // One batched bias GEMV per step: markov_w2(markov_w1[prev]) over all
    // requests at once, exactly as upstream shapes it (:110-113).
    const std::vector<float> bias =
        Qwen3DSparkModel::MarkovBiasForTokens(prev, weights, queue);
    VT_CHECK(static_cast<int64_t>(bias.size()) ==
                 static_cast<int64_t>(num_reqs) * draft_vocab,
             "SampleDsparkBlockDrafts: Markov bias shape");

    for (int r = 0; r < num_reqs; ++r) {
      // Request r's step-i prediction row. With sample_from_anchor the anchor row
      // itself predicts (offset 0); otherwise the anchor is a bonus token and the
      // predictions start at offset 1.
      const int64_t global_row =
          static_cast<int64_t>(r) * nqpr + sample_off + i;
      const float* base =
          block_logits.data() + static_cast<size_t>(global_row) * draft_vocab;
      const float* b = bias.data() + static_cast<size_t>(r) * draft_vocab;

      // Greedy argmax over base + Markov bias, lowest-index tie-break (the same
      // argmax convention as the landed DFlash/MTP sampling gates).
      int32_t best_idx = 0;
      float best_val = base[0] + b[0];
      for (int64_t v = 1; v < draft_vocab; ++v) {
        const float val = base[static_cast<size_t>(v)] + b[static_cast<size_t>(v)];
        if (val > best_val) {
          best_val = val;
          best_idx = static_cast<int32_t>(v);
        }
      }
      // The sampled id is a DRAFT-vocab id; the target verifies TARGET ids
      // (map_draft_to_target, qwen3_dspark.py:137-141), and `prev` indexes
      // markov_w1, which is the TARGET vocab.
      const int32_t target_id = Qwen3DSparkModel::MapDraftToTarget(best_idx, weights);
      drafts[static_cast<size_t>(r)][static_cast<size_t>(i)] = target_id;
      prev[static_cast<size_t>(r)] = target_id;
    }
  }
  return drafts;
}

DsparkProposeResult DsparkProposeBlock(
    const Qwen3DSparkWeights& weights, const HfConfig& config,
    const std::vector<float>& context_states,
    const std::vector<int32_t>& context_positions, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions, const std::vector<int32_t>& block_cu,
    const std::vector<int32_t>& anchor_ids, const DsparkBlockLayout& layout,
    vt::Queue& queue) {
  const int num_reqs = static_cast<int>(anchor_ids.size());
  const int nqpr = layout.num_query_per_req();
  VT_CHECK(num_reqs > 0 && layout.num_speculative_steps > 0,
           "DsparkProposeBlock: bad batch/k");
  VT_CHECK(static_cast<int64_t>(block_input_ids.size()) ==
               static_cast<int64_t>(num_reqs) * nqpr,
           "DsparkProposeBlock: block_input_ids must be "
           "[num_reqs*num_query_per_req]");
  VT_CHECK(block_positions.size() == block_input_ids.size(),
           "DsparkProposeBlock: block_positions length must match input_ids");
  VT_CHECK(static_cast<int64_t>(block_cu.size()) == num_reqs + 1 &&
               static_cast<int64_t>(ctx_cu.size()) == num_reqs + 1,
           "DsparkProposeBlock: cu vectors must be [num_reqs+1]");

  // The parallel backbone forward is the INHERITED DFlash one, unchanged
  // (_generate_draft :158-165 calls DFlashSpeculator._run_model).
  const std::vector<float> block_logits =
      Qwen3DFlashModel::ForwardBlockLogitsWithContext(
          context_states, context_positions, ctx_cu, block_input_ids,
          block_positions, block_cu, weights.backbone, config, queue);

  DsparkProposeResult out;
  out.draft_token_ids =
      SampleDsparkBlockDrafts(block_logits, anchor_ids, layout, weights, queue);
  return out;
}

}  // namespace vllm::v1
