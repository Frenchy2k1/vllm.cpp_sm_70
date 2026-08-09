// Ported from: vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
// (DSparkSpeculator :37, _sample_sequential :100-149, _generate_draft :151-169)
// @ 555967922 (vLLM 0.26.0.dev0). SPEC-DSPARK W4, row SPEC-DSPARK.
//
// DSpark's draft step is DFlash's parallel block forward followed by a SEQUENTIAL
// pass. `DSparkSpeculator(DFlashSpeculator)` overrides exactly two things, and
// this header owns both:
//
//   1. ANCHOR-AS-FIRST-PREDICTION. Each request emits N = num_speculative_tokens
//      query tokens (anchor + N-1 noise), NOT DFlash's 1 + N. EVERY query
//      position is a prediction — the anchor predicts the first draft token — so
//      we sample at all N rows and sample_pos = query_pos + 1 (standard
//      next-token), where DFlash's masks sit AT the predicted position
//      (speculator.py:5-11,50-58). Speculators-format checkpoints keep the
//      DFlash 1 + N fill-in layout instead (sample_from_anchor false), which is
//      why both layouts are parameters here rather than a constant.
//      Our landed PrepareDflashInputs already implements both
//      (qwen3_dflash.cpp:126,169-174, `sample_from_anchor`); the caller sets
//      num_query_per_req = N and the flag.
//
//   2. SEQUENTIAL MARKOV SAMPLING. Instead of DFlash's single parallel argmax
//      over all block rows, DSpark samples left-to-right, adding a
//      prefix-dependent Markov bias derived from the PREVIOUSLY SAMPLED token at
//      each step (speculator.py:100-149). This is what gives a parallel block
//      draft intra-block dependency; without it the k tokens are conditionally
//      independent given the context, which is exactly DFlash.
//
// Upstream computes `sample_hidden = head_hidden[sample_indices]` and then
// `compute_draft_logits(sample_hidden)` per step. We take the block logits the
// backbone forward already produced, because compute_draft_logits is a per-row
// lm_head (qwen3_dspark.py:132-135) and the backbone's
// ForwardBlockLogitsWithContext applies exactly that lm_head to every block row —
// so selecting rows before or after the lm_head yields the same numbers, and this
// way the k steps do not re-enter the model.
//
// GREEDY here. The probabilistic branch (Gumbel + the reduced-vocab scatter into
// target columns, speculator.py:124-146) is the next slice: our landed spec-decode
// gates are greedy, and greedy-first is the honest order.
#ifndef VLLM_V1_WORKER_GPU_SPEC_DECODE_DSPARK_SPECULATOR_H_
#define VLLM_V1_WORKER_GPU_SPEC_DECODE_DSPARK_SPECULATOR_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_dspark.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm::v1 {

// The block-row layout of one request's query block. `num_query_per_req` is N
// when sampling from the anchor and 1 + N otherwise; `first_sample_offset` is the
// block offset of the FIRST predicted row (0 with the anchor, 1 without) —
// exactly the `sample_off` our PrepareDflashInputs computes.
struct DsparkBlockLayout {
  int num_speculative_steps = 0;   // N (k)
  bool sample_from_anchor = true;  // native Qwen3 DSpark configs default to true
  int num_query_per_req() const {
    return sample_from_anchor ? num_speculative_steps : 1 + num_speculative_steps;
  }
  int first_sample_offset() const { return sample_from_anchor ? 0 : 1; }
};

// The sequential Markov sample loop (_sample_sequential :100-149, greedy).
//
//   block_logits  [num_reqs * num_query_per_req, draft_vocab] f32 — the backbone's
//                 base draft logits for every query row, in per-request block-row
//                 order (request r owns rows [r*nqpr, (r+1)*nqpr)).
//   anchor_ids    [num_reqs] — each request's anchor token in the TARGET vocab
//                 (the bonus/verified token, read upstream from
//                 input_ids[_anchor_idx], speculator.py:120-121). It seeds `prev`.
//
// Per step i in [0, N): bias = markov_w2(markov_w1[prev]); logits_i = base row +
// bias; draft = map_draft_to_target(argmax(logits_i)); prev = draft. Returns
// [num_reqs][N] TARGET-vocab draft ids.
//
// The whole batch takes ONE bias GEMV per step (prev is [num_reqs]), matching
// upstream's per-step batched markov_embed/markov_bias rather than looping
// requests.
std::vector<std::vector<int32_t>> SampleDsparkBlockDrafts(
    const std::vector<float>& block_logits, const std::vector<int32_t>& anchor_ids,
    const DsparkBlockLayout& layout, const Qwen3DSparkWeights& weights,
    vt::Queue& queue);

// One DSpark block propose: the DFlash context-aware block forward over the
// caller-accumulated context (unchanged, inherited) followed by the sequential
// Markov sampling above (_generate_draft :151-169). Arguments mirror
// DflashProposeBlock one-for-one; the only additions are the anchor ids and the
// block layout.
struct DsparkProposeResult {
  std::vector<std::vector<int32_t>> draft_token_ids;  // [num_reqs][N], target vocab
};

DsparkProposeResult DsparkProposeBlock(
    const Qwen3DSparkWeights& weights, const HfConfig& config,
    const std::vector<float>& context_states,
    const std::vector<int32_t>& context_positions, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions, const std::vector<int32_t>& block_cu,
    const std::vector<int32_t>& anchor_ids, const DsparkBlockLayout& layout,
    vt::Queue& queue);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_SPEC_DECODE_DSPARK_SPECULATOR_H_
