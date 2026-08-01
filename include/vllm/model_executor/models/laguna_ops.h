// Laguna-S-2.1 — the three genuinely-NEW small host ops the ~85-90%-reuse forward
// composes (W3). Everything else Laguna needs is LANDED reuse (ds4 Q4_K decode,
// ds2 MoE, gemma sliding-window, olmo3 YaRN, phi partial-rope); these are the
// "~10-15% new", and each is a pure, CPU-testable numeric function — NOT a new
// compute kernel. See .agents/specs/laguna-s21-w3-2026-07-31.md.
//
//   (a) LagunaSoftplusHeadGate   — per-head softplus attention OUTPUT gate
//                                  (laguna.py LagunaAttention: gate =
//                                   softplus(g_proj(x).float()); attn *= gate).
//   (b) LagunaUngroupedRouterTopK — ungrouped sigmoid noaux_tc router: sigmoid
//                                  scoring + e_score_correction_bias for
//                                  SELECTION, UNBIASED sigmoid weights, renorm,
//                                  routed_scaling. DeepSeek-V3 noaux_tc MINUS the
//                                  group step (use_grouped_topk=False).
//   (c) BuildLaguna{FullYarn,Sliding}CosSin — the dual per-layer RoPE cos/sin
//                                  caches: full-attn YaRN over the PARTIAL 64 dims
//                                  (reuses rotary_embedding_detail::
//                                   compute_yarn_inv_freq) + sliding plain 128-dim.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/laguna.h"  // LagunaParams

namespace vllm {

// (a) Per-head SOFTPLUS attention OUTPUT gate (laguna.py LagunaAttention.forward).
//   gate[h] = softplus(gate_logits[h])   (fp32; torch F.softplus, threshold 20)
//   attn[h*head_dim + d] *= gate[h]       (broadcast the scalar over head_dim)
// `attn` is ONE token's attention output [num_heads*head_dim], heads-major, f32.
// `gate_logits` is g_proj(hidden) for that token [num_heads], f32. In place.
void LagunaSoftplusHeadGate(std::vector<float>& attn,
                            const std::vector<float>& gate_logits,
                            int64_t num_heads, int64_t head_dim);

// torch F.softplus(x) with the default beta=1, threshold=20 (linear above 20).
float LagunaSoftplus(float x);

// (b) UNGROUPED sigmoid noaux_tc router top-k (one token).
// Mirrors vLLM FusedMoE select_experts, sigmoid + e_score_correction_bias,
// use_grouped_topk=False:
//   scores        = sigmoid(router_logits)                    [num_experts]
//   choice_scores = scores + e_score_correction_bias          (SELECTION only)
//   ids           = top_k of choice_scores (tie-break: LOWER index wins)
//   weights       = scores.gather(ids)   (UNBIASED sigmoid weights)
//   if norm_topk_prob: weights /= sum(weights)
//   weights      *= routed_scaling        (== scaling the routed OUTPUT, linear)
// `e_score_bias` may be empty (treated as all-zero).
struct LagunaRouterSelection {
  std::vector<int32_t> ids;    // [top_k] selected expert indices (selection order)
  std::vector<float> weights;  // [top_k] combine weights (renormed, scaled)
};
LagunaRouterSelection LagunaUngroupedRouterTopK(
    const std::vector<float>& router_logits,
    const std::vector<float>& e_score_bias, int64_t top_k, bool norm_topk_prob,
    float routed_scaling);

// (c) Dual per-layer RoPE cos/sin caches, laid out [rows, rotary_dim] with the
// NeoX cos|sin half split per row (matching rotary_embedding_detail's cache):
//   row r, i in [0, rotary_dim/2): cache[r*rd + i]           = cos(r*inv_freq[i])*m
//                                  cache[r*rd + rd/2 + i]     = sin(r*inv_freq[i])*m
// FULL-attention (global) layers: YaRN inv_freq (theta_full, factor, beta_fast/
// slow) over rotary_dim_full (=64), mscale = LagunaYarnMscale(factor, attn_factor).
// SLIDING layers: plain inv_freq[i]=theta_sliding^(-(2i)/rotary_dim_sliding),
// mscale = 1, over rotary_dim_sliding (=128). `rows` sizes the (position) axis;
// `pos0` is the global position of the FIRST row (default 0 → rows cover [0,rows)).
// The resident decode reads only row `pos`, so it builds a single row with rows=1,
// pos0=pos (avoids the O(pos)/token full-table rebuild = O(n^2) over a generation).
std::vector<float> BuildLagunaFullYarnCosSin(const LagunaParams& p, int64_t rows, int64_t pos0 = 0);
std::vector<float> BuildLagunaSlidingCosSin(const LagunaParams& p, int64_t rows, int64_t pos0 = 0);

// llama.cpp YaRN mscale: yarn_attn_factor * (1 + 0.1*ln(factor)) for factor>1,
// else yarn_attn_factor. Reproduces HF's precomputed attention_factor and the
// GGUF's yarn_attn_factor=1.0 + factor 32. Exposed for the RoPE bit-match gate.
double LagunaYarnMscale(double factor, double yarn_attn_factor);

}  // namespace vllm
