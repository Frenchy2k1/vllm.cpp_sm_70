// Laguna-S-2.1 — the three genuinely-NEW small host ops (W3). Pure numeric
// functions, no device/kernel dependency, unit-gated on CPU (test_laguna_scaffold
// laguna-ops cases). Grounding: vllm/model_executor/models/laguna.py
// (LagunaAttention softplus out-gate, LagunaMoE sigmoid FusedMoE) + vLLM
// FusedMoE.select_experts (sigmoid + e_score_correction_bias, use_grouped_topk=
// False) + rotary_embedding/yarn_scaling_rope.py (the YaRN inv_freq we REUSE).
#include "vllm/model_executor/models/laguna_ops.h"

#include <algorithm>
#include <cmath>

#include "vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {

float LagunaSoftplus(float x) {
  // torch F.softplus default (beta=1, threshold=20): linear above the threshold
  // to avoid exp() overflow; log1p(exp(x)) below. Computed in fp32 to mirror
  // laguna.py's `softplus(g_proj(x).float())`.
  if (x > 20.0F) return x;
  return std::log1p(std::exp(x));
}

void LagunaSoftplusHeadGate(std::vector<float>& attn,
                            const std::vector<float>& gate_logits,
                            int64_t num_heads, int64_t head_dim) {
  VT_CHECK(num_heads > 0 && head_dim > 0,
           "laguna gate: num_heads/head_dim must be positive");
  VT_CHECK(static_cast<int64_t>(gate_logits.size()) == num_heads,
           "laguna gate: gate_logits length must equal num_heads");
  VT_CHECK(static_cast<int64_t>(attn.size()) == num_heads * head_dim,
           "laguna gate: attn length must equal num_heads*head_dim");
  for (int64_t h = 0; h < num_heads; ++h) {
    const float g = LagunaSoftplus(gate_logits[static_cast<size_t>(h)]);
    float* row = attn.data() + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
    for (int64_t d = 0; d < head_dim; ++d) row[d] *= g;
  }
}

LagunaRouterSelection LagunaUngroupedRouterTopK(
    const std::vector<float>& router_logits,
    const std::vector<float>& e_score_bias, int64_t top_k, bool norm_topk_prob,
    float routed_scaling) {
  const int64_t E = static_cast<int64_t>(router_logits.size());
  VT_CHECK(E > 0, "laguna router: empty router_logits");
  VT_CHECK(top_k > 0 && top_k <= E, "laguna router: top_k out of range");
  VT_CHECK(e_score_bias.empty() ||
               static_cast<int64_t>(e_score_bias.size()) == E,
           "laguna router: e_score_bias length must equal num_experts (or empty)");

  // scores = sigmoid(logits); choice = scores + e_score_correction_bias.
  std::vector<float> scores(static_cast<size_t>(E));
  std::vector<float> choice(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    const float s = 1.0F / (1.0F + std::exp(-router_logits[static_cast<size_t>(e)]));
    scores[static_cast<size_t>(e)] = s;
    choice[static_cast<size_t>(e)] =
        s + (e_score_bias.empty() ? 0.0F : e_score_bias[static_cast<size_t>(e)]);
  }

  // top_k of `choice`, TIE-BREAK by LOWER expert index (the razor). Selection is
  // a partial sort over (choice desc, index asc); deterministic on ties.
  std::vector<int32_t> order(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) order[static_cast<size_t>(e)] = static_cast<int32_t>(e);
  std::partial_sort(
      order.begin(), order.begin() + top_k, order.end(),
      [&](int32_t a, int32_t b) {
        const float ca = choice[static_cast<size_t>(a)];
        const float cb = choice[static_cast<size_t>(b)];
        if (ca != cb) return ca > cb;
        return a < b;  // tie -> lower index wins
      });

  LagunaRouterSelection sel;
  sel.ids.resize(static_cast<size_t>(top_k));
  sel.weights.resize(static_cast<size_t>(top_k));
  float wsum = 0.0F;
  for (int64_t j = 0; j < top_k; ++j) {
    const int32_t id = order[static_cast<size_t>(j)];
    sel.ids[static_cast<size_t>(j)] = id;
    // UNBIASED sigmoid weight (gather from `scores`, NOT `choice`).
    const float w = scores[static_cast<size_t>(id)];
    sel.weights[static_cast<size_t>(j)] = w;
    wsum += w;
  }
  if (norm_topk_prob && wsum > 0.0F)
    for (float& w : sel.weights) w /= wsum;
  for (float& w : sel.weights) w *= routed_scaling;
  return sel;
}

namespace {
// Fill a [rows, rotary_dim] NeoX cos|sin cache from a precomputed inv_freq
// (length rotary_dim/2) with a scalar mscale. Layout matches
// rotary_embedding_detail::compute_yarn_cos_sin_cache.
std::vector<float> FillCosSin(const std::vector<float>& inv_freq,
                              int64_t rotary_dim, int64_t rows, float mscale) {
  const int64_t half = rotary_dim / 2;
  VT_CHECK(static_cast<int64_t>(inv_freq.size()) == half,
           "laguna rope: inv_freq length must equal rotary_dim/2");
  VT_CHECK(rows > 0, "laguna rope: rows must be positive");
  std::vector<float> cache(static_cast<size_t>(rows) *
                           static_cast<size_t>(rotary_dim));
  for (int64_t r = 0; r < rows; ++r) {
    const float p = static_cast<float>(r);
    for (int64_t i = 0; i < half; ++i) {
      const float freq = p * inv_freq[static_cast<size_t>(i)];
      const size_t base = static_cast<size_t>(r) * static_cast<size_t>(rotary_dim);
      cache[base + static_cast<size_t>(i)] = std::cos(freq) * mscale;
      cache[base + static_cast<size_t>(half + i)] = std::sin(freq) * mscale;
    }
  }
  return cache;
}
}  // namespace

std::vector<float> BuildLagunaFullYarnCosSin(const LagunaParams& p, int64_t rows) {
  VT_CHECK(p.rotary_dim_full > 0 && p.rotary_dim_full % 2 == 0,
           "laguna rope: rotary_dim_full must be positive/even");
  // REUSE the pinned YaRN inv_freq (linear-ramp interp/extrap blend) — the exact
  // numerics gated by test_rotary_embedding. Laguna supplies attention_factor
  // explicitly, so mscale = yarn_attention_factor (no yarn_get_mscale multiply).
  const std::vector<float> inv_freq = rotary_embedding_detail::compute_yarn_inv_freq(
      p.rotary_dim_full, p.rope_theta_full, p.yarn_orig_max_pos, p.yarn_factor,
      /*extrapolation_factor=*/1.0, static_cast<int64_t>(p.yarn_beta_fast),
      static_cast<int64_t>(p.yarn_beta_slow), /*truncate=*/false);
  return FillCosSin(inv_freq, p.rotary_dim_full, rows,
                    static_cast<float>(p.yarn_attention_factor));
}

std::vector<float> BuildLagunaSlidingCosSin(const LagunaParams& p, int64_t rows) {
  const int64_t rd = p.rotary_dim_sliding;
  VT_CHECK(rd > 0 && rd % 2 == 0,
           "laguna rope: rotary_dim_sliding must be positive/even");
  const int64_t half = rd / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  const double base = p.rope_theta_sliding;
  for (int64_t i = 0; i < half; ++i) {
    const double exponent = static_cast<double>(2 * i) / static_cast<double>(rd);
    inv_freq[static_cast<size_t>(i)] =
        static_cast<float>(1.0 / std::pow(base, exponent));
  }
  return FillCosSin(inv_freq, rd, rows, /*mscale=*/1.0F);
}

}  // namespace vllm
