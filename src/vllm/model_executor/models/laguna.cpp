// Laguna-S-2.1 forward — W3 REAL host-reference composition. Replaces the W1/W2
// `VT_CHECK(false, ...)` stub with a runnable per-layer forward that COMPOSES the
// ~85-90% reuse + the three genuinely-NEW small host ops (laguna_ops.cpp):
//
//   x_res = x; h = RMSNorm(x, input_norm)                    [shared RMSNorm]
//   q/k/v = {q,k,v}_proj(h)   (q width is PER-LAYER: 48 global / 72 sliding)
//   q,k  = dual per-layer RoPE (global -> YaRN partial-64 [olmo3 inv_freq];
//          sliding -> plain-128), applied NeoX-style on the first rotary_dim dims
//   attn = GQA(q,k,v, mask = global ? full-causal : sliding-window(512))
//                                                            [gemma2/3 is_sliding]
//   attn = per-head SOFTPLUS out-gate(attn, g_proj(h))       [NEW op (a)]
//   x   += o_proj(attn)
//   h2   = RMSNorm(x, post_attn_norm)
//   f    = dense SwiGLU MLP (layer 0)  |  ungrouped sigmoid-noaux MoE (1..47):
//            sel = UngroupedRouterTopK(router(h2), e_score_bias, top_k, renorm,
//                                      routed_scaling)        [NEW op (b), ds2-minus-group]
//            f   = sum_i sel.w_i * expert_{sel.id_i}(h2) + shared_expert(h2)
//   x   += f
//   final: RMSNorm(x, norm) -> lm_head (untied) -> logits
//
// SCOPE HONESTY: this is the whole-sequence (prefill) REFERENCE forward in f32 —
// runnable + unit-gated on synthetic weights (test_laguna_scaffold laguna-fwd
// case), the vehicle that verifies the composition + the new ops fire and the
// per-layer variable-Q-head shapes flow. It does NOT consume the paged KV cache
// (`attn_kv`/`attn_meta` are accepted but the reference recomputes attention over
// the token window); the device/paged production path (bf16 vt:: ops off the ds4
// keep-quant towers, olmo2.cpp ForwardBody pattern) + the tower materialization +
// the strict dual-oracle greedy gate (llama.cpp-Q4_K token-exact + vLLM-NVFP4
// near-tie) are W4. See .agents/specs/laguna-s21-w3-2026-07-31.md.
#include "vllm/model_executor/models/laguna.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "vllm/model_executor/models/laguna_ops.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/dtype.h"

namespace vllm {
namespace {

// Decode an OwnedTensor (host bytes) to a flat f32 vector. Reference path handles
// the two host dtypes a synthetic/dequantized tower carries: f32 and bf16.
std::vector<float> ReadF32(const OwnedTensor& t) {
  const int64_t n = t.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  const uint8_t* raw = t.bytes.data();
  if (t.dtype == vt::DType::kF32) {
    std::memcpy(out.data(), raw, static_cast<size_t>(n) * sizeof(float));
  } else if (t.dtype == vt::DType::kBF16) {
    const auto* b = reinterpret_cast<const uint16_t*>(raw);
    for (int64_t i = 0; i < n; ++i) {
      const uint32_t bits = static_cast<uint32_t>(b[i]) << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(float));
    }
  } else {
    VT_CHECK(false, "laguna reference forward: weight dtype must be f32/bf16 "
                    "(quant keep-quant decode is the W4 device path)");
  }
  return out;
}

// out[T,N] = x[T,K] @ W_nk[N,K]^T  (raw-NK torch Linear weight, no bias).
std::vector<float> MatmulNK(const std::vector<float>& x, const std::vector<float>& w,
                            int64_t T, int64_t N, int64_t K) {
  VT_CHECK(static_cast<int64_t>(x.size()) == T * K, "laguna matmul: x shape");
  VT_CHECK(static_cast<int64_t>(w.size()) == N * K, "laguna matmul: w shape");
  std::vector<float> out(static_cast<size_t>(T * N), 0.0F);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t nn = 0; nn < N; ++nn) {
      float acc = 0.0F;
      const float* xr = x.data() + static_cast<size_t>(t * K);
      const float* wr = w.data() + static_cast<size_t>(nn * K);
      for (int64_t kk = 0; kk < K; ++kk) acc += xr[kk] * wr[kk];
      out[static_cast<size_t>(t * N + nn)] = acc;
    }
  return out;
}

// RMSNorm(x, weight, eps): variance in f32.
std::vector<float> RmsNorm(const std::vector<float>& x, const std::vector<float>& w,
                           int64_t T, int64_t H, float eps) {
  std::vector<float> out(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const float* xr = x.data() + static_cast<size_t>(t * H);
    float ss = 0.0F;
    for (int64_t i = 0; i < H; ++i) ss += xr[i] * xr[i];
    const float inv = 1.0F / std::sqrt(ss / static_cast<float>(H) + eps);
    float* orow = out.data() + static_cast<size_t>(t * H);
    for (int64_t i = 0; i < H; ++i)
      orow[i] = xr[i] * inv * w[static_cast<size_t>(i)];
  }
  return out;
}

inline float Silu(float x) { return x / (1.0F + std::exp(-x)); }

// SwiGLU: merged gate_up[T,2I] -> silu(gate)*up -> [T,I].
std::vector<float> SiluAndMul(const std::vector<float>& gate_up, int64_t T,
                              int64_t I) {
  std::vector<float> act(static_cast<size_t>(T * I));
  for (int64_t t = 0; t < T; ++t) {
    const float* g = gate_up.data() + static_cast<size_t>(t * 2 * I);
    const float* u = g + I;
    float* a = act.data() + static_cast<size_t>(t * I);
    for (int64_t i = 0; i < I; ++i) a[i] = Silu(g[i]) * u[i];
  }
  return act;
}

// In-place NeoX partial RoPE over the first `rd` dims of each head. `cache` is
// [rows, rd] with the cos|sin half split (BuildLaguna*CosSin layout).
void ApplyRope(std::vector<float>& x, int64_t T, int64_t heads, int64_t Dh,
               int64_t rd, const std::vector<float>& cache,
               const std::vector<int32_t>& positions) {
  const int64_t half = rd / 2;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t pos = positions[static_cast<size_t>(t)];
    const float* crow = cache.data() + static_cast<size_t>(pos * rd);
    for (int64_t h = 0; h < heads; ++h) {
      float* xv = x.data() + static_cast<size_t>((t * heads + h) * Dh);
      for (int64_t i = 0; i < half; ++i) {
        const float c = crow[i];
        const float s = crow[half + i];
        const float x0 = xv[i];
        const float x1 = xv[half + i];
        xv[i] = x0 * c - x1 * s;
        xv[half + i] = x1 * c + x0 * s;
      }
    }
  }
}

std::vector<float> ExpertMlp(const std::vector<float>& h_row,
                             const std::vector<float>& gate_up_w,
                             const std::vector<float>& down_w, int64_t H,
                             int64_t I) {
  const std::vector<float> gu = MatmulNK(h_row, gate_up_w, 1, 2 * I, H);
  const std::vector<float> act = SiluAndMul(gu, 1, I);
  return MatmulNK(act, down_w, 1, H, I);  // [H]
}

}  // namespace

std::vector<float> LagunaModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;   // reference forward recomputes attention (paged path = W4)
  (void)config;
  (void)queue;
  const LagunaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const int64_t Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t kvdim = Hkv * Dh;
  const float eps = p.rms_norm_eps;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "laguna: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == p.num_hidden_layers,
           "laguna: one LagunaLayerWeights per layer required");

  // Size the RoPE caches to the max position (+1). Built once per regime.
  int64_t max_pos = 0;
  for (int32_t ps : positions) max_pos = std::max<int64_t>(max_pos, ps);
  const int64_t rope_rows = max_pos + 1;
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, rope_rows);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, rope_rows);

  // Embed: hidden[T,H] = embed[token_ids].
  const std::vector<float> embed = ReadF32(weights.embed);
  std::vector<float> hidden(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = token_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < Vsz, "laguna: token id out of range");
    std::memcpy(hidden.data() + static_cast<size_t>(t * H),
                embed.data() + static_cast<size_t>(tok * H),
                static_cast<size_t>(H) * sizeof(float));
  }

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l);
    const int64_t qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    VT_CHECK(group > 0 && Hq == group * Hkv,
             "laguna: per-layer Q-head count must be a multiple of KV heads");
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);

    // --- attention ---
    const std::vector<float> hn =
        RmsNorm(hidden, ReadF32(lw.input_norm), T, H, eps);
    std::vector<float> q = MatmulNK(hn, ReadF32(lw.attn.q_proj), T, qdim, H);
    std::vector<float> k = MatmulNK(hn, ReadF32(lw.attn.k_proj), T, kvdim, H);
    std::vector<float> v = MatmulNK(hn, ReadF32(lw.attn.v_proj), T, kvdim, H);
    const std::vector<float>& cache = global ? yarn_cache : slide_cache;
    ApplyRope(q, T, Hq, Dh, rd, cache, positions);
    ApplyRope(k, T, Hkv, Dh, rd, cache, positions);

    // GQA attention with the per-layer mask (global full-causal / sliding-window).
    std::vector<float> attn(static_cast<size_t>(T * qdim), 0.0F);
    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t kvh = h / group;
      for (int64_t i = 0; i < T; ++i) {
        const int64_t pi = positions[static_cast<size_t>(i)];
        // score over all j with pos_j <= pos_i (causal) and, for sliding layers,
        // pos_i - pos_j < window (FA window convention).
        float maxs = -std::numeric_limits<float>::infinity();
        std::vector<float> logit(static_cast<size_t>(T),
                                 -std::numeric_limits<float>::infinity());
        for (int64_t j = 0; j < T; ++j) {
          const int64_t pj = positions[static_cast<size_t>(j)];
          if (pj > pi) continue;
          if (window > 0 && pi - pj >= window) continue;
          const float* qv = q.data() + static_cast<size_t>((i * Hq + h) * Dh);
          const float* kvp = k.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
          float dot = 0.0F;
          for (int64_t d = 0; d < Dh; ++d) dot += qv[d] * kvp[d];
          dot *= scale;
          logit[static_cast<size_t>(j)] = dot;
          maxs = std::max(maxs, dot);
        }
        float denom = 0.0F;
        for (int64_t j = 0; j < T; ++j) {
          if (logit[static_cast<size_t>(j)] ==
              -std::numeric_limits<float>::infinity())
            continue;
          const float e = std::exp(logit[static_cast<size_t>(j)] - maxs);
          logit[static_cast<size_t>(j)] = e;
          denom += e;
        }
        float* ao = attn.data() + static_cast<size_t>((i * Hq + h) * Dh);
        for (int64_t j = 0; j < T; ++j) {
          const float w = logit[static_cast<size_t>(j)];
          if (w == -std::numeric_limits<float>::infinity() || w == 0.0F) continue;
          const float pw = w / denom;
          const float* vv = v.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
          for (int64_t d = 0; d < Dh; ++d) ao[d] += pw * vv[d];
        }
      }
    }

    // NEW op (a): per-head softplus attention OUTPUT gate.
    const std::vector<float> glogits =
        MatmulNK(hn, ReadF32(lw.attn.g_proj), T, Hq, H);  // [T,Hq]
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> row(attn.begin() + static_cast<int64_t>(i * qdim),
                             attn.begin() + static_cast<int64_t>((i + 1) * qdim));
      std::vector<float> gl(glogits.begin() + static_cast<int64_t>(i * Hq),
                            glogits.begin() + static_cast<int64_t>((i + 1) * Hq));
      LagunaSoftplusHeadGate(row, gl, Hq, Dh);
      std::copy(row.begin(), row.end(),
                attn.begin() + static_cast<int64_t>(i * qdim));
    }

    // o_proj + residual.
    const std::vector<float> o = MatmulNK(attn, ReadF32(lw.attn.o_proj), T, H, qdim);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

    // --- FFN: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE ---
    const std::vector<float> hn2 =
        RmsNorm(hidden, ReadF32(lw.post_attn_norm), T, H, eps);
    std::vector<float> f(static_cast<size_t>(T * H), 0.0F);
    if (lw.is_dense) {
      const int64_t I = p.intermediate_size;
      const std::vector<float> gu = MatmulNK(hn2, ReadF32(lw.mlp.gate_up_proj), T, 2 * I, H);
      const std::vector<float> act = SiluAndMul(gu, T, I);
      f = MatmulNK(act, ReadF32(lw.mlp.down_proj), T, H, I);
    } else {
      const int64_t E = p.num_experts;
      const int64_t moe_I = p.moe_intermediate_size;
      const std::vector<float> router_w = ReadF32(lw.moe.router);
      std::vector<float> bias;
      if (!lw.moe.e_score_correction_bias.Empty())
        bias = ReadF32(lw.moe.e_score_correction_bias);
      const std::vector<float> exp_gu = ReadF32(lw.moe.experts_gate_up);  // [E,2moeI,H]
      const std::vector<float> exp_dn = ReadF32(lw.moe.experts_down);     // [E,H,moeI]
      const int64_t gu_stride = 2 * moe_I * H;
      const int64_t dn_stride = H * moe_I;
      const bool has_shared = !lw.moe.shared_gate_up.Empty();
      std::vector<float> shared_gu, shared_dn;
      if (has_shared) {
        shared_gu = ReadF32(lw.moe.shared_gate_up);
        shared_dn = ReadF32(lw.moe.shared_down);
      }
      for (int64_t i = 0; i < T; ++i) {
        std::vector<float> hrow(hn2.begin() + static_cast<int64_t>(i * H),
                                hn2.begin() + static_cast<int64_t>((i + 1) * H));
        const std::vector<float> rlog = MatmulNK(hrow, router_w, 1, E, H);  // [E]
        const LagunaRouterSelection sel = LagunaUngroupedRouterTopK(
            rlog, bias, p.num_experts_per_tok, p.norm_topk_prob,
            p.moe_routed_scaling_factor);
        std::vector<float> acc(static_cast<size_t>(H), 0.0F);
        for (size_t s = 0; s < sel.ids.size(); ++s) {
          const int64_t id = sel.ids[s];
          std::vector<float> egu(exp_gu.begin() + static_cast<int64_t>(id * gu_stride),
                                 exp_gu.begin() + static_cast<int64_t>((id + 1) * gu_stride));
          std::vector<float> edn(exp_dn.begin() + static_cast<int64_t>(id * dn_stride),
                                 exp_dn.begin() + static_cast<int64_t>((id + 1) * dn_stride));
          const std::vector<float> eo = ExpertMlp(hrow, egu, edn, H, moe_I);
          const float w = sel.weights[s];
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += w * eo[static_cast<size_t>(d)];
        }
        if (has_shared) {
          const std::vector<float> so = ExpertMlp(hrow, shared_gu, shared_dn, H, moe_I);
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += so[static_cast<size_t>(d)];
        }
        std::copy(acc.begin(), acc.end(),
                  f.begin() + static_cast<int64_t>(i * H));
      }
    }
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += f[static_cast<size_t>(i)];
  }

  // Final RMSNorm -> lm_head (untied) -> logits.
  const std::vector<float> hn = RmsNorm(hidden, ReadF32(weights.norm), T, H, eps);
  const bool gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  std::vector<float> src;
  int64_t n_out;
  if (gather) {
    n_out = static_cast<int64_t>(logits_indices.size());
    src.resize(static_cast<size_t>(n_out * H));
    for (int64_t r = 0; r < n_out; ++r)
      std::memcpy(src.data() + static_cast<size_t>(r * H),
                  hn.data() + static_cast<size_t>(logits_indices[static_cast<size_t>(r)] * H),
                  static_cast<size_t>(H) * sizeof(float));
  } else {
    n_out = T;
    src = hn;
  }
  const bool tied = p.tie_word_embeddings || weights.lm_head.Empty();
  return MatmulNK(src, ReadF32(tied ? weights.embed : weights.lm_head), n_out, Vsz, H);
}

ForwardLogits LagunaModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  // The device path reuses the host reference until the W4 device assembly lands
  // (mirrors ds4's ForwardDevice-after-Forward staging).
  return HostLogits(
      LagunaModel::Forward(token_ids, positions, attn_meta, attn_kv, weights,
                           config, queue, logits_indices),
      weights.params.vocab_size);
}

}  // namespace vllm
