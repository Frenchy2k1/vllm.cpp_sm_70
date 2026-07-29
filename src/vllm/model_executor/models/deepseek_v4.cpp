// DeepSeek-V4-Flash forward — W7 ASSEMBLY. The `VT_CHECK(false, "W3-W8 pending")`
// stub is replaced by a REAL `DeepseekV4Model::Forward` that COMPOSES the four
// landed host-reference primitive stacks (W3 DSA Lightning-Indexer + 512-wide MLA
// output seams, W4 compressor + fp8_ds_mla KV state, W5 Manifold Hyper-Connections
// + Sinkhorn, W6 sqrtsoftplus/hash MoE + clamped SwiGLU) into an end-to-end logits
// producer on the portable CPU path at a SMALL synthetic config.
//
// ─── HONEST SCOPE (mirrors W3-W6) ───────────────────────────────────────────
// The fixed-config 167B V4 does NOT fit ONE GB10 (156.7 GiB, see deepseek_v4.h)
// and its weights are NOT materialized (W2b residual), so W7 is DERIVED +
// BUILD-VERIFIED: it assembles the interleave + is STRUCTURALLY gated at a tiny
// synthetic shape (test_deepseek_v4_forward.cpp) — NOT a real-checkpoint token
// gate (that is W8, multi-Spark). This does NOT claim V4 "runs" a real model; it
// claims the forward ASSEMBLES + is structurally gated at tiny shape. The device
// kernels (MHC Sinkhorn, DSA indexer/compressor, sqrtsoftplus router, clamped
// SwiGLU — the expert GEMM REUSES the existing NVFP4/FP8 grouped-GEMM) + the real
// e2e are named residuals (W7-device + W8).
//
// ─── INTERLEAVE (grounded, file:line on both sides, @ pin 555967922) ─────────
// vllm/models/deepseek_v4/nvidia/model.py:1080-1148 (DeepseekV4Model.forward) +
// :866-957 (DeepseekV4DecoderLayer.forward):
//   embed -> for each layer:
//     [first layer] MHC-pre BROADCAST expand [T,H] -> [T,hc,H] (mhc_pre_broadcast,
//        :880-897) ; [else] MHC fused-post-pre = MhcPost(prev-ffn-out) + MhcPre(attn)
//     512-wide MLA attn: q(wq_a->q_norm->wq_b) + kv(wkv->kv_norm), dual-theta RoPE,
//        DSA indexer->topk->compressor->fp8_ds_mla KV, sink softmax, grouped o-LoRA
//     MHC fused-post-pre = MhcPost(attn-out) + MhcPre(ffn)   (:934-957)
//     MoE: sqrtsoftplus/hash router + shared+routed clamped-SwiGLU experts
//   final MhcPost(last-ffn-out) -> hc_head collapse (:1136) -> norm -> lm_head.
//
// Where the tiny-config forward diverges from the 167B structure (documented, not
// silent): (i) the compressor pools a fixed W=2 window rather than the real
// (1+overlap)*compress_ratio window (the state-cache gather addressing is a W7
// device concern, deepseek_v4_compressor.h note); (ii) the MLA value is the full
// decoded latent (the W_UK/W_UV absorption geometry is a shared-MLA-extraction W7
// follow-on); (iii) a single rope_theta is used for all layers (the compressed
// layers' dual compress_rope_theta is a device-RoPE seam); (iv) the fp8_ds_mla
// quant_block == nope_head_dim (one block) at tiny width. Each reuses the SAME
// landed primitive math the device kernels will call.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_compressor.h"
#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/deepseek_v4_moe.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using deepseek_v4::ClampedSwiGLU;
using deepseek_v4::CompressorPoolNorm;
using deepseek_v4::CompressorSaveScoreApe;
using deepseek_v4::DsaIndexerLogits;
using deepseek_v4::DsaIndexerWeightFold;
using deepseek_v4::DsaTopkSelect;
using deepseek_v4::Fp8DsMlaDecodeToken;
using deepseek_v4::Fp8DsMlaEncodeToken;
using deepseek_v4::Fp8DsMlaLayout;
using deepseek_v4::HcHeadCollapse;
using deepseek_v4::MakeFp8DsMlaLayout;
using deepseek_v4::MhcPost;
using deepseek_v4::MhcPre;
using deepseek_v4::MhcPreResult;
using deepseek_v4::MoeRouteResult;
using deepseek_v4::SoftmaxWithSink;
using deepseek_v4::SqrtSoftplusRouteTopk;

// ── small portable linear-algebra helpers ────────────────────────────────────
float Dot(const float* a, const float* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

// y[o] = Σ_i W[o*in + i] * x[i]  (W is [out, in] row-major).
std::vector<float> MatVec(const std::vector<float>& w, const float* x, int64_t out,
                          int64_t in) {
  VT_CHECK(static_cast<int64_t>(w.size()) == out * in, "MatVec weight size mismatch");
  std::vector<float> y(static_cast<size_t>(out));
  for (int64_t o = 0; o < out; ++o) y[static_cast<size_t>(o)] = Dot(&w[o * in], x, in);
  return y;
}

// Weighted RMSNorm (the standard DeepSeek/vLLM RMSNorm).
std::vector<float> RmsNorm(const std::vector<float>& x, const std::vector<float>& w,
                           float eps) {
  const int64_t n = static_cast<int64_t>(x.size());
  double ss = 0.0;
  for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float r = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
  std::vector<float> y(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    y[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] * r * w[static_cast<size_t>(i)];
  return y;
}

// Decoupled NeoX-free pairwise RoPE over an `r`-wide (r even) rope subvector
// (common/rope.py deepseek_yarn, mscale disabled — the tiny forward uses a single
// theta; the compressed-layer dual compress_rope_theta is a device-RoPE seam).
void RopeInplace(float* v, int64_t r, int64_t pos, double theta) {
  for (int64_t i = 0; i < r / 2; ++i) {
    const double freq =
        std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(r));
    const double ang = static_cast<double>(pos) * freq;
    const float c = static_cast<float>(std::cos(ang));
    const float s = static_cast<float>(std::sin(ang));
    const float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * c - b * s;
    v[2 * i + 1] = a * s + b * c;
  }
}

std::vector<float> Slice(const std::vector<float>& v, int64_t off, int64_t len) {
  return std::vector<float>(v.begin() + off, v.begin() + off + len);
}

// ── 512-wide MLA attention block (W3 + W4 primitives) : [T,H] -> [T,H] ────────
std::vector<float> AttentionBlock(const DeepseekV4LayerHostWeights& L,
                                  const DeepseekV4Params& p,
                                  const std::vector<float>& x,
                                  const std::vector<int32_t>& positions, int64_t layer,
                                  V4Miswire miswire, V4ForwardTrace* trace) {
  const int64_t T = static_cast<int64_t>(positions.size());
  const int64_t H = p.hidden_size;
  const int64_t nh = p.num_attention_heads;
  const int64_t hd = p.head_dim;
  const int64_t rope = p.qk_rope_head_dim;
  const int64_t nope = hd - rope;
  const int64_t qlr = p.q_lora_rank;
  const float eps = p.rms_norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const bool is_indexer = p.has_indexer(layer);
  const bool is_comp = p.has_compressor(layer);

  // 1. per-token q [T,nh,hd] and raw kv latent [T,hd] (num_key_value_heads=1 MLA).
  std::vector<float> q(static_cast<size_t>(T) * nh * hd);
  std::vector<float> kraw(static_cast<size_t>(T) * hd);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> qa =
        RmsNorm(MatVec(L.wq_a, &x[t * H], qlr, H), L.q_norm_weight, eps);
    std::vector<float> qf = MatVec(L.wq_b, qa.data(), nh * hd, qlr);
    for (int64_t h = 0; h < nh; ++h)
      RopeInplace(&qf[h * hd + nope], rope, positions[static_cast<size_t>(t)], p.rope_theta);
    for (int64_t i = 0; i < nh * hd; ++i) q[t * nh * hd + i] = qf[static_cast<size_t>(i)];
    std::vector<float> kv = RmsNorm(MatVec(L.wkv, &x[t * H], hd, H), L.kv_norm_weight, eps);
    RopeInplace(&kv[nope], rope, positions[static_cast<size_t>(t)], p.rope_theta);
    for (int64_t d = 0; d < hd; ++d) kraw[t * hd + d] = kv[static_cast<size_t>(d)];
  }

  // 2. compressor (compressor layers): softmax-window POOL + save-time APE + RMSNorm
  //    into the cached latent (deepseek_v4_compressor.h : CompressorSaveScoreApe /
  //    CompressorPoolNorm). Non-compressor layers cache the raw latent directly.
  std::vector<float> latent = kraw;
  if (is_comp) {
    const int64_t cr = p.compress_ratio(layer);
    const int64_t win = 2;  // tiny pooling window (device gather addressing = W7 seam)
    std::vector<float> score(static_cast<size_t>(T) * hd);
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> s = MatVec(L.comp_wgate, &x[t * H], hd, H);
      for (int64_t d = 0; d < hd; ++d) score[t * hd + d] = s[static_cast<size_t>(d)];
    }
    std::vector<int64_t> pos64(positions.begin(), positions.end());
    score = CompressorSaveScoreApe(score, L.comp_ape, pos64, T, hd, cr);
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> kvwin(static_cast<size_t>(win) * hd, 0.0f);
      std::vector<float> scwin(static_cast<size_t>(win) * hd, 0.0f);
      std::vector<uint8_t> valid(static_cast<size_t>(win), 0);
      for (int64_t i = 0; i < win; ++i) {
        const int64_t row = t - (win - 1) + i;
        if (row < 0) continue;
        valid[static_cast<size_t>(i)] = 1;
        for (int64_t d = 0; d < hd; ++d) {
          kvwin[i * hd + d] = kraw[row * hd + d];
          scwin[i * hd + d] = score[row * hd + d];
        }
      }
      std::vector<float> comp =
          CompressorPoolNorm(kvwin, scwin, valid, L.comp_norm_weight, eps, win, hd);
      for (int64_t d = 0; d < hd; ++d) latent[t * hd + d] = comp[static_cast<size_t>(d)];
    }
    if (trace != nullptr) trace->layer_compressor_ran[static_cast<size_t>(layer)] = 1;
  }

  // 3. fp8_ds_mla KV-state round-trip — EXERCISE the paged cache layout (W4).
  const Fp8DsMlaLayout ly = MakeFp8DsMlaLayout(nope, rope, /*quant_block=*/nope);
  std::vector<float> deck(static_cast<size_t>(T) * hd);
  for (int64_t t = 0; t < T; ++t) {
    const auto tok = Fp8DsMlaEncodeToken(Slice(latent, t * hd, hd), ly);
    const std::vector<float> dec = Fp8DsMlaDecodeToken(tok, ly);
    for (int64_t d = 0; d < hd; ++d) deck[t * hd + d] = dec[static_cast<size_t>(d)];
  }

  // 4. selection: DSA Lightning-Indexer top-k on indexer layers, else dense causal.
  std::vector<std::vector<int64_t>> sel(static_cast<size_t>(T));
  if (is_indexer) {
    const int64_t inh = p.index_n_heads, ihd = p.index_head_dim, itopk = p.index_topk;
    std::vector<float> iq(static_cast<size_t>(T) * inh * ihd);
    std::vector<float> ik(static_cast<size_t>(T) * ihd);
    std::vector<float> wproj(static_cast<size_t>(T) * inh);
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> a = MatVec(L.idx_wq, &x[t * H], inh * ihd, H);
      for (int64_t i = 0; i < inh * ihd; ++i) iq[t * inh * ihd + i] = a[static_cast<size_t>(i)];
      std::vector<float> b = MatVec(L.idx_wk, &x[t * H], ihd, H);
      for (int64_t d = 0; d < ihd; ++d) ik[t * ihd + d] = b[static_cast<size_t>(d)];
      std::vector<float> c = MatVec(L.idx_wproj, &x[t * H], inh, H);
      for (int64_t h = 0; h < inh; ++h) wproj[t * inh + h] = c[static_cast<size_t>(h)];
    }
    const std::vector<float> folded = DsaIndexerWeightFold(wproj, T, inh, ihd);
    std::vector<int64_t> ws(static_cast<size_t>(T)), we(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      ws[static_cast<size_t>(t)] = 0;
      we[static_cast<size_t>(t)] = t + 1;  // causal candidate window
    }
    const std::vector<float> logits =
        DsaIndexerLogits(iq, ik, folded, ws, we, T, T, inh, ihd);
    const std::vector<int64_t> topk = DsaTopkSelect(logits, ws, we, T, T, itopk);
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < itopk; ++j) {
        const int64_t s = topk[t * itopk + j];
        if (s >= 0) sel[static_cast<size_t>(t)].push_back(s);
      }
    if (trace != nullptr) {
      trace->layer_is_indexer[static_cast<size_t>(layer)] = 1;
      trace->layer_indexer_selected[static_cast<size_t>(layer)] =
          T > 0 ? static_cast<int>(sel[static_cast<size_t>(T - 1)].size()) : 0;
    }
  } else {
    for (int64_t t = 0; t < T; ++t)
      for (int64_t s = 0; s <= t; ++s) sel[static_cast<size_t>(t)].push_back(s);
  }

  // 5. attention with per-head sink softmax; value = the decoded latent (W3 seams).
  std::vector<float> o(static_cast<size_t>(T) * nh * hd, 0.0f);
  const float kNegInf = -std::numeric_limits<float>::infinity();
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<int64_t>& S = sel[static_cast<size_t>(t)];
    for (int64_t h = 0; h < nh; ++h) {
      std::vector<float> sc(S.size());
      const float* qh = &q[(t * nh + h) * hd];
      for (size_t si = 0; si < S.size(); ++si)
        sc[si] = Dot(qh, &deck[S[si] * hd], hd) * scale;
      const float sink = (miswire == V4Miswire::kNoAttnSink)
                             ? kNegInf
                             : L.attn_sink[static_cast<size_t>(h)];
      const std::vector<float> probs = SoftmaxWithSink(sc, sink);
      float* oh = &o[(t * nh + h) * hd];
      for (size_t si = 0; si < S.size(); ++si) {
        const float w = probs[si];
        const float* v = &deck[S[si] * hd];
        for (int64_t d = 0; d < hd; ++d) oh[d] += w * v[d];
      }
    }
  }

  // 6. grouped OUTPUT-LoRA (W3) : [T,nh,hd] -> [T,H].
  return deepseek_v4::GroupedOutputLora(o, L.wo_a, L.wo_b, T, nh, hd, p.o_groups,
                                        p.o_lora_rank, H);
}

// ── DeepSeek-V4 MoE block (W6 primitives) : [T,H] -> [T,H] ────────────────────
std::vector<float> MoeBlock(const DeepseekV4LayerHostWeights& L,
                            const DeepseekV4Params& p, const std::vector<float>& x,
                            const std::vector<int32_t>& token_ids, int64_t layer,
                            V4Miswire miswire, V4ForwardTrace* trace) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t ne = p.n_routed_experts;
  const int64_t topk = p.num_experts_per_tok;
  const int64_t mi = p.moe_intermediate_size;
  const float lim = static_cast<float>(p.swiglu_limit);
  const bool cfg_hash = p.is_hash_layer(layer);
  const bool hash_route = cfg_hash && miswire != V4Miswire::kAllLayersGated;

  // router gating logits [T, ne].
  std::vector<float> gating(static_cast<size_t>(T) * ne);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> g = MatVec(L.gate_weight, &x[t * H], ne, H);
    for (int64_t e = 0; e < ne; ++e) gating[t * ne + e] = g[static_cast<size_t>(e)];
  }
  std::vector<int64_t> in_tokens;
  std::vector<int32_t> hashtab;
  std::vector<float> bias;
  if (hash_route) {
    in_tokens.assign(token_ids.begin(), token_ids.end());
    hashtab = L.tid2eid;
  } else {
    bias = L.gate_bias;  // may be empty (then plain top-k on the unbiased scores)
  }
  const MoeRouteResult route =
      SqrtSoftplusRouteTopk(gating, T, ne, topk, bias, p.norm_topk_prob,
                            static_cast<float>(p.routed_scaling_factor), in_tokens,
                            hashtab, p.vocab_size);
  if (trace != nullptr) {
    trace->layer_is_hash[static_cast<size_t>(layer)] = cfg_hash ? 1 : 0;
    trace->layer_hash_routed[static_cast<size_t>(layer)] = hash_route ? 1 : 0;
  }

  // one clamped-SwiGLU expert: w1/w3 [mi,H], w2 [H,mi].
  const auto expert = [&](const float* w1, const float* w3, const float* w2,
                          const float* xin) -> std::vector<float> {
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) {
      gate_up[static_cast<size_t>(r)] = Dot(&w1[r * H], xin, H);
      gate_up[static_cast<size_t>(mi + r)] = Dot(&w3[r * H], xin, H);
    }
    const std::vector<float> act = ClampedSwiGLU(gate_up, mi, lim, 1.0f, 0.0f);
    std::vector<float> out(static_cast<size_t>(H));
    for (int64_t hh = 0; hh < H; ++hh)
      out[static_cast<size_t>(hh)] = Dot(&w2[hh * mi], act.data(), mi);
    return out;
  };

  std::vector<float> out(static_cast<size_t>(T) * H, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    // shared expert (always active).
    const std::vector<float> sh =
        expert(L.shared_w1.data(), L.shared_w3.data(), L.shared_w2.data(), &x[t * H]);
    for (int64_t hh = 0; hh < H; ++hh) out[t * H + hh] += sh[static_cast<size_t>(hh)];
    // routed experts.
    for (int64_t j = 0; j < topk; ++j) {
      const int64_t e = route.topk_ids[t * topk + j];
      const float w = route.topk_weights[t * topk + j];
      const std::vector<float> eo = expert(&L.exp_w1[e * mi * H], &L.exp_w3[e * mi * H],
                                            &L.exp_w2[e * H * mi], &x[t * H]);
      for (int64_t hh = 0; hh < H; ++hh) out[t * H + hh] += w * eo[static_cast<size_t>(hh)];
    }
  }
  return out;
}

constexpr const char* kHostPending =
    "DeepseekV4 forward: host-float weight tower not materialized — the W7 "
    "tiny-config CPU composition runs off DeepseekV4Weights::host (populated by the "
    "structural gate, test_deepseek_v4_forward.cpp); the real-checkpoint FP8-block + "
    "NVFP4 tower materialization is the named W2b residual and the device forward is "
    "W7-device. See .agents/specs/deepseek-v4-flash.md §W7.";

constexpr const char* kDevicePending =
    "DeepseekV4 DEVICE forward (W7-device) not implemented — the tiny-config CPU "
    "composition lands in DeepseekV4Model::Forward / DeepseekV4ForwardHost; the CUDA "
    "kernels (MHC Sinkhorn, DSA indexer/compressor, sqrtsoftplus router, clamped "
    "SwiGLU; the expert GEMM REUSES the existing NVFP4/FP8 grouped-GEMM) + the real "
    "multi-Spark e2e (W8) are named residuals. See .agents/specs/deepseek-v4-flash.md §W7.";

}  // namespace

// ── the W7 composition (host, tiny-config; also the device kernels' oracle) ───
std::vector<float> DeepseekV4ForwardHost(const DeepseekV4HostWeights& hw,
                                         const DeepseekV4Params& p,
                                         const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const std::vector<int32_t>& logits_indices,
                                         V4Miswire miswire, V4ForwardTrace* trace) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t hc = p.hc_mult;
  const int64_t nlayers = p.num_hidden_layers;
  const int64_t V = p.vocab_size;
  const float eps = p.rms_norm_eps;
  const float hc_eps = static_cast<float>(p.hc_eps);
  const int64_t iters = p.hc_sinkhorn_iters;

  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "positions/token_ids length mismatch");
  VT_CHECK(static_cast<int64_t>(hw.layers.size()) == nlayers, "host layer count mismatch");
  VT_CHECK(hc > 0 && H > 0 && nlayers > 0, "degenerate config");

  if (trace != nullptr) {
    trace->hc_mult = hc;
    trace->hidden = H;
    trace->num_tokens = T;
    trace->residual_stream_elems = T * hc * H;
    trace->layer_is_hash.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_hash_routed.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_is_indexer.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_indexer_selected.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_compressor_ran.assign(static_cast<size_t>(nlayers), 0);
  }

  // embed lookup -> the [T,H] token hidden stream.
  std::vector<float> x(static_cast<size_t>(T) * H);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = token_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < V, "token id out of range");
    for (int64_t h = 0; h < H; ++h) x[t * H + h] = hw.embed[tok * H + h];
  }

  // MHC residual manifold [T,hc,H] + the per-token post/comb mixes.
  std::vector<float> residual(static_cast<size_t>(T) * hc * H, 0.0f);
  std::vector<float> post_mix(static_cast<size_t>(T) * hc, 0.0f);
  std::vector<float> res_mix(static_cast<size_t>(T) * hc * hc, 0.0f);
  bool have_residual = false;

  for (int64_t layer = 0; layer < nlayers; ++layer) {
    const DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(layer)];

    // ── attn sub-block MHC-pre: first layer BROADCAST-expands [T,H] -> [T,hc,H];
    //    subsequent layers fuse MhcPost(prev-ffn-out) + MhcPre(attn) (model.py:878-933).
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> res_t(static_cast<size_t>(hc) * H);
      if (!have_residual) {
        for (int64_t i = 0; i < hc; ++i)
          for (int64_t h = 0; h < H; ++h) res_t[i * H + h] = x[t * H + h];
      } else {
        res_t = MhcPost(Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                        Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc),
                        hc, H);
      }
      const MhcPreResult pre =
          MhcPre(res_t, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, hc, H, eps, hc_eps,
                 hc_eps, 2.0f, iters, L.attn_norm_weight, eps);
      for (int64_t i = 0; i < hc * H; ++i)
        residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc * hc; ++i)
        res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
      for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
    }
    have_residual = true;

    // ── 512-wide MLA attention (W3+W4).
    x = AttentionBlock(L, p, x, positions, layer, miswire, trace);

    // ── ffn sub-block MHC fused-post-pre = MhcPost(attn-out) + MhcPre(ffn).
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> res_t =
          MhcPost(Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                  Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc), hc, H);
      const MhcPreResult pre =
          MhcPre(res_t, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, hc, H, eps, hc_eps,
                 hc_eps, 2.0f, iters, L.ffn_norm_weight, eps);
      for (int64_t i = 0; i < hc * H; ++i)
        residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc * hc; ++i)
        res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
      for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
    }

    // ── DeepSeek-V4 MoE (W6).
    x = MoeBlock(L, p, x, token_ids, layer, miswire, trace);
  }

  // final MhcPost(last-ffn-out) -> hc_head collapse -> norm -> lm_head (model.py:1128-1146).
  std::vector<float> hidden(static_cast<size_t>(T) * H);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> res_t;
    if (miswire == V4Miswire::kSkipFinalMhcPost) {
      res_t = Slice(residual, t * hc * H, hc * H);  // skip the fold (RED-first)
    } else {
      res_t = MhcPost(Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                      Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc), hc,
                      H);
    }
    std::vector<float> h = HcHeadCollapse(res_t, hw.hc_head_fn, hw.hc_head_scale,
                                          hw.hc_head_base, hc, H, eps, hc_eps);
    h = RmsNorm(h, hw.final_norm_weight, eps);
    for (int64_t d = 0; d < H; ++d) hidden[t * H + d] = h[static_cast<size_t>(d)];
  }

  // gather the requested rows (all rows if logits_indices empty) and project.
  std::vector<int32_t> rows = logits_indices;
  if (rows.empty()) {
    rows.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) rows[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  std::vector<float> logits(rows.size() * static_cast<size_t>(V));
  for (size_t ri = 0; ri < rows.size(); ++ri) {
    const int64_t r = rows[ri];
    VT_CHECK(r >= 0 && r < T, "logits index out of range");
    for (int64_t v = 0; v < V; ++v)
      logits[ri * V + v] = Dot(&hw.lm_head[v * H], &hidden[r * H], H);
  }
  return logits;
}

std::vector<float> DeepseekV4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;
  (void)queue;
  VT_CHECK(weights.has_host_weights, kHostPending);
  return DeepseekV4ForwardHost(weights.host, weights.params, token_ids, positions,
                               logits_indices);
}

ForwardLogits DeepseekV4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false, kDevicePending);
  return {};
}

}  // namespace vllm
