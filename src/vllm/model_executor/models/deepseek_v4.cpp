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
#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/deepseek_v4_moe.h"
#include "vt/dtype.h"
#include "vt/ops.h"      // vt::MatmulBT (auto-dispatches kMatmulBTQuant on block weights)
#include "vt/tensor.h"   // vt::Tensor::Contiguous

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

// ── backend policy: HOST refs (the oracle) OR the W7-device CUDA kernels ───────
// The composition below is written ONCE and run either on the portable host
// references (DeepseekV4ForwardHost, the oracle the device kernels are gated
// against) or on the CUDA kernels through the OpProvider seam
// (DeepseekV4Model::ForwardDevice). Only the four NEW V4 op families
// (MHC / DSA indexer+seams / compressor+fp8_ds_mla / sqrtsoftplus-hash MoE +
// clamped SwiGLU) branch; the small linear projections stay host in both modes
// (in the real device path they REUSE the existing GEMM/MLA/MoE-grouped kernels —
// a documented W7 seam, not re-ported here). device==host at the tiny structural
// shape is the ForwardDevice composition gate (test_cuda_deepseek_v4.cpp).
// ── W2C: keep-quant weight SOURCE ─────────────────────────────────────────────
// When `gguf != nullptr` the big MLA/MoE/lm_head GEMMs consume the keep-quant
// `weights.gguf` OwnedTensor blocks DIRECTLY via vt::MatmulBT (which dispatches to
// the landed CPU kMatmulBTQuant CIQ GEMM for a block-quant weight) — NO per-layer
// f32 tower. The small non-GEMM tensors (norms, sinks, MHC/DSA mixing, ape, the
// hash table, embed) still come from the SMALL `hw` host tower, dequant-f32 exactly
// as our other GGUF models keep them (qwen3_5_gguf_weights.cpp). When `gguf ==
// nullptr` every GEMM reads the f32 `hw` tower (the safetensors/NVFP4 + the tiny
// synthetic structural gate), byte-for-byte the pre-W2C behavior. `device` selects
// the CUDA V4-primitive kernels for the four NEW op families (orthogonal to the
// weight source; the GGUF keep-quant path runs device=false, CPU).
struct V4Backend {
  bool device = false;
  vt::Queue* q = nullptr;
  const DeepseekV4GgufWeights* gguf = nullptr;
};

deepseek_v4::MhcPreResult DispMhcPre(const V4Backend& be, const std::vector<float>& residual,
                                     const std::vector<float>& fn,
                                     const std::vector<float>& scale,
                                     const std::vector<float>& base, int64_t hc, int64_t hidden,
                                     float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                     float hc_post_mult, int64_t iters,
                                     const std::vector<float>& norm_weight, float norm_eps) {
  if (be.device)
    return deepseek_v4::MhcDevice()->pre(*be.q, residual, fn, scale, base, hc, hidden, rms_eps,
                                         hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, iters,
                                         norm_weight, norm_eps);
  return MhcPre(residual, fn, scale, base, hc, hidden, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult, iters, norm_weight, norm_eps);
}
std::vector<float> DispMhcPost(const V4Backend& be, const std::vector<float>& x,
                               const std::vector<float>& residual,
                               const std::vector<float>& post_mix,
                               const std::vector<float>& comb, int64_t hc, int64_t hidden) {
  if (be.device) return deepseek_v4::MhcDevice()->post(*be.q, x, residual, post_mix, comb, hc, hidden);
  return MhcPost(x, residual, post_mix, comb, hc, hidden);
}
std::vector<float> DispHcHead(const V4Backend& be, const std::vector<float>& x,
                              const std::vector<float>& fn, float scale,
                              const std::vector<float>& base, int64_t hc, int64_t hidden,
                              float rms_eps, float hc_eps) {
  if (be.device) return deepseek_v4::MhcDevice()->head(*be.q, x, fn, scale, base, hc, hidden, rms_eps, hc_eps);
  return HcHeadCollapse(x, fn, scale, base, hc, hidden, rms_eps, hc_eps);
}
std::vector<float> DispSaveScoreApe(const V4Backend& be, const std::vector<float>& score,
                                    const std::vector<float>& ape,
                                    const std::vector<int64_t>& positions, int64_t T,
                                    int64_t width, int64_t cr) {
  if (be.device) return deepseek_v4::CompressorDevice()->save_score_ape(*be.q, score, ape, positions, T, width, cr);
  return CompressorSaveScoreApe(score, ape, positions, T, width, cr);
}
std::vector<float> DispPoolNorm(const V4Backend& be, const std::vector<float>& kv,
                                const std::vector<float>& score,
                                const std::vector<uint8_t>& valid,
                                const std::vector<float>& rms_w, float eps, int64_t window,
                                int64_t hd) {
  if (be.device) return deepseek_v4::CompressorDevice()->pool_norm(*be.q, kv, score, valid, rms_w, eps, window, hd);
  return CompressorPoolNorm(kv, score, valid, rms_w, eps, window, hd);
}
deepseek_v4::Fp8DsMlaToken DispEncode(const V4Backend& be, const std::vector<float>& head,
                                      const Fp8DsMlaLayout& ly) {
  if (be.device) return deepseek_v4::CompressorDevice()->encode(*be.q, head, ly);
  return Fp8DsMlaEncodeToken(head, ly);
}
std::vector<float> DispDecode(const V4Backend& be, const deepseek_v4::Fp8DsMlaToken& tok,
                              const Fp8DsMlaLayout& ly) {
  if (be.device) return deepseek_v4::CompressorDevice()->decode(*be.q, tok, ly);
  return Fp8DsMlaDecodeToken(tok, ly);
}
std::vector<float> DispWeightFold(const V4Backend& be, const std::vector<float>& wp, int64_t T,
                                  int64_t inh, int64_t ihd) {
  if (be.device) return deepseek_v4::DsaDevice()->weight_fold(*be.q, wp, T, inh, ihd);
  return DsaIndexerWeightFold(wp, T, inh, ihd);
}
std::vector<float> DispLogits(const V4Backend& be, const std::vector<float>& q,
                              const std::vector<float>& k, const std::vector<float>& folded,
                              const std::vector<int64_t>& ws, const std::vector<int64_t>& we,
                              int64_t T, int64_t nk, int64_t inh, int64_t ihd) {
  if (be.device) return deepseek_v4::DsaDevice()->logits(*be.q, q, k, folded, ws, we, T, nk, inh, ihd);
  return DsaIndexerLogits(q, k, folded, ws, we, T, nk, inh, ihd);
}
std::vector<int64_t> DispTopk(const V4Backend& be, const std::vector<float>& logits,
                              const std::vector<int64_t>& ws, const std::vector<int64_t>& we,
                              int64_t T, int64_t nk, int64_t topk) {
  if (be.device) return deepseek_v4::DsaDevice()->topk(*be.q, logits, ws, we, T, nk, topk);
  return DsaTopkSelect(logits, ws, we, T, nk, topk);
}
std::vector<float> DispSoftmaxSink(const V4Backend& be, const std::vector<float>& scores,
                                   float sink) {
  if (be.device) return deepseek_v4::DsaDevice()->softmax_sink(*be.q, scores, sink);
  return SoftmaxWithSink(scores, sink);
}
std::vector<float> DispGroupedOLora(const V4Backend& be, const std::vector<float>& o,
                                    const std::vector<float>& wo_a,
                                    const std::vector<float>& wo_b, int64_t T, int64_t nh,
                                    int64_t hd, int64_t ng, int64_t olr, int64_t H) {
  if (be.device) return deepseek_v4::DsaDevice()->grouped_olora(*be.q, o, wo_a, wo_b, T, nh, hd, ng, olr, H);
  return deepseek_v4::GroupedOutputLora(o, wo_a, wo_b, T, nh, hd, ng, olr, H);
}
deepseek_v4::MoeRouteResult DispRoute(const V4Backend& be, const std::vector<float>& gating,
                                      int64_t T, int64_t E, int64_t topk,
                                      const std::vector<float>& bias, bool renorm, float scale,
                                      const std::vector<int64_t>& in_tokens,
                                      const std::vector<int32_t>& hashtab, int64_t vocab) {
  if (be.device)
    return deepseek_v4::MoeDevice()->route(*be.q, gating, T, E, topk, bias, renorm, scale,
                                           in_tokens, hashtab, vocab);
  return SqrtSoftplusRouteTopk(gating, T, E, topk, bias, renorm, scale, in_tokens, hashtab, vocab);
}
std::vector<float> DispClampedSwiGLU(const V4Backend& be, const std::vector<float>& gate_up,
                                     int64_t d, float limit, float alpha, float beta) {
  if (be.device) return deepseek_v4::MoeDevice()->clamped_swiglu(*be.q, gate_up, d, limit, alpha, beta);
  return ClampedSwiGLU(gate_up, d, limit, alpha, beta);
}

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

// ── W2C keep-quant GEMM: Y[T,N] = X[T,K] @ W[N,K]^T ───────────────────────────
// `wq` (the keep-quant / bf16 OwnedTensor block, [N,K] nk=true as on GGUF disk) is
// consumed IN PLACE via vt::MatmulBT — a block-quant dtype routes to the CPU
// kMatmulBTQuant CIQ GEMM (cpu_quant_gemm.cpp, quantizes the activation once then
// integer vec_dot per output, weights never expanded); a bf16 dtype (the expand
// oracle) routes to the elementwise MatmulBT. When `wq` is null/absent (host
// source) it falls back to the per-row f32 MatVec — BIT-IDENTICAL to the pre-W2C
// host composition. Grounded in qwen3_5.cpp:786-838 (host MatmulBT off an
// OwnedTensor.View()) + vt/ops.cpp:134-171 (block-quant dispatch).
std::vector<float> Gemm(const V4Backend& be, const OwnedTensor* wq,
                        const std::vector<float>& wf32, const std::vector<float>& x,
                        int64_t T, int64_t N, int64_t K) {
  if (be.gguf != nullptr && wq != nullptr && !wq->Empty()) {
    VT_CHECK(be.q != nullptr, "deepseek-v4 keep-quant GEMM needs a queue");
    VT_CHECK(wq->rank == 2 && wq->shape[0] == N && wq->shape[1] == K,
             "deepseek-v4 keep-quant GEMM: weight shape mismatch: want [N=" +
                 std::to_string(N) + ",K=" + std::to_string(K) + "] got [" +
                 std::to_string(wq->shape[0]) + "," + std::to_string(wq->shape[1]) +
                 "] rank=" + std::to_string(wq->rank));
    std::vector<float> out(static_cast<size_t>(T) * N);
    vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                          vt::DType::kF32, be.q->device, {T, K});
    vt::Tensor o =
        vt::Tensor::Contiguous(out.data(), vt::DType::kF32, be.q->device, {T, N});
    vt::Tensor w = wq->View();
    vt::MatmulBT(*be.q, o, a, w);
    return out;
  }
  std::vector<float> out(static_cast<size_t>(T) * N);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> y = MatVec(wf32, &x[t * K], N, K);
    for (int64_t n = 0; n < N; ++n) out[t * N + n] = y[static_cast<size_t>(n)];
  }
  return out;
}

// Keep-quant GEMM against a ROW-SLICE [row_off, row_off+N) of a stacked block
// weight `w` ([E*out, K] nk=true) — the per-expert (moe_*_exps) / per-group (wo_a)
// slice. Rows are whole blocks (RowSizeBytes), so the offset is a byte offset and
// no block is ever cut (mirrors the loader's OwnGgufQuantBlocks row_offset slice,
// qwen3_5_gguf_weights.cpp:57-101, and the kStackedExpertWeight contract). Returns
// [T,N] f32.
std::vector<float> GemmRowSlice(const V4Backend& be, const OwnedTensor& w,
                                const std::vector<float>& x, int64_t T, int64_t N,
                                int64_t K, int64_t row_off) {
  VT_CHECK(be.q != nullptr, "deepseek-v4 keep-quant expert GEMM needs a queue");
  VT_CHECK(!w.repacked,
           "deepseek-v4 keep-quant expert/group slice requires non-repacked blocks "
           "(disable VT_CPU_QUANT_REPACK for the stacked-expert weights)");
  VT_CHECK(w.rank == 2 && row_off >= 0 && row_off + N <= w.shape[0] && w.shape[1] == K,
           "deepseek-v4 keep-quant expert GEMM: slice out of range");
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  std::vector<float> out(static_cast<size_t>(T) * N);
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()), vt::DType::kF32,
                                        be.q->device, {T, K});
  vt::Tensor o =
      vt::Tensor::Contiguous(out.data(), vt::DType::kF32, be.q->device, {T, N});
  vt::Tensor wt;
  wt.data = const_cast<uint8_t*>(w.bytes.data()) +
            static_cast<size_t>(row_off) * row_bytes;
  wt.dtype = w.dtype;
  wt.device = be.q->device;
  wt.rank = 2;
  wt.shape[0] = N;
  wt.shape[1] = K;
  wt.stride[0] = K;  // inert for a block-quant weight; correct for the bf16 oracle
  wt.stride[1] = 1;
  vt::MatmulBT(*be.q, o, a, wt);
  return out;
}

// Grouped OUTPUT-LoRA on the keep-quant tower (the GGUF mirror of
// deepseek_v4::GroupedOutputLora): z[t, g*olr+d] = Σ_r wo_a[g,d,r]·o[t,g,r]
// (per-group block-diagonal, so one row-slice quant GEMM per group), then
// out[t] = wo_b @ z. wo_a keep-quant [ng*olr, in_per_group], wo_b keep-quant
// [H, ng*olr]. o_proj.py:58-73.
std::vector<float> GroupedOutputLoraGguf(const V4Backend& be, const OwnedTensor& wo_a,
                                         const OwnedTensor& wo_b,
                                         const std::vector<float>& o, int64_t T,
                                         int64_t nh, int64_t hd, int64_t ng,
                                         int64_t olr, int64_t H) {
  VT_CHECK(ng > 0 && nh % ng == 0, "grouped o-LoRA: n_heads % n_groups != 0");
  const int64_t ipg = nh * hd / ng;  // in_per_group
  const int64_t z_dim = ng * olr;
  std::vector<float> z(static_cast<size_t>(T) * z_dim);
  std::vector<float> og(static_cast<size_t>(T) * ipg);
  for (int64_t g = 0; g < ng; ++g) {
    for (int64_t t = 0; t < T; ++t)
      for (int64_t r = 0; r < ipg; ++r)
        og[t * ipg + r] = o[t * nh * hd + g * ipg + r];
    const std::vector<float> zg =
        GemmRowSlice(be, wo_a, og, T, olr, ipg, /*row_off=*/g * olr);  // [T,olr]
    for (int64_t t = 0; t < T; ++t)
      for (int64_t d = 0; d < olr; ++d) z[t * z_dim + g * olr + d] = zg[t * olr + d];
  }
  return Gemm(be, &wo_b, /*wf32=*/{}, z, T, H, z_dim);  // [T,H]
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
                                  const DeepseekV4GgufLayerWeights* Lq,
                                  const DeepseekV4Params& p,
                                  const std::vector<float>& x,
                                  const std::vector<int32_t>& positions, int64_t layer,
                                  V4Miswire miswire, V4ForwardTrace* trace,
                                  const V4Backend& be) {
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

  // 1. q [T,nh,hd] and raw kv latent [T,hd] (num_key_value_heads=1 MLA). The MLA
  //    linears (wq_a, wq_b, wkv) run the keep-quant GEMM (Gemm) — the whole batch
  //    at once — then the per-token RMSNorm(q_norm/kv_norm) + per-head RoPE.
  std::vector<float> qa = Gemm(be, Lq != nullptr ? &Lq->wq_a : nullptr, L.wq_a, x, T, qlr, H);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> n = RmsNorm(Slice(qa, t * qlr, qlr), L.q_norm_weight, eps);
    for (int64_t i = 0; i < qlr; ++i) qa[t * qlr + i] = n[static_cast<size_t>(i)];
  }
  std::vector<float> q =
      Gemm(be, Lq != nullptr ? &Lq->wq_b : nullptr, L.wq_b, qa, T, nh * hd, qlr);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h)
      RopeInplace(&q[t * nh * hd + h * hd + nope], rope,
                  positions[static_cast<size_t>(t)], p.rope_theta);
  std::vector<float> kraw = Gemm(be, Lq != nullptr ? &Lq->wkv : nullptr, L.wkv, x, T, hd, H);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> kv = RmsNorm(Slice(kraw, t * hd, hd), L.kv_norm_weight, eps);
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
    // compressor pool-score projection (keep-quant comp_wgate) : [T,H] -> [T,hd].
    std::vector<float> score =
        Gemm(be, Lq != nullptr ? &Lq->comp_wgate : nullptr, L.comp_wgate, x, T, hd, H);
    std::vector<int64_t> pos64(positions.begin(), positions.end());
    score = DispSaveScoreApe(be, score, L.comp_ape, pos64, T, hd, cr);
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
          DispPoolNorm(be, kvwin, scwin, valid, L.comp_norm_weight, eps, win, hd);
      for (int64_t d = 0; d < hd; ++d) latent[t * hd + d] = comp[static_cast<size_t>(d)];
    }
    if (trace != nullptr) trace->layer_compressor_ran[static_cast<size_t>(layer)] = 1;
  }

  // 3. fp8_ds_mla KV-state round-trip — EXERCISE the paged cache layout (W4).
  const Fp8DsMlaLayout ly = MakeFp8DsMlaLayout(nope, rope, /*quant_block=*/nope);
  std::vector<float> deck(static_cast<size_t>(T) * hd);
  for (int64_t t = 0; t < T; ++t) {
    const auto tok = DispEncode(be, Slice(latent, t * hd, hd), ly);
    const std::vector<float> dec = DispDecode(be, tok, ly);
    for (int64_t d = 0; d < hd; ++d) deck[t * hd + d] = dec[static_cast<size_t>(d)];
  }

  // 4. selection: DSA Lightning-Indexer top-k on indexer layers, else dense causal.
  std::vector<std::vector<int64_t>> sel(static_cast<size_t>(T));
  if (is_indexer) {
    const int64_t inh = p.index_n_heads, ihd = p.index_head_dim, itopk = p.index_topk;
    // indexer q/k projections keep-quant (idx_wq_b / indexer_compressor_kv); the
    // weights_proj (idx_wproj) is a small V role and stays f32 (host).
    const std::vector<float> iq =
        Gemm(be, Lq != nullptr ? &Lq->idx_wq_b : nullptr, L.idx_wq, x, T, inh * ihd, H);
    const std::vector<float> ik =
        Gemm(be, Lq != nullptr ? &Lq->idx_comp_wkv : nullptr, L.idx_wk, x, T, ihd, H);
    const std::vector<float> wproj = Gemm(be, nullptr, L.idx_wproj, x, T, inh, H);
    const std::vector<float> folded = DispWeightFold(be, wproj, T, inh, ihd);
    std::vector<int64_t> ws(static_cast<size_t>(T)), we(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      ws[static_cast<size_t>(t)] = 0;
      we[static_cast<size_t>(t)] = t + 1;  // causal candidate window
    }
    const std::vector<float> logits =
        DispLogits(be, iq, ik, folded, ws, we, T, T, inh, ihd);
    const std::vector<int64_t> topk = DispTopk(be, logits, ws, we, T, T, itopk);
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
      const std::vector<float> probs = DispSoftmaxSink(be, sc, sink);
      float* oh = &o[(t * nh + h) * hd];
      for (size_t si = 0; si < S.size(); ++si) {
        const float w = probs[si];
        const float* v = &deck[S[si] * hd];
        for (int64_t d = 0; d < hd; ++d) oh[d] += w * v[d];
      }
    }
  }

  // 6. grouped OUTPUT-LoRA (W3) : [T,nh,hd] -> [T,H]. Keep-quant wo_a/wo_b on the
  //    GGUF source; the host/device-synthetic path keeps the f32 primitive.
  if (be.gguf != nullptr && Lq != nullptr) {
    return GroupedOutputLoraGguf(be, Lq->wo_a, Lq->wo_b, o, T, nh, hd, p.o_groups,
                                 p.o_lora_rank, H);
  }
  return DispGroupedOLora(be, o, L.wo_a, L.wo_b, T, nh, hd, p.o_groups, p.o_lora_rank, H);
}

// ── DeepSeek-V4 MoE block (W6 primitives) : [T,H] -> [T,H] ────────────────────
std::vector<float> MoeBlock(const DeepseekV4LayerHostWeights& L,
                            const DeepseekV4GgufLayerWeights* Lq,
                            const DeepseekV4Params& p, const std::vector<float>& x,
                            const std::vector<int32_t>& token_ids, int64_t layer,
                            V4Miswire miswire, V4ForwardTrace* trace, const V4Backend& be) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t ne = p.n_routed_experts;
  const int64_t topk = p.num_experts_per_tok;
  const int64_t mi = p.moe_intermediate_size;
  const float lim = static_cast<float>(p.swiglu_limit);
  const bool cfg_hash = p.is_hash_layer(layer);
  const bool hash_route = cfg_hash && miswire != V4Miswire::kAllLayersGated;
  const bool kq = be.gguf != nullptr && Lq != nullptr;

  // router gating logits [T, ne] (keep-quant moe_gate).
  const std::vector<float> gating =
      Gemm(be, kq ? &Lq->moe_gate : nullptr, L.gate_weight, x, T, ne, H);
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
      DispRoute(be, gating, T, ne, topk, bias, p.norm_topk_prob,
                static_cast<float>(p.routed_scaling_factor), in_tokens, hashtab, p.vocab_size);
  if (trace != nullptr) {
    trace->layer_is_hash[static_cast<size_t>(layer)] = cfg_hash ? 1 : 0;
    trace->layer_hash_routed[static_cast<size_t>(layer)] = hash_route ? 1 : 0;
  }

  // one clamped-SwiGLU expert on the f32 host tower: w1/w3 [mi,H], w2 [H,mi].
  const auto expert_f32 = [&](const float* w1, const float* w3, const float* w2,
                              const float* xin) -> std::vector<float> {
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) {
      gate_up[static_cast<size_t>(r)] = Dot(&w1[r * H], xin, H);
      gate_up[static_cast<size_t>(mi + r)] = Dot(&w3[r * H], xin, H);
    }
    const std::vector<float> act = DispClampedSwiGLU(be, gate_up, mi, lim, 1.0f, 0.0f);
    std::vector<float> out(static_cast<size_t>(H));
    for (int64_t hh = 0; hh < H; ++hh)
      out[static_cast<size_t>(hh)] = Dot(&w2[hh * mi], act.data(), mi);
    return out;
  };
  // clamped-SwiGLU over a keep-quant expert e: w13 = stacked [E*mi,H] row-slice at
  // e*mi, w2 = stacked [E*H,mi] row-slice at e*H. `x1` is the single token [1,H].
  const auto expert_kq = [&](const OwnedTensor& w1s, const OwnedTensor& w3s,
                             const OwnedTensor& w2s, int64_t e,
                             const std::vector<float>& x1) -> std::vector<float> {
    const std::vector<float> g = GemmRowSlice(be, w1s, x1, 1, mi, H, e * mi);
    const std::vector<float> u = GemmRowSlice(be, w3s, x1, 1, mi, H, e * mi);
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) {
      gate_up[static_cast<size_t>(r)] = g[static_cast<size_t>(r)];
      gate_up[static_cast<size_t>(mi + r)] = u[static_cast<size_t>(r)];
    }
    const std::vector<float> act = DispClampedSwiGLU(be, gate_up, mi, lim, 1.0f, 0.0f);
    return GemmRowSlice(be, w2s, act, 1, H, mi, e * H);  // [H]
  };
  // the shared expert is a plain (non-stacked) keep-quant triple.
  const auto shared_kq = [&](const std::vector<float>& x1) -> std::vector<float> {
    const std::vector<float> g = Gemm(be, &Lq->shared_gate, {}, x1, 1, mi, H);
    const std::vector<float> u = Gemm(be, &Lq->shared_up, {}, x1, 1, mi, H);
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) {
      gate_up[static_cast<size_t>(r)] = g[static_cast<size_t>(r)];
      gate_up[static_cast<size_t>(mi + r)] = u[static_cast<size_t>(r)];
    }
    const std::vector<float> act = DispClampedSwiGLU(be, gate_up, mi, lim, 1.0f, 0.0f);
    return Gemm(be, &Lq->shared_down, {}, act, 1, H, mi);  // [H]
  };

  std::vector<float> out(static_cast<size_t>(T) * H, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> x1(x.begin() + t * H, x.begin() + (t + 1) * H);
    // shared expert (always active).
    const std::vector<float> sh =
        kq ? shared_kq(x1)
           : expert_f32(L.shared_w1.data(), L.shared_w3.data(), L.shared_w2.data(),
                        &x[t * H]);
    for (int64_t hh = 0; hh < H; ++hh) out[t * H + hh] += sh[static_cast<size_t>(hh)];
    // routed experts.
    for (int64_t j = 0; j < topk; ++j) {
      const int64_t e = route.topk_ids[t * topk + j];
      const float w = route.topk_weights[t * topk + j];
      const std::vector<float> eo =
          kq ? expert_kq(Lq->moe_gate_exps, Lq->moe_up_exps, Lq->moe_down_exps, e, x1)
             : expert_f32(&L.exp_w1[e * mi * H], &L.exp_w3[e * mi * H],
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

// ── the W7 composition, written ONCE and run on HOST refs OR the device kernels
//    (backend policy `be`). DeepseekV4ForwardHost binds host; ForwardDevice binds
//    the CUDA kernels through the seam. ───────────────────────────────────────
static std::vector<float> ForwardComposeImpl(const DeepseekV4HostWeights& hw,
                                             const DeepseekV4Params& p,
                                             const std::vector<int32_t>& token_ids,
                                             const std::vector<int32_t>& positions,
                                             const std::vector<int32_t>& logits_indices,
                                             V4Miswire miswire, V4ForwardTrace* trace,
                                             const V4Backend& be) {
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

  // Keep-quant weight source (GGUF): the big MLA/MoE/lm_head GEMMs read the
  // compressed `be.gguf` blocks; small tensors stay in the f32 `hw` tower. Null
  // on the safetensors/NVFP4 + tiny-synthetic host path (every GEMM reads `hw`).
  const bool kq_src = be.gguf != nullptr;
  if (kq_src)
    VT_CHECK(static_cast<int64_t>(be.gguf->layers.size()) == nlayers,
             "deepseek-v4 keep-quant: gguf layer count mismatch");

  for (int64_t layer = 0; layer < nlayers; ++layer) {
    const DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(layer)];
    const DeepseekV4GgufLayerWeights* Lq =
        kq_src ? &be.gguf->layers[static_cast<size_t>(layer)] : nullptr;

    // ── attn sub-block MHC-pre: first layer BROADCAST-expands [T,H] -> [T,hc,H];
    //    subsequent layers fuse MhcPost(prev-ffn-out) + MhcPre(attn) (model.py:878-933).
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> res_t(static_cast<size_t>(hc) * H);
      if (!have_residual) {
        for (int64_t i = 0; i < hc; ++i)
          for (int64_t h = 0; h < H; ++h) res_t[i * H + h] = x[t * H + h];
      } else {
        res_t = DispMhcPost(be, Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                            Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc),
                            hc, H);
      }
      const MhcPreResult pre =
          DispMhcPre(be, res_t, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, hc, H, eps,
                     hc_eps, hc_eps, 2.0f, iters, L.attn_norm_weight, eps);
      for (int64_t i = 0; i < hc * H; ++i)
        residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc * hc; ++i)
        res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
      for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
    }
    have_residual = true;

    // ── 512-wide MLA attention (W3+W4).
    x = AttentionBlock(L, Lq, p, x, positions, layer, miswire, trace, be);

    // ── ffn sub-block MHC fused-post-pre = MhcPost(attn-out) + MhcPre(ffn).
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> res_t =
          DispMhcPost(be, Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                      Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc), hc, H);
      const MhcPreResult pre =
          DispMhcPre(be, res_t, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, hc, H, eps, hc_eps,
                     hc_eps, 2.0f, iters, L.ffn_norm_weight, eps);
      for (int64_t i = 0; i < hc * H; ++i)
        residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc * hc; ++i)
        res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
      for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
    }

    // ── DeepSeek-V4 MoE (W6).
    x = MoeBlock(L, Lq, p, x, token_ids, layer, miswire, trace, be);
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
    std::vector<float> h = DispHcHead(be, res_t, hw.hc_head_fn, hw.hc_head_scale,
                                      hw.hc_head_base, hc, H, eps, hc_eps);
    h = RmsNorm(h, hw.final_norm_weight, eps);
    for (int64_t d = 0; d < H; ++d) hidden[t * H + d] = h[static_cast<size_t>(d)];
  }

  // gather the requested rows (all rows if logits_indices empty) then project
  // through the lm_head — keep-quant (GGUF) or f32 (host) via the same Gemm.
  std::vector<int32_t> rows = logits_indices;
  if (rows.empty()) {
    rows.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) rows[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  const int64_t R = static_cast<int64_t>(rows.size());
  std::vector<float> hsel(static_cast<size_t>(R) * H);
  for (int64_t ri = 0; ri < R; ++ri) {
    const int64_t r = rows[static_cast<size_t>(ri)];
    VT_CHECK(r >= 0 && r < T, "logits index out of range");
    for (int64_t d = 0; d < H; ++d) hsel[ri * H + d] = hidden[r * H + d];
  }
  const OwnedTensor* lmq = be.gguf != nullptr ? &be.gguf->lm_head : nullptr;
  return Gemm(be, lmq, hw.lm_head, hsel, R, V, H);
}

// Public host oracle: the composition on the portable host references.
std::vector<float> DeepseekV4ForwardHost(const DeepseekV4HostWeights& hw,
                                         const DeepseekV4Params& p,
                                         const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const std::vector<int32_t>& logits_indices,
                                         V4Miswire miswire, V4ForwardTrace* trace) {
  return ForwardComposeImpl(hw, p, token_ids, positions, logits_indices, miswire, trace,
                            V4Backend{/*device=*/false, /*q=*/nullptr, /*gguf=*/nullptr});
}

// W2C — the GGUF keep-quant forward. The SAME composition as the host oracle, but
// the big MLA/MoE/lm_head GEMMs consume the COMPRESSED `weights.gguf` blocks in
// place via vt::MatmulBT -> kMatmulBTQuant (no per-layer f32 tower). The small
// tensors (norms/sinks/MHC/DSA mixing/ape/hash/embed) come from the SMALL
// `weights.host` tower the GGUF loader still dequants. Requires a queue (the CPU
// quant GEMM consumer).
std::vector<float> DeepseekV4ForwardGguf(const DeepseekV4Weights& weights,
                                         vt::Queue& queue,
                                         const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const std::vector<int32_t>& logits_indices,
                                         V4Miswire miswire, V4ForwardTrace* trace) {
  VT_CHECK(weights.has_gguf_weights,
           "DeepseekV4ForwardGguf: no keep-quant tower (call LoadDeepseekV4FromGguf)");
  VT_CHECK(weights.has_host_weights,
           "DeepseekV4ForwardGguf: the small f32 host tower (norms/embed/mixing) is "
           "absent");
  return ForwardComposeImpl(
      weights.host, weights.params, token_ids, positions, logits_indices, miswire, trace,
      V4Backend{/*device=*/false, /*q=*/&queue, /*gguf=*/&weights.gguf});
}

std::vector<float> DeepseekV4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;
  // GGUF source: consume the keep-quant tower (memory-bounded — no ~1 TiB f32
  // tower). Safetensors/NVFP4 + the tiny-synthetic gate: the f32 host oracle.
  if (weights.has_gguf_weights) {
    return DeepseekV4ForwardGguf(weights, queue, token_ids, positions, logits_indices);
  }
  (void)queue;
  VT_CHECK(weights.has_host_weights, kHostPending);
  return DeepseekV4ForwardHost(weights.host, weights.params, token_ids, positions,
                               logits_indices);
}

// ── W7-DEVICE: the DEVICE forward. Runs the SAME composition as
//    DeepseekV4ForwardHost but routes the four NEW V4 op families through the
//    CUDA kernels (kDeepseekV4{Mhc,Dsa,Compressor,Moe}) via the OpProvider seam,
//    at the tiny structural config. device==host within near-tie is the
//    ForwardDevice composition gate (test_cuda_deepseek_v4.cpp). The small linear
//    projections stay host in both modes (the real path REUSES the existing
//    MLA/MoE-grouped/NVFP4 GEMM kernels — a documented W7 seam). The full paged
//    engine over a materialized 167B checkpoint is the W8 residual. ────────────
ForwardLogits DeepseekV4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;
  VT_CHECK(weights.has_host_weights, kHostPending);
  VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending);
  std::vector<float> flat = ForwardComposeImpl(
      weights.host, weights.params, token_ids, positions, logits_indices,
      V4Miswire::kNone, /*trace=*/nullptr,
      V4Backend{/*device=*/true, /*q=*/&queue, /*gguf=*/nullptr});
  ForwardLogits out;
  out.vocab = weights.params.vocab_size;
  out.rows = out.vocab > 0 ? static_cast<int64_t>(flat.size()) / out.vocab : 0;
  out.host = std::move(flat);
  return out;
}

}  // namespace vllm
