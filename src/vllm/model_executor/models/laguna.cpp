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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vllm/model_executor/models/laguna_ops.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/backend.h"  // vt::GetBackend (device drain for the keep-quant GEMMs)
#include "vt/dtype.h"    // vt::IsBlockQuant / RowSizeBytes
#include "vt/ops.h"      // vt::MatmulBT (dispatches kMatmulBTQuant on block weights)
#include "vt/tensor.h"

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

// Per-head RMSNorm over the head_dim axis (Laguna QK-norm, VERIFIED W4): x is
// [T, heads, Dh] flattened; each length-Dh head vector is RMS-normed with the
// shared weight w[Dh]. Variance in f32 (Qwen3/OLMo-2 qk_layernorm semantics).
void RmsNormHeads(std::vector<float>& x, const std::vector<float>& w, int64_t T,
                  int64_t heads, int64_t Dh, float eps) {
  for (int64_t r = 0; r < T * heads; ++r) {
    float* v = x.data() + static_cast<size_t>(r * Dh);
    float ss = 0.0F;
    for (int64_t d = 0; d < Dh; ++d) ss += v[d] * v[d];
    const float inv = 1.0F / std::sqrt(ss / static_cast<float>(Dh) + eps);
    for (int64_t d = 0; d < Dh; ++d) v[d] = v[d] * inv * w[static_cast<size_t>(d)];
  }
}

inline float Silu(float x) { return x / (1.0F + std::exp(-x)); }

// Separate gate/up SwiGLU: silu(gate)*up per element -> [T,I].
std::vector<float> GateUpSilu(const std::vector<float>& gate,
                              const std::vector<float>& up, int64_t T, int64_t I) {
  std::vector<float> act(static_cast<size_t>(T * I));
  for (int64_t i = 0; i < T * I; ++i)
    act[static_cast<size_t>(i)] = Silu(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
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

// One expert / shared-MLP forward from SEPARATE gate/up/down weights (the real
// GGUF name-map): silu(gate@h)*(up@h) then down. [H] out.
std::vector<float> ExpertMlp(const std::vector<float>& h_row,
                             const std::vector<float>& gate_w,
                             const std::vector<float>& up_w,
                             const std::vector<float>& down_w, int64_t H,
                             int64_t I) {
  const std::vector<float> g = MatmulNK(h_row, gate_w, 1, I, H);
  const std::vector<float> u = MatmulNK(h_row, up_w, 1, I, H);
  const std::vector<float> act = GateUpSilu(g, u, 1, I);
  return MatmulNK(act, down_w, 1, H, I);  // [H]
}

// ── W5 keep-quant GEMM (mirror deepseek_v4.cpp Gemm/GemmRowSlice) ─────────────
// Drain a device queue's stream before the host reads the output (no-op on CPU).
inline void DrainQueue(vt::Queue& q) {
  if (q.device.type != vt::DeviceType::kCPU)
    vt::GetBackend(q.device).Synchronize(q);
}

// Y[T,N] = X[T,K] @ W[N,K]^T. W is a keep-quant (block) OR f32/bf16 OwnedTensor in
// the file's own [N,K] order (nk=true, as on GGUF disk). A block-quant weight
// routes to vt::MatmulBT's kMatmulBTQuant provider (CPU always; CUDA when the
// queue is a device queue — reads the unified-memory blocks in place). An f32/bf16
// weight (the router, or a bf16 oracle expansion) falls to the host MatmulNK
// reference — BIT-IDENTICAL to the pre-W5 path. Mirrors deepseek_v4.cpp:410.
std::vector<float> LqGemm(vt::Queue& q, const OwnedTensor& w,
                          const std::vector<float>& x, int64_t T, int64_t N,
                          int64_t K) {
  VT_CHECK(w.rank == 2 && w.shape[0] == N && w.shape[1] == K,
           "laguna keep-quant GEMM: weight shape mismatch");
  if (!vt::IsBlockQuant(w.dtype)) return MatmulNK(x, ReadF32(w), T, N, K);
  std::vector<float> out(static_cast<size_t>(T) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                        vt::DType::kF32, gq.device, {T, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {T, N});
  vt::Tensor wt = w.View();
  wt.device = gq.device;  // unified-memory block view retag (ds4 ops.cpp:198 check)
  vt::MatmulBT(gq, o, a, wt);
  if (on_dev) DrainQueue(gq);
  return out;
}

// Keep-quant GEMM against a ROW-SLICE [row_off, row_off+N) of a stacked block
// weight `w` [E*out, K] — the per-expert (moe_*_exps) slice. Rows are whole
// blocks (RowSizeBytes), so the offset is a byte offset and no block is cut.
// Mirrors deepseek_v4.cpp GemmRowSlice.
std::vector<float> LqGemmRowSlice(vt::Queue& q, const OwnedTensor& w,
                                  const std::vector<float>& x, int64_t T, int64_t N,
                                  int64_t K, int64_t row_off) {
  VT_CHECK(vt::IsBlockQuant(w.dtype),
           "laguna keep-quant expert GEMM requires a block-quant stacked weight");
  VT_CHECK(!w.repacked,
           "laguna keep-quant expert slice requires non-repacked blocks");
  VT_CHECK(w.rank == 2 && row_off >= 0 && row_off + N <= w.shape[0] && w.shape[1] == K,
           "laguna keep-quant expert GEMM: slice out of range");
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  std::vector<float> out(static_cast<size_t>(T) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                        vt::DType::kF32, gq.device, {T, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {T, N});
  vt::Tensor wt;
  wt.data = const_cast<uint8_t*>(w.bytes.data()) +
            static_cast<size_t>(row_off) * row_bytes;
  wt.dtype = w.dtype;
  wt.device = gq.device;
  wt.rank = 2;
  wt.shape[0] = N;
  wt.shape[1] = K;
  wt.stride[0] = K;
  wt.stride[1] = 1;
  vt::MatmulBT(gq, o, a, wt);
  if (on_dev) DrainQueue(gq);
  return out;
}

// W8 lever #2 (grouped-expert GEMM): default-ON, `VT_LAGUNA_GROUPED_MOE=0` restores
// the byte-exact per-expert loop in the SAME binary. Mirrors ds4 GroupedMoeEnabled.
inline bool LagunaGroupedMoeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_GROUPED_MOE");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Grouped keep-quant expert GEMM over the stacked [E*N,K] tower: out[P,N] where
// out[p,:] = act[p,:] · weight[expert_ids[p]*N .. +N] (the block row-slice for that
// expert). ONE vt::MatmulBTQuantGrouped launch replaces P per-expert LqGemmRowSlice
// matvecs. BYTE-IDENTICAL to the per-expert path — the grouped op's CPU provider loops
// the EXACT kMatmulBTQuant per group; the CUDA provider is the same integer-dot core
// (ds4-gated byte-exact). Routes through the SHARED vt keep-quant grouped op (fold
// policy: no hand-rolled per-model kernel). Mirror of deepseek_v4.cpp
// GemmGroupedExpertsKq (drains before the local eids/out buffers leave scope).
std::vector<float> LqGemmGrouped(vt::Queue& q, const OwnedTensor& w,
                                 const std::vector<float>& act,
                                 const std::vector<int32_t>& expert_ids, int64_t P,
                                 int64_t N, int64_t K) {
  VT_CHECK(vt::IsBlockQuant(w.dtype) && !w.repacked,
           "laguna grouped expert GEMM requires a non-repacked block-quant stacked weight");
  VT_CHECK(w.rank == 2 && w.shape[1] == K,
           "laguna grouped expert GEMM: weight K mismatch");
  VT_CHECK(static_cast<int64_t>(act.size()) == P * K,
           "laguna grouped expert GEMM: act size mismatch");
  VT_CHECK(static_cast<int64_t>(expert_ids.size()) == P,
           "laguna grouped expert GEMM: expert_ids size mismatch");
  std::vector<float> out(static_cast<size_t>(P) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  std::vector<int32_t> eids = expert_ids;  // stable buffer for the (unified) tensor
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(act.data()),
                                        vt::DType::kF32, gq.device, {P, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {P, N});
  vt::Tensor eid =
      vt::Tensor::Contiguous(eids.data(), vt::DType::kI32, gq.device, {P});
  vt::Tensor wt = w.View();
  wt.device = gq.device;  // unified-memory block view retag
  vt::MatmulBTQuantGrouped(gq, o, a, wt, eid);
  if (on_dev) DrainQueue(gq);
  return out;
}

// ── W6 shared building blocks (used by BOTH the stateless full-recompute and the
//    KV-cached incremental forward, so the two paths are bit-identical BY SHARING
//    the exact same float ops — the moved code is verbatim from the W5 forward). ──

// Embed gather: hidden[T,H] = embed_table[token_ids]. Gathers ONLY the T needed
// rows directly from the (f32/bf16) table bytes — BIT-IDENTICAL to the prior
// ReadF32(whole-table)-then-gather (same per-element f32/bf16→f32 conversion,
// same rows), but avoids materializing the full [Vsz,H] table (~1.23 GB, ~311M
// element-converts) on EVERY decode token — the dominant host-orchestration
// waste measured in the W7 speed profile (laguna-s21-w7-speed-2026-07-31.md #5).
std::vector<float> LagunaEmbed(const OwnedTensor& embed_t,
                               const std::vector<int32_t>& token_ids, int64_t H,
                               int64_t Vsz) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  std::vector<float> hidden(static_cast<size_t>(T * H));
  const uint8_t* raw = embed_t.bytes.data();
  const bool is_bf16 = embed_t.dtype == vt::DType::kBF16;
  VT_CHECK(embed_t.dtype == vt::DType::kF32 || is_bf16,
           "laguna embed: table dtype must be f32/bf16 (matches ReadF32)");
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = token_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < Vsz, "laguna: token id out of range");
    float* dst = hidden.data() + static_cast<size_t>(t * H);
    if (is_bf16) {
      const auto* b =
          reinterpret_cast<const uint16_t*>(raw) + static_cast<size_t>(tok * H);
      for (int64_t i = 0; i < H; ++i) {
        const uint32_t bits = static_cast<uint32_t>(b[i]) << 16;
        std::memcpy(&dst[i], &bits, sizeof(float));
      }
    } else {
      std::memcpy(dst,
                  reinterpret_cast<const float*>(raw) + static_cast<size_t>(tok * H),
                  static_cast<size_t>(H) * sizeof(float));
    }
  }
  return hidden;
}

// GQA attention of `Tq` queries (rows of `q`, global positions `q_pos`) against
// `kv_rows` cached K/V (rows of `k`/`v`, global positions `kv_pos`). window==0 =>
// full causal; window>0 => sliding (score only kv with 0 <= q_pos - kv_pos <
// window). Returns attn[Tq, Hq*Dh]. This is the VERBATIM W5 inner loop, generalized
// to distinct query/kv row sets so the KV-cached decode (Tq=1, kv=history) reuses
// the identical float ops as the full-recompute (Tq==kv_rows, q_pos==kv_pos).
std::vector<float> LagunaAttention(const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v, int64_t Tq,
                                   int64_t kv_rows, int64_t Hq, int64_t Hkv,
                                   int64_t Dh, int64_t group,
                                   const std::vector<int64_t>& q_pos,
                                   const std::vector<int64_t>& kv_pos,
                                   int64_t window) {
  const int64_t qdim = Hq * Dh;
  std::vector<float> attn(static_cast<size_t>(Tq * qdim), 0.0F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  for (int64_t h = 0; h < Hq; ++h) {
    const int64_t kvh = h / group;
    for (int64_t i = 0; i < Tq; ++i) {
      const int64_t pi = q_pos[static_cast<size_t>(i)];
      float maxs = -std::numeric_limits<float>::infinity();
      std::vector<float> logit(static_cast<size_t>(kv_rows),
                               -std::numeric_limits<float>::infinity());
      for (int64_t j = 0; j < kv_rows; ++j) {
        const int64_t pj = kv_pos[static_cast<size_t>(j)];
        if (pj > pi) continue;
        if (window > 0 && pi - pj >= window) continue;
        const float* qrow = q.data() + static_cast<size_t>((i * Hq + h) * Dh);
        const float* krow = k.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
        float dot = 0.0F;
        for (int64_t d = 0; d < Dh; ++d) dot += qrow[d] * krow[d];
        dot *= scale;
        logit[static_cast<size_t>(j)] = dot;
        maxs = std::max(maxs, dot);
      }
      float denom = 0.0F;
      for (int64_t j = 0; j < kv_rows; ++j) {
        if (logit[static_cast<size_t>(j)] == -std::numeric_limits<float>::infinity())
          continue;
        const float e = std::exp(logit[static_cast<size_t>(j)] - maxs);
        logit[static_cast<size_t>(j)] = e;
        denom += e;
      }
      float* ao = attn.data() + static_cast<size_t>((i * Hq + h) * Dh);
      for (int64_t j = 0; j < kv_rows; ++j) {
        const float ww = logit[static_cast<size_t>(j)];
        if (ww == -std::numeric_limits<float>::infinity() || ww == 0.0F) continue;
        const float pw = ww / denom;
        const float* vrow = v.data() + static_cast<size_t>((j * Hkv + kvh) * Dh);
        for (int64_t d = 0; d < Dh; ++d) ao[d] += pw * vrow[d];
      }
    }
  }
  return attn;
}

// FFN block: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE (layers 1..47).
// Consumes hn2[T,H] (post-attn RMSNorm) and returns f[T,H]. VERBATIM keep-quant W5
// FFN, shared by both forwards.
std::vector<float> LagunaFfnBlock(vt::Queue& q, const LagunaLayerWeights& lw,
                                  const LagunaParams& p,
                                  const std::vector<float>& hn2, int64_t T) {
  const int64_t H = p.hidden_size;
  std::vector<float> f(static_cast<size_t>(T * H), 0.0F);
  if (lw.is_dense) {
    const int64_t I = p.intermediate_size;
    const std::vector<float> g = LqGemm(q, lw.mlp.gate_proj, hn2, T, I, H);
    const std::vector<float> u = LqGemm(q, lw.mlp.up_proj, hn2, T, I, H);
    const std::vector<float> act = GateUpSilu(g, u, T, I);
    f = LqGemm(q, lw.mlp.down_proj, act, T, H, I);
    return f;
  }
  const int64_t moe_I = p.moe_intermediate_size;
  const std::vector<float> router_w = ReadF32(lw.moe.router);
  std::vector<float> bias;
  if (!lw.moe.e_score_correction_bias.Empty())
    bias = ReadF32(lw.moe.e_score_correction_bias);
  const bool has_shared = !lw.moe.shared_gate.Empty();
  for (int64_t i = 0; i < T; ++i) {
    std::vector<float> hrow(hn2.begin() + static_cast<int64_t>(i * H),
                            hn2.begin() + static_cast<int64_t>((i + 1) * H));
    const std::vector<float> rlog = MatmulNK(hrow, router_w, 1, p.num_experts, H);  // [E]
    const LagunaRouterSelection sel = LagunaUngroupedRouterTopK(
        rlog, bias, p.num_experts_per_tok, p.norm_topk_prob,
        p.moe_routed_scaling_factor);
    std::vector<float> acc(static_cast<size_t>(H), 0.0F);
    const int64_t Pk = static_cast<int64_t>(sel.ids.size());
    if (LagunaGroupedMoeEnabled() && Pk > 0) {
      // W8 lever #2: collapse this token's Pk per-expert gate/up/down matvecs into 3
      // grouped launches. Row s := (this token's hrow, expert sel.ids[s]) IN SLOT
      // ORDER, so eo[s] == the per-expert down output for expert sel.ids[s] and the
      // acc combine below runs in the exact same order as the per-expert fallback →
      // byte-identical. gate/up/down each = ONE vt::MatmulBTQuantGrouped.
      std::vector<int32_t> eids(sel.ids.begin(), sel.ids.end());
      std::vector<float> arep(static_cast<size_t>(Pk) * H);
      for (int64_t s = 0; s < Pk; ++s)
        std::memcpy(arep.data() + static_cast<size_t>(s) * H, hrow.data(),
                    static_cast<size_t>(H) * sizeof(float));
      const std::vector<float> eg =
          LqGemmGrouped(q, lw.moe.experts_gate, arep, eids, Pk, moe_I, H);
      const std::vector<float> eu =
          LqGemmGrouped(q, lw.moe.experts_up, arep, eids, Pk, moe_I, H);
      const std::vector<float> eact = GateUpSilu(eg, eu, Pk, moe_I);
      const std::vector<float> eo =
          LqGemmGrouped(q, lw.moe.experts_down, eact, eids, Pk, H, moe_I);
      for (int64_t s = 0; s < Pk; ++s) {
        const float wgt = sel.weights[static_cast<size_t>(s)];
        const float* eor = eo.data() + static_cast<size_t>(s) * H;
        for (int64_t d = 0; d < H; ++d)
          acc[static_cast<size_t>(d)] += wgt * eor[d];
      }
    } else {
      for (size_t s = 0; s < sel.ids.size(); ++s) {
        const int64_t id = sel.ids[s];
        const std::vector<float> eg =
            LqGemmRowSlice(q, lw.moe.experts_gate, hrow, 1, moe_I, H, id * moe_I);
        const std::vector<float> eu =
            LqGemmRowSlice(q, lw.moe.experts_up, hrow, 1, moe_I, H, id * moe_I);
        const std::vector<float> eact = GateUpSilu(eg, eu, 1, moe_I);
        const std::vector<float> eo =
            LqGemmRowSlice(q, lw.moe.experts_down, eact, 1, H, moe_I, id * H);
        const float wgt = sel.weights[s];
        for (int64_t d = 0; d < H; ++d)
          acc[static_cast<size_t>(d)] += wgt * eo[static_cast<size_t>(d)];
      }
    }
    if (has_shared) {
      const std::vector<float> sg = LqGemm(q, lw.moe.shared_gate, hrow, 1, moe_I, H);
      const std::vector<float> su = LqGemm(q, lw.moe.shared_up, hrow, 1, moe_I, H);
      const std::vector<float> sact = GateUpSilu(sg, su, 1, moe_I);
      const std::vector<float> so = LqGemm(q, lw.moe.shared_down, sact, 1, H, moe_I);
      for (int64_t d = 0; d < H; ++d)
        acc[static_cast<size_t>(d)] += so[static_cast<size_t>(d)];
    }
    std::copy(acc.begin(), acc.end(), f.begin() + static_cast<int64_t>(i * H));
  }
  return f;
}

// Final RMSNorm -> lm_head (untied keep-quant / tied f32) -> logits, with an
// optional gather over LOCAL row indices. VERBATIM W5 tail, shared by both forwards.
std::vector<float> LagunaFinalLogits(vt::Queue& q, const LagunaWeights& weights,
                                     const std::vector<float>& hidden, int64_t T,
                                     const std::vector<int32_t>& logits_indices) {
  const LagunaParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const float eps = p.rms_norm_eps;
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
  if (tied) return MatmulNK(src, ReadF32(weights.embed), n_out, Vsz, H);
  return LqGemm(q, weights.lm_head, src, n_out, Vsz, H);
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
    // Per-head QK-RMSNorm BEFORE RoPE (VERIFIED W4 from the GGUF attn_q/k_norm).
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      RmsNormHeads(q, ReadF32(lw.attn.q_norm), T, Hq, Dh, eps);
      RmsNormHeads(k, ReadF32(lw.attn.k_norm), T, Hkv, Dh, eps);
    }
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
      const std::vector<float> g = MatmulNK(hn2, ReadF32(lw.mlp.gate_proj), T, I, H);
      const std::vector<float> u = MatmulNK(hn2, ReadF32(lw.mlp.up_proj), T, I, H);
      const std::vector<float> act = GateUpSilu(g, u, T, I);
      f = MatmulNK(act, ReadF32(lw.mlp.down_proj), T, H, I);
    } else {
      const int64_t E = p.num_experts;
      const int64_t moe_I = p.moe_intermediate_size;
      const std::vector<float> router_w = ReadF32(lw.moe.router);
      std::vector<float> bias;
      if (!lw.moe.e_score_correction_bias.Empty())
        bias = ReadF32(lw.moe.e_score_correction_bias);
      const std::vector<float> exp_g = ReadF32(lw.moe.experts_gate);  // [E,moeI,H]
      const std::vector<float> exp_u = ReadF32(lw.moe.experts_up);    // [E,moeI,H]
      const std::vector<float> exp_dn = ReadF32(lw.moe.experts_down); // [E,H,moeI]
      const int64_t gu_stride = moe_I * H;
      const int64_t dn_stride = H * moe_I;
      const bool has_shared = !lw.moe.shared_gate.Empty();
      std::vector<float> shared_g, shared_u, shared_dn;
      if (has_shared) {
        shared_g = ReadF32(lw.moe.shared_gate);
        shared_u = ReadF32(lw.moe.shared_up);
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
          std::vector<float> eg(exp_g.begin() + static_cast<int64_t>(id * gu_stride),
                                exp_g.begin() + static_cast<int64_t>((id + 1) * gu_stride));
          std::vector<float> eu(exp_u.begin() + static_cast<int64_t>(id * gu_stride),
                                exp_u.begin() + static_cast<int64_t>((id + 1) * gu_stride));
          std::vector<float> edn(exp_dn.begin() + static_cast<int64_t>(id * dn_stride),
                                 exp_dn.begin() + static_cast<int64_t>((id + 1) * dn_stride));
          const std::vector<float> eo = ExpertMlp(hrow, eg, eu, edn, H, moe_I);
          const float w = sel.weights[s];
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += w * eo[static_cast<size_t>(d)];
        }
        if (has_shared) {
          const std::vector<float> so = ExpertMlp(hrow, shared_g, shared_u, shared_dn, H, moe_I);
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

// ════════════════════════════════════════════════════════════════════════════
// W5 — the REAL keep-quant GGUF forward. The `LagunaModel::Forward` composition
// with the ~9 GEMM sites routed through the keep-quant LqGemm/LqGemmRowSlice
// (vt::MatmulBT on the block-typed weights) instead of MatmulNK(ReadF32). All the
// glue (dual-RoPE, per-head QK-RMSNorm, per-head softplus out-gate, ungrouped
// sigmoid-noaux router + routed_scaling, shared expert) is IDENTICAL to the
// unit-gated f32 reference — only the matmul operands change from f32 to
// keep-quant blocks. Stateless whole-sequence recompute (mirrors
// DeepseekV4ForwardGguf); the greedy driver loops it.
std::vector<float> LagunaForwardGguf(const LagunaWeights& weights, vt::Queue& q,
                                     const std::vector<int32_t>& token_ids,
                                     const std::vector<int32_t>& positions,
                                     const std::vector<int32_t>& logits_indices) {
  const LagunaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const int64_t Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t kvdim = Hkv * Dh;
  const float eps = p.rms_norm_eps;
  VT_CHECK(weights.has_gguf_weights, "laguna gguf forward: no keep-quant tower");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "laguna gguf: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == p.num_hidden_layers,
           "laguna gguf: one LagunaLayerWeights per layer required");

  int64_t max_pos = 0;
  for (int32_t ps : positions) max_pos = std::max<int64_t>(max_pos, ps);
  const int64_t rope_rows = max_pos + 1;
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, rope_rows);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, rope_rows);

  // Embed: hidden[T,H] = embed[token_ids] (f32 gather table).
  std::vector<float> hidden = LagunaEmbed(weights.embed, token_ids, H, Vsz);

  std::vector<int64_t> pos64(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos64[static_cast<size_t>(t)] = positions[static_cast<size_t>(t)];

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l);
    const int64_t qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    VT_CHECK(group > 0 && Hq == group * Hkv,
             "laguna gguf: per-layer Q-head count must be a multiple of KV heads");
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);

    // --- attention ---
    const std::vector<float> hn = RmsNorm(hidden, ReadF32(lw.input_norm), T, H, eps);
    std::vector<float> qv = LqGemm(q, lw.attn.q_proj, hn, T, qdim, H);
    std::vector<float> kv = LqGemm(q, lw.attn.k_proj, hn, T, kvdim, H);
    std::vector<float> vv = LqGemm(q, lw.attn.v_proj, hn, T, kvdim, H);
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      RmsNormHeads(qv, ReadF32(lw.attn.q_norm), T, Hq, Dh, eps);
      RmsNormHeads(kv, ReadF32(lw.attn.k_norm), T, Hkv, Dh, eps);
    }
    const std::vector<float>& cache = global ? yarn_cache : slide_cache;
    ApplyRope(qv, T, Hq, Dh, rd, cache, positions);
    ApplyRope(kv, T, Hkv, Dh, rd, cache, positions);

    std::vector<float> attn = LagunaAttention(qv, kv, vv, T, T, Hq, Hkv, Dh, group,
                                              pos64, pos64, window);

    // per-head softplus attention OUTPUT gate.
    const std::vector<float> glogits = LqGemm(q, lw.attn.g_proj, hn, T, Hq, H);
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> row(attn.begin() + static_cast<int64_t>(i * qdim),
                             attn.begin() + static_cast<int64_t>((i + 1) * qdim));
      std::vector<float> gl(glogits.begin() + static_cast<int64_t>(i * Hq),
                            glogits.begin() + static_cast<int64_t>((i + 1) * Hq));
      LagunaSoftplusHeadGate(row, gl, Hq, Dh);
      std::copy(row.begin(), row.end(), attn.begin() + static_cast<int64_t>(i * qdim));
    }

    const std::vector<float> o = LqGemm(q, lw.attn.o_proj, attn, T, H, qdim);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

    // --- FFN: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE ---
    const std::vector<float> hn2 = RmsNorm(hidden, ReadF32(lw.post_attn_norm), T, H, eps);
    const std::vector<float> f = LagunaFfnBlock(q, lw, p, hn2, T);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += f[static_cast<size_t>(i)];
  }

  // Final RMSNorm -> lm_head (untied keep-quant) -> logits.
  return LagunaFinalLogits(q, weights, hidden, T, logits_indices);
}

// ── W6: the KV-CACHED incremental forward. Same keep-quant composition as
//    LagunaForwardGguf, but binds a LagunaKvCache: this call's T tokens append their
//    per-layer POST-RoPE K + RAW V to the cache and the queries attend over the FULL
//    cached history (global layers) / the last-512 window (sliding layers, evicted
//    beyond the window). Prefill = first call (cache.len==0, all prompt tokens);
//    decode = later calls (ONE new token, positions={cache.len}). Token-IDENTICAL to
//    LagunaForwardGguf over the growing context — the KV-cache identity (a token's
//    cached K/V equal what full-recompute would recompute, since RoPE/QK-norm depend
//    only on the token's own position and attention is causal). Mirror of
//    DeepseekV4ForwardGgufCached, extended MLA-latent -> GQA multi-head K/V, plus the
//    NEW sliding-window eviction (grounded in gemma2/3 is_sliding).
std::vector<float> LagunaForwardGgufCached(const LagunaWeights& weights, vt::Queue& q,
                                           LagunaKvCache& cache,
                                           const std::vector<int32_t>& token_ids,
                                           const std::vector<int32_t>& positions,
                                           const std::vector<int32_t>& logits_indices) {
  const LagunaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t Vsz = p.vocab_size;
  const int64_t Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads;
  const int64_t kvdim = Hkv * Dh;
  const float eps = p.rms_norm_eps;
  const int64_t nlayers = p.num_hidden_layers;
  VT_CHECK(weights.has_gguf_weights, "laguna gguf cached: no keep-quant tower");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "laguna gguf cached: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == nlayers,
           "laguna gguf cached: one LagunaLayerWeights per layer required");
  if (cache.k.empty()) cache.Reset(nlayers, Dh, Hkv);  // lazy init on first call
  VT_CHECK(static_cast<int64_t>(cache.k.size()) == nlayers && cache.head_dim == Dh &&
               cache.kv_heads == Hkv,
           "laguna gguf cached: cache not sized for this model");
  // The base global position of this call's first token = cache.len. Positions must
  // be contiguous from there (prefill: 0..T-1 with len==0; decode: {len}).
  const int64_t base = cache.len;
  VT_CHECK(positions[0] == static_cast<int32_t>(base),
           "laguna gguf cached: positions[0] must equal cache.len (contiguous decode)");

  int64_t max_pos = 0;
  for (int32_t ps : positions) max_pos = std::max<int64_t>(max_pos, ps);
  const int64_t rope_rows = max_pos + 1;
  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, rope_rows);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, rope_rows);

  std::vector<float> hidden = LagunaEmbed(weights.embed, token_ids, H, Vsz);

  std::vector<int64_t> q_pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) q_pos[static_cast<size_t>(t)] = positions[static_cast<size_t>(t)];

  for (int64_t l = 0; l < nlayers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l);
    const int64_t qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    VT_CHECK(group > 0 && Hq == group * Hkv,
             "laguna gguf cached: per-layer Q-head count must be a multiple of KV heads");
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);

    // --- attention: project THIS call's T tokens' q/k/v, qk-norm + RoPE ---
    const std::vector<float> hn = RmsNorm(hidden, ReadF32(lw.input_norm), T, H, eps);
    std::vector<float> qv = LqGemm(q, lw.attn.q_proj, hn, T, qdim, H);
    std::vector<float> knew = LqGemm(q, lw.attn.k_proj, hn, T, kvdim, H);
    std::vector<float> vnew = LqGemm(q, lw.attn.v_proj, hn, T, kvdim, H);
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      RmsNormHeads(qv, ReadF32(lw.attn.q_norm), T, Hq, Dh, eps);
      RmsNormHeads(knew, ReadF32(lw.attn.k_norm), T, Hkv, Dh, eps);
    }
    const std::vector<float>& rope = global ? yarn_cache : slide_cache;
    ApplyRope(qv, T, Hq, Dh, rd, rope, positions);
    ApplyRope(knew, T, Hkv, Dh, rd, rope, positions);

    // Append the T new K/V rows to this layer's cache. The cached K is POST-RoPE /
    // POST-QK-norm and V is raw — both position-only functions, so bit-exact.
    std::vector<float>& kc = cache.k[static_cast<size_t>(l)];
    std::vector<float>& vc = cache.v[static_cast<size_t>(l)];
    kc.insert(kc.end(), knew.begin(), knew.end());
    vc.insert(vc.end(), vnew.begin(), vnew.end());
    // Sliding-window eviction (gemma2/3 is_sliding): keep only the last `window`
    // rows — a query at global position pi attends kv with pi - pj < window, so once
    // more than `window` rows are cached the oldest can never be scored again. Global
    // layers (window==0) keep the whole history. first_pos tracks the evicted base.
    int64_t rows = static_cast<int64_t>(kc.size()) / kvdim;
    if (window > 0 && rows > window) {
      const int64_t drop = rows - window;
      const size_t off = static_cast<size_t>(drop * kvdim);
      kc.erase(kc.begin(), kc.begin() + static_cast<int64_t>(off));
      vc.erase(vc.begin(), vc.begin() + static_cast<int64_t>(off));
      cache.first_pos[static_cast<size_t>(l)] += drop;
      rows = window;
    }
    // Global positions of the cached rows: first_pos .. first_pos+rows-1.
    const int64_t fp = cache.first_pos[static_cast<size_t>(l)];
    std::vector<int64_t> kv_pos(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) kv_pos[static_cast<size_t>(r)] = fp + r;

    std::vector<float> attn = LagunaAttention(qv, kc, vc, T, rows, Hq, Hkv, Dh, group,
                                              q_pos, kv_pos, window);

    // per-head softplus attention OUTPUT gate.
    const std::vector<float> glogits = LqGemm(q, lw.attn.g_proj, hn, T, Hq, H);
    for (int64_t i = 0; i < T; ++i) {
      std::vector<float> row(attn.begin() + static_cast<int64_t>(i * qdim),
                             attn.begin() + static_cast<int64_t>((i + 1) * qdim));
      std::vector<float> gl(glogits.begin() + static_cast<int64_t>(i * Hq),
                            glogits.begin() + static_cast<int64_t>((i + 1) * Hq));
      LagunaSoftplusHeadGate(row, gl, Hq, Dh);
      std::copy(row.begin(), row.end(), attn.begin() + static_cast<int64_t>(i * qdim));
    }

    const std::vector<float> o = LqGemm(q, lw.attn.o_proj, attn, T, H, qdim);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

    // --- FFN: dense SwiGLU (layer 0) or ungrouped sigmoid-noaux MoE ---
    const std::vector<float> hn2 = RmsNorm(hidden, ReadF32(lw.post_attn_norm), T, H, eps);
    const std::vector<float> f = LagunaFfnBlock(q, lw, p, hn2, T);
    for (int64_t i = 0; i < T * H; ++i) hidden[static_cast<size_t>(i)] += f[static_cast<size_t>(i)];
  }

  cache.len += T;  // every layer appended its T rows
  return LagunaFinalLogits(q, weights, hidden, T, logits_indices);
}

}  // namespace vllm
