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

#include <mutex>
#include <unordered_map>

#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // Dev/DBuf/ResidentNvfp4 + Marlin (B2)
#include "vllm/model_executor/models/laguna_device.h"
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
  if (!vt::IsBlockQuant(w.dtype)) {
    // N5 lever #2: the bf16/f32 tower (attention/dense/router/shared/lm_head on the
    // NVFP4 arm) — the nsys trace found it runs the HOST MatmulNK reference even on
    // the CUDA queue, CPU-bound at ~4.8 s/tok (the lm_head [Vsz,H] MatmulNK + its
    // per-token ReadF32 dominate). On the DEVICE, keep the weight bf16 (no ReadF32),
    // cast the SMALL [T,K] f32 activation to bf16 on-GPU, and run the bf16×bf16->f32
    // MatmulBT (cuBLASLt nvjet on GB10; MatmulBT needs matched dtypes). f32 accum;
    // near-tie vs MatmulNK (different algo + bf16 activation rounding). The CPU path
    // keeps the exact MatmulNK reference (the run-gate stays byte-identical); the GGUF
    // keep-quant tower is block-quant here so this branch is nvfp4-arm-only.
    const bool bf16_dev = q.device.type != vt::DeviceType::kCPU && w.dtype == vt::DType::kBF16;
    if (!bf16_dev) return MatmulNK(x, ReadF32(w), T, N, K);
    std::vector<float> out(static_cast<size_t>(T) * N);
    std::vector<uint16_t> a_bf16(static_cast<size_t>(T) * K);  // bf16 activation (unified mem)
    vt::Tensor xf = vt::Tensor::Contiguous(const_cast<float*>(x.data()), vt::DType::kF32,
                                           q.device, {T, K});
    vt::Tensor ab = vt::Tensor::Contiguous(a_bf16.data(), vt::DType::kBF16, q.device, {T, K});
    vt::CastBf16(q, ab, xf);
    vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {T, N});
    vt::Tensor wt = w.View();
    wt.device = q.device;  // unified-memory bf16 weight retag (like the block-quant path)
    vt::MatmulBT(q, o, ab, wt);
    DrainQueue(q);
    return out;
  }
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

// N2 (task #230): TRUE-W4A4 NVFP4 GEMM for the safetensors arm's routed experts.
// Y[M,N] f32 = fp4(X[M,K] / input_divisor) @ dequant(w).T, accumulator * alpha.
// The activation is quantized to fp4 with w.input_global_scale_inv (NOT scale2),
// then vt::MatmulNvfp4Fp4 scales the fp4xfp4 accumulator by w.alpha (which folds
// BOTH the activation and weight reciprocated global scales). Mirrors LqGemm's
// unified-memory pattern (host ptrs as device tensors on GB10; raw weight view
// retagged). CPU-runnable (ScaledFp4QuantKernel + MatmulNvfp4Fp4Kernel exist), so
// the fixture gate exercises the exact numerics. If the DGX gate shows all-zeros,
// the whole-weight raw view needs ResidentWeight staging (keepquant-device-slice
// note) — a one-line switch to explicit residency.
std::vector<float> LqGemmNvfp4Fp4(vt::Queue& q, const Nvfp4Weight& w,
                                  const std::vector<float>& x, int64_t M, int64_t N,
                                  int64_t K) {
  VT_CHECK(w.n == N && w.k == K && K % 16 == 0,
           "laguna nvfp4 GEMM: weight shape / K%16 mismatch");
  VT_CHECK(w.IsTrueW4A4(), "laguna nvfp4 GEMM: expected true-W4A4 (alpha>0) weight");
  std::vector<float> out(static_cast<size_t>(M) * N);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = q.device.type != vt::DeviceType::kCPU;
  vt::Queue& gq = on_dev ? q : cpuq;
  // fp4 activation intermediates (unified-memory host buffers: i8 packed [M,K/2] +
  // fp8-e4m3 block scale [M,K/16]).
  std::vector<int8_t> ap_buf(static_cast<size_t>(M) * (K / 2));
  std::vector<int8_t> as_buf(static_cast<size_t>(M) * (K / 16));
  vt::Tensor xt = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                         vt::DType::kF32, gq.device, {M, K});
  vt::Tensor ap = vt::Tensor::Contiguous(ap_buf.data(), vt::DType::kI8, gq.device, {M, K / 2});
  vt::Tensor as = vt::Tensor::Contiguous(as_buf.data(), vt::DType::kI8, gq.device, {M, K / 16});
  vt::ScaledFp4Quant(gq, ap, as, xt, w.input_global_scale_inv);  // kLinear layout
  vt::Tensor ot = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {M, N});
  vt::Tensor bp = w.packed.View();
  bp.device = gq.device;  // unified-memory retag (mirror LqGemm)
  vt::Tensor bs = w.scale.View();
  bs.device = gq.device;
  vt::MatmulNvfp4Fp4(gq, ot, ap, as, bp, bs, w.alpha);
  if (on_dev) DrainQueue(gq);
  return out;
}

// N5 lever (task #234): DEVICE-RESIDENT W4A4 MoE block for ONE decode token. The
// per-expert `LqGemmNvfp4Fp4` loop drains the queue after EVERY GEMM (it returns a
// host vector) → ~top_k×3 `cudaStreamSynchronize`/layer, the 78.6%-of-API-time wall
// the nsys found. Here the WHOLE token's routed-expert MoE runs as one async device
// chain — per expert: fp4-quant(hrow, w.igs) → MatmulNvfp4Fp4 for gate/up, MoeSiluMul,
// fp4-quant(act, down.igs) → MatmulNvfp4Fp4 for down into a stacked [top_k,H] device
// buffer — then ONE MoeCombine (weighted sum) and ONE DrainQueue. Unified-memory
// buffers (host ptrs retagged device); every op stays on `q` so CUDA orders them (the
// gate GEMM reads `a_p` before the up-quant overwrites it — same-stream serial). CUDA
// only (M=1 decode); the CPU/per-expert path stays the byte-exact reference.
std::vector<float> LagunaMoeResidentFp4(vt::Queue& q, const LagunaMoeWeights& moe,
                                        const std::vector<float>& hrow, int64_t moe_I,
                                        int64_t H, const LagunaRouterSelection& sel) {
  const int64_t Pk = static_cast<int64_t>(sel.ids.size());
  std::vector<float> acc(static_cast<size_t>(H), 0.0F);
  if (Pk == 0) return acc;
  const vt::Device dev = q.device;
  // fp4 activation intermediates (K=H for gate/up, K=moe_I for down) + per-expert
  // gate/up/silu buffers + the stacked expert-output [Pk,H] the combine reduces.
  std::vector<int8_t> ap_h((H / 2)), as_h((H / 16));        // gate/up activation fp4
  std::vector<int8_t> ap_d((moe_I / 2)), as_d((moe_I / 16));  // down activation fp4
  std::vector<float> dg(moe_I), du(moe_I), dact(moe_I);
  std::vector<float> deo(static_cast<size_t>(Pk) * H);       // [Pk,H] stacked outputs
  std::vector<float> wgt(static_cast<size_t>(Pk));
  for (int64_t s = 0; s < Pk; ++s) wgt[static_cast<size_t>(s)] = sel.weights[static_cast<size_t>(s)];
  auto wview = [&](const OwnedTensor& w) { vt::Tensor t = w.View(); t.device = dev; return t; };
  vt::Tensor xt = vt::Tensor::Contiguous(const_cast<float*>(hrow.data()), vt::DType::kF32, dev, {1, H});
  vt::Tensor ap = vt::Tensor::Contiguous(ap_h.data(), vt::DType::kI8, dev, {1, H / 2});
  vt::Tensor as = vt::Tensor::Contiguous(as_h.data(), vt::DType::kI8, dev, {1, H / 16});
  vt::Tensor apd = vt::Tensor::Contiguous(ap_d.data(), vt::DType::kI8, dev, {1, moe_I / 2});
  vt::Tensor asd = vt::Tensor::Contiguous(as_d.data(), vt::DType::kI8, dev, {1, moe_I / 16});
  vt::Tensor dgt = vt::Tensor::Contiguous(dg.data(), vt::DType::kF32, dev, {1, moe_I});
  vt::Tensor dut = vt::Tensor::Contiguous(du.data(), vt::DType::kF32, dev, {1, moe_I});
  vt::Tensor dat = vt::Tensor::Contiguous(dact.data(), vt::DType::kF32, dev, {1, moe_I});
  for (int64_t s = 0; s < Pk; ++s) {
    const int64_t id = sel.ids[static_cast<size_t>(s)];
    const Nvfp4Weight& gw = moe.experts_gate_fp4[static_cast<size_t>(id)];
    const Nvfp4Weight& uw = moe.experts_up_fp4[static_cast<size_t>(id)];
    const Nvfp4Weight& dw = moe.experts_down_fp4[static_cast<size_t>(id)];
    vt::ScaledFp4Quant(q, ap, as, xt, gw.input_global_scale_inv);
    vt::MatmulNvfp4Fp4(q, dgt, ap, as, wview(gw.packed), wview(gw.scale), gw.alpha);
    vt::ScaledFp4Quant(q, ap, as, xt, uw.input_global_scale_inv);
    vt::MatmulNvfp4Fp4(q, dut, ap, as, wview(uw.packed), wview(uw.scale), uw.alpha);
    vt::MoeSiluMul(q, dat, dgt, dut);
    vt::ScaledFp4Quant(q, apd, asd, dat, dw.input_global_scale_inv);
    vt::Tensor eo = vt::Tensor::Contiguous(deo.data() + static_cast<size_t>(s) * H,
                                           vt::DType::kF32, dev, {1, H});
    vt::MatmulNvfp4Fp4(q, eo, apd, asd, wview(dw.packed), wview(dw.scale), dw.alpha);
  }
  vt::Tensor eostack = vt::Tensor::Contiguous(deo.data(), vt::DType::kF32, dev, {1, Pk, H});
  vt::Tensor wt = vt::Tensor::Contiguous(wgt.data(), vt::DType::kF32, dev, {1, Pk});
  vt::Tensor at = vt::Tensor::Contiguous(acc.data(), vt::DType::kF32, dev, {1, H});
  vt::MoeCombine(q, at, eostack, wt);
  DrainQueue(q);  // the ONLY sync for the whole token's routed-expert MoE
  return acc;
}

// ── N5 campaign-B (task #234): route the routed experts to the MARLIN W4A16
//    grouped MoE GEMM — vLLM's ACTUAL 18.8-tok/s kernel (VLLM_TEST_FORCE_FP8_MARLIN=1),
//    LOW-M-optimized for decode. Mirrors qwen3_5.cpp BuildMoeMarlinResident +
//    MoeBlockFusedMarlinCuda (the validated 27B/35B path: default-ON there,
//    16/16 token-for-token vs oracle, +22% gate / +80% decode), reusing the SHARED
//    dense_nvfp4::Dev/DBuf/ResidentNvfp4 + the shared vt::cuda Marlin repack/align
//    ops + vt::MoeGroupedGemmNvfp4Marlin. The SACRED qwen3_5 path is BYTE-UNTOUCHED
//    (this is a Laguna-local reconstruction over the shared primitives, split-w13
//    layout for a simple first landing). Gated OFF by default (VT_LAGUNA_MARLIN_MOE=1
//    opt-in) until the DGX near-tie + ncu gate lands — so the current default GEMV
//    path is unchanged. CUDA-only (VT_MARLIN_NVFP4). ──────────────────────────────
#ifdef VT_MARLIN_NVFP4
namespace {
// Resident Marlin-repacked routed experts (split w13). Mirror MoeMarlinResident.
struct LagunaMoeMarlinResident {
  void* w_gate = nullptr;  // i32 [E, H/16, moe_I*2]
  void* w_up = nullptr;    // i32 [E, H/16, moe_I*2]
  void* w_down = nullptr;  // i32 [E, moe_I/16, H*2]
  void* s_gate = nullptr;  // u8  [E, H/16, moe_I]
  void* s_up = nullptr;
  void* s_down = nullptr;  // u8  [E, moe_I/16, H]
  void* g_gate = nullptr;  // f32 [E]
  void* g_up = nullptr;
  void* g_down = nullptr;
  void* workspace = nullptr;  // i32 [sms*4]
  int sms = 0;
  bool ready = false;
};

LagunaMoeMarlinResident& LagunaMoeMarlinResidentFor(const LagunaMoeWeights* w) {
  static std::mutex mu;
  static std::unordered_map<const LagunaMoeWeights*, LagunaMoeMarlinResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[w];
}

// Repack every routed expert's fp4 gate/up/down into the resident Marlin layout —
// the per-expert body of qwen3_5.cpp BuildMoeMarlinResident (split branch),
// reusing the identical shared vt::cuda primitives. K=H (gate/up in), N=moe_I
// (gate/up out); down is K=moe_I, N=H.
void BuildLagunaMoeMarlinResident(vllm::dense_nvfp4::Dev d, const LagunaMoeWeights& moe, int E,
                                  int H, int I, LagunaMoeMarlinResident& mr) {
  void* stream = d.q.handle;
  mr.sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
  const size_t wg_i32 = static_cast<size_t>(H / 16) * (I * 2);  // gate/up weight elems
  const size_t wd_i32 = static_cast<size_t>(I / 16) * (H * 2);  // down weight elems
  const size_t sg_b = static_cast<size_t>(H / 16) * I;          // gate/up scale bytes
  const size_t sd_b = static_cast<size_t>(I / 16) * H;          // down scale bytes
  mr.w_gate = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
  mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_gate = d.b.Alloc(static_cast<size_t>(E) * sg_b);
  mr.s_up = d.b.Alloc(static_cast<size_t>(E) * sg_b);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_gate = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.g_down = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.workspace = d.b.Alloc(static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));

  // combined_scale_factor: gate+up jointly (w13), down alone (w2).
  std::vector<const uint8_t*> gu_bufs, dn_bufs;
  std::vector<size_t> gu_lens, dn_lens;
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(moe.experts_gate_fp4[se].scale.bytes.data()));
    gu_lens.push_back(moe.experts_gate_fp4[se].scale.bytes.size());
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(moe.experts_up_fp4[se].scale.bytes.data()));
    gu_lens.push_back(moe.experts_up_fp4[se].scale.bytes.size());
    dn_bufs.push_back(reinterpret_cast<const uint8_t*>(moe.experts_down_fp4[se].scale.bytes.data()));
    dn_lens.push_back(moe.experts_down_fp4[se].scale.bytes.size());
  }
  const float sf_gu = vt::cuda::MarlinNvfp4CombinedScaleFactor(gu_bufs, gu_lens);
  const float sf_dn = vt::cuda::MarlinNvfp4CombinedScaleFactor(dn_bufs, dn_lens);

  std::vector<float> gg(E), gu(E), gd(E);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    vllm::dense_nvfp4::Nvfp4Dev g = vllm::dense_nvfp4::ResidentNvfp4(d, moe.experts_gate_fp4[se]);
    vllm::dense_nvfp4::Nvfp4Dev u = vllm::dense_nvfp4::ResidentNvfp4(d, moe.experts_up_fp4[se]);
    vllm::dense_nvfp4::Nvfp4Dev dn = vllm::dense_nvfp4::ResidentNvfp4(d, moe.experts_down_fp4[se]);
    auto* wg = static_cast<uint32_t*>(mr.w_gate) + se * wg_i32;
    auto* wu = static_cast<uint32_t*>(mr.w_up) + se * wg_i32;
    auto* wd = static_cast<uint32_t*>(mr.w_down) + se * wd_i32;
    auto* sgp = static_cast<uint8_t*>(mr.s_gate) + se * sg_b;
    auto* sup = static_cast<uint8_t*>(mr.s_up) + se * sg_b;
    auto* sdp = static_cast<uint8_t*>(mr.s_down) + se * sd_b;
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wg,
                                       static_cast<const uint8_t*>(g.packed.data), H, I);
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wu,
                                       static_cast<const uint8_t*>(u.packed.data), H, I);
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wd,
                                       static_cast<const uint8_t*>(dn.packed.data), I, H);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(g.scale.data), sgp, H, I,
                                        sf_gu);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(u.scale.data), sup, H, I,
                                        sf_gu);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dn.scale.data), sdp, I, H,
                                        sf_dn);
    gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_gate_fp4[se].scale2, sf_gu);
    gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_up_fp4[se].scale2, sf_gu);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(moe.experts_down_fp4[se].scale2, sf_dn);
  }
  d.b.Copy(d.q, mr.g_gate, gg.data(), gg.size() * sizeof(float));
  d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // repack done → safe to free the fp4 originals

  // CRITICAL (matches qwen3_5.cpp BuildMoeMarlinResident tail): the Marlin resident
  // is now the committed compute path, so FREE the per-expert fp4 originals — both
  // the ResidentNvfp4 device transients (d_packed/d_scale, ~expert-tower-sized) and
  // the HOST mirror (packed/scale .bytes). Without this, peak = host copies + device
  // transients + Marlin resident ≈ 3× the expert tower, blowing past the 119 GiB
  // unified pool (a failed Alloc mid-build → null → silent device fault). Safe here:
  // BuildLagunaMoeMarlinResident runs ONLY under LagunaMarlinMoeEnabled(), so the
  // GEMV/CPU paths that read these bytes can never run in this process.
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    moe.experts_gate_fp4[se].d_packed.reset();
    moe.experts_gate_fp4[se].d_scale.reset();
    moe.experts_up_fp4[se].d_packed.reset();
    moe.experts_up_fp4[se].d_scale.reset();
    moe.experts_down_fp4[se].d_packed.reset();
    moe.experts_down_fp4[se].d_scale.reset();
    moe.experts_gate_fp4[se].packed.ReleaseHost();
    moe.experts_gate_fp4[se].scale.ReleaseHost();
    moe.experts_up_fp4[se].packed.ReleaseHost();
    moe.experts_up_fp4[se].scale.ReleaseHost();
    moe.experts_down_fp4[se].packed.ReleaseHost();
    moe.experts_down_fp4[se].scale.ReleaseHost();
  }
  mr.ready = true;
}
}  // namespace

// Single-token (T=1) routed-expert MoE via Marlin W4A16. Returns the combined
// routed-expert contribution [H] (shared expert handled by the caller, as for
// LagunaMoeResidentFp4). Mirrors MoeBlockFusedMarlinCuda's split-w13 body with
// T=1, top_k=Pk. bf16 activation (W4A16), so it IGNORES the W4A4 activation-quant
// fields and uses scale2 — exactly vLLM's Marlin config.
std::vector<float> LagunaMoeResidentMarlin(vt::Queue& q, const LagunaMoeWeights& moe,
                                           const std::vector<float>& hrow, int64_t moe_I, int64_t H,
                                           int64_t E, const LagunaRouterSelection& sel) {
  using vt::DType;
  const int64_t Pk = static_cast<int64_t>(sel.ids.size());
  std::vector<float> acc(static_cast<size_t>(H), 0.0F);
  if (Pk == 0) return acc;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  vllm::dense_nvfp4::Dev d{bk, q};
  LagunaMoeMarlinResident& mr = LagunaMoeMarlinResidentFor(&moe);
  if (!mr.ready)
    BuildLagunaMoeMarlinResident(d, moe, static_cast<int>(E), static_cast<int>(H),
                                 static_cast<int>(moe_I), mr);
  void* stream = q.handle;
  const vt::Device dev = q.device;

  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(1, static_cast<int>(Pk),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(1, static_cast<int>(Pk), static_cast<int>(E), block, &max_tok,
                                &max_blk);

  // bf16 activation [1,H] (upload f32 hrow, cast on device).
  vllm::dense_nvfp4::DBuf dh(d, DType::kBF16, {1, H});
  {
    vllm::dense_nvfp4::DBuf xf(d, DType::kF32, {1, H}, hrow.data());
    vt::CastBf16(q, dh.t(), xf.t());
  }
  // router top-k ids/weights for this one token = the Pk selected experts.
  std::vector<int32_t> ids32(static_cast<size_t>(Pk));
  std::vector<float> w1(static_cast<size_t>(Pk));
  for (int64_t s = 0; s < Pk; ++s) {
    ids32[static_cast<size_t>(s)] = static_cast<int32_t>(sel.ids[static_cast<size_t>(s)]);
    w1[static_cast<size_t>(s)] = sel.weights[static_cast<size_t>(s)];
  }
  vllm::dense_nvfp4::DBuf dtid(d, DType::kI32, {1, Pk}, ids32.data());
  vllm::dense_nvfp4::DBuf dtw(d, DType::kF32, {1, Pk}, w1.data());
  vllm::dense_nvfp4::DBuf sorted(d, DType::kI32, {max_tok});
  vllm::dense_nvfp4::DBuf eids(d, DType::kI32, {max_blk});
  vllm::dense_nvfp4::DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()), 1,
                                    static_cast<int>(Pk), static_cast<int>(E), block,
                                    static_cast<int32_t*>(sorted.ptr()),
                                    static_cast<int32_t*>(eids.ptr()),
                                    static_cast<int32_t*>(npad.ptr()));

  vt::Tensor wg = vllm::dense_nvfp4::MakeTensor(mr.w_gate, DType::kI32, dev, {E, H / 16, moe_I * 2});
  vt::Tensor wu = vllm::dense_nvfp4::MakeTensor(mr.w_up, DType::kI32, dev, {E, H / 16, moe_I * 2});
  vt::Tensor wd = vllm::dense_nvfp4::MakeTensor(mr.w_down, DType::kI32, dev, {E, moe_I / 16, H * 2});
  vt::Tensor sg = vllm::dense_nvfp4::MakeTensor(mr.s_gate, DType::kI8, dev, {E, H / 16, moe_I});
  vt::Tensor su = vllm::dense_nvfp4::MakeTensor(mr.s_up, DType::kI8, dev, {E, H / 16, moe_I});
  vt::Tensor sd = vllm::dense_nvfp4::MakeTensor(mr.s_down, DType::kI8, dev, {E, moe_I / 16, H});
  vt::Tensor gg = vllm::dense_nvfp4::MakeTensor(mr.g_gate, DType::kF32, dev, {E});
  vt::Tensor gu = vllm::dense_nvfp4::MakeTensor(mr.g_up, DType::kF32, dev, {E});
  vt::Tensor gd = vllm::dense_nvfp4::MakeTensor(mr.g_down, DType::kF32, dev, {E});
  vt::Tensor ws = vllm::dense_nvfp4::MakeTensor(mr.workspace, DType::kI32, dev, {mr.sms * 4});

  const int bi = block, Pki = static_cast<int>(Pk), Hi = static_cast<int>(H),
            Ii = static_cast<int>(moe_I);
  vllm::dense_nvfp4::DBuf dgate(d, DType::kBF16, {Pk, moe_I});
  vllm::dense_nvfp4::DBuf dup(d, DType::kBF16, {Pk, moe_I});
  vt::MoeGroupedGemmNvfp4Marlin(q, dgate.t(), dh.t(), wg, sg, gg, ws, sorted.t(),
                                eids.t(), npad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
  vt::MoeGroupedGemmNvfp4Marlin(q, dup.t(), dh.t(), wu, su, gu, ws, sorted.t(),
                                eids.t(), npad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
  vllm::dense_nvfp4::DBuf dact(d, DType::kBF16, {Pk, moe_I});
  vt::MoeSiluMul(q, dact.t(), dgate.t(), dup.t());

  vllm::dense_nvfp4::DBuf ddown(d, DType::kBF16, {Pk, H});
  vt::MoeGroupedGemmNvfp4Marlin(q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted.t(),
                                eids.t(), npad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, 1, Pki, Hi, Ii, false});
  vt::Tensor expert_out = vllm::dense_nvfp4::MakeTensor(ddown.ptr(), DType::kBF16, dev, {1, Pk, H});
  vllm::dense_nvfp4::DBuf dout(d, DType::kBF16, {1, H});
  vt::MoeCombine(q, dout.t(), expert_out, dtw.t());

  // bf16 [H] -> f32 acc (bf16 is the top 16 bits of f32; exact).
  std::vector<uint16_t> hb(static_cast<size_t>(H));
  dout.Download(d, hb.data());
  for (int64_t i = 0; i < H; ++i) {
    const uint32_t u = static_cast<uint32_t>(hb[static_cast<size_t>(i)]) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    acc[static_cast<size_t>(i)] = f;
  }
  return acc;
}

// Device-in/device-out variant for the resident decode: takes a DEVICE f32
// activation [H] + DEVICE i32 ids[Pk] + DEVICE f32 weights[Pk] (from the on-device
// router), runs the SAME Marlin W4A16 grouped chain, and writes the f32 [H] combine
// into out_dev (no host download → no per-MoE-layer drain). Same kernels ⇒ same
// device-regime near-tie as LagunaMoeResidentMarlin.
void LagunaMoeResidentMarlinInto(vt::Queue& q, const LagunaMoeWeights& moe, const float* hn_dev,
                                 int64_t moe_I, int64_t H, int64_t E, const int32_t* ids_dev,
                                 const float* w_dev, int64_t Pk, float* out_dev) {
  using vt::DType;
  if (Pk == 0) return;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  vllm::dense_nvfp4::Dev d{bk, q};
  LagunaMoeMarlinResident& mr = LagunaMoeMarlinResidentFor(&moe);
  if (!mr.ready)
    BuildLagunaMoeMarlinResident(d, moe, static_cast<int>(E), static_cast<int>(H),
                                 static_cast<int>(moe_I), mr);
  void* stream = q.handle;
  const vt::Device dev = q.device;
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(1, static_cast<int>(Pk),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(1, static_cast<int>(Pk), static_cast<int>(E), block, &max_tok,
                                &max_blk);
  // Wrap the async-produced device inputs (hn_dev/ids_dev/w_dev, written on-stream by
  // the router GEMM + sigmoid_topk with NO drain before) with MakeTensor — read them
  // ON-STREAM (ordered), NOT via a DBuf host-Copy which would snapshot stale data.
  vllm::dense_nvfp4::DBuf dh(d, DType::kBF16, {1, H});
  {
    vt::Tensor xf = vllm::dense_nvfp4::MakeTensor(const_cast<float*>(hn_dev), DType::kF32, dev,
                                                 {1, H});
    vt::CastBf16(q, dh.t(), xf);
  }
  vt::Tensor dtw = vllm::dense_nvfp4::MakeTensor(const_cast<float*>(w_dev), DType::kF32, dev,
                                               {1, Pk});
  vllm::dense_nvfp4::DBuf sorted(d, DType::kI32, {max_tok});
  vllm::dense_nvfp4::DBuf eids(d, DType::kI32, {max_blk});
  vllm::dense_nvfp4::DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(stream, ids_dev, 1, static_cast<int>(Pk), static_cast<int>(E),
                                    block, static_cast<int32_t*>(sorted.ptr()),
                                    static_cast<int32_t*>(eids.ptr()),
                                    static_cast<int32_t*>(npad.ptr()));
  vt::Tensor wg = vllm::dense_nvfp4::MakeTensor(mr.w_gate, DType::kI32, dev, {E, H / 16, moe_I * 2});
  vt::Tensor wu = vllm::dense_nvfp4::MakeTensor(mr.w_up, DType::kI32, dev, {E, H / 16, moe_I * 2});
  vt::Tensor wd = vllm::dense_nvfp4::MakeTensor(mr.w_down, DType::kI32, dev, {E, moe_I / 16, H * 2});
  vt::Tensor sg = vllm::dense_nvfp4::MakeTensor(mr.s_gate, DType::kI8, dev, {E, H / 16, moe_I});
  vt::Tensor su = vllm::dense_nvfp4::MakeTensor(mr.s_up, DType::kI8, dev, {E, H / 16, moe_I});
  vt::Tensor sd = vllm::dense_nvfp4::MakeTensor(mr.s_down, DType::kI8, dev, {E, moe_I / 16, H});
  vt::Tensor gg = vllm::dense_nvfp4::MakeTensor(mr.g_gate, DType::kF32, dev, {E});
  vt::Tensor gu = vllm::dense_nvfp4::MakeTensor(mr.g_up, DType::kF32, dev, {E});
  vt::Tensor gd = vllm::dense_nvfp4::MakeTensor(mr.g_down, DType::kF32, dev, {E});
  vt::Tensor ws = vllm::dense_nvfp4::MakeTensor(mr.workspace, DType::kI32, dev, {mr.sms * 4});
  const int bi = block, Pki = static_cast<int>(Pk), Hi = static_cast<int>(H),
            Ii = static_cast<int>(moe_I);
  vllm::dense_nvfp4::DBuf dgate(d, DType::kBF16, {Pk, moe_I});
  vllm::dense_nvfp4::DBuf dup(d, DType::kBF16, {Pk, moe_I});
  vt::MoeGroupedGemmNvfp4Marlin(q, dgate.t(), dh.t(), wg, sg, gg, ws, sorted.t(), eids.t(),
                                npad.t(), dtw, vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
  vt::MoeGroupedGemmNvfp4Marlin(q, dup.t(), dh.t(), wu, su, gu, ws, sorted.t(), eids.t(), npad.t(),
                                dtw, vt::MoeMarlinArgs{bi, Pki, 1, Ii, Hi, false});
  vllm::dense_nvfp4::DBuf dact(d, DType::kBF16, {Pk, moe_I});
  vt::MoeSiluMul(q, dact.t(), dgate.t(), dup.t());
  vllm::dense_nvfp4::DBuf ddown(d, DType::kBF16, {Pk, H});
  vt::MoeGroupedGemmNvfp4Marlin(q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted.t(), eids.t(),
                                npad.t(), dtw, vt::MoeMarlinArgs{bi, 1, Pki, Hi, Ii, false});
  vt::Tensor expert_out = vllm::dense_nvfp4::MakeTensor(ddown.ptr(), DType::kBF16, dev, {1, Pk, H});
  vllm::dense_nvfp4::DBuf dout(d, DType::kBF16, {1, H});
  vt::MoeCombine(q, dout.t(), expert_out, dtw);
  vt::Tensor ot = vllm::dense_nvfp4::MakeTensor(out_dev, DType::kF32, dev, {1, H});
  vt::CastF32(q, ot, dout.t());  // bf16 -> f32, device (no download)
}
#endif  // VT_MARLIN_NVFP4

// N5 campaign-B gate: route the routed experts through the Marlin W4A16 grouped
// MoE GEMM (vLLM's 18.8 kernel). Default OFF (opt-in) until the DGX near-tie + ncu
// gate lands; then flip to default-ON per the parity-enablers-ship-as-defaults
// policy. VT_LAGUNA_MARLIN_MOE=1 enables. Only meaningful when VT_MARLIN_NVFP4
// compiled the path in.
inline bool LagunaMarlinMoeEnabled() {
  // DEFAULT ON: the Marlin W4A16 grouped MoE is vLLM's own 18.8-tok/s kernel and the
  // validated fast Laguna-NVFP4 decode path (reproduced 3× on GB10, golden-matching,
  // ~10 tok/s = ~1.5× the fp4-GEMV fallback). It just works with no env; only an
  // explicit VT_LAGUNA_MARLIN_MOE=0 opts back out to the fp4 GEMV/resident path (the
  // same-binary A/B escape hatch). CUDA + VT_MARLIN_NVFP4 only; a CPU/non-marlin build
  // never compiles the branch, so it falls to the fp4 path automatically.
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_MARLIN_MOE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
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
  // N2 (task #230): safetensors NVFP4 arm — routed experts are per-expert TRUE-W4A4
  // Nvfp4Weight (experts_*_fp4), not the stacked keep-quant OwnedTensor. Everything
  // else on this path (router BF16, shared expert BF16, attention BF16) flows through
  // ReadF32/LqGemm unchanged. The grouped fast-path is keep-quant-only (its op is
  // W4A16 grouped, wrong numerics for W4A4), so it is gated off when fp4.
  const bool fp4 = !lw.moe.experts_gate_fp4.empty();
  if (fp4) {
    VT_CHECK(lw.moe.experts_gate_fp4[0].IsTrueW4A4(),
             "laguna nvfp4 MoE: routed experts must be true-W4A4 (alpha>0)");
    VT_CHECK(lw.moe.experts_up_fp4.size() == lw.moe.experts_gate_fp4.size() &&
                 lw.moe.experts_down_fp4.size() == lw.moe.experts_gate_fp4.size(),
             "laguna nvfp4 MoE: gate/up/down expert counts must match");
  }
  for (int64_t i = 0; i < T; ++i) {
    std::vector<float> hrow(hn2.begin() + static_cast<int64_t>(i * H),
                            hn2.begin() + static_cast<int64_t>((i + 1) * H));
    const std::vector<float> rlog = MatmulNK(hrow, router_w, 1, p.num_experts, H);  // [E]
    const LagunaRouterSelection sel = LagunaUngroupedRouterTopK(
        rlog, bias, p.num_experts_per_tok, p.norm_topk_prob,
        p.moe_routed_scaling_factor);
    std::vector<float> acc(static_cast<size_t>(H), 0.0F);
    const int64_t Pk = static_cast<int64_t>(sel.ids.size());
    if (!fp4 && LagunaGroupedMoeEnabled() && Pk > 0) {
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
    } else if (fp4 && q.device.type != vt::DeviceType::kCPU && Pk > 0) {
      // N5 device-resident MoE (task #234): the whole token's routed experts as ONE
      // async device chain, draining ONCE (vs ~Pk×3 per-GEMM syncs). Byte-neutral to
      // the per-expert path up to the fp4-op near-tie; the CPU path below stays the
      // reference so the run-gate is unchanged. Measures whether killing the syncs
      // wins EAGER on Laguna's fast-kernel profile (ds4's slow-glue precedent may not
      // transfer). VT_LAGUNA_RESIDENT_MOE=0 forces the per-expert path for an A/B.
      const char* dis = std::getenv("VT_LAGUNA_RESIDENT_MOE");
#ifdef VT_MARLIN_NVFP4
      if (LagunaMarlinMoeEnabled()) {
        // N5 campaign-B: route the routed experts through vLLM's 18.8 Marlin W4A16
        // grouped GEMM (opt-in VT_LAGUNA_MARLIN_MOE=1 until the DGX gate lands).
        const int64_t E = static_cast<int64_t>(lw.moe.experts_gate_fp4.size());
        const std::vector<float> racc =
            LagunaMoeResidentMarlin(q, lw.moe, hrow, moe_I, H, E, sel);
        for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += racc[static_cast<size_t>(d)];
      } else
#endif
      if (dis == nullptr || dis[0] != '0') {
        const std::vector<float> racc = LagunaMoeResidentFp4(q, lw.moe, hrow, moe_I, H, sel);
        for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += racc[static_cast<size_t>(d)];
      } else {
        for (size_t s = 0; s < sel.ids.size(); ++s) {
          const int64_t id = sel.ids[s];
          const std::vector<float> eg = LqGemmNvfp4Fp4(
              q, lw.moe.experts_gate_fp4[static_cast<size_t>(id)], hrow, 1, moe_I, H);
          const std::vector<float> eu = LqGemmNvfp4Fp4(
              q, lw.moe.experts_up_fp4[static_cast<size_t>(id)], hrow, 1, moe_I, H);
          const std::vector<float> eact = GateUpSilu(eg, eu, 1, moe_I);
          const std::vector<float> eo = LqGemmNvfp4Fp4(
              q, lw.moe.experts_down_fp4[static_cast<size_t>(id)], eact, 1, H, moe_I);
          const float wgt = sel.weights[s];
          for (int64_t d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += wgt * eo[static_cast<size_t>(d)];
        }
      }
    } else {
      for (size_t s = 0; s < sel.ids.size(); ++s) {
        const int64_t id = sel.ids[s];
        // NVFP4 arm: per-expert TRUE-W4A4 GEMMs (fp4 activation + alpha-scaled
        // accumulate); keep-quant arm: stacked-block row-slice matvecs. Both
        // produce [1,moe_I] gate/up then [1,H] down, so the SwiGLU + combine below
        // are shared verbatim.
        const std::vector<float> eg =
            fp4 ? LqGemmNvfp4Fp4(q, lw.moe.experts_gate_fp4[static_cast<size_t>(id)],
                                 hrow, 1, moe_I, H)
                : LqGemmRowSlice(q, lw.moe.experts_gate, hrow, 1, moe_I, H, id * moe_I);
        const std::vector<float> eu =
            fp4 ? LqGemmNvfp4Fp4(q, lw.moe.experts_up_fp4[static_cast<size_t>(id)],
                                 hrow, 1, moe_I, H)
                : LqGemmRowSlice(q, lw.moe.experts_up, hrow, 1, moe_I, H, id * moe_I);
        const std::vector<float> eact = GateUpSilu(eg, eu, 1, moe_I);
        const std::vector<float> eo =
            fp4 ? LqGemmNvfp4Fp4(q, lw.moe.experts_down_fp4[static_cast<size_t>(id)],
                                 eact, 1, H, moe_I)
                : LqGemmRowSlice(q, lw.moe.experts_down, eact, 1, H, moe_I, id * H);
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

// N5 campaign-B: build ALL routed-expert Marlin residents at model-LOAD time
// (mirrors vLLM's process_weights_after_loading, marlin_utils_fp4.py), instead of
// lazily on the first forward — so the 48L×256E repack cost is paid once at load,
// not as a first-token TTFT spike. No-op unless the Marlin path is enabled + on GPU
// (+ built with VT_MARLIN_NVFP4). The forward's lazy `if (!mr.ready) Build…` then
// finds every resident ready. Safe: only runs under LagunaMarlinMoeEnabled(), so the
// GEMV/CPU paths (which read the fp4 originals it frees) can never run this process.
// External linkage (declared in laguna.h); the anon-namespace helpers it calls stay
// visible here via the anonymous namespace's implicit using-directive.
void LagunaBuildMarlinResidents(vt::Queue& q, const LagunaWeights& w) {
#ifdef VT_MARLIN_NVFP4
  if (q.device.type == vt::DeviceType::kCPU || !LagunaMarlinMoeEnabled()) return;
  vt::Backend& bk = vt::GetBackend(q.device.type);
  vllm::dense_nvfp4::Dev d{bk, q};
  for (const LagunaLayerWeights& lw : w.layers) {
    if (lw.is_dense || lw.moe.experts_gate_fp4.empty()) continue;
    const int E = static_cast<int>(lw.moe.experts_gate_fp4.size());
    const int N = static_cast<int>(lw.moe.experts_gate_fp4[0].n);  // moe_intermediate
    const int K = static_cast<int>(lw.moe.experts_gate_fp4[0].k);  // hidden_size
    LagunaMoeMarlinResident& mr = LagunaMoeMarlinResidentFor(&lw.moe);
    if (!mr.ready) BuildLagunaMoeMarlinResident(d, lw.moe, E, K, N, mr);
  }
#else
  (void)q;
  (void)w;
#endif
}

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
  VT_CHECK(weights.has_gguf_weights || weights.has_nvfp4_weights,
           "laguna forward: no keep-quant or nvfp4 tower");
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
// ── N5 device-resident T=1 decode (VT_LAGUNA_RESIDENT_DECODE, default OFF) ─────
// The NVFP4/Marlin arm's per-GEMM host drains (~432 cudaStreamSynchronize/token,
// MEASURED) are the 1.9x gap to vLLM 18.8. This v1 runs the ATTENTION block fully
// device-resident (the 5 kLaguna glue kernels + no-sync bf16 GEMMs), collapsing the
// ~5 per-layer attention-GEMM drains into 2; the FFN reuses the existing host path
// (its own minimal drains) for now (v2 = device-resident FFN). BYTE-EXACT: the
// kLaguna kernels use sequential reductions matching the host float order, and the
// GEMMs are the same bf16 MatmulBT as the current golden. Gated on the golden ids
// staying identical (VT_LAGUNA_RESIDENT_DECODE=0/1 A/B).
inline bool LagunaResidentDecodeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_RESIDENT_DECODE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

bool LagunaCanRunResidentDecode(const LagunaParams& p, vt::Queue& q, const LagunaWeights& w,
                                int64_t T) {
#ifndef VT_MARLIN_NVFP4
  (void)p; (void)q; (void)w; (void)T;
  return false;  // the resident FFN routes the MoE through the Marlin W4A16 path
#else
  if (!LagunaResidentDecodeEnabled() || T != 1) return false;
  if (q.device.type == vt::DeviceType::kCPU) return false;
  if (!w.has_nvfp4_weights) return false;
  if (p.tie_word_embeddings || w.lm_head.Empty()) return false;
  if (!laguna::LagunaDeviceKernelsAvailable()) return false;
  return true;
#endif
}

std::vector<float> LagunaForwardResidentDecode(const LagunaWeights& weights, vt::Queue& q,
                                               LagunaKvCache& cache,
                                               const std::vector<int32_t>& token_ids,
                                               const std::vector<int32_t>& positions,
                                               const std::vector<int32_t>& logits_indices) {
  using vt::DType;
  const LagunaParams& p = weights.params;
  const int64_t H = p.hidden_size, Vsz = p.vocab_size, Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads, kvdim = Hkv * Dh;
  const int64_t nlayers = p.num_hidden_layers;
  const float eps = p.rms_norm_eps;
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  const vt::Device dev = q.device;
  const int64_t pos = positions[0];
  if (cache.k.empty()) cache.Reset(nlayers, Dh, Hkv);

  const laguna::LagunaDeviceKernels* LAG = laguna::LagunaDevice();

  int64_t Hq_max = p.num_attention_heads;
  for (int64_t l = 0; l < nlayers; ++l) Hq_max = std::max(Hq_max, p.QHeadsForLayer(l));
  const int64_t qdim_max = Hq_max * Dh;
  const int64_t E = p.num_experts, topk = p.num_experts_per_tok;
  const int64_t moe_I = p.moe_intermediate_size, dense_I = p.intermediate_size;
  const int64_t maxI = std::max(moe_I, dense_I);
  const int64_t maxK = std::max({H, qdim_max, moe_I, dense_I});

  const std::vector<float> yarn_cache = BuildLagunaFullYarnCosSin(p, pos + 1);
  const std::vector<float> slide_cache = BuildLagunaSlidingCosSin(p, pos + 1);

  std::vector<float> hidden = LagunaEmbed(weights.embed, token_ids, H, Vsz);
  std::vector<float> hn(static_cast<size_t>(H)), qv(static_cast<size_t>(qdim_max)),
      gl(static_cast<size_t>(Hq_max)), attn(static_cast<size_t>(qdim_max)),
      o(static_cast<size_t>(H));
  std::vector<uint16_t> abf(static_cast<size_t>(maxK));
  // device-FFN scratch (unified) + persistent f32 norm/bias keep-alive (no per-FFN drain)
  std::vector<float> gating(static_cast<size_t>(E)), dg(static_cast<size_t>(maxI)),
      du(static_cast<size_t>(maxI)), dact(static_cast<size_t>(maxI)), fdn(static_cast<size_t>(H)),
      so(static_cast<size_t>(H)), doutb(static_cast<size_t>(H));
  std::vector<int32_t> eids32(static_cast<size_t>(topk));
  std::vector<float> topw(static_cast<size_t>(topk));
  std::vector<std::vector<float>> keep;
  auto Keep = [&](std::vector<float> v) -> const float* {
    keep.push_back(std::move(v));
    return keep.back().data();
  };
  auto DevT = [&](float* p2, int64_t I) {
    return vt::Tensor::Contiguous(p2, DType::kF32, dev, {1, I});
  };
  auto SiluMul = [&](float* out, float* g, float* u, int64_t I) {
    vt::Tensor ot = DevT(out, I);
    vt::MoeSiluMul(q, ot, DevT(g, I), DevT(u, I));
  };
  auto AddInto = [&](float* a, float* b) {  // a += b  (both [1,H])
    vt::Tensor at = DevT(a, H);
    vt::Add(q, at, at, DevT(b, H));
  };

  // no-sync bf16 GEMM: out[1,N] f32 = cast(x[1,K])·w[N,K]^T (mirror LqGemm device path)
  auto GemmBf16Into = [&](float* out, const OwnedTensor& w, const float* x, int64_t N, int64_t K) {
    vt::Tensor xf = vt::Tensor::Contiguous(const_cast<float*>(x), DType::kF32, dev, {1, K});
    vt::Tensor ab = vt::Tensor::Contiguous(abf.data(), DType::kBF16, dev, {1, K});
    vt::CastBf16(q, ab, xf);
    vt::Tensor ot = vt::Tensor::Contiguous(out, DType::kF32, dev, {1, N});
    vt::Tensor wt = w.View();
    wt.device = dev;
    vt::MatmulBT(q, ot, ab, wt);
  };

  for (int64_t l = 0; l < nlayers; ++l) {
    const LagunaLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const int64_t Hq = p.QHeadsForLayer(l), qdim = Hq * Dh;
    const int64_t group = p.GqaGroupForLayer(l);
    const bool global = p.IsGlobalLayer(l);
    const int64_t rd = p.RotaryDimForLayer(l);
    const int64_t window = p.WindowForLayer(l);
    const float* rcache = (global ? yarn_cache : slide_cache).data();

    // Norm weights via ReadF32 (bf16->f32 exact, matching the host golden); layer-scoped
    // so they outlive both DrainQueues.
    const std::vector<float> w_in = ReadF32(lw.input_norm);
    LAG->rms_norm_seq(q, hn.data(), hidden.data(), w_in.data(), 1, H, eps, true);
    std::vector<float> knew(static_cast<size_t>(kvdim)), vnew(static_cast<size_t>(kvdim));
    GemmBf16Into(qv.data(), lw.attn.q_proj, hn.data(), qdim, H);
    GemmBf16Into(knew.data(), lw.attn.k_proj, hn.data(), kvdim, H);
    GemmBf16Into(vnew.data(), lw.attn.v_proj, hn.data(), kvdim, H);
    std::vector<float> w_qn, w_kn;
    if (p.has_qk_norm && !lw.attn.q_norm.Empty()) {
      w_qn = ReadF32(lw.attn.q_norm);
      w_kn = ReadF32(lw.attn.k_norm);
      LAG->rms_norm_seq(q, qv.data(), qv.data(), w_qn.data(), Hq, Dh, eps, true);
      LAG->rms_norm_seq(q, knew.data(), knew.data(), w_kn.data(), Hkv, Dh, eps, true);
    }
    LAG->rope_from_cache(q, qv.data(), rcache, Hq, Dh, rd, pos);
    LAG->rope_from_cache(q, knew.data(), rcache, Hkv, Dh, rd, pos);
    DrainQueue(q);  // knew/vnew host-readable for the cache append + eviction

    std::vector<float>& kc = cache.k[static_cast<size_t>(l)];
    std::vector<float>& vc = cache.v[static_cast<size_t>(l)];
    kc.insert(kc.end(), knew.begin(), knew.end());
    vc.insert(vc.end(), vnew.begin(), vnew.end());
    int64_t rows = static_cast<int64_t>(kc.size()) / kvdim;
    if (window > 0 && rows > window) {
      const int64_t drop = rows - window;
      const size_t off = static_cast<size_t>(drop * kvdim);
      kc.erase(kc.begin(), kc.begin() + static_cast<int64_t>(off));
      vc.erase(vc.begin(), vc.begin() + static_cast<int64_t>(off));
      cache.first_pos[static_cast<size_t>(l)] += drop;
      rows = window;
    }
    const int64_t fp = cache.first_pos[static_cast<size_t>(l)];

    LAG->decode_attn_gqa(q, attn.data(), qv.data(), kc.data(), vc.data(), Hq, Hkv, Dh, group, rows,
                         pos, fp, window, scale);
    GemmBf16Into(gl.data(), lw.attn.g_proj, hn.data(), Hq, H);
    LAG->softplus_head_gate(q, attn.data(), gl.data(), Hq, Dh);
    GemmBf16Into(o.data(), lw.attn.o_proj, attn.data(), H, qdim);
    {
      vt::Tensor ht = vt::Tensor::Contiguous(hidden.data(), DType::kF32, dev, {1, H});
      vt::Tensor ot = vt::Tensor::Contiguous(o.data(), DType::kF32, dev, {1, H});
      vt::Add(q, ht, ht, ot);
    }
    // post-attn RMSNorm (device); FFN fully device-resident — NO per-layer FFN drain.
    LAG->rms_norm_seq(q, hn.data(), hidden.data(), Keep(ReadF32(lw.post_attn_norm)), 1, H, eps,
                      true);
    if (lw.is_dense) {
      GemmBf16Into(dg.data(), lw.mlp.gate_proj, hn.data(), dense_I, H);
      GemmBf16Into(du.data(), lw.mlp.up_proj, hn.data(), dense_I, H);
      SiluMul(dact.data(), dg.data(), du.data(), dense_I);
      GemmBf16Into(fdn.data(), lw.mlp.down_proj, dact.data(), H, dense_I);
      AddInto(hidden.data(), fdn.data());
    } else {
#ifdef VT_MARLIN_NVFP4
      GemmBf16Into(gating.data(), lw.moe.router, hn.data(), E, H);  // router (bf16 weight)
      const float* bias = lw.moe.e_score_correction_bias.Empty()
                              ? nullptr
                              : Keep(ReadF32(lw.moe.e_score_correction_bias));
      LAG->sigmoid_topk(q, eids32.data(), topw.data(), gating.data(), bias, bias != nullptr, E,
                        topk, p.norm_topk_prob, p.moe_routed_scaling_factor);
      LagunaMoeResidentMarlinInto(q, lw.moe, hn.data(), moe_I, H, E, eids32.data(), topw.data(),
                                  topk, doutb.data());
      GemmBf16Into(dg.data(), lw.moe.shared_gate, hn.data(), moe_I, H);  // shared expert
      GemmBf16Into(du.data(), lw.moe.shared_up, hn.data(), moe_I, H);
      SiluMul(dact.data(), dg.data(), du.data(), moe_I);
      GemmBf16Into(so.data(), lw.moe.shared_down, dact.data(), H, moe_I);
      AddInto(hidden.data(), doutb.data());  // hidden += routed
      AddInto(hidden.data(), so.data());     // hidden += shared
#else
      VT_CHECK(false, "laguna resident decode: MoE requires the VT_MARLIN_NVFP4 build");
#endif
    }
  }
  DrainQueue(q);  // the ONE step-boundary drain: hidden coherent for the final logits
  return LagunaFinalLogits(q, weights, hidden, 1, logits_indices);
}

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
  VT_CHECK(weights.has_gguf_weights || weights.has_nvfp4_weights,
           "laguna cached forward: no keep-quant or nvfp4 tower");
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

  // N5 device-resident T=1 decode fast path (VT_LAGUNA_RESIDENT_DECODE, default OFF).
  if (LagunaCanRunResidentDecode(p, q, weights, T) && logits_indices.size() == 1 &&
      logits_indices[0] == 0) {
    std::vector<float> lg =
        LagunaForwardResidentDecode(weights, q, cache, token_ids, positions, logits_indices);
    cache.len += T;
    return lg;
  }

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
