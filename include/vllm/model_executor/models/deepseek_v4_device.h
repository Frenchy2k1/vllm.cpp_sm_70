// DeepSeek-V4-Flash W7-device — the CUDA kernel seam for the four NEW V4 op
// families. This header declares the DEVICE launchers (1:1 CUDA ports of the
// landed portable HOST references) and the OpProvider-seam resolvers the device
// forward (DeepseekV4Model::ForwardDevice) and the unit gate dispatch through.
//
// ─── WHAT THIS IS A PORT OF (the four families; file:line on BOTH sides) ─────
//   MHC        <- deepseek_v4_mhc.{h,cpp}        (Sinkhorn / MhcPre / MhcPost /
//                                                 HcHeadCollapse) @ kernels/mhc/*.py
//   DSA        <- deepseek_v4_dsa.{h,cpp}        (indexer weight-fold / MQA logits /
//                                                 causal top-k / sink softmax /
//                                                 grouped output-LoRA)
//   Compressor <- deepseek_v4_compressor.{h,cpp} (save-time APE / pool+norm /
//                                                 fp8_ds_mla KV encode+decode)
//   MoE        <- deepseek_v4_moe.{h,cpp}         (sqrtsoftplus / hash+bias router /
//                                                 clamped SwiGLU)
// The 512-wide MLA attention + expert grouped-GEMM REUSE the existing NVFP4/FP8
// CUDA paths (cuda_mla_attn.cu, cuda_moe*.cu) and are NOT re-ported here — only
// these four NEW glue families need dedicated V4 kernels.
//
// ─── HONEST SCOPE (mirrors W3-W7) ────────────────────────────────────────────
// The launchers take/return host std::vectors and upload/run/download internally
// (via the CUDA backend). That is a STRUCTURAL, correctness-grade path — each
// kernel is unit-gated on the DGX GB10 against its landed host reference at a
// SMALL synthetic shape (test_cuda_deepseek_v4.cpp) — NOT a fused/perf path. The
// per-op host round-trip lets a CPU-compiled TU (deepseek_v4.cpp ForwardDevice)
// drive the kernels through the seam without linking CUDA symbols; the real
// paged-engine e2e over a materialized checkpoint stays the W8 residual (the
// fixed-config 167B does not fit ONE GB10 — see deepseek_v4.h).
//
// SEAM: each family registers ONE OpProvider under a dedicated OpId
// (vt/ops.h: kDeepseekV4{Mhc,Dsa,Compressor,Moe}); the provider `fn` points at a
// static kernels-struct of typed device launchers. The resolvers below cast the
// GetOp() result; they THROW on a CPU-only build (nothing registered for kCUDA),
// which is correct — ForwardDevice is a device-only path.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_compressor.h"  // Fp8DsMlaLayout / Fp8DsMlaToken
#include "vllm/model_executor/models/deepseek_v4_mhc.h"          // MhcPreResult
#include "vllm/model_executor/models/deepseek_v4_moe.h"          // MoeRouteResult
#include "vt/device.h"                                           // vt::Queue

namespace vllm::deepseek_v4 {

// ── (1) MHC family device kernels ─────────────────────────────────────────────
struct MhcDeviceKernels {
  std::vector<float> (*sinkhorn)(vt::Queue&, const std::vector<float>& comb_logits,
                                 int64_t hc, int64_t iters, float eps);
  MhcPreResult (*pre)(vt::Queue&, const std::vector<float>& residual,
                      const std::vector<float>& fn, const std::vector<float>& scale,
                      const std::vector<float>& base, int64_t hc, int64_t hidden,
                      float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                      float hc_post_mult, int64_t iters,
                      const std::vector<float>& norm_weight, float norm_eps);
  std::vector<float> (*post)(vt::Queue&, const std::vector<float>& x,
                             const std::vector<float>& residual,
                             const std::vector<float>& post_layer_mix,
                             const std::vector<float>& comb_res_mix, int64_t hc,
                             int64_t hidden);
  std::vector<float> (*head)(vt::Queue&, const std::vector<float>& x,
                             const std::vector<float>& fn, float scale,
                             const std::vector<float>& base, int64_t hc, int64_t hidden,
                             float rms_eps, float hc_eps);
  // Brick B — IN-PLACE MHC glue (unified memory, no Upload/Download/Sync; caller
  // drains). Same MhcPost/HcHead/MhcPreKernel as above. `mix_scratch` is a
  // caller-provided [(2+hc)*hc] unified buffer for pre's intermediate mixes.
  void (*post_ip)(vt::Queue&, float* out, const float* x, const float* residual,
                  const float* post_mix, const float* comb, int64_t hc, int64_t hidden);
  void (*head_ip)(vt::Queue&, float* out, const float* x, const float* fn, float scale,
                  const float* base, int64_t hc, int64_t hidden, float rms_eps, float hc_eps);
  void (*pre_ip)(vt::Queue&, float* pre_mix, float* post_mix, float* comb_mix,
                 float* layer_input, float* mix_scratch, const float* residual, const float* fn,
                 const float* scale, const float* base, int64_t hc, int64_t hidden, float rms_eps,
                 float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult, int64_t iters,
                 const float* norm_weight, bool has_norm, float norm_eps);
};

// ── (2) DSA family device kernels ─────────────────────────────────────────────
struct DsaDeviceKernels {
  std::vector<float> (*weight_fold)(vt::Queue&, const std::vector<float>& weights_proj,
                                    int64_t num_tokens, int64_t index_n_heads,
                                    int64_t index_head_dim);
  std::vector<float> (*logits)(vt::Queue&, const std::vector<float>& q,
                               const std::vector<float>& k,
                               const std::vector<float>& folded_weights,
                               const std::vector<int64_t>& win_start,
                               const std::vector<int64_t>& win_end, int64_t num_tokens,
                               int64_t num_keys, int64_t index_n_heads,
                               int64_t index_head_dim);
  std::vector<int64_t> (*topk)(vt::Queue&, const std::vector<float>& logits,
                               const std::vector<int64_t>& win_start,
                               const std::vector<int64_t>& win_end, int64_t num_tokens,
                               int64_t num_keys, int64_t topk);
  std::vector<float> (*softmax_sink)(vt::Queue&, const std::vector<float>& scores, float sink);
  std::vector<float> (*grouped_olora)(vt::Queue&, const std::vector<float>& o,
                                      const std::vector<float>& wo_a,
                                      const std::vector<float>& wo_b, int64_t num_tokens,
                                      int64_t n_heads, int64_t head_dim, int64_t n_groups,
                                      int64_t o_lora_rank, int64_t hidden_size);
  // Brick A — device MLA decode/prefill attention over the unified KV-cache latent.
  // Unlike the launchers above (host-vector, Upload/Download/Sync), this reads/writes
  // UNIFIED-MEMORY raw pointers IN PLACE on the queue stream (no round-trip) — the
  // first real device V4 forward kernel, toward a capturable decode graph. q
  // [T*nh*hd], kv [n_keys*hd] (cached deck; num KV heads = 1, shared across heads),
  // sink [nh], o [T*nh*hd], all on the queue device. Causal: query t attends keys
  // [0, kv_base+t]. no_sink = the kNoAttnSink miswire (sink -> -inf). Matches
  // SoftmaxWithSink (deepseek_v4_dsa.cpp:116) with host accumulation order preserved.
  // Launches async on q's stream; the CALLER drains (Brick A) or captures (Brick D).
  void (*decode_attn)(vt::Queue&, float* o, const float* q, const float* kv,
                      const float* sink, int64_t nh, int64_t hd, int64_t kv_base,
                      int64_t num_tokens, float scale, bool no_sink);
  // Brick C — folded-in device glue (in place on unified/device buffers, no
  // Upload/Download/Sync; caller drains at Brick C / captures at Brick D).
  // rms_norm: weighted RMSNorm over [n] (has_w=false → the per-head q-RMS). Near-tie
  // (block reduction reorders vs host double-sequential).
  void (*rms_norm)(vt::Queue&, float* out, const float* x, const float* w, int64_t n, float eps,
                   bool has_w);
  // rope: sequential-recurrence RoPE over `num_rows` rows (each v[row*row_stride+off..+r]),
  // per-row position `row_pos[row]`. inverse flips the sin sign. Near-tie (cos/sin lib).
  void (*rope)(vt::Queue&, float* v, int64_t num_rows, int64_t row_stride, int64_t off, int64_t r,
               const int* row_pos, double base, double freq_scale, double ext_factor,
               int64_t n_ctx_orig, double beta_fast, double beta_slow, bool inverse);
  // Brick C part 2 — BATCHED RMSNorm over `rows` independent [n] segments in ONE
  // launch (the per-head q-RMS: rows=nh, has_w=false). Per-row identical to rms_norm
  // above (same block reduction; a shared weight w[n] applies to every row when has_w).
  void (*rms_norm_rows)(vt::Queue&, float* out, const float* x, const float* w, int64_t rows,
                        int64_t n, float eps, bool has_w);
};

// ── (3) Compressor family device kernels ──────────────────────────────────────
struct CompressorDeviceKernels {
  std::vector<float> (*save_score_ape)(vt::Queue&, const std::vector<float>& score,
                                       const std::vector<float>& ape,
                                       const std::vector<int64_t>& positions,
                                       int64_t num_tokens, int64_t width,
                                       int64_t compress_ratio);
  std::vector<float> (*pool_norm)(vt::Queue&, const std::vector<float>& kv,
                                  const std::vector<float>& score,
                                  const std::vector<uint8_t>& valid,
                                  const std::vector<float>& rms_weight, float eps,
                                  int64_t window, int64_t head_dim);
  Fp8DsMlaToken (*encode)(vt::Queue&, const std::vector<float>& head,
                          const Fp8DsMlaLayout& layout);
  std::vector<float> (*decode)(vt::Queue&, const Fp8DsMlaToken& token,
                               const Fp8DsMlaLayout& layout);
};

// ── (4) MoE family device kernels ─────────────────────────────────────────────
struct MoeDeviceKernels {
  // Elementwise sqrt(softplus(x)) over an arbitrary buffer (the router score).
  std::vector<float> (*sqrtsoftplus)(vt::Queue&, const std::vector<float>& x);
  MoeRouteResult (*route)(vt::Queue&, const std::vector<float>& gating, int64_t num_tokens,
                          int64_t num_experts, int64_t topk,
                          const std::vector<float>& e_score_correction_bias, bool renormalize,
                          float routed_scaling_factor,
                          const std::vector<int64_t>& input_tokens,
                          const std::vector<int32_t>& hash_indices_table, int64_t vocab_size);
  std::vector<float> (*clamped_swiglu)(vt::Queue&, const std::vector<float>& gate_up,
                                       int64_t d, float limit, float alpha, float beta);
  // Brick B — IN-PLACE clamped-SwiGLU: reads gate_up[2*d], writes out[d] on the
  // queue device (unified memory), NO Upload/Download/Sync (caller drains at Brick
  // B / captures at Brick D). Same ClampedSwiGLUKernel math as clamped_swiglu above
  // ⇒ bit-identical (elementwise, no reduction).
  void (*clamped_swiglu_ip)(vt::Queue&, float* out, const float* gate_up, int64_t d,
                            float limit, float alpha, float beta);
  // Brick B — IN-PLACE router: same RouteKernel; writes topk_ids[T*topk] (i32) +
  // topk_weights[T*topk] on the queue device (unified), no Upload/Download/Sync.
  void (*route_ip)(vt::Queue&, int32_t* topk_ids, float* topk_weights, const float* gating,
                   int64_t T, int64_t E, int64_t topk, const float* bias, bool has_bias,
                   const int64_t* in_tokens, bool is_hash, const int32_t* hashtab,
                   int64_t vocab, bool renorm, float scale);
  // Brick C — MoE combine: out[h] = Σ_a weights[a]*eo[a*H+h] (per-h sequential over
  // the A experts; near-tie vs host — device FMA contraction). In place on the queue.
  void (*moe_combine)(vt::Queue&, float* out, const float* eo, const float* weights, int64_t A,
                      int64_t H);
};

// Resolve a family's device kernels through the vt OpProvider seam. THROWS on a
// CPU-only build (no kCUDA provider registered) — ForwardDevice is device-only.
const MhcDeviceKernels* MhcDevice();
const DsaDeviceKernels* DsaDevice();
const CompressorDeviceKernels* CompressorDevice();
const MoeDeviceKernels* MoeDevice();

// True iff the CUDA backend registered the four V4 families (a device build with
// a live CUDA backend). ForwardDevice checks this before dispatch.
bool V4DeviceKernelsAvailable();

}  // namespace vllm::deepseek_v4
