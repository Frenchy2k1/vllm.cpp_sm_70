// Kimi Delta Attention (KDA) — the genuinely-NEW gated-linear-attention
// primitives that KDA adds ON TOP of plain GDN, as portable host (CPU)
// reference implementations. Shared unblocker for BOTH Kimi-Linear-48B
// (MODEL-TEXT-kimi-linear-*) and Kimi-K3 (MODEL-MM-kimi-k3-*, W4).
//
// KimiGatedDeltaNetAttention SUBCLASSES GatedDeltaNetAttention
// (kimi_gdn_linear_attn.py:85) — the conv-state/cache layout,
// GDNAttentionMetadata, conv update, chunked-delta recurrence and WY solve are
// REUSED from our landed GDN (src/vt/cuda/cuda_gdn.cu,
// src/vllm/v1/attention/backends/gdn_attn.cpp). What plain GDN does NOT have,
// and what THIS file references, are the four KDA deltas:
//
//   (1) a per-channel [H,D] LOW-RANK DECAY (f_a_proj -> f_b_proj bottleneck),
//       where plain GDN has only a per-HEAD scalar decay from A_log;
//   (2) the DECAY GATE itself: g = -exp(A_log[h]) * softplus(g1 + dt_bias),
//       per channel (kda_gate_fwd_kernel), plus its chunk-local cumulative-sum
//       prefill variant (kda_gate_cumsum_fwd_kernel);
//   (3) the SIGMOID-gated output norm FusedRMSNormGated(head_dim,
//       activation="sigmoid") — the gated-linear-attention output norm plain
//       GDN lacks;
//   (4) THREE separate q/k/v short causal convs (conv_size=4, silu), and the
//       q/k L2-norm preprocessing (use_qk_l2norm_in_kernel=True).
//
// WHY host/CPU reference (honest scope): the full KDA forward is a chunked
// gated-delta recurrence that reuses the GDN device machinery; the KDA-specific
// deltas above are the net-new numerics, and they are the cleanest first
// bricks. The REAL end-to-end gate is the Kimi-Linear-48B-A3B proxy vs the
// pinned vLLM oracle on GB10 (DGX-blocked — a NAMED residual; the 2.8T K3 does
// not fit one GB10). So this file lands + UNIT-GATES the KDA deltas against
// hand-derived literal cases and a from-first-principles double-precision
// reference (tests/vllm/models/test_kimi_kda.cpp), NOT a dumped-oracle rel-L2.
// The eventual GPU forward will call the SAME math from a CUDA kernel; this
// file pins the numerics portably so the kernel port has an oracle.
//
// SACRED-inert: additive TU only. It does NOT touch the shared GDN kernels
// (src/vt/cuda/cuda_gdn.cu, src/vllm/v1/attention/backends/gdn_attn.cpp), so the
// Qwen3.6-27B/35B GDN gate is structurally untouched (exactly as the DeepSeek-V4
// DSA lane kept the shared MLA path untouched). KDA lands as KDA-specific refs.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   OURS                   <-  UPSTREAM (vllm/, @ 0.26.0.dev0)
//   KdaLowRankDecay        <-  models/layers/mamba/gdn/kimi_gdn_linear_attn.py
//                              :142-156 (f_a_proj hidden->head_dim,
//                               f_b_proj head_dim->H*D) + :245
//                               (g1 = f_b_proj(f_a_proj(x)))
//   KdaDecayGate           <-  third_party/flash_linear_attention/ops/kda.py
//                              :1541-1600 (kda_gate_fwd_kernel) + :1603-1646
//                              (fused_kda_gate); model call
//                               kimi_gdn_linear_attn.py:420-425 (decode path)
//   KdaDecayGateChunkCumsum<-  kda.py:1182-1254 (kda_gate_cumsum_fwd_kernel:
//                               same gate then chunk-local cumsum * RCP_LN2) +
//                               :1257-1303 (fused_kda_gate_chunk_cumsum);
//                               model call kimi_gdn_linear_attn.py:403-415
//   FusedRMSNormGated      <-  kda.py:463-487 (forward_native) + :436
//                              (FusedRMSNormGated ctor, eps=1e-5);
//                               model kimi_gdn_linear_attn.py:219,:266
//   KdaShortConv           <-  kimi_gdn_linear_attn.py:171-198 (q/k/v_conv1d)
//                              + :324-356 (causal_conv1d_fn activation="silu");
//                               ops/causal_conv1d.py
//   L2NormRows             <-  kda.py:1511-1513 (use_qk_l2norm_in_kernel) +
//                               ops/l2norm.py:42-43,:96 (eps=1e-6, sum not mean)
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::kimi_kda {

// The natural-log -> log2 conversion constant the chunk-cumsum kernel folds in
// (kda.py:1295 cumsum_scale=RCP_LN2) so downstream exp2 kernels reproduce
// exp(g). = 1 / ln(2).
inline constexpr double kRcpLn2 = 1.4426950408889634;

// ── (1) per-channel [H,D] low-rank decay projection ──────────────────────────

// The f_a_proj -> f_b_proj bottleneck that produces the per-channel decay input
// g1 (kimi_gdn_linear_attn.py:142-156,:245). f_a projects hidden -> head_dim
// (the RANK), f_b projects head_dim -> H*D. Plain GDN has NO such projection —
// its decay is a per-head scalar. Both projections are bias-free linears; there
// is NO activation between them (a pure low-rank linear map).
//
//   x     : [num_tokens, hidden_size]              row-major
//   f_a   : [head_dim, hidden_size]                row-major (out=head_dim)
//   f_b   : [num_heads*head_dim, head_dim]         row-major (out=H*D)
// Returns g1 : [num_tokens, num_heads*head_dim]    row-major.
std::vector<float> KdaLowRankDecay(const std::vector<float>& x,
                                   const std::vector<float>& f_a,
                                   const std::vector<float>& f_b,
                                   int64_t num_tokens, int64_t hidden_size,
                                   int64_t num_heads, int64_t head_dim);

// ── (2) the KDA decay gate ───────────────────────────────────────────────────

// fused_kda_gate (kda_gate_fwd_kernel, kda.py:1541-1600): per (token,head,chan)
//     b_g  = g1[t,h,d] + dt_bias[h,d]                    (dt_bias optional)
//     b_a  = -exp(A_log[h])                              (per-head scalar)
//     sp   = softplus_beta(b_g)                          (beta,threshold below)
//     y    = b_a * sp
// with softplus_beta(x) = x                if beta*x > threshold
//                       = (1/beta)*log(1+exp(beta*x))  otherwise.
// The model uses beta=1.0, threshold=20.0 (fused_kda_gate defaults). This is
// the per-channel [H,D] log-decay plain GDN's per-head scalar decay lacks.
//
//   g1      : [num_tokens, num_heads*head_dim]  row-major (KdaLowRankDecay out)
//   a_log   : [num_heads]                        row-major
//   dt_bias : [num_heads*head_dim] or empty      row-major (empty => no bias)
// Returns y : [num_tokens, num_heads, head_dim]  row-major.
std::vector<float> KdaDecayGate(const std::vector<float>& g1,
                                const std::vector<float>& a_log,
                                const std::vector<float>& dt_bias,
                                int64_t num_tokens, int64_t num_heads,
                                int64_t head_dim, float beta = 1.0f,
                                float threshold = 20.0f);

// fused_kda_gate_chunk_cumsum (kda_gate_cumsum_fwd_kernel, kda.py:1182-1254):
// the PREFILL variant — the SAME per-channel gate as KdaDecayGate, followed by a
// chunk-local (reset at every `chunk_size` boundary) cumulative sum along time,
// scaled by RCP_LN2 (kda.py:1252-1253). The cumsum turns per-step log-decays
// into the cumulative log-decay the chunked recurrence consumes; the RCP_LN2
// fold is a representation detail so the downstream exp2 kernels reproduce
// exp(g) — pass log2_domain=false to get the plain natural-log cumulative sum.
//
//   g1..dt_bias, dims : as KdaDecayGate
//   chunk_size        : FLA chunk length (cumsum resets at each boundary)
//   log2_domain       : true => scale by RCP_LN2 (kernel-exact); false => raw
// Returns y : [num_tokens, num_heads, head_dim] row-major cumulative log-decay.
std::vector<float> KdaDecayGateChunkCumsum(const std::vector<float>& g1,
                                           const std::vector<float>& a_log,
                                           const std::vector<float>& dt_bias,
                                           int64_t num_tokens, int64_t num_heads,
                                           int64_t head_dim, int64_t chunk_size,
                                           bool log2_domain = true,
                                           float beta = 1.0f,
                                           float threshold = 20.0f);

// ── (3) the sigmoid-gated output norm ────────────────────────────────────────

enum class GatedNormActivation { kSigmoid, kSwish };

// FusedRMSNormGated.forward_native (kda.py:463-487). RMS-normalise x over the
// last (head_dim) axis, scale by the affine weight, then apply the gate:
//     x_normed[d] = x[t,h,d] * rsqrt(mean_d(x^2) + eps) * weight[d]
//     sigmoid: out = x_normed * sigmoid(g)
//     swish  : out = x_normed * g * sigmoid(g)
// KDA uses activation="sigmoid" (kimi_gdn_linear_attn.py:219), eps=1e-5
// (the FusedRMSNormGated ctor default, kda.py:441). `weight` may be empty for
// the no-affine case. Applied per (token,head) over head_dim.
//
//   x, g   : [num_tokens, num_heads, head_dim]  row-major
//   weight : [head_dim] or empty                row-major
// Returns out : [num_tokens, num_heads, head_dim] row-major.
std::vector<float> FusedRMSNormGated(
    const std::vector<float>& x, const std::vector<float>& g,
    const std::vector<float>& weight, int64_t num_tokens, int64_t num_heads,
    int64_t head_dim, GatedNormActivation activation = GatedNormActivation::kSigmoid,
    float eps = 1e-5f);

// ── (4) short causal conv + q/k L2-norm ──────────────────────────────────────

// One of the three separate q/k/v depthwise causal short convs
// (kimi_gdn_linear_attn.py:171-198), applied with activation="silu"
// (:324-356, causal_conv1d_fn). Depthwise (per channel), causal (only current
// and past positions), zero initial state (fresh sequence):
//     pre[t,c] = bias[c] + sum_{j=0}^{K-1} w[c,j] * x[t-(K-1)+j, c]
//                (indices < 0 read 0 — zero left-pad / empty conv state)
//     y[t,c]   = silu(pre[t,c]) = pre * sigmoid(pre)
//
//   x      : [num_tokens, channels]      row-major
//   weight : [channels, kernel_size]     row-major (K = short_conv_kernel_size)
//   bias   : [channels] or empty         row-major (KDA convs are bias-free)
// Returns y : [num_tokens, channels]     row-major.
std::vector<float> KdaShortConv(const std::vector<float>& x,
                                const std::vector<float>& weight,
                                const std::vector<float>& bias,
                                int64_t num_tokens, int64_t channels,
                                int64_t kernel_size);

// The q/k L2-norm preprocessing (use_qk_l2norm_in_kernel=True, kda.py:1511-1513;
// l2norm_fwd l2norm.py:42-43). Normalise each row over the last axis by its L2
// norm — note this divides by sqrt(SUM of squares + eps), NOT the mean (eps=1e-6
// default, l2norm.py:96):
//     y[t,d] = x[t,d] / sqrt(sum_d(x^2) + eps)
//
//   x   : [num_rows, dim]  row-major
// Returns y : [num_rows, dim] row-major.
std::vector<float> L2NormRows(const std::vector<float>& x, int64_t num_rows,
                              int64_t dim, float eps = 1e-6f);

}  // namespace vllm::kimi_kda
