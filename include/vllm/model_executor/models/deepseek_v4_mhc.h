// DeepSeek-V4-Flash W5 — Manifold / Markov Hyper-Connections (MHC), as portable
// host (CPU) reference implementations. This is the residual-topology brick: V4
// replaces the plain `residual + RMSNorm` stream with an `[tokens, hc_mult, H]`
// MANIFOLD of hc_mult parallel residual streams, mixed at every attn/ffn
// boundary by a Sinkhorn-normalized (doubly-stochastic) mixing matrix, with the
// per-layer RMSNorms FOLDED into the mix kernels. W5 owns the four pieces of
// that topology, each ported 1:1 with `file:line` on BOTH sides:
//
//   (1) MhcSinkhorn      — the `hc_sinkhorn_iters` (=20) Sinkhorn normalization
//                          of the hc_mult×hc_mult mixing matrix: a row-softmax
//                          seed, then alternating row/column normalization toward
//                          a doubly-stochastic matrix.
//   (2) MhcPre           — the mHC "pre" block: flatten the hc_mult streams,
//                          project through `hc_*_fn` with a FOLDED weight-free
//                          RMSNorm, split the projection into pre / post / comb
//                          coefficients (sigmoid gates + the Sinkhorn matrix),
//                          collapse the streams by the pre-gate into the single
//                          `layer_input` fed to attn/ffn, and optionally FOLD the
//                          model's attn_norm/ffn_norm RMSNorm into that output.
//   (3) MhcPost          — the mHC "post" block: fold the block output `x` back
//                          into the hc_mult-stream residual manifold via the
//                          Sinkhorn comb matrix (mix) + the post gate (add).
//   (4) HcHeadCollapse   — the final `hc_head` collapse: fold the hc_mult streams
//                          back into ONE hidden vector (weight-free RMSNorm →
//                          `hc_head_fn` projection → sigmoid gate → weighted sum)
//                          before the model's final norm + lm_head.
//
// ─── EAGER-REFERENCE FINDING (corrects the W0 spike premise) ─────────────────
// The W0 scope and the model-matrix row assert MHC has "ZERO eager reference and
// no numerical test upstream" — that is the single hardest correctness item. In
// fact the pinned vLLM DOES ship an eager PyTorch reference for the mHC kernels:
// `vllm/model_executor/kernels/mhc/torch.py` (`mhc_pre_torch` / `mhc_post_torch`)
// and `triton.py` (`hc_head_reduce_triton_kernel`, the head collapse). These are
// the canonical numerics the TileLang / Triton / CUDA / AITER kernels all match.
// W5 ports THAT eager reference 1:1 as portable host code and, per the brief,
// ALSO derives an INDEPENDENT double-precision reference from the mathematical
// definition (row/col-normalization to a doubly-stochastic matrix) so the two
// agree — the eager ref pins the exact epsilon/axis conventions, the derived ref
// proves the port is not a transcription of a bug. Four upstream implementations
// agree byte-for-byte on the Sinkhorn (see the port table), so the numerics are
// unambiguous. (OPEN QUESTION: none of the resolved constants were guessed; the
// one detail we CANNOT gate at tiny shape is end-to-end bf16 residual rounding
// between steps — a W7 device concern, documented below, not folded into these
// f32/f64 references.)
//
// WHY host/CPU reference (honest scope, mirrors W3/W4): the full V4 forward is a
// multi-Spark campaign — the checkpoint is 156.7 GiB (does not fit one GB10, see
// deepseek_v4.h) and the forward also needs the sqrtsoftplus/hash MoE (W6) +
// device assembly (W7), none landed. So W5 lands + UNIT-GATES these primitives
// against hand-derived small cases with literal expected numbers verified from
// the vLLM source AND from-first-principles double-precision references on
// randomized shapes (tests/vllm/models/test_deepseek_v4_mhc.cpp), rather than a
// full-model dumped-oracle rel-L2 (the fixed-config 167B arch cannot be built at
// a tiny shape). The eventual GPU forward (W7) ports the SAME math into a CUDA
// kernel; this file pins the numerics portably so the kernel port has an oracle.
//
// SACRED-inert: additive TU only. It does NOT touch any existing forward (in
// particular the shared DeepSeek-V2 MLA path stays untouched — MHC is a V4-only
// topology). The device kernels + folding these into DeepseekV4Model::Forward
// are NAMED W7 residuals.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   OURS              <-  UPSTREAM (vllm/, @ 0.26.0.dev0)  [+ SGLang v0.5.15 cross-ref]
//   MhcSinkhorn       <-  model_executor/kernels/mhc/torch.py:75-82 (softmax(-1)+eps
//                         seed, col-norm, then (iters-1)× [row-norm, col-norm]);
//                         IDENTICAL in tilelang_kernels.py:126-153 (`_sinkhorn_fwd`)
//                         and SGLang python/sglang/srt/layers/mhc.py:110-126
//                         (`hc_split_sinkhorn_kernel`)
//   MhcPre            <-  torch.py:56-91 (`mhc_pre_torch`) — flatten, `x @ fn.T`,
//                         folded weight-free RMSNorm `rsqrt(sqrsum/(hc*H)+rms_eps)`,
//                         pre=sigmoid(·)+hc_pre_eps, post=sigmoid(·)*hc_post_mult,
//                         comb=Sinkhorn, layer_input=Σ_i pre[i]·residual[i]; the
//                         optional attn/ffn RMSNorm fold from
//                         tilelang.py mhc_pre_big_fuse_with_norm (norm_weight/eps);
//                         constants hc_post_alpha=2.0, hc_pre_eps=hc_sinkhorn_eps=
//                         hc_eps @ nvidia/model.py:818-821,:886-894
//   MhcPost           <-  torch.py:94-106 (`mhc_post_torch`) — einsum
//                         "ij,ih->jh" (comb mix) + post_layer_mix·x (post add)
//   HcHeadCollapse    <-  triton.py:108-140 (`hc_head_reduce_triton_kernel`) +
//                         tilelang.py hc_head_fused_kernel_tilelang:720-748;
//                         hc_head_fn [hc,hc*H], hc_head_base [hc], hc_head_scale
//                         scalar [1] @ nvidia/model.py:1023-1041,:1137-1145
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::deepseek_v4 {

// ── (1) Sinkhorn normalization of the hc_mult×hc_mult mixing matrix ───────────
//
// The comb mixing logits are normalized to a (near-)doubly-stochastic matrix by
// `iters` (=hc_sinkhorn_iters, 20) Sinkhorn steps (torch.py:75-82; identical in
// tilelang_kernels.py:126-153 `_sinkhorn_fwd`; SGLang mhc.py:110-126). For the
// input logit matrix L[j,k] (`comb_logits`, [hc,hc] row-major):
//
//   seed:              M[j,k] = softmax_k(L[j,:])[k] + eps      (row softmax, +eps)
//   col-norm (dim=-2): M[j,k] /= (Σ_j M[j,k] + eps)             (normalize each col k)
//   repeat iters-1×:
//     row-norm (dim=-1): M[j,k] /= (Σ_k M[j,k] + eps)           (normalize each row j)
//     col-norm (dim=-2): M[j,k] /= (Σ_j M[j,k] + eps)
//
// The `+eps` is added to the numerator ONLY in the softmax seed and to every
// normalization DENOMINATOR (exactly as all four upstream impls do). With eps=0
// this converges to a doubly-stochastic matrix (row sums = col sums = 1); with
// eps>0 it is approximately doubly-stochastic. The AXIS ALTERNATION and the
// ITERATION COUNT are load-bearing — perturbing either changes the fixed point
// (proven RED-first in the unit gate).
//
//   comb_logits : [hc*hc] row-major (L[j,k] at j*hc+k)
// Returns the normalized mixing matrix [hc*hc] row-major.
std::vector<float> MhcSinkhorn(const std::vector<float>& comb_logits, int64_t hc,
                               int64_t iters, float eps);

// ── (2) mHC "pre" block ──────────────────────────────────────────────────────
//
// Per token (torch.py:56-91). `residual` is the [hc, hidden] stream manifold,
// `fn` is [hc3, hc*hidden] (hc3 = (2+hc)*hc), `scale` is [3], `base` is [hc3]:
//
//   x           = residual flattened to [hc*hidden]
//   mixes[o]    = Σ_{i,h} residual[i,h] · fn[o, i*hidden+h]          (o in [0,hc3))
//   sqrsum      = Σ_{i,h} residual[i,h]^2
//   mixes[o]   *= rsqrt(sqrsum/(hc*hidden) + rms_eps)                (folded RMSNorm)
//   pre[j]      = sigmoid(mixes[j]        · scale[0] + base[j])        + hc_pre_eps
//   post[j]     = sigmoid(mixes[hc+j]     · scale[1] + base[hc+j]) · hc_post_mult_value
//   comb_logits[j*hc+k] = mixes[2hc + j*hc+k] · scale[2] + base[2hc + j*hc+k]
//   comb        = MhcSinkhorn(comb_logits, hc, sinkhorn_iters, hc_sinkhorn_eps)
//   layer_input[h] = Σ_j pre[j] · residual[j,h]                       ([hidden])
//   if norm_weight non-empty (the FOLDED attn_norm/ffn_norm, tilelang.py
//   mhc_pre_big_fuse_with_norm):
//     layer_input[h] *= rsqrt(mean_h(layer_input^2) + norm_eps) · norm_weight[h]
//
// In the model hc_pre_eps == hc_sinkhorn_eps == config.hc_eps and
// hc_post_mult_value == hc_post_alpha == 2.0 (nvidia/model.py:818-821); they are
// separate params here for gateability. `norm_weight` empty ⇒ no fold (matches
// the raw `mhc_pre_torch` output exactly). NOTE (W7 seam): the model rounds
// `residual` and `layer_input` to bf16 between steps; this reference stays in f32
// (the kernels compute the mix/Sinkhorn in fp32) and leaves bf16 storage rounding
// to the device brick, exactly like W3/W4 left their RoPE/fp8 seams.
struct MhcPreResult {
  std::vector<float> pre_mix;      // [hc]        sigmoid gate for the stream collapse
  std::vector<float> post_mix;     // [hc]        post gate (consumed by MhcPost)
  std::vector<float> comb_mix;     // [hc*hc]     Sinkhorn mixing matrix (→ MhcPost)
  std::vector<float> layer_input;  // [hidden]    the collapsed input fed to attn/ffn
};
MhcPreResult MhcPre(const std::vector<float>& residual, const std::vector<float>& fn,
                    const std::vector<float>& scale, const std::vector<float>& base,
                    int64_t hc, int64_t hidden, float rms_eps, float hc_pre_eps,
                    float hc_sinkhorn_eps, float hc_post_mult_value,
                    int64_t sinkhorn_iters, const std::vector<float>& norm_weight,
                    float norm_eps);

// ── (3) mHC "post" block ─────────────────────────────────────────────────────
//
// Per token (torch.py:94-106). Folds the block output `x` [hidden] back into the
// hc_mult-stream residual manifold:
//
//   mixed[j,h] = Σ_i comb_res_mix[i,j] · residual[i,h]   (einsum "ij,ih->jh")
//   post[j,h]  = post_layer_mix[j] · x[h]
//   new[j,h]   = mixed[j,h] + post[j,h]                  ([hc, hidden] row-major)
//
// `comb_res_mix` is the Sinkhorn matrix from the matching MhcPre ([hc*hc] row-
// major, indexed [i,j] at i*hc+j — the mix SUMS over the first index i).
std::vector<float> MhcPost(const std::vector<float>& x, const std::vector<float>& residual,
                           const std::vector<float>& post_layer_mix,
                           const std::vector<float>& comb_res_mix, int64_t hc,
                           int64_t hidden);

// ── (4) hc_head collapse ─────────────────────────────────────────────────────
//
// The final collapse of the hc_mult streams to one hidden vector
// (triton.py:108-140 `hc_head_reduce_triton_kernel`; tilelang.py:720-748). Per
// token, `x` is [hc, hidden], `fn` is hc_head_fn [hc, hc*hidden], `base` is
// hc_head_base [hc], `scale` is the SCALAR hc_head_scale ([1] upstream):
//
//   x_flat    = x flattened to [hc*hidden]
//   x_normed  = x_flat · rsqrt(mean(x_flat^2) + rms_eps)   (weight-free RMSNorm)
//   mixes[m]  = Σ_{i,h} x_normed[i*hidden+h] · fn[m, i*hidden+h]   (m in [0,hc))
//   pre[m]    = sigmoid(mixes[m] · scale + base[m]) + hc_eps
//   out[h]    = Σ_m pre[m] · x[m,h]                         ([hidden])
//
// The model applies its final RMSNorm(weight) to `out` afterward (a separate
// standard RMSNorm, not folded here).
std::vector<float> HcHeadCollapse(const std::vector<float>& x, const std::vector<float>& fn,
                                  float scale, const std::vector<float>& base, int64_t hc,
                                  int64_t hidden, float rms_eps, float hc_eps);

}  // namespace vllm::deepseek_v4
