// Laguna-S-2.1 device-resident decode — the 5 small glue kernels the NVFP4/Marlin
// arm still runs on the host (RMSNorm, dual-RoPE, GQA decode attention, per-head
// softplus out-gate, sigmoid-noaux top-k), ported to device kernels so
// LagunaForwardResidentDecode keeps the activation on-GPU across all 48 layers and
// drains ONCE per token (kills the measured ~432 cudaStreamSynchronize + ~188
// cudaMemcpyAsync/token — see .agents/specs/laguna-device-resident-decode.md).
//
// BYTE-EXACT by construction: every reduction (RMSNorm sum-of-squares, attention
// softmax) is SEQUENTIAL (single accumulator per row, T=1), so it matches the host
// reference bit-for-bit — NOT the block-reduced near-tie DeepSeek uses. The GEMMs
// (vt::MatmulBT bf16) + Marlin MoE are the SAME kernels as the current Marlin-default
// golden, so they are byte-identical there already. Ports:
//   softplus_head_gate  <- LagunaSoftplusHeadGate  (laguna_ops.cpp:25)
//   rope_from_cache      <- ApplyRope              (laguna.cpp:136)
//   rms_norm_seq         <- RmsNorm / RmsNormHeads (laguna.cpp:94 / :112)
//   decode_attn_gqa      <- LagunaAttention        (laguna.cpp:701)
//   sigmoid_topk         <- LagunaUngroupedRouterTopK (laguna_ops.cpp:41)
// The kernel table is registered through the vt OpProvider seam under OpId::kLaguna
// on kCUDA (mirrors deepseek_v4_device.h). On a CPU-only build nothing registers
// for (kLaguna,kCUDA), so GetOp throws and LagunaDeviceKernelsAvailable() is false
// (the resident path stays gated off, host compose runs).
#pragma once

#include <cstdint>

#include "vt/device.h"  // vt::Queue

namespace vllm::laguna {

// The five device launchers. All take/return DEVICE pointers into the queue's
// unified memory; NONE syncs (the resident driver drains once at the step boundary).
struct LagunaDeviceKernels {
  // out[rows,n] = rmsnorm(x[rows,n]) [* w[n] when has_w]; SEQUENTIAL sum-of-squares
  // per row (bit-exact to RmsNorm:94 / RmsNormHeads:112). rows=1,n=H for the block
  // norms; rows=Hq/Hkv,n=Dh for the per-head QK-norm. in/out may alias.
  void (*rms_norm_seq)(vt::Queue&, float* out, const float* x, const float* w, int64_t rows,
                       int64_t n, float eps, bool has_w);
  // In-place partial-NeoX RoPE from a precomputed half-split [rope_rows,rd] cos/sin
  // cache: for each (head,i<rd/2) x[i]=x0*c-x1*s, x[half+i]=x1*c+x0*s, c=cache[pos*rd+i],
  // s=cache[pos*rd+half+i]; dims [rd,Dh) untouched. Bit-exact to ApplyRope:136.
  void (*rope_from_cache)(vt::Queue&, float* x, const float* cache, int64_t heads, int64_t Dh,
                          int64_t rd, int64_t pos);
  // GQA T=1 decode attention: o[Hq,Dh] over K/V[kv_rows,Hkv,Dh], head h reads KV head
  // h/group; SEQUENTIAL host-order 3-pass softmax (bit-exact to LagunaAttention:701).
  // Physical cache row r has GLOBAL position kv_pos=first_pos+r (matches the host's
  // sliding-window eviction bookkeeping); causal (skip kv_pos>q_pos) + per-layer window
  // (skip if window>0 and q_pos-kv_pos>=window); NO attn-sink. scale=1/sqrt(Dh).
  void (*decode_attn_gqa)(vt::Queue&, float* o, const float* q, const float* k, const float* v,
                          int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group, int64_t kv_rows,
                          int64_t q_pos, int64_t first_pos, int64_t window, float scale);
  // Per-head softplus OUT-gate in place: attn[h,d] *= softplus(gate_logits[h]),
  // softplus(x)=(x>20)?x:log1p(exp(x)) in f32. Bit-exact to LagunaSoftplusHeadGate:25.
  void (*softplus_head_gate)(vt::Queue&, float* attn, const float* gate_logits, int64_t Hq,
                             int64_t Dh);
  // Sigmoid-noaux top-k router (single block over E): scores=sigmoid(logits),
  // choice=scores+bias, select top-k by (choice desc, index asc); ids[topk] +
  // weights[topk] = UNBIASED scores[id], /wsum if renorm, *scale. Bit-exact selection
  // (integer tie-break) to LagunaUngroupedRouterTopK:41.
  void (*sigmoid_topk)(vt::Queue&, int32_t* ids, float* weights, const float* logits,
                       const float* bias, bool has_bias, int64_t E, int64_t topk, bool renorm,
                       float scale);
  // Brick A2 GRAPH variant of decode_attn_gqa (capturable). Identical math but the
  // two per-step-varying scalars come from DEVICE buffers so a captured CUDA graph
  // reads them at REPLAY: q_pos=*pos_dev, kv_rows=*len_dev (fixed pointers, contents
  // refreshed by the host outside capture). The current token's row is passed
  // SEPARATELY (knew/vnew, [Hkv,Dh]) rather than pre-appended to the cache — the
  // graph appends it BETWEEN replays at a host-known slot (a varying offset must
  // never be a captured arg). Attends cache rows j in [0,len) (global pos
  // first_pos+j) PLUS the new row (index len, global pos q_pos). first_pos / window /
  // Hq / Hkv / Dh / group / scale are per-layer constants baked at capture. The key
  // set == decode_attn_gqa's cache[0..rows) AFTER the between-replay append, so the
  // replayed output is bit-identical to the eager decode_attn_gqa. NO attn-sink.
  void (*decode_attn_gqa_g)(vt::Queue&, float* o, const float* q, const float* k, const float* v,
                            const float* knew, const float* vnew, int64_t Hq, int64_t Hkv,
                            int64_t Dh, int64_t group, int64_t first_pos, int64_t window,
                            float scale, const int* len_dev, const int* pos_dev);
};

// Resolver (throws on a CPU-only build where nothing registered for kLaguna,kCUDA).
const LagunaDeviceKernels* LagunaDevice();
// True iff the CUDA laguna kernel table is registered (guards the resident decode).
bool LagunaDeviceKernelsAvailable();

}  // namespace vllm::laguna
