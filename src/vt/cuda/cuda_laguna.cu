// Laguna-S-2.1 device-resident-decode glue kernels (OpId::kLaguna). The 5 small ops
// the NVFP4/Marlin decode still ran on the host — ported to CUDA so
// LagunaForwardResidentDecode keeps the activation on-GPU across all 48 layers and
// drains ONCE/token. GB10 unified memory: the caller passes raw std::vector data()
// pointers (device-accessible), no upload/download. BYTE-EXACT to the host reference:
// every reduction is SEQUENTIAL (single accumulator, matching host float order) — not
// the block-reduced near-tie DeepSeek uses. Registered on kCUDA via the OpProvider
// seam; laguna::LagunaDevice() resolves it. Ports (host ref -> kernel):
//   RmsNorm/RmsNormHeads (laguna.cpp:94/:112) -> RmsNormSeqKernel
//   ApplyRope (laguna.cpp:136)                -> RopeFromCacheKernel
//   LagunaAttention (laguna.cpp:701)          -> DecodeAttnGqaKernel
//   LagunaSoftplusHeadGate (laguna_ops.cpp:25)-> SoftplusHeadGateKernel
//   LagunaUngroupedRouterTopK (laguna_ops.cpp:41) -> SigmoidTopKKernel
#include <cuda_bf16.h>       // __nv_bfloat16 / __nv_bfloat162 / __bfloat1622float2
#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF_F

#include <cfloat>
#include <cstdint>

#include "vllm/model_executor/models/laguna_device.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vllm::laguna {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

cudaStream_t AsStream(Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// ── RMSNorm (block-per-row, block-reduced SoS; NEAR-TIE vs host RmsNorm:94/:112, in
// the accepted device regime — user-ratified 2026-08-01: gate the device path vs vLLM).
// One BLOCK per row: blockDim threads reduce Σ x[i]²; inv = 1/sqrtf(ss/n + eps).
__global__ void RmsNormSeqKernel(float* out, const float* x, const float* w, int64_t rows,
                                 int64_t n, float eps, bool has_w) {
  const int64_t r = static_cast<int64_t>(blockIdx.x);
  if (r >= rows) return;
  const float* xr = x + r * n;
  float* orow = out + r * n;
  float local = 0.0f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) local += xr[i] * xr[i];
  __shared__ float sh[256];
  sh[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(sh[0] / static_cast<float>(n) + eps);
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
    orow[i] = xr[i] * inv * (has_w ? w[i] : 1.0f);
}

// ── partial-NeoX RoPE from a half-split [rope_rows,rd] cache (bit-exact to ApplyRope) ──
// One thread per (head, i<rd/2): c=cache[pos*rd+i], s=cache[pos*rd+half+i].
__global__ void RopeFromCacheKernel(float* x, const float* cache, int64_t heads, int64_t Dh,
                                    int64_t rd, int64_t pos) {
  const int64_t half = rd / 2;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= heads * half) return;
  const int64_t h = t / half, i = t % half;
  const float* crow = cache + pos * rd;
  const float c = crow[i];
  const float s = crow[half + i];
  float* xv = x + h * Dh;
  const float x0 = xv[i];
  const float x1 = xv[half + i];
  xv[i] = x0 * c - x1 * s;
  xv[half + i] = x1 * c + x0 * s;
}

// ── Brick B: one-pass GQA-broadcast flash decode attention ──────────────────
// Dh is fixed at 128 for Laguna → 32 lanes × kLagEpl=4 head-dim elems/lane. Keys are
// staged into shared memory kLagTile at a time (each DRAM K/V row read ONCE) and the
// online-softmax O accumulator lives in registers (per lane), so this REPLACES the
// old 3-pass, block-per-Q-head, atomicAdd kernel (which re-read K 3× AND re-read the
// SAME KV once per Q-head in a group → 12-18× the minimum KV bytes).
constexpr int kLagDh = 128;              // Laguna head_dim (fixed; see laguna.h:90)
constexpr int kLagEpl = kLagDh / 32;     // 4 head-dim elems per lane (128 == 32*4)
constexpr int kLagTile = 32;             // keys staged per shared tile

// GQA T=1 decode attention (NEAR-TIE vs the host 3-pass softmax — the online rescale
// reorders the float adds, accepted in the device regime). ONE BLOCK per KV head g;
// block = QG*32 threads = QG warps, warp w owns query head h = g*QG + w (kvh=h/group=g,
// QG=group). Each key's K/V row is staged into shared ONCE by the whole block, then
// every warp attends it from shared (GQA broadcast). Register-resident online softmax
// (running m,l,o), warp-shuffle Q·K reduce, no atomicAdd. K/V laid out [kv_rows,Hkv,Dh];
// row j global pos = first_pos+j, q_pos=pos; causal + per-layer sliding window.
__global__ void DecodeAttnGqaKernel(float* o, const float* q, const float* k, const float* v,
                                    int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group,
                                    int64_t kv_rows, int64_t q_pos, int64_t first_pos,
                                    int64_t window, float scale) {
  const int64_t g = static_cast<int64_t>(blockIdx.x);  // KV head this block owns
  if (g >= Hkv) return;
  const int warp = static_cast<int>(threadIdx.x) >> 5;   // == q-head within the group
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int nth = static_cast<int>(blockDim.x);
  const int64_t QG = group;                              // q-heads per KV head (6 or 9)
  const int64_t h = g * QG + warp;                       // global q-head index
  const bool active = (warp < QG) && (h < Hq);           // (always true when block==QG*32)

  // This warp's query slice (this lane owns kLagEpl contiguous head dims), loaded once.
  float q_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i)
    q_reg[i] = active ? q[h * Dh + lane * kLagEpl + i] : 0.0f;
  float m = -CUDART_INF_F, l = 0.0f;
  float o_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i) o_reg[i] = 0.0f;

  __shared__ float ksh[kLagTile * kLagDh];  // 16 KiB
  __shared__ float vsh[kLagTile * kLagDh];  // 16 KiB

  for (int64_t base = 0; base < kv_rows; base += kLagTile) {
    const int64_t cnt = (kv_rows - base < kLagTile) ? (kv_rows - base) : kLagTile;
    const int64_t nload = cnt * Dh;  // elems per K (or V) block
    // Cooperative stage: [0,nload) → K rows, [nload,2*nload) → V rows (each read ONCE).
    for (int64_t idx = threadIdx.x; idx < 2 * nload; idx += nth) {
      const bool isv = idx >= nload;
      const int64_t e = isv ? (idx - nload) : idx;
      const int64_t row = e / Dh, col = e % Dh;
      const float* src = isv ? v : k;
      (isv ? vsh : ksh)[row * Dh + col] = src[((base + row) * Hkv + g) * Dh + col];
    }
    __syncthreads();
    if (active) {
      for (int64_t r = 0; r < cnt; ++r) {
        const int64_t pj = first_pos + base + r;
        if (pj > q_pos) continue;                          // causal (mask is lane-uniform)
        if (window > 0 && q_pos - pj >= window) continue;  // sliding window (0 => full causal)
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kLagEpl; ++i) dot += q_reg[i] * ksh[r * Dh + lane * kLagEpl + i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;    // full-head score to all lanes
        const float m_new = fmaxf(m, dot);
        const float corr = expf(m - m_new);                // 0 on the first key (m == -inf)
        const float pw = expf(dot - m_new);
#pragma unroll
        for (int i = 0; i < kLagEpl; ++i)
          o_reg[i] = o_reg[i] * corr + pw * vsh[r * Dh + lane * kLagEpl + i];
        l = l * corr + pw;
        m = m_new;
      }
    }
    __syncthreads();  // done reading shared before the next tile overwrites it
  }

  if (active) {
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;      // no visible key -> 0 (matches host)
#pragma unroll
    for (int i = 0; i < kLagEpl; ++i) o[h * Dh + lane * kLagEpl + i] = o_reg[i] * inv;
  }
}

// ── Brick A2+B: GRAPH one-pass GQA-broadcast flash decode (capturable) ──────
// Same one-pass flash kernel as DecodeAttnGqaKernel, but the two per-step-varying
// scalars come from DEVICE buffers so a captured CUDA graph reads them at REPLAY:
// len=*len_dev (prior cache rows), q_pos=*pos_dev. The current token's row is passed
// SEPARATELY (knew/vnew, [Hkv,Dh]) — NOT yet appended to the cache; the driver appends
// it between replays. Attends cache rows j in [0,len) (row j global pos first_pos+j)
// PLUS the new row (index len, global pos q_pos), so the key set == the eager kernel's
// cache[0..len+1) once appended ⇒ same math (near-tie float order). The LAUNCH shape
// (grid=Hkv, block=QG*32, static shared) is FIXED per layer; only the internal loop
// trip count varies via *len_dev at replay. first_pos/window/Hq/Hkv/Dh/group/scale are
// per-layer constants baked at capture.
__global__ void DecodeAttnGqaGKernel(float* o, const float* q, const float* k, const float* v,
                                     const float* knew, const float* vnew, int64_t Hq, int64_t Hkv,
                                     int64_t Dh, int64_t group, int64_t first_pos, int64_t window,
                                     float scale, const int* len_dev, const int* pos_dev) {
  const int64_t g = static_cast<int64_t>(blockIdx.x);  // KV head this block owns
  if (g >= Hkv) return;
  const int warp = static_cast<int>(threadIdx.x) >> 5;   // == q-head within the group
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int nth = static_cast<int>(blockDim.x);
  const int64_t QG = group;                              // q-heads per KV head (6 or 9)
  const int64_t h = g * QG + warp;                       // global q-head index
  const bool active = (warp < QG) && (h < Hq);
  const int64_t len = static_cast<int64_t>(*len_dev);    // prior cache rows
  const int64_t q_pos = static_cast<int64_t>(*pos_dev);  // this token's global position
  const int64_t total = len + 1;                         // cache rows + the new row

  float q_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i)
    q_reg[i] = active ? q[h * Dh + lane * kLagEpl + i] : 0.0f;
  float m = -CUDART_INF_F, l = 0.0f;
  float o_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i) o_reg[i] = 0.0f;

  __shared__ float ksh[kLagTile * kLagDh];  // 16 KiB
  __shared__ float vsh[kLagTile * kLagDh];  // 16 KiB

  for (int64_t base = 0; base < total; base += kLagTile) {
    const int64_t cnt = (total - base < kLagTile) ? (total - base) : kLagTile;
    const int64_t nload = cnt * Dh;
    // Cooperative stage: rows [0,len) from the k/v cache, row == len from knew/vnew.
    for (int64_t idx = threadIdx.x; idx < 2 * nload; idx += nth) {
      const bool isv = idx >= nload;
      const int64_t e = isv ? (idx - nload) : idx;
      const int64_t row = e / Dh, col = e % Dh;
      const int64_t gj = base + row;  // global key index in [0,total)
      float val;
      if (gj < len) {
        const float* src = isv ? v : k;
        val = src[(gj * Hkv + g) * Dh + col];
      } else {  // gj == len: the new (not-yet-appended) row, layout [Hkv,Dh]
        const float* src = isv ? vnew : knew;
        val = src[g * Dh + col];
      }
      (isv ? vsh : ksh)[row * Dh + col] = val;
    }
    __syncthreads();
    if (active) {
      for (int64_t r = 0; r < cnt; ++r) {
        const int64_t gj = base + r;
        const int64_t pj = (gj < len) ? (first_pos + gj) : q_pos;  // new row pos == q_pos
        if (pj > q_pos) continue;                          // causal
        if (window > 0 && q_pos - pj >= window) continue;  // sliding window
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kLagEpl; ++i) dot += q_reg[i] * ksh[r * Dh + lane * kLagEpl + i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;
        const float m_new = fmaxf(m, dot);
        const float corr = expf(m - m_new);
        const float pw = expf(dot - m_new);
#pragma unroll
        for (int i = 0; i < kLagEpl; ++i)
          o_reg[i] = o_reg[i] * corr + pw * vsh[r * Dh + lane * kLagEpl + i];
        l = l * corr + pw;
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (active) {
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
    for (int i = 0; i < kLagEpl; ++i) o[h * Dh + lane * kLagEpl + i] = o_reg[i] * inv;
  }
}

// ── per-head softplus OUT-gate (bit-exact to LagunaSoftplusHeadGate:25) ──
// softplus(x) = (x>20)? x : log1pf(expf(x)); attn[h,d] *= softplus(gate_logits[h]).
__global__ void SoftplusHeadGateKernel(float* attn, const float* gate_logits, int64_t Hq,
                                       int64_t Dh) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= Hq * Dh) return;
  const int64_t h = t / Dh;
  const float x = gate_logits[h];
  const float g = (x > 20.0f) ? x : log1pf(expf(x));
  attn[t] *= g;
}

// ── sigmoid-noaux top-k router (bit-exact selection to LagunaUngroupedRouterTopK:41) ──
// scores=sigmoid(logits); choice=scores+bias; pick top-k by (choice desc, idx asc);
// weights = UNBIASED scores[id], /wsum if renorm, *scale. Single block; E ≤ 1024.
__global__ void SigmoidTopKKernel(int32_t* ids, float* weights, const float* logits,
                                  const float* bias, bool has_bias, int64_t E, int64_t topk,
                                  bool renorm, float scale) {
  extern __shared__ float sh[];       // [E] scores | [E] choice | [E] selected(0/1)
  float* scores = sh;
  float* choice = sh + E;
  float* sel = sh + 2 * E;
  for (int64_t e = threadIdx.x; e < E; e += blockDim.x) {
    const float s = 1.0f / (1.0f + expf(-logits[e]));
    scores[e] = s;
    choice[e] = s + (has_bias ? bias[e] : 0.0f);
    sel[e] = 0.0f;
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  float wsum = 0.0f;
  for (int64_t j = 0; j < topk; ++j) {
    int64_t best = -1;
    float bestc = -CUDART_INF_F;
    for (int64_t e = 0; e < E; ++e) {          // ascending e + strict > ⇒ lower idx wins ties
      if (sel[e] != 0.0f) continue;
      if (choice[e] > bestc) { bestc = choice[e]; best = e; }
    }
    sel[best] = 1.0f;
    ids[j] = static_cast<int32_t>(best);
    const float w = scores[best];              // UNBIASED weight
    weights[j] = w;
    wsum += w;
  }
  for (int64_t j = 0; j < topk; ++j) {
    if (renorm && wsum > 0.0f) weights[j] /= wsum;
    weights[j] *= scale;
  }
}

// ── Laguna lm_head M=1 decode GEMV (coalesced, roofline-bound) ───────────────
// out[N] f32 = W[N,K] (bf16, row-major) · x[K] (f32), M=1. cuBLASLt's heuristic
// mis-routes this M=1×N=vocab×K=hidden GEMM to a BATCHED wmma tile algo (fills 1 of
// 16 tile rows → ~20% of roofline, the measured #1 Laguna decode GPU cost). This
// dedicated GEMV streams the ~616 MB weight ONCE at ~roofline. ONE BLOCK per output
// row n (grid = N): the block's 256 threads stride over K, each reading W[n*K+k]
// (bf16) — consecutive threads hit consecutive bf16 addresses ⇒ COALESCED. The K loop
// reads bf16 in __nv_bfloat162 PAIRS (4-byte loads; each row start n*K is 2-aligned
// for K even ⇒ a warp's 32 pairs = one 128 B transaction), multiply-accumulates in
// f32 against x (also read as float2 pairs), then a power-of-two block tree-reduce
// writes out[n]. NEAR-TIE vs the MatmulBT reference (block-reduced sum reorders the
// float adds; accepted device regime, gated vs vLLM). K odd falls back to a scalar
// tail (not exercised by Laguna: hidden=3072 is even).
__global__ void LmHeadGemvKernel(float* __restrict__ out, const __nv_bfloat16* __restrict__ w,
                                 const float* __restrict__ x, int64_t N, int64_t K) {
  const int64_t n = static_cast<int64_t>(blockIdx.x);
  if (n >= N) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int nth = static_cast<int>(blockDim.x);  // 256 (power of two; matches sh[256])
  const int64_t kpairs = K >> 1;                 // bf16 pairs (K even for Laguna)
  const __nv_bfloat162* __restrict__ w2 =
      reinterpret_cast<const __nv_bfloat162*>(w) + n * kpairs;  // this row, as pairs
  const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
  float acc = 0.0f;
  for (int64_t pdx = tid; pdx < kpairs; pdx += nth) {  // coalesced 4-byte bf162 stream
    const float2 wf = __bfloat1622float2(w2[pdx]);
    const float2 xv = x2[pdx];
    acc += wf.x * xv.x + wf.y * xv.y;
  }
  if ((K & 1) && tid == 0)  // odd-K scalar tail (not Laguna: hidden is even)
    acc += __bfloat162float(w[n * K + (K - 1)]) * x[K - 1];
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int s = nth >> 1; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[n] = sh[0];
}

// ── launchers (no sync — resident) ──
constexpr int kTPB = 128;
inline int Blocks(int64_t n) { return static_cast<int>((n + kTPB - 1) / kTPB); }

void RmsNormSeqLaunch(Queue& q, float* out, const float* x, const float* w, int64_t rows,
                      int64_t n, float eps, bool has_w) {
  // one block per row; 256 threads (matches the __shared__ sh[256] reduction).
  RmsNormSeqKernel<<<static_cast<unsigned>(rows), 256, 0, AsStream(q)>>>(out, x, w, rows, n, eps,
                                                                         has_w);
}
void RopeFromCacheLaunch(Queue& q, float* x, const float* cache, int64_t heads, int64_t Dh,
                         int64_t rd, int64_t pos) {
  RopeFromCacheKernel<<<Blocks(heads * (rd / 2)), kTPB, 0, AsStream(q)>>>(x, cache, heads, Dh, rd,
                                                                          pos);
}
void DecodeAttnGqaLaunch(Queue& q, float* o, const float* qd, const float* k, const float* v,
                         int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group, int64_t kv_rows,
                         int64_t q_pos, int64_t first_pos, int64_t window, float scale) {
  // ONE block per KV head (GQA broadcast); block = group*32 = QG warps, warp w owns
  // query head g*QG+w. Each K/V row staged into shared ONCE, reused across all QG heads.
  const unsigned grid = static_cast<unsigned>(Hkv);
  const unsigned blk = static_cast<unsigned>(group * 32);
  DecodeAttnGqaKernel<<<grid, blk, 0, AsStream(q)>>>(o, qd, k, v, Hq, Hkv, Dh, group, kv_rows,
                                                     q_pos, first_pos, window, scale);
}
void DecodeAttnGqaGLaunch(Queue& q, float* o, const float* qd, const float* k, const float* v,
                          const float* knew, const float* vnew, int64_t Hq, int64_t Hkv, int64_t Dh,
                          int64_t group, int64_t first_pos, int64_t window, float scale,
                          const int* len_dev, const int* pos_dev) {
  // ONE block per KV head; block = group*32 (QG warps). FIXED launch shape: grid=Hkv,
  // block=QG*32, STATIC shared (ksh/vsh) — never varies per step, so it is capturable.
  // Only the kernel's internal tile-loop trip count varies via *len_dev at replay.
  const unsigned grid = static_cast<unsigned>(Hkv);
  const unsigned blk = static_cast<unsigned>(group * 32);
  DecodeAttnGqaGKernel<<<grid, blk, 0, AsStream(q)>>>(o, qd, k, v, knew, vnew, Hq, Hkv, Dh, group,
                                                      first_pos, window, scale, len_dev, pos_dev);
}
void SoftplusHeadGateLaunch(Queue& q, float* attn, const float* gl, int64_t Hq, int64_t Dh) {
  SoftplusHeadGateKernel<<<Blocks(Hq * Dh), kTPB, 0, AsStream(q)>>>(attn, gl, Hq, Dh);
}
void SigmoidTopKLaunch(Queue& q, int32_t* ids, float* weights, const float* logits,
                       const float* bias, bool has_bias, int64_t E, int64_t topk, bool renorm,
                       float scale) {
  const int threads = E < 256 ? static_cast<int>(E) : 256;
  const size_t shmem = static_cast<size_t>(3 * E) * sizeof(float);
  SigmoidTopKKernel<<<1, threads, shmem, AsStream(q)>>>(ids, weights, logits, bias, has_bias, E,
                                                        topk, renorm, scale);
}
void LmHeadGemvLaunch(Queue& q, float* out, const void* w_bf16, const float* x, int64_t N,
                      int64_t K) {
  // ONE block per output row n (grid = N); 256 threads (power of two; matches the
  // sh[256] tree-reduce). Fixed grid + fixed pointers ⇒ CUDA-graph capturable.
  LmHeadGemvKernel<<<static_cast<unsigned>(N), 256, 0, AsStream(q)>>>(
      out, reinterpret_cast<const __nv_bfloat16*>(w_bf16), x, N, K);
}

const LagunaDeviceKernels kLaguna = {&RmsNormSeqLaunch,    &RopeFromCacheLaunch,
                                     &DecodeAttnGqaLaunch, &SoftplusHeadGateLaunch,
                                     &SigmoidTopKLaunch,   &DecodeAttnGqaGLaunch,
                                     &LmHeadGemvLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLaguna, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kLaguna)));
  }
} registrar;

}  // namespace
}  // namespace vllm::laguna
