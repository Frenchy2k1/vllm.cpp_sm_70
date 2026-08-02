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
#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/laguna_device.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vllm::laguna {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

cudaStream_t AsStream(Queue& q) { return static_cast<cudaStream_t>(q.handle); }

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("vt cuda laguna: ") + what + ": " +
                             cudaGetErrorString(e));
}

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

// ── Brick A2b GRAPH RoPE (capturable): row index from a DEVICE buffer ────────
// Identical to RopeFromCacheKernel except pos = *pos_dev, read at REPLAY, so ONE
// position-indexed cos/sin table (built once, rows [0,max_cap)) serves every replay —
// no per-step host cos/sin rebuild. Row *pos_dev == the old single-row build for pos.
__global__ void RopeFromCacheGKernel(float* x, const float* cache, int64_t heads, int64_t Dh,
                                     int64_t rd, const int* pos_dev) {
  const int64_t half = rd / 2;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= heads * half) return;
  const int64_t h = t / half, i = t % half;
  const float* crow = cache + static_cast<int64_t>(*pos_dev) * rd;
  const float c = crow[i];
  const float s = crow[half + i];
  float* xv = x + h * Dh;
  const float x0 = xv[i];
  const float x1 = xv[half + i];
  xv[i] = x0 * c - x1 * s;
  xv[half + i] = x1 * c + x0 * s;
}

// ── Brick A2b GRAPH KV-append (capturable): write the new token's K/V into the cache ──
// cache_k/cache_v[*len_dev * kvdim + i] = knew/vnew[i], for i in [0,kvdim). One thread per
// element. *len_dev is read at REPLAY (host refreshes outside capture); the slot varies
// across replays but the POINTERS are fixed ⇒ capture-safe. Called AFTER the graph decode
// attention consumed knew/vnew, so appending row *len_dev creates no intra-replay hazard.
__global__ void AppendKvRowKernel(float* cache_k, float* cache_v, const float* knew,
                                  const float* vnew, int64_t kvdim, const int* len_dev) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= kvdim) return;
  const int64_t off = static_cast<int64_t>(*len_dev) * kvdim;
  cache_k[off + i] = knew[i];
  cache_v[off + i] = vnew[i];
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
                                    int64_t window, float scale, const float* gate) {
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
    // L1 (VT_LAGUNA_GLUE_FUSED): fold the per-head softplus out-gate into the store.
    // BYTE-EXACT vs the separate SoftplusHeadGateKernel: it stored o=o_reg*inv (f32,
    // lossless) then reloaded+×g; folded computes the SAME (o_reg*inv)*g in registers.
    // gate==nullptr keeps the un-gated store bit-for-bit (no ×1.0 rounding risk).
    if (gate != nullptr) {
      const float gx = gate[h];
      const float gv = (gx > 20.0f) ? gx : log1pf(expf(gx));
#pragma unroll
      for (int i = 0; i < kLagEpl; ++i) o[h * Dh + lane * kLagEpl + i] = o_reg[i] * inv * gv;
    } else {
#pragma unroll
      for (int i = 0; i < kLagEpl; ++i) o[h * Dh + lane * kLagEpl + i] = o_reg[i] * inv;
    }
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
                                     float scale, const int* len_dev, const int* pos_dev,
                                     const float* gate) {
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
    if (gate != nullptr) {  // L1: fold softplus out-gate (byte-exact; see DecodeAttnGqaKernel)
      const float gx = gate[h];
      const float gv = (gx > 20.0f) ? gx : log1pf(expf(gx));
#pragma unroll
      for (int i = 0; i < kLagEpl; ++i) o[h * Dh + lane * kLagEpl + i] = o_reg[i] * inv * gv;
    } else {
#pragma unroll
      for (int i = 0; i < kLagEpl; ++i) o[h * Dh + lane * kLagEpl + i] = o_reg[i] * inv;
    }
  }
}

// ── Split-K decode attention (occupancy fill for GB10's many SMs) ───────────
// grid=Hkv (block-per-KV-head) launches only Hkv(=8) blocks → severe under-occupancy on
// GB10 for this memory-bound decode attention (which grows with context). SPLIT the
// KV-row range across SPLIT blocks per KV head: grid=dim3(Hkv,SPLIT); block (g,s) owns KV
// head g and KV rows [s*rps, min((s+1)*rps,rows)) (rps=ceil(rows/SPLIT)). Each block runs
// the SAME masked online-softmax over its slice and writes a PARTIAL — running max m_s,
// denom l_s, UN-normalized numerator o_s[Dh] (NOT divided by l_s) — to scratch[h,s]. A
// combine kernel then merges the SPLIT partials per q-head with the flash split-KV
// reduction (mirrors the cross-warp flash merge in cuda_paged_attn.cu:373/:386 — global
// max, exp-rescale each partial by exp(m_s-gm)):
//   gm=max_s m_s ; denom=Σ_s l_s·exp(m_s-gm) ; acc=Σ_s exp(m_s-gm)·o_s ; out=acc/denom.
// NEAR-TIE vs the single-block kernel (the sub-range running-max + merge reorders the
// float adds — accepted device regime, gated vs vLLM). SPLIT==1 collapses exactly:
// gm=m_0, acc=o_0, denom=l_0, out=o_0/l_0 == the single-block kernel's o_reg/l — so the
// launcher runs the original single-block kernel for SPLIT==1 (byte-exact tiny-kv fallback).

// Partial online-softmax over KV rows [row_begin,row_end) of KV head g → scratch[h,s].
__global__ void DecodeAttnGqaSplitKernel(float* mp, float* lp, float* op, const float* q,
                                         const float* k, const float* v, int64_t Hq, int64_t Hkv,
                                         int64_t Dh, int64_t group, int64_t kv_rows, int64_t q_pos,
                                         int64_t first_pos, int64_t window, float scale, int SPLIT) {
  const int64_t g = static_cast<int64_t>(blockIdx.x);  // KV head this block owns
  if (g >= Hkv) return;
  const int sp = static_cast<int>(blockIdx.y);         // split index within head g
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int nth = static_cast<int>(blockDim.x);
  const int64_t QG = group;
  const int64_t h = g * QG + warp;
  const bool active = (warp < QG) && (h < Hq);
  const int64_t rps = (kv_rows + SPLIT - 1) / SPLIT;   // rows per split (ceil)
  const int64_t row_begin = static_cast<int64_t>(sp) * rps;
  const int64_t row_end = (row_begin + rps < kv_rows) ? (row_begin + rps) : kv_rows;

  float q_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i)
    q_reg[i] = active ? q[h * Dh + lane * kLagEpl + i] : 0.0f;
  float m = -CUDART_INF_F, l = 0.0f;
  float o_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i) o_reg[i] = 0.0f;

  __shared__ float ksh[kLagTile * kLagDh];
  __shared__ float vsh[kLagTile * kLagDh];

  for (int64_t base = row_begin; base < row_end; base += kLagTile) {
    const int64_t cnt = (row_end - base < kLagTile) ? (row_end - base) : kLagTile;
    const int64_t nload = cnt * Dh;
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

  if (active) {  // UN-normalized partial (empty slice ⇒ m=-inf, l=0, o=0 — combine ignores)
#pragma unroll
    for (int i = 0; i < kLagEpl; ++i)
      op[(h * SPLIT + sp) * Dh + lane * kLagEpl + i] = o_reg[i];
    if (lane == 0) {
      mp[h * SPLIT + sp] = m;
      lp[h * SPLIT + sp] = l;
    }
  }
}

// GRAPH split-K partial (capturable): same as DecodeAttnGqaSplitKernel, but len/q_pos come
// from DEVICE buffers at replay and the new (not-yet-appended) row is knew/vnew (index len).
// total=len+1 rows; slice [sp*rps, min((sp+1)*rps,total)) with rps=ceil(total/SPLIT). SPLIT
// is FIXED at capture (grid baked); rps is computed IN-KERNEL from *len_dev so the SAME
// grid covers a growing context across replays. Empty splits write (-inf,0,0).
__global__ void DecodeAttnGqaSplitGKernel(float* mp, float* lp, float* op, const float* q,
                                          const float* k, const float* v, const float* knew,
                                          const float* vnew, int64_t Hq, int64_t Hkv, int64_t Dh,
                                          int64_t group, int64_t first_pos, int64_t window,
                                          float scale, const int* len_dev, const int* pos_dev,
                                          int SPLIT) {
  const int64_t g = static_cast<int64_t>(blockIdx.x);
  if (g >= Hkv) return;
  const int sp = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int nth = static_cast<int>(blockDim.x);
  const int64_t QG = group;
  const int64_t h = g * QG + warp;
  const bool active = (warp < QG) && (h < Hq);
  const int64_t len = static_cast<int64_t>(*len_dev);
  const int64_t q_pos = static_cast<int64_t>(*pos_dev);
  const int64_t total = len + 1;
  const int64_t rps = (total + SPLIT - 1) / SPLIT;
  const int64_t row_begin = static_cast<int64_t>(sp) * rps;
  const int64_t row_end = (row_begin + rps < total) ? (row_begin + rps) : total;

  float q_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i)
    q_reg[i] = active ? q[h * Dh + lane * kLagEpl + i] : 0.0f;
  float m = -CUDART_INF_F, l = 0.0f;
  float o_reg[kLagEpl];
#pragma unroll
  for (int i = 0; i < kLagEpl; ++i) o_reg[i] = 0.0f;

  __shared__ float ksh[kLagTile * kLagDh];
  __shared__ float vsh[kLagTile * kLagDh];

  for (int64_t base = row_begin; base < row_end; base += kLagTile) {
    const int64_t cnt = (row_end - base < kLagTile) ? (row_end - base) : kLagTile;
    const int64_t nload = cnt * Dh;
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
        if (pj > q_pos) continue;
        if (window > 0 && q_pos - pj >= window) continue;
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
#pragma unroll
    for (int i = 0; i < kLagEpl; ++i)
      op[(h * SPLIT + sp) * Dh + lane * kLagEpl + i] = o_reg[i];
    if (lane == 0) {
      mp[h * SPLIT + sp] = m;
      lp[h * SPLIT + sp] = l;
    }
  }
}

// Merge the SPLIT partials per q-head (flash split-KV reduction; mirrors the cross-warp
// merge in cuda_paged_attn.cu:386). ONE block per q-head h, block = kLagDh threads (one
// per head dim). gm=max_sp m_sp; sc_sp=exp(m_sp-gm) (0 for empty splits where m_sp==-inf);
// denom=Σ l_sp·sc_sp; out[h,d]=(Σ_sp sc_sp·o_sp[d])/denom. gm==-inf (no visible key in ANY
// split — impossible here since the diagonal is always visible) ⇒ write 0. Shared holds
// the SPLIT scalars: [SPLIT] m | [SPLIT] l | [SPLIT] sc.
__global__ void DecodeAttnCombineKernel(float* o, const float* mp, const float* lp,
                                        const float* op, int64_t Hq, int64_t Dh, int SPLIT,
                                        const float* gate) {
  const int64_t h = static_cast<int64_t>(blockIdx.x);
  if (h >= Hq) return;
  // L1 (VT_LAGUNA_GLUE_FUSED): fold the per-head softplus out-gate into the combine
  // epilogue. gate[h] is head-uniform, so softplus is computed ONCE per block; the
  // final store becomes (acc*inv)*gv — byte-exact vs the separate SoftplusHeadGate
  // pass (which stored acc*inv f32 then reloaded ×gv). gate==nullptr => un-gated store.
  float gv = 1.0f;
  const bool has_gate = (gate != nullptr);
  if (has_gate) {
    const float gx = gate[h];
    gv = (gx > 20.0f) ? gx : log1pf(expf(gx));
  }
  extern __shared__ float csh[];  // [SPLIT] m | [SPLIT] l | [SPLIT] sc
  float* m_sp = csh;
  float* l_sp = csh + SPLIT;
  float* sc_sp = csh + 2 * SPLIT;
  for (int t = static_cast<int>(threadIdx.x); t < SPLIT; t += static_cast<int>(blockDim.x)) {
    m_sp[t] = mp[h * SPLIT + t];
    l_sp[t] = lp[h * SPLIT + t];
  }
  __syncthreads();
  float gm = -CUDART_INF_F;
  for (int t = 0; t < SPLIT; ++t) gm = fmaxf(gm, m_sp[t]);
  for (int t = static_cast<int>(threadIdx.x); t < SPLIT; t += static_cast<int>(blockDim.x))
    sc_sp[t] = (m_sp[t] == -CUDART_INF_F || gm == -CUDART_INF_F) ? 0.0f : expf(m_sp[t] - gm);
  __syncthreads();
  float denom = 0.0f;
  for (int t = 0; t < SPLIT; ++t) denom += l_sp[t] * sc_sp[t];
  const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < Dh;
       d += static_cast<int64_t>(blockDim.x)) {
    float acc = 0.0f;
    for (int t = 0; t < SPLIT; ++t) acc += sc_sp[t] * op[(h * SPLIT + t) * Dh + d];
    o[h * Dh + d] = has_gate ? (acc * inv) * gv : (acc * inv);
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
  // Parallel top-K selection (was a thread-0 serial O(topk*E) scan = ~77us/call,
  // 5.3% of decode GPU time). Each round is a BLOCK-WIDE argmax over the UNSELECTED
  // experts with the lower-index-wins tie-break (mirrors "ascending e + strict >").
  // BYTE-IDENTICAL to the serial version (same picks, same weights, same order).
  const int t = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);  // power-of-two (launcher pads to 256)
  __shared__ float rval[256];
  __shared__ int ridx[256];
  for (int64_t j = 0; j < topk; ++j) {
    float bc = -CUDART_INF_F;
    int bi = static_cast<int>(E);  // sentinel "none": loses every tie (val & idx)
    for (int64_t e = t; e < E; e += nt) {
      if (sel[e] != 0.0f) continue;
      const float c = choice[e];
      if (c > bc || (c == bc && static_cast<int>(e) < bi)) { bc = c; bi = static_cast<int>(e); }
    }
    rval[t] = bc;
    ridx[t] = bi;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
      if (t < s) {
        const float ov = rval[t + s];
        const int oi = ridx[t + s];
        if (ov > rval[t] || (ov == rval[t] && oi < ridx[t])) {
          rval[t] = ov;
          ridx[t] = oi;
        }
      }
      __syncthreads();
    }
    const int best = ridx[0];
    if (t == 0) {
      sel[best] = 1.0f;
      ids[j] = best;
      weights[j] = scores[best];  // UNBIASED weight
    }
    __syncthreads();  // sel[best] visible to all threads before the next round
  }
  if (t == 0) {  // renorm+scale over the topk selected (topk small; thread 0)
    float wsum = 0.0f;
    for (int64_t j = 0; j < topk; ++j) wsum += weights[j];
    for (int64_t j = 0; j < topk; ++j) {
      if (renorm && wsum > 0.0f) weights[j] /= wsum;
      weights[j] *= scale;
    }
  }
}

// ── Laguna lm_head M=1 decode GEMV (coalesced, roofline-bound) ───────────────
// out[N] f32 = W[N,K] (bf16, row-major) · x[K] (f32), M=1. cuBLASLt's heuristic
// mis-routes this M=1×N=vocab×K=hidden GEMM to a BATCHED wmma tile algo (fills 1 of
// 16 tile rows → ~20% of roofline, the measured #1 Laguna decode GPU cost). This
// dedicated GEMV streams the ~616 MB weight ONCE at ~roofline. ONE WARP (32 lanes) per
// output row n: block = (blockDim/32) warps, so warp w owns row
// n = blockIdx.x*(blockDim/32) + (threadIdx.x>>5). warp_id is warp-UNIFORM ⇒ the n<N
// guard never splits a warp. Each lane strides over K reading W as __nv_bfloat162 PAIRS
// (lane l reads pair l, l+32, … — 32 consecutive 4-byte pairs = one 128 B COALESCED
// warp transaction; each row start n*K is 2-aligned for K even), multiply-accumulates
// in f32 against x (read as float2 pairs). The 32 partial sums reduce via
// __shfl_down_sync (full-mask warp shuffle — every lane of the owning warp stays active,
// the K-loop trip count is lane-uniform for even K — NO __shared__, NO __syncthreads);
// lane 0 writes out[n]. This replaces the old ONE-BLOCK-per-row sh[256] tree-reduce (8
// __syncthreads/row): warp-per-row removes all block sync + raises occupancy toward the
// GB10 BW roofline. NEAR-TIE vs the MatmulBT reference (warp-reduced sum reorders the
// float adds; accepted device regime, gated vs vLLM). K odd falls back to a scalar tail
// on lane 0 (not exercised by Laguna: hidden=3072 is even).
__global__ void LmHeadGemvKernel(float* __restrict__ out, const __nv_bfloat16* __restrict__ w,
                                 const float* __restrict__ x, int64_t N, int64_t K) {
  const int warp_id = static_cast<int>(threadIdx.x) >> 5;  // warp within block
  const int lane = static_cast<int>(threadIdx.x) & 31;     // lane within warp
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;  // 8 (256 threads/block)
  const int64_t n = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;  // this row
  if (n >= N) return;  // warp-UNIFORM (whole warp owns row n) ⇒ never splits a warp
  const int64_t kpairs = K >> 1;  // bf16 pairs (K even for Laguna)
  const __nv_bfloat162* __restrict__ w2 =
      reinterpret_cast<const __nv_bfloat162*>(w) + n * kpairs;  // this row, as pairs
  const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
  float acc = 0.0f;
  for (int64_t pdx = lane; pdx < kpairs; pdx += 32) {  // coalesced 128 B/warp bf162 stream
    const float2 wf = __bfloat1622float2(w2[pdx]);
    const float2 xv = x2[pdx];
    acc += wf.x * xv.x + wf.y * xv.y;
  }
  for (int off = 16; off > 0; off >>= 1)  // warp-shuffle tree-reduce (no __syncthreads)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) {
    if (K & 1)  // odd-K scalar tail (not Laguna: hidden is even)
      acc += __bfloat162float(w[n * K + (K - 1)]) * x[K - 1];
    out[n] = acc;
  }
}

// ── Laguna decode embed-gather (VT_LAGUNA_ONDEV_SAMPLE) ──────────────────────
// out[H] f32 = embed_table[*tok][0..H) — one output element per thread, the token id
// read from a DEVICE buffer (tok[0]) so it is capture-safe INSIDE the decode graph
// (fixed pointers, grid=ceil(H/TPB)). The bf16 table widens EXACTLY as the host
// LagunaGraph::Step embed loop does (bits<<16 zero-pad ⇒ byte-identical f32); an f32
// table is a plain copy. Replaces the between-replay HOST embed gather (+ the full-
// vocab host argmax) so the graph replay for step N+1 launches with no host work on
// step N's logits (the ~527 us GPU-idle step gap).
__global__ void EmbedGatherKernel(float* __restrict__ out, const void* __restrict__ table,
                                  int is_bf16, const int64_t* __restrict__ tok, int64_t H) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= H) return;
  const int64_t id = tok[0];
  if (is_bf16) {
    const uint16_t* __restrict__ t = reinterpret_cast<const uint16_t*>(table) + id * H;
    out[i] = __uint_as_float(static_cast<uint32_t>(t[i]) << 16);  // bf16→f32 (exact)
  } else {
    out[i] = reinterpret_cast<const float*>(table)[id * H + i];
  }
}

// ── launchers (no sync — resident) ──
constexpr int kTPB = 128;
inline int Blocks(int64_t n) { return static_cast<int>((n + kTPB - 1) / kTPB); }

// Split-K occupancy tuning. grid=Hkv(=8) starves GB10's many SMs; split each KV head's
// KV rows across SPLIT blocks to reach ~kLagSplitTargetBlocks total. The EAGER SPLIT adapts
// to kv_rows (more splits as context grows); the GRAPH uses a FIXED SPLIT (grid baked at
// capture, rps computed in-kernel from *len_dev so one capture covers a growing context).
constexpr int kLagSplitTargetBlocks = 256;     // aim to fill the machine (a few hundred blocks)
constexpr int kLagSplitMax = 32;               // cap: bounds partials scratch + combine cost
constexpr int kLagMinRowsPerSplit = kLagTile;  // never split below one staged tile (32 rows)
constexpr int kLagSplitGraph = 16;             // FIXED graph SPLIT (Hkv*16 blocks; capture-safe)

// EAGER split factor: enough blocks to fill the GPU, but never below one tile of rows per
// split and never above kLagSplitMax. Returns 1 (single-block byte-exact path) for tiny kv.
inline int ChooseSplitEager(int64_t kv_rows, int64_t Hkv) {
  if (kv_rows <= kLagMinRowsPerSplit || Hkv <= 0) return 1;
  const int by_rows = static_cast<int>((kv_rows + kLagMinRowsPerSplit - 1) / kLagMinRowsPerSplit);
  const int by_fill = static_cast<int>((kLagSplitTargetBlocks + Hkv - 1) / Hkv);
  int sp = (by_rows < by_fill) ? by_rows : by_fill;
  if (sp > kLagSplitMax) sp = kLagSplitMax;
  return (sp < 1) ? 1 : sp;
}

// Persistent grow-only device scratch for the split partials (o | m | l per q-head per
// split). Grown ONLY outside a CUDA-graph capture: the graph's gstate-0 eager warm-run
// (laguna.cpp LagunaGraph::Step) runs the SAME launcher sequence and sizes this to the max
// (Hq_max*SPLIT*(Dh+2)) BEFORE the gstate-1 capture, so the captured combine's baked
// scratch pointer never reallocs → it stays valid across every replay. cudaMalloc is thus
// never invoked inside capture. Single decode stream (no concurrency guard, matching the
// rest of the resident path).
float* EnsureSplitScratch(size_t floats) {
  static float* buf = nullptr;
  static size_t cap = 0;
  if (floats > cap) {
    if (buf != nullptr) cudaFree(buf);
    Check(cudaMalloc(reinterpret_cast<void**>(&buf), floats * sizeof(float)),
          "split-K partials scratch alloc");
    cap = floats;
  }
  return buf;
}

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
void RopeFromCacheGLaunch(Queue& q, float* x, const float* cache, int64_t heads, int64_t Dh,
                          int64_t rd, const int* pos_dev) {
  RopeFromCacheGKernel<<<Blocks(heads * (rd / 2)), kTPB, 0, AsStream(q)>>>(x, cache, heads, Dh, rd,
                                                                           pos_dev);
}
void AppendKvRowLaunch(Queue& q, float* cache_k, float* cache_v, const float* knew,
                       const float* vnew, int64_t kvdim, const int* len_dev) {
  AppendKvRowKernel<<<Blocks(kvdim), kTPB, 0, AsStream(q)>>>(cache_k, cache_v, knew, vnew, kvdim,
                                                             len_dev);
}
void DecodeAttnGqaLaunch(Queue& q, float* o, const float* qd, const float* k, const float* v,
                         int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group, int64_t kv_rows,
                         int64_t q_pos, int64_t first_pos, int64_t window, float scale,
                         const float* gate) {
  // block = group*32 = QG warps (warp w owns query head g*QG+w); each K/V row staged into
  // shared ONCE, reused across all QG heads. Split the KV rows across SPLIT blocks per KV
  // head to fill GB10 (grid=Hkv=8 alone starves the SMs on this memory-bound decode attn).
  // L1: `gate` (or nullptr) folds the softplus out-gate into the normalized store below.
  const unsigned blk = static_cast<unsigned>(group * 32);
  cudaStream_t st = AsStream(q);
  const int SPLIT = ChooseSplitEager(kv_rows, Hkv);
  if (SPLIT <= 1) {  // byte-exact single-block fallback (tiny kv_rows)
    DecodeAttnGqaKernel<<<static_cast<unsigned>(Hkv), blk, 0, st>>>(
        o, qd, k, v, Hq, Hkv, Dh, group, kv_rows, q_pos, first_pos, window, scale, gate);
    return;
  }
  const size_t need = static_cast<size_t>(Hq) * static_cast<size_t>(SPLIT) *
                      static_cast<size_t>(Dh + 2);
  float* sc = EnsureSplitScratch(need);
  float* op = sc;  // [Hq*SPLIT*Dh] partial numerators | [Hq*SPLIT] m | [Hq*SPLIT] l
  float* mp = sc + static_cast<size_t>(Hq) * static_cast<size_t>(SPLIT) * static_cast<size_t>(Dh);
  float* lp = mp + static_cast<size_t>(Hq) * static_cast<size_t>(SPLIT);
  const dim3 grid(static_cast<unsigned>(Hkv), static_cast<unsigned>(SPLIT));
  DecodeAttnGqaSplitKernel<<<grid, blk, 0, st>>>(mp, lp, op, qd, k, v, Hq, Hkv, Dh, group, kv_rows,
                                                 q_pos, first_pos, window, scale, SPLIT);
  const size_t csh = static_cast<size_t>(3 * SPLIT) * sizeof(float);
  DecodeAttnCombineKernel<<<static_cast<unsigned>(Hq), static_cast<unsigned>(kLagDh), csh, st>>>(
      o, mp, lp, op, Hq, Dh, SPLIT, gate);
}
void DecodeAttnGqaGLaunch(Queue& q, float* o, const float* qd, const float* k, const float* v,
                          const float* knew, const float* vnew, int64_t Hq, int64_t Hkv, int64_t Dh,
                          int64_t group, int64_t first_pos, int64_t window, float scale,
                          const int* len_dev, const int* pos_dev, const float* gate) {
  // block = group*32 (QG warps). CAPTURE-SAFE split-K: grid=dim3(Hkv,SPLIT) with a FIXED
  // SPLIT (baked at capture); the per-split KV-row range (rps) is computed IN-KERNEL from
  // *len_dev at replay so one capture covers a growing context. The scratch is pre-sized by
  // the gstate-0 eager warm-run, so no cudaMalloc happens inside the gstate-1 capture and
  // the combine's baked scratch pointer stays valid across replays. STATIC shared (ksh/vsh).
  const unsigned blk = static_cast<unsigned>(group * 32);
  cudaStream_t st = AsStream(q);
  constexpr int SPLIT = kLagSplitGraph;
  if (SPLIT <= 1) {  // byte-exact single-block fallback (keeps the original kernel live)
    DecodeAttnGqaGKernel<<<static_cast<unsigned>(Hkv), blk, 0, st>>>(
        o, qd, k, v, knew, vnew, Hq, Hkv, Dh, group, first_pos, window, scale, len_dev, pos_dev,
        gate);
    return;
  }
  const size_t need = static_cast<size_t>(Hq) * static_cast<size_t>(SPLIT) *
                      static_cast<size_t>(Dh + 2);
  float* sc = EnsureSplitScratch(need);
  float* op = sc;
  float* mp = sc + static_cast<size_t>(Hq) * static_cast<size_t>(SPLIT) * static_cast<size_t>(Dh);
  float* lp = mp + static_cast<size_t>(Hq) * static_cast<size_t>(SPLIT);
  const dim3 grid(static_cast<unsigned>(Hkv), static_cast<unsigned>(SPLIT));
  DecodeAttnGqaSplitGKernel<<<grid, blk, 0, st>>>(mp, lp, op, qd, k, v, knew, vnew, Hq, Hkv, Dh,
                                                  group, first_pos, window, scale, len_dev, pos_dev,
                                                  SPLIT);
  const size_t csh = static_cast<size_t>(3 * SPLIT) * sizeof(float);
  DecodeAttnCombineKernel<<<static_cast<unsigned>(Hq), static_cast<unsigned>(kLagDh), csh, st>>>(
      o, mp, lp, op, Hq, Dh, SPLIT, gate);
}
void SoftplusHeadGateLaunch(Queue& q, float* attn, const float* gl, int64_t Hq, int64_t Dh) {
  SoftplusHeadGateKernel<<<Blocks(Hq * Dh), kTPB, 0, AsStream(q)>>>(attn, gl, Hq, Dh);
}
void SigmoidTopKLaunch(Queue& q, int32_t* ids, float* weights, const float* logits,
                       const float* bias, bool has_bias, int64_t E, int64_t topk, bool renorm,
                       float scale) {
  const int threads = 256;  // power-of-two for the block tree-reduce (E<=256 experts
                            // handled by the strided argmax + idx=E sentinel).
  const size_t shmem = static_cast<size_t>(3 * E) * sizeof(float);
  SigmoidTopKKernel<<<1, threads, shmem, AsStream(q)>>>(ids, weights, logits, bias, has_bias, E,
                                                        topk, renorm, scale);
}
void LmHeadGemvLaunch(Queue& q, float* out, const void* w_bf16, const float* x, int64_t N,
                      int64_t K) {
  // ONE WARP per output row n. Block = kLmHeadWarps warps (kLmHeadWarps*32 threads);
  // grid = ceil(N / kLmHeadWarps) so each of the 8 warps/block owns an independent row.
  // Fixed grid + fixed pointers ⇒ CUDA-graph capturable. Warp-shuffle reduce (no shared,
  // no __syncthreads) replaces the old block-per-row sh[256] tree-reduce.
  constexpr int kLmHeadWarps = 8;  // 8 warps = 256 threads/block
  const unsigned grid = static_cast<unsigned>((N + kLmHeadWarps - 1) / kLmHeadWarps);
  LmHeadGemvKernel<<<grid, kLmHeadWarps * 32, 0, AsStream(q)>>>(
      out, reinterpret_cast<const __nv_bfloat16*>(w_bf16), x, N, K);
}
void EmbedGatherLaunch(Queue& q, float* out, const void* table, bool is_bf16, const int64_t* tok,
                       int64_t H) {
  // grid=ceil(H/kTPB), fixed pointers, reads tok[0] on-device ⇒ CUDA-graph capturable.
  EmbedGatherKernel<<<Blocks(H), kTPB, 0, AsStream(q)>>>(out, table, is_bf16 ? 1 : 0, tok, H);
}

const LagunaDeviceKernels kLaguna = {&RmsNormSeqLaunch,    &RopeFromCacheLaunch,
                                     &DecodeAttnGqaLaunch, &SoftplusHeadGateLaunch,
                                     &SigmoidTopKLaunch,   &DecodeAttnGqaGLaunch,
                                     &LmHeadGemvLaunch,    &AppendKvRowLaunch,
                                     &RopeFromCacheGLaunch, &EmbedGatherLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLaguna, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kLaguna)));
  }
} registrar;

}  // namespace
}  // namespace vllm::laguna
