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

// ── GQA T=1 decode attention (bit-exact to LagunaAttention:701) ──
// One thread per query head h (kvh=h/group). 3-pass host-order softmax: (A) global
// max over masked keys, (B) denom = Σ exp(logit-max), (C) o += (exp/denom)·v. The dot
// is recomputed each pass (deterministic → identical value) to avoid per-key storage.
// K/V laid out [kv_rows, Hkv, Dh]; kv_pos[j]=j, q_pos=pos; causal + per-layer window.
// ONE BLOCK per query head (blockDim threads cooperate over kv_rows). 3-pass
// host-order softmax parallelized: block-reduce max, block-reduce denom, atomicAdd
// weighted-V into a shared [Dh] accumulator. NEAR-TIE vs the host sequential softmax
// (block reduction reorders) — accepted in the device regime. Dh<=128.
__global__ void DecodeAttnGqaKernel(float* o, const float* q, const float* k, const float* v,
                                    int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group,
                                    int64_t kv_rows, int64_t q_pos, int64_t first_pos,
                                    int64_t window, float scale) {
  const int64_t h = static_cast<int64_t>(blockIdx.x);
  if (h >= Hq) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int nth = static_cast<int>(blockDim.x);
  const int64_t kvh = h / group;
  const float* qrow = q + h * Dh;
  float* orow = o + h * Dh;
  __shared__ float red[256];
  __shared__ float sao[128];
  for (int d = tid; d < Dh; d += nth) sao[d] = 0.0f;
  __syncthreads();

  auto masked = [&](int64_t j) -> bool {
    const int64_t pj = first_pos + j;
    if (pj > q_pos) return true;
    if (window > 0 && q_pos - pj >= window) return true;
    return false;
  };
  auto dotj = [&](int64_t j) -> float {
    const float* krow = k + (j * Hkv + kvh) * Dh;
    float dot = 0.0f;
    for (int64_t d = 0; d < Dh; ++d) dot += qrow[d] * krow[d];
    return dot * scale;
  };
  // (A) block-reduce max
  float lmax = -CUDART_INF_F;
  for (int64_t j = tid; j < kv_rows; j += nth)
    if (!masked(j)) lmax = fmaxf(lmax, dotj(j));
  red[tid] = lmax;
  __syncthreads();
  for (int s = nth / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float maxs = red[0];
  __syncthreads();
  if (maxs == -CUDART_INF_F) {  // no visible key -> attn 0 (matches host)
    for (int d = tid; d < Dh; d += nth) orow[d] = 0.0f;
    return;
  }
  // (B) block-reduce denom
  float lsum = 0.0f;
  for (int64_t j = tid; j < kv_rows; j += nth)
    if (!masked(j)) lsum += expf(dotj(j) - maxs);
  red[tid] = lsum;
  __syncthreads();
  for (int s = nth / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float denom = red[0];
  __syncthreads();
  // (C) weighted V -> shared accumulator
  for (int64_t j = tid; j < kv_rows; j += nth) {
    if (masked(j)) continue;
    const float e = expf(dotj(j) - maxs);
    if (e == 0.0f) continue;
    const float pw = e / denom;
    const float* vrow = v + (j * Hkv + kvh) * Dh;
    for (int64_t d = 0; d < Dh; ++d) atomicAdd(&sao[d], pw * vrow[d]);
  }
  __syncthreads();
  for (int d = tid; d < Dh; d += nth) orow[d] = sao[d];
}

// ── Brick A2: GRAPH GQA T=1 decode attention (capturable variant of DecodeAttnGqaKernel) ──
// DecodeAttnGqaKernel bakes `kv_rows`/`q_pos` into the launch (host args), so a
// captured graph would freeze the KV length + position. This variant reads BOTH from
// DEVICE buffers (len_dev/pos_dev) at runtime and takes the current token's row
// (knew/vnew, [Hkv,Dh]) SEPARATELY — it is NOT yet appended to the cache (the driver
// appends it between replays). It attends physical cache rows j in [0,len) (K/V laid
// out [rows,Hkv,Dh]; row j global pos = first_pos+j) PLUS the new row (index len,
// global pos = q_pos). Same 3-pass block-reduced softmax as the eager kernel; the key
// set {cache[0..len), knew} == the eager kernel's cache[0..len+1) once appended ⇒
// bit-identical. first_pos/window/Hq/Hkv/Dh/group/scale are baked per-layer constants.
__global__ void DecodeAttnGqaGKernel(float* o, const float* q, const float* k, const float* v,
                                     const float* knew, const float* vnew, int64_t Hq, int64_t Hkv,
                                     int64_t Dh, int64_t group, int64_t first_pos, int64_t window,
                                     float scale, const int* len_dev, const int* pos_dev) {
  const int64_t h = static_cast<int64_t>(blockIdx.x);
  if (h >= Hq) return;
  const int tid = static_cast<int>(threadIdx.x);
  const int nth = static_cast<int>(blockDim.x);
  const int64_t kvh = h / group;
  const int64_t len = static_cast<int64_t>(*len_dev);   // prior cache rows (== dev_rows)
  const int64_t q_pos = static_cast<int64_t>(*pos_dev);  // this token's global position
  const int64_t total = len + 1;                         // cache rows + the new row
  const float* qrow = q + h * Dh;
  float* orow = o + h * Dh;
  __shared__ float red[256];
  __shared__ float sao[128];
  for (int d = tid; d < Dh; d += nth) sao[d] = 0.0f;
  __syncthreads();

  auto pos_of = [&](int64_t j) -> int64_t { return (j < len) ? (first_pos + j) : q_pos; };
  auto masked = [&](int64_t j) -> bool {
    const int64_t pj = pos_of(j);
    if (pj > q_pos) return true;
    if (window > 0 && q_pos - pj >= window) return true;
    return false;
  };
  auto krow_of = [&](int64_t j) -> const float* {
    return (j < len) ? (k + (j * Hkv + kvh) * Dh) : (knew + kvh * Dh);
  };
  auto vrow_of = [&](int64_t j) -> const float* {
    return (j < len) ? (v + (j * Hkv + kvh) * Dh) : (vnew + kvh * Dh);
  };
  auto dotj = [&](int64_t j) -> float {
    const float* krow = krow_of(j);
    float dot = 0.0f;
    for (int64_t d = 0; d < Dh; ++d) dot += qrow[d] * krow[d];
    return dot * scale;
  };
  // (A) block-reduce max
  float lmax = -CUDART_INF_F;
  for (int64_t j = tid; j < total; j += nth)
    if (!masked(j)) lmax = fmaxf(lmax, dotj(j));
  red[tid] = lmax;
  __syncthreads();
  for (int s = nth / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float maxs = red[0];
  __syncthreads();
  if (maxs == -CUDART_INF_F) {  // no visible key -> attn 0 (matches host)
    for (int d = tid; d < Dh; d += nth) orow[d] = 0.0f;
    return;
  }
  // (B) block-reduce denom
  float lsum = 0.0f;
  for (int64_t j = tid; j < total; j += nth)
    if (!masked(j)) lsum += expf(dotj(j) - maxs);
  red[tid] = lsum;
  __syncthreads();
  for (int s = nth / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float denom = red[0];
  __syncthreads();
  // (C) weighted V -> shared accumulator
  for (int64_t j = tid; j < total; j += nth) {
    if (masked(j)) continue;
    const float e = expf(dotj(j) - maxs);
    if (e == 0.0f) continue;
    const float pw = e / denom;
    const float* vrow = vrow_of(j);
    for (int64_t d = 0; d < Dh; ++d) atomicAdd(&sao[d], pw * vrow[d]);
  }
  __syncthreads();
  for (int d = tid; d < Dh; d += nth) orow[d] = sao[d];
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
  // one block per query head; 128 threads cooperate over kv_rows.
  DecodeAttnGqaKernel<<<static_cast<unsigned>(Hq), 128, 0, AsStream(q)>>>(
      o, qd, k, v, Hq, Hkv, Dh, group, kv_rows, q_pos, first_pos, window, scale);
}
void DecodeAttnGqaGLaunch(Queue& q, float* o, const float* qd, const float* k, const float* v,
                          const float* knew, const float* vnew, int64_t Hq, int64_t Hkv, int64_t Dh,
                          int64_t group, int64_t first_pos, int64_t window, float scale,
                          const int* len_dev, const int* pos_dev) {
  // one block per query head; 128 threads cooperate over (len+1) rows read from the
  // DEVICE len_dev at replay. FIXED launch shape (no kv_rows-dependent shmem) — the
  // __shared__ red[256]/sao[128] are static, so the grid/shmem never vary per step.
  DecodeAttnGqaGKernel<<<static_cast<unsigned>(Hq), 128, 0, AsStream(q)>>>(
      o, qd, k, v, knew, vnew, Hq, Hkv, Dh, group, first_pos, window, scale, len_dev, pos_dev);
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

const LagunaDeviceKernels kLaguna = {&RmsNormSeqLaunch,       &RopeFromCacheLaunch,
                                     &DecodeAttnGqaLaunch,    &SoftplusHeadGateLaunch,
                                     &SigmoidTopKLaunch,      &DecodeAttnGqaGLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLaguna, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kLaguna)));
  }
} registrar;

}  // namespace
}  // namespace vllm::laguna
