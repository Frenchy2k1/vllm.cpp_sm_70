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

// ── RMSNorm (sequential SoS per row; bit-exact to RmsNorm:94 / RmsNormHeads:112) ──
// One thread per row: ss = Σ x[i]²  (host order); inv = 1/sqrtf(ss/n + eps).
__global__ void RmsNormSeqKernel(float* out, const float* x, const float* w, int64_t rows,
                                 int64_t n, float eps, bool has_w) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + r * n;
  float* orow = out + r * n;
  float ss = 0.0f;
  for (int64_t i = 0; i < n; ++i) ss += xr[i] * xr[i];
  const float inv = 1.0f / sqrtf(ss / static_cast<float>(n) + eps);
  for (int64_t i = 0; i < n; ++i) orow[i] = xr[i] * inv * (has_w ? w[i] : 1.0f);
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
__global__ void DecodeAttnGqaKernel(float* o, const float* q, const float* k, const float* v,
                                    int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group,
                                    int64_t kv_rows, int64_t q_pos, int64_t window, float scale) {
  const int64_t h = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (h >= Hq) return;
  const int64_t kvh = h / group;
  const float* qrow = q + h * Dh;
  float* orow = o + h * Dh;
  for (int64_t d = 0; d < Dh; ++d) orow[d] = 0.0f;

  auto masked = [&](int64_t j) -> bool {
    if (j > q_pos) return true;                          // causal (kv_pos[j]=j)
    if (window > 0 && q_pos - j >= window) return true;  // sliding-window eviction
    return false;
  };
  auto dotj = [&](int64_t j) -> float {
    const float* krow = k + (j * Hkv + kvh) * Dh;
    float dot = 0.0f;
    for (int64_t d = 0; d < Dh; ++d) dot += qrow[d] * krow[d];
    return dot * scale;
  };
  // (A) global max
  float maxs = -CUDART_INF_F;
  for (int64_t j = 0; j < kv_rows; ++j) {
    if (masked(j)) continue;
    const float dv = dotj(j);
    if (dv > maxs) maxs = dv;
  }
  if (maxs == -CUDART_INF_F) return;  // no visible key (matches host: attn stays 0)
  // (B) denom
  float denom = 0.0f;
  for (int64_t j = 0; j < kv_rows; ++j) {
    if (masked(j)) continue;
    denom += expf(dotj(j) - maxs);
  }
  // (C) weighted V
  for (int64_t j = 0; j < kv_rows; ++j) {
    if (masked(j)) continue;
    const float e = expf(dotj(j) - maxs);
    if (e == 0.0f) continue;  // host skips ww==0
    const float pw = e / denom;
    const float* vrow = v + (j * Hkv + kvh) * Dh;
    for (int64_t d = 0; d < Dh; ++d) orow[d] += pw * vrow[d];
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

// ── launchers (no sync — resident) ──
constexpr int kTPB = 128;
inline int Blocks(int64_t n) { return static_cast<int>((n + kTPB - 1) / kTPB); }

void RmsNormSeqLaunch(Queue& q, float* out, const float* x, const float* w, int64_t rows,
                      int64_t n, float eps, bool has_w) {
  RmsNormSeqKernel<<<Blocks(rows), kTPB, 0, AsStream(q)>>>(out, x, w, rows, n, eps, has_w);
}
void RopeFromCacheLaunch(Queue& q, float* x, const float* cache, int64_t heads, int64_t Dh,
                         int64_t rd, int64_t pos) {
  RopeFromCacheKernel<<<Blocks(heads * (rd / 2)), kTPB, 0, AsStream(q)>>>(x, cache, heads, Dh, rd,
                                                                          pos);
}
void DecodeAttnGqaLaunch(Queue& q, float* o, const float* qd, const float* k, const float* v,
                         int64_t Hq, int64_t Hkv, int64_t Dh, int64_t group, int64_t kv_rows,
                         int64_t q_pos, int64_t window, float scale) {
  DecodeAttnGqaKernel<<<Blocks(Hq), kTPB, 0, AsStream(q)>>>(o, qd, k, v, Hq, Hkv, Dh, group,
                                                            kv_rows, q_pos, window, scale);
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
                                     &SigmoidTopKLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLaguna, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kLaguna)));
  }
} registrar;

}  // namespace
}  // namespace vllm::laguna
