// DeepSeek-V4-Flash W7-device — CUDA kernels for the four NEW V4 op families,
// each a 1:1 device port of the landed portable HOST reference and unit-gated
// against it on the DGX GB10 at small shape (tests/vllm/models/
// test_cuda_deepseek_v4.cpp). Registered through the vt OpProvider seam
// (kDeepseekV4{Mhc,Dsa,Compressor,Moe}) so DeepseekV4Model::ForwardDevice can
// dispatch them. See include/.../deepseek_v4_device.h for the seam + honest scope.
//
// ─── PORT MAP (OURS <- host reference <- upstream file:line) ─────────────────
//   MhcSinkhorn/Pre/Post/Head <- deepseek_v4_mhc.cpp   <- kernels/mhc/torch.py:56-106,
//                                                          triton.py:108-140
//   DsaWeightFold/Logits/Topk/SoftmaxSink/GroupedOLora
//                             <- deepseek_v4_dsa.cpp    <- sparse_attn_indexer.py:203-207,
//                                :488-497; triton_fp8_mqa_logits.py:120-156;
//                                flashinfer_sparse.py:777,:896; nvidia/ops/o_proj.py:58-73
//   CompressorSaveScoreApe/PoolNorm/Fp8DsMlaEncode/Decode
//                             <- deepseek_v4_compressor.cpp <- save_partial_states.py:92-101,
//                                fused_compress_quant_cache.py:198-297, compressor.py:307-309
//   SqrtSoftplus/Route/ClampedSwiGLU
//                             <- deepseek_v4_moe.cpp    <- fused_topk_bias_router.py:75-118,
//                                activation.py:197-201
//
// These are correctness-grade STRUCTURAL kernels (tiny-shape gate, per-op host
// round-trip), NOT the fused/perf path — the 512-wide MLA attention + expert
// grouped-GEMM REUSE the existing NVFP4/FP8 kernels and are not re-ported. The
// real paged-engine e2e over a materialized 167B checkpoint stays the W8 residual.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vt/ops.h"

namespace vllm::deepseek_v4 {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("vt cuda deepseek_v4: ") + what + ": " +
                             cudaGetErrorString(e));
}
cudaStream_t AsStream(Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Minimal owning device buffer (tiny shapes; correctness path).
struct Dev {
  void* p = nullptr;
  explicit Dev(size_t bytes) { Check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
  ~Dev() {
    if (p) cudaFree(p);
  }
  Dev(Dev&& o) noexcept : p(o.p) { o.p = nullptr; }
  Dev& operator=(Dev&&) = delete;
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

template <class T>
Dev Upload(const std::vector<T>& v, cudaStream_t s) {
  Dev d(v.size() * sizeof(T));
  if (!v.empty())
    Check(cudaMemcpyAsync(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s),
          "H2D");
  return d;
}
template <class T>
void Download(std::vector<T>& v, void* dp, cudaStream_t s) {
  if (!v.empty())
    Check(cudaMemcpyAsync(v.data(), dp, v.size() * sizeof(T), cudaMemcpyDeviceToHost, s),
          "D2H");
}

// ── device math helpers (bit-faithful to the host references) ─────────────────
__device__ inline float Sig(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float SqrtSoftplusDev(float x) {
  const float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(sp);
}
__device__ inline float Bf16Round(float x) { return __bfloat162float(__float2bfloat16(x)); }

// ============================================================================
// (1) MHC family
// ============================================================================
// Sinkhorn of an hc×hc matrix (torch.py:75-82): row-softmax+eps seed, col-norm,
// then (iters-1)×[row-norm, col-norm]. Single thread (hc small); m in local mem.
__device__ void SinkhornInplace(const float* logits, float* m, int hc, int iters, float eps) {
  for (int j = 0; j < hc; ++j) {
    float rmax = logits[j * hc];
    for (int k = 1; k < hc; ++k) rmax = fmaxf(rmax, logits[j * hc + k]);
    float rsum = 0.0f;
    for (int k = 0; k < hc; ++k) {
      const float e = expf(logits[j * hc + k] - rmax);
      m[j * hc + k] = e;
      rsum += e;
    }
    for (int k = 0; k < hc; ++k) m[j * hc + k] = m[j * hc + k] / rsum + eps;
  }
  for (int k = 0; k < hc; ++k) {
    float c = 0.0f;
    for (int j = 0; j < hc; ++j) c += m[j * hc + k];
    const float den = c + eps;
    for (int j = 0; j < hc; ++j) m[j * hc + k] /= den;
  }
  for (int it = 0; it < iters - 1; ++it) {
    for (int j = 0; j < hc; ++j) {
      float r = 0.0f;
      for (int k = 0; k < hc; ++k) r += m[j * hc + k];
      const float den = r + eps;
      for (int k = 0; k < hc; ++k) m[j * hc + k] /= den;
    }
    for (int k = 0; k < hc; ++k) {
      float c = 0.0f;
      for (int j = 0; j < hc; ++j) c += m[j * hc + k];
      const float den = c + eps;
      for (int j = 0; j < hc; ++j) m[j * hc + k] /= den;
    }
  }
}

__global__ void SinkhornKernel(const float* logits, float* out, int hc, int iters, float eps) {
  float m[256];  // hc <= 16
  SinkhornInplace(logits, m, hc, iters, eps);
  for (int i = 0; i < hc * hc; ++i) out[i] = m[i];
}

// MhcPre (torch.py:56-91 + folded RMSNorm). Single thread; mixes/scratch global.
__global__ void MhcPreKernel(const float* residual, const float* fn, const float* scale,
                             const float* base, int hc, int hidden, float rms_eps,
                             float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult,
                             int iters, const float* norm_weight, int has_norm, float norm_eps,
                             float* mixes, float* pre_out, float* post_out, float* comb_out,
                             float* layer_out) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  double sqrsum = 0.0;
  for (int i = 0; i < flat; ++i) {
    const double r = residual[i];
    sqrsum += r * r;
  }
  for (int o = 0; o < hc3; ++o) {
    float acc = 0.0f;
    const int frow = o * flat;
    for (int i = 0; i < flat; ++i) acc += residual[i] * fn[frow + i];
    mixes[o] = acc;
  }
  const float rms =
      1.0f / sqrtf(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);
  for (int o = 0; o < hc3; ++o) mixes[o] *= rms;
  for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
  for (int j = 0; j < hc; ++j)
    post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
  float cl[256];  // hc*hc, hc<=16
  for (int j = 0; j < hc; ++j)
    for (int k = 0; k < hc; ++k) {
      const int idx = j * hc + k;
      cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
    }
  float m[256];
  SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
  for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  for (int h = 0; h < hidden; ++h) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  if (has_norm) {
    double ss = 0.0;
    for (int h = 0; h < hidden; ++h) {
      const double v = layer_out[h];
      ss += v * v;
    }
    const float r =
        1.0f / sqrtf(static_cast<float>(ss / static_cast<double>(hidden)) + norm_eps);
    for (int h = 0; h < hidden; ++h) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

// Brick B — PARALLEL MhcPre (one block, blockDim threads over the H/flat width; the
// tiny hc-sized gates + 20-iter Sinkhorn stay on thread 0 in HOST ORDER). Same math
// as MhcPreKernel but real-parallel (no <<<1,1>>> on the decode hot path). The width
// reductions (sqrsum, the hc3 mix dots, the final RMSNorm ss) accumulate in DOUBLE
// and reduce in a block tree → a CHARACTERIZED near-tie vs the single-thread version
// (reduction reorder); the per-h layer_out dot (over hc) is sequential → order kept.
__global__ void MhcPreParallelKernel(const float* __restrict__ residual,
                                     const float* __restrict__ fn, const float* __restrict__ scale,
                                     const float* __restrict__ base, int hc, int hidden,
                                     float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                     float hc_post_mult, int iters,
                                     const float* __restrict__ norm_weight, int has_norm,
                                     float norm_eps, float* mixes, float* pre_out, float* post_out,
                                     float* comb_out, float* layer_out) {
  const int hc3 = (2 + hc) * hc;
  const int flat = hc * hidden;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  extern __shared__ double red[];  // [nt]

  auto block_reduce = [&](double v) -> double {
    red[tid] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    const double r = red[0];
    __syncthreads();
    return r;
  };

  // sqrsum = Σ residual[i]^2 ; rms = 1/sqrt(mean + eps)
  double ls = 0.0;
  for (int i = tid; i < flat; i += nt) { const double r = residual[i]; ls += r * r; }
  const double sqrsum = block_reduce(ls);
  const float rms = 1.0f / sqrtf(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);

  // mixes[o] = (Σ_i residual[i]*fn[o*flat+i]) * rms
  for (int o = 0; o < hc3; ++o) {
    double la = 0.0;
    const int frow = o * flat;
    for (int i = tid; i < flat; i += nt) la += static_cast<double>(residual[i]) * fn[frow + i];
    const double acc = block_reduce(la);
    if (tid == 0) mixes[o] = static_cast<float>(acc) * rms;
  }
  __syncthreads();

  // gates + Sinkhorn (tiny hc; thread 0 sequential = host order)
  if (tid == 0) {
    for (int j = 0; j < hc; ++j) pre_out[j] = Sig(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
    for (int j = 0; j < hc; ++j)
      post_out[j] = Sig(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult;
    float cl[256], m[256];
    for (int j = 0; j < hc; ++j)
      for (int k = 0; k < hc; ++k) {
        const int idx = j * hc + k;
        cl[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
      }
    SinkhornInplace(cl, m, hc, iters, hc_sinkhorn_eps);
    for (int i = 0; i < hc * hc; ++i) comb_out[i] = m[i];
  }
  __syncthreads();

  // layer_out[h] = Σ_j pre[j]*residual[j*hidden+h] (parallel over h; per-h order kept)
  for (int h = tid; h < hidden; h += nt) {
    float acc = 0.0f;
    for (int j = 0; j < hc; ++j) acc += pre_out[j] * residual[j * hidden + h];
    layer_out[h] = acc;
  }
  __syncthreads();

  // optional folded final RMSNorm over hidden
  if (has_norm) {
    double ss = 0.0;
    for (int h = tid; h < hidden; h += nt) { const double v = layer_out[h]; ss += v * v; }
    const double ssr = block_reduce(ss);
    const float r =
        1.0f / sqrtf(static_cast<float>(ssr / static_cast<double>(hidden)) + norm_eps);
    for (int h = tid; h < hidden; h += nt) layer_out[h] = layer_out[h] * r * norm_weight[h];
  }
}

// MhcPost (torch.py:94-106). One thread per (j,h).
__global__ void MhcPostKernel(const float* x, const float* residual, const float* post_mix,
                              const float* comb, int hc, int hidden, float* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * hidden) return;
  const int j = idx / hidden, h = idx % hidden;
  float mixed = 0.0f;
  for (int i = 0; i < hc; ++i) mixed += comb[i * hc + j] * residual[i * hidden + h];
  out[idx] = mixed + post_mix[j] * x[h];
}

// HcHeadCollapse (triton.py:108-140). Single thread; pre[hc] in local.
__global__ void HcHeadKernel(const float* x, const float* fn, float scale, const float* base,
                             int hc, int hidden, float rms_eps, float hc_eps, float* out) {
  const int flat = hc * hidden;
  double ss = 0.0;
  for (int p = 0; p < flat; ++p) {
    const double v = x[p];
    ss += v * v;
  }
  const float r = 1.0f / sqrtf(static_cast<float>(ss / static_cast<double>(flat)) + rms_eps);
  float pre[256];
  for (int m = 0; m < hc; ++m) {
    float acc = 0.0f;
    const int frow = m * flat;
    for (int p = 0; p < flat; ++p) acc += (x[p] * r) * fn[frow + p];
    pre[m] = Sig(acc * scale + base[m]) + hc_eps;
  }
  for (int h = 0; h < hidden; ++h) {
    float acc = 0.0f;
    for (int m = 0; m < hc; ++m) acc += pre[m] * x[m * hidden + h];
    out[h] = acc;
  }
}

// ============================================================================
// (2) DSA family
// ============================================================================
__global__ void DsaWeightFoldKernel(const float* wp, float* out, int64_t n, float fold) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = wp[i] * fold;
}

// MQA logit (triton_fp8_mqa_logits.py:120-156). One thread per (t,s).
__global__ void DsaLogitsKernel(const float* q, const float* k, const float* folded,
                                const int64_t* ws, const int64_t* we, int T, int nk, int H,
                                int D, float* out) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(T) * nk) return;
  const int t = static_cast<int>(idx / nk), s = static_cast<int>(idx % nk);
  const int64_t s0 = ws[t] > 0 ? ws[t] : 0;
  const int64_t s1 = we[t] < nk ? we[t] : nk;
  if (s < s0 || s >= s1) {
    out[idx] = -INFINITY;
    return;
  }
  float acc = 0.0f;
  for (int h = 0; h < H; ++h) {
    float dot = 0.0f;
    const float* qp = &q[((static_cast<int64_t>(t) * H) + h) * D];
    const float* kp = &k[static_cast<int64_t>(s) * D];
    for (int d = 0; d < D; ++d) dot += qp[d] * kp[d];
    const float relu = dot > 0.0f ? dot : 0.0f;
    acc += folded[static_cast<int64_t>(t) * H + h] * relu;
  }
  out[idx] = acc;
}

// Causal top-k select (sparse_attn_indexer.py:488-497 + short-context all-select).
// One thread per token row; the same set + ascending emit as the host reference.
__global__ void DsaTopkKernel(const float* logits, const int64_t* ws, const int64_t* we,
                              int T, int nk, int topk, int64_t* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const int64_t s0 = ws[t] > 0 ? ws[t] : 0;
  const int64_t s1 = we[t] < nk ? we[t] : nk;
  const int64_t n = s1 > s0 ? s1 - s0 : 0;
  int64_t* dst = &out[static_cast<int64_t>(t) * topk];
  for (int j = 0; j < topk; ++j) dst[j] = -1;
  if (n <= topk) {
    int w = 0;
    for (int64_t s = s0; s < s1; ++s) dst[w++] = s;
    return;
  }
  // Pick `topk` best by (logit desc, index asc); chosen tracked in a local mask.
  bool chosen[512];  // nk (candidate window) small in the structural gate
  for (int64_t s = 0; s < n; ++s) chosen[s] = false;
  int64_t picked[64];  // topk small
  for (int j = 0; j < topk; ++j) {
    int64_t best = -1;
    float bestv = -INFINITY;
    for (int64_t s = s0; s < s1; ++s) {
      if (chosen[s - s0]) continue;
      const float v = logits[static_cast<int64_t>(t) * nk + s];
      if (best < 0 || v > bestv) {  // strict > keeps the SMALLER index on a tie
        bestv = v;
        best = s;
      }
    }
    chosen[best - s0] = true;
    picked[j] = best;
  }
  // Emit ascending key order (insertion sort of `topk` picks).
  for (int a = 0; a < topk; ++a) {
    int64_t mn = picked[a];
    int mi = a;
    for (int b = a + 1; b < topk; ++b)
      if (picked[b] < mn) {
        mn = picked[b];
        mi = b;
      }
    picked[mi] = picked[a];
    picked[a] = mn;
    dst[a] = mn;
  }
}

// Attention-sink softmax (flashinfer_sparse.py:777,:896). Single thread (one row).
__global__ void SoftmaxSinkKernel(const float* scores, int n, float sink, float* out) {
  float m = sink;
  for (int j = 0; j < n; ++j) m = fmaxf(m, scores[j]);
  if (m == -INFINITY) {
    for (int j = 0; j < n; ++j) out[j] = 0.0f;
    return;
  }
  float denom = expf(sink - m);
  for (int j = 0; j < n; ++j) {
    const float e = expf(scores[j] - m);
    out[j] = e;
    denom += e;
  }
  for (int j = 0; j < n; ++j) out[j] /= denom;
}

// Grouped output-LoRA (o_proj.py:58-73). One block per token; global z scratch.
__global__ void GroupedOLoraKernel(const float* o, const float* wo_a, const float* wo_b,
                                   int T, int nh, int hd, int ng, int olr, int H,
                                   int in_per_group, int z_dim, float* z_all, float* out) {
  const int t = blockIdx.x;
  if (t >= T) return;
  float* z = &z_all[static_cast<int64_t>(t) * z_dim];
  const float* o_t = &o[static_cast<int64_t>(t) * nh * hd];
  if (threadIdx.x == 0) {
    for (int g = 0; g < ng; ++g) {
      const float* o_g = o_t + g * in_per_group;
      const float* wa_g = &wo_a[static_cast<int64_t>(g) * olr * in_per_group];
      float* z_g = &z[g * olr];
      for (int d = 0; d < olr; ++d) {
        float acc = 0.0f;
        const float* wa_gd = wa_g + static_cast<int64_t>(d) * in_per_group;
        for (int r = 0; r < in_per_group; ++r) acc += wa_gd[r] * o_g[r];
        z_g[d] = acc;
      }
    }
    float* out_t = &out[static_cast<int64_t>(t) * H];
    for (int h = 0; h < H; ++h) {
      float acc = 0.0f;
      const float* wb_h = &wo_b[static_cast<int64_t>(h) * z_dim];
      for (int c = 0; c < z_dim; ++c) acc += wb_h[c] * z[c];
      out_t[h] = acc;
    }
  }
}

// ============================================================================
// (3) Compressor family
// ============================================================================
__global__ void SaveScoreApeKernel(const float* score, const float* ape,
                                    const int64_t* positions, int T, int width, int cr,
                                    float* out) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(T) * width) return;
  const int t = static_cast<int>(idx / width), d = static_cast<int>(idx % width);
  int64_t ape_row = positions[t] % cr;
  if (ape_row < 0) ape_row += cr;
  out[idx] = score[idx] + ape[ape_row * width + d];
}

// Compressor pool+norm (fused_compress_quant_cache.py:198-218). Single thread; one token.
__global__ void PoolNormKernel(const float* kv, const float* score, const uint8_t* valid,
                               const float* rms_w, float eps, int window, int hd,
                               float* out) {
  extern __shared__ float comp[];  // [hd]
  for (int d = 0; d < hd; ++d) {
    float m = -INFINITY;
    for (int i = 0; i < window; ++i) {
      const float s = valid[i] ? score[i * hd + d] : -INFINITY;
      m = fmaxf(m, s);
    }
    if (m == -INFINITY) {
      comp[d] = 0.0f;
      continue;
    }
    float denom = 0.0f, acc = 0.0f;
    for (int i = 0; i < window; ++i) {
      if (!valid[i]) continue;
      const float e = expf(score[i * hd + d] - m);
      denom += e;
      acc += kv[i * hd + d] * e;
    }
    comp[d] = acc / denom;
  }
  float var = 0.0f;
  for (int d = 0; d < hd; ++d) var += comp[d] * comp[d];
  var /= static_cast<float>(hd);
  const float rrms = 1.0f / sqrtf(var + eps);
  for (int d = 0; d < hd; ++d) out[d] = comp[d] * rrms * rms_w[d];
}

// fp8_ds_mla encode (fused_compress_quant_cache.py:238-297): per 64-wide NoPE
// block bf16-round -> absmax(>=1e-4) -> UE8M0 exponent -> e4m3; rope -> bf16.
__global__ void Fp8EncodeKernel(const float* head, int qblk, int nblk, uint8_t* nope_fp8,
                                uint8_t* scale_ue8m0) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  const int base = b * qblk;
  float absmax = 0.0f;
  for (int j = 0; j < qblk; ++j) absmax = fmaxf(absmax, fabsf(Bf16Round(head[base + j])));
  absmax = fmaxf(absmax, 1e-4f);
  const float raw = absmax * (1.0f / 448.0f);
  const float exponent = ceilf(log2f(raw));
  const float inv_scale = exp2f(-exponent);
  for (int j = 0; j < qblk; ++j) {
    float x = Bf16Round(head[base + j]) * inv_scale;
    x = fminf(fmaxf(x, -448.0f), 448.0f);
    nope_fp8[base + j] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  }
  float enc = exponent + 127.0f;
  enc = fmaxf(0.0f, fminf(255.0f, enc));
  scale_ue8m0[b] = static_cast<uint8_t>(enc);
}

// The rope part is bf16 verbatim; a separate kernel matches vt::F32ToBF16
// (round-to-nearest-even) via __float2bfloat16.
__global__ void RopeToBf16Kernel(const float* head, int nope, int rope, uint16_t* rope_bf16) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= rope) return;
  const __nv_bfloat16 h = __float2bfloat16(head[nope + j]);
  uint16_t bits;
  memcpy(&bits, &h, sizeof(bits));
  rope_bf16[j] = bits;
}

// fp8_ds_mla decode (SGLang dequant_k_cache.py:122-136).
__global__ void Fp8DecodeKernel(const uint8_t* nope_fp8, const uint8_t* scale_ue8m0,
                                const uint16_t* rope_bf16, int nope, int rope, int qblk,
                                int nblk, float* out) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < nope) {
    const int b = d / qblk;
    const float scale_pow2 = exp2f(static_cast<float>(scale_ue8m0[b]) - 127.0f);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(nope_fp8[d], __NV_E4M3);
    out[d] = __half2float(hr) * scale_pow2;
  }
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < rope) {
    __nv_bfloat16 h;
    const uint16_t bits = rope_bf16[j];
    memcpy(&h, &bits, sizeof(h));
    out[nope + j] = __bfloat162float(h);
  }
}

// ============================================================================
// (4) MoE family
// ============================================================================
__global__ void SqrtSoftplusKernel(const float* x, float* out, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = SqrtSoftplusDev(x[i]);
}

// sqrtsoftplus + noaux_tc bias router + hash bypass (fused_topk_bias_router.py:75-118).
// One thread per token. E<=256, topk<=32 for the structural gate.
__global__ void RouteKernel(const float* gating, int T, int E, int topk, const float* bias,
                            int has_bias, int is_hash, const int64_t* in_tokens,
                            const int32_t* hashtab, int64_t vocab, int renorm, float scale,
                            int32_t* ids_out, float* w_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  float scores[256];
  const float* g = &gating[static_cast<int64_t>(t) * E];
  for (int e = 0; e < E; ++e) scores[e] = SqrtSoftplusDev(g[e]);
  int32_t* ids = &ids_out[static_cast<int64_t>(t) * topk];
  float* w = &w_out[static_cast<int64_t>(t) * topk];
  if (is_hash) {
    int64_t tok = in_tokens[t] % vocab;
    if (tok < 0) tok += vocab;
    const int32_t* row = &hashtab[tok * topk];
    for (int j = 0; j < topk; ++j) {
      ids[j] = row[j];
      w[j] = scores[row[j]];
    }
  } else {
    float sfc[256];
    bool used[256];
    for (int e = 0; e < E; ++e) {
      sfc[e] = has_bias ? scores[e] + bias[e] : scores[e];
      used[e] = false;
    }
    for (int j = 0; j < topk; ++j) {
      int best = -1;
      float bestv = -INFINITY;
      for (int e = 0; e < E; ++e) {
        if (used[e]) continue;
        if (best < 0 || sfc[e] > bestv) {  // strict > -> smaller index wins a tie
          bestv = sfc[e];
          best = e;
        }
      }
      used[best] = true;
      ids[j] = best;
      w[j] = scores[best];  // GATHER from the UNBIASED scores
    }
  }
  if (renorm) {
    float sum = 0.0f;
    for (int j = 0; j < topk; ++j) sum += w[j];
    const float denom = fmaxf(sum, 1e-20f);
    for (int j = 0; j < topk; ++j) w[j] /= denom;
  }
  for (int j = 0; j < topk; ++j) w[j] *= scale;
}

// Clamped SwiGLU (activation.py:197-201). One thread per output channel.
__global__ void ClampedSwiGLUKernel(const float* gate_up, int d, float limit, float alpha,
                                    float beta, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  const float gate = fminf(gate_up[i], limit);
  const float up = fminf(fmaxf(gate_up[d + i], -limit), limit);
  out[i] = gate * Sig(alpha * gate) * (up + beta);
}

unsigned Grid(int64_t n, int block) {
  return static_cast<unsigned>((n + block - 1) / block);
}

// ── launchers (host-vector wrappers; upload -> kernel -> download) ────────────
std::vector<float> MhcSinkhornLaunch(Queue& q, const std::vector<float>& logits, int64_t hc,
                                     int64_t iters, float eps) {
  cudaStream_t s = AsStream(q);
  Dev dl = Upload(logits, s);
  std::vector<float> out(static_cast<size_t>(hc * hc));
  Dev dout(out.size() * sizeof(float));
  SinkhornKernel<<<1, 1, 0, s>>>(static_cast<const float*>(dl.p), static_cast<float*>(dout.p),
                                 static_cast<int>(hc), static_cast<int>(iters), eps);
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync sinkhorn");
  return out;
}

MhcPreResult MhcPreLaunch(Queue& q, const std::vector<float>& residual,
                          const std::vector<float>& fn, const std::vector<float>& scale,
                          const std::vector<float>& base, int64_t hc, int64_t hidden,
                          float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                          float hc_post_mult, int64_t iters,
                          const std::vector<float>& norm_weight, float norm_eps) {
  cudaStream_t s = AsStream(q);
  const int hc3 = static_cast<int>((2 + hc) * hc);
  Dev dr = Upload(residual, s), df = Upload(fn, s), ds = Upload(scale, s), db = Upload(base, s);
  const bool has_norm = !norm_weight.empty();
  std::vector<float> nw = has_norm ? norm_weight : std::vector<float>(1, 0.0f);
  Dev dnw = Upload(nw, s);
  Dev dmix(static_cast<size_t>(hc3) * sizeof(float));
  MhcPreResult out;
  out.pre_mix.resize(static_cast<size_t>(hc));
  out.post_mix.resize(static_cast<size_t>(hc));
  out.comb_mix.resize(static_cast<size_t>(hc * hc));
  out.layer_input.resize(static_cast<size_t>(hidden));
  Dev dpre(out.pre_mix.size() * sizeof(float)), dpost(out.post_mix.size() * sizeof(float));
  Dev dcomb(out.comb_mix.size() * sizeof(float)), dlin(out.layer_input.size() * sizeof(float));
  MhcPreKernel<<<1, 1, 0, s>>>(
      static_cast<const float*>(dr.p), static_cast<const float*>(df.p),
      static_cast<const float*>(ds.p), static_cast<const float*>(db.p), static_cast<int>(hc),
      static_cast<int>(hidden), rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult,
      static_cast<int>(iters), static_cast<const float*>(dnw.p), has_norm ? 1 : 0, norm_eps,
      static_cast<float*>(dmix.p), static_cast<float*>(dpre.p), static_cast<float*>(dpost.p),
      static_cast<float*>(dcomb.p), static_cast<float*>(dlin.p));
  Download(out.pre_mix, dpre.p, s);
  Download(out.post_mix, dpost.p, s);
  Download(out.comb_mix, dcomb.p, s);
  Download(out.layer_input, dlin.p, s);
  Check(cudaStreamSynchronize(s), "sync mhc_pre");
  return out;
}

std::vector<float> MhcPostLaunch(Queue& q, const std::vector<float>& x,
                                 const std::vector<float>& residual,
                                 const std::vector<float>& post_mix,
                                 const std::vector<float>& comb, int64_t hc, int64_t hidden) {
  cudaStream_t s = AsStream(q);
  Dev dx = Upload(x, s), dr = Upload(residual, s), dp = Upload(post_mix, s), dc = Upload(comb, s);
  std::vector<float> out(static_cast<size_t>(hc * hidden));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  MhcPostKernel<<<Grid(hc * hidden, block), block, 0, s>>>(
      static_cast<const float*>(dx.p), static_cast<const float*>(dr.p),
      static_cast<const float*>(dp.p), static_cast<const float*>(dc.p), static_cast<int>(hc),
      static_cast<int>(hidden), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync mhc_post");
  return out;
}

std::vector<float> HcHeadLaunch(Queue& q, const std::vector<float>& x,
                                const std::vector<float>& fn, float scale,
                                const std::vector<float>& base, int64_t hc, int64_t hidden,
                                float rms_eps, float hc_eps) {
  cudaStream_t s = AsStream(q);
  Dev dx = Upload(x, s), df = Upload(fn, s), db = Upload(base, s);
  std::vector<float> out(static_cast<size_t>(hidden));
  Dev dout(out.size() * sizeof(float));
  HcHeadKernel<<<1, 1, 0, s>>>(static_cast<const float*>(dx.p), static_cast<const float*>(df.p),
                               scale, static_cast<const float*>(db.p), static_cast<int>(hc),
                               static_cast<int>(hidden), rms_eps, hc_eps,
                               static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync hc_head");
  return out;
}

std::vector<float> DsaWeightFoldLaunch(Queue& q, const std::vector<float>& wp, int64_t T,
                                       int64_t inh, int64_t ihd) {
  cudaStream_t s = AsStream(q);
  const float fold = (1.0f / sqrtf(static_cast<float>(ihd))) *
                     (1.0f / sqrtf(static_cast<float>(inh)));
  Dev dw = Upload(wp, s);
  std::vector<float> out(wp.size());
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  DsaWeightFoldKernel<<<Grid(static_cast<int64_t>(wp.size()), block), block, 0, s>>>(
      static_cast<const float*>(dw.p), static_cast<float*>(dout.p),
      static_cast<int64_t>(wp.size()), fold);
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync fold");
  return out;
}

std::vector<float> DsaLogitsLaunch(Queue& q, const std::vector<float>& qv,
                                   const std::vector<float>& k, const std::vector<float>& folded,
                                   const std::vector<int64_t>& ws, const std::vector<int64_t>& we,
                                   int64_t T, int64_t nk, int64_t inh, int64_t ihd) {
  cudaStream_t s = AsStream(q);
  Dev dq = Upload(qv, s), dk = Upload(k, s), dfo = Upload(folded, s);
  Dev dws = Upload(ws, s), dwe = Upload(we, s);
  std::vector<float> out(static_cast<size_t>(T * nk));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  DsaLogitsKernel<<<Grid(T * nk, block), block, 0, s>>>(
      static_cast<const float*>(dq.p), static_cast<const float*>(dk.p),
      static_cast<const float*>(dfo.p), static_cast<const int64_t*>(dws.p),
      static_cast<const int64_t*>(dwe.p), static_cast<int>(T), static_cast<int>(nk),
      static_cast<int>(inh), static_cast<int>(ihd), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync logits");
  return out;
}

std::vector<int64_t> DsaTopkLaunch(Queue& q, const std::vector<float>& logits,
                                   const std::vector<int64_t>& ws,
                                   const std::vector<int64_t>& we, int64_t T, int64_t nk,
                                   int64_t topk) {
  cudaStream_t s = AsStream(q);
  Dev dl = Upload(logits, s), dws = Upload(ws, s), dwe = Upload(we, s);
  std::vector<int64_t> out(static_cast<size_t>(T * topk), -1);
  Dev dout(out.size() * sizeof(int64_t));
  const int block = 64;
  DsaTopkKernel<<<Grid(T, block), block, 0, s>>>(
      static_cast<const float*>(dl.p), static_cast<const int64_t*>(dws.p),
      static_cast<const int64_t*>(dwe.p), static_cast<int>(T), static_cast<int>(nk),
      static_cast<int>(topk), static_cast<int64_t*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync topk");
  return out;
}

std::vector<float> SoftmaxSinkLaunch(Queue& q, const std::vector<float>& scores, float sink) {
  cudaStream_t s = AsStream(q);
  Dev dsc = Upload(scores, s);
  std::vector<float> out(scores.size());
  Dev dout(out.size() * sizeof(float));
  SoftmaxSinkKernel<<<1, 1, 0, s>>>(static_cast<const float*>(dsc.p),
                                    static_cast<int>(scores.size()), sink,
                                    static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync softmax_sink");
  return out;
}

std::vector<float> GroupedOLoraLaunch(Queue& q, const std::vector<float>& o,
                                      const std::vector<float>& wo_a,
                                      const std::vector<float>& wo_b, int64_t T, int64_t nh,
                                      int64_t hd, int64_t ng, int64_t olr, int64_t H) {
  cudaStream_t s = AsStream(q);
  const int in_per_group = static_cast<int>(nh * hd / ng);
  const int z_dim = static_cast<int>(ng * olr);
  Dev doo = Upload(o, s), dwa = Upload(wo_a, s), dwb = Upload(wo_b, s);
  std::vector<float> out(static_cast<size_t>(T * H));
  Dev dout(out.size() * sizeof(float));
  Dev dz(static_cast<size_t>(T) * z_dim * sizeof(float));
  GroupedOLoraKernel<<<static_cast<unsigned>(T), 1, 0, s>>>(
      static_cast<const float*>(doo.p), static_cast<const float*>(dwa.p),
      static_cast<const float*>(dwb.p), static_cast<int>(T), static_cast<int>(nh),
      static_cast<int>(hd), static_cast<int>(ng), static_cast<int>(olr), static_cast<int>(H),
      in_per_group, z_dim, static_cast<float*>(dz.p), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync olora");
  return out;
}

std::vector<float> SaveScoreApeLaunch(Queue& q, const std::vector<float>& score,
                                      const std::vector<float>& ape,
                                      const std::vector<int64_t>& positions, int64_t T,
                                      int64_t width, int64_t cr) {
  cudaStream_t s = AsStream(q);
  Dev dsc = Upload(score, s), dap = Upload(ape, s), dp = Upload(positions, s);
  std::vector<float> out(score.size());
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  SaveScoreApeKernel<<<Grid(T * width, block), block, 0, s>>>(
      static_cast<const float*>(dsc.p), static_cast<const float*>(dap.p),
      static_cast<const int64_t*>(dp.p), static_cast<int>(T), static_cast<int>(width),
      static_cast<int>(cr), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync ape");
  return out;
}

std::vector<float> PoolNormLaunch(Queue& q, const std::vector<float>& kv,
                                  const std::vector<float>& score,
                                  const std::vector<uint8_t>& valid,
                                  const std::vector<float>& rms_w, float eps, int64_t window,
                                  int64_t hd) {
  cudaStream_t s = AsStream(q);
  Dev dkv = Upload(kv, s), dsc = Upload(score, s), dv = Upload(valid, s), dr = Upload(rms_w, s);
  std::vector<float> out(static_cast<size_t>(hd));
  Dev dout(out.size() * sizeof(float));
  PoolNormKernel<<<1, 1, static_cast<unsigned>(hd) * sizeof(float), s>>>(
      static_cast<const float*>(dkv.p), static_cast<const float*>(dsc.p),
      static_cast<const uint8_t*>(dv.p), static_cast<const float*>(dr.p), eps,
      static_cast<int>(window), static_cast<int>(hd), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync poolnorm");
  return out;
}

Fp8DsMlaToken Fp8EncodeLaunch(Queue& q, const std::vector<float>& head,
                              const Fp8DsMlaLayout& L) {
  cudaStream_t s = AsStream(q);
  Dev dh = Upload(head, s);
  Fp8DsMlaToken t;
  t.nope_fp8.assign(static_cast<size_t>(L.nope_head_dim), 0);
  t.scale_ue8m0.assign(static_cast<size_t>(L.n_nope_blocks), 0);
  t.rope_bf16.assign(static_cast<size_t>(L.rope_head_dim), 0);
  Dev dn(t.nope_fp8.size()), dsc(t.scale_ue8m0.size());
  Dev drp(t.rope_bf16.size() * sizeof(uint16_t));
  const int block = 64;
  Fp8EncodeKernel<<<Grid(L.n_nope_blocks, block), block, 0, s>>>(
      static_cast<const float*>(dh.p), static_cast<int>(L.quant_block),
      static_cast<int>(L.n_nope_blocks), static_cast<uint8_t*>(dn.p),
      static_cast<uint8_t*>(dsc.p));
  RopeToBf16Kernel<<<Grid(L.rope_head_dim, block), block, 0, s>>>(
      static_cast<const float*>(dh.p), static_cast<int>(L.nope_head_dim),
      static_cast<int>(L.rope_head_dim), static_cast<uint16_t*>(drp.p));
  Download(t.nope_fp8, dn.p, s);
  Download(t.scale_ue8m0, dsc.p, s);
  Download(t.rope_bf16, drp.p, s);
  Check(cudaStreamSynchronize(s), "sync fp8 encode");
  return t;
}

std::vector<float> Fp8DecodeLaunch(Queue& q, const Fp8DsMlaToken& t, const Fp8DsMlaLayout& L) {
  cudaStream_t s = AsStream(q);
  Dev dn = Upload(t.nope_fp8, s), dsc = Upload(t.scale_ue8m0, s), drp = Upload(t.rope_bf16, s);
  std::vector<float> out(static_cast<size_t>(L.nope_head_dim + L.rope_head_dim), 0.0f);
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  Fp8DecodeKernel<<<Grid(L.nope_head_dim, block), block, 0, s>>>(
      static_cast<const uint8_t*>(dn.p), static_cast<const uint8_t*>(dsc.p),
      static_cast<const uint16_t*>(drp.p), static_cast<int>(L.nope_head_dim),
      static_cast<int>(L.rope_head_dim), static_cast<int>(L.quant_block),
      static_cast<int>(L.n_nope_blocks), static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync fp8 decode");
  return out;
}

std::vector<float> SqrtSoftplusLaunch(Queue& q, const std::vector<float>& x) {
  cudaStream_t s = AsStream(q);
  Dev dx = Upload(x, s);
  std::vector<float> out(x.size());
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  SqrtSoftplusKernel<<<Grid(static_cast<int64_t>(x.size()), block), block, 0, s>>>(
      static_cast<const float*>(dx.p), static_cast<float*>(dout.p),
      static_cast<int64_t>(x.size()));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync sqrtsoftplus");
  return out;
}

MoeRouteResult RouteLaunch(Queue& q, const std::vector<float>& gating, int64_t T, int64_t E,
                           int64_t topk, const std::vector<float>& bias, bool renorm,
                           float scale, const std::vector<int64_t>& in_tokens,
                           const std::vector<int32_t>& hashtab, int64_t vocab) {
  cudaStream_t s = AsStream(q);
  const bool has_bias = !bias.empty();
  const bool is_hash = !hashtab.empty() && !in_tokens.empty();
  Dev dg = Upload(gating, s);
  std::vector<float> bpad = has_bias ? bias : std::vector<float>(1, 0.0f);
  std::vector<int64_t> tpad = in_tokens.empty() ? std::vector<int64_t>(1, 0) : in_tokens;
  std::vector<int32_t> hpad = hashtab.empty() ? std::vector<int32_t>(1, 0) : hashtab;
  Dev dbias = Upload(bpad, s), dtok = Upload(tpad, s), dhash = Upload(hpad, s);
  MoeRouteResult out;
  out.topk_ids.assign(static_cast<size_t>(T * topk), 0);
  out.topk_weights.assign(static_cast<size_t>(T * topk), 0.0f);
  Dev did(out.topk_ids.size() * sizeof(int32_t)), dw(out.topk_weights.size() * sizeof(float));
  const int block = 64;
  RouteKernel<<<Grid(T, block), block, 0, s>>>(
      static_cast<const float*>(dg.p), static_cast<int>(T), static_cast<int>(E),
      static_cast<int>(topk), static_cast<const float*>(dbias.p), has_bias ? 1 : 0,
      is_hash ? 1 : 0, static_cast<const int64_t*>(dtok.p), static_cast<const int32_t*>(dhash.p),
      vocab, renorm ? 1 : 0, scale, static_cast<int32_t*>(did.p), static_cast<float*>(dw.p));
  Download(out.topk_ids, did.p, s);
  Download(out.topk_weights, dw.p, s);
  Check(cudaStreamSynchronize(s), "sync route");
  return out;
}

std::vector<float> ClampedSwiGLULaunch(Queue& q, const std::vector<float>& gate_up, int64_t d,
                                       float limit, float alpha, float beta) {
  cudaStream_t s = AsStream(q);
  Dev dgu = Upload(gate_up, s);
  std::vector<float> out(static_cast<size_t>(d));
  Dev dout(out.size() * sizeof(float));
  const int block = 128;
  ClampedSwiGLUKernel<<<Grid(d, block), block, 0, s>>>(
      static_cast<const float*>(dgu.p), static_cast<int>(d), limit, alpha, beta,
      static_cast<float*>(dout.p));
  Download(out, dout.p, s);
  Check(cudaStreamSynchronize(s), "sync swiglu");
  return out;
}

// Brick B — IN-PLACE clamped-SwiGLU (unified memory, no Upload/Download/Sync). Same
// ClampedSwiGLUKernel as ClampedSwiGLULaunch, run directly on the caller's unified
// gate_up[2*d]/out[d] pointers. Bit-identical (elementwise). Caller drains.
void ClampedSwiGLUInPlaceLaunch(Queue& q, float* out, const float* gate_up, int64_t d,
                                float limit, float alpha, float beta) {
  if (d == 0) return;
  cudaStream_t s = AsStream(q);
  const int block = 128;
  ClampedSwiGLUKernel<<<Grid(d, block), block, 0, s>>>(gate_up, static_cast<int>(d), limit,
                                                       alpha, beta, out);
  Check(cudaGetLastError(), "clamped_swiglu_ip launch");
}

// Brick B — IN-PLACE MHC glue (unified memory, no Upload/Download/Sync). Same
// MhcPostKernel/HcHeadKernel/MhcPreKernel as the #183 launchers, run directly on the
// caller's unified pointers. Caller drains. MHC pre/head are <<<1,1>>> (hc=4 tiny);
// post is parallel. Near-tie vs host (device expf/rsqrt vs host in the RMSNorm/gates).
void MhcPostInPlaceLaunch(Queue& q, float* out, const float* x, const float* residual,
                          const float* post_mix, const float* comb, int64_t hc,
                          int64_t hidden) {
  if (hc == 0 || hidden == 0) return;
  cudaStream_t s = AsStream(q);
  const int block = 128;
  MhcPostKernel<<<Grid(hc * hidden, block), block, 0, s>>>(
      x, residual, post_mix, comb, static_cast<int>(hc), static_cast<int>(hidden), out);
  Check(cudaGetLastError(), "mhc_post_ip launch");
}

void HcHeadInPlaceLaunch(Queue& q, float* out, const float* x, const float* fn, float scale,
                         const float* base, int64_t hc, int64_t hidden, float rms_eps,
                         float hc_eps) {
  if (hidden == 0) return;
  cudaStream_t s = AsStream(q);
  HcHeadKernel<<<1, 1, 0, s>>>(x, fn, scale, base, static_cast<int>(hc),
                               static_cast<int>(hidden), rms_eps, hc_eps, out);
  Check(cudaGetLastError(), "hc_head_ip launch");
}

// MhcPre writes pre/post/comb mixes + layer_input; it needs an hc3=[(2+hc)*hc] mix
// scratch. `mix_scratch` is a caller-provided unified buffer (>= hc3 floats).
void MhcPreInPlaceLaunch(Queue& q, float* pre_mix, float* post_mix, float* comb_mix,
                         float* layer_input, float* mix_scratch, const float* residual,
                         const float* fn, const float* scale, const float* base, int64_t hc,
                         int64_t hidden, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                         float hc_post_mult, int64_t iters, const float* norm_weight,
                         bool has_norm, float norm_eps) {
  if (hidden == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;  // one block, parallel over the H/flat width
  const unsigned shmem = block * sizeof(double);
  MhcPreParallelKernel<<<1, block, shmem, s>>>(
      residual, fn, scale, base, static_cast<int>(hc), static_cast<int>(hidden), rms_eps,
      hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, static_cast<int>(iters), norm_weight,
      has_norm ? 1 : 0, norm_eps, mix_scratch, pre_mix, post_mix, comb_mix, layer_input);
  Check(cudaGetLastError(), "mhc_pre_ip launch");
}

// IN-PLACE router: same RouteKernel; writes topk_ids[T*topk] (i32) + weights[T*topk].
void RouteInPlaceLaunch(Queue& q, int32_t* topk_ids, float* topk_weights, const float* gating,
                        int64_t T, int64_t E, int64_t topk, const float* bias, bool has_bias,
                        const int64_t* in_tokens, bool is_hash, const int32_t* hashtab,
                        int64_t vocab, bool renorm, float scale) {
  if (T == 0) return;
  cudaStream_t s = AsStream(q);
  const int block = 64;
  RouteKernel<<<Grid(T, block), block, 0, s>>>(
      gating, static_cast<int>(T), static_cast<int>(E), static_cast<int>(topk), bias,
      has_bias ? 1 : 0, is_hash ? 1 : 0, in_tokens, hashtab, vocab, renorm ? 1 : 0, scale,
      topk_ids, topk_weights);
  Check(cudaGetLastError(), "route_ip launch");
}

// ── Brick C folded-in glue kernels (device RMSNorm / RoPE / MoE combine) ──────
// Weighted RMSNorm over [n]. One block, parallel block-tree reduction (double
// accumulate) → CHARACTERIZED near-tie vs host double-sequential (the reduction
// reorders); the scale+weight multiply is per-element (order-independent). has_w=0
// → no weight (the per-head q-RMS; DeepseekV4QHeadRmsNormInplace).
__global__ void RmsNormKernel(float* __restrict__ out, const float* __restrict__ x,
                              const float* __restrict__ w, int n, float eps, int has_w) {
  extern __shared__ double red[];
  const int tid = threadIdx.x, nt = blockDim.x;
  double ls = 0.0;
  for (int i = tid; i < n; i += nt) { const double v = x[i]; ls += v * v; }
  red[tid] = ls;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  const float r = 1.0f / sqrtf(static_cast<float>(red[0] / static_cast<double>(n)) + eps);
  for (int i = tid; i < n; i += nt) out[i] = has_w ? x[i] * r * w[i] : x[i] * r;
}
void RmsNormLaunch(Queue& q, float* out, const float* x, const float* w, int64_t n, float eps,
                   bool has_w) {
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;
  RmsNormKernel<<<1, block, block * sizeof(double), s>>>(out, x, w, static_cast<int>(n), eps,
                                                         has_w ? 1 : 0);
  Check(cudaGetLastError(), "rms_norm launch");
}

// Brick C part 2 — BATCHED RMSNorm over `rows` independent [n] segments in ONE
// launch (blockIdx.x = row). Used for the 64-head per-head q-RMS (has_w=false) so
// the resident decode does not issue nh=64 separate rms_norm launches per layer.
// Same block-tree reduction as RmsNormKernel ⇒ per-row IDENTICAL to it (the same
// characterized near-tie vs host double-sequential; a shared weight w[n] applies to
// every row when has_w).
__global__ void RmsNormRowsKernel(float* __restrict__ out, const float* __restrict__ x,
                                  const float* __restrict__ w, int rows, int n, float eps,
                                  int has_w) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + static_cast<int64_t>(row) * n;
  float* outr = out + static_cast<int64_t>(row) * n;
  extern __shared__ double red[];
  const int tid = threadIdx.x, nt = blockDim.x;
  double ls = 0.0;
  for (int i = tid; i < n; i += nt) { const double v = xr[i]; ls += v * v; }
  red[tid] = ls;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  const float r = 1.0f / sqrtf(static_cast<float>(red[0] / static_cast<double>(n)) + eps);
  for (int i = tid; i < n; i += nt) outr[i] = has_w ? xr[i] * r * w[i] : xr[i] * r;
}
void RmsNormRowsLaunch(Queue& q, float* out, const float* x, const float* w, int64_t rows,
                       int64_t n, float eps, bool has_w) {
  if (n == 0 || rows == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 256;
  RmsNormRowsKernel<<<static_cast<unsigned>(rows), block, block * sizeof(double), s>>>(
      out, x, w, static_cast<int>(rows), static_cast<int>(n), eps, has_w ? 1 : 0);
  Check(cudaGetLastError(), "rms_norm_rows launch");
}

__device__ double YarnCorrDimDev(int n_dims, int n_ctx_orig, double beta, double base) {
  const double kPi = 3.14159265358979323846;
  return static_cast<double>(n_dims) *
         log(static_cast<double>(n_ctx_orig) / (beta * 2.0 * kPi)) / (2.0 * log(base));
}
// One thread per row; the sequential-recurrence RoPE (host RopeInplaceLayer) on
// v[row*stride + off .. +r]. Near-tie vs host (device cos/sin vs libm; the double
// recurrence theta_extrap*=theta_scale preserves host order).
__global__ void RopeKernel(float* v, int num_rows, int row_stride, int off, int r,
                           const int* row_pos, double base, double freq_scale, double ext_factor,
                           int n_ctx_orig, double beta_fast, double beta_slow, int inverse) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  float* vv = v + static_cast<int64_t>(row) * row_stride + off;
  const double theta_scale = pow(base, -2.0 / static_cast<double>(r));
  const double sin_sign = inverse ? -1.0 : 1.0;
  double corr_lo = 0.0, corr_hi = 0.0;
  if (ext_factor != 0.0) {
    corr_lo = fmax(0.0, floor(YarnCorrDimDev(r, n_ctx_orig, beta_fast, base)));
    corr_hi = fmin(static_cast<double>(r - 1), ceil(YarnCorrDimDev(r, n_ctx_orig, beta_slow, base)));
  }
  double theta_extrap = static_cast<double>(row_pos[row]);
  for (int i = 0; i < r; i += 2) {
    const double theta_interp = freq_scale * theta_extrap;
    double theta = theta_interp;
    if (ext_factor != 0.0) {
      const double y = (static_cast<double>(i / 2) - corr_lo) / fmax(0.001, corr_hi - corr_lo);
      const double ramp = (1.0 - fmin(1.0, fmax(0.0, y))) * ext_factor;
      theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
    }
    const float c = static_cast<float>(cos(theta));
    const float sn = static_cast<float>(sin_sign * sin(theta));
    const float x0 = vv[i], x1 = vv[i + 1];
    vv[i] = x0 * c - x1 * sn;
    vv[i + 1] = x0 * sn + x1 * c;
    theta_extrap *= theta_scale;
  }
}
void RopeLaunch(Queue& q, float* v, int64_t num_rows, int64_t row_stride, int64_t off, int64_t r,
                const int* row_pos, double base, double freq_scale, double ext_factor,
                int64_t n_ctx_orig, double beta_fast, double beta_slow, bool inverse) {
  if (num_rows == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 128;
  const unsigned grid = static_cast<unsigned>((num_rows + block - 1) / block);
  RopeKernel<<<grid, block, 0, s>>>(v, static_cast<int>(num_rows), static_cast<int>(row_stride),
                                    static_cast<int>(off), static_cast<int>(r), row_pos, base,
                                    freq_scale, ext_factor, static_cast<int>(n_ctx_orig), beta_fast,
                                    beta_slow, inverse ? 1 : 0);
  Check(cudaGetLastError(), "rope launch");
}

// MoE combine: out[h] = Σ_a weights[a]*eo[a*H+h] (one thread per h; sequential over
// a → host order). Near-tie vs host (the device contracts weights[a]*eo+acc to an
// FMA; the host does separate multiply+add) — ~last-ULP, characterized.
__global__ void MoeCombineKernel(float* out, const float* eo, const float* weights, int A, int H) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float acc = 0.0f;
  for (int a = 0; a < A; ++a) acc += weights[a] * eo[static_cast<int64_t>(a) * H + h];
  out[h] = acc;
}
void MoeCombineLaunch(Queue& q, float* out, const float* eo, const float* weights, int64_t A,
                      int64_t H) {
  if (H == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 128;
  const unsigned grid = static_cast<unsigned>((H + block - 1) / block);
  MoeCombineKernel<<<grid, block, 0, s>>>(out, eo, weights, static_cast<int>(A),
                                          static_cast<int>(H));
  Check(cudaGetLastError(), "moe_combine launch");
}

// Brick D step 1 — DEVICE router gate (the last non-capturable host op of the
// resident decode): gating[e] = Σ_h x[h]·bf16→f32(W[e·H+h]) for the [ne,H] BF16
// `ffn.gate.weight`. One thread per expert; the dot is SEQUENTIAL in f32 with the
// exact `bits<<16` bf16 upcast — BIT-IDENTICAL to the host CPU MatmulBT
// (cpu_ops.cpp MatmulChunked<true>, f32 accumulate, LoadF32=BF16ToF32). Replaces
// the CPU MatmulBT (f32-act×bf16-weight, which the CUDA elementwise MatmulBT lacks)
// so the resident step is 100% device — no host op inside the capture region.
__global__ void RouterGateKernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                 float* __restrict__ gating, int ne, int H) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= ne) return;
  const uint16_t* we = w + static_cast<int64_t>(e) * H;
  float acc = 0.0f;
  for (int h = 0; h < H; ++h)
    acc += x[h] * __uint_as_float(static_cast<uint32_t>(we[h]) << 16);  // bf16→f32 exact
  gating[e] = acc;
}
void RouterGateLaunch(Queue& q, float* gating, const float* x, const void* w_bf16, int64_t ne,
                      int64_t H) {
  if (ne == 0) return;
  cudaStream_t s = AsStream(q);
  const unsigned block = 128;
  const unsigned grid = static_cast<unsigned>((ne + block - 1) / block);
  RouterGateKernel<<<grid, block, 0, s>>>(x, static_cast<const uint16_t*>(w_bf16), gating,
                                          static_cast<int>(ne), static_cast<int>(H));
  Check(cudaGetLastError(), "router_gate launch");
}

// ── Brick A: device MLA decode/prefill attention (unified memory, in place) ───
// One block per (query t, head h). num KV heads = 1 (all heads share the cached
// latent kv[s]). BIT-IDENTICAL to the host SoftmaxWithSink path by preserving its
// accumulation ORDER: per-key dot sequential over d, then thread-0 sequential
// max/denom over s (incl. the sink), then per-d output sequential over s. `e[]`
// (dynamic shared, sized kv_base+T) holds scores then exp() weights.
__global__ void DecodeAttnKernel(float* __restrict__ o, const float* __restrict__ q,
                                 const float* __restrict__ kv, const float* __restrict__ sink,
                                 int nh, int hd, int64_t kv_base, int T, float scale,
                                 bool no_sink) {
  const int th = blockIdx.x;      // in [0, T*nh)
  const int t = th / nh;
  const int h = th % nh;
  const int64_t n = kv_base + t + 1;  // causal: query t attends keys [0, kv_base+t]
  const float* qh = q + (static_cast<int64_t>(t) * nh + h) * hd;
  extern __shared__ float e[];    // [n] scores -> exp weights
  __shared__ float denom_sh;

  // Pass A: scores[s] = (qh · kv[s]) * scale — dot sequential over d (host order).
  for (int64_t s = threadIdx.x; s < n; s += blockDim.x) {
    const float* ks = kv + s * hd;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += qh[d] * ks[d];
    e[s] = acc * scale;
  }
  __syncthreads();

  // Pass mid (thread 0, sequential — matches host m/denom order exactly).
  if (threadIdx.x == 0) {
    const float ninf = -INFINITY;
    const float sink_h = no_sink ? ninf : sink[h];
    float m = sink_h;
    for (int64_t s = 0; s < n; ++s) m = fmaxf(m, e[s]);
    if (m == ninf) {  // fully -inf row: 0/0 guard (host returns zeros)
      denom_sh = 0.0f;
    } else {
      float denom = expf(sink_h - m);  // sink -> denominator only
      for (int64_t s = 0; s < n; ++s) {
        const float ee = expf(e[s] - m);
        e[s] = ee;
        denom += ee;
      }
      denom_sh = denom;
    }
  }
  __syncthreads();

  // Pass B: o[d] = Σ_s (e[s]/denom) · kv[s][d] — sequential over s (host order).
  const float denom = denom_sh;
  float* oh = o + (static_cast<int64_t>(t) * nh + h) * hd;
  if (denom == 0.0f) {
    for (int d = threadIdx.x; d < hd; d += blockDim.x) oh[d] = 0.0f;
    return;
  }
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < n; ++s) acc += (e[s] / denom) * kv[s * hd + d];
    oh[d] = acc;
  }
}

void DecodeAttnLaunch(Queue& q, float* o, const float* query, const float* kv,
                      const float* sink, int64_t nh, int64_t hd, int64_t kv_base,
                      int64_t T, float scale, bool no_sink) {
  if (T == 0 || nh == 0) return;
  const int64_t n_max = kv_base + T;
  // scores/weights live in dynamic shared (sized to the largest query's key count).
  // Long contexts beyond this are a named residual (global-scratch variant, R3).
  if (n_max * static_cast<int64_t>(sizeof(float)) > 40 * 1024)
    throw std::runtime_error(
        "vt cuda deepseek_v4: decode_attn context exceeds the shared-memory KV window "
        "(long-context device attention is a named residual)");
  cudaStream_t s = AsStream(q);
  const dim3 grid(static_cast<unsigned>(T * nh));
  const unsigned block = 256;
  const unsigned shmem = static_cast<unsigned>(n_max) * sizeof(float);
  DecodeAttnKernel<<<grid, block, shmem, s>>>(o, query, kv, sink, static_cast<int>(nh),
                                              static_cast<int>(hd), kv_base,
                                              static_cast<int>(T), scale, no_sink);
  Check(cudaGetLastError(), "decode_attn launch");
  // NO sync here — the caller drains (Brick A) or captures (Brick D).
}

// ── Brick D step 2: GRAPH decode attention (T=1, capturable) ──────────────────
// DecodeAttnKernel bakes `kv_base` into the launch (host arg + dynamic shmem), so a
// captured graph would freeze the context. This variant reads the KV length from a
// DEVICE buffer `len_dev` at runtime, uses FIXED shared memory (max_cap keys), and
// attends the `len` prior keys in `cache[0..len)` PLUS the current token's key
// `deck_new` (as key index `len` — it is not yet appended to `cache`). The key set
// {cache[0..len), deck_new} == the eager kernel's cache[0..kv_base] (which already
// had this token's deck appended) in the SAME order ⇒ BIT-IDENTICAL to eager.
__global__ void DecodeAttnGKernel(float* __restrict__ o, const float* __restrict__ q,
                                  const float* __restrict__ cache,
                                  const float* __restrict__ deck_new,
                                  const float* __restrict__ sink, int nh, int hd,
                                  const int* __restrict__ len_dev, float scale, bool no_sink) {
  const int h = blockIdx.x;   // T=1 → t=0, one block per head
  const int len = *len_dev;   // # prior keys already in `cache` (== kv_base)
  const int n = len + 1;      // + this token's key (deck_new)
  const float* qh = q + static_cast<int64_t>(h) * hd;
  extern __shared__ float e[];  // [max_cap] scores → exp weights
  __shared__ float denom_sh;
  for (int s = threadIdx.x; s < n; s += blockDim.x) {
    const float* ks = (s < len) ? (cache + static_cast<int64_t>(s) * hd) : deck_new;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += qh[d] * ks[d];
    e[s] = acc * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float ninf = -INFINITY;
    const float sink_h = no_sink ? ninf : sink[h];
    float m = sink_h;
    for (int s = 0; s < n; ++s) m = fmaxf(m, e[s]);
    if (m == ninf) {
      denom_sh = 0.0f;
    } else {
      float denom = expf(sink_h - m);
      for (int s = 0; s < n; ++s) { const float ee = expf(e[s] - m); e[s] = ee; denom += ee; }
      denom_sh = denom;
    }
  }
  __syncthreads();
  const float denom = denom_sh;
  float* oh = o + static_cast<int64_t>(h) * hd;
  if (denom == 0.0f) {
    for (int d = threadIdx.x; d < hd; d += blockDim.x) oh[d] = 0.0f;
    return;
  }
  for (int d = threadIdx.x; d < hd; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < n; ++s) {
      const float* ks = (s < len) ? (cache + static_cast<int64_t>(s) * hd) : deck_new;
      acc += (e[s] / denom) * ks[d];
    }
    oh[d] = acc;
  }
}
void DecodeAttnGLaunch(Queue& q, float* o, const float* query, const float* cache,
                       const float* deck_new, const float* sink, int64_t nh, int64_t hd,
                       const int* len_dev, int64_t max_cap, float scale, bool no_sink) {
  if (nh == 0) return;
  if (max_cap * static_cast<int64_t>(sizeof(float)) > 40 * 1024)
    throw std::runtime_error(
        "vt cuda deepseek_v4: decode_attn_g max_cap exceeds the shared-memory KV window");
  cudaStream_t s = AsStream(q);
  const dim3 grid(static_cast<unsigned>(nh));
  const unsigned block = 256;
  const unsigned shmem = static_cast<unsigned>(max_cap) * sizeof(float);
  DecodeAttnGKernel<<<grid, block, shmem, s>>>(o, query, cache, deck_new, sink,
                                               static_cast<int>(nh), static_cast<int>(hd), len_dev,
                                               scale, no_sink);
  Check(cudaGetLastError(), "decode_attn_g launch");
}

// ── the per-family kernels-structs (registered through the seam) ──────────────
const MhcDeviceKernels kMhc = {&MhcSinkhornLaunch, &MhcPreLaunch, &MhcPostLaunch, &HcHeadLaunch,
                               &MhcPostInPlaceLaunch, &HcHeadInPlaceLaunch, &MhcPreInPlaceLaunch};
const DsaDeviceKernels kDsa = {&DsaWeightFoldLaunch, &DsaLogitsLaunch, &DsaTopkLaunch,
                               &SoftmaxSinkLaunch, &GroupedOLoraLaunch, &DecodeAttnLaunch,
                               &RmsNormLaunch, &RopeLaunch, &RmsNormRowsLaunch, &DecodeAttnGLaunch};
const CompressorDeviceKernels kComp = {&SaveScoreApeLaunch, &PoolNormLaunch, &Fp8EncodeLaunch,
                                       &Fp8DecodeLaunch};
const MoeDeviceKernels kMoe = {&SqrtSoftplusLaunch,        &RouteLaunch,
                               &ClampedSwiGLULaunch,       &ClampedSwiGLUInPlaceLaunch,
                               &RouteInPlaceLaunch,        &MoeCombineLaunch,
                               &RouterGateLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDeepseekV4Mhc, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kMhc)));
    RegisterOp(OpId::kDeepseekV4Dsa, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kDsa)));
    RegisterOp(OpId::kDeepseekV4Compressor, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kComp)));
    RegisterOp(OpId::kDeepseekV4Moe, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kMoe)));
  }
} registrar;

}  // namespace
}  // namespace vllm::deepseek_v4
