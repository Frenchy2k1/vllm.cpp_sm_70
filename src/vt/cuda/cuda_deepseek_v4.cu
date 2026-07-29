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

// ── the per-family kernels-structs (registered through the seam) ──────────────
const MhcDeviceKernels kMhc = {&MhcSinkhornLaunch, &MhcPreLaunch, &MhcPostLaunch, &HcHeadLaunch};
const DsaDeviceKernels kDsa = {&DsaWeightFoldLaunch, &DsaLogitsLaunch, &DsaTopkLaunch,
                               &SoftmaxSinkLaunch, &GroupedOLoraLaunch, &DecodeAttnLaunch};
const CompressorDeviceKernels kComp = {&SaveScoreApeLaunch, &PoolNormLaunch, &Fp8EncodeLaunch,
                                       &Fp8DecodeLaunch};
const MoeDeviceKernels kMoe = {&SqrtSoftplusLaunch, &RouteLaunch, &ClampedSwiGLULaunch};

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
