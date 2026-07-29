// DeepSeek-V4-Flash W5 — Manifold / Markov Hyper-Connections (MHC) host
// reference. Ported 1:1 from the pinned vLLM eager reference @ 555967922:
//   - Sinkhorn + pre/post: vllm/model_executor/kernels/mhc/torch.py:56-106
//     (`mhc_pre_torch`, `mhc_post_torch`), byte-identical to
//     tilelang_kernels.py:126-153 (`_sinkhorn_fwd`) and SGLang v0.5.15
//     python/sglang/srt/layers/mhc.py:110-126.
//   - Head collapse: triton.py:108-140 (`hc_head_reduce_triton_kernel`) +
//     tilelang.py:720-748 (`hc_head_fused_kernel_tilelang`).
//   - The folded attn/ffn RMSNorm: tilelang.py mhc_pre_big_fuse_with_norm.
//   - Constants (hc_post_alpha=2.0, hc_pre_eps=hc_sinkhorn_eps=hc_eps):
//     vllm/models/deepseek_v4/nvidia/model.py:818-821,:886-894.
// See deepseek_v4_mhc.h for the full semantics + honest-scope note.
#include "vllm/model_executor/models/deepseek_v4_mhc.h"

#include <cmath>

namespace vllm::deepseek_v4 {

namespace {
inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

std::vector<float> MhcSinkhorn(const std::vector<float>& comb_logits, int64_t hc,
                               int64_t iters, float eps) {
  const size_t n = static_cast<size_t>(hc) * static_cast<size_t>(hc);
  std::vector<float> m(n, 0.0f);

  // Seed: row softmax over k (per row j), then +eps  (torch.py:78).
  for (int64_t j = 0; j < hc; ++j) {
    float rmax = comb_logits[j * hc];
    for (int64_t k = 1; k < hc; ++k) {
      rmax = std::max(rmax, comb_logits[j * hc + k]);
    }
    float rsum = 0.0f;
    for (int64_t k = 0; k < hc; ++k) {
      const float e = std::exp(comb_logits[j * hc + k] - rmax);
      m[j * hc + k] = e;
      rsum += e;
    }
    for (int64_t k = 0; k < hc; ++k) {
      m[j * hc + k] = m[j * hc + k] / rsum + eps;
    }
  }

  // First col-norm (dim=-2): divide each column k by (Σ_j M[j,k] + eps)
  // (torch.py:79).
  for (int64_t k = 0; k < hc; ++k) {
    float csum = 0.0f;
    for (int64_t j = 0; j < hc; ++j) csum += m[j * hc + k];
    const float denom = csum + eps;
    for (int64_t j = 0; j < hc; ++j) m[j * hc + k] /= denom;
  }

  // (iters-1)× [row-norm (dim=-1), col-norm (dim=-2)]  (torch.py:80-82).
  for (int64_t it = 0; it < iters - 1; ++it) {
    for (int64_t j = 0; j < hc; ++j) {
      float rsum = 0.0f;
      for (int64_t k = 0; k < hc; ++k) rsum += m[j * hc + k];
      const float denom = rsum + eps;
      for (int64_t k = 0; k < hc; ++k) m[j * hc + k] /= denom;
    }
    for (int64_t k = 0; k < hc; ++k) {
      float csum = 0.0f;
      for (int64_t j = 0; j < hc; ++j) csum += m[j * hc + k];
      const float denom = csum + eps;
      for (int64_t j = 0; j < hc; ++j) m[j * hc + k] /= denom;
    }
  }
  return m;
}

MhcPreResult MhcPre(const std::vector<float>& residual, const std::vector<float>& fn,
                    const std::vector<float>& scale, const std::vector<float>& base,
                    int64_t hc, int64_t hidden, float rms_eps, float hc_pre_eps,
                    float hc_sinkhorn_eps, float hc_post_mult_value,
                    int64_t sinkhorn_iters, const std::vector<float>& norm_weight,
                    float norm_eps) {
  const int64_t hc3 = (2 + hc) * hc;  // (2+hc_mult)*hc_mult
  const int64_t flat = hc * hidden;

  // mixes[o] = Σ residual · fn[o,:] ; sqrsum = Σ residual^2  (torch.py:62-64).
  std::vector<float> mixes(static_cast<size_t>(hc3), 0.0f);
  double sqrsum = 0.0;
  for (int64_t i = 0; i < hc; ++i) {
    for (int64_t h = 0; h < hidden; ++h) {
      const float r = residual[i * hidden + h];
      sqrsum += static_cast<double>(r) * static_cast<double>(r);
    }
  }
  for (int64_t o = 0; o < hc3; ++o) {
    float acc = 0.0f;
    const int64_t frow = o * flat;
    for (int64_t i = 0; i < hc; ++i) {
      for (int64_t h = 0; h < hidden; ++h) {
        acc += residual[i * hidden + h] * fn[frow + i * hidden + h];
      }
    }
    mixes[o] = acc;
  }
  // Folded weight-free RMSNorm of the projection (torch.py:65).
  const float rms = 1.0f / std::sqrt(static_cast<float>(sqrsum / static_cast<double>(flat)) + rms_eps);
  for (int64_t o = 0; o < hc3; ++o) mixes[o] *= rms;

  MhcPreResult out;
  out.pre_mix.resize(static_cast<size_t>(hc));
  out.post_mix.resize(static_cast<size_t>(hc));

  // pre = sigmoid(mixes[:hc]*scale0 + base[:hc]) + hc_pre_eps  (torch.py:67-68).
  for (int64_t j = 0; j < hc; ++j) {
    out.pre_mix[j] = Sigmoid(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
  }
  // post = sigmoid(mixes[hc:2hc]*scale1 + base[hc:2hc]) * hc_post_mult (torch.py:70-73).
  for (int64_t j = 0; j < hc; ++j) {
    out.post_mix[j] = Sigmoid(mixes[hc + j] * scale[1] + base[hc + j]) * hc_post_mult_value;
  }
  // comb logits = mixes[2hc:]*scale2 + base[2hc:] ; then Sinkhorn (torch.py:75-82).
  std::vector<float> comb_logits(static_cast<size_t>(hc) * static_cast<size_t>(hc));
  for (int64_t j = 0; j < hc; ++j) {
    for (int64_t k = 0; k < hc; ++k) {
      const int64_t idx = j * hc + k;
      comb_logits[idx] = mixes[2 * hc + idx] * scale[2] + base[2 * hc + idx];
    }
  }
  out.comb_mix = MhcSinkhorn(comb_logits, hc, sinkhorn_iters, hc_sinkhorn_eps);

  // layer_input[h] = Σ_j pre[j] · residual[j,h]  (torch.py:84-86).
  out.layer_input.assign(static_cast<size_t>(hidden), 0.0f);
  for (int64_t j = 0; j < hc; ++j) {
    const float p = out.pre_mix[j];
    for (int64_t h = 0; h < hidden; ++h) {
      out.layer_input[h] += p * residual[j * hidden + h];
    }
  }

  // Optional folded attn_norm/ffn_norm RMSNorm (tilelang.py mhc_pre_big_fuse_with_norm).
  if (!norm_weight.empty()) {
    double ss = 0.0;
    for (int64_t h = 0; h < hidden; ++h) {
      ss += static_cast<double>(out.layer_input[h]) * static_cast<double>(out.layer_input[h]);
    }
    const float r = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(hidden)) + norm_eps);
    for (int64_t h = 0; h < hidden; ++h) {
      out.layer_input[h] = out.layer_input[h] * r * norm_weight[h];
    }
  }
  return out;
}

std::vector<float> MhcPost(const std::vector<float>& x, const std::vector<float>& residual,
                           const std::vector<float>& post_layer_mix,
                           const std::vector<float>& comb_res_mix, int64_t hc,
                           int64_t hidden) {
  // new[j,h] = Σ_i comb[i,j]·residual[i,h] + post[j]·x[h]  (torch.py:100-106).
  std::vector<float> out(static_cast<size_t>(hc) * static_cast<size_t>(hidden), 0.0f);
  for (int64_t j = 0; j < hc; ++j) {
    const float p = post_layer_mix[j];
    for (int64_t h = 0; h < hidden; ++h) {
      float mixed = 0.0f;
      for (int64_t i = 0; i < hc; ++i) {
        mixed += comb_res_mix[i * hc + j] * residual[i * hidden + h];
      }
      out[j * hidden + h] = mixed + p * x[h];
    }
  }
  return out;
}

std::vector<float> HcHeadCollapse(const std::vector<float>& x, const std::vector<float>& fn,
                                  float scale, const std::vector<float>& base, int64_t hc,
                                  int64_t hidden, float rms_eps, float hc_eps) {
  const int64_t flat = hc * hidden;
  // Weight-free RMSNorm of the flattened streams (triton.py:118 `rmsnorm_nw`).
  double ss = 0.0;
  for (int64_t p = 0; p < flat; ++p) ss += static_cast<double>(x[p]) * static_cast<double>(x[p]);
  const float r = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(flat)) + rms_eps);

  // mixes[m] = Σ x_normed · fn[m,:] ; pre = sigmoid(mixes*scale + base) + hc_eps
  // (triton.py:119-120).
  std::vector<float> pre(static_cast<size_t>(hc));
  for (int64_t m = 0; m < hc; ++m) {
    float acc = 0.0f;
    const int64_t frow = m * flat;
    for (int64_t p = 0; p < flat; ++p) acc += (x[p] * r) * fn[frow + p];
    pre[m] = Sigmoid(acc * scale + base[m]) + hc_eps;
  }
  // out[h] = Σ_m pre[m]·x[m,h]  (triton.py:_hc_head_reduce_store_kernel:99).
  std::vector<float> out(static_cast<size_t>(hidden), 0.0f);
  for (int64_t m = 0; m < hc; ++m) {
    const float pm = pre[m];
    for (int64_t h = 0; h < hidden; ++h) out[h] += pm * x[m * hidden + h];
  }
  return out;
}

}  // namespace vllm::deepseek_v4
