// P4 — `ParakeetEncoder` / `ParakeetForCTC` MODULE gate against an INDEPENDENT
// in-test reference, in the discipline of tests/vt/test_ops_conv2d.cpp: the
// reference is written from the upstream definition, never from the production
// code, and it goes through NONE of the vt:: kernels the forward routes to.
//
// Upstream definitions transcribed here (transformers 5.3.0
// transformers/models/parakeet/modeling_parakeet.py — the module vLLM itself
// runs, parakeet.py:14,61):
//   ParakeetEncoderRelPositionalEncoding.forward :71-98
//   ParakeetEncoderFeedForward.forward           :109-113
//   ParakeetEncoderConvolutionModule.forward     :151-185
//   ParakeetEncoderAttention.forward             :291-346, _rel_shift :348-354
//   ParakeetEncoderSubsamplingConv2D.forward     :404-423
//   ParakeetEncoderBlock.forward                 :442-470
//   ParakeetEncoder.forward                      :576-640
//   ParakeetForCTC.forward :688-757 / .generate :759-811
//   tokenization_parakeet.py ParakeetTokenizer._decode :28-49
//
// The reference deliberately does the `_rel_shift` LITERALLY — pad one column,
// reinterpret as [2T, T], drop the first row, reinterpret as [T, 2T-1] — rather
// than the closed-form index vt::AttentionRelPos uses, so agreement is evidence
// for the closed form rather than a restatement of it.
//
// It also accumulates in DOUBLE while the forward is f32 throughout, so the
// comparison is tolerance-based (per-op byte-identity is already gated by the
// P1-P3 op suites); the measured margins are printed by every case.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vt/backend.h"

namespace {

using vllm::multimodal::ParakeetAttentionWeights;
using vllm::multimodal::ParakeetConvModuleWeights;
using vllm::multimodal::ParakeetCtcGreedyCollapse;
using vllm::multimodal::ParakeetEncoderConfig;
using vllm::multimodal::ParakeetEncoderForward;
using vllm::multimodal::ParakeetEncoderLayerWeights;
using vllm::multimodal::ParakeetEncoderWeights;
using vllm::multimodal::ParakeetFeedForwardWeights;
using vllm::multimodal::ParakeetForCTCForward;
using vllm::multimodal::ParakeetForCTCWeights;
using vllm::multimodal::ParakeetRelPositionalEncoding;
using vllm::multimodal::ParakeetSubsamplingOutputLength;
using vllm::multimodal::ParakeetSubsamplingWeights;

// Deterministic LCG, same shape as the P1-P3 op suites, so every case is
// reproducible with no seed corpus.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed)
      : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float Uniform(float scale = 1.0f) {
    return (static_cast<float>(Next() % 20001) / 10000.0f - 1.0f) * scale;
  }
  std::vector<float> Vec(int64_t n, float scale = 1.0f) {
    std::vector<float> v(static_cast<size_t>(n));
    for (float& x : v) x = Uniform(scale);
    return v;
  }
  // Strictly positive, for a BatchNorm running_var.
  std::vector<float> PosVec(int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (float& x : v) x = 0.5f + std::abs(Uniform());
    return v;
  }
};

struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
};

Err Compare(const std::vector<float>& got, const std::vector<double>& ref) {
  REQUIRE(got.size() == ref.size());
  double num = 0.0, den = 0.0, mx = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
    if (std::abs(d) > mx) mx = std::abs(d);
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

// ---------------------------------------------------------------------------
// The independent reference. Everything below is written from the upstream
// source, in double, with no call into vllm::multimodal except the type
// definitions.
// ---------------------------------------------------------------------------

using Mat = std::vector<double>;  // row-major

Mat ToDouble(const std::vector<float>& v) { return Mat(v.begin(), v.end()); }

// torch nn.Linear: out[m,n] = sum_k x[m,k]*w[n,k] + b[n].
Mat RefLinear(const Mat& x, int64_t m, int64_t k, const std::vector<float>& w,
              const std::vector<float>& b, int64_t n) {
  Mat out(static_cast<size_t>(m) * n, 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(j)]);
      for (int64_t kk = 0; kk < k; ++kk) {
        acc += x[static_cast<size_t>(i) * k + kk] *
               static_cast<double>(w[static_cast<size_t>(j) * k + kk]);
      }
      out[static_cast<size_t>(i) * n + j] = acc;
    }
  }
  return out;
}

// torch nn.LayerNorm over the last dim, BIASED variance.
Mat RefLayerNorm(const Mat& x, int64_t rows, int64_t dim, const std::vector<float>& w,
                 const std::vector<float>& b, double eps) {
  Mat out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const double* row = &x[static_cast<size_t>(r) * dim];
    double mean = 0.0;
    for (int64_t d = 0; d < dim; ++d) mean += row[d];
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t d = 0; d < dim; ++d) var += (row[d] - mean) * (row[d] - mean);
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t d = 0; d < dim; ++d) {
      out[static_cast<size_t>(r) * dim + d] =
          (row[d] - mean) * inv * static_cast<double>(w[static_cast<size_t>(d)]) +
          static_cast<double>(b[static_cast<size_t>(d)]);
    }
  }
  return out;
}

double RefSilu(double x) { return x / (1.0 + std::exp(-x)); }

// torch nn.Conv2d on [1, cin, h, w] with a [cout, cin/groups, kh, kw] weight.
Mat RefConv2d(const Mat& x, int64_t cin, int64_t h, int64_t wdim,
              const std::vector<float>& weight, const std::vector<float>& bias,
              int64_t cout, int64_t kh, int64_t kw, int64_t stride, int64_t pad,
              int64_t groups) {
  const int64_t hout = (h + 2 * pad - kh) / stride + 1;
  const int64_t wout = (wdim + 2 * pad - kw) / stride + 1;
  const int64_t cin_per_group = cin / groups;
  const int64_t cout_per_group = cout / groups;
  Mat out(static_cast<size_t>(cout) * hout * wout, 0.0);
  for (int64_t oc = 0; oc < cout; ++oc) {
    const int64_t g = oc / cout_per_group;
    for (int64_t oh = 0; oh < hout; ++oh) {
      for (int64_t ow = 0; ow < wout; ++ow) {
        double acc = 0.0;
        for (int64_t ic = 0; ic < cin_per_group; ++ic) {
          for (int64_t i = 0; i < kh; ++i) {
            for (int64_t j = 0; j < kw; ++j) {
              const int64_t ih = oh * stride + i - pad;
              const int64_t iw = ow * stride + j - pad;
              if (ih < 0 || ih >= h || iw < 0 || iw >= wdim) continue;
              const int64_t src_c = g * cin_per_group + ic;
              acc += x[(static_cast<size_t>(src_c) * h + ih) * wdim + iw] *
                     static_cast<double>(
                         weight[((static_cast<size_t>(oc) * cin_per_group + ic) * kh + i) *
                                    kw +
                                j]);
            }
          }
        }
        if (!bias.empty()) acc += static_cast<double>(bias[static_cast<size_t>(oc)]);
        out[(static_cast<size_t>(oc) * hout + oh) * wout + ow] = acc;
      }
    }
  }
  return out;
}

// torch nn.Conv1d(C, C, K, padding=(K-1)/2, groups=C) on [C, L].
Mat RefDepthwiseConv1d(const Mat& x, int64_t channels, int64_t length,
                       const std::vector<float>& weight, const std::vector<float>& bias,
                       int64_t kernel) {
  const int64_t pad = (kernel - 1) / 2;
  Mat out(static_cast<size_t>(channels) * length, 0.0);
  for (int64_t ch = 0; ch < channels; ++ch) {
    for (int64_t t = 0; t < length; ++t) {
      double acc = bias.empty() ? 0.0 : static_cast<double>(bias[static_cast<size_t>(ch)]);
      for (int64_t k = 0; k < kernel; ++k) {
        const int64_t src = t + k - pad;
        if (src < 0 || src >= length) continue;
        acc += x[static_cast<size_t>(ch) * length + src] *
               static_cast<double>(weight[static_cast<size_t>(ch) * kernel + k]);
      }
      out[static_cast<size_t>(ch) * length + t] = acc;
    }
  }
  return out;
}

// ParakeetEncoderRelPositionalEncoding.forward (:71-98).
Mat RefRelPositionalEncoding(int64_t seq_length, int64_t hidden) {
  const int64_t rows = 2 * seq_length - 1;
  Mat out(static_cast<size_t>(rows) * hidden);
  for (int64_t p = 0; p < rows; ++p) {
    const double pos = static_cast<double>(seq_length - 1 - p);
    for (int64_t i = 0; i < hidden / 2; ++i) {
      const double inv =
          std::pow(10000.0, -static_cast<double>(2 * i) / static_cast<double>(hidden));
      out[static_cast<size_t>(p) * hidden + 2 * i] = std::sin(pos * inv);
      out[static_cast<size_t>(p) * hidden + 2 * i + 1] = std::cos(pos * inv);
    }
  }
  return out;
}

// ParakeetEncoderAttention.forward (:291-346) with the LITERAL `_rel_shift`.
Mat RefAttention(const Mat& x, const Mat& pos_embed, int64_t rows, int64_t valid_rows,
                 const ParakeetAttentionWeights& w, const ParakeetEncoderConfig& cfg) {
  const int64_t heads = cfg.num_attention_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t p_rows = 2 * rows - 1;
  const double scaling = 1.0 / std::sqrt(static_cast<double>(hd));

  const Mat q = RefLinear(x, rows, cfg.hidden_size, w.q_w, w.q_b, heads * hd);
  const Mat k = RefLinear(x, rows, cfg.hidden_size, w.k_w, w.k_b, heads * hd);
  const Mat v = RefLinear(x, rows, cfg.hidden_size, w.v_w, w.v_b, heads * hd);
  const Mat rel =
      RefLinear(pos_embed, p_rows, cfg.hidden_size, w.relative_k_w, {}, heads * hd);

  Mat context(static_cast<size_t>(rows) * heads * hd, 0.0);
  for (int64_t h = 0; h < heads; ++h) {
    // (b)+(d): raw [rows, 2*rows-1].
    Mat raw(static_cast<size_t>(rows) * p_rows, 0.0);
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t p = 0; p < p_rows; ++p) {
        double acc = 0.0;
        for (int64_t d = 0; d < hd; ++d) {
          const double qi = q[(static_cast<size_t>(i) * heads + h) * hd + d] +
                            static_cast<double>(w.bias_v[static_cast<size_t>(h) * hd + d]);
          acc += qi * rel[(static_cast<size_t>(p) * heads + h) * hd + d];
        }
        raw[static_cast<size_t>(i) * p_rows + p] = acc;
      }
    }
    // `_rel_shift` (:348-354), performed literally.
    Mat padded(static_cast<size_t>(rows) * (p_rows + 1), 0.0);
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t p = 0; p < p_rows; ++p) {
        padded[static_cast<size_t>(i) * (p_rows + 1) + 1 + p] =
            raw[static_cast<size_t>(i) * p_rows + p];
      }
    }
    // view(-1, rows) is [2*rows, rows]; drop row 0; view(rows, p_rows).
    Mat shifted(static_cast<size_t>(rows) * p_rows, 0.0);
    for (size_t idx = 0; idx < shifted.size(); ++idx) {
      shifted[idx] = padded[idx + static_cast<size_t>(rows)];
    }

    for (int64_t i = 0; i < rows; ++i) {
      // A query row past the valid length has EVERY key masked upstream, and the
      // traced `sdpa` path returns exactly zero there (see the manifest note in
      // scripts/mm/p4_parakeet_oracle_dump.py).
      if (i >= valid_rows) continue;
      std::vector<double> scores(static_cast<size_t>(rows));
      double mx = -1e300;
      for (int64_t j = 0; j < rows; ++j) {
        double ac = 0.0;
        for (int64_t d = 0; d < hd; ++d) {
          const double qi = q[(static_cast<size_t>(i) * heads + h) * hd + d] +
                            static_cast<double>(w.bias_u[static_cast<size_t>(h) * hd + d]);
          ac += qi * k[(static_cast<size_t>(j) * heads + h) * hd + d];
        }
        // :324 matrix_bd *= scaling, then :246 attn = ac*scaling + matrix_bd.
        double s = ac * scaling + shifted[static_cast<size_t>(i) * p_rows + j] * scaling;
        if (j >= valid_rows) s = -1e300;  // :330 masked_fill(-inf)
        scores[static_cast<size_t>(j)] = s;
        if (s > mx) mx = s;
      }
      double denom = 0.0;
      for (double& s : scores) {
        s = std::exp(s - mx);
        denom += s;
      }
      for (int64_t d = 0; d < hd; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j < rows; ++j) {
          acc += scores[static_cast<size_t>(j)] / denom *
                 v[(static_cast<size_t>(j) * heads + h) * hd + d];
        }
        context[(static_cast<size_t>(i) * heads + h) * hd + d] = acc;
      }
    }
  }
  return RefLinear(context, rows, heads * hd, w.o_w, w.o_b, cfg.hidden_size);
}

// ParakeetEncoderConvolutionModule.forward (:151-185).
Mat RefConvModule(const Mat& x, int64_t rows, int64_t valid_rows,
                  const ParakeetConvModuleWeights& w, const ParakeetEncoderConfig& cfg) {
  const int64_t ch = cfg.hidden_size;
  const Mat gated = RefLinear(x, rows, ch, w.pointwise1_w, w.pointwise1_b, 2 * ch);
  // :169 F.glu over the channel axis.
  Mat h(static_cast<size_t>(rows) * ch, 0.0);
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t c = 0; c < ch; ++c) {
      const double a = gated[static_cast<size_t>(t) * 2 * ch + c];
      const double b = gated[static_cast<size_t>(t) * 2 * ch + ch + c];
      // :171-177 the padding mask, applied right after the GLU.
      h[static_cast<size_t>(t) * ch + c] =
          (t < valid_rows) ? a / (1.0 + std::exp(-b)) : 0.0;
    }
  }
  // transpose to [C, T] (:164, deferred), depthwise conv (:180).
  Mat ct(static_cast<size_t>(ch) * rows, 0.0);
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t c = 0; c < ch; ++c) {
      ct[static_cast<size_t>(c) * rows + t] = h[static_cast<size_t>(t) * ch + c];
    }
  }
  ct = RefDepthwiseConv1d(ct, ch, rows, w.depthwise_w, w.depthwise_b,
                          cfg.conv_kernel_size);
  // :181 BatchNorm1d in eval mode, :182 silu.
  for (int64_t c = 0; c < ch; ++c) {
    const double mean = w.norm_running_mean[static_cast<size_t>(c)];
    const double inv =
        1.0 / std::sqrt(static_cast<double>(w.norm_running_var[static_cast<size_t>(c)]) +
                        cfg.batch_norm_eps);
    for (int64_t t = 0; t < rows; ++t) {
      double& val = ct[static_cast<size_t>(c) * rows + t];
      val = (val - mean) * inv * static_cast<double>(w.norm_w[static_cast<size_t>(c)]) +
            static_cast<double>(w.norm_b[static_cast<size_t>(c)]);
      val = RefSilu(val);
    }
  }
  // :185 transpose back, :183 pointwise_conv2.
  Mat tc(static_cast<size_t>(rows) * ch, 0.0);
  for (int64_t c = 0; c < ch; ++c) {
    for (int64_t t = 0; t < rows; ++t) {
      tc[static_cast<size_t>(t) * ch + c] = ct[static_cast<size_t>(c) * rows + t];
    }
  }
  return RefLinear(tc, rows, ch, w.pointwise2_w, w.pointwise2_b, ch);
}

// ParakeetEncoderSubsamplingConv2D.forward (:404-423).
Mat RefSubsampling(const std::vector<float>& features, int64_t num_frames,
                   int64_t valid_frames, const ParakeetSubsamplingWeights& w,
                   const ParakeetEncoderConfig& cfg, int64_t* out_rows) {
  const int64_t k = cfg.subsampling_conv_kernel_size;
  const int64_t stride = cfg.subsampling_conv_stride;
  const int64_t pad = (k - 1) / 2;
  const int64_t ch = cfg.subsampling_conv_channels;

  auto strided = [&](int64_t len) { return (len + 2 * pad - k) / stride + 1; };
  auto mask = [&](Mat& t, int64_t h, int64_t wd, int64_t len) {
    for (int64_t c = 0; c < ch; ++c) {
      for (int64_t i = len; i < h; ++i) {
        for (int64_t j = 0; j < wd; ++j) t[(static_cast<size_t>(c) * h + i) * wd + j] = 0.0;
      }
    }
  };

  Mat x = ToDouble(features);
  int64_t h = num_frames, wd = cfg.num_mel_bins, len = valid_frames;
  x = RefConv2d(x, 1, h, wd, w.conv0_w, w.conv0_b, ch, k, k, stride, pad, 1);
  h = strided(h);
  wd = strided(wd);
  len = strided(len);
  mask(x, h, wd, len);
  for (double& v : x) v = std::max(0.0, v);  // :372 ReLU

  for (const auto& stage : w.stages) {
    x = RefConv2d(x, ch, h, wd, stage.depthwise_w, stage.depthwise_b, ch, k, k, stride,
                  pad, ch);
    h = strided(h);
    wd = strided(wd);
    len = strided(len);
    mask(x, h, wd, len);
    x = RefConv2d(x, ch, h, wd, stage.pointwise_w, stage.pointwise_b, ch, 1, 1, 1, 0, 1);
    mask(x, h, wd, len);  // a 1x1 conv does not change the length (:394, :402)
    for (double& v : x) v = std::max(0.0, v);  // :388 ReLU
  }

  // :420 transpose(1,2).reshape -> [T, C*F].
  Mat flat(static_cast<size_t>(h) * ch * wd, 0.0);
  for (int64_t t = 0; t < h; ++t) {
    for (int64_t c = 0; c < ch; ++c) {
      for (int64_t f = 0; f < wd; ++f) {
        flat[(static_cast<size_t>(t) * ch + c) * wd + f] =
            x[(static_cast<size_t>(c) * h + t) * wd + f];
      }
    }
  }
  *out_rows = h;
  return RefLinear(flat, h, ch * wd, w.linear_w, w.linear_b, cfg.hidden_size);
}

// ParakeetEncoder.forward (:576-640).
Mat RefEncoder(const std::vector<float>& features, int64_t num_frames,
               int64_t valid_frames, const ParakeetEncoderWeights& w,
               const ParakeetEncoderConfig& cfg, int64_t* out_rows) {
  int64_t rows = 0;
  Mat h = RefSubsampling(features, num_frames, valid_frames, w.subsampling, cfg, &rows);
  if (cfg.scale_input) {
    const double scale = std::sqrt(static_cast<double>(cfg.hidden_size));
    for (double& v : h) v *= scale;
  }
  const Mat pos = RefRelPositionalEncoding(rows, cfg.hidden_size);
  const int64_t valid_rows =
      std::min(ParakeetSubsamplingOutputLength(valid_frames, cfg), rows);

  for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
    const ParakeetEncoderLayerWeights& lw = w.layers[static_cast<size_t>(li)];
    const double eps = cfg.layer_norm_eps;

    auto feed_forward = [&](const Mat& in, const ParakeetFeedForwardWeights& ff) {
      Mat y = RefLinear(in, rows, cfg.hidden_size, ff.linear1_w, ff.linear1_b,
                        cfg.intermediate_size);
      for (double& v : y) v = RefSilu(v);
      return RefLinear(y, rows, cfg.intermediate_size, ff.linear2_w, ff.linear2_b,
                       cfg.hidden_size);
    };

    Mat n = RefLayerNorm(h, rows, cfg.hidden_size, lw.norm_feed_forward1_w,
                         lw.norm_feed_forward1_b, eps);
    Mat y = feed_forward(n, lw.feed_forward1);
    for (size_t i = 0; i < h.size(); ++i) h[i] += 0.5 * y[i];  // :451

    n = RefLayerNorm(h, rows, cfg.hidden_size, lw.norm_self_att_w, lw.norm_self_att_b,
                     eps);
    y = RefAttention(n, pos, rows, valid_rows, lw.self_attn, cfg);
    for (size_t i = 0; i < h.size(); ++i) h[i] += y[i];  // :460

    n = RefLayerNorm(h, rows, cfg.hidden_size, lw.norm_conv_w, lw.norm_conv_b, eps);
    y = RefConvModule(n, rows, valid_rows, lw.conv, cfg);
    for (size_t i = 0; i < h.size(); ++i) h[i] += y[i];  // :463

    n = RefLayerNorm(h, rows, cfg.hidden_size, lw.norm_feed_forward2_w,
                     lw.norm_feed_forward2_b, eps);
    y = feed_forward(n, lw.feed_forward2);
    for (size_t i = 0; i < h.size(); ++i) h[i] += 0.5 * y[i];  // :466

    h = RefLayerNorm(h, rows, cfg.hidden_size, lw.norm_out_w, lw.norm_out_b, eps);  // :468
  }
  *out_rows = rows;
  return h;
}

// ---------------------------------------------------------------------------
// A small but structurally complete model: two subsampling stages (so the
// depthwise+pointwise stage runs), multi-head attention with head_dim > 1, an
// odd conformer kernel, and two encoder blocks.
// ---------------------------------------------------------------------------
ParakeetEncoderConfig TinyConfig() {
  ParakeetEncoderConfig cfg;
  cfg.hidden_size = 24;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 3;
  cfg.num_key_value_heads = 3;
  cfg.intermediate_size = 40;
  cfg.conv_kernel_size = 5;
  cfg.subsampling_factor = 4;
  cfg.subsampling_conv_channels = 6;
  cfg.num_mel_bins = 12;
  cfg.subsampling_conv_kernel_size = 3;
  cfg.subsampling_conv_stride = 2;
  cfg.max_position_embeddings = 512;
  cfg.scale_input = true;
  cfg.vocab_size = 11;
  cfg.pad_token_id = 10;
  return cfg;
}

ParakeetForCTCWeights RandomWeights(const ParakeetEncoderConfig& cfg, Rng& rng) {
  const int64_t hidden = cfg.hidden_size;
  const int64_t inter = cfg.intermediate_size;
  const int64_t heads = cfg.num_attention_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t ch = cfg.subsampling_conv_channels;
  const int64_t k = cfg.subsampling_conv_kernel_size;

  ParakeetForCTCWeights w;
  ParakeetSubsamplingWeights& sub = w.encoder.subsampling;
  sub.conv0_w = rng.Vec(ch * 1 * k * k, 0.4f);
  sub.conv0_b = rng.Vec(ch, 0.2f);
  for (int64_t i = 0; i + 1 < cfg.num_subsampling_layers(); ++i) {
    ParakeetSubsamplingWeights::Stage stage;
    stage.depthwise_w = rng.Vec(ch * 1 * k * k, 0.4f);
    stage.depthwise_b = rng.Vec(ch, 0.2f);
    stage.pointwise_w = rng.Vec(ch * ch, 0.4f);
    stage.pointwise_b = rng.Vec(ch, 0.2f);
    sub.stages.push_back(std::move(stage));
  }
  sub.linear_w = rng.Vec(hidden * ch * cfg.subsampling_out_freq(), 0.3f);
  sub.linear_b = rng.Vec(hidden, 0.2f);

  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    ParakeetEncoderLayerWeights lw;
    for (ParakeetFeedForwardWeights* ff : {&lw.feed_forward1, &lw.feed_forward2}) {
      ff->linear1_w = rng.Vec(inter * hidden, 0.3f);
      ff->linear1_b = rng.Vec(inter, 0.2f);
      ff->linear2_w = rng.Vec(hidden * inter, 0.3f);
      ff->linear2_b = rng.Vec(hidden, 0.2f);
    }
    lw.self_attn.q_w = rng.Vec(heads * hd * hidden, 0.3f);
    lw.self_attn.q_b = rng.Vec(heads * hd, 0.2f);
    lw.self_attn.k_w = rng.Vec(heads * hd * hidden, 0.3f);
    lw.self_attn.k_b = rng.Vec(heads * hd, 0.2f);
    lw.self_attn.v_w = rng.Vec(heads * hd * hidden, 0.3f);
    lw.self_attn.v_b = rng.Vec(heads * hd, 0.2f);
    lw.self_attn.o_w = rng.Vec(hidden * heads * hd, 0.3f);
    lw.self_attn.o_b = rng.Vec(hidden, 0.2f);
    lw.self_attn.relative_k_w = rng.Vec(heads * hd * hidden, 0.3f);
    lw.self_attn.bias_u = rng.Vec(heads * hd, 0.3f);
    lw.self_attn.bias_v = rng.Vec(heads * hd, 0.3f);

    lw.conv.pointwise1_w = rng.Vec(2 * hidden * hidden, 0.3f);
    lw.conv.pointwise1_b = rng.Vec(2 * hidden, 0.2f);
    lw.conv.depthwise_w = rng.Vec(hidden * cfg.conv_kernel_size, 0.4f);
    lw.conv.depthwise_b = rng.Vec(hidden, 0.2f);
    lw.conv.norm_w = rng.Vec(hidden, 0.3f);
    lw.conv.norm_b = rng.Vec(hidden, 0.3f);
    lw.conv.norm_running_mean = rng.Vec(hidden, 0.4f);
    lw.conv.norm_running_var = rng.PosVec(hidden);
    lw.conv.pointwise2_w = rng.Vec(hidden * hidden, 0.3f);
    lw.conv.pointwise2_b = rng.Vec(hidden, 0.2f);

    for (std::vector<float>* v : {&lw.norm_feed_forward1_w, &lw.norm_self_att_w,
                                  &lw.norm_conv_w, &lw.norm_feed_forward2_w,
                                  &lw.norm_out_w}) {
      *v = rng.Vec(hidden, 0.3f);
      for (float& x : *v) x += 1.0f;
    }
    for (std::vector<float>* v : {&lw.norm_feed_forward1_b, &lw.norm_self_att_b,
                                  &lw.norm_conv_b, &lw.norm_feed_forward2_b,
                                  &lw.norm_out_b}) {
      *v = rng.Vec(hidden, 0.3f);
    }
    w.encoder.layers.push_back(std::move(lw));
  }

  w.ctc_head_w = rng.Vec(cfg.vocab_size * hidden, 0.5f);
  w.ctc_head_b = rng.Vec(cfg.vocab_size, 0.3f);
  return w;
}

}  // namespace

// `_get_subsampling_output_length` (:515-530), against the per-layer conv output
// formula computed independently here.
TEST_CASE("parakeet_subsampling_output_length") {
  ParakeetEncoderConfig cfg = TinyConfig();
  for (int64_t frames = 1; frames <= 200; ++frames) {
    int64_t expect = frames;
    for (int64_t i = 0; i < cfg.num_subsampling_layers(); ++i) {
      const int64_t pad = (cfg.subsampling_conv_kernel_size - 1) / 2;
      expect = (expect + 2 * pad - cfg.subsampling_conv_kernel_size) /
                   cfg.subsampling_conv_stride +
               1;
    }
    CHECK(ParakeetSubsamplingOutputLength(frames, cfg) == expect);
  }
  // The published checkpoints: subsampling_factor 8 (three stages), k=3, s=2.
  ParakeetEncoderConfig big;
  CHECK(big.num_subsampling_layers() == 3);
  CHECK(big.subsampling_out_freq() == 10);  // 80 mel bins / 2^3
  CHECK(ParakeetSubsamplingOutputLength(3000, big) == 375);
}

TEST_CASE("parakeet_rel_positional_encoding_matches_reference") {
  for (int64_t seq : {1, 2, 7, 33}) {
    for (int64_t hidden : {8, 24, 64}) {
      const std::vector<float> got = ParakeetRelPositionalEncoding(seq, hidden);
      const std::vector<double> ref = RefRelPositionalEncoding(seq, hidden);
      const Err e = Compare(got, ref);
      CHECK(e.max_abs < 1e-6);
    }
  }
  // Row seq-1 is position 0, so it is sin(0)=0 / cos(0)=1, interleaved.
  const std::vector<float> t = ParakeetRelPositionalEncoding(4, 8);
  for (int i = 0; i < 4; ++i) {
    CHECK(t[static_cast<size_t>(3) * 8 + 2 * i] == doctest::Approx(0.0));
    CHECK(t[static_cast<size_t>(3) * 8 + 2 * i + 1] == doctest::Approx(1.0));
  }
}

// tokenization_parakeet.py:38-42. Group FIRST, drop the blank SECOND — the order
// is what lets a blank separate two genuinely repeated tokens.
TEST_CASE("parakeet_ctc_greedy_collapse") {
  const int32_t blank = 10;
  CHECK(ParakeetCtcGreedyCollapse({}, blank) == std::vector<int32_t>{});
  CHECK(ParakeetCtcGreedyCollapse({10, 10, 10}, blank) == std::vector<int32_t>{});
  CHECK(ParakeetCtcGreedyCollapse({1, 1, 2, 2, 2, 3}, blank) ==
        std::vector<int32_t>{1, 2, 3});
  // A blank BETWEEN two equal tokens keeps them apart; without it they merge.
  CHECK(ParakeetCtcGreedyCollapse({4, 10, 4}, blank) == std::vector<int32_t>{4, 4});
  CHECK(ParakeetCtcGreedyCollapse({4, 4}, blank) == std::vector<int32_t>{4});
  CHECK(ParakeetCtcGreedyCollapse({10, 5, 10, 10, 5, 5, 10, 6}, blank) ==
        std::vector<int32_t>{5, 5, 6});
  // Against an independently written reference over a pseudo-random corpus.
  Rng rng(7);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<int32_t> ids(static_cast<size_t>(1 + rng.Next() % 40));
    for (int32_t& v : ids) v = static_cast<int32_t>(rng.Next() % 12);
    std::vector<int32_t> grouped;
    for (int32_t v : ids) {
      if (grouped.empty() || grouped.back() != v) grouped.push_back(v);
    }
    std::vector<int32_t> expect;
    for (int32_t v : grouped) {
      if (v != blank) expect.push_back(v);
    }
    CHECK(ParakeetCtcGreedyCollapse(ids, blank) == expect);
  }
}

TEST_CASE("parakeet_encoder_forward_matches_in_test_reference") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const ParakeetEncoderConfig cfg = TinyConfig();
  Rng rng(20260806);
  const ParakeetForCTCWeights w = RandomWeights(cfg, rng);

  // Two shapes: an UNPADDED clip and a padded one, so the subsampling mask, the
  // attention key mask and the convolution-module mask are all exercised.
  struct Case {
    int64_t frames;
    int64_t valid;
  };
  for (const Case& tc : {Case{21, 21}, Case{21, 13}, Case{40, 27}, Case{9, 4}}) {
    CAPTURE(tc.frames);
    CAPTURE(tc.valid);
    const std::vector<float> features =
        rng.Vec(tc.frames * cfg.num_mel_bins, 1.0f);

    int64_t ref_rows = 0;
    const std::vector<double> ref =
        RefEncoder(features, tc.frames, tc.valid, w.encoder, cfg, &ref_rows);

    int64_t valid_rows = 0;
    const std::vector<float> got =
        ParakeetEncoderForward(features, tc.frames, tc.valid, w.encoder, cfg, cpu,
                               &valid_rows, nullptr);
    CHECK(static_cast<int64_t>(got.size()) == ref_rows * cfg.hidden_size);
    CHECK(valid_rows == ParakeetSubsamplingOutputLength(tc.valid, cfg));

    const Err e = Compare(got, ref);
    MESSAGE("encoder frames=", tc.frames, " valid=", tc.valid, ": rel_l2=", e.rel_l2,
            " max_abs=", e.max_abs);
    CHECK(e.rel_l2 < 1e-5);
    CHECK(e.max_abs < 1e-4);
  }
}

TEST_CASE("parakeet_ctc_head_and_greedy_match_in_test_reference") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const ParakeetEncoderConfig cfg = TinyConfig();
  Rng rng(20260807);
  const ParakeetForCTCWeights w = RandomWeights(cfg, rng);
  const int64_t frames = 33, valid = 25;
  const std::vector<float> features = rng.Vec(frames * cfg.num_mel_bins, 1.0f);

  int64_t ref_rows = 0;
  const std::vector<double> hidden =
      RefEncoder(features, frames, valid, w.encoder, cfg, &ref_rows);
  // :682/:722 — the kernel-1 Conv1d head is a Linear over the [T, hidden] view.
  const std::vector<double> ref_logits = RefLinear(hidden, ref_rows, cfg.hidden_size,
                                                   w.ctc_head_w, w.ctc_head_b,
                                                   cfg.vocab_size);
  const int64_t ref_valid_rows =
      std::min(ParakeetSubsamplingOutputLength(valid, cfg), ref_rows);
  std::vector<int32_t> ref_ids(static_cast<size_t>(ref_rows), cfg.pad_token_id);
  for (int64_t t = 0; t < ref_valid_rows; ++t) {
    int64_t best = 0;
    for (int64_t v = 1; v < cfg.vocab_size; ++v) {
      if (ref_logits[static_cast<size_t>(t) * cfg.vocab_size + v] >
          ref_logits[static_cast<size_t>(t) * cfg.vocab_size + best]) {
        best = v;
      }
    }
    ref_ids[static_cast<size_t>(t)] = static_cast<int32_t>(best);
  }

  const auto out = ParakeetForCTCForward(features, frames, valid, w, cfg, cpu);
  CHECK(out.num_output_frames == ref_rows);
  CHECK(out.valid_output_frames == ref_valid_rows);
  const Err e = Compare(out.logits, ref_logits);
  MESSAGE("ctc logits: rel_l2=", e.rel_l2, " max_abs=", e.max_abs);
  CHECK(e.rel_l2 < 1e-5);
  // The DISCRETE outputs must be exact, not merely close.
  CHECK(out.greedy_ids == ref_ids);
  CHECK(out.token_ids == ParakeetCtcGreedyCollapse(ref_ids, cfg.pad_token_id));
  // Padding frames really are blank, so they contribute nothing to the collapse.
  for (int64_t t = ref_valid_rows; t < ref_rows; ++t) {
    CHECK(out.greedy_ids[static_cast<size_t>(t)] == cfg.pad_token_id);
  }
}

// `torch.argmax` (:796) returns the index of the FIRST maximal value, so a
// tie must resolve to the LOWEST id. Random logits never tie, so the tie is
// forced: a zeroed CTC head makes every logit exactly equal, and the whole valid
// prefix must then decode to id 0 — which id 0 is, deliberately, not the blank.
TEST_CASE("parakeet_ctc_argmax_breaks_ties_toward_the_lowest_id") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const ParakeetEncoderConfig cfg = TinyConfig();
  Rng rng(11);
  ParakeetForCTCWeights w = RandomWeights(cfg, rng);
  std::fill(w.ctc_head_w.begin(), w.ctc_head_w.end(), 0.0f);
  std::fill(w.ctc_head_b.begin(), w.ctc_head_b.end(), 0.0f);
  const int64_t frames = 21, valid = 15;
  const std::vector<float> features = rng.Vec(frames * cfg.num_mel_bins, 1.0f);

  const auto out = ParakeetForCTCForward(features, frames, valid, w, cfg, cpu);
  REQUIRE(out.valid_output_frames > 0);
  REQUIRE(cfg.pad_token_id != 0);
  for (const float v : out.logits) CHECK(v == 0.0f);
  for (int64_t t = 0; t < out.valid_output_frames; ++t) {
    CHECK(out.greedy_ids[static_cast<size_t>(t)] == 0);
  }
  CHECK(out.token_ids == std::vector<int32_t>{0});
}

// A config the port does not cover must FAIL, not silently run the wrong
// activation: `hidden_act` drives both the feed-forward (:105) and the
// convolution module (:129), and every published checkpoint uses silu.
TEST_CASE("parakeet_encoder_rejects_an_unported_activation") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  ParakeetEncoderConfig cfg = TinyConfig();
  Rng rng(3);
  const ParakeetForCTCWeights w = RandomWeights(cfg, rng);
  const std::vector<float> features = rng.Vec(21 * cfg.num_mel_bins, 1.0f);
  cfg.hidden_act = "gelu";
  CHECK_THROWS(ParakeetEncoderForward(features, 21, 21, w.encoder, cfg, cpu));
}
