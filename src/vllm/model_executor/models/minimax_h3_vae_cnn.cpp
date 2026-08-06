// MiniMax-H3 video-VAE 3D-CNN ENCODER primitives — causal Conv3d, GroupNorm, and
// the ResnetBlock3D that is the repeated unit of the encoder stack.
//
// The video VAE's DECODER is a ViT (already ported); its ENCODER is this 3D CNN.
// The encoder is needed for image/video CONDITIONING (fl2va keyframes, ref2va
// references) — NOT for producing output frames, so a t2va path does not use it.
//
// Two details that are easy to get wrong and are gated here:
//   * the convolution is CAUSAL IN TIME: all temporal padding goes on the LEFT
//     (`padding[0] * 2` frames), none on the right, so a frame never sees the
//     future. Spatial padding uses the checkpoint's `reflect` mode.
//   * GroupNorm is 32 groups at eps 1e-6 over (C/32, T, H, W) — the statistics
//     span TIME as well as space, which is why a per-frame normalization would
//     silently differ.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

int64_t ReflectIndex(int64_t index, int64_t size) {
  // torch's "reflect" padding excludes the edge sample: [a b c] -> b a b c b.
  if (size == 1) return 0;
  while (index < 0 || index >= size) {
    if (index < 0) index = -index;
    if (index >= size) index = 2 * (size - 1) - index;
  }
  return index;
}

}  // namespace

// The same reflect rule the conv uses, needed by Downsample3D's asymmetric pad.
int64_t ReflectIndexPublic(int64_t index, int64_t size) { return ReflectIndex(index, size); }

// A causal 3D convolution with `reflect` spatial padding (conv.py:12-88).
// Input/output are [C, T, H, W]; weight is [out, in, kt, kh, kw].
std::vector<float> MiniMaxH3CausalConv3d(const std::vector<float>& in, const MiniMaxH3Conv3dSpec& spec,
                                         const std::vector<float>& weight,
                                         const std::vector<float>* bias) {
  const int64_t ci = spec.in_channels, co = spec.out_channels;
  const int64_t t = spec.t, h = spec.h, w = spec.w;
  const int64_t kt = spec.kernel_t, kh = spec.kernel_h, kw = spec.kernel_w;
  const int64_t pt = spec.pad_t, ph = spec.pad_h, pw = spec.pad_w;
  VT_CHECK(static_cast<int64_t>(in.size()) == ci * t * h * w,
           "minimax_h3 conv3d: input size does not match [C, T, H, W]");
  VT_CHECK(static_cast<int64_t>(weight.size()) == co * ci * kt * kh * kw,
           "minimax_h3 conv3d: weight size does not match the kernel");

  // CAUSAL: 2*pad_t frames on the LEFT, none on the right (conv.py:40-51).
  const int64_t pad_left_t = spec.causal ? pt * 2 : pt;
  const int64_t pad_right_t = spec.causal ? 0 : pt;
  const int64_t padded_t = t + pad_left_t + pad_right_t;
  const int64_t padded_h = h + 2 * ph;
  const int64_t padded_w = w + 2 * pw;

  std::vector<float> padded(static_cast<size_t>(ci * padded_t * padded_h * padded_w), 0.0f);
  for (int64_t c = 0; c < ci; ++c) {
    for (int64_t pti = 0; pti < padded_t; ++pti) {
      const int64_t src_t = pti - pad_left_t;
      // Temporal padding is CONSTANT (zeros) in causal mode.
      if (src_t < 0 || src_t >= t) continue;
      for (int64_t phi = 0; phi < padded_h; ++phi) {
        const int64_t src_h = ReflectIndex(phi - ph, h);
        for (int64_t pwi = 0; pwi < padded_w; ++pwi) {
          const int64_t src_w = ReflectIndex(pwi - pw, w);
          padded[static_cast<size_t>(((c * padded_t + pti) * padded_h + phi) * padded_w + pwi)] =
              in[static_cast<size_t>(((c * t + src_t) * h + src_h) * w + src_w)];
        }
      }
    }
  }

  const int64_t out_t = (padded_t - kt) / spec.stride_t + 1;
  const int64_t out_h = (padded_h - kh) / spec.stride_h + 1;
  const int64_t out_w = (padded_w - kw) / spec.stride_w + 1;
  VT_CHECK(out_t > 0 && out_h > 0 && out_w > 0, "minimax_h3 conv3d: empty output");
  std::vector<float> out(static_cast<size_t>(co * out_t * out_h * out_w));
  for (int64_t oc = 0; oc < co; ++oc) {
    for (int64_t ot = 0; ot < out_t; ++ot) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          double acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0;
          for (int64_t ic = 0; ic < ci; ++ic) {
            for (int64_t a = 0; a < kt; ++a) {
              for (int64_t b = 0; b < kh; ++b) {
                for (int64_t c2 = 0; c2 < kw; ++c2) {
                  const double v = padded[static_cast<size_t>(
                      ((ic * padded_t + ot * spec.stride_t + a) * padded_h + oh * spec.stride_h + b) *
                          padded_w +
                      ow * spec.stride_w + c2)];
                  const double k = weight[static_cast<size_t>(
                      (((oc * ci + ic) * kt + a) * kh + b) * kw + c2)];
                  acc += v * k;
                }
              }
            }
          }
          out[static_cast<size_t>(((oc * out_t + ot) * out_h + oh) * out_w + ow)] =
              static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

// GroupNorm over [C, T, H, W]: statistics span the group's channels AND all of
// time and space (norm.py:342-357; 32 groups, eps 1e-6, affine).
void MiniMaxH3GroupNorm3d(std::vector<float>& x, int64_t channels, int64_t spatial,
                          int64_t num_groups, const std::vector<float>& weight,
                          const std::vector<float>& bias, double eps) {
  VT_CHECK(channels % num_groups == 0,
           "minimax_h3 groupnorm: channels must be divisible by num_groups");
  const int64_t per_group = channels / num_groups;
  for (int64_t g = 0; g < num_groups; ++g) {
    const int64_t begin = g * per_group;
    double mean = 0.0;
    const int64_t count = per_group * spatial;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) mean += x[static_cast<size_t>(c * spatial + i)];
    }
    mean /= static_cast<double>(count);
    double var = 0.0;
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = x[static_cast<size_t>(c * spatial + i)] - mean;
        var += d * d;
      }
    }
    var /= static_cast<double>(count);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = begin; c < begin + per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double normed = (x[static_cast<size_t>(c * spatial + i)] - mean) * inv;
        x[static_cast<size_t>(c * spatial + i)] = static_cast<float>(
            normed * weight[static_cast<size_t>(c)] + bias[static_cast<size_t>(c)]);
      }
    }
  }
}

// ResnetBlock3D (vae_cnn.py:83-171): norm -> SiLU -> conv -> norm -> SiLU -> conv,
// plus a 1x1x1 `nin_shortcut` when the channel count changes.
std::vector<float> MiniMaxH3ResnetBlock3dForward(const MiniMaxH3ResnetBlock3dConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::string& prefix,
                                                 const std::vector<float>& x) {
  const int64_t ci = config.in_channels, co = config.out_channels;
  const int64_t t = config.t, h = config.h, w = config.w;
  const int64_t spatial = t * h * w;
  VT_CHECK(static_cast<int64_t>(x.size()) == ci * spatial,
           "minimax_h3 resnet3d: input size does not match [C, T, H, W]");

  auto conv_spec = [&](int64_t in_ch, int64_t out_ch, int64_t kernel, int64_t pad) {
    MiniMaxH3Conv3dSpec spec;
    spec.in_channels = in_ch;
    spec.out_channels = out_ch;
    spec.t = t;
    spec.h = h;
    spec.w = w;
    spec.kernel_t = spec.kernel_h = spec.kernel_w = kernel;
    spec.pad_t = spec.pad_h = spec.pad_w = pad;
    spec.causal = true;
    return spec;
  };

  std::vector<float> hbuf = x;
  MiniMaxH3GroupNorm3d(hbuf, ci, spatial, config.num_groups, weights.Get(prefix + ".norm1.weight"),
                       weights.Get(prefix + ".norm1.bias"), config.eps);
  for (float& v : hbuf) v = v / (1.0f + std::exp(-v));  // SiLU
  hbuf = MiniMaxH3CausalConv3d(hbuf, conv_spec(ci, co, 3, 1), weights.Get(prefix + ".conv1.weight"),
                               &weights.Get(prefix + ".conv1.bias"));

  MiniMaxH3GroupNorm3d(hbuf, co, spatial, config.num_groups, weights.Get(prefix + ".norm2.weight"),
                       weights.Get(prefix + ".norm2.bias"), config.eps);
  for (float& v : hbuf) v = v / (1.0f + std::exp(-v));
  hbuf = MiniMaxH3CausalConv3d(hbuf, conv_spec(co, co, 3, 1), weights.Get(prefix + ".conv2.weight"),
                               &weights.Get(prefix + ".conv2.bias"));

  std::vector<float> residual = x;
  if (ci != co) {
    // A 1x1x1 conv with NO padding — the causal path is a no-op at pad 0.
    residual = MiniMaxH3CausalConv3d(x, conv_spec(ci, co, 1, 0),
                                     weights.Get(prefix + ".nin_shortcut.weight"),
                                     &weights.Get(prefix + ".nin_shortcut.bias"));
  }
  VT_CHECK(residual.size() == hbuf.size(), "minimax_h3 resnet3d: residual and main-branch shapes must match");
  for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] += residual[i];
  return hbuf;
}

// Downsample3D (vae_cnn.py:34-81). When the spatial stride is 2 the input is
// first padded by ONE on the RIGHT of W and the BOTTOM of H (an ASYMMETRIC pad,
// `F.pad(x, (0,1,0,1,0,0))`), and only then convolved with padding (1, 0, 0).
// Padding symmetrically instead shifts the whole sampling lattice by half a pixel.
std::vector<float> MiniMaxH3Downsample3d(const std::vector<float>& x,
                                         const MiniMaxH3Downsample3dConfig& config,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& bias) {
  VT_CHECK(config.time_stride == 1 || config.time_stride == 2,
           "minimax_h3 downsample3d: time_stride must be 1 or 2");
  VT_CHECK(config.space_stride >= 1 && config.space_stride <= 3,
           "minimax_h3 downsample3d: space_stride must be 1, 2 or 3");

  std::vector<float> input = x;
  int64_t h = config.h, w = config.w;
  if (config.space_stride == 2) {
    const int64_t nh = h + 1, nw = w + 1;
    std::vector<float> padded(static_cast<size_t>(config.in_channels * config.t * nh * nw), 0.0f);
    for (int64_t c = 0; c < config.in_channels; ++c) {
      for (int64_t ti = 0; ti < config.t; ++ti) {
        for (int64_t hi = 0; hi < nh; ++hi) {
          for (int64_t wi = 0; wi < nw; ++wi) {
            // `reflect` on the added right/bottom edge, matching pad_mode.
            const int64_t sh = hi < h ? hi : ReflectIndexPublic(hi, h);
            const int64_t sw = wi < w ? wi : ReflectIndexPublic(wi, w);
            padded[static_cast<size_t>(((c * config.t + ti) * nh + hi) * nw + wi)] =
                x[static_cast<size_t>(((c * config.t + ti) * h + sh) * w + sw)];
          }
        }
      }
    }
    input.swap(padded);
    h = nh;
    w = nw;
  }

  MiniMaxH3Conv3dSpec spec;
  spec.in_channels = config.in_channels;
  spec.out_channels = config.out_channels;
  spec.t = config.t;
  spec.h = h;
  spec.w = w;
  spec.kernel_t = spec.kernel_h = spec.kernel_w = 3;
  spec.pad_t = 1;
  spec.pad_h = spec.pad_w = 0;  // padding=(1, 0, 0)
  spec.stride_t = config.time_stride;
  spec.stride_h = spec.stride_w = config.space_stride;
  spec.causal = true;
  return MiniMaxH3CausalConv3d(input, spec, weight, &bias);
}

// EncoderFCN3D (vae_cnn.py:177-297): conv_in -> per level [N x ResnetBlock3D,
// then an optional Downsample3D or a 1x1x1 channel-matching conv] -> GroupNorm ->
// SiLU -> conv_out. Channel plan per level i:
//   block_mid[i] = ch * ch_mult[i];  block_in[0] = block_mid[0];
//   block_in[i>0] = block_mid[i-1];  block_out[i] = block_mid[i].
// A level gets a Downsample3D when space_down[i]*time_down[i] > 1; otherwise it
// gets a 1x1x1 conv ONLY if its output channel count differs, and nothing at all
// when it does not.
std::vector<float> MiniMaxH3EncoderFcn3dForward(const MiniMaxH3EncoderFcn3dConfig& config,
                                                const MiniMaxH3AudioVaeWeights& weights,
                                                const std::vector<float>& x,
                                                MiniMaxH3VideoFrameShape* out_shape) {
  const int64_t levels = static_cast<int64_t>(config.ch_mult.size());
  VT_CHECK(levels > 0, "minimax_h3 encoder3d: ch_mult must not be empty");
  VT_CHECK(static_cast<int64_t>(config.space_down.size()) == levels &&
               static_cast<int64_t>(config.time_down.size()) == levels,
           "minimax_h3 encoder3d: space_down/time_down must match ch_mult");

  std::vector<int64_t> block_mid(static_cast<size_t>(levels));
  for (int64_t i = 0; i < levels; ++i) block_mid[static_cast<size_t>(i)] = config.ch * config.ch_mult[static_cast<size_t>(i)];
  std::vector<int64_t> block_in(static_cast<size_t>(levels));
  block_in[0] = block_mid[0];
  for (int64_t i = 1; i < levels; ++i) block_in[static_cast<size_t>(i)] = block_mid[static_cast<size_t>(i - 1)];

  int64_t t = config.t, h = config.h, w = config.w;
  auto conv_spec = [&](int64_t in_ch, int64_t out_ch, int64_t kernel, int64_t pad) {
    MiniMaxH3Conv3dSpec spec;
    spec.in_channels = in_ch;
    spec.out_channels = out_ch;
    spec.t = t;
    spec.h = h;
    spec.w = w;
    spec.kernel_t = spec.kernel_h = spec.kernel_w = kernel;
    spec.pad_t = spec.pad_h = spec.pad_w = pad;
    spec.causal = true;
    return spec;
  };

  std::vector<float> hbuf =
      MiniMaxH3CausalConv3d(x, conv_spec(config.in_channels, block_in[0], 3, 1),
                            weights.Get("conv_in.weight"), &weights.Get("conv_in.bias"));
  int64_t channels = block_in[0];

  for (int64_t level = 0; level < levels; ++level) {
    for (int64_t b = 0; b < config.num_res_blocks; ++b) {
      MiniMaxH3ResnetBlock3dConfig block;
      block.in_channels = (b == 0) ? block_in[static_cast<size_t>(level)]
                                   : block_mid[static_cast<size_t>(level)];
      block.out_channels = block_mid[static_cast<size_t>(level)];
      block.t = t;
      block.h = h;
      block.w = w;
      block.num_groups = config.num_groups;
      block.eps = config.eps;
      hbuf = MiniMaxH3ResnetBlock3dForward(
          block, weights,
          "down." + std::to_string(level) + ".block." + std::to_string(b), hbuf);
      channels = block.out_channels;
    }

    const int64_t sd = config.space_down[static_cast<size_t>(level)];
    const int64_t td = config.time_down[static_cast<size_t>(level)];
    const std::string prefix = "down." + std::to_string(level) + ".downsample";
    if (sd * td > 1) {
      MiniMaxH3Downsample3dConfig down;
      down.in_channels = channels;
      down.out_channels = block_mid[static_cast<size_t>(level)];
      down.t = t;
      down.h = h;
      down.w = w;
      down.time_stride = td;
      down.space_stride = sd;
      hbuf = MiniMaxH3Downsample3d(hbuf, down, weights.Get(prefix + ".conv.weight"),
                                   weights.Get(prefix + ".conv.bias"));
      // The causal temporal pad means T shrinks by the stride like H and W do.
      t = (t + 2 - 3) / td + 1;
      h = (h + 1 - 3) / sd + 1;
      w = (w + 1 - 3) / sd + 1;
      channels = down.out_channels;
    } else if (block_mid[static_cast<size_t>(level)] != channels) {
      hbuf = MiniMaxH3CausalConv3d(hbuf, conv_spec(channels, block_mid[static_cast<size_t>(level)], 1, 0),
                                   weights.Get(prefix + ".weight"), &weights.Get(prefix + ".bias"));
      channels = block_mid[static_cast<size_t>(level)];
    }
  }

  MiniMaxH3GroupNorm3d(hbuf, channels, t * h * w, config.num_groups,
                       weights.Get("norm_out.weight"), weights.Get("norm_out.bias"), config.eps);
  for (float& v : hbuf) v = v / (1.0f + std::exp(-v));
  hbuf = MiniMaxH3CausalConv3d(hbuf, conv_spec(channels, config.z_channels, 3, 1),
                               weights.Get("conv_out.weight"), &weights.Get("conv_out.bias"));
  if (out_shape != nullptr) {
    out_shape->channels = config.z_channels;
    out_shape->t = t;
    out_shape->h = h;
    out_shape->w = w;
  }
  return hbuf;
}

}  // namespace vllm
