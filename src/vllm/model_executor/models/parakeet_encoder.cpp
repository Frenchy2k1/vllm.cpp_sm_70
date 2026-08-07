// Parakeet / FastConformer AUDIO encoder + CTC head forward — spike row
// `MODEL-AUDIO-PARAKEET-ENCODER` (work item P4 of
// .agents/specs/parakeet-conformer-encoder.md).
//
// Ported 1:1 from transformers 5.3.0
// `transformers/models/parakeet/modeling_parakeet.py` — the module vLLM itself
// runs (`vllm/model_executor/models/parakeet.py:14,61` import and instantiate
// `transformers.ParakeetEncoder`), which is why the provenance below is a
// transformers path and not a vLLM one. Full citation table and the recorded
// mirror-source deviation are in
// include/vllm/model_executor/models/parakeet_encoder.h.
//
//   ParakeetEncoderRelPositionalEncoding.forward     :71-98
//   ParakeetEncoderFeedForward.forward               :109-113
//   ParakeetEncoderConvolutionModule.forward         :151-185
//   ParakeetEncoderAttention.forward                 :291-346
//   ParakeetEncoderSubsamplingConv2D.forward         :404-423
//   ParakeetEncoderBlock.forward                     :442-470
//   ParakeetEncoder.forward                          :576-640
//   ParakeetForCTC.forward :688-757 / .generate :759-811
//   _get_subsampling_output_length :515-530, _get_output_attention_mask :532-541
//   tokenization_parakeet.py ParakeetTokenizer._decode :28-49
//
// Composed from the public vt:: ops. The three primitives the tree had no device
// op for before this spike do the structural work: vt::Conv2d (the subsampling
// front end), vt::DepthwiseConv1d (the conformer convolution module) and
// vt::AttentionRelPos (the Transformer-XL relative-position self-attention).
// Everything else reuses tuned shared ops (vt::MatmulBT, vt::Add, vt::LayerNorm,
// vt::Relu). The genuinely-elementwise leftovers that vt has no standalone op for
// — SiLU, GLU, the eval-time BatchNorm1d affine, the conformer 0.5 residual
// scaling and the two channel/time transposes — are explicit host loops, marked
// individually below rather than smuggled into a new op, because P4 owns no
// kernel files.
//
// Numeric contract: f32 throughout (the dtype the HF reference runs at), so the
// only parity variable is float summation order.
#include "vllm/model_executor/models/parakeet_encoder.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm::multimodal {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// --- RAII device buffer (same shape as whisper_audio.cpp's Buf, f32). --------
struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;

  Buf(Backend& backend, Queue& q, const std::vector<int64_t>& shape,
      const void* host = nullptr, DType dtype = DType::kF32)
      : b(backend) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dtype);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    t.data = p;
    t.dtype = dtype;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    if (host != nullptr) b.Copy(q, p, host, bytes);
  }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;

  Tensor& tensor() { return t; }
  std::vector<float> Download(Queue& q) {
    std::vector<float> out(bytes / sizeof(float));
    b.Copy(q, out.data(), p, bytes);
    b.Synchronize(q);
    return out;
  }
};

// The forward's marshalling context. Every stage takes and returns HOST f32
// vectors; each vt:: call uploads its operands and downloads its result. This is
// deliberately the simple, obviously-correct shape: P4's gate is faithfulness,
// and the speed gate for this family is GB10-only (spike § Gates) and is not
// claimed by this row. A device-resident forward is a separate, measurable step.
struct Ctx {
  Backend& b;
  Queue q;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

std::unique_ptr<Buf> Up(Ctx& c, const std::vector<float>& host,
                        const std::vector<int64_t>& shape) {
  if (static_cast<int64_t>(host.size()) != Numel(shape)) {
    throw std::runtime_error("parakeet: tensor size mismatch on upload");
  }
  return std::make_unique<Buf>(c.b, c.q, shape, host.data());
}

// out[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]) — torch `nn.Linear`. Also serves every
// kernel-1 `nn.Conv1d` in the model (the two pointwise convs of the convolution
// module, :134-136/:147-149, and the CTC head, :682): a 1x1 conv over [C,T] is
// exactly this Linear over the [T,C] view, which is why no conv op is needed and
// no transpose is paid for them.
std::vector<float> Linear(Ctx& c, const std::vector<float>& x, int64_t M, int64_t K,
                          const std::vector<float>& w, const std::vector<float>& bias,
                          int64_t N) {
  auto xb = Up(c, x, {M, K});
  auto wb = Up(c, w, {N, K});
  Buf out(c.b, c.q, {M, N});
  vt::MatmulBT(c.q, out.tensor(), xb->tensor(), wb->tensor());
  if (!bias.empty()) {
    auto bb = Up(c, bias, {N});
    vt::Add(c.q, out.tensor(), out.tensor(), bb->tensor());
  }
  return out.Download(c.q);
}

// torch `nn.LayerNorm(D)` over the last dim (:436-440 constructs five per block).
std::vector<float> LayerNorm(Ctx& c, const std::vector<float>& x, int64_t M, int64_t D,
                             const std::vector<float>& w, const std::vector<float>& bias,
                             float eps) {
  auto xb = Up(c, x, {M, D});
  auto wb = Up(c, w, {D});
  auto bb = Up(c, bias, {D});
  Buf out(c.b, c.q, {M, D});
  vt::LayerNormArgs args;
  args.eps = eps;
  vt::LayerNorm(c.q, out.tensor(), xb->tensor(), &wb->tensor(), &bb->tensor(), args);
  return out.Download(c.q);
}

// torch `nn.Conv2d` — vt::Conv2d (spike P1). x [1,Cin,H,W], weight
// [Cout,Cin/groups,KH,KW].
std::vector<float> Conv2d(Ctx& c, const std::vector<float>& x, int64_t cin, int64_t hin,
                          int64_t win, const std::vector<float>& w,
                          const std::vector<float>& bias, int64_t cout, int64_t kh,
                          int64_t kw, const vt::Conv2dArgs& args, int64_t hout,
                          int64_t wout) {
  auto xb = Up(c, x, {1, cin, hin, win});
  auto wb = Up(c, w, {cout, cin / args.groups, kh, kw});
  Buf out(c.b, c.q, {1, cout, hout, wout});
  std::unique_ptr<Buf> bb;
  if (!bias.empty()) bb = Up(c, bias, {cout});
  vt::Conv2d(c.q, out.tensor(), xb->tensor(), wb->tensor(),
             bb ? &bb->tensor() : nullptr, args);
  return out.Download(c.q);
}

// `F.relu`, applied in place — the subsampling stack's activation (:372, :388).
void ReluInPlace(Ctx& c, std::vector<float>& x) {
  auto xb = Up(c, x, {static_cast<int64_t>(x.size())});
  vt::Relu(c.q, xb->tensor(), xb->tensor());
  x = xb->Download(c.q);
}

// NON-CAUSAL depthwise `nn.Conv1d(C,C,K,groups=C)` — vt::DepthwiseConv1d (spike
// P2). x [1,C,L] channel-major, weight [C,1,K].
std::vector<float> DepthwiseConv1d(Ctx& c, const std::vector<float>& x, int64_t channels,
                                   int64_t length, const std::vector<float>& w,
                                   const std::vector<float>& bias, int64_t kernel,
                                   int64_t padding) {
  auto xb = Up(c, x, {1, channels, length});
  auto wb = Up(c, w, {channels, 1, kernel});
  const int64_t lout = length + 2 * padding - (kernel - 1) - 1 + 1;
  Buf out(c.b, c.q, {1, channels, lout});
  std::unique_ptr<Buf> bb;
  if (!bias.empty()) bb = Up(c, bias, {channels});
  vt::DepthwiseConv1dArgs args;
  args.stride = 1;
  args.padding = padding;
  args.dilation = 1;
  vt::DepthwiseConv1d(c.q, out.tensor(), xb->tensor(), wb->tensor(),
                      bb ? &bb->tensor() : nullptr, args);
  return out.Download(c.q);
}

// --- host-only elementwise leftovers ----------------------------------------
// vt has no standalone op for these, and P4 owns no kernel file, so each is an
// explicit loop with its upstream line cited. All are pointwise, so they cannot
// hide a shape/ordering bug the way a fused kernel could.

// `ACT2FN["silu"]` — the feed-forward activation (:105, :110) and the convolution
// module's (:129, :182). silu(x) = x * sigmoid(x).
void SiluInPlace(std::vector<float>& x) {
  for (float& v : x) {
    v = v / (1.0f + std::exp(-v));
  }
}

// `nn.functional.glu(hidden_states, dim=1)` on [B, 2C, T] (:169). Our layout is
// the transposed [T, 2C] view, where the channel split is the LAST axis, so
// out[t,c] = x[t,c] * sigmoid(x[t, C+c]).
std::vector<float> Glu(const std::vector<float>& x, int64_t rows, int64_t channels) {
  std::vector<float> out(static_cast<size_t>(rows) * channels);
  for (int64_t t = 0; t < rows; ++t) {
    const float* row = &x[static_cast<size_t>(t) * 2 * channels];
    for (int64_t ch = 0; ch < channels; ++ch) {
      out[static_cast<size_t>(t) * channels + ch] =
          row[ch] / (1.0f + std::exp(-row[channels + ch]));
    }
  }
  return out;
}

// `nn.BatchNorm1d(channels)` in EVAL mode (:146, applied :181): the running
// statistics are the normaliser, never the batch's. x is [C, L] channel-major.
void BatchNorm1dInPlace(std::vector<float>& x, int64_t channels, int64_t length,
                        const std::vector<float>& weight, const std::vector<float>& bias,
                        const std::vector<float>& running_mean,
                        const std::vector<float>& running_var, float eps) {
  for (int64_t ch = 0; ch < channels; ++ch) {
    const float inv =
        1.0f / std::sqrt(running_var[static_cast<size_t>(ch)] + eps);
    const float mean = running_mean[static_cast<size_t>(ch)];
    const float w = weight.empty() ? 1.0f : weight[static_cast<size_t>(ch)];
    const float b = bias.empty() ? 0.0f : bias[static_cast<size_t>(ch)];
    float* row = &x[static_cast<size_t>(ch) * length];
    for (int64_t t = 0; t < length; ++t) {
      row[t] = (row[t] - mean) * inv * w + b;
    }
  }
}

// [rows, cols] -> [cols, rows]. The convolution module's two `transpose(1, 2)`
// calls (:164, :185) between the block's time-major layout and the conv stack's
// channel-major one.
std::vector<float> Transpose2d(const std::vector<float>& x, int64_t rows, int64_t cols) {
  std::vector<float> out(static_cast<size_t>(rows) * cols);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t cidx = 0; cidx < cols; ++cidx) {
      out[static_cast<size_t>(cidx) * rows + r] = x[static_cast<size_t>(r) * cols + cidx];
    }
  }
  return out;
}

// dst += scale * src, the conformer's half-weighted residual joins (:451, :466)
// and the plain ones (:460, :463) at scale 1.
void AddScaled(std::vector<float>& dst, const std::vector<float>& src, float scale) {
  for (size_t i = 0; i < dst.size(); ++i) dst[i] += scale * src[i];
}

// Zero every row at or past `valid_rows` of a [rows, cols] host matrix.
void ZeroRowsFrom(std::vector<float>& x, int64_t rows, int64_t cols, int64_t valid_rows) {
  for (int64_t r = std::max<int64_t>(valid_rows, 0); r < rows; ++r) {
    std::fill_n(&x[static_cast<size_t>(r) * cols], static_cast<size_t>(cols), 0.0f);
  }
}

// --- the modules ------------------------------------------------------------

// ParakeetEncoderFeedForward.forward (:109-113): linear1 -> silu -> linear2.
// `activation_dropout` (:111) is inference-inert.
std::vector<float> FeedForward(Ctx& c, const std::vector<float>& x, int64_t rows,
                               const ParakeetFeedForwardWeights& w,
                               const ParakeetEncoderConfig& cfg) {
  std::vector<float> h = Linear(c, x, rows, cfg.hidden_size, w.linear1_w, w.linear1_b,
                                cfg.intermediate_size);
  SiluInPlace(h);
  return Linear(c, h, rows, cfg.intermediate_size, w.linear2_w, w.linear2_b,
                cfg.hidden_size);
}

// ParakeetEncoderConvolutionModule.forward (:151-185). `x` is [rows, C]
// time-major (upstream's `(batch, time, channels)`, :156).
std::vector<float> ConvolutionModule(Ctx& c, const std::vector<float>& x, int64_t rows,
                                     int64_t valid_rows,
                                     const ParakeetConvModuleWeights& w,
                                     const ParakeetEncoderConfig& cfg) {
  const int64_t channels = cfg.hidden_size;

  // :167-169 pointwise_conv1 (k=1) then GLU over the channel axis. Done in the
  // [rows, 2C] view, so upstream's :164 transpose is not paid here.
  std::vector<float> gated =
      Linear(c, x, rows, channels, w.pointwise1_w, w.pointwise1_b, 2 * channels);
  std::vector<float> h = Glu(gated, rows, channels);

  // :171-177 "Apply padding mask before convolution": every time position that
  // no query attends to is zeroed, so the depthwise conv never mixes padding in.
  ZeroRowsFrom(h, rows, channels, valid_rows);

  // :180 depthwise conv, centre-padded by (K-1)//2 (:133, :142) — channel-major.
  std::vector<float> ct = Transpose2d(h, rows, channels);  // :164 (deferred to here)
  ct = DepthwiseConv1d(c, ct, channels, rows, w.depthwise_w, w.depthwise_b,
                       cfg.conv_kernel_size, (cfg.conv_kernel_size - 1) / 2);

  // :181-182 BatchNorm1d (eval) then silu.
  BatchNorm1dInPlace(ct, channels, rows, w.norm_w, w.norm_b, w.norm_running_mean,
                     w.norm_running_var, cfg.batch_norm_eps);
  SiluInPlace(ct);

  // :183-185 pointwise_conv2 (k=1) + the transpose back to time-major.
  std::vector<float> tc = Transpose2d(ct, channels, rows);
  return Linear(c, tc, rows, channels, w.pointwise2_w, w.pointwise2_b, channels);
}

// ParakeetEncoderAttention.forward (:291-346), through vt::AttentionRelPos
// (spike P3). `x` is the pre-normalised [rows, hidden]; `pos_embed` is the
// [2*rows-1, hidden] relative-position table.
std::vector<float> SelfAttention(Ctx& c, const std::vector<float>& x,
                                 const std::vector<float>& pos_embed, int64_t rows,
                                 int64_t valid_rows, const ParakeetAttentionWeights& w,
                                 const ParakeetEncoderConfig& cfg) {
  const int64_t heads = cfg.num_attention_heads;
  const int64_t kv_heads = cfg.num_key_value_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t pos_rows = 2 * rows - 1;

  // :302-304 q/k/v projections. [rows, H*Dh] row-major IS [rows, H, Dh].
  std::vector<float> qs = Linear(c, x, rows, cfg.hidden_size, w.q_w, w.q_b, heads * hd);
  std::vector<float> ks =
      Linear(c, x, rows, cfg.hidden_size, w.k_w, w.k_b, kv_heads * hd);
  std::vector<float> vs =
      Linear(c, x, rows, cfg.hidden_size, w.v_w, w.v_b, kv_heads * hd);
  // :317-318 W_{k,R} applied to the position table; never biased (:285).
  std::vector<float> rel = Linear(c, pos_embed, pos_rows, cfg.hidden_size,
                                  w.relative_k_w, {}, heads * hd);

  auto qb = Up(c, qs, {rows, heads, hd});
  auto kb = Up(c, ks, {rows, kv_heads, hd});
  auto vb = Up(c, vs, {rows, kv_heads, hd});
  auto rb = Up(c, rel, {pos_rows, heads, hd});
  auto ub = Up(c, w.bias_u, {heads, hd});   // :287, term (c)
  auto vvb = Up(c, w.bias_v, {heads, hd});  // :289, term (d)

  // :326-330 the additive mask. Upstream's [T,T] mask is the outer product of
  // the subsampled validity vector with itself (:617-620), so for every query
  // row that is itself valid it reduces EXACTLY to "key j is valid" — which is
  // what vt::AttentionRelPos takes. The rows it does NOT reduce for (fully
  // masked queries) are handled right after the call.
  std::vector<int32_t> key_mask(static_cast<size_t>(rows), 0);
  for (int64_t t = 0; t < std::min(valid_rows, rows); ++t) {
    key_mask[static_cast<size_t>(t)] = 1;
  }
  Buf mask_buf(c.b, c.q, {rows}, key_mask.data(), DType::kI32);

  Buf attn(c.b, c.q, {rows, heads, hd});
  vt::AttentionRelPosArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(hd));  // :268
  args.scale_after_sum = false;                           // HF's form (:324, :246)
  vt::AttentionRelPos(c.q, attn.tensor(), qb->tensor(), kb->tensor(), vb->tensor(),
                      rb->tensor(), &ub->tensor(), &vvb->tensor(), &mask_buf.tensor(),
                      args);
  std::vector<float> out = attn.Download(c.q);

  // A query row past the valid length has EVERY key masked upstream (:619, the
  // `attention_mask & attention_mask.transpose` conjunction). The path that
  // actually runs there is `sdpa`, which returns exactly ZERO for such a row
  // (traced; recorded in the fixture manifest) where `eager` would return NaN.
  // vt::AttentionRelPos only carries a per-KEY mask, so those rows come back
  // with a real attention output; zeroing them here reproduces the traced
  // upstream observable, and o_proj's bias is then added on top exactly as
  // upstream adds it to sdpa's zero.
  ZeroRowsFrom(out, rows, heads * hd, valid_rows);

  // :344-345 reshape + o_proj.
  return Linear(c, out, rows, heads * hd, w.o_w, w.o_b, cfg.hidden_size);
}

// ParakeetEncoderSubsamplingConv2D.forward (:404-423). `input_features` is
// [num_frames, num_mel_bins]; upstream's :405 `unsqueeze(1)` makes it
// [1, 1, time, mel], so H is TIME and W is the mel axis and both are subsampled.
std::vector<float> Subsampling(Ctx& c, const std::vector<float>& input_features,
                               int64_t num_frames, int64_t valid_frames,
                               const ParakeetSubsamplingWeights& w,
                               const ParakeetEncoderConfig& cfg, int64_t* out_frames) {
  const int64_t k = cfg.subsampling_conv_kernel_size;
  const int64_t stride = cfg.subsampling_conv_stride;
  const int64_t pad = cfg.subsampling_padding();
  const int64_t channels = cfg.subsampling_conv_channels;

  vt::Conv2dArgs conv_args;
  conv_args.stride_h = stride;
  conv_args.stride_w = stride;
  conv_args.pad_h = pad;
  conv_args.pad_w = pad;

  // `_get_output_length` (:393-402) for a STRIDED layer; a 1x1 pointwise layer
  // has stride (1,1) and returns its input unchanged.
  auto strided_out = [&](int64_t len) { return (len + 2 * pad - k) / stride + 1; };

  int64_t h = num_frames;
  int64_t wdim = cfg.num_mel_bins;
  int64_t cur_len = valid_frames;  // :406 attention_mask.sum(-1)

  // :415-418 zero every subsampled TIME row past the item's length, across all
  // channels and all frequency bins — the mask is over the H (time) axis of a
  // [1, C, H, W] tensor.
  auto mask_time = [&](std::vector<float>& t, int64_t chan, int64_t hh, int64_t ww,
                       int64_t len) {
    for (int64_t ci = 0; ci < chan; ++ci) {
      for (int64_t hi = std::max<int64_t>(len, 0); hi < hh; ++hi) {
        std::fill_n(&t[(static_cast<size_t>(ci) * hh + hi) * ww], static_cast<size_t>(ww),
                    0.0f);
      }
    }
  };

  // layers[0]: the dense 1 -> C strided conv (:369-371), then its mask, then the
  // ReLU at layers[1] (:372).
  int64_t hout = strided_out(h);
  int64_t wout = strided_out(wdim);
  std::vector<float> x = Conv2d(c, input_features, /*cin=*/1, h, wdim, w.conv0_w,
                                w.conv0_b, channels, k, k, conv_args, hout, wout);
  h = hout;
  wdim = wout;
  cur_len = strided_out(cur_len);  // :413
  mask_time(x, channels, h, wdim, cur_len);
  ReluInPlace(c, x);

  vt::Conv2dArgs depthwise_args = conv_args;
  depthwise_args.groups = channels;
  vt::Conv2dArgs pointwise_args;  // 1x1, stride 1, no padding (:386)

  for (const ParakeetSubsamplingWeights::Stage& stage : w.stages) {
    hout = strided_out(h);
    wout = strided_out(wdim);
    x = Conv2d(c, x, channels, h, wdim, stage.depthwise_w, stage.depthwise_b, channels,
               k, k, depthwise_args, hout, wout);
    h = hout;
    wdim = wout;
    cur_len = strided_out(cur_len);
    mask_time(x, channels, h, wdim, cur_len);

    x = Conv2d(c, x, channels, h, wdim, stage.pointwise_w, stage.pointwise_b, channels,
               1, 1, pointwise_args, h, wdim);
    // A 1x1 conv is not strided, so `_get_output_length` returns cur_len
    // unchanged (:394, :402) — but the mask IS re-applied (:412-418).
    mask_time(x, channels, h, wdim, cur_len);
    ReluInPlace(c, x);  // :388
  }

  if (wdim != cfg.subsampling_out_freq()) {
    throw std::runtime_error("parakeet: subsampling frequency extent mismatch");
  }

  // :420 transpose(1,2).reshape(B, T, -1): from [1, C, T, F] to [T, C*F] with the
  // CHANNEL as the major axis of the flattened feature.
  std::vector<float> flat(static_cast<size_t>(h) * channels * wdim);
  for (int64_t t = 0; t < h; ++t) {
    for (int64_t ci = 0; ci < channels; ++ci) {
      for (int64_t f = 0; f < wdim; ++f) {
        flat[(static_cast<size_t>(t) * channels + ci) * wdim + f] =
            x[(static_cast<size_t>(ci) * h + t) * wdim + f];
      }
    }
  }

  *out_frames = h;
  // :421 the projection onto hidden_size.
  return Linear(c, flat, h, channels * wdim, w.linear_w, w.linear_b, cfg.hidden_size);
}

}  // namespace

int64_t ParakeetEncoderConfig::num_subsampling_layers() const {
  int64_t n = 0;
  for (int64_t f = subsampling_factor; f > 1; f /= 2) ++n;
  return n;
}

int64_t ParakeetEncoderConfig::subsampling_out_freq() const {
  int64_t freq = num_mel_bins;
  for (int64_t i = 0; i < num_subsampling_layers(); ++i) freq /= subsampling_conv_stride;
  return freq;
}

int64_t ParakeetSubsamplingOutputLength(int64_t input_length,
                                        const ParakeetEncoderConfig& cfg) {
  // :522-529 — all_paddings = (k-1)//2*2, add_pad = all_paddings - k, then per
  // layer floor((L + add_pad)/stride) + 1.
  const int64_t all_paddings = (cfg.subsampling_conv_kernel_size - 1) / 2 * 2;
  const int64_t add_pad = all_paddings - cfg.subsampling_conv_kernel_size;
  int64_t length = input_length;
  for (int64_t i = 0; i < cfg.num_subsampling_layers(); ++i) {
    length = (length + add_pad) / cfg.subsampling_conv_stride + 1;
  }
  return length;
}

std::vector<float> ParakeetRelPositionalEncoding(int64_t seq_length,
                                                 int64_t hidden_size) {
  const int64_t half = hidden_size / 2;
  const int64_t rows = 2 * seq_length - 1;
  std::vector<float> out(static_cast<size_t>(rows) * hidden_size);
  for (int64_t p = 0; p < rows; ++p) {
    // :79 positions run seq_length-1 down to -(seq_length-1).
    const double pos = static_cast<double>(seq_length - 1 - p);
    for (int64_t i = 0; i < half; ++i) {
      // :60-66 inv_freq[i] = 1 / base^(2i / hidden_size), base = 10000.
      const double inv_freq =
          1.0 / std::pow(10000.0, static_cast<double>(2 * i) / static_cast<double>(hidden_size));
      const double freq = pos * inv_freq;
      // :95-96 stack([sin, cos], -1).reshape(...) — INTERLEAVED, not halves.
      out[static_cast<size_t>(p) * hidden_size + 2 * i] = static_cast<float>(std::sin(freq));
      out[static_cast<size_t>(p) * hidden_size + 2 * i + 1] = static_cast<float>(std::cos(freq));
    }
  }
  return out;
}

std::vector<float> ParakeetEncoderForward(const std::vector<float>& input_features,
                                          int64_t num_frames, int64_t valid_frames,
                                          const ParakeetEncoderWeights& w,
                                          const ParakeetEncoderConfig& cfg,
                                          Backend& backend, int64_t* out_valid_frames,
                                          ParakeetEncoderCapture* capture) {
  if (cfg.hidden_act != "silu") {
    throw std::runtime_error("parakeet: unsupported hidden_act '" + cfg.hidden_act +
                             "' (only silu is ported)");
  }
  if (static_cast<int64_t>(input_features.size()) != num_frames * cfg.num_mel_bins) {
    throw std::runtime_error("parakeet: input_features size mismatch");
  }
  if (static_cast<int64_t>(w.layers.size()) != cfg.num_hidden_layers) {
    throw std::runtime_error("parakeet: layer count mismatch");
  }
  valid_frames = std::min(std::max<int64_t>(valid_frames, 0), num_frames);

  Ctx c{backend, backend.CreateQueue()};

  int64_t rows = 0;
  std::vector<float> h =
      Subsampling(c, input_features, num_frames, valid_frames, w.subsampling, cfg, &rows);
  if (capture != nullptr) capture->subsampling_out = h;

  // :562 input_scale, :608 the multiply.
  if (cfg.scale_input) {
    const float scale = static_cast<float>(std::sqrt(static_cast<double>(cfg.hidden_size)));
    for (float& v : h) v *= scale;
  }

  // :73-77 the length guard, checked on the SUBSAMPLED length (:72).
  if (rows > cfg.max_position_embeddings) {
    throw std::runtime_error("parakeet: sequence length exceeds max_position_embeddings");
  }
  std::vector<float> pos_embed = ParakeetRelPositionalEncoding(rows, cfg.hidden_size);
  if (capture != nullptr) capture->pos_embed = pos_embed;

  // :617 the subsampled validity length (`_get_output_attention_mask`).
  const int64_t valid_rows =
      std::min(ParakeetSubsamplingOutputLength(valid_frames, cfg), rows);

  for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
    const ParakeetEncoderLayerWeights& lw = w.layers[static_cast<size_t>(li)];

    // :449-451 macaron half-step feed-forward.
    std::vector<float> normed = LayerNorm(c, h, rows, cfg.hidden_size,
                                          lw.norm_feed_forward1_w,
                                          lw.norm_feed_forward1_b, cfg.layer_norm_eps);
    std::vector<float> ff1 = FeedForward(c, normed, rows, lw.feed_forward1, cfg);
    if (li == 0 && capture != nullptr) capture->layer0_ff1 = ff1;
    AddScaled(h, ff1, 0.5f);

    // :453-460 self-attention.
    normed = LayerNorm(c, h, rows, cfg.hidden_size, lw.norm_self_att_w,
                       lw.norm_self_att_b, cfg.layer_norm_eps);
    std::vector<float> attn =
        SelfAttention(c, normed, pos_embed, rows, valid_rows, lw.self_attn, cfg);
    if (li == 0 && capture != nullptr) capture->layer0_attn = attn;
    AddScaled(h, attn, 1.0f);

    // :462-463 convolution module.
    normed = LayerNorm(c, h, rows, cfg.hidden_size, lw.norm_conv_w, lw.norm_conv_b,
                       cfg.layer_norm_eps);
    std::vector<float> conv =
        ConvolutionModule(c, normed, rows, valid_rows, lw.conv, cfg);
    if (li == 0 && capture != nullptr) capture->layer0_conv = conv;
    AddScaled(h, conv, 1.0f);

    // :465-466 the second macaron half-step.
    normed = LayerNorm(c, h, rows, cfg.hidden_size, lw.norm_feed_forward2_w,
                       lw.norm_feed_forward2_b, cfg.layer_norm_eps);
    std::vector<float> ff2 = FeedForward(c, normed, rows, lw.feed_forward2, cfg);
    AddScaled(h, ff2, 0.5f);

    // :468 the block's closing norm.
    h = LayerNorm(c, h, rows, cfg.hidden_size, lw.norm_out_w, lw.norm_out_b,
                  cfg.layer_norm_eps);
    if (li == 0 && capture != nullptr) capture->block0_out = h;
  }

  backend.DestroyQueue(c.q);
  if (out_valid_frames != nullptr) *out_valid_frames = valid_rows;
  return h;
}

std::vector<int32_t> ParakeetCtcGreedyCollapse(const std::vector<int32_t>& ids,
                                               int32_t blank_id) {
  std::vector<int32_t> out;
  out.reserve(ids.size());
  // tokenization_parakeet.py:38-39 — `itertools.groupby` keeps the FIRST element
  // of every run of equal ids; :42 then drops the blank (which is pad_token_id).
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i > 0 && ids[i] == ids[i - 1]) continue;
    if (ids[i] == blank_id) continue;
    out.push_back(ids[i]);
  }
  return out;
}

ParakeetCTCOutput ParakeetForCTCForward(const std::vector<float>& input_features,
                                        int64_t num_frames, int64_t valid_frames,
                                        const ParakeetForCTCWeights& w,
                                        const ParakeetEncoderConfig& cfg,
                                        Backend& backend) {
  ParakeetCTCOutput out;
  std::vector<float> hidden =
      ParakeetEncoderForward(input_features, num_frames, valid_frames, w.encoder, cfg,
                             backend, &out.valid_output_frames, nullptr);
  const int64_t rows = cfg.hidden_size == 0
                           ? 0
                           : static_cast<int64_t>(hidden.size()) / cfg.hidden_size;
  out.num_output_frames = rows;

  Ctx c{backend, backend.CreateQueue()};
  // :682/:722 the CTC head is a kernel-1 Conv1d over [hidden, T], i.e. a Linear
  // over the [T, hidden] view; the two transposes at :722 cancel.
  out.logits = Linear(c, hidden, rows, cfg.hidden_size, w.ctc_head_w, w.ctc_head_b,
                      cfg.vocab_size);
  backend.DestroyQueue(c.q);

  // :796 greedy argmax. torch.argmax returns the FIRST maximal index, so the
  // comparison must be strict.
  out.greedy_ids.assign(static_cast<size_t>(rows), cfg.pad_token_id);
  for (int64_t t = 0; t < rows; ++t) {
    const float* row = &out.logits[static_cast<size_t>(t) * cfg.vocab_size];
    int64_t best = 0;
    for (int64_t v = 1; v < cfg.vocab_size; ++v) {
      if (row[v] > row[best]) best = v;
    }
    out.greedy_ids[static_cast<size_t>(t)] = static_cast<int32_t>(best);
  }
  // :798-801 every frame the output mask rejects becomes the pad/blank id.
  for (int64_t t = out.valid_output_frames; t < rows; ++t) {
    out.greedy_ids[static_cast<size_t>(t)] = cfg.pad_token_id;
  }
  out.token_ids = ParakeetCtcGreedyCollapse(out.greedy_ids, cfg.pad_token_id);
  return out;
}

}  // namespace vllm::multimodal
