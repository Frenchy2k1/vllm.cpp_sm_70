// MiniMax-H3 VIDEO VAE decoder, device-resident.
//
// The portable decoder in minimax_h3_video_vae.cpp is a CORRECTNESS reference:
// scalar triple loops, double accumulation, one key at a time. That is the right
// shape for a golden, and the wrong shape for a 36-layer ViT3D over thousands of
// tokens -- it is what made a 256x256 decode time out. This is the same graph with
// every activation on the device and every heavy step routed to the tuned shared
// ops.
//
// NO NEW KERNELS. The mapping, mirroring minimax_h3_device.cpp's for the DiT:
//
//   Linear                  -> vt::MatmulBT + vt::Add (rank-1 row-broadcast bias)
//   RMSNorm                 -> vt::RmsNorm
//   qk-norm (no affine)     -> vt::RmsNorm with a ones weight
//   qkv split               -> vt::QkvSplit
//   3D RoPE                 -> vt::RopeFromCache over a prebuilt [seq, rot_dim] cache
//   gated SiLU feed-forward -> vt::SiluAndMul
//   attention               -> vt::DFlashBlockAttention(causal=false), one document
//   LayerNorm (norm_out)    -> vt::LayerNorm
//   residual add            -> vt::Add
//
// TWO LOAD-TIME WEIGHT FOLDS make that mapping possible, and both are exact
// rearrangements rather than approximations:
//
// 1. QKV ROW REORDER. This checkpoint stores to_qkv PER-HEAD INTERLEAVED
//    ([head][q|k|v] within a row), while vt::QkvSplit -- and every other model in
//    this tree -- expects [q_all | k_all | v_all]. Permuting the weight ROWS (and
//    the bias) once at stage time puts the GEMM output directly in QkvSplit's
//    layout. The alternative, a bespoke interleaved-split kernel, would buy nothing.
//
// 2. PER-CHANNEL SCALE FOLD. Each branch ends `h += scale * (x @ W^T + b)` with a
//    LEARNED PER-OUTPUT-CHANNEL `scale`. Since `scale` indexes the output channel,
//    scaling row d of W and element d of b by scale[d] is algebraically identical
//    and leaves a plain vt::Add. This is a fold of a rank-1 diagonal into a GEMM,
//    not a numerical shortcut -- but it does REASSOCIATE the rounding (the scale is
//    applied to the weights before the dot product rather than to the sum after
//    it), so this path is held to the decoder's tolerance gate, not to bit-equality
//    with the reference. That is the same standing the DiT device forward has.
//
// The reference accumulates its dot products in DOUBLE; the shared ops accumulate
// in f32, which is what upstream torch does. So this is deliberately not
// bit-identical to the portable decoder and is gated against the same goldens.
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::DType;
using vt::Tensor;

// Upload one f32 parameter and record its owner in `storage`, the same
// ownership shape StageMiniMaxH3DitWeights uses: the struct holds plain Tensor
// VIEWS, and `storage` keeps the allocations alive for exactly as long.
Tensor Upload(vt::Backend& backend, vt::Queue& queue, const std::vector<float>& host,
              const std::vector<int64_t>& shape,
              std::vector<std::shared_ptr<void>>* storage) {
  int64_t numel = 1;
  for (const int64_t s : shape) numel *= s;
  VT_CHECK(static_cast<int64_t>(host.size()) == numel,
           "minimax_h3 video vae device: host buffer does not match the requested shape");
  const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, host.data(), bytes);
  storage->push_back(std::move(owner));
  return MakeTensor(p, DType::kF32, queue.device, shape);
}

// vt::MatmulBT + optional rank-1 bias -- the device twin of the reference Linear.
void LinearDev(Dev d, const Tensor& in, int64_t rows, int64_t in_features, const Tensor& weight,
               const Tensor* bias, Tensor& out) {
  VT_CHECK(weight.rank == 2 && weight.shape[1] == in_features,
           "minimax_h3 video vae device: weight shape does not match input width");
  Tensor a = dense_attn::Reshape(in, {rows, in_features});
  Tensor o = dense_attn::Reshape(out, {rows, weight.shape[0]});
  vt::MatmulBT(d.q, o, a, weight);
  if (bias != nullptr && bias->data != nullptr) vt::Add(d.q, o, o, *bias);
}

// Fold #1: [head][q|k|v] rows -> [q_all | k_all | v_all] rows. Applies to both the
// to_qkv weight (row-major [3*inner, dim]) and its bias ([3*inner]).
std::vector<float> ReorderQkvRows(const std::vector<float>& src, int64_t heads, int64_t dim_head,
                                  int64_t width) {
  const int64_t inner = heads * dim_head;
  VT_CHECK(static_cast<int64_t>(src.size()) == 3 * inner * width,
           "minimax_h3 video vae device: to_qkv parameter is not [3 * heads * dim_head, width]");
  std::vector<float> out(src.size());
  for (int64_t head = 0; head < heads; ++head) {
    for (int64_t part = 0; part < 3; ++part) {
      // source rows for (head, part) are contiguous: head * 3 * dim_head + part * dim_head
      const int64_t src_row = head * 3 * dim_head + part * dim_head;
      const int64_t dst_row = part * inner + head * dim_head;
      for (int64_t r = 0; r < dim_head; ++r) {
        for (int64_t c = 0; c < width; ++c) {
          out[static_cast<size_t>((dst_row + r) * width + c)] =
              src[static_cast<size_t>((src_row + r) * width + c)];
        }
      }
    }
  }
  return out;
}

// Fold #2: scale row d of a [out, in] weight (or element d of a bias) by scale[d].
std::vector<float> FoldPerChannelScale(const std::vector<float>& src,
                                       const std::vector<float>& scale, int64_t width) {
  VT_CHECK(static_cast<int64_t>(src.size()) == static_cast<int64_t>(scale.size()) * width,
           "minimax_h3 video vae device: per-channel scale does not match the parameter rows");
  std::vector<float> out(src.size());
  for (size_t d = 0; d < scale.size(); ++d) {
    for (int64_t c = 0; c < width; ++c) {
      out[d * static_cast<size_t>(width) + static_cast<size_t>(c)] =
          src[d * static_cast<size_t>(width) + static_cast<size_t>(c)] * scale[d];
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Staged weights
// ---------------------------------------------------------------------------

MiniMaxH3VideoVaeDeviceWeights StageMiniMaxH3VideoVaeWeights(
    vt::Queue& queue, const MiniMaxH3VideoVaeDecoderConfig& config,
    const MiniMaxH3AudioVaeWeights& weights) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const int64_t dim = config.block.dim;
  const int64_t heads = config.block.heads;
  const int64_t dim_head = config.block.dim_head;
  const int64_t inner = heads * dim_head;
  const int64_t ff_inner = config.block.ff_inner;
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;

  MiniMaxH3VideoVaeDeviceWeights out;
  auto up = [&](const std::vector<float>& host, const std::vector<int64_t>& shape) {
    return Upload(backend, queue, host, shape, &out.storage);
  };

  out.blocks.reserve(static_cast<size_t>(config.num_layers));
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string p = "transformer_blocks." + std::to_string(layer);
    const std::vector<float>& scale1 = weights.Get(p + ".scale1");
    const std::vector<float>& scale2 = weights.Get(p + ".scale2");
    MiniMaxH3VideoVaeDeviceBlock b;
    b.norm1 = up(weights.Get(p + ".norm1.weight"), {dim});
    b.norm2 = up(weights.Get(p + ".norm2.weight"), {dim});
    b.qkv_weight = up(ReorderQkvRows(weights.Get(p + ".attn.to_qkv.weight"), heads, dim_head, dim),
                      {3 * inner, dim});
    b.qkv_bias =
        up(ReorderQkvRows(weights.Get(p + ".attn.to_qkv.bias"), heads, dim_head, 1), {3 * inner});
    b.out_weight =
        up(FoldPerChannelScale(weights.Get(p + ".attn.to_out.weight"), scale1, inner), {dim, inner});
    b.out_bias = up(FoldPerChannelScale(weights.Get(p + ".attn.to_out.bias"), scale1, 1), {dim});
    b.w1_weight = up(weights.Get(p + ".ff.w1.weight"), {2 * ff_inner, dim});
    b.w1_bias = up(weights.Get(p + ".ff.w1.bias"), {2 * ff_inner});
    b.w2_weight =
        up(FoldPerChannelScale(weights.Get(p + ".ff.w2.weight"), scale2, ff_inner), {dim, ff_inner});
    b.w2_bias = up(FoldPerChannelScale(weights.Get(p + ".ff.w2.bias"), scale2, 1), {dim});
    out.blocks.push_back(std::move(b));
  }

  out.x_embedder_weight = up(weights.Get("x_embedder.weight"), {dim, config.in_channels});
  out.x_embedder_bias = up(weights.Get("x_embedder.bias"), {dim});
  out.register_tokens = up(weights.Get("register_tokens"), {config.num_register_tokens, dim});
  out.norm_out_weight = up(weights.Get("norm_out.weight"), {dim});
  out.has_norm_out_bias = weights.Has("norm_out.bias");
  if (out.has_norm_out_bias) out.norm_out_bias = up(weights.Get("norm_out.bias"), {dim});
  out.proj_out_weight = up(weights.Get("proj_out.weight"), {patch_dim, dim});
  out.proj_out_bias = up(weights.Get("proj_out.bias"), {patch_dim});
  out.qk_norm_ones = up(std::vector<float>(static_cast<size_t>(dim_head), 1.0f), {dim_head});
  backend.Synchronize(queue);
  return out;
}

// ---------------------------------------------------------------------------
// The decode
// ---------------------------------------------------------------------------

std::vector<float> MiniMaxH3VideoVaeDecodeDevice(vt::Device device,
                                                 const MiniMaxH3VideoVaeDecoderConfig& config,
                                                 const MiniMaxH3VideoVaeDeviceWeights& staged,
                                                 const std::vector<float>& latent, int64_t latent_t,
                                                 int64_t latent_h, int64_t latent_w,
                                                 MiniMaxH3VideoFrameShape* out_shape) {
  vt::Backend& backend = vt::GetBackend(device.type);
  vt::Queue queue = backend.CreateQueue();
  Dev d{backend, queue};

  const int64_t dim = config.block.dim;
  const int64_t heads = config.block.heads;
  const int64_t dim_head = config.block.dim_head;
  const int64_t inner = heads * dim_head;
  const int64_t ff_inner = config.block.ff_inner;
  const int64_t patches = latent_t * latent_h * latent_w;
  const int64_t num_suffix = 1 + config.num_register_tokens;  // register tokens + cls
  const int64_t seq = patches + num_suffix;
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;
  VT_CHECK(static_cast<int64_t>(latent.size()) == config.in_channels * patches,
           "minimax_h3 video vae device: latent size does not match [C, T, H, W]");

  // _pack_tensors_3d(x, 1, 1): [C,T,H,W] -> [T*H*W, C]. Cheap and host-side; the
  // cost that mattered was never this transpose.
  std::vector<float> tokens(static_cast<size_t>(patches * config.in_channels));
  for (int64_t p = 0; p < patches; ++p) {
    for (int64_t c = 0; c < config.in_channels; ++c) {
      tokens[static_cast<size_t>(p * config.in_channels + c)] =
          latent[static_cast<size_t>(c * patches + p)];
    }
  }

  // hidden = [x_embedder(tokens) | register_tokens | zero cls row]. The cls row is
  // an explicit ZERO (vae_vit.py:311-313), which the zero-fill below provides.
  DBuf hidden(d, DType::kF32, {seq, dim});
  backend.Memset(queue, hidden.t().data, 0, hidden.t().Bytes());
  {
    DBuf tok(d, DType::kF32, {patches, config.in_channels}, tokens.data());
    Tensor embedded = MakeTensor(hidden.t().data, DType::kF32, device, {patches, dim});
    LinearDev(d, tok.t(), patches, config.in_channels, staged.x_embedder_weight,
              &staged.x_embedder_bias, embedded);
    backend.Copy(queue, static_cast<char*>(hidden.t().data) + patches * dim * sizeof(float),
                 staged.register_tokens.data,
                 static_cast<size_t>(config.num_register_tokens * dim) * sizeof(float));
  }

  // The RoPE cache. vt::RopeFromCache consumes [P, rot_dim] laid out as
  // [cos(half) | sin(half)] and indexes it by position, so the reference's own
  // cos/sin tables -- which are already per-token and already tiled so that
  // cos[i + half] == cos[i] -- transcribe straight into it with no rope kernel.
  const int64_t rot_dim = config.rope_apply_dim;
  std::vector<float> cos_tab, sin_tab;
  MiniMaxH3VideoVaeRope(latent_t, latent_h, latent_w, num_suffix, rot_dim, config.rope_theta,
                        &cos_tab, &sin_tab);
  const int64_t half = rot_dim / 2;
  std::vector<float> rope_cache(static_cast<size_t>(seq * rot_dim));
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t i = 0; i < half; ++i) {
      rope_cache[static_cast<size_t>(s * rot_dim + i)] =
          cos_tab[static_cast<size_t>(s * rot_dim + i)];
      rope_cache[static_cast<size_t>(s * rot_dim + half + i)] =
          sin_tab[static_cast<size_t>(s * rot_dim + i)];
    }
  }
  DBuf rope_dev(d, DType::kF32, {seq, rot_dim}, rope_cache.data());
  std::vector<int32_t> positions_host(static_cast<size_t>(seq));
  for (int64_t s = 0; s < seq; ++s) positions_host[static_cast<size_t>(s)] = static_cast<int32_t>(s);
  DBuf positions(d, DType::kI32, {seq}, positions_host.data());

  // One document of length `seq`, bidirectional.
  const int32_t cu_seqlens[2] = {0, static_cast<int32_t>(seq)};

  vt::RmsNormArgs norm_args;
  norm_args.eps = static_cast<float>(config.block.eps);

  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const MiniMaxH3VideoVaeDeviceBlock& b = staged.blocks[static_cast<size_t>(layer)];

    // --- attention branch ---
    {
      DBuf normed(d, DType::kF32, {seq, dim});
      vt::RmsNorm(d.q, normed.t(), hidden.t(), b.norm1, norm_args);

      DBuf qkv(d, DType::kF32, {seq, 3 * inner});
      LinearDev(d, normed.t(), seq, dim, b.qkv_weight, &b.qkv_bias, qkv.t());

      DBuf qb(d, DType::kF32, {seq, inner});
      DBuf kb(d, DType::kF32, {seq, inner});
      DBuf vb(d, DType::kF32, {seq, inner});
      vt::QkvSplit(d.q, qb.t(), kb.t(), vb.t(), qkv.t());

      // qk RMSNorm over dim_head, WITHOUT affine -- a ones weight is the exact
      // elementwise_affine=False case and costs one multiply by 1.0.
      Tensor qn = dense_attn::Reshape(qb.t(), {seq * heads, dim_head});
      Tensor kn = dense_attn::Reshape(kb.t(), {seq * heads, dim_head});
      vt::RmsNorm(d.q, qn, qn, staged.qk_norm_ones, norm_args);
      vt::RmsNorm(d.q, kn, kn, staged.qk_norm_ones, norm_args);

      if (rot_dim > 0) {
        Tensor q3 = dense_attn::Reshape(qb.t(), {seq, heads, dim_head});
        Tensor k3 = dense_attn::Reshape(kb.t(), {seq, heads, dim_head});
        vt::RopeArgs rope;
        rope.rotary_dim = static_cast<int>(rot_dim);
        rope.is_neox_style = true;
        vt::RopeFromCache(d.q, q3, &k3, positions.t(), rope_dev.t(), rope);
      }

      Tensor tq = dense_attn::Reshape(qb.t(), {seq, heads, dim_head});
      Tensor tk = dense_attn::Reshape(kb.t(), {seq, heads, dim_head});
      Tensor tv = dense_attn::Reshape(vb.t(), {seq, heads, dim_head});
      DBuf attn(d, DType::kF32, {seq, heads, dim_head});
      vt::DFlashBlockAttentionArgs args;
      args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim_head)));
      args.causal = false;
      args.sliding_window = 0;
      args.cu_seqlens = cu_seqlens;
      args.num_reqs = 1;
      vt::DFlashBlockAttention(d.q, attn.t(), tq, tk, tv, args);

      // scale1 is already folded into out_weight/out_bias, so this is a plain add.
      DBuf projected(d, DType::kF32, {seq, dim});
      LinearDev(d, attn.t(), seq, inner, b.out_weight, &b.out_bias, projected.t());
      vt::Add(d.q, hidden.t(), hidden.t(), projected.t());
    }

    // --- feed-forward branch (gated SiLU) ---
    {
      DBuf normed(d, DType::kF32, {seq, dim});
      vt::RmsNorm(d.q, normed.t(), hidden.t(), b.norm2, norm_args);

      // w1 emits [gate | up] per row, which is exactly vt::SiluAndMul's layout.
      DBuf fused(d, DType::kF32, {seq, 2 * ff_inner});
      LinearDev(d, normed.t(), seq, dim, b.w1_weight, &b.w1_bias, fused.t());
      DBuf act(d, DType::kF32, {seq, ff_inner});
      vt::SiluAndMul(d.q, act.t(), fused.t());

      DBuf projected(d, DType::kF32, {seq, dim});
      LinearDev(d, act.t(), seq, ff_inner, b.w2_weight, &b.w2_bias, projected.t());
      vt::Add(d.q, hidden.t(), hidden.t(), projected.t());
    }
  }

  // norm_out is a LAYER norm here, not an RMS norm.
  {
    vt::LayerNormArgs ln;
    ln.eps = static_cast<float>(config.block.eps);
    vt::LayerNorm(d.q, hidden.t(), hidden.t(), &staged.norm_out_weight,
                  staged.has_norm_out_bias ? &staged.norm_out_bias : nullptr, ln);
  }

  DBuf projected(d, DType::kF32, {seq, patch_dim});
  LinearDev(d, hidden.t(), seq, dim, staged.proj_out_weight, &staged.proj_out_bias,
            projected.t());

  // Only the PATCH rows are unpacked; the register and cls rows are dropped.
  std::vector<float> projected_host(static_cast<size_t>(patches * patch_dim));
  backend.Copy(queue, projected_host.data(), projected.t().data,
               projected_host.size() * sizeof(float));
  backend.Synchronize(queue);

  // _unpack_tensors_3d: [patches, C*pt*ps*ps] -> [C, T*pt, H*ps, W*ps].
  const int64_t pt = config.patch_size_t, ps = config.patch_size;
  const int64_t video_t = latent_t * pt, video_h = latent_h * ps, video_w = latent_w * ps;
  std::vector<float> frames(
      static_cast<size_t>(config.out_channels * video_t * video_h * video_w));
  for (int64_t ti = 0; ti < latent_t; ++ti) {
    for (int64_t hi = 0; hi < latent_h; ++hi) {
      for (int64_t wi = 0; wi < latent_w; ++wi) {
        const int64_t row = (ti * latent_h + hi) * latent_w + wi;
        int64_t k = 0;
        for (int64_t c = 0; c < config.out_channels; ++c) {
          for (int64_t r = 0; r < pt; ++r) {
            for (int64_t p = 0; p < ps; ++p) {
              for (int64_t q = 0; q < ps; ++q) {
                const int64_t dst =
                    ((c * video_t + ti * pt + r) * video_h + hi * ps + p) * video_w + wi * ps + q;
                frames[static_cast<size_t>(dst)] =
                    projected_host[static_cast<size_t>(row * patch_dim + k)];
                ++k;
              }
            }
          }
        }
      }
    }
  }
  if (out_shape != nullptr) {
    out_shape->channels = config.out_channels;
    out_shape->t = video_t;
    out_shape->h = video_h;
    out_shape->w = video_w;
  }
  return frames;
}


// ---------------------------------------------------------------------------
// SPATIAL TILING (klvae.py:192-250)
//
// This is NOT a memory strategy, which is how it first reads. The ViT3D decoder's
// RoPE coordinates are LENGTH-NORMALIZED -- `2*((i + 0.5)/n) - 1` over whatever
// grid it is handed (MiniMaxH3VideoVaeRope) -- so the grid EXTENT is part of the
// input, not an implementation detail. Upstream always decodes in 256-pixel tiles
// (16 latent units); handing the decoder a 32x32 latent instead of the 16x16 it
// was trained on gives every token a position the model has never seen.
//
// Observed, not theorized: at 512x512 (which upstream plans as 3 tiles per axis)
// an untiled decode produced globally-correct frames covered in a grid of small
// squares, while 256x256 -- the one size where tile_size >= input, so tiled and
// untiled are the SAME computation -- came out clean.
//
// Every plan value is a whole number of `vae_ratio` units (tile_size 256 = 16*16,
// and the slack is distributed in whole vae_ratio steps), so the pixel plan slices
// the latent exactly.
namespace {

// Cross-fade `b` onto `a` along the LAST axis of a [C, T, H, W] buffer, reusing the
// gated 1-D MiniMaxH3BlendTiles per scanline. Result width is wa - extent + wb.
std::vector<float> BlendAlongW(const std::vector<float>& a, const std::vector<float>& b, int64_t c,
                               int64_t t, int64_t h, int64_t wa, int64_t wb, int64_t extent) {
  const int64_t wout = wa - extent + wb;
  std::vector<float> out(static_cast<size_t>(c * t * h * wout));
  std::vector<float> la(static_cast<size_t>(wa)), lb(static_cast<size_t>(wb));
  for (int64_t i = 0; i < c * t * h; ++i) {
    for (int64_t x = 0; x < wa; ++x) la[static_cast<size_t>(x)] = a[static_cast<size_t>(i * wa + x)];
    for (int64_t x = 0; x < wb; ++x) lb[static_cast<size_t>(x)] = b[static_cast<size_t>(i * wb + x)];
    const std::vector<float> merged = MiniMaxH3BlendTiles(la, lb, extent);
    // `merged` covers the region starting at (wa - extent); everything before it is
    // `a` untouched.
    for (int64_t x = 0; x < wa - extent; ++x) {
      out[static_cast<size_t>(i * wout + x)] = la[static_cast<size_t>(x)];
    }
    for (size_t x = 0; x < merged.size(); ++x) {
      out[static_cast<size_t>(i * wout + wa - extent) + x] = merged[x];
    }
  }
  return out;
}

// The same cross-fade along H. Lines are strided by W, so they are gathered.
std::vector<float> BlendAlongH(const std::vector<float>& a, const std::vector<float>& b, int64_t c,
                               int64_t t, int64_t ha, int64_t hb, int64_t w, int64_t extent) {
  const int64_t hout = ha - extent + hb;
  std::vector<float> out(static_cast<size_t>(c * t * hout * w));
  std::vector<float> la(static_cast<size_t>(ha)), lb(static_cast<size_t>(hb));
  for (int64_t plane = 0; plane < c * t; ++plane) {
    for (int64_t x = 0; x < w; ++x) {
      for (int64_t y = 0; y < ha; ++y) {
        la[static_cast<size_t>(y)] = a[static_cast<size_t>((plane * ha + y) * w + x)];
      }
      for (int64_t y = 0; y < hb; ++y) {
        lb[static_cast<size_t>(y)] = b[static_cast<size_t>((plane * hb + y) * w + x)];
      }
      const std::vector<float> merged = MiniMaxH3BlendTiles(la, lb, extent);
      for (int64_t y = 0; y < ha - extent; ++y) {
        out[static_cast<size_t>((plane * hout + y) * w + x)] = la[static_cast<size_t>(y)];
      }
      for (size_t y = 0; y < merged.size(); ++y) {
        out[static_cast<size_t>((plane * hout + (ha - extent) + static_cast<int64_t>(y)) * w + x)] =
            merged[y];
      }
    }
  }
  return out;
}

}  // namespace

std::vector<float> MiniMaxH3VideoVaeDecodeTiledDevice(
    vt::Device device, const MiniMaxH3VideoVaeDecoderConfig& config,
    const MiniMaxH3VideoVaeDeviceWeights& staged, const std::vector<float>& latent,
    int64_t latent_t, int64_t latent_h, int64_t latent_w, MiniMaxH3VideoFrameShape* out_shape) {
  const int64_t ratio = kMiniMaxH3VaeRatio;
  const MiniMaxH3TilePlan plan_h = MiniMaxH3SplitTiles(
      latent_h * ratio, kMiniMaxH3VaeTileSize, kMiniMaxH3VaeTileOverlapMin, ratio);
  const MiniMaxH3TilePlan plan_w = MiniMaxH3SplitTiles(
      latent_w * ratio, kMiniMaxH3VaeTileSize, kMiniMaxH3VaeTileOverlapMin, ratio);

  // A single tile is the untiled decode, bit for bit -- no slicing, no blending.
  if (plan_h.starts.size() == 1 && plan_w.starts.size() == 1) {
    return MiniMaxH3VideoVaeDecodeDevice(device, config, staged, latent, latent_t, latent_h,
                                         latent_w, out_shape);
  }

  const int64_t channels = config.in_channels;
  const int64_t ps = config.patch_size, pt = config.patch_size_t;
  const int64_t frames_t = latent_t * pt;

  std::vector<float> assembled;  // [out_channels, frames_t, H, W] as it grows
  int64_t assembled_h = 0, assembled_w = 0;

  for (size_t i = 0; i < plan_h.starts.size(); ++i) {
    const int64_t h0 = plan_h.starts[i] / ratio, hl = plan_h.lengths[i] / ratio;
    std::vector<float> row;
    int64_t row_w = 0;
    for (size_t j = 0; j < plan_w.starts.size(); ++j) {
      const int64_t w0 = plan_w.starts[j] / ratio, wl = plan_w.lengths[j] / ratio;
      VT_CHECK(h0 + hl <= latent_h && w0 + wl <= latent_w,
               "minimax_h3 video vae tiling: tile exceeds the latent grid");

      std::vector<float> sub(static_cast<size_t>(channels * latent_t * hl * wl));
      for (int64_t c = 0; c < channels; ++c) {
        for (int64_t tt = 0; tt < latent_t; ++tt) {
          for (int64_t y = 0; y < hl; ++y) {
            for (int64_t x = 0; x < wl; ++x) {
              sub[static_cast<size_t>(((c * latent_t + tt) * hl + y) * wl + x)] =
                  latent[static_cast<size_t>(((c * latent_t + tt) * latent_h + h0 + y) * latent_w +
                                             w0 + x)];
            }
          }
        }
      }
      MiniMaxH3VideoFrameShape tile_shape;
      const std::vector<float> tile = MiniMaxH3VideoVaeDecodeDevice(
          device, config, staged, sub, latent_t, hl, wl, &tile_shape);
      const int64_t tw = wl * ps;
      if (j == 0) {
        row = tile;
        row_w = tw;
      } else {
        // The plan is in CANVAS pixels (vae_ratio per latent unit); the blend
        // happens in the DECODER'S output pixels (patch_size per latent unit).
        // They are both 16 on the real checkpoint, so converting through latent
        // units matters only for reduced-dimension configs -- which is exactly
        // where the gates run.
        const int64_t ov = (plan_w.overlaps[j - 1] / ratio) * ps;
        row = BlendAlongW(row, tile, config.out_channels, frames_t, hl * ps, row_w, tw, ov);
        row_w = row_w - ov + tw;
      }
    }
    const int64_t th = hl * ps;
    if (i == 0) {
      assembled = std::move(row);
      assembled_h = th;
      assembled_w = row_w;
    } else {
      VT_CHECK(row_w == assembled_w, "minimax_h3 video vae tiling: tile rows disagree on width");
      const int64_t ov = (plan_h.overlaps[i - 1] / ratio) * ps;
      assembled = BlendAlongH(assembled, row, config.out_channels, frames_t, assembled_h, th,
                              assembled_w, ov);
      assembled_h = assembled_h - ov + th;
    }
  }

  if (out_shape != nullptr) {
    out_shape->channels = config.out_channels;
    out_shape->t = frames_t;
    out_shape->h = assembled_h;
    out_shape->w = assembled_w;
  }
  return assembled;
}


// ---------------------------------------------------------------------------
// TEMPORAL CHUNKING (klvae.py decode_temporal, :678-786)
//
// This is the real shape of upstream's video decode, and the one thing our port
// was missing: `decode_base` routes video through `decode_temporal`, NOT through
// a single pass. The ViT sees `tokens_chunk_size + token_overlap` temporal tokens
// at a time -- 7 for the shipped config -- never the whole latent.
//
// It matters for the same reason the spatial extent does: the decoder's RoPE is
// LENGTH-NORMALIZED over the grid it is handed (MiniMaxH3VideoVaeRope takes
// latent_t/h/w), so the temporal EXTENT is part of the input. Handing it 12
// tokens when it was trained on 7 gives every token a position the model never
// saw. Every reduced-dimension gate in this tree runs at latent_t = 2 -- below one
// chunk, where chunked and unchunked decode are the SAME computation -- which is
// exactly why the suite could not see this.
//
// Shipped config (clip_length 17, token_drop 3, vae_ratio_t 4) gives
// tokens_chunk_size 5, token_overlap 2, frame_pre_padding 3, frame_overlap 5.
// isolated_first_frame / isolated_last_frame default FALSE and the shipped config
// sets neither, so the head/tail isolation branches are not taken; they are
// deliberately NOT implemented rather than implemented untested.
std::vector<float> MiniMaxH3VideoVaeDecodeTemporalDevice(
    vt::Device device, const MiniMaxH3VideoVaeDecoderConfig& config,
    const MiniMaxH3VideoVaeDeviceWeights& staged, const std::vector<float>& latent,
    int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t target_frames,
    MiniMaxH3VideoFrameShape* out_shape) {
  const int64_t chan = config.in_channels;
  const int64_t ratio_t = config.vae_ratio_t;
  const int64_t chunk_tokens = config.tokens_chunk_size();
  const int64_t overlap_tokens = config.token_overlap();
  const int64_t pre_pad = config.frame_pre_padding();
  const int64_t frame_ov = config.frame_overlap();
  VT_CHECK(chunk_tokens > 0 && ratio_t > 0, "minimax_h3 temporal decode: bad chunk geometry");

  // pseudo_total_tokens = T - isolated + token_drop, padded up to a whole number
  // of chunks by REPEATING the last latent frame (klvae.py:688-696).
  int64_t pseudo_total = latent_t + config.token_drop;
  const int64_t remainder = pseudo_total % chunk_tokens;
  const int64_t pad_tokens = remainder == 0 ? 0 : chunk_tokens - remainder;
  pseudo_total += pad_tokens;
  const int64_t num_chunks = pseudo_total / chunk_tokens - (config.token_drop > 0 ? 1 : 0);
  // A latent shorter than one chunk yields num_chunks == 0 (latent_t 2 gives
  // pseudo 5, i.e. exactly one chunk, minus the token_drop chunk). There is no
  // chunking to do, so decode it directly rather than aborting -- this is the
  // regime the reduced-dimension gates run in.
  if (num_chunks < 1) {
    MiniMaxH3VideoFrameShape s{};
    std::vector<float> whole =
        config.decoder_tiling
            ? MiniMaxH3VideoVaeDecodeTiledDevice(device, config, staged, latent, latent_t, latent_h,
                                                 latent_w, &s)
            : MiniMaxH3VideoVaeDecodeDevice(device, config, staged, latent, latent_t, latent_h,
                                            latent_w, &s);
    if (out_shape != nullptr) *out_shape = s;
    return whole;
  }

  std::vector<float> z = latent;
  int64_t z_t = latent_t;
  if (pad_tokens > 0) {
    // repeat the LAST temporal token pad_tokens times
    std::vector<float> padded(static_cast<size_t>(chan * (z_t + pad_tokens) * latent_h * latent_w));
    const int64_t plane = latent_h * latent_w;
    for (int64_t c = 0; c < chan; ++c) {
      for (int64_t tt = 0; tt < z_t + pad_tokens; ++tt) {
        const int64_t src_t = std::min(tt, z_t - 1);
        for (int64_t i = 0; i < plane; ++i) {
          padded[static_cast<size_t>((c * (z_t + pad_tokens) + tt) * plane + i)] =
              z[static_cast<size_t>((c * z_t + src_t) * plane + i)];
        }
      }
    }
    z = std::move(padded);
    z_t += pad_tokens;
  }

  const int64_t ps = config.patch_size;
  const int64_t out_h = latent_h * ps, out_w = latent_w * ps;
  const int64_t chunk_dec = chunk_tokens * ratio_t;
  const int64_t plane = out_h * out_w;
  const int64_t chans = config.out_channels;

  // Decode each chunk, then walk upstream's j-loop: slice `chunk_dec` frames at a
  // time, drop `frame_pre_padding` leading frames, cross-fade the carried overlap.
  std::vector<std::vector<float>> pieces;  // each [C, f, H, W]
  std::vector<int64_t> piece_frames;
  std::vector<float> carry;  // dec_overlap
  int64_t carry_frames = 0;

  for (int64_t i = 0; i < num_chunks; ++i) {
    const int64_t t0 = i * chunk_tokens;
    const int64_t t1 = std::min(t0 + chunk_tokens + overlap_tokens, z_t);
    const int64_t nt = t1 - t0;
    VT_CHECK(nt > 0, "minimax_h3 temporal decode: empty chunk");

    std::vector<float> sub(static_cast<size_t>(chan * nt * latent_h * latent_w));
    const int64_t lplane = latent_h * latent_w;
    for (int64_t c = 0; c < chan; ++c) {
      for (int64_t k = 0; k < nt; ++k) {
        for (int64_t e = 0; e < lplane; ++e) {
          sub[static_cast<size_t>((c * nt + k) * lplane + e)] =
              z[static_cast<size_t>((c * z_t + t0 + k) * lplane + e)];
        }
      }
    }
    // Upstream's chunk loop calls `self._adaptive_decode(clip_z)` -- so when
    // decoder tiling is on, each TEMPORAL chunk is ALSO spatially tiled. The two
    // compose; they are not alternatives. Tiled decode falls through to the
    // untiled one bit-for-bit when the canvas fits in a single tile, so this is a
    // no-op below 256 px and matches upstream's tiled mode above it.
    MiniMaxH3VideoFrameShape cs{};
    const std::vector<float> dec =
        config.decoder_tiling
            ? MiniMaxH3VideoVaeDecodeTiledDevice(device, config, staged, sub, nt, latent_h,
                                                 latent_w, &cs)
            : MiniMaxH3VideoVaeDecodeDevice(device, config, staged, sub, nt, latent_h, latent_w,
                                            &cs);
    const int64_t dec_frames = cs.t;

    for (int64_t j = 0; j < (config.token_drop > 0 ? 2 : 1); ++j) {
      const int64_t f0 = j * chunk_dec;
      if (f0 >= dec_frames) break;
      const int64_t f1 = std::min(f0 + chunk_dec, dec_frames);
      const int64_t keep0 = f0 + pre_pad;
      if (keep0 >= f1) continue;
      const int64_t nf = f1 - keep0;

      std::vector<float> piece(static_cast<size_t>(chans * nf * plane));
      for (int64_t c = 0; c < chans; ++c) {
        for (int64_t f = 0; f < nf; ++f) {
          for (int64_t e = 0; e < plane; ++e) {
            piece[static_cast<size_t>((c * nf + f) * plane + e)] =
                dec[static_cast<size_t>((c * dec_frames + keep0 + f) * plane + e)];
          }
        }
      }

      if (j == 0) {
        if (carry_frames > 0) {
          // blend(dec_overlap, piece, frame_overlap) along the FRAME axis
          const int64_t ext = std::min({carry_frames, nf, frame_ov});
          for (int64_t c = 0; c < chans; ++c) {
            for (int64_t e = 0; e < plane; ++e) {
              for (int64_t f = 0; f < ext; ++f) {
                const double wb = static_cast<double>(f) / static_cast<double>(ext);
                const float a_v = carry[static_cast<size_t>(
                    (c * carry_frames + carry_frames - ext + f) * plane + e)];
                float& b_v = piece[static_cast<size_t>((c * nf + f) * plane + e)];
                b_v = static_cast<float>(a_v * (1.0 - wb) + b_v * wb);
              }
            }
          }
          carry.clear();
          carry_frames = 0;
        }
        pieces.push_back(std::move(piece));
        piece_frames.push_back(nf);
      } else {
        carry = std::move(piece);
        carry_frames = nf;
      }
    }
  }
  if (carry_frames > 0) {
    pieces.push_back(std::move(carry));
    piece_frames.push_back(carry_frames);
  }

  int64_t total = 0;
  for (const int64_t f : piece_frames) total += f;
  std::vector<float> out(static_cast<size_t>(chans * total * plane));
  int64_t at = 0;
  for (size_t k = 0; k < pieces.size(); ++k) {
    const int64_t nf = piece_frames[k];
    for (int64_t c = 0; c < chans; ++c) {
      for (int64_t f = 0; f < nf; ++f) {
        for (int64_t e = 0; e < plane; ++e) {
          out[static_cast<size_t>((c * total + at + f) * plane + e)] =
              pieces[k][static_cast<size_t>((c * nf + f) * plane + e)];
        }
      }
    }
    at += nf;
  }

  // trim_output (klvae.py:452-459): non-causal decoders CENTER-crop to the frame
  // count the request asked for.
  if (target_frames > 0 && target_frames < total) {
    const int64_t start = (total - target_frames) / 2;
    std::vector<float> trimmed(static_cast<size_t>(chans * target_frames * plane));
    for (int64_t c = 0; c < chans; ++c) {
      for (int64_t f = 0; f < target_frames; ++f) {
        for (int64_t e = 0; e < plane; ++e) {
          trimmed[static_cast<size_t>((c * target_frames + f) * plane + e)] =
              out[static_cast<size_t>((c * total + start + f) * plane + e)];
        }
      }
    }
    out = std::move(trimmed);
    total = target_frames;
  }

  if (out_shape != nullptr) {
    out_shape->channels = chans;
    out_shape->t = total;
    out_shape->h = out_h;
    out_shape->w = out_w;
  }
  return out;
}

}  // namespace vllm
