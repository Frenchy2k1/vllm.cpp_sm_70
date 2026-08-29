// Muse Glimmer perception encoder — W3 forward. See
// include/vllm/model_executor/models/muse_glimmer_vision.h for the full port
// map, the off-pin honesty statement, and the four silent traps this file is
// written around.
//
// Ported from vllm PR #51655 head `075d645af`,
// vllm/model_executor/models/muse_glimmer.py:
//   MuseGlimmerVisionEncoder.forward     :937-1034
//   MuseGlimmerVisionBlock.forward       :669-689
//   MuseGlimmerVisionAttention.forward   :602-638
//   MuseGlimmerVisionMLP.forward         :647-648
//   MuseGlimmerVisionAdapter.forward     :1042-1043
//   _patchify :902-935, _get_pos_emb :761-820, _make_2d_rope :741-759,
//   _get_sparse_permutation :844-867, _pixel_shuffle_downsample :822-842
// plus ApplyRotaryEmb.forward_static (rotary_embedding/common.py:143-184,
// is_neox_style=True, fp32 compute).
//
// Composed from the public vt:: ops (MatmulBT / Add / LayerNorm / GeluErf /
// RopeFromCache / AttentionDenseFlash / IndexSelect) and the shared merged-QKV seam
// vllm::models::FusedMergedQkvBiasSplit — no hand-rolled parallel path. The
// host precomputes (patchify, positional interpolation, the 2D-RoPE cos|sin
// table, the window permutation, the pixel shuffle) are deterministic f32, as
// they are for the Qwen3-VL tower; upstream runs the first two on GPU.
//
// STRUCTURE, NOT SPEED. The tower converts and uploads its weights inside the
// forward. The Qwen3-VL tower learned that this dominates a per-image encode
// and split out PrepareVisionDeviceWeights; the same split belongs here, but it
// is deliberately NOT done in W3: there is no oracle that can run Muse Glimmer,
// so there is no denominator against which any such change could be justified
// or measured. W4 owns the loader and residency.
#include "vllm/model_executor/models/muse_glimmer_vision.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/merged_qkv_fold.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm::multimodal {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// --- RAII device buffer (movable, so blocks of them live in a vector) --------
struct Buf {
  Backend* b = nullptr;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t{};

  Buf() = default;
  Buf(Backend& backend, Queue& q, DType dt, const std::vector<int64_t>& shape,
      const void* host = nullptr)
      : b(&backend) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p = b->Alloc(bytes == 0 ? 1 : bytes);
    t.data = p;
    t.dtype = dt;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    if (host != nullptr && bytes != 0) b->Copy(q, p, host, bytes);
  }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept { *this = std::move(o); }
  Buf& operator=(Buf&& o) noexcept {
    if (this != &o) {
      if (b != nullptr && p != nullptr) b->Free(p);
      b = o.b;
      p = o.p;
      bytes = o.bytes;
      t = o.t;
      o.b = nullptr;
      o.p = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
  ~Buf() {
    if (b != nullptr && p != nullptr) b->Free(p);
  }
  Tensor& tensor() { return t; }
  const Tensor& tensor() const { return t; }
  void Download(Queue& q, void* dst) const {
    b->Copy(q, dst, p, bytes);
    b->Synchronize(q);
  }
};

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}

// Allocate [shape] in `dt` and fill it from host f32, converting when needed.
Buf Upload(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
           const std::vector<float>& host) {
  if (dt == DType::kBF16) {
    const std::vector<uint16_t> bf = ToBf16(host);
    return Buf(b, q, dt, shape, bf.data());
  }
  return Buf(b, q, dt, shape, host.data());
}

std::vector<float> DownloadF32(const Buf& buf, Queue& q, DType dt, size_t numel) {
  std::vector<float> out(numel);
  if (dt == DType::kBF16) {
    std::vector<uint16_t> tmp(numel);
    buf.Download(q, tmp.data());
    for (size_t i = 0; i < numel; ++i) out[i] = vt::BF16ToF32(tmp[i]);
  } else {
    buf.Download(q, out.data());
  }
  return out;
}

// A contiguous row-range view of `src` (rows [row, row+rows) of dim 0).
Tensor RowSlice(const Tensor& src, int64_t row, int64_t rows) {
  Tensor s = src;
  s.shape[0] = rows;
  int64_t inner = 1;
  for (int i = 1; i < src.rank; ++i) inner *= src.shape[i];
  s.data = static_cast<char*>(src.data) +
           static_cast<size_t>(row * inner) * vt::SizeOf(src.dtype);
  return s;
}

// Reinterpret a contiguous [rows, a, b] tensor as [rows, a*b] or the reverse.
Tensor Reshape2(const Tensor& src, int64_t d0, int64_t d1) {
  Tensor s = src;
  s.rank = 2;
  s.shape[0] = d0;
  s.shape[1] = d1;
  s.stride[0] = d1;
  s.stride[1] = 1;
  return s;
}

Tensor Reshape3(const Tensor& src, int64_t d0, int64_t d1, int64_t d2) {
  Tensor s = src;
  s.rank = 3;
  s.shape[0] = d0;
  s.shape[1] = d1;
  s.shape[2] = d2;
  s.stride[0] = d1 * d2;
  s.stride[1] = d2;
  s.stride[2] = 1;
  return s;
}

// out[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]).
void LinearBias(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                const Tensor* bias) {
  vt::MatmulBT(q, out, x, w);
  if (bias != nullptr) vt::Add(q, out, out, *bias);
}

// The per-layer device weights.
struct DevBlock {
  Buf ln_1_w, ln_1_b, ln_2_w, ln_2_b;
  Buf qkv_w, qkv_b;
  Buf o_w, o_b;
  Buf c_fc_w, c_fc_b;
  Buf c_proj_w, c_proj_b;
};

}  // namespace

bool MuseGlimmerVisionConfig::has_window_layers() const {
  // muse_glimmer.py:941-943 — anything that is not "full_attention" is windowed.
  for (const std::string& t : layer_types)
    if (t != "full_attention") return true;
  return false;
}

// --- _patchify (muse_glimmer.py:902-935) -------------------------------------
// The patch vector is (t, c, ph, pw)-major: `permute(0,2,3,1,4,5)` puts (c,ph,pw)
// innermost per grid cell, and the temporal axis is inserted at dim 3 — by
// BROADCAST for a 3-channel still (`unsqueeze(3).expand`, :917-919) or by
// STACKING the per-frame patches for a patch_temporal*3-channel clip (:920-931).
std::vector<float> MuseGlimmerVisionPatchify(const MuseGlimmerVisionImage& image,
                                             const MuseGlimmerVisionConfig& cfg) {
  const int64_t ps = cfg.patch_size;
  const int64_t tp = cfg.patch_temporal;
  VT_CHECK(image.height % ps == 0 && image.width % ps == 0,
           "MuseGlimmer vision input must divide the patch size");
  const bool still = image.channels == 3;
  VT_CHECK(still || image.channels == tp * 3,
           "MuseGlimmer vision input channels must be 3 or patch_temporal*3");
  const int64_t gh = image.height / ps;
  const int64_t gw = image.width / ps;
  const int64_t pd = cfg.patch_dim();
  VT_CHECK(static_cast<int64_t>(image.pixels.size()) ==
               image.channels * image.height * image.width,
           "MuseGlimmer vision pixel buffer does not match [C,H,W]");

  std::vector<float> out(static_cast<size_t>(gh * gw) * static_cast<size_t>(pd));
  const int64_t plane = image.height * image.width;
  for (int64_t gy = 0; gy < gh; ++gy) {
    for (int64_t gx = 0; gx < gw; ++gx) {
      float* dst = &out[static_cast<size_t>((gy * gw + gx) * pd)];
      int64_t k = 0;
      for (int64_t t = 0; t < tp; ++t) {
        for (int64_t c = 0; c < 3; ++c) {
          // A still image reuses channel `c` for every temporal slot; a clip
          // takes frame `t`'s channel block.
          const int64_t src_c = still ? c : t * 3 + c;
          for (int64_t py = 0; py < ps; ++py) {
            for (int64_t px = 0; px < ps; ++px) {
              const int64_t y = gy * ps + py;
              const int64_t x = gx * ps + px;
              dst[k++] = image.pixels[static_cast<size_t>(src_c * plane + y * image.width + x)];
            }
          }
        }
      }
    }
  }
  return out;
}

// --- _get_pos_emb (muse_glimmer.py:761-820) ----------------------------------
// The HALF-PIXEL convention `(i + 0.5) * (table/grid) - 0.5` maps grid CENTRES
// onto table CENTRES; dropping either half deforms the sampling by half a table
// cell, which no shape or range check can see. Corners that fall outside the
// table are masked to weight 0 (NOT clamped-and-weighted), so an edge sample
// keeps only its in-range corners and the interpolation is not renormalized.
std::vector<float> MuseGlimmerVisionPosEmbedInterpolate(
    const std::vector<float>& pos_emb, int64_t grid_height, int64_t grid_width,
    const MuseGlimmerVisionConfig& cfg) {
  const int64_t H = cfg.hidden_size;
  const int64_t th = cfg.pos_emb_height;
  const int64_t tw = cfg.pos_emb_width;
  VT_CHECK(static_cast<int64_t>(pos_emb.size()) == th * tw * H,
           "MuseGlimmer vision positional table has the wrong shape");

  std::vector<float> h_grid(static_cast<size_t>(grid_height));
  std::vector<float> w_grid(static_cast<size_t>(grid_width));
  for (int64_t i = 0; i < grid_height; ++i)
    h_grid[static_cast<size_t>(i)] =
        (static_cast<float>(i) + 0.5f) *
            (static_cast<float>(th) / static_cast<float>(grid_height)) -
        0.5f;
  for (int64_t i = 0; i < grid_width; ++i)
    w_grid[static_cast<size_t>(i)] =
        (static_cast<float>(i) + 0.5f) *
            (static_cast<float>(tw) / static_cast<float>(grid_width)) -
        0.5f;

  std::vector<float> out(static_cast<size_t>(grid_height * grid_width) *
                         static_cast<size_t>(H));
  for (int64_t y = 0; y < grid_height; ++y) {
    const float hg = h_grid[static_cast<size_t>(y)];
    const int64_t h_floor = static_cast<int64_t>(std::floor(hg));
    const int64_t h_ceil = h_floor + 1;
    const float h_frac = hg - static_cast<float>(h_floor);
    const bool hf_valid = h_floor >= 0 && h_floor < th;
    const bool hc_valid = h_ceil >= 0 && h_ceil < th;
    const int64_t hf = std::clamp<int64_t>(h_floor, 0, th - 1);
    const int64_t hc = std::clamp<int64_t>(h_ceil, 0, th - 1);
    for (int64_t x = 0; x < grid_width; ++x) {
      const float wg = w_grid[static_cast<size_t>(x)];
      const int64_t w_floor = static_cast<int64_t>(std::floor(wg));
      const int64_t w_ceil = w_floor + 1;
      const float w_frac = wg - static_cast<float>(w_floor);
      const bool wf_valid = w_floor >= 0 && w_floor < tw;
      const bool wc_valid = w_ceil >= 0 && w_ceil < tw;
      const int64_t wf = std::clamp<int64_t>(w_floor, 0, tw - 1);
      const int64_t wc = std::clamp<int64_t>(w_ceil, 0, tw - 1);

      const float w00 = (1.0f - h_frac) * (1.0f - w_frac) *
                        static_cast<float>(hf_valid && wf_valid);
      const float w01 = (1.0f - h_frac) * w_frac * static_cast<float>(hf_valid && wc_valid);
      const float w10 = h_frac * (1.0f - w_frac) * static_cast<float>(hc_valid && wf_valid);
      const float w11 = h_frac * w_frac * static_cast<float>(hc_valid && wc_valid);

      const float* e00 = &pos_emb[static_cast<size_t>((hf * tw + wf) * H)];
      const float* e01 = &pos_emb[static_cast<size_t>((hf * tw + wc) * H)];
      const float* e10 = &pos_emb[static_cast<size_t>((hc * tw + wf) * H)];
      const float* e11 = &pos_emb[static_cast<size_t>((hc * tw + wc) * H)];
      float* dst = &out[static_cast<size_t>((y * grid_width + x) * H)];
      for (int64_t d = 0; d < H; ++d) {
        const size_t i = static_cast<size_t>(d);
        dst[i] = w00 * e00[i] + w01 * e01[i] + w10 * e10[i] + w11 * e11[i];
      }
    }
  }
  return out;
}

// --- _make_2d_rope (muse_glimmer.py:741-759) ---------------------------------
// WIDTH FIRST: `freqs = cat([freq_w, freq_h])` (:757). The positions are also
// 1-BASED (`arange(1, grid+1)`, :750-751). Transposing the two halves — or
// zero-basing them — leaves every shape, norm and value range intact.
void MuseGlimmerVisionRopeCosSin(int64_t grid_height, int64_t grid_width,
                                 const MuseGlimmerVisionConfig& cfg,
                                 std::vector<float>* cos, std::vector<float>* sin) {
  const int64_t hd = cfg.head_dim();
  VT_CHECK(hd % 4 == 0, "MuseGlimmer vision head dimension must be divisible by 4");
  const int64_t spatial_dim = hd / 2;
  const int64_t nfreq = spatial_dim / 2;  // arange(0, spatial_dim, 2)
  const int64_t half = hd / 2;            // == 2 * nfreq, the cos|sin width
  std::vector<double> inv_freq(static_cast<size_t>(nfreq));
  for (int64_t i = 0; i < nfreq; ++i)
    inv_freq[static_cast<size_t>(i)] =
        1.0 / std::pow(10000.0, static_cast<double>(2 * i) / static_cast<double>(spatial_dim));

  const int64_t L = grid_height * grid_width;
  cos->assign(static_cast<size_t>(L * half), 0.0f);
  sin->assign(static_cast<size_t>(L * half), 0.0f);
  for (int64_t y = 0; y < grid_height; ++y) {
    for (int64_t x = 0; x < grid_width; ++x) {
      const int64_t r = y * grid_width + x;
      const double hpos = static_cast<double>(y + 1);  // 1-based
      const double wpos = static_cast<double>(x + 1);
      float* cr = &(*cos)[static_cast<size_t>(r * half)];
      float* sr = &(*sin)[static_cast<size_t>(r * half)];
      for (int64_t i = 0; i < nfreq; ++i) {
        const double aw = wpos * inv_freq[static_cast<size_t>(i)];
        const double ah = hpos * inv_freq[static_cast<size_t>(i)];
        cr[i] = static_cast<float>(std::cos(aw));  // width half first
        sr[i] = static_cast<float>(std::sin(aw));
        cr[nfreq + i] = static_cast<float>(std::cos(ah));
        sr[nfreq + i] = static_cast<float>(std::sin(ah));
      }
    }
  }
}

// --- _get_sparse_permutation (muse_glimmer.py:844-867) -----------------------
// The grid is padded to whole pos_emb_height x pos_emb_width blocks with -1,
// walked block-major, and the padding is then DROPPED — so the surviving tokens
// stay contiguous per block and each block's valid count is its attention
// seq_len. Edge blocks are therefore SHORTER, not padded with real tokens.
void MuseGlimmerVisionSparsePermutation(int64_t grid_height, int64_t grid_width,
                                        const MuseGlimmerVisionConfig& cfg,
                                        std::vector<int32_t>* permutation,
                                        std::vector<int32_t>* seq_lens) {
  const int64_t bh = cfg.pos_emb_height;
  const int64_t bw = cfg.pos_emb_width;
  VT_CHECK(bh > 0 && bw > 0, "MuseGlimmer vision pos_emb block must be positive");
  const int64_t nby = (grid_height + bh - 1) / bh;
  const int64_t nbx = (grid_width + bw - 1) / bw;
  permutation->clear();
  seq_lens->clear();
  permutation->reserve(static_cast<size_t>(grid_height * grid_width));
  for (int64_t by = 0; by < nby; ++by) {
    for (int64_t bx = 0; bx < nbx; ++bx) {
      int32_t valid = 0;
      for (int64_t i = 0; i < bh; ++i) {
        for (int64_t j = 0; j < bw; ++j) {
          const int64_t y = by * bh + i;
          const int64_t x = bx * bw + j;
          if (y >= grid_height || x >= grid_width) continue;  // the -1 padding
          permutation->push_back(static_cast<int32_t>(y * grid_width + x));
          ++valid;
        }
      }
      seq_lens->push_back(valid);
    }
  }
}

// --- _pixel_shuffle_downsample (muse_glimmer.py:822-842) ---------------------
// Two steps, and BOTH matter: the permutation gathers each merge x merge
// spatial group into consecutive rows, then `.permute(0, 2, 1)` transposes the
// group so the output is HIDDEN-major (all merge^2 values of channel 0, then of
// channel 1, ...). Dropping the transpose keeps the shape and every value.
std::vector<float> MuseGlimmerVisionPixelShuffle(const std::vector<float>& hidden,
                                                 int64_t grid_height, int64_t grid_width,
                                                 int64_t dim,
                                                 const MuseGlimmerVisionConfig& cfg) {
  const int64_t f = cfg.merge_kernel_size;
  VT_CHECK(grid_height % f == 0 && grid_width % f == 0,
           "MuseGlimmer vision grid must divide merge_kernel_size");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == grid_height * grid_width * dim,
           "MuseGlimmer vision pixel-shuffle input has the wrong shape");
  const int64_t hb = grid_height / f;
  const int64_t wb = grid_width / f;
  const int64_t tokens = hb * wb;
  const int64_t group = f * f;
  std::vector<float> out(static_cast<size_t>(tokens * dim * group));
  for (int64_t bi = 0; bi < hb; ++bi) {
    for (int64_t bj = 0; bj < wb; ++bj) {
      const int64_t t = bi * wb + bj;
      for (int64_t li = 0; li < f; ++li) {
        for (int64_t lj = 0; lj < f; ++lj) {
          const int64_t k = li * f + lj;                          // slot in the group
          const int64_t src = (bi * f + li) * grid_width + bj * f + lj;
          const float* s = &hidden[static_cast<size_t>(src * dim)];
          float* d = &out[static_cast<size_t>(t * dim * group)];
          for (int64_t c = 0; c < dim; ++c)
            d[static_cast<size_t>(c * group + k)] = s[static_cast<size_t>(c)];
        }
      }
    }
  }
  return out;
}

// --- MuseGlimmerVisionEncoder.forward (muse_glimmer.py:937-1034) -------------
std::vector<float> MuseGlimmerVisionForward(const std::vector<MuseGlimmerVisionImage>& images,
                                            const MuseGlimmerVisionWeights& weights,
                                            const MuseGlimmerVisionConfig& cfg,
                                            Backend& backend,
                                            MuseGlimmerVisionCapture* capture) {
  const int64_t H = cfg.hidden_size;
  const int64_t nh = cfg.num_attention_heads;
  const int64_t I = cfg.intermediate_size;
  const int64_t pd = cfg.patch_dim();
  const int64_t nl = cfg.num_hidden_layers;
  VT_CHECK(H % nh == 0, "MuseGlimmer vision hidden size must divide num heads");  // :580
  const int64_t hd = cfg.head_dim();
  VT_CHECK(hd % 4 == 0,
           "MuseGlimmer vision head dimension must be divisible by 4");  // :707-708
  VT_CHECK(static_cast<int64_t>(cfg.layer_types.size()) == nl,
           "MuseGlimmer vision layer_types must match num_hidden_layers");  // :705-706
  VT_CHECK(cfg.output_dim == H * cfg.merge_unit(),
           "MuseGlimmer vision output_dim does not match the pixel-shuffle output");  // :734-739
  VT_CHECK(static_cast<int64_t>(weights.blocks.size()) == nl,
           "MuseGlimmer vision weights do not carry num_hidden_layers blocks");
  VT_CHECK(!images.empty(), "MuseGlimmer vision forward needs at least one image");

  const DType dt = cfg.compute_dtype;
  const bool has_sparse = cfg.has_window_layers();
  Queue q = backend.CreateQueue();

  // --- per-image host precompute (:952-996) ----------------------------------
  struct Meta {
    int64_t grid_height = 0, grid_width = 0, tokens = 0, offset = 0;
    std::vector<int32_t> permutation;  // identity when there are no window layers
  };
  if (capture != nullptr) {
    capture->patchified.clear();
    capture->pos_embeds.clear();
    capture->ln_pre_out.clear();
    capture->block0_out.clear();
  }

  std::vector<Meta> meta;
  std::vector<float> patch_all, pos_all, cache_all;
  std::vector<int32_t> global_seq_lens, sparse_seq_lens;
  int64_t L = 0;
  const int64_t half = hd / 2;

  for (const MuseGlimmerVisionImage& image : images) {
    const int64_t stride = cfg.patch_size * cfg.merge_kernel_size;
    VT_CHECK(image.height % stride == 0 && image.width % stride == 0,
             "MuseGlimmer vision input dimensions must divide the merged patch "
             "stride");  // :955-960
    Meta m;
    m.grid_height = image.height / cfg.patch_size;
    m.grid_width = image.width / cfg.patch_size;
    m.tokens = m.grid_height * m.grid_width;
    m.offset = L;

    const std::vector<float> patched = MuseGlimmerVisionPatchify(image, cfg);
    const std::vector<float> pos =
        MuseGlimmerVisionPosEmbedInterpolate(weights.pos_emb, m.grid_height, m.grid_width, cfg);
    std::vector<float> cos, sin;
    MuseGlimmerVisionRopeCosSin(m.grid_height, m.grid_width, cfg, &cos, &sin);
    if (capture != nullptr) {
      capture->patchified.push_back(patched);
      capture->pos_embeds.push_back(pos);
    }

    if (has_sparse) {
      std::vector<int32_t> seq;
      MuseGlimmerVisionSparsePermutation(m.grid_height, m.grid_width, cfg, &m.permutation,
                                         &seq);
      sparse_seq_lens.insert(sparse_seq_lens.end(), seq.begin(), seq.end());
    } else {
      m.permutation.resize(static_cast<size_t>(m.tokens));
      std::iota(m.permutation.begin(), m.permutation.end(), 0);
    }

    patch_all.insert(patch_all.end(), patched.begin(), patched.end());
    pos_all.insert(pos_all.end(), pos.begin(), pos.end());
    // The rope table follows the permutation (:990-991), so it is built here in
    // the block-walked order the hidden states will be in.
    for (int64_t r = 0; r < m.tokens; ++r) {
      const size_t src = static_cast<size_t>(m.permutation[static_cast<size_t>(r)]);
      cache_all.insert(cache_all.end(), cos.begin() + static_cast<std::ptrdiff_t>(src * static_cast<size_t>(half)),
                       cos.begin() + static_cast<std::ptrdiff_t>((src + 1) * static_cast<size_t>(half)));
      cache_all.insert(cache_all.end(), sin.begin() + static_cast<std::ptrdiff_t>(src * static_cast<size_t>(half)),
                       sin.begin() + static_cast<std::ptrdiff_t>((src + 1) * static_cast<size_t>(half)));
    }

    global_seq_lens.push_back(static_cast<int32_t>(m.tokens));
    L += m.tokens;
    meta.push_back(std::move(m));
  }

  // --- conv1_linear + positional embedding + ln_pre (:964-970) ---------------
  // conv1_linear is a bias-free Linear over the PATCHIFIED input (:710), not a
  // Conv2d. Everything here is row-local, so it runs in natural token order and
  // the window permutation is applied afterwards, exactly as upstream does.
  Buf pre(backend, q, dt, {L, H});
  Buf ln_pre_out(backend, q, dt, {L, H});
  {
    const Buf pix = Upload(backend, q, dt, {L, pd}, patch_all);
    const Buf conv1 = Upload(backend, q, dt, {H, pd}, weights.conv1_w);
    vt::MatmulBT(q, pre.tensor(), pix.tensor(), conv1.tensor());
    const Buf pe = Upload(backend, q, dt, {L, H}, pos_all);
    vt::Add(q, pre.tensor(), pre.tensor(), pe.tensor());
    const Buf lw = Upload(backend, q, dt, {H}, weights.ln_pre_w);
    const Buf lb = Upload(backend, q, dt, {H}, weights.ln_pre_b);
    vt::LayerNorm(q, ln_pre_out.tensor(), pre.tensor(), &lw.tensor(), &lb.tensor(),
                  vt::LayerNormArgs{cfg.layer_norm_eps});
  }
  if (capture != nullptr) {
    const std::vector<float> all =
        DownloadF32(ln_pre_out, q, dt, static_cast<size_t>(L * H));
    for (const Meta& m : meta) {
      const auto first = all.begin() + static_cast<std::ptrdiff_t>(m.offset * H);
      capture->ln_pre_out.emplace_back(
          first, first + static_cast<std::ptrdiff_t>(m.tokens * H));
    }
  }

  // --- window permutation (:988-993) ----------------------------------------
  Buf hidden(backend, q, dt, {L, H});
  {
    std::vector<int32_t> gather(static_cast<size_t>(L));
    for (const Meta& m : meta)
      for (int64_t r = 0; r < m.tokens; ++r)
        gather[static_cast<size_t>(m.offset + r)] =
            static_cast<int32_t>(m.offset) + m.permutation[static_cast<size_t>(r)];
    const Buf idx(backend, q, DType::kI32, {L}, gather.data());
    vt::IndexSelect(q, hidden.tensor(), ln_pre_out.tensor(), idx.tensor());
  }

  // --- rope cache + positions -----------------------------------------------
  const Buf cache = Upload(backend, q, dt, {L, hd}, cache_all);
  std::vector<int32_t> pos_ids(static_cast<size_t>(L));
  std::iota(pos_ids.begin(), pos_ids.end(), 0);
  const Buf positions(backend, q, DType::kI32, {L}, pos_ids.data());

  // --- device block weights --------------------------------------------------
  std::vector<DevBlock> dev(static_cast<size_t>(nl));
  for (int64_t l = 0; l < nl; ++l) {
    const MuseGlimmerVisionBlockWeights& bw = weights.blocks[static_cast<size_t>(l)];
    DevBlock& d = dev[static_cast<size_t>(l)];
    d.ln_1_w = Upload(backend, q, dt, {H}, bw.ln_1_w);
    d.ln_1_b = Upload(backend, q, dt, {H}, bw.ln_1_b);
    d.ln_2_w = Upload(backend, q, dt, {H}, bw.ln_2_w);
    d.ln_2_b = Upload(backend, q, dt, {H}, bw.ln_2_b);
    d.qkv_w = Upload(backend, q, dt, {3 * H, H}, bw.qkv_w);
    d.qkv_b = Upload(backend, q, dt, {3 * H}, bw.qkv_b);
    d.o_w = Upload(backend, q, dt, {H, H}, bw.o_w);
    d.o_b = Upload(backend, q, dt, {H}, bw.o_b);
    d.c_fc_w = Upload(backend, q, dt, {I, H}, bw.c_fc_w);
    d.c_fc_b = Upload(backend, q, dt, {I}, bw.c_fc_b);
    d.c_proj_w = Upload(backend, q, dt, {H, I}, bw.c_proj_w);
    d.c_proj_b = Upload(backend, q, dt, {H}, bw.c_proj_b);
  }

  // --- the block stack (:1015-1028) -----------------------------------------
  Buf n1(backend, q, dt, {L, H});
  Buf qkv(backend, q, dt, {L, 3 * H});
  Buf qb(backend, q, dt, {L, H}), kb(backend, q, dt, {L, H}), vb(backend, q, dt, {L, H});
  Buf ao(backend, q, dt, {L, nh, hd});
  Buf attn(backend, q, dt, {L, H});
  Buf n2(backend, q, dt, {L, H});
  Buf f1(backend, q, dt, {L, I});
  Buf f2(backend, q, dt, {L, H});

  vt::RopeArgs rope_args;
  rope_args.rotary_dim = static_cast<int>(hd);
  rope_args.is_neox_style = true;  // common.py:170-172 chunks the head in halves
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));  // :607

  for (int64_t l = 0; l < nl; ++l) {
    const DevBlock& d = dev[static_cast<size_t>(l)];
    vt::LayerNorm(q, n1.tensor(), hidden.tensor(), &d.ln_1_w.tensor(), &d.ln_1_b.tensor(),
                  vt::LayerNormArgs{cfg.layer_norm_eps});
    {
      // qkv_proj is one merged [3H, H] Linear WITH bias (:576-585); upstream
      // views it [L, 3, heads, head_dim] and unbinds dim 1, which is exactly the
      // contiguous q|k|v thirds this shared fold produces.
      Tensor qkv_bias = d.qkv_b.tensor();
      vllm::models::FusedMergedQkvBiasSplit(q, qkv.tensor(), qb.tensor(), kb.tensor(),
                                            vb.tensor(), n1.tensor(), d.qkv_w.tensor(),
                                            &qkv_bias);
    }
    Tensor q3 = Reshape3(qb.tensor(), L, nh, hd);
    Tensor k3 = Reshape3(kb.tensor(), L, nh, hd);
    const Tensor v3 = Reshape3(vb.tensor(), L, nh, hd);
    vt::RopeFromCache(q, q3, &k3, positions.tensor(), cache.tensor(), rope_args);

    // Full-attention layers attend over one whole image; window layers over one
    // pos_emb block (:1017-1021). Both are a segmentation of the same token
    // axis, and each image's tokens stay contiguous under the permutation, so
    // the block-diagonal mask is a per-segment dense attention.
    const std::vector<int32_t>& segments =
        cfg.layer_types[static_cast<size_t>(l)] == "full_attention" ? global_seq_lens
                                                                    : sparse_seq_lens;
    VT_CHECK(!segments.empty(), "MuseGlimmer vision sparse attention metadata is missing");
    {
      // The segmentation is validated BEFORE it indexes anything: a seq_lens
      // vector that does not exactly cover the token axis would otherwise make
      // the slices below read and write past the buffers, which is a silent
      // corruption rather than a diagnosable failure.
      int64_t covered = 0;
      for (int32_t n : segments) {
        VT_CHECK(n > 0, "MuseGlimmer vision attention segment must be positive");
        covered += static_cast<int64_t>(n);
      }
      VT_CHECK(covered == L, "MuseGlimmer vision attention segments do not cover the tokens");

      int64_t off = 0;
      const vt::AttentionArgs aargs{scale, /*causal=*/false};
      // KERNEL CHOICE — `vt::AttentionDenseFlash`, never `vt::Attention` (#1545,
      // class issue #1544). `kAttention` is deliberately frozen on the kernel
      // whose own header calls itself "Correctness-grade (M0.9)"
      // (src/vt/cuda/cuda_ops.cu:1456-1460): one 256-thread block per
      // (query, head), a 256-wide shared-memory tree reduction for EVERY key,
      // and no K/V tiling. That freeze is right for text decode and wrong for a
      // 50-layer non-causal vision tower, which has no other attention path.
      // The flash-tiled op serves head_dim 96 — bf16 asks 2*64*96*2 = 24 KiB of
      // dynamic shared memory, half the 48 KiB a launch gets by default — and it
      // honours `args.causal`, so the per-segment slicing below is unchanged.
      // On CPU both ops resolve to the SAME `AttentionKernel`
      // (src/vt/cpu/cpu_ops.cpp:3750-3761), so every golden is byte-identical;
      // on CUDA this is the rung whisper_audio.cpp:310-322 and
      // qwen3_vl_vision.cpp:462-480 already default to. `AttentionDenseFa2` is
      // NOT an option here: its fast path is head_dim 64 only
      // (cuda_ops.cu:3396-3399) and every other shape falls through to this op.
      for (int32_t n : segments) {
        const int64_t seg = static_cast<int64_t>(n);
        Tensor qs = RowSlice(q3, off, seg);
        const Tensor ks = RowSlice(k3, off, seg);
        const Tensor vs = RowSlice(v3, off, seg);
        Tensor os = RowSlice(ao.tensor(), off, seg);
        vt::AttentionDenseFlash(q, os, qs, ks, vs, aargs);
        off += seg;
      }
    }
    const Tensor ao2 = Reshape2(ao.tensor(), L, H);
    LinearBias(q, attn.tensor(), ao2, d.o_w.tensor(), &d.o_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());

    vt::LayerNorm(q, n2.tensor(), hidden.tensor(), &d.ln_2_w.tensor(), &d.ln_2_b.tensor(),
                  vt::LayerNormArgs{cfg.layer_norm_eps});
    LinearBias(q, f1.tensor(), n2.tensor(), d.c_fc_w.tensor(), &d.c_fc_b.tensor());
    vt::GeluErf(q, f1.tensor(), f1.tensor());  // F.gelu == erf, not tanh (:648)
    LinearBias(q, f2.tensor(), f1.tensor(), d.c_proj_w.tensor(), &d.c_proj_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), f2.tensor());

    if (capture != nullptr && l == 0)
      capture->block0_out = DownloadF32(hidden, q, dt, static_cast<size_t>(L * H));
  }

  // --- inverse permutation + ln_post + pixel shuffle (:1023-1034) ------------
  Buf natural(backend, q, dt, {L, H});
  {
    std::vector<int32_t> gather(static_cast<size_t>(L));
    for (const Meta& m : meta) {
      std::vector<int32_t> inverse(static_cast<size_t>(m.tokens));
      for (int64_t r = 0; r < m.tokens; ++r)
        inverse[static_cast<size_t>(m.permutation[static_cast<size_t>(r)])] =
            static_cast<int32_t>(r);
      for (int64_t r = 0; r < m.tokens; ++r)
        gather[static_cast<size_t>(m.offset + r)] =
            static_cast<int32_t>(m.offset) + inverse[static_cast<size_t>(r)];
    }
    const Buf idx(backend, q, DType::kI32, {L}, gather.data());
    vt::IndexSelect(q, natural.tensor(), hidden.tensor(), idx.tensor());
  }
  Buf post(backend, q, dt, {L, H});
  {
    const Buf lw = Upload(backend, q, dt, {H}, weights.ln_post_w);
    const Buf lb = Upload(backend, q, dt, {H}, weights.ln_post_b);
    vt::LayerNorm(q, post.tensor(), natural.tensor(), &lw.tensor(), &lb.tensor(),
                  vt::LayerNormArgs{cfg.layer_norm_eps});
  }
  const std::vector<float> post_host =
      DownloadF32(post, q, dt, static_cast<size_t>(L * H));
  backend.DestroyQueue(q);

  std::vector<float> out;
  out.reserve(static_cast<size_t>(L / cfg.merge_unit()) * static_cast<size_t>(cfg.output_dim));
  for (const Meta& m : meta) {
    const std::vector<float> item(
        post_host.begin() + static_cast<std::ptrdiff_t>(m.offset * H),
        post_host.begin() + static_cast<std::ptrdiff_t>((m.offset + m.tokens) * H));
    const std::vector<float> merged =
        MuseGlimmerVisionPixelShuffle(item, m.grid_height, m.grid_width, H, cfg);
    out.insert(out.end(), merged.begin(), merged.end());
  }
  return out;
}

// --- MuseGlimmerVisionAdapter.forward (muse_glimmer.py:1036-1044) ------------
// gelu(c_proj(gelu(c_fc(x)))) — BOTH projections bias-free, and note the SECOND
// gelu is on the output, not just between the two layers.
std::vector<float> MuseGlimmerVisionAdapterForward(
    const std::vector<float>& features, int64_t num_tokens,
    const MuseGlimmerVisionAdapterWeights& weights, const MuseGlimmerVisionConfig& cfg,
    Backend& backend) {
  const int64_t D = cfg.output_dim;
  const int64_t A = cfg.adapter_dim;
  VT_CHECK(static_cast<int64_t>(features.size()) == num_tokens * D,
           "MuseGlimmer vision adapter input has the wrong shape");
  const DType dt = cfg.compute_dtype;
  Queue q = backend.CreateQueue();

  const Buf x = Upload(backend, q, dt, {num_tokens, D}, features);
  const Buf fc = Upload(backend, q, dt, {A, D}, weights.c_fc_w);
  const Buf proj = Upload(backend, q, dt, {A, A}, weights.c_proj_w);
  Buf h1(backend, q, dt, {num_tokens, A});
  Buf h2(backend, q, dt, {num_tokens, A});
  vt::MatmulBT(q, h1.tensor(), x.tensor(), fc.tensor());
  vt::GeluErf(q, h1.tensor(), h1.tensor());
  vt::MatmulBT(q, h2.tensor(), h1.tensor(), proj.tensor());
  vt::GeluErf(q, h2.tensor(), h2.tensor());
  const std::vector<float> out =
      DownloadF32(h2, q, dt, static_cast<size_t>(num_tokens * A));
  backend.DestroyQueue(q);
  return out;
}

}  // namespace vllm::multimodal
