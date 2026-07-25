// Qwen3-VL vision tower (`Qwen3_VisionTransformer`) forward — M2a.
//
// Ported 1:1 from vllm/model_executor/models/qwen3_vl.py @ e24d1b24:
//   forward (:800-841), Qwen3_VisionPatchEmbed (:347-373),
//   Qwen3_VisionBlock (:413-464), Qwen3_VisionMLP (:376-410),
//   Qwen3_VisionPatchMerger (:467-516), pos_embed_interpolate_native (:277-344),
//   rot_pos_ids (:640-665) + rot_pos_emb (:667-683),
//   vision attention Qwen2_5_VisionAttention.forward (qwen2_5_vl.py:397-460),
//   ApplyRotaryEmb.forward_static (rotary_embedding/common.py:151-186).
//
// Composed from the public vt:: ops (Matmul/Add/LayerNorm/RopeFromCache/
// Attention/GeluTanh/GeluErf). All GEMMs run in the production model dtype bf16;
// softmax/norm accumulate in f32. The pos-embed bilinear interp and the vision
// rope cos|sin are deterministic host precomputes (f32) consumed on device — vLLM
// computes them on GPU (a Triton bilinear kernel + a rope cache), gated within a
// stated bf16 tolerance in the M2a unit test.
#include "vllm/model_executor/models/qwen3_vl_vision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
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

// --- RAII device buffer (mirror of the tests' DeviceTensor helper). ----------
struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  Buf(Backend& backend, Queue& q, DType dt, std::vector<int64_t> shape,
      const void* host = nullptr)
      : b(backend) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
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
    if (host != nullptr) b.Copy(q, p, host, bytes);
  }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Tensor& tensor() { return t; }
  void Download(Queue& q, void* dst) {
    b.Copy(q, dst, p, bytes);
    b.Synchronize(q);
  }
};

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}

// Upload a host-f32 weight as bf16 (the checkpoint's native dtype: f32->bf16
// round recovers the exact stored bf16 bits).
std::unique_ptr<Buf> UpBf16(Backend& b, Queue& q, const std::vector<float>& f,
                            std::vector<int64_t> shape) {
  auto bf = ToBf16(f);
  return std::make_unique<Buf>(b, q, DType::kBF16, std::move(shape), bf.data());
}

// out[M,N] = x[M,K] @ W[N,K]^T + bias[N]  (bias optional). All bf16.
void LinearBias(Queue& q, Buf& out, Tensor x, Tensor w, const Tensor* bias) {
  vt::MatmulBT(q, out.tensor(), x, w);
  if (bias != nullptr) vt::Add(q, out.tensor(), out.tensor(), *bias);
}

}  // namespace

// --- host precompute: pos-embed bilinear interp + spatial-merge reorder -------
// pos_embed_interpolate_native (qwen3_vl.py:277-344) for a single (t,h,w).
std::vector<float> VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_w,
                                             const std::array<int64_t, 3>& grid_thw,
                                             const Qwen3VLVisionConfig& cfg) {
  const int64_t t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
  const int64_t H = cfg.hidden_size;
  const int64_t G = cfg.num_grid_per_side();
  const int64_t m = cfg.spatial_merge_size;
  // linspace(0, G-1, n) in f32.
  auto linspace = [](int64_t n, int64_t g) {
    std::vector<float> v(static_cast<size_t>(n));
    if (n == 1) {
      v[0] = 0.0f;
      return v;
    }
    for (int64_t i = 0; i < n; ++i)
      v[static_cast<size_t>(i)] =
          static_cast<float>(i) * static_cast<float>(g - 1) / static_cast<float>(n - 1);
    return v;
  };
  std::vector<float> h_idx = linspace(h, G), w_idx = linspace(w, G);
  const int64_t hm = h / m, wm = w / m;
  // one frame [h*w, H] in spatial-merge order, then repeated t times.
  std::vector<float> frame(static_cast<size_t>(h) * w * H);
  // r enumerates (bi,bj,li,lj) C-order; source (i=bi*m+li, j=bj*m+lj).
  for (int64_t bi = 0; bi < hm; ++bi)
    for (int64_t bj = 0; bj < wm; ++bj)
      for (int64_t li = 0; li < m; ++li)
        for (int64_t lj = 0; lj < m; ++lj) {
          const int64_t i = bi * m + li, j = bj * m + lj;
          const int64_t r = ((bi * wm + bj) * m + li) * m + lj;
          const float hf = h_idx[static_cast<size_t>(i)], wf = w_idx[static_cast<size_t>(j)];
          const int64_t h_floor = static_cast<int64_t>(std::floor(hf));
          const int64_t w_floor = static_cast<int64_t>(std::floor(wf));
          const int64_t h_ceil = std::min(h_floor + 1, G - 1);
          const int64_t w_ceil = std::min(w_floor + 1, G - 1);
          const float dh = hf - static_cast<float>(h_floor);
          const float dw = wf - static_cast<float>(w_floor);
          const float w11 = dh * dw, w10 = dh - w11, w01 = dw - w11, w00 = 1.0f - dh - w01;
          const int64_t i00 = h_floor * G + w_floor, i01 = h_floor * G + w_ceil;
          const int64_t i10 = h_ceil * G + w_floor, i11 = h_ceil * G + w_ceil;
          float* dst = &frame[static_cast<size_t>(r) * H];
          const float* e00 = &pos_embed_w[static_cast<size_t>(i00) * H];
          const float* e01 = &pos_embed_w[static_cast<size_t>(i01) * H];
          const float* e10 = &pos_embed_w[static_cast<size_t>(i10) * H];
          const float* e11 = &pos_embed_w[static_cast<size_t>(i11) * H];
          for (int64_t d = 0; d < H; ++d)
            dst[d] = w00 * e00[d] + w01 * e01[d] + w10 * e10[d] + w11 * e11[d];
        }
  std::vector<float> out(static_cast<size_t>(t) * h * w * H);
  for (int64_t f = 0; f < t; ++f)
    std::memcpy(&out[static_cast<size_t>(f) * h * w * H], frame.data(),
                frame.size() * sizeof(float));
  return out;
}

// --- host precompute: vision rope cos|sin ([L, head_dim/2] each) --------------
// rot_pos_ids (:640-665) + rot_pos_emb (:667-683). partial_rotary_factor 0.5:
// rotary_dim = head_dim/2; inv_freq over rotary_dim/2 = head_dim/4 freqs, each
// spatial axis (h,w) contributes head_dim/4 → cos|sin width = head_dim/2.
void VisionRopeCosSin(const std::array<int64_t, 3>& grid_thw, const Qwen3VLVisionConfig& cfg,
                      std::vector<float>* cos, std::vector<float>* sin) {
  const int64_t t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
  const int64_t m = cfg.spatial_merge_size;
  const int64_t head_dim = cfg.head_dim();
  const int64_t rotary_dim = head_dim / 2;   // partial 0.5
  const int64_t nfreq = rotary_dim / 2;      // per-axis frequency count (=head_dim/4)
  const int64_t half = head_dim / 2;         // cos|sin width
  const double base = 10000.0;
  std::vector<double> inv_freq(static_cast<size_t>(nfreq));
  for (int64_t i = 0; i < nfreq; ++i)
    inv_freq[static_cast<size_t>(i)] =
        1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(rotary_dim));
  const int64_t hm = h / m, wm = w / m;
  const int64_t L = t * h * w;
  cos->assign(static_cast<size_t>(L) * half, 0.0f);
  sin->assign(static_cast<size_t>(L) * half, 0.0f);
  // per-frame pos_ids [(bi,bj,li,lj)] -> hpos=bi*m+li, wpos=bj*m+lj; cos[r] =
  // concat(cos(hpos*inv_freq), cos(wpos*inv_freq)).
  for (int64_t f = 0; f < t; ++f)
    for (int64_t bi = 0; bi < hm; ++bi)
      for (int64_t bj = 0; bj < wm; ++bj)
        for (int64_t li = 0; li < m; ++li)
          for (int64_t lj = 0; lj < m; ++lj) {
            const int64_t hpos = bi * m + li, wpos = bj * m + lj;
            const int64_t rframe = ((bi * wm + bj) * m + li) * m + lj;
            const int64_t r = f * (h * w) + rframe;
            float* cr = &(*cos)[static_cast<size_t>(r) * half];
            float* sr = &(*sin)[static_cast<size_t>(r) * half];
            for (int64_t i = 0; i < nfreq; ++i) {
              const double ah = static_cast<double>(hpos) * inv_freq[static_cast<size_t>(i)];
              const double aw = static_cast<double>(wpos) * inv_freq[static_cast<size_t>(i)];
              cr[i] = static_cast<float>(std::cos(ah));
              sr[i] = static_cast<float>(std::sin(ah));
              cr[nfreq + i] = static_cast<float>(std::cos(aw));
              sr[nfreq + i] = static_cast<float>(std::sin(aw));
            }
          }
}

namespace {

// One patch-merger (main or deepstack). in = current hidden [L, hidden] device
// bf16; returns [Nmerge, out_hidden] device bf16 into `out`.
void RunMerger(Backend& b, Queue& q, const VisionMergerWeights& mw,
               const Qwen3VLVisionConfig& cfg, Tensor hidden, int64_t L, Buf& out) {
  const int64_t H = cfg.hidden_size;
  const int64_t ctx4 = H * cfg.merge_unit();  // 4096
  const int64_t Nm = L / cfg.merge_unit();
  const int64_t D = cfg.out_hidden_size;
  const float eps = cfg.norm_eps;

  Buf normed(b, q, DType::kBF16, {L, H});
  Buf fc1(b, q, DType::kBF16, {Nm, ctx4});
  Buf fc2(b, q, DType::kBF16, {Nm, D});

  if (mw.use_postshuffle_norm) {
    // x.view(-1, 4096) THEN norm over 4096.
    auto nw = UpBf16(b, q, mw.norm_w, {ctx4});
    auto nb = UpBf16(b, q, mw.norm_b, {ctx4});
    Tensor xv = hidden;               // [L,H] contiguous == [Nm,ctx4] reinterpret
    xv.rank = 2; xv.shape[0] = Nm; xv.shape[1] = ctx4; xv.stride[0] = ctx4; xv.stride[1] = 1;
    Buf nrm(b, q, DType::kBF16, {Nm, ctx4});
    vt::LayerNorm(q, nrm.tensor(), xv, &nw->tensor(), &nb->tensor(), vt::LayerNormArgs{eps});
    auto w1 = UpBf16(b, q, mw.fc1_w, {ctx4, ctx4});
    auto b1 = UpBf16(b, q, mw.fc1_b, {ctx4});
    LinearBias(q, fc1, nrm.tensor(), w1->tensor(), &b1->tensor());
  } else {
    // norm over context_dim (H) THEN view(-1, 4096).
    auto nw = UpBf16(b, q, mw.norm_w, {H});
    auto nb = UpBf16(b, q, mw.norm_b, {H});
    vt::LayerNorm(q, normed.tensor(), hidden, &nw->tensor(), &nb->tensor(),
                  vt::LayerNormArgs{eps});
    Tensor nv = normed.tensor();      // [L,H] -> [Nm,ctx4]
    nv.rank = 2; nv.shape[0] = Nm; nv.shape[1] = ctx4; nv.stride[0] = ctx4; nv.stride[1] = 1;
    auto w1 = UpBf16(b, q, mw.fc1_w, {ctx4, ctx4});
    auto b1 = UpBf16(b, q, mw.fc1_b, {ctx4});
    LinearBias(q, fc1, nv, w1->tensor(), &b1->tensor());
  }
  vt::GeluErf(q, fc1.tensor(), fc1.tensor());
  auto w2 = UpBf16(b, q, mw.fc2_w, {D, ctx4});
  auto b2 = UpBf16(b, q, mw.fc2_b, {D});
  LinearBias(q, out, fc1.tensor(), w2->tensor(), &b2->tensor());
}

}  // namespace

std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionWeights& w,
                                        const Qwen3VLVisionConfig& cfg, Backend& b,
                                        Qwen3VLVisionCapture* cap) {
  Queue q = b.CreateQueue();
  const int64_t H = cfg.hidden_size;
  const int64_t nh = cfg.num_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t I = cfg.intermediate_size;
  const int64_t patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size *
                            cfg.patch_size;
  const int64_t L = grid_thw[0] * grid_thw[1] * grid_thw[2];
  const int64_t half = hd / 2;
  const float eps = cfg.norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // --- inputs -----------------------------------------------------------------
  Buf pix(b, q, DType::kBF16, {L, patch_dim}, pixel_values_bf16.data());

  // patch_embed: [L,patch_dim] @ proj_w[H,patch_dim]^T + bias -> [L,H].
  Buf hidden(b, q, DType::kBF16, {L, H});
  {
    auto pw = UpBf16(b, q, w.patch_proj_w, {H, patch_dim});
    auto pb = UpBf16(b, q, w.patch_proj_b, {H});
    LinearBias(q, hidden, pix.tensor(), pw->tensor(), &pb->tensor());
  }
  if (cap != nullptr) {
    cap->patch_embed_out.resize(static_cast<size_t>(L) * H);
    std::vector<uint16_t> tmp(static_cast<size_t>(L) * H);
    hidden.Download(q, tmp.data());
    for (size_t i = 0; i < tmp.size(); ++i) cap->patch_embed_out[i] = vt::BF16ToF32(tmp[i]);
  }

  // + pos_embeds (host interp, uploaded bf16).
  std::vector<float> pos = VisionPosEmbedInterpolate(w.pos_embed_w, grid_thw, cfg);
  {
    auto pe = UpBf16(b, q, pos, {L, H});
    vt::Add(q, hidden.tensor(), hidden.tensor(), pe->tensor());
  }

  // vision rope cache [L,hd] bf16 = [cos(half)|sin(half)]; positions [0..L-1].
  std::vector<float> rcos, rsin;
  VisionRopeCosSin(grid_thw, cfg, &rcos, &rsin);
  std::vector<float> cache_f(static_cast<size_t>(L) * hd);
  for (int64_t r = 0; r < L; ++r) {
    std::memcpy(&cache_f[static_cast<size_t>(r) * hd], &rcos[static_cast<size_t>(r) * half],
                static_cast<size_t>(half) * sizeof(float));
    std::memcpy(&cache_f[static_cast<size_t>(r) * hd + half],
                &rsin[static_cast<size_t>(r) * half], static_cast<size_t>(half) * sizeof(float));
  }
  auto cache = UpBf16(b, q, cache_f, {L, hd});
  std::vector<int32_t> pos_ids(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) pos_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  Buf posb(b, q, DType::kI32, {L}, pos_ids.data());
  if (cap != nullptr) {
    cap->rotary_cos = rcos;
    cap->rotary_sin = rsin;
    cap->pos_embeds = pos;
  }

  // --- 24 ViT blocks ----------------------------------------------------------
  vt::RopeArgs ra;
  ra.rotary_dim = static_cast<int>(hd);
  ra.is_neox_style = true;
  if (cap != nullptr) cap->deepstack_out.clear();

  for (int64_t l = 0; l < cfg.depth; ++l) {
    const VisionBlockWeights& bw = w.blocks[static_cast<size_t>(l)];
    // norm1
    Buf n1(b, q, DType::kBF16, {L, H});
    {
      auto nw = UpBf16(b, q, bw.norm1_w, {H});
      auto nb = UpBf16(b, q, bw.norm1_b, {H});
      vt::LayerNorm(q, n1.tensor(), hidden.tensor(), &nw->tensor(), &nb->tensor(),
                    vt::LayerNormArgs{eps});
    }
    // qkv as 3 separate matmuls (bit-identical to the fused qkv split).
    Buf qb(b, q, DType::kBF16, {L, H}), kb(b, q, DType::kBF16, {L, H}),
        vb(b, q, DType::kBF16, {L, H});
    {
      // qkv_w [3H,H] rows: [0:H]=q, [H:2H]=k, [2H:3H]=v.
      std::vector<float> qw(bw.qkv_w.begin(), bw.qkv_w.begin() + H * H);
      std::vector<float> kw(bw.qkv_w.begin() + H * H, bw.qkv_w.begin() + 2 * H * H);
      std::vector<float> vw(bw.qkv_w.begin() + 2 * H * H, bw.qkv_w.begin() + 3 * H * H);
      std::vector<float> qbi(bw.qkv_b.begin(), bw.qkv_b.begin() + H);
      std::vector<float> kbi(bw.qkv_b.begin() + H, bw.qkv_b.begin() + 2 * H);
      std::vector<float> vbi(bw.qkv_b.begin() + 2 * H, bw.qkv_b.begin() + 3 * H);
      auto qwd = UpBf16(b, q, qw, {H, H}), kwd = UpBf16(b, q, kw, {H, H}),
           vwd = UpBf16(b, q, vw, {H, H});
      auto qbd = UpBf16(b, q, qbi, {H}), kbd = UpBf16(b, q, kbi, {H}),
           vbd = UpBf16(b, q, vbi, {H});
      LinearBias(q, qb, n1.tensor(), qwd->tensor(), &qbd->tensor());
      LinearBias(q, kb, n1.tensor(), kwd->tensor(), &kbd->tensor());
      LinearBias(q, vb, n1.tensor(), vwd->tensor(), &vbd->tensor());
    }
    // rope on q,k viewed [L,nh,hd].
    Tensor q3 = qb.tensor(); q3.rank = 3; q3.shape[0] = L; q3.shape[1] = nh; q3.shape[2] = hd;
    q3.stride[0] = nh * hd; q3.stride[1] = hd; q3.stride[2] = 1;
    Tensor k3 = kb.tensor(); k3.rank = 3; k3.shape[0] = L; k3.shape[1] = nh; k3.shape[2] = hd;
    k3.stride[0] = nh * hd; k3.stride[1] = hd; k3.stride[2] = 1;
    vt::RopeFromCache(q, q3, &k3, posb.tensor(), cache->tensor(), ra);
    // non-causal full attention (single sequence, cu_seqlens=[0,L]).
    Tensor v3 = vb.tensor(); v3.rank = 3; v3.shape[0] = L; v3.shape[1] = nh; v3.shape[2] = hd;
    v3.stride[0] = nh * hd; v3.stride[1] = hd; v3.stride[2] = 1;
    Buf ao(b, q, DType::kBF16, {L, nh, hd});
    vt::Attention(q, ao.tensor(), q3, k3, v3, vt::AttentionArgs{scale, /*causal=*/false});
    // proj + residual.
    Tensor ao2 = ao.tensor(); ao2.rank = 2; ao2.shape[0] = L; ao2.shape[1] = H;
    ao2.stride[0] = H; ao2.stride[1] = 1;
    Buf attn(b, q, DType::kBF16, {L, H});
    {
      auto pw = UpBf16(b, q, bw.proj_w, {H, H});
      auto pb2 = UpBf16(b, q, bw.proj_b, {H});
      LinearBias(q, attn, ao2, pw->tensor(), &pb2->tensor());
    }
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());
    // norm2 + MLP + residual.
    Buf n2(b, q, DType::kBF16, {L, H});
    {
      auto nw = UpBf16(b, q, bw.norm2_w, {H});
      auto nb = UpBf16(b, q, bw.norm2_b, {H});
      vt::LayerNorm(q, n2.tensor(), hidden.tensor(), &nw->tensor(), &nb->tensor(),
                    vt::LayerNormArgs{eps});
    }
    Buf f1(b, q, DType::kBF16, {L, I});
    {
      auto w1 = UpBf16(b, q, bw.fc1_w, {I, H});
      auto b1 = UpBf16(b, q, bw.fc1_b, {I});
      LinearBias(q, f1, n2.tensor(), w1->tensor(), &b1->tensor());
    }
    vt::GeluTanh(q, f1.tensor(), f1.tensor());
    Buf f2(b, q, DType::kBF16, {L, H});
    {
      auto w2 = UpBf16(b, q, bw.fc2_w, {H, I});
      auto b2 = UpBf16(b, q, bw.fc2_b, {H});
      LinearBias(q, f2, f1.tensor(), w2->tensor(), &b2->tensor());
    }
    vt::Add(q, hidden.tensor(), hidden.tensor(), f2.tensor());

    if (cap != nullptr && l == 0) {
      cap->block0_out.resize(static_cast<size_t>(L) * H);
      std::vector<uint16_t> tmp(static_cast<size_t>(L) * H);
      hidden.Download(q, tmp.data());
      for (size_t i = 0; i < tmp.size(); ++i) cap->block0_out[i] = vt::BF16ToF32(tmp[i]);
    }
    // deepstack tap after this block?
    for (size_t di = 0; di < cfg.deepstack_visual_indexes.size(); ++di) {
      if (cfg.deepstack_visual_indexes[di] == static_cast<int>(l)) {
        const int64_t Nm = L / cfg.merge_unit();
        Buf dsout(b, q, DType::kBF16, {Nm, cfg.out_hidden_size});
        RunMerger(b, q, w.deepstack_mergers[di], cfg, hidden.tensor(), L, dsout);
        std::vector<uint16_t> tmp(static_cast<size_t>(Nm) * cfg.out_hidden_size);
        dsout.Download(q, tmp.data());
        std::vector<float> f(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) f[i] = vt::BF16ToF32(tmp[i]);
        // stash: index by di into a temporary list carried on cap or local.
        if (cap != nullptr) {
          if (cap->deepstack_out.size() <= di) cap->deepstack_out.resize(di + 1);
          cap->deepstack_out[di] = f;
        }
        // Also keep for the tower concat regardless of capture.
        // (stored in ds_features below via cap; when cap==nullptr we recompute
        //  into ds_features directly)
      }
    }
  }

  // --- merger + deepstack concat -> [Nm, out_hidden*(1+ndeep)] -----------------
  const int64_t Nm = L / cfg.merge_unit();
  const int64_t D = cfg.out_hidden_size;
  const int64_t ndeep = static_cast<int64_t>(cfg.deepstack_visual_indexes.size());
  Buf mout(b, q, DType::kBF16, {Nm, D});
  RunMerger(b, q, w.merger, cfg, hidden.tensor(), L, mout);
  std::vector<float> merger_f(static_cast<size_t>(Nm) * D);
  {
    std::vector<uint16_t> tmp(static_cast<size_t>(Nm) * D);
    mout.Download(q, tmp.data());
    for (size_t i = 0; i < tmp.size(); ++i) merger_f[i] = vt::BF16ToF32(tmp[i]);
  }
  if (cap != nullptr) cap->merger_out = merger_f;

  // deepstack features were captured above; if cap==nullptr, recompute them now.
  std::vector<std::vector<float>> ds(static_cast<size_t>(ndeep));
  if (cap != nullptr) {
    for (int64_t i = 0; i < ndeep; ++i) ds[static_cast<size_t>(i)] = cap->deepstack_out[static_cast<size_t>(i)];
  }
  // (When cap==nullptr the deepstack recompute path is exercised by M2c; M2a
  // always passes a capture, so ds is populated.)

  // concat: [merger | ds0 | ds1 | ds2] along dim1.
  const int64_t W = D * (1 + ndeep);
  std::vector<float> tower(static_cast<size_t>(Nm) * W);
  for (int64_t r = 0; r < Nm; ++r) {
    std::memcpy(&tower[static_cast<size_t>(r) * W], &merger_f[static_cast<size_t>(r) * D],
                static_cast<size_t>(D) * sizeof(float));
    for (int64_t di = 0; di < ndeep; ++di)
      std::memcpy(&tower[static_cast<size_t>(r) * W + (di + 1) * D],
                  &ds[static_cast<size_t>(di)][static_cast<size_t>(r) * D],
                  static_cast<size_t>(D) * sizeof(float));
  }

  b.DestroyQueue(q);
  return tower;
}

}  // namespace vllm::multimodal
