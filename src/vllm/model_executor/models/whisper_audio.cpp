// Whisper-class AUDIO encoder tower forward — audio-track A2.
//
// Ported 1:1 from transformers models/whisper/modeling_whisper.py @ 5.13.1:
//   WhisperEncoder.forward (:641-721) — gelu(conv1) -> gelu(conv2) -> transpose ->
//     + embed_positions -> N pre-norm WhisperEncoderLayer -> final layer_norm;
//   WhisperEncoderLayer.forward (:400-430); WhisperAttention.forward (:298-368)
//     (q_proj*scaling, k_proj no-bias, v_proj, non-causal MHA scaling=1.0 folded,
//     out_proj); WhisperMLP fc1 -> gelu(erf) -> fc2.
//   Cross-checked against the faithful vLLM port vllm/model_executor/models/
//     whisper.py @ e24d1b24 (WhisperEncoder:458, WhisperEncoderLayer:353).
//
// Composed from the public vt:: ops (MatmulBT/Add/LayerNorm/Attention/GeluErf).
// The two Conv1d layers are expressed as host im2col + vt::MatmulBT — no new CUDA
// kernel (Whisper conv is a FULL cross-channel conv, not the depthwise
// vt::CausalConv1d used by Mamba/GDN). All GEMMs run bf16; softmax/norm accumulate
// f32, mirroring the M2a vision tower and matching vLLM's bf16 encoder.
#include "vllm/model_executor/models/whisper_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

// --- RAII device buffer (mirror of qwen3_vl_vision.cpp's Buf). ----------------
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

// Upload a host-f32 tensor as bf16 (checkpoint native round-trips exactly).
std::unique_ptr<Buf> UpBf16(Backend& b, Queue& q, const std::vector<float>& f,
                            std::vector<int64_t> shape) {
  auto bf = ToBf16(f);
  return std::make_unique<Buf>(b, q, DType::kBF16, std::move(shape), bf.data());
}

// out[M,N] = x[M,K] @ W[N,K]^T + bias[N] (bias optional). All bf16.
void LinearBias(Queue& q, Buf& out, Tensor x, Tensor w, const Tensor* bias) {
  vt::MatmulBT(q, out.tensor(), x, w);
  if (bias != nullptr) vt::Add(q, out.tensor(), out.tensor(), *bias);
}

// Download a device bf16 buffer [rows*cols] to host f32.
std::vector<float> DownloadF32(Buf& buf, Queue& q, int64_t rows, int64_t cols) {
  std::vector<uint16_t> tmp(static_cast<size_t>(rows) * cols);
  buf.Download(q, tmp.data());
  std::vector<float> out(tmp.size());
  for (size_t i = 0; i < tmp.size(); ++i) out[i] = vt::BF16ToF32(tmp[i]);
  return out;
}

// im2col for a Conv1d(in_ch -> out_ch, kernel=3, pad=1, stride=s) whose input is
// laid out [in_ch, T] with `x_ct(ci, time)` giving the (ci, time) element (0 when
// time is out of [0, T)). Returns the [T_out, in_ch*3] row-major matrix so that
// out[t] = MatmulBT with W_flat[out_ch, in_ch*3] (W stored [out_ch, in_ch, 3]).
template <typename XAccessor>
std::vector<float> Im2Col(int64_t in_ch, int64_t T, int64_t stride, int64_t T_out,
                          const XAccessor& x_ct) {
  const int64_t K = 3, pad = 1;
  std::vector<float> col(static_cast<size_t>(T_out) * in_ch * K);
  for (int64_t t = 0; t < T_out; ++t) {
    for (int64_t ci = 0; ci < in_ch; ++ci) {
      for (int64_t dk = 0; dk < K; ++dk) {
        const int64_t src = t * stride + dk - pad;
        const float v = (src >= 0 && src < T) ? x_ct(ci, src) : 0.0f;
        col[static_cast<size_t>(t) * in_ch * K + ci * K + dk] = v;
      }
    }
  }
  return col;
}

}  // namespace

std::vector<float> WhisperAudioEncoderForward(const std::vector<float>& input_features,
                                              const WhisperAudioEncoderWeights& w,
                                              const WhisperAudioEncoderConfig& cfg, Backend& b,
                                              WhisperAudioEncoderCapture* cap) {
  Queue q = b.CreateQueue();
  const int64_t H = cfg.d_model;
  const int64_t nh = cfg.num_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t I = cfg.ffn_dim;
  const int64_t Cmel = cfg.num_mel_bins;
  const int64_t Tin = cfg.n_frames;                 // 3000
  const int64_t L = cfg.max_source_positions;       // 1500
  const float eps = cfg.norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // --- conv1: Conv1d(80->768, k3, pad1, stride1) + gelu --------------------------
  // input_features is [Cmel, Tin]. im2col over Tin (stride 1) -> [Tin, Cmel*3];
  // MatmulBT with conv1_w flat [H, Cmel*3] gives conv1 output ALREADY transposed to
  // [Tin, H]. + bias, gelu(erf).
  std::vector<float> col1 = Im2Col(Cmel, Tin, /*stride=*/1, /*T_out=*/Tin,
                                   [&](int64_t ci, int64_t time) {
                                     return input_features[static_cast<size_t>(ci) * Tin + time];
                                   });
  Buf conv1(b, q, DType::kBF16, {Tin, H});
  {
    auto c1 = UpBf16(b, q, col1, {Tin, Cmel * 3});
    auto w1 = UpBf16(b, q, w.conv1_w, {H, Cmel * 3});
    auto b1 = UpBf16(b, q, w.conv1_b, {H});
    LinearBias(q, conv1, c1->tensor(), w1->tensor(), &b1->tensor());
    vt::GeluErf(q, conv1.tensor(), conv1.tensor());
  }
  std::vector<float> h1 = DownloadF32(conv1, q, Tin, H);  // [Tin, H]

  // --- conv2: Conv1d(768->768, k3, pad1, stride2) + gelu -------------------------
  // input h1 is [Tin, H] (time-major). im2col over Tin with stride 2 -> [L, H*3].
  std::vector<float> col2 = Im2Col(H, Tin, /*stride=*/2, /*T_out=*/L,
                                   [&](int64_t ci, int64_t time) {
                                     return h1[static_cast<size_t>(time) * H + ci];
                                   });
  Buf hidden(b, q, DType::kBF16, {L, H});
  {
    auto c2 = UpBf16(b, q, col2, {L, H * 3});
    auto w2 = UpBf16(b, q, w.conv2_w, {H, H * 3});
    auto b2 = UpBf16(b, q, w.conv2_b, {H});
    LinearBias(q, hidden, c2->tensor(), w2->tensor(), &b2->tensor());
    vt::GeluErf(q, hidden.tensor(), hidden.tensor());
  }
  if (cap != nullptr) cap->post_conv = DownloadF32(hidden, q, L, H);

  // --- + embed_positions (fixed sinusoid, first L rows) --------------------------
  {
    std::vector<float> pos(w.embed_positions_w.begin(),
                           w.embed_positions_w.begin() + static_cast<size_t>(L) * H);
    auto pe = UpBf16(b, q, pos, {L, H});
    vt::Add(q, hidden.tensor(), hidden.tensor(), pe->tensor());
  }
  if (cap != nullptr) cap->post_pos = DownloadF32(hidden, q, L, H);

  // --- N pre-norm encoder layers -------------------------------------------------
  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    const WhisperEncoderLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    // residual + self_attn_layer_norm.
    Buf n1(b, q, DType::kBF16, {L, H});
    {
      auto nw = UpBf16(b, q, lw.attn_ln_w, {H});
      auto nb = UpBf16(b, q, lw.attn_ln_b, {H});
      vt::LayerNorm(q, n1.tensor(), hidden.tensor(), &nw->tensor(), &nb->tensor(),
                    vt::LayerNormArgs{eps});
    }
    // q/k/v projections. k_proj has NO bias (Whisper). q is pre-scaled in
    // transformers; we fold the identical scaling into vt::Attention (scale =
    // head_dim**-0.5) so q carries only its bias here.
    Buf qb(b, q, DType::kBF16, {L, H}), kb(b, q, DType::kBF16, {L, H}),
        vb(b, q, DType::kBF16, {L, H});
    {
      auto qw = UpBf16(b, q, lw.q_w, {H, H});
      auto qbi = UpBf16(b, q, lw.q_b, {H});
      auto kw = UpBf16(b, q, lw.k_w, {H, H});
      auto vw = UpBf16(b, q, lw.v_w, {H, H});
      auto vbi = UpBf16(b, q, lw.v_b, {H});
      LinearBias(q, qb, n1.tensor(), qw->tensor(), &qbi->tensor());
      LinearBias(q, kb, n1.tensor(), kw->tensor(), /*bias=*/nullptr);
      LinearBias(q, vb, n1.tensor(), vw->tensor(), &vbi->tensor());
    }
    // full bidirectional attention, [L, nh, hd].
    Tensor q3 = qb.tensor(); q3.rank = 3; q3.shape[0] = L; q3.shape[1] = nh; q3.shape[2] = hd;
    q3.stride[0] = nh * hd; q3.stride[1] = hd; q3.stride[2] = 1;
    Tensor k3 = kb.tensor(); k3.rank = 3; k3.shape[0] = L; k3.shape[1] = nh; k3.shape[2] = hd;
    k3.stride[0] = nh * hd; k3.stride[1] = hd; k3.stride[2] = 1;
    Tensor v3 = vb.tensor(); v3.rank = 3; v3.shape[0] = L; v3.shape[1] = nh; v3.shape[2] = hd;
    v3.stride[0] = nh * hd; v3.stride[1] = hd; v3.stride[2] = 1;
    Buf ao(b, q, DType::kBF16, {L, nh, hd});
    // Whisper encoder attention is FULL bidirectional (non-causal) over the fixed
    // 1500-frame context. The naive vt::Attention (kAttention) is O(t^2) with a
    // per-key block __syncthreads reduction — one CTA per (query,head) streaming all
    // 1500 keys — so nsys attributes the overwhelming majority of the 32-layer
    // encoder forward to it (the exact anti-pattern §7 fixed for the Qwen3-VL vision
    // tower). Route to the warp-scoped online-softmax vt::AttentionDenseFast
    // (head_dim 64, non-causal), the SAME kernel that beat vLLM's eager vision
    // encode: one WARP per (query,head), the head_dim reduction a butterfly
    // __shfl_xor (no __syncthreads), accumulator in registers — the IDENTICAL f32
    // online-softmax recurrence within the bf16 envelope. kAttention (the text
    // decode path, qwen3_5.cpp / voxtral text) is untouched ⇒ byte-identical by
    // construction. Grounded 1:1 in vLLM's fast encoder attention:
    // vllm/model_executor/models/whisper.py WhisperEncoderAttention (:255) ->
    // forward (:298-317) self.attn(q,k,v) dispatches the encoder self-attention to
    // the flash-attn varlen (non-causal, full) backend @ e24d1b24 — a
    // fully-SM-filling flash kernel, never the O(t^2) scalar path. VT_WHISPER_ENC_EAGER=1
    // restores the naive kernel for the same-binary A/B.
    static const bool enc_eager = [] {
      const char* e = std::getenv("VT_WHISPER_ENC_EAGER");
      return e != nullptr && e[0] == '1';
    }();
    if (enc_eager)
      vt::Attention(q, ao.tensor(), q3, k3, v3, vt::AttentionArgs{scale, /*causal=*/false});
    else
      vt::AttentionDenseFast(q, ao.tensor(), q3, k3, v3, vt::AttentionArgs{scale, /*causal=*/false});
    // out_proj + residual.
    Tensor ao2 = ao.tensor(); ao2.rank = 2; ao2.shape[0] = L; ao2.shape[1] = H;
    ao2.stride[0] = H; ao2.stride[1] = 1;
    Buf attn(b, q, DType::kBF16, {L, H});
    {
      auto ow = UpBf16(b, q, lw.out_w, {H, H});
      auto ob = UpBf16(b, q, lw.out_b, {H});
      LinearBias(q, attn, ao2, ow->tensor(), &ob->tensor());
    }
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());
    // residual + final_layer_norm + MLP (fc1 -> gelu-erf -> fc2) + residual.
    Buf n2(b, q, DType::kBF16, {L, H});
    {
      auto nw = UpBf16(b, q, lw.final_ln_w, {H});
      auto nb = UpBf16(b, q, lw.final_ln_b, {H});
      vt::LayerNorm(q, n2.tensor(), hidden.tensor(), &nw->tensor(), &nb->tensor(),
                    vt::LayerNormArgs{eps});
    }
    Buf f1(b, q, DType::kBF16, {L, I});
    {
      auto w1 = UpBf16(b, q, lw.fc1_w, {I, H});
      auto b1 = UpBf16(b, q, lw.fc1_b, {I});
      LinearBias(q, f1, n2.tensor(), w1->tensor(), &b1->tensor());
    }
    vt::GeluErf(q, f1.tensor(), f1.tensor());
    Buf f2(b, q, DType::kBF16, {L, H});
    {
      auto w2 = UpBf16(b, q, lw.fc2_w, {H, I});
      auto b2 = UpBf16(b, q, lw.fc2_b, {H});
      LinearBias(q, f2, f1.tensor(), w2->tensor(), &b2->tensor());
    }
    vt::Add(q, hidden.tensor(), hidden.tensor(), f2.tensor());

    if (cap != nullptr && l == 0) cap->block0_out = DownloadF32(hidden, q, L, H);
  }

  // --- final encoder layer_norm --------------------------------------------------
  Buf out(b, q, DType::kBF16, {L, H});
  {
    auto nw = UpBf16(b, q, w.final_ln_w, {H});
    auto nb = UpBf16(b, q, w.final_ln_b, {H});
    vt::LayerNorm(q, out.tensor(), hidden.tensor(), &nw->tensor(), &nb->tensor(),
                  vt::LayerNormArgs{eps});
  }
  std::vector<float> result = DownloadF32(out, q, L, H);
  if (cap != nullptr) cap->final_ln_out = result;

  b.DestroyQueue(q);
  return result;
}

}  // namespace vllm::multimodal
