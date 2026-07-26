// DFlash draft model forward (SPEC-DFLASH D2, DF-DRAFT-MODEL). Ported from
// vllm/model_executor/models/qwen3_dflash.py @ 555967922. See qwen3_dflash.h.
//
// The ONE new brick is the attention: full-attention layers route through
// vt::DFlashBlockAttention with args.causal=false (BIDIRECTIONAL in-block); SWA
// layers use args.causal=true + the window. Every other op (embed, merged-qkv
// GEMM, per-head q/k RMSNorm, NeoX RoPE, SwiGLU, standard add+RMSNorm, lm_head) is
// reused from the landed Qwen3-dense block ops (dense_attn_block.h / vt::).
#include "vllm/model_executor/models/qwen3_dflash.h"

#include <cmath>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight/Reshape/MakeRopeArgs
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight, Reshape, MakeRopeArgs

}  // namespace

std::vector<float> Qwen3DFlashModel::CombineAuxFeatures(const std::vector<float>& aux_features,
                                                        int64_t T,
                                                        const Qwen3DFlashWeights& weights,
                                                        const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Fin = H * weights.num_taps;
  VT_CHECK(static_cast<int64_t>(aux_features.size()) == T * Fin,
           "qwen3_dflash fc: aux_features must be [T, H*num_taps]");
  // aux is [T, H*num_taps] f32 -> cast to bf16 -> fc MatmulBT -> [T,H] bf16.
  DBuf aux32(d, DType::kF32, {T, Fin}, aux_features.data());
  DBuf auxb(d, DType::kBF16, {T, Fin});
  vt::CastBf16(d.q, auxb.t(), aux32.t());
  Tensor wfc = ResidentWeight(d, weights.fc);  // [H, H*num_taps] nk
  DBuf comb(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, comb.t(), auxb.t(), wfc);
  DBuf comb32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, comb32.t(), comb.t());
  std::vector<float> out(static_cast<size_t>(T) * H);
  comb32.Download(d, out.data());
  return out;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogits(
    const std::vector<int32_t>& input_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(input_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_dflash: positions length must match input_ids");
  VT_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(T),
           "qwen3_dflash: cu_seqlens must span [0,T]");
  VT_CHECK(weights.layers.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3_dflash: one layer weight per config.num_hidden_layers");

  // Embed: hidden[T,H] bf16 = embed_tokens[input_ids]; mask slots take
  // embed_tokens[mask_token_id] naturally (in-vocab), or the dedicated mask
  // embedding when present (qwen3_dflash.py:432-438).
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H});
    DBuf dids(d, DType::kI32, {T}, input_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  if (!weights.mask_embedding.Empty() && weights.mask_token_id >= 0) {
    // Substitute the dedicated mask embedding for mask_token_id rows.
    Tensor mask_emb = ResidentWeight(d, weights.mask_embedding, {H});
    std::vector<float> mask_host(static_cast<size_t>(H));
    {
      DBuf tmp(d, DType::kF32, {H});
      vt::CastF32(d.q, tmp.t(), mask_emb);
      tmp.Download(d, mask_host.data());
    }
    std::vector<float> hidden_host(static_cast<size_t>(T) * H);
    {
      DBuf tmp(d, DType::kF32, {T, H});
      vt::CastF32(d.q, tmp.t(), hidden.t());
      tmp.Download(d, hidden_host.data());
    }
    for (int64_t r = 0; r < T; ++r)
      if (input_ids[static_cast<size_t>(r)] == weights.mask_token_id)
        for (int64_t j = 0; j < H; ++j)
          hidden_host[static_cast<size_t>(r * H + j)] = mask_host[static_cast<size_t>(j)];
    DBuf hf(d, DType::kF32, {T, H}, hidden_host.data());
    vt::CastBf16(d.q, hidden.t(), hf.t());
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);
  DBuf dpos(d, DType::kI32, {T}, positions.data());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    // input_layernorm (std add+RMSNorm): dhn = norm(hidden + res); res updated.
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {T, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

    // attention over the context-free block (routes through DFlashBlockAttention).
    // Reuse the block helper but feed the real positions to RoPE.
    DBuf attn = [&]() -> DBuf {
      const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
      DBuf q(d, DType::kBF16, {T, qdim});
      DBuf k(d, DType::kBF16, {T, kdim});
      DBuf v(d, DType::kBF16, {T, kdim});
      Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
      Tensor wq = wqkv.Slice(0, 0, qdim);
      Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
      Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
      vt::MatmulBT(d.q, q.t(), dhn.t(), wq);
      vt::MatmulBT(d.q, k.t(), dhn.t(), wk);
      vt::MatmulBT(d.q, v.t(), dhn.t(), wv);
      Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
      Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
      Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
      Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
      Tensor wqn = ResidentWeight(d, layer.q_norm, {Dh});
      Tensor wkn = ResidentWeight(d, layer.k_norm, {Dh});
      vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
      vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
      vt::RopeNeox(d.q, q3, k3, dpos.t(), MakeRopeArgs(config));
      Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
      DBuf a(d, DType::kBF16, {T, Hq, Dh});
      vt::DFlashBlockAttentionArgs pa;
      pa.scale = scale;
      pa.causal = layer.attn_mode.causal;
      pa.sliding_window = layer.attn_mode.sliding_window;
      pa.cu_seqlens = cu.data();
      pa.num_reqs = static_cast<int>(cu.size()) - 1;
      vt::DFlashBlockAttention(d.q, a.t(), q3, k3, v3, pa);
      Tensor o_in = Reshape(a.t(), {T, Hq * Dh});
      Tensor wo = ResidentWeight(d, layer.o_proj);
      DBuf o(d, DType::kBF16, {T, H});
      vt::MatmulBT(d.q, o.t(), o_in, wo);
      return o;
    }();

    // post_attention_layernorm (std add+RMSNorm).
    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {T, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

    // SwiGLU MLP: gate_up GEMM -> SiluAndMul -> down GEMM.
    const int64_t I = config.intermediate_size;
    Tensor wgu = ResidentWeight(d, layer.gate_up_proj);
    DBuf gu(d, DType::kBF16, {T, 2 * I});
    vt::MatmulBT(d.q, gu.t(), dh2.t(), wgu);
    DBuf act(d, DType::kBF16, {T, I});
    vt::SiluAndMul(d.q, act.t(), gu.t());
    Tensor wdn = ResidentWeight(d, layer.down_proj);
    DBuf down(d, DType::kBF16, {T, H});
    vt::MatmulBT(d.q, down.t(), act.t(), wdn);
    hidden = std::move(down);
  }

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {T, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);
  std::vector<float> out(static_cast<size_t>(T) * vocab);
  logits.Download(d, out.data());
  return out;
}

}  // namespace vllm
