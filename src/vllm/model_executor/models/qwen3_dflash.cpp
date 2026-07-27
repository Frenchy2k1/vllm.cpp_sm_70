// DFlash draft model forward (SPEC-DFLASH D2, DF-DRAFT-MODEL). Ported from
// vllm/model_executor/models/qwen3_dflash.py @ 555967922. See qwen3_dflash.h.
//
// The ONE new brick is the attention: full-attention layers route through
// vt::DFlashBlockAttention with args.causal=false (BIDIRECTIONAL in-block); SWA
// layers use args.causal=true + the window. Every other op (embed, merged-qkv
// GEMM, per-head q/k RMSNorm, NeoX RoPE, SwiGLU, standard add+RMSNorm, lm_head) is
// reused from the landed Qwen3-dense block ops (dense_attn_block.h / vt::).
#include "vllm/model_executor/models/qwen3_dflash.h"

#include <algorithm>
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

constexpr int64_t kPadSlotId = -1;  // vLLM PAD_SLOT_ID (attention/backends/utils.py:45)

// Device-resident per-layer context K/V (D7). The D5 path downloaded each layer's
// projected K/V to host (2 D->H copies/layer) and re-uploaded them in the block
// forward's [context;block] host interleave. This helper keeps the projected K/V
// ON DEVICE as bf16 [num_ctx, kv_size] buffers, so the block forward can build the
// combined sequence with device vt::IndexCopy instead of host round-trips. The
// float ops (cast/RMSNorm/GEMM/k-norm/RoPE) are IDENTICAL to the D5 path and run in
// the same order, so the stored K/V bits are bit-identical to the D5 download.
// Mirrors precompute_and_store_context_kv (qwen3_dflash.py:548-619), minus the
// paged-cache write (our within-step store is these DBufs).
struct ContextKVDev {
  std::vector<DBuf> k;  // per attention layer: bf16 [num_ctx, Hkv*Dh] (normed+RoPE'd)
  std::vector<DBuf> v;  // per attention layer: bf16 [num_ctx, Hkv*Dh] (raw)
  int64_t num_ctx = 0;
};

ContextKVDev PrecomputeContextKVDevice(Dev d, const float* context_states,
                                       const int32_t* context_positions, int64_t C,
                                       const Qwen3DFlashWeights& weights,
                                       const HfConfig& config) {
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const float eps = static_cast<float>(config.rms_norm_eps);

  ContextKVDev out;
  out.num_ctx = C;
  if (C == 0) return out;

  // normed = RMSNorm(context_states, hidden_norm) — the ONE shared hidden_norm
  // over the combined target features (qwen3_dflash.py:505-520).
  DBuf ctx32(d, DType::kF32, {C, H}, context_states);
  DBuf ctxb(d, DType::kBF16, {C, H});
  vt::CastBf16(d.q, ctxb.t(), ctx32.t());
  Tensor w_hn = ResidentWeight(d, weights.hidden_norm, {H});
  DBuf normed(d, DType::kBF16, {C, H});
  vt::RmsNorm(d.q, normed.t(), ctxb.t(), w_hn, vt::RmsNormArgs{eps, false});
  DBuf cpos(d, DType::kI32, {C}, context_positions);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
    Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
    Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
    DBuf k(d, DType::kBF16, {C, kdim});
    DBuf v(d, DType::kBF16, {C, kdim});
    vt::MatmulBT(d.q, k.t(), normed.t(), wk);
    vt::MatmulBT(d.q, v.t(), normed.t(), wv);
    // K-norm over head_dim, then NeoX RoPE on K at the context positions (V raw).
    Tensor k2 = Reshape(k.t(), {C * Hkv, Dh});
    Tensor wkn = ResidentWeight(d, layer.k_norm, {Dh});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
    Tensor k3 = Reshape(k.t(), {C, Hkv, Dh});
    DBuf rope_scratch(d, DType::kBF16, {C, Hkv, Dh});
    rope_scratch.Zero(d);
    Tensor scratch3 = rope_scratch.t();
    vt::RopeNeox(d.q, k3, scratch3, cpos.t(), MakeRopeArgs(config));
    out.k.push_back(std::move(k));  // bf16 [C, kdim] contiguous (RoPE'd view aliases it)
    out.v.push_back(std::move(v));  // bf16 [C, kdim] raw
  }
  return out;
}

}  // namespace

DflashPrepareOutputs PrepareDflashInputs(const DflashPrepareBatch& b) {
  // Pure-integer host port of _prepare_dflash_inputs_kernel (dflash/speculator.py:
  // 472-618). Every store the Triton kernel makes is reproduced here; there is no
  // float math, so this is bit-exact by construction.
  const int32_t num_reqs = static_cast<int32_t>(b.idx_mapping.size());
  VT_CHECK(num_reqs > 0, "prepare_dflash_inputs: num_reqs must be > 0");
  VT_CHECK(static_cast<int32_t>(b.target_query_start_loc.size()) == num_reqs + 1,
           "prepare_dflash_inputs: target_query_start_loc must be [num_reqs+1]");
  const int32_t nqpr = b.num_query_per_req;
  const int32_t nspec = b.num_speculative_steps;
  const int32_t stride = b.block_table_stride;
  const int32_t bs = b.block_size;
  const int64_t num_target_tokens = b.target_query_start_loc.back();

  DflashPrepareOutputs o;
  o.input_ids.assign(static_cast<size_t>(num_reqs) * nqpr, 0);
  o.query_positions.assign(static_cast<size_t>(num_reqs) * nqpr, 0);
  o.query_start_loc.assign(static_cast<size_t>(b.max_num_reqs) + 1, 0);
  o.seq_lens.assign(static_cast<size_t>(b.max_num_reqs), 0);
  o.query_slot_mapping.assign(static_cast<size_t>(b.max_num_tokens), 0);
  o.context_positions.assign(static_cast<size_t>(num_target_tokens), 0);
  o.context_slot_mapping.assign(static_cast<size_t>(num_target_tokens), 0);
  o.sample_indices.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);
  o.sample_pos.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);
  o.sample_idx_mapping.assign(static_cast<size_t>(b.max_num_reqs) * nspec, 0);

  const int32_t sample_off = b.sample_from_anchor ? 0 : 1;

  for (int32_t r = 0; r < num_reqs; ++r) {
    const int32_t req_state_idx = b.idx_mapping[static_cast<size_t>(r)];
    const int32_t ctx_start = b.target_query_start_loc[static_cast<size_t>(r)];
    const int32_t ctx_end = b.target_query_start_loc[static_cast<size_t>(r) + 1];
    const int32_t num_ctx = ctx_end - ctx_start;
    const int32_t num_rejected = b.num_rejected[static_cast<size_t>(r)];
    const int32_t valid_ctx_end = ctx_end - num_rejected;
    const int32_t num_sampled = b.num_sampled[static_cast<size_t>(r)];
    const int32_t bonus_token =
        num_sampled > 0 ? b.last_sampled[static_cast<size_t>(req_state_idx)]
                        : b.next_prefill_tokens[static_cast<size_t>(req_state_idx)];
    const int64_t last_valid_pos =
        b.target_positions[static_cast<size_t>(valid_ctx_end) - 1];
    const int32_t query_base = r * nqpr;

    // --- Context positions / slots (j in [0, num_ctx)) ---
    for (int32_t j = 0; j < num_ctx; ++j) {
      const int64_t ctx_pos = b.target_positions[static_cast<size_t>(ctx_start + j)];
      int32_t ctx_block_num = static_cast<int32_t>(ctx_pos / bs);
      if (ctx_block_num > stride - 1) ctx_block_num = stride - 1;
      const int64_t ctx_block_id =
          b.block_table[static_cast<size_t>(r) * stride + ctx_block_num];
      const int64_t ctx_slot = ctx_block_id * bs + (ctx_pos % bs);
      o.context_positions[static_cast<size_t>(ctx_start + j)] = ctx_pos;
      o.context_slot_mapping[static_cast<size_t>(ctx_start + j)] = ctx_slot;
    }

    // --- Query positions / input_ids / slots + sample maps (offset in [0,nqpr)) ---
    for (int32_t off = 0; off < nqpr; ++off) {
      const int64_t query_pos = last_valid_pos + 1 + off;
      const int32_t query_idx = query_base + off;
      const int32_t input_id = (off == 0) ? bonus_token : b.parallel_drafting_token_id;
      int32_t q_block_num = static_cast<int32_t>(query_pos / bs);
      if (q_block_num > stride - 1) q_block_num = stride - 1;
      const int64_t q_block_id =
          b.block_table[static_cast<size_t>(r) * stride + q_block_num];
      const int64_t q_slot = q_block_id * bs + (query_pos % bs);
      o.input_ids[static_cast<size_t>(query_idx)] = input_id;
      o.query_positions[static_cast<size_t>(query_idx)] =
          std::min<int64_t>(query_pos, b.max_model_len - 1);
      o.query_slot_mapping[static_cast<size_t>(query_idx)] = q_slot;
      if (off >= sample_off) {
        const int32_t sample_idx = r * nspec + (off - sample_off);
        const int64_t spos = b.sample_from_anchor ? query_pos + 1 : query_pos;
        o.sample_indices[static_cast<size_t>(sample_idx)] = query_idx;
        o.sample_pos[static_cast<size_t>(sample_idx)] = spos;
        o.sample_idx_mapping[static_cast<size_t>(sample_idx)] = req_state_idx;
      }
    }

    o.query_start_loc[static_cast<size_t>(r)] = query_base;
    o.seq_lens[static_cast<size_t>(r)] = static_cast<int32_t>(last_valid_pos) + 1 + nqpr;
  }

  // --- Padding for CUDA-graph replay safety (kernel block_idx==0, req==last) ---
  const int32_t last_query_end = num_reqs * nqpr;
  for (int32_t i = num_reqs; i <= b.max_num_reqs; ++i)
    o.query_start_loc[static_cast<size_t>(i)] = last_query_end;
  // seq_lens[num_reqs, max_num_reqs) already 0 from assign.
  for (int32_t i = num_reqs * nspec; i < b.max_num_reqs * nspec; ++i) {
    o.sample_indices[static_cast<size_t>(i)] = 0;
    o.sample_pos[static_cast<size_t>(i)] = 0;
    o.sample_idx_mapping[static_cast<size_t>(i)] = -1;
  }
  for (int32_t i = num_reqs * nqpr; i < b.max_num_tokens; ++i)
    o.query_slot_mapping[static_cast<size_t>(i)] = kPadSlotId;
  return o;
}

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
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
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
    if (per_layer_out != nullptr) {
      DBuf tmp(d, DType::kF32, {T, H});
      vt::CastF32(d.q, tmp.t(), down.t());
      std::vector<float> lh(static_cast<size_t>(T) * H);
      tmp.Download(d, lh.data());
      per_layer_out->push_back(std::move(lh));
    }
    hidden = std::move(down);
  }

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  if (final_out != nullptr) {
    DBuf tmp(d, DType::kF32, {T, H});
    vt::CastF32(d.q, tmp.t(), dnorm.t());
    final_out->assign(static_cast<size_t>(T) * H, 0.0f);
    tmp.Download(d, final_out->data());
  }

  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {T, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);
  std::vector<float> out(static_cast<size_t>(T) * vocab);
  logits.Download(d, out.data());
  return out;
}

Qwen3DFlashModel::ContextKV Qwen3DFlashModel::PrecomputeContextKV(
    const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t C = static_cast<int64_t>(context_positions.size());
  VT_CHECK(static_cast<int64_t>(context_states.size()) == C * H,
           "PrecomputeContextKV: context_states must be [num_ctx, H]");

  ContextKV ckv;
  ckv.num_ctx = C;
  if (C == 0) {
    ckv.k.assign(static_cast<size_t>(config.num_hidden_layers), {});
    ckv.v.assign(static_cast<size_t>(config.num_hidden_layers), {});
    return ckv;
  }

  // Device-resident projection (D7); download each layer's K/V to the host host
  // ContextKV the CPU/parity gates read. Bit-identical to the old inline path
  // (same ops, same order) — this public host API exists ONLY for the D3 kvprep
  // gates; production reaches the device buffers directly (ForwardBlockLogitsWithContext).
  ContextKVDev dev = PrecomputeContextKVDevice(d, context_states.data(),
                                               context_positions.data(), C, weights, config);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    std::vector<float> kh(static_cast<size_t>(C) * Hkv * Dh);
    std::vector<float> vh(static_cast<size_t>(C) * Hkv * Dh);
    DBuf tk(d, DType::kF32, {C, Hkv, Dh});
    vt::CastF32(d.q, tk.t(), Reshape(dev.k[static_cast<size_t>(l)].t(), {C, Hkv, Dh}));
    tk.Download(d, kh.data());
    DBuf tv(d, DType::kF32, {C, Hkv, Dh});
    vt::CastF32(d.q, tv.t(), Reshape(dev.v[static_cast<size_t>(l)].t(), {C, Hkv, Dh}));
    tv.Download(d, vh.data());
    ckv.k.push_back(std::move(kh));
    ckv.v.push_back(std::move(vh));
  }
  return ckv;
}

// Shared core (D9): the [context; block] block forward GIVEN a device-resident
// per-layer context K/V (ContextKVDev). Both ForwardBlockLogitsWithContext (which
// re-projects the whole context every step) and ForwardBlockLogitsWithPrecomputedKV
// (which uploads the persistent append-only store) build the ckv and delegate here,
// so the two paths are byte-identical downstream of how ckv's bits were obtained.
static std::vector<float> ForwardWithCtxKVDev(
    Dev d, const ContextKVDev& ckv, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    std::vector<std::vector<float>>* per_layer_out, std::vector<float>* final_out) {
  const int64_t Tq = static_cast<int64_t>(block_input_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const int64_t vocab = weights.draft_vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  const int num_reqs = static_cast<int>(cu.size()) - 1;
  VT_CHECK(static_cast<int64_t>(block_positions.size()) == Tq,
           "ForwardWithCtxKVDev: block_positions length must match input_ids");
  VT_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == static_cast<int32_t>(Tq),
           "ForwardWithCtxKVDev: cu must span [0,Tq]");
  VT_CHECK(static_cast<int>(ctx_cu.size()) == num_reqs + 1 && ctx_cu.front() == 0,
           "ForwardWithCtxKVDev: ctx_cu must be [num_reqs+1]");
  const int64_t C = ckv.num_ctx;
  VT_CHECK(ctx_cu.back() == static_cast<int32_t>(C),
           "ForwardWithCtxKVDev: ctx_cu.back() must equal num_ctx");

  // Combined [context; block] per-request layout for the attention (cu_comb), plus
  // the DEVICE index maps (D7) that place context/block rows into the combined
  // buffer with vt::IndexCopy and extract the block-query rows with vt::IndexSelect
  // — replacing the D5 host download + std::vector interleave + re-upload. These are
  // tiny integer maps computed once from the cu vectors and uploaded once.
  const int64_t Ncomb = C + Tq;
  std::vector<int32_t> cu_comb(static_cast<size_t>(num_reqs) + 1, 0);
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t cl = ctx_cu[static_cast<size_t>(r) + 1] - ctx_cu[static_cast<size_t>(r)];
    const int32_t bl = cu[static_cast<size_t>(r) + 1] - cu[static_cast<size_t>(r)];
    cu_comb[static_cast<size_t>(r) + 1] = cu_comb[static_cast<size_t>(r)] + cl + bl;
  }
  // ctx_dest[j] = combined row for context source row j (ctx_cu order).
  // blk_idx[i]  = combined row for block source row i (cu order); used BOTH to
  // scatter block q/k/v in (IndexCopy: comb[blk_idx[i]] = block[i]) AND to gather
  // block outputs back out (IndexSelect: out[i] = comb[blk_idx[i]]).
  std::vector<int32_t> ctx_dest(static_cast<size_t>(C));
  std::vector<int32_t> blk_idx(static_cast<size_t>(Tq));
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t c0 = ctx_cu[static_cast<size_t>(r)], c1 = ctx_cu[static_cast<size_t>(r) + 1];
    const int32_t b0 = cu[static_cast<size_t>(r)], b1 = cu[static_cast<size_t>(r) + 1];
    const int32_t base = cu_comb[static_cast<size_t>(r)];
    for (int32_t j = c0; j < c1; ++j) ctx_dest[static_cast<size_t>(j)] = base + (j - c0);
    const int32_t bbase = base + (c1 - c0);
    for (int32_t i = b0; i < b1; ++i) blk_idx[static_cast<size_t>(i)] = bbase + (i - b0);
  }
  DBuf ctx_dest_d(d, DType::kI32, {C}, ctx_dest.data());
  DBuf blk_idx_d(d, DType::kI32, {Tq}, blk_idx.data());

  // Embed block tokens; substitute the dedicated mask embedding when present.
  DBuf hidden(d, DType::kBF16, {Tq, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H});
    DBuf dids(d, DType::kI32, {Tq}, block_input_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }
  if (!weights.mask_embedding.Empty() && weights.mask_token_id >= 0) {
    Tensor mask_emb = ResidentWeight(d, weights.mask_embedding, {H});
    std::vector<float> mask_host(static_cast<size_t>(H));
    {
      DBuf tmp(d, DType::kF32, {H});
      vt::CastF32(d.q, tmp.t(), mask_emb);
      tmp.Download(d, mask_host.data());
    }
    std::vector<float> hidden_host(static_cast<size_t>(Tq) * H);
    {
      DBuf tmp(d, DType::kF32, {Tq, H});
      vt::CastF32(d.q, tmp.t(), hidden.t());
      tmp.Download(d, hidden_host.data());
    }
    for (int64_t rr = 0; rr < Tq; ++rr)
      if (block_input_ids[static_cast<size_t>(rr)] == weights.mask_token_id)
        for (int64_t j = 0; j < H; ++j)
          hidden_host[static_cast<size_t>(rr * H + j)] = mask_host[static_cast<size_t>(j)];
    DBuf hf(d, DType::kF32, {Tq, H}, hidden_host.data());
    vt::CastBf16(d.q, hidden.t(), hf.t());
  }

  DBuf res(d, DType::kBF16, {Tq, H});
  res.Zero(d);
  DBuf dpos(d, DType::kI32, {Tq}, block_positions.data());

  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3DFlashLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
    DBuf dhn(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

    // Block q/k/v: same per-layer path as the context-free forward.
    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    DBuf q(d, DType::kBF16, {Tq, qdim});
    DBuf k(d, DType::kBF16, {Tq, kdim});
    DBuf v(d, DType::kBF16, {Tq, kdim});
    Tensor wqkv = ResidentWeight(d, layer.qkv_proj);
    vt::MatmulBT(d.q, q.t(), dhn.t(), wqkv.Slice(0, 0, qdim));
    vt::MatmulBT(d.q, k.t(), dhn.t(), wqkv.Slice(0, qdim, qdim + kdim));
    vt::MatmulBT(d.q, v.t(), dhn.t(), wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim));
    Tensor q2 = Reshape(q.t(), {Tq * Hq, Dh});
    Tensor k2 = Reshape(k.t(), {Tq * Hkv, Dh});
    Tensor q3 = Reshape(q.t(), {Tq, Hq, Dh});
    Tensor k3 = Reshape(k.t(), {Tq, Hkv, Dh});
    vt::RmsNorm(d.q, q2, q2, ResidentWeight(d, layer.q_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, ResidentWeight(d, layer.k_norm, {Dh}), vt::RmsNormArgs{eps, false});
    vt::RopeNeox(d.q, q3, k3, dpos.t(), MakeRopeArgs(config));

    // Build the combined [context; block] q/k/v ON DEVICE (D7): scatter the layer's
    // device context K/V and this block's q/k/v into the packed combined buffer via
    // vt::IndexCopy — NO D->H download, NO host std::vector interleave, NO re-upload.
    // The bf16 values are bit-identical to the D5 f32-roundtrip path (bf16->f32->bf16
    // is an identity round-trip), so DFlashBlockAttention sees identical inputs.
    Tensor v3 = Reshape(v.t(), {Tq, Hkv, Dh});
    DBuf qcb(d, DType::kBF16, {Ncomb, Hq, Dh});
    DBuf kcb(d, DType::kBF16, {Ncomb, Hkv, Dh});
    DBuf vcb(d, DType::kBF16, {Ncomb, Hkv, Dh});
    qcb.Zero(d);  // context query rows are unused (their attn output is discarded)
    Tensor qcb3 = qcb.t(), kcb3 = kcb.t(), vcb3 = vcb.t();
    if (C > 0) {  // this layer's device context K/V -> combined at ctx_dest
      Tensor ck2 = Reshape(ckv.k[static_cast<size_t>(l)].t(), {C, Hkv, Dh});
      Tensor cv2 = Reshape(ckv.v[static_cast<size_t>(l)].t(), {C, Hkv, Dh});
      Tensor cdst = ctx_dest_d.t();
      vt::IndexCopy(d.q, kcb3, ck2, cdst);
      vt::IndexCopy(d.q, vcb3, cv2, cdst);
    }
    {  // block q/k/v -> combined at blk_idx
      Tensor bidx = blk_idx_d.t();
      vt::IndexCopy(d.q, qcb3, q3, bidx);
      vt::IndexCopy(d.q, kcb3, k3, bidx);
      vt::IndexCopy(d.q, vcb3, v3, bidx);
    }
    // Attention over the combined sequence via the UNCHANGED D2 primitive.
    DBuf acomb(d, DType::kBF16, {Ncomb, Hq, Dh});
    vt::DFlashBlockAttentionArgs pa;
    pa.scale = scale;
    pa.causal = layer.attn_mode.causal;
    pa.sliding_window = layer.attn_mode.sliding_window;
    pa.cu_seqlens = cu_comb.data();
    pa.num_reqs = num_reqs;
    vt::DFlashBlockAttention(d.q, acomb.t(), qcb.t(), kcb.t(), vcb.t(), pa);
    // Extract the block-query rows out of the combined output ON DEVICE (IndexSelect:
    // a[i] = acomb[blk_idx[i]]), replacing the D5 download + host row-scatter.
    DBuf a(d, DType::kBF16, {Tq, Hq * Dh});
    {
      Tensor acomb2 = Reshape(acomb.t(), {Ncomb, Hq * Dh});
      Tensor a2 = Reshape(a.t(), {Tq, Hq * Dh});
      Tensor bidx = blk_idx_d.t();
      vt::IndexSelect(d.q, a2, acomb2, bidx);
    }
    Tensor wo = ResidentWeight(d, layer.o_proj);
    DBuf attn(d, DType::kBF16, {Tq, H});
    vt::MatmulBT(d.q, attn.t(), a.t(), wo);

    // post_attention_layernorm + SwiGLU MLP (unchanged from ForwardBlockLogits).
    Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {Tq, H});
    if (FusedChainAdoptEnabled())
      vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
    else
      vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
    const int64_t I = config.intermediate_size;
    Tensor wgu = ResidentWeight(d, layer.gate_up_proj);
    DBuf gu(d, DType::kBF16, {Tq, 2 * I});
    vt::MatmulBT(d.q, gu.t(), dh2.t(), wgu);
    DBuf act(d, DType::kBF16, {Tq, I});
    vt::SiluAndMul(d.q, act.t(), gu.t());
    Tensor wdn = ResidentWeight(d, layer.down_proj);
    DBuf down(d, DType::kBF16, {Tq, H});
    vt::MatmulBT(d.q, down.t(), act.t(), wdn);
    if (per_layer_out != nullptr) {
      DBuf tmp(d, DType::kF32, {Tq, H});
      vt::CastF32(d.q, tmp.t(), down.t());
      std::vector<float> lh(static_cast<size_t>(Tq) * H);
      tmp.Download(d, lh.data());
      per_layer_out->push_back(std::move(lh));
    }
    hidden = std::move(down);
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {Tq, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  if (final_out != nullptr) {
    DBuf tmp(d, DType::kF32, {Tq, H});
    vt::CastF32(d.q, tmp.t(), dnorm.t());
    final_out->assign(static_cast<size_t>(Tq) * H, 0.0f);
    tmp.Download(d, final_out->data());
  }
  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {Tq, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);
  std::vector<float> out(static_cast<size_t>(Tq) * vocab);
  logits.Download(d, out.data());
  return out;
}

// Upload a persistent host bf16 PrecomputedContextKV into per-layer device buffers,
// producing the SAME ContextKVDev the full recompute (PrecomputeContextKVDevice) would
// build — the stored bf16 bits ARE the projection output (bit-identical, D9).
static ContextKVDev UploadContextKV(Dev d,
                                    const Qwen3DFlashModel::PrecomputedContextKV& store,
                                    const HfConfig& config) {
  const int64_t C = store.num_ctx;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  ContextKVDev out;
  out.num_ctx = C;
  VT_CHECK(store.k.size() == static_cast<size_t>(config.num_hidden_layers) &&
               store.v.size() == static_cast<size_t>(config.num_hidden_layers),
           "UploadContextKV: store must hold one K/V per hidden layer");
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    VT_CHECK(static_cast<int64_t>(store.k[static_cast<size_t>(l)].size()) == C * kdim &&
                 static_cast<int64_t>(store.v[static_cast<size_t>(l)].size()) == C * kdim,
             "UploadContextKV: per-layer K/V size must be num_ctx*kv_dim");
    DBuf k(d, DType::kBF16, {C, kdim},
           C > 0 ? store.k[static_cast<size_t>(l)].data() : nullptr);
    DBuf v(d, DType::kBF16, {C, kdim},
           C > 0 ? store.v[static_cast<size_t>(l)].data() : nullptr);
    out.k.push_back(std::move(k));
    out.v.push_back(std::move(v));
  }
  return out;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithContext(
    const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
    const std::vector<int32_t>& ctx_cu, const std::vector<int32_t>& block_input_ids,
    const std::vector<int32_t>& block_positions, const std::vector<int32_t>& cu,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue,
    std::vector<std::vector<float>>* per_layer_out, std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  // Per-step FULL recompute of the context K/V (D5/D7 path): re-projects the ENTIRE
  // growing context every step (O(context^2) total). The D9 persistent path
  // (ForwardBlockLogitsWithPrecomputedKV) replaces this with an append-only store.
  ContextKVDev ckv = PrecomputeContextKVDevice(d, context_states.data(),
                                               context_positions.data(),
                                               static_cast<int64_t>(context_positions.size()),
                                               weights, config);
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

void Qwen3DFlashModel::AppendContextKVHost(PrecomputedContextKV& store,
                                           const std::vector<float>& new_features,
                                           const std::vector<int32_t>& new_positions,
                                           const Qwen3DFlashWeights& weights,
                                           const HfConfig& config, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t H = config.hidden_size;
  const int64_t kdim = config.num_key_value_heads * config.head_dim;
  const int64_t L = config.num_hidden_layers;
  const int64_t count = static_cast<int64_t>(new_positions.size());
  if (store.k.empty()) {
    store.k.assign(static_cast<size_t>(L), {});
    store.v.assign(static_cast<size_t>(L), {});
  }
  VT_CHECK(store.k.size() == static_cast<size_t>(L) && store.v.size() == static_cast<size_t>(L),
           "AppendContextKVHost: store layer count mismatch");
  VT_CHECK(static_cast<int64_t>(new_features.size()) == count * H,
           "AppendContextKVHost: new_features must be [count, H]");
  if (count == 0) return;
  // Project ONLY the `count` new rows (positions == their absolute positions), reusing
  // the EXACT per-row projection the full recompute runs, then download the bf16 K/V
  // and append. Per-row independence (hidden_norm/KV-GEMM/k_norm/RoPE) => these bits
  // equal what a full C-row recompute would produce for these same rows.
  ContextKVDev dev = PrecomputeContextKVDevice(d, new_features.data(), new_positions.data(),
                                               count, weights, config);
  for (int64_t l = 0; l < L; ++l) {
    std::vector<uint16_t> kh(static_cast<size_t>(count) * kdim);
    std::vector<uint16_t> vh(static_cast<size_t>(count) * kdim);
    dev.k[static_cast<size_t>(l)].Download(d, kh.data());  // raw bf16 bits
    dev.v[static_cast<size_t>(l)].Download(d, vh.data());
    std::vector<uint16_t>& sk = store.k[static_cast<size_t>(l)];
    std::vector<uint16_t>& sv = store.v[static_cast<size_t>(l)];
    sk.insert(sk.end(), kh.begin(), kh.end());
    sv.insert(sv.end(), vh.begin(), vh.end());
  }
  store.num_ctx += count;
}

std::vector<float> Qwen3DFlashModel::ForwardBlockLogitsWithPrecomputedKV(
    const PrecomputedContextKV& ckv_host, const std::vector<int32_t>& ctx_cu,
    const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
    const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
    vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out,
    std::vector<float>* final_out) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  ContextKVDev ckv = UploadContextKV(d, ckv_host, config);
  return ForwardWithCtxKVDev(d, ckv, ctx_cu, block_input_ids, block_positions, cu, weights,
                             config, per_layer_out, final_out);
}

}  // namespace vllm
