// Voxtral-Mini-3B e2e AUDIO->TEXT forward + weight loader (audio-track A3).
// See include/vllm/model_executor/models/voxtral.h for full provenance.
//
// Ported 1:1 from vllm/model_executor/models/voxtral.py @ e24d1b24
// (embed_multimodal:382-412, AudioLanguageAdapter:660-668, load_weights:502-568,
// VoxtralEncoderModel.mistral_remapping:674-724). The audio encoder is the A2
// WhisperAudioEncoderForward at Voxtral's encoder config; the text decoder is the
// LANDED shared dense forward (dense_attn::AttnBlock) driven from a merged
// inputs_embeds. Pure additive TU (no shared forward / runner / registry edit).
#include "vllm/model_executor/models/voxtral.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"  // LinearMethod seam
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"   // AttnBlock, BuildStepInputs, ResidentWeight
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"       // Qwen3VLMergeMultimodal (modality-agnostic merge)
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

using dense_attn::AttnBlock;
using dense_attn::BuildStepInputs;
using dense_attn::Dev;
using dense_attn::DBuf;
using dense_attn::FusedChainAdoptEnabled;
using dense_attn::ResidentWeight;
using dense_attn::StepInputs;
using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// --- bf16 StTensor -> host f32 vector (encoder + adapter weights). --------------
std::vector<float> StBf16ToF32(const StTensor& t) {
  VT_CHECK(t.dtype == "BF16", "voxtral: expected BF16 tensor");
  const auto* src = reinterpret_cast<const uint16_t*>(t.data);
  const size_t n = t.nbytes / sizeof(uint16_t);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = vt::BF16ToF32(src[i]);
  return out;
}

std::vector<float> Bf16BitsToF32(const uint16_t* p, int64_t n) {
  std::vector<float> o(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) o[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
  return o;
}
std::vector<uint16_t> F32ToBf16Bits(const float* p, int64_t n) {
  std::vector<uint16_t> o(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) o[static_cast<size_t>(i)] = vt::F32ToBF16(p[i]);
  return o;
}
void RoundToBf16(std::vector<float>& v) {
  for (float& x : v) x = vt::BF16ToF32(vt::F32ToBF16(x));
}
int64_t ArgMax(const std::vector<float>& logits) {
  int64_t am = 0;
  for (int64_t v = 1; v < static_cast<int64_t>(logits.size()); ++v)
    if (logits[static_cast<size_t>(v)] > logits[static_cast<size_t>(am)]) am = v;
  return am;
}

// Single-sequence CommonAttentionMetadata (mirror qwen3_vl.cpp StepMeta).
CommonAttentionMetadata StepMeta(int64_t T, int64_t seq_len, int64_t first_slot) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq_len)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(seq_len);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(first_slot + t);
  m.causal = true;
  return m;
}

// One Mistral/Llama decoder layer (== qwen3.cpp RunLayer, no qk-norm branch since
// Voxtral text leaves q_norm/k_norm EMPTY): input norm -> AttnBlock -> post norm ->
// SwiGLU MLP. `hidden` is the delta stream, `res` the residual accumulator.
void RunTextLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
                  DBuf& hidden, DBuf& res, const StepInputs& si,
                  const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());

  DBuf attn = AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

  auto gate_up = layers::MakeMlpGateUpMethod(layer.mlp.gate_up_proj,
                                             layer.mlp.gate_proj_fp4,
                                             layer.mlp.up_proj_fp4, I);
  DBuf act = gate_up->Apply(d, dh2.t());
  auto down = layers::MakeLinearMethod(layer.mlp.down_proj, layer.mlp.down_proj_fp4);
  hidden = down->Apply(d, act.t(), DType::kBF16);
}

// Forked dense forward from a merged inputs_embeds: N layers -> final RMSNorm ->
// UNTIED lm_head, returning the LAST row's logits [vocab] host f32. Mirrors
// qwen3.cpp ForwardBody but (1) starts from provided embeds and (2) always untied.
std::vector<float> ForwardLastLogits(Dev d, const std::vector<uint16_t>& embeds_bf16,
                                     const std::vector<int32_t>& positions, int64_t T,
                                     const CommonAttentionMetadata& meta,
                                     const std::vector<PagedKvCache>& attn_kv,
                                     const Qwen3DenseWeights& weights,
                                     const HfConfig& config) {
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  DBuf hidden(d, DType::kBF16, {T, H}, embeds_bf16.data());
  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, meta, config);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunTextLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si, meta,
                 attn_kv[static_cast<size_t>(l)], T);

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());

  // Gather the LAST row, then UNTIED lm_head (Matmul-B [H, vocab]).
  DBuf last(d, DType::kBF16, {1, H});
  d.b.Copy(d.q, last.ptr(),
           static_cast<char*>(dnorm.ptr()) +
               static_cast<size_t>((T - 1) * H) * vt::SizeOf(DType::kBF16),
           static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));
  Tensor lm = ResidentWeight(d, weights.lm_head);  // [H, vocab]
  DBuf logits(d, DType::kF32, {1, vocab});
  vt::Matmul(d.q, logits.t(), last.t(), lm);

  std::vector<float> out(static_cast<size_t>(vocab));
  logits.Download(d, out.data());
  return out;
}

// --- weight loading helpers -----------------------------------------------------
using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;

void LoadEncoderWeights(const TensorResolver& get, const std::vector<float>& embed_positions,
                        const multimodal::WhisperAudioEncoderConfig& cfg,
                        multimodal::WhisperAudioEncoderWeights& w) {
  const std::string E = "mm_whisper_embeddings.whisper_encoder.";
  w.conv1_w = StBf16ToF32(get(E + "conv_layers.0.weight"));
  w.conv1_b = StBf16ToF32(get(E + "conv_layers.0.bias"));
  w.conv2_w = StBf16ToF32(get(E + "conv_layers.1.weight"));
  w.conv2_b = StBf16ToF32(get(E + "conv_layers.1.bias"));
  w.embed_positions_w = embed_positions;
  w.final_ln_w = StBf16ToF32(get(E + "transformer.norm.weight"));
  w.final_ln_b = StBf16ToF32(get(E + "transformer.norm.bias"));
  w.layers.resize(static_cast<size_t>(cfg.num_layers));
  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    const std::string p = E + "transformer.layers." + std::to_string(l) + ".";
    multimodal::WhisperEncoderLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.attn_ln_w = StBf16ToF32(get(p + "attention_norm.weight"));
    lw.attn_ln_b = StBf16ToF32(get(p + "attention_norm.bias"));
    lw.final_ln_w = StBf16ToF32(get(p + "ffn_norm.weight"));
    lw.final_ln_b = StBf16ToF32(get(p + "ffn_norm.bias"));
    lw.q_w = StBf16ToF32(get(p + "attention.wq.weight"));
    lw.q_b = StBf16ToF32(get(p + "attention.wq.bias"));
    lw.k_w = StBf16ToF32(get(p + "attention.wk.weight"));  // k_proj: NO bias
    lw.v_w = StBf16ToF32(get(p + "attention.wv.weight"));
    lw.v_b = StBf16ToF32(get(p + "attention.wv.bias"));
    lw.out_w = StBf16ToF32(get(p + "attention.wo.weight"));
    lw.out_b = StBf16ToF32(get(p + "attention.wo.bias"));
    lw.fc1_w = StBf16ToF32(get(p + "feed_forward.w1.weight"));
    lw.fc1_b = StBf16ToF32(get(p + "feed_forward.w1.bias"));
    lw.fc2_w = StBf16ToF32(get(p + "feed_forward.w2.weight"));
    lw.fc2_b = StBf16ToF32(get(p + "feed_forward.w2.bias"));
  }
}

// Permute the rows of a bf16 [n_heads*head_dim, K] q/k weight from the Meta-
// interleaved rope layout (mistral consolidated) to the HF NeoX layout vLLM's
// rotary_emb (is_neox_style=True) expects — the EXACT transform vLLM applies on
// the mistral load path (verified bit-exact: permute(wq)==vLLM q_proj). Row map
// per head: out(j*hd2 + i) <- in(2i + j), hd2 = head_dim/2. Pure byte reorder of
// bf16 values (no arithmetic) => bit-exact. wv/wo are NOT permuted.
std::vector<uint16_t> PermuteQKBf16(const StTensor& t, int64_t n_heads) {
  VT_CHECK(t.dtype == "BF16" && t.shape.size() == 2, "voxtral: q/k permute needs 2-D BF16");
  const int64_t d1 = t.shape[0], K = t.shape[1];
  const int64_t hd = d1 / n_heads, hd2 = hd / 2;
  VT_CHECK(hd * n_heads == d1 && hd2 * 2 == hd, "voxtral: q/k permute head split mismatch");
  const auto* src = reinterpret_cast<const uint16_t*>(t.data);
  std::vector<uint16_t> out(static_cast<size_t>(d1) * K);
  for (int64_t h = 0; h < n_heads; ++h)
    for (int64_t i = 0; i < hd2; ++i)
      for (int64_t j = 0; j < 2; ++j) {
        const int64_t out_row = h * hd + j * hd2 + i;
        const int64_t in_row = h * hd + 2 * i + j;
        std::memcpy(&out[static_cast<size_t>(out_row) * K],
                    &src[static_cast<size_t>(in_row) * K],
                    static_cast<size_t>(K) * sizeof(uint16_t));
      }
  return out;
}

// Build the merged qkv OwnedTensor [Hq*Dh + 2*Hkv*Dh, K] (rows q|k|v) with q/k
// rope-permuted and v raw — the mistral-format analog of LoadMergedBf16RawNK.
OwnedTensor BuildPermutedQKV(const TensorResolver& get, const std::string& b,
                             const HfConfig& config) {
  const StTensor& wq = get(b + "attention.wq.weight");
  const StTensor& wk = get(b + "attention.wk.weight");
  const StTensor& wv = get(b + "attention.wv.weight");
  const int64_t K = wq.shape[1];
  const int64_t qd = wq.shape[0], kd = wk.shape[0], vd = wv.shape[0];
  std::vector<uint16_t> pq = PermuteQKBf16(wq, config.num_attention_heads);
  std::vector<uint16_t> pk = PermuteQKBf16(wk, config.num_key_value_heads);
  OwnedTensor m = dense_loaders::MakeOwned(DType::kBF16, {qd + kd + vd, K});
  auto* dst = reinterpret_cast<uint16_t*>(m.bytes.data());
  std::memcpy(dst, pq.data(), static_cast<size_t>(qd) * K * sizeof(uint16_t));
  std::memcpy(dst + qd * K, pk.data(), static_cast<size_t>(kd) * K * sizeof(uint16_t));
  std::memcpy(dst + (qd + kd) * K, wv.data, static_cast<size_t>(vd) * K * sizeof(uint16_t));
  m.nk = true;
  return m;
}

void LoadTextWeights(const TensorResolver& get, const HfConfig& config,
                     Qwen3DenseWeights& w) {
  w.tie_word_embeddings = false;
  w.attention_bias = false;
  w.embed_tokens = LoadBf16Direct(get, "mm_whisper_embeddings.tok_embeddings.weight");
  w.final_norm = LoadBf16Direct(get, "norm.weight");
  w.lm_head = LoadBf16Transposed(get, "output.weight");  // untied [vocab,H] -> [H,vocab]
  w.layers.resize(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    Qwen3DenseLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.input_layernorm = LoadBf16Direct(get, b + "attention_norm.weight");
    lw.post_attention_layernorm = LoadBf16Direct(get, b + "ffn_norm.weight");
    lw.attn.qkv_proj = BuildPermutedQKV(get, b, config);  // q/k rope-permuted, v raw
    lw.attn.o_proj = LoadMergedBf16RawNK(get, {b + "attention.wo.weight"});
    // SwiGLU: w1 = gate, w3 = up, w2 = down (mistral feed_forward naming).
    lw.mlp.gate_up_proj = LoadMergedBf16RawNK(
        get, {b + "feed_forward.w1.weight", b + "feed_forward.w3.weight"});
    lw.mlp.down_proj = LoadMergedBf16RawNK(get, {b + "feed_forward.w2.weight"});
  }
}

}  // namespace

multimodal::WhisperAudioEncoderConfig VoxtralEncoderConfig() {
  multimodal::WhisperAudioEncoderConfig c;
  c.d_model = 1280;
  c.num_heads = 20;   // head_dim = 1280/20 = 64
  c.num_layers = 32;
  c.ffn_dim = 5120;
  c.num_mel_bins = 128;
  c.max_source_positions = 1500;
  c.n_frames = 3000;
  c.norm_eps = 1e-5f;
  return c;
}

VoxtralWeights LoadVoxtralWeights(const SafetensorsFile& st,
                                  const std::vector<float>& embed_positions,
                                  const HfConfig& text_config) {
  const TensorResolver get = [&st](const std::string& name) -> const StTensor& {
    return st.Get(name);
  };
  VoxtralWeights w;
  w.encoder_cfg = VoxtralEncoderConfig();
  w.downsample_factor = 4;
  w.text_hidden = text_config.hidden_size;
  LoadEncoderWeights(get, embed_positions, w.encoder_cfg, w.encoder);
  w.adapter_w_in = StBf16ToF32(get("mm_whisper_embeddings.audio_language_projection.0.weight"));
  w.adapter_w_out = StBf16ToF32(get("mm_whisper_embeddings.audio_language_projection.2.weight"));
  LoadTextWeights(get, text_config, w.text);
  return w;
}

std::vector<float> VoxtralProjectAudio(const std::vector<float>& enc_out,
                                       const VoxtralWeights& w, Backend& b) {
  const int64_t D = w.encoder_cfg.d_model;                 // 1280
  const int64_t ds = w.downsample_factor;                  // 4
  const int64_t n_enc = static_cast<int64_t>(enc_out.size()) / D;
  // Pad n_enc up to a multiple of ds, then reshape (concat ds consecutive frames).
  const int64_t n_tok = (n_enc + ds - 1) / ds;
  const int64_t Kin = D * ds;                              // 5120
  const int64_t Ht = w.text_hidden;                        // 3072
  std::vector<float> reshaped(static_cast<size_t>(n_tok) * Kin, 0.0f);
  for (int64_t t = 0; t < n_enc; ++t)
    std::memcpy(&reshaped[static_cast<size_t>(t) * D], &enc_out[static_cast<size_t>(t) * D],
                static_cast<size_t>(D) * sizeof(float));

  Queue q = b.CreateQueue();
  auto up = [&](const std::vector<float>& f, std::vector<int64_t> shape) {
    auto bf = F32ToBf16Bits(f.data(), static_cast<int64_t>(f.size()));
    void* p = b.Alloc(bf.size() * sizeof(uint16_t));
    b.Copy(q, p, bf.data(), bf.size() * sizeof(uint16_t));
    Tensor t;
    t.data = p;
    t.dtype = DType::kBF16;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    return std::pair<void*, Tensor>{p, t};
  };
  auto x = up(reshaped, {n_tok, Kin});
  auto wi = up(w.adapter_w_in, {Ht, Kin});
  auto wo = up(w.adapter_w_out, {Ht, Ht});
  void* p_h = b.Alloc(static_cast<size_t>(n_tok) * Ht * sizeof(uint16_t));
  void* p_o = b.Alloc(static_cast<size_t>(n_tok) * Ht * sizeof(uint16_t));
  Tensor hbuf, obuf;
  auto mk = [&](void* p, int64_t r, int64_t c) {
    Tensor t;
    t.data = p; t.dtype = DType::kBF16; t.device = q.device; t.rank = 2;
    t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
  };
  hbuf = mk(p_h, n_tok, Ht);
  obuf = mk(p_o, n_tok, Ht);
  vt::MatmulBT(q, hbuf, x.second, wi.second);   // w_in: [n_tok,Kin] @ [Ht,Kin]^T -> [n_tok,Ht]
  vt::GeluErf(q, hbuf, hbuf);                   // nn.GELU()
  vt::MatmulBT(q, obuf, hbuf, wo.second);       // w_out: [n_tok,Ht] @ [Ht,Ht]^T -> [n_tok,Ht]

  std::vector<uint16_t> out_bits(static_cast<size_t>(n_tok) * Ht);
  b.Copy(q, out_bits.data(), p_o, out_bits.size() * sizeof(uint16_t));
  b.Synchronize(q);
  std::vector<float> out = Bf16BitsToF32(out_bits.data(), n_tok * Ht);

  b.Free(x.first); b.Free(wi.first); b.Free(wo.first); b.Free(p_h); b.Free(p_o);
  b.DestroyQueue(q);
  return out;
}

std::vector<int32_t> VoxtralGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& audio_embeds,
    int32_t audio_token_id, int32_t eos_token_id, const VoxtralWeights& weights,
    const HfConfig& config, Queue& queue, int max_new_tokens) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t Dh = config.head_dim;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());

  // Audio placeholder mask.
  std::vector<bool> mask(static_cast<size_t>(T0), false);
  int64_t n_audio = 0;
  for (int64_t t = 0; t < T0; ++t)
    if (prompt_ids[static_cast<size_t>(t)] == audio_token_id) { mask[static_cast<size_t>(t)] = true; ++n_audio; }
  const int64_t N = static_cast<int64_t>(audio_embeds.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_audio, "voxtral: audio_embeds rows != audio-token count");

  // KV caches: one big block per layer sized for T0 + max_new_tokens.
  const int64_t block_size = T0 + max_new_tokens + 8;
  const size_t kv_bytes =
      static_cast<size_t>(2 * block_size * Hkv * Dh) * vt::SizeOf(DType::kBF16);
  std::vector<std::shared_ptr<void>> kv_storage;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    void* p = backend.Alloc(kv_bytes);
    backend.Memset(queue, p, 0, kv_bytes);
    kv_storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
    PagedKvCache kv;
    kv.data = p; kv.dtype = DType::kBF16; kv.num_blocks = 1; kv.block_size = block_size;
    kv.num_kv_heads = Hkv; kv.head_size = Dh;
    attn_kv.push_back(kv);
  }

  // PREFILL: embed prompt ids, download, masked-scatter audio embeds, re-upload.
  std::vector<uint16_t> emb_bits(static_cast<size_t>(T0 * H));
  {
    DBuf ids(d, DType::kI32, {T0}, prompt_ids.data());
    DBuf emb(d, DType::kBF16, {T0, H});
    Tensor tab = ResidentWeight(d, weights.text.embed_tokens, {config.vocab_size, H});
    vt::Embedding(d.q, emb.t(), tab, ids.t());
    emb.Download(d, emb_bits.data());
  }
  std::vector<float> embeds = Bf16BitsToF32(emb_bits.data(), T0 * H);
  std::vector<float> aud = audio_embeds;
  RoundToBf16(aud);
  multimodal::Qwen3VLMergeMultimodal(embeds, T0, H, aud, mask);
  std::vector<uint16_t> merged_bits = F32ToBf16Bits(embeds.data(), T0 * H);

  std::vector<int32_t> pos_prefill(static_cast<size_t>(T0));
  for (int64_t t = 0; t < T0; ++t) pos_prefill[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  const CommonAttentionMetadata pm = StepMeta(T0, T0, 0);
  std::vector<float> logits =
      ForwardLastLogits(d, merged_bits, pos_prefill, T0, pm, attn_kv, weights.text, config);

  std::vector<int32_t> generated;
  int32_t next = static_cast<int32_t>(ArgMax(logits));
  generated.push_back(next);

  // DECODE: one token per step (no audio), 1-D position abs_idx.
  for (int step = 1; step < max_new_tokens; ++step) {
    if (next == eos_token_id) break;
    const int64_t abs_idx = T0 + (step - 1);
    std::vector<uint16_t> tok_emb(static_cast<size_t>(H));
    {
      const std::vector<int32_t> one = {next};
      DBuf ids(d, DType::kI32, {1}, one.data());
      DBuf emb(d, DType::kBF16, {1, H});
      Tensor tab = ResidentWeight(d, weights.text.embed_tokens, {config.vocab_size, H});
      vt::Embedding(d.q, emb.t(), tab, ids.t());
      emb.Download(d, tok_emb.data());
    }
    const std::vector<int32_t> pos1 = {static_cast<int32_t>(abs_idx)};
    const int64_t seq_len = abs_idx + 1;
    const CommonAttentionMetadata dm = StepMeta(1, seq_len, abs_idx);
    logits = ForwardLastLogits(d, tok_emb, pos1, 1, dm, attn_kv, weights.text, config);
    next = static_cast<int32_t>(ArgMax(logits));
    generated.push_back(next);
  }
  return generated;
}

}  // namespace vllm
