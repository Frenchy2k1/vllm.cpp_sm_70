// Voxtral-Mini-3B (`VoxtralForConditionalGeneration`) — audio-track A3: the first
// e2e AUDIO->TEXT understanding in the tree. A Whisper-large-v3-class audio encoder
// (A2 tower at Voxtral's config) + an AudioLanguageAdapter projector + a masked-
// scatter merge into the LANDED Mistral/Llama text decoder, forked greedy decode.
//
// Ported 1:1 from vllm/model_executor/models/voxtral.py @ e24d1b24:
//   VoxtralForConditionalGeneration.embed_multimodal:382-412 — encoder ->
//     downsample-concat reshape ([n_enc, d_model] -> [n_enc/ds, d_model*ds]) ->
//     AudioLanguageAdapter -> split back;
//   AudioLanguageAdapter:660-668 — w_in(Linear,no-bias) -> nn.GELU() -> w_out
//     (Linear,no-bias);
//   VoxtralEncoderModel.forward:819-839 (chunked WhisperEncoder), compute_whisper_
//     melspec:754-786 (torch.stft log-mel — A1 processor at Voxtral config);
//   load_weights:502-568 + VoxtralEncoderModel.mistral_remapping:674-724 — the
//     consolidated (mistral) checkpoint name map;
//   config: audio_token_id=24, audio_config {d_model 1280, 32 layers, 20 heads,
//     head_dim 64, ffn 5120, 128 mel bins, max_source_positions 1500}, downsample
//     _factor 4 (=> 1500/4 = 375 audio tokens per 30 s chunk), text_config (llama:
//     hidden 3072, 30 layers, 32 heads, 8 kv, head_dim 128, ffn 8192, rope_theta
//     1e8, vocab 131072, untied lm_head, NO qk-norm, NO attention bias).
//
// The audio encoder is the A2 `WhisperAudioEncoderForward` instantiated at
// Voxtral's encoder config (128 mel / 1280 d_model / 32 layers / head_dim 64) —
// structurally identical to the whisper-small A2 tower, only bigger; k_proj carries
// no bias exactly as A2. The text decoder is the LANDED shared dense forward
// (dense_attn::AttnBlock — qk-norm-optional, 1-D NeoX RoPE from config, GQA, paged
// FA2) driven from a merged inputs_embeds; the ONLY fork vs the plain Mistral text
// path is (1) inputs_embeds start (audio rows masked-scattered) and (2) untied
// lm_head. This TU is PURE ADDITIVE — it does NOT touch the shared dense forward /
// model runner / registry / any other model TU, so the Mistral text SACRED gate and
// every image/video/audio-pipeline/encoder gate is byte-identical by construction.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"           // Qwen3DenseWeights, PagedKvCache
#include "vllm/model_executor/models/whisper_audio.h"    // A2 encoder tower
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"

namespace vllm {

class SafetensorsFile;

// The full Voxtral checkpoint: the A2 Whisper-class encoder tower (at Voxtral's
// encoder config) + the AudioLanguageAdapter projector + the Mistral/Llama text
// backbone (shared dense container, untied lm_head, no qk-norm).
struct VoxtralWeights {
  multimodal::WhisperAudioEncoderWeights encoder;
  multimodal::WhisperAudioEncoderConfig encoder_cfg;
  // AudioLanguageAdapter (no bias). Host f32 in torch Linear [out, in] layout:
  //   w_in  [text_hidden, d_model*downsample_factor]
  //   w_out [text_hidden, text_hidden]
  std::vector<float> adapter_w_in;
  std::vector<float> adapter_w_out;
  int64_t downsample_factor = 4;
  int64_t text_hidden = 3072;
  Qwen3DenseWeights text;
};

// Default encoder config for Voxtral-Mini-3B (Whisper-large-v3 class). d_model 1280,
// 20 heads (head_dim 64), 32 layers, ffn 5120, 128 mel bins, 1500 source positions,
// 3000 input frames.
multimodal::WhisperAudioEncoderConfig VoxtralEncoderConfig();

// Load the full Voxtral consolidated.safetensors (mistral naming) into VoxtralWeights.
// `embed_positions` is the [max_source_positions*d_model] sinusoid golden (not in the
// checkpoint — vLLM computes it via transformers.sinusoids). `text_config` is the
// Mistral/Llama text HfConfig.
VoxtralWeights LoadVoxtralWeights(const SafetensorsFile& st,
                                  const std::vector<float>& embed_positions,
                                  const HfConfig& text_config);

// Project the encoder output [n_enc, d_model] to audio embeddings [n_enc/ds,
// text_hidden]: pad n_enc to a multiple of ds, reshape (concat ds consecutive
// frames) -> [n_enc/ds, d_model*ds], then w_in -> GELU(erf) -> w_out. Host f32.
// (voxtral.py embed_multimodal:392-407 + AudioLanguageAdapter:667.)
std::vector<float> VoxtralProjectAudio(const std::vector<float>& enc_out,
                                       const VoxtralWeights& w, vt::Backend& backend);

// Single-clip, single-sequence GREEDY audio->text generation (the A3 gate driver).
// `prompt_ids` is the placeholder-expanded model input (audio_token_id repeated
// num_audio_tokens times at the audio span). `audio_embeds` [num_audio_tokens *
// text_hidden] (== VoxtralProjectAudio output) is masked-scattered into the audio
// rows of the text embeddings; then the LANDED Mistral decoder runs greedy (1-D
// NeoX RoPE, untied lm_head) — text-only Mistral is byte-identical since this is an
// additive driver. Returns the generated token ids (<= max_new_tokens; stops on eos).
std::vector<int32_t> VoxtralGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& audio_embeds,
    int32_t audio_token_id, int32_t eos_token_id, const VoxtralWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens);

}  // namespace vllm
