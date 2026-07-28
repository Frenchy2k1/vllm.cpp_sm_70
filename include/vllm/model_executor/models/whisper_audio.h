// Whisper-class AUDIO encoder tower (`WhisperEncoder`) — audio-track A2 standalone
// forward, proven faithful to the transformers/vLLM reference in ISOLATION.
//
// Ported from:
//   transformers models/whisper/modeling_whisper.py @ 5.13.1
//     WhisperEncoder (:586), forward (:641-721) — conv1/conv2 + gelu + transpose +
//       embed_positions add + N pre-norm encoder layers + final layer_norm;
//     WhisperEncoderLayer (:371), forward (:400-430) — self_attn_layer_norm ->
//       self_attn -> residual -> final_layer_norm -> mlp(fc1,gelu,fc2) -> residual;
//     WhisperAttention (:244), forward (:298-368) — q_proj(bias)*scaling, k_proj
//       (NO bias), v_proj(bias), full non-causal MHA (scaling folded into the call
//       as 1.0), out_proj(bias); scaling = head_dim**-0.5;
//     sinusoids (:54) — the fixed embed_positions.weight [max_source_positions,
//       d_model] (dumped as a golden constant like A1's mel filterbank, so the
//       encoder-block math is the parity variable).
//   vllm/model_executor/models/whisper.py @ e24d1b24: WhisperEncoder (:458),
//     WhisperEncoderLayer (:353), WhisperMLP (:322), conv stride (:473-476) — the
//     faithful vLLM port this mirrors (identical structure).
//
// This is the A2 increment: the encoder tower proven faithful vs a dumped oracle
// reference in isolation (the audio->text e2e is A3 on Voxtral-Mini-3B over the
// LANDED Mistral backbone). Pure additive TU — it does NOT touch the shared model
// runner / registry / any other model forward, so the text/image/video/audio-
// pipeline gates are byte-identical by construction.
//
// Delta from the M2a Qwen3-VL VISION tower (include/.../qwen3_vl_vision.h): NO
// patch-merger, NO DeepStack, NO vision-RoPE (the Whisper encoder is fully
// bidirectional, no positional rotation — a fixed additive sinusoid instead), a
// CONV frontend (2x Conv1d, the 2nd stride-2, halving 3000 mel frames -> 1500
// encoder tokens) rather than a patchify matmul, GELU-erf everywhere (both the
// conv activations and the MLP; vision used tanh-GELU in the MLP), and pre-norm
// blocks with LayerNorm names self_attn_layer_norm / final_layer_norm. The convs
// are composed as im2col + the existing vt::MatmulBT (no new CUDA kernel).
//
// Numeric contract: production dtype bf16 (conv/attn/mlp GEMMs bf16, softmax/norm
// accumulation f32), mirroring the M2a vision tower and matching vLLM's bf16
// encoder. The reference is dumped in bf16 too, so the only parity variable is the
// bf16-kernel-stack accumulation divergence (the measured bf16-depth envelope).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "vt/backend.h"

namespace vllm::multimodal {

struct WhisperAudioEncoderConfig {
  int64_t d_model = 768;               // whisper-small hidden size
  int64_t num_heads = 12;              // encoder_attention_heads
  int64_t num_layers = 12;             // encoder_layers
  int64_t ffn_dim = 3072;             // encoder_ffn_dim
  int64_t num_mel_bins = 80;          // conv1 in-channels (== A1 n_mels)
  int64_t max_source_positions = 1500;  // encoder output length (== A1 num_audio_tokens)
  int64_t n_frames = 3000;            // input_features time dim (== A1 nb_max_frames)
  // nn.LayerNorm default eps (transformers WhisperEncoder uses nn.LayerNorm() with
  // the default 1e-5 for every layer norm — NOT the 1e-6 of the vision tower).
  float norm_eps = 1e-5f;

  int64_t head_dim() const { return d_model / num_heads; }
};

// All weights are host-side row-major f32 (as stored by torch: Linear weight is
// [out, in], Conv1d weight is [out_channels, in_channels, kernel]). LayerNorm
// weight/bias are [d_model]. k_proj has NO bias in Whisper (k_bias stays empty).
struct WhisperEncoderLayerWeights {
  std::vector<float> attn_ln_w, attn_ln_b;    // self_attn_layer_norm [d_model]
  std::vector<float> final_ln_w, final_ln_b;  // final_layer_norm [d_model]
  std::vector<float> q_w, q_b;                 // q_proj [d_model,d_model], bias
  std::vector<float> k_w;                      // k_proj [d_model,d_model], NO bias
  std::vector<float> v_w, v_b;                 // v_proj [d_model,d_model], bias
  std::vector<float> out_w, out_b;             // out_proj [d_model,d_model], bias
  std::vector<float> fc1_w, fc1_b;             // [ffn_dim,d_model], [ffn_dim]
  std::vector<float> fc2_w, fc2_b;             // [d_model,ffn_dim], [d_model]

  // Lazily-populated device-resident bf16 copies (CUDA/device forward only; null
  // before first use). Each host-f32 weight above is f32->bf16 converted and
  // uploaded to the device ONCE, then reused across every encoder forward — so
  // the per-call weight marshalling (host f32->bf16 conversion + H2D upload) that
  // dominated the encoder's host-side TTFT stops repeating. Mirrors the d_dev
  // residency seam in qwen3_5_weights.h (WhisperAudioEncoderForward populates
  // these on a const weight, exactly as the Qwen forward does; the shared_ptr
  // deleter frees through the vt::Backend, so the weights must NOT outlive the
  // backend — the same lifetime contract the decoder residents carry).
  mutable std::shared_ptr<void> d_attn_ln_w, d_attn_ln_b;
  mutable std::shared_ptr<void> d_final_ln_w, d_final_ln_b;
  mutable std::shared_ptr<void> d_q_w, d_q_b, d_k_w, d_v_w, d_v_b, d_out_w, d_out_b;
  mutable std::shared_ptr<void> d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b;
};

struct WhisperAudioEncoderWeights {
  std::vector<float> conv1_w, conv1_b;   // [d_model, num_mel_bins, 3], [d_model]
  std::vector<float> conv2_w, conv2_b;   // [d_model, d_model, 3], [d_model]
  std::vector<float> embed_positions_w;  // [max_source_positions, d_model] (sinusoid)
  std::vector<WhisperEncoderLayerWeights> layers;  // num_layers
  std::vector<float> final_ln_w, final_ln_b;       // final encoder layer_norm [d_model]

  // Device-resident bf16 copies of the non-layer weights, uploaded once and
  // reused (see the residency note in WhisperEncoderLayerWeights). d_embed_pos
  // holds only the first max_source_positions rows actually added (the sinusoid
  // table can be longer than L).
  mutable std::shared_ptr<void> d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b;
  mutable std::shared_ptr<void> d_embed_pos;
  mutable std::shared_ptr<void> d_final_ln_w, d_final_ln_b;
};

// Optional intermediate capture for the A2 unit gates (all host f32). When a
// pointer is passed, the forward downloads the gated stages; a production caller
// (A3) passes nullptr and pays nothing.
struct WhisperAudioEncoderCapture {
  std::vector<float> post_conv;     // [max_source_positions, d_model] after gelu(conv2)+transpose
  std::vector<float> post_pos;      // [max_source_positions, d_model] after + embed_positions
  std::vector<float> block0_out;    // [max_source_positions, d_model] after encoder layer 0
  std::vector<float> final_ln_out;  // [max_source_positions, d_model] after final layer_norm
};

// Runs the Whisper encoder on one clip. `input_features` is the A1 log-mel
// `[num_mel_bins, n_frames]` = [80, 3000] host f32. Returns the encoder hidden
// states `[max_source_positions, d_model]` = [1500, 768] as host f32. If capture
// != nullptr, the gated intermediate stages are filled.
std::vector<float> WhisperAudioEncoderForward(const std::vector<float>& input_features,
                                              const WhisperAudioEncoderWeights& w,
                                              const WhisperAudioEncoderConfig& cfg,
                                              vt::Backend& backend,
                                              WhisperAudioEncoderCapture* capture = nullptr);

}  // namespace vllm::multimodal
