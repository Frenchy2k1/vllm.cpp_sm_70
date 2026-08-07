// Parakeet / FastConformer AUDIO encoder + CTC head — spike row
// `MODEL-AUDIO-PARAKEET-ENCODER` (work item P4 of
// .agents/specs/parakeet-conformer-encoder.md), the model layer over the P1-P3
// kernels vt::Conv2d / vt::DepthwiseConv1d / vt::AttentionRelPos.
//
// Ported from transformers 5.3.0
// `transformers/models/parakeet/modeling_parakeet.py`:
//   ParakeetEncoderRelPositionalEncoding :51  (forward :71-98)
//   ParakeetEncoderFeedForward           :101 (forward :109-113)
//   ParakeetEncoderConvolutionModule     :116 (ctor :123-149, forward :151-185)
//   ParakeetEncoderAttention             :259 (ctor :262-289, forward :291-346,
//                                              _rel_shift :348-354)
//   ParakeetEncoderSubsamplingConv2D     :357 (ctor :358-391,
//                                              _get_output_length :393-402,
//                                              forward :404-423)
//   ParakeetEncoderBlock                 :426 (forward :442-470)
//   ParakeetEncoder                      :549 (forward :576-640)
//   ParakeetForCTC                       :675 (ctor :678-684, forward :688-757,
//                                              generate :759-811)
//   ParakeetPreTrainedModel._get_subsampling_output_length :515-530,
//                          _get_output_attention_mask      :532-541
//   configuration_parakeet.py ParakeetEncoderConfig :23, ParakeetCTCConfig :152
//   tokenization_parakeet.py  ParakeetTokenizer._decode :28-49 (the CTC collapse)
//
// **RECORDED DEVIATION — the mirror source is HF, not vLLM.** vLLM does NOT
// implement this encoder: `vllm/model_executor/models/parakeet.py:14` does
// `from transformers import ParakeetEncoder as HFParakeetEncoder` and `:61`
// (`ProjectedParakeet.__init__`) instantiates it, so HF transformers IS what
// vLLM runs and IS the honest provenance for every line below. The vLLM-native
// pieces (`ParakeetProjection:27`, `ProjectedParakeet:48`, `ParakeetExtractor:138`,
// `vllm/transformers_utils/configs/parakeet.py` `ParakeetConfig:8` /
// `ExtractorConfig:41`) are cited where they apply — the extractor drives
// include/vllm/multimodal/parakeet_audio_processor.h. The structurally identical
// vLLM-native conformer `vllm/model_executor/models/conformer_encoder.py`
// (`ConformerEncoder:289`) is the P5 reuse target, not this row's source.
//
// **SCOPE: the CTC head.** `ParakeetForCTC` (:675) is upstream-defined, so the
// CTC head and its greedy collapse are mirror work.
//
// **CORRECTED 2026-08-07.** This block used to end "The RNN-T / TDT transducer
// has NO upstream in either vLLM or HF and is deliberately NOT implemented; it
// stays a product call." That was measured against the transformers installed on
// the spike box, 5.3.0, which ships only `ParakeetForCTC`. It is FALSE of
// upstream `main`, which implements `ParakeetRNNTDecoder:831`,
// `ParakeetRNNTJointNetwork:879`, `ParakeetForRNNT:922`,
// `ParakeetTDTJointNetwork:1035` and `ParakeetForTDT:1052`, plus the greedy
// loops in `generation_parakeet.py`. The transducer is therefore MIRROR work and
// it lives in parakeet_transducer.h, which composes over this encoder.
//
// **TRACED, not read (T0).** `ParakeetEncoderAttention` selects its attention
// path at runtime (:306-308). On the pinned oracle dump the path that ACTUALLY
// ran is `sdpa` (recorded in the fixture manifest by
// scripts/mm/p4_parakeet_oracle_dump.py). That matters for one observable: a
// query row whose every key is masked yields exactly ZERO under `sdpa`, where
// `eager` would yield NaN. vt::AttentionRelPos documents zero, so our forward
// mirrors the path that runs.
//
// Numeric contract: f32 end to end (activations, weights, accumulation), which
// is the dtype the HF reference runs at. This is a CORRECTNESS-grade forward:
// each stage marshals through the vt:: op boundary rather than staying
// device-resident, because P4's gate is faithfulness and the speed gate for
// this family is GB10-only (spike § Gates) and not claimed here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"

namespace vllm::multimodal {

// Mirror of `ParakeetEncoderConfig` (configuration_parakeet.py:23, defaults
// :96-119) plus the two `ParakeetCTCConfig` (:152, :196-204) fields the head
// needs. Field names match upstream 1:1.
struct ParakeetEncoderConfig {
  int64_t hidden_size = 1024;
  int64_t num_hidden_layers = 24;
  int64_t num_attention_heads = 8;
  int64_t num_key_value_heads = 8;  // == num_attention_heads upstream (:124)
  int64_t intermediate_size = 4096;
  bool attention_bias = true;
  bool convolution_bias = true;
  int64_t conv_kernel_size = 9;
  int64_t subsampling_factor = 8;
  int64_t subsampling_conv_channels = 256;
  int64_t num_mel_bins = 80;
  int64_t subsampling_conv_kernel_size = 3;
  int64_t subsampling_conv_stride = 2;
  int64_t max_position_embeddings = 5000;
  bool scale_input = true;
  // `hidden_act` is "silu" for every published Parakeet checkpoint and is the
  // upstream default (:102); both the feed-forward (:105) and the convolution
  // module (:129) read it. Kept as a string so a checkpoint that sets anything
  // else FAILS LOUDLY at load rather than silently running the wrong activation.
  std::string hidden_act = "silu";

  // torch `nn.LayerNorm` / `nn.BatchNorm1d` defaults — upstream constructs both
  // with no explicit eps (:146, :436-440).
  float layer_norm_eps = 1e-5f;
  float batch_norm_eps = 1e-5f;

  // ParakeetCTCConfig (:196-204). `pad_token_id` IS the CTC blank (:747).
  int64_t vocab_size = 1025;
  int32_t pad_token_id = 1024;

  int64_t head_dim() const { return hidden_size / num_attention_heads; }
  // int(math.log2(subsampling_factor)) (:365, :520).
  int64_t num_subsampling_layers() const;
  // (kernel_size - 1) // 2 (:364).
  int64_t subsampling_padding() const { return (subsampling_conv_kernel_size - 1) / 2; }
  // config.num_mel_bins // (stride ** num_layers) (:390) — the frequency extent
  // that feeds the subsampling projection.
  int64_t subsampling_out_freq() const;
};

// `_get_subsampling_output_length` (:515-530): per subsampling stage
// floor((L + add_pad) / stride) + 1 with add_pad = (k-1)//2*2 - k. Identical to
// `ParakeetEncoderSubsamplingConv2D._get_output_length` (:393-402) applied to
// the strided layers, which is why one helper serves both call sites.
int64_t ParakeetSubsamplingOutputLength(int64_t input_length,
                                        const ParakeetEncoderConfig& cfg);

// `ParakeetEncoderRelPositionalEncoding.forward` (:71-98). Returns the
// [2*seq_length-1, hidden_size] f32 table whose row p holds, INTERLEAVED,
// sin(pos[p]*inv_freq[i]) then cos(pos[p]*inv_freq[i]) for i in [0,hidden/2),
// with pos = seq_length-1, seq_length-2, ..., -(seq_length-1) (:79) and
// inv_freq[i] = 1 / 10000^(2i/hidden_size) (:60-66). Computed in f32 exactly as
// upstream forces (`maybe_autocast(enabled=False)`, :90).
std::vector<float> ParakeetRelPositionalEncoding(int64_t seq_length,
                                                 int64_t hidden_size);

// --- weights ----------------------------------------------------------------
// All host-side row-major f32 in TORCH's own parameter layout, so a reader can
// map each field to a state-dict key by name:
//   nn.Linear.weight  [out_features, in_features]
//   nn.Conv1d.weight  [out_channels, in_channels/groups, kernel]
//   nn.Conv2d.weight  [out_channels, in_channels/groups, kH, kW]
// A bias vector is EMPTY when the checkpoint has none (config.attention_bias or
// config.convolution_bias false — vLLM parakeet.py:112-131 documents that a
// transformers-v5 `convolution_bias=False` config drops those params entirely).

// ParakeetEncoderSubsamplingConv2D (:357). `layers` is
// [Conv2d(1,C), ReLU, (Conv2d(C,C,groups=C), Conv2d(C,C,1x1), ReLU) * (n-1)]
// (:369-388), so state-dict indices are 0, then 2/3, then 5/6, ...
struct ParakeetSubsamplingWeights {
  struct Stage {
    std::vector<float> depthwise_w, depthwise_b;  // [C,1,k,k], [C] (:375-384)
    std::vector<float> pointwise_w, pointwise_b;  // [C,C,1,1], [C] (:386)
  };
  std::vector<float> conv0_w, conv0_b;  // [C,1,k,k], [C] (:369-371)
  std::vector<Stage> stages;            // num_subsampling_layers() - 1
  std::vector<float> linear_w, linear_b;  // [hidden, C*out_freq], [hidden] (:391)
};

// ParakeetEncoderAttention (:262-289).
struct ParakeetAttentionWeights {
  std::vector<float> q_w, q_b;  // [H*Dh, hidden] (:272-274)
  std::vector<float> k_w, k_b;  // [Hkv*Dh, hidden] (:275-277)
  std::vector<float> v_w, v_b;  // [Hkv*Dh, hidden] (:278-280)
  std::vector<float> o_w, o_b;  // [hidden, H*Dh] (:281-283)
  std::vector<float> relative_k_w;  // [H*Dh, hidden], W_{k,R}, NEVER biased (:285)
  std::vector<float> bias_u;        // [H, Dh] global CONTENT bias  (:287)
  std::vector<float> bias_v;        // [H, Dh] global POSITION bias (:289)
};

// ParakeetEncoderFeedForward (:102-107).
struct ParakeetFeedForwardWeights {
  std::vector<float> linear1_w, linear1_b;  // [intermediate, hidden]
  std::vector<float> linear2_w, linear2_b;  // [hidden, intermediate]
};

// ParakeetEncoderConvolutionModule (:134-149). `norm` is a BatchNorm1d, so the
// eval-time affine needs its RUNNING BUFFERS, not just weight/bias.
struct ParakeetConvModuleWeights {
  std::vector<float> pointwise1_w, pointwise1_b;  // [2C, C, 1], [2C] (:134-136)
  std::vector<float> depthwise_w, depthwise_b;    // [C, 1, K], [C]   (:137-145)
  std::vector<float> norm_w, norm_b;              // [C], [C]         (:146)
  std::vector<float> norm_running_mean, norm_running_var;  // [C], [C]
  std::vector<float> pointwise2_w, pointwise2_b;  // [C, C, 1], [C]   (:147-149)
};

// ParakeetEncoderBlock (:431-440).
struct ParakeetEncoderLayerWeights {
  ParakeetFeedForwardWeights feed_forward1, feed_forward2;
  ParakeetAttentionWeights self_attn;
  ParakeetConvModuleWeights conv;
  std::vector<float> norm_feed_forward1_w, norm_feed_forward1_b;
  std::vector<float> norm_self_att_w, norm_self_att_b;
  std::vector<float> norm_conv_w, norm_conv_b;
  std::vector<float> norm_feed_forward2_w, norm_feed_forward2_b;
  std::vector<float> norm_out_w, norm_out_b;
};

struct ParakeetEncoderWeights {
  ParakeetSubsamplingWeights subsampling;
  std::vector<ParakeetEncoderLayerWeights> layers;  // num_hidden_layers
};

// ParakeetForCTC (:678-682): the encoder plus a kernel-1 Conv1d head, which is
// a Linear in disguise ("Conv rather than linear to be consistent with NeMO
// decoding layer", :681).
struct ParakeetForCTCWeights {
  ParakeetEncoderWeights encoder;
  std::vector<float> ctc_head_w;  // [vocab_size, hidden, 1]
  std::vector<float> ctc_head_b;  // [vocab_size]
};

// Optional per-stage capture for the unit gates; a production caller passes
// nullptr and pays nothing. Each is host f32, row-major, in the shape noted.
struct ParakeetEncoderCapture {
  // [T_out, hidden] as `self.subsampling(...)` returns it (:607), i.e. BEFORE
  // the `* self.input_scale` on the next line (:608).
  std::vector<float> subsampling_out;
  std::vector<float> pos_embed;        // [2*T_out-1, hidden] (:609)
  std::vector<float> layer0_ff1;       // [T_out, hidden] (:450)
  std::vector<float> layer0_attn;      // [T_out, hidden] (:454-459)
  std::vector<float> layer0_conv;      // [T_out, hidden] (:462)
  std::vector<float> block0_out;       // [T_out, hidden] (:470)
};

// --- forward ----------------------------------------------------------------

// `ParakeetEncoder.forward` (:576-640) for ONE clip.
//   input_features : [num_frames, num_mel_bins] host f32, the extractor output
//                    for a single item (upstream's [B, T, mel] with B == 1).
//   valid_frames   : `attention_mask.sum(-1)` for that item (:406). Pass
//                    num_frames for an unpadded clip.
// Returns the encoder hidden states [T_out, hidden_size] host f32, where
// T_out == ParakeetSubsamplingOutputLength(num_frames, cfg) — the PADDED
// length, exactly as upstream returns (:638). `out_valid_frames`, when non-null,
// receives ParakeetSubsamplingOutputLength(valid_frames, cfg), i.e. the count of
// rows the caller may trust (:537-541).
std::vector<float> ParakeetEncoderForward(const std::vector<float>& input_features,
                                          int64_t num_frames, int64_t valid_frames,
                                          const ParakeetEncoderWeights& w,
                                          const ParakeetEncoderConfig& cfg,
                                          vt::Backend& backend,
                                          int64_t* out_valid_frames = nullptr,
                                          ParakeetEncoderCapture* capture = nullptr);

// `ParakeetForCTC.forward` (:688-757) + `generate` (:759-811) for one clip.
struct ParakeetCTCOutput {
  std::vector<float> logits;        // [num_output_frames, vocab_size] (:722)
  int64_t num_output_frames = 0;    // padded T_out
  int64_t valid_output_frames = 0;  // trustworthy prefix (:537-541)
  // argmax over the vocabulary per frame, with every frame at or past
  // valid_output_frames forced to the blank/pad id (:796-801).
  std::vector<int32_t> greedy_ids;
  // greedy_ids after the CTC collapse (tokenization_parakeet.py:38-42).
  std::vector<int32_t> token_ids;
};

ParakeetCTCOutput ParakeetForCTCForward(const std::vector<float>& input_features,
                                        int64_t num_frames, int64_t valid_frames,
                                        const ParakeetForCTCWeights& w,
                                        const ParakeetEncoderConfig& cfg,
                                        vt::Backend& backend);

// `ParakeetTokenizer._decode` (tokenization_parakeet.py:38-42): collapse runs of
// EQUAL consecutive ids to one (`itertools.groupby`), THEN drop every remaining
// blank. Order matters — collapsing after dropping blanks would merge two
// genuinely repeated tokens that a blank separates.
std::vector<int32_t> ParakeetCtcGreedyCollapse(const std::vector<int32_t>& ids,
                                               int32_t blank_id);

// --- checkpoint loading ------------------------------------------------------

// Load a HF-format `ParakeetForCTC` checkpoint from `dir`, which must hold a
// `config.json` and either `model.safetensors` or a
// `model.safetensors.index.json` shard set. Fills `cfg` from config.json
// (`ParakeetCTCConfig` with its nested `encoder_config`) and every tensor from
// the state dict, converting f16/bf16 storage to f32. Throws std::runtime_error
// naming the offending key on any missing/misshaped tensor, so a checkpoint that
// does not match the ported architecture fails loudly instead of silently
// producing garbage.
ParakeetForCTCWeights LoadParakeetForCTC(const std::string& dir,
                                         ParakeetEncoderConfig* cfg);

// Parse just the config half of the above (config.json in `dir`). Exposed so a
// caller can size buffers before paying for the weights.
ParakeetEncoderConfig LoadParakeetConfig(const std::string& dir);

}  // namespace vllm::multimodal
