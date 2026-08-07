// Parakeet RNN-T / TDT TRANSDUCER heads: the decode half of the Parakeet
// family, over the P4 encoder in parakeet_encoder.h.
//
// Ported from transformers `main`
// `transformers/models/parakeet/modeling_parakeet.py`:
//   ParakeetRNNTDecoder        :831 (ctor :834-844, forward :846-876)
//   ParakeetRNNTJointNetwork   :879 (ctor :882-886, forward :888-894)
//   ParakeetForRNNT            :922 (ctor :927-935, get_audio_features :938-950,
//                                    forward :954-1032)
//   ParakeetTDTJointNetwork    :1035 (ctor :1042-1044, the duration head)
//   ParakeetForTDT             :1052 (ctor :1055-1059, forward :1063-1142)
// `transformers/models/parakeet/generation_parakeet.py`:
//   ParakeetRNNTDecoderCache      :23  (lazy_initialization :31-58, update :60-82)
//   ParakeetRNNTGenerationMixin   :125 (_update_model_kwargs_for_generation :141-163,
//                                       _prepare_generated_length :165-190,
//                                       _prepare_model_inputs :192-228,
//                                       prepare_inputs_for_generation :233-248,
//                                       generate :250-268)
//   ParakeetTDTGenerationMixin    :271 (_update_model_kwargs_for_generation :280-299)
// `transformers/models/parakeet/configuration_parakeet.py`:
//   ParakeetRNNTConfig :136, ParakeetTDTConfig :188
//
// **CORRECTION: this head DOES have an upstream.** The P4 spike
// (.agents/specs/parakeet-conformer-encoder.md) recorded the transducer as
// having "NO upstream in either vLLM or HF transformers" and therefore being a
// product call rather than mirror work. That was measured against the LOCALLY
// INSTALLED transformers 5.3.0, which ships only `ParakeetForCTC`, and it is
// FALSE of current upstream: `main` implements the entire stack at the lines
// cited above. Everything in this file is therefore a MIRROR with citable
// file:line, exactly like the encoder. The spike is corrected in the same change
// that adds this header.
//
// **The mirror source is HF, not vLLM**, for the same reason the encoder's is:
// vLLM does not implement Parakeet, it imports it
// (`vllm/model_executor/models/parakeet.py:37,62`). vLLM wraps only the ENCODER
// (as the audio tower of Nemotron-VL), so it has no transducer call site at all
// and cannot be the provenance here.
//
// **TRACED, not read (T0).** The greedy loop's per-step decoder input shape is
// not obvious from `generate()`'s source, because `ParakeetRNNTDecoderCache` is
// NOT a `past_key_values` cache and so the base
// `prepare_inputs_for_generation` slicing rules do not visibly apply. A forward
// hook on `model.decoder` over a full `generate()` run records what ACTUALLY
// ran: every step feeds exactly `[1, 1]`, one token, so the LSTM advances one
// step per emission and never re-runs the prefix. That trace is recorded in the
// fixture manifest (`decoder_trace`) and asserted by the gate.
//
// Numeric contract: f32 end to end, the dtype the HF reference runs at, matching
// the encoder's contract. Batch 1, greedy only: `ParakeetForRNNT` declares
// `_supported_generation_modes = [GenerationMode.GREEDY_SEARCH]` (:925), so
// greedy is not a simplification, it is the whole supported surface.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vt/backend.h"

namespace vllm::multimodal {

// `ParakeetRNNTConfig` (configuration_parakeet.py:136) plus the `durations`
// field `ParakeetTDTConfig` (:188, :225) adds, plus the three
// generation_config.json fields the greedy loop reads. Field names match
// upstream 1:1.
struct ParakeetTransducerConfig {
  int64_t vocab_size = 8193;
  int64_t decoder_hidden_size = 640;
  int64_t num_decoder_layers = 2;
  // The JOINT's activation (:884). "relu" for every published transducer
  // checkpoint and the upstream default (:170). Distinct from the ENCODER's
  // `hidden_act`, which is "silu". Kept as a string so a checkpoint that sets
  // anything else FAILS LOUDLY rather than silently running the wrong function.
  std::string hidden_act = "relu";
  int64_t max_symbols_per_step = 10;
  int32_t blank_token_id = 8192;
  int32_t pad_token_id = 2;

  // TDT only: EMPTY means RNN-T. `ParakeetTDTConfig.durations` (:225), which
  // both widens the joint head (:1044) and drives the frame advance (:288-293).
  std::vector<int64_t> durations;

  // generation_config.json. `decoder_start_token_id` is the blank on every
  // published checkpoint (parakeet-rnnt-0.6b: 1024 == blank_token_id;
  // parakeet-tdt-0.6b-v3: 8192 == blank_token_id), which is also the token NeMo's
  // greedy transducer feeds at u=0. -1 means "not set": resolve to the blank.
  int32_t decoder_start_token_id = -1;
  // May be empty. parakeet-tdt-0.6b-v3 sets 3 (`<|endoftext|>`); rnnt-0.6b sets
  // none. A step that emits one of these ends the decode AFTER appending it,
  // which is `EosTokenCriteria`'s contract inside `generate()`.
  std::vector<int32_t> eos_token_ids;

  bool is_tdt() const { return !durations.empty(); }
  // `nn.Linear(decoder_hidden_size, vocab_size)` (:885) for RNN-T, widened to
  // `vocab_size + len(durations)` for TDT (:1044).
  int64_t joint_output_size() const {
    return vocab_size + static_cast<int64_t>(durations.size());
  }
  int32_t start_token() const {
    return decoder_start_token_id >= 0 ? decoder_start_token_id : blank_token_id;
  }
};

// --- weights ----------------------------------------------------------------
// Host-side row-major f32 in TORCH's own parameter layout, same convention as
// parakeet_encoder.h, so every field maps to a state-dict key by name.

// One `nn.LSTM` layer (:838-843). Torch packs the four gates ROW-WISE in the
// order INPUT, FORGET, CELL, OUTPUT: `weight_ih_l{k}` is
// [4*hidden, input_size] and `weight_hh_l{k}` is [4*hidden, hidden]. Both bias
// vectors are kept because torch applies BOTH (`b_ih + b_hh`), a redundancy it
// carries for cuDNN compatibility; dropping either would be wrong.
struct ParakeetLstmLayerWeights {
  std::vector<float> weight_ih;  // [4H, input_size]
  std::vector<float> weight_hh;  // [4H, H]
  std::vector<float> bias_ih;    // [4H]
  std::vector<float> bias_hh;    // [4H]
};

// `ParakeetRNNTDecoder` (:834-844): the prediction network.
struct ParakeetRNNTDecoderWeights {
  std::vector<float> embedding;                // [vocab_size, H]      (:837)
  std::vector<ParakeetLstmLayerWeights> lstm;  // num_decoder_layers   (:838-843)
  std::vector<float> projector_w, projector_b;  // [H, H], [H]         (:844)
};

// `ParakeetForRNNT` (:927-932) / `ParakeetForTDT` (:1055-1057).
struct ParakeetForTransducerWeights {
  ParakeetEncoderWeights encoder;                                // (:929)
  std::vector<float> encoder_projector_w, encoder_projector_b;   // [D, hidden],[D] (:930)
  ParakeetRNNTDecoderWeights decoder;                            // (:931)
  std::vector<float> joint_head_w, joint_head_b;  // [joint_output_size, D] (:885/:1044)
};

// --- prediction network ------------------------------------------------------

// `ParakeetRNNTDecoderCache` (generation_parakeet.py:23) for ONE sequence: the
// per-layer LSTM (h, c) plus `cache.cache`, the LAST decoder output. Zeroed at
// `lazy_initialization` (:31-58); `initialized` mirrors `is_initialized` (:29),
// which is what gates the blank fast path.
struct ParakeetRNNTDecoderState {
  std::vector<float> hidden;  // [num_decoder_layers, H]
  std::vector<float> cell;    // [num_decoder_layers, H]
  std::vector<float> output;  // [H], `cache.cache`, the projected decoder output
  bool initialized = false;
};

// One torch `nn.LSTMCell` step, the primitive `nn.LSTM` is built from:
//   i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
//   f = sigmoid(W_if x + b_if + W_hf h + b_hf)
//   g = tanh   (W_ig x + b_ig + W_hg h + b_hg)
//   o = sigmoid(W_io x + b_io + W_ho h + b_ho)
//   c' = f*c + i*g ; h' = o*tanh(c')
// Exposed so the unit gate can drive it against an independent in-test scalar
// reference without building a whole decoder.
void ParakeetLstmCell(const std::vector<float>& x, const ParakeetLstmLayerWeights& w,
                      int64_t input_size, int64_t hidden_size, std::vector<float>* h,
                      std::vector<float>* c);

// `ParakeetRNNTDecoder.forward` (:846-876) for ONE token, with the cache.
//
// The BLANK FAST PATH (:851-855) is the piece a transducer port most often gets
// wrong, so it is stated explicitly: when the state is already initialized and
// the input token IS the blank, upstream returns the cached output WITHOUT
// running the LSTM, so the hidden and cell states do NOT advance. A blank
// emission moves the encoder frame pointer and leaves the prediction network
// exactly where it was.
void ParakeetRNNTDecoderStep(int32_t input_id, const ParakeetRNNTDecoderWeights& w,
                             const ParakeetTransducerConfig& cfg,
                             ParakeetRNNTDecoderState* state);

// `ParakeetRNNTJointNetwork.forward` (:888-894): head(act(encoder + decoder)).
// `encoder_frame` is one row of the PROJECTED encoder output (:949), i.e.
// `pooler_output[t]`, length decoder_hidden_size. Returns
// [joint_output_size()]: for TDT the token logits occupy [0, vocab_size) and
// the duration logits the tail (:1124-1125).
std::vector<float> ParakeetTransducerJoint(const float* encoder_frame,
                                           const std::vector<float>& decoder_output,
                                           const ParakeetForTransducerWeights& w,
                                           const ParakeetTransducerConfig& cfg,
                                           vt::Backend& backend);

// --- greedy decode ------------------------------------------------------------

struct ParakeetTransducerOutput {
  // `ParakeetRNNTGenerateOutput.sequences` (generation_parakeet.py:107) exactly
  // as `generate()` returns it: the prepended `decoder_start_token_id` first,
  // then EVERY emitted token INCLUDING blanks.
  std::vector<int32_t> sequences;
  // `.durations` (:108), aligned with `sequences` and likewise zero-prefixed
  // (:259-262). For RNN-T this is the 0/1 encoder-frame ADVANCE (:156-160); for
  // TDT it is the predicted duration (:288-294).
  std::vector<int32_t> durations;
  // `sequences` with the blanks and the start token dropped: the ids a
  // transcript decodes from, since a transducer does NOT collapse repeats
  // (processing_parakeet.py:129 passes group_tokens=False for rnnt/tdt).
  std::vector<int32_t> token_ids;
  int64_t encoder_frames = 0;        // padded T
  int64_t valid_encoder_frames = 0;  // the frames the decode may walk
};

// The greedy transducer loop of `generate()` under
// `ParakeetRNNTGenerationMixin` / `ParakeetTDTGenerationMixin`, for one clip.
//
// `encoder_projected` is [frames, decoder_hidden_size]: `pooler_output`
// (:949), NOT the raw encoder output.
//
// The loop, with its upstream line for each rule:
//   * start from `decoder_start_token_id`, frame pointer 0 (:222-226);
//   * per step feed the LAST emitted token to the decoder and joint it with
//     `pooler_output[min(t, frames-1)]` (:243-246, where the CLAMP is upstream's: it
//     keeps the gather in range on the step that exhausts the encoder);
//   * pick the token by argmax over the VOCABULARY columns only. For TDT that is
//     what the published `suppress_tokens`
//     (= [vocab_size, vocab_size+len(durations)) ) achieves inside `generate()`;
//     for RNN-T the joint is only vocab_size wide, so the restriction is vacuous;
//   * RNN-T: advance one frame on a blank, or when `max_symbols_per_step`
//     consecutive non-blanks have been emitted at this frame (:148-157);
//   * TDT: advance by the predicted duration, forced to at least 1 on a blank
//     so the walk cannot stall (:286-293);
//   * stop once the pointer reaches `valid_frames` (:161, :113-122), on an
//     `eos_token_id`, or at the `max_symbols_per_step * frames` output-buffer
//     bound (:177-181).
ParakeetTransducerOutput ParakeetTransducerGreedyDecode(
    const std::vector<float>& encoder_projected, int64_t frames, int64_t valid_frames,
    const ParakeetForTransducerWeights& w, const ParakeetTransducerConfig& cfg,
    vt::Backend& backend);

// `ParakeetForRNNT` / `ParakeetForTDT` end to end for one clip: encoder
// (:944-948) -> `encoder_projector` (:949) -> greedy transducer decode.
// `valid_frames` is the extractor's valid prefix, in INPUT (mel) frames, exactly
// as `ParakeetForCTCForward` takes it.
ParakeetTransducerOutput ParakeetForTransducerForward(
    const std::vector<float>& input_features, int64_t num_frames, int64_t valid_frames,
    const ParakeetForTransducerWeights& w, const ParakeetEncoderConfig& enc_cfg,
    const ParakeetTransducerConfig& cfg, vt::Backend& backend);

// --- checkpoint loading ------------------------------------------------------

// Load an HF-format `ParakeetForRNNT` or `ParakeetForTDT` checkpoint from `dir`
// (config.json + model.safetensors or a shard index), reusing the encoder half
// of the P4 loader. `generation_config.json` is read when present for
// `decoder_start_token_id` / `eos_token_id`. Throws std::runtime_error naming
// the offending key on any missing or misshaped tensor.
ParakeetForTransducerWeights LoadParakeetTransducer(const std::string& dir,
                                                    ParakeetEncoderConfig* enc_cfg,
                                                    ParakeetTransducerConfig* cfg);

// Parse just the transducer half of config.json (+ generation_config.json).
ParakeetTransducerConfig LoadParakeetTransducerConfig(const std::string& dir);

// The `model_type` field of config.json: "parakeet_ctc", "parakeet_rnnt" or
// "parakeet_tdt" (configuration_parakeet.py:116, :164, :224). Lets a caller pick
// a head without guessing from the tensor names.
std::string LoadParakeetModelType(const std::string& dir);

}  // namespace vllm::multimodal
