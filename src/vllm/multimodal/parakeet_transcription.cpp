// Parakeet ASR transcription seam — see include/vllm/multimodal/
// parakeet_transcription.h for the fold contract and upstream mirror shape.
//
// This is the pipeline `examples/parakeet_transcribe/main.cpp` owned privately
// before the ARCH-ONE-SURFACE ROW 1 fold (main.cpp:176-264 @ f98e1e48), moved
// into the library verbatim in ORDER and SEMANTICS so the fold gate
// (tests/vllm/models/test_parakeet_transcription_fold.cpp) can hold it
// byte-identical to the pre-refactor transcript goldens:
//   WAV -> f32 mono            DecodeWavPcm16Mono (was: ReadWav16BitMono)
//   config.json model_type     LoadParakeetModelType (dispatch was main.cpp:210-238)
//   log-mel                    ParakeetAudioProcessor, feature_size = num_mel_bins
//   encoder + head             ParakeetForCTCForward / ParakeetForTransducerForward
//   ids -> text                vllm::tok::Tokenizer (was: LoadVocab + DecodeIds —
//                              the Metaspace split=true decoder now lives in the
//                              tokenizer, so the private copy is deleted)
#include "vllm/multimodal/parakeet_transcription.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vllm/model_executor/models/parakeet_transducer.h"
#include "vllm/multimodal/audio_processor.h"
#include "vllm/multimodal/parakeet_audio_processor.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vt/backend.h"

namespace vllm::multimodal {

struct ParakeetTranscriber::Impl {
  std::string model_type;
  ParakeetEncoderConfig enc_cfg;
  // Exactly one head is engaged, per config.json `model_type`.
  std::optional<ParakeetForCTCWeights> ctc;
  std::optional<ParakeetForTransducerWeights> transducer;
  ParakeetTransducerConfig transducer_cfg;
  std::optional<tok::Tokenizer> tokenizer;
};

ParakeetTranscriber::ParakeetTranscriber() = default;
ParakeetTranscriber::ParakeetTranscriber(ParakeetTranscriber&&) noexcept = default;
ParakeetTranscriber& ParakeetTranscriber::operator=(ParakeetTranscriber&&) noexcept =
    default;
ParakeetTranscriber::~ParakeetTranscriber() = default;

ParakeetTranscriber ParakeetTranscriber::FromDir(const std::string& dir) {
  ParakeetTranscriber t;
  t.impl_ = std::make_unique<Impl>();
  Impl& impl = *t.impl_;

  impl.model_type = LoadParakeetModelType(dir);
  if (impl.model_type == "parakeet_rnnt" || impl.model_type == "parakeet_tdt") {
    impl.transducer =
        LoadParakeetTransducer(dir, &impl.enc_cfg, &impl.transducer_cfg);
  } else if (impl.model_type == "parakeet_ctc" || impl.model_type.empty()) {
    // An empty model_type falls through to CTC exactly as the pre-refactor
    // example did (main.cpp:231-238: only a NON-empty, non-parakeet_ctc value
    // was refused).
    impl.ctc = LoadParakeetForCTC(dir, &impl.enc_cfg);
  } else {
    throw std::runtime_error("parakeet: unsupported model_type '" +
                             impl.model_type + "' (expected parakeet_ctc, "
                             "parakeet_rnnt or parakeet_tdt)");
  }

  // tokenizer.json is optional (ids-only checkpoints stay usable); a PRESENT
  // but unloadable one fails loudly rather than degrading to ids silently.
  const std::string tok_path = dir + "/tokenizer.json";
  if (std::filesystem::exists(tok_path)) {
    impl.tokenizer.emplace(tok::Tokenizer::FromHfJson(tok_path));
  }
  return t;
}

const std::string& ParakeetTranscriber::model_type() const {
  return impl_->model_type;
}

bool ParakeetTranscriber::has_tokenizer() const {
  return impl_->tokenizer.has_value();
}

ParakeetTranscription ParakeetTranscriber::Transcribe(const float* samples,
                                                      int64_t num_samples,
                                                      int sample_rate) const {
  const Impl& impl = *impl_;

  // The extractor is driven by the checkpoint's own num_mel_bins (80 on the
  // CTC/RNN-T checkpoints, 128 on parakeet-tdt-0.6b-v3), exactly as the
  // pre-refactor example configured it (main.cpp:196-202).
  ParakeetExtractorConfig ecfg;
  ecfg.feature_size = static_cast<int>(impl.enc_cfg.num_mel_bins);
  const ParakeetAudioProcessor proc(ecfg);
  const ParakeetAudioFeatures feats =
      proc.ProcessWaveform(samples, num_samples, sample_rate);

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);

  ParakeetTranscription out;
  if (impl.transducer.has_value()) {
    const ParakeetTransducerOutput head = ParakeetForTransducerForward(
        feats.input_features, feats.num_frames, feats.valid_frames,
        *impl.transducer, impl.enc_cfg, impl.transducer_cfg, cpu);
    out.token_ids = head.token_ids;
  } else {
    const ParakeetCTCOutput head =
        ParakeetForCTCForward(feats.input_features, feats.num_frames,
                              feats.valid_frames, *impl.ctc, impl.enc_cfg, cpu);
    out.token_ids = head.token_ids;
  }
  if (impl.tokenizer.has_value()) {
    out.text = impl.tokenizer->Decode(out.token_ids);
    out.has_text = true;
  }
  return out;
}

ParakeetTranscription ParakeetTranscriber::TranscribeWavBytes(
    const uint8_t* wav_bytes, size_t num_bytes) const {
  const DecodedAudio audio = DecodeWavPcm16Mono(wav_bytes, num_bytes);
  return Transcribe(audio.samples.data(),
                    static_cast<int64_t>(audio.samples.size()),
                    audio.sampling_rate);
}

ParakeetTranscription ParakeetTranscriber::TranscribeWavFile(
    const std::string& wav_path) const {
  std::ifstream f(wav_path, std::ios::binary);
  if (!f.good()) {
    throw std::runtime_error("parakeet: cannot open " + wav_path);
  }
  const std::string bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  return TranscribeWavBytes(reinterpret_cast<const uint8_t*>(bytes.data()),
                            bytes.size());
}

}  // namespace vllm::multimodal
