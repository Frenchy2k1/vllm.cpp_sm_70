// Parakeet log-mel FEATURE EXTRACTION — spike row
// `MODEL-AUDIO-PARAKEET-ENCODER` (work item P4 of
// .agents/specs/parakeet-conformer-encoder.md), the front end that produces the
// `input_features` `ParakeetEncoder` consumes.
//
// Ported from BOTH sides of the upstream chain, which agree on the algorithm:
//   vLLM (the vLLM-NATIVE half of the Parakeet port — vLLM implements the
//   extractor even though it delegates the encoder to HF):
//     vllm/model_executor/models/parakeet.py
//       ParakeetExtractor:138, _get_window:149-152, _get_mel_filters:154-168,
//       _torch_extract_fbank_features:170-187, _apply_mel_filters:189-196,
//       _apply_preemphasis:198-214, _normalize_mel_features:216-236,
//       EPSILON:134, LOG_ZERO_GUARD_VALUE:135
//     vllm/transformers_utils/configs/parakeet.py ExtractorConfig:41 (defaults
//       :47-55, from_hf_config:57-72)
//   transformers 5.3.0:
//     models/parakeet/feature_extraction_parakeet.py ParakeetFeatureExtractor:36
//       (ctor :65-97, _torch_extract_fbank_features:99-125, __call__:127-282)
//     audio_utils.py mel_filter_bank:453, hertz_to_mel:263, mel_to_hertz:299,
//       _create_triangular_filter_bank:356
//
// **RECORDED DEVIATIONS.**
//  1. No torch, no torchaudio, no librosa: the STFT is a direct DFT of the 257
//     needed bins per frame rather than an FFT, so the result differs from
//     `torch.stft` only in float summation order. Same deviation, and the same
//     justification, as the Whisper path in
//     include/vllm/multimodal/audio_processor.h.
//  2. The mel filter bank is CONSTRUCTED from the config here, in double
//     precision, following transformers `audio_utils.mel_filter_bank`
//     (slaney/slaney, `norm="slaney"`) — which is exactly what vLLM's
//     `ParakeetExtractor._get_mel_filters` (:154-168) calls. HF's own
//     `ParakeetFeatureExtractor` instead calls `librosa.filters.mel`, and says
//     in a comment (:83-93) that the ONLY reason is that `mel_filter_bank` works
//     in float64 while librosa works in float32 — same formula, different
//     rounding. Both are gated, at their own tolerances, in
//     tests/vllm/multimodal/test_parakeet_audio_processor.cpp.
//  3. vLLM's 30-second CLIP SPLITTING (`_clip_sizes:253-259`,
//     `split_audio_into_clips:271-284`) belongs to the Nemotron-VL multimodal
//     path, not to the ASR model: it exists so a long audio maps to a fixed
//     token budget. This class extracts ONE clip; a caller that wants vLLM's
//     splitting slices the waveform first. Recorded rather than silently
//     dropped.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vllm::multimodal {

// `ExtractorConfig` (vllm/transformers_utils/configs/parakeet.py:41) and the
// `ParakeetFeatureExtractor.__init__` defaults (feature_extraction_parakeet.py
// :65-75). The two agree field for field.
struct ParakeetExtractorConfig {
  int feature_size = 80;     // == config.num_mel_bins
  int sampling_rate = 16000;
  int hop_length = 160;
  int n_fft = 512;
  int win_length = 400;
  float preemphasis = 0.97f;
  float padding_value = 0.0f;

  int num_freq_bins() const { return n_fft / 2 + 1; }
};

// `audio_utils.hertz_to_mel(freq, mel_scale="slaney")` (:285-296) and its
// inverse `mel_to_hertz` (:321-331). Exposed because they are the piece a
// reviewer is most likely to want to check independently.
double ParakeetHertzToMelSlaney(double hertz);
double ParakeetMelToHertzSlaney(double mels);

// `audio_utils.mel_filter_bank(num_frequency_bins=n_fft//2+1,
// num_mel_filters=feature_size, min_frequency=0.0, max_frequency=sr/2,
// sampling_rate=sr, norm="slaney", mel_scale="slaney")` (:453), i.e. exactly the
// call vLLM's `ParakeetExtractor._get_mel_filters` makes (parakeet.py:159-167).
//
// Returned TRANSPOSED relative to `mel_filter_bank`, as [num_mel_filters,
// num_freq_bins] row-major — the orientation both upstreams multiply in
// (`torch.from_numpy(filter_bank.T)`, parakeet.py:168; `mel_filters @
// magnitudes`, parakeet.py:194 / feature_extraction_parakeet.py:119).
std::vector<float> ParakeetMelFilterBank(const ParakeetExtractorConfig& cfg);

// The extractor output for ONE clip.
struct ParakeetAudioFeatures {
  // [num_frames, feature_size] row-major, the encoder's `input_features`
  // (parakeet.py:196 permutes the mel spectrogram to (frames, mels)).
  std::vector<float> input_features;
  int64_t num_frames = 0;
  // `features_lengths` (parakeet.py:220-223) — the prefix the attention mask
  // marks valid. Frames at or past it are ZERO in `input_features` (:236).
  int64_t valid_frames = 0;
};

class ParakeetAudioProcessor {
 public:
  explicit ParakeetAudioProcessor(ParakeetExtractorConfig cfg);
  // Construct with an EXTERNAL filter bank ([feature_size, num_freq_bins]),
  // so a checkpoint that ships one, or a test that wants librosa's float32
  // rounding, can bypass the built-in construction.
  ParakeetAudioProcessor(ParakeetExtractorConfig cfg, std::vector<float> mel_filters);

  const ParakeetExtractorConfig& config() const { return cfg_; }
  const std::vector<float>& mel_filters() const { return mel_filters_; }

  // preemphasis -> STFT -> power -> mel -> log -> per-bin normalisation, i.e.
  // `ParakeetExtractor.__call__` (parakeet.py:286-330) for a single item.
  // `sample_rate` must equal cfg.sampling_rate: upstream REFUSES a mismatch
  // rather than resampling (feature_extraction_parakeet.py:195-201).
  ParakeetAudioFeatures ProcessWaveform(const float* samples, int64_t num_samples,
                                        int sample_rate) const;

 private:
  ParakeetExtractorConfig cfg_;
  std::vector<float> mel_filters_;  // [feature_size, num_freq_bins]
};

}  // namespace vllm::multimodal
