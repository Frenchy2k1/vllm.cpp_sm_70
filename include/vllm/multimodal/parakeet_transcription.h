// Parakeet ASR transcription seam — the ONE library entry point that composes
// parakeet_audio_processor -> parakeet_encoder -> the head config.json names
// (CTC / RNN-T / TDT) -> vllm::Tokenizer decode, and that every consumer
// (C ABI `vllm_transcribe`, the OpenAI server's /v1/audio/transcriptions, the
// `parakeet-transcribe` example) drives.
//
// ARCH-ONE-SURFACE ROW 1 (fold #4, audio transcription). This file ABSORBS the
// pipeline `examples/parakeet_transcribe/main.cpp` used to own privately
// (pre-refactor main.cpp:176-264 @ f98e1e48: WAV read, model_type dispatch,
// head forward, id->text). Per the ONE SURFACE directive the example keeps
// argv parsing and printing ONLY; the capability lives here, reachable by any
// embedder. The id->text step routes through vllm::Tokenizer (which now
// implements the Metaspace split=true decoder the example carried a private
// DecodeIds for), and the WAV ingest through the library's DecodeWavPcm16Mono.
//
// Upstream mirror shape: vLLM serves ASR through SupportsTranscription models
// behind vllm/entrypoints/speech_to_text/ (transcription/serving.py:29
// OpenAIServingTranscription -> base/serving.py `_create_speech_to_text`);
// this seam is the C++ engine-side equivalent of that handler's model call for
// the Parakeet family, whose forwards live in parakeet_encoder.h /
// parakeet_transducer.h (HF-mirrored; vLLM delegates the encoder to HF,
// parakeet.py:37,62). Greedy only — upstream's whole supported surface
// (`_supported_generation_modes`, modeling_parakeet.py main:925).
//
// CPU-only correctness-grade forward, matching the P4/P6 rows it composes; no
// speed claim is made here (spike § Gates: the family's speed gate is GB10).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vllm::multimodal {

// One transcription result. `token_ids` is exactly what the head decode
// emitted (CTC: greedy ids after the collapse; transducer: the emitted ids
// with blanks and the start token dropped) — the ids the pre-refactor example
// printed. `text` is the vllm::Tokenizer decode of those ids; `has_text` is
// false when the checkpoint ships no tokenizer.json (ids only, as before).
struct ParakeetTranscription {
  std::vector<int32_t> token_ids;
  std::string text;
  bool has_text = false;
};

// A loaded Parakeet checkpoint (any head) ready to transcribe. Construction
// loads config.json, dispatches on `model_type` ("parakeet_ctc" ->
// ParakeetForCTC; "parakeet_rnnt"/"parakeet_tdt" -> the transducer heads;
// anything else throws), loads the weights, and loads tokenizer.json through
// vllm::Tokenizer when the checkpoint ships one. Throws std::runtime_error
// naming the problem on any mismatch — never a silently wrong transcript.
class ParakeetTranscriber {
 public:
  // `dir` is an HF-format Parakeet checkpoint directory (config.json +
  // model.safetensors [+ index] [+ tokenizer.json] [+ generation_config.json]).
  static ParakeetTranscriber FromDir(const std::string& dir);

  ParakeetTranscriber(ParakeetTranscriber&&) noexcept;
  ParakeetTranscriber& operator=(ParakeetTranscriber&&) noexcept;
  ~ParakeetTranscriber();

  // "parakeet_ctc", "parakeet_rnnt" or "parakeet_tdt".
  const std::string& model_type() const;
  // True when tokenizer.json was present and loaded (text will be produced).
  bool has_tokenizer() const;

  // Transcribe a mono float32 waveform. `sample_rate` must match the
  // extractor's rate (16 kHz): the extractor refuses to resample, mirroring
  // feature_extraction_parakeet.py:195-201.
  ParakeetTranscription Transcribe(const float* samples, int64_t num_samples,
                                   int sample_rate) const;

  // Transcribe a canonical 16-bit PCM mono RIFF/WAVE byte buffer
  // (DecodeWavPcm16Mono — the library WAV ingest the example used to
  // duplicate as ReadWav16BitMono).
  ParakeetTranscription TranscribeWavBytes(const uint8_t* wav_bytes,
                                           size_t num_bytes) const;

  // Read `wav_path` and TranscribeWavBytes it.
  ParakeetTranscription TranscribeWavFile(const std::string& wav_path) const;

 private:
  ParakeetTranscriber();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::multimodal
