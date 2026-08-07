// Parakeet ASR end to end on real audio, real pretrained weights, CPU only,
// for the WHOLE published model family, not just CTC.
//
// This exists to answer one question the unit gates deliberately do NOT answer:
// does the port actually TRANSCRIBE? The oracles in
// tests/vllm/models/test_parakeet_ctc_engine.cpp and
// tests/vllm/models/test_parakeet_transducer.cpp gate our forward against real
// HF modules, but with RANDOM weights, so they prove the math and nothing about
// speech. Neither produces a transcript, on purpose.
//
// So: 16 kHz mono WAV -> log-mel (ParakeetAudioProcessor) -> encoder -> the head
// config.json names -> token ids -> text.
//
//   parakeet-transcribe <ckpt-dir> <audio.wav>
//
// `<ckpt-dir>` is any HF-format Parakeet checkpoint; the head is taken from
// `model_type` in config.json, so one binary covers all three:
//   parakeet_ctc  -> ParakeetForCTC   (nvidia/parakeet-ctc-0.6b, -ctc-1.1b)
//   parakeet_rnnt -> ParakeetForRNNT  (nvidia/parakeet-rnnt-0.6b, -rnnt-1.1b)
//   parakeet_tdt  -> ParakeetForTDT   (nvidia/parakeet-tdt-0.6b-v3)
//
// Token ids always go to stdout, diffable against HF's own `generate()`. When
// the checkpoint ships a `tokenizer.json` the decoded TEXT is printed too; see
// the decoder note on `DecodeIds` for exactly which upstream rule that mirrors
// and why it does not reuse vllm::Tokenizer.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vllm/model_executor/models/parakeet_transducer.h"
#include "vllm/multimodal/parakeet_audio_processor.h"
#include "vt/backend.h"

namespace {

// Minimal RIFF/WAVE reader: 16-bit PCM mono, which is what the LibriSpeech and
// NeMo sample clips are. Anything else is refused loudly rather than resampled,
// mirroring the extractor's own refusal to resample
// (feature_extraction_parakeet.py:195-201).
bool ReadWav16BitMono(const std::string& path, std::vector<float>* out, int* sample_rate) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  char riff[12];
  f.read(riff, 12);
  if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0) {
    std::fprintf(stderr, "%s is not a RIFF/WAVE file\n", path.c_str());
    return false;
  }
  int channels = 0;
  int bits = 0;
  while (f) {
    char id[4];
    uint32_t sz = 0;
    f.read(id, 4);
    f.read(reinterpret_cast<char*>(&sz), 4);
    if (!f) break;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      std::vector<char> fmt(sz);
      f.read(fmt.data(), sz);
      uint16_t ch = 0, bps = 0;
      uint32_t sr = 0;
      std::memcpy(&ch, fmt.data() + 2, 2);
      std::memcpy(&sr, fmt.data() + 4, 4);
      std::memcpy(&bps, fmt.data() + 14, 2);
      channels = ch;
      bits = bps;
      *sample_rate = static_cast<int>(sr);
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (channels != 1 || bits != 16) {
        std::fprintf(stderr, "need 16-bit mono, got %d-bit %d-channel\n", bits, channels);
        return false;
      }
      const size_t n = sz / 2;
      std::vector<int16_t> pcm(n);
      f.read(reinterpret_cast<char*>(pcm.data()), sz);
      out->resize(n);
      for (size_t i = 0; i < n; ++i) {
        (*out)[i] = static_cast<float>(pcm[i]) / 32768.0F;
      }
      return true;
    } else {
      f.seekg(sz, std::ios::cur);
    }
  }
  std::fprintf(stderr, "%s has no data chunk\n", path.c_str());
  return false;
}

// id -> piece from a `tokenizer.json`, taking both the BPE vocab and the
// `added_tokens` (which is where `<blank>` / `<pad>` / the multilingual prompt
// tokens live). Empty when the checkpoint ships no tokenizer.
std::map<int32_t, std::string> LoadVocab(const std::string& dir) {
  std::map<int32_t, std::string> vocab;
  std::ifstream f(dir + "/tokenizer.json", std::ios::binary);
  if (!f.good()) return vocab;
  nlohmann::json doc;
  f >> doc;
  const auto model = doc.find("model");
  if (model != doc.end()) {
    const auto v = model->find("vocab");
    if (v != model->end() && v->is_object()) {
      for (auto it = v->begin(); it != v->end(); ++it) {
        vocab[it.value().get<int32_t>()] = it.key();
      }
    }
  }
  const auto added = doc.find("added_tokens");
  if (added != doc.end() && added->is_array()) {
    for (const auto& t : *added) {
      vocab[t.at("id").get<int32_t>()] = t.at("content").get<std::string>();
    }
  }
  return vocab;
}

// The `Metaspace` DECODER rule, which is what every published Parakeet
// tokenizer.json declares (`{"type": "Metaspace", "replacement": "▁",
// "prepend_scheme": "always", "split": true}`): inside the FIRST piece the
// replacement character is DROPPED, inside every later piece it becomes a space,
// and the pieces are concatenated. Mirrors HF tokenizers
// `decoders::metaspace::Metaspace::decode_chain`.
//
// This does not go through `vllm::Tokenizer` on purpose: `Tokenizer::FromHfJson`
// refuses a Metaspace pre_tokenizer with `split: true` ("no golden in scope",
// src/vllm/tokenizer/tokenizer.cpp), which is an ENCODE-side restriction. This
// example only ever decodes, so it implements the decoder rule directly rather
// than weakening that guard to get at it.
std::string DecodeIds(const std::vector<int32_t>& ids,
                      const std::map<int32_t, std::string>& vocab) {
  static const std::string kReplacement = "\xe2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK
  std::string text;
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto it = vocab.find(ids[i]);
    if (it == vocab.end()) continue;
    const std::string& piece = it->second;
    for (size_t p = 0; p < piece.size();) {
      if (piece.compare(p, kReplacement.size(), kReplacement) == 0) {
        if (i != 0) text.push_back(' ');
        p += kReplacement.size();
      } else {
        text.push_back(piece[p]);
        ++p;
      }
    }
  }
  return text;
}

void PrintIdsAndText(const std::vector<int32_t>& ids,
                     const std::map<int32_t, std::string>& vocab) {
  for (size_t i = 0; i < ids.size(); ++i) {
    std::printf("%s%d", i ? " " : "", ids[i]);
  }
  std::printf("\n");
  if (!vocab.empty()) {
    std::printf("%s\n", DecodeIds(ids, vocab).c_str());
  } else {
    std::fprintf(stderr, "no tokenizer.json in the checkpoint: ids only\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <hf-parakeet-dir> <audio.wav>\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  const std::string wav = argv[2];

  std::vector<float> samples;
  int sample_rate = 0;
  if (!ReadWav16BitMono(wav, &samples, &sample_rate)) return 1;
  std::fprintf(stderr, "audio: %zu samples @ %d Hz (%.2f s)\n", samples.size(), sample_rate,
               static_cast<double>(samples.size()) / sample_rate);

  const std::string model_type = vllm::multimodal::LoadParakeetModelType(ckpt);
  const vllm::multimodal::ParakeetEncoderConfig probe =
      vllm::multimodal::LoadParakeetConfig(ckpt);
  std::fprintf(stderr, "model_type: %s (%lld mel bins)\n", model_type.c_str(),
               static_cast<long long>(probe.num_mel_bins));

  // The extractor is driven by the checkpoint's own `num_mel_bins`: 80 on the
  // CTC and RNN-T checkpoints, 128 on parakeet-tdt-0.6b-v3.
  vllm::multimodal::ParakeetExtractorConfig ecfg;
  ecfg.feature_size = static_cast<int>(probe.num_mel_bins);
  const vllm::multimodal::ParakeetAudioProcessor proc(ecfg);
  const vllm::multimodal::ParakeetAudioFeatures feats =
      proc.ProcessWaveform(samples.data(), static_cast<int64_t>(samples.size()), sample_rate);
  std::fprintf(stderr, "features: %lld frames (%lld valid)\n",
               static_cast<long long>(feats.num_frames),
               static_cast<long long>(feats.valid_frames));

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const std::map<int32_t, std::string> vocab = LoadVocab(ckpt);

  if (model_type == "parakeet_rnnt" || model_type == "parakeet_tdt") {
    vllm::multimodal::ParakeetEncoderConfig enc_cfg;
    vllm::multimodal::ParakeetTransducerConfig cfg;
    const vllm::multimodal::ParakeetForTransducerWeights w =
        vllm::multimodal::LoadParakeetTransducer(ckpt, &enc_cfg, &cfg);
    std::fprintf(stderr, "weights: %zu encoder layers, %lld decoder LSTM layers, vocab %lld%s\n",
                 w.encoder.layers.size(), static_cast<long long>(cfg.num_decoder_layers),
                 static_cast<long long>(cfg.vocab_size),
                 cfg.is_tdt() ? " (+ TDT duration head)" : "");

    const vllm::multimodal::ParakeetTransducerOutput out =
        vllm::multimodal::ParakeetForTransducerForward(feats.input_features, feats.num_frames,
                                                       feats.valid_frames, w, enc_cfg, cfg, cpu);
    std::fprintf(stderr, "encoder: %lld frames (%lld valid); decode: %zu steps, %zu tokens\n",
                 static_cast<long long>(out.encoder_frames),
                 static_cast<long long>(out.valid_encoder_frames),
                 out.sequences.size() - 1, out.token_ids.size());
    PrintIdsAndText(out.token_ids, vocab);
    return 0;
  }

  if (model_type != "parakeet_ctc" && !model_type.empty()) {
    std::fprintf(stderr, "unsupported model_type '%s'\n", model_type.c_str());
    return 1;
  }

  vllm::multimodal::ParakeetEncoderConfig cfg;
  const vllm::multimodal::ParakeetForCTCWeights w =
      vllm::multimodal::LoadParakeetForCTC(ckpt, &cfg);
  std::fprintf(stderr, "weights: %zu encoder layers, vocab %lld\n", w.encoder.layers.size(),
               static_cast<long long>(cfg.vocab_size));

  // Optional third argument: repeat the FORWARD only, so the checkpoint load and
  // the mel front end stay OUT of the timed region. Without it the wall clock is
  // dominated by reading a multi-GiB safetensors, and a 3 s clip can time SLOWER
  // than a 10 s one: a loader measurement wearing a model's clothes.
  const int reps = (argc > 3) ? std::atoi(argv[3]) : 1;
  vllm::multimodal::ParakeetCTCOutput out;
  double best_ms = 0.0;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    out = vllm::multimodal::ParakeetForCTCForward(
        feats.input_features, feats.num_frames, feats.valid_frames, w, cfg, cpu);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r == 0 || ms < best_ms) best_ms = ms;
    std::fprintf(stderr, "forward rep %d: %.1f ms\n", r, ms);
  }
  std::fprintf(stderr, "forward best: %.1f ms\n", best_ms);
  std::fprintf(stderr, "encoder: %lld output frames (%lld valid)\n",
               static_cast<long long>(out.num_output_frames),
               static_cast<long long>(out.valid_output_frames));
  PrintIdsAndText(out.token_ids, vocab);
  return 0;
}
