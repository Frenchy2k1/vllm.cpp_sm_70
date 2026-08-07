// ARCH-ONE-SURFACE ROW 1 fold gate: the library transcription seam
// (vllm::multimodal::ParakeetTranscriber) must reproduce the EXACT ids and
// transcript the PRE-refactor examples/parakeet_transcribe binary produced.
//
// Three arms, all on the committed deterministic fixtures
// (tests/vllm/models/fixtures/parakeet_e2e, scripts/mm/
// parakeet_e2e_fixture_gen.py):
//   1. OLD-vs-NEW byte identity: the pre-refactor pipeline is replicated here
//      verbatim (main.cpp @ f98e1e48: ReadWav16BitMono :49-99, head dispatch
//      :210-238, LoadVocab :104-126 + DecodeIds :140-159) and must agree with
//      the seam id-for-id and byte-for-byte on text.
//   2. GOLDEN identity: both must equal the COMMITTED golden files captured by
//      running the actual pre-refactor binary (golden_ctc.txt /
//      golden_rnnt.txt) — so the in-test replica cannot drift into agreeing
//      with a subtly changed library.
//   3. Refuse-by-task: the registered Parakeet archs resolve from config.json
//      with the SupportsTranscription-only ModelInfo, and the text-generation
//      entry (LoadedEngine::FromModelDir) refuses them with the actionable
//      message instead of crashing (interfaces.py:1110-1118 mirror).
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vllm/model_executor/models/parakeet_transducer.h"
#include "vllm/multimodal/parakeet_audio_processor.h"
#include "vllm/multimodal/parakeet_transcription.h"
#include "vt/backend.h"

namespace {

std::string Fix() { return std::string(PARAKEET_E2E_FIXTURE_DIR); }

// ── the PRE-refactor example pipeline, replicated verbatim ──────────────────
// examples/parakeet_transcribe/main.cpp:49-99 @ f98e1e48.
bool RefReadWav16BitMono(const std::string& path, std::vector<float>* out,
                         int* sample_rate) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char riff[12];
  f.read(riff, 12);
  if (std::memcmp(riff, "RIFF", 4) != 0 ||
      std::memcmp(riff + 8, "WAVE", 4) != 0) {
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
      f.read(fmt.data(), static_cast<std::streamsize>(sz));
      uint16_t ch = 0, bps = 0;
      uint32_t sr = 0;
      std::memcpy(&ch, fmt.data() + 2, 2);
      std::memcpy(&sr, fmt.data() + 4, 4);
      std::memcpy(&bps, fmt.data() + 14, 2);
      channels = ch;
      bits = bps;
      *sample_rate = static_cast<int>(sr);
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (channels != 1 || bits != 16) return false;
      const size_t n = sz / 2;
      std::vector<int16_t> pcm(n);
      f.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(sz));
      out->resize(n);
      for (size_t i = 0; i < n; ++i) {
        (*out)[i] = static_cast<float>(pcm[i]) / 32768.0F;
      }
      return true;
    } else {
      f.seekg(sz, std::ios::cur);
    }
  }
  return false;
}

// main.cpp:104-126 @ f98e1e48.
std::map<int32_t, std::string> RefLoadVocab(const std::string& dir) {
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

// main.cpp:140-159 @ f98e1e48.
std::string RefDecodeIds(const std::vector<int32_t>& ids,
                         const std::map<int32_t, std::string>& vocab) {
  static const std::string kReplacement = "\xe2\x96\x81";
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

// The old pipeline end to end (main.cpp:176-264): WAV -> features -> the head
// config.json names -> ids -> text.
struct OldResult {
  std::vector<int32_t> ids;
  std::string text;
};

OldResult RunOldPipeline(const std::string& ckpt, const std::string& wav) {
  namespace mm = vllm::multimodal;
  OldResult r;
  std::vector<float> samples;
  int sample_rate = 0;
  REQUIRE(RefReadWav16BitMono(wav, &samples, &sample_rate));

  const std::string model_type = mm::LoadParakeetModelType(ckpt);
  const mm::ParakeetEncoderConfig probe = mm::LoadParakeetConfig(ckpt);
  mm::ParakeetExtractorConfig ecfg;
  ecfg.feature_size = static_cast<int>(probe.num_mel_bins);
  const mm::ParakeetAudioProcessor proc(ecfg);
  const mm::ParakeetAudioFeatures feats = proc.ProcessWaveform(
      samples.data(), static_cast<int64_t>(samples.size()), sample_rate);

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  if (model_type == "parakeet_rnnt" || model_type == "parakeet_tdt") {
    mm::ParakeetEncoderConfig enc_cfg;
    mm::ParakeetTransducerConfig cfg;
    const mm::ParakeetForTransducerWeights w =
        mm::LoadParakeetTransducer(ckpt, &enc_cfg, &cfg);
    r.ids = mm::ParakeetForTransducerForward(feats.input_features,
                                             feats.num_frames,
                                             feats.valid_frames, w, enc_cfg,
                                             cfg, cpu)
                .token_ids;
  } else {
    mm::ParakeetEncoderConfig cfg;
    const mm::ParakeetForCTCWeights w = mm::LoadParakeetForCTC(ckpt, &cfg);
    r.ids = mm::ParakeetForCTCForward(feats.input_features, feats.num_frames,
                                      feats.valid_frames, w, cfg, cpu)
                .token_ids;
  }
  r.text = RefDecodeIds(r.ids, RefLoadVocab(ckpt));
  return r;
}

// Parse a committed golden file: line 1 = space-joined ids, line 2 = text.
OldResult ReadGolden(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open golden: ", path);
  OldResult g;
  std::string ids_line;
  REQUIRE(static_cast<bool>(std::getline(f, ids_line)));
  std::istringstream is(ids_line);
  int32_t id = 0;
  while (is >> id) g.ids.push_back(id);
  REQUIRE(static_cast<bool>(std::getline(f, g.text)));
  return g;
}

}  // namespace

TEST_CASE("fold gate: seam == pre-refactor pipeline == committed golden") {
  const std::string wav = Fix() + "/audio.wav";
  for (const char* head : {"ctc", "rnnt"}) {
    CAPTURE(head);
    const std::string ckpt = Fix() + "/" + head;

    const OldResult old_r = RunOldPipeline(ckpt, wav);
    const OldResult golden = ReadGolden(Fix() + "/golden_" + head + ".txt");

    const auto t = vllm::multimodal::ParakeetTranscriber::FromDir(ckpt);
    CHECK(t.has_tokenizer());
    const vllm::multimodal::ParakeetTranscription got =
        t.TranscribeWavFile(wav);

    // Old vs new, byte-identical.
    CHECK(got.token_ids == old_r.ids);
    CHECK(got.has_text);
    CHECK(got.text == old_r.text);
    // Both vs the transcript the pre-refactor BINARY printed.
    CHECK(old_r.ids == golden.ids);
    CHECK(old_r.text == golden.text);
    CHECK(got.token_ids == golden.ids);
    CHECK(got.text == golden.text);
  }
}

TEST_CASE("fold gate: PCM entry equals WAV entry") {
  // The seam's raw-PCM entry (the C-ABI `pcm` arm) must agree with the WAV
  // path on the same samples.
  const std::string wav = Fix() + "/audio.wav";
  std::vector<float> samples;
  int rate = 0;
  REQUIRE(RefReadWav16BitMono(wav, &samples, &rate));
  const auto t =
      vllm::multimodal::ParakeetTranscriber::FromDir(Fix() + "/ctc");
  const auto via_wav = t.TranscribeWavFile(wav);
  const auto via_pcm =
      t.Transcribe(samples.data(), static_cast<int64_t>(samples.size()), rate);
  CHECK(via_pcm.token_ids == via_wav.token_ids);
  CHECK(via_pcm.text == via_wav.text);
}

TEST_CASE("registry: Parakeet archs resolve as transcription-only") {
  for (const char* arch :
       {"ParakeetForCTC", "ParakeetForRNNT", "ParakeetForTDT"}) {
    CAPTURE(arch);
    const std::vector<std::string> archs = {arch};
    const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(
        std::span<const std::string>(archs));
    CHECK(reg.info.supports_transcription);
    CHECK(reg.info.supports_transcription_only);
    CHECK_FALSE(reg.info.is_text_generation_model);
  }
}

TEST_CASE("refuse-by-task: the text engine refuses a Parakeet dir cleanly") {
  // LoadedEngine::FromModelDir must throw the actionable transcription-only
  // message (never a crash, never a deep loader error about missing text
  // tensors).
  const std::string ckpt = Fix() + "/ctc";
  try {
    vllm::entrypoints::EngineParams params;
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(ckpt, params);
    FAIL("FromModelDir was expected to refuse a transcription-only arch");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    MESSAGE("refusal message: ", msg);
    CHECK(msg.find("supports transcription only") != std::string::npos);
    CHECK(msg.find("/v1/audio/transcriptions") != std::string::npos);
  }
}
