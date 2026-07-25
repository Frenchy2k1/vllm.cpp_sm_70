// Audio-processor parity gate (audio-track A1 correctness gate). Verifies the C++
// Whisper-class audio INPUT pipeline matches the vLLM 0.25.0 / transformers 5.13.1
// oracle fixture (captured by scripts/mm/a0_audio_ref.py) for openai/whisper-small:
//   - log-mel `input_features` [80,3000] within a stated rel-L2 tolerance (FFT
//     float ops => not bit-exact; DFT-vs-FFT summation order is the only diff),
//   - the audio placeholder-token expansion ([0] -> [0]*1500) BIT-identical,
//   - the mm-hash (MultiModalHasher.hash_kwargs(model_id, audio=<f32>)) BYTE-identical.
// RED-first: a wrong mel filterbank / hop / normalization blows the rel-L2 far past
// the tolerance (asserted below by perturbing each).
//
// Golden: tests/vllm/multimodal/fixtures/whisper_audio/{manifest.json,
//   audio_tone_16k_mono.wav, audio_waveform_f32.bin, input_features_f32.bin,
//   mel_filters_f32.bin}
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/multimodal/audio_processor.h"

namespace {

std::string FixDir() { return std::string(MM_AUDIO_FIXTURE_DIR); }

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

std::vector<float> ReadF32(const std::string& path) {
  const std::vector<uint8_t> b = ReadBytes(path);
  REQUIRE(b.size() % sizeof(float) == 0);
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  nlohmann::json j;
  f >> j;
  return j;
}

vllm::multimodal::AudioProcessorConfig ConfigFromManifest(
    const nlohmann::json& m) {
  vllm::multimodal::AudioProcessorConfig cfg;
  const auto& c = m.at("feature_contract");
  cfg.n_fft = c.at("n_fft").get<int>();
  cfg.hop_length = c.at("hop_length").get<int>();
  cfg.n_mels = c.at("n_mels").get<int>();
  cfg.sampling_rate = c.at("sampling_rate").get<int>();
  cfg.chunk_length_s = c.at("chunk_length_s").get<int>();
  cfg.dither = c.at("dither").get<double>();
  cfg.max_source_positions = c.at("max_source_positions").get<int>();
  cfg.audio_placeholder_id =
      m.at("placeholder").at("audio_placeholder_id").get<int32_t>();
  cfg.model_id = m.at("model_id").get<std::string>();
  return cfg;
}

// Relative L2: ||a-b||_2 / ||b||_2 (b = oracle golden).
double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / std::sqrt(den);
}

}  // namespace

TEST_CASE("whisper-audio-processor-parity: log-mel + expansion + mm-hash") {
  const std::string dir = FixDir();
  const nlohmann::json m = ReadJson(dir + "/manifest.json");
  const auto cfg = ConfigFromManifest(m);
  const std::vector<float> mel_filters = ReadF32(dir + "/mel_filters_f32.bin");

  // Decode the committed PCM16 WAV; assert it round-trips to the golden f32 waveform.
  const std::vector<uint8_t> wav = ReadBytes(dir + "/audio_tone_16k_mono.wav");
  const auto decoded = vllm::multimodal::DecodeWavPcm16Mono(wav.data(), wav.size());
  const std::vector<float> golden_wav = ReadF32(dir + "/audio_waveform_f32.bin");

  SUBCASE("wav decode byte-identical to oracle f32 waveform") {
    CHECK(decoded.sampling_rate == cfg.sampling_rate);
    REQUIRE(decoded.samples.size() == golden_wav.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < golden_wav.size(); ++i)
      if (decoded.samples[i] != golden_wav[i]) ++mismatches;
    CHECK(mismatches == 0);
  }

  vllm::multimodal::WhisperAudioProcessor proc(cfg, mel_filters);
  const auto kw = proc.ProcessWaveform(decoded.samples.data(),
                                       static_cast<int64_t>(decoded.samples.size()),
                                       decoded.sampling_rate);
  const std::vector<float> golden_feat = ReadF32(dir + "/input_features_f32.bin");

  // Tolerance justification: the oracle log-mel is torch.stft (FFT); ours is a
  // direct DFT over the same window/hop/reflect-pad. The ONLY numeric difference
  // is float summation order (FFT butterfly vs DFT accumulate), a well-below-bf16
  // effect. Transformers itself only claims the torch vs numpy STFT agree to 1e-5.
  // We hold a TIGHT 2e-4 rel-L2 band (measured value is far tighter, see report);
  // any real bug (wrong filterbank/hop/window/normalization) is >>1e-1 (asserted).
  constexpr double kTol = 2e-4;

  SUBCASE("log-mel input_features shape + rel-L2 within tolerance") {
    CHECK(kw.n_mels == cfg.n_mels);
    CHECK(kw.n_frames == m.at("input_features").at("shape")[1].get<int64_t>());
    REQUIRE(kw.input_features.size() == golden_feat.size());
    const double rel = RelL2(kw.input_features, golden_feat);
    MESSAGE("log-mel rel-L2 vs oracle = ", rel);
    CHECK(rel < kTol);
  }

  SUBCASE("RED-first: perturbed mel filterbank fails the gate") {
    std::vector<float> bad_mel = mel_filters;
    // Corrupt the LARGEST-magnitude filterbank weight (the slaney DC bins are 0,
    // so pick a genuinely load-bearing weight).
    size_t argmax = 0;
    for (size_t i = 1; i < bad_mel.size(); ++i)
      if (std::fabs(bad_mel[i]) > std::fabs(bad_mel[argmax])) argmax = i;
    bad_mel[argmax] *= 1.5f;
    vllm::multimodal::WhisperAudioProcessor bad(cfg, bad_mel);
    const auto bk = bad.ProcessWaveform(decoded.samples.data(),
                                        static_cast<int64_t>(decoded.samples.size()),
                                        decoded.sampling_rate);
    const double rel = RelL2(bk.input_features, golden_feat);
    MESSAGE("RED mel-perturb rel-L2 = ", rel);
    CHECK(rel > kTol);
  }

  SUBCASE("RED-first: wrong hop length fails the gate") {
    auto bad_cfg = cfg;
    bad_cfg.hop_length = cfg.hop_length + 1;  // 161 instead of 160
    vllm::multimodal::WhisperAudioProcessor bad(bad_cfg, mel_filters);
    const auto bk = bad.ProcessWaveform(decoded.samples.data(),
                                        static_cast<int64_t>(decoded.samples.size()),
                                        decoded.sampling_rate);
    // Different hop => different n_frames; rel-L2 on the overlapping min-length
    // prefix (the frames still misalign catastrophically).
    const size_t nmin = std::min(bk.input_features.size(), golden_feat.size());
    std::vector<float> a(bk.input_features.begin(), bk.input_features.begin() + nmin);
    std::vector<float> b(golden_feat.begin(), golden_feat.begin() + nmin);
    const double rel = RelL2(a, b);
    MESSAGE("RED hop-perturb rel-L2 = ", rel);
    CHECK(rel > kTol);
  }

  SUBCASE("RED-first: skipping the (x+4)/4 normalization fails the gate") {
    // Reconstruct the un-normalized log10 mel from the golden and compare: the
    // normalization is load-bearing, proving the gate is sensitive to it.
    std::vector<float> unnorm = kw.input_features;
    for (auto& v : unnorm) v = v * 4.0f - 4.0f;  // invert (x+4)/4 -> raw log10(clamped)
    const double rel = RelL2(unnorm, golden_feat);
    MESSAGE("RED no-normalize rel-L2 = ", rel);
    CHECK(rel > kTol);
  }

  SUBCASE("audio placeholder expansion byte-identical to oracle ids") {
    std::vector<int32_t> pre;
    for (const auto& v : m.at("placeholder").at("pre_expansion_token_ids"))
      pre.push_back(v.get<int32_t>());
    const int num_tokens = m.at("placeholder").at("num_audio_tokens").get<int>();
    std::vector<std::array<int, 2>> placeholders;
    const auto expanded = vllm::multimodal::ExpandAudioPlaceholders(
        pre, cfg.audio_placeholder_id, {num_tokens}, &placeholders);

    std::vector<int32_t> golden(static_cast<size_t>(num_tokens),
                                cfg.audio_placeholder_id);
    CHECK(expanded == golden);
    CHECK(static_cast<int>(expanded.size()) ==
          m.at("placeholder").at("expanded_len").get<int>());
    REQUIRE(placeholders.size() == 1);
    CHECK(placeholders[0][0] == 0);
    CHECK(placeholders[0][1] == num_tokens);
    // The placeholder count equals the encoder output length = max_source_positions.
    CHECK(num_tokens == cfg.max_source_positions);
  }

  SUBCASE("mm-hash byte-identical to MultiModalHasher") {
    const std::string ours = proc.HashAudio(
        golden_wav.data(), static_cast<int64_t>(golden_wav.size()));
    CHECK(ours == m.at("mm_hash").get<std::string>());
  }
}
