// P4 — Parakeet log-mel FRONT-END gate.
//
// Two independent references, both upstream:
//   * the mel filter bank vs a dump of transformers `audio_utils.mel_filter_bank`
//     (audio_utils.py:453, slaney/slaney) — the EXACT function vLLM's
//     `ParakeetExtractor._get_mel_filters` calls
//     (vllm/model_executor/models/parakeet.py:154-168) — and vs a dump of
//     librosa's float32 bank, which is what HF's own `ParakeetFeatureExtractor`
//     uses (feature_extraction_parakeet.py:94-97);
//   * the full `input_features` vs a dump of `ParakeetFeatureExtractor.__call__`
//     (feature_extraction_parakeet.py:127-282) on a fixed waveform.
// Fixture generator: scripts/mm/p4_parakeet_extractor_ref.py; the fixture is
// COMMITTED (228 KiB) so the gate runs in CI with no download and no torch.
//
// Plus in-test structural invariants that do not depend on any dump: the
// triangular filters' support and peak, and the post-normalisation zero
// mean / unit variance over the valid frames.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/multimodal/parakeet_audio_processor.h"

namespace {

using vllm::multimodal::ParakeetAudioProcessor;
using vllm::multimodal::ParakeetExtractorConfig;
using vllm::multimodal::ParakeetHertzToMelSlaney;
using vllm::multimodal::ParakeetMelFilterBank;
using vllm::multimodal::ParakeetMelToHertzSlaney;

std::string Fix() { return std::string(MM_PARAKEET_AUDIO_FIXTURE_DIR); }

template <typename T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<T> v(static_cast<size_t>(n) / sizeof(T));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
};

Err Compare(const std::vector<float>& got, const std::vector<float>& ref) {
  REQUIRE(got.size() == ref.size());
  double num = 0.0, den = 0.0, mx = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    if (std::abs(d) > mx) mx = std::abs(d);
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

}  // namespace

// The slaney mel scale is a piecewise linear/log map; `mel_to_hertz` must invert
// `hertz_to_mel` exactly at the breakpoint and everywhere else
// (audio_utils.py:285-296 / :321-331).
TEST_CASE("parakeet_mel_scale_roundtrip") {
  CHECK(ParakeetHertzToMelSlaney(0.0) == doctest::Approx(0.0));
  // The breakpoint: 1000 Hz is 15 mel in BOTH branches, so the map is continuous.
  CHECK(ParakeetHertzToMelSlaney(1000.0) == doctest::Approx(15.0));
  CHECK(ParakeetMelToHertzSlaney(15.0) == doctest::Approx(1000.0));
  CHECK(ParakeetHertzToMelSlaney(999.999) == doctest::Approx(14.9999850));
  for (double hz : {1.0, 100.0, 700.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0}) {
    CHECK(ParakeetMelToHertzSlaney(ParakeetHertzToMelSlaney(hz)) ==
          doctest::Approx(hz).epsilon(1e-12));
  }
}

TEST_CASE("parakeet_mel_filter_bank_matches_transformers_and_librosa") {
  ParakeetExtractorConfig cfg;  // the upstream defaults (80/16k/160/512/400)
  const std::vector<float> ours = ParakeetMelFilterBank(cfg);
  REQUIRE(ours.size() ==
          static_cast<size_t>(cfg.feature_size) * cfg.num_freq_bins());

  const std::vector<float> tf = ReadBin<float>(Fix() + "/mel_filters_transformers_f32.bin");
  const std::vector<float> lib = ReadBin<float>(Fix() + "/mel_filters_librosa_f32.bin");

  // Ours reconstructs `mel_filter_bank` in double, exactly as transformers does,
  // so this arm is a construction gate and must be near float32 resolution.
  const Err vs_tf = Compare(ours, tf);
  MESSAGE("mel bank vs transformers: rel_l2=", vs_tf.rel_l2, " max_abs=", vs_tf.max_abs);
  CHECK(vs_tf.max_abs < 1e-8);
  CHECK(vs_tf.rel_l2 < 1e-7);

  // The librosa arm is the DOCUMENTED float32-vs-float64 disagreement HF calls
  // out at feature_extraction_parakeet.py:83-93. It is bounded, not zero.
  const Err vs_lib = Compare(ours, lib);
  MESSAGE("mel bank vs librosa: rel_l2=", vs_lib.rel_l2, " max_abs=", vs_lib.max_abs);
  CHECK(vs_lib.max_abs < 1e-7);
}

// Structural invariants that need no dump: filter m is supported on
// (filter_freqs[m], filter_freqs[m+2]), peaks at filter_freqs[m+1], and is
// non-negative everywhere (audio_utils.py:371-375 + the slaney norm :532-535).
TEST_CASE("parakeet_mel_filter_bank_is_a_triangular_bank") {
  ParakeetExtractorConfig cfg;
  const std::vector<float> bank = ParakeetMelFilterBank(cfg);
  const int n_freq = cfg.num_freq_bins();

  const double mel_max = ParakeetHertzToMelSlaney(cfg.sampling_rate / 2.0);
  for (int m = 0; m < cfg.feature_size; ++m) {
    const double lo = ParakeetMelToHertzSlaney(mel_max * m / (cfg.feature_size + 1));
    const double hi = ParakeetMelToHertzSlaney(mel_max * (m + 2) / (cfg.feature_size + 1));
    const double bin_hz = static_cast<double>(cfg.sampling_rate / 2) / (n_freq - 1);
    double peak = 0.0;
    for (int k = 0; k < n_freq; ++k) {
      const float v = bank[static_cast<size_t>(m) * n_freq + k];
      CHECK(v >= 0.0f);
      const double hz = bin_hz * k;
      // Strictly outside the triangle's base the weight must be exactly zero.
      if (hz < lo - 1e-9 || hz > hi + 1e-9) {
        CHECK(v == 0.0f);
      }
      if (v > peak) peak = v;
    }
    // Every band must catch at least one FFT bin at this resolution.
    CHECK(peak > 0.0);
  }
}

TEST_CASE("parakeet_log_mel_matches_hf_feature_extractor") {
  const std::vector<float> wav = ReadBin<float>(Fix() + "/waveform_f32.bin");
  const std::vector<float> ref = ReadBin<float>(Fix() + "/input_features_f32.bin");
  const std::vector<int32_t> mask = ReadBin<int32_t>(Fix() + "/attention_mask_i32.bin");
  REQUIRE(!wav.empty());

  ParakeetExtractorConfig cfg;
  const int64_t frames = static_cast<int64_t>(mask.size());
  int64_t valid = 0;
  for (int32_t v : mask) valid += v;

  // Arm 1 — our own constructed bank (the vLLM `_get_mel_filters` path).
  ParakeetAudioProcessor proc(cfg);
  const auto got = proc.ProcessWaveform(wav.data(), static_cast<int64_t>(wav.size()),
                                        cfg.sampling_rate);
  CHECK(got.num_frames == frames);
  CHECK(got.valid_frames == valid);
  REQUIRE(got.input_features.size() == ref.size());
  const Err e1 = Compare(got.input_features, ref);
  MESSAGE("log-mel vs HF FE (own bank): rel_l2=", e1.rel_l2, " max_abs=", e1.max_abs);
  // Deviation 1 (direct DFT instead of torch's FFT) plus deviation 2 (float64
  // bank instead of librosa's float32) — bounded, and stated in the header.
  CHECK(e1.rel_l2 < 2e-5);

  // Arm 2 — HF's OWN librosa bank injected, which removes deviation 2 and
  // isolates the DFT-vs-FFT summation order. It must be strictly tighter, and
  // that ordering is the evidence the residual really is the bank.
  ParakeetAudioProcessor proc_lib(cfg,
                                  ReadBin<float>(Fix() + "/mel_filters_librosa_f32.bin"));
  const auto got_lib = proc_lib.ProcessWaveform(
      wav.data(), static_cast<int64_t>(wav.size()), cfg.sampling_rate);
  const Err e2 = Compare(got_lib.input_features, ref);
  MESSAGE("log-mel vs HF FE (librosa bank): rel_l2=", e2.rel_l2,
          " max_abs=", e2.max_abs);
  CHECK(e2.rel_l2 < 2e-5);
  CHECK(e2.rel_l2 <= e1.rel_l2);

  // Frames past the valid prefix are zeroed by the normalisation mask
  // (parakeet.py:236) — an exact, dump-independent property.
  for (int64_t f = valid; f < frames; ++f) {
    for (int m = 0; m < cfg.feature_size; ++m) {
      CHECK(got.input_features[static_cast<size_t>(f) * cfg.feature_size + m] == 0.0f);
    }
  }
}

// The normalisation is per MEL BIN over the valid frames (parakeet.py:216-236):
// mean 0, and a variance of 1 up to the `std/(std+EPSILON)` shrink the upstream
// epsilon placement causes. Both are checked against an in-test reference
// computed straight from the definition, so this arm needs no oracle.
TEST_CASE("parakeet_log_mel_normalisation_is_per_bin") {
  const std::vector<float> wav = ReadBin<float>(Fix() + "/waveform_f32.bin");
  ParakeetExtractorConfig cfg;
  ParakeetAudioProcessor proc(cfg);
  const auto got = proc.ProcessWaveform(wav.data(), static_cast<int64_t>(wav.size()),
                                        cfg.sampling_rate);
  const int64_t valid = got.valid_frames;
  REQUIRE(valid > 1);

  for (int m = 0; m < cfg.feature_size; ++m) {
    double sum = 0.0;
    for (int64_t f = 0; f < valid; ++f) {
      sum += got.input_features[static_cast<size_t>(f) * cfg.feature_size + m];
    }
    const double mean = sum / static_cast<double>(valid);
    double var = 0.0;
    for (int64_t f = 0; f < valid; ++f) {
      const double d =
          got.input_features[static_cast<size_t>(f) * cfg.feature_size + m] - mean;
      var += d * d;
    }
    var /= static_cast<double>(valid - 1);
    CHECK(std::abs(mean) < 1e-4);
    // (x-mu)/(sigma+eps) has unbiased variance (sigma/(sigma+eps))^2 <= 1.
    CHECK(var <= 1.0 + 1e-6);
    CHECK(var > 0.99);
  }
}

// A refusal, not a silent resample — feature_extraction_parakeet.py:195-201.
TEST_CASE("parakeet_extractor_refuses_a_sample_rate_mismatch") {
  ParakeetExtractorConfig cfg;
  ParakeetAudioProcessor proc(cfg);
  const std::vector<float> wav(1600, 0.1f);
  CHECK_THROWS(proc.ProcessWaveform(wav.data(), static_cast<int64_t>(wav.size()), 8000));
}
