// Parakeet log-mel feature extraction — spike row
// `MODEL-AUDIO-PARAKEET-ENCODER` (P4). Full provenance, and the three recorded
// deviations, are in include/vllm/multimodal/parakeet_audio_processor.h.
//
// Ported from vllm/model_executor/models/parakeet.py `ParakeetExtractor`:138
// (the vLLM-NATIVE half of the Parakeet port) and transformers 5.3.0
// models/parakeet/feature_extraction_parakeet.py `ParakeetFeatureExtractor`:36,
// plus transformers audio_utils.py mel_filter_bank:453 / hertz_to_mel:263 /
// mel_to_hertz:299 / _create_triangular_filter_bank:356.
#include "vllm/multimodal/parakeet_audio_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vllm::multimodal {
namespace {

constexpr double kPi = 3.14159265358979323846;

// parakeet.py:134 / feature_extraction_parakeet.py:28.
constexpr double kEpsilon = 1e-5;
// parakeet.py:135 / feature_extraction_parakeet.py:29 — 2**-24, the log guard.
constexpr double kLogZeroGuard = 5.9604644775390625e-08;

// np.linspace(start, stop, num): step = (stop-start)/(num-1), and the LAST
// element is set to `stop` exactly.
std::vector<double> Linspace(double start, double stop, int num) {
  std::vector<double> out(static_cast<size_t>(num));
  if (num == 1) {
    out[0] = start;
    return out;
  }
  const double step = (stop - start) / static_cast<double>(num - 1);
  for (int i = 0; i < num; ++i) out[static_cast<size_t>(i)] = start + step * i;
  out[static_cast<size_t>(num - 1)] = stop;
  return out;
}

}  // namespace

double ParakeetHertzToMelSlaney(double hertz) {
  // audio_utils.py:285-296.
  constexpr double kMinLogHertz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double logstep = 27.0 / std::log(6.4);
  if (hertz >= kMinLogHertz) {
    return kMinLogMel + std::log(hertz / kMinLogHertz) * logstep;
  }
  return 3.0 * hertz / 200.0;
}

double ParakeetMelToHertzSlaney(double mels) {
  // audio_utils.py:321-331.
  constexpr double kMinLogHertz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double logstep = std::log(6.4) / 27.0;
  if (mels >= kMinLogMel) {
    return kMinLogHertz * std::exp(logstep * (mels - kMinLogMel));
  }
  return 200.0 * mels / 3.0;
}

std::vector<float> ParakeetMelFilterBank(const ParakeetExtractorConfig& cfg) {
  const int n_freq = cfg.num_freq_bins();
  const int n_mels = cfg.feature_size;
  if (n_freq < 2) throw std::runtime_error("parakeet mel: num_frequency_bins < 2");

  // audio_utils.py:516-519 — the filter centres, equally spaced in mel space.
  const double mel_min = ParakeetHertzToMelSlaney(0.0);
  const double mel_max =
      ParakeetHertzToMelSlaney(static_cast<double>(cfg.sampling_rate) / 2.0);
  const std::vector<double> mel_freqs = Linspace(mel_min, mel_max, n_mels + 2);
  std::vector<double> filter_freqs(mel_freqs.size());
  for (size_t i = 0; i < mel_freqs.size(); ++i) {
    filter_freqs[i] = ParakeetMelToHertzSlaney(mel_freqs[i]);
  }
  // :528 — `np.linspace(0, sampling_rate // 2, num_frequency_bins)`, so the top
  // is the INTEGER-divided Nyquist.
  const std::vector<double> fft_freqs =
      Linspace(0.0, static_cast<double>(cfg.sampling_rate / 2), n_freq);

  // :371-375 `_create_triangular_filter_bank`, then :532-535 the slaney area
  // normalisation. Emitted transposed to [n_mels, n_freq].
  std::vector<float> out(static_cast<size_t>(n_mels) * n_freq, 0.0f);
  for (int m = 0; m < n_mels; ++m) {
    const double diff_lo = filter_freqs[static_cast<size_t>(m) + 1] -
                           filter_freqs[static_cast<size_t>(m)];
    const double diff_hi = filter_freqs[static_cast<size_t>(m) + 2] -
                           filter_freqs[static_cast<size_t>(m) + 1];
    const double enorm = 2.0 / (filter_freqs[static_cast<size_t>(m) + 2] -
                                filter_freqs[static_cast<size_t>(m)]);
    for (int k = 0; k < n_freq; ++k) {
      const double down =
          -(filter_freqs[static_cast<size_t>(m)] - fft_freqs[static_cast<size_t>(k)]) /
          diff_lo;
      const double up =
          (filter_freqs[static_cast<size_t>(m) + 2] - fft_freqs[static_cast<size_t>(k)]) /
          diff_hi;
      const double tri = std::max(0.0, std::min(down, up));
      out[static_cast<size_t>(m) * n_freq + k] = static_cast<float>(tri * enorm);
    }
  }
  return out;
}

ParakeetAudioProcessor::ParakeetAudioProcessor(ParakeetExtractorConfig cfg)
    : cfg_(cfg), mel_filters_(ParakeetMelFilterBank(cfg)) {}

ParakeetAudioProcessor::ParakeetAudioProcessor(ParakeetExtractorConfig cfg,
                                               std::vector<float> mel_filters)
    : cfg_(cfg), mel_filters_(std::move(mel_filters)) {
  const size_t expect =
      static_cast<size_t>(cfg_.feature_size) * static_cast<size_t>(cfg_.num_freq_bins());
  if (mel_filters_.size() != expect) {
    throw std::runtime_error("ParakeetAudioProcessor: mel_filters size mismatch");
  }
}

ParakeetAudioFeatures ParakeetAudioProcessor::ProcessWaveform(const float* samples,
                                                              int64_t num_samples,
                                                              int sample_rate) const {
  if (sample_rate != cfg_.sampling_rate) {
    // feature_extraction_parakeet.py:195-201 raises rather than resampling.
    throw std::runtime_error(
        "ParakeetAudioProcessor: sampling rate mismatch; upstream refuses to "
        "resample");
  }
  const int n_fft = cfg_.n_fft;
  const int hop = cfg_.hop_length;
  const int win_length = cfg_.win_length;
  const int n_freq = cfg_.num_freq_bins();
  const int n_mels = cfg_.feature_size;

  // --- preemphasis (parakeet.py:198-214 / feature_extraction_parakeet.py:251-258)
  // y[0] = x[0]; y[i] = x[i] - preemphasis * x[i-1]. The `masked_fill` on the
  // same lines only zeroes BATCH padding, which a single clip has none of.
  std::vector<double> wav(static_cast<size_t>(std::max<int64_t>(num_samples, 0)));
  for (int64_t i = 0; i < num_samples; ++i) {
    const double prev = (i == 0) ? 0.0 : static_cast<double>(samples[i - 1]);
    wav[static_cast<size_t>(i)] =
        static_cast<double>(samples[i]) -
        (i == 0 ? 0.0 : static_cast<double>(cfg_.preemphasis) * prev);
  }

  // --- torch.stft(center=True, pad_mode="constant") (parakeet.py:175-183):
  // ZERO-pad by n_fft//2 on both sides — note this is NOT the reflect padding
  // the Whisper path uses.
  const int pad = n_fft / 2;
  const int64_t padded_len = static_cast<int64_t>(wav.size()) + 2 * pad;
  std::vector<double> padded(static_cast<size_t>(padded_len), 0.0);
  for (size_t i = 0; i < wav.size(); ++i) padded[static_cast<size_t>(pad) + i] = wav[i];

  const int64_t num_frames =
      padded_len >= n_fft ? 1 + (padded_len - n_fft) / hop : 0;

  // `torch.hann_window(win_length, periodic=False)`: w[j] = 0.5 - 0.5*cos(2*pi*j
  // /(win_length-1)) (parakeet.py:152). torch.stft then CENTRE-pads the window
  // out to n_fft with zeros.
  std::vector<double> window(static_cast<size_t>(n_fft), 0.0);
  const int win_offset = (n_fft - win_length) / 2;
  for (int j = 0; j < win_length; ++j) {
    window[static_cast<size_t>(win_offset + j)] =
        0.5 - 0.5 * std::cos(2.0 * kPi * j / static_cast<double>(win_length - 1));
  }

  // DFT twiddles for the n_freq one-sided bins (deviation 1 in the header).
  std::vector<double> cos_tab(static_cast<size_t>(n_freq) * n_fft);
  std::vector<double> sin_tab(static_cast<size_t>(n_freq) * n_fft);
  for (int k = 0; k < n_freq; ++k) {
    for (int j = 0; j < n_fft; ++j) {
      const double angle = 2.0 * kPi * k * j / static_cast<double>(n_fft);
      cos_tab[static_cast<size_t>(k) * n_fft + j] = std::cos(angle);
      sin_tab[static_cast<size_t>(k) * n_fft + j] = std::sin(angle);
    }
  }

  // --- power spectrum -> mel -> log (parakeet.py:189-196).
  std::vector<float> feats(static_cast<size_t>(num_frames) * n_mels, 0.0f);
  std::vector<double> frame(static_cast<size_t>(n_fft));
  std::vector<double> power(static_cast<size_t>(n_freq));
  for (int64_t f = 0; f < num_frames; ++f) {
    const int64_t start = f * hop;
    for (int j = 0; j < n_fft; ++j) {
      frame[static_cast<size_t>(j)] =
          padded[static_cast<size_t>(start + j)] * window[static_cast<size_t>(j)];
    }
    for (int k = 0; k < n_freq; ++k) {
      double re = 0.0, im = 0.0;
      const double* ck = &cos_tab[static_cast<size_t>(k) * n_fft];
      const double* sk = &sin_tab[static_cast<size_t>(k) * n_fft];
      for (int j = 0; j < n_fft; ++j) {
        re += frame[static_cast<size_t>(j)] * ck[j];
        im -= frame[static_cast<size_t>(j)] * sk[j];  // e^{-i2*pi*k*j/N}
      }
      power[static_cast<size_t>(k)] = re * re + im * im;
    }
    for (int m = 0; m < n_mels; ++m) {
      double acc = 0.0;
      const float* filt = &mel_filters_[static_cast<size_t>(m) * n_freq];
      for (int k = 0; k < n_freq; ++k) {
        acc += static_cast<double>(filt[k]) * power[static_cast<size_t>(k)];
      }
      feats[static_cast<size_t>(f) * n_mels + m] =
          static_cast<float>(std::log(acc + kLogZeroGuard));
    }
  }

  // --- per-bin normalisation over the VALID frames (parakeet.py:216-236).
  // features_lengths = (audio_len + n_fft//2*2 - n_fft) // hop_length (:220-223),
  // which for center=True padding is simply audio_len // hop_length.
  ParakeetAudioFeatures out;
  out.num_frames = num_frames;
  out.valid_frames = std::min<int64_t>(
      num_samples < 0 ? 0 : (num_samples + 2 * pad - n_fft) / hop, num_frames);
  if (out.valid_frames < 0) out.valid_frames = 0;

  const int64_t valid = out.valid_frames;
  for (int m = 0; m < n_mels; ++m) {
    double sum = 0.0;
    for (int64_t f = 0; f < valid; ++f) {
      sum += static_cast<double>(feats[static_cast<size_t>(f) * n_mels + m]);
    }
    // The mean divides by `features_lengths` and the variance by
    // `features_lengths - 1` (the UNBIASED denominator, :232-234) — an upstream
    // asymmetry we keep rather than "fix".
    const double mean = valid > 0 ? sum / static_cast<double>(valid) : 0.0;
    double var = 0.0;
    for (int64_t f = 0; f < valid; ++f) {
      const double d = static_cast<double>(feats[static_cast<size_t>(f) * n_mels + m]) - mean;
      var += d * d;
    }
    var = valid > 1 ? var / static_cast<double>(valid - 1) : 0.0;
    const double inv = 1.0 / (std::sqrt(var) + kEpsilon);
    for (int64_t f = 0; f < num_frames; ++f) {
      float& v = feats[static_cast<size_t>(f) * n_mels + m];
      // :236 `(x - mean) / (std + EPSILON) * mask` — padding frames become 0.
      v = (f < valid) ? static_cast<float>((static_cast<double>(v) - mean) * inv) : 0.0f;
    }
  }

  out.input_features = std::move(feats);
  return out;
}

}  // namespace vllm::multimodal
