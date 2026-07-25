// Ported from: transformers WhisperFeatureExtractor
// (feature_extraction_whisper.py `_torch_extract_fbank_features`) — the torch STFT
// log-mel path that runs when torch is installed; audio_utils.mel_filter_bank
// (slaney) dumped as a golden constant; vllm/model_executor/models/whisper.py
// (get_num_audio_tokens:656, _get_prompt_updates:740). @ vLLM e24d1b24 /
// transformers 5.13.1. See audio_processor.h for full provenance.
//
// STFT parity note: the oracle uses torch.stft (FFT). Our direct DFT (only the
// 201 needed bins per frame) differs from FFT only in float summation order, so
// the log-mel matches to a small rel-L2 (measured, gated), NOT bit-exact — this is
// stated + justified. The placeholder ids and mm-hash are bit/byte-exact.
#include "vllm/multimodal/audio_processor.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "vllm/multimodal/hasher.h"

namespace vllm::multimodal {

namespace {

// Read a little-endian unsigned integer of `n` bytes from p.
uint32_t ReadLE(const uint8_t* p, int n) {
  uint32_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}

constexpr double kPi = 3.14159265358979323846;

}  // namespace

DecodedAudio DecodeWavPcm16Mono(const uint8_t* wav, size_t n) {
  if (n < 44 || std::memcmp(wav, "RIFF", 4) != 0 ||
      std::memcmp(wav + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("DecodeWavPcm16Mono: not a RIFF/WAVE buffer");
  }
  // Walk chunks after the 12-byte RIFF header.
  size_t pos = 12;
  int channels = 0, bits = 0, rate = 0;
  const uint8_t* data = nullptr;
  size_t data_len = 0;
  bool have_fmt = false;
  while (pos + 8 <= n) {
    const uint8_t* id = wav + pos;
    const uint32_t sz = ReadLE(wav + pos + 4, 4);
    const size_t body = pos + 8;
    if (body + sz > n) break;
    if (std::memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
      const uint16_t fmt = static_cast<uint16_t>(ReadLE(wav + body, 2));
      channels = static_cast<int>(ReadLE(wav + body + 2, 2));
      rate = static_cast<int>(ReadLE(wav + body + 4, 4));
      bits = static_cast<int>(ReadLE(wav + body + 14, 2));
      if (fmt != 1) throw std::runtime_error("DecodeWavPcm16Mono: not PCM (fmt!=1)");
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      data = wav + body;
      data_len = sz;
    }
    pos = body + sz + (sz & 1);  // chunks are word-aligned
  }
  if (!have_fmt || data == nullptr) {
    throw std::runtime_error("DecodeWavPcm16Mono: missing fmt/data chunk");
  }
  if (channels != 1) throw std::runtime_error("DecodeWavPcm16Mono: not mono");
  if (bits != 16) throw std::runtime_error("DecodeWavPcm16Mono: not 16-bit PCM");

  DecodedAudio out;
  out.sampling_rate = rate;
  const size_t num = data_len / 2;
  out.samples.resize(num);
  for (size_t i = 0; i < num; ++i) {
    const int16_t s = static_cast<int16_t>(ReadLE(data + 2 * i, 2));
    out.samples[i] = static_cast<float>(s) / 32768.0f;
  }
  return out;
}

WhisperAudioProcessor::WhisperAudioProcessor(AudioProcessorConfig cfg,
                                             std::vector<float> mel_filters)
    : cfg_(std::move(cfg)), mel_filters_(std::move(mel_filters)) {
  const size_t expect =
      static_cast<size_t>(cfg_.num_freq_bins()) * static_cast<size_t>(cfg_.n_mels);
  if (mel_filters_.size() != expect) {
    throw std::runtime_error("WhisperAudioProcessor: mel_filters size mismatch");
  }
}

AudioKwargs WhisperAudioProcessor::ProcessWaveform(const float* samples,
                                                   int64_t num_samples,
                                                   int sample_rate) const {
  if (sample_rate != cfg_.sampling_rate) {
    // Genuine resample (windowed sinc, à la librosa) is deferred — mirror the
    // image SmartResize/bicubic identity-only guard. The whisper-small fixture is
    // already at cfg.sampling_rate (16 kHz), so this path is never taken here.
    throw std::runtime_error(
        "WhisperAudioProcessor: resample deferred; provide audio at "
        "cfg.sampling_rate (16 kHz)");
  }
  const int n_fft = cfg_.n_fft;
  const int hop = cfg_.hop_length;
  const int n_mels = cfg_.n_mels;
  const int n_freq = cfg_.num_freq_bins();
  const int n_pad = cfg_.n_samples_padded();  // 480000

  // ---- pad / truncate the waveform to n_pad (WhisperFeatureExtractor.pad,
  // padding="max_length", max_length=n_samples, truncation=True) ----
  std::vector<double> wav(static_cast<size_t>(n_pad), 0.0);
  const int64_t copy = std::min<int64_t>(num_samples, n_pad);
  for (int64_t i = 0; i < copy; ++i) wav[static_cast<size_t>(i)] = samples[i];

  // ---- torch.stft(center=True) reflect-pads by n_fft/2 each side ----
  const int p = n_fft / 2;  // 200
  const int L = n_pad;
  const int Lp = L + 2 * p;
  std::vector<double> padded(static_cast<size_t>(Lp), 0.0);
  for (int j = 0; j < L; ++j) padded[static_cast<size_t>(p + j)] = wav[static_cast<size_t>(j)];
  // reflect (no edge repeat): left[i]=a[p-i] (i=0..p-1); right[k]=a[L-2-k] (k=0..p-1)
  for (int i = 0; i < p; ++i) padded[static_cast<size_t>(i)] = wav[static_cast<size_t>(p - i)];
  for (int k = 0; k < p; ++k) {
    const int src = L - 2 - k;
    padded[static_cast<size_t>(p + L + k)] =
        (src >= 0) ? wav[static_cast<size_t>(src)] : 0.0;
  }

  // number of STFT frames = 1 + (Lp - n_fft)/hop; drop the last (stft[...,:-1]).
  const int n_frames_full = 1 + (Lp - n_fft) / hop;
  const int n_frames = n_frames_full - 1;

  // ---- periodic Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n / n_fft) ----
  std::vector<double> win(static_cast<size_t>(n_fft));
  for (int nn = 0; nn < n_fft; ++nn) {
    win[static_cast<size_t>(nn)] = 0.5 - 0.5 * std::cos(2.0 * kPi * nn / n_fft);
  }

  // ---- precompute DFT twiddles cos/sin[k*n_fft + j] for k in [0,n_freq), j<n_fft ----
  std::vector<double> cosk(static_cast<size_t>(n_freq) * n_fft);
  std::vector<double> sink(static_cast<size_t>(n_freq) * n_fft);
  for (int k = 0; k < n_freq; ++k) {
    for (int j = 0; j < n_fft; ++j) {
      const double ang = 2.0 * kPi * k * j / n_fft;
      cosk[static_cast<size_t>(k) * n_fft + j] = std::cos(ang);
      sink[static_cast<size_t>(k) * n_fft + j] = std::sin(ang);
    }
  }

  // ---- log-mel spectrogram [n_mels, n_frames] ----
  std::vector<float> feat(static_cast<size_t>(n_mels) * n_frames);
  std::vector<double> mag2(static_cast<size_t>(n_freq));
  std::vector<double> frame(static_cast<size_t>(n_fft));
  double gmax = -1e300;  // running max over log10(mel) for the -8 clamp

  // First pass: fill log10(clamp(mel_spec, 1e-10)) and track the global max.
  for (int f = 0; f < n_frames; ++f) {
    const int start = f * hop;
    for (int j = 0; j < n_fft; ++j) {
      frame[static_cast<size_t>(j)] =
          padded[static_cast<size_t>(start + j)] * win[static_cast<size_t>(j)];
    }
    for (int k = 0; k < n_freq; ++k) {
      double re = 0.0, im = 0.0;
      const double* ck = &cosk[static_cast<size_t>(k) * n_fft];
      const double* sk = &sink[static_cast<size_t>(k) * n_fft];
      for (int j = 0; j < n_fft; ++j) {
        re += frame[static_cast<size_t>(j)] * ck[j];
        im -= frame[static_cast<size_t>(j)] * sk[j];  // e^{-i...}
      }
      mag2[static_cast<size_t>(k)] = re * re + im * im;
    }
    for (int m = 0; m < n_mels; ++m) {
      double acc = 0.0;  // mel_filters.T @ magnitudes: sum_k filt[k,m]*mag2[k]
      for (int k = 0; k < n_freq; ++k) {
        acc += static_cast<double>(mel_filters_[static_cast<size_t>(k) * n_mels + m]) *
               mag2[static_cast<size_t>(k)];
      }
      if (acc < 1e-10) acc = 1e-10;  // torch.clamp(min=1e-10)
      const double lg = std::log10(acc);
      feat[static_cast<size_t>(m) * n_frames + f] = static_cast<float>(lg);
      if (lg > gmax) gmax = lg;
    }
  }

  // Second pass: x=max(x, gmax-8); x=(x+4)/4 (whisper normalization).
  const double floor = gmax - 8.0;
  for (auto& v : feat) {
    double x = v;
    if (x < floor) x = floor;
    x = (x + 4.0) / 4.0;
    v = static_cast<float>(x);
  }

  AudioKwargs out;
  out.n_mels = n_mels;
  out.n_frames = n_frames;
  out.input_features = std::move(feat);
  return out;
}

std::string WhisperAudioProcessor::HashAudio(const float* samples,
                                             int64_t num_samples) const {
  return MultiModalHasher::HashAudioF32(cfg_.model_id, samples, num_samples);
}

std::vector<int32_t> ExpandAudioPlaceholders(
    const std::vector<int32_t>& prompt_ids, int32_t audio_placeholder_id,
    const std::vector<int>& num_audio_tokens_per_item,
    std::vector<std::array<int, 2>>* placeholders) {
  std::vector<int32_t> out;
  out.reserve(prompt_ids.size());
  if (placeholders) placeholders->clear();
  size_t item = 0;
  for (int32_t id : prompt_ids) {
    if (id == audio_placeholder_id && item < num_audio_tokens_per_item.size()) {
      const int n = num_audio_tokens_per_item[item++];
      const int offset = static_cast<int>(out.size());
      for (int i = 0; i < n; ++i) out.push_back(audio_placeholder_id);
      if (placeholders) placeholders->push_back({offset, n});
    } else {
      out.push_back(id);
    }
  }
  return out;
}

}  // namespace vllm::multimodal
