// MiniMax-H3 audio output: RIFF/WAVE serialization of the decoded waveform.
//
// The audio VAE emits a CHANNEL-MAJOR float waveform (all of channel 0, then all
// of channel 1) at 32 kHz. Every consumer wants interleaved PCM, so this does the
// interleave and the container in one place.
//
// WHY THIS EXISTS SEPARATELY FROM MP4 MUXING. `/v1/videos` ultimately returns MP4
// (H.264 video + audio), and how this project should obtain a muxer is an open
// DEPENDENCY DECISION (upstream vLLM-Omni shells out to ffmpeg; this tree has no
// subprocess precedent in the core library). WAV needs none of that: it is a
// 44-byte header plus samples, it is dependency-free, and it is required under
// EITHER outcome of that decision — a muxer will take PCM, and a standalone audio
// artifact is useful on its own. So it is implemented now and the container
// question stays open.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

void PutU32(std::string& out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

void PutU16(std::string& out, uint16_t value) {
  for (int i = 0; i < 2; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

}  // namespace

std::string MiniMaxH3WriteWav(const std::vector<float>& waveform, int64_t channels,
                              int64_t samples_per_channel, int64_t sample_rate) {
  VT_CHECK(channels > 0 && samples_per_channel > 0 && sample_rate > 0,
           "minimax_h3 wav: channels, samples and sample_rate must be positive");
  VT_CHECK(static_cast<int64_t>(waveform.size()) == channels * samples_per_channel,
           "minimax_h3 wav: waveform size does not match channels * samples_per_channel");

  const int64_t total = channels * samples_per_channel;
  const uint32_t data_bytes = static_cast<uint32_t>(total * 2);  // 16-bit PCM
  const uint16_t block_align = static_cast<uint16_t>(channels * 2);

  std::string out;
  out.reserve(44 + static_cast<size_t>(data_bytes));
  out += "RIFF";
  PutU32(out, 36u + data_bytes);  // chunk size = header remainder + payload
  out += "WAVE";
  out += "fmt ";
  PutU32(out, 16u);                                        // PCM fmt chunk size
  PutU16(out, 1u);                                         // format = PCM
  PutU16(out, static_cast<uint16_t>(channels));
  PutU32(out, static_cast<uint32_t>(sample_rate));
  PutU32(out, static_cast<uint32_t>(sample_rate) * block_align);  // byte rate
  PutU16(out, block_align);
  PutU16(out, 16u);                                        // bits per sample
  out += "data";
  PutU32(out, data_bytes);

  // The VAE's layout is CHANNEL-MAJOR; WAV is INTERLEAVED. Getting this backwards
  // produces audio that plays but with the channels time-smeared into each other.
  for (int64_t s = 0; s < samples_per_channel; ++s) {
    for (int64_t c = 0; c < channels; ++c) {
      float value = waveform[static_cast<size_t>(c * samples_per_channel + s)];
      value = std::min(1.0f, std::max(-1.0f, value));
      // Symmetric scaling by 32767 so +1.0 and -1.0 map to the extremes without
      // wrapping; -32768 is never produced.
      const int32_t quantized = static_cast<int32_t>(std::lround(value * 32767.0f));
      PutU16(out, static_cast<uint16_t>(static_cast<int16_t>(quantized)));
    }
  }
  return out;
}

// The INVERSE, needed once REFERENCE AUDIO became expressible: a ref2va request
// conditions on a supplied waveform, and the waveform arrives as a file.
//
// Deliberately NOT a resampler. The checkpoint's audio VAE is 32 kHz; encoding a
// 44.1 kHz file as if it were 32 kHz would silently shift every latent frame, so a
// rate mismatch is REFUSED and the caller resamples. Upstream can be lax here only
// because it has torchaudio (vae.py:298-306); this library has no audio dependency
// and will not pretend to.
std::vector<float> MiniMaxH3ReadWav(const std::string& bytes, int64_t want_channels,
                                    int64_t want_sample_rate, int64_t* out_samples_per_channel) {
  VT_CHECK(want_channels > 0, "minimax_h3 wav: want_channels must be positive");
  VT_CHECK(bytes.size() >= 44, "minimax_h3 wav: too short to be a RIFF/WAVE file");
  VT_CHECK(bytes.compare(0, 4, "RIFF") == 0 && bytes.compare(8, 4, "WAVE") == 0,
           "minimax_h3 wav: not a RIFF/WAVE file");

  auto u16 = [&](size_t at) -> uint32_t {
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[at])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[at + 1])) << 8);
  };
  auto u32 = [&](size_t at) -> uint32_t { return u16(at) | (u16(at + 2) << 16); };

  int64_t channels = 0, sample_rate = 0, bits = 0;
  bool have_fmt = false;
  size_t data_at = 0, data_len = 0;
  // Walk the CHUNK LIST rather than assuming a 44-byte header: real files carry
  // LIST/fact chunks before `data`, and a fixed offset would read metadata as audio.
  for (size_t at = 12; at + 8 <= bytes.size();) {
    const size_t len = u32(at + 4);
    const size_t body = at + 8;
    if (bytes.compare(at, 4, "fmt ") == 0 && len >= 16 && body + 16 <= bytes.size()) {
      VT_CHECK(u16(body) == 1u, "minimax_h3 wav: only uncompressed PCM is supported");
      channels = u16(body + 2);
      sample_rate = u32(body + 4);
      bits = u16(body + 14);
      have_fmt = true;
    } else if (bytes.compare(at, 4, "data") == 0) {
      data_at = body;
      data_len = std::min(len, bytes.size() - body);
      break;
    }
    at = body + len + (len & 1u);  // chunks are word-aligned
  }
  VT_CHECK(have_fmt, "minimax_h3 wav: no fmt chunk");
  VT_CHECK(data_at != 0, "minimax_h3 wav: no data chunk");
  VT_CHECK(bits == 16, "minimax_h3 wav: only 16-bit PCM is supported");
  VT_CHECK(channels > 0, "minimax_h3 wav: the fmt chunk declares no channels");
  VT_CHECK(want_sample_rate <= 0 || sample_rate == want_sample_rate,
           "minimax_h3 wav: sample rate does not match the model's (resample the file first; "
           "this reader deliberately does not)");

  const int64_t frames = static_cast<int64_t>(data_len) / (2 * channels);
  VT_CHECK(frames > 0, "minimax_h3 wav: the data chunk holds no samples");

  // Interleaved -> CHANNEL-MAJOR, the layout every other H3 audio surface uses.
  // A MONO source is REPEATED across the model's channels and anything wider is
  // truncated, mirroring vae.py:305-313.
  std::vector<float> out(static_cast<size_t>(want_channels * frames), 0.0f);
  for (int64_t t = 0; t < frames; ++t) {
    for (int64_t c = 0; c < want_channels; ++c) {
      const int64_t src = channels == 1 ? 0 : std::min<int64_t>(c, channels - 1);
      const int16_t pcm = static_cast<int16_t>(static_cast<uint16_t>(
          u16(data_at + static_cast<size_t>((t * channels + src) * 2))));
      out[static_cast<size_t>(c * frames + t)] = static_cast<float>(pcm) / 32768.0f;
    }
  }
  if (out_samples_per_channel != nullptr) *out_samples_per_channel = frames;
  return out;
}

}  // namespace vllm
