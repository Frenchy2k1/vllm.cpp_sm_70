// ROAD-V1-MM serving W1 — OpenAI chat multimodal content-part DECODE + ROUTE.
// See include/vllm/entrypoints/openai/chat_mm.h for the ported-from map and the
// named residuals.
#include "vllm/entrypoints/openai/chat_mm.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm::entrypoints::openai {

namespace {

// RFC 4648 reverse alphabet: value 0..63 for a base64 char, -1 invalid, -2 skip
// (ASCII whitespace), -3 padding '='.
int Base64Value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  if (c == '=') return -3;
  if (c == ' ' || c == '\t' || c == '\r' || c == '\n') return -2;
  return -1;
}

}  // namespace

std::vector<uint8_t> DecodeBase64(const std::string& b64) {
  std::vector<uint8_t> out;
  out.reserve((b64.size() / 4) * 3);
  int quad[4];
  int n = 0;         // how many real (non-skip) symbols collected in this group
  int pad = 0;       // padding chars seen in this group
  bool ended = false;
  for (const char ch : b64) {
    const int v = Base64Value(static_cast<unsigned char>(ch));
    if (v == -2) continue;  // whitespace
    if (v == -1) {
      throw std::runtime_error("DecodeBase64: invalid base64 character");
    }
    if (ended) {
      // Data after a completed padded group is malformed.
      throw std::runtime_error("DecodeBase64: trailing data after padding");
    }
    if (v == -3) {  // padding
      if (n < 2) throw std::runtime_error("DecodeBase64: misplaced padding");
      quad[n++] = 0;
      ++pad;
    } else {
      if (pad > 0) throw std::runtime_error("DecodeBase64: data after padding");
      quad[n++] = v;
    }
    if (n == 4) {
      const uint32_t triple = (static_cast<uint32_t>(quad[0]) << 18) |
                              (static_cast<uint32_t>(quad[1]) << 12) |
                              (static_cast<uint32_t>(quad[2]) << 6) |
                              static_cast<uint32_t>(quad[3]);
      out.push_back(static_cast<uint8_t>((triple >> 16) & 0xFF));
      if (pad < 2) out.push_back(static_cast<uint8_t>((triple >> 8) & 0xFF));
      if (pad < 1) out.push_back(static_cast<uint8_t>(triple & 0xFF));
      n = 0;
      if (pad > 0) ended = true;
      pad = 0;
    }
  }
  if (n != 0) {
    throw std::runtime_error("DecodeBase64: truncated base64 group");
  }
  return out;
}

DecodedMedia DecodeDataUri(const std::string& uri) {
  // RFC 2397: data:[<mediatype>][;base64],<payload>
  static const std::string kScheme = "data:";
  if (uri.rfind(kScheme, 0) != 0) {
    throw std::runtime_error(
        "DecodeDataUri: not a data: URI (http(s) media fetch is a named "
        "residual)");
  }
  const std::string::size_type comma = uri.find(',');
  if (comma == std::string::npos) {
    throw std::runtime_error("DecodeDataUri: malformed data URI (no comma)");
  }
  // Header between "data:" and the comma: <mediatype>[;base64].
  const std::string header =
      uri.substr(kScheme.size(), comma - kScheme.size());
  const std::string payload = uri.substr(comma + 1);

  const std::string::size_type semi = header.find(';');
  DecodedMedia media;
  media.media_type = header.substr(0, semi);  // may be empty
  const bool is_base64 =
      header.size() >= 7 && header.compare(header.size() - 7, 7, ";base64") == 0;
  if (!is_base64) {
    throw std::runtime_error(
        "DecodeDataUri: only ;base64 data URIs are supported");
  }
  media.bytes = DecodeBase64(payload);
  return media;
}

bool HasMultiModalParts(const ChatMessage& m) {
  if (!m.content_parts.has_value()) return false;
  for (const ChatContentPart& p : *m.content_parts) {
    if (p.type != "text") return true;
  }
  return false;
}

DecodedMedia DecodeImageUrlPart(const ChatContentPart& part) {
  return DecodeDataUri(part.url);
}

DecodedMedia DecodeInputAudioPart(const ChatContentPart& part) {
  DecodedMedia media;
  media.media_type = part.audio_format.empty()
                         ? std::string()
                         : "audio/" + part.audio_format;
  media.bytes = DecodeBase64(part.audio_data);
  return media;
}

multimodal::MultiModalInputs RouteAudioWav(
    const multimodal::WhisperAudioProcessor& proc, const DecodedMedia& audio,
    const std::vector<int32_t>& prompt_ids) {
  const multimodal::AudioProcessorConfig& cfg = proc.config();

  const multimodal::DecodedAudio decoded =
      multimodal::DecodeWavPcm16Mono(audio.bytes.data(), audio.bytes.size());
  multimodal::AudioKwargs kw = proc.ProcessWaveform(
      decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()),
      decoded.sampling_rate);

  const int num_audio_tokens = cfg.max_source_positions;
  std::vector<std::array<int, 2>> placeholders;
  std::vector<int32_t> expanded = multimodal::ExpandAudioPlaceholders(
      prompt_ids, cfg.audio_placeholder_id, {num_audio_tokens}, &placeholders);

  multimodal::MultiModalInputs out;
  out.prompt_token_ids = std::move(expanded);
  if (!placeholders.empty()) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = "audio";
    spec.offset = placeholders[0][0];
    spec.length = placeholders[0][1];
    spec.audio_data = std::make_shared<multimodal::AudioKwargs>(std::move(kw));
    spec.mm_hash = proc.HashAudio(
        decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()));
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

multimodal::MultiModalInputs RouteImageRgb(
    const multimodal::Qwen3VLImageProcessor& proc, const uint8_t* rgb,
    int64_t height, int64_t width, const std::vector<int32_t>& prompt_ids) {
  const multimodal::Qwen3VLProcessorConfig& cfg = proc.config();

  multimodal::ImageKwargs kw = proc.ProcessImage(rgb, height, width);
  const std::array<int64_t, 3> grid = kw.image_grid_thw;

  std::vector<std::array<int64_t, 3>> grids{grid};
  std::vector<std::array<int, 2>> placeholders;
  std::vector<int32_t> expanded = multimodal::ExpandImagePlaceholders(
      prompt_ids, cfg.image_token_id, cfg.merge_size, grids, &placeholders);

  multimodal::MultiModalInputs out;
  out.prompt_token_ids = std::move(expanded);
  if (!placeholders.empty()) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = "image";
    spec.offset = placeholders[0][0];
    spec.length = placeholders[0][1];
    spec.mm_hash = proc.HashImage(rgb, height, width);
    spec.data = std::make_shared<multimodal::ImageKwargs>(std::move(kw));
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

}  // namespace vllm::entrypoints::openai
