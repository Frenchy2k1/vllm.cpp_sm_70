// OpenAI chat multimodal content-part DECODE + ROUTE (ROAD-V1-MM serving W1).
//
// Ported from:
//   - vllm/entrypoints/chat_utils.py: MM_PARSER_MAP:1478 +
//     _parse_chat_message_content_mm_part:1524 (the content-part schema — the
//     PARSE half lives in protocol.cpp / ChatContentPart);
//   - vllm/multimodal/media.py + vllm/multimodal/utils.py:35-113 (the data: URI /
//     base64 media encode; we mirror the decode: data:{mimetype};base64,{b64}).
//
// This is the CPU-reachable half of wiring multimodal into the OpenAI server: the
// base64 / data-URI DECODE and the ROUTE of the decoded bytes through the EXISTING
// single-sequence mm processors (multimodal/qwen3vl_processor.h + audio_processor.h)
// to produce a MultiModalInputs (placeholder-expanded prompt ids + mm_features).
// It runs entirely on CPU with only the processor CONFIG (no model weights).
//
// NAMED RESIDUALS (out of this brick):
//   - the container-format image decode (PNG/JPEG -> RGB + dims): no codec is
//     vendored (the single-sequence e2e path itself consumes pre-decoded raw RGB),
//     so RouteImageRgb takes the raw RGB the processor expects;
//   - fetching an http(s) media URL (vs an inline data: URI);
//   - plumbing the produced MultiModalInputs into the engine request (the engine
//     add_request has no mm-features overload yet) and the mm model forward (GPU).
#ifndef VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_
#define VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/audio_processor.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/qwen3vl_processor.h"

namespace vllm::entrypoints::openai {

// A decoded media payload: the declared MIME type plus the raw bytes.
struct DecodedMedia {
  std::string media_type;      // e.g. "image/png", "audio/wav" (empty if absent)
  std::vector<uint8_t> bytes;  // raw decoded bytes
};

// Standard base64 decode (RFC 4648 alphabet, '=' padding). ASCII whitespace
// (space/tab/CR/LF) is ignored so wrapped payloads decode. Throws
// std::runtime_error on an invalid character or a truncated group.
std::vector<uint8_t> DecodeBase64(const std::string& b64);

// Decode an RFC 2397 data: URI — data:[<mediatype>][;base64],<payload>. Only the
// ;base64 form is supported (the shape OpenAI clients emit); throws on a
// non-`data:` URI (an http(s) fetch is a named residual) or a non-base64 data URI.
DecodedMedia DecodeDataUri(const std::string& uri);

// Whether a chat message carries any non-text (multimodal) content part.
bool HasMultiModalParts(const ChatMessage& m);

// Decode an image_url content part's `url` (a data: URI) to its raw bytes. The
// bytes are the container-format payload (PNG/JPEG/…); turning them into RGB +
// dims is the NAMED codec residual — RouteImageRgb consumes the raw RGB directly.
DecodedMedia DecodeImageUrlPart(const ChatContentPart& part);

// Decode an input_audio content part's base64 `data` to raw bytes (a container
// such as WAV). audio_url data: URIs go through DecodeDataUri instead.
DecodedMedia DecodeInputAudioPart(const ChatContentPart& part);

// Route a decoded PCM16 WAV through the EXISTING Whisper audio processor to
// produce the mm inputs + placeholder-expanded prompt, exactly as the single-
// sequence path does (DecodeWavPcm16Mono -> ProcessWaveform ->
// ExpandAudioPlaceholders). `prompt_ids` is the pre-tokenized prompt carrying one
// audio_placeholder_id per audio item (the tokenizer text path is out of scope,
// mirroring the M2c/A-track e2e fixtures). Throws on a non-PCM16/non-mono WAV.
multimodal::MultiModalInputs RouteAudioWav(
    const multimodal::WhisperAudioProcessor& proc, const DecodedMedia& audio,
    const std::vector<int32_t>& prompt_ids);

// Route already-decoded RGB image bytes (HWC uint8, height*width*3) through the
// EXISTING Qwen3-VL image processor (ProcessImage -> ExpandImagePlaceholders).
// The container-format decode (PNG/JPEG -> this RGB + dims) is the NAMED residual.
// `prompt_ids` carries one image_token_id per image item.
multimodal::MultiModalInputs RouteImageRgb(
    const multimodal::Qwen3VLImageProcessor& proc, const uint8_t* rgb,
    int64_t height, int64_t width, const std::vector<int32_t>& prompt_ids);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_
