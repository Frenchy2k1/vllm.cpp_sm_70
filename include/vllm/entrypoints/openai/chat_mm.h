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
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/audio_processor.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/qwen3vl_processor.h"

namespace vllm {
namespace tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h (the placeholder->id mapping)
}  // namespace tok
}  // namespace vllm

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

// ── Chat-template placeholder insertion (MM-SERVE-ENGINE) ──────────────────
//
// Ported from: vllm/entrypoints/chat_utils.py:886 `_add_placeholder` +
// :627 `get_placeholder_str`, and the per-model markers
// vllm/model_executor/models/qwen3_vl.py:1714 (image/video) +
// qwen2_audio.py:333 (audio). When a chat request carries a multimodal content
// part, vLLM inserts the model's placeholder STRING into the templated prompt at
// the part position; the tokenizer then maps it to ONE placeholder token, which
// the mm processor EXPANDS to N (= the grid/feature count) — the token-level
// expansion our RouteImageRgb / RouteAudioWav already perform
// (ExpandImagePlaceholders / ExpandAudioPlaceholders). These helpers own the
// STRING-level marker (the serving-layer half of the mirror); turning the marker
// into token ids needs the model tokenizer (MM-SERVE-E2E).

// The Qwen3-VL IMAGE placeholder marker: "<|vision_start|><|image_pad|><|vision_end|>"
// (qwen3_vl.py:1716). The single <|image_pad|> is what ExpandImagePlaceholders
// replaces with prod(grid_thw)/merge^2 copies of image_token_id.
std::string ImagePlaceholderString();

// The Qwen3-VL VIDEO placeholder marker: "<|vision_start|><|video_pad|><|vision_end|>"
// (qwen3_vl.py:1718).
std::string VideoPlaceholderString();

// The Qwen2-Audio placeholder marker: "Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"
// (qwen2_audio.py:335). `index` is the 1-based audio item number vLLM formats in.
std::string AudioPlaceholderString(int index);

// The placeholder marker for ONE parsed content part (get_placeholder_str
// dispatch): image_url -> ImagePlaceholderString; input_audio / audio_url ->
// AudioPlaceholderString(audio_index); "text" (and unrouted residual kinds) ->
// empty. `audio_index` is the running 1-based audio counter (Qwen numbers audios).
std::string ChatPlaceholderFor(const ChatContentPart& part, int audio_index);

// Collect the placeholder markers for a message's mm parts IN ORDER (mirrors the
// per-message mm_placeholder_storage, chat_utils.py:891). Text parts contribute
// no marker; audio parts are numbered 1..k. Empty when the message has no mm
// parts (a bare-string content is byte-identical — content_parts is nullopt).
std::vector<std::string> CollectChatPlaceholders(const ChatMessage& message);

// Build the marker-INJECTED content string for one message: walk its
// content_parts IN ORDER, appending each text part's text and each mm part's
// placeholder MARKER (ChatPlaceholderFor) at its position. Mirrors vLLM
// interleaving the placeholder string into the message content at the mm part
// offset (chat_utils.py:886 `_add_placeholder`); the single <|image_pad|> in the
// marker is what the tokenizer maps to ONE image_token_id, which the mm processor
// then EXPANDS to N. A bare-string message (content_parts nullopt) returns its
// content unchanged (byte-identical text path).
std::string BuildMarkerInjectedContent(const ChatMessage& message);

// ── The multimodal chat SEAM BODY (MM-SERVE-E2E) ───────────────────────────
//
// A decoded RGB image: raw HWC uint8 (height*width*3) + dims. Turning the
// container-format `image_url` bytes (PNG/JPEG/…) into this is the NAMED codec
// residual (no codec is vendored — the single-sequence e2e path itself consumes
// pre-decoded raw RGB); the production wiring supplies the codec.
struct DecodedImageRgb {
  std::vector<uint8_t> rgb;
  int64_t height = 0;
  int64_t width = 0;
};

// Image codec seam: decoded media bytes -> raw RGB + dims. Throws on an
// unsupported container. The default production codec rejects PNG/JPEG with a
// clear "codec residual" message; the e2e/test path supplies a raw-RGB
// passthrough (dims known from the fixture), exactly as the M2c single-sequence
// gate consumes raw 448x448x3 RGB (test_qwen3vl_e2e.cpp:116).
using ImageCodecFn = std::function<DecodedImageRgb(const DecodedMedia&)>;

// The chat-prompt renderer seam (structurally IDENTICAL to serving_chat.h
// ChatPromptFn — kept local so chat_mm.h need not pull serving_chat.h). The
// server's real chat-template renderer (MakeChatTemplatePromptFn) plugs in here.
using ChatPromptRenderFn = std::function<std::string(
    const std::vector<ChatMessage>&, bool,
    const std::vector<ChatCompletionToolsParam>&)>;

// Build the Qwen3-VL IMAGE multimodal chat seam body (the MultiModalChatFn the
// server sets via set_multimodal_chat_fn). The returned function turns chat
// `messages` into the engine's placeholder-EXPANDED MultiModalInputs:
//   1. inject the image placeholder marker at each image part's position
//      (BuildMarkerInjectedContent) and render the templated prompt via
//      `prompt_fn` (the real chat template);
//   2. tokenize the rendered prompt WITH special tokens — the tokenizer maps the
//      single <|image_pad|> marker to ONE `proc.config().image_token_id`
//      (tokenizer.h EncodeWithSpecialTokens, added tokens matched
//      leftmost-longest);
//   3. decode the image bytes (`codec`) and RouteImageRgb → EXPAND that single
//      id to N = prod(grid_thw)/merge^2 copies + build the mm_features handle
//      the engine mm generate overload carries onto Request.mm_features.
// Returns nullopt when no message carries an image part (the text path stays
// byte-identical). IMAGE only for now — video / audio / multiple-image are named
// residuals. `proc` and `tokenizer` must outlive the returned function (the
// server owns them for the process lifetime, like set_beam_search_tokenizer).
//
// This is the seam-body half of MM-SERVE-E2E: it produces the token-correct
// engine input; the GPU worker consuming Request.mm_features through the vision
// tower + merge + MRoPE/DeepStack forward is the remaining residual (the engine
// model runner has no mm-forward path yet — the M2c Qwen3VLGenerateGreedy driver
// runs it standalone, outside ModelRegistry::Forward).
std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&)>
MakeQwen3VLImageChatFn(const multimodal::Qwen3VLImageProcessor& proc,
                       const vllm::tok::Tokenizer& tokenizer,
                       ChatPromptRenderFn prompt_fn, ImageCodecFn codec);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_
