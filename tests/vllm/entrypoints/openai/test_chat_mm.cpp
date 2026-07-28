// ROAD-V1-MM serving W1 — CPU gate for the OpenAI multimodal content-part
// parse + decode + processor route (the first CPU-reachable brick of wiring
// multimodal into the OpenAI server).
//
// Proves, entirely on CPU with only the processor CONFIG (no model weights):
//   1. INERTNESS (RED line): a bare-string chat `content` parses byte-identically
//      to before — content set, content_parts == nullopt, no mm parts, the
//      DefaultChatPromptFallback prompt unchanged.
//   2. base64 / data-URI DECODE round-trips against known vectors + the committed
//      fixtures.
//   3. PARSE: an array-form `content` ([{text},{image_url},{input_audio}]) parses
//      into typed ChatContentParts with the mirrored schema; the RED contrast is
//      that the pre-wiring parser only handled `content.is_string()`, so an array
//      produced content_parts == nullopt / empty content (asserted inert above).
//   4. ROUTE: an input_audio part (base64 of the committed whisper WAV) runs the
//      EXISTING Whisper processor -> input_features [80,3000] + 1500 placeholder
//      tokens; an image_url part (data: URI of the committed raw-RGB fixture) runs
//      the EXISTING Qwen3-VL processor -> grid [1,28,28] + 196 merged tokens.
//      Concrete numbers vs the M1/A1 processor-parity fixtures.
//
// Fixtures: tests/vllm/multimodal/fixtures/{qwen3vl,whisper_audio} (the same M1 /
// A1 processor-parity goldens). NAMED residual: the container-format image decode
// (PNG/JPEG -> RGB) — no codec is vendored, so the image route consumes the raw
// RGB the processor expects (as the single-sequence e2e path itself does).
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

namespace {

using vllm::entrypoints::openai::ChatCompletionRequest;
using vllm::entrypoints::openai::ChatContentPart;
using vllm::entrypoints::openai::ChatMessage;

std::string ImgFixDir() { return std::string(MM_FIXTURE_DIR); }
std::string AudFixDir() { return std::string(MM_AUDIO_FIXTURE_DIR); }

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

// Standalone base64 encode (test-side; mirrors what an OpenAI client sends).
std::string EncodeBase64(const std::vector<uint8_t>& in) {
  static const char* kAlpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  size_t i = 0;
  for (; i + 3 <= in.size(); i += 3) {
    const uint32_t n = (static_cast<uint32_t>(in[i]) << 16) |
                       (static_cast<uint32_t>(in[i + 1]) << 8) |
                       static_cast<uint32_t>(in[i + 2]);
    out.push_back(kAlpha[(n >> 18) & 63]);
    out.push_back(kAlpha[(n >> 12) & 63]);
    out.push_back(kAlpha[(n >> 6) & 63]);
    out.push_back(kAlpha[n & 63]);
  }
  const size_t rem = in.size() - i;
  if (rem == 1) {
    const uint32_t n = static_cast<uint32_t>(in[i]) << 16;
    out.push_back(kAlpha[(n >> 18) & 63]);
    out.push_back(kAlpha[(n >> 12) & 63]);
    out.push_back('=');
    out.push_back('=');
  } else if (rem == 2) {
    const uint32_t n = (static_cast<uint32_t>(in[i]) << 16) |
                       (static_cast<uint32_t>(in[i + 1]) << 8);
    out.push_back(kAlpha[(n >> 18) & 63]);
    out.push_back(kAlpha[(n >> 12) & 63]);
    out.push_back(kAlpha[(n >> 6) & 63]);
    out.push_back('=');
  }
  return out;
}

vllm::multimodal::Qwen3VLProcessorConfig ImageConfigFromManifest(
    const nlohmann::json& m) {
  vllm::multimodal::Qwen3VLProcessorConfig cfg;
  const auto& c = m.at("config");
  cfg.patch_size = c.at("patch_size").get<int>();
  cfg.temporal_patch_size = c.at("temporal_patch_size").get<int>();
  cfg.merge_size = c.at("merge_size").get<int>();
  cfg.image_mean = c.at("image_mean")[0].get<double>();
  cfg.image_std = c.at("image_std")[0].get<double>();
  cfg.image_token_id = c.at("image_token_id").get<int32_t>();
  cfg.vision_start_token_id = c.at("vision_start_token_id").get<int32_t>();
  cfg.vision_end_token_id = c.at("vision_end_token_id").get<int32_t>();
  cfg.model_id = m.at("model_id").get<std::string>();
  return cfg;
}

vllm::multimodal::AudioProcessorConfig AudioConfigFromManifest(
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

}  // namespace

// ---------------------------------------------------------------------------
// 1. INERTNESS (the RED line): a bare-string content is byte-identical to today.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm inertness: bare-string content unchanged") {
  const nlohmann::json j = {
      {"messages",
       {{{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Hello there"}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();
  REQUIRE(req.messages.size() == 2);

  // Bare-string content: content set, content_parts stays nullopt, no mm parts.
  for (const ChatMessage& m : req.messages) {
    CHECK(m.content.has_value());
    CHECK_FALSE(m.content_parts.has_value());
    CHECK_FALSE(vllm::entrypoints::openai::HasMultiModalParts(m));
  }
  CHECK(*req.messages[1].content == "Hello there");

  // The rendered fallback prompt is unchanged from the pure-text path.
  const std::string prompt =
      vllm::entrypoints::openai::DefaultChatPromptFallback(
          req.messages, /*add_generation_prompt=*/true, {});
  CHECK(prompt ==
        "system: You are helpful.\nuser: Hello there\nassistant:");
}

// ---------------------------------------------------------------------------
// 2. base64 / data-URI decode round-trips.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm base64 decode: known vectors + round-trip") {
  using vllm::entrypoints::openai::DecodeBase64;
  auto S = [](const char* s) {
    return std::vector<uint8_t>(s, s + std::strlen(s));
  };
  CHECK(DecodeBase64("TWFu") == S("Man"));      // no padding
  CHECK(DecodeBase64("TWE=") == S("Ma"));        // one pad
  CHECK(DecodeBase64("TQ==") == S("M"));         // two pad
  CHECK(DecodeBase64("aGVsbG8gd29ybGQ=") == S("hello world"));
  CHECK(DecodeBase64("aGVsbG8g\nd29ybGQ=") == S("hello world"));  // whitespace ok
  CHECK_THROWS(DecodeBase64("****"));             // invalid chars
  CHECK_THROWS(DecodeBase64("TWF"));              // truncated group

  // Round-trip a byte ramp through the test encoder + the production decoder.
  std::vector<uint8_t> ramp(256);
  for (int i = 0; i < 256; ++i) ramp[i] = static_cast<uint8_t>(i);
  CHECK(DecodeBase64(EncodeBase64(ramp)) == ramp);
}

TEST_CASE("chat-mm data-URI decode: header + base64 payload") {
  using vllm::entrypoints::openai::DecodeDataUri;
  const auto m = DecodeDataUri("data:image/png;base64,TWFu");
  CHECK(m.media_type == "image/png");
  CHECK(m.bytes == std::vector<uint8_t>({'M', 'a', 'n'}));
  CHECK_THROWS(DecodeDataUri("https://example.com/a.png"));  // http residual
  CHECK_THROWS(DecodeDataUri("data:image/png,rawtext"));     // non-base64
}

// ---------------------------------------------------------------------------
// 3+4. PARSE + ROUTE: input_audio content part -> Whisper processor.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm audio: input_audio part -> processor -> features + expansion") {
  const std::string dir = AudFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = AudioConfigFromManifest(manifest);
  const std::vector<float> mel_filters = ReadF32(dir + "/mel_filters_f32.bin");
  const std::vector<uint8_t> wav = ReadBytes(dir + "/audio_tone_16k_mono.wav");

  // Build the OpenAI chat request with an inline base64 WAV input_audio part.
  const std::string wav_b64 = EncodeBase64(wav);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "Transcribe this:"}},
           {{"type", "input_audio"},
            {"input_audio", {{"data", wav_b64}, {"format", "wav"}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();

  // PARSE: the array form populated content_parts (the pre-wiring parser dropped
  // it — the inert bare-string case above is the RED contrast).
  REQUIRE(req.messages.size() == 1);
  const ChatMessage& msg = req.messages[0];
  REQUIRE(msg.content_parts.has_value());
  REQUIRE(msg.content_parts->size() == 2);
  CHECK((*msg.content_parts)[0].type == "text");
  CHECK((*msg.content_parts)[0].text == "Transcribe this:");
  CHECK((*msg.content_parts)[1].type == "input_audio");
  CHECK((*msg.content_parts)[1].audio_format == "wav");
  CHECK(vllm::entrypoints::openai::HasMultiModalParts(msg));
  // The joined text span drives the existing text prompt path.
  CHECK(*msg.content == "Transcribe this:");

  // DECODE: the base64 payload round-trips to the exact committed WAV bytes.
  const auto media =
      vllm::entrypoints::openai::DecodeInputAudioPart((*msg.content_parts)[1]);
  CHECK(media.media_type == "audio/wav");
  REQUIRE(media.bytes == wav);

  // ROUTE: the existing Whisper processor produces the log-mel features + the
  // placeholder-expanded prompt.
  vllm::multimodal::WhisperAudioProcessor proc(cfg, mel_filters);
  const std::vector<int32_t> prompt_ids = {100, 200, cfg.audio_placeholder_id,
                                           300};
  const vllm::multimodal::MultiModalInputs mm =
      vllm::entrypoints::openai::RouteAudioWav(proc, media, prompt_ids);

  const int num_tokens = cfg.max_source_positions;  // 1500
  const int64_t n_frames =
      manifest.at("input_features").at("shape")[1].get<int64_t>();  // 3000
  REQUIRE(mm.mm_features.size() == 1);
  CHECK(mm.mm_features[0].modality == "audio");
  CHECK(mm.mm_features[0].offset == 2);           // placeholder position
  CHECK(mm.mm_features[0].length == num_tokens);  // 1500
  REQUIRE(mm.mm_features[0].audio_data != nullptr);
  CHECK(mm.mm_features[0].audio_data->n_mels == cfg.n_mels);        // 80
  CHECK(mm.mm_features[0].audio_data->n_frames == n_frames);        // 3000
  // Expanded prompt: 3 real tokens + 1500 placeholder copies.
  CHECK(mm.prompt_token_ids.size() ==
        static_cast<size_t>(3 + num_tokens));
  CHECK(mm.mm_features[0].mm_hash == manifest.at("mm_hash").get<std::string>());
}

// ---------------------------------------------------------------------------
// 4. ROUTE: image_url content part (data: URI of the raw-RGB fixture) ->
//    Qwen3-VL processor. Container-format decode (PNG->RGB) is the named residual.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm image: image_url part -> processor -> grid + expansion") {
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == static_cast<size_t>(H * W * 3));

  // A data: URI carrying the raw RGB bytes. `image/x-raw-rgb` documents that the
  // container-format decode (PNG/JPEG -> RGB) is the NAMED residual; the parse +
  // base64 + route seams are exact.
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "What is in this image?"}},
           {{"type", "image_url"}, {"image_url", {{"url", uri}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();

  REQUIRE(req.messages.size() == 1);
  const ChatMessage& msg = req.messages[0];
  REQUIRE(msg.content_parts.has_value());
  REQUIRE(msg.content_parts->size() == 2);
  CHECK((*msg.content_parts)[1].type == "image_url");
  CHECK(vllm::entrypoints::openai::HasMultiModalParts(msg));

  // DECODE: the data URI round-trips to the exact raw RGB bytes.
  const auto media =
      vllm::entrypoints::openai::DecodeImageUrlPart((*msg.content_parts)[1]);
  CHECK(media.media_type == "image/x-raw-rgb");
  REQUIRE(media.bytes.size() == rgb.size());
  REQUIRE(media.bytes == rgb);

  // ROUTE: the existing Qwen3-VL processor produces the grid + merged tokens.
  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const std::vector<int32_t> prompt_ids = {5, cfg.image_token_id, 6};
  const vllm::multimodal::MultiModalInputs mm =
      vllm::entrypoints::openai::RouteImageRgb(proc, media.bytes.data(), H, W,
                                               prompt_ids);

  const auto g = manifest.at("image_grid_thw").at("values");
  const int64_t merged =
      (g[0].get<int64_t>() * g[1].get<int64_t>() * g[2].get<int64_t>()) /
      (cfg.merge_size * cfg.merge_size);  // (1*28*28)/4 = 196
  REQUIRE(mm.mm_features.size() == 1);
  CHECK(mm.mm_features[0].modality == "image");
  CHECK(mm.mm_features[0].offset == 1);          // placeholder position
  CHECK(mm.mm_features[0].length == merged);     // 196
  REQUIRE(mm.mm_features[0].data != nullptr);
  CHECK(mm.mm_features[0].data->image_grid_thw[0] == g[0].get<int64_t>());
  CHECK(mm.mm_features[0].data->image_grid_thw[1] == g[1].get<int64_t>());
  CHECK(mm.mm_features[0].data->image_grid_thw[2] == g[2].get<int64_t>());
  // Expanded prompt: 2 real tokens + 196 placeholder copies.
  CHECK(mm.prompt_token_ids.size() == static_cast<size_t>(2 + merged));
}

// ---------------------------------------------------------------------------
// 5. PLACEHOLDER STRINGS (MM-SERVE-ENGINE): the chat-template markers mirror
//    vLLM get_placeholder_str (qwen3_vl.py:1714 / qwen2_audio.py:333).
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm placeholder strings mirror vLLM get_placeholder_str") {
  using vllm::entrypoints::openai::AudioPlaceholderString;
  using vllm::entrypoints::openai::CollectChatPlaceholders;
  using vllm::entrypoints::openai::ImagePlaceholderString;
  using vllm::entrypoints::openai::VideoPlaceholderString;

  CHECK(ImagePlaceholderString() ==
        "<|vision_start|><|image_pad|><|vision_end|>");
  CHECK(VideoPlaceholderString() ==
        "<|vision_start|><|video_pad|><|vision_end|>");
  CHECK(AudioPlaceholderString(1) ==
        "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>");

  // CollectChatPlaceholders: one marker per mm part, audios numbered 1..k,
  // text parts contribute none.
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "look:"}},
           {{"type", "image_url"}, {"image_url", {{"url", "data:x"}}}},
           {{"type", "input_audio"},
            {"input_audio", {{"data", "AA=="}, {"format", "wav"}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();
  const std::vector<std::string> markers =
      CollectChatPlaceholders(req.messages[0]);
  REQUIRE(markers.size() == 2);
  CHECK(markers[0] == "<|vision_start|><|image_pad|><|vision_end|>");
  CHECK(markers[1] == "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>");

  // A bare-string message contributes no markers (byte-identical text path).
  ChatMessage bare;
  bare.role = "user";
  bare.content = "hello";
  CHECK(CollectChatPlaceholders(bare).empty());
}

// ---------------------------------------------------------------------------
// 6. FULL CHAIN (MM-SERVE-ENGINE, the RED-first "New" gate): parse an image_url
//    request -> route through the processor -> placeholder-EXPANDED prompt ->
//    the engine mm path (InputProcessor::process_inputs_mm) receives the
//    MultiModalInputs -> the built Request carries the mm handles + the expanded
//    prompt with the CORRECT feature count. Everything UP TO the mm forward (the
//    GPU consumer = MM-SERVE-E2E residual) is asserted on CPU.
// ---------------------------------------------------------------------------
TEST_CASE("chat-mm full chain: image request -> engine request carries mm") {
  const std::string dir = ImgFixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ImageConfigFromManifest(manifest);
  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");

  // Parse an OpenAI chat request with an image_url part.
  const std::string uri = "data:image/x-raw-rgb;base64," + EncodeBase64(rgb);
  const nlohmann::json j = {
      {"messages",
       {{{"role", "user"},
         {"content",
          {{{"type", "text"}, {"text", "What is in this image?"}},
           {{"type", "image_url"}, {"image_url", {{"url", uri}}}}}}}}}};
  const ChatCompletionRequest req = j.get<ChatCompletionRequest>();
  const ChatMessage& msg = req.messages[0];
  REQUIRE(vllm::entrypoints::openai::HasMultiModalParts(msg));

  // ROUTE (brick 1): decode + processor -> placeholder-EXPANDED MultiModalInputs.
  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const auto media =
      vllm::entrypoints::openai::DecodeImageUrlPart((*msg.content_parts)[1]);
  const std::vector<int32_t> base_prompt_ids = {5, cfg.image_token_id, 6};
  const vllm::multimodal::MultiModalInputs mm =
      vllm::entrypoints::openai::RouteImageRgb(proc, media.bytes.data(), H, W,
                                               base_prompt_ids);

  const auto g = manifest.at("image_grid_thw").at("values");
  const int64_t merged =
      (g[0].get<int64_t>() * g[1].get<int64_t>() * g[2].get<int64_t>()) /
      (cfg.merge_size * cfg.merge_size);  // 196
  // The placeholder-inserted prompt has EXACTLY the processor feature count of
  // image_pad slots (the count == the grid/feature count, MM-SERVE item 2).
  int64_t pad_slots = 0;
  for (int32_t id : mm.prompt_token_ids) {
    if (id == cfg.image_token_id) ++pad_slots;
  }
  CHECK(pad_slots == merged);
  CHECK(mm.mm_features.size() == 1);
  CHECK(mm.mm_features[0].length == merged);

  // ENGINE (MM-SERVE-ENGINE): process_inputs_mm is the exact call the engine mm
  // add_request overload makes. Feed the routed MultiModalInputs through it.
  const vllm::HfConfig hf = [] {
    vllm::HfConfig c;
    c.max_position_embeddings = 4096;
    c.raw = nlohmann::json::object();
    return c;
  }();
  static const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(
      std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36/tokenizer.json");
  vllm::v1::InputProcessor input_proc(tok, hf);

  vllm::SamplingParams params;
  vllm::v1::EngineCoreRequest core_req = input_proc.process_inputs_mm(
      "chatcmpl-0", mm.prompt_token_ids, mm.mm_features, params);

  // The engine request carries BOTH the expanded prompt AND the mm handles.
  CHECK(core_req.prompt_token_ids == mm.prompt_token_ids);
  REQUIRE(core_req.mm_features.size() == 1);
  CHECK(core_req.mm_features[0].modality == "image");
  CHECK(core_req.mm_features[0].length == merged);
  CHECK(core_req.mm_features[0].mm_hash ==
        manifest.at("mm_hash").get<std::string>());
  REQUIRE(core_req.mm_features[0].data != nullptr);

  // ...and the built Request (what the scheduler/encoder-cache consume).
  vllm::v1::Request built = vllm::v1::Request::FromEngineCoreRequest(core_req);
  REQUIRE(built.mm_features.size() == 1);
  CHECK(built.mm_features[0].length == merged);
  CHECK(built.prompt_token_ids == mm.prompt_token_ids);
  // E2E residual (MM-SERVE-E2E): the mm forward on the GPU worker consuming
  // built.mm_features to produce token-correct output on Qwen3-VL-4B.
}
