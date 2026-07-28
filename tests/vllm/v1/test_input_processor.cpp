// Tests for the V1 InputProcessor text path
// (vllm/v1/engine/input_processor.py::process_inputs @ e24d1b24).
//
// M1.8 Task 2 is a pure host-logic port (no goldens). The InputProcessor is the
// "constructing unit" that runs the SamplingParams __post_init__ (PostInit) our
// M1.1 SamplingParams deferred. Cases:
//   (a) text prompt   -> EngineCoreRequest.prompt_token_ids == tokenizer.Encode
//                        (the byte-level BPE path); request_id + arrival_time set.
//   (b) PostInit ran  -> a near-zero temperature is clamped to _MAX_TEMP; a
//                        greedy (temperature 0) params has its top_p/top_k/min_p
//                        normalized. (Verify() alone would not normalize these.)
//   (c) invalid params-> Verify (run inside PostInit) throws on an out-of-range
//                        field (e.g. top_p > 1).
//   (d) max_tokens    -> unset (nullopt) defaults to max_model_len - prompt_len.
//   (e) eos/stop      -> primary eos is written to sampling_params.eos_token_id
//                        unless ignore_eos; a secondary eos id (a list config)
//                        is merged into stop_token_ids unless ignore_eos.
#include <doctest/doctest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <memory>

#include <nlohmann/json.hpp>

#include "vllm/multimodal/inputs.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/request.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::SamplingParams;
using vllm::SamplingType;
using vllm::multimodal::AudioKwargs;
using vllm::multimodal::ImageKwargs;
using vllm::multimodal::MultiModalFeatureSpec;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCoreRequest;
using vllm::v1::InputProcessor;
using vllm::v1::Request;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36";

// One tokenizer load for the whole binary (the golden tokenizer.json is ~20MB).
const Tokenizer& GoldenTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJson(kGoldenDir + "/tokenizer.json");
  return tok;
}

// Builds a minimal HfConfig carrying just what the InputProcessor reads:
// max_position_embeddings (-> max_model_len) and raw["eos_token_id"].
HfConfig MakeConfig(int64_t max_len, const json& eos_token_id) {
  HfConfig cfg;
  cfg.model_type = "qwen3_5_moe";
  cfg.hidden_size = 8;
  cfg.num_hidden_layers = 1;
  cfg.max_position_embeddings = max_len;
  cfg.raw = json::object();
  if (!eos_token_id.is_null()) {
    cfg.raw["eos_token_id"] = eos_token_id;
  }
  return cfg;
}

}  // namespace

TEST_CASE("process_inputs tokenizes the text prompt (byte-level BPE path)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(/*max_len=*/4096, /*eos=*/json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt = "The capital of Germany is";
  SamplingParams params;  // defaults: max_tokens=16
  EngineCoreRequest req = proc.process_inputs("req-1", prompt, params,
                                              /*arrival_time=*/123.0);

  CHECK(req.request_id == "req-1");
  CHECK(req.arrival_time == doctest::Approx(123.0));
  CHECK(req.prompt_token_ids == tok.Encode(prompt));
  CHECK_FALSE(req.prompt_token_ids.empty());
  // Default max_tokens is preserved when the caller sets it.
  REQUIRE(req.sampling_params.max_tokens.has_value());
  CHECK(*req.sampling_params.max_tokens == 16);
}

TEST_CASE("process_inputs runs SamplingParams PostInit (normalization)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SUBCASE("near-zero temperature is clamped up to _MAX_TEMP") {
    SamplingParams params;
    params.temperature = 0.005;  // in (0, _MAX_TEMP=1e-2)
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK(req.sampling_params.temperature == doctest::Approx(vllm::kMaxTemp));
  }

  SUBCASE("greedy (temperature 0) normalizes top_p/top_k/min_p") {
    SamplingParams params;
    params.temperature = 0.0;
    params.top_p = 0.5;
    params.top_k = 20;
    params.min_p = 0.3;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK(req.sampling_params.temperature == doctest::Approx(0.0));
    CHECK(req.sampling_params.top_p == doctest::Approx(1.0));
    CHECK(req.sampling_params.top_k == 0);
    CHECK(req.sampling_params.min_p == doctest::Approx(0.0));
    CHECK(req.sampling_params.Type() == SamplingType::kGreedy);
  }
}

TEST_CASE("process_inputs throws on invalid SamplingParams (Verify)") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams bad;
  bad.top_p = 2.0;  // must be in (0, 1]
  CHECK_THROWS_AS(proc.process_inputs("r", "hi", bad), std::runtime_error);

  SamplingParams greedy_multi;
  greedy_multi.temperature = 0.0;
  greedy_multi.n = 2;  // n must be 1 under greedy sampling
  CHECK_THROWS_AS(proc.process_inputs("r", "hi", greedy_multi),
                  std::runtime_error);
}

TEST_CASE("process_inputs defaults max_tokens from max_model_len - prompt_len") {
  const Tokenizer& tok = GoldenTokenizer();
  const int64_t kMaxLen = 4096;
  HfConfig cfg = MakeConfig(kMaxLen, json(nullptr));
  InputProcessor proc(tok, cfg);

  const std::string prompt = "The capital of Germany is";
  const int prompt_len = static_cast<int>(tok.Encode(prompt).size());

  SamplingParams params;
  params.max_tokens = std::nullopt;  // "unset" -> generate up to max_model_len
  EngineCoreRequest req = proc.process_inputs("r", prompt, params);

  REQUIRE(req.sampling_params.max_tokens.has_value());
  CHECK(*req.sampling_params.max_tokens ==
        static_cast<int>(kMaxLen) - prompt_len);
}

TEST_CASE("process_inputs wires eos/stop token ids") {
  const Tokenizer& tok = GoldenTokenizer();

  SUBCASE("scalar eos is written to sampling_params.eos_token_id") {
    HfConfig cfg = MakeConfig(4096, json(151643));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    CHECK(*req.sampling_params.eos_token_id == 151643);
    // A single eos id has no secondary ids -> stop_token_ids untouched.
    CHECK(req.sampling_params.stop_token_ids.empty());
  }

  SUBCASE("ignore_eos leaves eos_token_id unset and stop_token_ids untouched") {
    HfConfig cfg = MakeConfig(4096, json(151643));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    CHECK(req.sampling_params.stop_token_ids.empty());
  }

  SUBCASE("list eos: primary is eos_token_id, secondary merged into stop ids") {
    HfConfig cfg = MakeConfig(4096, json::array({100, 200, 300}));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.stop_token_ids = {200, 999};  // 200 dedups against the eos ids
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    REQUIRE(req.sampling_params.eos_token_id.has_value());
    CHECK(*req.sampling_params.eos_token_id == 100);  // first element = primary
    // secondary eos ids {200, 300} unioned with existing {200, 999}.
    std::vector<int32_t> expected = {200, 300, 999};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }

  SUBCASE("list eos with ignore_eos does not touch stop_token_ids") {
    HfConfig cfg = MakeConfig(4096, json::array({100, 200, 300}));
    InputProcessor proc(tok, cfg);
    SamplingParams params;
    params.ignore_eos = true;
    params.stop_token_ids = {999};
    EngineCoreRequest req = proc.process_inputs("r", "hi", params);
    CHECK_FALSE(req.sampling_params.eos_token_id.has_value());
    std::vector<int32_t> expected = {999};
    CHECK(req.sampling_params.stop_token_ids == expected);
  }
}

// ─── ROAD-V1-C7 SAMPLE-LOGIT-FILTERS: bad_words + all_stop_token_ids wiring ───
TEST_CASE("process_inputs tokenizes bad_words into bad_words_token_ids") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  params.bad_words = {"hello"};
  EngineCoreRequest req = proc.process_inputs("r", "hi", params);

  // update_from_tokenizer produces the per-word n-grams (sampling_params.py:659).
  REQUIRE(req.sampling_params.bad_words_token_ids.has_value());
  const auto& bwti = *req.sampling_params.bad_words_token_ids;
  REQUIRE_FALSE(bwti.empty());
  // The no-prefix variant is the encode of the lstripped word.
  CHECK(bwti[0] == tok.Encode("hello"));
  // Every produced id is in range.
  const int32_t max_id = tok.VocabSize() - 1;
  for (const auto& ngram : bwti) {
    for (int32_t id : ngram) {
      CHECK(id >= 0);
      CHECK(id <= max_id);
    }
  }
}

TEST_CASE("process_inputs seeds all_stop_token_ids (stop_token_ids + eos)") {
  const Tokenizer& tok = GoldenTokenizer();
  // A primary eos id via the tokenizer fallback is used when config eos is null;
  // pass an explicit eos list to make the assertion deterministic.
  HfConfig cfg = MakeConfig(4096, json({1000, 1001}));
  InputProcessor proc(tok, cfg);

  SamplingParams params;
  params.stop_token_ids = {55};
  EngineCoreRequest req = proc.process_inputs("r", "hi", params);

  // all_stop_token_ids = stop_token_ids (PostInit) + eos ids (generation config).
  const auto& all = req.sampling_params.all_stop_token_ids;
  CHECK(all.count(55) == 1);
  CHECK(all.count(1000) == 1);
  CHECK(all.count(1001) == 1);
}

TEST_CASE("process_inputs: empty bad_words leaves bad_words_token_ids unset") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(nullptr));
  InputProcessor proc(tok, cfg);
  SamplingParams params;  // no bad_words
  EngineCoreRequest req = proc.process_inputs("r", "hi", params);
  CHECK_FALSE(req.sampling_params.bad_words_token_ids.has_value());
}

// ─── ROAD-V1-MM MM-SERVE-ENGINE: process_inputs_mm carries mm_features ───────
//
// The mm request path (input_processor.py:333-379): the prompt ids are the
// ALREADY placeholder-expanded stream (the serving-side processor ran) and the
// per-item mm_features ride onto the EngineCoreRequest alongside them. This is
// the CPU-verifiable half — everything UP TO the mm forward (which consumes
// mm_features on the GPU worker, MM-SERVE-E2E). RED-first: before this brick the
// engine had NO mm add_request path (process_inputs_tokens drops mm data);
// after, the request provably carries the mm handles + the expanded prompt.
TEST_CASE("process_inputs_mm carries mm_features + the expanded prompt") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(151643));
  InputProcessor proc(tok, cfg);

  // A placeholder-EXPANDED prompt: 2 real ids + 4 image_pad slots + 1 real id.
  const int32_t kImagePad = 151655;
  std::vector<int32_t> expanded = {5, 6, kImagePad, kImagePad,
                                   kImagePad, kImagePad, 7};

  // One image feature spec (the RouteImageRgb output shape): span [2,4] + an
  // opaque ImageKwargs handle + the mm-hash.
  MultiModalFeatureSpec spec;
  spec.mm_hash = "deadbeefcafe";
  spec.modality = "image";
  spec.offset = 2;
  spec.length = 4;
  auto kw = std::make_shared<ImageKwargs>();
  kw->num_patches = 16;
  kw->patch_feature_dim = 8;
  kw->image_grid_thw = {1, 8, 8};
  spec.data = kw;
  std::vector<MultiModalFeatureSpec> mm_features = {spec};

  SamplingParams params;  // default max_tokens=16
  EngineCoreRequest req = proc.process_inputs_mm(
      "mm-1", expanded, mm_features, params, /*arrival_time=*/42.0);

  // The EXPANDED prompt ids pass through verbatim (no tokenization).
  CHECK(req.request_id == "mm-1");
  CHECK(req.arrival_time == doctest::Approx(42.0));
  CHECK(req.prompt_token_ids == expanded);

  // The mm_features are carried onto the request (the handles + span + hash).
  REQUIRE(req.mm_features.size() == 1);
  CHECK(req.mm_features[0].modality == "image");
  CHECK(req.mm_features[0].mm_hash == "deadbeefcafe");
  CHECK(req.mm_features[0].offset == 2);
  CHECK(req.mm_features[0].length == 4);
  REQUIRE(req.mm_features[0].data != nullptr);
  CHECK(req.mm_features[0].data->num_patches == 16);
  CHECK(req.mm_features[0].data->image_grid_thw[1] == 8);

  // The eos/stop wiring is byte-for-byte the tokens path (scalar eos written).
  REQUIRE(req.sampling_params.eos_token_id.has_value());
  CHECK(*req.sampling_params.eos_token_id == 151643);

  // Request::FromEngineCoreRequest threads mm_features onto the built Request —
  // the exact handoff LLMEngine/AsyncLLM::add_request(MultiModalInputs) perform
  // before enqueueing to the EngineCore (the encoder-cache / vision consumer).
  Request built = Request::FromEngineCoreRequest(req);
  REQUIRE(built.mm_features.size() == 1);
  CHECK(built.mm_features[0].mm_hash == "deadbeefcafe");
  CHECK(built.mm_features[0].length == 4);
  CHECK(built.prompt_token_ids == expanded);
}

TEST_CASE("process_inputs_mm with EMPTY mm_features == the tokens path") {
  const Tokenizer& tok = GoldenTokenizer();
  HfConfig cfg = MakeConfig(4096, json(151643));
  InputProcessor proc(tok, cfg);

  std::vector<int32_t> ids = {10, 11, 12, 13};
  SamplingParams params;
  EngineCoreRequest mm_req = proc.process_inputs_mm(
      "r", ids, /*mm_features=*/{}, params, /*arrival_time=*/7.0);
  EngineCoreRequest tok_req =
      proc.process_inputs_tokens("r", ids, params, /*arrival_time=*/7.0);

  // INERTNESS: an empty mm_features request is identical to the tokens request
  // (same ids, same eos/max_tokens wiring) with an empty mm_features vector.
  CHECK(mm_req.mm_features.empty());
  CHECK(mm_req.prompt_token_ids == tok_req.prompt_token_ids);
  CHECK(mm_req.sampling_params.max_tokens == tok_req.sampling_params.max_tokens);
  CHECK(mm_req.sampling_params.eos_token_id ==
        tok_req.sampling_params.eos_token_id);
}
