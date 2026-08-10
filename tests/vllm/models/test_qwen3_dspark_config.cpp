// vllm.cpp original. Speculators-format DSpark config translation (SPEC-DSPARK
// W3). Ported semantics @ 555967922:
//   * vllm/transformers_utils/configs/speculators/base.py:47-64
//     extract_transformers_pre_trained_config — START from
//     `transformer_layer_config`, then apply the algorithm updater;
//     :90-95 validate_speculators_config — `transformer_layer_config` REQUIRED;
//   * vllm/transformers_utils/configs/speculators/algos.py:133-178 update_dspark
//     — architectures, sample_from_anchor (default FALSE on this path), the aux
//     layer ids and their `i - 1` target_layer_ids, and the copied DSpark keys;
//   * base.py:113-136 build_vllm_speculative_config — k comes from
//     `speculators_config.proposal_methods[0].speculative_tokens`.
//
// The fixtures are the REAL published configs (fetched 2026-08-09):
// RedHatAI/Qwen3.6-35B-A3B-speculator.dspark (our 35B gate model's draft) and
// RedHatAI/gemma-4-31B-it-speculator.dspark, reduced to the keys the translation
// reads. Both are speculators format; the deepseek-ai drafts are native and need
// no translation at all.
//
// RED-first: TranslateSpeculatorsDsparkConfig did not exist, so this did not
// compile.
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_dspark.h"

using namespace vllm;

namespace {
// RedHatAI/Qwen3.6-35B-A3B-speculator.dspark, verbatim on the keys that matter.
nlohmann::json Qwen35BSpeculatorsConfig() {
  return nlohmann::json::parse(R"({
    "architectures": ["Qwen3DSparkModel"],
    "aux_hidden_state_layer_ids": [2, 10, 20, 30, 37],
    "block_size": 8,
    "confidence_head_with_markov": true,
    "draft_vocab_size": 32000,
    "enable_confidence_head": true,
    "markov_head_type": "vanilla",
    "markov_rank": 256,
    "mask_token_id": 248077,
    "sample_from_anchor": true,
    "sliding_window_non_causal": false,
    "speculators_config": {
      "algorithm": "dspark",
      "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 8}],
      "verifier": {"name_or_path": "Qwen/Qwen3.6-35B-A3B"}
    },
    "speculators_model_type": "dspark",
    "transformer_layer_config": {
      "head_dim": 256,
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "layer_types": ["sliding_attention", "sliding_attention",
                      "sliding_attention", "sliding_attention",
                      "sliding_attention"],
      "model_type": "qwen3",
      "num_attention_heads": 16,
      "num_hidden_layers": 5,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-06,
      "sliding_window": 2048,
      "use_sliding_window": true,
      "vocab_size": 248320
    }
  })");
}
}  // namespace

TEST_CASE("speculators translation starts from transformer_layer_config") {
  const nlohmann::json out =
      Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(Qwen35BSpeculatorsConfig());
  // The backbone fields are the transformer_layer_config's, NOT the outer doc's.
  CHECK(out.at("hidden_size").get<int64_t>() == 2048);
  CHECK(out.at("num_hidden_layers").get<int64_t>() == 5);
  CHECK(out.at("num_key_value_heads").get<int64_t>() == 2);
  CHECK(out.at("vocab_size").get<int64_t>() == 248320);
  CHECK(out.at("model_type").get<std::string>() == "qwen3");
  CHECK(out.at("sliding_window").get<int64_t>() == 2048);
  REQUIRE(out.contains("layer_types"));
  CHECK(out.at("layer_types").size() == 5);
}

TEST_CASE("speculators translation applies the DSpark algorithm update") {
  const nlohmann::json out =
      Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(Qwen35BSpeculatorsConfig());
  REQUIRE(out.contains("architectures"));
  CHECK(out.at("architectures").at(0).get<std::string>() == "Qwen3DSparkModel");
  // aux_hidden_state_layer_ids are carried through AND turned into the i-1
  // target_layer_ids the aux tap plumbing reads (algos.py:162-165).
  CHECK(out.at("eagle_aux_hidden_state_layer_ids") ==
        nlohmann::json::array({2, 10, 20, 30, 37}));
  CHECK(out.at("target_layer_ids") == nlohmann::json::array({1, 9, 19, 29, 36}));
  // The copied DSpark keys (algos.py:167-178).
  CHECK(out.at("draft_vocab_size").get<int64_t>() == 32000);
  CHECK(out.at("markov_rank").get<int64_t>() == 256);
  CHECK(out.at("mask_token_id").get<int64_t>() == 248077);
  CHECK(out.at("block_size").get<int64_t>() == 8);
  CHECK(out.at("sample_from_anchor").get<bool>() == true);
}

TEST_CASE("sample_from_anchor defaults to FALSE on the speculators path") {
  // algos.py:157-159 — the DFlash 1+N fill-in layout is the default here, even
  // though a native Qwen3 DSpark config's absent key means True at the
  // speculator (dspark/speculator.py:50-52 getattr(..., True)). The two defaults
  // genuinely differ; the gemma-4-31B speculator ships sample_from_anchor false.
  nlohmann::json doc = Qwen35BSpeculatorsConfig();
  doc.erase("sample_from_anchor");
  const nlohmann::json out = Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(doc);
  CHECK(out.at("sample_from_anchor").get<bool>() == false);
}

TEST_CASE("speculators k comes from the first proposal method") {
  CHECK(Qwen3DSparkModel::SpeculatorsNumSpeculativeTokens(Qwen35BSpeculatorsConfig()) == 8);
  nlohmann::json doc = Qwen35BSpeculatorsConfig();
  doc["speculators_config"]["proposal_methods"] = nlohmann::json::array();
  CHECK_THROWS_AS(Qwen3DSparkModel::SpeculatorsNumSpeculativeTokens(doc), std::exception);
}

TEST_CASE("a speculators config without transformer_layer_config is rejected") {
  nlohmann::json doc = Qwen35BSpeculatorsConfig();
  doc.erase("transformer_layer_config");
  CHECK_THROWS_AS(Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(doc), std::exception);
}

TEST_CASE("a non-dspark speculators config is rejected") {
  nlohmann::json doc = Qwen35BSpeculatorsConfig();
  doc["speculators_model_type"] = "eagle3";
  CHECK_THROWS_AS(Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(doc), std::exception);
}

TEST_CASE("aux_hidden_state_layer_ids is required by the DSpark update") {
  nlohmann::json doc = Qwen35BSpeculatorsConfig();
  doc.erase("aux_hidden_state_layer_ids");
  CHECK_THROWS_AS(Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(doc), std::exception);
}

TEST_CASE("IsSpeculatorsDsparkConfig discriminates the two checkpoint layouts") {
  CHECK(Qwen3DSparkModel::IsSpeculatorsDsparkConfig(Qwen35BSpeculatorsConfig()));
  // A native deepseek-ai/dspark_qwen3_4b_block7 config: flat, no speculators keys.
  const nlohmann::json native = nlohmann::json::parse(R"({
    "architectures": ["Qwen3DSparkModel"],
    "block_size": 7,
    "hidden_size": 2560,
    "markov_rank": 256,
    "mask_token_id": 151669,
    "model_type": "qwen3",
    "num_hidden_layers": 5,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "vocab_size": 151936
  })");
  CHECK_FALSE(Qwen3DSparkModel::IsSpeculatorsDsparkConfig(native));
}
