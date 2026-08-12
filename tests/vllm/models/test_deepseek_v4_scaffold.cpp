// DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`) W1/W2 SCAFFOLDING gate. Proves the
// two things this pass can prove WITHOUT a checkpoint or a GPU:
//   (1) the arch RESOLVES through the registry (the additive TU registered it), and
//   (2) the config DESCENDS: ParseDeepseekV4Params reads the shipped
//       nvidia/DeepSeek-V4-Flash-NVFP4 config.json scalars and validates them.
// The forward + loader materialization + strict gate are NAMED W3-W8 residuals
// (see .agents/specs/deepseek-v4-flash.md §5); nothing here claims the model runs.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_probe.h"
#include "vllm/model_executor/models/model_registry.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using vllm::DeepseekV4Params;
using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ParseDeepseekV4Params;

namespace {
// The shipped nvidia/DeepSeek-V4-Flash-NVFP4 config.json (VERIFIED 2026-07-28),
// reduced to the scalars the parse consumes. Typed HfConfig fields are set from
// the same values; the V4-specific keys live in `raw`.
HfConfig RealConfig() {
  HfConfig c;
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = 4096;
  c.num_hidden_layers = 43;
  c.vocab_size = 129280;
  c.num_attention_heads = 64;
  c.num_key_value_heads = 1;
  c.head_dim = 512;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 1048576;
  nlohmann::json cr = nlohmann::json::array();
  for (int i = 0; i < 44; ++i) {  // 44-entry array (last entry is the MTP layer)
    if (i == 0 || i == 1 || i == 43)
      cr.push_back(0);
    else
      cr.push_back((i % 2 == 0) ? 4 : 128);
  }
  c.raw = {
      {"hidden_size", 4096},        {"num_hidden_layers", 43},
      {"vocab_size", 129280},       {"num_attention_heads", 64},
      {"num_key_value_heads", 1},   {"head_dim", 512},
      {"qk_rope_head_dim", 64},     {"q_lora_rank", 1024},
      {"o_lora_rank", 1024},        {"o_groups", 8},
      {"sliding_window", 128},      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 1048576},
      {"num_nextn_predict_layers", 1},
      {"n_routed_experts", 256},    {"num_experts_per_tok", 6},
      {"moe_intermediate_size", 2048}, {"n_shared_experts", 1},
      {"norm_topk_prob", true},     {"routed_scaling_factor", 1.5},
      {"swiglu_limit", 10.0},       {"scoring_func", "sqrtsoftplus"},
      {"topk_method", "noaux_tc"},  {"num_hash_layers", 3},
      {"expert_dtype", "fp4"},      {"hc_mult", 4},
      {"hc_sinkhorn_iters", 20},    {"hc_eps", 1e-6},
      {"index_head_dim", 128},      {"index_n_heads", 64},
      {"index_topk", 512},          {"compress_rope_theta", 160000},
      {"rope_theta", 10000},        {"tie_word_embeddings", false},
      {"compress_ratios", cr},
  };
  return c;
}
}  // namespace

TEST_CASE("deepseek-v4 scaffold: DeepseekV4ForCausalLM RESOLVES through the registry") {
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    for (std::string_view s : supported)
      if (s == a) return true;
    return false;
  };
  CHECK(has("DeepseekV4ForCausalLM"));

  HfConfig cfg;
  cfg.architectures = {"DeepseekV4ForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  CHECK(reg.architecture == "DeepseekV4ForCausalLM");
  CHECK(reg.info.is_text_generation_model);
  CHECK_FALSE(reg.info.supports_multimodal);
}

TEST_CASE("deepseek-v4 expert probe input stays in the float domain") {
  for (const float frequency : {0.017f, 0.013f}) {
    const std::vector<float> actual =
        vllm::detail::DeepseekV4ExpertProbeInput(10, frequency);
    REQUIRE(actual.size() == 10);
    for (int64_t i = 0; i < static_cast<int64_t>(actual.size()); ++i) {
      const float expected =
          0.5f * std::sin(frequency * static_cast<float>(i + 1));
      CAPTURE(frequency);
      CAPTURE(i);
      CHECK(actual[static_cast<size_t>(i)] == expected);
    }
  }
}

TEST_CASE("deepseek-v4 scaffold: config DESCENDS (ParseDeepseekV4Params)") {
  const DeepseekV4Params p = ParseDeepseekV4Params(RealConfig());
  // shared geometry
  CHECK(p.hidden_size == 4096);
  CHECK(p.num_hidden_layers == 43);
  CHECK(p.vocab_size == 129280);
  CHECK(p.num_attention_heads == 64);
  CHECK(p.num_key_value_heads == 1);
  CHECK(p.num_nextn_predict_layers == 1);
  // 512-wide MLA (NEW geometry)
  CHECK(p.head_dim == 512);
  CHECK(p.qk_rope_head_dim == 64);
  CHECK(p.q_lora_rank == 1024);
  CHECK(p.o_lora_rank == 1024);
  CHECK(p.o_groups == 8);
  CHECK(p.sliding_window == 128);
  // MoE
  CHECK(p.n_routed_experts == 256);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 2048);
  CHECK(p.n_shared_experts == 1);
  CHECK(p.num_hash_layers == 3);
  CHECK(p.scoring_func == "sqrtsoftplus");
  CHECK(p.expert_dtype == "fp4");
  CHECK(p.swiglu_limit == doctest::Approx(10.0));
  CHECK(p.routed_scaling_factor == doctest::Approx(1.5));
  // MHC
  CHECK(p.hc_mult == 4);
  CHECK(p.hc_sinkhorn_iters == 20);
  // DSA
  CHECK(p.index_head_dim == 128);
  CHECK(p.index_n_heads == 64);
  CHECK(p.index_topk == 512);
  CHECK(static_cast<int>(p.compress_ratios.size()) == 44);
}

TEST_CASE("deepseek-v4 scaffold: per-layer topology matches the verified schema") {
  const DeepseekV4Params p = ParseDeepseekV4Params(RealConfig());
  // Hash-routed: exactly layers 0,1,2 (num_hash_layers=3).
  CHECK(p.is_hash_layer(0));
  CHECK(p.is_hash_layer(2));
  CHECK_FALSE(p.is_hash_layer(3));
  // Compressor present on layers 2..42 (compress_ratio != 0) == 41 layers;
  // layers 0,1 have ratio 0. Indexer (ratio == 4) on 21 layers.
  int compressor = 0, indexer = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    if (p.has_compressor(l)) ++compressor;
    if (p.has_indexer(l)) ++indexer;
  }
  CHECK(compressor == 41);
  CHECK(indexer == 21);
  CHECK_FALSE(p.has_compressor(0));
  CHECK_FALSE(p.has_compressor(1));
}

TEST_CASE("deepseek-v4 scaffold: parse REJECTS an unrepresentable config") {
  HfConfig bad = RealConfig();
  bad.head_dim = 128;             // not the 512-wide MLA geometry
  bad.raw["head_dim"] = 128;
  CHECK_THROWS_AS(ParseDeepseekV4Params(bad), std::runtime_error);

  HfConfig bad2 = RealConfig();
  bad2.raw["scoring_func"] = "sigmoid";  // V4 is sqrtsoftplus-only
  CHECK_THROWS_AS(ParseDeepseekV4Params(bad2), std::runtime_error);
}
