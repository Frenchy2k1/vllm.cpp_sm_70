// Laguna-S-2.1 (`LagunaForCausalLM` / `model_type=laguna`) W1/W2 SCAFFOLDING
// gate. Proves the things this pass can prove WITHOUT the 73 GB checkpoint or a
// GPU:
//   (1) the arch RESOLVES through the registry (the additive TU registered it);
//   (2) the config DESCENDS: ParseLagunaParams reads the shipped
//       poolside/Laguna-S-2.1/config.json scalars — including the NESTED
//       per-layer-type rope_parameters (the OLMo-3 KeyError hazard) and the
//       per-layer VARIABLE Q-head array — and validates the invariants;
//   (3) the loaders + forward LOUDLY report the pending brick (throw), never a
//       silent wrong answer.
// The forward + loader materialization + strict dual-oracle gate are NAMED W3/W4
// residuals (see .agents/specs/laguna-s21-w1w2-2026-07-30.md); nothing here
// claims the model runs.
#include "vllm/model_executor/models/laguna.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <vector>

using vllm::HfConfig;
using vllm::LagunaParams;
using vllm::ModelRegistry;
using vllm::ParseLagunaConfig;
using vllm::ParseLagunaParams;

namespace {
// The shipped poolside/Laguna-S-2.1/config.json (VERIFIED 2026-07-30, scope spec
// §1), reduced to the scalars the parse consumes. Typed HfConfig fields are set
// from the same values; the laguna-specific keys live in `raw`.
HfConfig RealConfig() {
  HfConfig c;
  c.architectures = {"LagunaForCausalLM"};
  c.model_type = "laguna";
  c.hidden_size = 3072;
  c.num_hidden_layers = 48;
  c.vocab_size = 100352;
  c.num_attention_heads = 48;
  c.num_key_value_heads = 8;
  c.head_dim = 128;
  c.intermediate_size = 12288;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 1048576;
  c.sliding_window = 512;
  c.num_experts = 256;
  c.num_experts_per_tok = 10;
  c.moe_intermediate_size = 1024;

  // Interleaved 1:3 global:sliding layer_types + per-layer Q-head counts.
  nlohmann::json layer_types = nlohmann::json::array();
  nlohmann::json heads_per_layer = nlohmann::json::array();
  for (int l = 0; l < 48; ++l) {
    const bool global = (l % 4 == 0);
    layer_types.push_back(global ? "full_attention" : "sliding_attention");
    heads_per_layer.push_back(global ? 48 : 72);
    c.layer_types.push_back(global ? "full_attention" : "sliding_attention");
  }

  c.raw = {
      {"hidden_size", 3072},
      {"num_hidden_layers", 48},
      {"vocab_size", 100352},
      {"num_attention_heads", 48},
      {"num_key_value_heads", 8},
      {"head_dim", 128},
      {"intermediate_size", 12288},
      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 1048576},
      {"tie_word_embeddings", false},
      {"sliding_window", 512},
      {"gating", "per-head"},
      {"layer_types", layer_types},
      {"num_attention_heads_per_layer", heads_per_layer},
      {"num_experts", 256},
      {"num_experts_per_tok", 10},
      {"moe_intermediate_size", 1024},
      {"shared_expert_intermediate_size", 1024},
      {"norm_topk_prob", true},
      {"moe_routed_scaling_factor", 2.5},
      {"mlp_only_layers", nlohmann::json::array({0})},
      {"rope_parameters",
       {{"full_attention",
         {{"rope_type", "yarn"},
          {"rope_theta", 500000.0},
          {"factor", 128.0},
          {"original_max_position_embeddings", 8192},
          {"beta_slow", 1.0},
          {"beta_fast", 32.0},
          {"attention_factor", 1.4852030263919618},
          {"partial_rotary_factor", 0.5}}},
        {"sliding_attention",
         {{"rope_type", "default"},
          {"rope_theta", 10000.0},
          {"partial_rotary_factor", 1.0}}}}},
  };
  return c;
}
}  // namespace

TEST_CASE("laguna scaffold: LagunaForCausalLM RESOLVES through the registry") {
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    for (std::string_view s : supported)
      if (s == a) return true;
    return false;
  };
  CHECK(has("LagunaForCausalLM"));

  const std::vector<std::string> archs{"LagunaForCausalLM"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(archs);
  CHECK(reg.architecture == "LagunaForCausalLM");
  CHECK(reg.info.is_text_generation_model);
  CHECK_FALSE(reg.info.is_hybrid);
  CHECK_FALSE(reg.info.supports_multimodal);
}

TEST_CASE("laguna scaffold: config DESCENDS (arch + nested dual-rope + variable Q-head)") {
  const HfConfig c = RealConfig();
  const LagunaParams p = ParseLagunaParams(c);

  // Core geometry.
  CHECK(p.hidden_size == 3072);
  CHECK(p.num_hidden_layers == 48);
  CHECK(p.vocab_size == 100352);
  CHECK(p.num_attention_heads == 48);
  CHECK(p.num_key_value_heads == 8);
  CHECK(p.head_dim == 128);
  CHECK(p.intermediate_size == 12288);
  CHECK_FALSE(p.tie_word_embeddings);

  // Interleaved attention: 12 global + 36 sliding (1:3), window 512.
  CHECK(p.sliding_window == 512);
  int globals = 0;
  for (int l = 0; l < 48; ++l)
    if (p.IsGlobalLayer(l)) ++globals;
  CHECK(globals == 12);
  CHECK(p.IsGlobalLayer(0));
  CHECK_FALSE(p.IsGlobalLayer(1));

  // Per-layer VARIABLE Q-head count (48 global / 72 sliding) — the riskiest bit.
  REQUIRE(static_cast<int>(p.num_attention_heads_per_layer.size()) == 48);
  CHECK(p.num_attention_heads_per_layer[0] == 48);
  CHECK(p.num_attention_heads_per_layer[1] == 72);

  // MoE: sigmoid noaux_tc UNGROUPED + shared expert + routed_scaling 2.5.
  CHECK(p.num_experts == 256);
  CHECK(p.num_experts_per_tok == 10);
  CHECK(p.moe_intermediate_size == 1024);
  CHECK(p.shared_expert_intermediate_size == 1024);
  CHECK(p.norm_topk_prob);
  CHECK(p.moe_routed_scaling_factor == doctest::Approx(2.5));
  CHECK_FALSE(p.use_grouped_topk);       // the NEW ungrouped variant
  CHECK(p.has_e_score_correction_bias);
  // layer 0 dense (mlp_only_layers=[0]); layer 1 MoE.
  CHECK(p.IsDenseLayer(0));
  CHECK_FALSE(p.IsDenseLayer(1));

  // Per-head softplus output gate.
  CHECK(p.per_head_output_gate);

  // Dual per-layer RoPE: full-attn YaRN (partial 0.5 -> rotary_dim 64, mscale
  // 1.4852) + sliding plain (full 128-dim).
  CHECK(p.rope_theta_full == doctest::Approx(500000.0));
  CHECK(p.yarn_factor == doctest::Approx(128.0));
  CHECK(p.yarn_orig_max_pos == 8192);
  CHECK(p.yarn_attention_factor == doctest::Approx(1.4852030263919618));
  CHECK(p.rotary_dim_full == 64);
  CHECK(p.rope_theta_sliding == doctest::Approx(10000.0));
  CHECK(p.rotary_dim_sliding == 128);

  // The config hook itself must not throw (config-constructs check).
  CHECK_NOTHROW(ParseLagunaConfig(c));
}

TEST_CASE("laguna scaffold: forward + loaders LOUDLY report the W3/W4 residual") {
  const HfConfig c = RealConfig();
  // The GGUF/safetensors materialization is a W3 residual — must throw, not
  // silently return an empty/wrong model.
  const std::vector<vllm::SafetensorsFile> no_shards;
  CHECK_THROWS(vllm::LoadLagunaForCausalLMWeights(no_shards, c));
}
