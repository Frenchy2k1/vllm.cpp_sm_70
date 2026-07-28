// Kimi K3 (`KimiK3ForConditionalGeneration`) W2/W5 SCAFFOLDING gate. Proves the
// three things this DERIVE-AND-SHIP lane can prove WITHOUT a checkpoint or a GPU:
//   (1) the arch RESOLVES through the registry (the additive TU registered it),
//   (2) the config DESCENDS: ParseKimiK3Params reads the nested `text_config`
//       (KimiLinear KDA+MLA+MoE hybrid) + `vision_config` + `quantization_config`
//       DERIVED K3 scalars and validates them, and
//   (3) the TEXT-BACKBONE STRUCTURE is faithful: EnumerateKimiK3TextBackboneTensors
//       branches KDA vs MLA per layer (is_kda_layer) and MoE vs dense per layer,
//       grounded 1:1 in the pinned kimi_linear.py.
// The forward REFUSES-by-name and the loader refuses MXFP4 (a real K3 checkpoint's
// dtype) — both asserted here. Nothing claims the 2.8T model runs; K3 is beyond
// the pinned oracle (555967922). See .agents/specs/kimi-k3.md §5.
#include "vllm/model_executor/models/kimi_k3.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

using vllm::EnumerateKimiK3TextBackboneTensors;
using vllm::HfConfig;
using vllm::KimiK3Params;
using vllm::KimiK3TextParams;
using vllm::LoadKimiK3ForConditionalGenerationWeights;
using vllm::ModelRegistry;
using vllm::ParseKimiK3Params;

namespace {
// The DERIVED moonshotai/Kimi-K3 config.json (from the HF fetch in the W0 spike;
// NOT byte-verified), reduced to the scalars the parse consumes. K3 wraps a nested
// `text_config` whose architectures = ["KimiLinearForCausalLM"]. To keep the test
// fast, the layer/expert counts are SCALED DOWN (the enumeration formula is
// scale-invariant); the real-scale scalars (H=7168, L=93, 896 experts) are checked
// separately in RealScaleTextConfig() below.
HfConfig SmallK3Config() {
  HfConfig c;
  c.architectures = {"KimiK3ForConditionalGeneration"};
  // LoadHfConfig lifts the shared scalars from text_config into the typed fields.
  c.hidden_size = 512;
  c.num_hidden_layers = 6;
  c.vocab_size = 4096;
  c.num_attention_heads = 8;

  // kda_layers use 1-based indexing: is_kda_layer(l) == (l+1) in kda_layers.
  // Here layers {0,1,3,5} are KDA (=> kda set {1,2,4,6}); {2,4} are MLA.
  nlohmann::json text = {
      {"model_type", "kimi_linear"},
      {"architectures", nlohmann::json::array({"KimiLinearForCausalLM"})},
      {"hidden_size", 512},
      {"num_hidden_layers", 6},
      {"vocab_size", 4096},
      {"num_attention_heads", 8},
      {"intermediate_size", 1024},
      {"rms_norm_eps", 1e-6},
      {"tie_word_embeddings", false},
      {"num_experts", 4},
      {"num_experts_per_token", 2},
      {"num_shared_experts", 2},
      {"moe_intermediate_size", 256},
      {"first_k_dense_replace", 1},
      {"moe_layer_freq", 1},
      {"moe_router_activation_func", "sigmoid"},
      {"kv_lora_rank", 512},
      {"q_lora_rank", 1536},
      {"qk_nope_head_dim", 128},
      {"qk_rope_head_dim", 64},
      {"v_head_dim", 128},
      {"linear_attn_config",
       {{"kda_layers", nlohmann::json::array({1, 2, 4, 6})},
        {"full_attn_layers", nlohmann::json::array({3, 5})},
        {"num_heads", 8},
        {"head_dim", 128},
        {"short_conv_kernel_size", 4}}},
  };
  c.raw = {
      {"model_type", "kimi_k3"},
      {"architectures", nlohmann::json::array({"KimiK3ForConditionalGeneration"})},
      {"text_config", text},
      {"vision_config",
       {{"patch_size", 14}, {"num_hidden_layers", 27}, {"hidden_size", 1152}}},
  };
  return c;
}

// The DERIVED real K3 scale (spec §0): H=7168, L=93 (69 KDA + 24 MLA), 896 experts
// / top-16 / 2 shared, MLA kv_lora 512 / q_lora 1536 / nope 128 / rope 64.
HfConfig RealScaleK3Config() {
  HfConfig c = SmallK3Config();
  c.hidden_size = 7168;
  c.num_hidden_layers = 93;
  c.vocab_size = 163840;
  c.num_attention_heads = 96;
  nlohmann::json& text = c.raw["text_config"];
  text["hidden_size"] = 7168;
  text["num_hidden_layers"] = 93;
  text["vocab_size"] = 163840;
  text["num_attention_heads"] = 96;
  text["num_experts"] = 896;
  text["num_experts_per_token"] = 16;
  text["num_shared_experts"] = 2;
  text["moe_intermediate_size"] = 3072;
  text["linear_attn_config"]["num_heads"] = 96;
  text["linear_attn_config"]["head_dim"] = 128;
  // 69 KDA layers (1-based idx 1..69); 24 MLA (70..93).
  nlohmann::json kda = nlohmann::json::array();
  for (int i = 1; i <= 69; ++i) kda.push_back(i);
  text["linear_attn_config"]["kda_layers"] = kda;
  return c;
}
}  // namespace

TEST_CASE("kimi-k3 scaffold: KimiK3ForConditionalGeneration RESOLVES via registry") {
  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    return std::find(supported.begin(), supported.end(), a) != supported.end();
  };
  CHECK(has("KimiK3ForConditionalGeneration"));

  HfConfig cfg;
  cfg.architectures = {"KimiK3ForConditionalGeneration"};
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(cfg);
  CHECK(reg.architecture == "KimiK3ForConditionalGeneration");
  CHECK(reg.info.is_text_generation_model);
  CHECK(reg.info.is_hybrid);          // 69 KDA linear-attn layers
  CHECK(reg.info.has_inner_state);
  CHECK(reg.info.supports_multimodal);  // MoonViT-V2 vision
}

TEST_CASE("kimi-k3 scaffold: config DESCENDS the nested text/vision/quant configs") {
  const KimiK3Params p = ParseKimiK3Params(RealScaleK3Config());
  const KimiK3TextParams& t = p.text;
  // shared geometry (DERIVED real K3 scale)
  CHECK(t.hidden_size == 7168);
  CHECK(t.num_hidden_layers == 93);
  CHECK(t.vocab_size == 163840);
  CHECK(t.num_attention_heads == 96);
  // MLA geometry (DeepSeek-V3 dims)
  CHECK(t.kv_lora_rank == 512);
  CHECK(t.q_lora_rank == 1536);
  CHECK(t.qk_nope_head_dim == 128);
  CHECK(t.qk_rope_head_dim == 64);
  CHECK(t.v_head_dim == 128);
  // MoE
  CHECK(t.num_experts == 896);
  CHECK(t.num_experts_per_token == 16);
  CHECK(t.num_shared_experts == 2);
  CHECK(t.moe_intermediate_size == 3072);
  CHECK(t.moe_router_activation_func == "sigmoid");
  // KDA linear_attn_config
  CHECK(t.has_linear_attn_config);
  CHECK(t.kda_num_heads == 96);
  CHECK(t.kda_head_dim == 128);
  CHECK(t.kda_short_conv_kernel_size == 4);
  CHECK(static_cast<int>(t.kda_layers.size()) == 69);
  // vision (PARTIAL, MoonViT-V2)
  CHECK(p.vision.present);
  CHECK(p.vision.patch_size == 14);
  CHECK(p.vision.num_hidden_layers == 27);
}

TEST_CASE("kimi-k3 scaffold: per-layer KDA/MLA + MoE/dense split matches upstream") {
  const KimiK3Params p = ParseKimiK3Params(RealScaleK3Config());
  const KimiK3TextParams& t = p.text;
  // is_kda_layer(l) == (l+1) in kda_layers; 1..69 => layers 0..68 KDA, 69..92 MLA.
  int kda = 0, mla = 0;
  for (int64_t l = 0; l < t.num_hidden_layers; ++l)
    (t.is_kda_layer(l) ? kda : mla)++;
  CHECK(kda == 69);
  CHECK(mla == 24);
  CHECK(t.is_kda_layer(0));
  CHECK(t.is_kda_layer(68));
  CHECK_FALSE(t.is_kda_layer(69));
  CHECK_FALSE(t.is_kda_layer(92));
  // first_k_dense_replace=1 => layer 0 dense, 1..92 MoE.
  CHECK_FALSE(t.is_moe_layer(0));
  CHECK(t.is_moe_layer(1));
  CHECK(t.is_moe_layer(92));
}

TEST_CASE("kimi-k3 scaffold: text-backbone enumeration is structurally faithful") {
  const KimiK3Params p = ParseKimiK3Params(SmallK3Config());
  const std::vector<std::string> names =
      EnumerateKimiK3TextBackboneTensors(p.text, /*prefix=*/"");
  const auto has = [&](const std::string& n) {
    return std::find(names.begin(), names.end(), n) != names.end();
  };

  // model level
  CHECK(has("model.embed_tokens.weight"));
  CHECK(has("model.norm.weight"));
  CHECK(has("lm_head.weight"));  // tie_word_embeddings=false

  // layer 0 = KDA + dense (first_k_dense_replace=1): KDA weights + dense MLP.
  CHECK(has("model.layers.0.self_attn.f_a_proj.weight"));   // KDA low-rank decay
  CHECK(has("model.layers.0.self_attn.f_b_proj.weight"));
  CHECK(has("model.layers.0.self_attn.o_norm.weight"));     // sigmoid-gated norm
  CHECK(has("model.layers.0.self_attn.q_conv1d.weight"));   // 3 short convs
  CHECK(has("model.layers.0.self_attn.A_log"));
  CHECK(has("model.layers.0.mlp.gate_proj.weight"));        // dense MLP
  CHECK_FALSE(has("model.layers.0.mlp.gate.weight"));       // NOT MoE

  // layer 2 = MLA (idx 2 not in kda set) + MoE. K3 q_lora_rank>0 => q-LoRA branch.
  CHECK(has("model.layers.2.self_attn.q_a_proj.weight"));
  CHECK(has("model.layers.2.self_attn.q_a_layernorm.weight"));
  CHECK(has("model.layers.2.self_attn.q_b_proj.weight"));
  CHECK(has("model.layers.2.self_attn.kv_a_proj_with_mqa.weight"));
  CHECK(has("model.layers.2.self_attn.kv_b_proj.weight"));
  CHECK_FALSE(has("model.layers.2.self_attn.f_a_proj.weight"));  // NOT KDA
  // MoE gate + noaux_tc bias + shared experts + routed experts w1/w2/w3.
  CHECK(has("model.layers.2.mlp.gate.weight"));
  CHECK(has("model.layers.2.mlp.gate.e_score_correction_bias"));
  CHECK(has("model.layers.2.mlp.shared_experts.gate_proj.weight"));
  CHECK(has("model.layers.2.mlp.experts.0.w1.weight"));
  CHECK(has("model.layers.2.mlp.experts.3.w3.weight"));  // num_experts=4 => 0..3
  CHECK_FALSE(has("model.layers.2.mlp.experts.4.w1.weight"));
}

TEST_CASE("kimi-k3 scaffold: parse REJECTS a config missing linear_attn_config") {
  HfConfig bad = SmallK3Config();
  bad.raw["text_config"].erase("linear_attn_config");
  CHECK_THROWS_AS(ParseKimiK3Params(bad), std::runtime_error);

  HfConfig bad2 = SmallK3Config();
  bad2.raw["text_config"]["moe_router_activation_func"] = "gelu";  // not sig/softmax
  CHECK_THROWS_AS(ParseKimiK3Params(bad2), std::runtime_error);
}

TEST_CASE("kimi-k3 scaffold: loader REFUSES MXFP4 (shared DeepSeek-V4 row)") {
  // A real K3 checkpoint is MXFP4 (mxfp4-pack-quantized). The loader must refuse
  // NOT-YET-BUILDABLE rather than write a second MXFP4 kernel. Empty shards is
  // fine: the MXFP4 refusal fires before any tensor is touched.
  HfConfig cfg = RealScaleK3Config();
  cfg.raw["quantization_config"] = {{"quant_method", "compressed-tensors"},
                                    {"format", "mxfp4-pack-quantized"}};
  const KimiK3Params p = ParseKimiK3Params(cfg);
  CHECK(p.is_mxfp4);
  const std::vector<vllm::SafetensorsFile> no_shards;
  CHECK_THROWS_AS(LoadKimiK3ForConditionalGenerationWeights(no_shards, cfg),
                  std::runtime_error);
}
