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
#include "vllm/model_executor/models/laguna_ops.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
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

TEST_CASE("laguna scaffold: loaders LOUDLY report the W4 device-materialization residual") {
  const HfConfig c = RealConfig();
  // The GGUF/safetensors tower materialization is a W4 residual — must throw, not
  // silently return an empty/wrong model. (The forward COMPOSITION + the 3 new
  // ops are now REAL + unit-gated below; only device materialization + the strict
  // dual-oracle greedy gate remain.)
  const std::vector<vllm::SafetensorsFile> no_shards;
  CHECK_THROWS(vllm::LoadLagunaForCausalLMWeights(no_shards, c));
}

// ─── W3 NEW-OP UNIT GATES ─────────────────────────────────────────────────────
using vllm::LagunaRouterSelection;
using vllm::LagunaSoftplus;
using vllm::LagunaSoftplusHeadGate;
using vllm::LagunaUngroupedRouterTopK;

namespace {
// Build a host f32 OwnedTensor from a flat vector + shape.
vllm::OwnedTensor F32(const std::vector<float>& v,
                      std::vector<int64_t> shape) {
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kF32;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  std::vector<uint8_t> bytes(v.size() * sizeof(float));
  std::memcpy(bytes.data(), v.data(), bytes.size());
  t.bytes = vllm::OwnedBytes(std::move(bytes));
  return t;
}
// Deterministic pseudo-random small weights (LCG), centered near 0.
std::vector<float> Rand(int64_t n, uint32_t& s) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[static_cast<size_t>(i)] =
        (static_cast<float>(s >> 8) / static_cast<float>(1u << 24) - 0.5F) * 0.4F;
  }
  return v;
}
}  // namespace

TEST_CASE("laguna op (a): per-head softplus attention output gate") {
  // softplus(x) = log1p(exp(x)); threshold 20 -> linear. Bit-check a few points.
  CHECK(LagunaSoftplus(0.0F) == doctest::Approx(std::log(2.0F)));
  CHECK(LagunaSoftplus(-2.0F) == doctest::Approx(std::log1p(std::exp(-2.0F))));
  CHECK(LagunaSoftplus(25.0F) == doctest::Approx(25.0F));  // linear above 20

  // Two heads, head_dim 3; gate broadcasts the per-head softplus over head_dim.
  std::vector<float> attn = {1, 2, 3, /*head1*/ -1, -2, -3};
  const std::vector<float> gl = {0.0F, 1.5F};
  LagunaSoftplusHeadGate(attn, gl, /*num_heads=*/2, /*head_dim=*/3);
  const float g0 = std::log1p(std::exp(0.0F));
  const float g1 = std::log1p(std::exp(1.5F));
  CHECK(attn[0] == doctest::Approx(1 * g0));
  CHECK(attn[2] == doctest::Approx(3 * g0));
  CHECK(attn[3] == doctest::Approx(-1 * g1));
  CHECK(attn[5] == doctest::Approx(-3 * g1));
}

TEST_CASE("laguna op (b): ungrouped sigmoid-noaux router selection + tie-break + weights") {
  // 4 experts, top-2. Logits chosen so sigmoid(scores) has a clear order, but the
  // e_score_correction_bias FLIPS the selection (bias steers selection; the
  // COMBINE weight stays the UNBIASED sigmoid). This is the noaux_tc contract.
  const std::vector<float> logits = {0.0F, 0.5F, -0.5F, 2.0F};
  //   sigmoid ~ {0.500, 0.622, 0.378, 0.881}
  // bias lifts expert 2 (the lowest raw score) above everyone but expert 3.
  const std::vector<float> bias = {0.0F, 0.0F, 1.0F, 0.0F};
  //   choice ~ {0.500, 0.622, 1.378, 0.881} -> top-2 (choice desc) = {expert 2, expert 3}
  const LagunaRouterSelection sel = LagunaUngroupedRouterTopK(
      logits, bias, /*top_k=*/2, /*norm_topk_prob=*/true, /*routed_scaling=*/2.5F);
  REQUIRE(sel.ids.size() == 2);
  CHECK(sel.ids[0] == 2);  // bias-lifted to the highest choice score
  CHECK(sel.ids[1] == 3);  // next-highest choice
  // Weights are the UNBIASED sigmoid scores, renormalized, then *routed_scaling.
  const float s2 = 1.0F / (1.0F + std::exp(0.5F));   // sigmoid(-0.5), expert 2
  const float s3 = 1.0F / (1.0F + std::exp(-2.0F));  // sigmoid(2.0),  expert 3
  const float sum = s2 + s3;
  CHECK(sel.weights[0] == doctest::Approx(s2 / sum * 2.5F));
  CHECK(sel.weights[1] == doctest::Approx(s3 / sum * 2.5F));

  // TIE-BREAK RAZOR (RED-first): equal choice scores must select the LOWER index.
  // Before the (choice desc, index asc) tie-break this returned the higher index.
  const std::vector<float> tie_logits = {1.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<float> no_bias;  // all-zero
  const LagunaRouterSelection ts = LagunaUngroupedRouterTopK(
      tie_logits, no_bias, /*top_k=*/2, /*norm_topk_prob=*/false, 1.0F);
  CHECK(ts.ids[0] == 0);
  CHECK(ts.ids[1] == 1);
}

TEST_CASE("laguna op (c): dual per-layer RoPE cos/sin caches bit-match a hand reference") {
  const HfConfig c = RealConfig();
  const LagunaParams p = ParseLagunaParams(c);
  const int64_t rows = 4;

  // Sliding (plain) cache: rotary_dim 128, theta 10000, mscale 1.
  const std::vector<float> slide = vllm::BuildLagunaSlidingCosSin(p, rows);
  REQUIRE(static_cast<int64_t>(slide.size()) == rows * p.rotary_dim_sliding);
  const int64_t rds = p.rotary_dim_sliding;
  const int64_t halfs = rds / 2;
  for (int64_t r : {1, 3})
    for (int64_t i : {0, 1, 7}) {
      const double inv = 1.0 / std::pow(10000.0, static_cast<double>(2 * i) /
                                                     static_cast<double>(rds));
      const double freq = static_cast<double>(r) * inv;
      CHECK(slide[static_cast<size_t>(r * rds + i)] ==
            doctest::Approx(std::cos(static_cast<float>(freq))));
      CHECK(slide[static_cast<size_t>(r * rds + halfs + i)] ==
            doctest::Approx(std::sin(static_cast<float>(freq))));
    }

  // Full-attn YaRN cache: rotary_dim 64 (partial 0.5), mscale = attention_factor.
  // Reference inv_freq via the (separately gated) YaRN detail; the laguna-NEW
  // compose being checked here is partial rotary_dim + mscale + layout.
  const std::vector<float> yarn = vllm::BuildLagunaFullYarnCosSin(p, rows);
  REQUIRE(static_cast<int64_t>(yarn.size()) == rows * p.rotary_dim_full);
  const std::vector<float> inv = vllm::rotary_embedding_detail::compute_yarn_inv_freq(
      p.rotary_dim_full, p.rope_theta_full, p.yarn_orig_max_pos, p.yarn_factor,
      1.0, static_cast<int64_t>(p.yarn_beta_fast),
      static_cast<int64_t>(p.yarn_beta_slow), false);
  const int64_t rdf = p.rotary_dim_full;
  const int64_t halff = rdf / 2;
  const float m = static_cast<float>(p.yarn_attention_factor);
  for (int64_t r : {1, 2})
    for (int64_t i : {0, 5, 31}) {
      const float freq = static_cast<float>(r) * inv[static_cast<size_t>(i)];
      CHECK(yarn[static_cast<size_t>(r * rdf + i)] ==
            doctest::Approx(std::cos(freq) * m));
      CHECK(yarn[static_cast<size_t>(r * rdf + halff + i)] ==
            doctest::Approx(std::sin(freq) * m));
    }
  // The YaRN mscale (1.4852) genuinely scales the cache (not an identity).
  CHECK(m > 1.4F);
  CHECK(yarn[0] == doctest::Approx(m));  // r=0: cos(0)*m = m
}

TEST_CASE("laguna: per-layer VARIABLE Q-head shape wiring") {
  const HfConfig c = RealConfig();
  const LagunaParams p = ParseLagunaParams(c);
  // Global layer 0: 48 heads, group 6, q_dim 6144. Sliding layer 1: 72/9/9216.
  CHECK(p.QHeadsForLayer(0) == 48);
  CHECK(p.QHeadsForLayer(1) == 72);
  CHECK(p.GqaGroupForLayer(0) == 6);
  CHECK(p.GqaGroupForLayer(1) == 9);
  CHECK(p.QDimForLayer(0) == 48 * 128);
  CHECK(p.QDimForLayer(1) == 72 * 128);
  CHECK(p.RotaryDimForLayer(0) == 64);   // global -> YaRN partial-64
  CHECK(p.RotaryDimForLayer(1) == 128);  // sliding -> plain-128
  CHECK(p.WindowForLayer(0) == 0);       // global -> full causal
  CHECK(p.WindowForLayer(1) == 512);     // sliding -> window 512
}

// ─── W3 FORWARD COMPOSITION GATE (synthetic weights) ─────────────────────────
namespace {
// A tiny 2-layer Laguna: layer 0 global+dense, layer 1 sliding+MoE. Exercises the
// per-layer variable Q-head (2 vs 4), dual RoPE, sliding window, softplus gate,
// dense MLP + ungrouped MoE, untied lm_head. Numeric golden vs an oracle is W4;
// this gate proves the composition RUNS, is deterministic, shape-correct, and
// that the new ops are WIRED (perturbing g_proj changes the logits).
vllm::LagunaWeights TinyModel() {
  vllm::LagunaParams p;
  p.hidden_size = 8;
  p.num_hidden_layers = 2;
  p.vocab_size = 10;
  p.num_attention_heads = 2;
  p.num_key_value_heads = 1;
  p.head_dim = 4;
  p.intermediate_size = 12;
  p.rms_norm_eps = 1e-6F;
  p.tie_word_embeddings = false;
  p.max_position_embeddings = 64;
  p.sliding_window = 2;
  p.layer_types = {"full_attention", "sliding_attention"};
  p.num_attention_heads_per_layer = {2, 4};
  p.num_experts = 4;
  p.num_experts_per_tok = 2;
  p.moe_intermediate_size = 6;
  p.shared_expert_intermediate_size = 6;
  p.norm_topk_prob = true;
  p.moe_routed_scaling_factor = 2.5F;
  p.mlp_only_layers = {0};
  p.rope_theta_full = 500000.0;
  p.yarn_factor = 128.0;
  p.yarn_orig_max_pos = 8192;
  p.yarn_beta_fast = 32.0;
  p.yarn_beta_slow = 1.0;
  p.yarn_attention_factor = 1.4852030263919618;
  p.partial_rotary_factor_full = 0.5;
  p.rotary_dim_full = 2;      // head_dim(4) * 0.5
  p.rope_theta_sliding = 10000.0;
  p.rotary_dim_sliding = 4;   // full head_dim

  const int64_t H = p.hidden_size, V = p.vocab_size, Dh = p.head_dim;
  const int64_t Hkv = p.num_key_value_heads, E = p.num_experts;
  const int64_t moeI = p.moe_intermediate_size, I = p.intermediate_size;
  uint32_t s = 12345u;

  vllm::LagunaWeights w;
  w.params = p;
  w.embed = F32(Rand(V * H, s), {V, H});
  w.norm = F32(Rand(H, s), {H});
  w.lm_head = F32(Rand(V * H, s), {V, H});
  for (int64_t l = 0; l < 2; ++l) {
    const int64_t Hq = p.QHeadsForLayer(l);
    vllm::LagunaLayerWeights lw;
    lw.input_norm = F32(Rand(H, s), {H});
    lw.post_attn_norm = F32(Rand(H, s), {H});
    lw.attn.q_proj = F32(Rand(Hq * Dh * H, s), {Hq * Dh, H});
    lw.attn.k_proj = F32(Rand(Hkv * Dh * H, s), {Hkv * Dh, H});
    lw.attn.v_proj = F32(Rand(Hkv * Dh * H, s), {Hkv * Dh, H});
    lw.attn.o_proj = F32(Rand(H * Hq * Dh, s), {H, Hq * Dh});
    lw.attn.g_proj = F32(Rand(Hq * H, s), {Hq, H});
    if (l == 0) {
      lw.is_dense = true;
      lw.mlp.gate_up_proj = F32(Rand(2 * I * H, s), {2 * I, H});
      lw.mlp.down_proj = F32(Rand(H * I, s), {H, I});
    } else {
      lw.is_dense = false;
      lw.moe.router = F32(Rand(E * H, s), {E, H});
      lw.moe.e_score_correction_bias = F32(Rand(E, s), {E});
      lw.moe.experts_gate_up = F32(Rand(E * 2 * moeI * H, s), {E, 2 * moeI, H});
      lw.moe.experts_down = F32(Rand(E * H * moeI, s), {E, H, moeI});
      lw.moe.shared_gate_up = F32(Rand(2 * moeI * H, s), {2 * moeI, H});
      lw.moe.shared_down = F32(Rand(H * moeI, s), {H, moeI});
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}
}  // namespace

TEST_CASE("laguna W3 forward: composition RUNS, deterministic, shape-correct, ops wired") {
  vllm::LagunaWeights w = TinyModel();
  const std::vector<int32_t> tokens = {1, 4, 7};
  const std::vector<int32_t> positions = {0, 1, 2};
  const int64_t V = w.params.vocab_size;
  vllm::v1::CommonAttentionMetadata meta{};
  const std::vector<vllm::PagedKvCache> kv;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};  // reference ignores it

  const std::vector<float> logits = vllm::LagunaModel::Forward(
      tokens, positions, meta, kv, w, HfConfig{}, q, {});
  REQUIRE(static_cast<int64_t>(logits.size()) == 3 * V);
  for (float x : logits) CHECK(std::isfinite(x));

  // Deterministic across runs.
  const std::vector<float> logits2 = vllm::LagunaModel::Forward(
      tokens, positions, meta, kv, w, HfConfig{}, q, {});
  REQUIRE(logits2.size() == logits.size());
  for (size_t i = 0; i < logits.size(); ++i)
    CHECK(logits[i] == doctest::Approx(logits2[i]));

  // logits_indices gather: last row only -> equals the full forward's last row.
  const std::vector<float> last = vllm::LagunaModel::Forward(
      tokens, positions, meta, kv, w, HfConfig{}, q, {2});
  REQUIRE(static_cast<int64_t>(last.size()) == V);
  for (int64_t j = 0; j < V; ++j)
    CHECK(last[static_cast<size_t>(j)] ==
          doctest::Approx(logits[static_cast<size_t>(2 * V + j)]));

  // The per-head softplus gate is WIRED: perturbing g_proj shifts the logits.
  vllm::LagunaWeights w2 = TinyModel();
  {
    const vllm::OwnedTensor& gt = w2.layers[0].attn.g_proj;
    std::vector<float> g(reinterpret_cast<const float*>(gt.bytes.data()),
                         reinterpret_cast<const float*>(gt.bytes.data()) +
                             gt.Numel());
    for (float& x : g) x += 1.0F;
    w2.layers[0].attn.g_proj =
        F32(g, {w2.params.QHeadsForLayer(0), w2.params.hidden_size});
  }
  const std::vector<float> logits3 = vllm::LagunaModel::Forward(
      tokens, positions, meta, kv, w2, HfConfig{}, q, {});
  bool differs = false;
  for (size_t i = 0; i < logits.size(); ++i)
    if (std::abs(logits[i] - logits3[i]) > 1e-4F) differs = true;
  CHECK(differs);
}
