// N1b-close (task #230): run-gate for the NVFP4 Laguna safetensors loader
// (LoadLagunaForCausalLMWeights). Builds a tiny synthetic 2-layer NVFP4 checkpoint
// (L0 dense, L1 MoE with 2 W4A4 experts) matching the verified
// poolside/Laguna-S-2.1-NVFP4 name-map + dtypes, loads it through the REAL loader,
// and asserts every field round-trips byte-identically. RED-first: a wrong
// scale-global reciprocal and a missing tensor both fail.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/laguna.h"

using vllm::HfConfig;
using vllm::LagunaWeights;
using vllm::LoadLagunaForCausalLMWeights;
using vllm::SafetensorsFile;

namespace {

// ── synthetic safetensors builder ──────────────────────────────────────────
struct Fx {
  std::string name;
  std::string dtype;                 // "BF16" / "F32" / "U8" / "F8_E4M3"
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// Deterministic filler: byte b at index i = (seed*31 + i) & 0xff.
std::string Fill(size_t n, int seed) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i) s[i] = static_cast<char>((seed * 31 + static_cast<int>(i)) & 0xff);
  return s;
}
// f32 scalar as 4 raw bytes.
std::string F32Bytes(float v) {
  std::string s(4, '\0');
  std::memcpy(s.data(), &v, 4);
  return s;
}

std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype},
                   {"shape", t.shape},
                   {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("laguna_nvfp4_loader_" + std::to_string(counter++) + ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// ── tiny model geometry ─────────────────────────────────────────────────────
constexpr int H = 32, Dh = 16, NH = 2, KV = 1, DENSE_I = 64, MOE_I = 16, V = 8, E = 2;

HfConfig TinyConfig() {
  HfConfig c;
  c.architectures = {"LagunaForCausalLM"};
  c.model_type = "laguna";
  c.hidden_size = H;
  c.num_hidden_layers = 2;
  c.vocab_size = V;
  c.num_attention_heads = NH;
  c.num_key_value_heads = KV;
  c.head_dim = Dh;
  c.intermediate_size = DENSE_I;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 4096;
  c.sliding_window = 8;
  c.num_experts = E;
  c.num_experts_per_tok = 1;
  c.moe_intermediate_size = MOE_I;
  nlohmann::json layer_types = nlohmann::json::array();
  nlohmann::json heads = nlohmann::json::array();
  for (int l = 0; l < 2; ++l) {
    const bool global = (l == 0);
    layer_types.push_back(global ? "full_attention" : "sliding_attention");
    heads.push_back(NH);
    c.layer_types.push_back(global ? "full_attention" : "sliding_attention");
  }
  c.raw = {
      {"hidden_size", H}, {"num_hidden_layers", 2}, {"vocab_size", V},
      {"num_attention_heads", NH}, {"num_key_value_heads", KV}, {"head_dim", Dh},
      {"intermediate_size", DENSE_I}, {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 4096}, {"tie_word_embeddings", false},
      {"sliding_window", 8}, {"layer_types", layer_types},
      {"num_attention_heads_per_layer", heads}, {"num_experts", E},
      {"num_experts_per_tok", 1}, {"moe_intermediate_size", MOE_I},
      {"shared_expert_intermediate_size", MOE_I}, {"norm_topk_prob", true},
      {"moe_routed_scaling_factor", 2.5}, {"mlp_only_layers", nlohmann::json::array({0})},
      {"rope_parameters",
       {{"full_attention", {{"rope_type", "yarn"}, {"rope_theta", 500000.0},
                            {"factor", 8.0}, {"original_max_position_embeddings", 512},
                            {"beta_slow", 1.0}, {"beta_fast", 32.0},
                            {"attention_factor", 1.0}, {"partial_rotary_factor", 0.5}}},
        {"sliding_attention", {{"rope_type", "default"}, {"rope_theta", 10000.0},
                               {"partial_rotary_factor", 1.0}}}}},
  };
  return c;
}

// BF16 tensor (2 bytes/elem).
Fx Bf16(const std::string& n, std::vector<int64_t> shape, int seed) {
  int64_t ne = 1;
  for (int64_t d : shape) ne *= d;
  return {n, "BF16", std::move(shape), Fill(static_cast<size_t>(ne) * 2, seed)};
}
void AttnBf16(std::vector<Fx>& v, const std::string& b, int& s) {
  v.push_back(Bf16(b + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(b + "post_attention_layernorm.weight", {H}, s++));
  v.push_back(Bf16(b + "self_attn.q_proj.weight", {NH * Dh, H}, s++));
  v.push_back(Bf16(b + "self_attn.k_proj.weight", {KV * Dh, H}, s++));
  v.push_back(Bf16(b + "self_attn.v_proj.weight", {KV * Dh, H}, s++));
  v.push_back(Bf16(b + "self_attn.o_proj.weight", {H, NH * Dh}, s++));
  v.push_back(Bf16(b + "self_attn.g_proj.weight", {NH, H}, s++));
  v.push_back(Bf16(b + "self_attn.q_norm.weight", {Dh}, s++));
  v.push_back(Bf16(b + "self_attn.k_norm.weight", {Dh}, s++));
}
// One W4A4 NVFP4 projection: weight_packed U8 [N,K/2] + weight_scale F8 [N,K/16]
// + weight_global_scale/input_global_scale F32 scalars.
void ExpertProj(std::vector<Fx>& v, const std::string& p, int64_t n, int64_t k, int& s,
                float wgs, float igs) {
  v.push_back({p + ".weight_packed", "U8", {n, k / 2}, Fill(static_cast<size_t>(n * k / 2), s++)});
  v.push_back({p + ".weight_scale", "F8_E4M3", {n, k / 16}, Fill(static_cast<size_t>(n * k / 16), s++)});
  v.push_back({p + ".weight_global_scale", "F32", {}, F32Bytes(wgs)});
  v.push_back({p + ".input_global_scale", "F32", {}, F32Bytes(igs)});
}

std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("model.embed_tokens.weight", {V, H}, s++));
  v.push_back(Bf16("model.norm.weight", {H}, s++));
  v.push_back(Bf16("lm_head.weight", {V, H}, s++));
  // layer 0 dense
  AttnBf16(v, "model.layers.0.", s);
  v.push_back(Bf16("model.layers.0.mlp.gate_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16("model.layers.0.mlp.up_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16("model.layers.0.mlp.down_proj.weight", {H, DENSE_I}, s++));
  // layer 1 MoE
  AttnBf16(v, "model.layers.1.", s);
  v.push_back(Bf16("model.layers.1.mlp.gate.weight", {E, H}, s++));  // router BF16
  v.push_back({"model.layers.1.mlp.experts.e_score_correction_bias", "F32", {E},
               Fill(static_cast<size_t>(E) * 4, s++)});
  for (int e = 0; e < E; ++e) {
    const std::string ep = "model.layers.1.mlp.experts." + std::to_string(e) + ".";
    ExpertProj(v, ep + "gate_proj", MOE_I, H, s, 2.0F + e, 3.0F + e);
    ExpertProj(v, ep + "up_proj", MOE_I, H, s, 4.0F + e, 5.0F + e);
    ExpertProj(v, ep + "down_proj", H, MOE_I, s, 6.0F + e, 7.0F + e);
  }
  v.push_back(Bf16("model.layers.1.mlp.shared_expert.gate_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16("model.layers.1.mlp.shared_expert.up_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16("model.layers.1.mlp.shared_expert.down_proj.weight", {H, MOE_I}, s++));
  return v;
}

}  // namespace

TEST_CASE("laguna nvfp4 loader: synthetic checkpoint round-trips byte-identically") {
  const std::vector<Fx> ts = BuildTensors();
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));

  const LagunaWeights w = LoadLagunaForCausalLMWeights(shards, TinyConfig());

  // structure
  CHECK(w.params.num_hidden_layers == 2);
  REQUIRE(w.layers.size() == 2u);
  CHECK(w.layers[0].is_dense == true);
  CHECK(w.layers[1].is_dense == false);

  // model-level BF16 round-trip (embed bytes == the fixture's embed bytes)
  auto find = [&](const std::string& n) -> const Fx& {
    for (const Fx& t : ts) if (t.name == n) return t;
    FAIL("fixture tensor missing: " << n);
    return ts[0];
  };
  const Fx& embed = find("model.embed_tokens.weight");
  REQUIRE(w.embed.bytes.size() == embed.bytes.size());
  CHECK(std::memcmp(w.embed.bytes.data(), embed.bytes.data(), embed.bytes.size()) == 0);

  // dense L0 MLP present, MoE fp4 fields empty on the dense layer
  CHECK(w.layers[0].mlp.gate_proj.bytes.size() ==
        find("model.layers.0.mlp.gate_proj.weight").bytes.size());

  // MoE layer: 2 W4A4 experts loaded
  const auto& moe = w.layers[1].moe;
  REQUIRE(moe.experts_gate_fp4.size() == static_cast<size_t>(E));
  REQUIRE(moe.experts_up_fp4.size() == static_cast<size_t>(E));
  REQUIRE(moe.experts_down_fp4.size() == static_cast<size_t>(E));

  // expert-0 gate: shapes + packed/scale byte-identity + global-scale math
  const auto& g0 = moe.experts_gate_fp4[0];
  CHECK(g0.n == MOE_I);
  CHECK(g0.k == H);
  const Fx& g0p = find("model.layers.1.mlp.experts.0.gate_proj.weight_packed");
  const Fx& g0s = find("model.layers.1.mlp.experts.0.gate_proj.weight_scale");
  REQUIRE(g0.packed.bytes.size() == g0p.bytes.size());
  CHECK(std::memcmp(g0.packed.bytes.data(), g0p.bytes.data(), g0p.bytes.size()) == 0);
  REQUIRE(g0.scale.bytes.size() == g0s.bytes.size());
  CHECK(std::memcmp(g0.scale.bytes.data(), g0s.bytes.data(), g0s.bytes.size()) == 0);
  // gate_proj expert 0: wgs=2.0, igs=3.0 (from ExpertProj call)
  CHECK(g0.weight_global_scale_inv == doctest::Approx(2.0));
  CHECK(g0.scale2 == doctest::Approx(1.0 / 2.0));              // reciprocal of the divisor
  CHECK(g0.input_global_scale_inv == doctest::Approx(3.0));
  CHECK(g0.alpha == doctest::Approx((1.0 / 2.0) * (1.0 / 3.0)));  // scale2 * 1/igs

  // expert-1 down: distinct globals prove per-expert indexing (wgs=7.0, igs=8.0)
  const auto& d1 = moe.experts_down_fp4[1];
  CHECK(d1.n == H);
  CHECK(d1.k == MOE_I);
  CHECK(d1.scale2 == doctest::Approx(1.0 / 7.0));
  CHECK(d1.alpha == doctest::Approx((1.0 / 7.0) * (1.0 / 8.0)));

  // F32 bias round-trip
  const Fx& bias = find("model.layers.1.mlp.experts.e_score_correction_bias");
  REQUIRE(moe.e_score_correction_bias.bytes.size() == bias.bytes.size());
  CHECK(std::memcmp(moe.e_score_correction_bias.bytes.data(), bias.bytes.data(),
                    bias.bytes.size()) == 0);

  // shared expert is BF16 (not fp4)
  CHECK(moe.shared_gate.bytes.size() ==
        find("model.layers.1.mlp.shared_expert.gate_proj.weight").bytes.size());
  CHECK(moe.shared_gate_fp4.Empty());  // the fp4 shared field stays unused
}

TEST_CASE("laguna nvfp4 loader: missing tensor throws (RED-first)") {
  std::vector<Fx> ts = BuildTensors();
  // drop expert-0 gate weight_scale -> the loader must throw, not silently succeed
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const Fx& t) {
                            return t.name ==
                                   "model.layers.1.mlp.experts.0.gate_proj.weight_scale";
                          }),
           ts.end());
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  CHECK_THROWS(LoadLagunaForCausalLMWeights(shards, TinyConfig()));
}
