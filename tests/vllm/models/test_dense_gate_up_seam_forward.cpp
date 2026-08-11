// vllm.cpp original (FUSION-DENSE-MIGRATE executed-coverage doctest); no upstream
// mirror.
//
// EXECUTED CPU COVERAGE for the merged-GEMM fold — the four DENSE model TUs whose
// gate-up MLP moved onto layers::UnquantizedMlpGateUpMethod (commandr, glm4,
// minicpm, phi3) had, before this file, NO test that loaded them at all on a CPU
// box: their only gates (tests/parity/test_*_paged_engine.cpp) are checkpoint-gated
// and dgx-only, and the seam's own unit test (test_linear_method) exercises the
// METHOD, not the five call sites that now use it. A mutation of the intermediate
// size at a call site (`I` -> `I - 1`) therefore survived the whole CPU suite.
//
// ORACLE evidence needs the GPU; SELF-CONSISTENCY evidence does not. Each case
// drives the REAL model forward over in-memory synthetic weights (no checkpoint,
// no GPU) and pins the two structural properties the fold could break, both
// analytically — no golden, no tolerance:
//
//   (a) THE GATE/UP SPLIT AND THE MULTIPLICATIVE `up`. Zeroing the UP half of the
//       merged [2I, H] gate_up (rows [I, 2I)) makes silu(gate)*up EXACTLY zero, so
//       down_proj(0) is exactly zero — the MLP contributes nothing. Zeroing
//       down_proj instead makes the MLP contribute nothing for a different reason.
//       Both arms must produce BYTE-IDENTICAL logits. They only do so if the seam
//       splits the merged operand at exactly I: with any other split the "up"
//       slice picks up gate rows, act is nonzero, and the two arms diverge (or the
//       shape contract fails outright).
//   (b) THE HALF ORDER. Swapping the two halves must CHANGE the logits, since
//       silu(gate)*up != silu(up)*gate — so a gate/up transposition cannot hide.
//
//   Plus the vacuity guard: baseline != MLP-disabled, i.e. the MLP is live in the
//   forward these cases drive, so (a) is not two zeros compared to each other.
//
// MiniCPM3 shares the identical Qwen3DenseMlpWeights call shape with MiniCPM and
// Phi-3 (both covered here) but wraps it in an MLA attention block with a
// load-time kv_b_proj absorption; its synthetic harness is the DeepSeek-V2 one
// (tests/vllm/models/test_deepseek_v2_forward.cpp), not this dense one, so it is
// the one folded TU this file does not drive.
//
// The token-exact vs-oracle bar for all five stays tests/parity/
// test_{commandr,glm4,minicpm,minicpm3,phi3}_paged_engine.cpp (dgx-only).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/commandr.h"
#include "vllm/model_executor/models/glm4.h"
#include "vllm/model_executor/models/minicpm.h"
#include "vllm/model_executor/models/phi3.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

// ─── Tiny shared geometry (every model below is built at these dims) ─────────
constexpr int64_t kH = 64;    // hidden_size
constexpr int64_t kHq = 4;    // attention heads
constexpr int64_t kHkv = 2;   // kv heads (GQA 2)
constexpr int64_t kDh = 16;   // head_dim
constexpr int64_t kI = 32;    // intermediate_size (the split this file pins)
constexpr int64_t kV = 48;    // vocab
constexpr int64_t kL = 2;     // layers
constexpr int64_t kMaxPos = 32;
constexpr int64_t kT = 5;     // prefill tokens
constexpr int64_t kBlock = 8;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                     float scale = 0.08F) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (int64_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(dist(rng));
  return o;
}

// Plain per-position [cos|sin] rope cache, bf16 [rows, dim] — the layout
// vt::RopeFromCache consumes (Phi-3's LongRoPE cache and Command-R's full-rotary
// cache are this shape; the exact angles are irrelevant to what these cases pin,
// only that they are the SAME across arms).
OwnedTensor MakeRopeCache(int64_t rows, int64_t dim, double theta) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = false;
  o.rank = 2;
  o.shape[0] = rows;
  o.shape[1] = dim;
  o.bytes.resize(static_cast<size_t>(rows * dim) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  const int64_t half = dim / 2;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < half; ++i) {
      const double inv = 1.0 / std::pow(theta, static_cast<double>(2 * i) /
                                                   static_cast<double>(dim));
      const double ang = static_cast<double>(r) * inv;
      p[r * dim + i] = vt::F32ToBF16(static_cast<float>(std::cos(ang)));
      p[r * dim + half + i] = vt::F32ToBF16(static_cast<float>(std::sin(ang)));
    }
  }
  return o;
}

// ─── The four arms every case runs ───────────────────────────────────────────
enum class Arm {
  kBase,      // untouched weights
  kZeroUp,    // gate_up rows [I, 2I) = 0  -> silu(gate)*0 == 0  -> MLP contributes 0
  kZeroDown,  // down_proj = 0             -> down(act) == 0     -> MLP contributes 0
  kSwap,      // swap the gate and up halves
};

// Mutate ONE layer's MLP in place. Works for every merged-gate_up MLP struct
// (Qwen3DenseMlpWeights / Glm4MlpWeights / CommandrMlpWeights) — they all carry
// `gate_up_proj` [2I, H] raw-NK and `down_proj` [H, I] raw-NK.
template <typename MlpW>
void ApplyArm(MlpW& m, Arm arm) {
  auto* gu = reinterpret_cast<uint16_t*>(m.gate_up_proj.bytes.data());
  auto* dn = reinterpret_cast<uint16_t*>(m.down_proj.bytes.data());
  const size_t half = static_cast<size_t>(kI) * static_cast<size_t>(kH);
  switch (arm) {
    case Arm::kBase:
      break;
    case Arm::kZeroUp:
      std::fill(gu + half, gu + 2 * half, static_cast<uint16_t>(0));
      break;
    case Arm::kZeroDown:
      std::fill(dn, dn + static_cast<size_t>(kH) * static_cast<size_t>(kI),
                static_cast<uint16_t>(0));
      break;
    case Arm::kSwap:
      for (size_t i = 0; i < half; ++i) std::swap(gu[i], gu[half + i]);
      break;
  }
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(int64_t layers, int64_t num_blocks, int64_t block_size, int64_t kv_heads,
            int64_t head_size) {
    for (int64_t l = 0; l < layers; ++l)
      buf.emplace_back(
          static_cast<size_t>(num_blocks * 2 * block_size * kv_heads * head_size),
          0.0F);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = kv_heads;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

const std::vector<int32_t>& Tokens() {
  static const std::vector<int32_t> t = {3, 17, 42, 8, 31};
  return t;
}
const std::vector<int32_t>& Positions() {
  static const std::vector<int32_t> p = {0, 1, 2, 3, 4};
  return p;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The four assertions, shared by every model case. `run(Arm)` builds the model's
// weights, applies the arm to EVERY layer's MLP, and returns the [T, vocab] logits.
template <typename Run>
void CheckGateUpSeam(const std::string& model, Run run) {
  INFO("model=" << model);

  const std::vector<float> base = run(Arm::kBase);
  REQUIRE(base.size() == static_cast<size_t>(kT * kV));
  for (float x : base) REQUIRE(std::isfinite(x));

  // Deterministic: the same weights twice are byte-identical (so a difference
  // below is the arm, never run-to-run noise).
  CHECK(Same(base, run(Arm::kBase)));

  const std::vector<float> zero_up = run(Arm::kZeroUp);
  const std::vector<float> zero_down = run(Arm::kZeroDown);
  for (float x : zero_up) REQUIRE(std::isfinite(x));

  // (a) THE SPLIT. Two independent ways of making the MLP contribute exactly
  // zero must agree BIT FOR BIT. This is the assertion that dies when the
  // intermediate size handed to the seam is not the model's I.
  CHECK(Same(zero_up, zero_down));

  // Vacuity guard: the MLP is actually live in this forward.
  CHECK_FALSE(Same(base, zero_up));

  // (b) THE HALF ORDER: silu(gate)*up != silu(up)*gate.
  CHECK_FALSE(Same(base, run(Arm::kSwap)));
}

// ─── Per-model configs + synthetic weights ───────────────────────────────────

// Shared dense attention/MLP layer (MiniCPM and Phi-3 both consume
// Qwen3DenseLayerWeights; neither has qk-norm or qkv bias).
vllm::Qwen3DenseLayerWeights MakeDenseLayer(uint32_t& seed) {
  vllm::Qwen3DenseLayerWeights lw;
  lw.input_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
  lw.post_attention_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
  lw.attn.qkv_proj =
      MakeBf16({kHq * kDh + 2 * kHkv * kDh, kH}, /*nk=*/true, seed++);
  lw.attn.o_proj = MakeBf16({kH, kHq * kDh}, /*nk=*/true, seed++);
  lw.mlp.gate_up_proj = MakeBf16({2 * kI, kH}, /*nk=*/true, seed++);
  lw.mlp.down_proj = MakeBf16({kH, kI}, /*nk=*/true, seed++);
  return lw;
}

vllm::Qwen3DenseWeights MakeDenseWeights() {
  vllm::Qwen3DenseWeights w;
  w.tie_word_embeddings = true;
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({kV, kH}, false, 1);
  w.final_norm = MakeBf16({kH}, false, 2, 0.5F);
  uint32_t seed = 100;
  for (int64_t l = 0; l < kL; ++l) w.layers.push_back(MakeDenseLayer(seed));
  return w;
}

HfConfig BaseConfig() {
  HfConfig c;
  c.num_hidden_layers = kL;
  c.hidden_size = kH;
  c.num_attention_heads = kHq;
  c.num_key_value_heads = kHkv;
  c.head_dim = kDh;
  c.rotary_dim = kDh;
  c.intermediate_size = kI;
  c.rms_norm_eps = 1e-5;
  c.rope_theta = 10000.0;
  c.vocab_size = kV;
  c.max_position_embeddings = kMaxPos;
  c.raw = nlohmann::json::object();
  return c;
}

}  // namespace

// ─── MiniCPM (MiniCPMForCausalLM) ────────────────────────────────────────────
TEST_CASE("dense gate-up seam: MiniCPM forward pins the merged gate_up split") {
  HfConfig c = BaseConfig();
  c.model_type = "minicpm";
  c.architectures = {"MiniCPMForCausalLM"};
  c.raw["scale_emb"] = 12.0;    // the three MiniCPM scalar deltas, exercised
  c.raw["scale_depth"] = 1.4;   // so the MLP output rides the scaled residual add
  c.raw["dim_model_base"] = 16.0;

  CheckGateUpSeam("minicpm", [&](Arm arm) {
    vllm::MiniCPMWeights w = MakeDenseWeights();
    for (auto& l : w.layers) ApplyArm(l.mlp, arm);
    CachePool pool(kL, /*num_blocks=*/2, kBlock, kHkv, kDh);
    const CommonAttentionMetadata am = PrefillMeta(kT, kBlock);
    vt::Queue q = Q();
    return vllm::MiniCPMModel::Forward(Tokens(), Positions(), am, pool.attn_kv, w, c,
                                       q);
  });
}

// ─── Phi-3 / Phi-4 (Phi3ForCausalLM) ─────────────────────────────────────────
TEST_CASE("dense gate-up seam: Phi-3 forward pins the merged gate_up split") {
  HfConfig c = BaseConfig();
  c.model_type = "phi3";
  c.architectures = {"Phi3ForCausalLM"};

  CheckGateUpSeam("phi3", [&](Arm arm) {
    vllm::Phi3Weights w;
    w.dense = MakeDenseWeights();
    w.rope_cos_sin = MakeRopeCache(kMaxPos, c.rotary_dim, c.rope_theta);
    for (auto& l : w.dense.layers) ApplyArm(l.mlp, arm);
    CachePool pool(kL, /*num_blocks=*/2, kBlock, kHkv, kDh);
    const CommonAttentionMetadata am = PrefillMeta(kT, kBlock);
    vt::Queue q = Q();
    return vllm::Phi3Model::Forward(Tokens(), Positions(), am, pool.attn_kv, w, c, q);
  });
}

// ─── GLM-4 (Glm4ForCausalLM) ─────────────────────────────────────────────────
TEST_CASE("dense gate-up seam: GLM-4 forward pins the merged gate_up split") {
  HfConfig c = BaseConfig();
  c.model_type = "glm4";
  c.architectures = {"Glm4ForCausalLM"};
  c.rotary_dim = kDh / 2;  // GLM ropes a PARTIAL, interleaved slice

  CheckGateUpSeam("glm4", [&](Arm arm) {
    vllm::Glm4Weights w;
    w.tie_word_embeddings = false;
    w.embed_tokens = MakeBf16({kV, kH}, false, 1);
    w.final_norm = MakeBf16({kH}, false, 2, 0.5F);
    w.lm_head = MakeBf16({kH, kV}, /*nk=*/false, 3);
    uint32_t seed = 200;
    for (int64_t l = 0; l < kL; ++l) {
      vllm::Glm4LayerWeights lw;
      lw.input_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
      lw.post_attention_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
      lw.post_self_attn_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
      lw.post_mlp_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
      lw.attn.qkv_proj =
          MakeBf16({kHq * kDh + 2 * kHkv * kDh, kH}, /*nk=*/true, seed++);
      lw.attn.qkv_bias = MakeBf16({kHq * kDh + 2 * kHkv * kDh}, false, seed++);
      lw.attn.o_proj = MakeBf16({kH, kHq * kDh}, /*nk=*/true, seed++);
      lw.mlp.gate_up_proj = MakeBf16({2 * kI, kH}, /*nk=*/true, seed++);
      lw.mlp.down_proj = MakeBf16({kH, kI}, /*nk=*/true, seed++);
      w.layers.push_back(std::move(lw));
    }
    for (auto& l : w.layers) ApplyArm(l.mlp, arm);
    CachePool pool(kL, /*num_blocks=*/2, kBlock, kHkv, kDh);
    const CommonAttentionMetadata am = PrefillMeta(kT, kBlock);
    vt::Queue q = Q();
    return vllm::Glm4Model::Forward(Tokens(), Positions(), am, pool.attn_kv, w, c, q);
  });
}

// ─── Command-R (CohereForCausalLM) ───────────────────────────────────────────
TEST_CASE("dense gate-up seam: Command-R forward pins the merged gate_up split") {
  HfConfig c = BaseConfig();
  c.model_type = "cohere";
  c.architectures = {"CohereForCausalLM"};
  c.raw["layer_norm_eps"] = 1e-5;
  c.raw["logit_scale"] = 0.0625;

  CheckGateUpSeam("commandr", [&](Arm arm) {
    vllm::CommandrWeights w;
    w.logit_scale = vllm::CommandrLogitScale(c);
    w.embed_tokens = MakeBf16({kV, kH}, false, 1);
    w.final_norm = MakeBf16({kH}, false, 2, 0.5F);
    w.rope_cos_sin = MakeRopeCache(kMaxPos, kDh, c.rope_theta);
    uint32_t seed = 300;
    for (int64_t l = 0; l < kL; ++l) {
      vllm::CommandrLayerWeights lw;
      lw.input_layernorm = MakeBf16({kH}, false, seed++, 0.5F);
      lw.attn.qkv_proj =
          MakeBf16({kHq * kDh + 2 * kHkv * kDh, kH}, /*nk=*/true, seed++);
      lw.attn.o_proj = MakeBf16({kH, kHq * kDh}, /*nk=*/true, seed++);
      lw.mlp.gate_up_proj = MakeBf16({2 * kI, kH}, /*nk=*/true, seed++);
      lw.mlp.down_proj = MakeBf16({kH, kI}, /*nk=*/true, seed++);
      w.layers.push_back(std::move(lw));
    }
    for (auto& l : w.layers) ApplyArm(l.mlp, arm);
    CachePool pool(kL, /*num_blocks=*/2, kBlock, kHkv, kDh);
    const CommonAttentionMetadata am = PrefillMeta(kT, kBlock);
    vt::Queue q = Q();
    return vllm::CommandrModel::Forward(Tokens(), Positions(), am, pool.attn_kv, w, c,
                                        q);
  });
}
