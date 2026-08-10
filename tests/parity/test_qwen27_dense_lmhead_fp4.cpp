// PERF-27B-LMHEAD-FP4 (issue #213) — keep the ModelOpt NVFP4 `lm_head` PACKED.
// `nvidia/Qwen3.6-27B-NVFP4` ships a ModelOpt NVFP4 output head (`lm_head.weight`
// U8 + `lm_head.weight_scale` F8_E4M3 + `lm_head.weight_scale_2` f32). The dense
// loader used to DEQUANTIZE it into a bf16 [in,out] Matmul-B owner, so the logits
// GEMM re-read ~2.543 GB every decode step where the packed head is ~0.715 GB.
// vLLM keeps it quantized instead (anchors on `Qwen3_5DenseWeights::lm_head_fp4`,
// qwen3_5_dense.h).
//
// These cases pin the LOADER ROUTING, the NUMERICS and the RESIDENCY of that
// decision. They are synthetic (no checkpoint, no GPU) on purpose: the 235/235
// `test_qwen27_paged_engine` gate runs `unsloth`@890bdef7, whose head is BF16, so
// that gate is BLIND to this path.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::DenseCheckpointHasLmHead;
using vllm::HfConfig;
using vllm::LoadDenseLmHead;
using vllm::ModelRegistry;
using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseModel;
using vllm::Qwen3_5DenseWeights;
using vllm::StTensor;
using vt::DType;

namespace {

// --- an in-memory safetensors stand-in (mirrors test_qwen3_5_lm_head_dtypes) ---

struct Fake {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

class Bag {
 public:
  void Put(const std::string& name, Fake f) { items_[name] = std::move(f); }

  vllm::TensorResolver Resolver() {
    return [this](const std::string& name) -> const StTensor& {
      auto it = items_.find(name);
      REQUIRE_MESSAGE(it != items_.end(), "missing tensor: " << name);
      Fake& f = it->second;
      StTensor& v = views_[name];
      v.dtype = f.dtype;
      v.shape = f.shape;
      v.data = f.bytes.data();
      v.nbytes = f.bytes.size();
      return v;
    };
  }

  std::function<bool(const std::string&)> Has() {
    return [this](const std::string& n) { return items_.count(n) != 0; };
  }

 private:
  std::unordered_map<std::string, Fake> items_;
  std::unordered_map<std::string, StTensor> views_;
};

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

// One deterministic ModelOpt NVFP4 fixture. The nibbles and group scale bytes are
// chosen DIRECTLY (exact E2M1 codes, exact powers-of-two fp8 block scales) so the
// value a correct dequant must produce is exact and known here:
//   w[r][c] = sign * kE2M1Lut[idx] * F8E4M3ToF32(block_scale) * weight_scale_2
// `weight_scale_2` is the ModelOpt convention: the SCALE, not the CT divisor.
constexpr float kFixtureScale2 = 0.125F;

struct Nvfp4Fixture {
  std::vector<uint8_t> packed;  // [N, K/2]
  std::vector<uint8_t> scale;   // [N, K/16]
  std::vector<float> dequant;   // [N, K] the exact expected value
};

Nvfp4Fixture MakeNvfp4Fixture(int64_t n, int64_t k, uint64_t seed) {
  Nvfp4Fixture f;
  f.packed.assign(static_cast<size_t>(n) * static_cast<size_t>(k / 2), 0);
  f.scale.assign(static_cast<size_t>(n) * static_cast<size_t>(k / 16), 0);
  f.dequant.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0.0F);
  // Exact e4m3 powers of two: 0.25, 0.5, 1.0, 2.0.
  const float kBlockScales[4] = {0.25F, 0.5F, 1.0F, 2.0F};
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t g = 0; g < k / 16; ++g) {
      const float bs =
          kBlockScales[Mix(seed + static_cast<uint64_t>(r * 977 + g)) & 3U];
      f.scale[static_cast<size_t>(r) * static_cast<size_t>(k / 16) +
              static_cast<size_t>(g)] = vllm::F32ToF8E4M3(bs);
      for (int64_t j = 0; j < 16; ++j) {
        const int64_t c = g * 16 + j;
        const uint64_t h = Mix(seed + static_cast<uint64_t>(r * 131071 + c));
        const int idx = static_cast<int>(h % 8U);
        const bool neg = ((h >> 8) & 1U) != 0U;
        const float mag = vllm::kE2M1Lut[idx];
        const uint8_t nib = vllm::Fp4ToNibble(neg ? -mag : mag);
        const size_t byte = static_cast<size_t>(r) * static_cast<size_t>(k / 2) +
                            static_cast<size_t>(c / 2);
        // low nibble first (the on-disk E2M1 packing both dequants read).
        if (c % 2 == 0)
          f.packed[byte] = static_cast<uint8_t>(f.packed[byte] | (nib & 0x0FU));
        else
          f.packed[byte] = static_cast<uint8_t>(f.packed[byte] | (nib << 4));
        f.dequant[static_cast<size_t>(r) * static_cast<size_t>(k) +
                  static_cast<size_t>(c)] =
            (neg ? -mag : mag) * bs * kFixtureScale2;
      }
    }
  }
  return f;
}

void PutModelOptHead(Bag& bag, const std::string& proj, int64_t n, int64_t k,
                     const Nvfp4Fixture& f, bool with_input_scale) {
  bag.Put(proj + ".weight", Fake{"U8", {n, k / 2}, f.packed});
  bag.Put(proj + ".weight_scale", Fake{"F8_E4M3", {n, k / 16}, f.scale});
  Fake s2{"F32", {}, std::vector<uint8_t>(4)};
  const float v = kFixtureScale2;
  std::memcpy(s2.bytes.data(), &v, 4);
  bag.Put(proj + ".weight_scale_2", std::move(s2));
  if (with_input_scale) {
    // The gate checkpoint DOES ship `lm_head.input_scale`. Consuming it would flip
    // IsTrueW4A4() and select the W4A4 GEMM vLLM refuses here (modelopt.py:1365).
    Fake is{"F32", {}, std::vector<uint8_t>(4)};
    const float iv = 0.0625F;
    std::memcpy(is.bytes.data(), &iv, 4);
    bag.Put(proj + ".input_scale", std::move(is));
  }
}

// The SAME fp4 bytes under compressed-tensors names: `weight_global_scale` is a
// DIVISOR (1/scale), and the per-Linear `input_global_scale` shipped next to it
// is exactly the activation divisor that must NOT be consumed on an output head.
void PutCtNvfp4Head(Bag& bag, const std::string& proj, int64_t n, int64_t k,
                    const Nvfp4Fixture& f) {
  bag.Put(proj + ".weight_packed", Fake{"U8", {n, k / 2}, f.packed});
  bag.Put(proj + ".weight_scale", Fake{"F8_E4M3", {n, k / 16}, f.scale});
  Fake wgs{"F32", {}, std::vector<uint8_t>(4)};
  const float divisor = 1.0F / kFixtureScale2;  // CT stores the RECIPROCAL
  std::memcpy(wgs.bytes.data(), &divisor, 4);
  bag.Put(proj + ".weight_global_scale", std::move(wgs));
  Fake igs{"F32", {}, std::vector<uint8_t>(4)};
  const float iv = 16.0F;
  std::memcpy(igs.bytes.data(), &iv, 4);
  bag.Put(proj + ".input_global_scale", std::move(igs));
}

// The fixture bytes as an NVFP4 TOWER projection [N=out, K=in].
Nvfp4Weight MakeNvfp4Weight(int64_t n, int64_t k, uint64_t seed) {
  const Nvfp4Fixture f = MakeNvfp4Fixture(n, k, seed);
  Nvfp4Weight w;
  w.n = n;
  w.k = k;
  w.scale2 = kFixtureScale2;
  w.packed.dtype = w.scale.dtype = DType::kI8;
  w.packed.rank = w.scale.rank = 2;
  w.packed.shape[0] = w.scale.shape[0] = n;
  w.packed.shape[1] = k / 2;
  w.scale.shape[1] = k / 16;
  w.packed.bytes.assign(f.packed.begin(), f.packed.end());
  w.scale.bytes.assign(f.scale.begin(), f.scale.end());
  return w;
}

// One TOWER projection [N=out, K=in] under either spelling. `LoadNvfp4AnyNaming`
// has an arm for each — `LoadCtNvfp4Raw` for compressed-tensors, its own body for
// ModelOpt — so a residency opt-in added to one is invisible to a fixture that
// only ever exercises the other.
void PutNvfp4Proj(Bag& bag, const std::string& proj, int64_t n, int64_t k,
                  uint64_t seed, bool ct_naming) {
  const Nvfp4Fixture f = MakeNvfp4Fixture(n, k, seed);
  if (ct_naming) {
    PutCtNvfp4Head(bag, proj, n, k, f);
  } else {
    // ModelOpt ships an `input_scale` next to EVERY projection; a tower
    // projection is where the loader may legally ignore it at the default
    // `VT_MODELOPT_W4A4=0`.
    PutModelOptHead(bag, proj, n, k, f, /*with_input_scale=*/true);
  }
}

uint16_t F32ToBf16(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToF32(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float v = 0.0F;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

Fake MakeBf16(const std::vector<int64_t>& shape, uint64_t seed) {
  int64_t total = 1;
  for (int64_t s : shape) total *= s;
  Fake f{"BF16", shape, std::vector<uint8_t>(static_cast<size_t>(total) * 2)};
  for (int64_t i = 0; i < total; ++i) {
    const uint16_t h = F32ToBf16(RandV(seed + static_cast<uint64_t>(i)));
    std::memcpy(f.bytes.data() + static_cast<size_t>(i) * 2, &h, 2);
  }
  return f;
}

// One whole synthetic dense decoder layer with EVERY routed projection stored
// NVFP4 (ModelOpt spelling) and everything else BF16, under the exact tensor
// names `LoadQwen3_5DenseLayer` resolves. `linear_attention` quantizes the GDN
// `out_proj`, `full_attention` quantizes q/k/v/o_proj, and both quantize the MLP
// — which is every `LoadNvfp4AnyNaming` call site in the dense loader.
void PutNvfp4DenseLayer(Bag& bag, const HfConfig& c, const std::string& type,
                        int64_t idx, bool ct_naming) {
  const std::string base =
      "model.language_model.layers." + std::to_string(idx) + ".";
  const uint64_t s = 3000 + static_cast<uint64_t>(idx) * 700;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  bag.Put(base + "input_layernorm.weight", MakeBf16({H}, s + 1));
  bag.Put(base + "post_attention_layernorm.weight", MakeBf16({H}, s + 2));
  if (type == "linear_attention") {
    const std::string la = base + "linear_attn.";
    // The in-projections are on vLLM's `ignore` list and stay BF16 in the on-disk
    // torch-Linear [out, in] orientation; only `out_proj` is quantized.
    bag.Put(la + "in_proj_qkv.weight", MakeBf16({conv_dim, H}, s + 10));
    bag.Put(la + "in_proj_z.weight", MakeBf16({value_dim, H}, s + 20));
    bag.Put(la + "in_proj_b.weight", MakeBf16({Hv, H}, s + 30));
    bag.Put(la + "in_proj_a.weight", MakeBf16({Hv, H}, s + 40));
    bag.Put(la + "conv1d.weight", MakeBf16({conv_dim, 1, Kw}, s + 50));
    bag.Put(la + "A_log", MakeBf16({Hv}, s + 60));
    bag.Put(la + "dt_bias", MakeBf16({Hv}, s + 70));
    bag.Put(la + "norm.weight", MakeBf16({Dv}, s + 80));
    PutNvfp4Proj(bag, la + "out_proj", H, value_dim, s + 90, ct_naming);
  } else {
    const std::string sa = base + "self_attn.";
    PutNvfp4Proj(bag, sa + "q_proj", Hq * Dh, H, s + 110, ct_naming);
    PutNvfp4Proj(bag, sa + "k_proj", Hkv * Dh, H, s + 120, ct_naming);
    PutNvfp4Proj(bag, sa + "v_proj", Hkv * Dh, H, s + 130, ct_naming);
    PutNvfp4Proj(bag, sa + "o_proj", H, Hq * Dh, s + 140, ct_naming);
    bag.Put(sa + "q_norm.weight", MakeBf16({Dh}, s + 150));
    bag.Put(sa + "k_norm.weight", MakeBf16({Dh}, s + 160));
  }
  const std::string mlp = base + "mlp.";
  PutNvfp4Proj(bag, mlp + "gate_proj", I, H, s + 210, ct_naming);
  PutNvfp4Proj(bag, mlp + "up_proj", I, H, s + 220, ct_naming);
  PutNvfp4Proj(bag, mlp + "down_proj", H, I, s + 230, ct_naming);
}

// EVERY `Nvfp4Weight` a loaded dense layer can own, by struct field rather than
// by loader call site — which is what makes the sweep below survive a loader
// path that does not exist yet.
std::vector<std::pair<std::string, const Nvfp4Weight*>> LayerNvfp4Weights(
    const Qwen3_5DenseLayerWeights& lw) {
  return {
      {"linear_attn.out_proj", &lw.gdn.out_proj_fp4},
      {"self_attn.q_proj", &lw.attn.q_proj_fp4},
      {"self_attn.k_proj", &lw.attn.k_proj_fp4},
      {"self_attn.v_proj", &lw.attn.v_proj_fp4},
      {"self_attn.o_proj", &lw.attn.o_proj_fp4},
      {"mlp.gate_proj", &lw.mlp.gate_proj_fp4},
      {"mlp.up_proj", &lw.mlp.up_proj_fp4},
      {"mlp.down_proj", &lw.mlp.down_proj_fp4},
  };
}

// --- the small synthetic dense model (shape scaffold from
// test_qwen27_dense_forward.cpp; hidden/vocab widened to Marlin-shaped) ---

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 128;  // == the head's K
  c.num_hidden_layers = 2;
  c.vocab_size = 256;  // == the head's N
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

Qwen3_5DenseWeights MakeWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    const int64_t I = c.intermediate_size;
    lw.mlp.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 501);
    lw.mlp.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 502);
    lw.mlp.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 503);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

vt::Queue CpuQ() {
  return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
}

// The bf16 [K,N] Matmul-B owner the OLD loader produced for this head: dequant,
// round to bf16, transpose. Built from the fixture's own exact values, NOT by
// calling either production dequant, so the numerical case below compares against
// an INDEPENDENT reference.
OwnedTensor ReferenceBf16Head(const Nvfp4Fixture& f, int64_t n, int64_t k) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.rank = 2;
  o.shape[0] = k;
  o.shape[1] = n;
  o.bytes.resize(static_cast<size_t>(n) * static_cast<size_t>(k) * 2);
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < n; ++r)
    for (int64_t c = 0; c < k; ++c)
      p[static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(r)] =
          F32ToBf16(f.dequant[static_cast<size_t>(r) * static_cast<size_t>(k) +
                              static_cast<size_t>(c)]);
  return o;
}

}  // namespace

// ── 1. The loader keeps the ModelOpt NVFP4 head PACKED ───────────────────────
TEST_CASE("qwen27 dense lm_head: a ModelOpt NVFP4 head stays PACKED (no bf16 owner)") {
  constexpr int64_t N = 256, K = 128;
  const Nvfp4Fixture f = MakeNvfp4Fixture(N, K, 7);
  Bag bag;
  PutModelOptHead(bag, "lm_head", N, K, f, /*with_input_scale=*/true);

  OwnedTensor bf16;
  Nvfp4Weight fp4;
  LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", bf16, fp4);

  REQUIRE_FALSE(fp4.Empty());
  // The point: NOTHING is materialized to bf16 (the old U8 branch built a [K,N]
  // bf16 owner, ~2.543 GB at the real 248320x5120).
  CHECK(bf16.Empty());
  CHECK(fp4.n == N);
  CHECK(fp4.k == K);
  // Resident bytes: K*N/2 packed E2M1 + K*N/16 fp8-e4m3 block scales.
  CHECK(fp4.packed.bytes.size() == static_cast<size_t>(K) * static_cast<size_t>(N) / 2);
  CHECK(fp4.scale.bytes.size() == static_cast<size_t>(K) * static_cast<size_t>(N) / 16);
  // ModelOpt's weight_scale_2 IS the scale (not the CT divisor).
  CHECK(fp4.scale2 == doctest::Approx(kFixtureScale2));
  // ── 3. and the `lm_head.input_scale` this fixture ships must NOT flip the head
  // to W4A4: vLLM DELETES it on the W4A16 head path (modelopt.py:1365), and
  // consuming it would route the head to the fp4-activation GEMM.
  CHECK_FALSE(fp4.IsTrueW4A4());
  CHECK(fp4.alpha == 0.0F);
}

// ── 3b. The SAME rule under compressed-tensors names ─────────────────────────
// `LoadCtNvfp4Raw` consumes `input_global_scale` UNCONDITIONALLY, correct for a
// TOWER projection of the 27B CT checkpoint, which really is W4A4. An output head
// is not one (modelopt.py:1365): a W4A4 head would take the fp4-activation GEMM
// AND skip the pre-capture Marlin build, which early-returns on IsTrueW4A4().
TEST_CASE("qwen27 dense lm_head: a compressed-tensors NVFP4 head is W4A16, not W4A4") {
  constexpr int64_t N = 256, K = 128;
  const Nvfp4Fixture f = MakeNvfp4Fixture(N, K, 23);
  Bag bag;
  PutCtNvfp4Head(bag, "lm_head", N, K, f);

  // A CT head's ONLY weight tensor is `lm_head.weight_packed`, so a bare
  // `lm_head.weight` probe reads it as `tie_word_embeddings` and computes the
  // logits off the embedding table.
  CHECK(DenseCheckpointHasLmHead(bag.Has(), "lm_head"));
  Bag tied;  // a genuinely tied checkpoint ships no head under either name
  tied.Put("model.language_model.embed_tokens.weight", MakeBf16({8, 4}, 5));
  CHECK_FALSE(DenseCheckpointHasLmHead(tied.Has(), "lm_head"));

  OwnedTensor bf16;
  Nvfp4Weight fp4;
  LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", bf16, fp4);

  REQUIRE_FALSE(fp4.Empty());
  CHECK(bf16.Empty());
  CHECK(fp4.n == N);
  CHECK(fp4.k == K);
  // CT stores the global scale as a DIVISOR; scale2 is its reciprocal.
  CHECK(fp4.scale2 == doctest::Approx(kFixtureScale2));
  // The head must land on the SAME W4A16 dispatcher the ModelOpt spelling takes.
  CHECK(fp4.alpha == 0.0F);
  CHECK_FALSE(fp4.IsTrueW4A4());
}

TEST_CASE("qwen27 dense lm_head: a BF16 head is unaffected (the benchmarked form)") {
  Bag bag;
  bag.Put("lm_head.weight", MakeBf16({8, 4}, 3));

  OwnedTensor bf16;
  Nvfp4Weight fp4;
  LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", bf16, fp4);

  CHECK(fp4.Empty());
  REQUIRE_FALSE(bf16.Empty());
  REQUIRE(bf16.rank == 2);
  CHECK(bf16.shape[0] == 4);  // in
  CHECK(bf16.shape[1] == 8);  // out
}

TEST_CASE("qwen27 dense lm_head: VT_LMHEAD_FP4=0 restores the dequantized owner") {
  constexpr int64_t N = 256, K = 128;
  const Nvfp4Fixture f = MakeNvfp4Fixture(N, K, 13);
  Bag bag;
  PutModelOptHead(bag, "lm_head", N, K, f, /*with_input_scale=*/true);

#ifdef _WIN32
  _putenv_s("VT_LMHEAD_FP4", "0");
#else
  setenv("VT_LMHEAD_FP4", "0", 1);
#endif
  OwnedTensor bf16;
  Nvfp4Weight fp4;
  LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", bf16, fp4);
#ifdef _WIN32
  _putenv_s("VT_LMHEAD_FP4", "");
#else
  unsetenv("VT_LMHEAD_FP4");
#endif

  CHECK(fp4.Empty());
  REQUIRE_FALSE(bf16.Empty());
  CHECK(bf16.shape[0] == K);
  CHECK(bf16.shape[1] == N);
  // The in-binary rollback must reproduce the OLD dequant exactly.
  const auto* p = reinterpret_cast<const uint16_t*>(bf16.bytes.data());
  for (int64_t r = 0; r < N; r += 37)
    for (int64_t c = 0; c < K; c += 11)
      CHECK(Bf16ToF32(p[static_cast<size_t>(c) * static_cast<size_t>(N) +
                        static_cast<size_t>(r)]) ==
            doctest::Approx(f.dequant[static_cast<size_t>(r) * static_cast<size_t>(K) +
                                      static_cast<size_t>(c)]));
}

// ── 2. The packed head computes the SAME logits as the dequantized head ──────
TEST_CASE("qwen27 dense lm_head: packed-head logits match the dequant-then-GEMM reference") {
  const HfConfig c = MakeConfig();
  const int64_t N = c.vocab_size, K = c.hidden_size;
  const Nvfp4Fixture f = MakeNvfp4Fixture(N, K, 17);
  Bag bag;
  PutModelOptHead(bag, "lm_head", N, K, f, /*with_input_scale=*/true);

  OwnedTensor unused_bf16;
  Nvfp4Weight fp4;
  LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", unused_bf16, fp4);
  REQUIRE_FALSE(fp4.Empty());

  const std::vector<int32_t> ids{3, 11, 40, 200};
  const std::vector<int32_t> pos{0, 1, 2, 3};
  vt::Queue q = CpuQ();

  // Reference arm: the bf16 [K,N] owner the OLD loader produced.
  Qwen3_5DenseWeights ref = MakeWeights(c);
  ref.lm_head = ReferenceBf16Head(f, N, K);
  const std::vector<float> want =
      Qwen3_5DenseModel::ForwardDense(ids, pos, ref, c, q);

  // Packed arm: identical model, head kept packed.
  Qwen3_5DenseWeights got_w = MakeWeights(c);
  got_w.lm_head_fp4 = fp4;
  const std::vector<float> got =
      Qwen3_5DenseModel::ForwardDense(ids, pos, got_w, c, q);

  REQUIRE(got.size() == want.size());
  REQUIRE(want.size() == ids.size() * static_cast<size_t>(N));
  double max_abs = 0.0;
  double scale = 1e-6;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(std::isfinite(got[i]));
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i] - want[i])));
    scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
  }
  // Both arms consume the SAME fp4 codes; the only divergence allowed is the
  // reference arm's bf16 rounding plus GEMM accumulation order (Marlin W4A16
  // tolerance band).
  CHECK(max_abs / scale < 2e-2);
}

// ── 5. The packed head's resident is built ONCE, at PREPARE time, for the HEAD
// ── ALONE ────────────────────────────────────────────────────────────────────
// Backends with no fp4 GEMM (CPU here; Vulkan and Metal register neither
// kMatmulNvfp4 nor the Marlin grouped GEMM) fall back to `vt::Matmul` on a
// dequantized bf16 [K,N] operand. Three ways to lose that, none visible to a
// numerical assertion: rebuild it inside the GEMM (2.54 GB per decode step at the
// real 248320x5120); drop the PrepareLmHeadResident call so the forward builds it
// (on CUDA that same call is the PRE-CAPTURE Marlin build); or keep one for EVERY
// NVFP4 weight, quadrupling the tower's steady-state bytes on these same backends
// — the double-residency of issue #203.
TEST_CASE("qwen27 dense lm_head: the packed head's resident is built once, at prepare") {
  const HfConfig c = MakeConfig();
  const int64_t N = c.vocab_size, K = c.hidden_size;
  const Nvfp4Fixture f = MakeNvfp4Fixture(N, K, 37);
  Bag bag;
  PutModelOptHead(bag, "lm_head", N, K, f, /*with_input_scale=*/true);

  OwnedTensor unused_bf16;
  Nvfp4Weight fp4;
  LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", unused_bf16, fp4);
  REQUIRE_FALSE(fp4.Empty());

  Qwen3_5DenseWeights w = MakeWeights(c);
  w.lm_head_fp4 = fp4;
  // An NVFP4 TOWER as well as an NVFP4 head, both on the same fallback.
  const int64_t I = c.intermediate_size;
  for (Qwen3_5DenseLayerWeights& lw : w.layers) {
    lw.mlp.gate_proj_fp4 = MakeNvfp4Weight(I, K, 41);
    lw.mlp.up_proj_fp4 = MakeNvfp4Weight(I, K, 43);
    lw.mlp.down_proj_fp4 = MakeNvfp4Weight(K, I, 47);
  }
  vt::Queue q = CpuQ();

  // The registry `prepare` hook builds it BEFORE any forward runs.
  const std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  REQUIRE(w.lm_head_fp4.d_dequant_b == nullptr);
  ModelRegistry::Prepare(*model, c, q);
  REQUIRE(w.lm_head_fp4.d_dequant_b != nullptr);
  const void* first = w.lm_head_fp4.d_dequant_b.get();

  // ...and no forward rebuilds it.
  const std::vector<int32_t> ids{3, 11}, pos{0, 1};
  (void)Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);
  (void)Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);
  CHECK(w.lm_head_fp4.d_dequant_b.get() == first);

  // ...and the tower ran the same fallback GEMM without acquiring any.
  for (const Qwen3_5DenseLayerWeights& lw : w.layers) {
    CHECK_FALSE(lw.mlp.gate_proj_fp4.keep_dequant_b);
    CHECK(lw.mlp.gate_proj_fp4.d_dequant_b == nullptr);
    CHECK(lw.mlp.up_proj_fp4.d_dequant_b == nullptr);
    CHECK(lw.mlp.down_proj_fp4.d_dequant_b == nullptr);
  }

  // A BF16 head acquires none — the hook is inert off the packed path.
  Qwen3_5DenseWeights bf16_w = MakeWeights(c);
  bf16_w.lm_head = ReferenceBf16Head(f, N, K);
  const std::unique_ptr<vllm::LoadedModel> bf16_model =
      vllm::BorrowQwen3_5DenseLoadedModel(bf16_w);
  ModelRegistry::Prepare(*bf16_model, c, q);
  CHECK(bf16_w.lm_head_fp4.d_dequant_b == nullptr);
}

// ── 8. The opt-in belongs to ONE LOADER, and the LOADER is what proves it ────
// The case above builds its tower with `MakeNvfp4Weight` — direct struct
// construction — so it pins what the FORWARD does with a weight that did not opt
// in, and says nothing about which loader may set the flag. Adding
// `r.keep_dequant_b = true;` to `LoadNvfp4AnyNaming`, the one function every
// dense NVFP4 TOWER projection flows through (MLP gate/up/down via
// `LoadDenseMlp`, attention q/k/v/o via `LoadAttnDense`, GDN `out_proj` via
// `LoadGdnDense`), reopens the whole-tower bf16 expansion of #203 verbatim while
// every case above stays green — and the CUDA gate is blind by construction,
// because `kMatmulNvfp4` is registered there so `ResidentNvfp4DequantB` is never
// reached.
//
// So load a fully-NVFP4 layer of BOTH types, under BOTH namings, through the REAL
// loader, and sweep every `Nvfp4Weight` the layer struct owns. The sweep
// enumerates FIELDS, not call sites, so a fourth opt-in site anywhere in the
// dense loader fails it.
TEST_CASE("qwen27 dense lm_head: LoadDenseLmHead is the ONLY loader that opts a weight in") {
  const HfConfig c = MakeConfig();
  const int64_t N = c.vocab_size, K = c.hidden_size;
  const Nvfp4Fixture f = MakeNvfp4Fixture(N, K, 53);

  for (const bool ct_naming : {false, true}) {
    CAPTURE(ct_naming);
    Bag bag;  // outlives every weight below: the bf16 arms BORROW its bytes
    if (ct_naming) {
      PutCtNvfp4Head(bag, "lm_head", N, K, f);
    } else {
      PutModelOptHead(bag, "lm_head", N, K, f, /*with_input_scale=*/true);
    }
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      PutNvfp4DenseLayer(bag, c, c.layer_types[static_cast<size_t>(l)], l,
                         ct_naming);

    // The head, through the loader that IS allowed to opt in.
    OwnedTensor unused_bf16;
    Nvfp4Weight head;
    LoadDenseLmHead(bag.Resolver(), bag.Has(), "lm_head", unused_bf16, head);
    REQUIRE_FALSE(head.Empty());
    CHECK(head.keep_dequant_b);

    // The tower, through the loaders that are NOT.
    int populated = 0;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const std::string type = c.layer_types[static_cast<size_t>(l)];
      const Qwen3_5DenseLayerWeights lw =
          vllm::LoadQwen3_5DenseLayer(bag.Resolver(), bag.Has(), type, l);
      for (const auto& [name, w] : LayerNvfp4Weights(lw)) {
        // An empty slot would pass the opt-in check for free, so count what the
        // loader actually routed and require the census below: a fixture that
        // stopped producing NVFP4 tower weights must not read as a pass.
        if (w->Empty()) continue;
        ++populated;
        INFO(type << " layer " << l << " " << name);
        CHECK_FALSE(w->keep_dequant_b);
      }
    }
    // linear_attention: gdn.out_proj + mlp gate/up/down = 4.
    // full_attention:   attn q/k/v/o  + mlp gate/up/down = 7.
    CHECK(populated == 11);
  }
}
