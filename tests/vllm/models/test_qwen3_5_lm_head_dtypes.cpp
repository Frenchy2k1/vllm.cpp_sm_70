// Loader gate for issue #164: the Qwen3.6-27B NVFP4 publishers do not agree on how
// the OUTPUT HEAD is stored, and revisions of a SINGLE repo disagree with each
// other. Measured on the two snapshots we hold (headers read directly):
//
//   unsloth/Qwen3.6-27B-NVFP4 @890bdef7  lm_head.weight BF16    [248320, 5120]
//   unsloth/Qwen3.6-27B-NVFP4 @ccdaab7e  lm_head.weight F8_E4M3 [248320, 5120]
//                                        lm_head.weight_scale BF16 [248320, 1]
//
// @890bdef7 is the snapshot every recorded 27B-NVFP4 benchmark ran on, which is
// why the BF16-only assert survived: the head was never quantized under us until
// the repo was re-quantized. nvidia/Qwen3.6-27B-NVFP4 ships a third form, a
// ModelOpt NVFP4 head (`weight` U8 + `weight_scale` F8 + `weight_scale_2` f32).
//
// These cases pin the DISPATCH, not the kernels: the dequant math is already
// covered by the nvfp4-emulation and fp8 loader tests. Synthetic tensors only —
// no checkpoint, no GPU, so this runs everywhere.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"

using vllm::LoadLmHeadAnyDtype;
using vllm::OwnedTensor;
using vllm::StTensor;

namespace {

// Minimal in-memory stand-in for one resolved safetensors entry. The loader only
// reads dtype/shape/data/nbytes, so a backing vector is enough.
struct Fake {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

class Bag {
 public:
  void Put(const std::string& name, Fake f) { items_[name] = std::move(f); }

  // One STABLE StTensor per name. A single shared view would be a fixture bug:
  // the loader holds `const StTensor& w = get(name)` across the later
  // get(name + "_scale") call, exactly as a real safetensors resolver allows,
  // so resolving the scale must not disturb the weight it already bound.
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
  // std::unordered_map never invalidates references to existing elements on
  // insert, so every returned reference stays valid for the whole load.
  std::unordered_map<std::string, StTensor> views_;
};

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

Fake MakeBf16(const std::vector<int64_t>& shape, const std::vector<float>& vals) {
  Fake f{"BF16", shape, {}};
  f.bytes.resize(vals.size() * 2);
  for (size_t i = 0; i < vals.size(); ++i) {
    const uint16_t h = F32ToBf16(vals[i]);
    std::memcpy(f.bytes.data() + i * 2, &h, 2);
  }
  return f;
}

// e4m3 encode for the small exact powers of two this test uses (no rounding).
uint8_t EncodeE4M3(float v) {
  if (v == 0.0F) return 0;
  const uint8_t sign = v < 0 ? 0x80 : 0x00;
  float a = v < 0 ? -v : v;
  int exp = 0;
  while (a >= 2.0F) { a /= 2.0F; ++exp; }
  while (a < 1.0F) { a *= 2.0F; --exp; }
  const uint8_t biased = static_cast<uint8_t>(exp + 7);
  const uint8_t mant = static_cast<uint8_t>((a - 1.0F) * 8.0F + 0.5F);
  return static_cast<uint8_t>(sign | (biased << 3) | (mant & 0x7U));
}

}  // namespace

TEST_CASE("qwen3_5 lm_head: BF16 head is unchanged (the benchmarked @890bdef7 form)") {
  Bag bag;
  // [out=2, in=4] row-major, transposed to [in=4, out=2].
  bag.Put("lm_head.weight",
          MakeBf16({2, 4}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}));
  const OwnedTensor o = LoadLmHeadAnyDtype(bag.Resolver(), bag.Has(), "lm_head.weight");

  REQUIRE(o.rank == 2);
  CHECK(o.shape[0] == 4);  // in
  CHECK(o.shape[1] == 2);  // out
  const auto* d = reinterpret_cast<const uint16_t*>(o.bytes.data());
  CHECK(Bf16ToF32(d[0]) == doctest::Approx(1.0F));  // [in0,out0]
  CHECK(Bf16ToF32(d[1]) == doctest::Approx(5.0F));  // [in0,out1]
  CHECK(Bf16ToF32(d[6]) == doctest::Approx(4.0F));  // [in3,out0]
  CHECK(Bf16ToF32(d[7]) == doctest::Approx(8.0F));  // [in3,out1]
}

TEST_CASE("qwen3_5 lm_head: FP8 with a PER-OUTPUT-CHANNEL scale (@ccdaab7e form)") {
  Bag bag;
  // weight rows are e4m3 1.0/2.0; the per-row scale differs, which is exactly
  // what a per-tensor reader would get wrong.
  Fake w{"F8_E4M3", {2, 4}, {}};
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 4; ++c) {
      w.bytes.push_back(EncodeE4M3(r == 0 ? 1.0F : 2.0F));
    }
  }
  bag.Put("lm_head.weight", std::move(w));
  bag.Put("lm_head.weight_scale", MakeBf16({2, 1}, {0.5F, 4.0F}));

  const OwnedTensor o = LoadLmHeadAnyDtype(bag.Resolver(), bag.Has(), "lm_head.weight");
  REQUIRE(o.rank == 2);
  CHECK(o.shape[0] == 4);
  CHECK(o.shape[1] == 2);
  const auto* d = reinterpret_cast<const uint16_t*>(o.bytes.data());
  // row0 = 1.0 * 0.5, row1 = 2.0 * 4.0 -> distinct per-channel results.
  CHECK(Bf16ToF32(d[0]) == doctest::Approx(0.5F));
  CHECK(Bf16ToF32(d[1]) == doctest::Approx(8.0F));
  CHECK(Bf16ToF32(d[6]) == doctest::Approx(0.5F));
  CHECK(Bf16ToF32(d[7]) == doctest::Approx(8.0F));
}

TEST_CASE("qwen3_5 lm_head: FP8 with a single per-tensor scale") {
  Bag bag;
  Fake w{"F8_E4M3", {2, 4}, {}};
  for (int i = 0; i < 8; ++i) w.bytes.push_back(EncodeE4M3(2.0F));
  bag.Put("lm_head.weight", std::move(w));
  Fake sc{"F32", {1}, std::vector<uint8_t>(4)};
  const float s = 3.0F;
  std::memcpy(sc.bytes.data(), &s, 4);
  bag.Put("lm_head.weight_scale", std::move(sc));

  const OwnedTensor o = LoadLmHeadAnyDtype(bag.Resolver(), bag.Has(), "lm_head.weight");
  const auto* d = reinterpret_cast<const uint16_t*>(o.bytes.data());
  for (int i = 0; i < 8; ++i) CHECK(Bf16ToF32(d[i]) == doctest::Approx(6.0F));
}

TEST_CASE("qwen3_5 lm_head: an unsupported dtype names itself instead of asserting BF16") {
  Bag bag;
  Fake w{"I32", {2, 4}, std::vector<uint8_t>(32)};
  bag.Put("lm_head.weight", std::move(w));
  // The old code raised "expected BF16 for lm_head.weight" for EVERY quantized
  // head; the message must now name what was actually seen and what is accepted.
  CHECK_THROWS_WITH_AS(
      LoadLmHeadAnyDtype(bag.Resolver(), bag.Has(), "lm_head.weight"),
      doctest::Contains("unsupported dtype 'I32'"), std::runtime_error);
}

TEST_CASE("qwen3_5 lm_head: FP8 without a scale fails loudly, not silently") {
  Bag bag;
  Fake w{"F8_E4M3", {2, 4}, std::vector<uint8_t>(8, EncodeE4M3(1.0F))};
  bag.Put("lm_head.weight", std::move(w));
  CHECK_THROWS_AS(LoadLmHeadAnyDtype(bag.Resolver(), bag.Has(), "lm_head.weight"),
                  std::runtime_error);
}
