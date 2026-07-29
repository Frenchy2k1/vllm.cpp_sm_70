// `QUANT-GGUF-NVFP4` — ggml type 40 (NVFP4) materialization out of a GGUF.
//
// The binding gate is CROSS-FORMAT EQUIVALENCE, not a hand-rolled expectation:
// the two NVFP4 containers we support hold the SAME quantization of the SAME
// Qwen3.6-27B weights, so the GGUF dequant and the already-gated
// compressed-tensors/modelopt dequant (`DequantNvfp4ToBf16`, itself gated in
// test_nvfp4_dequant.cpp against torch semantics) must agree BIT FOR BIT.
//
// Evidence for that claim, measured on the real files (dgx.casa,
// ~/bench/q36-27b-nvfp4.gguf vs ~/bench/q36-27b-nvfp4-vllm/model.safetensors,
// see .agents/specs/gguf-nvfp4-notes.md Sec 5):
//   * the per-16 fp8-e4m3 scale bytes are BYTE-IDENTICAL between containers;
//   * the 4-bit nibbles are the SAME VALUES in a different order (the ggml
//     split-half packing rather than the torch pairwise packing);
//   * the GGUF `<stem>.scale` f32 sidecar is BIT-IDENTICAL to
//     float32(1)/float32(weight_global_scale) on the safetensors side.
// So an exact match is the right bar; an approximate one would hide a layout
// bug, which is the dangerous failure here (a wrong dequant still produces
// plausible logits).
//
// tests/vllm/gguf_nvfp4_goldens.inc carries real bytes from BOTH containers for
// three [4, 512] weight slices, so this gate needs no multi-GB asset and runs in
// CI. The full-tensor sweep over the real files is the asset-gated case at the
// bottom (VLLM_NVFP4_GGUF + VLLM_NVFP4_ST).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

#include "gguf_nvfp4_goldens.inc"

using vllm::DequantGgufRowToBf16;
using vllm::DequantGgufRowToF32;
using vllm::DequantNvfp4ToBf16;
using vllm::GgmlTypeNeedsGlobalScale;

namespace {

constexpr uint32_t kNvfp4 = 40;

float BitsToF32(uint32_t bits) {
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

bool HasGguf(const vllm::GgufFile& g, const std::string& name) {
  for (const vllm::GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

bool HasSt(const vllm::SafetensorsFile& s, const std::string& name) {
  for (const std::string& n : s.Names()) {
    if (n == name) return true;
  }
  return false;
}

}  // namespace

// --- The container gap is DECLARED, not silently defaulted. ---------------
//
// NVFP4 blocks are the one encoding we read whose blocks are NOT
// self-contained: the tensor value needs the per-tensor `<stem>.scale`
// sidecar. Dequantizing without it yields values off by a per-tensor factor of
// ~1e4 — finite, plausible-looking, and completely wrong. The 3-argument entry
// point must therefore REFUSE type 40 rather than assume 1.0.
TEST_CASE("gguf nvfp4: the scale-less entry point refuses type 40") {
  CHECK(GgmlTypeNeedsGlobalScale(kNvfp4));
  CHECK_FALSE(GgmlTypeNeedsGlobalScale(8));   // Q8_0 carries its own f16 scale
  CHECK_FALSE(GgmlTypeNeedsGlobalScale(12));  // Q4_K likewise
  CHECK_FALSE(GgmlTypeNeedsGlobalScale(0));   // F32

  std::vector<uint8_t> b(36, 0);
  CHECK_THROWS_AS(DequantGgufRowToF32(kNvfp4, b.data(), 64), std::runtime_error);
  CHECK_THROWS_AS(DequantGgufRowToBf16(kNvfp4, b.data(), 64), std::runtime_error);

  // And the converse: a self-contained encoding must not silently swallow a
  // global scale a caller passed by mistake.
  std::vector<uint8_t> q8(34, 0);
  CHECK_THROWS_AS(DequantGgufRowToF32(8, q8.data(), 32, 2.0F),
                  std::runtime_error);
  CHECK_NOTHROW(DequantGgufRowToF32(8, q8.data(), 32, 1.0F));
}

// --- Hand-computed single block: the ggml split-half nibble order. ---------
//
// block_nvfp4 = uint8 d[4] (one fp8-e4m3 scale per 16-element sub-block), then
// uint8 qs[32]. Sub-block s owns qs[s*8 .. s*8+8); within it, byte j holds
// element j in the LOW nibble and element j+8 in the HIGH nibble. That split is
// the whole difference from the torch pairwise packing, and getting it wrong
// permutes weights inside each group of 16 while keeping the value histogram
// (and therefore the norms) intact — which is why this case pins exact
// positions, not statistics.
TEST_CASE("gguf nvfp4: one block, hand-placed nibbles and sub-block scales") {
  std::vector<uint8_t> blk(36, 0);
  // Sub-block fp8-e4m3 scales: 0x38 = 1.0, 0x3C = 1.5, 0x40 = 2.0, 0x30 = 0.5.
  blk[0] = 0x38; blk[1] = 0x3C; blk[2] = 0x40; blk[3] = 0x30;
  // Sub-block 0, byte 0: low nibble 0x2 (e2m1 1.0) -> element 0,
  //                      high nibble 0xD (sign|0x5 -> -3.0) -> element 8.
  blk[4] = 0xD2;
  // Sub-block 0, byte 7: low 0x7 (6.0) -> element 7, high 0x0 (0.0) -> element 15.
  blk[11] = 0x07;
  // Sub-block 2, byte 3: low 0x4 (2.0) -> element 32+3, high 0xC (-2.0) -> 32+11.
  blk[4 + 16 + 3] = 0xC4;
  // Sub-block 3, byte 5: low 0x6 (4.0) -> element 48+5, high 0x1 (0.5) -> 48+13.
  blk[4 + 24 + 5] = 0x16;

  const float global = 3.0F;
  const std::vector<float> y = DequantGgufRowToF32(kNvfp4, blk.data(), 64, global);
  REQUIRE(y.size() == 64);

  CHECK(y[0] == doctest::Approx(1.0F * 1.0F * global));    // sub 0, scale 1.0
  CHECK(y[8] == doctest::Approx(-3.0F * 1.0F * global));
  CHECK(y[7] == doctest::Approx(6.0F * 1.0F * global));
  CHECK(y[15] == doctest::Approx(0.0F));
  CHECK(y[35] == doctest::Approx(2.0F * 2.0F * global));   // sub 2, scale 2.0
  CHECK(y[43] == doctest::Approx(-2.0F * 2.0F * global));
  CHECK(y[53] == doctest::Approx(4.0F * 0.5F * global));   // sub 3, scale 0.5
  CHECK(y[61] == doctest::Approx(0.5F * 0.5F * global));
  // Everything else is a zero nibble.
  for (int i = 0; i < 64; ++i) {
    if (i == 0 || i == 7 || i == 8 || i == 35 || i == 43 || i == 53 || i == 61) {
      continue;
    }
    CHECK(y[i] == 0.0F);
  }
}

// --- THE GATE: cross-format equivalence on real Qwen3.6-27B-NVFP4 bytes. ---
TEST_CASE("gguf nvfp4: real GGUF blocks dequant bit-identically to the "
          "compressed-tensors path") {
  for (const auto& c : gguf_nvfp4_goldens::kCases) {
    INFO("case " << std::string(c.name));
    const int64_t numel = c.out_dim * c.in_dim;
    const float global = BitsToF32(c.global_scale_bits);
    REQUIRE(global > 0.0F);

    const std::vector<uint16_t> from_gguf =
        DequantGgufRowToBf16(kNvfp4, c.gguf, numel, global);
    REQUIRE(from_gguf.size() == static_cast<size_t>(numel));

    std::vector<uint16_t> from_st(static_cast<size_t>(numel), 0);
    DequantNvfp4ToBf16(c.st_packed, c.st_scales, global, c.out_dim, c.in_dim,
                       from_st.data());

    // Bit-for-bit, not approximately: identical inputs through identical
    // arithmetic. A permutation bug shows up as a large count here, and a
    // scale-decode bug as a total mismatch.
    size_t diff = 0;
    for (size_t i = 0; i < from_gguf.size(); ++i) {
      if (from_gguf[i] != from_st[i]) ++diff;
    }
    CHECK(diff == 0);

    // The slice must not be degenerate, or "all equal" would be vacuous.
    size_t nonzero = 0;
    for (uint16_t v : from_gguf) {
      if ((v & 0x7FFFU) != 0) ++nonzero;
    }
    CHECK(nonzero > static_cast<size_t>(numel) / 2);
  }
}

// --- The f32 and bf16 entry points agree (the bf16 one is the rounded f32). --
TEST_CASE("gguf nvfp4: bf16 entry point is the rounded f32 entry point") {
  const auto& c = gguf_nvfp4_goldens::kCases[0];
  const int64_t numel = c.out_dim * c.in_dim;
  const float global = BitsToF32(c.global_scale_bits);
  const std::vector<float> f32 =
      DequantGgufRowToF32(kNvfp4, c.gguf, numel, global);
  const std::vector<uint16_t> bf16 =
      DequantGgufRowToBf16(kNvfp4, c.gguf, numel, global);
  REQUIRE(f32.size() == bf16.size());
  for (size_t i = 0; i < f32.size(); ++i) {
    REQUIRE(bf16[i] == vt::F32ToBF16(f32[i]));
  }
}

// --- Guards. ---------------------------------------------------------------
TEST_CASE("gguf nvfp4: geometry and scale guards") {
  std::vector<uint8_t> b(36 * 2, 0);
  // numel must be a whole number of 64-element blocks.
  CHECK_THROWS_AS(DequantGgufRowToF32(kNvfp4, b.data(), 63, 1.0F),
                  std::runtime_error);
  CHECK_THROWS_AS(DequantGgufRowToF32(kNvfp4, b.data(), 100, 1.0F),
                  std::runtime_error);
  // A non-finite or non-positive global scale is a loader bug, not a weight.
  CHECK_THROWS_AS(DequantGgufRowToF32(kNvfp4, b.data(), 64, 0.0F),
                  std::runtime_error);
  CHECK_THROWS_AS(DequantGgufRowToF32(kNvfp4, b.data(), 64, -1.0F),
                  std::runtime_error);
  // The reader's traits and this decoder must agree on the block geometry.
  CHECK(vllm::GgmlTraits(kNvfp4).block_elems == 64);
  CHECK(vllm::GgmlTraits(kNvfp4).block_bytes == 36);
}

// ===========================================================================
// THE `C` COLUMN GATE: the ggml blocks REPACK, byte for byte, into the operand
// pair the NVFP4 GEMMs consume.
//
// Column `C` (native quantized compute) is entirely this permutation: if the
// repacked (weight_packed, weight_scale) streams are BIT-IDENTICAL to the ones
// the compressed-tensors container of the same quantization run stores, then a
// GGUF weight entering vt::MatmulNvfp4* is entering it with operands the fp4
// kernels are already gated on, and no numerics are re-litigated. The same
// goldens the value-level gate above uses carry both containers' bytes, so this
// runs in CI with no asset. See .agents/specs/gguf-nvfp4-native-compute.md.
// ===========================================================================
TEST_CASE("gguf nvfp4 repack: ggml blocks -> the fp4 GEMM operand pair, "
          "byte-identical to the compressed-tensors container") {
  for (const auto& c : gguf_nvfp4_goldens::kCases) {
    INFO("case " << std::string(c.name));
    std::vector<uint8_t> packed(static_cast<size_t>(c.out_dim * c.in_dim / 2));
    std::vector<uint8_t> scale(static_cast<size_t>(c.out_dim * c.in_dim / 16));
    vllm::RepackGgufNvfp4Rows(c.gguf, c.out_dim, c.in_dim, packed.data(),
                              scale.data());

    // Count rather than compare containers directly: doctest stringifies both
    // operands of a reported comparison.
    size_t pdiff = 0;
    for (size_t i = 0; i < packed.size(); ++i) {
      if (packed[i] != c.st_packed[i]) ++pdiff;
    }
    size_t sdiff = 0;
    for (size_t i = 0; i < scale.size(); ++i) {
      if (scale[i] != c.st_scales[i]) ++sdiff;
    }
    CHECK(pdiff == 0);
    CHECK(sdiff == 0);

    // Not vacuous: the slice must actually carry nibbles and scales.
    size_t nonzero_packed = 0;
    for (uint8_t v : packed) {
      if (v != 0) ++nonzero_packed;
    }
    CHECK(nonzero_packed > packed.size() / 2);
  }
}

// The gate must DISCRIMINATE THE LAYOUT, not merely the symbol. The dangerous
// bug here is the same one the value-level gate was built against: assuming the
// GGUF already packs nibbles pairwise (i.e. copying qs[] straight through)
// preserves every value and every scale, so it produces finite, plausible
// logits. Run that wrong repack deliberately and require the comparison above
// to reject it.
TEST_CASE("gguf nvfp4 repack: the straight-through (wrong) packing is "
          "REJECTED") {
  const auto& c = gguf_nvfp4_goldens::kCases[0];
  const int64_t nblocks = c.in_dim / 64;
  std::vector<uint8_t> wrong(static_cast<size_t>(c.out_dim * c.in_dim / 2));
  for (int64_t r = 0; r < c.out_dim; ++r) {
    for (int64_t b = 0; b < nblocks; ++b) {
      // qs[32] copied verbatim: right values, ggml's split-half order retained.
      std::memcpy(wrong.data() + (r * c.in_dim + b * 64) / 2,
                  c.gguf + (r * nblocks + b) * 36 + 4, 32);
    }
  }
  size_t diff = 0;
  for (size_t i = 0; i < wrong.size(); ++i) {
    if (wrong[i] != c.st_packed[i]) ++diff;
  }
  CHECK(diff > wrong.size() / 4);
}

// The repack and the dequant must agree about the SAME bytes: decoding the
// repacked operands through the already-gated compressed-tensors decoder must
// reproduce the GGUF decoder's values exactly. This ties the new primitive to
// the value-level gate instead of letting the two drift.
TEST_CASE("gguf nvfp4 repack: repacked operands dequant to the GGUF values") {
  for (const auto& c : gguf_nvfp4_goldens::kCases) {
    INFO("case " << std::string(c.name));
    const int64_t numel = c.out_dim * c.in_dim;
    const float global = BitsToF32(c.global_scale_bits);
    std::vector<uint8_t> packed(static_cast<size_t>(numel / 2));
    std::vector<uint8_t> scale(static_cast<size_t>(numel / 16));
    vllm::RepackGgufNvfp4Rows(c.gguf, c.out_dim, c.in_dim, packed.data(),
                              scale.data());
    std::vector<uint16_t> via_repack(static_cast<size_t>(numel), 0);
    DequantNvfp4ToBf16(packed.data(), scale.data(), global, c.out_dim, c.in_dim,
                       via_repack.data());
    const std::vector<uint16_t> direct =
        DequantGgufRowToBf16(kNvfp4, c.gguf, numel, global);
    size_t diff = 0;
    for (size_t i = 0; i < direct.size(); ++i) {
      if (direct[i] != via_repack[i]) ++diff;
    }
    CHECK(diff == 0);
  }
}

TEST_CASE("gguf nvfp4 repack: geometry guards") {
  std::vector<uint8_t> src(36 * 2, 0);
  std::vector<uint8_t> packed(64, 0);
  std::vector<uint8_t> scale(8, 0);
  // K must be a whole number of 64-element NVFP4 blocks.
  CHECK_THROWS_AS(vllm::RepackGgufNvfp4Rows(src.data(), 1, 63, packed.data(),
                                            scale.data()),
                  std::runtime_error);
  CHECK_THROWS_AS(vllm::RepackGgufNvfp4Rows(src.data(), 1, 32, packed.data(),
                                            scale.data()),
                  std::runtime_error);
  CHECK_THROWS_AS(vllm::RepackGgufNvfp4Rows(nullptr, 1, 64, packed.data(),
                                            scale.data()),
                  std::runtime_error);
  // 128 elements = 2 blocks is fine and fills both outputs.
  std::vector<uint8_t> src2(72, 0xABU);
  std::vector<uint8_t> p2(64, 0), s2(8, 0);
  vllm::RepackGgufNvfp4Rows(src2.data(), 1, 128, p2.data(), s2.data());
  CHECK(p2[0] != 0);
  CHECK(s2[0] == 0xABU);
}

// --- Asset-gated: the whole real files, every NVFP4 tensor they share. ------
//
// VLLM_NVFP4_GGUF=~/bench/q36-27b-nvfp4.gguf
// VLLM_NVFP4_ST=~/bench/q36-27b-nvfp4-vllm/model.safetensors
// Without both, this case is a no-op so CI stays asset-free.
TEST_CASE("gguf nvfp4: full-tensor cross-format sweep on the real files") {
  const char* gguf_path = std::getenv("VLLM_NVFP4_GGUF");
  const char* st_path = std::getenv("VLLM_NVFP4_ST");
  if (gguf_path == nullptr || st_path == nullptr) return;  // asset-gated

  vllm::GgufFile g = vllm::GgufFile::Open(gguf_path);
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(st_path);

  // Two reasons this is not simply "every NVFP4 tensor":
  //  * the GGUF quantizes MORE projections than this checkpoint does (the
  //    recipe ignores the GDN in_proj_* family, which the GGUF still stores as
  //    NVFP4), so only the shared projections have a counterpart at all;
  //  * blk.N.ssm_out is shared but additionally carries the GGUF v-head TILING
  //    (tiled head r*num_k + k == HF grouped head k*num_v_per_k + r, verified
  //    on layer 0 of this file), which the loader undoes with ReorderVCols.
  //    Comparing it RAW would fail for a layout reason that has nothing to do
  //    with the NVFP4 container, so it is deliberately out of this sweep.
  struct Pair { const char* gguf; const char* st; };
  const Pair kPairs[] = {
      {"blk.%d.ffn_gate.weight", "model.language_model.layers.%d.mlp.gate_proj"},
      {"blk.%d.ffn_up.weight", "model.language_model.layers.%d.mlp.up_proj"},
      {"blk.%d.ffn_down.weight", "model.language_model.layers.%d.mlp.down_proj"},
  };

  int compared = 0;
  for (int layer = 0; layer < 64; ++layer) {
    for (const Pair& p : kPairs) {
      char gname[128];
      char sname[192];
      std::snprintf(gname, sizeof(gname), p.gguf, layer);
      std::snprintf(sname, sizeof(sname), p.st, layer);
      const std::string packed = std::string(sname) + ".weight_packed";
      if (!HasGguf(g, gname) || !HasSt(st, packed)) continue;

      const vllm::GgufTensorInfo& t = g.Get(gname);
      if (t.ggml_type != kNvfp4) continue;
      const int64_t in_dim = t.shape[1];
      const int64_t out_dim = t.shape[0];

      const vllm::StTensor& sp = st.Get(packed);
      const vllm::StTensor& ss = st.Get(std::string(sname) + ".weight_scale");
      const vllm::StTensor& sg =
          st.Get(std::string(sname) + ".weight_global_scale");
      if (sp.shape.size() != 2 || sp.shape[0] != out_dim ||
          sp.shape[1] * 2 != in_dim) {
        continue;  // a differently-shaped export; not this gate's business
      }
      float wgs = 0.0F;
      std::memcpy(&wgs, sg.data, sizeof(wgs));
      REQUIRE(wgs > 0.0F);

      const vllm::GgufTensorInfo& sc = g.Get(
          std::string(gname).substr(0, std::string(gname).size() - 7) +
          ".scale");
      float global = 0.0F;
      std::memcpy(&global, sc.data, sizeof(global));

      // The sidecar IS the reciprocal of the safetensors divisor, bit for bit.
      const float inv = 1.0F / wgs;
      REQUIRE(std::memcmp(&global, &inv, sizeof(float)) == 0);

      const std::vector<uint16_t> from_gguf =
          DequantGgufRowToBf16(kNvfp4, t.data, out_dim * in_dim, global);
      std::vector<uint16_t> from_st(static_cast<size_t>(out_dim * in_dim), 0);
      DequantNvfp4ToBf16(sp.data, ss.data, global, out_dim, in_dim,
                         from_st.data());
      // Count, never compare the vectors directly: doctest stringifies both
      // operands of a failing (or -s reported) comparison, and these hold tens
      // of millions of elements.
      size_t diff = 0;
      for (size_t i = 0; i < from_gguf.size(); ++i) {
        if (from_gguf[i] != from_st[i]) ++diff;
      }
      INFO("tensor " << std::string(gname) << " numel " << (out_dim * in_dim));
      REQUIRE(diff == 0);
      ++compared;
    }
  }
  MESSAGE("compared " << compared << " NVFP4 tensors across both containers");
  CHECK(compared > 0);
}

// ===========================================================================
// The MoE path: ONE sidecar scale PER EXPERT.
//
// A stacked expert tensor's `<stem>.scale` holds E scalars (256 on the real
// 35B A3B), so the loader must dequantize one expert slab per scale. The bug
// this rules out is the easy one: applying scale[0] to every expert. The
// fixture gives all three experts BYTE-IDENTICAL blocks and scales 1, 2, 4, so
// a correct load yields experts that are exact 1x/2x/4x multiples of each
// other, and a scale[0]-for-everyone load yields three identical experts.
// ===========================================================================

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

constexpr uint32_t kF32Type = 0;

std::string LeF32(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  std::string s(4, '\0');
  for (int k = 0; k < 4; ++k) s[k] = static_cast<char>((bits >> (8 * k)) & 0xFF);
  return s;
}

std::string F32Blob(int64_t n, float base) {
  std::string s;
  for (int64_t i = 0; i < n; ++i) {
    s += LeF32(base + 0.125F * static_cast<float>(i % 11));
  }
  return s;
}

// Deterministic NVFP4 blocks. Scale bytes are held inside 0x30..0x47 so no
// sub-block scale is a NaN encoding (0x7F/0xFF) or a subnormal.
std::string Nvfp4Blocks(int64_t rows, int64_t k, uint32_t seed) {
  REQUIRE(k % 64 == 0);
  std::string out;
  uint32_t s = seed | 1U;
  auto next = [&s]() {
    s = s * 1664525U + 1013904223U;
    return (s >> 16) & 0xFFU;
  };
  for (int64_t b = 0; b < rows * (k / 64); ++b) {
    for (int i = 0; i < 4; ++i) {
      out.push_back(static_cast<char>(0x30 + (next() % 0x18)));
    }
    for (int i = 0; i < 32; ++i) out.push_back(static_cast<char>(next()));
  }
  return out;
}

void AddF32Torch(gguf_test::GgufModelBuilder& b, const std::string& name,
                 const std::vector<int64_t>& torch_shape, float base) {
  std::vector<uint64_t> dims;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it) {
    dims.push_back(static_cast<uint64_t>(*it));
  }
  int64_t n = 1;
  for (int64_t d : torch_shape) n *= d;
  b.AddTensor(name, dims, kF32Type, F32Blob(n, base));
}

struct MoeFixture {
  int64_t H = 64, vocab = 32, n_head = 2, n_head_kv = 1, head_dim = 32;
  int64_t E = 3, used = 2, I = 64, Is = 64, n_layer = 1;
  std::vector<float> expert_scales{1.0F, 2.0F, 4.0F};
  // One expert's blocks, repeated E times, so only the scale distinguishes them.
  std::string one_expert_gate, one_expert_up, one_expert_down;
};

std::string BuildMoeNvfp4Gguf(MoeFixture& d) {
  using gguf_test::F32Kv;
  using gguf_test::StrKv;
  using gguf_test::U32Kv;
  gguf_test::GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35moe"));
  b.AddKv(U32Kv("qwen35moe.embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(U32Kv("qwen35moe.block_count", static_cast<uint32_t>(d.n_layer)));
  b.AddKv(U32Kv("qwen35moe.attention.head_count",
                static_cast<uint32_t>(d.n_head)));
  b.AddKv(U32Kv("qwen35moe.attention.head_count_kv",
                static_cast<uint32_t>(d.n_head_kv)));
  b.AddKv(U32Kv("qwen35moe.attention.key_length",
                static_cast<uint32_t>(d.head_dim)));
  b.AddKv(U32Kv("qwen35moe.expert_count", static_cast<uint32_t>(d.E)));
  b.AddKv(U32Kv("qwen35moe.expert_used_count", static_cast<uint32_t>(d.used)));
  b.AddKv(U32Kv("qwen35moe.expert_feed_forward_length",
                static_cast<uint32_t>(d.I)));
  b.AddKv(U32Kv("qwen35moe.expert_shared_feed_forward_length",
                static_cast<uint32_t>(d.Is)));
  b.AddKv(F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(U32Kv("qwen35moe.full_attention_interval", 1));
  b.AddKv(U32Kv("qwen35moe.context_length", 4096));

  AddF32Torch(b, "token_embd.weight", {d.vocab, d.H}, 0.5F);
  AddF32Torch(b, "output_norm.weight", {d.H}, 1.5F);
  AddF32Torch(b, "output.weight", {d.vocab, d.H}, 0.25F);

  d.one_expert_gate = Nvfp4Blocks(d.I, d.H, 4242);
  d.one_expert_up = Nvfp4Blocks(d.I, d.H, 5353);
  d.one_expert_down = Nvfp4Blocks(d.H, d.I, 6464);

  const std::string p = "blk.0.";
  AddF32Torch(b, p + "attn_norm.weight", {d.H}, 1.25F);
  AddF32Torch(b, p + "post_attention_norm.weight", {d.H}, 1.75F);
  AddF32Torch(b, p + "attn_q.weight", {d.n_head * d.head_dim, d.H}, 0.1F);
  AddF32Torch(b, p + "attn_k.weight", {d.n_head_kv * d.head_dim, d.H}, 0.2F);
  AddF32Torch(b, p + "attn_v.weight", {d.n_head_kv * d.head_dim, d.H}, 0.3F);
  AddF32Torch(b, p + "attn_output.weight", {d.H, d.n_head * d.head_dim}, 0.4F);
  AddF32Torch(b, p + "attn_q_norm.weight", {d.head_dim}, 1.5F);
  AddF32Torch(b, p + "attn_k_norm.weight", {d.head_dim}, 1.5F);
  AddF32Torch(b, p + "ffn_gate_inp.weight", {d.E, d.H}, 0.05F);
  AddF32Torch(b, p + "ffn_gate_inp_shexp.weight", {d.H}, 0.75F);

  struct Stack { const char* name; const std::string* blocks;
                 int64_t out_dim, in_dim; };
  const Stack stacks[] = {
      {"ffn_gate_exps.weight", &d.one_expert_gate, d.I, d.H},
      {"ffn_up_exps.weight", &d.one_expert_up, d.I, d.H},
      {"ffn_down_exps.weight", &d.one_expert_down, d.H, d.I},
  };
  for (const Stack& s : stacks) {
    std::string stacked;
    for (int64_t e = 0; e < d.E; ++e) stacked += *s.blocks;
    b.AddTensor(p + s.name,
                {static_cast<uint64_t>(s.in_dim),
                 static_cast<uint64_t>(s.out_dim),
                 static_cast<uint64_t>(d.E)},
                kNvfp4, stacked);
    std::string sc;
    for (int64_t e = 0; e < d.E; ++e) {
      sc += LeF32(d.expert_scales[static_cast<size_t>(e)]);
    }
    b.AddTensor(p + std::string(s.name).substr(
                    0, std::string(s.name).size() - 7) + ".scale",
                {static_cast<uint64_t>(d.E)}, kF32Type, sc);
  }

  AddF32Torch(b, p + "ffn_gate_shexp.weight", {d.Is, d.H}, 0.15F);
  AddF32Torch(b, p + "ffn_up_shexp.weight", {d.Is, d.H}, 0.25F);
  AddF32Torch(b, p + "ffn_down_shexp.weight", {d.H, d.Is}, 0.35F);
  return b.Build();
}

}  // namespace

TEST_CASE("gguf nvfp4: a stacked expert tensor uses ITS OWN per-expert scale") {
  MoeFixture d;
  const gguf_test::TempFile f(BuildMoeNvfp4Gguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  const vllm::GgufLoadPolicy expand;  // everything expands; no keep-quant
  const vllm::Qwen3_5MoeWeights w = vllm::LoadQwen3_5MoeFromGguf(g, c, &expand);
  const auto& moe = w.layers[0].moe;

  struct Stack {
    const std::vector<vllm::OwnedTensor>* got;
    const std::string* blocks;
    int64_t out_dim, in_dim;
    const char* name;
  };
  const Stack stacks[] = {
      {&moe.expert_gate, &d.one_expert_gate, d.I, d.H, "ffn_gate_exps"},
      {&moe.expert_up, &d.one_expert_up, d.I, d.H, "ffn_up_exps"},
      {&moe.expert_down, &d.one_expert_down, d.H, d.I, "ffn_down_exps"},
  };

  for (const Stack& s : stacks) {
    INFO("stack " << std::string(s.name));
    REQUIRE(s.got->size() == static_cast<size_t>(d.E));
    const int64_t numel = s.out_dim * s.in_dim;

    for (int64_t e = 0; e < d.E; ++e) {
      INFO("expert " << e);
      const float scale = d.expert_scales[static_cast<size_t>(e)];
      // Reference: this expert's OWN slab bytes with this expert's OWN scale,
      // transposed to the loader's [in, out] Matmul-B layout.
      const std::vector<uint16_t> ref = DequantGgufRowToBf16(
          kNvfp4, reinterpret_cast<const uint8_t*>(s.blocks->data()), numel,
          scale);
      const vllm::OwnedTensor& got = (*s.got)[static_cast<size_t>(e)];
      REQUIRE(got.dtype == vt::DType::kBF16);
      REQUIRE(got.shape[0] == s.in_dim);
      REQUIRE(got.shape[1] == s.out_dim);
      const auto* gp = reinterpret_cast<const uint16_t*>(got.bytes.data());
      size_t diff = 0;
      for (int64_t o = 0; o < s.out_dim; ++o) {
        for (int64_t i = 0; i < s.in_dim; ++i) {
          if (gp[i * s.out_dim + o] != ref[static_cast<size_t>(o * s.in_dim + i)]) {
            ++diff;
          }
        }
      }
      CHECK(diff == 0);
    }

    // The experts share their blocks and differ only in scale, so a correct
    // load makes expert 1 exactly 2x expert 0 and expert 2 exactly 4x. Powers
    // of two are bf16-exact, so this is an equality, and it FAILS LOUDLY if the
    // loader ever reuses scale[0] (all three would be identical instead).
    const auto* e0 = reinterpret_cast<const uint16_t*>(
        (*s.got)[0].bytes.data());
    const auto* e1 = reinterpret_cast<const uint16_t*>(
        (*s.got)[1].bytes.data());
    const auto* e2 = reinterpret_cast<const uint16_t*>(
        (*s.got)[2].bytes.data());
    size_t scaled_ok = 0;
    size_t distinct = 0;
    for (int64_t i = 0; i < numel; ++i) {
      const float v0 = vt::BF16ToF32(e0[i]);
      if (vt::BF16ToF32(e1[i]) == 2.0F * v0 &&
          vt::BF16ToF32(e2[i]) == 4.0F * v0) {
        ++scaled_ok;
      }
      if (e1[i] != e0[i]) ++distinct;
    }
    CHECK(scaled_ok == static_cast<size_t>(numel));
    CHECK(distinct > static_cast<size_t>(numel) / 2);
  }
}
