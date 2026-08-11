// PERF-27B-DENSE-MARLIN-GATEUP (issue #365, spec
// .agents/specs/perf-27b-dense-marlin-gateup.md) — the DECISIONS that route the
// 27B's dense W4A16 MLP through the fused Marlin gate_up pair.
//
// WHAT THIS FILE CAN AND CANNOT PIN, stated plainly. The fused gate_up GEMM is
// `vt::MoeGroupedGemmNvfp4Marlin` / `vt::MarlinDenseGemm` over an N-concatenated
// [2N,K] NVFP4 operand: CUDA-only, and only in a build that has the vendored
// Marlin NVFP4 kernel (VT_MARLIN_NVFP4). Its ARITHMETIC therefore cannot execute
// here, and this file does not pretend to pin numbers. What it pins is every
// decision that is made BEFORE the kernel and that decides whether the kernel is
// legal to use at all:
//
//   * the shape/scale PRECONDITION that makes N-concatenating gate and up into
//     one operand legal — asserted term by term, including the `scale2` equality
//     that the merged single global scale depends on (spec §4/§6: the fusion is
//     unreachable, not silently relaxable, if the two shards disagree);
//   * the row's toggle and its DEFAULT, which is now ON — spec §3.3 made the
//     flip conditional on a same-binary A/B, and that A/B measured a win at
//     both c1 and c8 with complete separation and identical tokens;
//   * that a device/build with no Marlin NVFP4 op registered NEVER selects the
//     fused path, so every non-CUDA backend keeps the split pair byte-for-byte;
//   * that the fused pair resident is the EXISTING one, keyed on the gate
//     weight's own `resident_marlin_pair` slot — a dense gate weight keys it
//     exactly like a shared-expert gate weight, so there is no second cache.
//
// The numeric bar (fused == split within the split-K reduce's grouping, and
// token-exactness against the pinned oracle) belongs to the GPU box. Note that
// tests/vllm/model_executor/layers/test_linear_method.cpp:202 already records,
// from a measured run, that the fused and split Marlin paths are NOT
// bit-identical — Marlin's fp32 split-K reduce groups the K-slices differently
// for a [2N,K] operand than for two [N,K] operands — so "byte-identical to the
// split path" is the wrong bar and this file does not assert it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/device.h"

namespace {

using vllm::DenseMlpWeights;
using vllm::Nvfp4Weight;
using vt::DType;

namespace dn = vllm::dense_nvfp4;

// A non-empty NVFP4 W4A16 weight (alpha == 0 => IsTrueW4A4() false), shaped like
// one half of a dense MLP gate_up: [N=intermediate, K=hidden]. The bytes are
// never read here — only the shape/scale fields the precondition inspects are.
Nvfp4Weight MakeW4A16(int64_t N, int64_t K, float scale2) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.scale2 = scale2;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 2), 0);
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 16;
  w.scale.bytes.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 16), 0);
  return w;
}

// Qwen3.6-27B dense MLP: hidden 5120, intermediate 25600.
constexpr int64_t kN = 25600;
constexpr int64_t kK = 5120;

}  // namespace

TEST_CASE("dense gate_up fusion: the pair precondition is asserted, not assumed") {
  const Nvfp4Weight gate = MakeW4A16(kN, kK, 0.5F);

  SUBCASE("a matching W4A16 pair is fusable") {
    const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F);
    CHECK(dn::GateUpPairFusableShape(gate, up));
  }

  SUBCASE("UNEQUAL scale2 is NOT fusable") {
    // The load-bearing term. The merged resident emits ONE global scale over
    // both shards (vLLM's merged parameter has exactly one weight_global_scale);
    // two shards with different scale2 cannot share it, so relaxing this would
    // change numerics rather than fuse them. Spec §6 stop condition.
    const Nvfp4Weight up = MakeW4A16(kN, kK, 0.25F);
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, up));
  }

  SUBCASE("a scale2 that differs only in the last bit is NOT fusable") {
    const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F + 6e-8F);
    REQUIRE(up.scale2 != gate.scale2);
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, up));
  }

  SUBCASE("an empty half is NOT fusable") {
    const Nvfp4Weight empty;
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, empty));
    CHECK_FALSE(dn::GateUpPairFusableShape(empty, gate));
    CHECK_FALSE(dn::GateUpPairFusableShape(empty, empty));
  }

  SUBCASE("a true-W4A4 half is NOT fusable (that is the CUTLASS merged path)") {
    Nvfp4Weight w4a4 = MakeW4A16(kN, kK, 0.5F);
    w4a4.alpha = 0.125F;  // IsTrueW4A4()
    REQUIRE(w4a4.IsTrueW4A4());
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, w4a4));
    CHECK_FALSE(dn::GateUpPairFusableShape(w4a4, gate));
  }

  SUBCASE("mismatched N or K is NOT fusable") {
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, MakeW4A16(kN / 2, kK, 0.5F)));
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, MakeW4A16(kN, kK / 2, 0.5F)));
  }

  SUBCASE("a mismatched block-scale FORMAT is NOT fusable") {
    // Row-stacking the two shards' scales is only meaningful when both are read
    // with the same group size and the same scale encoding.
    Nvfp4Weight mx = MakeW4A16(kN, kK, 0.5F);
    mx.is_mxfp4 = true;
    mx.group_size = 32;
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, mx));
    Nvfp4Weight group_only = MakeW4A16(kN, kK, 0.5F);
    group_only.group_size = 32;
    CHECK_FALSE(dn::GateUpPairFusableShape(gate, group_only));
  }
}

TEST_CASE("dense gate_up fusion: VT_DENSE_MARLIN_GATEUP defaults ON, opt out with =0") {
  // Spec §3.3 asked for the default to move on a measured same-binary A/B, and
  // it did: interleaved 4-rep A/B on nvidia/Qwen3.6-27B-NVFP4@0893e160 (GB10),
  // toggle the only variable, gave fused +2.12% at c1 (12.0823 vs 11.8313) and
  // +1.70% at c8 (83.6186 vs 82.2217) with COMPLETE SEPARATION at both — every
  // fused rep beat every split rep — and a byte-identical 64-token greedy
  // continuation on both arms. So the lever now ships ON, per the standing
  // project rule that a parity enabler's default flips before a binding grid.
  //
  // UNSET must be ON, which is what a leftover opt-IN
  // `e && e[0]=='1' && e[1]=='\0'` spelling would break; only an explicit
  // leading '0' opts back out to the split pair.
  CHECK(dn::DenseMarlinGateUpEnabledFor(nullptr));
  CHECK(dn::DenseMarlinGateUpEnabledFor(""));
  CHECK(dn::DenseMarlinGateUpEnabledFor("1"));
  CHECK(dn::DenseMarlinGateUpEnabledFor("true"));
  CHECK(dn::DenseMarlinGateUpEnabledFor("10"));
  CHECK_FALSE(dn::DenseMarlinGateUpEnabledFor("0"));

  // The opt-out convention is the one the NEAREST parity levers use — the
  // sibling fused-gate_up toggles in the same header, VT_MARLIN_DENSE_PAIR
  // (MarlinDensePairEnabled) and VT_MOE_FUSED_W13 (FusedGateUpEnabled), both
  // `!(e != nullptr && e[0] == '0')`. That inspects the FIRST character only,
  // so a leading '0' opts out whatever follows it. Pinned so the choice is a
  // decision on record rather than an accident of spelling.
  CHECK_FALSE(dn::DenseMarlinGateUpEnabledFor("00"));
  CHECK_FALSE(dn::DenseMarlinGateUpEnabledFor("0x"));

  // The cached reader is that pure parser applied to the process environment —
  // it must not carry a second, differently-spelled default.
  CHECK(dn::DenseMarlinGateUpEnabled() ==
        dn::DenseMarlinGateUpEnabledFor(std::getenv("VT_DENSE_MARLIN_GATEUP")));
}

TEST_CASE("dense gate_up fusion: a backend with no Marlin NVFP4 op never selects it") {
  // The composed guard has no build-time gate of its own: `vt::OpRegistered` is
  // what makes a CPU/Vulkan/Metal device — and a build without VT_MARLIN_NVFP4 —
  // answer false. A perfectly-shaped pair must still be REFUSED here, because
  // there is no fused kernel to substitute; the split pair is unchanged.
  const Nvfp4Weight gate = MakeW4A16(kN, kK, 0.5F);
  const Nvfp4Weight up = MakeW4A16(kN, kK, 0.5F);
  REQUIRE(dn::GateUpPairFusableShape(gate, up));
  REQUIRE_FALSE(vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin,
                                 vt::DeviceType::kCPU));
  CHECK_FALSE(
      dn::DenseMlpGateUpFusedMarlinEligible(gate, up, vt::DeviceType::kCPU));
}

TEST_CASE("dense gate_up fusion reuses the EXISTING pair resident, not a new cache") {
  // MarlinDensePairResident is held on the GATE weight's own
  // `resident_marlin_pair` slot (issue #237: residency is a member of the
  // weights, never a static address-keyed map). A dense MLP's `gate_proj_fp4`
  // carries that same slot, so the dense path keys the pair resident exactly
  // like the shared expert does — which is why wiring it needed no second cache
  // and no extension of the existing one.
  DenseMlpWeights w;
  CHECK(w.gate_proj_fp4.resident_marlin_pair.state == nullptr);
  CHECK(w.up_proj_fp4.resident_marlin_pair.state == nullptr);
  // The pair slot is distinct from the single-projection slot the SPLIT path
  // uses, so selecting one layout never marks the other ready.
  CHECK(w.gate_proj_fp4.resident_marlin.state == nullptr);

  w.gate_proj_fp4.resident_marlin_pair.state = std::make_shared<int>(1);
  CHECK(w.gate_proj_fp4.resident_marlin.state == nullptr);
  CHECK(w.up_proj_fp4.resident_marlin_pair.state == nullptr);
}
