// SPDX-License-Identifier: Apache-2.0
// sm70 W4A16 NVFP4 SIMT decode — tactic-registry contract test (Phase 2, A).
//
// Host-side only (the vt test TUs are compiled as plain C++; no CUDA calls
// here). It asserts the ADDITIVITY contract a server relies on: the tactic is
// registered statically, `SelectArchTactic` offers it for sm_70 devices, and
// the family's selection stats exist. The KERNEL-side numerics are exercised
// by the TU's own live self-check (the skinny contract) on the first real
// launch — that bit of the proof lives on the device (ci-cuda-sm70 lane /
// the V100 box), not in this file.

#include <doctest/doctest.h>

#ifdef VLLM_CPP_CUDA

#include "vt/cuda/cuda_arch_tactics.h"
#include "vt/cuda/cuda_device_caps.h"

namespace vt::cuda {

TEST_CASE("sm70 nvfp4-w4a16 tactic is registered and selectable for sm70") {
  const DeviceCaps caps = GetDeviceCaps();
  const ArchTactic* tactic = SelectArchTactic(TacticFamily::kSm70Nvfp4W4a16, caps);
  if (!caps.valid || caps.sm_major != 7) {
    // Not running on the hardware: the registry must still be present
    // (additivity is a compile-time surface), but selection is device-bound.
    MESSAGE("no sm_7 CUDA device visible; registry-only assertions");
    CHECK(RegisteredArchTacticCount(TacticFamily::kSm70Nvfp4W4a16) == 1);
    return;
  }
  REQUIRE(RegisteredArchTacticCount(TacticFamily::kSm70Nvfp4W4a16) == 1);
  REQUIRE(tactic != nullptr);
  CHECK(tactic->supports(caps));
  CHECK(GetArchTacticStats(TacticFamily::kSm70Nvfp4W4a16).selections >= 0);
}

}  // namespace vt::cuda

// On-box device driver for the W4A16 family (cuda_sm70_nvfp4_gemm.cu):
// SIMT decode, fused greedy-argmax vs CPU max, and the k%128 decline path.
extern "C" int vt_sm70_nvfp4_selfcheck(void);

extern "C" int vt_sm70_nvfp4_microbench(void);

extern "C" int vt_sm70_fp8w8a16_microbench(void);

TEST_CASE("sm70 nvfp4-w4a16 device self-check (SIMT + fused-argmax + decline)") {
  const int rc = vt_sm70_nvfp4_selfcheck();
  if (rc == 2) {
    MESSAGE("not an sm_70 CUDA device: device self-check skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("sm70 nvfp4-w4a16 kernel throughput microbench (effective GB/s, informational)") {
  const int rc = vt_sm70_nvfp4_microbench();
  if (rc == 2) {
    MESSAGE("not an sm_70 CUDA device: microbench skipped");
    return;
  }
  // The microbench prints GB/s; it has no pass/fail claim of its own (a
  // benchmark is a number, not an assertion). A nonzero return is a run error.
  CHECK(rc == 0);
}

TEST_CASE("sm70 fp8-w8a16 kernel throughput microbench (effective GB/s, informational)") {
  const int rc = vt_sm70_fp8w8a16_microbench();
  if (rc == 2) {
    MESSAGE("not an sm_70 CUDA device: microbench skipped");
    return;
  }
  // Same contract as the W4A16 sibling: GB/s is informational; a nonzero
  // return is a run error.
  CHECK(rc == 0);
}
#else
TEST_CASE("sm70 nvfp4-w4a16 / fp8-w8a16 kernels (CUDA not built)") {
  MESSAGE("CUDA support not built; sm70 kernel tests skipped");
  CHECK(true);
}
#endif
