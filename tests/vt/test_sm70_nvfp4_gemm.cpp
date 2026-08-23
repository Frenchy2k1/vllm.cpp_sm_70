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