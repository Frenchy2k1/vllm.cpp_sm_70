// SPDX-License-Identifier: Apache-2.0
// sm70 FA2 decode — device parity self-check driver (Phase 2 brick F).
//
// Host-side only (plain C++, no CUDA calls here). The numeric work runs in the
// cuda_sm70_flash_attn.cu TU's own self-check (`vt_sm70_fa2_self_check`: my
// Volta-WMMA decode fast path vs the reference PagedAttentionKernel on the
// SAME paged fp16 tensors). This test only asserts that end-to-end contract.

#include <doctest/doctest.h>

extern "C" int vt_sm70_fa2_self_check(float tol_rel, int verbose);

TEST_CASE("sm70 fa2 decode fast path matches the reference (fp32 rel tolerance)") {
  // fp16-MMA vs fp32-fma score rounding: expect sub-percent rel deviation.
  const int rc = vt_sm70_fa2_self_check(/*tol_rel=*/3e-2f, /*verbose=*/1);
  CHECK(rc == 0);
}