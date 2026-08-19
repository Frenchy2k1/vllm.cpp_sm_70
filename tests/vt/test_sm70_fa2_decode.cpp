// SPDX-License-Identifier: Apache-2.0
// sm70 FA2 decode — device parity self-check + op-dispatch driver.
//
// Host-side only (plain C++, no CUDA calls here). Numerics run in the CUDA TUs:
//   vt_sm70_fa2_self_check  — my Volta-WMMA decode fast path vs the reference
//                             PagedAttentionKernel, two configs (single head +
//                             GQA), device-synced, fractional values.
//   vt_sm70_fa2_op_parity   — end-to-end DISPATCH proof: the real op path
//                             (LaunchDecode) with the fast path forced ON vs
//                             forced OFF; asserts routing, not just kernels.

#include <doctest/doctest.h>

extern "C" int vt_sm70_fa2_self_check(float tol_rel, int verbose);
extern "C" int vt_sm70_fa2_op_parity(float tol_rel, int verbose);
extern "C" int vt_sm70_fa2_prefill_self_check(float tol_rel, int verbose);
extern "C" int vt_sm70_fa2_prefill_op_parity(float tol_rel, int verbose);

TEST_CASE("sm70 fa2 decode fast path matches the reference (kernel parity)") {
  // fp16-WMMA vs fp32-fma rounding -> sub-percent rel deviation expected.
  const int rc = vt_sm70_fa2_self_check(/*tol_rel=*/3e-2f, /*verbose=*/1);
  CHECK(rc == 0);  // tests the standalone kernel against the oracle.
}

TEST_CASE("sm70 fa2 decode is wired into the op dispatch (routing parity)") {
  const int rc = vt_sm70_fa2_op_parity(/*tol_rel=*/3e-2f, /*verbose=*/1);
  if (rc == 2) {
    MESSAGE("not an sm_70 device: dispatch cannot engage, skipping");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("sm70 fa2 prefill fast path matches CPU flash (causal, 2 req)") {
  const int rc = vt_sm70_fa2_prefill_self_check(/*tol_rel=*/3e-2f, /*verbose=*/1);
  if (rc == 2) {
    MESSAGE("not an sm_70 CUDA device: prefill self-check skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("sm70 fa2 prefill matches the engine's paged prefill op (routing parity)") {
  const int rc = vt_sm70_fa2_prefill_op_parity(/*tol_rel=*/3e-2f, /*verbose=*/1);
  if (rc == 2) {
    MESSAGE("not an sm_70 CUDA device: prefill op-parity skipped");
    return;
  }
  CHECK(rc == 0);
}