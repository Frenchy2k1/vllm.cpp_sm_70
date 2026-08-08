// vllm.cpp original (vt runtime env-flag plumbing); the kernels these flags select
// are the GDN PREFILL causal_conv1d forward register-window kernel
// (CausalConv1dFwdRegKernel) and the split fused post-conv kernel
// (GdnPostConvSplitKernel), both in cuda_gdn.cu. Each is BIT-IDENTICAL (0-ulp) to the
// shipped kernel it replaces (CausalConv1dFwdTiledKernel / GdnPostConvKernel).
//
// CPU-tier contract for the env-flag plumbing that selects these prefill fast paths
// (src/vt/cuda/gdn_prefill_conv.h): the VT_CONV_REG and VT_GDN_POSTCONV_SPLIT flag
// predicates. The kernels themselves are CUDA-only; their BIT-EXACT (0-ulp) parity
// vs the shipped kernels is a DGX-gated CUDA check (tests/vt/test_ops_gdn.cpp). This
// suite pins the portable default/rollback parse so the contract is regression-covered
// on every platform, not just CUDA. The register-window and fast-megablock paths default
// ON; the slower split and not-yet-gate-validated token tile remain opt-in.
#include <doctest/doctest.h>

#include "vt/cuda/gdn_prefill_conv.h"

using vt::cuda::ConvRegFlagIsOn;
using vt::cuda::ConvExactChunksFlagIsOn;
using vt::cuda::GdnPostConvFastFlagIsOn;
using vt::cuda::GdnPostConvSplitFlagIsOn;
using vt::cuda::GdnPostConvTokenTileFlagIsOn;
using vt::cuda::GdnPostConvTokenTileGridX;

TEST_CASE("VT_CONV_REG defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: CausalConv1dFwdRegKernel's output (both `out` and the
  // rolled conv_state) is BIT-IDENTICAL (0-ulp) to the shipped tiled/scalar prefill
  // conv kernels by construction (same bias init, same j=0..k-1 tap accumulation over
  // the same f32 window values, same silu/identity epilogue + round-to-store, same
  // (K-1) state write-back), and it mirrors vLLM's register-resident FLA conv.
  CHECK(ConvRegFlagIsOn(nullptr));
  CHECK(ConvRegFlagIsOn(""));
  CHECK(ConvRegFlagIsOn("1"));
  CHECK(ConvRegFlagIsOn("off"));
  CHECK(ConvRegFlagIsOn("false"));
  CHECK(ConvRegFlagIsOn("2"));
  CHECK(ConvRegFlagIsOn(" 0"));  // leading space, not '0'
  // Roll back to the shipped tiled kernel: FIRST character '0'.
  CHECK_FALSE(ConvRegFlagIsOn("0"));
  CHECK_FALSE(ConvRegFlagIsOn("0abc"));
  CHECK_FALSE(ConvRegFlagIsOn("00"));
}

TEST_CASE("VT_CONV_EXACT_CHUNKS defaults ON; only a '0'-leading value rolls back") {
  CHECK(ConvExactChunksFlagIsOn(nullptr));
  CHECK(ConvExactChunksFlagIsOn(""));
  CHECK_FALSE(ConvExactChunksFlagIsOn("0"));
  CHECK_FALSE(ConvExactChunksFlagIsOn("0abc"));
  CHECK(ConvExactChunksFlagIsOn("1"));
  CHECK(ConvExactChunksFlagIsOn("on"));
}

TEST_CASE("VT_GDN_POSTCONV_SPLIT defaults OFF (opt-in); a non-'0' value enables it") {
  // Default (unset) is OFF: GdnPostConvSplitKernel is BIT-IDENTICAL (0-ulp) to the
  // shipped GdnPostConvKernel by construction (byte-for-byte q/k L2-norm branch; same
  // per-element V copy; same per-head gating math), but the DGX A/B measured it
  // near-neutral, so per the house "neutral ⇒ opt-in" convention it ships OFF. It is
  // enabled only when the value is present AND its first character is not '0'.
  CHECK_FALSE(GdnPostConvSplitFlagIsOn(nullptr));  // unset → shipped megablock
  CHECK_FALSE(GdnPostConvSplitFlagIsOn("0"));
  CHECK_FALSE(GdnPostConvSplitFlagIsOn("0abc"));
  CHECK_FALSE(GdnPostConvSplitFlagIsOn("00"));
  // Enable the split kernel: a present, non-'0'-leading value.
  CHECK(GdnPostConvSplitFlagIsOn(""));  // present (empty) → not '0'-leading → on
  CHECK(GdnPostConvSplitFlagIsOn("1"));
  CHECK(GdnPostConvSplitFlagIsOn("on"));
  CHECK(GdnPostConvSplitFlagIsOn("2"));
  CHECK(GdnPostConvSplitFlagIsOn(" 0"));  // leading space, not '0'
}

TEST_CASE("VT_GDN_POSTCONV_FAST defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: GdnPostConvFastKernel is BIT-IDENTICAL (0-ulp) to the
  // shipped GdnPostConvKernel for the Dk==Dv==128 gate dims (128-thread reduction is
  // the 256-thread tree minus a leading +0 step; the V copy is a 128-bit-staged pure
  // copy/convert), and it measured -24% per-call on GB10, so per the parity-enabler
  // policy (byte-exact ⇒ never-slower + token-safe) it ships ON. It rolls back to the
  // megablock only when the value is present AND its first character is '0'.
  CHECK(GdnPostConvFastFlagIsOn(nullptr));  // unset → fast megablock
  CHECK(GdnPostConvFastFlagIsOn(""));
  CHECK(GdnPostConvFastFlagIsOn("1"));
  CHECK(GdnPostConvFastFlagIsOn("on"));
  CHECK(GdnPostConvFastFlagIsOn("2"));
  CHECK(GdnPostConvFastFlagIsOn(" 0"));  // leading space, not '0'
  CHECK_FALSE(GdnPostConvFastFlagIsOn("0"));
  CHECK_FALSE(GdnPostConvFastFlagIsOn("0abc"));
  CHECK_FALSE(GdnPostConvFastFlagIsOn("00"));
}

TEST_CASE("VT_GDN_POSTCONV_TOKEN_TILE defaults OFF; a non-'0' value enables it") {
  CHECK_FALSE(GdnPostConvTokenTileFlagIsOn(nullptr));
  CHECK_FALSE(GdnPostConvTokenTileFlagIsOn("0"));
  CHECK_FALSE(GdnPostConvTokenTileFlagIsOn("0abc"));
  CHECK_FALSE(GdnPostConvTokenTileFlagIsOn("00"));
  CHECK(GdnPostConvTokenTileFlagIsOn(""));
  CHECK(GdnPostConvTokenTileFlagIsOn("1"));
  CHECK(GdnPostConvTokenTileFlagIsOn("on"));
  CHECK(GdnPostConvTokenTileFlagIsOn(" 0"));
}

TEST_CASE("GDN post-conv token tile covers each ceil(T/16) work item") {
  CHECK(GdnPostConvTokenTileGridX(0) == 0);
  CHECK(GdnPostConvTokenTileGridX(1) == 1);
  CHECK(GdnPostConvTokenTileGridX(15) == 1);
  CHECK(GdnPostConvTokenTileGridX(16) == 1);
  CHECK(GdnPostConvTokenTileGridX(17) == 2);
  CHECK(GdnPostConvTokenTileGridX(127) == 8);
  CHECK(GdnPostConvTokenTileGridX(128) == 8);
  CHECK(GdnPostConvTokenTileGridX(2048) == 128);
}
