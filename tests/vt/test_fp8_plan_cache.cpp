// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CPU-tier contract for the cuBLASLt FP8 plan-cache pure plumbing
// (src/vt/cuda/fp8_plan_cache.h): the VT_FP8_PLAN_CACHE flag predicate (default
// ON, "0" rollback) and the Fp8PlanKey equality + hash. The cache map itself
// holds cuBLASLt handles and is CUDA-only (lives in cuda_matmul.cu); the byte-
// exact cached-vs-fresh GEMM proof is the CUDA-tier test_ops_fp8_cutlass.cpp.
// This suite pins the KEY completeness (every descriptor/algo-affecting field
// distinguishes plans — a collision here would be a wrong-algo correctness bug)
// on every platform, not just DGX.
#include <doctest/doctest.h>

#include <string>
#include <unordered_map>

#include "vt/cuda/fp8_plan_cache.h"

using vt::cuda::Fp8AlphaVecCapSupported;
using vt::cuda::Fp8AlphaVecEpilogueFlagIsOn;
using vt::cuda::Fp8PlanCacheFlagIsOn;
using vt::cuda::Fp8PlanKey;
using vt::cuda::Fp8PlanKeyHash;
using vt::cuda::Fp8PlanRefusal;
using vt::cuda::Fp8PlanRefusalFor;
using vt::cuda::Fp8PlanRefusalTag;
using vt::cuda::Fp8ScaleModeFor;

TEST_CASE("VT_FP8_PLAN_CACHE is OFF by default; ON only for exactly \"1\"") {
  CHECK_FALSE(Fp8PlanCacheFlagIsOn(nullptr));  // unset -> OFF (default: rebuild per call)
  CHECK(Fp8PlanCacheFlagIsOn("1"));            // the opt-in: cache the plan
  CHECK_FALSE(Fp8PlanCacheFlagIsOn(""));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("2"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("on"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("true"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("0"));
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("11"));  // only the exact "1" enables
  CHECK_FALSE(Fp8PlanCacheFlagIsOn("1 "));  // trailing space must not enable
  CHECK_FALSE(Fp8PlanCacheFlagIsOn(" 1"));  // leading space must not enable
}

namespace {
// A canonical fp8 TN plan key (the 35B-shape family: e4m3 A/B, bf16 out, TN
// transposes, host-alpha scale). Each test perturbs ONE field.
Fp8PlanKey Base() {
  Fp8PlanKey k;
  k.device = 0;
  k.m = 8;
  k.n = 6144;
  k.k = 2048;
  k.out_type = 1;      // CUDA_R_16BF stand-in (values are opaque ints here)
  k.a_type = 28;       // CUDA_R_8F_E4M3 stand-in
  k.compute_type = 68; // CUBLAS_COMPUTE_32F stand-in
  k.scale_type = 0;    // CUDA_R_32F stand-in
  k.trans_a = 1;       // CUBLAS_OP_T
  k.trans_b = 0;       // CUBLAS_OP_N
  k.epilogue = 1;      // CUBLASLT_EPILOGUE_DEFAULT
  k.scale_mode = 0;    // host-alpha folded
  return k;
}
}  // namespace

TEST_CASE("Fp8PlanKey: two identical keys are equal and hash the same") {
  const Fp8PlanKey a = Base(), b = Base();
  CHECK(a == b);
  CHECK(Fp8PlanKeyHash{}(a) == Fp8PlanKeyHash{}(b));
}

TEST_CASE("Fp8PlanKey: perturbing ANY descriptor/algo field makes a DISTINCT key") {
  const Fp8PlanKey base = Base();
  // Every field is part of what determines the cuBLASLt descriptor or the
  // selected algo; a missed field would let a different shape/config reuse the
  // wrong plan. Each perturbation must break equality (and, being distinct keys,
  // must not silently alias in the map).
  auto differs = [&](Fp8PlanKey k) {
    CHECK_FALSE(base == k);
    // A hash collision is legal but not expected for these small perturbations;
    // equality is the authority, so we assert the map treats them as 2 entries.
    std::unordered_map<Fp8PlanKey, int, Fp8PlanKeyHash> m;
    m[base] = 1;
    m[k] = 2;
    CHECK(m.size() == 2);
  };
  { Fp8PlanKey k = base; k.device = 1;       differs(k); }
  { Fp8PlanKey k = base; k.m = 1;            differs(k); }
  { Fp8PlanKey k = base; k.n = 4096;         differs(k); }
  { Fp8PlanKey k = base; k.k = 1024;         differs(k); }
  { Fp8PlanKey k = base; k.out_type = 0;     differs(k); }  // bf16 out vs f32 out
  { Fp8PlanKey k = base; k.a_type = 29;      differs(k); }
  { Fp8PlanKey k = base; k.compute_type = 0; differs(k); }
  { Fp8PlanKey k = base; k.scale_type = 1;   differs(k); }
  { Fp8PlanKey k = base; k.trans_a = 0;      differs(k); }
  { Fp8PlanKey k = base; k.trans_b = 1;      differs(k); }
  { Fp8PlanKey k = base; k.epilogue = 2;     differs(k); }
  { Fp8PlanKey k = base; k.scale_mode = 1;   differs(k); }
}

// --- PERF-FP8-ALPHA-FOLD (spec .agents/specs/perf-fp8-alpha-fold.md) --------
// The vector-alpha epilogue arm applies each output COLUMN's folded fp8 alpha
// inside the cuBLASLt epilogue (CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO)
// instead of paying a separate full-tensor f32 read-modify-write pass. Its three
// pure pieces are CUDA-free and therefore pinned here, on every platform: the
// opt-in flag parse, the plan-key separation that stops a vector-alpha matmul
// reusing a scalar-alpha algo, and the algo-capability predicate that is the only
// thing standing between an algo that does NOT support the mode and a wrong
// result. The byte-exact vector-alpha-vs-two-launch GEMM proof is the CUDA-tier
// test_ops_fp8_cutlass.cpp; it cannot run on a CPU box.

TEST_CASE("VT_FP8_ALPHA_VEC_EPILOGUE is OFF by default; ON only for exactly \"1\"") {
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn(nullptr));  // unset -> OFF (the shipped default)
  CHECK(Fp8AlphaVecEpilogueFlagIsOn("1"));            // the opt-in: fold alpha into the epilogue
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn(""));
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn("0"));
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn("2"));
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn("on"));
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn("true"));
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn("11"));  // only the exact "1" enables
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn("1 "));  // trailing space must not enable
  CHECK_FALSE(Fp8AlphaVecEpilogueFlagIsOn(" 1"));  // leading space must not enable
}

TEST_CASE("Fp8ScaleModeFor: a vector-alpha plan can NEVER alias a scalar-alpha plan") {
  // The pointer mode is set on the matmul DESCRIPTOR before the heuristic runs,
  // so it can change the selected algo (including the split-K factor). Two plans
  // that differ only in scale_mode must therefore be two cache entries; if they
  // collapsed, a vector-alpha matmul would execute an algo chosen for a host
  // scalar — a silent wrong-result bug that no shape/dtype field would catch.
  CHECK(Fp8ScaleModeFor(false) == 0);              // host scalar alpha: the shipped mode
  CHECK(Fp8ScaleModeFor(true) != Fp8ScaleModeFor(false));
  Fp8PlanKey host = Base();
  host.scale_mode = Fp8ScaleModeFor(false);
  Fp8PlanKey vec = Base();
  vec.scale_mode = Fp8ScaleModeFor(true);
  CHECK_FALSE(host == vec);
  std::unordered_map<Fp8PlanKey, int, Fp8PlanKeyHash> m;
  m[host] = 1;
  m[vec] = 2;
  CHECK(m.size() == 2);
}

TEST_CASE("Fp8AlphaVecCapSupported: ONLY the ALPHA_DEVICE_VECTOR_BETA_ZERO bit qualifies") {
  // cublasLtPointerModeMask_t (cublasLt.h): HOST=1, DEVICE=2, DEVICE_VECTOR=4,
  // ALPHA_DEVICE_VECTOR_BETA_ZERO=8, ALPHA_DEVICE_VECTOR_BETA_HOST=16. We issue
  // the BETA_ZERO form, so ONLY bit 8 authorizes it. Accepting any other bit
  // would run an algo that does not implement the mode we asked for.
  CHECK_FALSE(Fp8AlphaVecCapSupported(0U));   // no capability reported -> fall back
  CHECK_FALSE(Fp8AlphaVecCapSupported(1U));   // HOST only (the classic scalar algo)
  CHECK_FALSE(Fp8AlphaVecCapSupported(2U));   // DEVICE scalar
  CHECK_FALSE(Fp8AlphaVecCapSupported(4U));   // DEVICE_VECTOR, but not the BETA_ZERO form
  CHECK_FALSE(Fp8AlphaVecCapSupported(7U));   // HOST|DEVICE|DEVICE_VECTOR
  CHECK_FALSE(Fp8AlphaVecCapSupported(16U));  // BETA_HOST only: a DIFFERENT mode
  CHECK_FALSE(Fp8AlphaVecCapSupported(23U));  // every neighbouring bit EXCEPT 8
  CHECK(Fp8AlphaVecCapSupported(8U));         // exactly the mode we set
  CHECK(Fp8AlphaVecCapSupported(9U));         // HOST|BETA_ZERO
  CHECK(Fp8AlphaVecCapSupported(31U));        // a fully capable algo
  CHECK(Fp8AlphaVecCapSupported(0xFFFFFFFFU));
}

TEST_CASE("Fp8PlanKey: same shape but different output dtype -> distinct plans") {
  // The f32-out and bf16-out fp8 GEMMs share (m,n,k) but select different C/D
  // layouts and can latch different algos; they must never share a cached plan.
  Fp8PlanKey bf16_out = Base();
  bf16_out.out_type = 1;
  Fp8PlanKey f32_out = Base();
  f32_out.out_type = 0;
  CHECK_FALSE(bf16_out == f32_out);
  std::unordered_map<Fp8PlanKey, int, Fp8PlanKeyHash> m;
  m[bf16_out] = 1;
  m[f32_out] = 2;
  CHECK(m.size() == 2);
}

TEST_CASE("Fp8PlanRefusalFor: names WHICH refusal, and never guesses a cap it never read") {
  // The 2026-08-11 GB10 run saw ZERO TN-fp8-alphavec lines and could not tell
  // "cuBLASLt offers no fp8 algo once the pointer mode is on the descriptor"
  // from "it offered one whose cap mask refuses the mode" — two causes with
  // different next steps. The dominance rule is the load-bearing part: with no
  // algo returned there is no capability to have read, so a cap refusal must
  // NOT be reported even when the caller's pointer_mode_ok defaulted false.
  CHECK(Fp8PlanRefusalFor(true, true) == Fp8PlanRefusal::kNone);
  CHECK(Fp8PlanRefusalFor(false, true) == Fp8PlanRefusal::kNoHeuristic);
  CHECK(Fp8PlanRefusalFor(false, false) == Fp8PlanRefusal::kNoHeuristic);
  CHECK(Fp8PlanRefusalFor(true, false) == Fp8PlanRefusal::kPointerModeUnsupported);
  // Tags are distinct and stable: the operator greps these out of the log.
  CHECK(std::string(Fp8PlanRefusalTag(Fp8PlanRefusal::kNone)) == "none");
  CHECK(std::string(Fp8PlanRefusalTag(Fp8PlanRefusal::kNoHeuristic)) == "no-heuristic");
  CHECK(std::string(Fp8PlanRefusalTag(Fp8PlanRefusal::kPointerModeUnsupported)) ==
        "pointer-mode-unsupported");
  CHECK(std::string(Fp8PlanRefusalTag(Fp8PlanRefusal::kNoHeuristic)) !=
        std::string(Fp8PlanRefusalTag(Fp8PlanRefusal::kPointerModeUnsupported)));
}
