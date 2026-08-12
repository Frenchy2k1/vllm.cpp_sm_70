// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Per-device plan cache key + flag plumbing for the cuBLASLt FP8 (e4m3) TN dense
// GEMM path in cuda_matmul.cu (MatmulFp8CublasLtKernelCuda). The heavy cuBLASLt
// call cublasLtMatmulAlgoGetHeuristic — plus the matmul descriptor + three matrix
// layouts it needs — was rebuilt on EVERY fp8 GEMM call, causing a recurring
// ~0.8 ms host gap before the sm_121 fp8 GEMM (nsys, dgx:~/work/prefill-attr-35b/)
// that hurts 35B prefill TTFT and c1-c4 decode. vLLM reuses an in-graph plan; we
// mirror that with a per-device {desc, A/B/C layouts, heuristic algo} cache keyed
// on the FULL shape/config that determines them, so a cache hit skips the
// descriptor/layout creation + heuristic and goes straight to cublasLtMatmul.
//
// This header holds ONLY the pure, CUDA-free pieces (the VT_FP8_PLAN_CACHE flag
// predicate and the plan KEY: fields + equality + hash) so they are unit-testable
// on the CPU tier (tests/vt/test_fp8_plan_cache.cpp), exactly like gemm_algo_log.h.
// The cache map itself holds cuBLASLt handles and therefore lives in cuda_matmul.cu.
//
// Bit-exactness: cuBLASLt algo selection is process-deterministic (the same shape
// selects the same algo per the algo-latching forensic record; see
// gemm_algo_log.h / .agents/state.md), so pinning the first-selected plan is
// numerically identical to rebuilding it — exactly what a captured graph does.
// Verified byte-exact vs a fresh-plan GEMM in test_ops_fp8_cutlass.cpp.
//
// DEFAULT OFF (opt-in, VT_FP8_PLAN_CACHE=1). The lever's premise — that the
// per-call cublasLtMatmulAlgoGetHeuristic + descriptor/layout rebuild is a
// removable ~0.8 ms host gap before the fp8 GEMM — was NOT reproduced on GB10
// (2026-07-18, CLAIM-FP8-PLAN-CACHE-1): a same-binary 35B A/B is wall-clock
// NEUTRAL on prefill TTFT (async on AND off) and c1/c4 decode TPOT, and nsys
// shows the pre-fp8-GEMM GPU-timeline gap is UNCHANGED by the cache (~210 µs
// with cache off vs ~204 µs on) — the heuristic host cost is negligible/hidden
// (prefill is GPU-bound so it overlaps GPU work; decode is CUDA-graph-captured
// so the heuristic runs once at capture, not per replay-step). The cache is a
// correct, bit-exact structural mirror of vLLM's in-graph plan reuse kept behind
// an opt-in flag for eager/non-graph regimes; it does not flip the default
// because the "faster" condition is unmet.
#ifndef VT_CUDA_FP8_PLAN_CACHE_H_
#define VT_CUDA_FP8_PLAN_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace vt::cuda {

// Pure predicate for the VT_FP8_PLAN_CACHE contract: the cache is OFF by default
// and ENABLED only for the exact value "1" (the opt-in). nullptr (unset) and
// every other value are OFF. Kept separate from the cached getter below so the
// parse is unit-testable without touching the process-global cache.
inline bool Fp8PlanCacheFlagIsOn(const char* env_value) {
  return env_value != nullptr && std::string_view(env_value) == "1";
}

// Process-cached gate, read from the environment exactly once (getenv on the
// first call only; every later hot-path call pays a single bool load). Kept out
// of the CPU unit test because the cache latches on first read; the parse itself
// is covered via Fp8PlanCacheFlagIsOn.
inline bool Fp8PlanCacheEnabled() {
  static const bool enabled = Fp8PlanCacheFlagIsOn(std::getenv("VT_FP8_PLAN_CACHE"));
  return enabled;
}

// --- PERF-FP8-ALPHA-FOLD: the vector-alpha epilogue arm ---------------------
// Spec: .agents/specs/perf-fp8-alpha-fold.md. Issue #402 (§3 "Lever B").
//
// When two FP8 shards are N-concatenated into ONE operand but carry DIFFERENT
// folded alphas, no single host scalar reproduces both halves, so the model
// today runs the GEMM at alpha=1 and applies the per-output-COLUMN alpha in a
// second full-tensor pass (vt::MulColVecF32). At T=4096 prefill that pass is a
// read-modify-write of a [T,16384] f32 tensor per GDN layer, measured at
// 209.5 GB/s = 77% of the GB10's 273.1 GB/s peak — 122.99 ms/request over 48
// calls, 43.6% of the whole measured 27B prefill deficit. It is bandwidth-bound,
// so its cost is its WIDTH; this is deliberately NOT the launch-count regime
// that #402 §4 sized as neutral on the decode step.
//
// cuBLASLt applies exactly this vector in the epilogue for free:
// CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO (cublasLt.h, "alpha
// pointer targets an array in device memory, beta is zero"), whose documented
// length rule is that the vector must "match number of output matrix ROWS"
// (CUBLASLT_MATMUL_DESC_POINTER_MODE). Our fp8 D is created COLUMN-MAJOR as
// (rows=n, cols=m, ld=n) — see the TN derivation in cuda_matmul.cu — so
// cuBLASLt's output ROWS are our row-major output's COLUMNS, and the model's
// resident f32 [N] alpha vector is already the right vector in the right layout.
//
// Pure (CUDA-free) pieces live here so the flag parse, the plan-key separation
// and the algo-capability predicate are unit-testable on every platform
// (tests/vt/test_fp8_plan_cache.cpp), exactly like the plan-cache flag above.

// Pure predicate for the VT_FP8_ALPHA_VEC_EPILOGUE contract: DEFAULT OFF, and
// enabled only for the exact value "1". nullptr (unset) and every other value
// are OFF. The default stays OFF until an operator-run same-binary prefill A/B
// (and the CUDA-tier bitwise case) says otherwise; the fallback below is the
// current two-launch behavior, byte for byte.
inline bool Fp8AlphaVecEpilogueFlagIsOn(const char* env_value) {
  return env_value != nullptr && std::string_view(env_value) == "1";
}

// Process-cached gate, read from the environment exactly once. Same shape as
// Fp8PlanCacheEnabled/GemmAlgoLogEnabled: the parse is what the unit test pins;
// this latching getter is deliberately not unit-tested.
inline bool Fp8AlphaVecEpilogueEnabled() {
  static const bool enabled =
      Fp8AlphaVecEpilogueFlagIsOn(std::getenv("VT_FP8_ALPHA_VEC_EPILOGUE"));
  return enabled;
}

// Values for Fp8PlanKey::scale_mode. The pointer mode is set on the matmul
// DESCRIPTOR before cublasLtMatmulAlgoGetHeuristic runs, so it can change the
// selected algo — including the split-K factor, which changes the f32 reduction
// order. A vector-alpha plan must therefore never reuse a scalar-alpha plan for
// the same shape, and vice versa.
enum : int {
  kFp8ScaleModeHostAlpha = 0,       // per-tensor scale folded into the host alpha
  kFp8ScaleModeAlphaDeviceVec = 1,  // per-column alpha vector in the epilogue
};

// The scale_mode a plan key must carry for the requested alpha form. Trivial by
// construction; it exists so the two constants can never be swapped or collapsed
// silently at the one call site that builds the key.
inline int Fp8ScaleModeFor(bool alpha_device_vector) {
  return alpha_device_vector ? kFp8ScaleModeAlphaDeviceVec : kFp8ScaleModeHostAlpha;
}

// Mirror of cublasLtPointerModeMask_t's
// CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO. Kept as a local
// constant so this header stays CUDA-free; cuda_matmul.cu static_asserts it
// against the real enum, so a header change cannot drift past the build.
inline constexpr unsigned int kFp8PointerModeMaskAlphaDeviceVectorBetaZero = 8U;

// Does an algo's CUBLASLT_ALGO_CAP_POINTER_MODE_MASK authorize the mode we set?
// ONLY the BETA_ZERO bit does: DEVICE_VECTOR (4) and ALPHA_DEVICE_VECTOR_BETA_HOST
// (16) are different modes, and an algo reporting them does not implement ours.
// A false here is not an error — it selects the two-launch fallback.
inline bool Fp8AlphaVecCapSupported(unsigned int pointer_mode_cap_mask) {
  return (pointer_mode_cap_mask & kFp8PointerModeMaskAlphaDeviceVectorBetaZero) != 0U;
}

// The FULL key that determines the cuBLASLt descriptor + selected algo for the
// fp8 (e4m3) TN dense GEMM path in cuda_matmul.cu. EVERY input that changes the
// descriptor OR the heuristic-selected algo MUST appear here — a missed field =
// wrong algo/desc reuse = a correctness bug. Fields are plain ints (the cuBLASLt
// enums cast to int at the call site) so this stays CUDA-free and CPU-testable.
//
// Field rationale (all captured from MatmulFp8CublasLtKernelCuda):
//   device       — one cuBLASLt handle + cached plans per device index.
//   m, n, k      — the GEMM shape; drives all three layout extents/leading dims
//                  (A=[K,N] ld=K, B=[K,M] ld=K, C=D=[N,M] ld=N) AND the algo the
//                  heuristic selects. m=a_fp8.shape[0], n=b_fp8.shape[0],
//                  k=a_fp8.shape[1].
//   out_type     — cudaDataType_t of C/D (CUDA_R_32F for f32 out, CUDA_R_16BF for
//                  bf16 out); the ONLY dtype that varies (A/B are always e4m3).
//                  Changes the C/D layout AND can change the selected algo.
//   a_type       — cudaDataType_t of the A/B operands (always CUDA_R_8F_E4M3
//                  here); pinned in the key so a future dtype split can't alias.
//   compute_type — cublasComputeType_t on the descriptor (CUBLAS_COMPUTE_32F).
//   scale_type   — cudaDataType_t scale on the descriptor (CUDA_R_32F).
//   trans_a/b    — CUBLASLT_MATMUL_DESC_TRANSA/TRANSB (OP_T / OP_N for the TN form).
//   epilogue     — cublasLtEpilogue_t (DEFAULT here; no bias/act fusion).
//   scale_mode   — which alpha FORM the descriptor carries, per Fp8ScaleModeFor:
//                  kFp8ScaleModeHostAlpha (0) = per-tensor scale folded into the
//                  host alpha, no pointer mode set; kFp8ScaleModeAlphaDeviceVec
//                  (1) = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO on
//                  the descriptor, alpha a device f32 [N] vector. The VALUE of a
//                  host alpha does not affect the descriptor or the algo, so it
//                  is deliberately NOT part of the key — but the pointer MODE is
//                  on the descriptor the heuristic reads, so the two forms must
//                  never share a plan.
struct Fp8PlanKey {
  int device = 0;
  int64_t m = 0, n = 0, k = 0;
  int out_type = 0;
  int a_type = 0;
  int compute_type = 0;
  int scale_type = 0;
  int trans_a = 0;
  int trans_b = 0;
  int epilogue = 0;
  int scale_mode = 0;

  bool operator==(const Fp8PlanKey& o) const {
    return device == o.device && m == o.m && n == o.n && k == o.k &&
           out_type == o.out_type && a_type == o.a_type && compute_type == o.compute_type &&
           scale_type == o.scale_type && trans_a == o.trans_a && trans_b == o.trans_b &&
           epilogue == o.epilogue && scale_mode == o.scale_mode;
  }
};

// FNV-1a-style hash over every key field (order-independent correctness: the ==
// above is the authority; the hash only needs to spread). Mixing each field in
// keeps distinct shapes/dtypes/transposes in different buckets.
struct Fp8PlanKeyHash {
  std::size_t operator()(const Fp8PlanKey& kk) const {
    std::size_t h = 1469598103934665603ull;  // FNV offset basis
    auto mix = [&h](std::uint64_t v) {
      h ^= static_cast<std::size_t>(v);
      h *= 1099511628211ull;  // FNV prime
    };
    mix(static_cast<std::uint64_t>(kk.device));
    mix(static_cast<std::uint64_t>(kk.m));
    mix(static_cast<std::uint64_t>(kk.n));
    mix(static_cast<std::uint64_t>(kk.k));
    mix(static_cast<std::uint64_t>(kk.out_type));
    mix(static_cast<std::uint64_t>(kk.a_type));
    mix(static_cast<std::uint64_t>(kk.compute_type));
    mix(static_cast<std::uint64_t>(kk.scale_type));
    mix(static_cast<std::uint64_t>(kk.trans_a));
    mix(static_cast<std::uint64_t>(kk.trans_b));
    mix(static_cast<std::uint64_t>(kk.epilogue));
    mix(static_cast<std::uint64_t>(kk.scale_mode));
    return h;
  }
};

}  // namespace vt::cuda

#endif  // VT_CUDA_FP8_PLAN_CACHE_H_
