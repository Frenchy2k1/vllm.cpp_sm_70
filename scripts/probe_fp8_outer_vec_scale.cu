// PERF-FP8-ALPHA-FOLD attempt 3 — standalone cuBLASLt capability probe.
// Spec: .agents/specs/perf-fp8-alpha-fold.md (§"Next traceable hypothesis").
//
// QUESTION: can cuBLASLt apply our resident per-output-column f32 alpha vector
// inside the fp8 GEMM epilogue via CUBLASLT_MATMUL_DESC_A_SCALE_POINTER (17)
// with CUBLASLT_MATMUL_DESC_A_SCALE_MODE (31) =
// CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F (3) — the per-channel/rowwise fp8
// scaling path — at the REAL 27B and 35B GDN in_proj_qkvz gate shapes?
//
// WHY A PROBE AND NOT AN A/B: attempt 1 (POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO)
// was integrated first, and its refusal was only discovered from a null A/B —
// a VOID measurement of the fallback against itself. This settles the capability
// question before a line of integration is written.
//
// WHY IT MUST CALL cublasLtMatmul AND NOT ONLY THE HEURISTIC: cublasLt.h:1420-1430
// documents A_SCALE_POINTER as "If set for an unsupported matrix data, scale,
// and compute type combination, calling cublasLtMatmul() will return
// CUBLAS_INVALID_VALUE" — the refusal surfaces at MATMUL time, not at heuristic
// time. A green heuristic proves nothing here.
//
// LAYOUT (mirrors MatmulFp8CublasLtKernelCuda in src/vt/cuda/cuda_matmul.cu):
//   row-major out[M_tok, N_out] computed as col-major out^T[N_out, M_tok]:
//     A = weight : col-major [K, N_out] ld=K, TRANSA=OP_T, e4m3
//     B = act    : col-major [K, M_tok] ld=K, TRANSB=OP_N, e4m3
//     C = D = out: col-major [N_out, M_tok] ld=N_out, f32
//   so the cuBLASLt matmul dims are (M=N_out, N=M_tok, K=K), and the A-side
//   OUTER_VEC length "M elements" is EXACTLY N_out — our resident f32 [N]
//   alpha_vec, unrepacked. That equivalence is the whole reason this API is a
//   candidate; the probe asserts it by checking the numbers, not by assuming it.
//
// Build:  nvcc -O2 -std=c++17 -o probe probe_fp8_outer_vec_scale.cu -lcublasLt -lcudart
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t kWorkspaceBytes = 32ull * 1024 * 1024;
constexpr int kHeuristicAlgos = 1;  // mirrors kGemvHeuristicAlgos (cuda_matmul.cu:53)

const char* StatusName(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
    default: return "UNKNOWN";
  }
}

// The four descriptor forms the probe distinguishes. Together they separate
// "the shape has no fp8 heuristic at all" from "the A-scale POINTER is
// unsupported" from "the pointer works but the OUTER_VEC MODE is not" from
// "it works" — the exact distinction attempt 1 could not make.
enum Variant {
  kHostScalar = 0,       // control: today's shipped form, host scalar alpha
  kAScaleScalarDev = 1,  // A_SCALE_POINTER -> device f32 SCALAR (mode 0, the default)
  kAScaleOuterVec = 2,   // A_SCALE_POINTER -> device f32 [N_out], A_SCALE_MODE=OUTER_VEC_32F
  kAScaleOuterVecB = 3,  // as above + B_SCALE_POINTER -> ones[M_tok], B_SCALE_MODE=OUTER_VEC_32F
};

const char* VariantName(Variant v) {
  switch (v) {
    case kHostScalar: return "V0-host-scalar-alpha      ";
    case kAScaleScalarDev: return "V1-A_SCALE_PTR-scalar32f  ";
    case kAScaleOuterVec: return "V2-A_SCALE_PTR-OUTER_VEC  ";
    case kAScaleOuterVecB: return "V3-A+B_SCALE-OUTER_VEC    ";
  }
  return "?";
}

struct RunResult {
  cublasStatus_t heur_status = CUBLAS_STATUS_SUCCESS;
  int returned = 0;
  cublasStatus_t matmul_status = CUBLAS_STATUS_SUCCESS;
  cudaError_t sync_status = cudaSuccess;
  uint32_t algo_id = 0, tile = 0, stages = 0, split_k = 0;
  uint32_t pointer_mode_cap = 0;
  size_t ws = 0;
  bool ran = false;  // matmul actually executed and synced clean
};

// One descriptor build + heuristic + matmul, for one variant. d_out is filled
// only when ran==true. Every failure is REPORTED, never fatal: a refusal is the
// answer the probe exists to collect.
RunResult RunVariant(cublasLtHandle_t lt, Variant v, int64_t m_tok, int64_t n_out, int64_t k,
                     const void* d_a_w, const void* d_b_act, float* d_out, const float* d_alpha_vec,
                     const float* d_alpha_scalar, const float* d_ones_n, void* d_ws,
                     cudaStream_t stream) {
  RunResult r;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

  // Report the SetAttribute statuses too: a refusal could in principle land here
  // rather than at matmul, and conflating the two is how attempt 1 lost a week.
  cublasStatus_t set_ptr = CUBLAS_STATUS_SUCCESS, set_mode = CUBLAS_STATUS_SUCCESS;
  if (v == kAScaleScalarDev) {
    set_ptr = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                             &d_alpha_scalar, sizeof(d_alpha_scalar));
  } else if (v == kAScaleOuterVec || v == kAScaleOuterVecB) {
    set_ptr = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                             &d_alpha_vec, sizeof(d_alpha_vec));
    const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    set_mode = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode,
                                              sizeof(mode));
    if (v == kAScaleOuterVecB) {
      cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_ones_n,
                                     sizeof(d_ones_n));
      cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    }
  }
  if (set_ptr != CUBLAS_STATUS_SUCCESS || set_mode != CUBLAS_STATUS_SUCCESS) {
    printf("      setattr: A_SCALE_POINTER=%s A_SCALE_MODE=%s\n", StatusName(set_ptr),
           StatusName(set_mode));
  }

  cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, static_cast<uint64_t>(k),
                             static_cast<uint64_t>(n_out), k);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, static_cast<uint64_t>(k),
                             static_cast<uint64_t>(m_tok), k);
  cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, static_cast<uint64_t>(n_out),
                             static_cast<uint64_t>(m_tok), n_out);

  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &kWorkspaceBytes, sizeof(kWorkspaceBytes));

  cublasLtMatmulHeuristicResult_t heur{};
  r.heur_status = cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, kHeuristicAlgos,
                                                 &heur, &r.returned);
  if (r.heur_status == CUBLAS_STATUS_SUCCESS && r.returned > 0) {
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_ID, &r.algo_id,
                                         sizeof(r.algo_id), &w);
    cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &r.tile,
                                         sizeof(r.tile), &w);
    cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &r.stages,
                                         sizeof(r.stages), &w);
    cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &r.split_k,
                                         sizeof(r.split_k), &w);
    cublasLtMatmulAlgoCapGetAttribute(&heur.algo, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
                                      &r.pointer_mode_cap, sizeof(r.pointer_mode_cap), &w);
    r.ws = heur.workspaceSize;

    // alpha is the HOST scalar in every variant: in V1-V3 the per-column factor
    // rides on the A scale, so alpha stays 1.0 and the vector does the work.
    const float alpha = 1.0f, beta = 0.0f;
    r.matmul_status = cublasLtMatmul(lt, desc, &alpha, d_a_w, la, d_b_act, lb, &beta, d_out, lc,
                                     d_out, lc, &heur.algo, d_ws, kWorkspaceBytes, stream);
    if (r.matmul_status == CUBLAS_STATUS_SUCCESS) {
      r.sync_status = cudaStreamSynchronize(stream);
      r.ran = r.sync_status == cudaSuccess;
    }
  }

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatmulDescDestroy(desc);
  return r;
}

void PrintRun(Variant v, const RunResult& r) {
  printf("      %s heur=%-13s ret=%d", VariantName(v), StatusName(r.heur_status), r.returned);
  if (r.returned > 0) {
    printf(" algoId=%u tile=%u stages=%u splitK=%u ws=%zu ptrModeCap=%u matmul=%s", r.algo_id,
           r.tile, r.stages, r.split_k, r.ws, r.pointer_mode_cap, StatusName(r.matmul_status));
    if (r.matmul_status == CUBLAS_STATUS_SUCCESS)
      printf(" sync=%s", r.sync_status == cudaSuccess ? "ok" : cudaGetErrorName(r.sync_status));
  }
  printf("\n");
}

struct Shape {
  const char* label;
  int64_t m_tok, n_qkv, n_z, k;
};

}  // namespace

int main() {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);
  printf("PROBE PERF-FP8-ALPHA-FOLD attempt 3: A_SCALE_POINTER + OUTER_VEC_32F\n");
  printf("device=%s sm_%d%d cublasLt=%zu runtime=%d driver=%d\n", prop.name, prop.major,
         prop.minor, cublasLtGetVersion(), rt, drv);
  printf("cublasLt enums: A_SCALE_POINTER=%d A_SCALE_MODE=%d OUTER_VEC_32F=%d\n\n",
         (int)CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (int)CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
         (int)CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);

  cublasLtHandle_t lt = nullptr;
  cublasLtCreate(&lt);
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  void* d_ws = nullptr;
  cudaMalloc(&d_ws, kWorkspaceBytes);

  // The real gate shapes. n_qkv/n_z are the two FP8 shards whose DIFFERENT
  // folded alphas are the entire reason a per-column vector exists here
  // (qwen3_5.cpp ResidentFp8Qkvz: folded = (qkv.alpha == z.alpha) is FALSE).
  const Shape shapes[] = {
      {"27B GDN in_proj_qkvz", 1, 10240, 6144, 5120},
      {"27B GDN in_proj_qkvz", 3, 10240, 6144, 5120},
      {"27B GDN in_proj_qkvz", 128, 10240, 6144, 5120},
      {"27B GDN in_proj_qkvz", 1024, 10240, 6144, 5120},
      {"27B GDN in_proj_qkvz", 4096, 10240, 6144, 5120},  // the MEASURED prefill regime
      {"35B GDN in_proj_qkvz", 1, 4096, 2048, 2048},
      {"35B GDN in_proj_qkvz", 3, 4096, 2048, 2048},
      {"35B GDN in_proj_qkvz", 128, 4096, 2048, 2048},
      {"35B GDN in_proj_qkvz", 1024, 4096, 2048, 2048},
      {"35B GDN in_proj_qkvz", 4096, 4096, 2048, 2048},
  };

  int works = 0, refused = 0;
  for (const Shape& sh : shapes) {
    const int64_t n_out = sh.n_qkv + sh.n_z, m_tok = sh.m_tok, k = sh.k;
    const float alpha_qkv = 0.035f, alpha_z = 0.017f;  // the test's two shard alphas
    printf("--- %s M=%lld N=%lld (qkv %lld + z %lld) K=%lld\n", sh.label, (long long)m_tok,
           (long long)n_out, (long long)sh.n_qkv, (long long)sh.n_z, (long long)k);

    std::mt19937 rng(static_cast<uint32_t>(m_tok * 7919 + k));
    std::uniform_int_distribution<int> ub(0, 255);
    auto rand_fp8 = [&](size_t n) {
      std::vector<uint8_t> vv(n);
      for (auto& x : vv) {
        int byte = ub(rng);
        if ((byte & 0x7F) == 0x7F) byte &= ~0x7;  // no NaN encodings
        x = static_cast<uint8_t>(byte);
      }
      return vv;
    };
    const std::vector<uint8_t> h_act = rand_fp8(static_cast<size_t>(m_tok) * k);
    const std::vector<uint8_t> h_w = rand_fp8(static_cast<size_t>(n_out) * k);
    std::vector<float> h_alpha(static_cast<size_t>(n_out));
    for (int64_t i = 0; i < n_out; ++i) h_alpha[i] = (i < sh.n_qkv) ? alpha_qkv : alpha_z;
    const std::vector<float> h_ones(static_cast<size_t>(m_tok), 1.0f);
    const float h_one = 1.0f;

    void *d_w = nullptr, *d_act = nullptr;
    float *d_out = nullptr, *d_alpha = nullptr, *d_scalar = nullptr, *d_ones = nullptr;
    const size_t out_elems = static_cast<size_t>(m_tok) * n_out;
    cudaMalloc(&d_w, h_w.size());
    cudaMalloc(&d_act, h_act.size());
    cudaMalloc(&d_out, out_elems * sizeof(float));
    cudaMalloc(&d_alpha, h_alpha.size() * sizeof(float));
    cudaMalloc(&d_scalar, sizeof(float));
    cudaMalloc(&d_ones, h_ones.size() * sizeof(float));
    cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, h_act.data(), h_act.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ones, h_ones.data(), h_ones.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Reference = TODAY'S SHIPPED FORM: GEMM at host alpha=1, then the per-column
    // vector applied as a separate f32 pass. The host multiply below is the exact
    // IEEE operation MulColVecF32Kernel performs (one f32 multiply per element),
    // so a byte comparison against it is the byte-exactness gate, not an epsilon.
    std::vector<float> ref(out_elems);
    RunResult base = RunVariant(lt, kHostScalar, m_tok, n_out, k, d_w, d_act, d_out, d_alpha,
                                d_scalar, d_ones, d_ws, stream);
    PrintRun(kHostScalar, base);
    if (!base.ran) {
      printf("      -> control FAILED; no fp8 heuristic for this shape, nothing to compare\n\n");
      cudaFree(d_w); cudaFree(d_act); cudaFree(d_out);
      cudaFree(d_alpha); cudaFree(d_scalar); cudaFree(d_ones);
      continue;
    }
    cudaMemcpy(ref.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost);
    // D is col-major [n_out, m_tok]: element (row=col_of_our_output, col=token).
    for (size_t t = 0; t < static_cast<size_t>(m_tok); ++t)
      for (size_t c = 0; c < static_cast<size_t>(n_out); ++c)
        ref[t * n_out + c] = ref[t * n_out + c] * h_alpha[c];

    for (Variant v : {kAScaleScalarDev, kAScaleOuterVec, kAScaleOuterVecB}) {
      cudaMemset(d_out, 0, out_elems * sizeof(float));
      RunResult r = RunVariant(lt, v, m_tok, n_out, k, d_w, d_act, d_out, d_alpha, d_scalar, d_ones,
                               d_ws, stream);
      PrintRun(v, r);
      if (!r.ran) continue;
      if (v == kAScaleScalarDev) continue;  // scalar 1.0: a support probe, not a numeric arm
      std::vector<float> got(out_elems);
      cudaMemcpy(got.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost);
      size_t diff = 0;
      double max_rel = 0.0;
      for (size_t i = 0; i < out_elems; ++i) {
        if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
          ++diff;
          const double d = std::abs((double)got[i] - (double)ref[i]);
          const double den = std::abs((double)ref[i]);
          if (den > 0) max_rel = std::max(max_rel, d / den);
        }
      }
      printf("        -> vs two-launch reference: %zu/%zu f32 words differ (%.4f%%), maxRel=%.3e\n",
             diff, out_elems, 100.0 * (double)diff / (double)out_elems, max_rel);
      if (v == kAScaleOuterVec) {
        if (diff == 0) {
          ++works;
          printf("        -> BYTE-EXACT and ACCEPTED\n");
        } else {
          ++refused;
        }
      }
    }
    printf("\n");
    cudaFree(d_w); cudaFree(d_act); cudaFree(d_out);
    cudaFree(d_alpha); cudaFree(d_scalar); cudaFree(d_ones);
  }

  printf("SUMMARY: OUTER_VEC byte-exact on %d shape(s), non-exact/refused on %d\n", works, refused);
  cudaFree(d_ws);
  cudaStreamDestroy(stream);
  cublasLtDestroy(lt);
  printf("PROBE_DONE\n");
  return 0;
}
