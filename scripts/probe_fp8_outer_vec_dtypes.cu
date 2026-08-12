// PERF-FP8-ALPHA-FOLD attempt 3, pass 2 — is OUTER_VEC_32F refused because of
// our OUTPUT DTYPE, or on this device at all?
// Spec: .agents/specs/perf-fp8-alpha-fold.md.
//
// Pass 1 (probe_fp8_outer_vec_scale.cu) measured, on all ten real gate shapes:
//   V0 host scalar alpha              -> SUCCESS (algoId=67, splitK=1)
//   V1 A_SCALE_POINTER, SCALAR_32F    -> SUCCESS (identical algo)
//   V2 A_SCALE_POINTER + OUTER_VEC_32F-> heuristic CUBLAS_STATUS_INVALID_VALUE
//   V3 A+B OUTER_VEC_32F              -> heuristic CUBLAS_STATUS_NOT_SUPPORTED
// so the A-scale POINTER mechanism works for our exact e4m3 TN config and it is
// the MODE that is refused. Pass 1 asked that question at ONE output dtype
// (CUDA_R_32F, what the op's contract stores today).
//
// That is not enough to call the mechanism unavailable, for a reason that
// matters to a NEIGHBOURING row: fp8 rowwise scaling upstream (torch._scaled_mm)
// normally produces BF16, not f32 — and #417 / row/PERF-27B-BF16-FP8-OUT wants
// this very buffer narrowed to bf16 anyway. If OUTER_VEC_32F is implemented for
// a bf16 (or fp8) D but not an f32 D, then the alpha fold and the bf16-output
// lever COMBINE instead of competing, and the refusal recorded in the spec would
// be wrong in the one direction that costs the most.
//
// So this pass sweeps the output dtype and adds a device-level control: a plain
// square fp8 shape. That separates three very different conclusions —
//   "OUTER_VEC needs a non-f32 D"        (a lever, jointly with #417)
//   "OUTER_VEC is absent for OUR shapes" (a shape limit)
//   "OUTER_VEC is absent on GB10"        (a device/driver limit)
// — which a single-dtype probe cannot tell apart.
//
// Build: nvcc -O2 -std=c++17 -arch=sm_121a -o probe2 probe_fp8_outer_vec_dtypes.cu \
//        -lcublasLt -lcudart
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t kWorkspaceBytes = 32ull * 1024 * 1024;

const char* StatusName(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    default: return "OTHER";
  }
}

const char* DtypeName(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_32F: return "f32   ";
    case CUDA_R_16BF: return "bf16  ";
    case CUDA_R_16F: return "f16   ";
    case CUDA_R_8F_E4M3: return "e4m3  ";
    default: return "?     ";
  }
}

// scale_mode_a/b: -1 leaves the scale unset entirely; otherwise it is a
// cublasLtMatmulMatrixScale_t applied with the matching device pointer.
struct Cfg {
  const char* name;
  int scale_mode_a;
  int scale_mode_b;
};

// Heuristic-only: pass 1 already established that when a plan IS returned the
// matmul runs and the numbers are checkable. Here the question is purely whether
// cuBLASLt offers a plan for the descriptor at all, and pass 1 measured that the
// refusal lands on the HEURISTIC (INVALID_VALUE / NOT_SUPPORTED), not on matmul.
cublasStatus_t Probe(cublasLtHandle_t lt, const Cfg& cfg, cudaDataType_t out_type, int64_t m_tok,
                     int64_t n_out, int64_t k, const float* d_vec_m, const float* d_vec_n,
                     uint32_t* algo_id_out) {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
  if (cfg.scale_mode_a >= 0) {
    // A-side outer vector has M elements = D's rows = our N_out.
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_vec_m,
                                   sizeof(d_vec_m));
    const int32_t mode = cfg.scale_mode_a;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
  }
  if (cfg.scale_mode_b >= 0) {
    // B-side outer vector has N elements = D's cols = our token count.
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_vec_n,
                                   sizeof(d_vec_n));
    const int32_t mode = cfg.scale_mode_b;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
  }
  cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, (uint64_t)k, (uint64_t)n_out, k);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, (uint64_t)k, (uint64_t)m_tok, k);
  cublasLtMatrixLayoutCreate(&lc, out_type, (uint64_t)n_out, (uint64_t)m_tok, n_out);
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &kWorkspaceBytes, sizeof(kWorkspaceBytes));
  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  const cublasStatus_t st =
      cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &returned);
  *algo_id_out = 0;
  if (st == CUBLAS_STATUS_SUCCESS && returned > 0) {
    size_t w = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_ID, algo_id_out,
                                         sizeof(*algo_id_out), &w);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatmulDescDestroy(desc);
  return (st == CUBLAS_STATUS_SUCCESS && returned == 0) ? CUBLAS_STATUS_NOT_SUPPORTED : st;
}

struct Shape {
  const char* label;
  int64_t m_tok, n_out, k;
};

}  // namespace

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  printf("PROBE pass 2: OUTER_VEC_32F across OUTPUT DTYPES\n");
  printf("device=%s sm_%d%d cublasLt=%zu\n\n", prop.name, prop.major, prop.minor,
         cublasLtGetVersion());

  cublasLtHandle_t lt = nullptr;
  cublasLtCreate(&lt);

  const Cfg cfgs[] = {
      {"none (control)          ", -1, -1},
      {"A=SCALAR_32F            ", CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, -1},
      {"A=OUTER_VEC_32F         ", CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F, -1},
      {"A=B=OUTER_VEC_32F       ", CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F},
      {"A=OUTER_VEC B=SCALAR    ", CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
       CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F},
  };
  const cudaDataType_t out_types[] = {CUDA_R_32F, CUDA_R_16BF, CUDA_R_16F, CUDA_R_8F_E4M3};
  const Shape shapes[] = {
      {"27B gate M=1   ", 1, 16384, 5120},
      {"27B gate M=4096", 4096, 16384, 5120},
      {"35B gate M=4096", 4096, 6144, 2048},
      {"square  M=4096 ", 4096, 4096, 4096},  // device-level control: a canonical shape
  };

  // One vector per side, sized for the largest case; contents are irrelevant to a
  // heuristic query, only the pointers' presence and the declared modes matter.
  float *d_vec_m = nullptr, *d_vec_n = nullptr;
  cudaMalloc(&d_vec_m, 16384 * sizeof(float));
  cudaMalloc(&d_vec_n, 16384 * sizeof(float));
  cudaMemset(d_vec_m, 0, 16384 * sizeof(float));
  cudaMemset(d_vec_n, 0, 16384 * sizeof(float));

  int outer_vec_ok = 0;
  for (const Shape& sh : shapes) {
    printf("--- %s (M_tok=%lld N_out=%lld K=%lld)\n", sh.label, (long long)sh.m_tok,
           (long long)sh.n_out, (long long)sh.k);
    printf("      %-26s", "config \\ D dtype");
    for (cudaDataType_t t : out_types) printf(" %s", DtypeName(t));
    printf("\n");
    for (const Cfg& cfg : cfgs) {
      printf("      %-26s", cfg.name);
      for (cudaDataType_t t : out_types) {
        uint32_t algo = 0;
        const cublasStatus_t st =
            Probe(lt, cfg, t, sh.m_tok, sh.n_out, sh.k, d_vec_m, d_vec_n, &algo);
        if (st == CUBLAS_STATUS_SUCCESS) {
          printf(" ok#%-3u", algo);
          if (cfg.scale_mode_a == CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) ++outer_vec_ok;
        } else {
          printf(" %-6.6s", StatusName(st));
        }
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("SUMMARY: OUTER_VEC_32F accepted in %d of the swept (shape x dtype x cfg) cells\n",
         outer_vec_ok);
  cudaFree(d_vec_m);
  cudaFree(d_vec_n);
  cublasLtDestroy(lt);
  printf("PROBE2_DONE\n");
  return 0;
}
