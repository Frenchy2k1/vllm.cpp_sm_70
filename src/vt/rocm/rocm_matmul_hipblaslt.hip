// ROCm dense GEMM via hipBLAS (hipblasGemmEx) + optional hipBLASLt.
// Primary path: hipblasGemmEx with the standard row-major TN trick used by
// many ROCm LLM stacks. Falls back is not needed — GemmEx is required.
//
// MatmulBT: out[M,N] = a[M,K] @ b[N,K]^T  (b = Linear weight row-major)
// Matmul:   out[M,N] = a[M,K] @ b[K,N]
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

void CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: matmul: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

const char* BlasName(hipblasStatus_t st) {
  switch (st) {
    case HIPBLAS_STATUS_SUCCESS: return "SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM: return "INVALID_ENUM";
    default: return "UNKNOWN";
  }
}

void CheckBlas(hipblasStatus_t st, const char* what) {
  if (st != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("vt rocm: matmul: ") + what +
                             ": hipblas " + std::to_string(static_cast<int>(st)) +
                             " (" + BlasName(st) + ")");
  }
}

struct BlasCtx {
  hipblasHandle_t handle = nullptr;
};

BlasCtx GetBlas(int device, hipStream_t stream) {
  static std::mutex mu;
  static std::unordered_map<int, BlasCtx> ctxs;
  std::lock_guard<std::mutex> lock(mu);
  auto it = ctxs.find(device);
  if (it == ctxs.end()) {
    BlasCtx c;
    CheckBlas(hipblasCreate(&c.handle), "hipblasCreate");
    ctxs.emplace(device, c);
    it = ctxs.find(device);
  }
  CheckBlas(hipblasSetStream(it->second.handle, stream), "hipblasSetStream");
  return it->second;
}

std::string ComboName(const Tensor& a, const Tensor& b, const Tensor& out) {
  return std::string("(") + Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(out.dtype);
}

hipDataType ToBlasType(DType dt) {
  switch (dt) {
    case DType::kF32: return HIP_R_32F;
    case DType::kBF16: return HIP_R_16BF;
    case DType::kF16: return HIP_R_16F;
    default:
      throw std::runtime_error("vt rocm: matmul: unsupported dtype");
  }
}

}  // namespace

// out[M,N] = a[M,K] @ b[K,N]  (row-major contiguous / a may have row stride)
// Col-major BLAS view: C^T = B^T @ A^T
//   gemm(OP_T, OP_T, N, M, K, B, ldb=N? wait)
// Standard row-major C = A @ B via:
//   gemm(OP_N, OP_N, N, M, K, B_col, N, A_col, K, C_col, N)
// where A_row[M,K] is treated as A_col with ld=K meaning A_col is K x M...
//
// PyTorch/rocBLAS convention for row-major C(m,n) = A(m,k) @ B(k,n):
//   hipblasGemm(handle, OP_N, OP_N, n, m, k,
//               B, n, A, k, C, n)
// with A,B,C stored row-major — i.e. pass dimensions swapped so BLAS
// writes the transpose layout that coincides with row-major C.
void MatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16 = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32 = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16 && !f32) {
    throw std::runtime_error("vt rocm: matmul: unsupported " + ComboName(a, b, out));
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: matmul: bad out dtype");
  }
  const int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
  if (M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "k0");
    return;
  }
  // Require contiguous B rows; A may be strided in dim0.
  if (b.stride[0] != N) {
    throw std::runtime_error("vt rocm: matmul: b must be row-contiguous");
  }
  if (a.stride[0] < K) {
    throw std::runtime_error("vt rocm: matmul: bad a stride");
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(a.dtype);
  const hipDataType ot = ToBlasType(out.dtype);

  // C_rm = A_rm @ B_rm  <=>  treat as gemm(N,N) writing C with ld=N, dims n,m,k
  // A leading dim = a.stride[0] (row stride in elements)
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                          /*k=*/static_cast<int>(K), &alpha, b.data, at,
                          /*ldb=*/static_cast<int>(N), a.data, at,
                          /*lda=*/static_cast<int>(a.stride[0]), &beta, out.data, ot,
                          /*ldc=*/static_cast<int>(N), HIPBLAS_COMPUTE_32F,
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx NN");
}

// out[M,N] = a[M,K] @ b[N,K]^T
// Row-major trick: gemm(OP_T, OP_N, N, M, K, B, K, A, a_rs, C, N)
//   BLAS: C = op(A)*op(B) with opA=T => A is KxN in col form = row B[N,K]
//   opB=N, B is KxM col = row A[M,K] with ld=a_rs
//   C is NxM col = row out[M,N] ld=N
void MatmulBTKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16 = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32 = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16 && !f32) {
    throw std::runtime_error("vt rocm: matmul_bt: unsupported " + ComboName(a, b, out));
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: matmul_bt: bad out dtype");
  }
  const int64_t M = a.shape[0], K = a.shape[1], N = b.shape[0];
  if (M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "bt k0");
    return;
  }
  if (b.stride[0] != K) {
    throw std::runtime_error("vt rocm: matmul_bt: b must be [N,K] row-contiguous");
  }
  if (a.stride[0] < K) {
    throw std::runtime_error("vt rocm: matmul_bt: bad a stride");
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(a.dtype);
  const hipDataType ot = ToBlasType(out.dtype);

  // C = B^T @ A^T in row-major disguise:
  // gemm(OP_T, OP_N, N, M, K, B, ldb=K, A, lda=a_rs, C, ldc=N)
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                          /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                          /*k=*/static_cast<int>(K), &alpha, b.data, at,
                          /*ldb=*/static_cast<int>(K), a.data, at,
                          /*lda=*/static_cast<int>(a.stride[0]), &beta, out.data, ot,
                          /*ldc=*/static_cast<int>(N), HIPBLAS_COMPUTE_32F,
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx BT");
}

}  // namespace vt::rocm
