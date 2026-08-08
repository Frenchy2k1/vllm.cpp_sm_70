// ROCm: batched MatmulBT helpers for MoE top-k fuse.
#pragma once

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {

void MatmulBTStridedBatchKernelRocm(Queue& q, void* out, const void* a, const void* b,
                                    int batch, int M, int N, int K, DType dtype);

void MatmulBTStridedBatchFullKernelRocm(Queue& q, void* out, const void* a, const void* b,
                                        int batch, int M, int N, int K, long long strideA,
                                        DType dtype);

// Pointer-array batch (no gather): out_ptrs[g] = a @ b_ptrs[g]^T
void MatmulBTPointerBatchKernelRocm(Queue& q, void** out_ptrs, const void* a,
                                    void** b_ptrs, int batch, int M, int N, int K,
                                    DType dtype);

// Both A and B per-batch: out_ptrs[g] = a_ptrs[g] @ b_ptrs[g]^T
void MatmulBTPointerBatchABKernelRocm(Queue& q, void** out_ptrs, void** a_ptrs,
                                      void** b_ptrs, int batch, int M, int N, int K,
                                      DType dtype);

// out = alpha * (a @ b^T) + beta * out   — MoE expert mix fuse
// a [M,K], b [N,K], out [M,N], contiguous rows
void MatmulBTAlphaBetaRocm(Queue& q, void* out, const void* a, const void* b, int M, int N,
                           int K, float alpha, float beta, DType dtype);

// M=1 BF16 act × FP8 weight [N,K] with BF16 channel scale[N]
void MatmulBTFp8ChannelRocm(Queue& q, void* out, const void* a, const void* b_fp8,
                            const void* scale_bf16, int M, int N, int K, float alpha,
                            float beta);

}  // namespace vt::rocm
