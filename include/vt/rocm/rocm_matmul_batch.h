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

// out_bf16[N,K] = scale[n] * f8_e4m3(w[n,k])  (device; for hipBLAS prefill)
void DequantFp8ChannelBf16Rocm(Queue& q, void* out_bf16, const void* fp8,
                               const void* scale_bf16, int N, int K);

// Fused Expert GeGLU FP8 decode (T=1). Faster than 3× MatmulBTFp8Channel.
bool ExpertGeGLUFp8M1Rocm(Queue& q, void* y, const void* x, const void* fp8_gu, const void* s_gu,
                          const void* fp8_dn, const void* s_dn, int I, int H, float alpha,
                          float beta);
bool ExpertGeGLUFp8TopKM1Rocm(Queue& q, void* ysum, const void* x, const void* const* fp8_gu,
                              const void* const* s_gu, const void* const* fp8_dn,
                              const void* const* s_dn, const float* wts, int G, int I, int H);
// Contiguous resident FP8 packs + device idx/wts (decode T=1, no host gather).
bool ExpertGeGLUFp8TopKIndexedRocm(Queue& q, void* ysum, const void* x, const void* gu_base,
                                   const void* dn_base, const void* sgu_base, const void* sdn_base,
                                   const int32_t* idx_dev, const float* wts_dev, int G, int I,
                                   int H);
void ApplyExpertScaleRwRocm(Queue& q, float* rw_dev, const int32_t* ri_dev, const float* escale_dev,
                            int G, int E);
bool PrewarmExpertGeGLUFp8TopKIndexedRocm(int dev, int G, int I, int H);

// Prefill MoE helpers (GPU-only gather / weighted scatter — no host hacc).
// out[n,H] = in[token_ids[i], H] for i in [0,n)
void MoeGatherRowsRocm(Queue& q, void* out_bf16, const void* in_bf16, const int32_t* token_ids,
                       int n, int H);
// acc[token_ids[i], :] += weight[i] * y[i, :]  (bf16 acc + bf16 y, float weights)
void MoeWeightedScatterAddRocm(Queue& q, void* acc_bf16, const void* y_bf16,
                               const int32_t* token_ids, const float* weights, int n, int H);
// Zero bf16 buffer [rows*H]
void MoeZeroBf16Rocm(Queue& q, void* buf_bf16, int64_t nelem);

}  // namespace vt::rocm
