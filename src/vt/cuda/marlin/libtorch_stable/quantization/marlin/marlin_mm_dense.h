// vt lift — declaration of the vendored DENSE Marlin dispatcher (marlin::marlin_mm),
// defined in marlin_mm_dense.cu (= vLLM csrc/libtorch_stable/quantization/marlin/
// marlin.cu:326-541 `marlin::marlin_mm`, with the torch::stable `marlin_gemm`
// host wrapper stripped). The vt::Tensor launcher (src/vt/cuda/cuda_marlin_dense.cu)
// calls this directly.
//
// This is the DIRECT-A, tile-per-CTA dense GEMM — vLLM's OWN dense W4A16 numerics,
// distinct from the moe/marlin_moe_wna16 dispatcher (marlin_mm.h): no
// sorted_token_ids / expert_ids / top_k gather, an `lda` (A.stride(0)) parameter,
// and its own par-split fp32 C_tmp reduce structure. It is the byte-preserving
// replacement for the single-expert MoE-marlin route the dense E=1 projections use
// today (dense_nvfp4_gemm.h), whose par regrouping of the fp32 C_tmp reduce costs
// one bf16 ULP vs the oracle (row QUANT-CT-MXFP4-MARLIN-STRUCT / #50 / #54).
#pragma once

#include <cuda_runtime.h>

#include "core/scalar_type.hpp"

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias, bool has_act_order,
               bool is_k_full, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k, int thread_n, int sms,
               bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);

}  // namespace MARLIN_NAMESPACE_NAME
