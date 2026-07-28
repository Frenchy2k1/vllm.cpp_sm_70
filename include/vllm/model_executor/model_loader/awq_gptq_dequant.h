// Ported from:
//   vllm/model_executor/layers/quantization/awq_triton.py (awq_dequantize_kernel) @ 555967922
//   vllm/model_executor/layers/quantization/auto_awq.py:73-168 (_REVERSE_AWQ_PACK_ORDER,
//     _convert_awq_to_standard_format) @ 555967922
//   csrc/libtorch_stable/quantization/gptq/qdq_4.cuh (dequant_4bit_8_gptq) @ 555967922
//   csrc/libtorch_stable/quantization/gptq/q_gemm.cu:201-202,256 (zero_offset v1/v2) @ 555967922
//
// AWQ / GPTQ INT4 weight-only dequant utilities. Both formats pack 8 4-bit
// values per int32; AWQ and GPTQ differ ONLY in packing axis, bit order and
// zero-point convention (this file mirrors both exactly). These materialize a
// bf16 weight matrix from the on-disk community-format tensors, the CPU
// reference the later Marlin GPU compute path (kernel-matrix) is gated against.
//
//   -- AWQ (activation-aware weight quant, asymmetric, group-wise) --
//   qweight  int32 [K, N/8]     packed along OUTPUT dim N, AWQ order
//                               [0,4,1,5,2,6,3,7] (auto_awq.py:77)
//   qzeros   int32 [K/G, N/8]   packed along N, same AWQ order
//   scales   f32   [K/G, N]     one scale per (group, output-col); caller
//                               decodes the on-disk fp16/bf16 -> f32
//   w[k,n] = (nib_w(k,n) - nib_z(k/G,n)) * scales[k/G, n]     (awq_triton.py:101)
//
//   -- GPTQ (AutoGPTQ / GPTQModel, symmetric or asymmetric, group-wise) --
//   qweight  int32 [K/8, N]     packed along INPUT dim K, STANDARD order
//                               (quant_utils.gptq_pack == pack_rows)
//   qzeros   int32 [K/G, N/8]   packed along N, standard order
//   scales   f32   [num_groups, N]
//   g_idx    int32 [K] or null  act-order (desc_act) row->group map; when null,
//                               group(k) = k / group_size
//   w[k,n] = (nib_w(k,n) - (nib_z(g,n) + zero_offset)) * scales[g, n]
//            with g = g_idx ? g_idx[k] : k/G, zero_offset = 1 (GPTQv1) | 0 (v2)
//            (qdq_4.cuh dequant_4bit_8_gptq; q_gemm.cu:201-202 zero_offset)
//
// int32 payloads are read as uint32 for the shifts so sign extension can never
// corrupt a nibble. Output bf16 bit patterns are row-major [K, N].
#pragma once

#include <cstdint>

namespace vllm {

// 4-bit AWQ/GPTQ: 8 nibbles per packed int32.
inline constexpr int kAwqGptqPackFactor = 8;

// AWQ reverse-pack order: output element e reads nibble at position
// kAwqReverseOrder[e] within its int32 (auto_awq.py:77, awq_triton.py:54-64).
inline constexpr int kAwqReverseOrder[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// Dequantize an AWQ INT4 weight matrix to bf16.
//   qweight  [K, N/8]   int32 (row-major), AWQ-packed along N
//   scales   [K/G, N]   f32
//   qzeros   [K/G, N/8] int32, AWQ-packed along N
//   k, n     full output dims (n must be a multiple of 8)
//   group_size G (>0); use group_size == k for a single per-column group
//   out_bf16 [K, N]     bf16 bit patterns (caller-owned, K*N entries)
// Aborts (VT_CHECK) on null buffers, non-positive dims, n % 8 != 0, or a
// group_size that does not divide k.
void DequantAwq4ToBf16(const int32_t* qweight, const float* scales,
                       const int32_t* qzeros, int64_t k, int64_t n,
                       int64_t group_size, uint16_t* out_bf16);

// Dequantize a GPTQ INT4 weight matrix to bf16.
//   qweight  [K/8, N]         int32 (row-major), packed along K, standard order
//   scales   [num_groups, N]  f32
//   qzeros   [num_groups, N/8] int32, packed along N, standard order
//   g_idx    [K] int32 or nullptr (nullptr => group(k) = k / group_size)
//   k, n     full output dims (k must be a multiple of 8, n % 8 == 0)
//   group_size G (>0)
//   zero_offset 1 for GPTQv1 (classic AutoGPTQ), 0 for GPTQv2 (q_gemm.cu:201-202)
//   out_bf16 [K, N]          bf16 bit patterns (caller-owned, K*N entries)
// Aborts (VT_CHECK) on null buffers, non-positive dims, k % 8 != 0, n % 8 != 0,
// a group_size that does not divide k, or zero_offset outside {0, 1}.
void DequantGptq4ToBf16(const int32_t* qweight, const float* scales,
                        const int32_t* qzeros, const int32_t* g_idx, int64_t k,
                        int64_t n, int64_t group_size, int zero_offset,
                        uint16_t* out_bf16);

}  // namespace vllm
