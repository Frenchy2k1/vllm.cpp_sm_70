// vllm.cpp — load-time Marlin NVFP4 repack (host-callable, torch-free lift of
// vLLM's marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin). See
// src/vt/cuda/cuda_marlin_repack.cu. Built only under VT_MARLIN_NVFP4.
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace vt::cuda {

// Repack one expert's modelopt packed fp4 weight [N=out, K/2] uint8 (device)
// into Marlin interleaved layout [K/16, N*2] int32 (device, caller-allocated,
// zero-init not required). size_k = in features (K), size_n = out features (N).
void MarlinRepackExpertWeight(void* stream, int device, uint32_t* out_weight,
                              const uint8_t* packed_nk2, int size_k, int size_n);

// Process one expert's fp8-e4m3 block scales [N=out, K/16] uint8 (device) into
// Marlin S0E5M3 scales [K/16, N] uint8 (device, caller-allocated). scale_factor
// is the shared per-GEMM combined_scale_factor (power of 2 >= 1).
void MarlinProcessExpertScales(void* stream, const uint8_t* scale_nk16, uint8_t* out_scale,
                               int size_k, int size_n, float scale_factor);

// MXFP4 variant: process one expert's E8M0 (UE8M0) block scales
// [N=out, K/32] uint8 (device, group_size 32) into Marlin's permuted E8M0 scale
// layout [K/32, N] uint8 (device, caller-allocated). Mirrors vLLM
// mxfp4_marlin_process_scales (marlin_utils_fp4.py:125-139, bf16 activation):
// marlin_permute_scales(group_size=32) + the within-4 [0,2,1,3] "fit the fp8
// dequant layout" reorder, then keep the value as an E8M0 byte (NO scale_factor,
// NO *128/S0E5M3, NO global — the `.to(bf16).to(e8m0)` round-trip is identity for
// the exact-pow2 E8M0 values). The Marlin kernel decodes each byte as
// 2^(byte-127) (marlin_template.h:1497-1510). No combined scale factor, no global.
void MarlinProcessExpertScalesMxfp4(void* stream, const uint8_t* scale_nk32,
                                    uint8_t* out_scale, int size_k, int size_n);

// combined_scale_factor across a set of experts' host fp8 scale buffers (all
// experts of one Marlin GEMM share it: gate+up for w13, down alone for w2).
float MarlinNvfp4CombinedScaleFactor(const std::vector<const uint8_t*>& host_scale_bufs,
                                     const std::vector<size_t>& lens);

// nvfp4_marlin_process_global_scale(scale2, bf16) / combined_scale_factor.
float MarlinNvfp4ProcessGlobalScale(float scale2, float combined_sf);

// SM count for the device (workspace sizing).
int MarlinDeviceSms(int device);

// --- moe_align_block_size (torch-free) ---------------------------------------
// Pick block_size_m (vLLM marlin_moe.py logic, clamped >=16).
int MarlinMoeAlignBlockSizeSelect(int num_tokens, int top_k, int num_experts);
// Output buffer sizes for the given (num_tokens, top_k, E, block_size).
void MarlinMoeAlignSizes(int num_tokens, int top_k, int num_experts, int block_size,
                         int* max_num_tokens_padded, int* max_num_blocks);
// Launch moe_align. topk_ids [num_tokens, top_k] i32 (device). Outputs (device,
// caller-allocated to the MarlinMoeAlignSizes sizes): sorted_ids, expert_ids,
// num_tokens_post_pad[1].
void MarlinMoeAlignBlockSize(void* stream, const int32_t* topk_ids, int num_tokens, int top_k,
                             int num_experts, int block_size, int32_t* sorted_ids,
                             int32_t* expert_ids, int32_t* num_tokens_post_pad);

// TEST-ONLY: the original single-block serial-scan align reference, kept so the
// byte-exact parity test can assert the parallel default matches it. Same
// outputs and sizing as MarlinMoeAlignBlockSize. Not on any production path.
void MarlinMoeAlignBlockSizeSerial(void* stream, const int32_t* topk_ids, int num_tokens,
                                   int top_k, int num_experts, int block_size, int32_t* sorted_ids,
                                   int32_t* expert_ids, int32_t* num_tokens_post_pad);

}  // namespace vt::cuda
