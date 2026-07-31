#pragma once
// Tier C2 of the cross-arch merged-GEMM fold plan
// (.agents/specs/arch-fusion-fold-plan-2026-07-30.md): the vision-tower
// attention QKV fold where the weight is a single resident merged
// [q_dim+k_dim+v_dim, K] tensor and only the compute is split.
//
// FusedMergedQkvBiasSplit performs ONE vt::MatmulBT over the merged-QKV weight,
// an OPTIONAL fused per-[q_dim+k_dim+v_dim] BIAS epilogue, then a contiguous
// vt::QkvSplit into three dense outputs. It is BYTE-IDENTICAL to the unfused
// baseline 3x { vt::MatmulBT(row-slice) [+ vt::Add(bias-slice)] } because:
//   - the merged GEMM's per-output-row reduction is exactly the math of each
//     separate row-slice GEMM (wider N, no cross-row mixing, same K accum);
//   - the merged [3d] bias add over the [M, 3d] output broadcasts per COLUMN,
//     so it equals three [d] bias adds applied to the three contiguous thirds;
//   - vt::QkvSplit is a pure contiguous chunk (copy) with no arithmetic.
//
// The NEW piece vs the text bf16 merged-QKV descriptor (D1), whose AttnBlock
// VT_CHECKs qkv_bias.Empty(), is the fused per-[3d] BIAS epilogue carried here
// for the bias-bearing vision path (e.g. Qwen3-VL vision qkv). Towers without a
// qkv bias (e.g. Gemma-4 vision) pass qkv_bias == nullptr and reuse the same
// merged-GEMM + contiguous-split path (their per-slice clamp epilogue stays
// outside, on the split outputs, exactly as before).
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm::models {

// out layout: qkv_scratch is [M, q_dim + k_dim + v_dim] (contiguous); q_out,
// k_out, v_out are dense [M, q_dim/k_dim/v_dim]. qkv_w is the resident merged
// weight [q_dim + k_dim + v_dim, K]; x is [M, K]. qkv_bias (optional) is the
// merged rank-1 [q_dim + k_dim + v_dim] bias.
inline void FusedMergedQkvBiasSplit(vt::Queue& q, vt::Tensor& qkv_scratch,
                                    vt::Tensor& q_out, vt::Tensor& k_out,
                                    vt::Tensor& v_out, const vt::Tensor& x,
                                    const vt::Tensor& qkv_w,
                                    const vt::Tensor* qkv_bias) {
  vt::MatmulBT(q, qkv_scratch, x, qkv_w);
  if (qkv_bias != nullptr) vt::Add(q, qkv_scratch, qkv_scratch, *qkv_bias);
  vt::QkvSplit(q, q_out, k_out, v_out, qkv_scratch);
}

}  // namespace vllm::models
