// Portable fused helpers used by Gemma4 (and reusable elsewhere).
// Model code MUST call these — never vt::rocm::* — so CPU/CUDA/Vulkan link.
#pragma once

#include <cstdint>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vt {

void RmsNormPlusAdd(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                    const Tensor& addend, const RmsNormArgs& args);

void DualRmsNormPlusRes(Queue& q, Tensor& out, const Tensor& x1, const Tensor& w1,
                        const Tensor& x2, const Tensor& w2, const Tensor& w3,
                        const Tensor& residual, const RmsNormArgs& args);

void GeluMulSeparate(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                     DType dtype);

void MatmulBTAlphaBeta(Queue& q, void* out, const void* a, const void* b, int M, int N, int K,
                       float alpha, float beta, DType dtype);

void MatmulBTFp8Channel(Queue& q, void* out, const void* a, const void* b_fp8,
                        const void* scale_bf16, int M, int N, int K, float alpha, float beta);

bool ExpertGeGLUBf16TopKM1(Queue& q, void* ysum, const void* x, const void* const* w_gu,
                           const void* const* w_dn, const float* wts, int G, int I, int H);

}  // namespace vt
