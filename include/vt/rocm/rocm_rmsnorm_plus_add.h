#pragma once
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vt::rocm {
// out = rmsnorm(x, w) + addend  (Gemma-4 residual join)
void RmsNormPlusAddRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                        const Tensor& addend, const RmsNormArgs& args);

// out = rmsnorm(rmsnorm(x1,w1)+rmsnorm(x2,w2), w3) + residual
void DualRmsNormPlusResRocm(Queue& q, Tensor& out, const Tensor& x1, const Tensor& w1,
                            const Tensor& x2, const Tensor& w2, const Tensor& w3,
                            const Tensor& residual, const RmsNormArgs& args);
}  // namespace vt::rocm
