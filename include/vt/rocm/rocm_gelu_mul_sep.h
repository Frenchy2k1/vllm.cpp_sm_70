#pragma once
#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {
// out[i] = gelu_tanh(gate[i]) * up[i]  for i in [0,n)
void GeluMulSeparateRocm(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                         DType dtype);
}  // namespace vt::rocm
