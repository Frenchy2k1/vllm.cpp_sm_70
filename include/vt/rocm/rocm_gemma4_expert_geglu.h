// Custom RDNA4 Gemma-4 Expert GeGLU (decode T=1).
#pragma once

#include "vt/device.h"

namespace vt::rocm {

// y[H] = beta*y + alpha * Down(Gelu(Gate(x))*Up(x))
// w_gu [2I,H] bf16, w_dn [H,I] bf16, x[H] bf16
bool ExpertGeGLUBf16M1Rocm(Queue& q, void* y, const void* x, const void* w_gu, const void* w_dn,
                           int I, int H, float alpha, float beta);

// Top-k sequential mix into ysum.
bool ExpertGeGLUBf16TopKM1Rocm(Queue& q, void* ysum, const void* x, const void* const* w_gu,
                               const void* const* w_dn, const float* wts, int G, int I, int H);

}  // namespace vt::rocm
