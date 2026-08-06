// MiniMax-H3 DiT — the device-resident-forward kernel seam (brick H3-2b).
//
// The CPU reference (`MiniMaxH3DitForward`, minimax_h3.cpp) computes into host
// `std::vector<float>` buffers. `MiniMaxH3DitForwardDevice` runs the SAME graph
// with every activation resident in device memory, so the block stack never
// round-trips through the host.
//
// ─── WHY THIS TABLE IS SO SMALL ──────────────────────────────────────────────
// Almost the whole forward is already covered by tuned SHARED vt:: ops, and the
// port reuses them rather than growing a private kernel set:
//
//   Linear                 -> vt::MatmulBT + vt::Add (rank-1 row-broadcast bias)
//   RmsNormRows            -> vt::RmsNorm
//   qkv split              -> vt::QkvSplit   ([q_all|k_all|v_all] is its exact layout)
//   ApplyRope (3-axis)     -> vt::RopeFromCache
//   MLP silu(gate)*up      -> vt::SiluAndMul (fc1 emits [gate;up] per row)
//   residual add           -> vt::Add
//   row gather / scatter   -> vt::IndexSelect / vt::IndexCopy
//   packed varlen attention-> vt::DFlashBlockAttention(causal=false)
//
// H3's RoPE looks exotic but is NOT: `_apply_rope`
// (minimax_h3_transformer.py:233-244) is plain NeoX rotate_half over the leading
// `rot_dim` head dims. Only the ANGLES are unusual — three axes (t,h,w) derived
// from the fp64 position grid instead of one scalar position. Since
// vt::RopeFromCache takes a per-row [S, rot_dim] cos/sin cache and an index, the
// port builds that cache from the position grid and passes positions=arange(S),
// which reproduces ApplyRope without a bespoke kernel.
//
// That leaves exactly THREE ops the shared surface does not provide.
//
// SEAM: one OpProvider entry (vt/ops.h: kMiniMaxH3) whose `fn` points at a static
// kernels-struct of typed launchers, mirroring the kLaguna precedent. Registered
// for BOTH kCPU (cpu_ops.cpp) and kCUDA (cuda_minimax_h3.cu) — unlike Laguna,
// which is CUDA-only — so the device forward is covered by CPU CI as well as by
// a real GPU.
#pragma once

#include <cstdint>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm::minimax_h3 {

struct MiniMaxH3DeviceKernels {
  // _modulate_scale_shift (minimax_h3_transformer.py:183-192), in place:
  //   x[r, i] = x[r, i] * (1 + scale[idx[r], i]) + shift[idx[r], i]
  // `idx` is the per-row AdaLN row selector (combined_indices for a block,
  // inverse_indices for the final layer), i32 [rows] on the device.
  //
  // `src_stride` is the ROW stride of shift/scale, which are strided CHUNK VIEWS
  // into the flat AdaLN projection [rows, expand*width] -- chunk c of row r lives
  // at r*src_stride + c*width, so the rows are width-wide but src_stride apart.
  // Passing it explicitly is deliberate: treating the views as contiguous reads
  // the wrong memory and yields a plausible-but-wrong result.
  //
  // `dtype` is the STREAM dtype (kF32 or kBF16) of x/shift/scale. Arithmetic is
  // always f32; a bf16 stream rounds on STORE, which is what makes the cast point
  // identical to upstream's -- there is no separate rounding pass.
  void (*modulate_scale_shift)(vt::Queue&, void* x, const void* shift, const void* scale,
                               const int32_t* idx, int64_t rows, int64_t width,
                               int64_t src_stride, vt::DType dtype);
  // _modulate_gate (minimax_h3_transformer.py:195-204), accumulating in place:
  //   residual[r, i] += gate[idx[r], i] * other[r, i]
  // `src_stride` and `dtype` as for modulate_scale_shift.
  void (*modulate_gate)(vt::Queue&, void* residual, const void* gate, const void* other,
                        const int32_t* idx, int64_t rows, int64_t width, int64_t src_stride,
                        vt::DType dtype);
  // Plain elementwise SiLU in place: x[i] = x[i] * sigmoid(x[i]). The shared op
  // set has SiluAndMul / MoeSiluMul (both GATED forms) but no ungated SiLU, which
  // the time embedder (:272-285) and the AdaLN projection (:555-561) both need.
  // f32 only: both call sites are fp32 ISLANDS.
  void (*silu)(vt::Queue&, float* x, int64_t n);
};

// Resolver. Throws when nothing is registered for (kMiniMaxH3, device) — which
// cannot happen for kCPU/kCUDA in a normal build, but keeps the failure explicit
// rather than a null dereference on an unexpected backend.
const MiniMaxH3DeviceKernels* MiniMaxH3Device(vt::DeviceType device);
// True iff the table is registered for `device` (guards the device forward).
bool MiniMaxH3DeviceKernelsAvailable(vt::DeviceType device);

}  // namespace vllm::minimax_h3
