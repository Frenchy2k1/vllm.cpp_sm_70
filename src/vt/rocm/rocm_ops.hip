// ROCm op-table registration (BACKEND-ROCM, W0) — seam 3 of .agents/backends.md.
//
// **UNBUILT.** Never compiled: no AMD GPU on the authoring machine.
//
// This file is the whole of "adding an op to a backend": a static Registrar that
// calls RegisterOp(OpId, DeviceType::kROCM, fn) once per kernel, exactly as
// src/vt/cuda/cuda_ops.cu:3512-3520 and src/vt/vulkan/vulkan_ops.cpp do. There is
// no selector to edit, no model to teach, no dispatch table to extend: an op is
// registered or it is not, and vt::GetOp answers accordingly.
//
// ONE op is registered today (RmsNorm). What happens to the other 105 OpIds
// depends on the board, and both behaviours are correct:
//   * UNIFIED memory (Radeon 780M, Strix Halo): the portable reference tier
//     (include/vt/op_provider.h:186-224) installs lazily on the first GetOp miss
//     and serves them from the CPU kernels, so a model runs end to end, slowly.
//     VT_OP_PROVIDER_STATS=1 prints each fallback the first time it is taken —
//     that print-out IS the prioritized kernel to-do list for M3.
//   * DISCRETE (7900 XTX): the tier is not eligible and GetOp throws. Register
//     the op or it does not run. This is the memory-safety gate, not a policy.
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt::rocm {

// Defined in rocm_rmsnorm.hip.
void RmsNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual);

namespace {

// Registers during static init, before main(), like every other backend's op
// table. Harmless on a machine with no AMD GPU: filling the table registers
// nothing about devices, and the kROCM backend never registers there, so no
// kROCM queue can exist to dispatch with.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernelRocm)));
  }
} registrar;

}  // namespace
}  // namespace vt::rocm
