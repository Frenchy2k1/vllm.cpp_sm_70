// Resolvers for the MiniMax-H3 device glue table (vt::OpId::kMiniMaxH3).
// The tables themselves are registered by src/vt/cpu/cpu_minimax_h3.cpp and
// src/vt/cuda/cuda_minimax_h3.cu; this TU only casts the OpProvider result, so it
// links in both CPU-only and CUDA builds.
#include "vllm/model_executor/models/minimax_h3_device.h"

#include "vt/ops.h"

namespace vllm::minimax_h3 {

const MiniMaxH3DeviceKernels* MiniMaxH3Device(vt::DeviceType device) {
  return static_cast<const MiniMaxH3DeviceKernels*>(vt::GetOp(vt::OpId::kMiniMaxH3, device));
}

bool MiniMaxH3DeviceKernelsAvailable(vt::DeviceType device) {
  return vt::OpRegistered(vt::OpId::kMiniMaxH3, device);
}

}  // namespace vllm::minimax_h3
