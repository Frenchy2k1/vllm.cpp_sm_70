// Laguna device-kernel resolver — the OpProvider-seam lookup for the kLaguna glue
// table (registered by cuda_laguna.cu on kCUDA). Always compiled (CPU + CUDA); holds
// NO CUDA code. On a CPU-only build nothing is registered for (kLaguna,kCUDA), so
// GetOp throws and LagunaDeviceKernelsAvailable() returns false — the resident decode
// path stays gated off and the host compose path runs. See laguna_device.h.
#include "vllm/model_executor/models/laguna_device.h"

#include "vt/ops.h"  // OpId, GetOp, OpRegistered

namespace vllm::laguna {

const LagunaDeviceKernels* LagunaDevice() {
  return static_cast<const LagunaDeviceKernels*>(
      vt::GetOp(vt::OpId::kLaguna, vt::DeviceType::kCUDA));
}

bool LagunaDeviceKernelsAvailable() {
  return vt::OpRegistered(vt::OpId::kLaguna, vt::DeviceType::kCUDA);
}

}  // namespace vllm::laguna
