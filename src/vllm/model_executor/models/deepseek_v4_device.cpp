// DeepSeek-V4-Flash W7-device — the OpProvider-seam resolvers for the four V4
// device kernel families. Always compiled (CPU + CUDA); it holds NO CUDA code —
// it only looks up the per-family kernels-struct the CUDA TU (cuda_deepseek_v4.cu)
// registered under kDeepseekV4{Mhc,Dsa,Compressor,Moe} on kCUDA. On a CPU-only
// build nothing is registered for those (op, kCUDA), so GetOp() throws and
// ForwardDevice surfaces a clean device-only error. See deepseek_v4_device.h.
#include "vllm/model_executor/models/deepseek_v4_device.h"

#include "vt/ops.h"  // OpId, GetOp, OpRegistered

namespace vllm::deepseek_v4 {

const MhcDeviceKernels* MhcDevice() {
  return static_cast<const MhcDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kCUDA));
}
const DsaDeviceKernels* DsaDevice() {
  return static_cast<const DsaDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Dsa, vt::DeviceType::kCUDA));
}
const CompressorDeviceKernels* CompressorDevice() {
  return static_cast<const CompressorDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Compressor, vt::DeviceType::kCUDA));
}
const MoeDeviceKernels* MoeDevice() {
  return static_cast<const MoeDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Moe, vt::DeviceType::kCUDA));
}

bool V4DeviceKernelsAvailable() {
  return vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kDeepseekV4Dsa, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kDeepseekV4Compressor, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kDeepseekV4Moe, vt::DeviceType::kCUDA);
}

}  // namespace vllm::deepseek_v4
