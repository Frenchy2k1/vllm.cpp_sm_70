// See include/vllm/config/device.h. Ported from: vllm/config/device.py @
// 555967922 (Device Literal:13 — the name set; DeviceConfig:61-66 — an explicit
// device is assigned verbatim, never substituted).
#include "vllm/config/device.h"

#include <stdexcept>

namespace vllm {

Device DeviceFromString(const std::string& value) {
  if (value == "auto") {
    return Device::kAuto;
  }
  if (value == "cpu") {
    return Device::kCPU;
  }
  if (value == "cuda") {
    return Device::kNamedPlatform;
  }
  // Mirrors pydantic rejecting a value outside the Device Literal
  // (vllm/config/device.py:13). "tpu"/"xpu" are upstream names this build
  // cannot serve explicitly yet; they are rejected with the same message shape
  // rather than mapped to something else.
  throw std::invalid_argument("Unknown device: " + value +
                              " (expected one of: auto, cpu, cuda)");
}

const char* DeviceName(Device device) {
  switch (device) {
    case Device::kAuto:
      return "auto";
    case Device::kCPU:
      return "cpu";
    case Device::kNamedPlatform:
      return "cuda";
  }
  return "invalid";
}

}  // namespace vllm
