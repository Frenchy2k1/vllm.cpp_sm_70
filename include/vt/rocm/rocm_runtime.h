// HIP-free declarations of the two probes the ROCm backend exposes upward
// (BACKEND-ROCM, W0). Mirrors the role of src/vt/vulkan/vulkan_context.h in the
// Vulkan skeleton: the engine-side platform TU asks "is there a device?" without
// ever including <hip/hip_runtime.h>, which is what keeps src/vllm/ free of
// vendor headers and lets the platform leg be read as plain C++.
//
// Both are defined in src/vt/rocm/rocm_backend.hip and both are noexcept: they
// are called from static-init registrars, where throwing would abort the process
// at load time on a machine that merely happens to have HIP installed.
#pragma once

#include <string>

namespace vt::rocm {

// True when the HIP runtime is present AND reports at least one usable device.
// False on any error, which is the conservative answer: a platform whose
// backend() would throw is worse than an unregistered platform, because
// CurrentPlatform() must be able to fall through to CPU.
bool DeviceAvailable() noexcept;

// hipDeviceProp_t::gcnArchName for `index` (e.g. "gfx1100", or
// "gfx942:sramecc+:xnack-"), empty when the device is not present. For error
// messages, test output and bug reports: a ROCm issue that names the arch is
// actionable, one that says "an AMD GPU" is not.
std::string DeviceArchName(int index) noexcept;

}  // namespace vt::rocm
