// Tenstorrent leg of the Platform seam (BACKEND-TENSTORRENT, W0 skeleton).
// Self-registers kTENSTORRENT via a static Registrar, copying the
// `src/vllm/platforms/cpu.cpp` / `cuda.cpp` / `vulkan.cpp` registrar idiom.
// Compiled only in Tenstorrent builds (CMake target_sources gate).
//
// NO UPSTREAM MIRROR. vLLM has no `vllm/platforms/tenstorrent.py` and no
// Tenstorrent path anywhere in its tree; this is a recorded extension of the
// `vllm/platforms/interface.py:134-229 class Platform` seam
// (.agents/porting-inventory.md §9, item 15). Where a value has an upstream
// analogue the analogue is cited; where it does not, that is said outright.
//
// Deliberately plain C++, not a ttnn TU — everything Tenstorrent-specific is
// reached through the vt::Backend virtuals, so the engine-side platform tree
// stays free of ttnn headers.
#include "vllm/platforms/interface.h"

#include <vector>

#include "vt/backend.h"
#include "vt/tenstorrent/tenstorrent_device.h"

namespace vllm::platforms {
namespace {

class TenstorrentPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kTENSTORRENT; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kTENSTORRENT); }

  // interface.py:409-415 get_device_capability. Tenstorrent's Tensix cores
  // have no CUDA-SM-shaped "compute capability" to report; the base {0, 0}
  // ("no meaningful compute capability", backend.h) is the honest answer,
  // same as CPU.
  DeviceCapability get_device_capability() const override { return DeviceCapability{}; }

  // W0 registers exactly one op (kMatmul) in F32 only — see tenstorrent_ops.cpp.
  // Widening this list ahead of the kernels that back it would be a claim we
  // cannot honour.
  std::vector<DType> supported_dtypes() const override { return {DType::kF32}; }

  // Discrete PCIe device (tenstorrent_backend.cpp UnifiedMemory()==false); no
  // host-weight-release/pool-cap policy has been worked out for it yet, so the
  // default (empty) ResidencyPolicy is the honest answer for W0 — same
  // non-decision Vulkan's W0 makes, for the same reason (no discrete-GPU
  // staging path implemented yet).
  ResidencyPolicy residency_policy() const override { return {}; }

  // No attention kernel exists for this device (only kMatmul is registered).
  // An EMPTY list is the honest and mechanically correct answer:
  // SelectAttentionBackendName walks the list and takes the first REGISTERED
  // name, so returning one here would let selection hand back a backend whose
  // kernels do not exist instead of throwing loudly.
  std::vector<std::string> get_attn_backend_priority(const AttnSelectorConfig&) const override {
    return {};
  }
};

// Registers kTENSTORRENT during static init (registration completes before
// main() per the interface.h contract). Stays silent on a Tenstorrent-enabled
// build running where no Blackhole card is present — the exact shape of
// vulkan.cpp's and metal.cpp's registrars, which likewise probe the DEVICE
// rather than trusting another TU's initializer (static-init order across TUs
// is unspecified). CurrentPlatform() (platform.cpp) walks the priority array
// and must be able to fall through past an unregistered kTENSTORRENT to CPU.
struct Registrar {
  Registrar() noexcept {
    if (!vt::tenstorrent::DeviceAvailable()) return;
    static TenstorrentPlatform platform;
    RegisterPlatform(DeviceType::kTENSTORRENT, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
