// Tenstorrent backend — the `vt::Backend` implementation + its static
// registrar. BACKEND-TENSTORRENT, W0 skeleton
// (.agents/specs/tenstorrent-backend.md). vllm.cpp original: vLLM has no
// Tenstorrent platform anywhere, so there is no upstream mirror; the SHAPE is
// the CPU reference `src/vt/cpu/cpu_backend.cpp` (the 6 pure virtuals).
//
// SCOPE / STUBS — stated plainly so nothing here reads as more than it is:
//   * Blackhole is a DISCRETE PCIe device (no shared host/device address
//     space), unlike Vulkan's W0 here (unified on its GB10 target, so its
//     Alloc returns a directly host-dereferenceable mapped pointer). That
//     trick does not apply here: vt::Tensor.data for this backend is plain
//     HOST memory (identical to the CPU backend's aligned_alloc), and every
//     registered op stages host<->device itself via ttnn::Tensor::from_vector
//     / to_vector (hands-on validated on real Blackhole hardware — see the
//     spec's "Resolved: hands-on spike result"). That is CORRECT — every op
//     result is bit-for-bit what the device actually computed — but it pays a
//     host round-trip per call; avoiding that (keeping vt::Tensor storage
//     device-resident between ops) is exactly the deferred work the spec
//     flags as the shipping-performance follow-up, not attempted in W0.
//   * `SupportsGraphCapture()` stays FALSE. tt_metal trace capture
//     (begin_trace_capture/end_trace_capture/replay_trace) is the eventual
//     mapping the spec already names; not implemented here.
//   * `UnifiedMemory()` is `false`: this is the real hardware property (a
//     discrete card over PCIe), independent of this W0's host-staging
//     implementation detail above. It also means op_provider.h's portable CPU
//     reference tier stays gated off for this device — no free correctness
//     net for an unregistered op.
#include "vt/backend.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <cstdlib>
#include <cstring>

namespace vt::tenstorrent {
namespace {

class TenstorrentBackend final : public Backend {
 public:
  void* Alloc(size_t bytes) override {
    VT_CHECK(bytes <= SIZE_MAX - 63, "tenstorrent alloc size overflow");
    void* p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
    VT_CHECK(p != nullptr, "tenstorrent alloc failed");
    return p;
  }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int value, size_t bytes) override { std::memset(p, value, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kTENSTORRENT, 0}, nullptr}; }

  // See the file-level SCOPE note: vt::Tensor storage is host memory for this
  // backend, the real device is discrete, and unified-memory-gated code paths
  // (the op_provider.h reference tier in particular) must stay off.
  bool UnifiedMemory() const override { return false; }
};

struct Registrar {
  Registrar() noexcept {
    // Same runtime-probe-before-registering shape as Vulkan's
    // VulkanDeviceAvailable() gate: a Tenstorrent-enabled build on a host
    // with no Blackhole card registers nothing instead of throwing during
    // static init (unspecified TU order rules out trusting another TU's
    // initializer to have probed already).
    if (!DeviceAvailable()) return;
    static TenstorrentBackend backend;
    RegisterBackend(DeviceType::kTENSTORRENT, &backend);
  }
} registrar;

}  // namespace
}  // namespace vt::tenstorrent
