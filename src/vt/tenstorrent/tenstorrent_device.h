// Shared Tenstorrent mesh-device lifecycle for the backend, op providers, and
// the platform registrar (BACKEND-TENSTORRENT W0, .agents/specs/
// tenstorrent-backend.md). vllm.cpp original; no upstream mirror.
#pragma once

#include <memory>

// ttnn::MeshDevice (ttnn/api/ttnn/device.hpp) is a `using` alias for this real
// type, brought into ttnn:: scope via `using namespace device;` there — the
// alias itself is not forward-declarable, so this header names the concrete
// type it resolves to.
namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace vt::tenstorrent {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// True iff at least one Tenstorrent device is enumerable on this host. Cheap
// (tt::tt_metal::GetNumAvailableDevices(), no device open) — the same
// runtime-probe-before-registering shape vulkan_context.h's Available() uses,
// so a Tenstorrent-enabled build on a host with no Blackhole card registers
// nothing instead of throwing during static init.
bool DeviceAvailable();

// Opens (lazily, once) and returns the single process-wide mesh device this
// W0 skeleton targets — device index 0 only, no multi-device mesh. Throws if
// DeviceAvailable() was not already checked true. DELIBERATELY LEAKED (see
// the .cpp): a plain static shared_ptr's destructor reproducibly segfaults at
// process exit, inside MeshDevice's own teardown chain, a cross-DSO static
// destruction ordering hazard the real device-driver state survives fine
// either way.
MeshDevice& SharedMeshDevice();

}  // namespace vt::tenstorrent
