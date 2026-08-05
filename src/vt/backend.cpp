// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/backend.h"

#include <array>
#include <atomic>

namespace vt {

uint64_t NextQueueId() noexcept {
  static std::atomic<uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

// Async-output primitives (async_utils.py:12-70). The base implementations suit
// any synchronous/unified backend: pinned host memory degenerates to ordinary
// host memory (Copy is already a memcpy) and every event op is a no-op because
// all work submitted to a queue has completed by the time the host observes it.
// CUDA overrides all six with cudaHostAlloc + cudaEvent_t. (CPU inherits these.)
void* Backend::AllocPinned(size_t bytes) { return Alloc(bytes == 0 ? 1 : bytes); }
void Backend::FreePinned(void* p) { Free(p); }
Event Backend::CreateEvent(bool /*blocking*/) { return Event{}; }
void Backend::DestroyEvent(Event&) {}
void Backend::RecordEvent(Event&, Queue&) {}
void Backend::SynchronizeEvent(Event&) {}
// Synchronous backends: every previously submitted op has already completed.
bool Backend::QueryEvent(Event&) { return true; }
void Backend::QueueWaitEvent(Queue&, Event&) {}

void Backend::BeginCapture(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::EndCapture(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::Replay(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void* Backend::EndCaptureGraph(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); return nullptr; }
void Backend::ReplayGraph(Queue&, void*) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::DestroyGraph(void*) {}

namespace {
struct RegistryEntry {
  Backend* backend = nullptr;
  const DeviceResourceOps* resources = nullptr;
};

// One row of `kMaxDevicesPerType` slots per DeviceType (W2 multi-device registry).
// Slot [type][0] is the single-device path the type-level API reads/writes, so a
// build that only touches index 0 is byte-identical to the pre-W2 one-entry-per-
// type table (the whole row is zero-initialized; only [type][0] is ever
// populated on a single-GPU host).
using DeviceRow = std::array<RegistryEntry, kMaxDevicesPerType>;

std::array<DeviceRow, kNumDeviceTypes>& Registry() {
  static std::array<DeviceRow, kNumDeviceTypes> registry{};
  return registry;
}

size_t DeviceIndex(DeviceType type) {
  const size_t index = static_cast<size_t>(type);
  VT_CHECK(index < kNumDeviceTypes, "invalid device type");
  return index;
}

RegistryEntry& Entry(Device device) {
  const size_t slot = static_cast<size_t>(device.index);
  VT_CHECK(device.index >= 0 && slot < kMaxDevicesPerType,
           "device index out of range for the multi-device registry");
  return Registry()[DeviceIndex(device.type)][slot];
}
}  // namespace

Backend& GetBackend(DeviceType type) {
  Backend* b = Registry()[DeviceIndex(type)][0].backend;
  VT_CHECK(b != nullptr, std::string("no backend registered for device type ") +
                             std::to_string(static_cast<int>(type)));
  return *b;
}

Backend* TryGetBackend(DeviceType type) {
  const size_t index = static_cast<size_t>(type);
  if (index >= kNumDeviceTypes) return nullptr;
  return Registry()[index][0].backend;
}

void RegisterBackend(DeviceType type, Backend* backend) {
  VT_CHECK(backend != nullptr, "cannot register a null backend");
  Registry()[DeviceIndex(type)][0].backend = backend;
}

void RegisterDeviceResourceOps(DeviceType type, const DeviceResourceOps* ops) {
  VT_CHECK(ops != nullptr && ops->alloc != nullptr && ops->free != nullptr &&
               ops->create_queue != nullptr && ops->destroy_queue != nullptr,
           "device resource table is incomplete");
  Registry()[DeviceIndex(type)][0].resources = ops;
}

Backend& GetBackend(Device device) {
  Backend* b = Entry(device).backend;
  VT_CHECK(b != nullptr,
           std::string("no backend registered for device type ") +
               std::to_string(static_cast<int>(device.type)) + " index " +
               std::to_string(device.index));
  return *b;
}

Backend* TryGetBackend(Device device) {
  const size_t type = static_cast<size_t>(device.type);
  const size_t slot = static_cast<size_t>(device.index);
  if (type >= kNumDeviceTypes || device.index < 0 || slot >= kMaxDevicesPerType)
    return nullptr;
  return Registry()[type][slot].backend;
}

void RegisterBackend(Device device, Backend* backend) {
  VT_CHECK(backend != nullptr, "cannot register a null backend");
  Entry(device).backend = backend;
}

void RegisterDeviceResourceOps(Device device, const DeviceResourceOps* ops) {
  VT_CHECK(ops != nullptr && ops->alloc != nullptr && ops->free != nullptr &&
               ops->create_queue != nullptr && ops->destroy_queue != nullptr,
           "device resource table is incomplete");
  Entry(device).resources = ops;
}

void* Alloc(Device device, size_t bytes) {
  VT_CHECK(bytes > 0, "device allocation size must be positive");
  RegistryEntry& entry = Entry(device);
  if (entry.resources != nullptr) return entry.resources->alloc(device, bytes);
  if (entry.backend != nullptr) return entry.backend->Alloc(bytes);
  VT_CHECK(device.index == 0,
           "legacy backend allocation shim only supports device index 0");
  return GetBackend(device.type).Alloc(bytes);
}

void Free(Device device, void* p) {
  RegistryEntry& entry = Entry(device);
  if (entry.resources != nullptr) {
    entry.resources->free(device, p);
    return;
  }
  if (entry.backend != nullptr) {
    entry.backend->Free(p);
    return;
  }
  VT_CHECK(device.index == 0, "legacy backend free shim only supports device index 0");
  GetBackend(device.type).Free(p);
}

Queue CreateQueue(Device device) {
  RegistryEntry& entry = Entry(device);
  if (entry.resources != nullptr) return entry.resources->create_queue(device);
  if (entry.backend != nullptr) {
    Queue q = entry.backend->CreateQueue();
    VT_CHECK(q.device == device, "backend returned a queue for the wrong device");
    return q;
  }
  VT_CHECK(device.index == 0,
           "legacy backend queue-creation shim only supports device index 0");
  Queue q = GetBackend(device.type).CreateQueue();
  VT_CHECK(q.device == device, "legacy backend returned a queue for the wrong device");
  return q;
}

void DestroyQueue(Queue& q) {
  if (q.id == 0) return;
  RegistryEntry& entry = Entry(q.device);
  if (entry.resources != nullptr) {
    entry.resources->destroy_queue(q);
  } else {
    Backend& backend =
        entry.backend != nullptr ? *entry.backend : GetBackend(q.device.type);
    VT_CHECK(entry.backend != nullptr || q.device.index == 0,
             "legacy backend queue-destroy shim only supports device index 0");
    backend.Synchronize(q);
    backend.DestroyQueue(q);
  }
  q.handle = nullptr;
  q.id = 0;
}

}  // namespace vt
