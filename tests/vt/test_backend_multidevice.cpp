// Scale-out W2 multi-device backend registry (BACKEND-DISTRIBUTED-TP,
// .agents/specs/scale-out-distributed.md §W2). Proves the registry now addresses
// a SPECIFIC Device{type,index} — the seam tensor/pipeline parallel needs to hold
// one Backend* per discrete GPU — while the single-device path (index 0) stays
// BYTE-NEUTRAL: the type-level API and Device{type,0} resolve the identical
// Backend*, and registering a second device at index 1 leaves index 0 untouched.
//
// Hardware-free: two fake backends stand in on the otherwise-unused kXPU slots
// (the same technique test_op_provider / test_reference_tier use), so no GPU is
// required to exercise per-index registration + Alloc/Free/CreateQueue routing.
#include <doctest/doctest.h>

#include <cstdint>

#include "vt/backend.h"
#include "vt/device.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;

namespace {

// A distinguishable fake: every Alloc returns its unique tag so a test can prove
// WHICH device's backend served the call. Owns nothing real.
class FakeBackend final : public Backend {
 public:
  explicit FakeBackend(uintptr_t tag) : tag_(tag) {}

  void* Alloc(size_t) override { return reinterpret_cast<void*>(tag_); }
  void Free(void*) override { ++frees_; }
  void Memset(Queue&, void*, int, size_t) override {}
  void Copy(Queue&, void*, const void*, size_t) override {}
  Queue CreateQueue() override {
    Queue q;
    q.device = Device{DeviceType::kXPU, index_};
    q.handle = reinterpret_cast<void*>(tag_);
    return q;
  }
  bool UnifiedMemory() const override { return true; }

  void set_index(int32_t i) { index_ = i; }
  int frees() const { return frees_; }

 private:
  uintptr_t tag_;
  int32_t index_ = 0;
  int frees_ = 0;
};

}  // namespace

TEST_CASE("Device{type,0} and the type-level API resolve the identical backend (byte-neutral)") {
  static FakeBackend dev0(0xA0);
  dev0.set_index(0);
  vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &dev0);

  // Registering at Device{...,0} writes the SAME slot the type-level getter reads.
  CHECK(&vt::GetBackend(DeviceType::kXPU) == &dev0);
  CHECK(&vt::GetBackend(Device{DeviceType::kXPU, 0}) == &dev0);
  CHECK(vt::TryGetBackend(Device{DeviceType::kXPU, 0}) == &dev0);

  // The device-explicit Alloc for index 0 goes through that identical backend.
  CHECK(vt::Alloc(Device{DeviceType::kXPU, 0}, 16) == reinterpret_cast<void*>(0xA0));
}

TEST_CASE("A second device at index 1 is addressable and leaves index 0 untouched") {
  static FakeBackend dev0(0xB0);
  static FakeBackend dev1(0xB1);
  dev0.set_index(0);
  dev1.set_index(1);
  vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &dev0);
  vt::RegisterBackend(Device{DeviceType::kXPU, 1}, &dev1);

  // Each index resolves its OWN backend.
  CHECK(&vt::GetBackend(Device{DeviceType::kXPU, 0}) == &dev0);
  CHECK(&vt::GetBackend(Device{DeviceType::kXPU, 1}) == &dev1);

  // The type-level getter (and thus every single-device call site) still sees
  // index 0 only — the multi-device change did not perturb it.
  CHECK(&vt::GetBackend(DeviceType::kXPU) == &dev0);

  // Alloc routes by index: device 1 is served by dev1, device 0 by dev0.
  CHECK(vt::Alloc(Device{DeviceType::kXPU, 1}, 8) == reinterpret_cast<void*>(0xB1));
  CHECK(vt::Alloc(Device{DeviceType::kXPU, 0}, 8) == reinterpret_cast<void*>(0xB0));

  // CreateQueue/DestroyQueue route to the index's backend too.
  Queue q = vt::CreateQueue(Device{DeviceType::kXPU, 1});
  CHECK(q.device == (Device{DeviceType::kXPU, 1}));
  vt::DestroyQueue(q);
  CHECK(q.id == 0);
}

TEST_CASE("An unregistered index is a clean nullptr, not a stale index-0 hit") {
  static FakeBackend dev0(0xC0);
  dev0.set_index(0);
  vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &dev0);

  // Index 5 was never registered: the non-throwing probe is null (it must NOT
  // fall back to index 0), and out-of-range indices are null too.
  CHECK(vt::TryGetBackend(Device{DeviceType::kXPU, 5}) == nullptr);
  CHECK(vt::TryGetBackend(Device{DeviceType::kXPU, -1}) == nullptr);
  CHECK(vt::TryGetBackend(Device{DeviceType::kXPU, 999}) == nullptr);
}
