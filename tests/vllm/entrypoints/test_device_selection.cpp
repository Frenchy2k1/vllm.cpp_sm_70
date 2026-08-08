// ARCH-ONE-SURFACE ROW 8: mutation-sensitive integration gate for canonical
// platform lookup and explicit queue selection. This is a separate executable
// because it deliberately registers a distinctive fake platform/backend in the
// otherwise-unused XPU slot; process isolation keeps the global registries from
// affecting the general platform and loader suites.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include "vllm/config/device.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"

namespace {

class FakeXpuBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override { std::free(p); }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    ++create_queue_calls;
    return vt::Queue{vt::Device{vt::DeviceType::kXPU, 17}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }

  int create_queue_calls = 0;
};

class FakeXpuPlatform final : public vllm::platforms::Platform {
 public:
  explicit FakeXpuPlatform(FakeXpuBackend& backend) : backend_(backend) {}

  vt::DeviceType device_type() const override { return vt::DeviceType::kXPU; }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {};
  }
  std::vector<vt::DType> supported_dtypes() const override {
    return {vt::DType::kBF16};
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    return {};
  }

 private:
  FakeXpuBackend& backend_;
};

FakeXpuBackend& Backend() {
  static FakeXpuBackend backend;
  return backend;
}

FakeXpuPlatform& Platform() {
  static FakeXpuPlatform platform(Backend());
  return platform;
}

void RegisterDistinctivePlatform() {
  vt::RegisterBackend(vt::DeviceType::kXPU, &Backend());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kXPU, &Platform());
}

}  // namespace

TEST_CASE("platform lookup preserves a distinctive non-CPU platform identity") {
  RegisterDistinctivePlatform();

  vllm::platforms::Platform* found =
      vllm::platforms::FindPlatformByName("xpu");
  REQUIRE(found == &Platform());
  CHECK(found->device_type() == vt::DeviceType::kXPU);

  for (std::string_view invalid : {"xpu-junk", "xpu ", " xpu", "XPU",
                                   "xp", "xpu-extra"}) {
    CAPTURE(invalid);
    CHECK(vllm::platforms::FindPlatformByName(invalid) == nullptr);
  }
}

TEST_CASE("explicit named-device queue propagates the found platform type") {
  RegisterDistinctivePlatform();
  // Slot 2 remains publicly named "cuda". Registering the XPU-shaped fake in
  // that lookup slot separates the canonical public name from the platform's
  // returned type: a hidden CPU/CUDA constant cannot satisfy this assertion.
  vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, &Platform());
  Backend().create_queue_calls = 0;

  vt::Queue queue = vllm::entrypoints::SelectQueueForModel(
      "DistinctiveArchitecture", vllm::Device::kNamedPlatform);
  CHECK(queue.device.type == vt::DeviceType::kXPU);
  CHECK(queue.device.index == 17);
  CHECK(Backend().create_queue_calls == 1);
}
