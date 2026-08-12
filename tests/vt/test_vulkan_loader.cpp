#include <doctest/doctest.h>

#include <cstring>
#include <string>
#include <vector>

#include "vt/vulkan/vulkan_loader.h"

namespace {

struct FakeLoader {
  bool open_succeeds = true;
  bool lookup_succeeds = true;
  bool create_succeeds = true;
  int closes = 0;
  std::wstring opened;
  std::vector<std::string> lookups;
};

FakeLoader* g_fake = nullptr;

VKAPI_ATTR void VKAPI_CALL FakeCreateInstance() {}

PFN_vkVoidFunction VKAPI_CALL FakeGetInstanceProcAddr(VkInstance, const char* name) {
  if (g_fake != nullptr && g_fake->create_succeeds &&
      std::strcmp(name, "vkCreateInstance") == 0) {
    return reinterpret_cast<PFN_vkVoidFunction>(&FakeCreateInstance);
  }
  return nullptr;
}

void* FakeOpen(void* context, const wchar_t* name) {
  auto* fake = static_cast<FakeLoader*>(context);
  fake->opened = name;
  return fake->open_succeeds ? fake : nullptr;
}

void* FakeLookup(void* context, void*, const char* name) {
  auto* fake = static_cast<FakeLoader*>(context);
  fake->lookups.emplace_back(name);
  return fake->lookup_succeeds
             ? reinterpret_cast<void*>(&FakeGetInstanceProcAddr)
             : nullptr;
}

void FakeClose(void* context, void*) {
  ++static_cast<FakeLoader*>(context)->closes;
}

vt::vulkan::VulkanLibraryOps Ops(FakeLoader* fake) {
  return {fake, FakeOpen, FakeLookup, FakeClose};
}

}  // namespace

TEST_CASE("Win32 Vulkan loader ops require the mandatory entry point and clean up") {
  for (int failure = 0; failure < 3; ++failure) {
    FakeLoader fake;
    fake.open_succeeds = failure != 0;
    fake.lookup_succeeds = failure != 1;
    fake.create_succeeds = failure != 2;
    g_fake = &fake;
    CHECK_FALSE(vt::vulkan::ProbeVulkanLibraryForTesting(Ops(&fake)));
    CHECK(fake.opened == L"vulkan-1.dll");
    CHECK(fake.closes == (fake.open_succeeds ? 1 : 0));
  }
  FakeLoader fake;
  g_fake = &fake;
  CHECK(vt::vulkan::ProbeVulkanLibraryForTesting(Ops(&fake)));
  CHECK(fake.lookups == std::vector<std::string>{"vkGetInstanceProcAddr"});
  CHECK(fake.closes == 1);
}
