// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "vt/backend.h"

namespace vt::cpu {
namespace {

void* AllocAligned64(size_t bytes) {
  const size_t padded = ((bytes + 63) / 64) * 64;
#if defined(_WIN32)
  return _aligned_malloc(padded, 64);
#else
  return std::aligned_alloc(64, padded);
#endif
}

void FreeAligned64(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

class CpuBackend final : public Backend {
  // Host and device memory are the SAME allocation here, so reading the sampled
  // token id back between steps is always valid: the depth-2 async input-combine
  // path stays default-ON for CPU (it was correct and contract-tested before the
  // ROCm work; regressing it would silently drop the CPU backend to depth-1).
 public:
  bool SupportsAsyncSampledTokenReadback() const override { return true; }

  void* Alloc(size_t bytes) override {
    VT_CHECK(bytes <= SIZE_MAX - 63, "cpu alloc size overflow");
    void* p = AllocAligned64(bytes);  // 64B-aligned, padded size
    VT_CHECK(p != nullptr, "cpu alloc failed");
    return p;
  }
  void Free(void* p) override { FreeAligned64(p); }
  void Memset(Queue&, void* p, int value, size_t bytes) override { std::memset(p, value, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kCPU, 0}, nullptr}; }
  bool UnifiedMemory() const override { return true; }
};

struct Registrar {
  Registrar() {
    static CpuBackend backend;
    RegisterBackend(DeviceType::kCPU, &backend);
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
