// ROCm leg of the vt::Backend seam (BACKEND-ROCM, W0 skeleton).
//
// **UNBUILT.** No AMD GPU exists on the authoring machine, so this translation
// unit has never been compiled by anyone. That is weaker than the project's
// "build-supported" label (which means it compiles and emits real machine code)
// and it must not be quoted as such until a contributor reports a build. It is
// written to be the smallest possible thing a first compile can be wrong about:
// the one piece with a DECISION in it (the gfx capability parse) was lifted out
// into include/vt/rocm/rocm_arch.h, which IS compiled and unit-tested on a
// CPU-only box (tests/vt/test_rocm_arch.cpp, 40 assertions).
//
// SHAPE: src/vt/cuda/cuda_backend.cu, virtual for virtual, including its
// unified-memory reasoning (see UnifiedMemory below) and its multi-device
// registrar. The HIP runtime API is a deliberate near-copy of the CUDA runtime
// API, which is why upstream vLLM compiles most of csrc/ for both through a
// hipify pass (cmake/hipify.py); this file is what that pass produces for the
// backend, written out by hand so a reader can diff the two.
//
// SCOPE / what is NOT here, stated plainly:
//   * `SupportsGraphCapture()` stays FALSE. hipGraph exists and is the eventual
//     mapping, but a graph that captures a wrong stream is a silent correctness
//     bug (see .agents/ on CUDA-graph capture baking stack addresses), so it is
//     not something to write blind.
//   * The async-output primitives (AllocPinned / events) inherit the vt::Backend
//     defaults. src/vt/backend.cpp documents those as already correct for
//     unified-memory backends; on a discrete AMD card they are correct but
//     synchronous, and hipHostMalloc / hipEvent* are the obvious later port.
//   * Exactly ONE op is registered (RmsNorm, see rocm_ops.hip). Everything else
//     resolves through the portable reference tier on a unified-memory part and
//     THROWS on a discrete one, which is the honest state, not a defect.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/rocm/rocm_arch.h"

namespace vt::rocm {
namespace {

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " + hipGetErrorString(err));
  }
}

hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// One probe per device, cached: the registrar needs the residency attributes and
// the capability together, and hipGetDeviceProperties is not free. Mirrors
// vt/cuda/cuda_device_caps.h, kept local because there is no second consumer yet.
struct DeviceCaps {
  bool valid = false;
  bool integrated = false;
  bool pageable_memory_access = false;
  int major = 0;
  int minor = 0;
  std::string gcn_arch;
};

DeviceCaps ProbeDevice(int index) noexcept {
  DeviceCaps caps;
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, index) != hipSuccess) return caps;

  caps.gcn_arch = props.gcnArchName;
  // Prefer the string HIP itself reports over props.major/.minor: gcnArchName is
  // what upstream's _capability_from_gcn_arch parses (rocm.py:223), so both
  // layers answer from the same source. Fall back to the driver's own fields
  // when the string is unparseable rather than claiming (0, 0).
  if (const auto parsed = CapabilityFromGcnArch(caps.gcn_arch)) {
    caps.major = parsed->first;
    caps.minor = parsed->second;
  } else {
    caps.major = props.major;
    caps.minor = props.minor;
  }

  int integrated = 0;
  int pageable = 0;
  if (hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, index) != hipSuccess ||
      hipDeviceGetAttribute(&pageable, hipDeviceAttributePageableMemoryAccess, index) !=
          hipSuccess) {
    return caps;
  }
  caps.integrated = integrated != 0;
  caps.pageable_memory_access = pageable != 0;
  caps.valid = true;
  return caps;
}

class RocmBackend final : public Backend {
 public:
  // Non-throwing by design: constructed during static init by the registrar,
  // which probes the device beforehand. Same contract as CudaBackend.
  RocmBackend(int device, bool unified_memory, int major, int minor) noexcept
      : device_(device), unified_memory_(unified_memory), major_(major), minor_(minor) {}

  int DeviceCapabilityMajor() const override { return major_; }
  int DeviceCapabilityMinor() const override { return minor_; }

  // hipMalloc's alignment is at least 256 bytes, as cudaMalloc's is, which
  // satisfies the >= 64B contract Backend::Alloc owes StepArena.
  void* Alloc(size_t bytes) override {
    void* p = nullptr;
    Check(hipMalloc(&p, bytes), "hipMalloc");
    return p;
  }
  void Free(void* p) override { Check(hipFree(p), "hipFree"); }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    Check(hipMemsetAsync(p, value, bytes, AsStream(q)), "hipMemsetAsync");
  }
  // hipMemcpyDefault infers direction from the pointer values, so one entry
  // point covers h2d, d2h and d2d — and pageable host pointers on an APU.
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    Check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, AsStream(q)), "hipMemcpyAsync");
  }
  Queue CreateQueue() override {
    hipStream_t stream = nullptr;
    Check(hipStreamCreate(&stream), "hipStreamCreate");
    return Queue{Device{DeviceType::kROCM, device_}, stream};
  }
  void DestroyQueue(Queue& q) override {
    if (q.handle == nullptr) return;
    Check(hipStreamDestroy(AsStream(q)), "hipStreamDestroy");
    q.handle = nullptr;
  }
  void Synchronize(Queue& q) override {
    Check(hipStreamSynchronize(AsStream(q)), "hipStreamSynchronize");
  }

  // THE LOAD-BEARING BOOL. This is what decides whether the portable CPU
  // reference tier installs for kROCM (include/vt/op_provider.h:197-201): a CPU
  // kernel dereferences HOST pointers, so it is correct only where host and
  // device memory alias. On an APU (780M, Strix Halo) that makes every
  // unimplemented op fall back and a model run end to end; on a discrete card it
  // would be memory corruption, so the tier must stay off and unimplemented ops
  // must throw instead.
  //
  // The conjunction is deliberate and is CudaBackend's own (cuda_backend.cu:295-
  // 303): pageable-memory-access ALONE is not enough, because a discrete part can
  // report it while the driver services host pointers through HMM page migration.
  // That is not the zero-copy aliasing this contract needs, so an INTEGRATED GPU
  // is additionally required. Probed, never inferred from the gfx name.
  bool UnifiedMemory() const override { return unified_memory_; }

 private:
  int device_ = 0;
  bool unified_memory_ = false;
  int major_ = 0;
  int minor_ = 0;
};

// Registers every visible AMD GPU at its own Device{kROCM, i} slot, mirroring
// CudaBackend's multi-device registrar (cuda_backend.cu:304-323) — the 4x 7900
// XTX box on issue #41 is exactly why that is not a device-0 hardcode.
// `RegisterBackend(Device{kROCM,0}, ...)` writes the same slot
// `GetBackend(kROCM)` reads, so the single-GPU path is unaffected.
//
// Stays SILENT when there is no HIP runtime or no device: the platform registrar
// probes independently (static-init order across TUs is unspecified), and
// CurrentPlatform() must be able to fall through to CPU on a machine that merely
// happened to be built with -DVLLM_CPP_HIP=ON.
struct Registrar {
  Registrar() noexcept {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) return;
    static std::vector<std::unique_ptr<RocmBackend>> backends;
    const int count = n < static_cast<int>(vt::kMaxDevicesPerType)
                          ? n
                          : static_cast<int>(vt::kMaxDevicesPerType);
    for (int i = 0; i < count; ++i) {
      const DeviceCaps caps = ProbeDevice(i);
      if (!caps.valid) continue;
      backends.push_back(std::make_unique<RocmBackend>(
          i, caps.pageable_memory_access && caps.integrated, caps.major, caps.minor));
      RegisterBackend(Device{DeviceType::kROCM, i}, backends.back().get());
    }
  }
} registrar;

}  // namespace

// --- The HIP-free probes declared in include/vt/rocm/rocm_runtime.h ----------

bool DeviceAvailable() noexcept {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

std::string DeviceArchName(int index) noexcept {
  const DeviceCaps caps = ProbeDevice(index);
  return caps.gcn_arch;
}

}  // namespace vt::rocm
