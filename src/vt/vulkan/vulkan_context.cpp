// Vulkan backend — instance/device/queue/memory/pipeline scaffolding.
// See vulkan_context.h for the port map (llama.cpp `ggml/src/ggml-vulkan/` @
// 237ad9b96). BACKEND-VULKAN, W0 skeleton.
//
// § RELAXED PRECISION — the knobs that had to be pinned.
// The Metal skeleton found that Metal's DEFAULT fast-math would have silently
// voided its CPU comparison and pinned MTLMathModeSafe. Vulkan/SPIR-V has the
// same class of trap in three places, handled as follows:
//   1. `inversesqrt()` — GLSL only requires ~2 ULP and drivers lower it to the
//      hardware reciprocal-sqrt approximation. llama.cpp uses it in both norm
//      shaders (rms_norm.comp:86, norm.comp:39). We use `1.0 / sqrt(x)`, which
//      is literally what the CPU reference computes. Pinned in the SHADER, so it
//      cannot be undone by a driver flag.
//   2. `RelaxedPrecision` decorations — emitted only for mediump/lowp
//      qualifiers, which none of our shaders use, and glslang applies no
//      fast-math relaxation of its own (there is no -ffast-math equivalent). The
//      committed SPIR-V is therefore IEEE-as-written by construction and is
//      regenerated only through scripts/gen-vulkan-spirv.py.
//   3. FLOAT CONTROLS — denormal flush-to-zero and signed-zero/Inf/NaN
//      preservation for fp32 are IMPLEMENTATION-DEFINED in Vulkan
//      (VkPhysicalDeviceFloatControlsProperties). These are the one knob we
//      cannot pin from the shader without SPV_KHR_float_controls execution
//      modes, so instead they are PROBED, recorded on the context, and asserted
//      by the unit gate (tests/vt/test_vulkan_backend.cpp), which reports what
//      the device actually does rather than assuming. They matter only for
//      denormal inputs and NaN/±0 payloads; the bit-exact tier of the
//      cross-device harness covers exactly those cases through the bf16 codec,
//      which is integer arithmetic and therefore unaffected either way.
#include "vulkan_context.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "vulkan_loader.h"
#include "vulkan_spirv.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vt::vulkan {
namespace {

// Dispatch accounting, enabled by VT_VULKAN_DISPATCH_STATS (VK-E deep dive).
const bool kDispatchStats = [] {
  const char* v = std::getenv("VT_VULKAN_DISPATCH_STATS");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();
// COMMAND-BUFFER BATCHING (VK-A2), VT_VULKAN_BATCH.
//
// DEFAULT ON. `=0` forces the per-dispatch submit-and-wait path, as a bisect
// lever and for the same-binary A/B that measured this (decode 2.62x, 8 of 8
// interleaved pairs on GB10).
//
// Batching is sound only if EVERY host read of device memory drains the batch
// first, and there are exactly three such paths. Backend::Copy and Memset are
// host memcpy over the persistently mapped allocation, and Synchronize is the
// caller's explicit "make results readable" point -- all three flush. The third
// is not a method on this backend at all: the PORTABLE REFERENCE TIER runs CPU
// kernels DIRECTLY over device memory for any op with no native Vulkan kernel,
// which is sound only because this backend is unified-memory.
//
// That last one is covered by `Backend::FlushPending` (backend.h:44-49), which
// op_provider.cpp calls before dispatching a reference-tier kernel and which the
// Vulkan backend implements. Without it a host kernel would read bytes a pending
// dispatch had not written -- silently, with no error, which is the failure mode
// this campaign has already paid for twice.
const bool kBatchDispatch = [] {
  const char* v = std::getenv("VT_VULKAN_BATCH");
  return v == nullptr || std::strcmp(v, "0") != 0;
}();

// Cap on dispatches per submit. Bounded so a batch cannot pin an unbounded number
// of descriptor sets, and so the fence granularity stays coarse enough to be
// worth batching but fine enough to bound latency.
constexpr uint32_t kMaxBatch = 512;

const std::chrono::steady_clock::time_point g_dispatch_t0 =
    std::chrono::steady_clock::now();

// The void* handle smuggling in vulkan_context.h is only sound while every
// Vulkan handle fits in a pointer. Dispatchable handles are pointers by
// definition; non-dispatchable ones are uint64_t on a 64-bit build.
static_assert(sizeof(VkInstance) <= sizeof(void*), "VkInstance must fit in void*");
static_assert(sizeof(VkBuffer) <= sizeof(void*), "VkBuffer must fit in void*");
static_assert(sizeof(VkDeviceMemory) <= sizeof(void*), "VkDeviceMemory must fit in void*");

template <typename H>
void* Pack(H h) {
  void* p = nullptr;
  std::memcpy(&p, &h, sizeof(H));
  return p;
}
template <typename H>
H Unpack(void* p) {
  H h{};
  std::memcpy(&h, &p, sizeof(H));
  return h;
}

void Check(VkResult r, const char* what) {
  VT_CHECK(r == VK_SUCCESS,
           std::string("vulkan: ") + what + " failed with VkResult " + std::to_string(r));
}

// llama.cpp `ggml_vk_find_memory_properties` (ggml-vulkan.cpp:2957): walk the
// memory types the buffer's requirements allow and take the first that carries
// every required property flag. We keep its ORDERED FALLBACK shape
// (`ggml_vk_create_buffer`, :3065-3090) — first choice DEVICE_LOCAL as well as
// host-visible/coherent (the unified case: GB10 exposes exactly such a type on
// its single 89.72 GiB heap, and so does llvmpipe), falling back to plain
// host-visible/coherent on a discrete GPU.
constexpr VkMemoryPropertyFlags kHostFlags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

int FindMemoryType(const VkPhysicalDeviceMemoryProperties& props, uint32_t type_bits,
                   VkMemoryPropertyFlags required) {
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) == 0) continue;
    if ((props.memoryTypes[i].propertyFlags & required) == required) return static_cast<int>(i);
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Device selection. Runs during Available(), i.e. possibly on a machine with no
// Vulkan at all, so it must never throw and never leave an instance behind.
// ---------------------------------------------------------------------------
struct Probe {
  bool ok = false;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  uint32_t api_version = 0;
  char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};
};

VkInstance CreateInstance() {
  const VulkanApi& vk = Api();
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vllm.cpp";
  // Ask for 1.1: this backend needs VK_KHR_16bit_storage, which is CORE in 1.1
  // (see the shaders' § STORAGE MODEL). A 1.0-only loader has no
  // vkEnumerateInstanceVersion and would reject this, which is the answer we
  // want — the backend cannot run there.
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  if (vk.vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return VK_NULL_HANDLE;
  return instance;
}

bool HasStorageBuffer16BitAccess(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  VkPhysicalDevice16BitStorageFeatures f16{};
  f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  VkPhysicalDeviceFeatures2 f2{};
  f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  f2.pNext = &f16;
  vk.vkGetPhysicalDeviceFeatures2(pd, &f2);
  return f16.storageBuffer16BitAccess == VK_TRUE;
}

bool HasDeviceExtension(VkPhysicalDevice pd, const char* want) {
  const VulkanApi& vk = Api();
  uint32_t count = 0;
  if (vk.vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> exts(count);
  if (vk.vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, exts.data()) != VK_SUCCESS) {
    return false;
  }
  for (const auto& e : exts) {
    if (std::strcmp(e.extensionName, want) == 0) return true;
  }
  return false;
}

// The ONE configuration the committed coopmat SPIR-V is written to:
// 16x16x16, A/B bf16, C/Result f32, SUBGROUP scope. Vulkan requires an EXACT
// match against a reported configuration -- there is no "nearest" -- so this
// asks for exactly that tuple and nothing else.
bool HasBf16F32CoopMatConfig(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR == nullptr) return false;
  uint32_t count = 0;
  if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkCooperativeMatrixPropertiesKHR> cfg(count);
  for (auto& c : cfg) c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  if (vk.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &count, cfg.data()) != VK_SUCCESS) {
    return false;
  }
  for (const auto& c : cfg) {
    if (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
        c.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
        c.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
        c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        c.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        c.scope == VK_SCOPE_SUBGROUP_KHR) {
      return true;
    }
  }
  return false;
}

int FindComputeQueueFamily(VkPhysicalDevice pd) {
  const VulkanApi& vk = Api();
  uint32_t count = 0;
  vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && families[i].queueCount > 0) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Picks the first physical device that satisfies every requirement. Preference
// order mirrors llama.cpp's device selection intent (`ggml_vk_instance_init`):
// a real GPU before a software rasterizer, so a box that has BOTH — like the dev
// box, which enumerates llvmpipe — still runs on the GPU when there is one.
// VK_VT_DEVICE lets a caller force a specific index, which is how the llvmpipe
// CI path is exercised on a machine that also has a GPU.
Probe ProbeDevice(VkInstance instance) {
  const VulkanApi& vk = Api();
  Probe best;
  uint32_t count = 0;
  if (vk.vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
    return best;
  }
  std::vector<VkPhysicalDevice> devices(count);
  if (vk.vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return best;

  const char* forced = std::getenv("VT_VULKAN_DEVICE");
  int forced_index = forced != nullptr ? std::atoi(forced) : -1;

  int best_rank = -1;
  for (uint32_t i = 0; i < count; ++i) {
    if (forced_index >= 0 && static_cast<int>(i) != forced_index) continue;
    VkPhysicalDeviceProperties props{};
    vk.vkGetPhysicalDeviceProperties(devices[i], &props);
    if (VK_API_VERSION_MAJOR(props.apiVersion) < 1) continue;
    if (VK_API_VERSION_MAJOR(props.apiVersion) == 1 && VK_API_VERSION_MINOR(props.apiVersion) < 1) {
      continue;  // needs 1.1 for VK_KHR_16bit_storage in core
    }
    if (!HasStorageBuffer16BitAccess(devices[i])) continue;
    const int qf = FindComputeQueueFamily(devices[i]);
    if (qf < 0) continue;

    VkPhysicalDeviceMemoryProperties mem{};
    vk.vkGetPhysicalDeviceMemoryProperties(devices[i], &mem);
    if (FindMemoryType(mem, ~0u, kHostFlags) < 0) continue;

    // Rank: integrated/discrete GPU (2) > virtual GPU (1) > CPU/other (0).
    int rank = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      rank = 2;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
      rank = 1;
    }
    if (rank <= best_rank) continue;
    best_rank = rank;
    best.ok = true;
    best.physical_device = devices[i];
    best.queue_family = static_cast<uint32_t>(qf);
    best.api_version = props.apiVersion;
    std::memcpy(best.name, props.deviceName, sizeof(best.name));
  }
  return best;
}

}  // namespace

// ---------------------------------------------------------------------------

// How many descriptor sets each pipeline owns (VK-A2).
//
// A DESCRIPTOR SET IS READ AT EXECUTION TIME, NOT RECORD TIME. With one set per
// pipeline -- the pre-VK-A2 shape -- recording two dispatches of the same
// pipeline into one command buffer would have the second vkUpdateDescriptorSets
// overwrite the first's operands before the GPU ran either. That was sound only
// because every dispatch waited on a fence before the next one touched the set.
// Batching therefore needs a set PER RECORDED DISPATCH, which is the substantive
// part of this row; deferring the submit alone would silently compute garbage.
//
// The ring is bounded, so a pipeline used more than kDescriptorRing times in one
// batch forces a flush. That is a throughput ceiling, never a correctness
// question: the flush happens before the set is reused.
// MEASURED: 16 was the batch-length limiter, not kMaxBatch. vt_rms_norm runs 112
// times per forward pass (4 per layer x 28 layers), so it exhausted a 16-deep ring
// seven times per pass and forced a flush each time -- observed batches capped at
// 40-46 dispatches against a kMaxBatch of 128. Every forced flush is a
// vkQueueSubmit plus a blocking vkWaitForFences, and a host profile with the idle
// CPU-threadpool spin suppressed puts 62% of on-CPU time in the kernel and the
// NVIDIA driver against only 14% in our own code. So the submits ARE the host
// cost, and the ring depth is what sets how many there are.
constexpr uint32_t kDescriptorRing = 128;

// EFFECTIVE ring depth, clamped to the allocated one. Exists so the ring can be
// A/B'd in ONE binary: it was 16, which capped batches at 40-46 dispatches, and a
// cross-BUILD comparison of two depths is the shape that produced a false 1.2x
// reading for the subgroup tactic earlier in this campaign.
const uint32_t kRingDepth = [] {
  const char* v = std::getenv("VT_VULKAN_RING");
  if (v == nullptr) return kDescriptorRing;
  const int n = std::atoi(v);
  if (n < 1) return 1u;
  return n > static_cast<int>(kDescriptorRing) ? kDescriptorRing : static_cast<uint32_t>(n);
}();

struct VulkanContext::Pipeline {
  VkShaderModule module = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSet sets[kDescriptorRing] = {};
  uint32_t used_this_batch = 0;  // reset by FlushBatchLocked
  uint32_t buffer_count = 0;
  uint32_t push_size = 0;
};

bool VulkanContext::Available() {
  // Cached: probing creates and destroys an instance, and both registrars plus
  // the platform TU ask.
  static const bool available = [] {
    if (!LoadVulkanLibrary()) return false;
    const VulkanApi& vk = Api();
    // A 1.0-only loader cannot give us the 1.1 core features this backend needs.
    if (vk.vkEnumerateInstanceVersion == nullptr) return false;
    uint32_t loader_version = 0;
    if (vk.vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) return false;
    if (VK_API_VERSION_MAJOR(loader_version) == 1 &&
        VK_API_VERSION_MINOR(loader_version) < 1) {
      return false;
    }
    VkInstance instance = CreateInstance();
    if (instance == VK_NULL_HANDLE) return false;
    LoadInstanceFunctions(instance);
    const Probe probe = ProbeDevice(instance);
    vk.vkDestroyInstance(instance, nullptr);
    return probe.ok;
  }();
  return available;
}

bool VulkanDeviceAvailable() { return VulkanContext::Available(); }

uint32_t FlatGroupCount(int64_t n) {
  if (n <= 0) return 0;
  return static_cast<uint32_t>((n + kWorkgroupSize - 1) / kWorkgroupSize);
}

VulkanContext::VulkanContext() {
  VT_CHECK(LoadVulkanLibrary(), "vulkan: no Vulkan loader (libvulkan.so.1) on this machine");
  const VulkanApi& vk = Api();

  VkInstance instance = CreateInstance();
  VT_CHECK(instance != VK_NULL_HANDLE, "vulkan: vkCreateInstance failed");
  LoadInstanceFunctions(instance);
  instance_ = Pack(instance);

  const Probe probe = ProbeDevice(instance);
  VT_CHECK(probe.ok, "vulkan: no physical device meets the backend's requirements "
                     "(Vulkan >= 1.1, a compute queue, storageBuffer16BitAccess, and a "
                     "HOST_VISIBLE|HOST_COHERENT memory type)");
  physical_device_ = Pack(probe.physical_device);
  queue_family_ = probe.queue_family;
  api_major_ = static_cast<int>(VK_API_VERSION_MAJOR(probe.api_version));
  api_minor_ = static_cast<int>(VK_API_VERSION_MINOR(probe.api_version));
  device_name_ = probe.name;

  // Float controls — probed and recorded, not pinned; see § RELAXED PRECISION.
  VkPhysicalDeviceFloatControlsProperties fc{};
  fc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &fc;
  vk.vkGetPhysicalDeviceProperties2(probe.physical_device, &props2);
  denorm_preserve_f32_ = fc.shaderDenormPreserveFloat32 == VK_TRUE;
  sz_inf_nan_preserve_f32_ = fc.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;
  max_workgroup_count_x_ = props2.properties.limits.maxComputeWorkGroupCount[0];
  // GPU TIMESTAMP SUPPORT, probed rather than assumed. `timestampPeriod` is
  // nanoseconds per tick; a device reporting 0 cannot timestamp at all, and
  // `timestampComputeAndGraphics == VK_FALSE` means the compute queue family may
  // not support it even when the device nominally does. Either way profiling
  // silently stays off rather than producing nonsense numbers.
  if (props2.properties.limits.timestampComputeAndGraphics == VK_TRUE) {
    timestamp_period_ns_ = static_cast<double>(props2.properties.limits.timestampPeriod);
  }
  VT_CHECK(props2.properties.limits.maxComputeWorkGroupInvocations >= kWorkgroupSize,
           "vulkan: device reports maxComputeWorkGroupInvocations below the Vulkan "
           "guaranteed minimum of 128, which the committed SPIR-V is compiled for");

  // storageBuffer16BitAccess must be ENABLED, not merely supported.
  VkPhysicalDevice16BitStorageFeatures f16{};
  f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  f16.storageBuffer16BitAccess = VK_TRUE;

  // --- COOPERATIVE MATRIX (VK-C), enabled only where the device actually has it.
  //
  // Enablement is CONDITIONAL for a load-bearing reason: naming an unsupported
  // extension in VkDeviceCreateInfo makes vkCreateDevice FAIL OUTRIGHT, so an
  // unconditional request would take the whole backend down on llvmpipe -- which
  // is the only Vulkan device CI can reach, and which exposes the extension not
  // at all (measured 2026-08-07). The scalar tactic stays correct there.
  //
  // The predicate is deliberately narrow: extension present, AND a bf16 x bf16
  // -> f32 16x16x16 SUBGROUP configuration reported, AND the subgroup size known.
  // A device with cooperative matrix but only f16 or int8 configurations gets the
  // scalar path, because the committed coopmat SPIR-V is written to the bf16
  // shape and Vulkan gives no way to "almost" match a configuration.
  std::vector<const char*> device_exts;
  const bool has_coopmat_ext = HasDeviceExtension(probe.physical_device,
                                                  "VK_KHR_cooperative_matrix");
  const bool has_bf16_ext = HasDeviceExtension(probe.physical_device,
                                               "VK_KHR_shader_bfloat16");
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_feat{};
  coop_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_feat{};
  bf16_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;

  // Subgroup size, which the coopmat shader's workgroup shape depends on.
  VkPhysicalDeviceSubgroupProperties sub{};
  sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 sprops{};
  sprops.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  sprops.pNext = &sub;
  Api().vkGetPhysicalDeviceProperties2(probe.physical_device, &sprops);
  subgroup_size_ = sub.subgroupSize;

  if (has_coopmat_ext && has_bf16_ext &&
      HasBf16F32CoopMatConfig(probe.physical_device) && subgroup_size_ > 0) {
    coopmat_bf16_f32_ = true;
    coop_feat.cooperativeMatrix = VK_TRUE;
    bf16_feat.shaderBFloat16Type = VK_TRUE;
    bf16_feat.shaderBFloat16CooperativeMatrix = VK_TRUE;
    device_exts.push_back("VK_KHR_cooperative_matrix");
    device_exts.push_back("VK_KHR_shader_bfloat16");
    // Chain: f16 -> coopmat -> bf16.
    coop_feat.pNext = &bf16_feat;
    f16.pNext = &coop_feat;
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = &f16;
  dci.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
  dci.ppEnabledExtensionNames = device_exts.empty() ? nullptr : device_exts.data();
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice device = VK_NULL_HANDLE;
  Check(vk.vkCreateDevice(probe.physical_device, &dci, nullptr, &device), "vkCreateDevice");
  LoadDeviceFunctions(device);
  device_ = Pack(device);

  VkQueue queue = VK_NULL_HANDLE;
  Api().vkGetDeviceQueue(device, queue_family_, 0, &queue);
  queue_ = Pack(queue);

  // Memory type, chosen once for every allocation this backend makes. Ordered
  // fallback, llama.cpp `ggml_vk_create_buffer`:3065-3090 shape.
  VkPhysicalDeviceMemoryProperties mem{};
  Api().vkGetPhysicalDeviceMemoryProperties(probe.physical_device, &mem);
  int type = FindMemoryType(mem, ~0u, kHostFlags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  unified_memory_ = type >= 0;
  if (type < 0) type = FindMemoryType(mem, ~0u, kHostFlags);
  VT_CHECK(type >= 0, "vulkan: no HOST_VISIBLE|HOST_COHERENT memory type");
  memory_type_index_ = static_cast<uint32_t>(type);

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = queue_family_;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  Check(Api().vkCreateCommandPool(device, &cpci, nullptr, &command_pool), "vkCreateCommandPool");
  command_pool_ = Pack(command_pool);

  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  Check(Api().vkAllocateCommandBuffers(device, &cbai, &cmd), "vkAllocateCommandBuffers");
  command_buffer_ = Pack(cmd);

  // One descriptor set per kernel, each holding at most kMaxBindings storage
  // buffers. Sized for the whole committed shader table so the pool is never
  // reallocated. (llama.cpp grows a vector of pools instead; a fixed pool is
  // enough here because dispatch is synchronous and sets are re-updated.)
  constexpr uint32_t kMaxBindings = 12;  // llama.cpp's MAX_PARAMETER_COUNT
  // ONE DESCRIPTOR SET PER PIPELINE, AND PIPELINES NOW OUTNUMBER MODULES.
  // Since VK-A1 a module can be specialized into several pipelines — vt_cast
  // alone reaches one per (src, dst) dtype pair — and each allocates its own set
  // from this pool. Sizing it by module count would exhaust the pool on the Nth
  // specialization and fail in vkAllocateDescriptorSets, far from the cause. The
  // headroom factor is deliberate slack, not a computed bound; GetPipeline
  // reports pool exhaustion with the kernel name if it is ever hit.
  constexpr uint32_t kSpecializationHeadroom = 16;
  const uint32_t kernels =
      static_cast<uint32_t>(kSpirvModuleCount) * kSpecializationHeadroom;
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = kernels * kDescriptorRing * kMaxBindings;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // kDescriptorRing sets per pipeline now, not one -- see the ring's comment for
  // why batching cannot share a set across recorded dispatches.
  dpci.maxSets = kernels * kDescriptorRing;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  Check(Api().vkCreateDescriptorPool(device, &dpci, nullptr, &descriptor_pool),
        "vkCreateDescriptorPool");
  descriptor_pool_ = Pack(descriptor_pool);

  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  Check(Api().vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence");
  fence_ = Pack(fence);

  scratch_mapped_ = AllocBuffer(kScratchBytes, &scratch_buffer_, &scratch_memory_);

  pipelines_ = new std::map<std::string, Pipeline>();
  dispatch_hist_ = new std::map<std::string, uint64_t>();
  dispatch_ms_ = new std::map<std::string, double>();
  batch_names_ = new std::vector<std::string>();
  // TWO timestamps per dispatch (before and after), for a whole batch. Created
  // only under the stats flag: a query pool is cheap, but writing timestamps adds
  // commands to every dispatch and production must not pay for a diagnostic.
  if (kDispatchStats && timestamp_period_ns_ > 0.0) {
    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = kMaxBatch * 2;
    VkQueryPool qp = VK_NULL_HANDLE;
    if (Api().vkCreateQueryPool(Unpack<VkDevice>(device_), &qpci, nullptr, &qp) == VK_SUCCESS) {
      query_pool_ = Pack(qp);
    }
  }

  // VT_VULKAN_DISPATCH_STATS=1 dumps the per-shader histogram at exit. Registered
  // with atexit rather than printed from a destructor because this context is a
  // never-destroyed process singleton, and because a run that is KILLED by a
  // timeout -- which is exactly how the VK-E runs ended -- still unwinds atexit
  // handlers on SIGTERM only if the handler is installed. A run that never
  // reaches a clean exit still leaves the counters readable via the accessors.
  if (const char* v = std::getenv("VT_VULKAN_DISPATCH_STATS");
      v != nullptr && std::strcmp(v, "0") != 0) {
    std::atexit([] {
      if (!Available()) return;
      const VulkanContext& ctx = Get();
      std::fprintf(stderr, "[vt vulkan] TOTAL DISPATCHES: %llu\n",
                   static_cast<unsigned long long>(ctx.dispatch_count()));
      std::map<std::string, uint64_t> counts;
      for (const auto& kv : ctx.DispatchHistogram()) counts[kv.first] = kv.second;
      const auto times = ctx.DispatchTimeMs();
      double total_ms = 0.0;
      for (const auto& kv : times) total_ms += kv.second;
      // Sorted by TIME, not count. The ordering is the point: reading a
      // count-sorted list is how a cheap shader that runs often gets mistaken
      // for the bottleneck.
      // PER-SHADER TIME IS UNAVAILABLE UNDER BATCHING, and printing 0.0 for
      // every row would read as "these kernels are free" rather than "this was
      // not measured". The wait that attributed time to a shader was the
      // per-dispatch fence; batching submits many dispatches under ONE fence, so
      // there is nothing to attribute. Counts stay exact either way.
      //
      // Getting per-kernel time back under batching needs GPU timestamp queries
      // (vkCmdWriteTimestamp around each dispatch), which is the proper Vulkan
      // answer and is not built. Until then, profile with VT_VULKAN_BATCH=0:
      // relative KERNEL cost is still meaningful there, it just also carries the
      // per-dispatch floor that batching removes.
      if (!ctx.batching_enabled() || total_ms > 0.0) {
        std::fprintf(stderr, "[vt vulkan] %-24s %8s %10s %7s %10s\n", "shader",
                     "count", "total ms", "%", "ms/call");
      } else {
        // Reached only when batching is on AND the device could not timestamp
        // (timestampComputeAndGraphics false, or the pool failed to create).
        std::fprintf(stderr,
                     "[vt vulkan] per-shader TIME unavailable: batching is on and this "
                     "device reports no compute timestamps.\n"
                     "[vt vulkan] Re-run with VT_VULKAN_BATCH=0 for per-kernel ms. "
                     "Counts below are exact.\n");
        std::fprintf(stderr, "[vt vulkan] %-24s %8s\n", "shader", "count");
        for (const auto& kv : counts) {
          std::fprintf(stderr, "[vt vulkan] %-24s %8llu\n", kv.first.c_str(),
                       static_cast<unsigned long long>(kv.second));
        }
        std::fprintf(stderr, "[vt vulkan] %-24s %8llu\n", "TOTAL",
                     static_cast<unsigned long long>(ctx.dispatch_count()));
        return;
      }
      for (const auto& kv : times) {
        const uint64_t n = counts[kv.first];
        std::fprintf(stderr, "[vt vulkan] %-24s %8llu %10.1f %6.1f%% %10.4f\n",
                     kv.first.c_str(), static_cast<unsigned long long>(n), kv.second,
                     total_ms > 0 ? 100.0 * kv.second / total_ms : 0.0,
                     n > 0 ? kv.second / double(n) : 0.0);
      }
      std::fprintf(stderr, "[vt vulkan] %-24s %8llu %10.1f\n", "TOTAL",
                   static_cast<unsigned long long>(ctx.dispatch_count()), total_ms);
    });
  }
  mutex_ = new std::mutex();
}

VulkanContext& VulkanContext::Get() {
  // Function-local static: thread-safe initialization, constructed on first use
  // and deliberately never destroyed (process lifetime, matching llama.cpp's
  // `vk_instance` singleton and the Metal skeleton's MetalContext::Get).
  static VulkanContext* ctx = new VulkanContext();
  return *ctx;
}

void* VulkanContext::AllocBuffer(size_t bytes, void** out_buffer, void** out_memory) {
  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  // ROUNDED UP TO A WHOLE 32-BIT WORD, and that is a CORRECTNESS requirement of
  // this backend's storage model, not tidiness.
  //
  // Every operand is bound as a `uint32_t[]` view (and, for float operands, also
  // as a `uint16_t[]` one) over the WHOLE buffer — vt_common.glsl § STORAGE
  // MODEL. An array of `uint` over a buffer of N bytes has floor(N/4) elements,
  // so a buffer whose length is not a multiple of 4 has a TRUNCATED 32-bit view
  // and its last partial word is unreachable. Under robustBufferAccess that read
  // returns zero; without it, it is undefined. Either way it is silent.
  //
  // MEASURED (BACKEND-VULKAN-GDN): a 3-byte i8 `has_initial_state[3]` — the GDN
  // per-request "does this row have a prior state" flag, which the gather shader
  // reads byte-wise through the 32-bit view precisely so it need not require
  // VK_KHR_8bit_storage — produced a 0-element view, every flag read back as
  // false, and the gather ZEROED rows it should have copied. The gate caught it
  // as a memcmp mismatch against the CPU oracle.
  //
  // Nothing before that read a non-multiple-of-4 buffer through the 32-bit view
  // (f32/i32/i64 lengths are multiples of 4 by construction, and 16-bit dtypes go
  // through the 16-bit view), which is why the skeleton lived with it. The fix
  // belongs HERE rather than in one shader: any future byte- or word-granular
  // read of a small operand would hit the same edge.
  //
  // A zero-length VkBuffer is also invalid, and the rounding covers that too: a
  // 0-byte request still yields a distinct freeable pointer, which is the CPU
  // backend's contract and what vt::StepArena relies on. Only the BUFFER LENGTH
  // grows; the mapped pointer and every byte the caller wrote are untouched, so
  // Copy/Memset stay bit-exact.
  const VkDeviceSize len = bytes == 0 ? 4 : static_cast<VkDeviceSize>((bytes + 3) & ~size_t{3});

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = len;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  Check(vk.vkCreateBuffer(device, &bci, nullptr, &buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vk.vkGetBufferMemoryRequirements(device, buffer, &req);
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = memory_type_index_;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  Check(vk.vkAllocateMemory(device, &mai, nullptr, &memory), "vkAllocateMemory");
  Check(vk.vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

  void* mapped = nullptr;
  Check(vk.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
  // vt::StepArena depends on >= 64-byte alignment (include/vt/backend.h:26).
  // A whole VkDeviceMemory mapping is at least `minMemoryMapAlignment`
  // (>= 64 by spec) aligned, so this holds by construction; assert it anyway
  // because everything downstream silently depends on it.
  VT_CHECK(reinterpret_cast<uintptr_t>(mapped) % 64 == 0,
           "vulkan: mapped allocation is not 64-byte aligned");

  *out_buffer = Pack(buffer);
  *out_memory = Pack(memory);
  return mapped;
}

void VulkanContext::FreeBuffer(void* buffer, void* memory) {
  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  vk.vkUnmapMemory(device, Unpack<VkDeviceMemory>(memory));
  vk.vkDestroyBuffer(device, Unpack<VkBuffer>(buffer), nullptr);
  vk.vkFreeMemory(device, Unpack<VkDeviceMemory>(memory), nullptr);
}

namespace {

// The pipeline cache key: the module name plus its specialization values, which
// together identify one VkPipeline. Decimal so the key reads back in the
// VT_CHECK messages below.
std::string PipelineKey(const std::string& name, const uint32_t* spec_values,
                        uint32_t spec_count) {
  std::string key = name;
  for (uint32_t i = 0; i < spec_count; ++i) {
    key += (i == 0 ? '|' : ',');
    key += std::to_string(spec_values[i]);
  }
  return key;
}

}  // namespace

VulkanContext::Pipeline& VulkanContext::GetPipeline(const std::string& name,
                                                    uint32_t buffer_count, uint32_t push_size,
                                                    const uint32_t* spec_values,
                                                    uint32_t spec_count) {
  const std::string key = PipelineKey(name, spec_values, spec_count);
  auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  auto it = cache.find(key);
  if (it != cache.end()) {
    // A kernel's binding count and push-constant size are properties of its
    // SPIR-V; a mismatch means the host and the committed shader have drifted,
    // which would corrupt memory rather than fail cleanly.
    VT_CHECK(it->second.buffer_count == buffer_count && it->second.push_size == push_size,
             "vulkan: pipeline " + key + " re-requested with a different binding layout");
    return it->second;
  }

  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);

  const SpirvModule* module = nullptr;
  for (size_t i = 0; i < kSpirvModuleCount; ++i) {
    if (name == kSpirvModules[i].name) { module = &kSpirvModules[i]; break; }
  }
  VT_CHECK(module != nullptr,
           "vulkan: no committed SPIR-V for kernel '" + name +
               "' — regenerate with scripts/gen-vulkan-spirv.py");

  // Specialization values are passed POSITIONALLY against the module's declared
  // SpecIds. Vulkan SILENTLY IGNORES a map entry whose constantID the module does
  // not declare, so an unchecked host/shader drift is wrong numbers rather than an
  // error — the same class as the binding-layout check above, and just as fatal.
  VT_CHECK(spec_count == module->spec_id_count,
           "vulkan: kernel '" + name + "' declares " +
               std::to_string(module->spec_id_count) +
               " specialization constant(s) but " + std::to_string(spec_count) +
               " value(s) were supplied — host and committed SPIR-V have drifted;"
               " regenerate with scripts/gen-vulkan-spirv.py");

  Pipeline p;
  p.buffer_count = buffer_count;
  p.push_size = push_size;

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = module->word_count * sizeof(uint32_t);
  smci.pCode = module->words;
  Check(vk.vkCreateShaderModule(device, &smci, nullptr, &p.module), "vkCreateShaderModule");

  // One descriptor-set layout per pipeline, with exactly its binding count.
  // llama.cpp instead shares ONE 12-binding layout across every pipeline
  // (ggml-vulkan.cpp:6424-6437); per-pipeline is simpler here because the
  // pipeline LAYOUT already has to be per-pipeline (push-constant sizes differ)
  // and it removes any question about descriptors a shader never declares.
  std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_count);
  for (uint32_t i = 0; i < buffer_count; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = buffer_count;
  dslci.pBindings = bindings.data();
  Check(vk.vkCreateDescriptorSetLayout(device, &dslci, nullptr, &p.set_layout),
        "vkCreateDescriptorSetLayout");

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = push_size;
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.set_layout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  Check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &p.layout), "vkCreatePipelineLayout");

  // One uint32 per constant, tightly packed; entry i carries the module's i-th
  // declared SpecId (the generator emits them sorted ascending). These two must
  // OUTLIVE vkCreateComputePipelines below — a temporary whose address is
  // captured and read later is the use-after-free class this project has already
  // hit twice with CUDA-graph capture, so they are named locals in this scope,
  // never a nested block.
  std::vector<VkSpecializationMapEntry> spec_entries(spec_count);
  for (uint32_t i = 0; i < spec_count; ++i) {
    spec_entries[i].constantID = module->spec_ids[i];
    spec_entries[i].offset = i * static_cast<uint32_t>(sizeof(uint32_t));
    spec_entries[i].size = sizeof(uint32_t);
  }
  VkSpecializationInfo spec_info{};
  spec_info.mapEntryCount = spec_count;
  spec_info.pMapEntries = spec_entries.data();
  spec_info.dataSize = spec_count * sizeof(uint32_t);
  spec_info.pData = spec_values;

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p.module;
  cpci.stage.pName = "main";
  cpci.stage.pSpecializationInfo = spec_count != 0 ? &spec_info : nullptr;
  cpci.layout = p.layout;
  Check(vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline),
        "vkCreateComputePipelines");

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = Unpack<VkDescriptorPool>(descriptor_pool_);
  dsai.descriptorSetCount = kDescriptorRing;
  VkDescriptorSetLayout layouts[kDescriptorRing];
  for (uint32_t i = 0; i < kDescriptorRing; ++i) layouts[i] = p.set_layout;
  dsai.pSetLayouts = layouts;
  // Named, because the plausible cause is pool exhaustion from specialization
  // (one set per PIPELINE, and a module can have many), which is otherwise a bare
  // VkResult a long way from its reason.
  Check(vk.vkAllocateDescriptorSets(device, &dsai, p.sets),
        ("vkAllocateDescriptorSets for pipeline '" + key +
         "' (descriptor pool may be exhausted by specialized pipelines)").c_str());

  return cache.emplace(key, p).first->second;
}

bool VulkanContext::PipelineExistsFor(const std::string& name) const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  const auto& cache = *static_cast<std::map<std::string, Pipeline>*>(pipelines_);
  // Keys are "<module>" or "<module>|<spec values>", so a prefix match up to the
  // separator identifies every specialization of one module.
  for (const auto& kv : cache) {
    if (kv.first == name || kv.first.compare(0, name.size() + 1, name + "|") == 0) {
      return true;
    }
  }
  return false;
}

uint64_t VulkanContext::dispatch_count() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return dispatch_total_;
}

std::vector<std::pair<std::string, double>> VulkanContext::DispatchTimeMs() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  const auto& ms = *static_cast<std::map<std::string, double>*>(dispatch_ms_);
  std::vector<std::pair<std::string, double>> out(ms.begin(), ms.end());
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return out;
}

std::vector<std::pair<std::string, uint64_t>> VulkanContext::DispatchHistogram() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  const auto& hist = *static_cast<std::map<std::string, uint64_t>*>(dispatch_hist_);
  std::vector<std::pair<std::string, uint64_t>> out(hist.begin(), hist.end());
  std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return out;
}

size_t VulkanContext::PipelineCacheSize() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return static_cast<std::map<std::string, Pipeline>*>(pipelines_)->size();
}

bool VulkanContext::batching_enabled() const { return kBatchDispatch; }

uint32_t VulkanContext::pending_batch() const {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  return batch_count_;
}

void VulkanContext::FlushBatch() {
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));
  FlushBatchLocked();
}

// Ends the open command buffer, submits it, and WAITS. The wait is what makes
// every descriptor set in the batch free to reuse and every write visible to the
// host, so it is not an optimisation to drop: without it the reset below would
// race the GPU.
void VulkanContext::FlushBatchLocked() {
  if (!batch_open_) return;
  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  auto cmd = Unpack<VkCommandBuffer>(command_buffer_);
  Check(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  auto fence = Unpack<VkFence>(fence_);
  Check(vk.vkResetFences(device, 1, &fence), "vkResetFences");
  Check(vk.vkQueueSubmit(Unpack<VkQueue>(queue_), 1, &si, fence), "vkQueueSubmit");
  if (kDispatchStats) {
    std::fprintf(stderr, "[vt vulkan] FLUSH %u dispatches in one submit\n", batch_count_);
    std::fflush(stderr);
  }
  Check(vk.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

  // Read the timestamps back, now that the batch has certainly completed. This is
  // what restores the per-shader TIME profile that batching removed -- and it
  // measures GPU execution directly instead of a host-side fence wait.
  auto* names = static_cast<std::vector<std::string>*>(batch_names_);
  if (query_pool_ != nullptr && !names->empty()) {
    const uint32_t n = static_cast<uint32_t>(names->size());
    std::vector<uint64_t> ticks(n * 2, 0);
    if (vk.vkGetQueryPoolResults(device, Unpack<VkQueryPool>(query_pool_), 0, n * 2,
                                 ticks.size() * sizeof(uint64_t), ticks.data(),
                                 sizeof(uint64_t),
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
        VK_SUCCESS) {
      auto& acc = *static_cast<std::map<std::string, double>*>(dispatch_ms_);
      for (uint32_t i = 0; i < n; ++i) {
        const uint64_t t0 = ticks[i * 2], t1 = ticks[i * 2 + 1];
        if (t1 > t0) acc[(*names)[i]] += double(t1 - t0) * timestamp_period_ns_ / 1.0e6;
      }
    }
    names->clear();
  }

  // Every pipeline's ring is free again only AFTER the wait above.
  for (auto& kv : *static_cast<std::map<std::string, Pipeline>*>(pipelines_)) {
    kv.second.used_this_batch = 0;
  }
  batch_open_ = false;
  batch_count_ = 0;
}

void VulkanContext::Dispatch(const std::string& name, const void* const* buffers,
                             uint32_t buffer_count, const void* push_constants,
                             uint32_t push_size, uint32_t group_count_x,
                             const uint32_t* spec_values, uint32_t spec_count) {
  if (group_count_x == 0) return;  // nothing to do; an empty dispatch is illegal
  VT_CHECK(group_count_x <= max_workgroup_count_x_,
           "vulkan: dispatch needs " + std::to_string(group_count_x) +
               " workgroups, above the device limit of " +
               std::to_string(max_workgroup_count_x_));

  const VulkanApi& vk = Api();
  auto device = Unpack<VkDevice>(device_);
  // The single command buffer and each pipeline's single descriptor set are
  // re-recorded per dispatch, so the whole record-submit-wait must be
  // serialized. Correct, not fast — the same trade the Metal skeleton makes with
  // one command buffer per op (src/vt/metal/metal_ops.mm § DISPATCH MODEL).
  std::lock_guard<std::mutex> guard(*static_cast<std::mutex*>(mutex_));

  // Counted under the mutex the dispatch already holds -- see the header for why
  // this is measured on OUR side rather than inferred from context switches.
  ++dispatch_total_;
  ++(*static_cast<std::map<std::string, uint64_t>*>(dispatch_hist_))[name];
  // PERIODIC dump, not just atexit: `timeout` sends SIGTERM, whose default action
  // terminates WITHOUT running atexit handlers, and every VK-E run so far ended
  // exactly that way. A diagnostic that only reports on a clean exit would have
  // reported nothing on precisely the runs worth diagnosing.
  if (kDispatchStats && dispatch_total_ % 100 == 0) {
    const auto now = std::chrono::steady_clock::now();
    const double secs =
        std::chrono::duration<double>(now - g_dispatch_t0).count();
    std::fprintf(stderr, "[vt vulkan] dispatches=%llu  elapsed=%.1fs  rate=%.0f/s\n",
                 static_cast<unsigned long long>(dispatch_total_), secs,
                 secs > 0 ? dispatch_total_ / secs : 0.0);
  }

  Pipeline& p = GetPipeline(name, buffer_count, push_size, spec_values, spec_count);

  // A pipeline that has consumed its whole descriptor ring must flush before it
  // can reuse set 0, because the GPU has not necessarily read the earlier ones
  // yet. Flushing also resets every pipeline's counter.
  if (kBatchDispatch && (p.used_this_batch >= kRingDepth || batch_count_ >= kMaxBatch)) {
    FlushBatchLocked();
  }

  VkDescriptorSet set = kBatchDispatch ? p.sets[p.used_this_batch] : p.sets[0];

  std::vector<VkDescriptorBufferInfo> infos(buffer_count);
  std::vector<VkWriteDescriptorSet> writes(buffer_count);
  for (uint32_t i = 0; i < buffer_count; ++i) {
    infos[i].buffer = Unpack<VkBuffer>(const_cast<void*>(buffers[i]));
    infos[i].offset = 0;  // always WHOLE; the element offset rides push constants
    infos[i].range = VK_WHOLE_SIZE;
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vk.vkUpdateDescriptorSets(device, buffer_count, writes.data(), 0, nullptr);

  auto cmd = Unpack<VkCommandBuffer>(command_buffer_);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (!kBatchDispatch) {
    Check(vk.vkResetCommandPool(device, Unpack<VkCommandPool>(command_pool_), 0),
          "vkResetCommandPool");
    Check(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
  } else if (!batch_open_) {
    Check(vk.vkResetCommandPool(device, Unpack<VkCommandPool>(command_pool_), 0),
          "vkResetCommandPool");
    Check(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    // The pool must be reset on the DEVICE timeline, inside the command buffer,
    // before any query in it is written.
    if (query_pool_ != nullptr) {
      vk.vkCmdResetQueryPool(cmd, Unpack<VkQueryPool>(query_pool_), 0, kMaxBatch * 2);
      static_cast<std::vector<std::string>*>(batch_names_)->clear();
    }
    batch_open_ = true;
  } else {
    // BETWEEN recorded dispatches: the ops in a decode step are sequentially
    // dependent (norm feeds projection feeds attention), so every dispatch must
    // see the previous one's writes. Without this the batch would run them
    // concurrently and compute garbage. This is the cost batching pays back --
    // a barrier is far cheaper than a fence round-trip to the host.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vk.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                            nullptr);
  }

  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0,
                             nullptr);
  vk.vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                        push_constants);
  const bool timed = kBatchDispatch && query_pool_ != nullptr && batch_count_ < kMaxBatch;
  if (timed) {
    // TOP_OF_PIPE before / BOTTOM_OF_PIPE after brackets this dispatch's execution
    // on the GPU. Because a barrier separates consecutive dispatches, the interval
    // is this kernel's own time rather than an overlap with its neighbours.
    vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           Unpack<VkQueryPool>(query_pool_), batch_count_ * 2);
  }
  vk.vkCmdDispatch(cmd, group_count_x, 1, 1);
  if (timed) {
    vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                           Unpack<VkQueryPool>(query_pool_), batch_count_ * 2 + 1);
    static_cast<std::vector<std::string>*>(batch_names_)->push_back(name);
  }

  if (kBatchDispatch) {
    ++p.used_this_batch;
    ++batch_count_;
    return;  // submitted by FlushBatch, at the next host read or Synchronize
  }
  Check(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  auto fence = Unpack<VkFence>(fence_);
  Check(vk.vkResetFences(device, 1, &fence), "vkResetFences");
  Check(vk.vkQueueSubmit(Unpack<VkQueue>(queue_), 1, &si, fence), "vkQueueSubmit");
  // Blocking wait: the whole backend is synchronous in W0, so by the time an op
  // returns the host may read the mapped memory directly. Host-coherent memory
  // needs no invalidate, and vkQueueSubmit itself makes prior host writes
  // visible to the device (the host-write ordering guarantee), so there is no
  // flush on the way in either.
  // PER-DISPATCH TIMING. The periodic counter showed fewer than 2000 dispatches
  // in 150 s, which rules out "many cheap submits" and points at a few very
  // expensive ones -- so the useful diagnostic is WHICH shader is slow, not how
  // many ran. Anything over the threshold names itself.
  // Printed BEFORE the wait, and flushed. The timing print below runs only if the
  // wait RETURNS -- so if a fence never completes, the post-wait line never
  // appears and the hang is invisible. The last line printed here names the
  // dispatch that hung.
  if (kDispatchStats) {
    std::fprintf(stderr, "[vt vulkan] submit #%llu %-22s groups=%u\n",
                 static_cast<unsigned long long>(dispatch_total_), name.c_str(),
                 group_count_x);
    std::fflush(stderr);
  }
  const auto wait_t0 = std::chrono::steady_clock::now();
  Check(vk.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  if (kDispatchStats) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - wait_t0).count();
    // Attributed to the shader that just ran. Accumulated OUTSIDE the dispatch
    // mutex would race; this whole function already holds it.
    (*static_cast<std::map<std::string, double>*>(dispatch_ms_))[name] += ms;
    if (ms > 200.0) {
      std::fprintf(stderr, "[vt vulkan] SLOW dispatch %-22s %8.1f ms  groups=%u\n",
                   name.c_str(), ms, group_count_x);
    }
  }
}

}  // namespace vt::vulkan
