// Vulkan backend — shared instance/device/queue/pipeline context
// (BACKEND-VULKAN, W0 skeleton). vllm.cpp original (vt runtime, inventory
// deviation §9.1): vLLM has no Vulkan platform anywhere, so the DESIGN is ported
// from llama.cpp's Vulkan backend (`ggml/src/ggml-vulkan/ggml-vulkan.cpp` @ pin
// 237ad9b96). Specifically:
//
//   * one process-wide VkInstance + VkPhysicalDevice + VkDevice + compute
//     VkQueue, created lazily and kept for the process — llama.cpp
//     `ggml_vk_instance_init` / `ggml_vk_device_init` and its `vk_instance`
//     singleton;
//   * host-visible storage buffers whose memory type is chosen by walking
//     VkPhysicalDeviceMemoryProperties for the required property flags with an
//     ordered fallback list — llama.cpp `ggml_vk_find_memory_properties`
//     (ggml-vulkan.cpp:2957) and `ggml_vk_create_buffer` (:2971-3100);
//   * a NAME -> compute-pipeline cache so each kernel is specialized once —
//     llama.cpp `ggml_vk_create_pipeline_func` (:2460-2560) and its per-device
//     pipeline map;
//   * push constants for the small per-dispatch parameter block — llama.cpp
//     `ggml_vk_dispatch_pipeline` (:7507-7530).
//
// This header is deliberately PLAIN C++ — it does NOT include vulkan_core.h — so
// the op TU, the platform TU and the tests can include it without pulling the
// Vulkan API into their translation units. Handles cross the boundary as void*,
// exactly as the Metal skeleton does for its ObjC types
// (src/vt/metal/metal_context.h). vulkan_context.cpp static_asserts that the
// real handle types fit.
#ifndef VT_VULKAN_VULKAN_CONTEXT_H_
#define VT_VULKAN_VULKAN_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vt::vulkan {

// Process-wide Vulkan context. Created on first use, never destroyed (the
// process outlives it; matching llama.cpp's `vk_instance` singleton lifetime and
// the Metal skeleton's MetalContext).
class VulkanContext {
 public:
  // Returns the singleton, creating instance/device/queue and the descriptor and
  // command pools on first call. Throws (VT_CHECK) if anything fails — by the
  // time this is called, Available() has already said a usable device exists, so
  // a failure here is a broken driver, not an absent one.
  static VulkanContext& Get();

  // True iff a Vulkan loader is present AND it enumerates a physical device that
  // satisfies this backend's requirements (Vulkan >= 1.1, a compute queue
  // family, VK_KHR_16bit_storage's storageBuffer16BitAccess, and a
  // HOST_VISIBLE|HOST_COHERENT memory type usable for storage buffers). Safe on
  // a machine with no GPU and no loader: it does NOT throw. This is the
  // predicate both registrars use, so a Vulkan-enabled BUILD on a machine
  // without Vulkan simply does not register kVULKAN rather than aborting during
  // static initialization.
  static bool Available();

  // Dispatch one compute kernel, SYNCHRONOUSLY (record, submit, wait). `name` is
  // a key in the committed SPIR-V table (src/vt/vulkan/vulkan_spirv.h).
  // `buffers` are the VkBuffer handles for descriptor bindings 0..n-1, in order;
  // this backend binds every buffer WHOLE (offset 0, VK_WHOLE_SIZE) and carries
  // the element offset in the push constants, so any interior tensor pointer
  // works regardless of minStorageBufferOffsetAlignment (see
  // src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL).
  // Serialized by an internal mutex: the command buffer and each pipeline's
  // descriptor set are single instances that are re-recorded per dispatch.
  //
  // `spec_values` are SPECIALIZATION CONSTANT values, supplied in ASCENDING
  // constantID order and matching the module's declared `spec_ids` (recorded in
  // the committed SPIR-V table) exactly. They are part of the pipeline cache KEY:
  // each distinct combination becomes its own VkPipeline, specialized once and
  // reused, with the driver folding the constant and eliminating the branches it
  // kills. This is the variant mechanism that replaces llama.cpp's
  // one-module-per-#define explosion (its vulkan-shaders-gen has 242
  // `string_to_spv` call sites at pin 237ad9b96, most inside dtype/quant/coopmat
  // loops) — here one module covers the axis and the count of committed artifacts
  // tracks shader FILES instead of their cross product.
  void Dispatch(const std::string& name, const void* const* buffers, uint32_t buffer_count,
                const void* push_constants, uint32_t push_size, uint32_t group_count_x,
                const uint32_t* spec_values = nullptr, uint32_t spec_count = 0);

  // Was a pipeline for this SPIR-V module ever created? The cache key is the
  // module name plus its specialization values, so this asks "did any variant of
  // `name` get built", which is the only honest way for a test to prove that a
  // TACTIC actually ran rather than merely that the results were right. A gate
  // that checks numbers alone passes identically when the fallback served the
  // call -- the same trap the op-provider decline counters exist for.
  bool PipelineExistsFor(const std::string& name) const;

  // DISPATCH ACCOUNTING (VK-E deep dive). Total submits, and a per-shader
  // histogram. This exists because a wall-clock number could not distinguish two
  // very different stories: a reasonable dispatch count each paying a large
  // fence-wait, versus dispatching far more times than the model should need.
  // Context-switch counts could not separate them either -- the driver's fence
  // wait is a poll() loop that may wake more than once per fence -- so the count
  // has to come from OUR side of the boundary.
  //
  // Always-on and lock-free-ish (guarded by the same mutex the dispatch already
  // takes), because the cost is one increment against a submit that already costs
  // milliseconds. `VT_VULKAN_DISPATCH_STATS=1` prints the histogram at exit.
  uint64_t dispatch_count() const;
  // name -> count, sorted by count descending. For the diagnostic dump.
  std::vector<std::pair<std::string, uint64_t>> DispatchHistogram() const;

  // name -> total fence-wait milliseconds, sorted descending. COUNTS NAME THE
  // SHAPE OF A RUN; ONLY TIME NAMES THE LEVER. A measured 0.046 ms per-dispatch
  // floor against a 0.357 ms observed average proved that 87% of this backend's
  // dispatch cost is real kernel execution, not submission overhead -- so the
  // question stopped being "how many dispatches" and became "which kernel", and
  // a histogram of counts cannot answer that. The two differ wildly: the most
  // FREQUENT shader is routinely not the most EXPENSIVE one.
  //
  // Submission is synchronous, so the fence wait brackets that dispatch and
  // nothing else, and these sum to the run's GPU time rather than overlapping.
  std::vector<std::pair<std::string, double>> DispatchTimeMs() const;

  // COMMAND-BUFFER BATCHING (VK-A2). Records dispatches into ONE command buffer
  // and submits once, instead of submit+fence-wait per op.
  //
  // WHY: a measured per-dispatch floor of 0.046 ms times 2,952 dispatches is
  // 135.8 ms of a 275.8 ms decode run -- 49% of GPU time, up from 13% before the
  // argmax and GEMV kernels landed. Three shaders now cost AT OR BELOW that
  // floor, meaning they do no meaningful work relative to being launched.
  //
  // FLUSH is mandatory before ANY host read of device memory. This backend's
  // Copy/Memset are plain memcpy over the persistently mapped, host-coherent
  // allocation, so a pending batch means the host reads STALE bytes -- silently,
  // with no error. Backend::Copy, Memset and Synchronize all flush.
  void FlushBatch();
  // Whether dispatch batching is active. Exposed so a test never has to restate
  // the default: the VK-A2 gate originally re-derived it from the environment
  // variable and silently asserted the wrong branch the moment the default
  // flipped from off to on. A predicate duplicated between an implementation and
  // its gate is a predicate that will disagree with itself.
  bool batching_enabled() const;
  // Dispatches currently recorded and not yet submitted. For the gate: batching
  // is invisible in results by construction (same kernels, same order), so a test
  // has to assert the MECHANISM rather than the numbers.
  uint32_t pending_batch() const;

  // Number of distinct pipelines currently cached. Exposed for the unit gate: it
  // is how a test proves a new specialization produced a NEW pipeline rather than
  // silently reusing an existing one — which would look identical in the results.
  size_t PipelineCacheSize() const;

  // Allocate one host-visible, host-coherent storage buffer of `bytes` and keep
  // it PERSISTENTLY MAPPED. Returns the mapped host pointer — which is what
  // vt::Tensor::data carries — and hands back the VkBuffer / VkDeviceMemory
  // handles for the allocation registry. Persistent mapping is what makes
  // Backend::Copy/Memset plain memcpy/memset and keeps them BIT-EXACT.
  void* AllocBuffer(size_t bytes, void** out_buffer, void** out_memory);
  void FreeBuffer(void* buffer, void* memory);

  // A small device-visible scratch buffer for per-dispatch data that is too big
  // for push constants (the fused-chain recipe step list). Returns the VkBuffer
  // handle; `Data()` is its persistently mapped host pointer. Reused across
  // dispatches, which is safe because dispatch is synchronous.
  void* ScratchBuffer() const { return scratch_buffer_; }
  void* ScratchData() const { return scratch_mapped_; }
  static constexpr size_t kScratchBytes = 1024;

  // --- Capability data mirrored onto the Platform seam (src/vllm/platforms/
  // vulkan.cpp) and onto vt::Backend.
  // The VULKAN API VERSION is what we expose as the DeviceCapability
  // major/minor pair — {1, 4} on GB10 (API 1.4.312). CUDA answers this question
  // with sm_XY and Metal with the Apple GPU family; the Vulkan analogue is the
  // API level, so has_device_capability(1, 1) reads as "Vulkan >= 1.1", the same
  // shape of question the CUDA code already asks.
  // The shared VkQueue, as the opaque handle vt::Queue carries.
  void* queue_handle() const { return queue_; }

  int api_major() const { return api_major_; }
  int api_minor() const { return api_minor_; }
  bool unified_memory() const { return unified_memory_; }
  const std::string& device_name() const { return device_name_; }
  uint32_t max_workgroup_count_x() const { return max_workgroup_count_x_; }
  // The two float-controls properties that decide whether our f32 arithmetic is
  // IEEE as written. Probed, recorded, and asserted by the unit gate; see
  // vulkan_context.cpp § RELAXED PRECISION.
  bool denorm_preserve_f32() const { return denorm_preserve_f32_; }
  bool signed_zero_inf_nan_preserve_f32() const { return sz_inf_nan_preserve_f32_; }

  // --- COOPERATIVE MATRIX (VK-C). True iff the device exposes
  // VK_KHR_cooperative_matrix AND reports a bf16 x bf16 -> f32 configuration at
  // 16x16x16 with SUBGROUP scope AND has the subgroup size the committed SPIR-V
  // assumes -- all four, because any one of them missing makes the coopmat
  // pipeline unusable and the scalar tactic is always correct.
  //
  // MEASURED 2026-08-07: GB10 reports 11 configurations, all SUBGROUP scope,
  // among them 16x16x16 bf16/bf16/f32/f32; llvmpipe exposes the extension NOT AT
  // ALL. So this predicate is genuinely false on the only device CI can reach,
  // which is why the scalar fallback is the tested-everywhere path and the
  // coopmat path is dgx-gated.
  bool coopmat_bf16_f32() const { return coopmat_bf16_f32_; }
  uint32_t subgroup_size() const { return subgroup_size_; }

 private:
  VulkanContext();
  struct Pipeline;
  Pipeline& GetPipeline(const std::string& name, uint32_t buffer_count, uint32_t push_size,
                        const uint32_t* spec_values, uint32_t spec_count);

  void* instance_ = nullptr;         // VkInstance
  void* physical_device_ = nullptr;  // VkPhysicalDevice
  void* device_ = nullptr;           // VkDevice
  void* queue_ = nullptr;            // VkQueue
  void* command_pool_ = nullptr;     // VkCommandPool
  void* command_buffer_ = nullptr;   // VkCommandBuffer
  void* descriptor_pool_ = nullptr;  // VkDescriptorPool
  void* fence_ = nullptr;            // VkFence
  void* scratch_buffer_ = nullptr;   // VkBuffer
  void* scratch_memory_ = nullptr;   // VkDeviceMemory
  void* scratch_mapped_ = nullptr;   // host pointer
  void FlushBatchLocked();           // caller holds mutex_
  // GPU TIMESTAMP PROFILING. Batching submits many dispatches under ONE fence,
  // so the per-dispatch fence wait that used to attribute time to a shader no
  // longer exists. Timestamps written into the command buffer are the only way to
  // recover per-kernel GPU time once submissions are batched -- and they measure
  // the GPU directly rather than a host-side wait, so they are strictly better
  // evidence than what they replace.
  //
  // Allocated and written ONLY when VT_VULKAN_DISPATCH_STATS is set, so a
  // production dispatch pays nothing.
  void* query_pool_ = nullptr;       // VkQueryPool
  double timestamp_period_ns_ = 0.0; // 0 => device cannot timestamp; profiling off
  void* batch_names_ = nullptr;      // std::vector<std::string>*, one per recorded dispatch
  bool batch_open_ = false;          // a command buffer is recording
  uint32_t batch_count_ = 0;         // dispatches recorded into it
  void* dispatch_hist_ = nullptr;    // std::map<std::string, uint64_t>*
  void* dispatch_ms_ = nullptr;      // std::map<std::string, double>*
  uint64_t dispatch_total_ = 0;
  void* pipelines_ = nullptr;        // std::map<std::string, Pipeline>*
  void* mutex_ = nullptr;            // std::mutex*
  uint32_t queue_family_ = 0;
  uint32_t memory_type_index_ = 0;
  int api_major_ = 0;
  int api_minor_ = 0;
  bool unified_memory_ = false;
  bool denorm_preserve_f32_ = false;
  bool sz_inf_nan_preserve_f32_ = false;
  bool coopmat_bf16_f32_ = false;
  uint32_t subgroup_size_ = 0;
  uint32_t max_workgroup_count_x_ = 0;
  std::string device_name_;

  friend class VulkanAllocator;
};

// Plain-C++ spelling of VulkanContext::Available(), so the engine-side platform
// TU (src/vllm/platforms/vulkan.cpp) can ask "is there a Vulkan device?" without
// depending on static-initialization ORDER — asking "did the backend registrar
// already run?" from another TU's initializer is unspecified-order and would
// intermittently skip platform registration. Same reasoning, same shape, as
// vt::metal::MetalDeviceAvailable().
bool VulkanDeviceAvailable();

// Workgroup size every kernel in this backend is compiled with. Mirrors VT_TG in
// src/vt/vulkan/shaders/vt_common.glsl; the host must agree with the SPIR-V
// because the flat kernels compute their workgroup COUNT from it.
inline constexpr uint32_t kWorkgroupSize = 128;

// Number of workgroups needed to cover `n` elements at kWorkgroupSize threads
// each.
uint32_t FlatGroupCount(int64_t n);

}  // namespace vt::vulkan

#endif  // VT_VULKAN_VULKAN_CONTEXT_H_
