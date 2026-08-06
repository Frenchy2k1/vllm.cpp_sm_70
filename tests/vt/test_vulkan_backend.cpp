// Vulkan backend skeleton unit gates (BACKEND-VULKAN, W0). Newly authored — vLLM
// has no Vulkan tests to port, and llama.cpp's `test-backend-ops` is a ggml
// harness with no vt:: analogue. Mirrors the shape of
// tests/vt/test_metal_backend.cpp (and through it tests/vt/test_backend.cpp) so
// the three are read side by side.
//
// This TU is COMPILED ONLY in a Vulkan build (tests/CMakeLists.txt gates it on
// VLLM_CPP_VULKAN) and every assertion goes through the public vt:: seam — if
// the skeleton needed Vulkan headers in a test to be checkable, the seam would
// be leaking.
//
// Cross-device NUMERIC equality vs an oracle is NOT here; it lives in
// tests/vt/test_backend_cross_device.cpp, which runs against every registered
// non-CPU backend and so covers Vulkan automatically — and which, on the GB10
// box, compares Vulkan against a CUDA build in the SAME binary, the strongest
// cross-backend oracle in the project.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/vulkan/vulkan_context.h"
#include "vt/vulkan/vulkan_spirv.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

// A Vulkan-ENABLED build can legitimately run where there is no loader or no
// conformant device (a headless CI container), in which case the registrars stay
// silent by design. Every case below is skipped in that state rather than
// failing — but the skip is REPORTED, so a silently-unregistered backend on a
// box that does have one cannot masquerade as a pass.
bool VulkanPresent() { return vt::vulkan::VulkanDeviceAvailable(); }

}  // namespace

TEST_CASE("the committed SPIR-V table is present and well-formed") {
  // Independent of any device: this is a property of the CHECKED-IN artifact, so
  // it also gates the generator (scripts/gen-vulkan-spirv.py) on a box with no
  // Vulkan at all.
  // The blobs live in vulkan_spirv.cpp, so the array is `extern` and of unknown
  // bound here and the generated count is the only way to size it. That is the
  // point of the split: at the target shader surface the words must not be
  // re-parsed by every TU that merely needs the table.
  const size_t n = vt::vulkan::kSpirvModuleCount;
  CHECK(n == 8);
  for (size_t mi = 0; mi < n; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    REQUIRE(m.word_count > 5);          // a SPIR-V header alone is 5 words
    CHECK(m.words[0] == 0x07230203u);   // SPIR-V magic
  }
  // The eight registered ops are served by exactly these seven modules (kCastBf16
  // and kCastF32 share vt_cast), so a rename in either direction breaks here
  // rather than at pipeline-creation time on a device we might not have.
  for (const char* want : {"vt_add", "vt_cast", "vt_fused_chain", "vt_layer_norm", "vt_matmul",
                           "vt_relu", "vt_rms_norm", "vt_silu_and_mul"}) {
    bool found = false;
    for (size_t mi = 0; mi < vt::vulkan::kSpirvModuleCount; ++mi) {
      if (std::strcmp(vt::vulkan::kSpirvModules[mi].name, want) == 0) found = true;
    }
    CAPTURE(want);
    CHECK(found);
  }
}

TEST_CASE("the committed SPIR-V table records each module's specialization constants") {
  // Device-independent: a property of the checked-in artifact, so this also gates
  // the generator on a box with no Vulkan.
  //
  // The host passes specialization values by constantID. Vulkan SILENTLY IGNORES
  // a VkSpecializationMapEntry whose ID the module does not declare, so a drift
  // between host and shader is WRONG NUMBERS, not a clean error. Recording the
  // declared IDs alongside each blob is what lets GetPipeline check it.
  for (size_t mi = 0; mi < vt::vulkan::kSpirvModuleCount; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    // Structural: the pointer and the count agree, and the IDs are sorted with no
    // duplicates — GetPipeline builds VkSpecializationMapEntry positionally from
    // this array, so an unsorted or duplicated ID would bind the wrong value.
    CHECK((m.spec_ids == nullptr) == (m.spec_id_count == 0));
    for (size_t i = 1; i < m.spec_id_count; ++i) {
      CHECK(m.spec_ids[i - 1] < m.spec_ids[i]);
    }
    // vt_cast is the backend's FIRST variant axis: constants 0 and 1 are its
    // source and destination dtype, so one module serves every (src, dst) pair
    // instead of a module per pair. Every other W0 shader still declares none.
    //
    // The workgroup size is deliberately NOT such a constant — see the measured
    // reason in src/vt/vulkan/shaders/vt_common.glsl (local_size_x_id emits
    // LocalSize 1 1 1 at the vulkan1.1 target and computes ~1/128 of the tensor).
    if (std::strcmp(m.name, "vt_cast") == 0) {
      REQUIRE(m.spec_id_count == 2);  // src dtype, dst dtype
      CHECK(m.spec_ids[0] == 0u);
      CHECK(m.spec_ids[1] == 1u);
    } else if (std::strcmp(m.name, "vt_matmul") == 0) {
      // a dtype, b dtype, out dtype, orientation: 3*3*3*2 = 54 variants served by
      // ONE committed module, which is the argument for specialization constants
      // over llama.cpp's module-per-#define in miniature.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else {
      CHECK(m.spec_id_count == 0);
    }
  }
}

TEST_CASE("Vulkan specializes pipelines per dtype pair and caches them separately") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // Two DIFFERENT dtype pairs through the same committed module. The results
  // being right is necessary but not sufficient: a specialization that silently
  // did nothing would also produce right results here, because the shader's
  // defaults happen to be f32->f32. What proves the mechanism engaged is that the
  // pipeline cache GREW BY TWO — one specialized pipeline per pair.
  const size_t before = ctx.PipelineCacheSize();

  const int64_t n = 300;  // not a multiple of the workgroup size
  std::vector<float> src(n);
  for (int64_t i = 0; i < n; ++i) src[i] = static_cast<float>(i) - 150.5f;

  auto* f32_in = static_cast<float*>(vk.Alloc(n * sizeof(float)));
  auto* bf16_mid = static_cast<uint16_t*>(vk.Alloc(n * sizeof(uint16_t)));
  auto* f32_out = static_cast<float*>(vk.Alloc(n * sizeof(float)));
  vk.Copy(q, f32_in, src.data(), n * sizeof(float));

  Tensor t_f32_in = Tensor::Contiguous(f32_in, vt::DType::kF32, d, {n});
  Tensor t_bf16 = Tensor::Contiguous(bf16_mid, vt::DType::kBF16, d, {n});
  Tensor t_f32_out = Tensor::Contiguous(f32_out, vt::DType::kF32, d, {n});

  vt::CastBf16(q, t_bf16, t_f32_in);   // f32 -> bf16 : one specialization
  vt::CastF32(q, t_f32_out, t_bf16);   // bf16 -> f32 : a different one
  vk.Synchronize(q);

  CHECK(ctx.PipelineCacheSize() == before + 2);

  // The round trip must be EXACTLY the CPU codec's, so this stays in the
  // bit-exact tier rather than the NMSE tier: bf16 keeps the high 16 bits under
  // round-to-nearest-even, so a value that survives the narrowing must come back
  // identical.
  std::vector<float> back(n);
  vk.Copy(q, back.data(), f32_out, n * sizeof(float));
  vk.Synchronize(q);
  for (int64_t i = 0; i < n; ++i) {
    CAPTURE(i);
    CHECK(back[i] == vt::BF16ToF32(vt::F32ToBF16(src[i])));
  }

  vk.Free(f32_in);
  vk.Free(bf16_mid);
  vk.Free(f32_out);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan backend is registered on a Vulkan-capable host") {
  if (!VulkanPresent()) {
    MESSAGE("no Vulkan loader or no conformant device on this host; skipping");
    return;
  }
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);

  // GB10 exposes one 89.72 GiB DEVICE_LOCAL|HOST_VISIBLE heap, and llvmpipe is a
  // CPU device, so both report unified. This is load-bearing beyond a hardware
  // fact: vt::Backend's SEVEN async-output primitive defaults
  // (src/vt/backend.cpp:19-32) are documented correct exactly for unified
  // backends, so the skeleton inherits them instead of implementing them.
  CHECK(vk.UnifiedMemory());

  // A pre-recorded VkCommandBuffer is the eventual mapping
  // (include/vt/backend.h:92) but is NOT implemented; the honest answer today is
  // false, and the base class makes BeginCapture throw loudly rather than
  // silently no-op.
  CHECK_FALSE(vk.SupportsGraphCapture());
  Queue q = vk.CreateQueue();
  CHECK_THROWS_AS(vk.BeginCapture(q), std::runtime_error);

  CHECK(q.device.type == DeviceType::kVULKAN);
  CHECK(q.handle != nullptr);  // the shared VkQueue
  CHECK(q.id != 0);            // a live identity for the workspace-key machinery

  // The Vulkan API version as the capability pair. The assertion is deliberately
  // ">= 1.1", not "== 1.4": the gate is that a REAL probe ran AND that the
  // version floor this backend needs (16-bit storage in core) actually holds,
  // not that we are on one specific GPU.
  CHECK(vk.DeviceCapabilityMajor() >= 1);
  CHECK((vk.DeviceCapabilityMajor() > 1 || vk.DeviceCapabilityMinor() >= 1));

  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan allocations are 64B-aligned, byte-exact and freeable") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();

  void* p = vk.Alloc(64);
  REQUIRE(p != nullptr);
  // include/vt/backend.h:26 — vt::StepArena depends on >= 64-byte alignment.
  CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);

  vk.Memset(q, p, 0xAB, 64);
  vk.Synchronize(q);
  unsigned char dst[64];
  vk.Copy(q, dst, p, 64);
  vk.Synchronize(q);
  CHECK(dst[0] == 0xAB);
  CHECK(dst[63] == 0xAB);
  vk.Free(p);

  // A zero-byte request still yields a valid, distinct, freeable block (the CPU
  // backend's contract, which the arena relies on).
  void* z = vk.Alloc(0);
  CHECK(z != nullptr);
  vk.Free(z);
  vk.Free(nullptr);  // no-op

  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan resolves INTERIOR pointers (tensor views/slices) to the owning buffer") {
  if (!VulkanPresent()) return;
  // vt::Tensor::Slice / ::View hand out pointers INTO an allocation, while Vulkan
  // binds resources, not pointers. The allocation registry
  // (src/vt/vulkan/vulkan_buffers.h) is what bridges that; this case is its gate.
  // It is a STRONGER gate on Vulkan than on Metal, because Vulkan additionally
  // has a descriptor-offset ALIGNMENT rule that this backend sidesteps by binding
  // whole buffers and passing the byte offset in push constants — if that ever
  // regressed to a descriptor offset, a non-zero interior offset would either
  // fail validation or silently read shifted data, and this case catches both.
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  const int64_t rows = 4, cols = 8;
  auto* base = static_cast<float*>(vk.Alloc(rows * cols * sizeof(float)));
  std::vector<float> host(rows * cols);
  for (size_t i = 0; i < host.size(); ++i) host[i] = -1.0f * static_cast<float>(i + 1);
  vk.Copy(q, base, host.data(), host.size() * sizeof(float));

  // Operate on rows [1,3) only — an INTERIOR pointer at byte offset 32, which is
  // NOT a multiple of a typical minStorageBufferOffsetAlignment of 256.
  Tensor sub = Tensor::Contiguous(base + cols, vt::DType::kF32, d, {2, cols});
  vt::Relu(q, sub, sub);
  vk.Synchronize(q);

  std::vector<float> back(host.size());
  vk.Copy(q, back.data(), base, back.size() * sizeof(float));
  vk.Synchronize(q);
  // Rows 0 and 3 untouched (bit-exact); rows 1-2 relu'd to zero (input was all
  // negative), which also proves the buffer OFFSET was applied and not ignored.
  CHECK(back[0] == host[0]);
  CHECK(back[cols * 3] == host[cols * 3]);
  for (int64_t i = cols; i < cols * 3; ++i) CHECK(back[i] == 0.0f);

  vk.Free(base);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan rejects memory it did not allocate, loudly") {
  if (!VulkanPresent()) return;
  // Handing a Vulkan kernel a host std::vector is THE bring-up mistake; it must
  // throw, never read garbage.
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};
  std::vector<float> host(64, 1.0f);
  Tensor t = Tensor::Contiguous(host.data(), vt::DType::kF32, d, {8, 8});
  CHECK_THROWS_AS(vt::Relu(q, t, t), std::runtime_error);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan platform is registered and reports unified/no-pool residency") {
  if (!VulkanPresent()) return;
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kVULKAN);
  CHECK(p.device_type() == DeviceType::kVULKAN);
  CHECK_FALSE(p.is_cuda());
  CHECK_FALSE(p.is_cpu());
  CHECK(p.is_unified_memory());
  CHECK_FALSE(p.supports_graph_capture());

  CHECK(p.get_device_capability().present());
  CHECK(p.get_device_capability().major >= 1);

  // interface.py:181-187 order — bf16 is the default fallback.
  REQUIRE(p.supported_dtypes().size() == 3);
  CHECK(p.supported_dtypes()[0] == vt::DType::kBF16);

  // Unified memory: never free the only copy, never pool device scratch.
  const auto rp = p.residency_policy();
  CHECK_FALSE(rp.release_host_weights_after_upload);
  CHECK_FALSE(rp.uses_device_memory_pool);

  // No Vulkan attention kernel exists yet, so the priority list is EMPTY by
  // design (see src/vllm/platforms/vulkan.cpp) — selection must fail loudly
  // rather than name a backend whose kernels are absent.
  CHECK(p.get_attn_backend_priority().empty());
}

TEST_CASE("Vulkan registers the W0 op set and NOT the unimplemented rest") {
  if (!VulkanPresent()) return;
  // The skeleton's registered surface, stated as an executable fact so a later
  // work row cannot quietly claim more than it implements.
  for (vt::OpId op : {vt::OpId::kAdd, vt::OpId::kRelu, vt::OpId::kSiluAndMul,
                      vt::OpId::kCastBf16, vt::OpId::kCastF32, vt::OpId::kLayerNorm,
                      vt::OpId::kRmsNorm, vt::OpId::kFusedChain,
                      // VK-B: the dense path's GEMM, both orientations.
                      vt::OpId::kMatmul, vt::OpId::kMatmulBT}) {
    CHECK(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // No NATIVE Vulkan kernel yet for attention, KV cache, embedding, sampling.
  for (vt::OpId op : {vt::OpId::kPagedAttention, vt::OpId::kReshapeAndCache,
                      vt::OpId::kEmbedding, vt::OpId::kGreedyArgmax}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // ...but they no longer THROW, and this assertion used to say they did.
  //
  // Accelerator-seam work row S5 (af0b21ba) added the PORTABLE REFERENCE TIER:
  // for a unified-memory device, a missed GetOp lazily installs the CPU kernel as
  // a priority -1000 provider, below every native kernel. Vulkan is eligible (GB10
  // integrated and llvmpipe both report unified), so every op the CPU backend has
  // resolves here and runs ON THE HOST against shared memory — correct, and
  // arbitrarily slow.
  //
  // The Metal sibling was updated for this (test_metal_backend.cpp:215-231); THIS
  // file was not, and the assertion sat RED from the moment S5 landed because no
  // CI leg builds the Vulkan backend and nobody built it locally. The mirrored
  // form below is deliberate: the two backends should fail the same way.
  //
  // Measured on this tree (VK-A1, 2026-08-06): of 87 CPU-registered ops, 8 are
  // NATIVE on Vulkan, 79 are served by the reference tier, and ZERO throw.
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kVULKAN));
  // kPagedAttention, NOT kMatmul: matmul now has a native Vulkan kernel, so it
  // would no longer exercise the tier. This op must stay one that is genuinely
  // unimplemented, and it moves to the next such op as the backend fills in.
  void* paged = nullptr;
  CHECK_NOTHROW(paged = vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kVULKAN));
  CHECK(paged != nullptr);
  // BY NAME, so a host kernel can never masquerade as a native Vulkan one (Risk 7).
  const auto stats = vt::GetOpProviderStats(vt::OpId::kPagedAttention, DeviceType::kVULKAN);
  REQUIRE(stats.last_selected != nullptr);
  CHECK(std::string(stats.last_selected) == std::string(vt::kReferenceProviderName));
  CHECK(vt::GetReferenceTierHits() > 0);
}

TEST_CASE("Vulkan float-controls are PROBED and reported, not assumed") {
  if (!VulkanPresent()) return;
  // The relaxed-precision knobs Vulkan leaves implementation-defined. We cannot
  // pin fp32 denormal/signed-zero behaviour from GLSL without
  // SPV_KHR_float_controls execution modes, so the honest gate is to RECORD what
  // this device does. Both outcomes are acceptable — the shaders avoid
  // `inversesqrt` and carry integer bf16/f16 codecs precisely so that neither
  // knob can move a gated result — but a silent change here would be the first
  // clue if a future NMSE regression appeared, so it is printed rather than
  // asserted to a particular value.
  auto& ctx = vt::vulkan::VulkanContext::Get();
  MESSAGE("vulkan device: " << ctx.device_name() << " (API " << ctx.api_major() << "."
                            << ctx.api_minor() << ")");
  MESSAGE("shaderDenormPreserveFloat32 = " << ctx.denorm_preserve_f32());
  MESSAGE("shaderSignedZeroInfNanPreserveFloat32 = "
          << ctx.signed_zero_inf_nan_preserve_f32());
  // What we DO require: the workgroup-count limit must cover the dispatches the
  // skeleton makes (one workgroup per row).
  CHECK(ctx.max_workgroup_count_x() >= 65535u);
}
