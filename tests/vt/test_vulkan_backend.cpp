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
#include <limits>
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
  CHECK(n == 15);
  for (size_t mi = 0; mi < n; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    REQUIRE(m.word_count > 5);          // a SPIR-V header alone is 5 words
    CHECK(m.words[0] == 0x07230203u);   // SPIR-V magic
  }
  // The eight registered ops are served by exactly these seven modules (kCastBf16
  // and kCastF32 share vt_cast), so a rename in either direction breaks here
  // rather than at pipeline-creation time on a device we might not have.
  for (const char* want : {"vt_add", "vt_cast", "vt_embedding", "vt_fused_chain",
                           "vt_greedy_argmax", "vt_layer_norm", "vt_matmul",
                           "vt_matmul_coopmat", "vt_paged_attn", "vt_qkv_split",
                           "vt_relu", "vt_reshape_and_cache", "vt_rms_norm",
                           "vt_rope_from_cache", "vt_silu_and_mul"}) {
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
    } else if (std::strcmp(m.name, "vt_embedding") == 0) {
      // table dtype, out dtype, id width (i32 vs i64).
      REQUIRE(m.spec_id_count == 3);
      for (uint32_t want = 0; want < 3; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_rope_from_cache") == 0) {
      // q / k / cache dtype, the NeoX-vs-GPT-J pairing, and the position width.
      REQUIRE(m.spec_id_count == 5);
      for (uint32_t want = 0; want < 5; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_qkv_split") == 0) {
      // source dtype, destination dtype.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_reshape_and_cache") == 0) {
      // A single WIDTH selector, not a dtype code: this op moves bytes and
      // converts nothing, so 32-bit and 16-bit are the only two paths.
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0u);
    } else if (std::strcmp(m.name, "vt_paged_attn") == 0) {
      // query / k-cache / v-cache / out dtype.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul_coopmat") == 0) {
      // Only the b orientation and the output dtype: A and B are bf16 by the
      // hardware configuration this shader is written to, so they are not axes.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
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

  // Vulkan now HAS native kPagedAttention + kReshapeAndCache reading and writing
  // the NHD layout FlashAttentionBackend::get_kv_cache_shape allocates, so the
  // selector may reach FLASH_ATTN — on exactly the footing Metal reached it.
  // This is what let OPT-125m run end to end on Vulkan.
  {
    vllm::platforms::AttnSelectorConfig cfg;
    const auto prio = p.get_attn_backend_priority(cfg);
    REQUIRE(prio.size() == 1);
    CHECK(prio[0] == "FLASH_ATTN");
  }
  // MLA stays EMPTY, and that is a capability statement rather than a stub:
  // kMlaDecodeAttention / kMlaPrefillAttention / kConcatAndCacheMla have no
  // Vulkan kernel, so naming a backend here would route an MLA model into one
  // that cannot serve it. Selection must fail loudly instead.
  {
    vllm::platforms::AttnSelectorConfig mla;
    mla.use_mla = true;
    CHECK(p.get_attn_backend_priority(mla).empty());
  }
}

TEST_CASE("Vulkan registers the W0 op set and NOT the unimplemented rest") {
  if (!VulkanPresent()) return;
  // The skeleton's registered surface, stated as an executable fact so a later
  // work row cannot quietly claim more than it implements.
  for (vt::OpId op : {vt::OpId::kAdd, vt::OpId::kRelu, vt::OpId::kSiluAndMul,
                      vt::OpId::kCastBf16, vt::OpId::kCastF32, vt::OpId::kLayerNorm,
                      vt::OpId::kRmsNorm, vt::OpId::kFusedChain,
                      // VK-B: the dense path's GEMM (both orientations) and the
                      // two ends of the model, token ids in and out.
                      vt::OpId::kMatmul, vt::OpId::kMatmulBT,
                      vt::OpId::kEmbedding, vt::OpId::kGreedyArgmax,
                      // The attention block: paged attention (the one kernel
                      // with no llama.cpp Vulkan counterpart), the KV write, the
                      // QKV split and the rotary APPLY.
                      vt::OpId::kPagedAttention, vt::OpId::kReshapeAndCache,
                      vt::OpId::kQkvSplit, vt::OpId::kRopeFromCache}) {
    CHECK(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // No NATIVE Vulkan kernel yet for the rotary TABLE BUILD (kRopeCosSinCache and
  // kRopeNeox both construct the angle in double -- deliberately left on the
  // portable tier, mirroring vLLM's own split), quant, MoE, or the sampler beyond
  // greedy argmax.
  for (vt::OpId op : {vt::OpId::kRopeNeox, vt::OpId::kRopeCosSinCache,
                      vt::OpId::kApplyTemperature, vt::OpId::kMoeRouterTopK}) {
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
  // This must stay an op that is GENUINELY unimplemented natively, and it moves
  // on as the backend fills in: kMatmul, then kPagedAttention, then
  // kReshapeAndCache all had their turn and now have native kernels. kRopeNeox is
  // the current one.
  void* rope = nullptr;
  CHECK_NOTHROW(rope = vt::GetOp(vt::OpId::kRopeNeox, DeviceType::kVULKAN));
  CHECK(rope != nullptr);
  // BY NAME, so a host kernel can never masquerade as a native Vulkan one (Risk 7).
  const auto stats = vt::GetOpProviderStats(vt::OpId::kRopeNeox, DeviceType::kVULKAN);
  REQUIRE(stats.last_selected != nullptr);
  CHECK(std::string(stats.last_selected) == std::string(vt::kReferenceProviderName));
  CHECK(vt::GetReferenceTierHits() > 0);
}

TEST_CASE("cooperative-matrix capability is PROBED, and absent on llvmpipe") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  MESSAGE("vulkan device: " << ctx.device_name());
  // Assembled BEFORE the macro. MESSAGE(x << y) expands to
  // `MessageBuilder << x << y`, so an expression written inside it is consumed by
  // the builder rather than evaluated first -- `MESSAGE("text" << flag)` renders
  // as "1", which reads as if the capability were TRUE. For a line whose entire
  // job is to report a capability honestly, that is the worst failure mode.
  const std::string coop_line = std::string("coopmat bf16xbf16->f32 16x16x16 SUBGROUP: ") +
                                (ctx.coopmat_bf16_f32() ? "YES" : "no");
  MESSAGE(coop_line);
  MESSAGE("subgroup size: " << ctx.subgroup_size());

  // The predicate must be a REPORT, never an assumption, so this asserts the
  // property rather than a specific answer: a device may or may not have it.
  // What IS asserted unconditionally is that the backend stayed usable either
  // way -- enabling an absent extension would have failed vkCreateDevice
  // outright, so merely getting here proves the enablement is conditional.
  CHECK(ctx.subgroup_size() > 0);

  // MEASURED 2026-08-07: llvmpipe exposes VK_KHR_cooperative_matrix NOT AT ALL,
  // so on the software rasterizer -- the only Vulkan device CI can reach -- the
  // answer must be NO, and the scalar GEMM tactic is what runs. This is pinned
  // because it is what makes the CI leg a real test of the FALLBACK path rather
  // than an accident.
  if (ctx.device_name().find("llvmpipe") != std::string::npos) {
    CHECK_FALSE(ctx.coopmat_bf16_f32());
  }
}

TEST_CASE("bf16 GEMM takes the COOPMAT tactic where available, scalar where not") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // K = 32 is a multiple of 16 (the tactic requires it); M = 20 and N = 12 are
  // deliberately RAGGED so the shader's bounds-checked store is exercised rather
  // than only whole tiles.
  constexpr int64_t kM = 32, kK = 32, kN = 16;

  std::vector<float> a(kM * kK), b(kN * kK);
  for (int64_t i = 0; i < kM * kK; ++i) a[i] = 0.5f * static_cast<float>((i % 7) - 3);
  for (int64_t i = 0; i < kN * kK; ++i) b[i] = 0.25f * static_cast<float>((i % 5) - 2);

  // bf16 device operands. The values above are chosen to be exactly
  // representable in bf16, so the ORACLE below can be computed in f32 without the
  // narrowing itself becoming the error under test.
  std::vector<uint16_t> a_bf(kM * kK), b_bf(kN * kK);
  for (size_t i = 0; i < a.size(); ++i) a_bf[i] = vt::F32ToBF16(a[i]);
  for (size_t i = 0; i < b.size(); ++i) b_bf[i] = vt::F32ToBF16(b[i]);

  // Oracle: MatmulBT semantics, b is [N,K]. Sequential f32 accumulation, which is
  // the CPU kernel's contract; the coopmat tile order differs, hence the NMSE bar.
  std::vector<float> ref(kM * kN, 0.0f);
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t j = 0; j < kN; ++j) {
      float acc = 0.0f;
      for (int64_t p2 = 0; p2 < kK; ++p2) {
        acc += vt::BF16ToF32(a_bf[i * kK + p2]) * vt::BF16ToF32(b_bf[j * kK + p2]);
      }
      ref[i * kN + j] = acc;
    }
  }

  void* da = vk.Alloc(a_bf.size() * sizeof(uint16_t));
  void* db = vk.Alloc(b_bf.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(kM * kN * sizeof(float)));
  vk.Copy(q, da, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  vk.Copy(q, db, b_bf.data(), b_bf.size() * sizeof(uint16_t));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kM, kN});
  vt::MatmulBT(q, to, ta, tb);
  vk.Synchronize(q);

  // WHICH TACTIC RAN. This is the load-bearing assertion, not the numbers: the
  // scalar kernel would produce results just as correct, so a numeric check alone
  // cannot tell a working coopmat path from a silent fallback. Same reasoning as
  // the op-provider decline counters.
  //
  // NOTE the shape: M and N are WHOLE TILES here (32 and 16), because the tactic
  // now requires that. The previous version of this case used M=20, N=12 to
  // "exercise raggedness" and PASSED while the kernel was reading past the end of
  // its operand -- the out-of-bounds read stayed inside the allocation and the
  // garbage rows were discarded by the bounds-checked store. See the ragged case
  // below, which asserts the tactic DECLINES rather than trying to be correct.
  const bool coop_expected = ctx.coopmat_bf16_f32() && ctx.subgroup_size() == 32;
  const std::string tactic_line =
      std::string("bf16 GEMM tactic: ") + (coop_expected ? "COOPMAT" : "scalar");
  MESSAGE(tactic_line);
  CHECK(ctx.PipelineExistsFor(coop_expected ? "vt_matmul_coopmat" : "vt_matmul"));
  if (!coop_expected) {
    // On a device without the configuration the coopmat module must NEVER be
    // built -- selecting it there would fail at pipeline creation.
    CHECK_FALSE(ctx.PipelineExistsFor("vt_matmul_coopmat"));
  }

  std::vector<float> got(kM * kN);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double diff = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += diff * diff;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  const double nmse = den == 0.0 ? num : num / den;
  const std::string nmse_line =
      std::string("bf16 GEMM NMSE vs the f32 oracle: ") + std::to_string(nmse);
  MESSAGE(nmse_line);
  CHECK(nmse <= 5e-4);

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("a PARTIAL-TILE GEMM declines coopmat -- the shape that hung a GPU") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // M = 1 is the DECODE shape, and it is what hung GB10: lm_head dispatched
  // vt_matmul_coopmat with 9,496 workgroups, coopMatLoad read a full 16-row tile
  // from a 1-row activation -- ~30 KB past the buffer -- the GPU faulted and
  // vkWaitForFences(UINT64_MAX) never returned. A hang, not an error.
  //
  // N is also deliberately partial (17) so both edges are covered.
  constexpr int64_t kM = 1, kK = 32, kN = 17;

  std::vector<float> a(kM * kK), b(kN * kK);
  for (int64_t i = 0; i < kM * kK; ++i) a[i] = 0.5f * static_cast<float>((i % 7) - 3);
  for (int64_t i = 0; i < kN * kK; ++i) b[i] = 0.25f * static_cast<float>((i % 5) - 2);
  std::vector<uint16_t> a_bf(a.size()), b_bf(b.size());
  for (size_t i = 0; i < a.size(); ++i) a_bf[i] = vt::F32ToBF16(a[i]);
  for (size_t i = 0; i < b.size(); ++i) b_bf[i] = vt::F32ToBF16(b[i]);

  std::vector<float> ref(kM * kN, 0.0f);
  for (int64_t i = 0; i < kM; ++i) {
    for (int64_t j = 0; j < kN; ++j) {
      float acc = 0.0f;
      for (int64_t p2 = 0; p2 < kK; ++p2) {
        acc += vt::BF16ToF32(a_bf[i * kK + p2]) * vt::BF16ToF32(b_bf[j * kK + p2]);
      }
      ref[i * kN + j] = acc;
    }
  }

  void* da = vk.Alloc(a_bf.size() * sizeof(uint16_t));
  void* db = vk.Alloc(b_bf.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(kM * kN * sizeof(float)));
  vk.Copy(q, da, a_bf.data(), a_bf.size() * sizeof(uint16_t));
  vk.Copy(q, db, b_bf.data(), b_bf.size() * sizeof(uint16_t));
  vk.Synchronize(q);

  const size_t before = ctx.PipelineCacheSize();
  Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {kM, kK});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kM, kN});
  vt::MatmulBT(q, to, ta, tb);
  vk.Synchronize(q);

  // THE ASSERTION THAT MATTERS: on a partial tile the tactic must DECLINE. A
  // numeric check cannot express this -- the scalar kernel is equally correct, and
  // the coopmat kernel would not return a wrong answer here, it would HANG.
  CHECK(ctx.PipelineExistsFor("vt_matmul"));
  const size_t after = ctx.PipelineCacheSize();
  CAPTURE(before);
  CAPTURE(after);

  std::vector<float> got(kM * kN);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);
  for (int64_t i = 0; i < kM * kN; ++i) {
    CAPTURE(i);
    CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-3));
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("greedy argmax tree-reduces the vocabulary and keeps the first-wins tie-break") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // The kernel changed from one INVOCATION per row scanning the vocabulary
  // serially (10.03 ms/call measured, 10% of all GPU time in an e2e run) to one
  // WORKGROUP per row with a tree reduction. A reduction can be fast and still
  // wrong in ways a single max VALUE never reveals, so what is asserted here is
  // the INDEX, under the two conditions a reduction actually breaks.
  //
  // kV is deliberately NOT a multiple of the 128-lane workgroup, so the last
  // lane's chunk is short and the empty-range path is exercised.
  constexpr int64_t kV = 1000;

  auto run = [&](const std::vector<float>& logits) {
    void* dl = vk.Alloc(logits.size() * sizeof(float));
    void* dt = vk.Alloc(2 * sizeof(int64_t));
    vk.Copy(q, dl, logits.data(), logits.size() * sizeof(float));
    vk.Synchronize(q);
    Tensor tl = Tensor::Contiguous(dl, vt::DType::kF32, d, {1, kV});
    Tensor tt = Tensor::Contiguous(dt, vt::DType::kI64, d, {1});
    vt::GreedyArgmax(q, tt, tl);
    vk.Synchronize(q);
    int64_t got = -1;
    vk.Copy(q, &got, dt, sizeof(int64_t));
    vk.Synchronize(q);
    vk.Free(dl);
    vk.Free(dt);
    return got;
  };

  SUBCASE("a plain maximum is found across the whole vocabulary") {
    std::vector<float> l(kV, 0.0f);
    // Past lane 0's chunk, so a scan that only ever covered chunk 0 -- the shape
    // of the old single-lane kernel's parallel replacement done wrong -- misses it.
    l[777] = 5.0f;
    CHECK(run(l) == 777);
  }

  SUBCASE("ties resolve to the LOWEST index, including across lane boundaries") {
    std::vector<float> l(kV, 0.0f);
    // 8 and 900 land in different lanes' chunks, so the winner is decided by the
    // MERGE, not by either lane's own scan. A merge written with `>=` instead of
    // `>` -- the natural way to write it -- returns 900 here and passes every
    // check that only looks at the maximum value.
    l[8] = 3.0f;
    l[900] = 3.0f;
    CHECK(run(l) == 8);
  }

  SUBCASE("ties within a single lane's chunk also resolve to the lowest index") {
    std::vector<float> l(kV, 0.0f);
    l[3] = 2.0f;
    l[4] = 2.0f;
    CHECK(run(l) == 3);
  }

  SUBCASE("a NaN POISONS the scan exactly as the CPU kernel's does") {
    // cpu_sample.cpp:49 compares with `x > best`, which is false for every NaN.
    // A NaN adopted as the running best therefore blocks every later candidate,
    // and the CPU returns the index it was holding -- here 0, the initial one.
    // This is the case a STRIDED split would get wrong: the lane covering 500
    // would never see the NaN at 0 and would return 500, disagreeing with the CPU
    // oracle on a diverged model. Contiguous chunks are what make it agree.
    std::vector<float> l(kV, 0.0f);
    l[0] = std::numeric_limits<float>::quiet_NaN();
    l[500] = 9.0f;
    CHECK(run(l) == 0);
  }

  SUBCASE("a NaN AFTER the running maximum does not displace it") {
    std::vector<float> l(kV, 0.0f);
    l[10] = 7.0f;
    l[600] = std::numeric_limits<float>::quiet_NaN();
    CHECK(run(l) == 10);
  }

  vk.DestroyQueue(q);
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
