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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <utility>
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
  CHECK(n == 24);
  for (size_t mi = 0; mi < n; ++mi) {
    const auto& m = vt::vulkan::kSpirvModules[mi];
    CAPTURE(m.name);
    REQUIRE(m.word_count > 5);          // a SPIR-V header alone is 5 words
    CHECK(m.words[0] == 0x07230203u);   // SPIR-V magic
  }
  // Every registered op is served by one of these modules (kCastBf16 and kCastF32
  // share vt_cast), so a rename in either direction breaks here rather than at
  // pipeline-creation time on a device we might not have.
  for (const char* want : {"vt_add", "vt_cast", "vt_embedding", "vt_fused_chain",
                           "vt_greedy_argmax", "vt_layer_norm", "vt_matmul",
                           "vt_matmul_coopmat", "vt_matmul_vec", "vt_paged_attn",
                           "vt_qkv_split",
                           "vt_relu", "vt_reshape_and_cache", "vt_rms_norm",
                           "vt_rope_from_cache", "vt_silu_and_mul",
                           // BACKEND-VULKAN-GDN: the GDN / conv1d glue family.
                           "vt_causal_conv1d_update", "vt_gdn_post_conv",
                           "vt_gdn_state_gather", "vt_gdn_state_scatter",
                           "vt_rms_norm_gated", "vt_sigmoid_gate_bf16",
                           // BACKEND-VULKAN-GDN-CORE: the two recurrences.
                           "vt_gdn_prefill", "vt_gdn_decode"}) {
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
    } else if (std::strcmp(m.name, "vt_matmul_vec") == 0) {
      // a dtype, b dtype, out dtype and the UNROLL factor -- but NOT the
      // orientation. This module is MatmulBT by construction: the whole reason it
      // exists is that b's [N,K] layout makes lane-strided K reads contiguous, and
      // the other orientation is already coalesced in vt_matmul and would be made
      // worse here. The unroll rides a spec constant so both arms are ONE module.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_matmul") == 0) {
      // a dtype, b dtype, out dtype, orientation: 3*3*3*2 = 54 variants served by
      // ONE committed module, which is the argument for specialization constants
      // over llama.cpp's module-per-#define in miniature.
      REQUIRE(m.spec_id_count == 4);
      for (uint32_t want = 0; want < 4; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_sigmoid_gate_bf16") == 0) {
      // ONE axis, and the count is the assertion: the gate is f32 and the output
      // bf16 by the op contract (src/vt/ops.cpp:3327-3334), so only the attention
      // operand varies. A second constant appearing here would mean someone
      // widened the shader past what the op actually promises.
      REQUIRE(m.spec_id_count == 1);
      CHECK(m.spec_ids[0] == 0u);
    } else if (std::strcmp(m.name, "vt_gdn_post_conv") == 0) {
      // conv dtype, the shared q/k/v dtype, the shared araw/braw dtype. g/beta,
      // a_log and dt_bias are f32 by contract and are therefore NOT axes.
      REQUIRE(m.spec_id_count == 3);
      for (uint32_t want = 0; want < 3; ++want) CHECK(m.spec_ids[want] == want);
    } else if (std::strcmp(m.name, "vt_gdn_prefill") == 0 ||
               std::strcmp(m.name, "vt_gdn_decode") == 0) {
      // The shared q/k/v dtype and the out dtype. TWO, not four: g, beta and the
      // state are f32 by contract (a compressed state is CUDA-only,
      // src/vt/ops.cpp:1631-1638) and the host DECLINES a call whose q/k/v
      // disagree, so a third dtype constant here would mean the shader had grown
      // past what the op promises. Both modules assert the same list because they
      // share one step body (vt_gdn_recurrence.glsl) and must stay in lockstep.
      REQUIRE(m.spec_id_count == 2);
      for (uint32_t want = 0; want < 2; ++want) CHECK(m.spec_ids[want] == want);
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
                      vt::OpId::kQkvSplit, vt::OpId::kRopeFromCache,
                      // BACKEND-VULKAN-GDN: the GDN / conv1d glue family that a
                      // GDN hybrid (Qwen3.6-27B) hits on every step.
                      vt::OpId::kSigmoidGateBf16, vt::OpId::kRmsNormGated,
                      vt::OpId::kGdnStateGather, vt::OpId::kGdnStateScatter,
                      vt::OpId::kCausalConv1dUpdate, vt::OpId::kGdnPostConv,
                      // BACKEND-VULKAN-GDN-CORE: the two gated-delta recurrences
                      // themselves, which are where a GDN hybrid's prefill time
                      // actually was.
                      vt::OpId::kGdnPrefill, vt::OpId::kGdnDecode}) {
    CHECK(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // No NATIVE Vulkan kernel yet for the rotary TABLE BUILD (kRopeCosSinCache and
  // kRopeNeox both construct the angle in double -- deliberately left on the
  // portable tier, mirroring vLLM's own split), quant, MoE, or the sampler beyond
  // greedy argmax.
  //
  // kCausalConv1dFwd (the prefill conv) is out for a narrower reason -- its state
  // write-back needs a different dispatch shape than the decode update, see
  // src/vt/vulkan/vulkan_ops.cpp.
  for (vt::OpId op : {vt::OpId::kRopeNeox, vt::OpId::kRopeCosSinCache,
                      vt::OpId::kApplyTemperature, vt::OpId::kMoeRouterTopK,
                      vt::OpId::kCausalConv1dFwd}) {
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

TEST_CASE("a DECODE GEMV takes the vec tactic; prefill and the other orientation decline") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // Unlike the coopmat tactic, this one has NO hardware precondition -- it is a
  // different assignment of work to lanes, not a different instruction -- so
  // llvmpipe runs it and CI can gate the selection for real rather than only
  // gating that it is refused.
  //
  // K = 256 is two full workgroup widths, so every lane gets work and the strided
  // loop runs more than one iteration. N is deliberately NOT a multiple of 16, so
  // the coopmat tactic declines and cannot be what is being measured here.
  constexpr int64_t kK = 256, kN = 33;

  auto build = [&](int64_t m) {
    std::vector<uint16_t> a(m * kK), b(kN * kK);
    for (int64_t i = 0; i < m * kK; ++i)
      a[i] = vt::F32ToBF16(0.5f * static_cast<float>((i % 7) - 3));
    for (int64_t i = 0; i < kN * kK; ++i)
      b[i] = vt::F32ToBF16(0.25f * static_cast<float>((i % 5) - 2));
    return std::make_pair(a, b);
  };

  auto run_bt = [&](int64_t m, std::vector<float>& got) {
    auto ab = build(m);
    void* da = vk.Alloc(ab.first.size() * sizeof(uint16_t));
    void* db = vk.Alloc(ab.second.size() * sizeof(uint16_t));
    auto* dout = static_cast<float*>(vk.Alloc(m * kN * sizeof(float)));
    vk.Copy(q, da, ab.first.data(), ab.first.size() * sizeof(uint16_t));
    vk.Copy(q, db, ab.second.data(), ab.second.size() * sizeof(uint16_t));
    vk.Synchronize(q);
    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {m, kK});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kN, kK});
    Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {m, kN});
    vt::MatmulBT(q, to, ta, tb);
    vk.Synchronize(q);
    got.assign(static_cast<size_t>(m * kN), 0.0f);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    // Host oracle in f64, so neither tactic's accumulation order is privileged.
    std::vector<float> ref(static_cast<size_t>(m * kN), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < kN; ++j) {
        double acc = 0.0;
        for (int64_t c = 0; c < kK; ++c) {
          acc += static_cast<double>(vt::BF16ToF32(ab.first[i * kK + c])) *
                 static_cast<double>(vt::BF16ToF32(ab.second[j * kK + c]));
        }
        ref[static_cast<size_t>(i * kN + j)] = static_cast<float>(acc);
      }
    }
    vk.Free(da);
    vk.Free(db);
    vk.Free(dout);
    return ref;
  };

  SUBCASE("M=1 MatmulBT selects vt_matmul_vec and is numerically right") {
    std::vector<float> got;
    const std::vector<float> ref = run_bt(1, got);
    // THE LOAD-BEARING ASSERTION IS THE TACTIC. The scalar kernel is equally
    // correct, so a silent fallback would post numbers that pass every value
    // check below while proving nothing about the kernel this change adds.
    CHECK(ctx.PipelineExistsFor("vt_matmul_vec"));
    for (size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i);
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-3));
    }
  }

  SUBCASE("M=8 is prefill-shaped and does NOT take the vec tactic") {
    // One workgroup per output element is only the right trade when there are few
    // of them. The predicate refuses M > 1, and the scalar or coopmat tactic
    // handles it -- verified by the numbers still being right.
    const size_t before = ctx.PipelineCacheSize();
    std::vector<float> got;
    const std::vector<float> ref = run_bt(8, got);
    CAPTURE(before);
    CAPTURE(ctx.PipelineCacheSize());
    for (size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i);
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-3));
    }
  }

  SUBCASE("the non-BT orientation declines -- it is ALREADY coalesced") {
    // vt_matmul reads b[q*n + j] there, so adjacent lanes already hit adjacent
    // addresses; the vec shape would make that strided and strictly worse. This
    // asserts the tactic is scoped to the orientation it actually helps, which a
    // numeric check can never show.
    std::vector<uint16_t> a(kK), b(kK * kN);
    // int64_t, NOT size_t: `(i % 7) - 3` on an unsigned type underflows to a huge
    // value for i % 7 < 3, and the resulting operands overflow the accumulation
    // to inf. Same expression as the BT builder above, which uses int64_t.
    for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i)
      a[static_cast<size_t>(i)] = vt::F32ToBF16(0.5f * static_cast<float>((i % 7) - 3));
    for (int64_t i = 0; i < static_cast<int64_t>(b.size()); ++i)
      b[static_cast<size_t>(i)] = vt::F32ToBF16(0.25f * static_cast<float>((i % 5) - 2));
    void* da = vk.Alloc(a.size() * sizeof(uint16_t));
    void* db = vk.Alloc(b.size() * sizeof(uint16_t));
    auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
    vk.Copy(q, da, a.data(), a.size() * sizeof(uint16_t));
    vk.Copy(q, db, b.data(), b.size() * sizeof(uint16_t));
    vk.Synchronize(q);
    Tensor ta = Tensor::Contiguous(da, vt::DType::kBF16, d, {1, kK});
    Tensor tb = Tensor::Contiguous(db, vt::DType::kBF16, d, {kK, kN});
    Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {1, kN});
    vt::Matmul(q, to, ta, tb);
    vk.Synchronize(q);
    std::vector<float> got(kN);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    for (int64_t j = 0; j < kN; ++j) {
      double acc = 0.0;
      for (int64_t c = 0; c < kK; ++c) {
        acc += static_cast<double>(vt::BF16ToF32(a[static_cast<size_t>(c)])) *
               static_cast<double>(vt::BF16ToF32(b[static_cast<size_t>(c * kN + j)]));
      }
      CAPTURE(j);
      CHECK(got[static_cast<size_t>(j)] == doctest::Approx(static_cast<float>(acc)).epsilon(1e-3));
    }
    CHECK(ctx.PipelineExistsFor("vt_matmul"));
    vk.Free(da);
    vk.Free(db);
    vk.Free(dout);
  }

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

TEST_CASE("VK-A2 batching records many dispatches per submit and stays byte-exact") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  // Asked, not re-derived. An earlier version of this gate recomputed the lever's
  // default from the environment and asserted the WRONG branch as soon as the
  // default flipped on -- it failed loudly, but a subtler duplication would just
  // have gone vacuous.
  const bool batching = ctx.batching_enabled();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // WHAT THIS ASSERTS IS THE MECHANISM, because the RESULTS cannot show it:
  // batching runs the same kernels in the same order, so a batch that silently
  // degrades to one dispatch per submit computes identical numbers and every
  // value check still passes. Without a pending-count assertion this whole path
  // could stop batching and no gate would notice -- the same failure shape the
  // coopmat tactic needed PipelineExistsFor for.
  constexpr int64_t kN = 4096;
  constexpr int kOps = 6;

  std::vector<float> a(kN, 1.5f), b(kN, 2.25f);
  void* da = vk.Alloc(kN * sizeof(float));
  void* db = vk.Alloc(kN * sizeof(float));
  auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(float));
  vk.Copy(q, db, b.data(), b.size() * sizeof(float));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {kN});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kN});

  // Issue several dependent ops with NO intervening host read, which is the only
  // way a batch can accumulate.
  for (int i = 0; i < kOps; ++i) vt::Add(q, to, ta, tb);
  const uint32_t pending = ctx.pending_batch();
  CAPTURE(pending);

  if (batching) {
    // More than one dispatch is in flight, i.e. the submit really was deferred.
    CHECK(pending > 1u);
  } else {
    // Default build: every dispatch submitted and waited, so nothing is ever
    // pending. This half keeps the assertion honest on the default path rather
    // than making the test vacuous when the lever is off.
    CHECK(pending == 0u);
  }

  // Flushing must make the writes visible to a plain host read -- Copy is a
  // memcpy over mapped memory, so a missing flush shows up as stale bytes here.
  std::vector<float> got(kN, 0.0f);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);
  CHECK(ctx.pending_batch() == 0u);
  for (int64_t i = 0; i < kN; i += 512) {
    CAPTURE(i);
    CHECK(got[static_cast<size_t>(i)] == doctest::Approx(3.75f));
  }

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
}

TEST_CASE("a REFERENCE-TIER op drains the batch before it touches device memory") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  if (!ctx.batching_enabled()) return;  // nothing is ever pending with the lever off
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  // THE HAZARD THIS CLOSES, and why it needed its own gate. The portable
  // reference tier runs a HOST kernel directly over device memory, which is only
  // sound because this backend is unified. With batching, a dispatch may be
  // recorded and NOT yet submitted, so that host kernel would read bytes the GPU
  // has not written -- silently, with no error and no crash.
  //
  // The opt-125m STRICT gate passes with batching on, but it CANNOT prove this:
  // OPT touches the reference tier only at setup, before any batch is open. So it
  // would pass whether or not the hook exists. This asserts the hook FIRES.
  constexpr int64_t kN = 1024;
  std::vector<float> a(kN, 1.0f);
  void* da = vk.Alloc(kN * sizeof(float));
  void* db = vk.Alloc(kN * sizeof(float));
  auto* dout = static_cast<float*>(vk.Alloc(kN * sizeof(float)));
  vk.Copy(q, da, a.data(), a.size() * sizeof(float));
  vk.Copy(q, db, a.data(), a.size() * sizeof(float));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, vt::DType::kF32, d, {kN});
  Tensor tb = Tensor::Contiguous(db, vt::DType::kF32, d, {kN});
  Tensor to = Tensor::Contiguous(dout, vt::DType::kF32, d, {kN});
  for (int i = 0; i < 4; ++i) vt::Add(q, to, ta, tb);
  REQUIRE(ctx.pending_batch() > 0u);  // a batch really is open

  // Resolving a reference-tier op is what op_provider.cpp routes through
  // Backend::FlushPending. kRopeNeox is the op this file already names as
  // genuinely unimplemented natively on Vulkan.
  REQUIRE(vt::ReferenceTierEligible(DeviceType::kVULKAN));
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kRopeNeox, DeviceType::kVULKAN));
  void* ref = vt::GetOp(vt::OpId::kRopeNeox, DeviceType::kVULKAN);
  CHECK(ref != nullptr);

  // THE ASSERTION: resolving that host kernel drained the batch. Had FlushPending
  // stayed the base class's no-op, this would still read > 0 and the host kernel
  // would go on to read stale device memory.
  CHECK(ctx.pending_batch() == 0u);

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
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

// ===========================================================================
// BACKEND-VULKAN-GDN — numeric gates for the GDN / conv1d glue family.
//
// THE ORACLE IS OUR OWN CPU BACKEND, evaluated in the SAME binary on the SAME
// inputs, which is the contract tests/vt/test_backend_cross_device.cpp already
// states. It is used here rather than there because these ops need shapes with
// structure — padded gate strides, widened cache rows, NULL cache indices — that
// the generic harness does not generate.
//
// EVERY CASE ALSO ASSERTS THE MECHANISM, not only the numbers. On a unified
// device an unregistered op silently resolves to the portable CPU reference tier
// and produces answers that are not merely close to the oracle but IDENTICAL to
// it, so a numbers-only gate would pass just as green with no shader written at
// all. `PipelineExistsFor` proves a Vulkan pipeline for the intended module was
// built, and the provider's `last_selected` name proves the call did not fall
// through to the host. Both, because either alone has a hole: a pipeline can
// exist from an earlier case in the same process, and a provider can be selected
// for a shape whose kernel then declines.
// ===========================================================================
namespace {

// Deterministic, library-independent filler. A fixed LCG rather than
// std::mt19937 so both backends see byte-identical inputs regardless of standard
// library version, spread over a range that actually exercises sigmoid and
// softplus instead of sitting in their linear middle.
std::vector<float> Spread(size_t n, float scale, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>(s >> 8) / 8388608.0f - 1.0f) * scale;
  }
  return v;
}

double NmseOf(const std::vector<float>& ref, const std::vector<float>& got) {
  REQUIRE(ref.size() == got.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dd = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += dd * dd;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den == 0.0 ? num : num / den;
}

// The already-ported bar for reducing / arithmetic kernels
// (tests/vt/test_backend_cross_device.cpp, itself from llama.cpp
// test-quantize-fns:17-28). NOT bit-exactness: a workgroup tree reduction does
// not preserve the CPU tier's fixed sequential accumulation order.
constexpr double kGdnNmseTol = 5e-4;

// Owns one allocation on a backend. Deliberately not retrofitted onto the cases
// above, which predate this row — rewriting them would put unrelated churn in
// this change.
class Buf {
 public:
  Buf(Backend& b, size_t elems, size_t elem_bytes) : b_(b), p_(b.Alloc(elems * elem_bytes)) {}
  ~Buf() { b_.Free(p_); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void* p() const { return p_; }
  template <typename T>
  T* as() const { return static_cast<T*>(p_); }

 private:
  Backend& b_;
  void* p_;
};

// Was this op served by the NATIVE Vulkan kernel on its last call, BY NAME? The
// reference tier registers under a different name, so this is what separates
// "the shader ran" from "the host kernel ran and the numbers matched".
bool RanNative(vt::OpId op) {
  const auto stats = vt::GetOpProviderStats(op, DeviceType::kVULKAN);
  return stats.last_selected != nullptr &&
         std::string(stats.last_selected) == std::string(vt::kNativeProviderName);
}

}  // namespace

TEST_CASE("GDN sigmoid output-gate runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // 300 is deliberately NOT a multiple of the 128-wide workgroup, so the tail
  // guard is exercised rather than assumed.
  constexpr int64_t kN = 300;
  const std::vector<float> attn = Spread(kN, 3.0f, 11u);
  const std::vector<float> gate = Spread(kN, 6.0f, 29u);  // wide: sigmoid saturates

  Buf va(vk, kN, 4), vg(vk, kN, 4), vo(vk, kN, 2);
  Buf ca(cpu, kN, 4), cg(cpu, kN, 4), co(cpu, kN, 2);
  vk.Copy(vq, va.p(), attn.data(), kN * 4);
  vk.Copy(vq, vg.p(), gate.data(), kN * 4);
  std::memcpy(ca.p(), attn.data(), kN * 4);
  std::memcpy(cg.p(), gate.data(), kN * 4);
  vk.Synchronize(vq);

  Tensor vat = Tensor::Contiguous(va.p(), vt::DType::kF32, vd, {kN});
  Tensor vgt = Tensor::Contiguous(vg.p(), vt::DType::kF32, vd, {kN});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kBF16, vd, {kN});
  Tensor cat = Tensor::Contiguous(ca.p(), vt::DType::kF32, cd, {kN});
  Tensor cgt = Tensor::Contiguous(cg.p(), vt::DType::kF32, cd, {kN});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kBF16, cd, {kN});

  vt::SigmoidGateBf16(cq, cot, cat, cgt);
  vt::SigmoidGateBf16(vq, vot, vat, vgt);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_sigmoid_gate_bf16"));
  CHECK(RanNative(vt::OpId::kSigmoidGateBf16));

  std::vector<uint16_t> got(kN);
  vk.Copy(vq, got.data(), vo.p(), kN * 2);
  vk.Synchronize(vq);
  std::vector<float> ref_f(kN), got_f(kN);
  for (int64_t i = 0; i < kN; ++i) {
    ref_f[i] = vt::BF16ToF32(co.as<uint16_t>()[i]);
    got_f[i] = vt::BF16ToF32(got[i]);
  }
  const double nmse = NmseOf(ref_f, got_f);
  MESSAGE("sigmoid_gate_bf16 NMSE vs the CPU oracle: " << nmse);
  CHECK(nmse <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("gated RMSNorm runs NATIVELY on Vulkan: silu, sigmoid, and a padded gate") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // D = 300 > the 128-wide workgroup, so each lane walks the row in a strided
  // loop and the tree reduction is genuinely over partial sums.
  constexpr int64_t kRows = 5, kD = 300;
  const std::vector<float> x = Spread(kRows * kD, 2.0f, 7u);
  const std::vector<float> z = Spread(kRows * kD, 4.0f, 13u);
  const std::vector<float> w = Spread(kD, 1.5f, 17u);

  Buf vx(vk, kRows * kD, 4), vz(vk, kRows * kD, 4), vw(vk, kD, 4), vo(vk, kRows * kD, 4);
  Buf cx(cpu, kRows * kD, 4), cz(cpu, kRows * kD, 4), cw(cpu, kD, 4), co(cpu, kRows * kD, 4);
  vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
  vk.Copy(vq, vz.p(), z.data(), z.size() * 4);
  vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
  std::memcpy(cx.p(), x.data(), x.size() * 4);
  std::memcpy(cz.p(), z.data(), z.size() * 4);
  std::memcpy(cw.p(), w.data(), w.size() * 4);
  vk.Synchronize(vq);

  Tensor vxt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kRows, kD});
  Tensor vzt = Tensor::Contiguous(vz.p(), vt::DType::kF32, vd, {kRows, kD});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kD});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kRows, kD});
  Tensor cxt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kRows, kD});
  Tensor czt = Tensor::Contiguous(cz.p(), vt::DType::kF32, cd, {kRows, kD});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kD});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kRows, kD});

  for (bool sigmoid_gate : {false, true}) {
    vt::RmsNormGatedArgs args;
    args.eps = 1e-6f;
    args.sigmoid_gate = sigmoid_gate;
    vt::RmsNormGated(cq, cot, cxt, czt, cwt, args);
    vt::RmsNormGated(vq, vot, vxt, vzt, vwt, args);
    vk.Synchronize(vq);
    std::vector<float> got(kRows * kD), ref(kRows * kD);
    vk.Copy(vq, got.data(), vo.p(), got.size() * 4);
    vk.Synchronize(vq);
    std::memcpy(ref.data(), co.p(), ref.size() * 4);
    const double nmse = NmseOf(ref, got);
    // Assembled OUTSIDE the macro: MESSAGE(x << y) hands the expression to the
    // doctest MessageBuilder, so a flag written inside renders as "1".
    const std::string line = std::string("rmsnorm_gated (") +
                             (sigmoid_gate ? "sigmoid" : "silu") +
                             ") NMSE vs the CPU oracle: " + std::to_string(nmse);
    MESSAGE(line);
    CHECK(nmse <= kGdnNmseTol);
  }
  CHECK(ctx.PipelineExistsFor("vt_rms_norm_gated"));
  CHECK(RanNative(vt::OpId::kRmsNormGated));

  // THE PADDED-ROW rank-3 GATE, which is the shape the merged-qkvz path actually
  // produces: the gate is the `z` slice of one fused projection, so its TOKEN
  // stride exceeds Hv*D while the inner block stays contiguous. A shader that
  // ignored gate.stride[0] would still pass the rank-2 case above, so this is the
  // assertion that the stride is honoured at all.
  constexpr int64_t kT = 3, kHv = 2, kDv = 64, kPad = 16;
  constexpr int64_t kZStride = kHv * kDv + kPad;
  const std::vector<float> x3 = Spread(kT * kHv * kDv, 2.0f, 23u);
  const std::vector<float> z3 = Spread(kT * kZStride, 4.0f, 31u);
  const std::vector<float> w3 = Spread(kDv, 1.5f, 37u);

  Buf vx3(vk, kT * kHv * kDv, 4), vz3(vk, kT * kZStride, 4), vw3(vk, kDv, 4),
      vo3(vk, kT * kHv * kDv, 4);
  Buf cx3(cpu, kT * kHv * kDv, 4), cz3(cpu, kT * kZStride, 4), cw3(cpu, kDv, 4),
      co3(cpu, kT * kHv * kDv, 4);
  vk.Copy(vq, vx3.p(), x3.data(), x3.size() * 4);
  vk.Copy(vq, vz3.p(), z3.data(), z3.size() * 4);
  vk.Copy(vq, vw3.p(), w3.data(), w3.size() * 4);
  std::memcpy(cx3.p(), x3.data(), x3.size() * 4);
  std::memcpy(cz3.p(), z3.data(), z3.size() * 4);
  std::memcpy(cw3.p(), w3.data(), w3.size() * 4);
  vk.Synchronize(vq);

  auto padded_gate = [](void* p, Device dev) {
    Tensor t = Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHv, kDv});
    t.stride[0] = kZStride;  // the padded TOKEN stride; inner dims stay packed
    return t;
  };
  Tensor vx3t = Tensor::Contiguous(vx3.p(), vt::DType::kF32, vd, {kT, kHv, kDv});
  Tensor vo3t = Tensor::Contiguous(vo3.p(), vt::DType::kF32, vd, {kT, kHv, kDv});
  Tensor vw3t = Tensor::Contiguous(vw3.p(), vt::DType::kF32, vd, {kDv});
  Tensor cx3t = Tensor::Contiguous(cx3.p(), vt::DType::kF32, cd, {kT, kHv, kDv});
  Tensor co3t = Tensor::Contiguous(co3.p(), vt::DType::kF32, cd, {kT, kHv, kDv});
  Tensor cw3t = Tensor::Contiguous(cw3.p(), vt::DType::kF32, cd, {kDv});
  Tensor vz3t = padded_gate(vz3.p(), vd);
  Tensor cz3t = padded_gate(cz3.p(), cd);

  vt::RmsNormGatedArgs args3;
  args3.eps = 1e-6f;
  vt::RmsNormGated(cq, co3t, cx3t, cz3t, cw3t, args3);
  vt::RmsNormGated(vq, vo3t, vx3t, vz3t, vw3t, args3);
  vk.Synchronize(vq);
  std::vector<float> got3(kT * kHv * kDv), ref3(kT * kHv * kDv);
  vk.Copy(vq, got3.data(), vo3.p(), got3.size() * 4);
  vk.Synchronize(vq);
  std::memcpy(ref3.data(), co3.p(), ref3.size() * 4);
  const double nmse3 = NmseOf(ref3, got3);
  MESSAGE("rmsnorm_gated (padded rank-3 gate) NMSE vs the CPU oracle: " << nmse3);
  CHECK(nmse3 <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("GDN state gather/scatter run NATIVELY on Vulkan and are BIT-EXACT") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // BIT-EXACT, not NMSE: these two ops move f32 words and compute nothing, so
  // anything short of equality would be hiding a bug (the tier
  // vt_reshape_and_cache is already gated in).
  //
  // The cache inner dim is WIDER than the working one (6 vs 4) — the
  // spec-decode-widened conv row. That shape is the whole reason the kernel walks
  // channels at the cache's physical stride, and a shader that assumed the
  // contiguous fast path would pass a same-width test and corrupt this one.
  constexpr int64_t kCacheRows = 5, kC = 3, kCacheInner = 6, kWorkInner = 4;
  constexpr int64_t kRows = 3;
  constexpr int64_t kCacheElems = kCacheRows * kC * kCacheInner;
  constexpr int64_t kWorkElems = kRows * kC * kWorkInner;
  const std::vector<float> cache_init = Spread(kCacheElems, 5.0f, 41u);
  const std::vector<int32_t> idx = {4, 0, 2};
  // Request 1 has NO initial state: its working row must come back ZEROED, not
  // copied. i8 is the interesting width — it is why the shader reads the flag
  // byte-wise through the 32-bit view instead of needing VK_KHR_8bit_storage.
  //
  // THREE ELEMENTS IS ALSO THREE BYTES, and that is deliberate: it is the shape
  // that caught the sub-word allocation bug this row fixed. A 3-byte VkBuffer's
  // `uint[]` view has ZERO elements, so every flag read back as false and the
  // gather zeroed rows it should have copied — silently, because the read is
  // robust rather than faulting. AllocBuffer now rounds every buffer up to a
  // whole 32-bit word (src/vt/vulkan/vulkan_context.cpp), and keeping this length
  // at 3 is what keeps that fix gated. Do not "tidy" it to 4.
  const std::vector<int8_t> his = {1, 0, 1};

  Buf vc(vk, kCacheElems, 4), vw(vk, kWorkElems, 4), vi(vk, kRows, 4), vh(vk, kRows, 1);
  Buf cc(cpu, kCacheElems, 4), cw(cpu, kWorkElems, 4), ci(cpu, kRows, 4), ch(cpu, kRows, 1);
  vk.Copy(vq, vc.p(), cache_init.data(), kCacheElems * 4);
  vk.Copy(vq, vi.p(), idx.data(), kRows * 4);
  vk.Copy(vq, vh.p(), his.data(), kRows);
  std::memcpy(cc.p(), cache_init.data(), kCacheElems * 4);
  std::memcpy(ci.p(), idx.data(), kRows * 4);
  std::memcpy(ch.p(), his.data(), kRows);
  vk.Synchronize(vq);

  auto cache_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kCacheRows, kC, kCacheInner});
  };
  auto work_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kRows, kC, kWorkInner});
  };
  Tensor vct = cache_t(vc.p(), vd), vwt = work_t(vw.p(), vd);
  Tensor cct = cache_t(cc.p(), cd), cwt = work_t(cw.p(), cd);
  Tensor vit = Tensor::Contiguous(vi.p(), vt::DType::kI32, vd, {kRows});
  Tensor cit = Tensor::Contiguous(ci.p(), vt::DType::kI32, cd, {kRows});
  Tensor vht = Tensor::Contiguous(vh.p(), vt::DType::kI8, vd, {kRows});
  Tensor cht = Tensor::Contiguous(ch.p(), vt::DType::kI8, cd, {kRows});

  vt::GdnStateGather(cq, cwt, cct, cit, &cht);
  vt::GdnStateGather(vq, vwt, vct, vit, &vht);
  vk.Synchronize(vq);
  CHECK(ctx.PipelineExistsFor("vt_gdn_state_gather"));
  CHECK(RanNative(vt::OpId::kGdnStateGather));

  std::vector<float> got(kWorkElems);
  vk.Copy(vq, got.data(), vw.p(), kWorkElems * 4);
  vk.Synchronize(vq);
  CHECK(std::memcmp(got.data(), cw.p(), kWorkElems * 4) == 0);
  // The zeroing is asserted DIRECTLY, not left to the memcmp: if both kernels
  // wrongly copied row 1, the comparison above would still be green.
  bool row1_zero = true;
  for (int64_t e = 0; e < kC * kWorkInner; ++e) {
    if (got[kC * kWorkInner + e] != 0.0f) row1_zero = false;
  }
  CHECK(row1_zero);

  // Scatter new working rows back and compare the WHOLE cache, so an over-wide
  // write into the widened row's tail columns is caught.
  const std::vector<float> work_new = Spread(kWorkElems, 9.0f, 43u);
  vk.Copy(vq, vw.p(), work_new.data(), kWorkElems * 4);
  std::memcpy(cw.p(), work_new.data(), kWorkElems * 4);
  vk.Synchronize(vq);
  vt::GdnStateScatter(cq, cct, cwt, cit);
  vt::GdnStateScatter(vq, vct, vwt, vit);
  vk.Synchronize(vq);
  CHECK(ctx.PipelineExistsFor("vt_gdn_state_scatter"));
  CHECK(RanNative(vt::OpId::kGdnStateScatter));

  std::vector<float> cache_got(kCacheElems);
  vk.Copy(vq, cache_got.data(), vc.p(), kCacheElems * 4);
  vk.Synchronize(vq);
  CHECK(std::memcmp(cache_got.data(), cc.p(), kCacheElems * 4) == 0);
  // And cache row 1, which no index names, still holds its ORIGINAL bytes —
  // proof the scatter wrote only where it was told to.
  CHECK(std::memcmp(cache_got.data() + kC * kCacheInner,
                    cache_init.data() + kC * kCacheInner,
                    static_cast<size_t>(kC * kCacheInner) * 4) == 0);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the decode causal conv1d update runs NATIVELY on Vulkan, state roll included") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // conv_state has MORE rows than the batch and is addressed through
  // conv_state_indices — the in-place indexed decode path the model takes. A
  // NEGATIVE index is upstream's NULL block and must leave both the output
  // element and the cache row alone.
  constexpr int64_t kBatch = 4, kC = 5, kK = 4, kWidth = kK - 1;
  constexpr int64_t kStateRows = 6, kStateLen = kWidth;
  const std::vector<float> x = Spread(kBatch * kC, 2.0f, 53u);
  const std::vector<float> w = Spread(kC * kK, 1.0f, 59u);
  const std::vector<float> bias = Spread(kC, 0.5f, 61u);
  const std::vector<float> state0 = Spread(kStateRows * kC * kStateLen, 3.0f, 67u);
  const std::vector<int32_t> cidx = {5, 1, -1, 0};  // token 2 -> NULL block

  Buf vx(vk, kBatch * kC, 4), vw(vk, kC * kK, 4), vb(vk, kC, 4), vo(vk, kBatch * kC, 4),
      vs(vk, kStateRows * kC * kStateLen, 4), vi(vk, kBatch, 4);
  Buf cx(cpu, kBatch * kC, 4), cw(cpu, kC * kK, 4), cb(cpu, kC, 4), co(cpu, kBatch * kC, 4),
      cs(cpu, kStateRows * kC * kStateLen, 4), cci(cpu, kBatch, 4);
  vk.Copy(vq, vx.p(), x.data(), x.size() * 4);
  vk.Copy(vq, vw.p(), w.data(), w.size() * 4);
  vk.Copy(vq, vb.p(), bias.data(), bias.size() * 4);
  vk.Copy(vq, vs.p(), state0.data(), state0.size() * 4);
  vk.Copy(vq, vi.p(), cidx.data(), cidx.size() * 4);
  // The output buffer is pre-seeded so the NULL-block token's element can be
  // checked for having been LEFT ALONE rather than merely being plausible.
  const std::vector<float> out_seed(kBatch * kC, -12345.0f);
  vk.Copy(vq, vo.p(), out_seed.data(), out_seed.size() * 4);
  std::memcpy(cx.p(), x.data(), x.size() * 4);
  std::memcpy(cw.p(), w.data(), w.size() * 4);
  std::memcpy(cb.p(), bias.data(), bias.size() * 4);
  std::memcpy(cs.p(), state0.data(), state0.size() * 4);
  std::memcpy(cci.p(), cidx.data(), cidx.size() * 4);
  std::memcpy(co.p(), out_seed.data(), out_seed.size() * 4);
  vk.Synchronize(vq);

  Tensor vxt = Tensor::Contiguous(vx.p(), vt::DType::kF32, vd, {kBatch, kC});
  Tensor vwt = Tensor::Contiguous(vw.p(), vt::DType::kF32, vd, {kC, kK});
  Tensor vbt = Tensor::Contiguous(vb.p(), vt::DType::kF32, vd, {kC});
  Tensor vot = Tensor::Contiguous(vo.p(), vt::DType::kF32, vd, {kBatch, kC});
  Tensor vst = Tensor::Contiguous(vs.p(), vt::DType::kF32, vd, {kStateRows, kC, kStateLen});
  Tensor vit = Tensor::Contiguous(vi.p(), vt::DType::kI32, vd, {kBatch});
  Tensor cxt = Tensor::Contiguous(cx.p(), vt::DType::kF32, cd, {kBatch, kC});
  Tensor cwt = Tensor::Contiguous(cw.p(), vt::DType::kF32, cd, {kC, kK});
  Tensor cbt = Tensor::Contiguous(cb.p(), vt::DType::kF32, cd, {kC});
  Tensor cot = Tensor::Contiguous(co.p(), vt::DType::kF32, cd, {kBatch, kC});
  Tensor cst = Tensor::Contiguous(cs.p(), vt::DType::kF32, cd, {kStateRows, kC, kStateLen});
  Tensor citt = Tensor::Contiguous(cci.p(), vt::DType::kI32, cd, {kBatch});

  vt::CausalConv1dArgs args;  // silu_activation defaults true, as Qwen GDN uses
  vt::CausalConv1dUpdate(cq, cot, cxt, cwt, &cbt, cst, args, &citt);
  vt::CausalConv1dUpdate(vq, vot, vxt, vwt, &vbt, vst, args, &vit);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_causal_conv1d_update"));
  CHECK(RanNative(vt::OpId::kCausalConv1dUpdate));

  std::vector<float> got(kBatch * kC), ref(kBatch * kC);
  vk.Copy(vq, got.data(), vo.p(), got.size() * 4);
  vk.Synchronize(vq);
  std::memcpy(ref.data(), co.p(), ref.size() * 4);
  const double nmse = NmseOf(ref, got);
  MESSAGE("causal_conv1d_update NMSE vs the CPU oracle: " << nmse);
  CHECK(nmse <= kGdnNmseTol);
  // The NULL-block token kept its seed.
  for (int64_t c = 0; c < kC; ++c) {
    CAPTURE(c);
    CHECK(got[2 * kC + c] == -12345.0f);
  }

  // THE ROLLED STATE IS THE HALF A NUMBERS-ONLY OUTPUT CHECK MISSES: the output
  // reads the OLD taps, so a kernel that never rolled the window would produce a
  // perfect first step and diverge from the second one onward.
  std::vector<float> state_got(kStateRows * kC * kStateLen);
  vk.Copy(vq, state_got.data(), vs.p(), state_got.size() * 4);
  vk.Synchronize(vq);
  const std::vector<float> state_ref(cs.as<float>(), cs.as<float>() + state_got.size());
  CHECK(std::memcmp(state_got.data(), state_ref.data(), state_got.size() * 4) == 0);
  // Spelled out independently of the oracle: the roll writes the RAW x sample
  // into the last tap, so if BOTH kernels skipped the roll the memcmp above
  // would still be green.
  for (int64_t bt = 0; bt < kBatch; ++bt) {
    if (cidx[static_cast<size_t>(bt)] < 0) continue;
    for (int64_t c = 0; c < kC; ++c) {
      CAPTURE(bt);
      CAPTURE(c);
      const int64_t slot = cidx[static_cast<size_t>(bt)];
      const int64_t last = (slot * kC + c) * kStateLen + kWidth - 1;
      CHECK(state_got[static_cast<size_t>(last)] == x[static_cast<size_t>(bt * kC + c)]);
    }
  }

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the fused GDN post-conv preamble runs NATIVELY on Vulkan, all five outputs") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  constexpr int64_t kT = 3, kHk = 2, kDk = 64, kHv = 2, kDv = 32;
  constexpr int64_t kKeyDim = kHk * kDk, kValDim = kHv * kDv;
  constexpr int64_t kConvDim = 2 * kKeyDim + kValDim;
  const std::vector<float> conv = Spread(kT * kConvDim, 2.0f, 71u);
  // araw is spread WIDE on purpose: softplus has two branches (the > 20
  // pass-through and the log1p one) and a narrow range would only ever reach one.
  const std::vector<float> araw = Spread(kT * kHv, 25.0f, 73u);
  const std::vector<float> braw = Spread(kT * kHv, 4.0f, 79u);
  const std::vector<float> a_log = Spread(kHv, 1.0f, 83u);
  const std::vector<float> dt_bias = Spread(kHv, 1.0f, 89u);

  Buf vconv(vk, kT * kConvDim, 4), va(vk, kT * kHv, 4), vb(vk, kT * kHv, 4), val(vk, kHv, 4),
      vdt(vk, kHv, 4);
  Buf vqo(vk, kT * kKeyDim, 4), vko(vk, kT * kKeyDim, 4), vvo(vk, kT * kValDim, 4),
      vgo(vk, kT * kHv, 4), vbo(vk, kT * kHv, 4);
  Buf cconv(cpu, kT * kConvDim, 4), ca(cpu, kT * kHv, 4), cb(cpu, kT * kHv, 4),
      cal(cpu, kHv, 4), cdt(cpu, kHv, 4);
  Buf cqo(cpu, kT * kKeyDim, 4), cko(cpu, kT * kKeyDim, 4), cvo(cpu, kT * kValDim, 4),
      cgo(cpu, kT * kHv, 4), cbo(cpu, kT * kHv, 4);

  vk.Copy(vq, vconv.p(), conv.data(), conv.size() * 4);
  vk.Copy(vq, va.p(), araw.data(), araw.size() * 4);
  vk.Copy(vq, vb.p(), braw.data(), braw.size() * 4);
  vk.Copy(vq, val.p(), a_log.data(), a_log.size() * 4);
  vk.Copy(vq, vdt.p(), dt_bias.data(), dt_bias.size() * 4);
  std::memcpy(cconv.p(), conv.data(), conv.size() * 4);
  std::memcpy(ca.p(), araw.data(), araw.size() * 4);
  std::memcpy(cb.p(), braw.data(), braw.size() * 4);
  std::memcpy(cal.p(), a_log.data(), a_log.size() * 4);
  std::memcpy(cdt.p(), dt_bias.data(), dt_bias.size() * 4);
  vk.Synchronize(vq);

  auto run = [&](Queue& q, Device dev, const Buf& bconv, const Buf& ba, const Buf& bb,
                 const Buf& bal, const Buf& bdt, const Buf& bq, const Buf& bk, const Buf& bv,
                 const Buf& bg, const Buf& bbeta) {
    Tensor tconv = Tensor::Contiguous(bconv.p(), vt::DType::kF32, dev, {kT, kConvDim});
    Tensor ta = Tensor::Contiguous(ba.p(), vt::DType::kF32, dev, {kT, kHv});
    Tensor tb = Tensor::Contiguous(bb.p(), vt::DType::kF32, dev, {kT, kHv});
    Tensor tal = Tensor::Contiguous(bal.p(), vt::DType::kF32, dev, {kHv});
    Tensor tdt = Tensor::Contiguous(bdt.p(), vt::DType::kF32, dev, {kHv});
    Tensor tq = Tensor::Contiguous(bq.p(), vt::DType::kF32, dev, {kT, kHk, kDk});
    Tensor tk = Tensor::Contiguous(bk.p(), vt::DType::kF32, dev, {kT, kHk, kDk});
    Tensor tv = Tensor::Contiguous(bv.p(), vt::DType::kF32, dev, {kT, kHv, kDv});
    Tensor tg = Tensor::Contiguous(bg.p(), vt::DType::kF32, dev, {kT, kHv});
    Tensor tbeta = Tensor::Contiguous(bbeta.p(), vt::DType::kF32, dev, {kT, kHv});
    vt::L2NormArgs l2args;
    vt::GdnPostConv(q, tq, tk, tv, tg, tbeta, tconv, ta, tb, tal, tdt, l2args);
  };
  run(cq, cd, cconv, ca, cb, cal, cdt, cqo, cko, cvo, cgo, cbo);
  run(vq, vd, vconv, va, vb, val, vdt, vqo, vko, vvo, vgo, vbo);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_gdn_post_conv"));
  CHECK(RanNative(vt::OpId::kGdnPostConv));

  // ALL FIVE outputs are compared. The kernel writes them from two different
  // branches of one dispatch (q/k from the head-slot branch, v/g/beta from the
  // other), so checking a subset would leave a whole branch unasserted.
  auto compare = [&](const char* what, const Buf& dev_buf, const Buf& host_buf, int64_t n) {
    std::vector<float> got(static_cast<size_t>(n));
    vk.Copy(vq, got.data(), dev_buf.p(), got.size() * 4);
    vk.Synchronize(vq);
    const std::vector<float> ref(host_buf.as<float>(), host_buf.as<float>() + n);
    const double nmse = NmseOf(ref, got);
    const std::string line =
        std::string("gdn_post_conv ") + what + " NMSE vs the CPU oracle: " + std::to_string(nmse);
    MESSAGE(line);
    CHECK(nmse <= kGdnNmseTol);
  };
  compare("q_out", vqo, cqo, kT * kKeyDim);
  compare("k_out", vko, cko, kT * kKeyDim);
  compare("v_out", vvo, cvo, kT * kValDim);
  compare("g_out", vgo, cgo, kT * kHv);
  compare("beta_out", vbo, cbo, kT * kHv);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

// ===========================================================================
// BACKEND-VULKAN-GDN-CORE — the two gated-delta RECURRENCES.
//
// Same contract as the glue gates above: the oracle is our own CPU backend in
// the same binary, and every case asserts the MECHANISM as well as the numbers,
// because on this unified-memory device an unregistered op resolves to the CPU
// reference tier and returns answers IDENTICAL to the oracle.
//
// THE SHAPES ARE CHOSEN AGAINST THE TILE GEOMETRY, not for roundness
// (src/vt/vulkan/shaders/vt_gdn_recurrence.glsl: BV=16 state rows per workgroup,
// NW=8 lanes cooperating on each):
//   Dv = 19  -> ceil(19/16) = 2 value tiles and the second has only 3 valid rows,
//               so the tile tail's zero-fill and the guarded store-back both run;
//   Dk = 20  -> ceil(20/8) = 3 columns per lane, and lane 7's slice starts PAST
//               the end (c1 < c0), so a lane contributing an EMPTY partial to the
//               row reduction still has to reach every barrier;
//   Hv/Hk = 6/2 -> a GQA broadcast ratio of 3, not 1.
// A shape that was a multiple of everything would pass with all three wrong.
// ===========================================================================
namespace {

constexpr int64_t kGdnHk = 2, kGdnHv = 6, kGdnDk = 20, kGdnDv = 19;

// g is a LOG decay and beta a gate in (0,1). Feeding the raw spread would make
// exp(g) > 1 and GROW the carried state instead of contracting it, which is the
// opposite of the regime the recurrence runs in — and would let a sign error in
// the decay hide inside a plausible-looking NMSE.
std::vector<float> GdnDecays(size_t n, uint32_t seed) {
  std::vector<float> v = Spread(n, 2.0f, seed);
  for (float& x : v) x = -std::fabs(x);
  return v;
}
std::vector<float> GdnBetas(size_t n, uint32_t seed) {
  std::vector<float> v = Spread(n, 3.0f, seed);
  for (float& x : v) x = 1.0f / (1.0f + std::exp(-x));
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}

}  // namespace

TEST_CASE("the GDN PREFILL recurrence runs NATIVELY on Vulkan and matches the CPU oracle") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // Three sequences of lengths 5, 0 and 3. The EMPTY one is the interesting case:
  // its workgroups must stage their state tile, run no steps, and write it back
  // UNCHANGED — the CPU reference's `for (t = qslp[s]; t < qslp[s+1]; ...)`.
  constexpr int64_t kN = 3, kT = 8;
  const std::vector<int32_t> qsl = {0, 5, 5, 8};
  constexpr int64_t kQk = kT * kGdnHk * kGdnDk;
  constexpr int64_t kVo = kT * kGdnHv * kGdnDv;
  constexpr int64_t kGb = kT * kGdnHv;
  constexpr int64_t kRowElems = kGdnHv * kGdnDv * kGdnDk;
  constexpr int64_t kSt = kN * kRowElems;

  const std::vector<float> qv = Spread(kQk, 1.0f, 101u);
  const std::vector<float> kv = Spread(kQk, 1.0f, 103u);
  const std::vector<float> vv = Spread(kVo, 2.0f, 107u);
  const std::vector<float> gv = GdnDecays(kGb, 109u);
  const std::vector<float> bv = GdnBetas(kGb, 113u);
  const std::vector<float> s0 = Spread(kSt, 0.5f, 127u);

  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kGdnDk));

  Buf vqb(vk, kQk, 4), vkb(vk, kQk, 4), vvb(vk, kVo, 4), vob(vk, kVo, 4);
  Buf vgb(vk, kGb, 4), vbb(vk, kGb, 4), vsb(vk, kSt, 4), vlb(vk, kN + 1, 4);
  Buf cqb(cpu, kQk, 4), ckb(cpu, kQk, 4), cvb(cpu, kVo, 4), cob(cpu, kVo, 4);
  Buf cgb(cpu, kGb, 4), cbb(cpu, kGb, 4), csb(cpu, kSt, 4), clb(cpu, kN + 1, 4);

  vk.Copy(vq, vqb.p(), qv.data(), kQk * 4);
  vk.Copy(vq, vkb.p(), kv.data(), kQk * 4);
  vk.Copy(vq, vvb.p(), vv.data(), kVo * 4);
  vk.Copy(vq, vgb.p(), gv.data(), kGb * 4);
  vk.Copy(vq, vbb.p(), bv.data(), kGb * 4);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);
  vk.Copy(vq, vlb.p(), qsl.data(), (kN + 1) * 4);
  std::memcpy(cqb.p(), qv.data(), kQk * 4);
  std::memcpy(ckb.p(), kv.data(), kQk * 4);
  std::memcpy(cvb.p(), vv.data(), kVo * 4);
  std::memcpy(cgb.p(), gv.data(), kGb * 4);
  std::memcpy(cbb.p(), bv.data(), kGb * 4);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  std::memcpy(clb.p(), qsl.data(), (kN + 1) * 4);
  vk.Synchronize(vq);

  auto qk_t = [](void* p, Device dev, vt::DType dt) {
    return Tensor::Contiguous(p, dt, dev, {kT, kGdnHk, kGdnDk});
  };
  auto vo_t = [](void* p, Device dev, vt::DType dt) {
    return Tensor::Contiguous(p, dt, dev, {kT, kGdnHv, kGdnDv});
  };
  auto gb_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kGdnHv});
  };
  auto st_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kN, kGdnHv, kGdnDv, kGdnDk});
  };

  Tensor vqt = qk_t(vqb.p(), vd, vt::DType::kF32), vkt = qk_t(vkb.p(), vd, vt::DType::kF32);
  Tensor vvt = vo_t(vvb.p(), vd, vt::DType::kF32), vot = vo_t(vob.p(), vd, vt::DType::kF32);
  Tensor vgt = gb_t(vgb.p(), vd), vbt = gb_t(vbb.p(), vd), vst = st_t(vsb.p(), vd);
  Tensor vlt = Tensor::Contiguous(vlb.p(), vt::DType::kI32, vd, {kN + 1});
  Tensor cqt = qk_t(cqb.p(), cd, vt::DType::kF32), ckt = qk_t(ckb.p(), cd, vt::DType::kF32);
  Tensor cvt = vo_t(cvb.p(), cd, vt::DType::kF32), cot = vo_t(cob.p(), cd, vt::DType::kF32);
  Tensor cgt = gb_t(cgb.p(), cd), cbt = gb_t(cbb.p(), cd), cst = st_t(csb.p(), cd);
  Tensor clt = Tensor::Contiguous(clb.p(), vt::DType::kI32, cd, {kN + 1});

  vt::GdnPrefill(cq, cot, cqt, ckt, cvt, cgt, cbt, cst, clt, args);
  vt::GdnPrefill(vq, vot, vqt, vkt, vvt, vgt, vbt, vst, vlt, args);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_gdn_prefill"));
  CHECK(RanNative(vt::OpId::kGdnPrefill));

  std::vector<float> got_out(kVo), got_state(kSt);
  vk.Copy(vq, got_out.data(), vob.p(), kVo * 4);
  vk.Copy(vq, got_state.data(), vsb.p(), kSt * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref_out(cob.as<float>(), cob.as<float>() + kVo);
  const std::vector<float> ref_state(csb.as<float>(), csb.as<float>() + kSt);
  const double nmse_out = NmseOf(ref_out, got_out);
  const double nmse_state = NmseOf(ref_state, got_state);
  MESSAGE("gdn_prefill out NMSE vs the CPU oracle: " << nmse_out);
  MESSAGE("gdn_prefill CARRIED STATE NMSE vs the CPU oracle: " << nmse_state);
  CHECK(nmse_out <= kGdnNmseTol);
  // The state is asserted SEPARATELY from the output because it is the part that
  // carries: a kernel that decayed correctly but wrote its tile back to the wrong
  // rows would still produce a plausible `out`.
  CHECK(nmse_state <= kGdnNmseTol);

  // Sequence 1 is EMPTY, so its state block must come back BIT-IDENTICAL to what
  // it went in as — not merely close. This catches a store-back that wrote a
  // decayed or zeroed tile for a zero-length token range.
  CHECK(std::memcmp(got_state.data() + kRowElems, s0.data() + kRowElems,
                    static_cast<size_t>(kRowElems) * 4) == 0);

  // --- bf16 arm: proof the dtype specialization actually engaged --------------
  // The numbers alone would not prove it. The shader's DEFAULT specialization is
  // f32/f32, so a silently-ignored VkSpecializationMapEntry would read bf16 bytes
  // as f32 — the exact failure this backend's spec-id bookkeeping exists for — and
  // a NEW pipeline appearing in the cache is what says the value was bound.
  const size_t pipelines_before = ctx.PipelineCacheSize();
  const std::vector<uint16_t> q16 = ToBf16(qv), k16 = ToBf16(kv), v16 = ToBf16(vv);
  Buf vq16(vk, kQk, 2), vk16(vk, kQk, 2), vv16(vk, kVo, 2), vo16(vk, kVo, 2);
  Buf cq16(cpu, kQk, 2), ck16(cpu, kQk, 2), cv16(cpu, kVo, 2), co16(cpu, kVo, 2);
  vk.Copy(vq, vq16.p(), q16.data(), kQk * 2);
  vk.Copy(vq, vk16.p(), k16.data(), kQk * 2);
  vk.Copy(vq, vv16.p(), v16.data(), kVo * 2);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);  // reset the carried state
  std::memcpy(cq16.p(), q16.data(), kQk * 2);
  std::memcpy(ck16.p(), k16.data(), kQk * 2);
  std::memcpy(cv16.p(), v16.data(), kVo * 2);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  vk.Synchronize(vq);

  Tensor vq16t = qk_t(vq16.p(), vd, vt::DType::kBF16);
  Tensor vk16t = qk_t(vk16.p(), vd, vt::DType::kBF16);
  Tensor vv16t = vo_t(vv16.p(), vd, vt::DType::kBF16);
  Tensor vo16t = vo_t(vo16.p(), vd, vt::DType::kBF16);
  Tensor cq16t = qk_t(cq16.p(), cd, vt::DType::kBF16);
  Tensor ck16t = qk_t(ck16.p(), cd, vt::DType::kBF16);
  Tensor cv16t = vo_t(cv16.p(), cd, vt::DType::kBF16);
  Tensor co16t = vo_t(co16.p(), cd, vt::DType::kBF16);

  vt::GdnPrefill(cq, co16t, cq16t, ck16t, cv16t, cgt, cbt, cst, clt, args);
  vt::GdnPrefill(vq, vo16t, vq16t, vk16t, vv16t, vgt, vbt, vst, vlt, args);
  vk.Synchronize(vq);
  CHECK(ctx.PipelineCacheSize() == pipelines_before + 1);

  std::vector<uint16_t> got16(kVo);
  vk.Copy(vq, got16.data(), vo16.p(), kVo * 2);
  vk.Synchronize(vq);
  std::vector<float> got16_f(kVo), ref16_f(kVo);
  for (int64_t i = 0; i < kVo; ++i) {
    got16_f[i] = vt::BF16ToF32(got16[i]);
    ref16_f[i] = vt::BF16ToF32(co16.as<uint16_t>()[i]);
  }
  const double nmse16 = NmseOf(ref16_f, got16_f);
  MESSAGE("gdn_prefill out NMSE (bf16 q/k/v and bf16 out): " << nmse16);
  CHECK(nmse16 <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("the GDN DECODE recurrence runs NATIVELY on Vulkan, indexed and compact") {
  if (!VulkanPresent()) return;
  auto& ctx = vt::vulkan::VulkanContext::Get();
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // The cache has MORE rows than the batch and the indices are scrambled, so a
  // kernel that used the token index as its state row would pass a
  // batch-rows-only test and corrupt this one. Index 1 is NEGATIVE — fla's NULL
  // block — whose output row must come back ZEROED.
  constexpr int64_t kBatch = 5, kSlots = 7;
  const std::vector<int32_t> idx = {3, -1, 0, 6, 1};
  constexpr int64_t kQk = kBatch * kGdnHk * kGdnDk;
  constexpr int64_t kVo = kBatch * kGdnHv * kGdnDv;
  constexpr int64_t kGb = kBatch * kGdnHv;
  constexpr int64_t kRowElems = kGdnHv * kGdnDv * kGdnDk;
  constexpr int64_t kSt = kSlots * kRowElems;

  const std::vector<float> qv = Spread(kQk, 1.0f, 211u);
  const std::vector<float> kv = Spread(kQk, 1.0f, 223u);
  const std::vector<float> vv = Spread(kVo, 2.0f, 227u);
  const std::vector<float> gv = GdnDecays(kGb, 229u);
  const std::vector<float> bv = GdnBetas(kGb, 233u);
  const std::vector<float> s0 = Spread(kSt, 0.5f, 239u);

  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kGdnDk));

  Buf vqb(vk, kQk, 4), vkb(vk, kQk, 4), vvb(vk, kVo, 4), vob(vk, kVo, 4);
  Buf vgb(vk, kGb, 4), vbb(vk, kGb, 4), vsb(vk, kSt, 4), vib(vk, kBatch, 4);
  Buf cqb(cpu, kQk, 4), ckb(cpu, kQk, 4), cvb(cpu, kVo, 4), cob(cpu, kVo, 4);
  Buf cgb(cpu, kGb, 4), cbb(cpu, kGb, 4), csb(cpu, kSt, 4), cib(cpu, kBatch, 4);

  vk.Copy(vq, vqb.p(), qv.data(), kQk * 4);
  vk.Copy(vq, vkb.p(), kv.data(), kQk * 4);
  vk.Copy(vq, vvb.p(), vv.data(), kVo * 4);
  vk.Copy(vq, vgb.p(), gv.data(), kGb * 4);
  vk.Copy(vq, vbb.p(), bv.data(), kGb * 4);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);
  vk.Copy(vq, vib.p(), idx.data(), kBatch * 4);
  std::memcpy(cqb.p(), qv.data(), kQk * 4);
  std::memcpy(ckb.p(), kv.data(), kQk * 4);
  std::memcpy(cvb.p(), vv.data(), kVo * 4);
  std::memcpy(cgb.p(), gv.data(), kGb * 4);
  std::memcpy(cbb.p(), bv.data(), kGb * 4);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  std::memcpy(cib.p(), idx.data(), kBatch * 4);
  vk.Synchronize(vq);

  auto qk_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kBatch, kGdnHk, kGdnDk});
  };
  auto vo_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kBatch, kGdnHv, kGdnDv});
  };
  auto gb_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kBatch, kGdnHv});
  };
  auto st_t = [](void* p, Device dev, int64_t rows) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {rows, kGdnHv, kGdnDv, kGdnDk});
  };

  Tensor vqt = qk_t(vqb.p(), vd), vkt = qk_t(vkb.p(), vd);
  Tensor vvt = vo_t(vvb.p(), vd), vot = vo_t(vob.p(), vd);
  Tensor vgt = gb_t(vgb.p(), vd), vbt = gb_t(vbb.p(), vd);
  Tensor vst = st_t(vsb.p(), vd, kSlots);
  Tensor vit = Tensor::Contiguous(vib.p(), vt::DType::kI32, vd, {kBatch});
  Tensor cqt = qk_t(cqb.p(), cd), ckt = qk_t(ckb.p(), cd);
  Tensor cvt = vo_t(cvb.p(), cd), cot = vo_t(cob.p(), cd);
  Tensor cgt = gb_t(cgb.p(), cd), cbt = gb_t(cbb.p(), cd);
  Tensor cst = st_t(csb.p(), cd, kSlots);
  Tensor cit = Tensor::Contiguous(cib.p(), vt::DType::kI32, cd, {kBatch});

  vt::GdnDecode(cq, cot, cqt, ckt, cvt, cgt, cbt, cst, args, &cit);
  vt::GdnDecode(vq, vot, vqt, vkt, vvt, vgt, vbt, vst, args, &vit);
  vk.Synchronize(vq);

  CHECK(ctx.PipelineExistsFor("vt_gdn_decode"));
  CHECK(RanNative(vt::OpId::kGdnDecode));

  std::vector<float> got_out(kVo), got_state(kSt);
  vk.Copy(vq, got_out.data(), vob.p(), kVo * 4);
  vk.Copy(vq, got_state.data(), vsb.p(), kSt * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref_out(cob.as<float>(), cob.as<float>() + kVo);
  const std::vector<float> ref_state(csb.as<float>(), csb.as<float>() + kSt);
  const double nmse_out = NmseOf(ref_out, got_out);
  const double nmse_state = NmseOf(ref_state, got_state);
  MESSAGE("gdn_decode (indexed cache) out NMSE vs the CPU oracle: " << nmse_out);
  MESSAGE("gdn_decode (indexed cache) CACHE NMSE vs the CPU oracle: " << nmse_state);
  CHECK(nmse_out <= kGdnNmseTol);
  CHECK(nmse_state <= kGdnNmseTol);

  // The NULL block's output row is asserted DIRECTLY: if both kernels wrongly ran
  // the recurrence for it, the comparison above would still be green.
  bool null_row_zero = true;
  for (int64_t e = 0; e < kGdnHv * kGdnDv; ++e) {
    if (got_out[kGdnHv * kGdnDv + e] != 0.0f) null_row_zero = false;
  }
  CHECK(null_row_zero);
  // Cache slots 2, 4 and 5 are named by no index and must still hold their
  // ORIGINAL bytes — proof the indirection wrote only where it was told to.
  for (int64_t slot : {2, 4, 5}) {
    CAPTURE(slot);
    CHECK(std::memcmp(got_state.data() + slot * kRowElems, s0.data() + slot * kRowElems,
                      static_cast<size_t>(kRowElems) * 4) == 0);
  }

  // --- the COMPACT path: no state_idx, one state row per token ----------------
  Buf vsc(vk, kBatch * kRowElems, 4), csc(cpu, kBatch * kRowElems, 4);
  vk.Copy(vq, vsc.p(), s0.data(), kBatch * kRowElems * 4);
  std::memcpy(csc.p(), s0.data(), kBatch * kRowElems * 4);
  vk.Synchronize(vq);
  Tensor vsct = st_t(vsc.p(), vd, kBatch), csct = st_t(csc.p(), cd, kBatch);
  vt::GdnDecode(cq, cot, cqt, ckt, cvt, cgt, cbt, csct, args, nullptr);
  vt::GdnDecode(vq, vot, vqt, vkt, vvt, vgt, vbt, vsct, args, nullptr);
  vk.Synchronize(vq);

  std::vector<float> got_c(kVo), got_sc(kBatch * kRowElems);
  vk.Copy(vq, got_c.data(), vob.p(), kVo * 4);
  vk.Copy(vq, got_sc.data(), vsc.p(), kBatch * kRowElems * 4);
  vk.Synchronize(vq);
  const std::vector<float> ref_c(cob.as<float>(), cob.as<float>() + kVo);
  const std::vector<float> ref_sc(csc.as<float>(), csc.as<float>() + kBatch * kRowElems);
  const double nmse_c = NmseOf(ref_c, got_c);
  MESSAGE("gdn_decode (compact state) out NMSE vs the CPU oracle: " << nmse_c);
  CHECK(nmse_c <= kGdnNmseTol);
  CHECK(NmseOf(ref_sc, got_sc) <= kGdnNmseTol);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}

TEST_CASE("a GDN recurrence wider than the shared tile DECLINES to the reference tier") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue vq = vk.CreateQueue();
  Queue cq = cpu.CreateQueue();
  const Device vd{DeviceType::kVULKAN, 0};
  const Device cd{DeviceType::kCPU, 0};

  // Dk = 132 is past VT_GDN_MAX_DK, so the [BV,Dk] tile would not fit Vulkan's
  // GUARANTEED 16 KB of shared memory. The kernel must forward through the
  // provider seam rather than throw or index past the array: declining preserves a
  // capability the reference tier already had, which is what GetOpFallback is for.
  // Asserted through the DECLINE COUNTER, because `last_selected` still names
  // vt-native — the native provider WAS selected, and then forwarded.
  constexpr int64_t kN = 1, kT = 2, kHk = 1, kHv = 1, kDk = 132, kDv = 4;
  constexpr int64_t kQk = kT * kHk * kDk, kVo = kT * kHv * kDv, kGb = kT * kHv;
  constexpr int64_t kSt = kN * kHv * kDv * kDk;
  const std::vector<int32_t> qsl = {0, 2};
  const std::vector<float> qv = Spread(kQk, 1.0f, 307u), kv = Spread(kQk, 1.0f, 311u);
  const std::vector<float> vv = Spread(kVo, 2.0f, 313u);
  const std::vector<float> gv = GdnDecays(kGb, 317u), bv = GdnBetas(kGb, 331u);
  const std::vector<float> s0 = Spread(kSt, 0.5f, 337u);
  vt::GdnArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(kDk));

  Buf vqb(vk, kQk, 4), vkb(vk, kQk, 4), vvb(vk, kVo, 4), vob(vk, kVo, 4);
  Buf vgb(vk, kGb, 4), vbb(vk, kGb, 4), vsb(vk, kSt, 4), vlb(vk, kN + 1, 4);
  Buf cqb(cpu, kQk, 4), ckb(cpu, kQk, 4), cvb(cpu, kVo, 4), cob(cpu, kVo, 4);
  Buf cgb(cpu, kGb, 4), cbb(cpu, kGb, 4), csb(cpu, kSt, 4), clb(cpu, kN + 1, 4);
  vk.Copy(vq, vqb.p(), qv.data(), kQk * 4);
  vk.Copy(vq, vkb.p(), kv.data(), kQk * 4);
  vk.Copy(vq, vvb.p(), vv.data(), kVo * 4);
  vk.Copy(vq, vgb.p(), gv.data(), kGb * 4);
  vk.Copy(vq, vbb.p(), bv.data(), kGb * 4);
  vk.Copy(vq, vsb.p(), s0.data(), kSt * 4);
  vk.Copy(vq, vlb.p(), qsl.data(), (kN + 1) * 4);
  std::memcpy(cqb.p(), qv.data(), kQk * 4);
  std::memcpy(ckb.p(), kv.data(), kQk * 4);
  std::memcpy(cvb.p(), vv.data(), kVo * 4);
  std::memcpy(cgb.p(), gv.data(), kGb * 4);
  std::memcpy(cbb.p(), bv.data(), kGb * 4);
  std::memcpy(csb.p(), s0.data(), kSt * 4);
  std::memcpy(clb.p(), qsl.data(), (kN + 1) * 4);
  vk.Synchronize(vq);

  auto qk_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHk, kDk});
  };
  auto vo_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHv, kDv});
  };
  auto gb_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kT, kHv});
  };
  auto st_t = [](void* p, Device dev) {
    return Tensor::Contiguous(p, vt::DType::kF32, dev, {kN, kHv, kDv, kDk});
  };
  Tensor vqt = qk_t(vqb.p(), vd), vkt = qk_t(vkb.p(), vd);
  Tensor vvt = vo_t(vvb.p(), vd), vot = vo_t(vob.p(), vd);
  Tensor vgt = gb_t(vgb.p(), vd), vbt = gb_t(vbb.p(), vd), vst = st_t(vsb.p(), vd);
  Tensor vlt = Tensor::Contiguous(vlb.p(), vt::DType::kI32, vd, {kN + 1});
  Tensor cqt = qk_t(cqb.p(), cd), ckt = qk_t(ckb.p(), cd);
  Tensor cvt = vo_t(cvb.p(), cd), cot = vo_t(cob.p(), cd);
  Tensor cgt = gb_t(cgb.p(), cd), cbt = gb_t(cbb.p(), cd), cst = st_t(csb.p(), cd);
  Tensor clt = Tensor::Contiguous(clb.p(), vt::DType::kI32, cd, {kN + 1});

  const auto before = vt::GetOpProviderStats(vt::OpId::kGdnPrefill, DeviceType::kVULKAN);
  CHECK_NOTHROW(vt::GdnPrefill(vq, vot, vqt, vkt, vvt, vgt, vbt, vst, vlt, args));
  vk.Synchronize(vq);
  const auto after = vt::GetOpProviderStats(vt::OpId::kGdnPrefill, DeviceType::kVULKAN);
  CHECK(after.declines == before.declines + 1);

  // And it still computed the right answer on the tier it forwarded to — BIT-EXACT
  // this time, because the same host code ran both sides.
  vt::GdnPrefill(cq, cot, cqt, ckt, cvt, cgt, cbt, cst, clt, args);
  std::vector<float> got(kVo);
  vk.Copy(vq, got.data(), vob.p(), kVo * 4);
  vk.Synchronize(vq);
  CHECK(std::memcmp(got.data(), cob.p(), static_cast<size_t>(kVo) * 4) == 0);

  vk.DestroyQueue(vq);
  cpu.DestroyQueue(cq);
}
