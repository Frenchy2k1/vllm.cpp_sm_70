// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE 27B FP8-TOWER GREEDY ACCEPTANCE GATE — GATE-27B-FP8-TOWER-GOLDEN,
// issue #466. See .agents/specs/gate-27b-fp8-tower-golden.md.
//
// A SECOND ARM, not a replacement. test_qwen27_paged_engine.cpp keeps its
// checkpoint, its goldens and its assertions untouched; this file adds the arm
// that one structurally cannot be.
//
// WHY IT EXISTS. The dense 27B gate pins `unsloth/Qwen3.6-27B-NVFP4`@`890bdef7`
// (tests/parity/hf_snapshot.h). Read off the checkpoint itself, that revision
// lists `linear_attn.in_proj_{qkv,z,a,b}` in its compressed-tensors `ignore`
// set and ships ZERO `*.input_scale` tensors: it has no FP8 W8A8 weight
// anywhere. Every fp8-tower lever — the fp8 GDN input projection and its
// merged-qkvz collapse, the fp8 out_proj, the fp8 attention qkv — is selected by
// probing an on-disk tensor dtype (`qwen3_5_dense_weights.cpp:425-432,437-447,
// 472-485`: `dtype == "F8_E4M3"`), so on that checkpoint every one of those
// branches is DEAD CODE and the gate cannot fail for an fp8 defect.
//
// The sibling gate concedes exactly this and then does nothing about it:
// test_qwen27_paged_engine.cpp:227 wraps the whole FP8 dispatch-count contract
// in `if (fp8_inproj.Total() != 0)`, which is always false there. An
// unexercised contract that prints nothing and reports success is
// indistinguishable from a satisfied one. That is the defect this file closes,
// and the single inverted guard below — Total() != 0 asserted, not assumed — is
// the whole point of the row.
//
// `nvidia/Qwen3.6-27B-NVFP4`@`0893e160` is `modelopt_mixed`: W4A16 NVFP4 MLP,
// FP8 W8A8 tower, NVFP4 head. It is the checkpoint every recorded 27B
// performance ratio was taken on. Its goldens live in
// tests/parity/goldens/qwen36_logits_27n/, captured by the SAME
// tools/parity/dump_qwen36.py recipe as the @890bdef7 goldens (same prompt,
// same 16 greedy tokens, same temperature-0 sampling, same engine knobs), from
// the pinned oracle.
//
// Checkpoint-GATED + dgx-only: on the CPU dev box / CI the snapshot is absent,
// so the body emits a refusal that CANNOT be read as coverage (see the SKIP
// block) and returns. It compiles and links everywhere; it only RUNS where the
// checkpoint is.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime_api.h>
#include "vt/cuda/cuda_gdn_internal.h"
#endif

#include "npy.h"
#include "hf_snapshot.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

// Qwen3.6-27B has 48 GDN (`linear_attention`) layers. One merged FP8 qkvz GEMM
// per layer on the default arm; two split GEMMs per layer on the
// VT_GDN_MERGED_QKVZ_FP8=0 rollback.
constexpr uint64_t kGdnLayers = 48;

// Snapshot dir of the FP8-tower 27B, or "" to refuse. Pinned to the revision
// its goldens were captured against; a cache holding some other revision of the
// same repo skips rather than being substituted (tests/parity/hf_snapshot.h).
std::string Find27nSnapshot() { return parity::Qwen27nFp8TowerSnapshot(); }

std::vector<int32_t> LoadI32Npy(const fs::path& p) {
  const parity::NpyArray a = parity::LoadNpy(p.string());
  REQUIRE(a.dtype == "<i4");
  const size_t n = a.data.size() / sizeof(int32_t);
  const auto* src = reinterpret_cast<const int32_t*>(a.data.data());
  return std::vector<int32_t>(src, src + n);
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

TEST_CASE("qwen27n fp8-tower paged-engine greedy acceptance gate (dgx-only)") {
  const std::string snap = Find27nSnapshot();

  // ---------------------------------------------------------------- THE SKIP
  //
  // A skipped arm that prints like a pass is what let the fp8 tower go
  // ungated for an entire campaign, so this refusal is built to be unmistakable
  // three separate ways:
  //
  //   1. A banner carrying the literal token NO-FP8-TOWER-COVERAGE, the repo,
  //      and the exact revision. It says what was NOT measured, not that
  //      something was skipped.
  //   2. Exactly ONE assertion is recorded on this path, against many on the
  //      covered path. Assertion COUNT is the signal a gate reader actually
  //      diffs, and a changed count is the house RED signal
  //      (.agents/verification.md); a run that reports one assertion for this
  //      case measured no tokens, full stop.
  //   3. VT_REQUIRE_27N_FP8_GATE=1 turns absence into a hard FAILURE. Any
  //      harness that intends this gate as evidence sets it, and then the
  //      checkpoint being absent can never be recorded as a pass.
  //
  // The proof line at the bottom is printed ONLY after tokens have been
  // compared, mirroring the `MODEL_GATE_CONTRACTS` proof discipline in
  // tools/bench/online_gate.py: `ctest` exit 0 alone says nothing about whether
  // anything was compared.
  if (snap.empty()) {
    const char* required = std::getenv("VT_REQUIRE_27N_FP8_GATE");
    MESSAGE(
        "\n"
        "======================================================================\n"
        "  NO-FP8-TOWER-COVERAGE — THIS GATE MEASURED NOTHING\n"
        "  nvidia/Qwen3.6-27B-NVFP4 @0893e1606ff3d5f97a441f405d5fc541a6bdf404\n"
        "  is not cached under $HOME/.cache/huggingface/hub (or the\n"
        "  VT_QWEN27N_SNAPSHOT override names no config.json).\n"
        "  NO fp8 W8A8 GDN input projection, out_proj or attention qkv was\n"
        "  executed, and NO token was compared. test_qwen27_paged_engine's\n"
        "  235/235 does NOT cover this path: its checkpoint has no fp8 tensor.\n"
        "  Set VT_REQUIRE_27N_FP8_GATE=1 to make this absence a FAILURE.\n"
        "======================================================================");
    if (required != nullptr && required[0] == '1' && required[1] == '\0') {
      throw std::runtime_error(
          "VT_REQUIRE_27N_FP8_GATE=1 but the FP8-tower 27B checkpoint "
          "(nvidia/Qwen3.6-27B-NVFP4@0893e160) is absent: this run has NO fp8 "
          "tower coverage and must not be recorded as one");
    }
    // The single sentinel assertion of the refusal path (see 2 above).
    CHECK(snap.empty());
    return;
  }

  // BUILD-CONFIGURATION PRECONDITION, mirroring the sibling gate. A gate
  // configured without the production kernel set measures the build, not the
  // code, and has voided work on this box three times (.agents/environment.md).
#ifdef VLLM_CPP_CUDA
  constexpr bool kCutlassNvfp4Compiled =
#ifdef VT_CUTLASS_NVFP4
      true;
#else
      false;
#endif
  constexpr bool kTritonAotCompiled =
#ifdef VLLM_CPP_TRITON
      true;
#else
      false;
#endif
  if (!kCutlassNvfp4Compiled || !kTritonAotCompiled) {
    throw std::runtime_error(
        "qwen27n fp8-tower gate requires the production kernel build: "
        "configure with -DVLLM_CPP_CUTLASS_DIR=<cutlass 4.5.0+> (defines "
        "VT_CUTLASS_NVFP4) AND -DVLLM_CPP_TRITON=ON (vendored Triton-AOT GDN).");
  }
#endif

  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_27n";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / "token_ids.npy");
  const std::vector<int32_t> want_prod = LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_prod.size());  // 16
  REQUIRE(kMaxTokens > 0);

  // No `greedy_ids_emulation` negative control on this arm. That fixture exists
  // on the @890bdef7 arm for a documented W4A4 tok6 whitespace near-tie between
  // the production and emulation NVFP4 kernels, and
  // tools/parity/dump_27b_emulation_greedy.py hardcodes EXPECT_TOK6 = 271 for
  // that checkpoint. @0893e160 is W4A16 with an FP8 tower — a different kernel
  // set and a different near-tie question — so importing that fixture would
  // assert a property nobody measured. Recorded as a limitation in the spec,
  // not silently reused.

  MESSAGE("qwen27n_fp8_tower: loading nvidia/Qwen3.6-27B-NVFP4@0893e160 via "
          "FromModelDir(" << snap << ") — modelopt_mixed, FP8 W8A8 tower...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

#ifdef VLLM_CPP_CUDA
  REQUIRE(loaded->runner().kv_cache_backend_resident());
  const auto check_device_pointer = [](const void* pointer) {
    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
    CHECK(attributes.type == cudaMemoryTypeDevice);
  };
  for (const vllm::PagedKvCache& cache : loaded->runner().attn_kv())
    check_device_pointer(cache.data);
  for (const vllm::GdnStateCache& cache : loaded->runner().gdn_state()) {
    check_device_pointer(cache.ssm_state.data);
    check_device_pointer(cache.conv_state.data);
  }
#endif

  MESSAGE("qwen27n_fp8_tower: greedy-decoding " << kMaxTokens
          << " tokens through the PAGED dense engine...");
  loaded->engine().add_request("gate", kPrompt, Greedy(kMaxTokens));
  std::optional<vllm::RequestOutput> final;
  auto consume = [&](std::vector<vllm::RequestOutput> batch) {
    for (vllm::RequestOutput& item : batch)
      if (item.finished) final = std::move(item);
  };

  consume(loaded->engine().step());  // prefill

#ifdef VLLM_CPP_CUDA
  vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
  vllm::detail::ResetGdnFp8InProjDebugStats();
#endif
  consume(loaded->engine().step());  // one-token pure non-spec decode
#ifdef VLLM_CPP_CUDA
  const vllm::detail::GdnFp8InProjDebugStats fp8_inproj =
      vllm::detail::GetGdnFp8InProjDebugStats();
  const uint64_t packed_launches =
      vt::cuda::testing::GetGdnPackedDecodeDebugStats().launches;
  vllm::detail::DisableGdnFp8InProjDebugStats();
  vt::cuda::testing::DisableGdnPackedDecodeDebugStats();

  // ---- THE INVERTED GUARD. The sibling gate asks `if (Total() != 0)` and so
  // never runs this contract; here a ZERO total is the FAILURE, because zero is
  // the exact signature of gating a checkpoint whose tower is BF16 — which is
  // how this hole stayed open. Assert it before the shape, so a wrong
  // checkpoint fails on WHAT IT IS rather than on a count that happens to
  // differ.
  if (fp8_inproj.Total() == 0) {
    throw std::runtime_error(
        "qwen27n fp8-tower gate executed ZERO fp8 GDN input projections: the "
        "loaded checkpoint has no FP8 W8A8 tower, so this run covers nothing "
        "it claims to. Expected nvidia/Qwen3.6-27B-NVFP4@0893e160 "
        "(modelopt_mixed); a BF16-owner checkpoint such as "
        "unsloth/Qwen3.6-27B-NVFP4@890bdef7 totals 0 here.");
  }

  // PERF-27B-GDN-FP8-QKVZ structural contract, now on a checkpoint that can
  // actually run it: vLLM issues ONE merged qkvz GEMM per GDN layer, so the
  // default arm must be exactly 48 merged + 0 split, and the
  // VT_GDN_MERGED_QKVZ_FP8=0 rollback exactly 0 merged + 96 split.
  const bool fp8_merged = vllm::detail::MergedGdnFp8QkvzEnvSelected(
      vllm::detail::GdnMergedFp8QkvzEnvConfig{
          std::getenv("VT_GDN_MERGED_PROJ"),
          std::getenv("VT_GDN_MERGED_QKVZ"),
          std::getenv("VT_GDN_MERGED_QKVZ_FP8")});
  const uint64_t expected_merged = fp8_merged ? kGdnLayers : 0U;
  const uint64_t expected_split = fp8_merged ? 0U : 2U * kGdnLayers;
  CHECK(fp8_inproj.merged_launches == expected_merged);
  CHECK(fp8_inproj.split_launches == expected_split);
  if (fp8_inproj.merged_launches != expected_merged ||
      fp8_inproj.split_launches != expected_split) {
    throw std::runtime_error(
        "qwen27n GDN fp8 input-projection dispatch count mismatch");
  }

  // Packed GDN decode is NOT selectable on an FP8 tower and must issue ZERO
  // launches on every arm. `ShouldUsePackedGdnDecode`'s dtype_compatible term
  // requires `in_proj_qkv_fp8.Empty() && in_proj_z_fp8.Empty()`
  // (qwen3_5.cpp:4161-4165) because vt::GdnPackedDecode rejects fp8 shards.
  //
  // Deliberately NOT expressed through `detail::PackedGdnDecodeEnvSelected`,
  // which the sibling gate uses: that helper mirrors the ENV couplings only and
  // knows nothing about the weight dtype, so on this checkpoint it would demand
  // 48 and throw for the wrong reason. The env mirror's divergence from the
  // real predicate is a separate finding; this arm pins the real behaviour.
  CHECK(packed_launches == 0U);
  if (packed_launches != 0U) {
    throw std::runtime_error(
        "qwen27n packed GDN decode selected on an FP8 tower (expected 0)");
  }
#endif

  while (loaded->engine().has_unfinished_requests())
    consume(loaded->engine().step());
  if (!final.has_value())
    throw std::runtime_error("qwen27n paged engine produced no final output");
  const vllm::RequestOutput& out = *final;

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;

  // Tokenizer diagnostic: a greedy divergence must not be a tokenizer mismatch.
  CHECK(out.prompt_token_ids == want_prompt_ids);

  MESSAGE("qwen27n_fp8_tower: produced " << got.size() << "/" << kMaxTokens
          << " tokens; continuation=\"" << out.outputs[0].text << "\"");

  REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
  CHECK(got == want_prod);

  // Printed ONLY after tokens were compared.
  MESSAGE("qwen27n_fp8_tower: FP8 tower 16/16 token-exact vs vLLM "
          "@0893e160");
}
