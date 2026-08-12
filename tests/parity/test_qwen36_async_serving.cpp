// vllm.cpp original (checkpoint-gated acceptance gate); mirrors the scheduling
// intent of vllm/v1/engine/test_async_llm.py (greedy served decode is
// token-deterministic regardless of step interleave) against the SAME pinned 35B
// oracle continuation the SACRED sync gate (test_qwen36_paged_engine.cpp) uses.
//
// THE ASYNC-SERVING 35B GREEDY TOKEN-EXACT GATE — the missing counterpart to
// test_qwen36_paged_engine.cpp. That gate drives the SYNC LLMEngine
// (LLMEngine::step -> EngineCore::step, depth-1); this one drives the ASYNC
// SERVING frontend (LoadedEngine::async_engine() -> AsyncLLM ->
// EngineCoreProc::step_with_batch_queue, depth-2 / max_concurrent_batches==2),
// which is the path the production OpenAI server runs and the ONLY path that
// pipelines two steps (async_forward_in_flight_) — so it is the only path that
// exposes the ROW-SERVE-ASYNC-LLM P0: async batch-1 greedy decode
// nondeterministically degenerating into repeated token-0 garbage.
//
// WHY THE SYNC SACRED GATE MISSES IT (root cause, recorded 2026-08-06): on the
// async path sample_tokens_async DELETES the synchronous token_ids_cpu write-back
// (runner.cpp: "The token VALUE append to token_ids_cpu ... is DELETED on the
// async path"), so the next step's prepare_inputs reads a STALE/zero placeholder
// for every decode row and RELIES on the device combine splicing the real token
// from last_sampled_tokens. On the DEFAULT (VT_ASYNC_DEVICE_MIRROR=0) integrated
// path the combine patches step.input_token_ids on the MAIN QUEUE while the
// Qwen3.5 decode graph reads that same host vector on the CPU (BuildPaddedDecode
// -> CopyInPlace -> EmbedInto's host->device upload) WITHOUT an intervening
// sync — an unsynchronized device-write/host-read race. When the CPU wins, it
// embeds the zero placeholder -> token-0 degeneration. The sync LLMEngine never
// pipelines, so sample_tokens writes token_ids_cpu synchronously and the combine
// is redundant, hiding the race. The device-resident mirror
// (VT_ASYNC_DEVICE_MIRROR=1) routes device_token_ids into the embed via
// ApplyDeviceTokenIdsOverride (main-queue-ordered after the combine), so the
// embed never does the racing host read — deterministic, coherent, token-exact.
//
// GATE POLARITY (greedy is deterministic; vLLM guarantees the same tokens for a
// request regardless of async step interleave, async_llm.py):
//   VT_ASYNC_DEVICE_MIRROR=0 (pre-fix production default) => RED (P0 repro),
//   default / VT_ASYNC_DEVICE_MIRROR=1 (the fix)          => GREEN (token-exact).
//
// Checkpoint-GATED + dgx-only, exactly like the SACRED gate: it resolves the real
// nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot; on the CPU dev box / CI the snapshot is
// absent, so the body emits a loud SKIP and returns (compiles + links on CPU, but
// only RUNS on dgx.casa GB10, where the CUDA decode-graph + real GPU overlap the
// bug needs exist). scripts/dgx-bringup.sh invokes it under both env arms.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/async_llm.h"

#include "hf_snapshot.h"

namespace fs = std::filesystem;

namespace {

// Load an i32 (.npy "<i4") vector from the committed golden.
std::vector<int32_t> LoadI32Npy(const fs::path& p) {
  const parity::NpyArray a = parity::LoadNpy(p.string());
  REQUIRE(a.dtype == "<i4");
  const size_t n = a.data.size() / sizeof(int32_t);
  const auto* src = reinterpret_cast<const int32_t*>(a.data.data());
  return std::vector<int32_t>(src, src + n);
}

// Greedy (argmax) sampling params — temperature 0 => deterministic, matching the
// oracle's temperature-0 continuation. Identical to the SACRED gate's Greedy().
vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// Snapshot dir of the pinned 35B checkpoint (contains config.json), or "".
// IDENTICAL resolution to test_qwen36_paged_engine.cpp's Find35BSnapshot.
// GATE-PIN-UNPINNED-SNAPSHOTS (#471). This used to take the first
// `directory_iterator` entry under `<repo>/snapshots/`. The 35B goldens name the
// revision they were captured against (`oracle.model` of
// goldens/qwen36_*_35b/manifest.json), so there was never a reason not to
// enforce it. Now pinned; a cache holding another revision skips.
std::string Find35BSnapshot() { return parity::Qwen36A3bNvfP4Snapshot(); }

}  // namespace

// The M0-exit prompt + its oracle greedy continuation (the SAME golden the SACRED
// sync gate reproduces). Greedy is deterministic and vLLM produces the identical
// tokens on the async serving path — so this golden IS the SYNC-engine anchor the
// async output must match.
TEST_CASE("qwen36 async-serving greedy token-exact gate (dgx-only, 35B) — "
          "ROW-SERVE-ASYNC-LLM P0") {
  const std::string snap = Find35BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "35B checkpoint absent; skipping (dgx-only) — "
        "nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot not present. The P0 needs the "
        "CUDA decode-graph + real GPU overlap; a CPU-only run cannot reproduce "
        "it (the eager backend serializes the queue), so this trivially passes.");
    return;
  }

  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_35b";
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16

  MESSAGE("qwen36_async_serving: loading full 35B via FromModelDir(" << snap
          << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  // The gate is only meaningful when the depth-2 async serving path is engaged —
  // that is the path that pipelines two steps (async_forward_in_flight_) and
  // exposes the P0. If async scheduling resolved OFF (VT_ASYNC_RUNNER=0 /
  // VT_ASYNC_SCHED=0) the frontend would run depth-1 and the race cannot occur,
  // so a pass would be vacuous. Fail loud instead of silently passing.
  REQUIRE(loaded->async_scheduling_enabled());
  REQUIRE(loaded->max_concurrent_batches() == 2);

  vllm::v1::AsyncLLM& aengine = loaded->async_engine();

  // ── ARM 1: BATCH-1 (single-request) served greedy decode ──────────────────
  // THE P0. Each independent single-request generation MUST reproduce the oracle
  // continuation token-for-token. Several reps because the bug is nondeterministic
  // (the host<->combine race is a coin flip per decode step); on the buggy
  // (mirror-OFF) path the first generated token degenerates to ~0 and stays there,
  // so any one rep diverging fails the gate. On the fixed (mirror-ON) path greedy
  // is deterministic and every rep is byte-exact.
  constexpr int kBatch1Reps = 5;
  for (int r = 0; r < kBatch1Reps; ++r) {
    const std::string id = "b1-r" + std::to_string(r);
    MESSAGE("qwen36_async_serving[batch1 rep " << r << "]: served greedy-decoding "
            << kMaxTokens << " tokens through AsyncLLM (depth-2)...");
    const vllm::RequestOutput out =
        aengine.generate(kPrompt, Greedy(kMaxTokens), id);
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    MESSAGE("qwen36_async_serving[batch1 rep " << r << "]: produced "
            << got.size() << "/" << kMaxTokens << " tokens; continuation=\""
            << out.outputs[0].text << "\"");
    REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
    // TOKEN-EXACT vs the SACRED sync oracle continuation.
    CHECK(got == want_greedy_ids);
  }

  // ── ARM 2: CONCURRENCY bracket ────────────────────────────────────────────
  // N independent greedy requests submitted together so the engine runs them as
  // pure-decode batched steps (num_reqs==N) through the batched decode graph +
  // async combine while the depth-2 loop pipelines. Each request is independent
  // (own KV + GDN state), so each MUST reproduce the same oracle continuation
  // regardless of the step interleave (vLLM's greedy determinism guarantee).
  //
  // Default N=4 is a SMALL bracket. Under VT_ASYNC_EXECUTOR (Option A) the runner
  // skips the depth-2 drain and the decode-graph stages each step's input H2D OUT of
  // the captured replay into persistent device buffers, guarded by an input-staged
  // event; the next same-slot Refresh waits only that tiny copy. Higher concurrency
  // (VT_ASYNC_SERVING_CONC) makes the batched replay heavy enough that the host runs
  // ahead — the regime the input-staged event must cover. The DETERMINISTIC RED is
  // VT_ASYNC_EXECUTOR_POISON=1, which overwrites the PINNED H2D source immediately
  // after the async copy is enqueued (a true-async DMA reads the garbage): if the
  // event boundary were unsound this is exactly what a Refresh that ran ahead would
  // do, so the gate must FAIL. GREEN (VT_ASYNC_EXECUTOR=1, no poison) => token-exact.
  int kN = 4;
  if (const char* c = std::getenv("VT_ASYNC_SERVING_CONC")) {
    const int v = std::atoi(c);
    if (v > 0) kN = v;
  }
  MESSAGE("qwen36_async_serving: concurrency bracket N=" << kN);
  std::vector<vllm::v1::AsyncRequest> reqs;
  reqs.reserve(static_cast<size_t>(kN));
  for (int i = 0; i < kN; ++i) {
    reqs.push_back(aengine.add_request("c" + std::to_string(i), kPrompt,
                                       Greedy(kMaxTokens)));
  }
  // Drain each request to its TERMINAL output (get_output returns one coalesced,
  // possibly non-final, RequestOutput — generate() loops until finished; mirror
  // that here). Blocking on request i does NOT serialize the engine: all kN were
  // enqueued first and the engine thread steps them concurrently, so requests j!=i
  // keep decoding (and coalescing their terminal outputs) while we collect i.
  for (int i = 0; i < kN; ++i) {
    const vllm::v1::AsyncRequest& req = reqs[static_cast<size_t>(i)];
    vllm::RequestOutput out;
    for (;;) {
      std::optional<vllm::RequestOutput> ready = aengine.get_output_nowait(req);
      out = ready.has_value() ? std::move(*ready) : aengine.get_output(req);
      if (out.finished) break;
    }
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    MESSAGE("qwen36_async_serving[conc " << i << "]: produced " << got.size()
            << "/" << kMaxTokens << " tokens; continuation=\""
            << out.outputs[0].text << "\"");
    REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
    CHECK(got == want_greedy_ids);
  }

  aengine.shutdown();
}
