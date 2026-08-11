// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror. The
// CLASSIC-DENSE sibling of test_qwen36_async_serving.cpp (which owns the gate
// models). Mirrors its scheduling intent — greedy served decode is
// token-deterministic regardless of async step interleave — for the shared
// pure-dense driver (Qwen3ForCausalLM and every registry that routes through
// Qwen3DenseModel / qwen3.cpp EmbedInto).
//
// THE ASYNC-SERVING CLASSIC-DENSE GREEDY GATE — the missing counterpart to
// test_qwen3_paged_engine.cpp (the SYNC SACRED gate). That gate drives the SYNC
// LLMEngine (depth-1); this one drives the ASYNC SERVING frontend
// (LoadedEngine::async_engine() -> AsyncLLM -> step_with_batch_queue, depth-2 /
// max_concurrent_batches==2), the path the production OpenAI server runs and the
// ONLY path that pipelines two steps — so it is the only path that exposes the
// ROW-SERVE-ASYNC-DENSE-MIRROR P0: classic-dense Qwen3 async batch-1 greedy decode
// nondeterministically degenerating into repeated token-0 garbage.
//
// ROOT CAUSE (the #31 class, ported). On the async path the sampled token is NOT
// written to token_ids_cpu synchronously (sample_tokens_async); the runner's device
// combine splices each decode row's real token into the DEVICE input-ids on the
// main queue while the host `token_ids` vector stays stale. Before this row,
// classic dense Qwen3's EmbedInto (qwen3.cpp) IGNORED the device ids and uploaded
// the stale host vector — racing the combine's device write (unsynchronized
// device-write/host-read). When the host read wins it embeds the zero placeholder
// -> token-0 degeneration. The gate models (qwen3_5) already consumed the device
// override via ApplyDeviceTokenIdsOverride; this row wires the identical consumer
// into the shared dense EmbedInto + publishes the override from the classic-dense
// registry forward (ForwardQwen3ForCausalLM's DeviceTokenIdsScope).
//
// WHY THE ORACLE IS THE IN-PROCESS SYNC ENGINE, not a committed strict golden. The
// classic dense checkpoints (Qwen3-0.6B/4B, bf16) are NEAR-TIE models — vLLM's own
// prefill argmax disagrees with its incremental decode token at genuine ties (see
// test_qwen3_paged_engine.cpp), so a STRICT token-exact bar against a committed
// vLLM golden is ill-posed here. But the async serving path and the sync engine run
// the IDENTICAL per-step DEVICE forward on the SAME runner: with the fix the async
// embed reads exactly the ids the sync path appends to token_ids_cpu, so async ==
// sync token-for-token (CUDA argmax is deterministic given identical inputs). The
// sync LLMEngine is race-free (it writes token_ids_cpu synchronously, so its combine
// is redundant — the SACRED gate proves it correct), making its continuation the
// exact, drift-proof, near-tie-proof anchor the async output must reproduce. The P0
// degeneration is repeated token-0 garbage, far outside any near-tie band, so a
// divergence is unambiguous.
//
// GATE POLARITY (greedy is deterministic; async interleave must not change tokens):
//   VT_ASYNC_DEVICE_MIRROR=0 (rollback: dense EmbedInto races) => RED (P0 repro),
//   default / VT_ASYNC_DEVICE_MIRROR=1 (the fix)              => GREEN (async==sync).
//
// Checkpoint-GATED + dgx-only, exactly like the SACRED gate: it resolves the real
// Qwen3-0.6B / Qwen3-4B snapshots under ~/.cache/huggingface/hub. On the CPU dev
// box / CI the snapshots are absent, so each case emits a loud SKIP and returns
// (compiles + links on CPU, but only RUNS on dgx.casa GB10, where the CUDA forward
// + real GPU overlap the bug needs exist).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/async_llm.h"

namespace fs = std::filesystem;

namespace {

// Greedy (argmax) sampling params — temperature 0 => deterministic. Identical to
// the SACRED gate's Greedy().
vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// Resolve the newest HF snapshot dir for a "models--<repo>" cache entry, or "".
// IDENTICAL resolution to test_qwen3_paged_engine.cpp's FindSnapshot.
std::string FindSnapshot(const std::string& repo_dir) {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) / ".cache/huggingface/hub" / repo_dir / "snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}

// The classic-dense async-serving gate. Loads the checkpoint ONCE, computes the
// race-free SYNC continuation as the anchor, then requires every async batch-1 rep
// and every concurrent request to reproduce it token-for-token.
void RunAsyncGate(const std::string& repo_dir, const std::string& label) {
  const std::string snap = FindSnapshot(repo_dir);
  if (snap.empty()) {
    MESSAGE(label << " checkpoint absent; skipping (dgx-only) — " << repo_dir
            << " snapshot not present. The P0 needs the CUDA forward + real GPU "
               "overlap; a CPU-only run cannot reproduce it (the eager backend "
               "serializes the queue), so this trivially passes.");
    return;
  }

  // Prompt long enough that the served decode runs many steps (the race is per
  // decode step). A short greedy continuation exposes any degeneration immediately.
  const std::string kPrompt = "The capital of France is Paris, and the";
  constexpr int kMaxTokens = 24;

  MESSAGE(label << ": loading via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  // The gate is only meaningful when the depth-2 async serving path is engaged —
  // the path that pipelines two steps and exposes the P0. If async scheduling
  // resolved OFF (VT_ASYNC_RUNNER=0 / VT_ASYNC_SCHED=0) the frontend runs depth-1
  // and the race cannot occur, so a pass would be vacuous. Fail loud instead.
  REQUIRE(loaded->async_scheduling_enabled());
  REQUIRE(loaded->max_concurrent_batches() == 2);

  // ── THE ANCHOR: the race-free SYNC engine continuation ────────────────────────
  // Driven to completion BEFORE the async frontend threads start, so the shared
  // scheduler/executor is idle when async takes over. The sync LLMEngine writes
  // token_ids_cpu synchronously (its combine is redundant), so this is the correct,
  // drift-proof, near-tie-proof reference the async output must match.
  const vllm::RequestOutput sync_out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "sync-anchor");
  REQUIRE(sync_out.finished);
  REQUIRE(sync_out.outputs.size() == 1);
  const std::vector<int32_t> want = sync_out.outputs[0].token_ids;
  REQUIRE(static_cast<int>(want.size()) == kMaxTokens);
  MESSAGE(label << ": sync anchor continuation=\"" << sync_out.outputs[0].text
          << "\" (" << want.size() << " tokens)");

  vllm::v1::AsyncLLM& aengine = loaded->async_engine();

  // ── ARM 1: BATCH-1 (single-request) served greedy decode ──────────────────────
  // THE P0. Each independent single-request generation MUST reproduce the sync
  // anchor token-for-token. Several reps because the bug is nondeterministic (the
  // host<->combine race is a coin flip per decode step); on the buggy (mirror-OFF)
  // path the first generated token degenerates and stays there, so any one rep
  // diverging fails the gate. On the fixed (mirror-ON) path every rep is byte-exact.
  constexpr int kBatch1Reps = 5;
  for (int r = 0; r < kBatch1Reps; ++r) {
    const std::string id = "b1-r" + std::to_string(r);
    const vllm::RequestOutput out =
        aengine.generate(kPrompt, Greedy(kMaxTokens), id);
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    MESSAGE(label << "[batch1 rep " << r << "]: produced " << got.size() << "/"
            << kMaxTokens << " tokens; continuation=\"" << out.outputs[0].text
            << "\"");
    REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
    CHECK(got == want);
  }

  // ── ARM 2: CONCURRENCY bracket ────────────────────────────────────────────────
  // N independent greedy requests submitted together so the engine runs them as
  // pure-decode batched steps (num_reqs==N) while the depth-2 loop pipelines. Each
  // request is independent (own KV state), so each MUST reproduce the same sync
  // anchor regardless of the step interleave (vLLM's greedy determinism guarantee).
  int kN = 4;
  if (const char* c = std::getenv("VT_ASYNC_SERVING_CONC")) {
    const int v = std::atoi(c);
    if (v > 0) kN = v;
  }
  MESSAGE(label << ": concurrency bracket N=" << kN);
  std::vector<vllm::v1::AsyncRequest> reqs;
  reqs.reserve(static_cast<size_t>(kN));
  for (int i = 0; i < kN; ++i) {
    reqs.push_back(aengine.add_request("c" + std::to_string(i), kPrompt,
                                       Greedy(kMaxTokens)));
  }
  // Drain each request to its TERMINAL output. Blocking on request i does NOT
  // serialize the engine: all kN were enqueued first and the engine thread steps
  // them concurrently, so requests j!=i keep decoding while we collect i.
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
    MESSAGE(label << "[conc " << i << "]: produced " << got.size() << "/"
            << kMaxTokens << " tokens; continuation=\"" << out.outputs[0].text
            << "\"");
    REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
    CHECK(got == want);
  }

  aengine.shutdown();
}

}  // namespace

// Qwen3-0.6B (dense) — the primary classic-dense P0 vehicle (smallest/fastest).
TEST_CASE("qwen3-0.6B dense async-serving greedy token-exact gate (dgx-only) — "
          "ROW-SERVE-ASYNC-DENSE-MIRROR") {
  RunAsyncGate("models--Qwen--Qwen3-0.6B", "qwen3-0.6B");
}

// Qwen3-4B (dense) — the bigger-model confirmation (36 layers, GQA 32/8), same
// shared driver, same async serving path.
TEST_CASE("qwen3-4B dense async-serving greedy token-exact gate (dgx-only) — "
          "ROW-SERVE-ASYNC-DENSE-MIRROR") {
  RunAsyncGate("models--Qwen--Qwen3-4B", "qwen3-4B");
}

// ─── Sibling families — the #323 regression gates ────────────────────────────
// Llama / Mistral / InternLM2 are all `using <X>Model = Qwen3DenseModel`, so they
// share this exact forward. These three were RED until the decode-graph decline
// (#323): the graph replayed against the stale HOST token_ids, so every
// concurrent request past slot 0 degenerated. Mistral and InternLM2 reproduced
// it; Llama did not, purely because it did not engage the graph in this battery.
// Keep all three: the failure is a property of the shared path, not of a family.
TEST_CASE("llama-3.2-1B dense async-serving greedy token-exact gate (dgx-only) — #323") {
  RunAsyncGate("models--meta-llama--Llama-3.2-1B", "llama-3.2-1B");
}
TEST_CASE("mistral-7B-v0.3 dense async-serving greedy token-exact gate (dgx-only) — #323") {
  RunAsyncGate("models--mistralai--Mistral-7B-v0.3", "mistral-7B-v0.3");
}
TEST_CASE("internlm2-chat-1.8B dense async-serving greedy token-exact gate (dgx-only) — #323") {
  RunAsyncGate("models--internlm--internlm2-chat-1_8b", "internlm2-chat-1.8B");
}

// ─── Sibling families (ROW-SERVE-ASYNC-DENSE-MIRROR sibling scope) ───────────
// Llama / Mistral / InternLM2 each have their OWN Forward*ForCausalLM entry but
// route through the SAME shared EmbedInto, so the device token-ids mirror has to
// be established per entry point. It was NOT, until the sibling scope landed:
// the stale host token_ids raced the combine's device write exactly as in #31.
// These cases are the regression that catches a future entry point being added
// without the guard — the previous coverage was Qwen3-only, so the three
// siblings could regress silently.

TEST_CASE("llama-3.2-1B dense async-serving greedy token-exact gate (dgx-only) — "
          "ROW-SERVE-ASYNC-DENSE-MIRROR sibling scope") {
  RunAsyncGate("models--meta-llama--Llama-3.2-1B", "llama-3.2-1B");
}

TEST_CASE("mistral-7B-v0.3 dense async-serving greedy token-exact gate (dgx-only) — "
          "ROW-SERVE-ASYNC-DENSE-MIRROR sibling scope") {
  RunAsyncGate("models--mistralai--Mistral-7B-v0.3", "mistral-7B-v0.3");
}

TEST_CASE("internlm2-chat-1.8B dense async-serving greedy token-exact gate (dgx-only) — "
          "ROW-SERVE-ASYNC-DENSE-MIRROR sibling scope") {
  RunAsyncGate("models--internlm--internlm2-chat-1_8b", "internlm2-chat-1.8B");
}
