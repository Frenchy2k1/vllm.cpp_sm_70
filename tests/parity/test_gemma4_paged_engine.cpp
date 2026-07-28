// vllm.cpp original (Gemma-4 G1b — THE SACRED correctness gate for the first
// Gemma-4-family model); no upstream mirror.
//
// THE PAGED-ENGINE Gemma4ForConditionalGeneration (unsloth/gemma-4-E4B-it, bf16)
// GREEDY CORRECTNESS GATE. Drives the pinned golden prompt token ids through the
// FULL PAGED LLMEngine stack (InputProcessor -> Scheduler -> per-layer paged
// attention + KV growth -> Sampler -> OutputProcessor) via LoadedEngine::
// FromModelDir, and checks the greedy (temperature-0) decode against the pinned
// vLLM 0.25.0 STRICT golden (tests/parity/goldens/gemma4_e4b_text/gen_manifest.json
// ref_token_ids). The manifest is deterministic (K=5 identical runs), so the gate
// is TOKEN-EXACT: our 32 generated ids MUST equal ref_token_ids exactly.
//
// This is the RED->GREEN proof for the G1b runner per-layer-KV change: Gemma-4's
// heterogeneous per-layer head_dim (256 sliding / 512 full) means the runner must
// allocate a per-layer KV cache. Before G1b the forward's
// VT_CHECK(kv.head_size == Dh) aborted on the first full-attention layer; after
// G1b the whole stack runs and this gate can pass. A first-divergence index +
// value is printed on any mismatch (the honest divergence-map outcome).
//
// Checkpoint-GATED + dgx-only: resolves ~/.cache/huggingface/hub/models--unsloth
// --gemma-4-E4B-it. On CPU/CI the snapshot is absent, so the case emits a loud
// SKIP and returns (compiles + links on CPU, RUNS only on dgx).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

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

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

TEST_CASE("gemma4-e4b-text-strict-greedy-gate") {
  const std::string snap = FindSnapshot("models--unsloth--gemma-4-E4B-it");
  if (snap.empty()) {
    MESSAGE("gemma4-E4B checkpoint absent; skipping (dgx-only) — "
            "models--unsloth--gemma-4-E4B-it");
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "gemma4_e4b_text";
  const fs::path manifest = gdir / "gen_manifest.json";
  REQUIRE(fs::exists(manifest));

  nlohmann::json m;
  {
    std::ifstream f(manifest.string());
    f >> m;
  }
  REQUIRE(m.value("gate_form", "") == "STRICT");
  REQUIRE(m.value("deterministic", false));
  const int max_tokens = m.value("max_tokens", 32);
  std::vector<int32_t> prompt_ids =
      m.at("prompt_token_ids").get<std::vector<int32_t>>();
  std::vector<int32_t> ref = m.at("ref_token_ids").get<std::vector<int32_t>>();
  REQUIRE(static_cast<int>(ref.size()) == max_tokens);

  MESSAGE("gemma4-E4B: loading via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  const vllm::RequestOutput out = loaded->engine().generate(
      prompt_ids, Greedy(max_tokens), "gemma4gate");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;

  // First-divergence map (the honest partial outcome when not 32/32).
  int first_div = -1;
  const int n = static_cast<int>(std::min(got.size(), ref.size()));
  for (int j = 0; j < n; ++j) {
    if (got[static_cast<size_t>(j)] != ref[static_cast<size_t>(j)]) {
      first_div = j;
      break;
    }
  }
  if (first_div >= 0 || got.size() != ref.size()) {
    MESSAGE("gemma4-E4B STRICT DIVERGENCE: first mismatch idx="
            << first_div << " ours="
            << (first_div >= 0 ? got[static_cast<size_t>(first_div)] : -1)
            << " golden="
            << (first_div >= 0 ? ref[static_cast<size_t>(first_div)] : -1)
            << " (our_len=" << got.size() << " ref_len=" << ref.size() << ")");
  } else {
    MESSAGE("gemma4-E4B STRICT: 32/32 token-exact vs vLLM 0.25.0 golden.");
  }
  REQUIRE(static_cast<int>(got.size()) == max_tokens);
  CHECK(first_div == -1);
}
