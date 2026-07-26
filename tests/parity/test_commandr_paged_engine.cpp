// vllm.cpp original (Command-R / Cohere W4 — THE SACRED correctness gate for the
// first CohereForCausalLM model); no upstream mirror (upstream's Command-R coverage
// is the generic text-generation entry — spec §"Tests to port").
//
// THE PAGED-ENGINE CohereForCausalLM GREEDY CORRECTNESS GATE. Drives the standard
// prompt battery through the FULL PAGED LLMEngine stack via
// LoadedEngine::FromModelDir and checks the greedy decode against the pinned vLLM
// 0.25.0 oracle. This proves the ZERO-NEW-KERNEL Command-R deltas are wired
// correctly:
//   * PARALLEL residual (commandr.py:257-273): ONE weight-only LayerNorm feeds BOTH
//     attention and MLP, and h = residual + attn + mlp. Wiring the two blocks
//     sequentially (Llama-style, attn then a second norm then mlp) — the OPT
//     silent-corruption mode — emits FLUENT-WRONG tokens far outside the near-tie
//     band, which this gate FAILS on.
//   * logit_scale scalar (commandr.py:376): logits *= config.logit_scale before
//     sampling. Dropping the scale rescales every logit; on near-ties it flips the
//     greedy argmax and diverges (RED-tested).
//   * weight-only Cohere LayerNorm (mean-centred, NO bias) vs RMSNorm, GPT-J
//     (is_neox_style=False) full-width rotary, tied embeddings.
//
// GATE FORM (selected BY MEASUREMENT, near-tie-distributional methodology): capture
// vLLM's per-prompt (batch=1) greedy over K>=3 — deterministic ⇒ STRICT token-exact;
// else the ratified near-tie ROOT-divergence gate (a token OUTSIDE vLLM's
// teacher-forced band, gap > kNearTieMnats on OUR prefix, is a REAL forward
// divergence the gate FAILS on). Mirrors the stablelm/internlm2/phi3 gates.
//
// Goldens (tests/parity/goldens/commandr_greedy/, captured on dgx):
//   greedy_ids.npy / greedy_dist.npy / our_ids.npy / neartie_gap_mnats.npy.
// Checkpoint-GATED + dgx-only: resolves a CohereForCausalLM snapshot under
// ~/.cache/huggingface/hub/. On CPU/CI, or when the (HF-gated) checkpoint / the
// goldens are absent, it emits a loud SKIP.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vt/ops.h"

namespace fs = std::filesystem;

namespace {

constexpr int32_t kNearTieMnats = 500;

const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "Once upon a time,",
      "In the beginning God created",
      "The quick brown fox jumps over",
      "def fibonacci(n):",
      "Water boils at a temperature of",
      "The theory of relativity was developed by",
      "To be or not to be, that is",
      "The largest planet in our solar system is",
      "Machine learning is a subfield of",
      "The mitochondria is the powerhouse of",
      "Roses are red, violets are",
      "The first president of the United States was",
      "E equals m c",
      "A journey of a thousand miles begins with",
      "The chemical symbol for gold is",
  };
  return p;
}

std::string FindSnapshot(const std::vector<std::string>& repo_dirs) {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  for (const std::string& repo_dir : repo_dirs) {
    const fs::path snaps =
        fs::path(home) / ".cache/huggingface/hub" / repo_dir / "snapshots";
    std::error_code ec;
    if (!fs::is_directory(snaps, ec)) continue;
    for (const auto& e : fs::directory_iterator(snaps, ec)) {
      if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
    }
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

const int32_t* AsI32(const parity::NpyArray& a) {
  return reinterpret_cast<const int32_t*>(a.data.data());
}

void RunGate(const std::vector<std::string>& repo_dirs,
             const std::string& golden_subdir, const char* label) {
  const std::string snap = FindSnapshot(repo_dirs);
  if (snap.empty()) {
    MESSAGE(label << " checkpoint absent; skipping (dgx-only, HF-gated) — "
            << repo_dirs.front());
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE(label << " greedy golden absent; skipping — capture on dgx: "
            "commandr-oracle-capture.py --per-prompt");
    return;
  }
  if (dump && !have_gap) {
    MESSAGE(label << ": BOOTSTRAP dump (gap golden absent)...");
    std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
        vllm::entrypoints::LoadedEngine::FromModelDir(
            snap, vllm::entrypoints::EngineParams{});
    const parity::NpyArray gg = parity::LoadNpy((gdir / "greedy_ids.npy").string());
    const int64_t NN = gg.shape[0], TT = gg.shape[1];
    std::vector<int32_t> buf(static_cast<size_t>(NN * TT), -1);
    for (int64_t i = 0; i < NN; ++i) {
      const vllm::RequestOutput out = le->engine().generate(
          Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(TT)),
          "boot" + std::to_string(i));
      const std::vector<int32_t>& got = out.outputs[0].token_ids;
      for (int64_t j = 0; j < TT && j < static_cast<int64_t>(got.size()); ++j)
        buf[static_cast<size_t>(i * TT + j)] = got[static_cast<size_t>(j)];
    }
    const std::string path = (gdir / "our_ids.i32").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) { std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f); std::fclose(f); }
    MESSAGE(label << " BOOTSTRAP dumped our token ids -> " << path);
    return;
  }
  if (!have_gap) {
    MESSAGE(label << " gap goldens absent; skipping — run under VT_DUMP_IDS=1 "
            "then commandr-neartie-gap.py");
    return;
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap = parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(o.dtype == "<i4");
  REQUIRE(gap.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];
  REQUIRE(o.shape.size() == 2);
  REQUIRE(o.shape[0] == N);
  REQUIRE(o.shape[1] == T);
  REQUIRE(gap.shape.size() == 2);
  REQUIRE(gap.shape[0] == N);
  REQUIRE(gap.shape[1] == T);
  REQUIRE(static_cast<size_t>(N) == Prompts().size());
  const int32_t* gd = AsI32(g);
  const int32_t* od = AsI32(o);
  const int32_t* gapd = AsI32(gap);

  std::vector<int32_t> our_dump;
  if (dump) our_dump.assign(static_cast<size_t>(N * T), -1);

  MESSAGE(label << ": loading via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  int strict_exact = 0;
  int neartie_only = 0;
  int fail = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "gate" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    if (dump) {
      for (int64_t j = 0; j < T; ++j)
        our_dump[static_cast<size_t>(i * T + j)] = got[static_cast<size_t>(j)];
    }

    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != od[i * T + j]) { first_div = static_cast<int>(j); break; }
    }
    REQUIRE_MESSAGE(first_div < 0,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                    << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                    << " committed anchor=" << (first_div < 0 ? -1 : od[i * T + first_div])
                    << " — re-run commandr-neartie-gap.py to refresh the gap golden");

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) { worst_gap = mn; worst_i = static_cast<int>(i); worst_j = static_cast<int>(j); }
      if (mn > kNearTieMnats) { prompt_ok = false; if (first_bad < 0) first_bad = static_cast<int>(j); }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
              << " our=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ") \"" << out.outputs[0].text << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK(prompt_ok);
  }

  if (dump) {
    const std::string path = (gdir / "our_ids.i32").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
      std::fwrite(our_dump.data(), sizeof(int32_t), our_dump.size(), f);
      std::fclose(f);
      MESSAGE(label << " dumped our token ids -> " << path);
    }
  }
  MESSAGE(label << " correctness gate: " << (strict_exact + neartie_only) << "/" << N
          << " prompts PASS  (STRICT token-exact vs vLLM per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail << " forward-divergent)");
  REQUIRE(fail == 0);
}

}  // namespace

// CohereForCausalLM (dense, weight-only LayerNorm + GPT-J parallel residual +
// logit_scale + tied embeddings) — the first Command-R / Cohere SACRED gate. Proves
// the parallel-residual re-join and the logit_scale scalar are wired correctly.
// Resolves the first available (HF-gated) Command-R snapshot; SKIPs otherwise.
TEST_CASE("CohereForCausalLM dense paged-engine greedy correctness gate (dgx-only, SACRED)") {
  RunGate({"models--CohereForAI--aya-expanse-8b",
           "models--CohereLabs--aya-expanse-8b",
           "models--CohereForAI--aya-23-8B",
           "models--CohereForAI--c4ai-command-r-v01"},
          "commandr_greedy", "command-r");
}
