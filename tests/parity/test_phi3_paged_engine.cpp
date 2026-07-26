// vllm.cpp original (Phi-3/Phi-4 W4 — THE SACRED correctness gate for the
// Phi3ForCausalLM family); no upstream mirror (upstream's Phi-3 coverage is the
// generic text-generation entry — spec §"Tests to port").
//
// THE PAGED-ENGINE Phi3ForCausalLM (Phi-4-mini-instruct, bf16) GREEDY
// CORRECTNESS GATE. Drives the standard prompt battery through the FULL PAGED
// LLMEngine stack via LoadedEngine::FromModelDir and checks the greedy decode
// against the pinned vLLM 0.25.0 oracle. This proves the FOUR scalar multipliers
// (embedding / residual / attention / logits) are wired correctly — a mis-set
// scalar or the wrong attention scale (0.015625 vs 1/sqrt(64)) emits FLUENT WRONG
// tokens far outside the near-tie band (the OPT silent-corruption mode).
//
// GATE FORM (selected BY MEASUREMENT, RATIFIED near-tie ROOT-divergence
// methodology): run vLLM's OWN per-prompt greedy K=5 first (glm4-oracle-capture.py
// --per-prompt --model microsoft/Phi-4-mini-instruct) — Phi-4-mini is
// ALL-DETERMINISTIC over K=5, so the STRICT bar is well-posed. The gate teacher-
// forces vLLM on OUR exact sequence and, at the FIRST position where our greedy
// token diverges from vLLM's greedy (the ROOT — the last position whose prefix is
// bit-identical to vLLM's greedy path), requires vLLM's teacher-forced argmax to
// beat our token by <= kNearTieMnats. STRICT wherever our token equals vLLM's; a
// bf16 near-tie is the honest fallback where vLLM's own prefill/decode disagree.
// A root gap > kNearTieMnats is a REAL forward divergence the gate FAILS on;
// positions DOWNSTREAM of the root are cascade on a forked (equally-valid)
// continuation and are not counted. RCA (measured, current main): every root
// divergence <= 0.5 nats (max 0.5 @ p6); p12's only >0.5 positions (tok6/tok8 =
// 1.0) are cascade downstream of an exact-tie root (tok4, gap 0.0000).
//
// Goldens (tests/parity/goldens/phi4_mini_greedy/, captured on dgx):
//   greedy_ids.npy / greedy_dist.npy / our_ids.npy / neartie_gap_mnats.npy.
// Checkpoint-GATED + dgx-only: resolves ~/.cache/huggingface/hub/
// models--microsoft--Phi-4-mini-instruct. On CPU/CI it emits a loud SKIP.
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

const int32_t* AsI32(const parity::NpyArray& a) {
  return reinterpret_cast<const int32_t*>(a.data.data());
}

void RunGate(const std::string& repo_dir, const std::string& golden_subdir,
             const char* label) {
  const std::string snap = FindSnapshot(repo_dir);
  if (snap.empty()) {
    MESSAGE(label << " checkpoint absent; skipping (dgx-only) — " << repo_dir);
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE(label << " greedy golden absent; skipping — capture on dgx: "
            "glm4-oracle-capture.py --per-prompt --model "
            "microsoft/Phi-4-mini-instruct");
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
            "then glm4-neartie-gap.py --model microsoft/Phi-4-mini-instruct");
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
    // EOS-aware: greedy decode STOPS at EOS, so a prompt may yield fewer than T
    // tokens (e.g. phi-4-14B gives short factual answers). The real length is L; the
    // committed goldens pad post-termination positions (our_ids with -1, the gap
    // golden with 0). We compare only the L real tokens. (Phi-4-mini never hits EOS
    // in T tokens, so L==T there and the behavior is unchanged.)
    const int64_t L = static_cast<int64_t>(got.size());
    REQUIRE(L >= 1);
    REQUIRE(L <= T);
    if (dump) {
      for (int64_t j = 0; j < T; ++j)
        our_dump[static_cast<size_t>(i * T + j)] =
            (j < L) ? got[static_cast<size_t>(j)] : -1;
    }

    int first_div = -1;
    for (int64_t j = 0; j < L; ++j) {
      if (got[static_cast<size_t>(j)] != od[i * T + j]) { first_div = static_cast<int>(j); break; }
    }
    REQUIRE_MESSAGE(first_div < 0,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                    << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                    << " committed anchor=" << (first_div < 0 ? -1 : od[i * T + first_div])
                    << " — re-run glm4-neartie-gap.py to refresh the gap golden");
    // The committed anchor must terminate at the SAME point (padded with -1).
    for (int64_t j = L; j < T; ++j)
      REQUIRE_MESSAGE(od[i * T + j] == -1,
                      label << " anchor length drift prompt[" << i << "] tok=" << j
                      << " committed=" << od[i * T + j] << " (expected -1 post-EOS)");

    // RATIFIED near-tie ROOT-divergence gate (see [[near-tie-distributional-gate]]).
    // The teacher-forced gap measures FORWARD PARITY only while OUR prefix is
    // bit-identical to vLLM's greedy path. At the FIRST position where our greedy
    // token diverges from vLLM's greedy token (the ROOT) the gap is the CLEAN
    // forward-parity measurement — our token vs vLLM's argmax on the SAME prefix —
    // and a root gap > kNearTieMnats is a REAL forward divergence the gate FAILS on.
    // Positions AFTER the root sit on an off-greedy-path prefix vLLM's greedy never
    // visits (our continuation is a legitimately different, equally-valid sentence),
    // so their gaps are CASCADE on that different continuation, NOT independent
    // forward errors, and are not counted. This is a strict generalization of the
    // per-position olmo2/qwen3-dense gate: on any prompt WITHOUT a cascade (no >
    // kNearTieMnats position downstream of the root) the verdict is identical; it
    // differs only where an exact-tie root (vLLM's own decode contradicting its
    // teacher-forced argmax) forks the continuation. Phi-4-mini: p12 root=tok4 gap
    // 0.0000 (vLLM's argmax IS our token), tok6/tok8 = 1.0 nats are cascade on the
    // forked sentence. A genuine forward bug diverges EARLY -> the root gap is > the
    // band and is caught (verified RED-first by disabling LongRoPE / the fused qkv).
    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < L; ++j) {
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) { worst_gap = mn; worst_i = static_cast<int>(i); worst_j = static_cast<int>(j); }
      if (mn > kNearTieMnats) { prompt_ok = false; if (first_bad < 0) first_bad = static_cast<int>(j); }
      // ROOT reached: our token diverged from vLLM greedy. Its gap (checked above)
      // is the last on-greedy-path measurement; downstream is cascade -> stop.
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) { exact = false; break; }
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

// Phi-4-mini-instruct (dense, pre-norm Llama forward + pre-fused qkv/gate_up loader
// + precomputed partial-rotary LongRoPE cache) — the Phi3ForCausalLM SACRED gate.
// Proves the pre-fused loader + the LongRoPE cos/sin cache are wired correctly.
TEST_CASE("Phi-4-mini dense paged-engine greedy correctness gate (dgx-only, SACRED)") {
  RunGate("models--microsoft--Phi-4-mini-instruct", "phi4_mini_greedy",
          "Phi-4-mini");
}

// phi-4 (14B, Phi3ForCausalLM) — the RATIFIED BIGGER-DENSE anchor for the same
// forward code path. The near-tie methodology requires a bigger deterministic dense
// model to prove the forward is genuinely correct (not just near-tie-lucky). Same
// generic RunGate: STRICT wherever our token equals vLLM's per-prompt greedy, the
// near-tie ROOT-divergence band only where vLLM's own bf16 prefill/decode disagree.
// dgx-only + checkpoint-gated (models--microsoft--phi-4, ~29 GiB bf16; fits the GB10
// 119 GiB unified pool); emits a loud SKIP where the checkpoint/goldens are absent.
TEST_CASE("phi-4-14B dense paged-engine greedy STRICT anchor (dgx-only, SACRED)") {
  RunGate("models--microsoft--phi-4", "phi4_14b_greedy", "phi-4-14B");
}
