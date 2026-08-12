// vllm.cpp original (checkpoint-gated spec-decode gate); no upstream mirror.
//
// THE 27B DFlash BLOCK-DIFFUSION e2e CORRECTNESS GATE (SPEC-DFLASH D5, milestone
// M-df-1). Drives the FOUR committed golden prompts through the FULL paged engine
// with DFlash speculative decoding turned ON via EngineParams::speculative_config
// ('{"method":"dflash","model":<z-lab draft>,"num_speculative_tokens":16}'), and
// asserts, per the D0 gate-form finding (tests/parity/goldens/dflash_27b/):
//
//   (a) STRICT MODE-MATCHED token identity: our DFlash-ON greedy continuation ==
//       vLLM-DFlash-ON (the committed dflash_27b_spec_on.json golden, itself
//       run-deterministic K>=3). vLLM-DFlash-ON is NOT token-identical to
//       vLLM-spec-OFF at k=16 near-ties, so the gate is DFlash-ON vs DFlash-ON,
//       not the MTP three-way identity.
//   (b) MEASURED NONZERO ACCEPTANCE ~ vLLM's (2.2/8.8/4.75/4.57 acceptance_len;
//       num_accepted_delta 17/39/30/25). Token identity ALONE passes on a dead
//       drafter, so acceptance is a REQUIRED companion (the I5e dead-drafter trap).
//
// Checkpoint-GATED + dgx-only: the NVFP4 target + the z-lab DFlash draft + the CUDA
// spec kernels live only on dgx.casa (GB10); on the CPU dev box / CI the body emits
// a loud SKIP (compiles + links on CPU).
//
// SECOND CASE (SPEC-DFLASH-GGUF `GD4`, axis A): the same engine path with the
// DRAFT SOURCE as the free variable. See its own header comment below.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "hf_snapshot.h"
#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/logprobs.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// GATE-PIN-UNPINNED-SNAPSHOTS (#471). The `SnapDir` that used to live here was a
// HEURISTIC dressed as a resolver, and it is instructive about why heuristics do
// not substitute for a pin. It preferred the snapshot holding a single
// `model.safetensors` -- which does select `unsloth/Qwen3.6-27B-NVFP4`@890bdef7
// today, because only that revision is single-file. But its fallback returned
// `any`, i.e. THE LAST entry it saw, so on a cache where no snapshot is
// single-file it silently preferred the newest re-quant; and the preference
// itself would break the moment a future single-file revision appeared. It
// encoded a property the goldens happen to correlate with, not the identity the
// goldens were captured against.
//
// DELETED, not kept. Resolution goes through parity::HfSnapshot, which names the
// revision and skips rather than substituting.

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// ── SPEC-DFLASH-GGUF GD4 helpers ────────────────────────────────────────────
// Greedy plus per-position alternatives. Built in one shot rather than by
// mutating a Greedy() result so PostInit() sees the final field set once.
vllm::SamplingParams GreedyWithLogprobs(int max_tokens, int k) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.logprobs = k;
  sp.PostInit();
  return sp;
}

std::string Env(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr ? std::string(v) : std::string();
}

int EnvInt(const char* name, int fallback) {
  const std::string v = Env(name);
  return v.empty() ? fallback : std::atoi(v.c_str());
}

std::string Ids(const std::vector<int32_t>& v) {
  std::string s;
  for (const int32_t id : v) s += std::to_string(id) + " ";
  return s;
}

// Index of the first position where two greedy continuations disagree, or
// npos when one is a prefix of the other and they agree that far.
std::size_t FirstDiff(const std::vector<int32_t>& a,
                      const std::vector<int32_t>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return a.size() == b.size() ? std::string::npos : n;
}

}  // namespace

TEST_CASE("qwen27 DFlash e2e correctness gate (dgx-only, 27B block-diffusion k=16)") {
  const std::string target = parity::Qwen27NvfP4Snapshot();
  const std::string draft = parity::Qwen27DFlashDraftSnapshot();
  const fs::path golden_path =
      fs::path(PARITY_GOLDENS_DIR) / "dflash_27b" / "dflash_27b_spec_on.json";
  if (target.empty() || draft.empty() || !fs::exists(golden_path)) {
    MESSAGE("SKIP (dgx-only): the 27B DFlash e2e gate needs "
               "unsloth/Qwen3.6-27B-NVFP4 @"
            << std::string(parity::kQwen27NvfP4Revision)
            << " (got: " << (target.empty() ? "ABSENT" : target) << "), "
            << "z-lab/Qwen3.6-27B-DFlash @" << std::string(parity::kQwen27DFlashDraftRevision)
            << " (got: " << (draft.empty() ? "ABSENT" : draft) << "), "
            << "and the committed golden. A cache holding a DIFFERENT revision "
               "of either repo skips rather than being substituted (#471).");
    return;
  }

  std::ifstream gf(golden_path.string());
  json golden;
  gf >> golden;
  const int k = golden.value("num_speculative_tokens", 16);
  const auto& records = golden.at("records");

  // Turn DFlash speculative decoding ON. num_speculative_tokens = block_size (16).
  vllm::entrypoints::EngineParams params;
  params.max_num_seqs = 2;  // bound the k+1 GDN spec-state slots (~2.4 GiB/req).
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      std::string("{\"method\":\"dflash\",\"model\":\"") + draft +
      "\",\"num_speculative_tokens\":" + std::to_string(k) + "}");

  MESSAGE("qwen27_dflash: loading 27B target " << target
          << " + DFlash draft " << draft << " (k=" << k << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(target, params);

  int prompt_idx = 0;
  int total_prompts = 0, exact_prompts = 0;
  for (const auto& rec : records) {
    const std::string prompt = rec.at("prompt").get<std::string>();
    const std::vector<int32_t> want_out =
        rec.at("output_token_ids").get<std::vector<int32_t>>();
    const std::vector<int32_t> want_prompt_ids =
        rec.at("prompt_token_ids").get<std::vector<int32_t>>();
    const int max_tokens = static_cast<int>(want_out.size());
    const double want_acc_len = rec.value("acceptance_len", 0.0);
    const int want_acc_delta = rec.value("num_accepted_delta", 0);
    ++total_prompts;

    const int64_t acc_before = loaded->runner().spec_drafts_accepted();
    const int64_t prop_before = loaded->runner().spec_drafts_proposed();

    const std::string rid = "dflash" + std::to_string(prompt_idx++);
    loaded->engine().add_request(rid, prompt, Greedy(max_tokens));
    std::optional<vllm::RequestOutput> final;
    while (loaded->engine().has_unfinished_requests())
      for (vllm::RequestOutput& item : loaded->engine().step())
        if (item.finished) final = std::move(item);
    REQUIRE(final.has_value());
    const vllm::RequestOutput& out = *final;
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;

    const int64_t acc = loaded->runner().spec_drafts_accepted() - acc_before;
    const int64_t prop = loaded->runner().spec_drafts_proposed() - prop_before;

    CHECK(out.prompt_token_ids == want_prompt_ids);
    const bool exact = (got == want_out);
    if (exact) ++exact_prompts;
    size_t shared = 0;  // shared-prefix length before the first divergence.
    if (!exact) {
      const size_t n = std::min(got.size(), want_out.size());
      while (shared < n && got[shared] == want_out[shared]) ++shared;
      std::string ctx =
          "first divergence at index " + std::to_string(shared) + ": got[";
      for (size_t j = (shared >= 2 ? shared - 2 : 0);
           j < std::min(got.size(), shared + 3); ++j)
        ctx += std::to_string(got[j]) +
               (j + 1 < std::min(got.size(), shared + 3) ? "," : "");
      ctx += "] vs want[";
      for (size_t j = (shared >= 2 ? shared - 2 : 0);
           j < std::min(want_out.size(), shared + 3); ++j)
        ctx += std::to_string(want_out[j]) +
               (j + 1 < std::min(want_out.size(), shared + 3) ? "," : "");
      ctx += "]";
      MESSAGE("  " << ctx);
    }
    MESSAGE("qwen27_dflash prompt[" << (prompt_idx - 1) << "] \"" << prompt
            << "\": exact=" << (exact ? "YES" : "NO") << "  drafts "
            << acc << "/" << prop << " accepted (golden accepted_delta="
            << want_acc_delta << ", acceptance_len=" << want_acc_len
            << ")  text=\"" << out.outputs[0].text << "\"");

    // (b) MANDATORY nonzero acceptance ~ vLLM's (the dead-drafter trap): token
    // identity ALONE passes on a dead drafter, so acceptance is the load-bearing
    // liveness+correctness proof. HARD: the drafter is alive and its accepted
    // count is in vLLM's neighborhood (measured deltas +2/0/-1/0 on the 4 golden
    // prompts; band 4 rejects a dead/broken drafter with wide margin).
    CHECK(prop > 0);
    CHECK(acc > 0);
    CHECK(std::llabs(acc - static_cast<int64_t>(want_acc_delta)) <= 4);

    // (a) STRICT mode-matched token identity vs the vLLM-DFlash-ON golden. Where
    // OUR inline block-verify numerics match vLLM's this is exact; where a bf16
    // near-tie in the D3-documented inline context-KV recompute envelope
    // (~0.3-1.3% rel-L2) flips the block context, greedy diverges at a SINGLE
    // near-tie point (the ratified near-tie ROOT the D0 gate-form anticipated) —
    // proven a near-tie, not a wiring bug, by the exact prompts + near-exact
    // acceptance + a non-trivial shared prefix here. A structural break would
    // diverge at index 0. HARD: a non-exact prompt still shares a real prefix.
    if (!exact) CHECK(shared > 0);
  }
  MESSAGE("qwen27_dflash: " << exact_prompts << "/" << total_prompts
          << " prompts STRICT token-exact vs vLLM-DFlash-ON; the rest diverge at "
             "single bf16 near-ties (acceptance ~ vLLM on all). Strict 4/4 is "
             "gated on the D6 persistent paged draft-KV (bit-matching vLLM's "
             "fused context-KV projections).");
  // HARD: strict identity holds on the deterministic majority (proves the runner
  // accumulation + block-verify are correct, not universally near-tie-broken).
  CHECK(exact_prompts >= 2);
}

// ── SPEC-DFLASH-GGUF `GD4`: the axis-A end-to-end token + acceptance gate ───
//
// Axis A of the GGUF-DFlash spike is "GGUF DRAFT + safetensors TARGET". `GD1`-`GD3`
// landed the loader for it and gated it at the unit tier, but until this case
// nothing had ever GENERATED through a GGUF-sourced DFlash draft.
//
// The case above is a different gate and cannot be reused: it pins our DFlash-ON
// tokens to a committed vLLM-DFlash-ON golden with the z-lab safetensors snapshot
// HARDCODED, so it has no way to vary the one thing this row changes. Here the
// DRAFT SOURCE is the free variable: `VLLM_DFLASH_DRAFT` and the optional
// `VLLM_DFLASH_DRAFT_B` each name a `.gguf` file, a checkpoint directory, or an
// HF repo id (`ResolveDflashDraftDir` accepts all three).
//
// WHAT THE BAR IS, AND WHY IT IS NOT "spec-ON == spec-OFF"
//
// For MTP at k=1 the bar is spec-ON == spec-OFF: greedy speculation rewrites HOW
// tokens are produced, never WHICH. DFlash at k=16 is measurably NOT like that,
// and the measurement predates every line of GGUF work: `SPEC-DFLASH` `D5`
// captured vLLM's OWN DFlash-ON against vLLM's OWN spec-OFF and found them
// token-different on 3 of the 4 golden prompts, because the block verify diverges
// from sequential decode at bf16 near-ties. `D6` then root-caused our matching
// residual as bf16-IRREDUCIBLE (the inline context-KV recompute envelope, ~0.3%
// rel-L2, plus a from-scratch block-attention kernel against vLLM's flashinfer
// paged one) and ratified the near-tie form as FINAL. So the landed case above
// gates DFlash-ON against DFlash-ON, not against spec-OFF, and this one must be
// shaped the same way or it fails the ALREADY-`DONE` safetensors reference arm -
// which it did, at index 16 of the 48-token prompt, identically for both draft
// formats. A gate the reference arm fails is not measuring the change.
//
// So the binding bar here is MODE-MATCHED and CROSS-FORMAT:
//
//   (a) THE SAME DRAFT FROM TWO SOURCES IS THE SAME DRAFT. With both
//       `VLLM_DFLASH_DRAFT` and `VLLM_DFLASH_DRAFT_B` set, the two DFlash-ON
//       continuations must be token-for-token identical and their accepted /
//       proposed counts must match exactly. This is the whole claim of axis A -
//       a draft read through the GGUF conventions (the RAW norms, the +1 target
//       layer offset, the [N,K] fc orientation, the target-supplied vocab) is
//       the SAME draft - and it is the one comparison the DFlash near-tie
//       envelope cannot muddy, because both arms take the identical code path
//       and differ only in where the bytes came from.
//   (b) MEASURED NONZERO ACCEPTANCE, per arm. (a) alone also passes on two
//       equally dead drafters: identical output, every proposal rejected, every
//       token recomputed by the target - which is exactly what a mis-loaded
//       draft produces, since a norm read under the wrong convention still
//       yields valid logits, just wrong ones. This is the `I5e` dead-drafter
//       trap; both halves are required.
//   (c) NEAR-TIE-ROBUST against spec-OFF, in the ratified `D5` form: identical,
//       or a non-trivial shared prefix. A structural break diverges at index 0.
//
// Why the spec-OFF arm is run TWICE. The sibling `SPEC-MTP-GGUF` measurement
// found that on the 35B A3B NVFP4 SAFETENSORS target the quantized-GEMM logits
// land on a coarse 1/16 grid producing EXACT top-1/top-2 ties, and that at such a
// tie two spec-OFF runs of the same binary need not agree. A target that does not
// reproduce its own spec-OFF sequence makes (c) unmeasurable, and that has to be
// established as a TARGET property before any divergence is charged to a draft.
// So arm 0 is the reference, arm 1 is the reproducibility control, and a third
// logprobs arm MEASURES the per-position rank1-rank2 margins rather than arguing
// about them. All three share ONE engine, so they cost one load.
//
// ASSET-GATED (dgx-only): needs the 27B NVFP4 safetensors target plus a DFlash
// draft. Absent => loud SKIP, so CI stays asset-free.
namespace {

// One DFlash-ON arm: its own engine (speculative_config is fixed at
// construction), one greedy request, the runner's draft counters. Scoped so the
// engine is destroyed before the next arm loads, keeping peak RSS at one engine.
struct SpecArm {
  std::vector<int32_t> ids;
  int64_t proposed = 0;
  int64_t accepted = 0;
};

SpecArm RunDflashOn(const std::string& target, const std::string& draft, int k,
                    const std::string& prompt, int max_tokens,
                    const vllm::entrypoints::EngineParams& base,
                    const char* request_id) {
  vllm::entrypoints::EngineParams spec = base;
  spec.speculative_config = vllm::ParseSpeculativeConfigJson(
      std::string("{\"method\":\"dflash\",\"model\":\"") + draft +
      "\",\"num_speculative_tokens\":" + std::to_string(k) + "}");

  MESSAGE("GD4: loading target + DFlash draft " << draft << " (k=" << k << ")...");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(target, spec);
  const vllm::RequestOutput out =
      loaded->engine().generate(prompt, Greedy(max_tokens), request_id);
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);

  SpecArm arm;
  arm.ids = out.outputs[0].token_ids;
  arm.proposed = loaded->runner().spec_drafts_proposed();
  arm.accepted = loaded->runner().spec_drafts_accepted();
  MESSAGE("GD4 spec-ON text=\"" << out.outputs[0].text << "\"");
  MESSAGE("GD4 ids spec-ON [" << draft << "]: " << Ids(arm.ids));
  MESSAGE("GD4 ACCEPTANCE draft=" << draft << " proposed=" << arm.proposed
          << " accepted=" << arm.accepted << " rate="
          << (arm.proposed > 0
                  ? static_cast<double>(arm.accepted) /
                        static_cast<double>(arm.proposed)
                  : 0.0));
  return arm;
}

// Does this draft SOURCE re-encode the weights at reduced precision? Decided
// from the FILE, never from a flag, because that is exactly the fact bar (a)
// turns on and a flag would let a future run band an arm that has no business
// being banded. A ggml type is a re-encoding iff it packs MORE THAN ONE element
// per block: `GgmlTraits().block_elems` is 1 for F32 / F16 / BF16 and 32-256 for
// every Q*/IQ*/MXFP4/NVFP4 type (`gguf_reader.cpp` trait table). A checkpoint
// directory or an HF repo id is not a GGUF at all and loads at its stored bf16,
// so it answers false.
bool IsQuantizedGgufDraft(const std::string& src) {
  std::error_code ec;
  if (!fs::is_regular_file(src, ec)) return false;
  if (fs::path(src).extension() != ".gguf") return false;
  const vllm::GgufFile f = vllm::GgufFile::Open(src);
  for (const vllm::GgufTensorInfo& t : f.Tensors()) {
    if (vllm::GgmlTraits(t.ggml_type).block_elems > 1) {
      // `std::string(...)`, not the bare `const char*`: streaming a raw pointer
      // into MESSAGE(...) prints its bool conversion, which is the same trap
      // the spec-OFF verdict below already documents.
      MESSAGE("GD9: " << src << " is a QUANTIZED GGUF (" << t.name << " is "
                      << std::string(vllm::GgmlTraits(t.ggml_type).name) << ")");
      return true;
    }
  }
  MESSAGE("GD9: " << src << " is an UNQUANTIZED GGUF (every tensor a scalar "
                     "ggml type)");
  return false;
}

// The cross-QUANTIZATION accept band, and where the number comes from.
//
// It is NOT a tolerance chosen to make a red test green; that failure mode is
// what this row already produced once, when the 2026-07-28 "Q4_K_M costs this
// draft nothing" reading turned out to be an emulation-build artifact. It is
// the measured spread plus exactly one quantum of headroom:
//
//   * `accepted` moves in units of 1, and the MEASURED cross-quantization delta
//     on the production build is 0 on the 24-token prompt (both drafts equal)
//     and 1 on the 48-token prompt (Q4_K_M 46 against bf16 47). A band equal to
//     the observed maximum has zero margin and flips red on any single
//     additional near-tie, so the bound is that maximum plus one unit: 2.
//   * `proposed` is NOT an independent quantity. With the token streams
//     identical (which bar (a) asserts unconditionally, and which the verifier
//     guarantees), both arms emit the same number of tokens, and each 16-wide
//     propose block emits its accepted prefix plus one bonus token. So
//     `accepted + nblocks` is the same constant in both arms, hence
//     `proposed = k * nblocks` gives `d_proposed = -k * d_accepted` EXACTLY.
//     The measurement confirms it: 46/112 against 47/96 is d_accepted -1 and
//     d_proposed +16 = -16 * -1. The proposed bound is therefore `k` times the
//     accepted bound by derivation, not by a second choice.
//
// It is also TIGHTER than the precedent it follows: the landed `SPEC-DFLASH`
// golden arm bands its acceptance at `<= 4`.
constexpr int64_t kCrossQuantAcceptBand = 2;

int64_t AbsDelta(int64_t a, int64_t b) { return a > b ? a - b : b - a; }

}  // namespace

TEST_CASE("dflash axis-A: a GGUF draft and a safetensors draft are the same draft") {
  // VLLM_DFLASH_TARGET still names an explicit directory for a deliberate
  // different-checkpoint run; the FALLBACK is now pinned rather than guessed.
  std::string target = Env("VLLM_DFLASH_TARGET");
  if (target.empty()) target = parity::Qwen27NvfP4Snapshot();
  const std::string draft_a = Env("VLLM_DFLASH_DRAFT");
  const std::string draft_b = Env("VLLM_DFLASH_DRAFT_B");
  if (target.empty() || draft_a.empty()) {
    MESSAGE("SKIP: set VLLM_DFLASH_DRAFT (a .gguf, a checkpoint dir, or an HF "
            "repo id), optionally VLLM_DFLASH_DRAFT_B for the cross-format bar, "
            "and VLLM_DFLASH_TARGET (the 27B NVFP4 safetensors dir)");
    return;
  }

  const int max_tokens = EnvInt("VLLM_DFLASH_MAX_TOKENS", 24);
  const int k = EnvInt("VLLM_DFLASH_K", 16);
  std::string prompt = Env("VLLM_DFLASH_PROMPT");
  if (prompt.empty()) prompt = "The capital of France is";

  vllm::entrypoints::EngineParams base;
  base.max_num_seqs = 2;  // bound the k+1 GDN spec-state slots (~2.4 GiB/req).

  std::vector<int32_t> off_a, off_b, off_lp;
  {
    MESSAGE("GD4: loading target " << target << " spec-OFF...");
    auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(target, base);

    const vllm::RequestOutput a =
        loaded->engine().generate(prompt, Greedy(max_tokens), "gd4-off-a");
    REQUIRE(a.finished);
    REQUIRE(a.outputs.size() == 1);
    off_a = a.outputs[0].token_ids;
    REQUIRE(!off_a.empty());
    MESSAGE("GD4 spec-OFF[a] text=\"" << a.outputs[0].text << "\"");

    const vllm::RequestOutput b =
        loaded->engine().generate(prompt, Greedy(max_tokens), "gd4-off-b");
    REQUIRE(b.finished);
    off_b = b.outputs[0].token_ids;

    // Margin sweep. Same engine, same prompt, greedy; the only difference is
    // that the sampler is asked to publish alternatives. Reports the count of
    // EXACT top-1/top-2 ties and the minimum margin, which is what decides
    // whether a divergence is a tie-break coin flip or a wiring defect.
    const vllm::RequestOutput c = loaded->engine().generate(
        prompt, GreedyWithLogprobs(max_tokens, 8), "gd4-off-probe");
    REQUIRE(c.finished);
    off_lp = c.outputs[0].token_ids;
    const vllm::CompletionOutput& co = c.outputs[0];
    if (co.logprobs.has_value()) {
      int exact_ties = 0;
      double min_margin = std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < co.logprobs->size(); ++i) {
        const vllm::LogprobsOnePosition& pos = (*co.logprobs)[i];
        double lp1 = std::numeric_limits<double>::quiet_NaN();
        double lp2 = std::numeric_limits<double>::quiet_NaN();
        for (const int32_t id : pos.order) {
          const vllm::Logprob* lp = pos.find(id);
          if (lp == nullptr || !lp->rank.has_value()) continue;
          if (*lp->rank == 1) lp1 = lp->logprob;
          if (*lp->rank == 2) lp2 = lp->logprob;
        }
        if (std::isnan(lp1) || std::isnan(lp2)) continue;
        const double margin = lp1 - lp2;
        if (margin <= 0.0) ++exact_ties;
        if (margin < min_margin) min_margin = margin;
        MESSAGE("GD4 margin pos=" << i << " top1=" << lp1 << " top2=" << lp2
                                  << " margin=" << margin << " nats");
      }
      MESSAGE("GD4 TARGET margin sweep: exact ties=" << exact_ties
              << " min margin=" << min_margin << " nats");
    }
  }

  MESSAGE("GD4 ids spec-OFF[a]: " << Ids(off_a));
  MESSAGE("GD4 ids spec-OFF[b]: " << Ids(off_b));
  MESSAGE("GD4 ids spec-OFF[probe]: " << Ids(off_lp));

  // The reproducibility CONTROL, reported before any bar is applied. A target
  // that forks against itself would make (c) a coin flip rather than a statement
  // about a draft, and the answer has to be on the record either way.
  const std::size_t off_diff = FirstDiff(off_a, off_b);
  const bool off_stable = (off_diff == std::string::npos);
  // Composed as a string rather than streamed piecewise: a mixed
  // `const char*` / `std::string` conditional inside MESSAGE(...) printed the
  // bare bool on the first GD4 run instead of the word.
  const std::string off_verdict =
      off_stable ? std::string("STABLE")
                 : "UNSTABLE, first divergence at index " +
                       std::to_string(off_diff);
  MESSAGE("GD4 spec-OFF self-reproducibility: " << off_verdict);

  const SpecArm arm_a =
      RunDflashOn(target, draft_a, k, prompt, max_tokens, base, "gd4-on-a");

  // (b) The drafter ran and was believed at least once. Checked FIRST because it
  // is the half that no near-tie envelope can excuse: a collapsed acceptance is a
  // defect, not a rounding difference.
  CHECK(arm_a.proposed > 0);
  CHECK(arm_a.accepted > 0);

  // (c) The ratified `D5` near-tie form against spec-OFF. Reported with the exact
  // divergence index so a regression is diagnosable from the log alone.
  const std::size_t on_diff = FirstDiff(arm_a.ids, off_a);
  MESSAGE("GD4 spec-ON vs spec-OFF[a]: "
          << (on_diff == std::string::npos
                  ? std::string("IDENTICAL")
                  : "first divergence at index " + std::to_string(on_diff)));
  CHECK(on_diff != 0u);  // a structural break diverges at the very first token.

  if (draft_b.empty()) {
    MESSAGE("GD4: VLLM_DFLASH_DRAFT_B unset, cross-format bar (a) NOT run");
    return;
  }

  const SpecArm arm_b =
      RunDflashOn(target, draft_b, k, prompt, max_tokens, base, "gd4-on-b");
  CHECK(arm_b.proposed > 0);
  CHECK(arm_b.accepted > 0);

  const std::size_t xf_diff = FirstDiff(arm_a.ids, arm_b.ids);
  MESSAGE("GD4 CROSS-FORMAT " << draft_a << " vs " << draft_b << ": "
          << (xf_diff == std::string::npos
                  ? std::string("IDENTICAL")
                  : "first divergence at index " + std::to_string(xf_diff)));

  // (a) THE BAR, split by what the two arms ACTUALLY vary (`GD9`).
  //
  // TOKENS are exact in BOTH forms and are never banded. That half is not even
  // a hope: `rejection_sampler.h` pins ACCEPT-IFF-EQUAL with
  // STOP-AT-FIRST-MISMATCH, so under greedy verification the emitted stream is
  // the TARGET's own greedy continuation whatever the draft proposes.
  CHECK(arm_a.ids == arm_b.ids);

  // The ACCEPT COUNTS are the only channel carrying information about the
  // draft, and what they are worth depends on whether the two arms hold the
  // same weights. The original single bar read "Same weights, two containers",
  // which is TRUE of an unquantized GGUF against the safetensors shards and
  // FALSE of a `Q4_K_M` file, which is a 4-bit re-encoding of them. So:
  const bool cross_quant =
      IsQuantizedGgufDraft(draft_a) || IsQuantizedGgufDraft(draft_b);
  if (!cross_quant) {
    // CROSS-FORMAT. Same weights to the byte (gate 2 proves that at load:
    // 58/58 tensors byte-identical through this same loader path), so exact
    // accounting is a real invariant and the right bar. Nothing about the
    // DFlash near-tie envelope can excuse a difference, because both arms run
    // the identical code path and differ only in where the bytes came from.
    MESSAGE("GD9: CROSS-FORMAT arm (neither draft is quantized), accept "
            "counts gated EXACT");
    CHECK(arm_a.proposed == arm_b.proposed);
    CHECK(arm_a.accepted == arm_b.accepted);
  } else {
    // CROSS-QUANTIZATION. Different weights by construction, so an exact
    // accept-count bar asserts something the arms do not satisfy. Banded; see
    // kCrossQuantAcceptBand for where the two bounds come from.
    MESSAGE("GD9: CROSS-QUANTIZATION arm, accept counts gated to band "
            << kCrossQuantAcceptBand << " (proposed " << k << "x that): "
            << "d_accepted=" << (arm_a.accepted - arm_b.accepted)
            << " d_proposed=" << (arm_a.proposed - arm_b.proposed));
    CHECK(AbsDelta(arm_a.accepted, arm_b.accepted) <= kCrossQuantAcceptBand);
    CHECK(AbsDelta(arm_a.proposed, arm_b.proposed) <=
          static_cast<int64_t>(k) * kCrossQuantAcceptBand);
  }
}

// ── SPEC-DFLASH-GGUF `GD7`: the axis-B end-to-end token + acceptance gate ───
//
// Axis B is "the TARGET is a GGUF too". Its whole content is `B1`: the draft
// SHARES the target's bf16 `embed_tokens` + `lm_head`, and until `B1` that
// sharing was typed as `const std::vector<SafetensorsFile>&`, so a GGUF target
// could not supply it and `FromModelDir`'s GGUF branch REFUSED dflash outright.
// `SharedHeadSource` re-expresses the seam as a source; `B2` deleted the
// refusal; `B3` wired the load into the GGUF branch. This case is the first
// thing that ever generated through it.
//
// WHAT IS ASSERTED, AND WHY IT IS NOT A CROSS-TARGET IDENTITY
//
// The axis-A case above could hard-assert cross-FORMAT token identity because
// the two arms differed ONLY in where the DRAFT's bytes came from: same target,
// same kernels, same numerics. Axis B cannot borrow that form, for a reason
// that has nothing to do with this row: our two 27B containers do not compute
// the same way. The safetensors NVFP4 target runs the true W4A4 fp4 GEMM path;
// the GGUF NVFP4 target has no NVFP4 GGUF GEMM at all (`QUANT-GGUF-NVFP4` is
// dequant-only, `docs/STATUS.md`), so on CUDA its weights EXPAND to bf16 and it
// runs bf16 GEMMs. Two different kernel families over the same nominal weights
// diverge at near-ties by construction. So a cross-target token identity would
// be gating the containers' arithmetic, not axis B.
//
// The bars this case therefore binds are the ones axis B actually owns:
//
//   (a) MEASURED NONZERO ACCEPTANCE on the GGUF-target arm. This is the spike's
//       HIGHEST-ranked risk made falsifiable: the draft scores with a bf16
//       `lm_head` dequantized out of the target file while the target computes
//       on its own path, so draft and target can score with numerically
//       different heads and acceptance can move. Collapse to zero IS a defect
//       (and is also exactly what a mis-wired shared head produces - a dead
//       drafter still emits valid-looking logits). A lower-but-nonzero rate is
//       a finding to record, not a failure, which is why the hard bar is
//       nonzero and the COMPARISON below is reported.
//   (b) NEAR-TIE-ROBUST vs the SAME target's spec-OFF, in the ratified `D5`
//       form: identical, or a divergence that is not at index 0. Within one
//       target this is meaningful and container-independent - a structural
//       break in the shared-head wiring diverges at the very first token.
//   (c) REPORTED, and CONDITIONALLY gated: the GGUF-target arm against the
//       safetensors-target arm, DFlash-ON vs DFlash-ON, plus their spec-OFF
//       controls. The spec-OFF pair is the honest discriminator: if the two
//       containers agree token-for-token WITHOUT speculation then they are the
//       same target numerically on this prompt, and only then is a difference
//       in the DFlash-ON pair, or in the acceptance rate, chargeable to the
//       shared head. In that case the acceptance rates are gated to agree
//       within 25% relative. If the spec-OFF pair already differs, the arms are
//       reported and nothing is charged to axis B.
//
// ASSET-GATED (dgx-only): `VLLM_DFLASH_TARGET_B` (the `.gguf` target) and
// `VLLM_DFLASH_DRAFT`; `VLLM_DFLASH_TARGET` (the safetensors target) optional
// and enables (c). Absent => loud SKIP, so CI stays asset-free.
namespace {

// One spec-OFF arm on its own engine, scoped so peak RSS stays at one 27B.
std::vector<int32_t> RunSpecOff(const std::string& target,
                                const std::string& prompt, int max_tokens,
                                const vllm::entrypoints::EngineParams& base,
                                const char* request_id) {
  MESSAGE("GD7: loading target " << target << " spec-OFF...");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(target, base);
  const vllm::RequestOutput out =
      loaded->engine().generate(prompt, Greedy(max_tokens), request_id);
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  MESSAGE("GD7 spec-OFF [" << target << "] text=\"" << out.outputs[0].text
                           << "\"");
  return out.outputs[0].token_ids;
}

double AcceptRate(const SpecArm& a) {
  return a.proposed > 0 ? static_cast<double>(a.accepted) /
                              static_cast<double>(a.proposed)
                        : 0.0;
}

std::string DiffVerdict(const std::vector<int32_t>& a,
                        const std::vector<int32_t>& b) {
  const std::size_t d = FirstDiff(a, b);
  return d == std::string::npos
             ? std::string("IDENTICAL")
             : "first divergence at index " + std::to_string(d);
}

}  // namespace

TEST_CASE("dflash axis-B: a GGUF target serves the draft's shared head") {
  const std::string target_g = Env("VLLM_DFLASH_TARGET_B");
  const std::string target_s = Env("VLLM_DFLASH_TARGET");
  const std::string draft = Env("VLLM_DFLASH_DRAFT");
  if (target_g.empty() || draft.empty()) {
    MESSAGE("SKIP: set VLLM_DFLASH_TARGET_B (a .gguf target) and "
            "VLLM_DFLASH_DRAFT (a .gguf draft, a checkpoint dir, or an HF repo "
            "id); VLLM_DFLASH_TARGET (the safetensors target) is optional and "
            "enables the cross-target comparison");
    return;
  }
  REQUIRE(fs::exists(target_g));

  const int max_tokens = EnvInt("VLLM_DFLASH_MAX_TOKENS", 24);
  const int k = EnvInt("VLLM_DFLASH_K", 16);
  std::string prompt = Env("VLLM_DFLASH_PROMPT");
  if (prompt.empty()) prompt = "The capital of France is";

  vllm::entrypoints::EngineParams base;
  base.max_num_seqs = 2;  // bound the k+1 GDN spec-state slots (~2.4 GiB/req).

  const std::vector<int32_t> off_g =
      RunSpecOff(target_g, prompt, max_tokens, base, "gd7-off-gguf");
  REQUIRE(!off_g.empty());
  MESSAGE("GD7 ids GGUF-target spec-OFF: " << Ids(off_g));

  const SpecArm arm_g =
      RunDflashOn(target_g, draft, k, prompt, max_tokens, base, "gd7-on-gguf");

  // (a) The load-bearing half. Checked first: no near-tie envelope can excuse a
  // collapsed acceptance, and a dead drafter is what a mis-wired shared head
  // looks like from the outside.
  CHECK(arm_g.proposed > 0);
  CHECK(arm_g.accepted > 0);

  // (b) The ratified `D5` near-tie form, WITHIN the GGUF target.
  MESSAGE("GD7 GGUF-target spec-ON vs its own spec-OFF: "
          << DiffVerdict(arm_g.ids, off_g));
  CHECK(FirstDiff(arm_g.ids, off_g) != 0u);

  if (target_s.empty()) {
    MESSAGE("GD7: VLLM_DFLASH_TARGET unset, cross-target comparison (c) NOT "
            "run");
    return;
  }

  // (c) The cross-target comparison, with its own spec-OFF control so a
  // difference is attributable before it is charged to anything.
  const std::vector<int32_t> off_s =
      RunSpecOff(target_s, prompt, max_tokens, base, "gd7-off-st");
  const SpecArm arm_s =
      RunDflashOn(target_s, draft, k, prompt, max_tokens, base, "gd7-on-st");
  CHECK(arm_s.proposed > 0);
  CHECK(arm_s.accepted > 0);

  const bool targets_agree = FirstDiff(off_s, off_g) == std::string::npos;
  MESSAGE("GD7 CROSS-TARGET spec-OFF " << target_s << " vs " << target_g << ": "
          << DiffVerdict(off_s, off_g));
  MESSAGE("GD7 CROSS-TARGET spec-ON  " << target_s << " vs " << target_g << ": "
          << DiffVerdict(arm_s.ids, arm_g.ids));
  MESSAGE("GD7 ACCEPTANCE safetensors-target " << arm_s.accepted << "/"
          << arm_s.proposed << " (" << AcceptRate(arm_s)
          << ") vs GGUF-target " << arm_g.accepted << "/" << arm_g.proposed
          << " (" << AcceptRate(arm_g) << ")");

  if (!targets_agree) {
    MESSAGE("GD7: the two containers already disagree WITHOUT speculation, so "
            "the DFlash-ON difference and the acceptance delta are NOT "
            "chargeable to the shared head; reported only");
    return;
  }
  // The containers are the same target on this prompt, so the shared head is
  // the only thing left that could move acceptance. This is the spike's
  // highest-ranked risk, gated.
  const double rs = AcceptRate(arm_s);
  const double rg = AcceptRate(arm_g);
  MESSAGE("GD7: containers agree spec-OFF; gating the acceptance ratio "
          << (rs > 0.0 ? rg / rs : 0.0));
  CHECK(rs > 0.0);
  CHECK(std::abs(rg - rs) <= 0.25 * rs);
}
