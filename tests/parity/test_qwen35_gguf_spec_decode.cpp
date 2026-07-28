// vllm.cpp original (asset-gated GGUF spec-decode gate); no upstream mirror.
//
// THE GGUF MTP TOKEN-IDENTITY GATE (`SPEC-MTP-GGUF`, spike row `G4`).
//
// The safetensors sibling (test_qwen27_spec_decode.cpp) establishes its
// three-way identity transitively through a pinned vLLM-oracle golden. There is
// no vLLM oracle for a GGUF target - vLLM has no GGUF MTP path at all - so this
// gate is TWO-way and self-referential, which is exactly the right bar for what
// this row changes: the head now loads from a different SOURCE, and the claim
// being tested is that speculating with it changes no output.
//
//   (a) SPEC-ON == SPEC-OFF, token for token, greedy, single request, same
//       .gguf file loaded twice. Greedy MTP is exactness-preserving, so any
//       divergence is a loader or wiring bug, not sampling noise. This is the
//       claim that matters: the head read through the GGUF conventions (the
//       (w+1) norm un-shift, the quantization routing, the [N,K] fc
//       orientation) is the SAME head, so it proposes tokens the target
//       accepts rather than tokens the target has to reject.
//   (b) MEASURED NONZERO ACCEPTANCE. Token identity ALONE also passes on a
//       completely dead drafter - every proposal rejected, every token
//       recomputed by the target - which is precisely the failure mode a
//       mis-loaded head produces (a norm off by one still yields valid logits,
//       just wrong ones). So identity without acceptance proves nothing about
//       the head, and both are required.
//
// ASSET-GATED: needs a Qwen3.5/3.6 GGUF converted WITH the MTP head. Point
// VLLM_MTP_GGUF_MODEL at one; absent => loud SKIP so CI stays asset-free.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/logprobs.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

constexpr int kMaxTokens = 24;
// Alternatives requested per position by the device-delta probe below. Wide
// enough that the OTHER device's pick is virtually certain to appear in the
// row, which is what makes the two rows directly comparable.
constexpr int kProbeTopK = 20;
// Deliberately prosaic and low-entropy: a factual continuation keeps the greedy
// path away from near-ties, so a divergence indicts the drafter rather than
// bf16 rounding at a coin-flip step.
constexpr const char* kPrompt = "The capital of France is";

std::string MtpGgufPath() {
  const char* env = std::getenv("VLLM_MTP_GGUF_MODEL");
  return env != nullptr ? std::string(env) : std::string();
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// Greedy, plus the per-position alternatives the probe needs. Built in one shot
// rather than by mutating a Greedy() result, so PostInit() sees the final field
// set exactly once.
vllm::SamplingParams GreedyWithLogprobs(int max_tokens, int k) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.logprobs = k;
  sp.PostInit();
  return sp;
}

vllm::entrypoints::EngineParams BaseParams() {
  vllm::entrypoints::EngineParams p;
  p.block_size = 32;
  p.num_blocks = 256;
  p.max_num_seqs = 1;
  return p;
}

}  // namespace

TEST_CASE("gguf mtp: spec-ON equals spec-OFF and the drafter is alive") {
  const std::string model = MtpGgufPath();
  if (model.empty() || !fs::exists(model)) {
    MESSAGE("SKIP: set VLLM_MTP_GGUF_MODEL to a Qwen3.5/3.6 GGUF converted "
            "WITH the MTP head (not --no-mtp)");
    return;
  }

  // (1) Baseline: the same file, no speculation. This is the reference the
  // spec-ON run must reproduce exactly.
  std::vector<int32_t> baseline_ids;
  {
    auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model,
                                                                BaseParams());
    const vllm::RequestOutput out =
        loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gguf-mtp-off");
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    baseline_ids = out.outputs[0].token_ids;
    REQUIRE(!baseline_ids.empty());
    MESSAGE("spec-OFF produced " << baseline_ids.size() << " tokens: \""
                                 << out.outputs[0].text << "\"");
  }

  // (2) Same file, speculation ON. num_speculative_tokens is left unset so the
  // engine resolves k from the checkpoint's own head depth - which is the value
  // this row taught HfConfigFromGguf to republish, so an unset k also exercises
  // that plumbing rather than papering over it with an explicit number.
  vllm::entrypoints::EngineParams spec_params = BaseParams();
  spec_params.speculative_config =
      vllm::ParseSpeculativeConfigJson(R"({"method":"mtp"})");

  auto loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(model, spec_params);
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gguf-mtp-on");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& spec_ids = out.outputs[0].token_ids;

  MESSAGE("spec-ON produced " << spec_ids.size() << " tokens: \""
                              << out.outputs[0].text << "\"");
  MESSAGE("drafts proposed/accepted: "
          << loaded->runner().spec_drafts_proposed() << "/"
          << loaded->runner().spec_drafts_accepted());

  // (a) Exactness. Greedy speculative decoding rewrites HOW tokens are produced,
  // never WHICH.
  CHECK(spec_ids == baseline_ids);

  // (b) The drafter actually ran and was believed at least once. Without this,
  // (a) is satisfied by a head that proposes garbage the target rejects every
  // step - identical output, zero speedup, and a silently broken loader.
  CHECK(loaded->runner().spec_drafts_accepted() > 0);
}

// MINIMAL REPRODUCER for `CPU-SPEC-DIVERGENCE`, deliberately with NO MTP head
// and NO draft model of any kind.
//
// `ngram` speculation proposes from the prompt's own suffix history. It sets a
// SpeculativeConfig, so the engine takes the SAME target-side speculative path
// (widened KV / conv row, spec metadata, spec conv update) as MTP - but it loads
// no head, reads no `nextn` tensors, and touches nothing SPEC-MTP-GGUF added. If
// this ALSO diverges, the defect is proven independent of MTP and of the GGUF
// head loader, and this becomes the reproducer to fix the engine against.
TEST_CASE("cpu spec divergence: ngram needs no draft head and still diverges") {
  const std::string model = MtpGgufPath();
  if (model.empty() || !fs::exists(model)) {
    MESSAGE("SKIP: set VLLM_MTP_GGUF_MODEL");
    return;
  }

  std::vector<int32_t> baseline_ids;
  {
    auto loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(model, BaseParams());
    const vllm::RequestOutput out =
        loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "ngram-off");
    REQUIRE(out.finished);
    baseline_ids = out.outputs[0].token_ids;
  }

  vllm::entrypoints::EngineParams spec_params = BaseParams();
  spec_params.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"method":"ngram","num_speculative_tokens":2})");

  auto loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(model, spec_params);
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "ngram-on");
  REQUIRE(out.finished);

  MESSAGE("ngram spec-ON: \"" << out.outputs[0].text << "\"");
  // Same exactness bar: greedy speculation never changes WHICH tokens come out.
  CHECK(out.outputs[0].token_ids == baseline_ids);
}

// DEVICE-DELTA PROBE: what the CPU-vs-GPU token difference actually IS.
//
// CPU-vs-GPU token equality is NOT this row's bar and is not generally
// expected. The binding bar is spec-ON == spec-OFF WITHIN one device (the two
// cases above). Across devices the kernels, the accumulation order and the
// rounding differ, so a greedy sequence legitimately forks at the first
// APPROXIMATE TIE - the same reason this repo gates several models on a
// near-tie-robust form rather than a strict one. What the row does owe is
// EVIDENCE for which of the two it is, so this probe MEASURES the margin
// instead of arguing about it.
//
// It runs spec-OFF ONLY. Speculation is deliberately absent, so whatever the
// probe shows can be neither blamed on nor credited to the draft head: this
// isolates the plain forward.
//
// Why the two devices' rows are comparable at the ROOT divergence, and only
// there: at the first position where the sampled tokens differ, both arms have
// consumed a BIT-IDENTICAL token prefix (the prompt plus the agreed tokens), so
// each arm's row is that same prefix's next-token distribution and the
// comparison is teacher-forced by construction. From the NEXT position on the
// prefixes differ and the rows are no longer comparable - which is exactly why
// a root-divergence margin, not a token diff count, is the thing to read.
//
// The repo's ratified near-tie band is 0.5 nats (`kNearTieMnats = 500`): a root
// margin inside it is bf16 rounding at a coin flip, one outside it is a
// forward-path defect that must be bisected.
//
// DOUBLE-GATED (asset + `VLLM_MTP_GGUF_PROBE=1`) so the two binding cases above
// keep their exact runtime, memory footprint and behaviour.
TEST_CASE("gguf mtp probe: spec-OFF per-position logprob margins") {
  const std::string model = MtpGgufPath();
  if (model.empty() || !fs::exists(model)) {
    MESSAGE("SKIP: set VLLM_MTP_GGUF_MODEL");
    return;
  }
  if (std::getenv("VLLM_MTP_GGUF_PROBE") == nullptr) {
    MESSAGE("SKIP: set VLLM_MTP_GGUF_PROBE=1 for the CPU-vs-GPU delta probe");
    return;
  }

  auto loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(model, BaseParams());
  const vllm::RequestOutput out = loaded->engine().generate(
      kPrompt, GreedyWithLogprobs(kMaxTokens, kProbeTopK), "gguf-mtp-probe");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const vllm::CompletionOutput& co = out.outputs[0];

  MESSAGE("PROBE text: \"" << co.text << "\"");
  std::string ids;
  for (const int32_t id : co.token_ids) ids += std::to_string(id) + " ";
  MESSAGE("PROBE ids: " << ids);

  REQUIRE(co.logprobs.has_value());
  REQUIRE(co.logprobs->size() == co.token_ids.size());

  // One greppable line per position: the sampled id, then every alternative as
  // id:rank:logprob in the sampler's own insertion order (sampled first, then
  // rank 1..k). Rank 1 and rank 2 give the margin at that position; the ranked
  // list lets the OTHER device's pick be looked up by id.
  for (std::size_t i = 0; i < co.logprobs->size(); ++i) {
    const vllm::LogprobsOnePosition& pos = (*co.logprobs)[i];
    std::string row;
    for (const int32_t id : pos.order) {
      const vllm::Logprob* lp = pos.find(id);
      REQUIRE(lp != nullptr);
      row += std::to_string(id) + ":" +
             std::to_string(lp->rank.value_or(-1)) + ":" +
             std::to_string(lp->logprob) + " ";
    }
    MESSAGE("PROBE pos=" << i << " sampled=" << co.token_ids[i] << " | "
                         << row);
  }
}
