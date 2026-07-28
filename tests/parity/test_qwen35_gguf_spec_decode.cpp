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
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

constexpr int kMaxTokens = 24;
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
