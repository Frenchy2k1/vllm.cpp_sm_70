// vllm.cpp original (checkpoint-gated spec-decode gate); no upstream mirror.
//
// THE 27B ngram SPECULATIVE-DECODE e2e GATE (SPEC-NGRAM, ROAD-V1-D3). Drives the
// committed vLLM-ngram-ON golden prompts through the FULL paged engine with the
// draft-FREE n-gram proposer turned ON via EngineParams::speculative_config
// ('{"method":"ngram","num_speculative_tokens":k,"prompt_lookup_min":m,
//   "prompt_lookup_max":n}'), and asserts:
//
//   (a) STRICT token identity: our ngram-ON greedy continuation == vLLM-ngram-ON
//       (the committed ngram_27b_spec_on.json golden). Unlike DFlash's k=16 block
//       diffusion, ngram uses the SAME greedy rejection sampler as MTP, which is
//       EXACTNESS-PRESERVING (accept a draft iff it equals the target argmax), so
//       vLLM-ngram-ON == vLLM-spec-OFF == our-ngram-ON position for position — the
//       full three-way identity, exact on EVERY prompt (no near-tie divergence).
//   (b) MEASURED NONZERO ACCEPTANCE ~ vLLM's. Token identity ALONE passes on a
//       drafter that never matches (the I5e dead-drafter trap), so nonzero
//       acceptance in vLLM's neighborhood is a REQUIRED companion. The repetitive
//       golden prompts guarantee the suffix-ngram matcher fires.
//
// Checkpoint-GATED + dgx-only: the NVFP4 27B target + the CUDA spec verify kernels
// live only on dgx.casa (GB10); on the CPU dev box / CI the body emits a loud SKIP
// (compiles + links on CPU). Golden gen: scripts/spec/ngram_27b_golden.py.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "hf_snapshot.h"

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// Resolve the 27B NVFP4 snapshot dir (the bf16 unsloth NVFP4 single-file snapshot,
// matching the ngram golden + the 27B SACRED gate), preferring the single
// model.safetensors snapshot over a sharded FP8-lm_head re-quant.
std::string Snap27B() {
  // Pinned to the goldens' revision; see tests/parity/hf_snapshot.h.
  return parity::Qwen27NvfP4Snapshot();
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

TEST_CASE("qwen27 ngram spec-decode e2e gate (dgx-only, 27B draft-free ngram)") {
  const std::string target = Snap27B();
  const fs::path golden_path =
      fs::path(PARITY_GOLDENS_DIR) / "ngram_27b" / "ngram_27b_spec_on.json";
  if (target.empty() || !fs::exists(golden_path)) {
    MESSAGE("27B NVFP4 target / ngram golden absent; skipping (dgx-only)");
    return;
  }

  std::ifstream gf(golden_path.string());
  json golden;
  gf >> golden;
  const int k = golden.value("num_speculative_tokens", 3);
  const int min_n = golden.value("prompt_lookup_min", 2);
  const int max_n = golden.value("prompt_lookup_max", 3);
  const auto& records = golden.at("records");

  // Turn the draft-FREE ngram proposer ON.
  vllm::entrypoints::EngineParams params;
  params.max_num_seqs = 2;  // bound the k+1 GDN spec-state slots.
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      std::string("{\"method\":\"ngram\",\"num_speculative_tokens\":") +
      std::to_string(k) + ",\"prompt_lookup_min\":" + std::to_string(min_n) +
      ",\"prompt_lookup_max\":" + std::to_string(max_n) + "}");

  MESSAGE("qwen27_ngram: loading 27B target " << target << " (ngram k=" << k
          << ", window [" << min_n << "," << max_n << "])...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(target, params);

  int prompt_idx = 0;
  int total_prompts = 0, exact_prompts = 0;
  int64_t total_acc = 0, total_prop = 0;
  for (const auto& rec : records) {
    const std::string prompt = rec.at("prompt").get<std::string>();
    const std::vector<int32_t> want_out =
        rec.at("output_token_ids").get<std::vector<int32_t>>();
    const std::vector<int32_t> want_prompt_ids =
        rec.at("prompt_token_ids").get<std::vector<int32_t>>();
    const int max_tokens = static_cast<int>(want_out.size());
    const int want_acc_delta = rec.value("num_accepted_delta", 0);
    const double want_acc_len = rec.value("acceptance_len", 0.0);
    ++total_prompts;

    const int64_t acc_before = loaded->runner().spec_drafts_accepted();
    const int64_t prop_before = loaded->runner().spec_drafts_proposed();

    const std::string rid = "ngram" + std::to_string(prompt_idx++);
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
    total_acc += acc;
    total_prop += prop;

    CHECK(out.prompt_token_ids == want_prompt_ids);
    const bool exact = (got == want_out);
    if (exact) ++exact_prompts;
    size_t shared = 0;
    if (!exact) {
      const size_t n = std::min(got.size(), want_out.size());
      while (shared < n && got[shared] == want_out[shared]) ++shared;
    }
    MESSAGE("qwen27_ngram prompt[" << (prompt_idx - 1) << "] \"" << prompt
            << "\": exact=" << (exact ? "YES" : "NO") << " (shared=" << shared
            << ")  drafts " << acc << "/" << prop << " accepted (golden "
            << "accepted_delta=" << want_acc_delta << ", acceptance_len="
            << want_acc_len << ")  text=\"" << out.outputs[0].text << "\"");

    // (a) STRICT token identity vs vLLM-ngram-ON. ngram greedy is exactness-
    // preserving, so this is exact on EVERY prompt.
    CHECK(got == want_out);

    // (b) Per-prompt acceptance matches vLLM's within a small band (the ngram
    // matcher is deterministic over the identical committed context, so the
    // accepted count tracks vLLM's closely). Band scales with the count (a
    // high-acceptance repetitive prompt reaches ~36; telemetry edge effects at
    // the generation boundary are a few tokens).
    const int64_t band = std::max<int64_t>(4, want_acc_delta / 4);
    CHECK(std::llabs(acc - static_cast<int64_t>(want_acc_delta)) <= band);
  }

  MESSAGE("qwen27_ngram: " << exact_prompts << "/" << total_prompts
          << " prompts STRICT token-exact vs vLLM-ngram-ON; total drafts "
          << total_acc << "/" << total_prop << " accepted");

  // Exactness-preserving: every prompt is token-identical to vLLM-ngram-ON.
  CHECK(exact_prompts == total_prompts);
  // Dead-drafter trap: the loop RAN and the draft-free proposer is alive.
  CHECK(total_prop > 0);
  CHECK(total_acc > 0);
}
