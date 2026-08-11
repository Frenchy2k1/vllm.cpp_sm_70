// Two engines in ONE process must decode identically (issue #237, dgx-only).
//
// This is the arm that proves the reported corruption is gone, as opposed to
// test_moe_resident_lifetime, which pins the ownership invariant on the CPU.
//
// WHY THIS MODEL. Qwen3-Coder-30B-A3B is the checkpoint the bf16 fast-MoE
// resident path was built for (see MoeBf16Resident in qwen3_5.cpp, "Qwen3-Coder
// Qwen3MoeForCausalLM, W5"). Its experts are ~94% of the checkpoint and every
// one of them goes through the per-expert device-pointer array that used to be
// cached in a process-lifetime map keyed on the ADDRESS of the MoeBlockWeights.
// A smaller or non-MoE model never touches that code and would pass whether or
// not the bug is present -- which is exactly how a first attempt at gating this
// produced a meaningless 10/10 on the pre-fix tree.
//
// WHY IT IS DETERMINISTIC. The original repro (richiejp, issue #237) needed a
// specific ctest ordering plus a benchmark fixture, and still only failed
// intermittently, because it depended on the allocator happening to hand the
// second engine an address the first had freed. This drives that directly: build
// an engine, decode greedily, DESTROY it, then build another in the same process
// and decode the same prompt. Identical construction in identical order is
// precisely the case where an allocator reuses addresses, so the collision the
// old map could not detect is the expected case here rather than a rare one.
//
// WHAT FAILURE LOOKS LIKE. On the pre-fix tree the second engine inherits an
// entry already marked ready whose device pointers were freed with the first
// engine. Nothing faults -- the CUDA context is never destroyed, so the memory
// stays mapped -- so the second decode returns corrupted or zeroed token ids
// while the first is correct. Hence both checks below: the sequences must match,
// AND the second must not be degenerate, because two runs of all-zeros would
// "match" while being exactly the bug.
//
// ROUNDS. Three engines, not two: round 2 is where reuse is most likely, and
// round 3 catches a cache that survives one destruction but not two.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <system_error>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

std::string FindQwen3CoderSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/"
      "models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;  // argmax => the same prompt must give the same ids
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// One full engine lifecycle: load, decode, destroy. The engine is scoped so the
// weights -- and now the resident state they own -- are released before the
// caller builds the next one.
std::vector<int32_t> DecodeOnce(const std::string& snap, const std::string& tag) {
  std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});
  const vllm::RequestOutput out = le->engine().generate(
      "def fibonacci(n):", Greedy(16), tag);
  return out.outputs[0].token_ids;
}

}  // namespace

TEST_CASE("qwen3-coder: rebuilding the engine in one process decodes identically (#237)") {
  const std::string snap = FindQwen3CoderSnapshot();
  if (snap.empty()) {
    MESSAGE("SKIP: Qwen3-Coder-30B-A3B snapshot absent (dgx-only gate)");
    return;
  }

  const std::vector<int32_t> first = DecodeOnce(snap, "engine1");
  REQUIRE_FALSE(first.empty());

  // The reported symptom was zeroed ids, so a baseline of all-zeros would make
  // every later comparison vacuously true. Establish the baseline is real first.
  bool baseline_nonzero = false;
  for (const int32_t t : first) {
    if (t != 0) { baseline_nonzero = true; break; }
  }
  REQUIRE(baseline_nonzero);

  for (int round = 2; round <= 3; ++round) {
    const std::vector<int32_t> again =
        DecodeOnce(snap, "engine" + std::to_string(round));
    INFO("round " << round << " of engine construction");
    REQUIRE(again.size() == first.size());
    CHECK(again == first);

    bool nonzero = false;
    for (const int32_t t : again) {
      if (t != 0) { nonzero = true; break; }
    }
    CHECK(nonzero);
  }
}
