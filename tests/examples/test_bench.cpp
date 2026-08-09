// Smoke test for the M2.1 benchmark harness (examples/bench/bench_core.h): drive
// the SYNTHETIC CPU engine through the production AsyncLLM measurement loop
// and assert it produces sane metrics. The NUMBERS are meaningless (toy weights)
// — this asserts the HARNESS: all N requests finish, throughput > 0, TTFT > 0,
// and the token accounting is coherent. The real parity numbers come from a GB10
// run with --model (dgx-pending), which this same code path drives.
#include "bench_core.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

using vllm::bench::BenchConfig;
using vllm::bench::BenchResult;
using vllm::bench::DispatchBenchPromptAdmission;
using vllm::bench::RunBench;

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, std::optional<std::string> value) : name_(name) {
    if (const char* previous = std::getenv(name)) previous_ = previous;
    if (value.has_value()) {
      REQUIRE(::setenv(name, value->c_str(), /*overwrite=*/1) == 0);
    } else {
      REQUIRE(::unsetenv(name) == 0);
    }
  }

  ~ScopedEnv() {
    if (previous_.has_value()) {
      (void)::setenv(name_.c_str(), previous_->c_str(), /*overwrite=*/1);
    } else {
      (void)::unsetenv(name_.c_str());
    }
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

struct AdmissionObservation {
  int result = 0;
  int pretokenized_calls = 0;
  int string_calls = 0;
};

AdmissionObservation ObserveAdmission(const char* env_value) {
  AdmissionObservation observed;
  observed.result = DispatchBenchPromptAdmission(
      env_value,
      [&]() {
        ++observed.pretokenized_calls;
        return 17;
      },
      [&]() {
        ++observed.string_calls;
        return 29;
      });
  return observed;
}

}  // namespace

TEST_CASE("bench: pretokenized admission dispatch is default-on with exact rollback") {
  const AdmissionObservation unset = ObserveAdmission(nullptr);
  CHECK(unset.result == 17);
  CHECK(unset.pretokenized_calls == 1);
  CHECK(unset.string_calls == 0);

  const AdmissionObservation enabled = ObserveAdmission("1");
  CHECK(enabled.result == 17);
  CHECK(enabled.pretokenized_calls == 1);
  CHECK(enabled.string_calls == 0);

  const AdmissionObservation rollback = ObserveAdmission("0");
  CHECK(rollback.result == 29);
  CHECK(rollback.pretokenized_calls == 0);
  CHECK(rollback.string_calls == 1);

  // Invalid spellings keep the safe, production-parity default rather than
  // silently restoring timed tokenization.
  const AdmissionObservation invalid = ObserveAdmission("banana");
  CHECK(invalid.result == 17);
  CHECK(invalid.pretokenized_calls == 1);
  CHECK(invalid.string_calls == 0);
}

TEST_CASE("bench: synthetic engine completes all requests with sane metrics") {
  BenchConfig cfg;
  cfg.num_prompts = 8;
  cfg.input_len = 16;
  cfg.output_len = 16;
  cfg.concurrency = 4;
  cfg.seed = 123;
  cfg.temperature = 0.0;  // greedy => deterministic, exactly output_len tokens.

  const BenchResult r = RunBench(cfg);

  // The comparison harness must exercise the production AsyncLLM frontend.
  // Before the SERVE-CLI-BENCH B1 repair it called synchronous
  // LoadedEngine::engine() even when async scheduling resolved enabled.
  CHECK(r.async_frontend);
  CHECK(r.max_concurrent_batches >= 1);

  // All N requests finished through the engine loop.
  CHECK(r.completed == cfg.num_prompts);
  // Wall time advanced and throughput is positive.
  CHECK(r.duration_s > 0.0);
  CHECK(r.request_throughput > 0.0);
  CHECK(r.output_throughput > 0.0);
  CHECK(r.total_token_throughput > 0.0);
  CHECK(r.input_throughput > 0.0);
  // Token accounting: greedy w/ no eos => exactly output_len tokens per request.
  CHECK(r.total_output == static_cast<int64_t>(cfg.num_prompts) * cfg.output_len);
  REQUIRE(r.output_token_ids.size() == static_cast<size_t>(cfg.num_prompts));
  for (const auto& ids : r.output_token_ids) {
    CHECK(ids.size() == static_cast<size_t>(cfg.output_len));
  }
  CHECK(r.total_input > 0);
  CHECK(r.total_token_throughput ==
        doctest::Approx(r.input_throughput + r.output_throughput));
  // Latency metrics are engaged (first token observed => TTFT > 0; multi-token
  // decode => TPOT/ITL > 0).
  CHECK(r.mean_ttft_ms > 0.0);
  CHECK(r.mean_tpot_ms > 0.0);
  CHECK(r.mean_itl_ms > 0.0);
  CHECK(r.mean_e2el_ms >= r.mean_ttft_ms);
  CHECK(r.mean_per_stream_decode > 0.0);
}

TEST_CASE("bench: output token IDs serialize in submission order") {
  BenchConfig cfg;
  cfg.num_prompts = 3;
  cfg.input_len = 8;
  cfg.output_len = 4;
  cfg.concurrency = 2;

  const BenchResult r = RunBench(cfg);
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "vllm_cpp_bench_token_ids.json";
  vllm::bench::WriteOutputTokenIds(path.string(), r);

  nlohmann::json saved;
  std::ifstream(path) >> saved;
  std::filesystem::remove(path);
  REQUIRE(saved.is_array());
  REQUIRE(saved.size() == 3);
  CHECK(saved == nlohmann::json(r.output_token_ids));
}

TEST_CASE("bench: concurrency=1 (serial) also completes and is coherent") {
  BenchConfig cfg;
  cfg.num_prompts = 4;
  cfg.input_len = 8;
  cfg.output_len = 8;
  cfg.concurrency = 1;
  cfg.seed = 7;

  const BenchResult r = RunBench(cfg);

  CHECK(r.completed == 4);
  CHECK(r.total_output == 4 * 8);
  CHECK(r.request_throughput > 0.0);
  CHECK(r.mean_ttft_ms > 0.0);
}

TEST_CASE("bench: ShareGPT dataset supplies exact prompts") {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "vllm_cpp_bench_sharegpt.json";
  {
    std::ofstream out(path);
    out << R"json([{"conversations":[{"from":"human","value":"hello world"},{"from":"gpt","value":"ok"}]},{"conversations":[{"from":"human","value":"world hello"},{"from":"gpt","value":"ok"}]}])json";
  }

  BenchConfig cfg;
  cfg.dataset_path = path.string();
  cfg.num_prompts = 2;
  cfg.input_len = 8;
  cfg.output_len = 4;
  cfg.concurrency = 2;
  const BenchResult r = RunBench(cfg);
  std::filesystem::remove(path);

  CHECK(r.completed == 2);
  CHECK(r.total_output == 8);
  CHECK(r.total_input > 0);
}

TEST_CASE("bench: pretokenized default preserves prompt and output token IDs") {
  BenchConfig cfg;
  cfg.num_prompts = 5;
  cfg.input_len = 12;
  cfg.output_len = 6;
  cfg.concurrency = 3;
  cfg.seed = 31;
  cfg.temperature = 0.0;

  BenchResult pretokenized;
  {
    ScopedEnv env("VT_BENCH_PRETOKENIZE", std::nullopt);
    pretokenized = RunBench(cfg);
  }

  BenchResult timed_string;
  {
    ScopedEnv env("VT_BENCH_PRETOKENIZE", std::string("0"));
    timed_string = RunBench(cfg);
  }

  CHECK(pretokenized.pretokenized_admission);
  CHECK_FALSE(timed_string.pretokenized_admission);
  REQUIRE(pretokenized.prompt_token_ids.size() ==
          static_cast<size_t>(cfg.num_prompts));
  REQUIRE(timed_string.prompt_token_ids.size() ==
          static_cast<size_t>(cfg.num_prompts));
  CHECK(pretokenized.prompt_token_ids == timed_string.prompt_token_ids);
  CHECK(pretokenized.output_token_ids == timed_string.output_token_ids);
  CHECK(pretokenized.total_input == timed_string.total_input);
  CHECK(pretokenized.total_output == timed_string.total_output);
}
