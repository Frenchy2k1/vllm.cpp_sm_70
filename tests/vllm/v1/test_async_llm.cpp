// Ported from: tests/v1/engine/test_async_llm.py @ e24d1b24
// W2 T-unit cases from specs/async-serving.md: test_load (:109), test_abort
// (:157), test_multi_abort (:228), test_finished_flag (:306),
// test_mid_stream_cancellation (:340), and test_abort_final_output (:598).
//
// The upstream suite drives a tiny real model under asyncio. This C++ port
// drives the real Scheduler/EngineCoreProc/OutputProcessor pipeline with the
// same canned one-token-per-step ModelRunnerBase seam used by
// test_engine_core_proc.cpp. It tests the asynchronous queue/collector and
// request lifecycle without GPU/model dependencies; GPU token-exactness and
// live-SSE arrival timing remain W2's explicit G1/G3 gates.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/metrics/loggers.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::AsyncLLM;
using vllm::v1::AsyncRequest;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerBase;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::OutputProcessor;
using vllm::v1::metrics::PrometheusStatLogger;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

constexpr int32_t kCannedToken = 17;  // fixture token " world"

class RunnerStub : public ModelRunnerBase {
 public:
  explicit RunnerStub(std::chrono::microseconds delay = {}) : delay_(delay) {}

  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    stashed_output_ = scheduler_output;
    return std::nullopt;
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    if (delay_.count() != 0) std::this_thread::sleep_for(delay_);
    ModelRunnerOutput output;
    int index = 0;
    for (const auto& [request_id, num_tokens] :
         stashed_output_.num_scheduled_tokens) {
      (void)num_tokens;
      output.req_ids.push_back(request_id);
      output.req_id_to_index[request_id] = index++;
      output.sampled_token_ids.push_back({kCannedToken});
    }
    return output;
  }

 private:
  SchedulerOutput stashed_output_;
  std::chrono::microseconds delay_;
};

// A runner whose engine-thread sampling hook throws, so the busy-loop fatal
// guard (core_client.cpp) is the only place holding the true root-cause string.
class ThrowingRunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    (void)scheduler_output;
    return std::nullopt;
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& /*grammar_output*/) override {
    throw std::runtime_error(
        "vt: DIAG_ROOT_CAUSE_SENTINEL at qwen3_5.cpp:0");
  }
};

// Scope-guarded process-wide std::cerr redirect. Installed BEFORE the engine
// starts and restored only when this object is destroyed; sequencing the
// restore after the engine (and its joined threads) guarantees no engine-thread
// write races the rdbuf swap.
class CerrRedirect {
 public:
  explicit CerrRedirect(std::streambuf* target)
      : previous_(std::cerr.rdbuf(target)) {}
  ~CerrRedirect() { std::cerr.rdbuf(previous_); }
  CerrRedirect(const CerrRedirect&) = delete;
  CerrRedirect& operator=(const CerrRedirect&) = delete;

 private:
  std::streambuf* previous_;
};

std::unique_ptr<Scheduler> CreateScheduler() {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 128;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;

  KVCacheConfig kv;
  kv.num_blocks = 10000;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, 1, 1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv, 16,
                                     /*enable_caching=*/true);
}

Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_async_llm_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array();
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},    {"l", 2},     {"o", 3},
                {"w", 4},   {"r", 5},    {"d", 6},     {"Ġ", 7},
                {"1", 8},   {"2", 9},    {"ll", 10},   {"he", 11},
                {"llo", 12}, {"hello", 13}, {"Ġw", 14}, {"or", 15},
                {"orld", 16}, {"Ġworld", 17}, {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 19;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 20;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  {
    std::ofstream out(path, std::ios::binary);
    out << doc.dump();
  }
  Tokenizer tokenizer = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tokenizer;
}

HfConfig MakeConfig() {
  HfConfig config;
  config.max_position_embeddings = 8192;
  config.raw = json::object();
  return config;
}

SamplingParams Params(int max_tokens, RequestOutputKind output_kind) {
  SamplingParams params;
  params.max_tokens = max_tokens;
  params.output_kind = output_kind;
  params.temperature = 0.0;
  return params;
}

int Drain(AsyncLLM& engine, const AsyncRequest& request,
          std::vector<RequestOutput>* outputs = nullptr) {
  int tokens = 0;
  for (;;) {
    RequestOutput output = engine.get_output(request);
    REQUIRE(output.outputs.size() == 1);
    tokens += static_cast<int>(output.outputs[0].token_ids.size());
    if (outputs != nullptr) outputs->push_back(output);
    if (output.finished) return tokens;
  }
}

// Parse the scalar a Prometheus text line carries: find "<series> " and read
// the value to end of line. Same shape as test_llm_engine.cpp's helper, but
// this one RETURNS a sentinel instead of failing when the series is absent, so
// it is safe to call inside a poll loop.
double MetricValue(const std::string& text, const std::string& series) {
  const std::string needle = series + " ";
  const size_t p = text.find(needle);
  if (p == std::string::npos) return -1.0;
  const size_t v = p + needle.size();
  const size_t e = text.find('\n', v);
  return std::stod(text.substr(v, e - v));
}

void InitHash() {
  static bool initialized = false;
  if (!initialized) {
    init_none_hash(sha256_cbor);
    initialized = true;
  }
}

}  // namespace

TEST_CASE("async_llm test_load: concurrent requests all finish with unique ids") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  constexpr int kNumRequests = 32;
  constexpr int kTokens = 10;
  std::vector<AsyncRequest> requests;
  std::set<std::string> ids;
  for (int i = 0; i < kNumRequests; ++i) {
    const std::string id = "request-" + std::to_string(i);
    requests.push_back(engine.add_request(
        id, "hello", Params(kTokens, RequestOutputKind::kDelta)));
    ids.insert(requests.back().request_id);
  }
  CHECK(ids.size() == kNumRequests);
  for (const AsyncRequest& request : requests) {
    CHECK(Drain(engine, request) == kTokens);
  }
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm test_abort and test_multi_abort leave other requests healthy") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::microseconds(200));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest abort_a = engine.add_request(
      "abort-a", "hello", Params(100000, RequestOutputKind::kDelta));
  AsyncRequest survivor = engine.add_request(
      "survivor", "hello", Params(8, RequestOutputKind::kDelta));
  AsyncRequest abort_b = engine.add_request(
      "abort-b", "hello", Params(100000, RequestOutputKind::kFinalOnly));

  // Let abort-a produce at least one partial delta, then multi-abort both ids.
  RequestOutput partial = engine.get_output(abort_a);
  CHECK_FALSE(partial.finished);
  engine.abort(std::vector<std::string>{"abort-a", "abort-b"});

  RequestOutput final_a = engine.get_output(abort_a);
  CHECK(final_a.finished);
  CHECK(*final_a.outputs[0].finish_reason == "abort");
  RequestOutput final_b = engine.get_output(abort_b);
  CHECK(final_b.finished);
  CHECK(*final_b.outputs[0].finish_reason == "abort");
  CHECK(Drain(engine, survivor) == 8);
  CHECK_FALSE(engine.has_unfinished_requests());

  // Reusing an aborted external id is valid once cleanup completes.
  AsyncRequest reused = engine.add_request(
      "abort-a", "hello", Params(3, RequestOutputKind::kDelta));
  CHECK(Drain(engine, reused) == 3);
}

TEST_CASE("async_llm test_finished_flag and mid_stream_cancellation") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest request = engine.add_request(
      "cancel", "hello", Params(1000, RequestOutputKind::kDelta));
  std::vector<RequestOutput> seen;
  int tokens = 0;
  while (tokens < 5) {
    RequestOutput output_value = engine.get_output(request);
    CHECK_FALSE(output_value.finished);
    tokens += static_cast<int>(output_value.outputs[0].token_ids.size());
    seen.push_back(std::move(output_value));
  }
  engine.abort("cancel");
  RequestOutput final_output = engine.get_output(request);
  CHECK(final_output.finished);
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK(tokens >= 5);
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm test_abort_final_output returns terminal metadata") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest request = engine.add_request(
      "abort-final", "hello",
      Params(3000, RequestOutputKind::kFinalOnly));
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  engine.abort(request.request_id);
  RequestOutput final_output = engine.get_output(request);
  CHECK(final_output.finished);
  REQUIRE(final_output.outputs.size() == 1);
  REQUIRE(final_output.outputs[0].finish_reason.has_value());
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK_FALSE(final_output.outputs[0].stop_reason.has_value());
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm rejects a duplicate live request id without replacing its collector") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, *scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor));

  AsyncRequest original = engine.add_request(
      "duplicate", "hello", Params(1000, RequestOutputKind::kDelta));
  CHECK_THROWS_AS(
      engine.add_request("duplicate", "hello",
                         Params(1, RequestOutputKind::kFinalOnly)),
      std::invalid_argument);

  engine.abort(original.request_id);
  RequestOutput final_output = engine.get_output(original);
  CHECK(final_output.finished);
  REQUIRE(final_output.outputs.size() == 1);
  REQUIRE(final_output.outputs[0].finish_reason.has_value());
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK_FALSE(engine.has_unfinished_requests());
}

TEST_CASE("async_llm shutdown wakes an active request with a terminal abort") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();
  auto scheduler = CreateScheduler();
  RunnerStub runner(std::chrono::milliseconds(1));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);

  AsyncRequest request;
  {
    auto engine = std::make_unique<AsyncLLM>(
        input, *scheduler, executor, output,
        get_request_block_hasher(16, sha256_cbor));
    request = engine->add_request(
        "shutdown", "hello", Params(1000, RequestOutputKind::kDelta));
  }

  RequestOutput final_output = request.collector->get();
  CHECK(final_output.finished);
  REQUIRE(final_output.outputs.size() == 1);
  REQUIRE(final_output.outputs[0].finish_reason.has_value());
  CHECK(*final_output.outputs[0].finish_reason == "abort");
  CHECK_FALSE(output.has_unfinished_requests());
}

TEST_CASE("async_llm concurrent submission cannot publish after shutdown sweep") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();

  // Exercise both valid outcomes of the admission boundary repeatedly: the
  // request is accepted before shutdown and receives a terminal output, or
  // shutdown wins and add_request raises EngineDeadError. In neither case may
  // a collector be registered after shutdown's abort-all sweep and hang.
  for (int attempt = 0; attempt < 64; ++attempt) {
    auto scheduler = CreateScheduler();
    RunnerStub runner(std::chrono::milliseconds(1));
    Executor executor(runner);
    InputProcessor input(tokenizer, config);
    OutputProcessor output(&tokenizer);
    AsyncLLM engine(input, *scheduler, executor, output,
                    get_request_block_hasher(16, sha256_cbor));

    std::optional<AsyncRequest> accepted;
    std::exception_ptr submit_error;
    std::thread submitter([&] {
      try {
        accepted = engine.add_request(
            "shutdown-race", "hello",
            Params(1000, RequestOutputKind::kDelta));
      } catch (...) {
        submit_error = std::current_exception();
      }
    });

    engine.shutdown();
    submitter.join();

    if (accepted.has_value()) {
      RequestOutput terminal = accepted->collector->get();
      CHECK(terminal.finished);
    } else {
      REQUIRE(submit_error != nullptr);
      CHECK_THROWS_AS(std::rethrow_exception(submit_error),
                      vllm::v1::EngineDeadError);
    }
    CHECK_FALSE(output.has_unfinished_requests());
  }
}

TEST_CASE(
    "async_llm engine-thread guard logs the raw root cause before poisoning") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();

  // Capture everything the engine thread emits, but assert only after the
  // engine (and both its joined threads) are gone so the rdbuf restore cannot
  // race an in-flight write and so no doctest failure text is swallowed.
  std::ostringstream captured;
  bool saw_engine_dead = false;
  std::string second_add_what;
  bool second_add_threw = false;
  {
    CerrRedirect redirect(captured.rdbuf());
    auto scheduler = CreateScheduler();
    ThrowingRunnerStub runner;
    Executor executor(runner);
    InputProcessor input(tokenizer, config);
    OutputProcessor output(&tokenizer);
    AsyncLLM engine(input, *scheduler, executor, output,
                    get_request_block_hasher(16, sha256_cbor));

    AsyncRequest request = engine.add_request(
        "diag", "hello", Params(128, RequestOutputKind::kDelta));

    // Drive the consumer until the poisoned engine surfaces EngineDeadError.
    for (int i = 0; i < 2000 && !saw_engine_dead; ++i) {
      try {
        (void)engine.get_output(request);
      } catch (const vllm::v1::EngineDeadError&) {
        saw_engine_dead = true;
      } catch (...) {
        // Any other terminal exception also ends the drive loop.
        saw_engine_dead = true;
      }
    }

    // A subsequent submission must fast-fail with the generic wrapper, never
    // leaking the raw root cause into the client-facing error.
    try {
      (void)engine.add_request("diag-after", "hello",
                               Params(1, RequestOutputKind::kDelta));
    } catch (const vllm::v1::EngineDeadError& e) {
      second_add_threw = true;
      second_add_what = e.what();
    }
  }

  const std::string log = captured.str();
  CHECK(saw_engine_dead);
  CHECK(log.find("engine-fatal:") != std::string::npos);
  CHECK(log.find("DIAG_ROOT_CAUSE_SENTINEL") != std::string::npos);
  CHECK(second_add_threw);
  CHECK(second_add_what.find("EngineCore encountered an issue") !=
        std::string::npos);
  CHECK(second_add_what.find("DIAG_ROOT_CAUSE_SENTINEL") == std::string::npos);
}

// ─── Metrics on the ASYNC-SCHEDULING (depth-2 batch-queue) step path (#277) ───
// `AsyncLLM` with max_concurrent_batches=2 runs EngineCore::step_with_batch_queue
// instead of step(). Upstream stamps `scheduler_stats` and `timestamp` inside
// `Scheduler.update_from_output` (scheduler.py:1938-1951) and
// `EngineCoreOutputs.__post_init__` (engine/__init__.py:249-251) — a path BOTH
// step functions share, so upstream's batch-queue path carries them too. Our
// port stamped them at EngineCore::step() only, so the depth-2 serving path
// (which is what `LoadedEngine` resolves to when the runner supports async
// scheduling) published a default-constructed SchedulerStats and timestamp 0.
//
// The gauge assertion is a POLL rather than a single scrape: the running batch
// is only observable while a request is in flight, and the delayed stub keeps
// one in flight for as long as the poll needs. It also exercises the recorder
// mutex, since Expose() here runs on the test thread concurrently with the
// output handler's Record().
//
// RED before the stamp: the running gauge never leaves 0 and the poll times out.
TEST_CASE("async_llm: depth-2 batch-queue step publishes live scheduler stats") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();

  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.async_scheduling = true;
  KVCacheConfig kv;
  kv.num_blocks = 10000;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, 1, 1, DType::kF32));
  vllm::v1::AsyncScheduler scheduler(cfg, kv, /*block_size=*/16,
                                     /*enable_caching=*/true);

  // 500 us per step keeps a long request in flight for the whole poll window.
  RunnerStub runner(std::chrono::microseconds(500));
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor),
                  /*shutdown_timeout_s=*/0, /*max_concurrent_batches=*/2);
  PrometheusStatLogger logger("m", /*max_model_len=*/8192);
  engine.set_stat_logger(&logger);

  AsyncRequest long_request = engine.add_request(
      "in-flight", "hello", Params(100000, RequestOutputKind::kDelta));

  // Poll until the running gauge reports the in-flight request. Generous
  // budget: this box is shared and the assertion is about the value ever being
  // published at all, not about how fast it appears.
  bool saw_running = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (std::chrono::steady_clock::now() < deadline) {
    if (MetricValue(logger.Expose(),
                    "vllm:num_requests_running{model_name=\"m\",engine=\"0\"}") ==
        1.0) {
      saw_running = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  CHECK(saw_running);

  engine.abort(long_request.request_id);
  engine.shutdown();
}

// The token counters and the engine-core-timestamp-derived histograms on the
// same depth-2 path. Without the timestamp stamp the TTFT/e2e observations are
// `0 - arrival_time`, i.e. strongly NEGATIVE, so a positive _sum is the exact
// discriminator. Counters/observation counts additionally gate the fold itself.
TEST_CASE("async_llm: depth-2 batch-queue step folds IterationStats") {
  InitHash();
  Tokenizer tokenizer = BuildFixture();
  HfConfig config = MakeConfig();

  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.async_scheduling = true;
  KVCacheConfig kv;
  kv.num_blocks = 10000;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, 1, 1, DType::kF32));
  vllm::v1::AsyncScheduler scheduler(cfg, kv, /*block_size=*/16,
                                     /*enable_caching=*/true);

  RunnerStub runner;
  Executor executor(runner);
  InputProcessor input(tokenizer, config);
  OutputProcessor output(&tokenizer);
  AsyncLLM engine(input, scheduler, executor, output,
                  get_request_block_hasher(16, sha256_cbor),
                  /*shutdown_timeout_s=*/0, /*max_concurrent_batches=*/2);
  PrometheusStatLogger logger("m", /*max_model_len=*/8192);
  engine.set_stat_logger(&logger);

  constexpr int kTokens = 8;
  AsyncRequest request = engine.add_request(
      "r", "hello", Params(kTokens, RequestOutputKind::kDelta));
  CHECK(Drain(engine, request) == kTokens);
  // Join the output handler so every fold has retired before the scrape.
  engine.shutdown();

  const std::string t = logger.Expose();
  const char* kLabels = "{model_name=\"m\",engine=\"0\"}";
  CHECK(MetricValue(t, std::string("vllm:prompt_tokens_total") + kLabels) == 1.0);
  CHECK(MetricValue(t, std::string("vllm:generation_tokens_total") + kLabels) ==
        static_cast<double>(kTokens));
  CHECK(MetricValue(t, std::string("vllm:time_to_first_token_seconds_count") +
                           kLabels) == 1.0);
  CHECK(MetricValue(t, std::string("vllm:e2e_request_latency_seconds_count") +
                           kLabels) == 1.0);
  // The discriminator for the timestamp stamp: an unstamped (0.0) engine-core
  // timestamp makes both of these negative.
  CHECK(MetricValue(t, std::string("vllm:time_to_first_token_seconds_sum") +
                           kLabels) > 0.0);
  CHECK(MetricValue(t, std::string("vllm:e2e_request_latency_seconds_sum") +
                           kLabels) > 0.0);
}
