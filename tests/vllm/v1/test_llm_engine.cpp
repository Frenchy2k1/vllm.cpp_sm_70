// End-to-end tests for the V1 LLMEngine (M1.8 Task 6) — the whole loop wired:
// InputProcessor -> EngineCore(Scheduler + Executor over the real GPUModelRunner)
// -> OutputProcessor -> RequestOutput. Ported from vllm/v1/engine/llm_engine.py @
// e24d1b24 (add_request/step) + the LLM.generate driver.
//
// Drives a SMALL SYNTHETIC hybrid-MoE Qwen3.6 model on CPU (the real 35B greedy
// through the full paged loop on dgx is the milestone acceptance gate — dgx-
// pending). This proves the pieces actually CONNECT and produce a deterministic
// stream. Cases:
//   1. greedy determinism + termination: generate(prompt, greedy, max_tokens=N)
//      yields exactly N tokens, the loop terminates, and two runs of the same
//      prompt+greedy produce the SAME tokens.
//   2. 2-request concurrent batch: two requests added, stepped together, each
//      gets its own correct stream (matches its standalone single-request run).
//   3. streaming (DELTA) vs non-streaming (CUMULATIVE): the concatenated deltas
//      equal the cumulative full text for the same prompt+greedy.
//   4. stop on max_tokens (LENGTH finish) end to end.
//   5. stop on a stop STRING: the OutputProcessor's string-stop -> reqs_to_abort
//      -> EngineCore aborts -> the loop ends (finish_reason "stop").
//
// The synthetic model uses vocab_size == the tiny BPE fixture's assigned id count
// (0..23, no holes) so every greedy argmax is decodable. The KV cache config uses
// a UNIFIED block_size == max_model_len == hash_block_size (the hybrid KV
// coordinator requires every group's block_size == hash_block_size); with prompts
// far shorter than a block, prefix caching stays inert (no block ever completes).
#include "vllm/v1/engine/llm_engine.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/scheduler.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/metrics/loggers.h"
#include "vllm/v1/metrics/prometheus.h"
#include "vllm/v1/metrics/stats.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5MoeWeights;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::EngineCore;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GPUModelRunner;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::LLMEngine;
using vllm::v1::MambaSpec;
using vllm::v1::OutputProcessor;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vllm::v1::metrics::PrometheusStatLogger;
using vt::DType;

namespace {

// ─── Synthetic weights (mirrors tests/vllm/v1/worker/test_runner.cpp) ────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

// Vocab size == the tiny BPE fixture's assigned ids (0..23): every greedy argmax
// over the lm_head is decodable by the detokenizer (no vocab holes).
constexpr int kVocab = 24;
constexpr int kBlockSize = 32;      // == max_model_len == hash_block_size (hybrid)
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 32;

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = kVocab;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();  // no eos_token_id -> generation runs to max_tokens.
  return c;
}

vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// Hybrid KV config: one full-attn group + one mamba (GDN) group, UNIFIED
// block_size == max_model_len == hash_block_size (hybrid coordinator constraint).
KVCacheConfig MakeKvConfig(const HfConfig& c) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa3"},
      std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh, DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn0", "gdn1", "gdn2"},
      std::make_shared<MambaSpec>(
          kBlockSize,
          std::vector<std::vector<int64_t>>{{conv_dim, Kw - 1},
                                            {Hv, Dv, Dk}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The tiny oracle-verified BPE fixture from tests/vllm/v1/test_output_processor
// (ids 0..23, no holes): "hello"=13, " world"=17, "1"=8, "2"=9, ...
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_llmengine_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
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
  json vocab = {{"h", 0},   {"e", 1},    {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},    {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11},  {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15},  {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
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
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

// Owns a fully-wired LLMEngine stack over the shared const config/weights/
// tokenizer. Members are declared in dependency order (Scheduler + runner ->
// Executor -> EngineCore; InputProcessor/OutputProcessor -> LLMEngine) so the
// by-reference seams stay valid for the stack's lifetime.
struct Harness {
  Harness(const HfConfig& c, const Qwen3_5MoeWeights& w, const Tokenizer& tok,
          int max_num_reqs = 8)
      : scheduler(MakeSchedulerConfig(), MakeKvConfig(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, w, MakeKvConfig(c), Q(), max_num_reqs, kMaxModelLen,
               /*max_num_batched_tokens=*/kMaxModelLen * max_num_reqs),
        executor(runner),
        engine_core(scheduler, executor),
        input_processor(tok, c),
        output_processor(&tok),
        engine(input_processor, engine_core, output_processor, Hasher()) {}

  static SchedulerConfig MakeSchedulerConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 8;
    cfg.max_num_batched_tokens = kMaxModelLen * 8;
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = kMaxModelLen;
    cfg.watermark = 0.0;
    return cfg;
  }

  static vllm::v1::BlockHasher Hasher() {
    static bool init = false;
    if (!init) {
      init_none_hash(sha256_cbor);
      init = true;
    }
    // block_size == hash_block_size; prompts are far shorter than a block, so no
    // block ever completes and this stays inert (prefix caching never fires).
    return get_request_block_hasher(kBlockSize, sha256_cbor);
  }

  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  EngineCore engine_core;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  LLMEngine engine;
};

// The {model_name, engine} label suffix every series carries for model "m".
constexpr const char* kL = "{model_name=\"m\",engine=\"0\"}";

// Parse the scalar a Prometheus text line carries: find "<series> " and read the
// value to end of line. `series` is the full "name{labels}" (or a
// histogram/counter suffixed name). Fails the test if the series is absent.
double MetricValue(const std::string& text, const std::string& series) {
  const std::string needle = series + " ";
  const size_t p = text.find(needle);
  REQUIRE_MESSAGE(p != std::string::npos, "series absent: " << series);
  const size_t v = p + needle.size();
  const size_t e = text.find('\n', v);
  return std::stod(text.substr(v, e - v));
}

// The observation count of a histogram family (its _count series).
int64_t HistogramCount(const std::string& text, const std::string& name) {
  return static_cast<int64_t>(
      MetricValue(text, name + "_count" + std::string(kL)));
}

// The observation SUM of a histogram family (its _sum series). This is what
// flips from 0 to positive once the QUEUED/SCHEDULED events populate the
// per-request timing intervals: before the event wiring the histogram is still
// observed once per finished request, but always with value 0.0, so _count is
// nonzero while _sum stays exactly 0.
double HistogramSum(const std::string& text, const std::string& name) {
  return MetricValue(text, name + "_sum" + std::string(kL));
}

}  // namespace

// ─── 1. Greedy determinism + termination ─────────────────────────────────────
TEST_CASE("llm_engine: greedy generate is deterministic and terminates at max_tokens") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kN = 6;

  RequestOutput run1;
  RequestOutput run2;
  {
    Harness h(c, w, tok);
    run1 = h.engine.generate(prompt, Greedy(kN), "req");
    CHECK_FALSE(h.engine.has_unfinished_requests());  // loop terminated.
  }
  {
    Harness h(c, w, tok);  // fresh stack -> fresh KV/scheduler state.
    run2 = h.engine.generate(prompt, Greedy(kN), "req");
  }

  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  // Exactly N tokens produced (no eos configured -> length finish).
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kN);
  REQUIRE(run1.outputs[0].finish_reason.has_value());
  CHECK(*run1.outputs[0].finish_reason == "length");

  // Deterministic: same prompt + greedy -> identical token stream + text.
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run1.outputs[0].token_ids == run2.outputs[0].token_ids);
  CHECK(run1.outputs[0].text == run2.outputs[0].text);
}

// ─── 2. Two-request concurrent batch ─────────────────────────────────────────
TEST_CASE("llm_engine: 2-request concurrent batch produces correct per-request streams") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 5;

  // Oracle: each request run on its OWN engine.
  RequestOutput solo_a;
  RequestOutput solo_b;
  {
    Harness h(c, w, tok);
    solo_a = h.engine.generate("hello", Greedy(kN), "A");
  }
  {
    Harness h(c, w, tok);
    solo_b = h.engine.generate("world", Greedy(kN), "B");
  }

  // Both concurrently on one engine, stepped together.
  Harness h(c, w, tok);
  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "world", Greedy(kN));

  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) finished[out.request_id] = out;
    }
  }

  REQUIRE(finished.count("A") == 1);
  REQUIRE(finished.count("B") == 1);
  // Each concurrent stream matches its standalone single-request run (the batch
  // path is per-request independent).
  CHECK(finished["A"].outputs[0].token_ids == solo_a.outputs[0].token_ids);
  CHECK(finished["B"].outputs[0].token_ids == solo_b.outputs[0].token_ids);
  CHECK(finished["A"].outputs[0].token_ids != finished["B"].outputs[0].token_ids);
}

// ─── 3. Streaming (DELTA) vs non-streaming (CUMULATIVE) equivalence ───────────
TEST_CASE("llm_engine: streaming deltas concatenate to the non-streaming full text") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";
  const int kN = 6;

  // Non-streaming: the finished CUMULATIVE output carries the full text.
  RequestOutput cumulative;
  {
    Harness h(c, w, tok);
    cumulative = h.engine.generate(prompt, Greedy(kN), "req");
  }

  // Streaming: DELTA output_kind — accumulate the per-step deltas.
  std::string streamed;
  std::vector<int32_t> streamed_ids;
  {
    Harness h(c, w, tok);
    SamplingParams sp = Greedy(kN);
    sp.output_kind = RequestOutputKind::kDelta;
    h.engine.add_request("req", prompt, sp);
    while (h.engine.has_unfinished_requests()) {
      for (const RequestOutput& out : h.engine.step()) {
        if (!out.outputs.empty()) {
          streamed += out.outputs[0].text;
          for (int32_t t : out.outputs[0].token_ids) streamed_ids.push_back(t);
        }
      }
    }
  }

  REQUIRE(cumulative.outputs.size() == 1);
  CHECK(streamed == cumulative.outputs[0].text);
  CHECK(streamed_ids == cumulative.outputs[0].token_ids);
}

// ─── 4. Stop on max_tokens (LENGTH finish), driven a step at a time ───────────
TEST_CASE("llm_engine: max_tokens length-stops the request end to end") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 3;

  Harness h(c, w, tok);
  h.engine.add_request("req", "hello", Greedy(kN));

  int steps = 0;
  RequestOutput result;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) result = std::move(out);
    }
    ++steps;
    REQUIRE(steps < 100);  // must terminate.
  }

  REQUIRE(result.finished);
  REQUIRE(result.outputs.size() == 1);
  CHECK(static_cast<int>(result.outputs[0].token_ids.size()) == kN);
  CHECK(*result.outputs[0].finish_reason == "length");
  CHECK_FALSE(h.engine.has_unfinished_requests());
}

// ─── 5. Stop on a stop STRING (OutputProcessor -> reqs_to_abort -> abort) ──────
TEST_CASE("llm_engine: a stop string ends the request through the full loop") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const std::string prompt = "hello";

  // First produce the deterministic full greedy text so we can pick a stop
  // substring that actually appears in it.
  std::string full_text;
  {
    Harness h(c, w, tok);
    RequestOutput r = h.engine.generate(prompt, Greedy(8), "probe");
    full_text = r.outputs[0].text;
  }
  REQUIRE(full_text.size() >= 2);
  // A stop string drawn from the middle of the produced text (skip the first
  // char so truncation leaves a non-trivial prefix).
  const std::string stop = full_text.substr(1, 1);

  Harness h(c, w, tok);
  SamplingParams sp = Greedy(8);
  sp.stop = {stop};
  h.engine.add_request("req", prompt, sp);

  int steps = 0;
  RequestOutput result;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& out : h.engine.step()) {
      if (out.finished) result = std::move(out);
    }
    ++steps;
    REQUIRE(steps < 100);
  }

  REQUIRE(result.finished);
  REQUIRE(result.outputs.size() == 1);
  REQUIRE(result.outputs[0].finish_reason.has_value());
  // The string stop is detected by the OutputProcessor (not the scheduler's
  // token-level check_stop) -> finish_reason "stop", stop_reason == the string.
  CHECK(*result.outputs[0].finish_reason == "stop");
  REQUIRE(result.outputs[0].stop_reason.has_value());
  CHECK(*result.outputs[0].stop_reason == stop);
  // The reqs_to_abort feedback tore the request down in the EngineCore too.
  CHECK_FALSE(h.engine.has_unfinished_requests());
  // ...AND reached the SCHEDULER: a string stop is invisible to the scheduler's
  // token-level check_stop, so ONLY the reqs_to_abort -> abort_requests feedback
  // removes it. has_unfinished_requests() above only reflects OutputProcessor
  // state (which erases on finish regardless); this asserts the abort genuinely
  // propagated to the scheduler, so the request is not silently left running
  // (leaked KV + wasted compute) if the feedback were dropped.
  CHECK(h.scheduler.get_num_unfinished_requests() == 0);
  // Output truncated before the stop string (include_stop_str_in_output=false).
  CHECK(result.outputs[0].text.find(stop) == std::string::npos);
}

// ─── 6. Live per-step metrics feed the Prometheus registry (SERVE-METRICS) ────
// The step loop folds this step's SchedulerStats + IterationStats into the
// attached PrometheusStatLogger (llm_engine.py:308-329). Before this wiring the
// registry stayed primed at zero forever (a scrape returned the schema, not the
// counts). This drives the CPU reference engine for several real steps and
// asserts the LIVE values are correct and evolve.
TEST_CASE("llm_engine: live per-step stats populate the Prometheus registry") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 4;  // max_tokens per request (length stop).

  Harness h(c, w, tok);
  PrometheusStatLogger logger("m", kMaxModelLen);
  h.engine.set_stat_logger(&logger);

  const std::string kRun = std::string("vllm:num_requests_running") + kL;
  const std::string kWait = std::string("vllm:num_requests_waiting") + kL;
  const std::string kPrompt = std::string("vllm:prompt_tokens_total") + kL;
  const std::string kGen = std::string("vllm:generation_tokens_total") + kL;
  const std::string kSuccessLen =
      "vllm:request_success_total{model_name=\"m\",engine=\"0\",finished_reason="
      "\"length\"}";

  // Baseline: primed at zero, no histogram observations — exactly the state a
  // scrape saw when the step site never recorded (the pre-wiring behaviour).
  {
    const std::string t = logger.Expose();
    CHECK(MetricValue(t, kPrompt) == 0.0);
    CHECK(MetricValue(t, kGen) == 0.0);
    CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 0);
    CHECK(HistogramCount(t, "vllm:e2e_request_latency_seconds") == 0);
  }

  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "hello", Greedy(kN));

  // Step 1 = prefill: both requests enter the running batch and emit their first
  // token. Gauges track the batch; prompt tokens (1 per "hello") are counted and
  // one TTFT sample is observed per request.
  std::map<std::string, RequestOutput> finished;
  for (RequestOutput& o : h.engine.step()) {
    if (o.finished) finished[o.request_id] = o;
  }
  {
    const std::string t = logger.Expose();
    CHECK(MetricValue(t, kRun) == 2.0);
    CHECK(MetricValue(t, kWait) == 0.0);
    CHECK(MetricValue(t, kPrompt) == 2.0);   // 1 token per "hello" x2 prefilled.
    CHECK(MetricValue(t, kGen) == 2.0);      // first token per request.
    CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 2);
    CHECK(HistogramCount(t, "vllm:inter_token_latency_seconds") == 0);
  }

  // Drive to completion.
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& o : h.engine.step()) {
      if (o.finished) finished[o.request_id] = o;
    }
  }
  REQUIRE(finished.size() == 2);

  int64_t total_gen = 0;
  for (const auto& kv : finished) {
    total_gen += static_cast<int64_t>(kv.second.outputs[0].token_ids.size());
  }
  CHECK(total_gen == 2 * kN);  // deterministic greedy, length-stopped.

  const std::string t = logger.Expose();
  // Token counters == the EXACT counts the run produced.
  CHECK(MetricValue(t, kPrompt) == 2.0);
  CHECK(MetricValue(t, kGen) == static_cast<double>(total_gen));
  // Both requests finished with the "length" reason.
  CHECK(MetricValue(t, kSuccessLen) == 2.0);
  // All requests drained: the gauges fall back to zero.
  CHECK(MetricValue(t, kRun) == 0.0);
  CHECK(MetricValue(t, kWait) == 0.0);
  // Histogram sample counts: TTFT once per request; ITL once per decode token
  // (kN-1 per request, the first token being the prefill TTFT); e2e + TPOT once
  // per finished request; iteration-tokens observed on at least the prefill step.
  CHECK(HistogramCount(t, "vllm:time_to_first_token_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:inter_token_latency_seconds") == 2 * (kN - 1));
  CHECK(HistogramCount(t, "vllm:e2e_request_latency_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:request_time_per_output_token_seconds") == 2);
  CHECK(HistogramCount(t, "vllm:iteration_tokens_total") >= 1);
  CHECK(HistogramCount(t, "vllm:request_generation_tokens") == 2);
}

// ─── 7. Per-request queue/prefill/inference timing from EngineCoreEvents ──────
// The scheduler records QUEUED (add_request), SCHEDULED (batch admission) and
// PREEMPTED events per request; the OutputProcessor folds them into the
// queue/prefill/inference timing intervals update_from_finished_request observes
// (stats.py:459-476), feeding the Prometheus timing histograms. Before this
// wiring these histograms were STILL observed once per finished request, but
// always with value 0.0 — so their _sum stayed exactly 0. This drives the real
// CPU engine to completion and asserts the sums now flip POSITIVE and the
// intervals are correctly ordered. RED-first: reverting the event
// emission/consumption leaves every _sum below at 0.0 and each CHECK fails.
TEST_CASE("llm_engine: per-request queue/prefill/inference timing populates") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const Tokenizer& tok = Fixture();
  const int kN = 4;  // max_tokens per request (length stop).

  Harness h(c, w, tok);
  PrometheusStatLogger logger("m", kMaxModelLen);
  h.engine.set_stat_logger(&logger);

  const char* kQueue = "vllm:request_queue_time_seconds";
  const char* kPrefill = "vllm:request_prefill_time_seconds";
  const char* kInference = "vllm:request_inference_time_seconds";
  const char* kDecode = "vllm:request_decode_time_seconds";
  const char* kE2E = "vllm:e2e_request_latency_seconds";

  // Baseline: nothing observed yet, so every timing _sum is 0.
  {
    const std::string t = logger.Expose();
    CHECK(HistogramSum(t, kQueue) == 0.0);
    CHECK(HistogramSum(t, kPrefill) == 0.0);
    CHECK(HistogramSum(t, kInference) == 0.0);
  }

  h.engine.add_request("A", "hello", Greedy(kN));
  h.engine.add_request("B", "hello", Greedy(kN));
  std::map<std::string, RequestOutput> finished;
  while (h.engine.has_unfinished_requests()) {
    for (RequestOutput& o : h.engine.step()) {
      if (o.finished) finished[o.request_id] = o;
    }
  }
  REQUIRE(finished.size() == 2);

  const std::string t = logger.Expose();
  // Every finished request contributes one observation to each timing family.
  CHECK(HistogramCount(t, kQueue) == 2);
  CHECK(HistogramCount(t, kPrefill) == 2);
  CHECK(HistogramCount(t, kInference) == 2);

  const double queue_sum = HistogramSum(t, kQueue);
  const double prefill_sum = HistogramSum(t, kPrefill);
  const double inference_sum = HistogramSum(t, kInference);
  const double decode_sum = HistogramSum(t, kDecode);
  const double e2e_sum = HistogramSum(t, kE2E);

  // THE FLIP: each interval is now a real positive duration (was exactly 0).
  CHECK(queue_sum > 0.0);
  CHECK(prefill_sum > 0.0);
  CHECK(inference_sum > 0.0);
  CHECK(decode_sum > 0.0);

  // Interval algebra (stats.py): inference = prefill + decode (both measured
  // from the SAME scheduled_ts / first_token_ts / last_token_ts endpoints), so
  // the aggregate sums satisfy it too, within float tolerance.
  CHECK(inference_sum == doctest::Approx(prefill_sum + decode_sum));
  // Ordering: the queued span precedes scheduling, inference sits inside e2e,
  // and prefill is a sub-interval of inference.
  CHECK(inference_sum <= e2e_sum);
  CHECK(prefill_sum <= inference_sum);
  CHECK(queue_sum > 0.0);
}
