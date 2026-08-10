// vllm.cpp original. DSpark draft weight loader (SPEC-DSPARK W3). Ported
// semantics: Qwen3DSparkForCausalLM.load_weights (qwen3_dspark.py:149-185
// @ 555967922) — the d2t rename, the t2d/mask_embedding/confidence_head skips,
// and the optional embed_tokens/lm_head that the draft shares from the target.
//
// The synthetic resolver reproduces the REAL on-disk key list of
// deepseek-ai/dspark_qwen3_4b_block7 (64 tensors: unprefixed, full vocab, no
// d2t) and RedHatAI/Qwen3.6-35B-A3B-speculator.dspark (66 tensors: same
// spelling plus `d2t` I64 [32000] and `t2d` BOOL), both dumped from the
// checkpoints on 2026-08-09. That key list is the thing a loader gets silently
// wrong, so it is pinned here rather than paraphrased.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dspark.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using namespace vllm;

namespace {

constexpr int64_t kH = 4, kHq = 2, kHkv = 1, kDh = 2, kI = 6, kLayers = 2, kTaps = 2;
constexpr int64_t kVocab = 8, kRank = 3;

// A tiny in-memory safetensors stand-in: name -> (dtype, shape, bytes).
class FakeCheckpoint {
 public:
  void AddBf16(const std::string& name, const std::vector<int64_t>& shape, double seed) {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(uint16_t));
    auto* p = reinterpret_cast<uint16_t*>(bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(static_cast<float>(0.3 * std::sin(seed + 0.7 * static_cast<double>(i))));
    Add(name, "BF16", shape, std::move(bytes));
  }
  void AddI64(const std::string& name, const std::vector<int64_t>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    Add(name, "I64", {static_cast<int64_t>(values.size())}, std::move(bytes));
  }
  void AddBool(const std::string& name, int64_t n) {
    Add(name, "BOOL", {n}, std::vector<uint8_t>(static_cast<size_t>(n), 1));
  }
  void Erase(const std::string& name) { entries_.erase(name); }

  TensorResolver Resolver() const {
    return [this](const std::string& name) -> const StTensor& {
      auto it = entries_.find(name);
      if (it == entries_.end()) throw std::runtime_error("missing tensor: " + name);
      return it->second.view;
    };
  }

 private:
  struct Entry {
    std::vector<uint8_t> bytes;
    StTensor view;
  };
  void Add(const std::string& name, const char* dtype, const std::vector<int64_t>& shape,
           std::vector<uint8_t>&& bytes) {
    Entry e;
    e.bytes = std::move(bytes);
    e.view.dtype = dtype;
    e.view.shape = shape;
    e.view.nbytes = e.bytes.size();
    auto [it, ok] = entries_.emplace(name, std::move(e));
    it->second.view.data = it->second.bytes.data();
  }
  mutable std::map<std::string, Entry> entries_;
};

HfConfig MakeConfig(int64_t draft_vocab) {
  HfConfig c;
  c.hidden_size = kH;
  c.num_attention_heads = kHq;
  c.num_key_value_heads = kHkv;
  c.head_dim = kDh;
  c.rotary_dim = kDh;
  c.rope_theta = 10000.0;
  c.intermediate_size = kI;
  c.vocab_size = kVocab;
  c.num_hidden_layers = kLayers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 64;
  c.layer_types = {"sliding_attention", "full_attention"};
  c.raw = nlohmann::json::object();
  c.raw["markov_rank"] = kRank;
  c.raw["mask_token_id"] = 7;
  if (draft_vocab != kVocab) c.raw["draft_vocab_size"] = draft_vocab;
  return c;
}

// The REAL key list, at toy dimensions.
FakeCheckpoint MakeCheckpoint(int64_t draft_vocab, bool with_d2t) {
  FakeCheckpoint ck;
  ck.AddBf16("embed_tokens.weight", {kVocab, kH}, 0.1);
  ck.AddBf16("fc.weight", {kH, kH * kTaps}, 0.2);
  ck.AddBf16("hidden_norm.weight", {kH}, 0.3);
  ck.AddBf16("norm.weight", {kH}, 0.4);
  ck.AddBf16("lm_head.weight", {draft_vocab, kH}, 0.5);
  for (int64_t l = 0; l < kLayers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    ck.AddBf16(p + "input_layernorm.weight", {kH}, 1.0 + l);
    ck.AddBf16(p + "post_attention_layernorm.weight", {kH}, 1.1 + l);
    ck.AddBf16(p + "self_attn.q_proj.weight", {kHq * kDh, kH}, 1.2 + l);
    ck.AddBf16(p + "self_attn.k_proj.weight", {kHkv * kDh, kH}, 1.3 + l);
    ck.AddBf16(p + "self_attn.v_proj.weight", {kHkv * kDh, kH}, 1.4 + l);
    ck.AddBf16(p + "self_attn.o_proj.weight", {kH, kHq * kDh}, 1.5 + l);
    ck.AddBf16(p + "self_attn.q_norm.weight", {kDh}, 1.6 + l);
    ck.AddBf16(p + "self_attn.k_norm.weight", {kDh}, 1.7 + l);
    ck.AddBf16(p + "mlp.gate_proj.weight", {kI, kH}, 1.8 + l);
    ck.AddBf16(p + "mlp.up_proj.weight", {kI, kH}, 1.9 + l);
    ck.AddBf16(p + "mlp.down_proj.weight", {kH, kI}, 2.0 + l);
  }
  // The DSpark-specific three, plus the two upstream skips.
  ck.AddBf16("markov_head.markov_w1.weight", {kVocab, kRank}, 3.1);
  ck.AddBf16("markov_head.markov_w2.weight", {draft_vocab, kRank}, 3.2);
  ck.AddBf16("confidence_head.proj.weight", {1, kH + 2}, 3.3);
  ck.AddBf16("confidence_head.proj.bias", {1}, 3.4);
  if (with_d2t) {
    // OFFSETS, not absolute ids: target = draft + d2t[draft].
    std::vector<int64_t> d2t(static_cast<size_t>(draft_vocab));
    for (int64_t i = 0; i < draft_vocab; ++i) d2t[static_cast<size_t>(i)] = i % 3;
    ck.AddI64("d2t", d2t);
    ck.AddBool("t2d", kVocab);
  }
  return ck;
}

}  // namespace

TEST_CASE("LoadQwen3DSpark loads the native full-vocab checkpoint layout") {
  const FakeCheckpoint ck = MakeCheckpoint(kVocab, /*with_d2t=*/false);
  const HfConfig cfg = MakeConfig(kVocab);
  const Qwen3DSparkWeights w = LoadQwen3DSpark(ck.Resolver(), cfg, kTaps, /*mask=*/7);

  CHECK(w.markov_rank == kRank);
  CHECK(w.vocab_size == kVocab);
  CHECK(w.draft_vocab_size == kVocab);
  CHECK(w.draft_id_to_target_id.empty());  // full vocab -> identity map
  // The Markov head: w1 is an embedding table (NOT nk), w2 an nk Linear.
  REQUIRE(w.markov_w1.rank == 2);
  CHECK(w.markov_w1.shape[0] == kVocab);
  CHECK(w.markov_w1.shape[1] == kRank);
  CHECK_FALSE(w.markov_w1.nk);
  REQUIRE(w.markov_w2.rank == 2);
  CHECK(w.markov_w2.shape[0] == kVocab);
  CHECK(w.markov_w2.shape[1] == kRank);
  CHECK(w.markov_w2.nk);
  // The inherited backbone came through the DFlash loader.
  CHECK(w.backbone.layers.size() == static_cast<size_t>(kLayers));
  CHECK(w.backbone.num_taps == kTaps);
  CHECK(w.backbone.mask_token_id == 7);
  CHECK(w.backbone.draft_vocab_size == kVocab);
  CHECK_FALSE(w.backbone.fc.Empty());
}

TEST_CASE("LoadQwen3DSpark reads d2t as an OFFSET table for a reduced draft vocab") {
  constexpr int64_t kDraftVocab = 5;
  const FakeCheckpoint ck = MakeCheckpoint(kDraftVocab, /*with_d2t=*/true);
  const HfConfig cfg = MakeConfig(kDraftVocab);
  const Qwen3DSparkWeights w = LoadQwen3DSpark(ck.Resolver(), cfg, kTaps, /*mask=*/7);

  CHECK(w.draft_vocab_size == kDraftVocab);
  REQUIRE(w.draft_id_to_target_id.size() == static_cast<size_t>(kDraftVocab));
  for (int64_t i = 0; i < kDraftVocab; ++i) {
    CHECK(w.draft_id_to_target_id[static_cast<size_t>(i)] == static_cast<int32_t>(i % 3));
    // The map the sampler will apply: target = draft + offset.
    CHECK(Qwen3DSparkModel::MapDraftToTarget(static_cast<int32_t>(i), w) ==
          static_cast<int32_t>(i + i % 3));
  }
  CHECK(w.markov_w2.shape[0] == kDraftVocab);
}

TEST_CASE("a reduced draft vocab without d2t is refused") {
  // Silently drafting draft-vocab ids into a target vocab would corrupt output.
  FakeCheckpoint ck = MakeCheckpoint(5, /*with_d2t=*/true);
  ck.Erase("d2t");
  CHECK_THROWS_AS(LoadQwen3DSpark(ck.Resolver(), MakeConfig(5), kTaps, 7), std::exception);
}

TEST_CASE("a checkpoint without the Markov head is refused by name") {
  FakeCheckpoint ck = MakeCheckpoint(kVocab, false);
  ck.Erase("markov_head.markov_w2.weight");
  CHECK_THROWS_AS(LoadQwen3DSpark(ck.Resolver(), MakeConfig(kVocab), kTaps, 7),
                  std::exception);
}

TEST_CASE("a markov_w2 disagreeing with the config's draft vocab is refused") {
  // config says 5, checkpoint ships 8 — the mismatch that would otherwise index
  // the wrong logit columns forever.
  const FakeCheckpoint ck = MakeCheckpoint(kVocab, /*with_d2t=*/false);
  CHECK_THROWS_AS(LoadQwen3DSpark(ck.Resolver(), MakeConfig(5), kTaps, 7), std::exception);
}
