// vllm.cpp original. DSpark Markov head + draft->target vocab map unit tests
// (SPEC-DSPARK W2). Ported semantics: vllm/model_executor/models/qwen3_dspark.py
// @ 555967922 — DSparkMarkovHead.embed (:61-63) / .bias (:65-67),
// map_draft_to_target (:137-141), the draft_vocab_size default (:99-100).
//
// These run on synthetic weights and pin the W2 load-bearing invariants:
//   (1) markov_embed gathers the right row of markov_w1 for each previous token;
//   (2) markov_bias equals an independent bf16 reference of markov_w2 @ embed^T,
//       scaled by logit_scale;
//   (3) RED — the bias DEPENDS on the previous token. This is the whole point of
//       the head: a purely parallel block draft (our landed DFlash) has no
//       intra-block dependency, and DSpark buys it here. A stubbed head that
//       ignored `prev` would pass (1) and (2) shape checks but fail this;
//   (4) map_draft_to_target is the identity for a full-vocab draft and
//       `draft_id + d2t[draft_id]` for a reduced one (the table holds the
//       OFFSET, not the absolute target id — getting that backwards silently
//       drafts the wrong tokens);
//   (5) ResolveDsparkDims defaults draft_vocab_size to vocab_size and refuses a
//       config without markov_rank (not a DSpark draft).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/qwen3_dspark.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using namespace vllm;

namespace {
vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

OwnedTensor MkBf16(const std::vector<int64_t>& shape, double seed, double amp, bool nk) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  t.nk = nk;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return t;
}

float Bf16At(const OwnedTensor& t, int64_t i) {
  const auto* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
  return vt::BF16ToF32(p[i]);
}

constexpr int64_t kVocab = 9;       // target vocab
constexpr int64_t kDraftVocab = 5;  // reduced draft vocab
constexpr int64_t kRank = 4;        // markov_rank

Qwen3DSparkWeights MakeWeights(bool reduced_vocab) {
  Qwen3DSparkWeights w;
  w.markov_rank = kRank;
  w.vocab_size = kVocab;
  w.draft_vocab_size = reduced_vocab ? kDraftVocab : kVocab;
  w.markov_w1 = MkBf16({kVocab, kRank}, 0.11, 0.5, false);
  w.markov_w2 = MkBf16({w.draft_vocab_size, kRank}, 0.37, 0.4, true);
  w.backbone.draft_vocab_size = w.draft_vocab_size;
  if (reduced_vocab) {
    // d2t holds the OFFSET: target = draft + d2t[draft].
    w.draft_id_to_target_id = {0, 1, 1, 3, 4};
  }
  return w;
}
}  // namespace

TEST_CASE("markov_embed gathers the previous token's markov_w1 row") {
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = MakeWeights(/*reduced_vocab=*/false);
  const std::vector<int32_t> prev = {3, 0, 8};
  const std::vector<float> embed = Qwen3DSparkModel::MarkovEmbed(prev, w, q);
  REQUIRE(embed.size() == prev.size() * static_cast<size_t>(kRank));
  for (size_t b = 0; b < prev.size(); ++b) {
    for (int64_t r = 0; r < kRank; ++r) {
      const float want = Bf16At(w.markov_w1, prev[b] * kRank + r);
      CHECK(embed[b * kRank + static_cast<size_t>(r)] == doctest::Approx(want).epsilon(1e-3));
    }
  }
}

TEST_CASE("markov_bias matches an independent reference of markov_w2 @ embed") {
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = MakeWeights(/*reduced_vocab=*/true);
  const std::vector<int32_t> prev = {2, 7};
  const std::vector<float> embed = Qwen3DSparkModel::MarkovEmbed(prev, w, q);
  const std::vector<float> bias = Qwen3DSparkModel::MarkovBias(
      embed, static_cast<int64_t>(prev.size()), w, q);
  REQUIRE(bias.size() == prev.size() * static_cast<size_t>(kDraftVocab));
  for (size_t b = 0; b < prev.size(); ++b) {
    for (int64_t v = 0; v < kDraftVocab; ++v) {
      double acc = 0.0;
      for (int64_t r = 0; r < kRank; ++r) {
        acc += static_cast<double>(embed[b * kRank + static_cast<size_t>(r)]) *
               static_cast<double>(Bf16At(w.markov_w2, v * kRank + r));
      }
      acc *= static_cast<double>(w.logit_scale);
      CHECK(bias[b * kDraftVocab + static_cast<size_t>(v)] ==
            doctest::Approx(static_cast<float>(acc)).epsilon(2e-2));
    }
  }
}

TEST_CASE("the Markov bias DEPENDS on the previous token") {
  // RED: this is the entire reason the head exists. DFlash's parallel block draft
  // has no intra-block dependency; DSpark's sequential stage adds one through
  // exactly this bias. A head that ignored `prev` would still return the right
  // shape.
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = MakeWeights(/*reduced_vocab=*/false);
  const std::vector<float> a = Qwen3DSparkModel::MarkovBiasForTokens({1}, w, q);
  const std::vector<float> b = Qwen3DSparkModel::MarkovBiasForTokens({6}, w, q);
  REQUIRE(a.size() == b.size());
  bool any_diff = false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > 1e-4f) any_diff = true;
  }
  CHECK(any_diff);
}

TEST_CASE("MarkovBiasForTokens equals the two-step composition") {
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = MakeWeights(/*reduced_vocab=*/true);
  const std::vector<int32_t> prev = {4, 4, 1};
  const std::vector<float> composed = Qwen3DSparkModel::MarkovBiasForTokens(prev, w, q);
  const std::vector<float> stepwise = Qwen3DSparkModel::MarkovBias(
      Qwen3DSparkModel::MarkovEmbed(prev, w, q), static_cast<int64_t>(prev.size()), w, q);
  REQUIRE(composed.size() == stepwise.size());
  for (size_t i = 0; i < composed.size(); ++i) CHECK(composed[i] == stepwise[i]);
  // The same previous token must produce the same bias row (a pure function of prev).
  for (int64_t v = 0; v < kDraftVocab; ++v) {
    CHECK(composed[static_cast<size_t>(v)] ==
          composed[static_cast<size_t>(kDraftVocab + v)]);
  }
}

TEST_CASE("map_draft_to_target is the identity for a full-vocab draft") {
  const Qwen3DSparkWeights w = MakeWeights(/*reduced_vocab=*/false);
  CHECK(w.draft_id_to_target_id.empty());
  for (int32_t id = 0; id < static_cast<int32_t>(kVocab); ++id) {
    CHECK(Qwen3DSparkModel::MapDraftToTarget(id, w) == id);
  }
}

TEST_CASE("map_draft_to_target adds the d2t OFFSET for a reduced draft vocab") {
  const Qwen3DSparkWeights w = MakeWeights(/*reduced_vocab=*/true);
  // d2t = {0, 1, 1, 3, 4} -> target = draft + d2t[draft].
  CHECK(Qwen3DSparkModel::MapDraftToTarget(0, w) == 0);
  CHECK(Qwen3DSparkModel::MapDraftToTarget(1, w) == 2);
  CHECK(Qwen3DSparkModel::MapDraftToTarget(2, w) == 3);
  CHECK(Qwen3DSparkModel::MapDraftToTarget(3, w) == 6);
  CHECK(Qwen3DSparkModel::MapDraftToTarget(4, w) == 8);
  CHECK_THROWS_AS(Qwen3DSparkModel::MapDraftToTarget(5, w), std::exception);
  CHECK_THROWS_AS(Qwen3DSparkModel::MapDraftToTarget(-1, w), std::exception);
}

TEST_CASE("ResolveDsparkDims mirrors the config defaults") {
  HfConfig c;
  c.vocab_size = 151936;
  c.raw = nlohmann::json::object();
  c.raw["markov_rank"] = 256;
  Qwen3DSparkWeights w;
  Qwen3DSparkModel::ResolveDsparkDims(c, w);
  CHECK(w.markov_rank == 256);
  CHECK(w.vocab_size == 151936);
  // draft_vocab_size defaults to vocab_size (qwen3_dspark.py:99-100).
  CHECK(w.draft_vocab_size == 151936);
  CHECK(w.logit_scale == doctest::Approx(1.0f));

  c.raw["draft_vocab_size"] = 32000;
  c.raw["logit_scale"] = 2.0;
  Qwen3DSparkWeights w2;
  Qwen3DSparkModel::ResolveDsparkDims(c, w2);
  CHECK(w2.draft_vocab_size == 32000);
  CHECK(w2.logit_scale == doctest::Approx(2.0f));

  // A config without markov_rank is not a DSpark draft (upstream reads
  // config.markov_rank unguarded, qwen3_dspark.py:86).
  HfConfig plain;
  plain.vocab_size = 1024;
  plain.raw = nlohmann::json::object();
  Qwen3DSparkWeights w3;
  CHECK_THROWS_AS(Qwen3DSparkModel::ResolveDsparkDims(plain, w3), std::exception);
}
