// vllm.cpp original. DSpark sequential Markov sampling + the anchor-as-first-
// prediction block layout (SPEC-DSPARK W4). Ported semantics:
// vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:100-149 (_sample_sequential)
// and :5-11,50-58 (the N vs 1+N query layout) @ 555967922.
//
// The decisive fixture is a Markov head built so the bias DICTATES a chain:
// markov_w1 is the identity ([V, r] with r == V), so embed(prev) = e_prev, and
// markov_w2[v][p] is large exactly when v == (p + 1) % V. The bias therefore
// forces draft[i] = (draft[i-1] + 1) % V regardless of the base logits, seeded by
// the anchor. A sampler that ignored `prev` — i.e. DFlash's parallel argmax, or a
// loop that fed the bias the anchor every step — cannot produce that ramp.
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vllm/v1/worker/gpu/spec_decode/dspark/speculator.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using namespace vllm;
using namespace vllm::v1;

namespace {
vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

constexpr int64_t kV = 6;  // target vocab == markov_rank, so markov_w1 is I
constexpr float kBig = 64.0f;

OwnedTensor MkBf16(const std::vector<int64_t>& shape, const std::vector<float>& vals,
                   bool nk) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  t.nk = nk;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  REQUIRE(static_cast<int64_t>(vals.size()) == n);
  t.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(vals[static_cast<size_t>(i)]);
  return t;
}

// markov_w1 = I [V, V]; markov_w2[v][p] = kBig iff v == (p+1) % V.
Qwen3DSparkWeights ChainWeights() {
  Qwen3DSparkWeights w;
  w.markov_rank = kV;
  w.vocab_size = kV;
  w.draft_vocab_size = kV;
  w.backbone.draft_vocab_size = kV;
  std::vector<float> w1(static_cast<size_t>(kV * kV), 0.0f);
  for (int64_t i = 0; i < kV; ++i) w1[static_cast<size_t>(i * kV + i)] = 1.0f;
  w.markov_w1 = MkBf16({kV, kV}, w1, /*nk=*/false);
  std::vector<float> w2(static_cast<size_t>(kV * kV), 0.0f);
  for (int64_t p = 0; p < kV; ++p) {
    const int64_t v = (p + 1) % kV;
    w2[static_cast<size_t>(v * kV + p)] = kBig;
  }
  w.markov_w2 = MkBf16({kV, kV}, w2, /*nk=*/true);
  return w;
}

// A head that contributes nothing: the drafts are then the plain per-row argmax,
// i.e. exactly what DFlash's parallel sampling produces.
Qwen3DSparkWeights ZeroHeadWeights() {
  Qwen3DSparkWeights w = ChainWeights();
  w.markov_w2 = MkBf16({kV, kV}, std::vector<float>(static_cast<size_t>(kV * kV), 0.0f),
                       /*nk=*/true);
  return w;
}

// Base logits where row j of every request peaks at column (j * 2) % kV, small
// enough that the chain bias dominates when present.
std::vector<float> BaseLogits(int num_reqs, int nqpr) {
  std::vector<float> logits(static_cast<size_t>(num_reqs) * nqpr * kV, 0.0f);
  for (int r = 0; r < num_reqs; ++r) {
    for (int j = 0; j < nqpr; ++j) {
      const size_t row = (static_cast<size_t>(r) * nqpr + j) * kV;
      logits[row + static_cast<size_t>((j * 2 + r) % kV)] = 1.0f;
    }
  }
  return logits;
}
}  // namespace

TEST_CASE("the sequential loop chains each step onto the PREVIOUS sampled token") {
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = ChainWeights();
  DsparkBlockLayout layout;
  layout.num_speculative_steps = 4;
  layout.sample_from_anchor = true;
  const int num_reqs = 2;
  const std::vector<int32_t> anchors = {1, 4};
  const std::vector<float> logits = BaseLogits(num_reqs, layout.num_query_per_req());

  const auto drafts = SampleDsparkBlockDrafts(logits, anchors, layout, w, q);
  REQUIRE(drafts.size() == static_cast<size_t>(num_reqs));
  for (int r = 0; r < num_reqs; ++r) {
    REQUIRE(drafts[static_cast<size_t>(r)].size() == 4);
    int32_t expect = anchors[static_cast<size_t>(r)];
    for (int i = 0; i < 4; ++i) {
      expect = static_cast<int32_t>((expect + 1) % kV);
      CHECK(drafts[static_cast<size_t>(r)][static_cast<size_t>(i)] == expect);
    }
  }
}

TEST_CASE("with a zero Markov head the drafts collapse to the parallel argmax") {
  // The DFlash behaviour. This is the control for the test above: same rows, same
  // base logits, and the only difference is whether the head contributes.
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = ZeroHeadWeights();
  DsparkBlockLayout layout;
  layout.num_speculative_steps = 3;
  layout.sample_from_anchor = true;
  const std::vector<int32_t> anchors = {2};
  const std::vector<float> logits = BaseLogits(1, layout.num_query_per_req());

  const auto drafts = SampleDsparkBlockDrafts(logits, anchors, layout, w, q);
  REQUIRE(drafts[0].size() == 3);
  for (int i = 0; i < 3; ++i) {
    CHECK(drafts[0][static_cast<size_t>(i)] == static_cast<int32_t>((i * 2) % kV));
  }
}

TEST_CASE("sample_from_anchor selects N rows starting at the anchor row") {
  // Anchor layout: N query rows, every one a prediction (speculator.py:5-11).
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = ZeroHeadWeights();
  DsparkBlockLayout layout;
  layout.num_speculative_steps = 3;
  layout.sample_from_anchor = true;
  CHECK(layout.num_query_per_req() == 3);
  CHECK(layout.first_sample_offset() == 0);

  // Distinctive peak per row: row j peaks at column j.
  std::vector<float> logits(static_cast<size_t>(3) * kV, 0.0f);
  for (int j = 0; j < 3; ++j) logits[static_cast<size_t>(j) * kV + static_cast<size_t>(j)] = 1.0f;
  const auto drafts = SampleDsparkBlockDrafts(logits, {0}, layout, w, q);
  CHECK(drafts[0] == std::vector<int32_t>{0, 1, 2});
}

TEST_CASE("without sample_from_anchor the anchor row is a bonus token, not sampled") {
  // The DFlash 1 + N fill-in layout, which the Speculators-format checkpoints use
  // (algos.py:157-159 defaults sample_from_anchor false).
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = ZeroHeadWeights();
  DsparkBlockLayout layout;
  layout.num_speculative_steps = 3;
  layout.sample_from_anchor = false;
  CHECK(layout.num_query_per_req() == 4);
  CHECK(layout.first_sample_offset() == 1);

  std::vector<float> logits(static_cast<size_t>(4) * kV, 0.0f);
  for (int j = 0; j < 4; ++j) logits[static_cast<size_t>(j) * kV + static_cast<size_t>(j)] = 1.0f;
  const auto drafts = SampleDsparkBlockDrafts(logits, {0}, layout, w, q);
  // Rows 1,2,3 -> columns 1,2,3. Row 0 (the anchor) is NOT sampled.
  CHECK(drafts[0] == std::vector<int32_t>{1, 2, 3});
}

TEST_CASE("a reduced draft vocab maps every sampled id into the target vocab") {
  vt::Queue q = Cpu();
  Qwen3DSparkWeights w = ZeroHeadWeights();
  w.draft_vocab_size = 3;
  w.backbone.draft_vocab_size = 3;
  w.markov_w2 = MkBf16({3, kV}, std::vector<float>(static_cast<size_t>(3 * kV), 0.0f),
                       /*nk=*/true);
  w.draft_id_to_target_id = {0, 2, 3};  // OFFSETS: 0->0, 1->3, 2->5

  DsparkBlockLayout layout;
  layout.num_speculative_steps = 3;
  layout.sample_from_anchor = true;
  std::vector<float> logits(static_cast<size_t>(3) * 3, 0.0f);
  for (int j = 0; j < 3; ++j) logits[static_cast<size_t>(j) * 3 + static_cast<size_t>(j)] = 1.0f;

  const auto drafts = SampleDsparkBlockDrafts(logits, {0}, layout, w, q);
  CHECK(drafts[0] == std::vector<int32_t>{0, 3, 5});
}

TEST_CASE("shape and range violations are refused") {
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = ZeroHeadWeights();
  DsparkBlockLayout layout;
  layout.num_speculative_steps = 2;
  layout.sample_from_anchor = true;
  const std::vector<float> good(static_cast<size_t>(2) * kV, 0.0f);
  CHECK_THROWS_AS(SampleDsparkBlockDrafts(std::vector<float>(3, 0.0f), {0}, layout, w, q),
                  std::exception);
  // An anchor id outside the TARGET vocab would index markov_w1 out of range.
  CHECK_THROWS_AS(SampleDsparkBlockDrafts(good, {static_cast<int32_t>(kV)}, layout, w, q),
                  std::exception);
}

// ---- W7 (#436): the device sequential sampler ------------------------------
//
// The host loop downloads [num_reqs, draft_vocab] f32 and scans it once per
// step, measured at 10.5 ms of a 37.9 ms draft step on the 27B lane (28%).
// SampleSequentialDevice keeps the bias and the argmax on device. It is a pure
// COST move, so the bar is exact token identity, not "close": these cases run
// both paths over the same inputs and compare ids.

TEST_CASE("device sequential sampling is TOKEN-IDENTICAL to the host loop") {
  vt::Queue q = Cpu();
  const Qwen3DSparkWeights w = ChainWeights();

  // Both layouts, several batch sizes: the device path computes its base row as
  // r*nqpr + first_sample_offset + i and must agree with the host indexing for
  // each, which is where an off-by-one would hide.
  for (bool from_anchor : {true, false}) {
    for (int num_reqs : {1, 2, 3}) {
      DsparkBlockLayout layout;
      layout.num_speculative_steps = 4;
      layout.sample_from_anchor = from_anchor;
      const int nqpr = layout.num_query_per_req();
      const std::vector<float> logits = BaseLogits(num_reqs, nqpr);
      std::vector<int32_t> anchors;
      for (int r = 0; r < num_reqs; ++r) anchors.push_back((r * 2) % static_cast<int>(kV));

      const auto device = Qwen3DSparkModel::SampleSequentialDevice(
          logits, anchors, nqpr, layout.first_sample_offset(),
          layout.num_speculative_steps, w, q);
      const auto host = SampleDsparkBlockDrafts(logits, anchors, layout, w, q);

      REQUIRE(device.size() == host.size());
      for (size_t r = 0; r < host.size(); ++r) {
        CHECK(device[r] == host[r]);
      }
    }
  }
}

TEST_CASE("device sampling reproduces the chain, not just the parallel argmax") {
  // Guards against a device path that silently drops the Markov bias and still
  // "matches" because the base argmax happens to agree: with ChainWeights the
  // bias DOMINATES (kBig vs 1.0), so each step must be (prev + 1) % kV. A zero
  // head must instead give the plain per-row argmax, and the two must DIFFER.
  vt::Queue q = Cpu();
  DsparkBlockLayout layout;
  layout.num_speculative_steps = 4;
  layout.sample_from_anchor = true;
  const int nqpr = layout.num_query_per_req();
  const std::vector<float> logits = BaseLogits(1, nqpr);
  const std::vector<int32_t> anchors = {1};

  const auto chained = Qwen3DSparkModel::SampleSequentialDevice(
      logits, anchors, nqpr, layout.first_sample_offset(),
      layout.num_speculative_steps, ChainWeights(), q);
  REQUIRE(chained.size() == 1);
  int32_t expect = anchors[0];
  for (int i = 0; i < layout.num_speculative_steps; ++i) {
    expect = static_cast<int32_t>((expect + 1) % kV);
    CHECK(chained[0][static_cast<size_t>(i)] == expect);
  }

  const auto flat = Qwen3DSparkModel::SampleSequentialDevice(
      logits, anchors, nqpr, layout.first_sample_offset(),
      layout.num_speculative_steps, ZeroHeadWeights(), q);
  CHECK(flat[0] != chained[0]);
}

TEST_CASE("device sampling maps a reduced draft vocab through d2t") {
  // The device argmax runs over draft-vocab columns; the emitted id must still
  // be the TARGET id (draft + d2t[draft]), exactly as the host path emits.
  vt::Queue q = Cpu();
  Qwen3DSparkWeights w = ChainWeights();
  w.draft_vocab_size = 3;
  w.backbone.draft_vocab_size = 3;
  std::vector<float> w2(static_cast<size_t>(3 * kV), 0.0f);
  for (int64_t p = 0; p < kV; ++p) w2[static_cast<size_t>(((p + 1) % 3) * kV + p)] = kBig;
  w.markov_w2 = MkBf16({3, kV}, w2, /*nk=*/true);
  w.draft_id_to_target_id = {2, 2, 2};  // OFFSETS: draft d -> target d + 2

  DsparkBlockLayout layout;
  layout.num_speculative_steps = 3;
  layout.sample_from_anchor = true;
  const int nqpr = layout.num_query_per_req();
  std::vector<float> logits(static_cast<size_t>(nqpr) * 3, 0.0f);
  const std::vector<int32_t> anchors = {0};

  const auto device = Qwen3DSparkModel::SampleSequentialDevice(
      logits, anchors, nqpr, layout.first_sample_offset(),
      layout.num_speculative_steps, w, q);
  const auto host = SampleDsparkBlockDrafts(logits, anchors, layout, w, q);
  CHECK(device[0] == host[0]);
  for (int32_t id : device[0]) CHECK(id >= 2);  // every id went through the +2 offset
}

TEST_CASE("the device sampler refuses what it cannot reproduce") {
  vt::Queue q = Cpu();
  Qwen3DSparkWeights w = ChainWeights();
  CHECK(Qwen3DSparkModel::CanSampleOnDevice(w));

  // The host path scales the BIAS ONLY by logit_scale; the device composition
  // does not, so a non-unit scale must route back to the host loop rather than
  // silently produce different tokens.
  w.logit_scale = 2.0f;
  CHECK_FALSE(Qwen3DSparkModel::CanSampleOnDevice(w));
  CHECK_THROWS(Qwen3DSparkModel::SampleSequentialDevice(
      BaseLogits(1, 4), std::vector<int32_t>{0}, 4, 0, 4, w, q));

  // And the shape contract is enforced, not assumed.
  Qwen3DSparkWeights ok = ChainWeights();
  CHECK_THROWS(Qwen3DSparkModel::SampleSequentialDevice(
      BaseLogits(1, 4), std::vector<int32_t>{0}, 4, /*first_sample_offset=*/2,
      /*num_spec=*/4, ok, q));  // 2 + 4 > 4 rows
  CHECK_THROWS(Qwen3DSparkModel::SampleSequentialDevice(
      BaseLogits(1, 4), std::vector<int32_t>{0}, /*nqpr=*/5, 0, 4, ok, q));  // size mismatch
}
