// Ported from:
//   tests/model_executor/layers/test_pooler_heads.py:56-294
//     (TestEmbeddingPoolerHead, TestClassifierPoolerHead)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING). The head cases are recomputed here in DOUBLE
// PRECISION (the projector/classifier are deterministic small matmuls evaluated
// twice — once by the head, once as the reference — so a head that drops the
// projector / matryoshka slice / calibration / activation fails). RED-first
// verified during bring-up by disabling each stage. Also covers the
// `SequencePooler` / `DispatchPooler` composite over the landed W1 pooling
// methods (no standalone upstream unit module — upstream exercises these through
// the pooling model tests; ported here as the composite gate).
//
// The upstream `test_list_input_gets_stacked` and `test_head_dtype` cases are
// torch return-type / dtype nuances (list-vs-stacked Tensor, fp16 cast) with no
// numeric effect on our per-sequence PoolerOutput / float32 CPU path; NOT
// ported (recorded deviation, see heads.h). The tokwise `TestTokenEmbedding/
// TokenClassifierPoolerHead` classes are the W5 residual (SKIPPED-DEFERRED).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "vllm/model_executor/layers/pooler/activations.h"
#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"
#include "vllm/model_executor/layers/pooler/heads.h"
#include "vllm/model_executor/layers/pooler/methods.h"
#include "vllm/model_executor/layers/pooler/poolers.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::ClassifierPoolerHead;
using vllm::EmbeddingPoolerHead;
using vllm::PooledData;
using vllm::PoolerNormalize;
using vllm::PoolerOutput;
using vllm::PoolingMetadata;
using vllm::PoolingParams;
using vllm::PoolingTask;
using vllm::SequencePoolingType;

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }

// A batch of pooled_data with deterministic distinct values.
PooledData MakePooled(int64_t rows, int64_t cols) {
  PooledData d{std::vector<float>(static_cast<size_t>(rows * cols)), rows, cols};
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      d.data[r * cols + c] = static_cast<float>(0.5 * (r + 1) - 0.25 * c + 0.1);
    }
  }
  return d;
}

PoolingMetadata MetaFor(std::vector<PoolingParams> params) {
  PoolingMetadata md;
  md.pooling_params = std::move(params);
  return md;
}

std::vector<PoolingParams> Params(int n, PoolingTask task,
                                  std::optional<int64_t> dims = std::nullopt,
                                  std::optional<bool> use_act = std::nullopt) {
  std::vector<PoolingParams> out;
  for (int i = 0; i < n; ++i) {
    PoolingParams p;
    p.task = task;
    p.dimensions = dims;
    p.use_activation = use_act;
    out.push_back(p);
  }
  return out;
}

// A deterministic bias-free linear W[out][in]; y[r][o] = sum_i x[r][i]*W[o][i].
struct Linear {
  int64_t in_f, out_f;
  std::vector<double> w;  // out_f x in_f
  Linear(int64_t in_f_, int64_t out_f_) : in_f(in_f_), out_f(out_f_), w(out_f_ * in_f_) {
    for (int64_t o = 0; o < out_f; ++o)
      for (int64_t i = 0; i < in_f; ++i)
        w[o * in_f + i] = 0.05 * (o + 1) - 0.03 * i + 0.02 * ((o + i) % 3);
  }
  PooledData operator()(const PooledData& x) const {
    PooledData y{std::vector<float>(static_cast<size_t>(x.rows * out_f)), x.rows, out_f};
    for (int64_t r = 0; r < x.rows; ++r)
      for (int64_t o = 0; o < out_f; ++o) {
        double acc = 0.0;
        for (int64_t i = 0; i < in_f; ++i) acc += static_cast<double>(x.At(r, i)) * w[o * in_f + i];
        y.data[r * out_f + o] = static_cast<float>(acc);
      }
    return y;
  }
};

double L2(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * x;
  return std::sqrt(s);
}

constexpr int64_t kHidden = 6;
constexpr int64_t kBatch = 3;

}  // namespace

// ===========================================================================
// EmbeddingPoolerHead (test_pooler_heads.py:56)
// ===========================================================================
TEST_CASE("EmbeddingPoolerHead supported tasks == {embed}") {
  CHECK(EmbeddingPoolerHead().GetSupportedTasks() ==
        std::set<PoolingTask>{PoolingTask::kEmbed});
}

TEST_CASE("EmbeddingPoolerHead passthrough (no projector/activation)") {
  EmbeddingPoolerHead head;
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed)));
  REQUIRE(out.size() == static_cast<size_t>(kBatch));
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < kHidden; ++c) CHECK(out[r][c] == x.At(r, c));
}

TEST_CASE("EmbeddingPoolerHead applies the projector") {
  Linear proj(kHidden, 4);
  EmbeddingPoolerHead head(proj, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed)));
  PooledData ref = proj(x);
  for (int64_t r = 0; r < kBatch; ++r) {
    REQUIRE(out[r].size() == 4u);
    for (int64_t c = 0; c < 4; ++c) CHECK(out[r][c] == doctest::Approx(ref.At(r, c)));
  }
}

TEST_CASE("EmbeddingPoolerHead matryoshka uniform truncation") {
  EmbeddingPoolerHead head;
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed, 4)));
  for (int64_t r = 0; r < kBatch; ++r) {
    REQUIRE(out[r].size() == 4u);
    for (int64_t c = 0; c < 4; ++c) CHECK(out[r][c] == x.At(r, c));
  }
}

TEST_CASE("EmbeddingPoolerHead matryoshka mixed dims") {
  EmbeddingPoolerHead head;
  PooledData x = MakePooled(2, kHidden);
  std::vector<PoolingParams> params = Params(2, PoolingTask::kEmbed);
  params[0].dimensions = 4;
  params[1].dimensions = 6;
  PoolerOutput out = head.Forward(x, MetaFor(params));
  CHECK(out[0].size() == 4u);
  CHECK(out[1].size() == 6u);
}

TEST_CASE("EmbeddingPoolerHead matryoshka mixed with a None dim") {
  EmbeddingPoolerHead head;
  PooledData x = MakePooled(2, kHidden);
  std::vector<PoolingParams> params = Params(2, PoolingTask::kEmbed);
  params[0].dimensions = 4;  // params[1].dimensions stays unset
  PoolerOutput out = head.Forward(x, MetaFor(params));
  REQUIRE(out[0].size() == 4u);
  for (int64_t c = 0; c < 4; ++c) CHECK(out[0][c] == x.At(0, c));
  REQUIRE(out[1].size() == static_cast<size_t>(kHidden));
  for (int64_t c = 0; c < kHidden; ++c) CHECK(out[1][c] == x.At(1, c));
}

TEST_CASE("EmbeddingPoolerHead activation uniform true yields unit-L2 rows") {
  EmbeddingPoolerHead head(nullptr, std::make_shared<PoolerNormalize>());
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out =
      head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed, std::nullopt, true)));
  for (int64_t r = 0; r < kBatch; ++r) CHECK(L2(out[r]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("EmbeddingPoolerHead activation uniform false is passthrough") {
  EmbeddingPoolerHead head(nullptr, std::make_shared<PoolerNormalize>());
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out =
      head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed, std::nullopt, false)));
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < kHidden; ++c) CHECK(out[r][c] == x.At(r, c));
}

TEST_CASE("EmbeddingPoolerHead activation mixed flags") {
  EmbeddingPoolerHead head(nullptr, std::make_shared<PoolerNormalize>());
  PooledData x = MakePooled(2, kHidden);
  std::vector<PoolingParams> params = Params(2, PoolingTask::kEmbed);
  params[0].use_activation = true;
  params[1].use_activation = false;
  PoolerOutput out = head.Forward(x, MetaFor(params));
  CHECK(L2(out[0]) == doctest::Approx(1.0).epsilon(1e-6));
  for (int64_t c = 0; c < kHidden; ++c) CHECK(out[1][c] == x.At(1, c));
}

TEST_CASE("EmbeddingPoolerHead projector then matryoshka") {
  Linear proj(kHidden, 5);
  EmbeddingPoolerHead head(proj, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed, 3)));
  PooledData ref = proj(x);
  for (int64_t r = 0; r < kBatch; ++r) {
    REQUIRE(out[r].size() == 3u);
    for (int64_t c = 0; c < 3; ++c) CHECK(out[r][c] == doctest::Approx(ref.At(r, c)));
  }
}

TEST_CASE("EmbeddingPoolerHead matryoshka then activation (order: slice, normalize)") {
  EmbeddingPoolerHead head(nullptr, std::make_shared<PoolerNormalize>());
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out =
      head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kEmbed, 4, true)));
  for (int64_t r = 0; r < kBatch; ++r) {
    REQUIRE(out[r].size() == 4u);
    // normalize is over the SLICED 4-vector: double reference of x[r][:4]/||.||.
    double n = 0.0;
    for (int64_t c = 0; c < 4; ++c) n += static_cast<double>(x.At(r, c)) * x.At(r, c);
    n = std::sqrt(n);
    for (int64_t c = 0; c < 4; ++c)
      CHECK(out[r][c] == doctest::Approx(x.At(r, c) / n).epsilon(1e-6));
  }
}

TEST_CASE("EmbeddingPoolerHead empty batch yields no rows") {
  EmbeddingPoolerHead head;
  PooledData x{{}, 0, kHidden};
  PoolerOutput out = head.Forward(x, MetaFor({}));
  CHECK(out.empty());
}

TEST_CASE("EmbeddingPoolerHead rejects a params-length mismatch") {
  EmbeddingPoolerHead head;
  PooledData x = MakePooled(kBatch, kHidden);
  CHECK_THROWS_AS(head.Forward(x, MetaFor(Params(kBatch - 1, PoolingTask::kEmbed))),
                  std::runtime_error);
}

// ===========================================================================
// ClassifierPoolerHead (test_pooler_heads.py:191)
// ===========================================================================
TEST_CASE("ClassifierPoolerHead supported tasks == {classify}") {
  CHECK(ClassifierPoolerHead().GetSupportedTasks() ==
        std::set<PoolingTask>{PoolingTask::kClassify});
}

TEST_CASE("ClassifierPoolerHead passthrough") {
  ClassifierPoolerHead head;
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kClassify)));
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < kHidden; ++c) CHECK(out[r][c] == x.At(r, c));
}

TEST_CASE("ClassifierPoolerHead applies the classifier") {
  Linear clf(kHidden, 3);
  ClassifierPoolerHead head(clf, std::nullopt, std::nullopt, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kClassify)));
  PooledData ref = clf(x);
  for (int64_t r = 0; r < kBatch; ++r) {
    REQUIRE(out[r].size() == 3u);
    for (int64_t c = 0; c < 3; ++c) CHECK(out[r][c] == doctest::Approx(ref.At(r, c)));
  }
}

TEST_CASE("ClassifierPoolerHead logit_mean subtracts") {
  ClassifierPoolerHead head(nullptr, 2.0, std::nullopt, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kClassify)));
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < kHidden; ++c)
      CHECK(out[r][c] == doctest::Approx(x.At(r, c) - 2.0));
}

TEST_CASE("ClassifierPoolerHead logit_sigma divides") {
  ClassifierPoolerHead head(nullptr, std::nullopt, 0.5, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kClassify)));
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < kHidden; ++c)
      CHECK(out[r][c] == doctest::Approx(x.At(r, c) / 0.5));
}

TEST_CASE("ClassifierPoolerHead Platt scaling (logit - mean) / sigma") {
  ClassifierPoolerHead head(nullptr, 1.0, 2.0, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kClassify)));
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < kHidden; ++c)
      CHECK(out[r][c] == doctest::Approx((x.At(r, c) - 1.0) / 2.0));
}

TEST_CASE("ClassifierPoolerHead classifier then Platt scaling") {
  Linear clf(kHidden, 3);
  ClassifierPoolerHead head(clf, 1.0, 2.0, nullptr);
  PooledData x = MakePooled(kBatch, kHidden);
  PoolerOutput out = head.Forward(x, MetaFor(Params(kBatch, PoolingTask::kClassify)));
  PooledData logits = clf(x);
  for (int64_t r = 0; r < kBatch; ++r)
    for (int64_t c = 0; c < 3; ++c)
      CHECK(out[r][c] == doctest::Approx((logits.At(r, c) - 1.0) / 2.0));
}

TEST_CASE("ClassifierPoolerHead activation true/false") {
  ClassifierPoolerHead head(nullptr, std::nullopt, std::nullopt,
                            std::make_shared<PoolerNormalize>());
  PooledData x = MakePooled(2, kHidden);
  std::vector<PoolingParams> params = Params(2, PoolingTask::kClassify);
  params[0].use_activation = true;
  params[1].use_activation = false;
  PoolerOutput out = head.Forward(x, MetaFor(params));
  CHECK(L2(out[0]) == doctest::Approx(1.0).epsilon(1e-6));
  for (int64_t c = 0; c < kHidden; ++c) CHECK(out[1][c] == x.At(1, c));
}

TEST_CASE("ClassifierPoolerHead empty batch yields no rows") {
  ClassifierPoolerHead head;
  PooledData x{{}, 0, kHidden};
  PoolerOutput out = head.Forward(x, MetaFor({}));
  CHECK(out.empty());
}

// ===========================================================================
// SequencePooler + DispatchPooler composite (over the W1 pooling methods)
// ===========================================================================
namespace {
// cumsum cursor helper (mirror test_pooler.cpp MakeCursor).
vllm::PoolingCursor MakeCursor(const std::vector<int64_t>& prompt_lens) {
  const int64_t n = static_cast<int64_t>(prompt_lens.size());
  std::vector<int64_t> cumsum(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) cumsum[i + 1] = cumsum[i] + prompt_lens[i];
  vllm::PoolingCursor c;
  c.prompt_lens = prompt_lens;
  c.num_scheduled_tokens = prompt_lens;
  c.seq_lens = prompt_lens;
  for (int64_t i = 0; i < n; ++i) {
    c.first_token_indices.push_back(cumsum[i]);
    c.last_token_indices.push_back(cumsum[i + 1] - 1);
  }
  return c;
}
vt::Tensor Hidden(std::vector<float>& s, int64_t rows, int64_t cols) {
  return vt::Tensor::Contiguous(s.data(), vt::DType::kF32, Cpu(), {rows, cols});
}
}  // namespace

TEST_CASE("SequencePooler (embed factory) pools LAST + normalizes") {
  vllm::PoolerConfig cfg;  // seq_pooling_type unset -> LAST default
  auto pooler = vllm::PoolerForEmbed(cfg, SequencePoolingType::kLast);
  CHECK(pooler->GetSupportedTasks() == std::set<PoolingTask>{PoolingTask::kEmbed});

  std::vector<float> h = {1, 2, 3, 4, 5, 6, 7, 8};  // 4 tokens x 2
  vt::Tensor hidden = Hidden(h, 4, 2);
  PoolingMetadata md;
  md.pooling_cursor = MakeCursor({2, 2});
  md.pooling_params = Params(2, PoolingTask::kEmbed, std::nullopt, true);
  PoolerOutput out = pooler->Forward(hidden, md);
  REQUIRE(out.size() == 2u);
  // seq0 last token = [3,4] -> /5 ; seq1 last = [7,8] -> /sqrt(113).
  CHECK(out[0][0] == doctest::Approx(3.0 / 5.0));
  CHECK(out[0][1] == doctest::Approx(4.0 / 5.0));
  const double n1 = std::sqrt(7.0 * 7.0 + 8.0 * 8.0);
  CHECK(out[1][0] == doctest::Approx(7.0 / n1));
  CHECK(L2(out[1]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("PoolerForClassify supports only classify") {
  vllm::PoolerConfig cfg;
  auto pooler = vllm::PoolerForClassify(cfg, SequencePoolingType::kLast);
  CHECK(pooler->GetSupportedTasks() == std::set<PoolingTask>{PoolingTask::kClassify});
}

TEST_CASE("DispatchPooler.ForEmbedding routes the embed task") {
  vllm::PoolerConfig cfg;
  auto disp = vllm::DispatchPooler::ForEmbedding(cfg, SequencePoolingType::kLast);
  CHECK(disp->GetSupportedTasks() == std::set<PoolingTask>{PoolingTask::kEmbed});

  std::vector<float> h = {1, 2, 3, 4, 5, 6};  // 3 tokens x 2
  vt::Tensor hidden = Hidden(h, 3, 2);
  PoolingMetadata md;
  md.pooling_cursor = MakeCursor({3});
  md.pooling_params = Params(1, PoolingTask::kEmbed, std::nullopt, true);
  md.tasks = {PoolingTask::kEmbed};
  PoolerOutput out = disp->Forward(hidden, md);
  REQUIRE(out.size() == 1u);
  CHECK(L2(out[0]) == doctest::Approx(1.0).epsilon(1e-6));  // last token normalized
}

TEST_CASE("DispatchPooler routes a MIXED embed+classify batch by task groups") {
  vllm::PoolerConfig cfg;
  std::map<PoolingTask, std::unique_ptr<vllm::Pooler>> poolers;
  poolers.emplace(PoolingTask::kEmbed,
                  vllm::PoolerForEmbed(cfg, SequencePoolingType::kLast));
  poolers.emplace(PoolingTask::kClassify,
                  vllm::PoolerForClassify(cfg, SequencePoolingType::kLast));
  vllm::DispatchPooler disp(std::move(poolers));
  CHECK(disp.GetSupportedTasks() ==
        std::set<PoolingTask>{PoolingTask::kEmbed, PoolingTask::kClassify});

  // 3 seqs: [embed, embed, classify], 1 token each.
  std::vector<float> h = {1, 0, 0, 1, 3, 4};
  vt::Tensor hidden = Hidden(h, 3, 2);
  PoolingMetadata md;
  md.pooling_cursor = MakeCursor({1, 1, 1});
  md.pooling_params = {PoolingParams{}, PoolingParams{}, PoolingParams{}};
  md.pooling_params[0].task = PoolingTask::kEmbed;
  md.pooling_params[0].use_activation = true;
  md.pooling_params[1].task = PoolingTask::kEmbed;
  md.pooling_params[1].use_activation = true;
  md.pooling_params[2].task = PoolingTask::kClassify;  // no activation set
  md.tasks = {PoolingTask::kEmbed, PoolingTask::kEmbed, PoolingTask::kClassify};
  PoolerOutput out = disp.Forward(hidden, md);
  REQUIRE(out.size() == 3u);
  // Rows 0,1 are embed -> unit norm; row 2 is classify passthrough of [3,4].
  CHECK(L2(out[0]) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(L2(out[1]) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(out[2][0] == doctest::Approx(3.0));
  CHECK(out[2][1] == doctest::Approx(4.0));
}

TEST_CASE("DispatchPooler ctor rejects a pooler that does not support its task") {
  vllm::PoolerConfig cfg;
  std::map<PoolingTask, std::unique_ptr<vllm::Pooler>> poolers;
  // Register an embed pooler under the classify key -> must throw.
  poolers.emplace(PoolingTask::kClassify,
                  vllm::PoolerForEmbed(cfg, SequencePoolingType::kLast));
  CHECK_THROWS_AS(vllm::DispatchPooler(std::move(poolers)), std::runtime_error);
}
