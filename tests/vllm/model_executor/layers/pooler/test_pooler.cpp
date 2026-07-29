// Ported from:
//   tests/model_executor/layers/test_pooler_methods.py:110-260
//     (TestCLSPool, TestLastPool, TestMeanPool, TestGetSeqPoolingMethod)
//   tests/model_executor/layers/test_pooler_activations.py:24-158
//     (TestPoolerIdentity, TestPoolerNormalize, TestPoolerMultiLabelClassify,
//      TestPoolerClassify)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W1 brick (CLAIM-POOLING). Every expected value is an independent
// DOUBLE-PRECISION reference computed here (not a torch dump): the CLS/LAST
// cases are exact gathers, MeanPool and the activations are recomputed in double
// and compared with a float tolerance. RED-first: with any pooling method or
// activation stubbed to a no-op / wrong reduction, the arithmetic subcases below
// fail (verified during bring-up by disabling the MeanPool division and the
// PoolerNormalize denominator).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/pooler/activations.h"
#include "vllm/model_executor/layers/pooler/methods.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }

// Mirror of the upstream _make_pooling_cursor helper
// (test_pooler_methods.py:27): cumsum over num_scheduled_tokens gives the flat
// first/last token indices into the packed hidden-state buffer.
vllm::PoolingCursor MakeCursor(const std::vector<int64_t>& prompt_lens,
                               std::vector<int64_t> num_scheduled = {},
                               std::vector<int64_t> seq_lens = {}) {
  if (num_scheduled.empty()) num_scheduled = prompt_lens;
  if (seq_lens.empty()) seq_lens = prompt_lens;
  const int64_t n = static_cast<int64_t>(prompt_lens.size());
  std::vector<int64_t> cumsum(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) cumsum[i + 1] = cumsum[i] + num_scheduled[i];
  vllm::PoolingCursor cursor;
  cursor.prompt_lens = prompt_lens;
  cursor.num_scheduled_tokens = num_scheduled;
  cursor.seq_lens = seq_lens;
  for (int64_t i = 0; i < n; ++i) {
    cursor.first_token_indices.push_back(cumsum[i]);
    cursor.last_token_indices.push_back(cumsum[i + 1] - 1);
  }
  return cursor;
}

vllm::PoolingMetadata MakeMetadata(const std::vector<int64_t>& prompt_lens,
                                   std::vector<int64_t> num_scheduled = {},
                                   std::vector<int64_t> seq_lens = {}) {
  vllm::PoolingMetadata md;
  md.pooling_cursor = MakeCursor(prompt_lens, std::move(num_scheduled),
                                 std::move(seq_lens));
  return md;
}

vt::Tensor Hidden(std::vector<float>& storage, int64_t rows, int64_t cols) {
  return vt::Tensor::Contiguous(storage.data(), vt::DType::kF32, Cpu(),
                                {rows, cols});
}

}  // namespace

using vllm::CLSPool;
using vllm::LastPool;
using vllm::MeanPool;
using vllm::PooledData;

// ---------------------------------------------------------------------------
// CLSPool (test_pooler_methods.py:110)
// ---------------------------------------------------------------------------
TEST_CASE("CLSPool extracts first token of each sequence") {
  std::vector<float> h = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  vt::Tensor hidden = Hidden(h, 5, 2);
  PooledData out = CLSPool().Forward(hidden, MakeMetadata({2, 3}));
  REQUIRE(out.rows == 2);
  REQUIRE(out.cols == 2);
  CHECK(out.At(0, 0) == 1.0f);  // token 0
  CHECK(out.At(0, 1) == 2.0f);
  CHECK(out.At(1, 0) == 5.0f);  // token 2
  CHECK(out.At(1, 1) == 6.0f);
}

TEST_CASE("CLSPool rejects partial prefill") {
  std::vector<float> h = {1, 2, 3, 4, 5, 6};
  vt::Tensor hidden = Hidden(h, 3, 2);
  CHECK_THROWS_WITH_AS(CLSPool().Forward(hidden, MakeMetadata({3}, {2})),
                       doctest::Contains("partial prefill"),
                       std::runtime_error);
}

// ---------------------------------------------------------------------------
// LastPool (test_pooler_methods.py:130)
// ---------------------------------------------------------------------------
TEST_CASE("LastPool extracts last token of each sequence") {
  std::vector<float> h = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  vt::Tensor hidden = Hidden(h, 5, 2);
  PooledData out = LastPool().Forward(hidden, MakeMetadata({2, 3}));
  CHECK(out.At(0, 0) == 3.0f);   // token 1
  CHECK(out.At(0, 1) == 4.0f);
  CHECK(out.At(1, 0) == 9.0f);   // token 4
  CHECK(out.At(1, 1) == 10.0f);
}

TEST_CASE("LastPool with partial prefill extracts last scheduled token") {
  std::vector<float> h = {1, 2, 3, 4};
  vt::Tensor hidden = Hidden(h, 2, 2);
  PooledData out = LastPool().Forward(hidden, MakeMetadata({4}, {2}));
  REQUIRE(out.rows == 1);
  CHECK(out.At(0, 0) == 3.0f);
  CHECK(out.At(0, 1) == 4.0f);
}

// ---------------------------------------------------------------------------
// MeanPool (test_pooler_methods.py:151) — double-precision reference
// ---------------------------------------------------------------------------
TEST_CASE("MeanPool computes the per-sequence mean") {
  std::vector<float> h = {1, 2, 3, 4, 10, 20};
  vt::Tensor hidden = Hidden(h, 3, 2);
  PooledData out = MeanPool().Forward(hidden, MakeMetadata({2, 1}));
  // seq0 = mean([[1,2],[3,4]]) = [2,3]; seq1 = [10,20]
  CHECK(out.At(0, 0) == doctest::Approx(2.0));
  CHECK(out.At(0, 1) == doctest::Approx(3.0));
  CHECK(out.At(1, 0) == doctest::Approx(10.0));
  CHECK(out.At(1, 1) == doctest::Approx(20.0));
}

TEST_CASE("MeanPool single token is identity") {
  std::vector<float> h = {5, 10};
  vt::Tensor hidden = Hidden(h, 1, 2);
  PooledData out = MeanPool().Forward(hidden, MakeMetadata({1}));
  CHECK(out.At(0, 0) == doctest::Approx(5.0));
  CHECK(out.At(0, 1) == doctest::Approx(10.0));
}

TEST_CASE("MeanPool over multiple sequences (double reference)") {
  std::vector<float> h = {0, 0, 2, 4, 4, 8, 10, 10};
  vt::Tensor hidden = Hidden(h, 4, 2);
  PooledData out = MeanPool().Forward(hidden, MakeMetadata({3, 1}));
  // seq0 = mean of rows 0..2 = [(0+2+4)/3, (0+4+8)/3] = [2,4]
  const double ref00 = (0.0 + 2.0 + 4.0) / 3.0;
  const double ref01 = (0.0 + 4.0 + 8.0) / 3.0;
  CHECK(out.At(0, 0) == doctest::Approx(ref00));
  CHECK(out.At(0, 1) == doctest::Approx(ref01));
  CHECK(out.At(1, 0) == doctest::Approx(10.0));
  CHECK(out.At(1, 1) == doctest::Approx(10.0));
}

TEST_CASE("MeanPool empty batch yields (0, hidden)") {
  // An empty batch (num_seqs == 0) still carries the hidden width; the vt CPU
  // tensor forbids a 0-length dim, so the buffer is shaped [1, 8] and the empty
  // cursor drives the (0, hidden) result — the pooler only reads shape[1].
  std::vector<float> h(8, 0.0f);
  vt::Tensor hidden = Hidden(h, 1, 8);
  PooledData out = MeanPool().Forward(hidden, MakeMetadata({}));
  CHECK(out.rows == 0);
  CHECK(out.cols == 8);
}

TEST_CASE("MeanPool rejects partial prefill") {
  std::vector<float> h = {1, 2, 3, 4};
  vt::Tensor hidden = Hidden(h, 2, 2);
  CHECK_THROWS_WITH_AS(MeanPool().Forward(hidden, MakeMetadata({3}, {2})),
                       doctest::Contains("partial prefill"),
                       std::runtime_error);
}

TEST_CASE("MeanPool upcasts a float16 input to float32") {
  // test_pooler_methods.py:250 — mean([[1,2],[3,4]]) = [2,3], f16 in.
  std::vector<uint16_t> h = {vt::F32ToF16(1.0f), vt::F32ToF16(2.0f),
                             vt::F32ToF16(3.0f), vt::F32ToF16(4.0f)};
  vt::Tensor hidden = vt::Tensor::Contiguous(h.data(), vt::DType::kF16, Cpu(),
                                             {2, 2});
  PooledData out = MeanPool().Forward(hidden, MakeMetadata({2}));
  CHECK(out.At(0, 0) == doctest::Approx(2.0).epsilon(0.01));
  CHECK(out.At(0, 1) == doctest::Approx(3.0).epsilon(0.01));
}

// ---------------------------------------------------------------------------
// get_seq_pooling_method factory (test_pooler_methods.py:264)
// ---------------------------------------------------------------------------
TEST_CASE("GetSeqPoolingMethod resolves CLS/LAST/MEAN and rejects unknown") {
  std::vector<float> h = {1, 2, 3, 4, 5, 6};
  vt::Tensor hidden = Hidden(h, 3, 2);
  auto md = MakeMetadata({3});

  // CLS -> first token [1,2]; LAST -> last [5,6]; MEAN -> [3,4].
  CHECK(vllm::GetSeqPoolingMethod("CLS")->Forward(hidden, md).At(0, 0) == 1.0f);
  CHECK(vllm::GetSeqPoolingMethod("LAST")->Forward(hidden, md).At(0, 0) == 5.0f);
  CHECK(vllm::GetSeqPoolingMethod("MEAN")->Forward(hidden, md).At(0, 0) ==
        doctest::Approx(3.0));
  CHECK(vllm::GetSeqPoolingMethod(vllm::SequencePoolingType::kMean)
            ->Forward(hidden, md)
            .At(0, 1) == doctest::Approx(4.0));
  CHECK_THROWS_WITH_AS(vllm::GetSeqPoolingMethod(std::string("UNKNOWN")),
                       doctest::Contains("UNKNOWN"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// PoolerIdentity (test_pooler_activations.py:24)
// ---------------------------------------------------------------------------
TEST_CASE("PoolerIdentity returns input unchanged") {
  PooledData d{{-1.0f, 2.0f, -3.0f, 4.0f}, 2, 2};
  const std::vector<float> before = d.data;
  vllm::PoolerIdentity().Apply(d);
  CHECK(d.data == before);
}

// ---------------------------------------------------------------------------
// PoolerNormalize (test_pooler_activations.py:44) — double-precision reference
// ---------------------------------------------------------------------------
TEST_CASE("PoolerNormalize yields unit-L2 rows and the exact direction") {
  PooledData d{{3.0f, 4.0f, 1.0f, 2.0f}, 2, 2};
  vllm::PoolerNormalize().Apply(d);
  // Row 0: [3,4]/5 = [0.6,0.8]; row 1: [1,2]/sqrt(5).
  for (int64_t r = 0; r < 2; ++r) {
    const double n = std::hypot(static_cast<double>(d.At(r, 0)),
                                static_cast<double>(d.At(r, 1)));
    CHECK(n == doctest::Approx(1.0).epsilon(1e-6));
  }
  CHECK(d.At(0, 0) == doctest::Approx(0.6));
  CHECK(d.At(0, 1) == doctest::Approx(0.8));
  const double denom = std::sqrt(5.0);
  CHECK(d.At(1, 0) == doctest::Approx(1.0 / denom));
  CHECK(d.At(1, 1) == doctest::Approx(2.0 / denom));
}

// ---------------------------------------------------------------------------
// PoolerMultiLabelClassify (test_pooler_activations.py:70)
// ---------------------------------------------------------------------------
TEST_CASE("PoolerMultiLabelClassify sigmoids into (0,1)") {
  PooledData d{{0.0f, 100.0f, -100.0f}, 1, 3};
  vllm::PoolerMultiLabelClassify().Apply(d);
  CHECK(d.At(0, 0) == doctest::Approx(0.5));
  CHECK(d.At(0, 1) == doctest::Approx(1.0).epsilon(1e-4));
  CHECK(d.At(0, 2) == doctest::Approx(0.0).epsilon(1e-4));
}

// ---------------------------------------------------------------------------
// PoolerClassify (test_pooler_activations.py:97)
// ---------------------------------------------------------------------------
TEST_CASE("PoolerClassify infers softmax from row width when unset") {
  PooledData d{{1.0f, 2.0f, 3.0f, 0.5f, 0.5f, 0.5f}, 2, 3};
  vllm::PoolerClassify().Apply(d);  // num_labels unset -> width 3 -> softmax
  for (int64_t r = 0; r < 2; ++r) {
    double s = 0.0;
    for (int64_t c = 0; c < 3; ++c) s += d.At(r, c);
    CHECK(s == doctest::Approx(1.0));
  }
  // Uniform row -> exactly 1/3 each (double reference).
  CHECK(d.At(1, 0) == doctest::Approx(1.0 / 3.0));
}

TEST_CASE("PoolerClassify uses sigmoid when num_labels < 2") {
  PooledData d{{0.0f}, 1, 1};
  vllm::PoolerClassify(1).Apply(d);
  CHECK(d.At(0, 0) == doctest::Approx(0.5));
}

TEST_CASE("PoolerClassify uses softmax when num_labels >= 2") {
  PooledData d{{1.0f, 1.0f, 1.0f, 1.0f}, 1, 4};
  vllm::PoolerClassify(4).Apply(d);
  double s = std::accumulate(d.data.begin(), d.data.end(), 0.0);
  CHECK(s == doctest::Approx(1.0));
  CHECK(d.At(0, 0) == doctest::Approx(0.25));
}
