// Ported from:
//   vllm/v1/worker/gpu/pool/pooling_runner.py:22-42 (get_supported_tasks, pool,
//     is_valid)
//   the embedding-parity intent of tests/models/language/pooling/test_embedding.py
//     (SKIPPED-DEFERRED as a REAL-model oracle gate — see below)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W3 brick (CLAIM-POOLING): the pooling RUNNER path — forward -> last hidden
// state -> Pooler -> embedding vector (no token sampling). This test drives the
// runner over a SYNTHETIC hidden-state buffer through a real DispatchPooler and
// gates the produced embedding with a COSINE-parity check against an independent
// DOUBLE-PRECISION reference of the exact pooling pipeline (LAST-token pool +
// L2 normalize).
//
// HONEST RESIDUAL: this is a STRUCTURAL (synthetic-weights) cosine gate — it
// proves the runner->pooler->normalize pipeline is correct and self-consistent,
// and that a WRONG pooling type or a MISSING normalize drops the cosine below
// threshold (RED-first). It is NOT the real-model oracle gate: a cosine-vs-vLLM
// check needs a registered concrete embedding model's forward producing real
// hidden states (no such model is registered in our tree yet — the model
// bring-up + `vllm.LLM(task="embed").encode(...)` oracle is the NAMED W3-model /
// W4 residual). We do NOT fabricate a cosine-vs-oracle number.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vllm/v1/worker/gpu/pool/pooling_runner.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::PoolerOutput;
using vllm::PoolingMetadata;
using vllm::PoolingParams;
using vllm::PoolingRunner;
using vllm::PoolingTask;
using vllm::SequencePoolingType;

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }

vt::Tensor Hidden(std::vector<float>& s, int64_t rows, int64_t cols) {
  return vt::Tensor::Contiguous(s.data(), vt::DType::kF32, Cpu(), {rows, cols});
}

vllm::PoolingCursor MakeCursor(const std::vector<int64_t>& prompt_lens,
                               std::vector<int64_t> seq_lens = {}) {
  const int64_t n = static_cast<int64_t>(prompt_lens.size());
  if (seq_lens.empty()) seq_lens = prompt_lens;
  std::vector<int64_t> cumsum(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) cumsum[i + 1] = cumsum[i] + prompt_lens[i];
  vllm::PoolingCursor c;
  c.prompt_lens = prompt_lens;
  c.num_scheduled_tokens = prompt_lens;
  c.seq_lens = seq_lens;
  for (int64_t i = 0; i < n; ++i) {
    c.first_token_indices.push_back(cumsum[i]);
    c.last_token_indices.push_back(cumsum[i + 1] - 1);
  }
  return c;
}

std::vector<PoolingParams> EmbedParams(int n, bool activation) {
  std::vector<PoolingParams> out;
  for (int i = 0; i < n; ++i) {
    PoolingParams p;
    p.task = PoolingTask::kEmbed;
    p.use_activation = activation;
    out.push_back(p);
  }
  return out;
}

// Double-precision cosine similarity between two equal-length vectors.
double Cosine(const std::vector<float>& a, const std::vector<double>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double L2(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * x;
  return std::sqrt(s);
}

// A hidden-state buffer whose LAST token and FIRST (CLS) token point in clearly
// different directions per sequence, so a wrong pool method is separable by
// cosine. The LAST tokens are deliberately NON-unit length (3 and 4) so the
// unit-L2 assertion in the cosine gate is a real RED-first for the normalize
// stage (a missing normalize leaves L2 = 3 / 4, not 1). 2 seqs x 3 tokens x 4
// hidden.
constexpr int64_t kHidden = 4;
std::vector<float> MakeHidden() {
  return {
      // seq0: t0 (CLS) points +x (unit), t1 mixed, t2 (LAST) points +y, length 3
      1, 0, 0, 0,   0.5f, 0.5f, 0, 0,   0, 3, 0, 0,
      // seq1: t0 (CLS) points +z (unit), t1 mixed, t2 (LAST) points +w, length 4
      0, 0, 1, 0,   0, 0, 0.5f, 0.5f,   0, 0, 0, 4,
  };
}

// Double-precision reference: LAST-token pool + L2 normalize, per sequence.
std::vector<std::vector<double>> LastNormalizedReference() {
  std::vector<float> h = MakeHidden();
  std::vector<std::vector<double>> ref;
  for (int64_t seq = 0; seq < 2; ++seq) {
    const int64_t last = seq * 3 + 2;  // last token flat index
    std::vector<double> v(kHidden);
    double n = 0.0;
    for (int64_t c = 0; c < kHidden; ++c) {
      v[c] = static_cast<double>(h[last * kHidden + c]);
      n += v[c] * v[c];
    }
    n = std::sqrt(n);
    for (double& x : v) x /= n;
    ref.push_back(v);
  }
  return ref;
}

}  // namespace

// ===========================================================================
// PoolingRunner.get_supported_tasks (pooling_runner.py:22)
// ===========================================================================
TEST_CASE("PoolingRunner reports the model pooler's supported tasks") {
  vllm::PoolerConfig cfg;
  auto pooler = vllm::DispatchPooler::ForEmbedding(cfg, SequencePoolingType::kLast);
  std::vector<PoolingTask> tasks = PoolingRunner::GetSupportedTasks(*pooler);
  REQUIRE(tasks.size() == 1u);
  CHECK(tasks[0] == PoolingTask::kEmbed);
}

// ===========================================================================
// The pooling runner path + STRUCTURAL cosine-parity gate
// ===========================================================================
TEST_CASE("PoolingRunner.pool produces LAST-normalized embeddings (cosine gate)") {
  vllm::PoolerConfig cfg;
  auto pooler = vllm::DispatchPooler::ForEmbedding(cfg, SequencePoolingType::kLast);
  PoolingRunner runner(*pooler);

  std::vector<float> h = MakeHidden();
  vt::Tensor hidden = Hidden(h, 6, kHidden);
  PoolingMetadata md;
  md.pooling_cursor = MakeCursor({3, 3});
  md.pooling_params = EmbedParams(2, /*activation=*/true);
  md.tasks = {PoolingTask::kEmbed, PoolingTask::kEmbed};

  PoolerOutput out = runner.Pool(hidden, md);
  REQUIRE(out.size() == 2u);

  const auto ref = LastNormalizedReference();
  for (int64_t seq = 0; seq < 2; ++seq) {
    // Each embedding is unit-L2 ...
    CHECK(L2(out[seq]) == doctest::Approx(1.0).epsilon(1e-6));
    // ... and cosine-identical to the double-precision LAST+normalize reference.
    CHECK(Cosine(out[seq], ref[seq]) == doctest::Approx(1.0).epsilon(1e-6));
  }
}

// RED-first: a WRONG pooling type (CLS instead of LAST) selects a token pointing
// in a different direction, so the cosine vs the LAST reference collapses.
TEST_CASE("RED-first: CLS pooling drops cosine vs the LAST reference") {
  vllm::PoolerConfig cfg;
  auto wrong = vllm::DispatchPooler::ForEmbedding(cfg, SequencePoolingType::kCLS);
  PoolingRunner runner(*wrong);

  std::vector<float> h = MakeHidden();
  vt::Tensor hidden = Hidden(h, 6, kHidden);
  PoolingMetadata md;
  md.pooling_cursor = MakeCursor({3, 3});
  md.pooling_params = EmbedParams(2, /*activation=*/true);
  md.tasks = {PoolingTask::kEmbed, PoolingTask::kEmbed};

  PoolerOutput out = runner.Pool(hidden, md);
  const auto ref = LastNormalizedReference();
  for (int64_t seq = 0; seq < 2; ++seq) {
    // CLS token is orthogonal to the LAST token here -> cosine ~ 0, well below
    // any parity threshold (0.99).
    CHECK(Cosine(out[seq], ref[seq]) < 0.5);
  }
}

// RED-first: dropping the normalize activation leaves a non-unit vector.
TEST_CASE("RED-first: missing normalize yields a non-unit embedding") {
  vllm::PoolerConfig cfg;
  auto pooler = vllm::DispatchPooler::ForEmbedding(cfg, SequencePoolingType::kLast);
  PoolingRunner runner(*pooler);

  std::vector<float> h = {3, 4, 0, 0, 6, 8, 0, 0};  // 2 tokens x 4, one seq
  vt::Tensor hidden = Hidden(h, 2, kHidden);
  PoolingMetadata md;
  md.pooling_cursor = MakeCursor({2});
  md.pooling_params = EmbedParams(1, /*activation=*/false);  // normalize OFF
  md.tasks = {PoolingTask::kEmbed};

  PoolerOutput out = runner.Pool(hidden, md);
  REQUIRE(out.size() == 1u);
  // LAST token = [6,8,0,0], norm 10 (not 1) because normalize is disabled.
  CHECK(L2(out[0]) == doctest::Approx(10.0));
}

// ===========================================================================
// PoolingRunner.is_valid (pooling_runner.py:41)
// ===========================================================================
TEST_CASE("PoolingRunner.is_valid flags fully-prefilled sequences") {
  vllm::PoolerConfig cfg;
  auto pooler = vllm::DispatchPooler::ForEmbedding(cfg, SequencePoolingType::kLast);
  PoolingRunner runner(*pooler);

  PoolingMetadata md;
  // seq0 fully prefilled (seq_len == prompt_len); seq1 not (seq_len < prompt_len).
  md.pooling_cursor = MakeCursor({2, 4}, /*seq_lens=*/{2, 3});
  std::vector<uint8_t> valid = runner.ComputeValid(md);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0] == 1);
  CHECK(valid[1] == 0);
}
