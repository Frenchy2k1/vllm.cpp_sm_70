// DeepSeek-V4-Flash W3 UNIT GATE — the genuinely-NEW attention primitives.
// HONEST scope: the full-model gate is multi-Spark-blocked (156.7 GiB, does not
// fit one GB10; forward also needs MHC/MoE, not ported). So W3 gates the MATH
// per-primitive against HAND-DERIVED small cases (literal expected numbers I can
// verify by hand from the vLLM source) PLUS a from-first-principles double-
// precision reference on randomized shapes (rel-L2). This is the "hand-case +
// structural review" bar named in the W3 brief, NOT a dumped-oracle rel-L2 (the
// arch cannot be constructed at a tiny shape — it is a fixed-config 167B). The
// eventual GPU forward (W7) ports the same math into a CUDA kernel; these tests
// are its portable oracle.
//
// Grounded in: v1/attention/ops/triton_fp8_mqa_logits.py:120-156 (MQA logit +
// ReLU), model_executor/layers/sparse_attn_indexer.py:203-207,:488-497,
// models/deepseek_v4/attention.py:70-86,:735,:843 (weight fold + short-context
// select), nvidia/ops/o_proj.py:58-73 (grouped output-LoRA), flashinfer_sparse.py
// :777,:896 + attention.py:219-222 (attention sinks).
#include "vllm/model_executor/models/deepseek_v4_dsa.h"

#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace vllm::deepseek_v4;

namespace {
const float kNegInf = -std::numeric_limits<float>::infinity();

// Relative L2: ||a-b||_2 / max(||b||_2, eps). Reference (b) in double.
double RelL2(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}
}  // namespace

// ── (A) DSA Lightning Indexer ────────────────────────────────────────────────

TEST_CASE("dsv4-dsa: indexer weight fold = weight * D^-0.5 * H^-0.5") {
  // H=4, D=4 -> softmax_scale=1/2, head_scale=1/2, fold=1/4.
  const std::vector<float> wproj = {2.0f, 4.0f, 8.0f, 0.0f};  // [1 token, 4 heads]
  const std::vector<float> folded = DsaIndexerWeightFold(wproj, 1, 4, 4);
  REQUIRE(folded.size() == 4);
  CHECK(folded[0] == doctest::Approx(0.5));   // 2 * 1/4
  CHECK(folded[1] == doctest::Approx(1.0));   // 4 * 1/4
  CHECK(folded[2] == doctest::Approx(2.0));   // 8 * 1/4
  CHECK(folded[3] == doctest::Approx(0.0));
}

TEST_CASE("dsv4-dsa: indexer MQA logit is sum_h w * ReLU(q.k) — ReLU is load-bearing") {
  // 1 query, 3 keys, H=2, D=2. folded_weights passed directly = [1,1].
  //   q_h0=[1,0], q_h1=[0,1];  k0=[1,0], k1=[0,1], k2=[-1,0]
  //   logit[0,0]=ReLU(1)+ReLU(0)=1 ; [0,1]=ReLU(0)+ReLU(1)=1 ;
  //   [0,2]=ReLU(-1->0)+ReLU(0)=0   (WITHOUT ReLU this would be -1)
  const std::vector<float> q = {1, 0, 0, 1};
  const std::vector<float> k = {1, 0, 0, 1, -1, 0};
  const std::vector<float> w = {1, 1};
  const std::vector<int64_t> ws = {0}, we = {3};
  const std::vector<float> lg = DsaIndexerLogits(q, k, w, ws, we, 1, 3, 2, 2);
  REQUIRE(lg.size() == 3);
  CHECK(lg[0] == doctest::Approx(1.0));
  CHECK(lg[1] == doctest::Approx(1.0));
  CHECK(lg[2] == doctest::Approx(0.0));  // ReLU clipped the -1
}

TEST_CASE("dsv4-dsa: indexer causal window sets out-of-range keys to -inf") {
  const std::vector<float> q = {1, 0, 0, 1};
  const std::vector<float> k = {1, 0, 0, 1, 1, 1};
  const std::vector<float> w = {1, 1};
  const std::vector<int64_t> ws = {0}, we = {2};  // key 2 excluded
  const std::vector<float> lg = DsaIndexerLogits(q, k, w, ws, we, 1, 3, 2, 2);
  CHECK(lg[0] == doctest::Approx(1.0));
  CHECK(lg[1] == doctest::Approx(1.0));
  CHECK(lg[2] == kNegInf);  // outside window
}

TEST_CASE("dsv4-dsa: top-k short-context selects EVERY candidate, ascending, -1 pad") {
  // n=3 candidates <= topk=5 -> [0,1,2,-1,-1] (attention.py:70-86).
  const std::vector<float> lg = {9.0f, 1.0f, 5.0f};
  const std::vector<int64_t> ws = {0}, we = {3};
  const std::vector<int64_t> sel = DsaTopkSelect(lg, ws, we, 1, 3, 5);
  REQUIRE(sel.size() == 5);
  CHECK(sel[0] == 0);
  CHECK(sel[1] == 1);
  CHECK(sel[2] == 2);
  CHECK(sel[3] == -1);
  CHECK(sel[4] == -1);
}

TEST_CASE("dsv4-dsa: top-k full picks largest-logit keys, emitted ascending") {
  // logits [5,1,9,3], topk=2 -> {idx2=9, idx0=5} -> ascending [0,2].
  const std::vector<float> lg = {5.0f, 1.0f, 9.0f, 3.0f};
  const std::vector<int64_t> ws = {0}, we = {4};
  const std::vector<int64_t> sel = DsaTopkSelect(lg, ws, we, 1, 4, 2);
  REQUIRE(sel.size() == 2);
  CHECK(sel[0] == 0);
  CHECK(sel[1] == 2);
}

TEST_CASE("dsv4-dsa: top-k ties break toward the smaller key index") {
  // logits [5,5,1], topk=2 -> tie on the two 5s -> {0,1}.
  const std::vector<float> lg = {5.0f, 5.0f, 1.0f};
  const std::vector<int64_t> ws = {0}, we = {3};
  const std::vector<int64_t> sel = DsaTopkSelect(lg, ws, we, 1, 3, 2);
  CHECK(sel[0] == 0);
  CHECK(sel[1] == 1);
}

TEST_CASE("dsv4-dsa: top-k respects the causal window offset") {
  // window [1,4): candidates {1,2,3}; logits there [_,2,9,4], topk=2 -> {2,3}.
  const std::vector<float> lg = {kNegInf, 2.0f, 9.0f, 4.0f};
  const std::vector<int64_t> ws = {1}, we = {4};
  const std::vector<int64_t> sel = DsaTopkSelect(lg, ws, we, 1, 4, 2);
  CHECK(sel[0] == 2);
  CHECK(sel[1] == 3);
}

// ── (B) 512-wide MLA output seams ────────────────────────────────────────────

TEST_CASE("dsv4-mla: sink=-inf reduces to a plain softmax") {
  const std::vector<float> s = {0.0f, 0.0f};
  const std::vector<float> p = SoftmaxWithSink(s, kNegInf);
  CHECK(p[0] == doctest::Approx(0.5));
  CHECK(p[1] == doctest::Approx(0.5));
}

TEST_CASE("dsv4-mla: attention sink removes probability mass from the keys") {
  // scores=[0,0], sink=0 -> denom=3, each key 1/3; probs sum to 2/3 (< 1).
  const std::vector<float> s = {0.0f, 0.0f};
  const std::vector<float> p = SoftmaxWithSink(s, 0.0f);
  CHECK(p[0] == doctest::Approx(1.0 / 3.0));
  CHECK(p[1] == doctest::Approx(1.0 / 3.0));
  CHECK((p[0] + p[1]) == doctest::Approx(2.0 / 3.0));
}

TEST_CASE("dsv4-mla: sink softmax is numerically stable at large logits") {
  const std::vector<float> s = {100.0f, 100.0f};
  const std::vector<float> p = SoftmaxWithSink(s, 100.0f);  // all equal
  CHECK(p[0] == doctest::Approx(1.0 / 3.0));
  CHECK(p[1] == doctest::Approx(1.0 / 3.0));
  CHECK(std::isfinite(p[0]));
}

TEST_CASE("dsv4-mla: grouped output-LoRA (wo_a per-group bmm + wo_b) — hand case") {
  // 1 token, n_heads=2, head_dim=2, n_groups=2 (heads_per_group=1, in_per_group=2),
  // o_lora_rank=2, hidden=2, z_dim=4.
  //   o = [head0=[1,2], head1=[3,4]] -> group0 in=[1,2], group1 in=[3,4]
  //   wo_a[g0]=I -> z0=[1,2] ; wo_a[g1]=[[1,1],[1,-1]] -> z1=[7,-1]
  //   z=[1,2,7,-1]; wo_b row0=[1,0,0,0]->1 ; row1=[0,0,1,1]->6
  const std::vector<float> o = {1, 2, 3, 4};
  const std::vector<float> wo_a = {/*g0*/ 1, 0, 0, 1, /*g1*/ 1, 1, 1, -1};
  const std::vector<float> wo_b = {1, 0, 0, 0, 0, 0, 1, 1};
  const std::vector<float> out =
      GroupedOutputLora(o, wo_a, wo_b, /*T*/ 1, /*heads*/ 2, /*hd*/ 2,
                        /*groups*/ 2, /*rank*/ 2, /*hidden*/ 2);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(1.0));
  CHECK(out[1] == doctest::Approx(6.0));
}

// ── from-first-principles double-precision references (randomized rel-L2) ─────

TEST_CASE("dsv4-dsa: indexer logits vs independent double reference (rel-L2)") {
  std::mt19937 rng(20260728);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  const int64_t T = 3, S = 6, H = 4, D = 8;
  std::vector<float> q(T * H * D), k(S * D), w(T * H);
  for (auto& x : q) x = u(rng);
  for (auto& x : k) x = u(rng);
  for (auto& x : w) x = u(rng);
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = t + 2; }  // varied causal

  const std::vector<float> got =
      DsaIndexerLogits(q, k, w, ws, we, T, S, H, D);

  // Independent double reference (different accumulation order via per-key first).
  std::vector<double> ref(static_cast<size_t>(T) * S, -1e300);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t s = 0; s < S; ++s) {
      if (s < ws[t] || s >= we[t]) continue;
      double acc = 0.0;
      for (int64_t h = 0; h < H; ++h) {
        double dot = 0.0;
        for (int64_t d = 0; d < D; ++d)
          dot += static_cast<double>(q[(t * H + h) * D + d]) * k[s * D + d];
        acc += static_cast<double>(w[t * H + h]) * (dot > 0.0 ? dot : 0.0);
      }
      ref[t * S + s] = acc;
    }
  }
  // Compare only finite (in-window) entries.
  std::vector<float> ga; std::vector<double> ra;
  for (size_t i = 0; i < got.size(); ++i)
    if (std::isfinite(got[i])) { ga.push_back(got[i]); ra.push_back(ref[i]); }
  CHECK(RelL2(ga, ra) < 1e-6);
}

TEST_CASE("dsv4-mla: grouped output-LoRA vs independent double reference (rel-L2)") {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  const int64_t T = 2, heads = 8, hd = 4, groups = 4, rank = 3, hidden = 6;
  const int64_t in_per_group = heads * hd / groups;  // 8
  const int64_t z_dim = groups * rank;               // 12
  std::vector<float> o(T * heads * hd), wo_a(groups * rank * in_per_group),
      wo_b(hidden * z_dim);
  for (auto& x : o) x = u(rng);
  for (auto& x : wo_a) x = u(rng);
  for (auto& x : wo_b) x = u(rng);

  const std::vector<float> got =
      GroupedOutputLora(o, wo_a, wo_b, T, heads, hd, groups, rank, hidden);

  std::vector<double> ref(static_cast<size_t>(T) * hidden, 0.0);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> z(static_cast<size_t>(z_dim), 0.0);
    for (int64_t g = 0; g < groups; ++g)
      for (int64_t d = 0; d < rank; ++d) {
        double acc = 0.0;
        for (int64_t r = 0; r < in_per_group; ++r)
          acc += static_cast<double>(wo_a[(g * rank + d) * in_per_group + r]) *
                 o[t * heads * hd + g * in_per_group + r];
        z[g * rank + d] = acc;
      }
    for (int64_t h = 0; h < hidden; ++h) {
      double acc = 0.0;
      for (int64_t c = 0; c < z_dim; ++c)
        acc += static_cast<double>(wo_b[h * z_dim + c]) * z[c];
      ref[t * hidden + h] = acc;
    }
  }
  CHECK(RelL2(got, ref) < 1e-6);
}
