// Kimi Delta Attention (KDA) net-new primitives — UNIT GATE.
// HONEST scope: the end-to-end gate is the Kimi-Linear-48B-A3B proxy vs the
// pinned vLLM oracle on GB10 (DGX-blocked; the 2.8T Kimi-K3 does not fit one
// GB10). So this gates the KDA-specific MATH per-primitive against HAND-DERIVED
// literal cases (expected numbers verifiable by hand from the vLLM source) PLUS
// a from-first-principles double-precision reference on randomized shapes
// (rel-L2). This is the "host-reference + structural-review" bar, NOT a
// dumped-oracle rel-L2. The eventual GPU forward ports the same math into a CUDA
// kernel; these tests are its portable oracle. Mirrors the DeepSeek-V4 DSA lane
// (tests/vllm/models/test_deepseek_v4_dsa.cpp).
//
// Grounded in: kimi_gdn_linear_attn.py:142-156,:171-198,:219,:245,:266,:324-356,
// :403-425; third_party/flash_linear_attention/ops/kda.py:463-487,:1182-1254,
// :1541-1600,:1603-1646; ops/l2norm.py:42-43,:96.
#include "vllm/model_executor/models/kimi_kda.h"

#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

using namespace vllm::kimi_kda;

namespace {
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
double Sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

// ── (1) low-rank decay projection ────────────────────────────────────────────

TEST_CASE("kda: low-rank decay is f_b @ f_a @ x (the bottleneck) — hand case") {
  // hidden=2, head_dim=2 (rank), H=1 -> g1 [1,2].
  // f_a = I -> r = x = [1,2]; f_b = [[1,1],[1,-1]] -> g1 = [3, -1].
  const std::vector<float> x = {1, 2};
  const std::vector<float> f_a = {1, 0, 0, 1};
  const std::vector<float> f_b = {1, 1, 1, -1};
  const std::vector<float> g1 = KdaLowRankDecay(x, f_a, f_b, 1, 2, 1, 2);
  REQUIRE(g1.size() == 2);
  CHECK(g1[0] == doctest::Approx(3.0));
  CHECK(g1[1] == doctest::Approx(-1.0));
}

// ── (2) the decay gate ───────────────────────────────────────────────────────

TEST_CASE("kda: decay gate = -exp(A_log)*softplus(g1), softplus linearises >thr") {
  // H=1, D=2, A_log=0 -> b_a=-1. g1=[0, 100]: softplus(0)=ln2, 100>20 -> linear.
  const std::vector<float> g1 = {0.0f, 100.0f};
  const std::vector<float> a_log = {0.0f};
  const std::vector<float> y = KdaDecayGate(g1, a_log, /*dt_bias*/ {}, 1, 1, 2);
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(-std::log(2.0)));  // -1 * ln2
  CHECK(y[1] == doctest::Approx(-100.0));          // -1 * 100 (linear regime)
}

TEST_CASE("kda: decay gate applies A_log per head and dt_bias per channel") {
  // A_log = ln2 -> b_a = -exp(ln2) = -2. g1=0, dt_bias=0 -> softplus(0)=ln2.
  const std::vector<float> g1 = {0.0f};
  const std::vector<float> a_log = {static_cast<float>(std::log(2.0))};
  const std::vector<float> dt = {0.0f};
  const std::vector<float> y = KdaDecayGate(g1, a_log, dt, 1, 1, 1);
  CHECK(y[0] == doctest::Approx(-2.0 * std::log(2.0)));  // -1.386294
}

TEST_CASE("kda: decay gate vs independent double reference (rel-L2)") {
  std::mt19937 rng(20260728);
  std::uniform_real_distribution<float> u(-3.0f, 3.0f);
  const int64_t T = 4, H = 3, D = 5, hd = H * D;
  std::vector<float> g1(T * hd), a_log(H), dt(hd);
  for (auto& z : g1) z = u(rng);
  for (auto& z : a_log) z = u(rng);
  for (auto& z : dt) z = u(rng);
  const std::vector<float> got = KdaDecayGate(g1, a_log, dt, T, H, D);

  std::vector<double> ref(static_cast<size_t>(T) * hd);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h) {
      const double ba = -std::exp(static_cast<double>(a_log[h]));
      for (int64_t d = 0; d < D; ++d) {
        const double bg =
            static_cast<double>(g1[t * hd + h * D + d]) + dt[h * D + d];
        // beta=1, threshold=20 -> softplus(x)=log(1+e^x) (all |x|<~6 here)
        const double sp = std::log1p(std::exp(bg));
        ref[t * hd + h * D + d] = ba * sp;
      }
    }
  CHECK(RelL2(got, ref) < 1e-6);
}

TEST_CASE("kda: chunk-cumsum resets at chunk boundary, folds RCP_LN2") {
  // A_log=0 -> b_a=-1; g1 all >20 -> softplus linear -> gate = -g1.
  // T=3, chunk_size=2: gate=[-30,-40,-50]; cumsum (reset at t=2) = [-30,-70,-50].
  const std::vector<float> g1 = {30.0f, 40.0f, 50.0f};
  const std::vector<float> a_log = {0.0f};
  const std::vector<float> raw =
      KdaDecayGateChunkCumsum(g1, a_log, {}, 3, 1, 1, /*chunk*/ 2,
                              /*log2_domain*/ false);
  REQUIRE(raw.size() == 3);
  CHECK(raw[0] == doctest::Approx(-30.0));
  CHECK(raw[1] == doctest::Approx(-70.0));
  CHECK(raw[2] == doctest::Approx(-50.0));  // chunk reset

  const std::vector<float> scaled =
      KdaDecayGateChunkCumsum(g1, a_log, {}, 3, 1, 1, 2, /*log2_domain*/ true);
  CHECK(scaled[0] == doctest::Approx(-30.0 * kRcpLn2));
  CHECK(scaled[1] == doctest::Approx(-70.0 * kRcpLn2));
  CHECK(scaled[2] == doctest::Approx(-50.0 * kRcpLn2));
}

// ── (3) the sigmoid-gated output norm ────────────────────────────────────────

TEST_CASE("kda: FusedRMSNormGated sigmoid = rmsnorm(x)*w*sigmoid(g) — hand case") {
  // x=[1,1], eps=0 -> var=1, normed=[1,1]; weight=[2,3]; g=[0,0]->sigmoid .5.
  const std::vector<float> x = {1, 1};
  const std::vector<float> g = {0, 0};
  const std::vector<float> w = {2, 3};
  const std::vector<float> out = FusedRMSNormGated(
      x, g, w, 1, 1, 2, GatedNormActivation::kSigmoid, /*eps*/ 0.0f);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(1.0));   // 1*2*0.5
  CHECK(out[1] == doctest::Approx(1.5));   // 1*3*0.5
}

TEST_CASE("kda: FusedRMSNormGated swish vs sigmoid branch differ by the g factor") {
  // x=[1,1] eps=0 no-affine -> normed=[1,1]; g=[2,2].
  const std::vector<float> x = {1, 1};
  const std::vector<float> g = {2, 2};
  const std::vector<float> sig = FusedRMSNormGated(
      x, g, {}, 1, 1, 2, GatedNormActivation::kSigmoid, 0.0f);
  const std::vector<float> sw = FusedRMSNormGated(
      x, g, {}, 1, 1, 2, GatedNormActivation::kSwish, 0.0f);
  CHECK(sig[0] == doctest::Approx(Sig(2.0)));           // 1*sigmoid(2)
  CHECK(sw[0] == doctest::Approx(2.0 * Sig(2.0)));      // 1*2*sigmoid(2)
}

TEST_CASE("kda: FusedRMSNormGated normalises over head_dim (not across heads)") {
  // 1 token, 2 heads, D=2. Head0 x=[3,4] (var 12.5), head1 x=[0,0] (var 0).
  // Independent double reference; g=head-wise = [0,0,0,0] -> sigmoid .5.
  const std::vector<float> x = {3, 4, 0, 0};
  const std::vector<float> g = {0, 0, 0, 0};
  const std::vector<float> out = FusedRMSNormGated(
      x, g, {}, 1, 2, 2, GatedNormActivation::kSigmoid, /*eps*/ 1e-5f);
  // head0: rstd=1/sqrt(12.5+1e-5); normed*0.5.
  const double rstd0 = 1.0 / std::sqrt(12.5 + 1e-5);
  CHECK(out[0] == doctest::Approx(3.0 * rstd0 * 0.5));
  CHECK(out[1] == doctest::Approx(4.0 * rstd0 * 0.5));
  CHECK(out[2] == doctest::Approx(0.0));  // head1 all-zero
  CHECK(out[3] == doctest::Approx(0.0));
}

TEST_CASE("kda: FusedRMSNormGated vs independent double reference (rel-L2)") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-2.0f, 2.0f);
  const int64_t T = 3, H = 2, D = 6, hd = H * D;
  const float eps = 1e-5f;
  std::vector<float> x(T * hd), g(T * hd), w(D);
  for (auto& z : x) z = u(rng);
  for (auto& z : g) z = u(rng);
  for (auto& z : w) z = u(rng);
  const std::vector<float> got =
      FusedRMSNormGated(x, g, w, T, H, D, GatedNormActivation::kSigmoid, eps);

  std::vector<double> ref(static_cast<size_t>(T) * hd);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h) {
      double var = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        const double v = x[t * hd + h * D + d];
        var += v * v;
      }
      var /= static_cast<double>(D);
      const double rstd = 1.0 / std::sqrt(var + eps);
      for (int64_t d = 0; d < D; ++d) {
        const double normed =
            static_cast<double>(x[t * hd + h * D + d]) * rstd * w[d];
        ref[t * hd + h * D + d] = normed * Sig(g[t * hd + h * D + d]);
      }
    }
  CHECK(RelL2(got, ref) < 1e-6);
}

// ── (4) short causal conv + q/k L2-norm ──────────────────────────────────────

TEST_CASE("kda: short conv is causal depthwise + silu — hand case") {
  // K=2, 1 channel, x=[1,2,3], w=[1,1]. pre[t]=x[t-1]+x[t] (x[-1]=0):
  //   pre = [1, 3, 5]; y = pre*sigmoid(pre).
  const std::vector<float> x = {1, 2, 3};
  const std::vector<float> w = {1, 1};
  const std::vector<float> y = KdaShortConv(x, w, {}, 3, 1, 2);
  REQUIRE(y.size() == 3);
  CHECK(y[0] == doctest::Approx(1.0 * Sig(1.0)));
  CHECK(y[1] == doctest::Approx(3.0 * Sig(3.0)));
  CHECK(y[2] == doctest::Approx(5.0 * Sig(5.0)));
}

TEST_CASE("kda: short conv respects the zero initial state at t=0") {
  // K=3: at t=0 only tap j=K-1 (pos 0) is in range; taps j=0,1 read 0.
  // x=[5], w=[9,9,2] -> pre = 2*5 = 10; y=10*sigmoid(10).
  const std::vector<float> x = {5};
  const std::vector<float> w = {9, 9, 2};
  const std::vector<float> y = KdaShortConv(x, w, {}, 1, 1, 3);
  CHECK(y[0] == doctest::Approx(10.0 * Sig(10.0)));
}

TEST_CASE("kda: short conv vs independent double reference (rel-L2, +bias)") {
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  const int64_t T = 5, C = 4, K = 4;
  std::vector<float> x(T * C), w(C * K), b(C);
  for (auto& z : x) z = u(rng);
  for (auto& z : w) z = u(rng);
  for (auto& z : b) z = u(rng);
  const std::vector<float> got = KdaShortConv(x, w, b, T, C, K);

  std::vector<double> ref(static_cast<size_t>(T) * C);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t c = 0; c < C; ++c) {
      double acc = b[c];
      for (int64_t j = 0; j < K; ++j) {
        const int64_t pos = t - (K - 1) + j;
        if (pos < 0) continue;
        acc += static_cast<double>(w[c * K + j]) * x[pos * C + c];
      }
      ref[t * C + c] = acc * Sig(acc);
    }
  CHECK(RelL2(got, ref) < 1e-6);
}

TEST_CASE("kda: q/k L2-norm divides by sqrt(sum sq + eps) — hand case") {
  // x=[3,4], eps=0 -> norm=5 -> [0.6, 0.8].
  const std::vector<float> x = {3, 4};
  const std::vector<float> y = L2NormRows(x, 1, 2, /*eps*/ 0.0f);
  REQUIRE(y.size() == 2);
  CHECK(y[0] == doctest::Approx(0.6));
  CHECK(y[1] == doctest::Approx(0.8));
}

TEST_CASE("kda: q/k L2-norm uses SUM of squares, not the mean") {
  // If it used mean (like RMSNorm) x=[1,1] would map to ~[1,1]; L2 maps to
  // [1/sqrt2, 1/sqrt2].
  const std::vector<float> x = {1, 1};
  const std::vector<float> y = L2NormRows(x, 1, 2, 0.0f);
  CHECK(y[0] == doctest::Approx(1.0 / std::sqrt(2.0)));
  CHECK(y[1] == doctest::Approx(1.0 / std::sqrt(2.0)));
}
