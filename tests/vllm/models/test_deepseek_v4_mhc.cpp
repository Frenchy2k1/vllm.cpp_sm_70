// DeepSeek-V4-Flash W5 UNIT GATE — Manifold / Markov Hyper-Connections (MHC):
// the Sinkhorn-normalized hc_mult-stream residual manifold + the mHC pre/post
// mixes + the hc_head collapse, with the folded RMSNorms.
//
// HONEST scope (mirrors W3/W4): the full-model gate is multi-Spark-blocked (the
// V4 checkpoint is 156.7 GiB, does not fit one GB10; the forward also needs the
// sqrtsoftplus/hash MoE (W6) + device assembly (W7)). So W5 gates the MATH per-
// primitive against (a) HAND-DERIVED small cases with literal expected numbers I
// verify by hand from the vLLM eager reference, and (b) from-first-principles
// DOUBLE-PRECISION references on randomized shapes (rel-L2 / independent
// recompute). This is the "derived-eager-reference + hand-case + structural-
// review" bar named in the W5 brief, NOT a dumped-oracle rel-L2 (the fixed-config
// 167B arch cannot be constructed at a tiny shape, AND — the brief's premise —
// upstream ships no golden numerical test for these kernels).
//
// DERIVATION NOTE: the W0 spike asserts MHC has "ZERO eager reference upstream".
// That is CORRECTED here: the pinned vLLM ships an eager PyTorch reference
// (`vllm/model_executor/kernels/mhc/torch.py` mhc_pre_torch/mhc_post_torch,
// `triton.py` hc_head_reduce_triton_kernel), which the TileLang/Triton/CUDA/AITER
// kernels all match, and four upstream impls (torch.py, tilelang_kernels.py
// `_sinkhorn_fwd`, tilelang.py, SGLang mhc.py) agree byte-for-byte on the
// Sinkhorn. We port THAT eager reference AND derive the Sinkhorn independently in
// double precision from its mathematical definition (alternating row/column
// normalization toward a doubly-stochastic matrix); the two agreeing is the gate.
//
// Grounded in: model_executor/kernels/mhc/torch.py:56-106, triton.py:108-140,
// tilelang_kernels.py:126-153, tilelang.py:720-748 (+ mhc_pre_big_fuse_with_norm),
// constants nvidia/model.py:818-821; cross-checked SGLang v0.5.15
// python/sglang/srt/layers/mhc.py:110-126. All @ vLLM pin 555967922.
#include "vllm/model_executor/models/deepseek_v4_mhc.h"

#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

using namespace vllm::deepseek_v4;

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

// ── Independent DOUBLE-PRECISION references (from-first-principles) ───────────
// These re-implement the math in double, NOT by calling the f32 impl, so an
// agreeing rel-L2 proves the port is not a transcription of a bug.

std::vector<double> SinkhornRefD(const std::vector<float>& logits, int64_t hc,
                                 int64_t iters, double eps) {
  std::vector<double> m(static_cast<size_t>(hc * hc));
  for (int64_t j = 0; j < hc; ++j) {  // row softmax + eps
    double rmax = logits[j * hc];
    for (int64_t k = 1; k < hc; ++k) rmax = std::max(rmax, static_cast<double>(logits[j * hc + k]));
    double s = 0.0;
    for (int64_t k = 0; k < hc; ++k) { m[j * hc + k] = std::exp(logits[j * hc + k] - rmax); s += m[j * hc + k]; }
    for (int64_t k = 0; k < hc; ++k) m[j * hc + k] = m[j * hc + k] / s + eps;
  }
  for (int64_t k = 0; k < hc; ++k) {  // col-norm
    double c = 0.0;
    for (int64_t j = 0; j < hc; ++j) c += m[j * hc + k];
    for (int64_t j = 0; j < hc; ++j) m[j * hc + k] /= (c + eps);
  }
  for (int64_t it = 0; it < iters - 1; ++it) {
    for (int64_t j = 0; j < hc; ++j) {  // row-norm
      double r = 0.0;
      for (int64_t k = 0; k < hc; ++k) r += m[j * hc + k];
      for (int64_t k = 0; k < hc; ++k) m[j * hc + k] /= (r + eps);
    }
    for (int64_t k = 0; k < hc; ++k) {  // col-norm
      double c = 0.0;
      for (int64_t j = 0; j < hc; ++j) c += m[j * hc + k];
      for (int64_t j = 0; j < hc; ++j) m[j * hc + k] /= (c + eps);
    }
  }
  return m;
}

double SigmoidD(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// (1) Sinkhorn normalization
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-mhc: Sinkhorn(all-zero logits) is the uniform doubly-stochastic matrix") {
  // hc=4, eps=0: row softmax of all-equal = 1/4 each; every norm is a no-op ⇒
  // M[j,k] = 1/hc exactly, an exact doubly-stochastic fixed point.
  const std::vector<float> logits(16, 0.0f);
  const std::vector<float> m = MhcSinkhorn(logits, 4, 20, 0.0f);
  REQUIRE(m.size() == 16);
  for (float v : m) CHECK(v == doctest::Approx(0.25));
  // Doubly-stochastic: every row sum and col sum == 1.
  for (int j = 0; j < 4; ++j) {
    double rs = 0, cs = 0;
    for (int k = 0; k < 4; ++k) { rs += m[j * 4 + k]; cs += m[k * 4 + j]; }
    CHECK(rs == doctest::Approx(1.0));
    CHECK(cs == doctest::Approx(1.0));
  }
}

TEST_CASE("dsv4-mhc: Sinkhorn symmetric 2x2 fixed point [[.75,.25],[.25,.75]]") {
  // logits [[ln3,0],[0,ln3]], eps=0. Row softmax: [3/4,1/4],[1/4,3/4]; columns
  // already sum to 1, so every subsequent normalization is a no-op ⇒ fixed point.
  const float l3 = std::log(3.0f);
  const std::vector<float> logits = {l3, 0, 0, l3};
  const std::vector<float> m = MhcSinkhorn(logits, 2, 20, 0.0f);
  CHECK(m[0] == doctest::Approx(0.75));
  CHECK(m[1] == doctest::Approx(0.25));
  CHECK(m[2] == doctest::Approx(0.25));
  CHECK(m[3] == doctest::Approx(0.75));
}

TEST_CASE("dsv4-mhc: Sinkhorn iteration count is LOAD-BEARING (col-norm alone != full)") {
  // Asymmetric logits: after the seed+first col-norm (iters=1) columns sum to 1
  // but rows do NOT; the full 20-iter run drives BOTH toward 1 and lands a
  // materially different matrix. Perturbing the iteration count changes the
  // result — the RED-first lever, pinned permanently.
  const float l3 = std::log(3.0f);
  const std::vector<float> logits = {l3, 0, 0, 0};  // 2x2, only [0,0] biased
  const std::vector<float> m1 = MhcSinkhorn(logits, 2, 1, 0.0f);
  const std::vector<float> m20 = MhcSinkhorn(logits, 2, 20, 0.0f);
  // iters=1: columns sum to 1 (last op was col-norm)...
  for (int k = 0; k < 2; ++k) {
    CHECK(m1[0 * 2 + k] + m1[1 * 2 + k] == doctest::Approx(1.0));
  }
  // ...but at least one ROW does not sum to 1 at iters=1.
  const double r0_1 = m1[0] + m1[1];
  CHECK(std::abs(r0_1 - 1.0) > 1e-3);
  // iters=20: both rows and columns sum to 1 (converged doubly-stochastic).
  for (int j = 0; j < 2; ++j) {
    double rs = 0, cs = 0;
    for (int k = 0; k < 2; ++k) { rs += m20[j * 2 + k]; cs += m20[k * 2 + j]; }
    CHECK(rs == doctest::Approx(1.0).epsilon(1e-4));
    CHECK(cs == doctest::Approx(1.0).epsilon(1e-4));
  }
  // The two matrices differ (iteration count matters).
  double diff = 0;
  for (size_t i = 0; i < m1.size(); ++i) diff += std::abs(m1[i] - m20[i]);
  CHECK(diff > 1e-2);
}

TEST_CASE("dsv4-mhc: Sinkhorn converges to doubly-stochastic (randomized, eps=0)") {
  std::mt19937 rng(0xD5F4);
  std::uniform_real_distribution<float> U(-3.0f, 3.0f);
  for (int64_t hc : {2, 3, 4, 6}) {
    std::vector<float> logits(static_cast<size_t>(hc * hc));
    for (auto& v : logits) v = U(rng);
    const std::vector<float> m = MhcSinkhorn(logits, hc, 60, 0.0f);
    for (int64_t j = 0; j < hc; ++j) {
      double rs = 0, cs = 0;
      for (int64_t k = 0; k < hc; ++k) { rs += m[j * hc + k]; cs += m[k * hc + j]; }
      CHECK(rs == doctest::Approx(1.0).epsilon(1e-4));
      CHECK(cs == doctest::Approx(1.0).epsilon(1e-4));
    }
  }
}

TEST_CASE("dsv4-mhc: Sinkhorn f32 == independent f64 reference (randomized)") {
  std::mt19937 rng(0x5117);
  std::uniform_real_distribution<float> U(-4.0f, 4.0f);
  const double eps = 1e-6;
  for (int64_t hc : {2, 3, 4, 8}) {
    std::vector<float> logits(static_cast<size_t>(hc * hc));
    for (auto& v : logits) v = U(rng);
    const std::vector<float> m = MhcSinkhorn(logits, hc, 20, static_cast<float>(eps));
    const std::vector<double> ref = SinkhornRefD(logits, hc, 20, eps);
    CHECK(RelL2(m, ref) < 1e-5);
  }
}

TEST_CASE("dsv4-mhc: Sinkhorn f32 == f64 ref at SMALL iteration counts (RED-first lever)") {
  // At 20 iters the Sinkhorn has CONVERGED, so ±1 iter is within any tolerance —
  // a gate at iters=20 alone canNOT catch an iteration-count bug. This case pins
  // the count at SMALL, NON-converged values (2,3,5) where each iteration MATTERS:
  // impl(n) vs the independent f64 ref(n) at the SAME small n must agree to float
  // epsilon, so perturbing the loop bound (n-1↔n-2) or swapping a normalization
  // axis diverges and FAILS here. Uses an asymmetric, wide-spread logit matrix
  // that converges slowly so the intermediate iterates are well-separated.
  std::mt19937 rng(0x51A7);
  std::uniform_real_distribution<float> U(-6.0f, 6.0f);
  const double eps = 1e-6;
  for (int64_t iters : {2, 3, 5}) {
    for (int64_t hc : {2, 3, 4}) {
      std::vector<float> logits(static_cast<size_t>(hc * hc));
      for (auto& v : logits) v = U(rng);
      const std::vector<float> m = MhcSinkhorn(logits, hc, iters, static_cast<float>(eps));
      const std::vector<double> ref = SinkhornRefD(logits, hc, iters, eps);
      CHECK(RelL2(m, ref) < 1e-5);
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// (2) mHC "pre" block
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-mhc: MhcPre hand case (fn=0 ⇒ mixes=0 ⇒ gate midpoints)") {
  // hc=2, hidden=2. fn=0 ⇒ mixes=0 regardless of residual/rms_eps.
  //   pre  = sigmoid(0)+0        = 0.5
  //   post = sigmoid(0)*2        = 1.0
  //   comb = Sinkhorn(all-zero)  = 0.5 everywhere
  //   layer_input[h] = Σ_j 0.5·residual[j,h]
  const int64_t hc = 2, hidden = 2, hc3 = (2 + hc) * hc;  // 8
  const std::vector<float> residual = {2, 0, 0, 6};  // stream0=[2,0], stream1=[0,6]
  const std::vector<float> fn(static_cast<size_t>(hc3 * hc * hidden), 0.0f);
  const std::vector<float> scale = {1, 1, 1};
  const std::vector<float> base(static_cast<size_t>(hc3), 0.0f);
  const MhcPreResult r = MhcPre(residual, fn, scale, base, hc, hidden, 1e-6f,
                                /*hc_pre_eps=*/0.0f, /*hc_sinkhorn_eps=*/0.0f,
                                /*hc_post_mult=*/2.0f, /*iters=*/20, {}, 0.0f);
  REQUIRE(r.pre_mix.size() == 2);
  CHECK(r.pre_mix[0] == doctest::Approx(0.5));
  CHECK(r.pre_mix[1] == doctest::Approx(0.5));
  CHECK(r.post_mix[0] == doctest::Approx(1.0));
  CHECK(r.post_mix[1] == doctest::Approx(1.0));
  for (float v : r.comb_mix) CHECK(v == doctest::Approx(0.5));
  CHECK(r.layer_input[0] == doctest::Approx(1.0));  // .5*2 + .5*0
  CHECK(r.layer_input[1] == doctest::Approx(3.0));  // .5*0 + .5*6
}

TEST_CASE("dsv4-mhc: MhcPre folds the attn/ffn RMSNorm when norm_weight given") {
  // Same fn=0 setup ⇒ layer_input pre-fold = [1,3]. Fold with weight=1, eps=0:
  //   rms = rsqrt((1+9)/2) = rsqrt(5) ⇒ out = [1/sqrt5, 3/sqrt5].
  const int64_t hc = 2, hidden = 2, hc3 = (2 + hc) * hc;
  const std::vector<float> residual = {2, 0, 0, 6};
  const std::vector<float> fn(static_cast<size_t>(hc3 * hc * hidden), 0.0f);
  const std::vector<float> scale = {1, 1, 1};
  const std::vector<float> base(static_cast<size_t>(hc3), 0.0f);
  const std::vector<float> nw = {1, 1};
  const MhcPreResult r = MhcPre(residual, fn, scale, base, hc, hidden, 1e-6f, 0.0f,
                                0.0f, 2.0f, 20, nw, /*norm_eps=*/0.0f);
  const double s5 = std::sqrt(5.0);
  CHECK(r.layer_input[0] == doctest::Approx(1.0 / s5));
  CHECK(r.layer_input[1] == doctest::Approx(3.0 / s5));
}

TEST_CASE("dsv4-mhc: MhcPre f32 == independent f64 reference (randomized, folded)") {
  std::mt19937 rng(0xABCD);
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  const float rms_eps = 1e-5f, norm_eps = 1e-6f, hc_pre_eps = 1e-6f, hc_sink_eps = 1e-6f;
  for (int64_t hc : {2, 4}) {
    for (int64_t hidden : {4, 8}) {
      const int64_t hc3 = (2 + hc) * hc, flat = hc * hidden;
      std::vector<float> residual(static_cast<size_t>(flat)), fn(static_cast<size_t>(hc3 * flat));
      std::vector<float> scale(3), base(static_cast<size_t>(hc3)), nw(static_cast<size_t>(hidden));
      for (auto& v : residual) v = U(rng);
      for (auto& v : fn) v = 0.1f * U(rng);
      for (auto& v : scale) v = U(rng);
      for (auto& v : base) v = U(rng);
      for (auto& v : nw) v = 0.5f + 0.5f * U(rng);
      const MhcPreResult r = MhcPre(residual, fn, scale, base, hc, hidden, rms_eps,
                                    hc_pre_eps, hc_sink_eps, 2.0f, 20, nw, norm_eps);
      // Independent f64 recompute of layer_input (the composed output).
      std::vector<double> mixes(static_cast<size_t>(hc3), 0.0), pre(static_cast<size_t>(hc));
      double sq = 0.0;
      for (int64_t p = 0; p < flat; ++p) sq += static_cast<double>(residual[p]) * residual[p];
      for (int64_t o = 0; o < hc3; ++o) {
        double a = 0.0;
        for (int64_t p = 0; p < flat; ++p) a += static_cast<double>(residual[p]) * fn[o * flat + p];
        mixes[o] = a;
      }
      const double rms = 1.0 / std::sqrt(sq / static_cast<double>(flat) + rms_eps);
      for (int64_t o = 0; o < hc3; ++o) mixes[o] *= rms;
      for (int64_t j = 0; j < hc; ++j)
        pre[j] = SigmoidD(mixes[j] * scale[0] + base[j]) + hc_pre_eps;
      std::vector<double> li(static_cast<size_t>(hidden), 0.0);
      for (int64_t j = 0; j < hc; ++j)
        for (int64_t h = 0; h < hidden; ++h) li[h] += pre[j] * residual[j * hidden + h];
      double ss = 0.0;
      for (int64_t h = 0; h < hidden; ++h) ss += li[h] * li[h];
      const double nr = 1.0 / std::sqrt(ss / static_cast<double>(hidden) + norm_eps);
      for (int64_t h = 0; h < hidden; ++h) li[h] = li[h] * nr * nw[h];
      CHECK(RelL2(r.layer_input, li) < 1e-4);
      // comb is a valid Sinkhorn matrix (near doubly-stochastic).
      for (int64_t j = 0; j < hc; ++j) {
        double cs = 0;
        for (int64_t k = 0; k < hc; ++k) cs += r.comb_mix[k * hc + j];
        CHECK(cs == doctest::Approx(1.0).epsilon(1e-3));
      }
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// (3) mHC "post" block
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-mhc: MhcPost hand case — identity comb + post add") {
  // hc=2, hidden=1. comb=identity[[1,0],[0,1]] (comb[i,j] at i*hc+j).
  //   new[0] = comb[0,0]*5 + comb[1,0]*7 + post[0]*10 = 5 + 20 = 25
  //   new[1] = comb[0,1]*5 + comb[1,1]*7 + post[1]*10 = 7 + 30 = 37
  const std::vector<float> x = {10};
  const std::vector<float> residual = {5, 7};       // stream0=[5], stream1=[7]
  const std::vector<float> post = {2, 3};
  const std::vector<float> comb = {1, 0, 0, 1};      // identity
  const std::vector<float> out = MhcPost(x, residual, post, comb, 2, 1);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(25.0));
  CHECK(out[1] == doctest::Approx(37.0));
}

TEST_CASE("dsv4-mhc: MhcPost mix SUMS over the first comb index i") {
  // Uniform comb 0.5, post=0, x=0 ⇒ both output streams = mean of residual.
  const std::vector<float> x = {0};
  const std::vector<float> residual = {5, 7};
  const std::vector<float> post = {0, 0};
  const std::vector<float> comb = {0.5f, 0.5f, 0.5f, 0.5f};
  const std::vector<float> out = MhcPost(x, residual, post, comb, 2, 1);
  CHECK(out[0] == doctest::Approx(6.0));  // .5*5 + .5*7
  CHECK(out[1] == doctest::Approx(6.0));
}

TEST_CASE("dsv4-mhc: MhcPost f32 == independent f64 reference (randomized)") {
  std::mt19937 rng(0x9001);
  std::uniform_real_distribution<float> U(-2.0f, 2.0f);
  for (int64_t hc : {2, 4}) {
    for (int64_t hidden : {3, 8}) {
      std::vector<float> x(static_cast<size_t>(hidden)), residual(static_cast<size_t>(hc * hidden));
      std::vector<float> post(static_cast<size_t>(hc)), comb(static_cast<size_t>(hc * hc));
      for (auto& v : x) v = U(rng);
      for (auto& v : residual) v = U(rng);
      for (auto& v : post) v = U(rng);
      for (auto& v : comb) v = U(rng);
      const std::vector<float> out = MhcPost(x, residual, post, comb, hc, hidden);
      std::vector<double> ref(static_cast<size_t>(hc * hidden));
      for (int64_t j = 0; j < hc; ++j)
        for (int64_t h = 0; h < hidden; ++h) {
          double mixed = 0.0;
          for (int64_t i = 0; i < hc; ++i)
            mixed += static_cast<double>(comb[i * hc + j]) * residual[i * hidden + h];
          ref[j * hidden + h] = mixed + static_cast<double>(post[j]) * x[h];
        }
      CHECK(RelL2(out, ref) < 1e-6);
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// (4) hc_head collapse
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("dsv4-mhc: HcHeadCollapse hand case (fn=0 ⇒ pre=0.5 ⇒ mean of streams)") {
  // hc=2, hidden=2, x=[[2,0],[0,0]], fn=0, base=0, scale=0, rms_eps=0, hc_eps=0.
  //   x_flat=[2,0,0,0], ss=4, r=rsqrt(4/4)=1. mixes=0 ⇒ pre=sigmoid(0)=0.5.
  //   out[0]=.5*2+.5*0=1 ; out[1]=.5*0+.5*0=0.
  const int64_t hc = 2, hidden = 2, flat = hc * hidden;
  const std::vector<float> x = {2, 0, 0, 0};
  const std::vector<float> fn(static_cast<size_t>(hc * flat), 0.0f);
  const std::vector<float> base(static_cast<size_t>(hc), 0.0f);
  const std::vector<float> out = HcHeadCollapse(x, fn, /*scale=*/0.0f, base, hc, hidden,
                                                /*rms_eps=*/0.0f, /*hc_eps=*/0.0f);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(1.0));
  CHECK(out[1] == doctest::Approx(0.0));
}

TEST_CASE("dsv4-mhc: HcHeadCollapse f32 == independent f64 reference (randomized)") {
  std::mt19937 rng(0x4EAD);
  std::uniform_real_distribution<float> U(-1.5f, 1.5f);
  const float rms_eps = 1e-5f, hc_eps = 1e-6f;
  for (int64_t hc : {2, 4}) {
    for (int64_t hidden : {4, 8}) {
      const int64_t flat = hc * hidden;
      std::vector<float> x(static_cast<size_t>(flat)), fn(static_cast<size_t>(hc * flat));
      std::vector<float> base(static_cast<size_t>(hc));
      const float scale = U(rng);
      for (auto& v : x) v = U(rng);
      for (auto& v : fn) v = 0.2f * U(rng);
      for (auto& v : base) v = U(rng);
      const std::vector<float> out = HcHeadCollapse(x, fn, scale, base, hc, hidden, rms_eps, hc_eps);
      double ss = 0.0;
      for (int64_t p = 0; p < flat; ++p) ss += static_cast<double>(x[p]) * x[p];
      const double r = 1.0 / std::sqrt(ss / static_cast<double>(flat) + rms_eps);
      std::vector<double> pre(static_cast<size_t>(hc));
      for (int64_t m = 0; m < hc; ++m) {
        double a = 0.0;
        for (int64_t p = 0; p < flat; ++p) a += (static_cast<double>(x[p]) * r) * fn[m * flat + p];
        pre[m] = SigmoidD(a * scale + base[m]) + hc_eps;
      }
      std::vector<double> ref(static_cast<size_t>(hidden), 0.0);
      for (int64_t m = 0; m < hc; ++m)
        for (int64_t h = 0; h < hidden; ++h) ref[h] += pre[m] * x[m * hidden + h];
      CHECK(RelL2(out, ref) < 1e-4);
    }
  }
}
