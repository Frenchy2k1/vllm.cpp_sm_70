// DeepSeek-V4-Flash W4 UNIT GATE — the DSA COMPRESSOR forward + the fp8_ds_mla
// KV-cache state read/write layout.
// HONEST scope: the full-model gate is multi-Spark-blocked (156.7 GiB, does not
// fit one GB10; forward also needs MHC/MoE, not ported). So W4 gates the MATH
// per-primitive against HAND-DERIVED small cases (literal expected numbers I can
// verify by hand from the vLLM source) PLUS from-first-principles double-
// precision references on randomized shapes (rel-L2 / independent recompute).
// This is the "hand-case + structural review" bar named in the W4 brief, NOT a
// dumped-oracle rel-L2 (the arch cannot be constructed at a tiny shape — it is a
// fixed-config 167B). The eventual GPU forward (W7) ports the same math into a
// CUDA kernel; these tests are its portable oracle.
//
// Grounded in: common/ops/save_partial_states.py:92-101 (APE add),
// common/ops/fused_compress_quant_cache.py:198-297 (pool+RMSNorm + fp8_ds_mla
// store), compressor.py:307-309 (layout), cross-checked against SGLang v0.5.15
// dsv4/fused_compress_triton.py + dsv4/quant_k_cache.py + dsv4/dequant_k_cache.py.
#include "vllm/model_executor/models/deepseek_v4_compressor.h"

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
}  // namespace

// ── (A1) Save-time APE fusion ────────────────────────────────────────────────

TEST_CASE("dsv4-compressor: save-time score += ape[position % compress_ratio]") {
  // T=2, width=2, compress_ratio=2. positions [0,1] -> ape rows [0,1].
  const std::vector<float> score = {1, 2, 3, 4};
  const std::vector<float> ape = {10, 20, 30, 40};  // [2 rows, 2 wide]
  const std::vector<int64_t> pos = {0, 1};
  const std::vector<float> out = CompressorSaveScoreApe(score, ape, pos, 2, 2, 2);
  REQUIRE(out.size() == 4);
  CHECK(out[0] == doctest::Approx(11));  // 1 + ape[0][0]=10
  CHECK(out[1] == doctest::Approx(22));  // 2 + ape[0][1]=20
  CHECK(out[2] == doctest::Approx(33));  // 3 + ape[1][0]=30
  CHECK(out[3] == doctest::Approx(44));  // 4 + ape[1][1]=40
}

TEST_CASE("dsv4-compressor: APE row wraps with position modulo compress_ratio") {
  // position 3, compress_ratio 2 -> ape row 1 (not 3).
  const std::vector<float> score = {0, 0};
  const std::vector<float> ape = {10, 20, 30, 40};
  const std::vector<int64_t> pos = {3};
  const std::vector<float> out = CompressorSaveScoreApe(score, ape, pos, 1, 2, 2);
  CHECK(out[0] == doctest::Approx(30));  // ape[3 % 2 = 1][0]
  CHECK(out[1] == doctest::Approx(40));  // ape[1][1]
}

// ── (A2) Compressor POOL + RMSNorm ───────────────────────────────────────────

TEST_CASE("dsv4-compressor: pool = softmax(score,dim=0) . kv, then RMSNorm — hand case") {
  // window=2, head_dim=2. score all 0 -> softmax = [0.5,0.5] per column.
  //   compressed[0] = 1*.5 + 3*.5 = 2 ; compressed[1] = 2*.5 + 4*.5 = 3
  //   var = (4+9)/2 = 6.5 ; rrms = 1/sqrt(6.5)
  const std::vector<float> kv = {1, 2, 3, 4};
  const std::vector<float> score = {0, 0, 0, 0};
  const std::vector<uint8_t> valid = {1, 1};
  const std::vector<float> w = {1, 1};
  const std::vector<float> out = CompressorPoolNorm(kv, score, valid, w, 0.0f, 2, 2);
  const double rrms = 1.0 / std::sqrt(6.5);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(2.0 * rrms));
  CHECK(out[1] == doctest::Approx(3.0 * rrms));
}

TEST_CASE("dsv4-compressor: masked window rows are excluded from the pool") {
  // valid=[1,0] -> only row0 pools; softmax over one row = 1.0.
  //   compressed = kv[0] = [1,2] ; var = (1+4)/2 = 2.5
  const std::vector<float> kv = {1, 2, 999, 999};  // row1 must NOT leak
  const std::vector<float> score = {0, 0, 0, 0};
  const std::vector<uint8_t> valid = {1, 0};
  const std::vector<float> w = {1, 1};
  const std::vector<float> out = CompressorPoolNorm(kv, score, valid, w, 0.0f, 2, 2);
  const double rrms = 1.0 / std::sqrt(2.5);
  CHECK(out[0] == doctest::Approx(1.0 * rrms));
  CHECK(out[1] == doctest::Approx(2.0 * rrms));
}

TEST_CASE("dsv4-compressor: softmax is PER-COLUMN (dim=0) — the load-bearing nuance") {
  // If pooling used one shared weight per row, both columns would pool
  // identically. Here column 1's score strongly favors row 0 while column 0 is
  // uniform, so the compressed columns differ in a way only per-column softmax
  // produces. kv=[[10,10],[0,0]], score=[[0,100],[0,0]].
  //   col0: softmax([0,0])=[.5,.5] -> 5 ; col1: softmax([100,0])~[1,0] -> 10
  // RMSNorm scales both columns by the same rrms, so the RATIO 10/5 = 2 survives.
  const std::vector<float> kv = {10, 10, 0, 0};
  const std::vector<float> score = {0, 100, 0, 0};
  const std::vector<uint8_t> valid = {1, 1};
  const std::vector<float> w = {1, 1};
  const std::vector<float> out = CompressorPoolNorm(kv, score, valid, w, 0.0f, 2, 2);
  CHECK(out[1] == doctest::Approx(2.0 * out[0]));
  CHECK(out[0] > 0.0f);
}

// ── (B1) fp8_ds_mla layout geometry ──────────────────────────────────────────

TEST_CASE("dsv4-fp8_ds_mla: V4 layout is 448 fp8 + 64 bf16, 576B stride, 7+1 scales") {
  const Fp8DsMlaLayout L = MakeFp8DsMlaLayout(448, 64, 64);
  CHECK(L.nope_head_dim == 448);
  CHECK(L.rope_head_dim == 64);
  CHECK(L.quant_block == 64);
  CHECK(L.n_nope_blocks == 7);         // 448 / 64
  CHECK(L.token_stride_bytes == 576);  // 448*1 + 64*2
  CHECK(L.scale_dim == 8);             // 7 real + 1 pad
}

// ── (B2) fp8_ds_mla UE8M0 scale-byte encoding (hand-derived) ─────────────────

TEST_CASE("dsv4-fp8_ds_mla: all-ones block -> UE8M0 scale byte 119, exact round-trip") {
  // absmax=1, raw=1/448 -> exponent=ceil(log2(1/448))=-8 -> byte=-8+127=119.
  // decode: fp8(256) * 2^(119-127)=2^-8 -> 1.0 (256 is e4m3-exact).
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(64, 0, 64);  // one nope block, no rope
  std::vector<float> head(64, 1.0f);
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  REQUIRE(t.scale_ue8m0.size() == 1);
  CHECK(static_cast<int>(t.scale_ue8m0[0]) == 119);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);
  REQUIRE(dec.size() == 64);
  for (float v : dec) CHECK(v == doctest::Approx(1.0));
}

TEST_CASE("dsv4-fp8_ds_mla: value 3.0 block -> UE8M0 scale byte 120, exact round-trip") {
  // absmax=3, raw=3/448 -> exponent=ceil(log2(3/448))=-7 -> byte=120.
  // decode: fp8(384=1.5*2^8) * 2^-7 -> 3.0.
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(64, 0, 64);
  std::vector<float> head(64, 3.0f);
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  CHECK(static_cast<int>(t.scale_ue8m0[0]) == 120);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);
  for (float v : dec) CHECK(v == doctest::Approx(3.0));
}

TEST_CASE("dsv4-fp8_ds_mla: RoPE part is stored/read bf16 verbatim") {
  // 448 nope (zeros) + rope [1.5, 0.0, -2.0, 0.25] — all bf16-exact.
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(448, 4, 64);
  std::vector<float> head(452, 0.0f);
  head[448] = 1.5f;
  head[449] = 0.0f;
  head[450] = -2.0f;
  head[451] = 0.25f;
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  REQUIRE(t.rope_bf16.size() == 4);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);
  CHECK(dec[448] == doctest::Approx(1.5));
  CHECK(dec[449] == doctest::Approx(0.0));
  CHECK(dec[450] == doctest::Approx(-2.0));
  CHECK(dec[451] == doctest::Approx(0.25));
}

// ── from-first-principles double-precision references (randomized) ────────────

TEST_CASE("dsv4-compressor: pool+norm vs independent double reference (rel-L2)") {
  std::mt19937 rng(20260729);
  std::uniform_real_distribution<float> u(-1.5f, 1.5f);
  const int64_t W = 5, D = 12;
  const float eps = 1e-6f;
  std::vector<float> kv(W * D), score(W * D), rms(D);
  std::vector<uint8_t> valid(W, 1);
  for (auto& x : kv) x = u(rng);
  for (auto& x : score) x = u(rng);
  for (auto& x : rms) x = u(rng);
  valid[1] = 0;  // exercise the mask
  valid[4] = 0;

  const std::vector<float> got = CompressorPoolNorm(kv, score, valid, rms, eps, W, D);

  // Independent double reference (per-column softmax pool, then RMSNorm).
  std::vector<double> comp(static_cast<size_t>(D), 0.0);
  for (int64_t d = 0; d < D; ++d) {
    double m = -1e300;
    for (int64_t i = 0; i < W; ++i)
      if (valid[i]) m = std::max(m, static_cast<double>(score[i * D + d]));
    double denom = 0.0, acc = 0.0;
    for (int64_t i = 0; i < W; ++i) {
      if (!valid[i]) continue;
      const double e = std::exp(static_cast<double>(score[i * D + d]) - m);
      denom += e;
      acc += static_cast<double>(kv[i * D + d]) * e;
    }
    comp[d] = acc / denom;
  }
  double var = 0.0;
  for (int64_t d = 0; d < D; ++d) var += comp[d] * comp[d];
  var /= D;
  const double rrms = 1.0 / std::sqrt(var + eps);
  std::vector<double> ref(static_cast<size_t>(D));
  for (int64_t d = 0; d < D; ++d) ref[d] = comp[d] * rrms * rms[d];

  CHECK(RelL2(got, ref) < 1e-6);
}

TEST_CASE("dsv4-fp8_ds_mla: UE8M0 scale bytes match an independent recompute") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-6.0f, 6.0f);
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(256, 0, 64);  // 4 nope blocks
  std::vector<float> head(256);
  for (auto& x : head) x = u(rng);
  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);

  // Independent recompute of each block's UE8M0 exponent byte.
  REQUIRE(t.scale_ue8m0.size() == 4);
  for (int64_t b = 0; b < 4; ++b) {
    double absmax = 0.0;
    for (int64_t j = 0; j < 64; ++j)
      absmax = std::max(absmax, std::fabs(static_cast<double>(head[b * 64 + j])));
    absmax = std::max(absmax, 1e-4);
    const double exponent = std::ceil(std::log2(absmax / 448.0));
    int expected = static_cast<int>(exponent) + 127;
    expected = std::max(0, std::min(255, expected));
    CHECK(static_cast<int>(t.scale_ue8m0[static_cast<size_t>(b)]) == expected);
  }
}

TEST_CASE("dsv4-fp8_ds_mla: decode(encode(x)) round-trips within fp8 granularity") {
  std::mt19937 rng(101);
  std::uniform_real_distribution<float> u(-4.0f, 4.0f);
  Fp8DsMlaLayout L = MakeFp8DsMlaLayout(448, 64, 64);
  std::vector<float> head(512);
  for (auto& x : head) x = u(rng);

  const Fp8DsMlaToken t = Fp8DsMlaEncodeToken(head, L);
  const std::vector<float> dec = Fp8DsMlaDecodeToken(t, L);

  // e4m3 carries 3 mantissa bits -> worst-case per-element relative step ~2^-4;
  // with per-64-block power-of-two scaling the magnitude-weighted rel-L2 sits
  // well under that. bf16 rope is near-exact. Honest granularity bound, not 1e-6.
  std::vector<double> ref(head.begin(), head.end());
  CHECK(RelL2(dec, ref) < 0.05);
}
