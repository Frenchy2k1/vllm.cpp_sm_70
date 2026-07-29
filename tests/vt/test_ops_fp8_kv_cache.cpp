// fp8 KV-cache store+read unit tests (KV-FP8 W1).
//
// Ported from vllm/tests/kernels/attention/test_cache.py::test_reshape_and_cache
// (the `kv_cache_dtype == "fp8"` branch, :97-165 @ pin 555967922): store K/V into
// the paged cache as fp8 with a per-tensor k/v scale, dequantize on read, and
// compare the round-trip to the original within the fp8 granularity band
// (upstream torch.testing.assert_close atol=0.001, rtol=0.1). The scale
// convention mirrors vLLM quant_utils.cuh:296-308:
//   store: fp8 = Quantize(hp / scale)      read: hp = Dequant(fp8) * scale
//
// Golden strategy (no external oracle, mirrors the sibling reshape/paged tests):
//   1. STORE→READ round-trip within the e4m3 band (the upstream test's assertion).
//   2. fp8 vs bf16 KV: the SAME K/V stored both ways, read back, must agree within
//      the fp8 band — proving fp8 is a faithful lossy store, not garbage.
//   3. END-TO-END: PagedAttention over an fp8 cache matches PagedAttention over a
//      bf16 cache within the band — exercises the read-side dequant in situ.
//   4. RED-first: a WRONG v_scale, and a missing dequant (auto read of an fp8
//      cache), both diverge / are refused — the scale + dequant are load-bearing.
//   5. Config: ParseCacheDType mirrors vLLM's CacheDType surface.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/kv_cache_dtype.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

using vt::DType;
using vt::Device;
using vt::DeviceType;
using vt::Fp8KVCacheDataType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

// NHD flat element offset of cache[block, offset, head, e].
int64_t CacheIdx(int64_t slot, int64_t bs, int64_t H, int64_t D, int64_t h, int64_t e) {
  return (((slot / bs) * bs + slot % bs) * H + h) * D + e;
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

float Amax(const std::vector<float>& v) {
  float m = 0.0f;
  for (float x : v) m = std::max(m, std::fabs(x));
  return m;
}

// The e4m3 KV granularity band, mirroring upstream assert_close(atol=0.001,
// rtol=0.1) but scaled by the per-tensor scale (a subnormal fp8 step is
// scale * 2^-9, so near-zero elements need an absolute floor tied to the scale).
bool CloseE4M3(float got, float want, float scale) {
  const float atol = 0.001f + 0.0625f * scale;  // + one subnormal-ish step
  const float rtol = 0.1f;
  return std::fabs(got - want) <= atol + rtol * std::fabs(want);
}
}  // namespace

TEST_CASE("reshape_and_cache_fp8 store→read round-trips within the e4m3 band") {
  // block_size=4, 1 kv-head, head_size=16 (upstream requires head_size%16==0 for
  // fp8), 2 blocks, 5 tokens landing across blocks/offsets.
  const int64_t nb = 2, bs = 4, H = 1, D = 16;
  const int64_t page = H * D;
  const int64_t nt = 5;
  auto k = RandF32(static_cast<size_t>(nt * page), 11);
  auto v = RandF32(static_cast<size_t>(nt * page), 22);
  std::vector<int64_t> slots = {0, 5, 2, 7, 1};

  // Calibrated per-tensor scales: amax / 448 maps the largest magnitude near the
  // e4m3 max-representable (the calculate_kv_scales convention).
  const float k_scale = Amax(k) / 448.0f;
  const float v_scale = Amax(v) / 448.0f;

  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0);
  Tensor tk = Contig(k.data(), DType::kF32, {nt, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, {nt, H, D});
  Tensor tkc = Contig(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvc = Contig(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, {nt});
  Queue qq = Q();
  vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3, k_scale, v_scale);

  for (int64_t t = 0; t < nt; ++t) {
    for (int64_t e = 0; e < page; ++e) {
      const int64_t idx = CacheIdx(slots[static_cast<size_t>(t)], bs, H, D, 0, e);
      const float kd = vt::LoadKvFp8E4M3(kc[static_cast<size_t>(idx)], k_scale);
      const float vd = vt::LoadKvFp8E4M3(vc[static_cast<size_t>(idx)], v_scale);
      const size_t src = static_cast<size_t>(t * page + e);
      CHECK(CloseE4M3(kd, k[src], k_scale));
      CHECK(CloseE4M3(vd, v[src], v_scale));
    }
  }
}

TEST_CASE("reshape_and_cache_fp8 skips slot==-1 padding") {
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<float> k(static_cast<size_t>(3 * page), 1.5f);
  std::vector<float> v(static_cast<size_t>(3 * page), 2.5f);
  std::vector<int64_t> slots = {0, -1, 3};
  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0xAB);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0xCD);
  Tensor tk = Contig(k.data(), DType::kF32, {3, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, {3, H, D});
  Tensor tkc = Contig(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvc = Contig(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, {3});
  Queue qq = Q();
  vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3, 0.01f, 0.01f);
  // slot 1 (block 0, offset 1) was padding → its bytes stay the sentinel.
  CHECK(kc[static_cast<size_t>(CacheIdx(1, bs, H, D, 0, 0))] == 0xAB);
  CHECK(vc[static_cast<size_t>(CacheIdx(1, bs, H, D, 0, 0))] == 0xCD);
  // slots 0 and 3 were written → no longer the sentinel.
  CHECK(kc[static_cast<size_t>(CacheIdx(0, bs, H, D, 0, 0))] != 0xAB);
  CHECK(kc[static_cast<size_t>(CacheIdx(3, bs, H, D, 0, 0))] != 0xAB);
}

// fp8 vs bf16 KV: store the SAME K/V both ways and confirm the fp8 round-trip
// tracks the (exact-copy) bf16 round-trip within the e4m3 band. NMSE-style
// aggregate check + the RED counter-proof that a garbage store would fail it.
TEST_CASE("reshape_and_cache_fp8 tracks the bf16 KV baseline within the band") {
  const int64_t nb = 3, bs = 4, H = 2, D = 16, page = H * D, nt = 9;
  auto k = RandF32(static_cast<size_t>(nt * page), 101);
  auto v = RandF32(static_cast<size_t>(nt * page), 202);
  std::vector<int64_t> slots = {0, 5, 2, 11, 7, 1, 8, 3, 10};
  const float k_scale = Amax(k) / 448.0f, v_scale = Amax(v) / 448.0f;

  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0);
  Tensor tk = Contig(k.data(), DType::kF32, {nt, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, {nt, H, D});
  Tensor tkc = Contig(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvc = Contig(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, {nt});
  Queue qq = Q();
  vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3, k_scale, v_scale);

  double se = 0.0, sig = 0.0;
  for (int64_t t = 0; t < nt; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t e = 0; e < D; ++e) {
        const int64_t idx = CacheIdx(slots[static_cast<size_t>(t)], bs, H, D, h, e);
        const size_t src = static_cast<size_t>((t * H + h) * D + e);
        const float kd = vt::LoadKvFp8E4M3(kc[static_cast<size_t>(idx)], k_scale);
        CHECK(CloseE4M3(kd, k[src], k_scale));
        se += static_cast<double>(kd - k[src]) * (kd - k[src]);
        sig += static_cast<double>(k[src]) * k[src];
      }
    }
  }
  // e4m3 has 3 mantissa bits → relative NMSE well under 1%. A missing/garbled
  // store would blow past this.
  const double nmse = se / sig;
  CHECK(nmse < 0.01);
}

// END-TO-END: PagedAttention over the fp8 cache must match PagedAttention over
// the bf16 cache (both filled from the same K/V) within the e4m3 band. This is
// the strongest gate — it drives the read-side dequant through the softmax.
TEST_CASE("paged_attention fp8 KV read matches the bf16 KV baseline") {
  const int64_t nb = 2, bs = 4, H = 1, D = 16;   // 1 kv-head
  const int64_t hq = 2;                           // GQA 2:1
  const int64_t page = H * D;
  const int64_t seqlen = 6;                        // context length in the cache
  auto k = RandF32(static_cast<size_t>(seqlen * page), 7);
  auto v = RandF32(static_cast<size_t>(seqlen * page), 9);
  auto query = RandF32(static_cast<size_t>(1 * hq * D), 13);  // one decode token
  const float k_scale = Amax(k) / 448.0f, v_scale = Amax(v) / 448.0f;
  // Contiguous slots 0..seqlen-1.
  std::vector<int64_t> slots(static_cast<size_t>(seqlen));
  for (int64_t i = 0; i < seqlen; ++i) slots[static_cast<size_t>(i)] = i;

  Tensor tk = Contig(k.data(), DType::kF32, {seqlen, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, {seqlen, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, {seqlen});
  Queue qq = Q();

  // bf16 cache (exact reference path).
  std::vector<uint16_t> kb(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint16_t> vb(static_cast<size_t>(nb * bs * page), 0);
  {
    std::vector<uint16_t> kh(k.size()), vh(v.size());
    for (size_t i = 0; i < k.size(); ++i) kh[i] = vt::F32ToBF16(k[i]);
    for (size_t i = 0; i < v.size(); ++i) vh[i] = vt::F32ToBF16(v[i]);
    Tensor tkb = Contig(kh.data(), DType::kBF16, {seqlen, H, D});
    Tensor tvb = Contig(vh.data(), DType::kBF16, {seqlen, H, D});
    Tensor tkcb = Contig(kb.data(), DType::kBF16, {nb, bs, H, D});
    Tensor tvcb = Contig(vb.data(), DType::kBF16, {nb, bs, H, D});
    vt::ReshapeAndCache(qq, tkb, tvb, tkcb, tvcb, ts);
  }

  // fp8 cache.
  std::vector<uint8_t> kf(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vf(static_cast<size_t>(nb * bs * page), 0);
  {
    Tensor tkcf = Contig(kf.data(), DType::kI8, {nb, bs, H, D});
    Tensor tvcf = Contig(vf.data(), DType::kI8, {nb, bs, H, D});
    vt::ReshapeAndCacheFp8(qq, tk, tv, tkcf, tvcf, ts, Fp8KVCacheDataType::kFp8E4M3, k_scale,
                           v_scale);
  }

  // Shared attention metadata: 1 request, all seqlen tokens context, 1 query.
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(seqlen)};
  std::vector<int32_t> qsl = {0, 1};
  std::vector<int32_t> btab = {0, 1};  // blocks 0,1 cover seqlen<=8 tokens
  Tensor tq = Contig(query.data(), DType::kF32, {1, hq, D});
  Tensor tseq = Contig(seq_lens.data(), DType::kI32, {1});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, {2});
  Tensor tbt = Contig(btab.data(), DType::kI32, {1, 2});

  PagedAttentionArgs base;
  base.scale = 1.0f / std::sqrt(static_cast<float>(D));
  base.causal = true;

  // bf16 output.
  std::vector<float> out_bf(static_cast<size_t>(1 * hq * D), 0.0f);
  Tensor tob = Contig(out_bf.data(), DType::kF32, {1, hq, D});
  Tensor tkcb = Contig(kb.data(), DType::kBF16, {nb, bs, H, D});
  Tensor tvcb = Contig(vb.data(), DType::kBF16, {nb, bs, H, D});
  vt::PagedAttention(qq, tob, tq, tkcb, tvcb, tbt, tseq, tqsl, base);

  // fp8 output.
  std::vector<float> out_fp(static_cast<size_t>(1 * hq * D), 0.0f);
  Tensor tof = Contig(out_fp.data(), DType::kF32, {1, hq, D});
  Tensor tkcf = Contig(kf.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvcf = Contig(vf.data(), DType::kI8, {nb, bs, H, D});
  PagedAttentionArgs fp8 = base;
  fp8.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
  fp8.k_scale = k_scale;
  fp8.v_scale = v_scale;
  vt::PagedAttention(qq, tof, tq, tkcf, tvcf, tbt, tseq, tqsl, fp8);

  for (size_t i = 0; i < out_bf.size(); ++i) {
    // The softmax-weighted average of fp8-quantized V, so a tight band suffices.
    CHECK(out_fp[i] == doctest::Approx(out_bf[i]).epsilon(0.05));
  }

  // RED-first #1: a WRONG v_scale (2x) must diverge from the baseline — proving
  // the read-side scale is load-bearing, not ignored.
  std::vector<float> out_bad(static_cast<size_t>(1 * hq * D), 0.0f);
  Tensor tobad = Contig(out_bad.data(), DType::kF32, {1, hq, D});
  PagedAttentionArgs badv = fp8;
  badv.v_scale = v_scale * 2.0f;
  vt::PagedAttention(qq, tobad, tq, tkcf, tvcf, tbt, tseq, tqsl, badv);
  double maxdiff = 0.0;
  for (size_t i = 0; i < out_bad.size(); ++i)
    maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(out_bad[i] - out_bf[i])));
  CHECK(maxdiff > 0.05);  // clearly outside the band → scale matters
}

// RED-first #2: reading an fp8 (kI8) cache with the AUTO (float) path is refused,
// not silently mis-read as floats. Guards against a missing-dequant regression.
TEST_CASE("paged_attention refuses an fp8 cache on the auto (no-dequant) path") {
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<uint8_t> kf(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vf(static_cast<size_t>(nb * bs * page), 0);
  std::vector<float> query(static_cast<size_t>(1 * 1 * D), 0.1f);
  std::vector<int32_t> seq_lens = {1}, qsl = {0, 1}, btab = {0};
  Tensor tq = Contig(query.data(), DType::kF32, {1, 1, D});
  Tensor tkcf = Contig(kf.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvcf = Contig(vf.data(), DType::kI8, {nb, bs, H, D});
  Tensor tseq = Contig(seq_lens.data(), DType::kI32, {1});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, {2});
  Tensor tbt = Contig(btab.data(), DType::kI32, {1, 1});
  std::vector<float> out(static_cast<size_t>(D), 0.0f);
  Tensor to = Contig(out.data(), DType::kF32, {1, 1, D});
  PagedAttentionArgs args;  // kv_cache_dtype defaults to kAuto
  args.scale = 0.25f;
  Queue qq = Q();
  CHECK_THROWS_AS(vt::PagedAttention(qq, to, tq, tkcf, tvcf, tbt, tseq, tqsl, args),
                  std::runtime_error);
}

TEST_CASE("reshape_and_cache_fp8 validates dtype/scale/kind") {
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<float> k(static_cast<size_t>(page), 1.0f), v(static_cast<size_t>(page), 1.0f);
  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<int64_t> slots = {0};
  Tensor tk = Contig(k.data(), DType::kF32, {1, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, {1, H, D});
  Tensor tkc = Contig(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvc = Contig(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, {1});
  Queue qq = Q();

  CHECK_NOTHROW(
      vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3, 0.01f, 0.01f));
  // Non-positive scale is refused.
  CHECK_THROWS_AS(
      vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3, 0.0f, 0.01f),
      std::runtime_error);
  // A float cache (not kI8) is refused (that is ReshapeAndCache's job).
  {
    std::vector<float> fc(static_cast<size_t>(nb * bs * page), 0.0f);
    Tensor tfc = Contig(fc.data(), DType::kF32, {nb, bs, H, D});
    CHECK_THROWS_AS(vt::ReshapeAndCacheFp8(qq, tk, tv, tfc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3,
                                           0.01f, 0.01f),
                    std::runtime_error);
  }
  // kAuto is refused (use ReshapeAndCache).
  CHECK_THROWS_AS(
      vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kAuto, 0.01f, 0.01f),
      std::runtime_error);
}

// fp8_e5m2 CPU compute is a NAMED later brick — it must be refused, not silently
// mis-stored. (Ported from test_cache.py KV_CACHE_DTYPE which includes e5m2; the
// e4m3 half is live, the e5m2 half is SKIPPED-with-reason via this refusal gate.)
TEST_CASE("reshape_and_cache_fp8 refuses e5m2 (later brick)") {
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<float> k(static_cast<size_t>(page), 1.0f), v(static_cast<size_t>(page), 1.0f);
  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<int64_t> slots = {0};
  Tensor tk = Contig(k.data(), DType::kF32, {1, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, {1, H, D});
  Tensor tkc = Contig(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvc = Contig(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, {1});
  Queue qq = Q();
  CHECK_THROWS_AS(
      vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E5M2, 0.01f, 0.01f),
      std::runtime_error);
}

// Config-field wiring: ParseCacheDType mirrors vLLM's CacheDType (config/cache.py:
// 19-36,76) for the fp8-e4m3 family this row owns.
TEST_CASE("ParseCacheDType mirrors the vLLM CacheDType surface") {
  using vllm::v1::IsQuantizedKvCache;
  using vllm::v1::ParseCacheDType;

  auto a = ParseCacheDType("auto", DType::kBF16);
  CHECK(a.is_fp8 == false);
  CHECK(a.storage == DType::kBF16);
  CHECK(a.fp8_kind == Fp8KVCacheDataType::kAuto);

  CHECK(ParseCacheDType("float16", DType::kBF16).storage == DType::kF16);
  CHECK(ParseCacheDType("bfloat16", DType::kF32).storage == DType::kBF16);

  for (const char* s : {"fp8", "fp8_e4m3"}) {
    auto r = ParseCacheDType(s, DType::kBF16);
    CHECK(r.is_fp8 == true);
    CHECK(r.storage == DType::kI8);
    CHECK(r.fp8_kind == Fp8KVCacheDataType::kFp8E4M3);
  }
  auto e5 = ParseCacheDType("fp8_e5m2", DType::kBF16);
  CHECK(e5.is_fp8 == true);
  CHECK(e5.fp8_kind == Fp8KVCacheDataType::kFp8E5M2);

  CHECK(IsQuantizedKvCache("auto") == false);
  CHECK(IsQuantizedKvCache("bfloat16") == false);
  CHECK(IsQuantizedKvCache("fp8") == true);
  CHECK(IsQuantizedKvCache("fp8_e5m2") == true);

  // Members owned by later rows are refused (mirrored surface, scoped compute).
  CHECK_THROWS_AS(ParseCacheDType("nvfp4", DType::kBF16), std::runtime_error);
  CHECK_THROWS_AS(ParseCacheDType("fp8_ds_mla", DType::kBF16), std::runtime_error);
}
