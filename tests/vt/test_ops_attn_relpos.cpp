// Byte-exactness + determinism gate for `vt::AttentionRelPos` (spike row P3,
// .agents/specs/parakeet-conformer-encoder.md; kernel
// src/vt/cpu/cpu_attn_relpos.cpp).
//
// The op mirrors `ParakeetEncoderAttention` (transformers 5.3.0
// transformers/models/parakeet/modeling_parakeet.py:259, forward :302-347,
// `_rel_shift` :349-355, softmax/value contraction in `eager_attention_forward`
// :225-255) — the module vLLM runs (vllm/model_executor/models/parakeet.py:37,62)
// — and vLLM's own native `RelPosMultiHeadAttention`
// (vllm/model_executor/models/conformer_encoder.py:170, forward :188-217,
// `_rel_shift` :179-186), whose only arithmetic difference is where the scale
// lands (`AttentionRelPosArgs::scale_after_sum`).
//
// GATE, matching tests/vt/test_ops_matmul_elem.cpp discipline: `memcmp` against
// an INDEPENDENT in-test scalar reference, NOT against the kernel. The reference
// deliberately performs `_rel_shift` LITERALLY — build the [T, 2T-1] matrix,
// left-pad a column, reinterpret as [2T, T], drop the first row, reinterpret as
// [T, 2T-1], truncate to T columns — exactly as both upstreams do. The kernel
// instead indexes rel_key at the closed form `T-1-i+j`. Byte-identity between
// the two IS the proof of that derivation, checked by execution rather than by
// reading the source.
//
// Coverage: T from 1 to 13 (including the P == 2T-1 == 1 degenerate case), MHA
// and GQA head ratios, head_dim 1..16, f32/f16/bf16 (including mixed operand
// dtypes), bias_u/bias_v present and absent, both scale placements, a padding
// key_mask, the all-masked row deviation, and thread counts 1/2/4/8.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (via -I src)
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionRelPosArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float Uniform() { return static_cast<float>(Next() % 20001) / 10000.0f - 1.0f; }
};

struct Buf {
  DType dtype;
  std::vector<uint8_t> bytes;
  std::vector<float> ref;

  Buf(DType dt, int64_t n, Rng& rng) : dtype(dt) {
    ref.resize(static_cast<size_t>(n));
    bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
    for (int64_t i = 0; i < n; ++i) {
      const float v = rng.Uniform();
      switch (dt) {
        case DType::kF32:
          reinterpret_cast<float*>(bytes.data())[i] = v;
          ref[i] = v;
          break;
        case DType::kF16: {
          const uint16_t h = vt::F32ToF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::F16ToF32(h);
          break;
        }
        case DType::kBF16: {
          const uint16_t h = vt::F32ToBF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::BF16ToF32(h);
          break;
        }
        default:
          break;
      }
    }
  }
  void* Data() { return bytes.data(); }
};

void StoreAt(std::vector<uint8_t>* out, int64_t i, DType dt, float v) {
  switch (dt) {
    case DType::kF32: reinterpret_cast<float*>(out->data())[i] = v; break;
    case DType::kF16: reinterpret_cast<uint16_t*>(out->data())[i] = vt::F32ToF16(v); break;
    case DType::kBF16: reinterpret_cast<uint16_t*>(out->data())[i] = vt::F32ToBF16(v); break;
    default: break;
  }
}

struct Case {
  const char* name;
  int64_t t, hq, hk, d;
};

const std::vector<Case>& Cases() {
  static const std::vector<Case> c = {
      {"T1 degenerate", 1, 2, 2, 4},
      {"T2", 2, 1, 1, 3},
      {"T3 mha", 3, 4, 4, 8},
      {"T5 odd d", 5, 3, 3, 5},
      {"T8 gqa 6/2", 8, 6, 2, 4},
      {"T13 gqa 3/1", 13, 3, 1, 16},
      {"d1", 7, 2, 2, 1},
      {"single head", 11, 1, 1, 9},
      {"wide heads", 4, 8, 4, 2},
  };
  return c;
}

// LITERAL `_rel_shift`, transcribed op for op from the upstreams:
//   HF   (modeling_parakeet.py:349-355):
//        pad(x, (1,0)) -> view(B,H,-1,T) -> [:, :, 1:] -> view(B,H,T,P)
//   vLLM (conformer_encoder.py:179-186): cat(zeros, x, -1) -> view(N,H,T2+1,T1)
//        -> [:, :, 1:] -> view_as(x) -> [..., : x.size(-1)//2 + 1]
// Both truncate to T columns when P == 2T-1. `raw` is [T, P] row-major; the
// result is [T, T].
std::vector<float> RelShift(const std::vector<float>& raw, int64_t t) {
  const int64_t p = 2 * t - 1;
  // left-pad one column: [T, P+1] == [T, 2T]
  std::vector<float> padded(static_cast<size_t>(t * (p + 1)), 0.0f);
  for (int64_t i = 0; i < t; ++i)
    for (int64_t c = 0; c < p; ++c)
      padded[static_cast<size_t>(i * (p + 1) + 1 + c)] = raw[static_cast<size_t>(i * p + c)];
  // reinterpret as [2T, T] and drop the first row -> [2T-1, T], flat length T*P
  std::vector<float> dropped(padded.begin() + t, padded.end());
  // reinterpret as [T, P] and keep the first T columns
  std::vector<float> out(static_cast<size_t>(t * t), 0.0f);
  for (int64_t i = 0; i < t; ++i)
    for (int64_t j = 0; j < t; ++j)
      out[static_cast<size_t>(i * t + j)] = dropped[static_cast<size_t>(i * p + j)];
  return out;
}

// INDEPENDENT scalar reference, written from the upstream forward:
//   q_u = q + bias_u ; q_v = q + bias_v                    (:295-300)
//   matrix_ac = q_u @ k^T                                  (eager :247 / :208)
//   matrix_bd = rel_shift(q_v @ rel_k^T)[..., :T]          (:317-320 / :209-210)
//   s = ac*scale + bd*scale   (HF)  |  (ac+bd)*scale  (vLLM native)
//   out = softmax_j(s) @ v                                 (:251-253)
void RefRelPos(const Case& c, const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v, const std::vector<float>& rk,
               const std::vector<float>* bu, const std::vector<float>* bv,
               const std::vector<int32_t>* mask, float scale, bool scale_after_sum, DType odt,
               std::vector<uint8_t>* out) {
  const int64_t t = c.t, hq = c.hq, hk = c.hk, d = c.d;
  const int64_t p = 2 * t - 1;
  const int64_t qpk = hq / hk;
  const float kNegInf = -std::numeric_limits<float>::infinity();
  out->assign(static_cast<size_t>(t * hq * d) * vt::SizeOf(odt), 0);
  for (int64_t h = 0; h < hq; ++h) {
    const int64_t g = h / qpk;
    // matrix_bd BEFORE the shift: [T, P], then shifted to [T, T].
    std::vector<float> raw(static_cast<size_t>(t * p), 0.0f);
    std::vector<float> ac(static_cast<size_t>(t * t), 0.0f);
    for (int64_t i = 0; i < t; ++i) {
      std::vector<float> qu(static_cast<size_t>(d)), qv(static_cast<size_t>(d));
      for (int64_t e = 0; e < d; ++e) {
        const float qe = q[static_cast<size_t>((i * hq + h) * d + e)];
        qu[static_cast<size_t>(e)] = bu != nullptr ? qe + (*bu)[static_cast<size_t>(h * d + e)] : qe;
        qv[static_cast<size_t>(e)] = bv != nullptr ? qe + (*bv)[static_cast<size_t>(h * d + e)] : qe;
      }
      for (int64_t pp = 0; pp < p; ++pp) {
        float s = 0.0f;
        for (int64_t e = 0; e < d; ++e)
          s += qv[static_cast<size_t>(e)] * rk[static_cast<size_t>((pp * hq + h) * d + e)];
        raw[static_cast<size_t>(i * p + pp)] = s;
      }
      for (int64_t j = 0; j < t; ++j) {
        float s = 0.0f;
        for (int64_t e = 0; e < d; ++e)
          s += qu[static_cast<size_t>(e)] * k[static_cast<size_t>((j * hk + g) * d + e)];
        ac[static_cast<size_t>(i * t + j)] = s;
      }
    }
    const std::vector<float> bd = RelShift(raw, t);
    for (int64_t i = 0; i < t; ++i) {
      std::vector<float> s(static_cast<size_t>(t), 0.0f);
      float m = kNegInf;
      for (int64_t j = 0; j < t; ++j) {
        if (mask != nullptr && (*mask)[static_cast<size_t>(j)] == 0) {
          s[static_cast<size_t>(j)] = kNegInf;
          continue;
        }
        const float a = ac[static_cast<size_t>(i * t + j)];
        const float b = bd[static_cast<size_t>(i * t + j)];
        s[static_cast<size_t>(j)] = scale_after_sum ? (a + b) * scale : scale * a + scale * b;
        if (s[static_cast<size_t>(j)] > m) m = s[static_cast<size_t>(j)];
      }
      if (m == kNegInf) {
        for (int64_t e = 0; e < d; ++e) StoreAt(out, (i * hq + h) * d + e, odt, 0.0f);
        continue;
      }
      float denom = 0.0f;
      for (int64_t j = 0; j < t; ++j) {
        const float e = std::exp(s[static_cast<size_t>(j)] - m);
        s[static_cast<size_t>(j)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      std::vector<float> acc(static_cast<size_t>(d), 0.0f);
      for (int64_t j = 0; j < t; ++j) {
        const float pw = s[static_cast<size_t>(j)] * inv;
        for (int64_t e = 0; e < d; ++e)
          acc[static_cast<size_t>(e)] += pw * v[static_cast<size_t>((j * hk + g) * d + e)];
      }
      for (int64_t e = 0; e < d; ++e)
        StoreAt(out, (i * hq + h) * d + e, odt, acc[static_cast<size_t>(e)]);
    }
  }
}

struct Fixture {
  Case c;
  Buf q, k, v, rk, bu, bv;
  std::vector<int32_t> mask;
  float scale;

  Fixture(const Case& cc, DType qdt, DType kdt, DType rdt, uint64_t seed, Rng& rng)
      : c(cc),
        q(qdt, cc.t * cc.hq * cc.d, rng),
        k(kdt, cc.t * cc.hk * cc.d, rng),
        v(kdt, cc.t * cc.hk * cc.d, rng),
        rk(rdt, (2 * cc.t - 1) * cc.hq * cc.d, rng),
        bu(qdt, cc.hq * cc.d, rng),
        bv(qdt, cc.hq * cc.d, rng),
        mask(static_cast<size_t>(cc.t), 1) {
    (void)seed;
    scale = 1.0f / std::sqrt(static_cast<float>(cc.d));
  }
};

void RunCase(const Case& c, DType qdt, DType kdt, DType rdt, DType odt, bool bias, bool masked,
             bool scale_after_sum, uint64_t seed) {
  Queue qq = CpuQueue();
  Rng rng(seed);
  Fixture f(c, qdt, kdt, rdt, seed, rng);
  if (masked) {
    // A trailing-padding mask, the shape a padded encoder batch produces.
    for (int64_t j = c.t - c.t / 3; j < c.t; ++j) f.mask[static_cast<size_t>(j)] = 0;
  }
  const int64_t p = 2 * c.t - 1;
  std::vector<uint8_t> got(static_cast<size_t>(c.t * c.hq * c.d) * vt::SizeOf(odt), 0xAB);
  Tensor tq = Tensor::Contiguous(f.q.Data(), qdt, Cpu(), {c.t, c.hq, c.d});
  Tensor tk = Tensor::Contiguous(f.k.Data(), kdt, Cpu(), {c.t, c.hk, c.d});
  Tensor tv = Tensor::Contiguous(f.v.Data(), kdt, Cpu(), {c.t, c.hk, c.d});
  Tensor tr = Tensor::Contiguous(f.rk.Data(), rdt, Cpu(), {p, c.hq, c.d});
  Tensor tbu = Tensor::Contiguous(f.bu.Data(), qdt, Cpu(), {c.hq, c.d});
  Tensor tbv = Tensor::Contiguous(f.bv.Data(), qdt, Cpu(), {c.hq, c.d});
  Tensor tm = Tensor::Contiguous(f.mask.data(), DType::kI32, Cpu(), {c.t});
  Tensor to = Tensor::Contiguous(got.data(), odt, Cpu(), {c.t, c.hq, c.d});
  AttentionRelPosArgs args;
  args.scale = f.scale;
  args.scale_after_sum = scale_after_sum;
  vt::AttentionRelPos(qq, to, tq, tk, tv, tr, bias ? &tbu : nullptr, bias ? &tbv : nullptr,
                      masked ? &tm : nullptr, args);

  std::vector<uint8_t> want;
  RefRelPos(c, f.q.ref, f.k.ref, f.v.ref, f.rk.ref, bias ? &f.bu.ref : nullptr,
            bias ? &f.bv.ref : nullptr, masked ? &f.mask : nullptr, f.scale, scale_after_sum, odt,
            &want);
  REQUIRE(want.size() == got.size());
  CHECK_MESSAGE(std::memcmp(got.data(), want.data(), want.size()) == 0, c.name);
}

}  // namespace

TEST_CASE("attention relpos: byte-identical to the literal _rel_shift reference, every dtype") {
  const DType kDt[3] = {DType::kF32, DType::kF16, DType::kBF16};
  uint64_t seed = 1;
  for (DType qdt : kDt) {
    for (DType odt : kDt) {
      for (const Case& c : Cases()) {
        RunCase(c, qdt, qdt, qdt, odt, /*bias=*/true, /*masked=*/false,
                /*scale_after_sum=*/false, seed++);
      }
    }
  }
}

TEST_CASE("attention relpos: mixed operand dtypes, bias variants, both scale placements") {
  uint64_t seed = 5000;
  for (bool bias : {false, true}) {
    for (bool scale_after_sum : {false, true}) {
      for (const Case& c : Cases()) {
        // q bf16, k/v f16, rel_k f32 — the operand dtypes are independent, so a
        // kernel that assumed one shared dtype would read garbage here.
        RunCase(c, DType::kBF16, DType::kF16, DType::kF32, DType::kF32, bias, /*masked=*/false,
                scale_after_sum, seed++);
        RunCase(c, DType::kF32, DType::kF32, DType::kF32, DType::kBF16, bias, /*masked=*/false,
                scale_after_sum, seed++);
      }
    }
  }
}

TEST_CASE("attention relpos: a padding key_mask matches the reference") {
  uint64_t seed = 9000;
  for (const Case& c : Cases()) {
    if (c.t < 3) continue;  // t/3 == 0 would mask nothing
    RunCase(c, DType::kF32, DType::kF32, DType::kF32, DType::kF32, /*bias=*/true, /*masked=*/true,
            /*scale_after_sum=*/false, seed++);
    RunCase(c, DType::kBF16, DType::kBF16, DType::kBF16, DType::kF32, /*bias=*/true,
            /*masked=*/true, /*scale_after_sum=*/true, seed++);
  }
}

TEST_CASE("attention relpos: an i8 key_mask is accepted like an i32 one") {
  Queue q = CpuQueue();
  const Case c = {"i8 mask", 6, 2, 2, 4};
  Rng rng(2468);
  Buf bq(DType::kF32, c.t * c.hq * c.d, rng);
  Buf bk(DType::kF32, c.t * c.hk * c.d, rng);
  Buf bv(DType::kF32, c.t * c.hk * c.d, rng);
  Buf br(DType::kF32, (2 * c.t - 1) * c.hq * c.d, rng);
  std::vector<int32_t> m32 = {1, 1, 1, 1, 0, 0};
  std::vector<int8_t> m8 = {1, 1, 1, 1, 0, 0};
  AttentionRelPosArgs args;
  args.scale = 0.5f;
  Tensor tq = Tensor::Contiguous(bq.Data(), DType::kF32, Cpu(), {c.t, c.hq, c.d});
  Tensor tk = Tensor::Contiguous(bk.Data(), DType::kF32, Cpu(), {c.t, c.hk, c.d});
  Tensor tv = Tensor::Contiguous(bv.Data(), DType::kF32, Cpu(), {c.t, c.hk, c.d});
  Tensor tr = Tensor::Contiguous(br.Data(), DType::kF32, Cpu(), {2 * c.t - 1, c.hq, c.d});
  std::vector<float> a(static_cast<size_t>(c.t * c.hq * c.d), 0.0f), b = a;
  Tensor ta = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {c.t, c.hq, c.d});
  Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {c.t, c.hq, c.d});
  Tensor t32 = Tensor::Contiguous(m32.data(), DType::kI32, Cpu(), {c.t});
  Tensor t8 = Tensor::Contiguous(m8.data(), DType::kI8, Cpu(), {c.t});
  vt::AttentionRelPos(q, ta, tq, tk, tv, tr, nullptr, nullptr, &t32, args);
  vt::AttentionRelPos(q, tb, tq, tk, tv, tr, nullptr, nullptr, &t8, args);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("attention relpos: an all-masked query row yields zeros, not NaN") {
  // Recorded deviation (include/vt/ops.h): upstream softmaxes an all -inf row
  // and gets NaN; such a row is padding whose output is discarded, so zeros are
  // emitted. Asserted rather than left to chance because a NaN would propagate
  // through the encoder's residual stream.
  Queue q = CpuQueue();
  const int64_t t = 4, hq = 2, d = 3;
  Rng rng(1357);
  Buf bq(DType::kF32, t * hq * d, rng);
  Buf bk(DType::kF32, t * hq * d, rng);
  Buf bv(DType::kF32, t * hq * d, rng);
  Buf br(DType::kF32, (2 * t - 1) * hq * d, rng);
  std::vector<int32_t> mask(static_cast<size_t>(t), 0);  // nothing valid at all
  std::vector<float> got(static_cast<size_t>(t * hq * d), 1.0f);
  Tensor tq = Tensor::Contiguous(bq.Data(), DType::kF32, Cpu(), {t, hq, d});
  Tensor tk = Tensor::Contiguous(bk.Data(), DType::kF32, Cpu(), {t, hq, d});
  Tensor tv = Tensor::Contiguous(bv.Data(), DType::kF32, Cpu(), {t, hq, d});
  Tensor tr = Tensor::Contiguous(br.Data(), DType::kF32, Cpu(), {2 * t - 1, hq, d});
  Tensor tm = Tensor::Contiguous(mask.data(), DType::kI32, Cpu(), {t});
  Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {t, hq, d});
  AttentionRelPosArgs args;
  args.scale = 0.25f;
  vt::AttentionRelPos(q, to, tq, tk, tv, tr, nullptr, nullptr, &tm, args);
  for (float x : got) CHECK(x == 0.0f);
}

TEST_CASE("attention relpos: byte-identical across thread counts") {
  // Determinism contract (src/vt/cpu/cpu_attn_relpos.cpp): parallelism
  // partitions (head, query) rows only. 3*13 == 39 rows does not divide by 8.
  Queue q = CpuQueue();
  const Case c = {"threads", 13, 3, 1, 16};
  Rng rng(777);
  Buf bq(DType::kBF16, c.t * c.hq * c.d, rng);
  Buf bk(DType::kBF16, c.t * c.hk * c.d, rng);
  Buf bv(DType::kBF16, c.t * c.hk * c.d, rng);
  Buf br(DType::kBF16, (2 * c.t - 1) * c.hq * c.d, rng);
  Buf bbu(DType::kBF16, c.hq * c.d, rng);
  Buf bbv(DType::kBF16, c.hq * c.d, rng);
  AttentionRelPosArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(c.d));
  std::vector<uint8_t> base;
  for (int nth : {1, 2, 4, 8}) {
    vt::cpu::Threadpool tp(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    std::vector<uint8_t> got(static_cast<size_t>(c.t * c.hq * c.d) * sizeof(float), 0xAB);
    Tensor tq = Tensor::Contiguous(bq.Data(), DType::kBF16, Cpu(), {c.t, c.hq, c.d});
    Tensor tk = Tensor::Contiguous(bk.Data(), DType::kBF16, Cpu(), {c.t, c.hk, c.d});
    Tensor tv = Tensor::Contiguous(bv.Data(), DType::kBF16, Cpu(), {c.t, c.hk, c.d});
    Tensor tr = Tensor::Contiguous(br.Data(), DType::kBF16, Cpu(), {2 * c.t - 1, c.hq, c.d});
    Tensor tbu = Tensor::Contiguous(bbu.Data(), DType::kBF16, Cpu(), {c.hq, c.d});
    Tensor tbv = Tensor::Contiguous(bbv.Data(), DType::kBF16, Cpu(), {c.hq, c.d});
    Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {c.t, c.hq, c.d});
    vt::AttentionRelPos(q, to, tq, tk, tv, tr, &tbu, &tbv, nullptr, args);
    vt::cpu::Threadpool::SwapForTesting(prev);
    if (base.empty()) {
      base = got;
      std::vector<uint8_t> want;
      RefRelPos(c, bq.ref, bk.ref, bv.ref, br.ref, &bbu.ref, &bbv.ref, nullptr, args.scale, false,
                DType::kF32, &want);
      CHECK(std::memcmp(base.data(), want.data(), want.size()) == 0);
    } else {
      CHECK(std::memcmp(base.data(), got.data(), base.size()) == 0);
    }
  }
}

TEST_CASE("attention relpos: the shape contract is enforced, not assumed") {
  Queue q = CpuQueue();
  const int64_t t = 5, hq = 4, hk = 2, d = 3;
  std::vector<float> qd(static_cast<size_t>(t * hq * d), 0.1f);
  std::vector<float> kd(static_cast<size_t>(t * hk * d), 0.2f);
  std::vector<float> vd(static_cast<size_t>(t * hk * d), 0.3f);
  std::vector<float> rd(static_cast<size_t>((2 * t - 1) * hq * d), 0.4f);
  std::vector<float> od(static_cast<size_t>(t * hq * d), 0.0f);
  Tensor tq = Tensor::Contiguous(qd.data(), DType::kF32, Cpu(), {t, hq, d});
  Tensor tk = Tensor::Contiguous(kd.data(), DType::kF32, Cpu(), {t, hk, d});
  Tensor tv = Tensor::Contiguous(vd.data(), DType::kF32, Cpu(), {t, hk, d});
  Tensor tr = Tensor::Contiguous(rd.data(), DType::kF32, Cpu(), {2 * t - 1, hq, d});
  Tensor to = Tensor::Contiguous(od.data(), DType::kF32, Cpu(), {t, hq, d});
  AttentionRelPosArgs args;
  args.scale = 0.5f;
  vt::AttentionRelPos(q, to, tq, tk, tv, tr, nullptr, nullptr, nullptr, args);  // well-formed

  // rel_key must be exactly [2T-1, Hq, D]
  std::vector<float> rbad(static_cast<size_t>(t * hq * d), 0.4f);
  Tensor trb = Tensor::Contiguous(rbad.data(), DType::kF32, Cpu(), {t, hq, d});
  CHECK_THROWS(vt::AttentionRelPos(q, to, tq, tk, tv, trb, nullptr, nullptr, nullptr, args));
  // scale must be set
  AttentionRelPosArgs noscale;
  CHECK_THROWS(vt::AttentionRelPos(q, to, tq, tk, tv, tr, nullptr, nullptr, nullptr, noscale));
  // Hq must be a multiple of Hkv
  std::vector<float> k3(static_cast<size_t>(t * 3 * d), 0.2f);
  Tensor tk3 = Tensor::Contiguous(k3.data(), DType::kF32, Cpu(), {t, 3, d});
  CHECK_THROWS(vt::AttentionRelPos(q, to, tq, tk3, tk3, tr, nullptr, nullptr, nullptr, args));
  // bias must be rank-2 [Hq, D]
  std::vector<float> bbad(static_cast<size_t>(hq * d), 0.0f);
  Tensor tbb = Tensor::Contiguous(bbad.data(), DType::kF32, Cpu(), {hq * d});
  CHECK_THROWS(vt::AttentionRelPos(q, to, tq, tk, tv, tr, &tbb, nullptr, nullptr, args));
  // key_mask must be i8/i32 [T]
  std::vector<int32_t> mbad(static_cast<size_t>(t + 1), 1);
  Tensor tmb = Tensor::Contiguous(mbad.data(), DType::kI32, Cpu(), {t + 1});
  CHECK_THROWS(vt::AttentionRelPos(q, to, tq, tk, tv, tr, nullptr, nullptr, &tmb, args));
}
