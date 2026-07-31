// vllm.cpp original (fold plan Tier D1 — the merged-QKV A/B); no upstream mirror.
//
// D1 folds the q/k/v projections that SHARE the layer input onto ONE bf16
// vt::MatmulBT over the merged [Nq+Nk+Nv, H] weight owner + a contiguous
// vt::QkvSplit (the OLMo-2/Granite/StableLM exemplar), flipping the default of the
// shared VT_QWEN3_QKV_MERGE gate ON for qwen3_dense, qwen3_coder, qwen3_dflash and
// Gemma-1/2/3/4. This is the RED-first proof of the merge's correctness claim:
//
//   ONE MatmulBT over the row-concatenated [wq;wk;wv] weight, then a contiguous
//   QkvSplit, is BYTE-IDENTICAL to three separate MatmulBT over wq/wk/wv.
//
// It is bit-exact GEMM math because MatmulBT computes each OUTPUT ROW n as an
// independent reduction out[t,n] = sum_k a[t,k]*b[n,k]: widening N (stacking the
// k/v weight rows after the q rows) changes neither the per-row operand nor the
// per-row reduction, and QkvSplit is a pure contiguous column copy. The ONLY
// byte-affecting part of D1 is downstream on hardware — cuBLASLt may pick a
// different K-reduction algo for the wider merged N — which is the CHARACTERIZED
// 0.6B near-tie handled by the batched golden regen, NOT this op-level identity.
//
// CPU-only (the reference MatmulBT is deterministic per output row on CPU, so the
// identity is exact here; the GPU cuBLASLt algo choice is the separately-gated
// near-tie). RED-first: a deliberately-WRONG split offset (k/v swapped) must fail
// the byte-check, proving the assertion has teeth.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

// GQA shape mirroring a small dense/Gemma block: q has Hq heads, k/v have Hkv,
// all of head_dim Dh, over hidden H.
struct Shape {
  int64_t T, H, Hq, Hkv, Dh;
  int64_t qdim() const { return Hq * Dh; }
  int64_t kdim() const { return Hkv * Dh; }
  int64_t nqkv() const { return qdim() + 2 * kdim(); }
};

std::vector<uint16_t> RandBf16(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = vt::F32ToBF16(dist(rng));
  return v;
}

// Separate path: three MatmulBT over the sliced q/k/v weight rows (the pre-D1
// baseline / VT_QWEN3_QKV_MERGE=0 arm).
void SeparateQkv(const Shape& s, const std::vector<uint16_t>& dhn,
                 const std::vector<uint16_t>& wqkv, std::vector<uint16_t>& q,
                 std::vector<uint16_t>& k, std::vector<uint16_t>& v) {
  Queue qq = CpuQueue();
  Tensor tdhn = Tensor::Contiguous(const_cast<uint16_t*>(dhn.data()), DType::kBF16,
                                   Cpu(), {s.T, s.H});
  Tensor twqkv = Tensor::Contiguous(const_cast<uint16_t*>(wqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv(), s.H});
  Tensor wq = twqkv.Slice(0, 0, s.qdim());
  Tensor wk = twqkv.Slice(0, s.qdim(), s.qdim() + s.kdim());
  Tensor wv = twqkv.Slice(0, s.qdim() + s.kdim(), s.qdim() + 2 * s.kdim());
  Tensor tq = Tensor::Contiguous(q.data(), DType::kBF16, Cpu(), {s.T, s.qdim()});
  Tensor tk = Tensor::Contiguous(k.data(), DType::kBF16, Cpu(), {s.T, s.kdim()});
  Tensor tv = Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {s.T, s.kdim()});
  vt::MatmulBT(qq, tq, tdhn, wq);
  vt::MatmulBT(qq, tk, tdhn, wk);
  vt::MatmulBT(qq, tv, tdhn, wv);
}

// Merged path: ONE MatmulBT over the whole owner + a contiguous QkvSplit (the D1
// / VT_QWEN3_QKV_MERGE=1 arm).
void MergedQkv(const Shape& s, const std::vector<uint16_t>& dhn,
               const std::vector<uint16_t>& wqkv, std::vector<uint16_t>& q,
               std::vector<uint16_t>& k, std::vector<uint16_t>& v) {
  Queue qq = CpuQueue();
  Tensor tdhn = Tensor::Contiguous(const_cast<uint16_t*>(dhn.data()), DType::kBF16,
                                   Cpu(), {s.T, s.H});
  Tensor twqkv = Tensor::Contiguous(const_cast<uint16_t*>(wqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv(), s.H});
  std::vector<uint16_t> qkv(static_cast<size_t>(s.T * s.nqkv()), 0);
  Tensor tqkv = Tensor::Contiguous(qkv.data(), DType::kBF16, Cpu(), {s.T, s.nqkv()});
  vt::MatmulBT(qq, tqkv, tdhn, twqkv);
  Tensor tq = Tensor::Contiguous(q.data(), DType::kBF16, Cpu(), {s.T, s.qdim()});
  Tensor tk = Tensor::Contiguous(k.data(), DType::kBF16, Cpu(), {s.T, s.kdim()});
  Tensor tv = Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {s.T, s.kdim()});
  vt::QkvSplit(qq, tq, tk, tv, tqkv);
}
}  // namespace

TEST_CASE("qkv-merge: merged MatmulBT+QkvSplit == separate q/k/v MatmulBT (bf16, byte-identical)") {
  // GQA (Qwen3/Gemma-shaped), MHA, and a Gemma-4-ish larger head_dim.
  const std::vector<Shape> shapes = {
      {5, 64, 8, 2, 8},    // GQA 8/2
      {3, 128, 4, 4, 16},  // MHA 4/4
      {7, 96, 6, 1, 16},   // MQA 6/1
      {4, 256, 8, 4, 32},  // wider head_dim (Gemma-4-ish)
  };
  int shape_idx = 0;
  for (const Shape& s : shapes) {
    const auto dhn = RandBf16(static_cast<size_t>(s.T * s.H), 100u + shape_idx);
    const auto wqkv = RandBf16(static_cast<size_t>(s.nqkv() * s.H), 200u + shape_idx);

    std::vector<uint16_t> qa(static_cast<size_t>(s.T * s.qdim()), 1);
    std::vector<uint16_t> ka(static_cast<size_t>(s.T * s.kdim()), 1);
    std::vector<uint16_t> va(static_cast<size_t>(s.T * s.kdim()), 1);
    SeparateQkv(s, dhn, wqkv, qa, ka, va);

    std::vector<uint16_t> qb(static_cast<size_t>(s.T * s.qdim()), 2);
    std::vector<uint16_t> kb(static_cast<size_t>(s.T * s.kdim()), 2);
    std::vector<uint16_t> vb(static_cast<size_t>(s.T * s.kdim()), 2);
    MergedQkv(s, dhn, wqkv, qb, kb, vb);

    CHECK(std::memcmp(qa.data(), qb.data(), qa.size() * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(ka.data(), kb.data(), ka.size() * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(va.data(), vb.data(), va.size() * sizeof(uint16_t)) == 0);
    ++shape_idx;
  }
}

TEST_CASE("qkv-merge: RED-first — a wrong k/v split offset must NOT match (the check has teeth)") {
  // Same math, but split the merged output at the WRONG boundary (swap where k and
  // v are read from). If the byte-check above were vacuous, this would still pass;
  // it must FAIL, proving the identity above is a real constraint.
  const Shape s{6, 128, 8, 2, 16};  // qdim=128, kdim=32 (k and v are distinct rows)
  const auto dhn = RandBf16(static_cast<size_t>(s.T * s.H), 7u);
  const auto wqkv = RandBf16(static_cast<size_t>(s.nqkv() * s.H), 9u);

  std::vector<uint16_t> qa(static_cast<size_t>(s.T * s.qdim()), 0);
  std::vector<uint16_t> ka(static_cast<size_t>(s.T * s.kdim()), 0);
  std::vector<uint16_t> va(static_cast<size_t>(s.T * s.kdim()), 0);
  SeparateQkv(s, dhn, wqkv, qa, ka, va);

  // Merged owner output.
  Queue qq = CpuQueue();
  Tensor tdhn = Tensor::Contiguous(const_cast<uint16_t*>(dhn.data()), DType::kBF16,
                                   Cpu(), {s.T, s.H});
  Tensor twqkv = Tensor::Contiguous(const_cast<uint16_t*>(wqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv(), s.H});
  std::vector<uint16_t> qkv(static_cast<size_t>(s.T * s.nqkv()), 0);
  Tensor tqkv = Tensor::Contiguous(qkv.data(), DType::kBF16, Cpu(), {s.T, s.nqkv()});
  vt::MatmulBT(qq, tqkv, tdhn, twqkv);

  // WRONG split: read k from the v column range and vice versa (offset bug).
  std::vector<uint16_t> kbad(static_cast<size_t>(s.T * s.kdim()), 0);
  std::vector<uint16_t> vbad(static_cast<size_t>(s.T * s.kdim()), 0);
  const int64_t row = s.nqkv();
  for (int64_t t = 0; t < s.T; ++t) {
    const uint16_t* src = qkv.data() + t * row;
    // v-range copied into k (wrong), k-range copied into v (wrong).
    std::memcpy(kbad.data() + t * s.kdim(), src + s.qdim() + s.kdim(),
                static_cast<size_t>(s.kdim()) * sizeof(uint16_t));
    std::memcpy(vbad.data() + t * s.kdim(), src + s.qdim(),
                static_cast<size_t>(s.kdim()) * sizeof(uint16_t));
  }
  // The wrong split must differ from the correct separate result (k != v generally).
  const bool k_differs =
      std::memcmp(ka.data(), kbad.data(), ka.size() * sizeof(uint16_t)) != 0;
  const bool v_differs =
      std::memcmp(va.data(), vbad.data(), va.size() * sizeof(uint16_t)) != 0;
  CHECK(k_differs);
  CHECK(v_differs);
}
