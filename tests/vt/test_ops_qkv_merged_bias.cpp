// vllm.cpp original (fold plan Tier C2 — the merged-QKV *with bias* A/B); no
// upstream mirror.
//
// C2 folds a vision-tower QKV whose weight is ALREADY a single resident merged
// [q_dim+k_dim+v_dim, H] tensor onto ONE bf16 vt::MatmulBT + a fused merged
// [q_dim+k_dim+v_dim] BIAS epilogue (vt::Add, broadcast per column) + a
// contiguous vt::QkvSplit. This is the exact op sequence run by
// models::FusedMergedQkvBiasSplit (src/vllm/model_executor/models/merged_qkv_fold.h).
//
// The NEW piece vs the Tier-D1 merged-QKV (test_ops_qkv_merge.cpp, which had NO
// bias — AttnBlock VT_CHECKs qkv_bias.Empty) is the fused per-[3d] bias add. This
// is the RED-first proof of that epilogue's correctness claim:
//
//   ONE MatmulBT over [wq;wk;wv], then ONE Add of the merged [bq;bk;bv] bias,
//   then a contiguous QkvSplit, is BYTE-IDENTICAL to three separate
//   { MatmulBT(w_slice) + Add(b_slice) }.
//
// It is bit-exact because MatmulBT computes each OUTPUT ROW n as an independent
// reduction out[t,n] = sum_k a[t,k]*b[n,k] (widening N does not touch the per-row
// operand or reduction), the merged bias add broadcasts per COLUMN n so it equals
// three per-slice [d] adds on the contiguous thirds, and QkvSplit is a pure
// contiguous copy. CPU-only (the CPU MatmulBT reference is deterministic per
// output row, so the identity is exact here; the separately-gated GPU cuBLASLt
// wider-N algo choice is the characterized near-tie, not this op-level identity).
//
// RED-first: a WRONG bias slicing (k/v bias offsets swapped) must fail the
// byte-check, proving the assertion has teeth.
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

// Vision-tower QKV: q/k/v all project to the same dim (MHA, Hq==Hkv), but the
// A/B holds for GQA too, so a couple of shapes vary the split.
struct Shape {
  int64_t T, H, qdim, kdim, vdim;
  int64_t nqkv() const { return qdim + kdim + vdim; }
};

std::vector<uint16_t> RandBf16(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = vt::F32ToBF16(dist(rng));
  return v;
}

// Baseline: three { MatmulBT(w_slice) + Add(bias_slice) } (the unfused vision
// qkv — Qwen3-VL's 3x LinearBias over the qkv_w row-slices).
void SeparateQkvBias(const Shape& s, const std::vector<uint16_t>& x,
                     const std::vector<uint16_t>& wqkv,
                     const std::vector<uint16_t>& bqkv, std::vector<uint16_t>& q,
                     std::vector<uint16_t>& k, std::vector<uint16_t>& v) {
  Queue qq = CpuQueue();
  Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, Cpu(),
                                 {s.T, s.H});
  Tensor twqkv = Tensor::Contiguous(const_cast<uint16_t*>(wqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv(), s.H});
  Tensor tbqkv = Tensor::Contiguous(const_cast<uint16_t*>(bqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv()});
  Tensor wq = twqkv.Slice(0, 0, s.qdim);
  Tensor wk = twqkv.Slice(0, s.qdim, s.qdim + s.kdim);
  Tensor wv = twqkv.Slice(0, s.qdim + s.kdim, s.nqkv());
  Tensor bq = tbqkv.Slice(0, 0, s.qdim);
  Tensor bk = tbqkv.Slice(0, s.qdim, s.qdim + s.kdim);
  Tensor bv = tbqkv.Slice(0, s.qdim + s.kdim, s.nqkv());
  Tensor tq = Tensor::Contiguous(q.data(), DType::kBF16, Cpu(), {s.T, s.qdim});
  Tensor tk = Tensor::Contiguous(k.data(), DType::kBF16, Cpu(), {s.T, s.kdim});
  Tensor tv = Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {s.T, s.vdim});
  vt::MatmulBT(qq, tq, tx, wq);
  vt::Add(qq, tq, tq, bq);
  vt::MatmulBT(qq, tk, tx, wk);
  vt::Add(qq, tk, tk, bk);
  vt::MatmulBT(qq, tv, tx, wv);
  vt::Add(qq, tv, tv, bv);
}

// Folded: ONE MatmulBT + ONE merged-bias Add + contiguous QkvSplit (exactly what
// models::FusedMergedQkvBiasSplit runs).
void MergedQkvBias(const Shape& s, const std::vector<uint16_t>& x,
                   const std::vector<uint16_t>& wqkv,
                   const std::vector<uint16_t>& bqkv, std::vector<uint16_t>& q,
                   std::vector<uint16_t>& k, std::vector<uint16_t>& v) {
  Queue qq = CpuQueue();
  Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, Cpu(),
                                 {s.T, s.H});
  Tensor twqkv = Tensor::Contiguous(const_cast<uint16_t*>(wqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv(), s.H});
  Tensor tbqkv = Tensor::Contiguous(const_cast<uint16_t*>(bqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv()});
  std::vector<uint16_t> qkv(static_cast<size_t>(s.T * s.nqkv()), 0);
  Tensor tqkv = Tensor::Contiguous(qkv.data(), DType::kBF16, Cpu(), {s.T, s.nqkv()});
  vt::MatmulBT(qq, tqkv, tx, twqkv);
  vt::Add(qq, tqkv, tqkv, tbqkv);  // merged [nqkv] bias, broadcast per column
  Tensor tq = Tensor::Contiguous(q.data(), DType::kBF16, Cpu(), {s.T, s.qdim});
  Tensor tk = Tensor::Contiguous(k.data(), DType::kBF16, Cpu(), {s.T, s.kdim});
  Tensor tv = Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {s.T, s.vdim});
  vt::QkvSplit(qq, tq, tk, tv, tqkv);
}
}  // namespace

TEST_CASE("qkv-merged-bias: merged MatmulBT+Add+QkvSplit == 3x {MatmulBT+Add} (bf16, byte-identical)") {
  const std::vector<Shape> shapes = {
      {5, 64, 64, 64, 64},     // Qwen3-VL-ish MHA (q=k=v=H)
      {3, 128, 128, 128, 128}, // wider MHA
      {7, 96, 48, 24, 24},     // GQA (q != k == v) — split boundaries distinct
      {4, 256, 256, 256, 256}, // Gemma-4-vision-ish (q=k=v=H)
  };
  int idx = 0;
  for (const Shape& s : shapes) {
    const auto x = RandBf16(static_cast<size_t>(s.T * s.H), 100u + idx);
    const auto wqkv = RandBf16(static_cast<size_t>(s.nqkv() * s.H), 200u + idx);
    const auto bqkv = RandBf16(static_cast<size_t>(s.nqkv()), 300u + idx);

    std::vector<uint16_t> qa(static_cast<size_t>(s.T * s.qdim), 1);
    std::vector<uint16_t> ka(static_cast<size_t>(s.T * s.kdim), 1);
    std::vector<uint16_t> va(static_cast<size_t>(s.T * s.vdim), 1);
    SeparateQkvBias(s, x, wqkv, bqkv, qa, ka, va);

    std::vector<uint16_t> qb(static_cast<size_t>(s.T * s.qdim), 2);
    std::vector<uint16_t> kb(static_cast<size_t>(s.T * s.kdim), 2);
    std::vector<uint16_t> vb(static_cast<size_t>(s.T * s.vdim), 2);
    MergedQkvBias(s, x, wqkv, bqkv, qb, kb, vb);

    CHECK(std::memcmp(qa.data(), qb.data(), qa.size() * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(ka.data(), kb.data(), ka.size() * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(va.data(), vb.data(), va.size() * sizeof(uint16_t)) == 0);
    ++idx;
  }
}

TEST_CASE("qkv-merged-bias: RED-first — a wrong k/v BIAS slice must NOT match (the check has teeth)") {
  // Correct GEMM + split, but slice the merged BIAS at the WRONG boundary (swap
  // where the k and v bias sub-vectors are read). If the byte-check above were
  // vacuous this would still pass; it must FAIL, proving the merged-bias epilogue
  // identity is a real constraint.
  const Shape s{6, 128, 64, 32, 32};  // distinct q/k/v dims and distinct biases
  const auto x = RandBf16(static_cast<size_t>(s.T * s.H), 7u);
  const auto wqkv = RandBf16(static_cast<size_t>(s.nqkv() * s.H), 9u);
  const auto bqkv = RandBf16(static_cast<size_t>(s.nqkv()), 11u);

  std::vector<uint16_t> qa(static_cast<size_t>(s.T * s.qdim), 0);
  std::vector<uint16_t> ka(static_cast<size_t>(s.T * s.kdim), 0);
  std::vector<uint16_t> va(static_cast<size_t>(s.T * s.vdim), 0);
  SeparateQkvBias(s, x, wqkv, bqkv, qa, ka, va);

  // Merged GEMM (correct), then add a WRONG-sliced bias: k gets v's bias and v
  // gets k's bias.
  Queue qq = CpuQueue();
  Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, Cpu(),
                                 {s.T, s.H});
  Tensor twqkv = Tensor::Contiguous(const_cast<uint16_t*>(wqkv.data()), DType::kBF16,
                                    Cpu(), {s.nqkv(), s.H});
  std::vector<uint16_t> qkv(static_cast<size_t>(s.T * s.nqkv()), 0);
  Tensor tqkv = Tensor::Contiguous(qkv.data(), DType::kBF16, Cpu(), {s.T, s.nqkv()});
  vt::MatmulBT(qq, tqkv, tx, twqkv);

  // Build a mis-ordered bias [bq; bv; bk] and add it (wrong).
  std::vector<uint16_t> bwrong(static_cast<size_t>(s.nqkv()));
  std::memcpy(bwrong.data(), bqkv.data(), s.qdim * sizeof(uint16_t));
  std::memcpy(bwrong.data() + s.qdim, bqkv.data() + s.qdim + s.kdim,
              s.vdim * sizeof(uint16_t));  // v-bias into k slot
  std::memcpy(bwrong.data() + s.qdim + s.vdim, bqkv.data() + s.qdim,
              s.kdim * sizeof(uint16_t));  // k-bias into v slot
  Tensor tbwrong = Tensor::Contiguous(bwrong.data(), DType::kBF16, Cpu(), {s.nqkv()});
  vt::Add(qq, tqkv, tqkv, tbwrong);

  std::vector<uint16_t> qbad(static_cast<size_t>(s.T * s.qdim), 0);
  std::vector<uint16_t> kbad(static_cast<size_t>(s.T * s.kdim), 0);
  std::vector<uint16_t> vbad(static_cast<size_t>(s.T * s.vdim), 0);
  Tensor tq = Tensor::Contiguous(qbad.data(), DType::kBF16, Cpu(), {s.T, s.qdim});
  Tensor tk = Tensor::Contiguous(kbad.data(), DType::kBF16, Cpu(), {s.T, s.kdim});
  Tensor tv = Tensor::Contiguous(vbad.data(), DType::kBF16, Cpu(), {s.T, s.vdim});
  vt::QkvSplit(qq, tq, tk, tv, tqkv);

  const bool k_differs =
      std::memcmp(ka.data(), kbad.data(), ka.size() * sizeof(uint16_t)) != 0;
  const bool v_differs =
      std::memcmp(va.data(), vbad.data(), va.size() * sizeof(uint16_t)) != 0;
  CHECK(k_differs);
  CHECK(v_differs);
}
