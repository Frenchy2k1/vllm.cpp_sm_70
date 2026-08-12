// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror —
// upstream's equivalent coverage is FlashAttention-2's own `tests/test_flash_attn.py`
// (vllm-project/flash-attention @ 2c839c33, `test_flash_attn_output` non-causal
// d=64), which cannot be ported directly because it is a torch/pytest harness
// comparing against `torch.nn.functional.scaled_dot_product_attention`. The
// adaptation is: same shape family (dense, non-paged, b=1, non-causal, head_dim 64,
// bf16), reference is our own byte-exact scalar `vt::AttentionDenseFlash` instead of
// SDPA, tolerance is a bf16-envelope rel-L2 instead of upstream's 2x-reference-error
// rule.
//
// WHAT THIS FILE GUARDS — `vt::AttentionDenseFa2` (`OpId::kAttentionDenseFa2`,
// multimodal-speed.md §17, issue #432). PR #439's review found the op shipped with
// ZERO tests, and demonstrated two defects that survived every gate in the tree:
//
//   M2a  Corrupt the params filler so the kernel attends only HALF the keys
//        (`p.seqlen_k = t/2`). No exception, no diagnostic, plausible tokens, and
//        the shipping default arm — which does not use this op — stayed 16/16.
//        Killed here by `fa2 dense attends the FULL key range`, which perturbs only
//        the tail of V and REQUIRES the output to move.
//
//   M3   Drop `!args.causal` from the dispatch gate and ask for causal attention.
//        The launcher hardcoded `p.is_causal = false`, so it returned an output
//        BIT-IDENTICAL to the non-causal one and refused nothing. Killed here by
//        `attention-dense-fa2 falls through for CAUSAL (M3)`, which requires
//        bit-equality with the causal reference. The launcher now also takes
//        `bool causal` and throws, so the mutation surfaces as a refusal; with the
//        guard also removed it surfaces as the bit-equality failure. Both are red.
//
// The op is TOTAL: its fast path is bf16 + head_dim 64 + non-causal + MHA + FA-2
// compiled, and every other shape must fall through to `AttentionDenseFlash`
// BIT-exactly. Both halves are covered below.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <stdlib.h>  // setenv/unsetenv (POSIX; not in <cstdlib>'s guaranteed set)

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionArgs;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Upload(Queue& q, const void* src) { b_.Copy(q, p_, src, bytes_); }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

// Deterministic LCG in [-1,1) — same shape as test_ops_attention.cpp's RandF32, so
// the two files cannot drift on <random> implementation differences.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
  }
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = vt::F32ToBF16(f[i]);
  return b;
}

std::vector<float> FromBf16(const std::vector<uint16_t>& b) {
  std::vector<float> f(b.size());
  for (size_t i = 0; i < b.size(); ++i) f[i] = vt::BF16ToF32(b[i]);
  return f;
}

double RelL2(const std::vector<float>& a, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return std::sqrt(num / (den + 1e-30));
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return m;
}

// Number of positions where two results differ at all. Used instead of
// `CHECK(a == b)` on the vectors: doctest stringifies both operands of a failing
// CHECK, and dumping two 65,000-element vectors buries the actual signal (it did,
// on the first M3-silent mutation run).
size_t Mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) ++n;
  return n;
}

double Rms(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(s / static_cast<double>(v.empty() ? 1 : v.size()));
}

enum class Op { kFa2, kFlash };

// Run one dense attention through either op on bf16 [T,H,D] host data, returning the
// bf16 output decoded to f32.
std::vector<float> RunBf16(Op op, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
                           const std::vector<uint16_t>& v, int64_t T, int64_t Hq, int64_t Hk,
                           int64_t D, float scale, bool causal) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kBF16, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kBF16, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kBF16, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kBF16, {T, Hq, D});
  const AttentionArgs args{scale, causal};
  if (op == Op::kFa2)
    vt::AttentionDenseFa2(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  else
    vt::AttentionDenseFlash(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  std::vector<uint16_t> got(static_cast<size_t>(T * Hq * D), 0);
  dout.Download(g.q, got.data());
  return FromBf16(got);
}

std::vector<float> RunF32(Op op, const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, int64_t T, int64_t Hq, int64_t Hk,
                          int64_t D, float scale, bool causal) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kF32, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {T, Hq, D});
  const AttentionArgs args{scale, causal};
  if (op == Op::kFa2)
    vt::AttentionDenseFa2(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  else
    vt::AttentionDenseFlash(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  std::vector<float> got(static_cast<size_t>(T * Hq * D), 0.0f);
  dout.Download(g.q, got.data());
  return got;
}

// The bf16 envelope. FA-2 and the scalar flash kernel compute the same function with
// different reduction orders AND different softmax formulations (exp2f on a
// log2-scaled score vs expf on a linearly-scaled one), and both store bf16, whose
// relative resolution is 2^-8 = 3.9e-3. A rel-L2 of a few e-3 is the expected
// agreement; anything approaching 1e-2 is a real defect. The M2a mutation (attend
// half the keys) lands at rel-L2 ~ O(1) against these bounds.
constexpr double kRelL2Bound = 1.0e-2;
constexpr double kMaxAbsVsRmsBound = 0.15;

}  // namespace

// ===========================================================================
// 1. The FAST PATH: bf16, head_dim 64, non-causal, MHA — the Whisper encoder shape.
//    FA-2 must agree with the shipping byte-exact scalar kernel inside the bf16
//    envelope.
TEST_CASE("attention-dense-fa2 bf16 hd-64 non-causal MHA matches AttentionDenseFlash") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 parity");
    return;
  }
  // (T, H): the real Voxtral/Whisper-large encoder geometry (1500 positions, 20
  // heads of 64) plus two shapes that are NOT multiples of FA-2's 128-wide K block,
  // so the epilogue masking is exercised too.
  const std::vector<std::pair<int64_t, int64_t>> shapes = {{1500, 20}, {257, 4}, {17, 2}};
  const int64_t D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (const auto& [T, H] : shapes) {
    const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 1000 + static_cast<uint32_t>(T)));
    const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 2000 + static_cast<uint32_t>(T)));
    const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 3000 + static_cast<uint32_t>(T)));

    const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, /*causal=*/false);
    const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, /*causal=*/false);

    const double rel = RelL2(got, ref);
    const double mad = MaxAbsDiff(got, ref);
    const double rms = Rms(ref);
    MESSAGE("T=", T, " H=", H, " D=64 non-causal: rel-L2 ", rel, "  max|diff| ", mad,
            "  (rms(ref) ", rms, ")");
    CHECK(rel < kRelL2Bound);
    CHECK(mad < kMaxAbsVsRmsBound * rms);
  }
}

// ===========================================================================
// 2. THE M2a KILLER — the op must attend the ENTIRE key range.
//    Reference-free: perturb ONLY the tail of V and require the output to move by
//    much more than the bf16 envelope. A params filler that clamps `seqlen_k`, or a
//    launcher that mis-strides K/V, silently drops those keys and this case goes red
//    while every token gate in the tree stays green (review of PR #439, M2a).
TEST_CASE("attention-dense-fa2 attends the FULL key range (M2a: p.seqlen_k = t/2)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 key-range coverage");
    return;
  }
  const int64_t T = 1500, H = 4, D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 4242));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 4243));
  auto vf = RandF32(static_cast<size_t>(T * H * D), 4244);
  const auto v_base = ToBf16(vf);

  // Perturb the SECOND HALF of the value rows only, by a large, unmistakable amount.
  for (int64_t t = T / 2; t < T; ++t)
    for (int64_t i = 0; i < H * D; ++i) vf[static_cast<size_t>(t * H * D + i)] += 8.0f;
  const auto v_tail = ToBf16(vf);

  const auto out_base = RunBf16(Op::kFa2, q, k, v_base, T, H, H, D, scale, false);
  const auto out_tail = RunBf16(Op::kFa2, q, k, v_tail, T, H, H, D, scale, false);

  const double moved = RelL2(out_tail, out_base);
  MESSAGE("perturbing V rows [", T / 2, ",", T, ") moved the output by rel-L2 ", moved,
          " (bf16 envelope is ", kRelL2Bound, ")");
  // With random q/k at hd-64 the softmax mass is spread over all 1500 keys, so
  // roughly half of it sits in the perturbed tail and the output must shift by O(1).
  // A kernel attending only keys [0, T/2) would shift by EXACTLY zero.
  CHECK(moved > 100.0 * kRelL2Bound);

  // And the scalar reference must see the same shift, so the case is measuring the
  // key range and not some FA-2-specific artefact.
  const auto ref_base = RunBf16(Op::kFlash, q, k, v_base, T, H, H, D, scale, false);
  const auto ref_tail = RunBf16(Op::kFlash, q, k, v_tail, T, H, H, D, scale, false);
  MESSAGE("  scalar reference moved by rel-L2 ", RelL2(ref_tail, ref_base));
  CHECK(RelL2(out_tail, ref_tail) < kRelL2Bound);
}

// ===========================================================================
// 3. TOTALITY / FALL-THROUGH — every shape outside the narrow fast path must reach
//    `AttentionDenseFlash`, and therefore be BIT-identical to calling it directly.
//
//    Four SEPARATE test cases, not four SUBCASEs of one: an uncaught exception ends
//    the whole enclosing TEST_CASE and doctest skips its remaining subcases. The M3
//    mutation throws, and under a single test case that silently dropped the GQA,
//    hd!=64 and f32 coverage from the run (14 assertions became 9). Separate cases
//    keep one failure from hiding three others.
//
//    The causal case is the M3 killer: with the dispatch gate's `!args.causal`
//    removed, FA-2 answered the NON-causal question instead and refused nothing.
TEST_CASE("attention-dense-fa2 falls through for CAUSAL (M3)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 causal fall-through");
    return;
  }
  // bf16, hd 64, MHA — everything the fast path wants EXCEPT non-causality.
  const int64_t T = 257, H = 4, D = 64;
  const float s64 = 1.0f / std::sqrt(64.0f);
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 51));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 52));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 53));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s64, /*causal=*/true);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, s64, /*causal=*/true);
  CHECK(Mismatches(got, ref) == 0);  // bit-exact: same kernel, same args
  // The causal answer must NOT be the non-causal one — otherwise "bit-exact vs the
  // reference" would be satisfiable by a kernel that ignores the mask entirely.
  const auto noncausal = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s64, /*causal=*/false);
  MESSAGE("causal vs non-causal rel-L2 (must be large): ", RelL2(ref, noncausal));
  CHECK(RelL2(ref, noncausal) > 100.0 * kRelL2Bound);
}

TEST_CASE("attention-dense-fa2 falls through for GQA (h_k != h)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 GQA fall-through");
    return;
  }
  const int64_t T = 96, Hq = 8, Hk = 2, D = 64;
  const float s64 = 1.0f / std::sqrt(64.0f);
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * Hq * D), 61));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * Hk * D), 62));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * Hk * D), 63));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, Hq, Hk, D, s64, false);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, Hq, Hk, D, s64, false);
  CHECK(Mismatches(got, ref) == 0);
}

TEST_CASE("attention-dense-fa2 falls through for head_dim != 64") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 head_dim fall-through");
    return;
  }
  const int64_t T = 96, H = 3, D = 80;  // no non-split FA-2 instantiation at 80
  const float s = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 71));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 72));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 73));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s, false);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, s, false);
  CHECK(Mismatches(got, ref) == 0);
}

TEST_CASE("attention-dense-fa2 falls through for f32") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 f32 fall-through");
    return;
  }
  const int64_t T = 96, H = 3, D = 64;
  const float s64 = 1.0f / std::sqrt(64.0f);
  const auto q = RandF32(static_cast<size_t>(T * H * D), 81);
  const auto k = RandF32(static_cast<size_t>(T * H * D), 82);
  const auto v = RandF32(static_cast<size_t>(T * H * D), 83);
  const auto ref = RunF32(Op::kFlash, q, k, v, T, H, H, D, s64, false);
  const auto got = RunF32(Op::kFa2, q, k, v, T, H, H, D, s64, false);
  CHECK(Mismatches(got, ref) == 0);
}

// ===========================================================================
// 4. THE A/B KNOB — `VT_FA2_DENSE=0` is the same-binary rollback arm recorded in
//    docs/ENVIRONMENT.md and used for every §17 measurement. If it stopped
//    selecting the scalar kernel, every A/B in the record would silently be
//    comparing the FA-2 arm against itself.
TEST_CASE("attention-dense-fa2 VT_FA2_DENSE=0 restores the scalar kernel bit-exactly") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping VT_FA2_DENSE A/B knob");
    return;
  }
  const int64_t T = 257, H = 4, D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 91));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 92));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 93));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, false);

  const char* prev = getenv("VT_FA2_DENSE");
  const std::string saved = prev == nullptr ? std::string() : std::string(prev);
  const bool had = prev != nullptr;
  (void)setenv("VT_FA2_DENSE", "0", 1);
  const auto off = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, false);
  if (had)
    (void)setenv("VT_FA2_DENSE", saved.c_str(), 1);
  else
    (void)unsetenv("VT_FA2_DENSE");

  CHECK(Mismatches(off, ref) == 0);  // the knob really routes back to AttentionDenseFlash
}
