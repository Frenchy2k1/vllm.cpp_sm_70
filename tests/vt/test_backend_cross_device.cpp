// Cross-device op-equality harness — the gap
// .agents/specs/backend-fanout-metal-vulkan-xpu.md § Gates calls out explicitly:
// "the seam for a second DeviceType exists but NOTHING exercises CPU-vs-device
// equality". This file is that harness. Newly authored (no upstream vLLM test
// mirrors it: vLLM's device-parameterized kernel tests compare against torch,
// which we do not have).
//
// CONTRACT — read before loosening anything here.
//   * The ORACLE is our own CPU backend, evaluated on the SAME host, from the
//     SAME binary, on the SAME inputs.
//   * The bar for REDUCING / arithmetic ops is NMSE <= 5e-4 — the already-ported
//     llama.cpp threshold (tests/vt/test_ops_quant_dot.cpp, itself ported
//     unwidened from llama.cpp test-quantize-fns:17-28 / test-backend-ops:4277).
//     It is NOT bit-exactness and must not be written as such: the CPU tier's
//     reproducibility comes from a FIXED SEQUENTIAL reduction order
//     (src/vt/cpu/cpu_quant_dot.cpp:22-28, deliberate) and no GPU cross-lane or
//     threadgroup tree reduction preserves it.
//   * The bar for PURE COPY / LAYOUT paths (Backend::Copy, Backend::Memset, a
//     same-dtype cast) IS bit-exactness — nothing is reassociated there, so
//     anything less would be hiding a bug.
//
// The harness runs against EVERY non-CPU backend that is registered in this
// build, so it is one file for Metal, and for CUDA/Vulkan/XPU when they arrive.
// A device that has not registered a given op is SKIPPED rather than failed:
// a partial backend is a supported, tested state (src/vt/ops.cpp:104-111).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// The already-ported bar. See the file header for why this is not memcmp.
constexpr double kNmseTol = 5e-4;

const char* DeviceName(DeviceType t) {
  switch (t) {
    case DeviceType::kCPU: return "CPU";
    case DeviceType::kCUDA: return "CUDA";
    case DeviceType::kMETAL: return "METAL";
    case DeviceType::kVULKAN: return "VULKAN";
    case DeviceType::kXPU: return "XPU";
    case DeviceType::kROCM: return "ROCM";
  }
  return "?";
}

// Normalized mean squared error, the same statistic
// tests/vt/test_ops_quant_dot.cpp gates on: sum((a-b)^2) / sum(a^2).
double Nmse(const std::vector<float>& ref, const std::vector<float>& got) {
  REQUIRE(ref.size() == got.size());
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(ref[i]) - static_cast<double>(got[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den == 0.0 ? num : num / den;
}

// Which non-CPU backends does THIS build actually have? GetBackend throws when a
// DeviceType is unregistered, which is the documented probe (no is-registered
// accessor exists on the vt:: seam).
std::vector<DeviceType> RegisteredDevices() {
  std::vector<DeviceType> out;
  for (DeviceType t : {DeviceType::kCUDA, DeviceType::kMETAL, DeviceType::kVULKAN,
                       DeviceType::kXPU, DeviceType::kROCM}) {
    try {
      (void)vt::GetBackend(t);
      out.push_back(t);
    } catch (const std::exception&) {
      // not built / no device present — nothing to compare against
    }
  }
  return out;
}

bool OpAvailable(vt::OpId op, DeviceType t) { return vt::OpRegistered(op, t); }

// A device-resident f32 buffer with host staging, so one body serves a unified
// backend (Metal, GB10) and a discrete one identically: every transfer goes
// through Backend::Copy rather than assuming the host can dereference the
// pointer.
class DevBuf {
 public:
  DevBuf(vt::Backend& b, Queue& q, size_t n) : b_(b), q_(q), n_(n) {
    ptr_ = b_.Alloc(n * sizeof(float));
  }
  ~DevBuf() { b_.Free(ptr_); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;

  void Upload(const std::vector<float>& src) {
    REQUIRE(src.size() == n_);
    b_.Copy(q_, ptr_, src.data(), n_ * sizeof(float));
  }
  std::vector<float> Download() {
    std::vector<float> out(n_);
    b_.Synchronize(q_);
    b_.Copy(q_, out.data(), ptr_, n_ * sizeof(float));
    b_.Synchronize(q_);
    return out;
  }
  void* ptr() const { return ptr_; }

 private:
  vt::Backend& b_;
  Queue& q_;
  size_t n_;
  void* ptr_ = nullptr;
};

std::vector<float> RandomVec(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

Tensor T2(void* p, Device d, int64_t r, int64_t c) {
  return Tensor::Contiguous(p, DType::kF32, d, {r, c});
}
Tensor T1(void* p, Device d, int64_t n) {
  return Tensor::Contiguous(p, DType::kF32, d, {n});
}
// Integer operands: embedding ids (i32 or i64, both accepted by vt::Embedding)
// and sampler token ids (i64 by contract).
Tensor TI32(void* p, Device d, int64_t n) {
  return Tensor::Contiguous(p, DType::kI32, d, {n});
}
Tensor TI64(void* p, Device d, int64_t n) {
  return Tensor::Contiguous(p, DType::kI64, d, {n});
}

}  // namespace

// ---------------------------------------------------------------------------
// Bit-exact tier: the pure byte paths. No arithmetic, so no tolerance.
// ---------------------------------------------------------------------------
TEST_CASE("device Copy/Memset are BIT-EXACT against the host bytes") {
  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();

    constexpr size_t kN = 977;  // deliberately not a round number
    std::vector<uint8_t> src(kN);
    for (size_t i = 0; i < kN; ++i) src[i] = static_cast<uint8_t>((i * 31u + 7u) & 0xFFu);

    void* p = dev.Alloc(kN);
    dev.Copy(q, p, src.data(), kN);
    dev.Synchronize(q);
    std::vector<uint8_t> back(kN, 0);
    dev.Copy(q, back.data(), p, kN);
    dev.Synchronize(q);
    CHECK(std::memcmp(src.data(), back.data(), kN) == 0);

    dev.Memset(q, p, 0x5A, kN);
    dev.Synchronize(q);
    dev.Copy(q, back.data(), p, kN);
    dev.Synchronize(q);
    std::vector<uint8_t> expect(kN, 0x5A);
    CHECK(std::memcmp(expect.data(), back.data(), kN) == 0);

    dev.Free(p);
    dev.DestroyQueue(q);
  }
}

// The bf16<->f32 casts are a pure ELEMENTWISE CODEC: no reduction, no
// reassociation, one rounding on store. So the bar here is BIT-EXACTNESS against
// the CPU reference, not NMSE — CastF32 (bf16 -> f32) is an exact widening, and
// CastBf16 (f32 -> bf16) must reproduce vt::F32ToBF16's round-to-nearest-EVEN
// (src/vt/dtype.cpp:224-233) exactly. A device that got the rounding "nearly
// right" would sail through an NMSE gate and still corrupt weights, so the
// rounding contract is checked with memcmp over every finite value, +-0, +-inf
// and 16 EXACT halfway ties.
//
// ONE DOCUMENTED CARVE-OUT: the NaN PAYLOAD. Measured on GB10 2026-07-22 with
// this very harness — for input 0x7FC00000 our CPU codec yields bf16 0x7FC0
// (`(u >> 16) | 0x0040`, i.e. truncate-and-quiet) while CUDA's
// `__float2bfloat16` yields 0x7FFF (canonical all-ones payload). Both are valid
// QUIET NaNs and IEEE-754 does not specify payload propagation across a
// narrowing conversion, so this is an architectural representation difference,
// NOT a rounding defect. It is carved out EXPLICITLY and narrowly: the payload
// bits are excluded, the quiet-NaN-ness is still asserted, and nothing about the
// rounding gate is weakened. (Metal, whose MSL codec is a literal transcription
// of vt::F32ToBF16 including its NaN branch, IS bit-exact here too — only CUDA
// differs, which is itself worth knowing.)
TEST_CASE("bf16<->f32 casts are BIT-EXACT against the CPU codec") {
  constexpr int64_t kRows = 8, kCols = 64;
  constexpr size_t kN = kRows * kCols;
  // Deliberately includes values that land ON a bf16 rounding tie, plus a NaN
  // and the infinities, so the tie-break and the NaN path are actually covered.
  std::vector<float> src = RandomVec(kN, 11, -8.0f, 8.0f);
  constexpr size_t kNanIdx = 0;  // the single payload carve-out; see the header
  src[kNanIdx] = std::numeric_limits<float>::quiet_NaN();
  src[1] = std::numeric_limits<float>::infinity();
  src[2] = -std::numeric_limits<float>::infinity();
  src[3] = 0.0f;
  src[4] = -0.0f;
  for (size_t i = 5; i < 21; ++i) {
    // Exact halfway cases for the bf16 mantissa: low 16 bits == 0x8000.
    uint32_t bits = (0x3F800000u + (static_cast<uint32_t>(i - 5) << 16)) | 0x8000u;
    std::memcpy(&src[i], &bits, sizeof(bits));
  }

  // CPU oracle: f32 -> bf16 -> f32.
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> cs = src;
  std::vector<uint16_t> ref_bf(kN);
  std::vector<float> ref_f32(kN);
  {
    Tensor tin = T2(cs.data(), cd, kRows, kCols);
    Tensor tbf = Tensor::Contiguous(ref_bf.data(), DType::kBF16, cd, {kRows, kCols});
    Tensor tf32 = T2(ref_f32.data(), cd, kRows, kCols);
    vt::CastBf16(cq, tbf, tin);
    vt::CastF32(cq, tf32, tbf);
  }
  cpu.DestroyQueue(cq);

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kCastBf16, dt) || !OpAvailable(vt::OpId::kCastF32, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    void* pin = dev.Alloc(kN * sizeof(float));
    void* pbf = dev.Alloc(kN * sizeof(uint16_t));
    void* pf32 = dev.Alloc(kN * sizeof(float));
    dev.Copy(q, pin, src.data(), kN * sizeof(float));
    Tensor tin = T2(pin, d, kRows, kCols);
    Tensor tbf = Tensor::Contiguous(pbf, DType::kBF16, d, {kRows, kCols});
    Tensor tf32 = T2(pf32, d, kRows, kCols);
    vt::CastBf16(q, tbf, tin);
    vt::CastF32(q, tf32, tbf);
    dev.Synchronize(q);

    std::vector<uint16_t> got_bf(kN);
    std::vector<float> got_f32(kN);
    dev.Copy(q, got_bf.data(), pbf, kN * sizeof(uint16_t));
    dev.Copy(q, got_f32.data(), pf32, kN * sizeof(float));
    dev.Synchronize(q);

    // Bit-exact everywhere EXCEPT the NaN payload slot. Compared as two
    // memcmp'd spans rather than a loop so a single differing bit anywhere in
    // the rounding-relevant data still fails hard.
    CHECK(std::memcmp(ref_bf.data(), got_bf.data(), kNanIdx * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(ref_bf.data() + kNanIdx + 1, got_bf.data() + kNanIdx + 1,
                      (kN - kNanIdx - 1) * sizeof(uint16_t)) == 0);
    CHECK(std::memcmp(ref_f32.data(), got_f32.data(), kNanIdx * sizeof(float)) == 0);
    CHECK(std::memcmp(ref_f32.data() + kNanIdx + 1, got_f32.data() + kNanIdx + 1,
                      (kN - kNanIdx - 1) * sizeof(float)) == 0);
    // The carve-out is on the PAYLOAD only: the value must still be a QUIET NaN
    // (bf16 exponent all ones + mantissa MSB set), and must still widen to a NaN.
    const uint16_t nan_bf = got_bf[kNanIdx];
    CHECK((nan_bf & 0x7F80u) == 0x7F80u);  // exponent all ones
    CHECK((nan_bf & 0x007Fu) != 0u);       // non-zero payload => NaN, not inf
    CHECK((nan_bf & 0x0040u) == 0x0040u);  // mantissa MSB set => QUIET
    CHECK(std::isnan(got_f32[kNanIdx]));

    dev.Free(pin);
    dev.Free(pbf);
    dev.Free(pf32);
    dev.DestroyQueue(q);
  }
}

// ---------------------------------------------------------------------------
// NMSE tier: everything with arithmetic. CPU is the oracle.
// ---------------------------------------------------------------------------
TEST_CASE("elementwise ops match the CPU oracle within NMSE <= 5e-4") {
  constexpr int64_t kRows = 17;
  constexpr int64_t kCols = 128;
  constexpr size_t kN = kRows * kCols;

  const std::vector<float> a = RandomVec(kN, 101);
  const std::vector<float> b = RandomVec(kN, 202);
  const std::vector<float> bias = RandomVec(kCols, 303);

  // --- CPU oracle, computed once through the very same vt:: entry points.
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ca = a, cb = b, cbias = bias;
  std::vector<float> ref_add(kN), ref_bias(kN), ref_relu(kN), ref_silu(kRows * kCols / 2);
  {
    Tensor ta = T2(ca.data(), cd, kRows, kCols);
    Tensor tb = T2(cb.data(), cd, kRows, kCols);
    Tensor tbias = T1(cbias.data(), cd, kCols);
    Tensor tadd = T2(ref_add.data(), cd, kRows, kCols);
    Tensor tbcast = T2(ref_bias.data(), cd, kRows, kCols);
    Tensor trelu = T2(ref_relu.data(), cd, kRows, kCols);
    Tensor tsilu = T2(ref_silu.data(), cd, kRows, kCols / 2);
    vt::Add(cq, tadd, ta, tb);
    vt::Add(cq, tbcast, ta, tbias);
    vt::Relu(cq, trelu, ta);
    vt::SiluAndMul(cq, tsilu, ta);
  }

  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf da(dev, q, kN), db(dev, q, kN), dbias(dev, q, kCols), dout(dev, q, kN);
    da.Upload(a);
    db.Upload(b);
    dbias.Upload(bias);
    Tensor ta = T2(da.ptr(), d, kRows, kCols);
    Tensor tb = T2(db.ptr(), d, kRows, kCols);
    Tensor tbias = T1(dbias.ptr(), d, kCols);

    if (OpAvailable(vt::OpId::kAdd, dt)) {
      Tensor to = T2(dout.ptr(), d, kRows, kCols);
      vt::Add(q, to, ta, tb);
      CHECK(Nmse(ref_add, dout.Download()) <= kNmseTol);
      // The rank-1 nn.Linear bias broadcast is a DIFFERENT indexing path.
      vt::Add(q, to, ta, tbias);
      CHECK(Nmse(ref_bias, dout.Download()) <= kNmseTol);
    }
    if (OpAvailable(vt::OpId::kRelu, dt)) {
      Tensor to = T2(dout.ptr(), d, kRows, kCols);
      vt::Relu(q, to, ta);
      CHECK(Nmse(ref_relu, dout.Download()) <= kNmseTol);
    }
    if (OpAvailable(vt::OpId::kSiluAndMul, dt)) {
      DevBuf dsilu(dev, q, kRows * kCols / 2);
      Tensor to = T2(dsilu.ptr(), d, kRows, kCols / 2);
      vt::SiluAndMul(q, to, ta);
      CHECK(Nmse(ref_silu, dsilu.Download()) <= kNmseTol);
    }
    dev.DestroyQueue(q);
  }
  cpu.DestroyQueue(cq);
}

TEST_CASE("RopeFromCache matches the CPU oracle within NMSE <= 5e-4, both styles") {
  // The APPLY half of vLLM's rotary split: the cos/sin table is built once (on
  // the portable tier, in double) and this rotates q and k with it.
  constexpr int64_t kTokens = 11, kHq = 4, kHk = 2, kD = 16, kRot = 16;
  constexpr int64_t kMaxPos = 64;

  const std::vector<float> q0 = RandomVec(kTokens * kHq * kD, 801);
  const std::vector<float> k0 = RandomVec(kTokens * kHk * kD, 802);
  const std::vector<float> cache = RandomVec(kMaxPos * kRot, 803, -1.0f, 1.0f);
  // Positions are NOT 0..n-1: a kernel that used the token index instead of the
  // position would pass on the identity mapping and fail here.
  std::vector<int32_t> pos(kTokens);
  for (int64_t i = 0; i < kTokens; ++i) pos[static_cast<size_t>(i)] = int32_t((i * 7 + 3) % kMaxPos);

  // NeoX rotates (pair, pair+half); GPT-J style rotates (2*pair, 2*pair+1). They
  // are different element pairings, so a kernel that hardcoded one passes half
  // the models and silently corrupts the other half.
  for (bool neox : {true, false}) {
    CAPTURE(neox);
    vt::RopeArgs args;
    args.rotary_dim = kRot;
    args.is_neox_style = neox;

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> refq = q0, refk = k0, ccache = cache;
    std::vector<int32_t> cpos = pos;
    {
      Tensor tq = Tensor::Contiguous(refq.data(), DType::kF32, cd, {kTokens, kHq, kD});
      Tensor tk = Tensor::Contiguous(refk.data(), DType::kF32, cd, {kTokens, kHk, kD});
      Tensor tc = Tensor::Contiguous(ccache.data(), DType::kF32, cd, {kMaxPos, kRot});
      Tensor tp = Tensor::Contiguous(cpos.data(), DType::kI32, cd, {kTokens});
      vt::RopeFromCache(cq, tq, &tk, tp, tc, args);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kRopeFromCache, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dq(dev, q, kTokens * kHq * kD), dk(dev, q, kTokens * kHk * kD),
          dc(dev, q, kMaxPos * kRot);
      dq.Upload(q0);   // rotation is IN PLACE, so re-upload the pristine input
      dk.Upload(k0);
      dc.Upload(cache);
      void* dpos = dev.Alloc(kTokens * sizeof(int32_t));
      dev.Copy(q, dpos, pos.data(), kTokens * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kTokens, kHq, kD});
      Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTokens, kHk, kD});
      Tensor tc = Tensor::Contiguous(dc.ptr(), DType::kF32, d, {kMaxPos, kRot});
      Tensor tp = Tensor::Contiguous(dpos, DType::kI32, d, {kTokens});
      vt::RopeFromCache(q, tq, &tk, tp, tc, args);
      dev.Synchronize(q);

      CHECK(Nmse(refq, dq.Download()) <= kNmseTol);
      CHECK(Nmse(refk, dk.Download()) <= kNmseTol);

      dev.Free(dpos);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("ReshapeAndCache scatters into the KV cache BIT-EXACTLY") {
  // Byte movement, so the bar is memcmp against the CPU oracle, not NMSE.
  constexpr int64_t kTokens = 9, kHk = 2, kD = 8, kBS = 4, kBlocks = 6;
  constexpr int64_t kElems = kHk * kD;          // one token's page payload
  constexpr int64_t kCacheN = kBlocks * kBS * kHk * kD;

  const std::vector<float> knew = RandomVec(kTokens * kElems, 701);
  const std::vector<float> vnew = RandomVec(kTokens * kElems, 702);
  // Pre-existing cache contents: the padded-token case must leave these INTACT,
  // so they cannot start as zeros or the check would pass vacuously.
  const std::vector<float> kc0 = RandomVec(kCacheN, 703);
  const std::vector<float> vc0 = RandomVec(kCacheN, 704);

  // Slots are deliberately SCATTERED and out of order, and two tokens carry -1.
  // Upstream pads the mapping and marks padded tokens negative (cpu_cache.cpp:60);
  // a kernel that clamped instead of skipping would corrupt a real page, and one
  // that read the i64 slot as unsigned would index astronomically out of range.
  const std::vector<int64_t> slots = {20, -1, 3, 11, -1, 0, 23, 7, 15};

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ck = knew, cv = vnew, ref_kc = kc0, ref_vc = vc0;
  std::vector<int64_t> cslots = slots;
  {
    Tensor tk = Tensor::Contiguous(ck.data(), DType::kF32, cd, {kTokens, kHk, kD});
    Tensor tv = Tensor::Contiguous(cv.data(), DType::kF32, cd, {kTokens, kHk, kD});
    Tensor tkc = Tensor::Contiguous(ref_kc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
    Tensor tvc = Tensor::Contiguous(ref_vc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
    Tensor tsm = Tensor::Contiguous(cslots.data(), DType::kI64, cd, {kTokens});
    vt::ReshapeAndCache(cq, tk, tv, tkc, tvc, tsm);
  }
  cpu.DestroyQueue(cq);

  for (DeviceType dt : RegisteredDevices()) {
    if (!OpAvailable(vt::OpId::kReshapeAndCache, dt)) continue;
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf dk(dev, q, kTokens * kElems), dv(dev, q, kTokens * kElems),
        dkc(dev, q, kCacheN), dvc(dev, q, kCacheN);
    dk.Upload(knew);
    dv.Upload(vnew);
    dkc.Upload(kc0);   // seeded, so an untouched page must survive
    dvc.Upload(vc0);
    void* dsm = dev.Alloc(kTokens * sizeof(int64_t));
    dev.Copy(q, dsm, slots.data(), kTokens * sizeof(int64_t));
    dev.Synchronize(q);

    Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTokens, kHk, kD});
    Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {kTokens, kHk, kD});
    Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
    Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
    Tensor tsm = Tensor::Contiguous(dsm, DType::kI64, d, {kTokens});
    vt::ReshapeAndCache(q, tk, tv, tkc, tvc, tsm);
    dev.Synchronize(q);

    const std::vector<float> got_kc = dkc.Download();
    const std::vector<float> got_vc = dvc.Download();
    CHECK(std::memcmp(ref_kc.data(), got_kc.data(), ref_kc.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(ref_vc.data(), got_vc.data(), ref_vc.size() * sizeof(float)) == 0);

    dev.Free(dsm);
    dev.DestroyQueue(q);
  }

  // --- Unbind flash layout: single (blocks,2,bs,H,D) allocation, K/V strided ---
  // Matches dense_attn::KvSlice — the layout the engine really feeds.
  {
    const int64_t within = kBS * kHk * kD;
    std::vector<float> combined(static_cast<size_t>(kBlocks * 2 * within));
    for (int64_t b = 0; b < kBlocks; ++b)
      for (int64_t e = 0; e < within; ++e) {
        combined[static_cast<size_t>((b * 2 + 0) * within + e)] =
            kc0[static_cast<size_t>(b * within + e)];
        combined[static_cast<size_t>((b * 2 + 1) * within + e)] =
            vc0[static_cast<size_t>(b * within + e)];
      }
    std::vector<float> ref_comb = combined;
    {
      vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
      Queue cq = cpu.CreateQueue();
      const Device cd{DeviceType::kCPU, 0};
      std::vector<float> ck = knew, cv = vnew, cslots_f;
      std::vector<int64_t> cslots = slots;
      Tensor tk = Tensor::Contiguous(ck.data(), DType::kF32, cd, {kTokens, kHk, kD});
      Tensor tv = Tensor::Contiguous(cv.data(), DType::kF32, cd, {kTokens, kHk, kD});
      Tensor tcomb =
          Tensor::Contiguous(ref_comb.data(), DType::kF32, cd, {kBlocks * 2 * within});
      auto slice = [&](int which) {
        Tensor t = tcomb;
        t.data = static_cast<char*>(t.data) +
                 static_cast<size_t>(which) * static_cast<size_t>(within) * sizeof(float);
        t.rank = 4;
        t.shape[0] = kBlocks;
        t.shape[1] = kBS;
        t.shape[2] = kHk;
        t.shape[3] = kD;
        t.stride[0] = 2 * within;
        t.stride[1] = kHk * kD;
        t.stride[2] = kD;
        t.stride[3] = 1;
        return t;
      };
      Tensor tsm = Tensor::Contiguous(cslots.data(), DType::kI64, cd, {kTokens});
      Tensor tkc = slice(0), tvc = slice(1);
      vt::ReshapeAndCache(cq, tk, tv, tkc, tvc, tsm);
      cpu.DestroyQueue(cq);
    }

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kReshapeAndCache, dt)) continue;
      CAPTURE(DeviceName(dt));
      CAPTURE("unbind");
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dk(dev, q, kTokens * kElems), dv(dev, q, kTokens * kElems),
          dcomb(dev, q, kBlocks * 2 * within);
      dk.Upload(knew);
      dv.Upload(vnew);
      dcomb.Upload(combined);
      void* dsm = dev.Alloc(kTokens * sizeof(int64_t));
      dev.Copy(q, dsm, slots.data(), kTokens * sizeof(int64_t));
      dev.Synchronize(q);
      Tensor tk = Tensor::Contiguous(dk.ptr(), DType::kF32, d, {kTokens, kHk, kD});
      Tensor tv = Tensor::Contiguous(dv.ptr(), DType::kF32, d, {kTokens, kHk, kD});
      Tensor tcomb =
          Tensor::Contiguous(dcomb.ptr(), DType::kF32, d, {kBlocks * 2 * within});
      auto slice = [&](int which) {
        Tensor t = tcomb;
        t.data = static_cast<char*>(t.data) +
                 static_cast<size_t>(which) * static_cast<size_t>(within) * sizeof(float);
        t.rank = 4;
        t.shape[0] = kBlocks;
        t.shape[1] = kBS;
        t.shape[2] = kHk;
        t.shape[3] = kD;
        t.stride[0] = 2 * within;
        t.stride[1] = kHk * kD;
        t.stride[2] = kD;
        t.stride[3] = 1;
        return t;
      };
      Tensor tsm = Tensor::Contiguous(dsm, DType::kI64, d, {kTokens});
      Tensor tkc = slice(0), tvc = slice(1);
      vt::ReshapeAndCache(q, tk, tv, tkc, tvc, tsm);
      dev.Synchronize(q);
      const std::vector<float> got = dcomb.Download();
      CHECK(std::memcmp(ref_comb.data(), got.data(), ref_comb.size() * sizeof(float)) == 0);
      dev.Free(dsm);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("paged attention matches the CPU oracle within NMSE <= 5e-4") {
  // GQA prefill: Hq=4 query heads over Hk=2 kv heads, head dim 8, block size 4,
  // one request of 37 tokens spanning 10 pages. Ragged on purpose -- 37 is not a
  // multiple of the block size, so the last page is partly occupied and a kernel
  // that walked whole pages would read past the sequence.
  constexpr int64_t kT = 37, kHq = 4, kHk = 2, kD = 8, kBS = 4;
  constexpr int64_t kBlocks = (kT + kBS - 1) / kBS;  // 10

  const std::vector<float> query = RandomVec(kT * kHq * kD, 601);
  const std::vector<float> kc = RandomVec(kBlocks * kBS * kHk * kD, 602);
  const std::vector<float> vc = RandomVec(kBlocks * kBS * kHk * kD, 603);
  std::vector<int32_t> block_table(kBlocks);
  // A NON-IDENTITY mapping, so a kernel that ignored the block table and indexed
  // the cache linearly would fail. Page j lives at cache block (kBlocks-1-j).
  for (int64_t b = 0; b < kBlocks; ++b) {
    block_table[static_cast<size_t>(b)] = static_cast<int32_t>(kBlocks - 1 - b);
  }
  const std::vector<int32_t> seq_lens = {static_cast<int32_t>(kT)};
  const std::vector<int32_t> qsl = {0, static_cast<int32_t>(kT)};

  // Three configurations, because they are different branches in the kernel and
  // a single causal case would leave two of them unexercised.
  struct Cfg { const char* name; bool causal; bool window; float softcap; };
  const Cfg cfgs[] = {
      {"causal", true, false, 0.0f},
      {"causal+softcap", true, false, 30.0f},   // cap * tanh(s / cap)
      {"sliding-window", true, true, 0.0f},     // window_left bounds jmin
  };

  for (const Cfg& cfg : cfgs) {
    CAPTURE(cfg.name);
    vt::PagedAttentionArgs args;
    args.scale = 0.353553f;
    args.causal = cfg.causal;
    args.logits_soft_cap = cfg.softcap;
    // Both bounds must be >= 0 (ops.cpp:2778). right = 0 is the causal
    // sliding window: no future keys, at most 8 past ones.
    if (cfg.window) args.window_size = vt::AttentionWindow{8, 0};

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cq_v = query, ckc = kc, cvc = vc, ref(kT * kHq * kD);
    std::vector<int32_t> cbt = block_table, csl = seq_lens, cqsl = qsl;
    {
      Tensor tq = Tensor::Contiguous(cq_v.data(), DType::kF32, cd, {kT, kHq, kD});
      Tensor tkc = Tensor::Contiguous(ckc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(cvc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(cbt.data(), DType::kI32, cd, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(csl.data(), DType::kI32, cd, {1});
      Tensor tqsl = Tensor::Contiguous(cqsl.data(), DType::kI32, cd, {2});
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kT, kHq, kD});
      vt::PagedAttention(cq, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kPagedAttention, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dq(dev, q, kT * kHq * kD), dkc(dev, q, kBlocks * kBS * kHk * kD),
          dvc(dev, q, kBlocks * kBS * kHk * kD), dout(dev, q, kT * kHq * kD);
      dq.Upload(query);
      dkc.Upload(kc);
      dvc.Upload(vc);
      void* dbt = dev.Alloc(kBlocks * sizeof(int32_t));
      void* dsl = dev.Alloc(sizeof(int32_t));
      void* dqsl = dev.Alloc(2 * sizeof(int32_t));
      dev.Copy(q, dbt, block_table.data(), kBlocks * sizeof(int32_t));
      dev.Copy(q, dsl, seq_lens.data(), sizeof(int32_t));
      dev.Copy(q, dqsl, qsl.data(), 2 * sizeof(int32_t));
      dev.Synchronize(q);

      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kT, kHq, kD});
      Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(dbt, DType::kI32, d, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(dsl, DType::kI32, d, {1});
      Tensor tqsl = Tensor::Contiguous(dqsl, DType::kI32, d, {2});
      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kT, kHq, kD});
      vt::PagedAttention(q, to, tq, tkc, tvc, tbt, tsl, tqsl, args);
      dev.Synchronize(q);

      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);

      dev.Free(dbt);
      dev.Free(dsl);
      dev.Free(dqsl);
      dev.DestroyQueue(q);
    }
  }

  // --- DECODE shape: one new query token over a filled cache (Tq=1, seq=kT).
  // This is the path multi-token generation hits after prefill; a prefill-only
  // test leaves it unexercised.
  {
    constexpr int64_t kTq = 1;
    const std::vector<float> q_dec = RandomVec(kTq * kHq * kD, 701);
    const std::vector<int32_t> sl_dec = {static_cast<int32_t>(kT)};
    const std::vector<int32_t> qsl_dec = {0, 1};
    vt::PagedAttentionArgs dargs;
    dargs.scale = 0.353553f;
    dargs.causal = true;

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cq_v = q_dec, ckc = kc, cvc = vc, ref(kTq * kHq * kD);
    std::vector<int32_t> cbt = block_table, csl = sl_dec, cqsl = qsl_dec;
    {
      Tensor tq = Tensor::Contiguous(cq_v.data(), DType::kF32, cd, {kTq, kHq, kD});
      Tensor tkc = Tensor::Contiguous(ckc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(cvc.data(), DType::kF32, cd, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(cbt.data(), DType::kI32, cd, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(csl.data(), DType::kI32, cd, {1});
      Tensor tqsl = Tensor::Contiguous(cqsl.data(), DType::kI32, cd, {2});
      Tensor to = Tensor::Contiguous(ref.data(), DType::kF32, cd, {kTq, kHq, kD});
      vt::PagedAttention(cq, to, tq, tkc, tvc, tbt, tsl, tqsl, dargs);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kPagedAttention, dt)) continue;
      CAPTURE(DeviceName(dt));
      CAPTURE("decode");
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};
      DevBuf dq(dev, q, kTq * kHq * kD), dkc(dev, q, kBlocks * kBS * kHk * kD),
          dvc(dev, q, kBlocks * kBS * kHk * kD), dout(dev, q, kTq * kHq * kD);
      dq.Upload(q_dec);
      dkc.Upload(kc);
      dvc.Upload(vc);
      void* dbt = dev.Alloc(kBlocks * sizeof(int32_t));
      void* dsl = dev.Alloc(sizeof(int32_t));
      void* dqsl = dev.Alloc(2 * sizeof(int32_t));
      dev.Copy(q, dbt, block_table.data(), kBlocks * sizeof(int32_t));
      dev.Copy(q, dsl, sl_dec.data(), sizeof(int32_t));
      dev.Copy(q, dqsl, qsl_dec.data(), 2 * sizeof(int32_t));
      dev.Synchronize(q);
      Tensor tq = Tensor::Contiguous(dq.ptr(), DType::kF32, d, {kTq, kHq, kD});
      Tensor tkc = Tensor::Contiguous(dkc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tvc = Tensor::Contiguous(dvc.ptr(), DType::kF32, d, {kBlocks, kBS, kHk, kD});
      Tensor tbt = Tensor::Contiguous(dbt, DType::kI32, d, {1, kBlocks});
      Tensor tsl = Tensor::Contiguous(dsl, DType::kI32, d, {1});
      Tensor tqsl = Tensor::Contiguous(dqsl, DType::kI32, d, {2});
      Tensor to = Tensor::Contiguous(dout.ptr(), DType::kF32, d, {kTq, kHq, kD});
      vt::PagedAttention(q, to, tq, tkc, tvc, tbt, tsl, tqsl, dargs);
      dev.Synchronize(q);
      CHECK(Nmse(ref, dout.Download()) <= kNmseTol);
      dev.Free(dbt);
      dev.Free(dsl);
      dev.Free(dqsl);
      dev.DestroyQueue(q);
    }
  }
}

TEST_CASE("Embedding gather and greedy argmax match the CPU oracle EXACTLY") {
  // Neither op is arithmetic, so neither gets the NMSE tier: a gather must move
  // the exact bytes, and an argmax must pick the exact index.
  constexpr int64_t kVocab = 61;
  constexpr int64_t kHidden = 40;
  constexpr int64_t kTokens = 7;

  const std::vector<float> table = RandomVec(kVocab * kHidden, 501);
  // Ids chosen to include 0 and the last row, and to REPEAT — a gather that
  // accidentally consumed ids positionally would pass on distinct ids.
  const std::vector<int32_t> ids32 = {0, 60, 13, 13, 1, 59, 0};
  std::vector<int64_t> ids64(ids32.begin(), ids32.end());

  // Logits with a DELIBERATE TIE: row 0 has its maximum twice, at columns 2 and
  // 5. The contract (cpu_sample.cpp:49, strict `>`) is that the FIRST wins, so a
  // kernel that used `>=` or a tie-indifferent tree reduction returns 5 and fails
  // here. That is a different token, not a rounding difference.
  constexpr int64_t kRows = 3;
  std::vector<float> logits(kRows * kVocab, 0.0f);
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kVocab; ++c) logits[r * kVocab + c] = -1.0f * float(c + 1);
  }
  logits[0 * kVocab + 2] = 9.0f;
  logits[0 * kVocab + 5] = 9.0f;   // tie with column 2; column 2 must win
  logits[1 * kVocab + 60] = 5.0f;  // last column
  logits[2 * kVocab + 0] = 5.0f;   // first column

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ctable = table, clogits = logits;
  std::vector<int32_t> cids = ids32;
  std::vector<float> ref_emb(kTokens * kHidden);
  std::vector<int64_t> ref_tok(kRows);
  {
    Tensor tt = T2(ctable.data(), cd, kVocab, kHidden);
    Tensor ti = TI32(cids.data(), cd, kTokens);
    Tensor to = T2(ref_emb.data(), cd, kTokens, kHidden);
    vt::Embedding(cq, to, tt, ti);
    Tensor tl = T2(clogits.data(), cd, kRows, kVocab);
    Tensor ttok = TI64(ref_tok.data(), cd, kRows);
    vt::GreedyArgmax(cq, ttok, tl);
  }
  REQUIRE(ref_tok[0] == 2);  // the oracle itself must honour the tie-break

  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    if (OpAvailable(vt::OpId::kEmbedding, dt)) {
      DevBuf dtable(dev, q, kVocab * kHidden), demb(dev, q, kTokens * kHidden);
      dtable.Upload(table);
      // i32 and i64 ids are DIFFERENT index paths, so both are exercised.
      for (bool wide : {false, true}) {
        CAPTURE(wide);
        void* dids = dev.Alloc(kTokens * (wide ? sizeof(int64_t) : sizeof(int32_t)));
        if (wide) {
          dev.Copy(q, dids, ids64.data(), kTokens * sizeof(int64_t));
        } else {
          dev.Copy(q, dids, ids32.data(), kTokens * sizeof(int32_t));
        }
        dev.Synchronize(q);
        Tensor tt = T2(dtable.ptr(), d, kVocab, kHidden);
        Tensor ti = wide ? TI64(static_cast<int64_t*>(dids), d, kTokens)
                         : TI32(static_cast<int32_t*>(dids), d, kTokens);
        Tensor to = T2(demb.ptr(), d, kTokens, kHidden);
        vt::Embedding(q, to, tt, ti);
        dev.Synchronize(q);
        const std::vector<float> got = demb.Download();
        // A gather moves bytes; equality is exact, not NMSE.
        CHECK(std::memcmp(ref_emb.data(), got.data(), ref_emb.size() * sizeof(float)) == 0);
        dev.Free(dids);
      }
    }

    if (OpAvailable(vt::OpId::kGreedyArgmax, dt)) {
      DevBuf dlog(dev, q, kRows * kVocab);
      dlog.Upload(logits);
      void* dtok = dev.Alloc(kRows * sizeof(int64_t));
      Tensor tl = T2(dlog.ptr(), d, kRows, kVocab);
      Tensor ttok = TI64(static_cast<int64_t*>(dtok), d, kRows);
      vt::GreedyArgmax(q, ttok, tl);
      dev.Synchronize(q);
      std::vector<int64_t> got(kRows);
      dev.Copy(q, got.data(), dtok, kRows * sizeof(int64_t));
      dev.Synchronize(q);
      for (int64_t r = 0; r < kRows; ++r) {
        CAPTURE(r);
        CHECK(got[r] == ref_tok[r]);
      }
      dev.Free(dtok);
    }
    dev.DestroyQueue(q);
  }
  cpu.DestroyQueue(cq);
}

TEST_CASE("GEMM matches the CPU oracle within NMSE <= 5e-4, both orientations") {
  // Shapes are deliberately RAGGED and not multiples of the workgroup size, so a
  // kernel that silently processed only whole tiles would fail rather than pass
  // on a friendly shape. K is the reduction length and gets the awkward value.
  constexpr int64_t kM = 13;
  constexpr int64_t kK = 37;
  constexpr int64_t kN = 9;

  const std::vector<float> a = RandomVec(kM * kK, 401);
  const std::vector<float> b = RandomVec(kK * kN, 402);   // [K,N] for Matmul
  const std::vector<float> bt = RandomVec(kN * kK, 403);  // [N,K] for MatmulBT

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ca = a, cb = b, cbt = bt;
  std::vector<float> ref_mm(kM * kN), ref_mmbt(kM * kN);
  {
    Tensor ta = T2(ca.data(), cd, kM, kK);
    Tensor tb = T2(cb.data(), cd, kK, kN);
    Tensor tbt = T2(cbt.data(), cd, kN, kK);
    Tensor tmm = T2(ref_mm.data(), cd, kM, kN);
    Tensor tmmbt = T2(ref_mmbt.data(), cd, kM, kN);
    vt::Matmul(cq, tmm, ta, tb);
    vt::MatmulBT(cq, tmmbt, ta, tbt);
  }

  for (DeviceType dt : RegisteredDevices()) {
    CAPTURE(DeviceName(dt));
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};

    DevBuf da(dev, q, kM * kK), dout(dev, q, kM * kN);
    da.Upload(a);
    Tensor ta = T2(da.ptr(), d, kM, kK);
    Tensor to = T2(dout.ptr(), d, kM, kN);

    if (OpAvailable(vt::OpId::kMatmul, dt)) {
      DevBuf db(dev, q, kK * kN);
      db.Upload(b);
      Tensor tb = T2(db.ptr(), d, kK, kN);
      vt::Matmul(q, to, ta, tb);
      CHECK(Nmse(ref_mm, dout.Download()) <= kNmseTol);
    }
    // MatmulBT is a DIFFERENT indexing path (the torch Linear [N,K] weight
    // layout), not a transpose of the same code, so it gets its own case.
    if (OpAvailable(vt::OpId::kMatmulBT, dt)) {
      DevBuf dbt(dev, q, kN * kK);
      dbt.Upload(bt);
      Tensor tbt = T2(dbt.ptr(), d, kN, kK);
      vt::MatmulBT(q, to, ta, tbt);
      CHECK(Nmse(ref_mmbt, dout.Download()) <= kNmseTol);
    }
    dev.DestroyQueue(q);
  }
  cpu.DestroyQueue(cq);
}

TEST_CASE("row-reducing ops match the CPU oracle within NMSE <= 5e-4") {
  // Widths chosen to exercise BOTH threadgroup regimes on a GPU: one that is a
  // clean power of two and one that is not (so the strided row loop has a
  // ragged tail), plus one narrower than a single 32-wide simd.
  for (int64_t cols : {128, 100, 17}) {
    CAPTURE(cols);
    const int64_t rows = 9;
    const size_t n = static_cast<size_t>(rows * cols);
    const std::vector<float> x = RandomVec(n, 404 + static_cast<uint32_t>(cols));
    const std::vector<float> w = RandomVec(static_cast<size_t>(cols), 505);
    const std::vector<float> bias = RandomVec(static_cast<size_t>(cols), 606);
    const std::vector<float> res0 = RandomVec(n, 707);

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cx = x, cw = w, cbias = bias;
    std::vector<float> ref_rms(n), ref_ln(n), ref_rms_res(n), ref_res_out = res0;
    {
      Tensor tx = T2(cx.data(), cd, rows, cols);
      Tensor tw = T1(cw.data(), cd, cols);
      Tensor tb = T1(cbias.data(), cd, cols);
      Tensor trms = T2(ref_rms.data(), cd, rows, cols);
      Tensor tln = T2(ref_ln.data(), cd, rows, cols);
      vt::RmsNorm(cq, trms, tx, tw, vt::RmsNormArgs{1e-6f, false}, nullptr);
      vt::LayerNorm(cq, tln, tx, &tw, &tb, vt::LayerNormArgs{1e-5f});
      // The in-place residual-stream form: residual is READ AND WRITTEN.
      Tensor tres = T2(ref_res_out.data(), cd, rows, cols);
      Tensor trr = T2(ref_rms_res.data(), cd, rows, cols);
      vt::RmsNorm(cq, trr, tx, tw, vt::RmsNormArgs{1e-6f, false}, &tres);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dx(dev, q, n), dw(dev, q, static_cast<size_t>(cols)),
          dbias(dev, q, static_cast<size_t>(cols)), dout(dev, q, n), dres(dev, q, n);
      dx.Upload(x);
      dw.Upload(w);
      dbias.Upload(bias);
      Tensor tx = T2(dx.ptr(), d, rows, cols);
      Tensor tw = T1(dw.ptr(), d, cols);
      Tensor tb = T1(dbias.ptr(), d, cols);
      Tensor to = T2(dout.ptr(), d, rows, cols);

      if (OpAvailable(vt::OpId::kRmsNorm, dt)) {
        vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, false}, nullptr);
        CHECK(Nmse(ref_rms, dout.Download()) <= kNmseTol);

        dres.Upload(res0);
        Tensor tres = T2(dres.ptr(), d, rows, cols);
        vt::RmsNorm(q, to, tx, tw, vt::RmsNormArgs{1e-6f, false}, &tres);
        CHECK(Nmse(ref_rms_res, dout.Download()) <= kNmseTol);
        // The residual stream itself is an OUTPUT and must agree too.
        CHECK(Nmse(ref_res_out, dres.Download()) <= kNmseTol);
      }
      if (OpAvailable(vt::OpId::kLayerNorm, dt)) {
        vt::LayerNorm(q, to, tx, &tw, &tb, vt::LayerNormArgs{1e-5f});
        CHECK(Nmse(ref_ln, dout.Download()) <= kNmseTol);
      }
      dev.DestroyQueue(q);
    }
  }
}

// The single kFusedChain registration is what earns a backend the whole portable
// fusion catalog, so it gets its own cross-device case. BOTH realization tiers
// are exercised on the same recipe (kFusedAddRmsNorm: add into the residual,
// then normalize it), because they are DIFFERENT code paths on a new backend:
//   Tier 0 (default) — the device-agnostic composite in src/vt/ops.cpp walks the
//     recipe dispatching each opcode to the backend's STANDALONE ops. A backend
//     inherits it for free; what is being proven is that its standalone ops
//     compose correctly, including the in-place residual fold.
//   Tier 1 (VT_FUSED_TIER=1) — the backend's OWN single-pass kFusedChain kernel.
// The CPU oracle is recomputed per tier so like is compared with like.
TEST_CASE("FusedChain matches the CPU oracle within NMSE <= 5e-4 (both tiers)") {
  const int64_t rows = 11, cols = 96;
  const size_t n = static_cast<size_t>(rows * cols);
  const std::vector<float> x = RandomVec(n, 808);
  const std::vector<float> w = RandomVec(static_cast<size_t>(cols), 909);
  const std::vector<float> res0 = RandomVec(n, 1010);
  const vt::FusedRecipe& recipe = vt::kFusedAddRmsNorm;

  // vt::FusedTier() re-reads the environment on every call (fused_recipe.h), so
  // the tier can be flipped within this process — the same mechanism the
  // existing tests/vt/test_ops_fused_chain.cpp parity cases rely on.
  const char* prev = std::getenv("VT_FUSED_TIER");
  const std::string saved = prev != nullptr ? std::string(prev) : std::string();
  const bool had_prev = prev != nullptr;

  for (int tier : {0, 1}) {
    CAPTURE(tier);
    setenv("VT_FUSED_TIER", tier == 0 ? "0" : "1", 1);
    // ASSERT the tier actually took effect rather than trusting the log: doctest
    // CAPTURE is lazily stringified, so a mis-set environment would silently
    // run the same path twice and still look like two-tier coverage.
    REQUIRE(vt::FusedTier() == tier);

    vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
    Queue cq = cpu.CreateQueue();
    const Device cd{DeviceType::kCPU, 0};
    std::vector<float> cx = x, cw = w, cres = res0, ref_out(n);
    {
      Tensor tx = T2(cx.data(), cd, rows, cols);
      Tensor tw = T1(cw.data(), cd, cols);
      Tensor tres = T2(cres.data(), cd, rows, cols);
      Tensor to = T2(ref_out.data(), cd, rows, cols);
      vt::FusedChain(cq, to, tx, tw, &tres, recipe, 1e-6f);
    }
    cpu.DestroyQueue(cq);

    for (DeviceType dt : RegisteredDevices()) {
      if (!OpAvailable(vt::OpId::kFusedChain, dt)) continue;
      CAPTURE(DeviceName(dt));
      vt::Backend& dev = vt::GetBackend(dt);
      Queue q = dev.CreateQueue();
      const Device d{dt, 0};

      DevBuf dx(dev, q, n), dw(dev, q, static_cast<size_t>(cols)), dres(dev, q, n),
          dout(dev, q, n);
      dx.Upload(x);
      dw.Upload(w);
      dres.Upload(res0);
      Tensor tx = T2(dx.ptr(), d, rows, cols);
      Tensor tw = T1(dw.ptr(), d, cols);
      Tensor tres = T2(dres.ptr(), d, rows, cols);
      Tensor to = T2(dout.ptr(), d, rows, cols);
      vt::FusedChain(q, to, tx, tw, &tres, recipe, 1e-6f);

      CHECK(Nmse(ref_out, dout.Download()) <= kNmseTol);
      CHECK(Nmse(cres, dres.Download()) <= kNmseTol);
      dev.DestroyQueue(q);
    }
  }

  if (had_prev) {
    setenv("VT_FUSED_TIER", saved.c_str(), 1);
  } else {
    unsetenv("VT_FUSED_TIER");
  }
}

// ---------------------------------------------------------------------------
// S5 PORTABLE REFERENCE TIER (accelerator-seam-audit.md, work row S5). The proof
// that op count is now a PERFORMANCE budget, not a CORRECTNESS gate: an op a
// UNIFIED-MEMORY device lacks a native kernel for falls back to the CPU reference
// and still returns the right answer, instead of throwing.
//
// Gated on Backend::UnifiedMemory() — THE safety invariant. On a discrete device
// the DevBuf pointer is a real device pointer a CPU kernel must never dereference,
// so the tier is neither installed nor exercised there (test_reference_tier.cpp
// asserts the refusal directly against a fake discrete backend). On this box's
// registered unified devices (Metal M4, GB10 CUDA/Vulkan) the pointer is
// host-accessible, so the fallback runs. On a plain CPU build there is no non-CPU
// device and the case is inert.
TEST_CASE("reference tier: an op with no native kernel matches the CPU oracle (unified only)") {
  constexpr int64_t kRows = 7, kCols = 48;
  constexpr size_t kN = kRows * kCols;
  const std::vector<float> in = RandomVec(kN, 1313, -4.0f, 4.0f);

  // CPU oracle through the same vt::Relu entry point (Relu is exact elementwise).
  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cq = cpu.CreateQueue();
  const Device cd{DeviceType::kCPU, 0};
  std::vector<float> ci = in, ref(kN);
  {
    Tensor ti = T2(ci.data(), cd, kRows, kCols);
    Tensor to = T2(ref.data(), cd, kRows, kCols);
    vt::Relu(cq, to, ti);
  }
  cpu.DestroyQueue(cq);

  for (DeviceType dt : RegisteredDevices()) {
    if (!vt::GetBackend(dt).UnifiedMemory()) continue;  // safety: unified only
    // Only meaningful where the device LACKS a native kernel for the op; where it
    // has one, the native path is already covered by the NMSE cases above.
    if (vt::OpRegistered(vt::OpId::kRelu, dt)) continue;
    CAPTURE(DeviceName(dt));

    const unsigned long long hits_before = vt::GetReferenceTierHits();
    vt::Backend& dev = vt::GetBackend(dt);
    Queue q = dev.CreateQueue();
    const Device d{dt, 0};
    DevBuf din(dev, q, kN), dout(dev, q, kN);
    din.Upload(in);
    Tensor ti = T2(din.ptr(), d, kRows, kCols);
    Tensor to = T2(dout.ptr(), d, kRows, kCols);
    vt::Relu(q, to, ti);  // no native kernel -> portable CPU fallback

    // Same host kernel, so bit-identical to the CPU oracle, not just close.
    const std::vector<float> got = dout.Download();
    CHECK(std::memcmp(ref.data(), got.data(), kN * sizeof(float)) == 0);
    // The fallback fired and it was not silent.
    CHECK(vt::GetReferenceTierHits() > hits_before);
    CHECK(std::string(vt::OpProviderNameAt(vt::OpId::kRelu, dt, 0)) ==
          vt::kReferenceProviderName);
    dev.DestroyQueue(q);
  }
}
