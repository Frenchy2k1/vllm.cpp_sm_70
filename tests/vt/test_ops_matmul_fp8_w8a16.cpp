// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Gate for vt::OpId::kMatmulFp8W8a16 — the keep-quant FP8 W8A16 dense GEMM
// (sm70 decode arm). Runs the op on BOTH the CPU queue (the dequant reference
// kernel, registered natively) and — when a CUDA backend is present — the GPU
// queue (LaunchSm70Fp8W8A16 through the op surface), comparing each arm to the
// same self-contained model-math oracle as the kernel's own case G:
//   out[m,n] = Σ_k bf16(act[m,k]) · f32(fp8(w[n,k])) · scale[n]
// over a f64 accumulation, at 2e-2 relative (fp8/bf16 tolerance). The CPU arm
// must agree much tighter (it is the reference); the CUDA arm is the decode
// kernel's non-vd round-trip measured against an independent oracle.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Device Cpu() { return Device{DeviceType::kCPU, 0}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
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

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first_bad = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    const float diff = std::fabs(got[i] - want[i]);
    if (diff > max_abs) max_abs = diff;
    if (!(diff <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
    CAPTURE(max_abs);
    if (std::getenv("VT_W8A16_DEBUG") != nullptr) {
      std::fprintf(stderr, "[w8a16dbg] bad=%zu first=%zu got=%g want=%g\n", bad,
                   first_bad, (double)got[first_bad], (double)want[first_bad]);
      for (size_t j = first_bad; j < first_bad + 8 && j < got.size(); ++j)
        std::fprintf(stderr, "  j=%zu got=%g want=%g ratio=%g\n", j, (double)got[j],
                     (double)want[j], want[j] != 0 ? (double)(got[j] / want[j]) : 0.0);
    }
  }
  CHECK(bad == 0);
}

// fp8-e4m3fn byte -> f32 (bit-matches vllm::F8E4M3ToF32 / the TU's
// Fp8ByteToF32; NaN excluded by construction below).
float Fp8ByteToF32(uint32_t byte) {
  const int sign = (byte & 0x80u) ? -1 : 1;
  const int exp = static_cast<int>((byte >> 3) & 0xFu);
  const int mant = static_cast<int>(byte & 0x7u);
  if (exp == 15 && mant == 7) return 0.0f;  // NaN
  if (exp == 0) return static_cast<float>(sign) * static_cast<float>(mant) / 512.0f;
  return static_cast<float>(sign) * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f,
                                               exp - 7);
}

// One (M, N, K) case across the CPU reference queue and (when the CUDA backend
// is registered) the GPU queue. The oracle is the case-G model math in f64:
// out[m,n] = Σ_k bf16(act[m,k]) · f32(fp8(w[n,k])) · scale[n] (the op doc).
static void RunCase(int M, int N, int K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(-2.0f, 2.0f);
  std::uniform_int_distribution<int> ub(0, 255);

  std::vector<uint16_t> act(static_cast<size_t>(M) * K);  // bf16 bits
  for (auto& v : act) v = vt::F32ToBF16(ux(rng));
  std::vector<uint8_t> w(static_cast<size_t>(N) * K);  // raw fp8-e4m3fn bytes
  for (auto& b : w) {
    int byte = ub(rng);
    if ((byte & 0x7F) == 0x7F) byte &= ~0x7;  // no NaN encodings (0x7F/0xFF)
    b = static_cast<uint8_t>(byte);
  }
  std::vector<float> scale(static_cast<size_t>(N));  // per-output-column f32
  for (auto& v : scale) v = 0.01f * static_cast<float>(1 + rng() % 25);

  std::vector<double> ref(static_cast<size_t>(M) * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += static_cast<double>(vt::BF16ToF32(act[static_cast<size_t>(m) * K + k])) *
               static_cast<double>(Fp8ByteToF32(w[static_cast<size_t>(n) * K + k]));
      // The op stores bf16 (out[m,n] = bf16(acc·scale[n])), so the oracle must
      // round through the same bf16 gate — otherwise the comparison would
      // count the store quantization itself as a difference.
      ref[static_cast<size_t>(m) * N + n] =
          static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(
              static_cast<float>(acc * scale[static_cast<size_t>(n)]))));
    }
  std::vector<float> want(ref.begin(), ref.end());

  // CPU leg — the registered reference arm must BE the oracle within fp32
  // accumulation tolerance (the op gate's truth for CPU builds).
  {
    Backend& b = vt::GetBackend(DeviceType::kCPU);
    QueueGuard g(b);
    Tensor ta = MakeTensor(act.data(), DType::kBF16, Cpu(), {M, K});
    Tensor tw = MakeTensor(w.data(), DType::kI8, Cpu(), {N, K});
    Tensor ts = MakeTensor(scale.data(), DType::kF32, Cpu(), {N});
    std::vector<uint16_t> out(static_cast<size_t>(M) * N);
    Tensor tout = MakeTensor(out.data(), DType::kBF16, Cpu(), {M, N});
    vt::MatmulFp8W8a16(g.q, tout, ta, tw, ts);
    std::vector<float> got(out.size());
    for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(out[i]);
    CheckClose(got, want, 1e-3f, 1e-3f);
  }

  if (!HasCuda()) {
    MESSAGE("no CUDA backend; device leg skipped");
    return;
  }

  // CUDA leg: the sm70 decode kernel through the vt:: op surface.
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  DeviceTensor dx(b, g.q, DType::kBF16, {M, K}, act.data());
  DeviceTensor dw(b, g.q, DType::kI8, {N, K}, w.data());
  DeviceTensor ds(b, g.q, DType::kF32, {N}, scale.data());
  DeviceTensor dout(b, g.q, DType::kBF16, {M, N});
  vt::MatmulFp8W8a16(g.q, dout.tensor(), dx.tensor(), dw.tensor(), ds.tensor());
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(g.q, out.data());
  std::vector<float> got(out.size());
  for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(out[i]);
  CheckClose(got, want, 2e-2f, 2e-2f);
}

TEST_CASE("matmul_fp8_w8a16 op surface: registered on CPU (+CUDA when present)") {
  CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8W8a16, DeviceType::kCPU));
  CHECK(std::string(vt::OpName(vt::OpId::kMatmulFp8W8a16)) == "MatmulFp8W8a16");
  if (HasCuda()) {
    CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8W8a16, DeviceType::kCUDA));
  }
}

TEST_CASE("matmul_fp8_w8a16 decode band parity (M<=16, case-G oracle, 2e-2)") {
  // M=1 (the pure decode step) and M=8/16 — the QPN band — at N%32==0 and
  // K%128==0 (the kernel's validated 8-warp band), plus a K%128!=0 shape that
  // must route through the naive fallback (the 4-warp band is not gated).
  RunCase(1, 32, 1024, 0x5EED01);
  RunCase(8, 64, 1024, 0x5EED02);
  RunCase(16, 64, 1024, 0x5EED0A);
  RunCase(16, 64, 512, 0x5EED03);
}

TEST_CASE("matmul_fp8_w8a16 prefill fallback parity (M>16, same oracle)") {
  // The decode band is M<=16; a larger M must still be serviced (the naive
  // fallback in the CUDA arm) and agree with the same oracle.
  RunCase(33, 64, 1024, 0x5EED04);
}

}  // namespace