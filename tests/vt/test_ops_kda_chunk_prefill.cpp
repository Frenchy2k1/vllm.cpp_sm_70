// KDA CHUNK-PREFILL (vt::KdaChunkPrefill) — UNIT GATE (RED-first).
//
// The chunked forward of the SAME per-K-channel gated-delta linear attention as
// vt::KdaGatedDeltaRule (the #104 recurrence), computed in BT=64 chunks through the
// vendored FLA Triton-AOT cubins (kda_gate_cumsum -> kkt(inter+intra) -> solve_tril
// -> recompute_w_u -> chunk_delta_h -> chunk_gla_o), mirroring vLLM's prefill path
// (kimi_gdn_linear_attn.py:141 chunk_kda_with_fused_gate; decode stays recurrent).
//
// ─── WHAT THIS GATES ───────────────────────────────────────────────────────
// (a) CPU: the op fuses the gate (g = -exp(a_log)*softplus(g_raw+dt_bias)) then runs
//     the proven recurrence, so it MUST equal vt::KdaGatedDeltaRule fed the same gate
//     — bit-for-bit (both the gate + recurrence). Ties the wrapper to a proven ref.
// (b) CUDA: the chunk op (6 cubins, all-bf16) MUST equal the recurrence up to the
//     chunked-vs-recurrent REDUCTION ORDER + bf16 rounding (an rtol band, NOT bit-
//     exact — §17.5). This is the ONLY place the KDA chunk kernel chain is exercised
//     numerically off the full 48.9B model.
// RED-first: a reference built with a PERTURBED gate blows past the band, proving the
// band has teeth (it would catch a mis-wired gate/kernel).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::GdnArgs;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

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

std::vector<float> RandF32(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// L2-normalize each [D] row (as the caller does before KDA q/k).
std::vector<float> L2NormRows(std::vector<float> v, size_t rows, size_t D) {
  for (size_t r = 0; r < rows; ++r) {
    double s = 0.0;
    for (size_t i = 0; i < D; ++i) s += static_cast<double>(v[r * D + i]) * v[r * D + i];
    const float inv = static_cast<float>(1.0 / std::sqrt(s + 1e-12));
    for (size_t i = 0; i < D; ++i) v[r * D + i] *= inv;
  }
  return v;
}

// The per-token per-channel log-decay FLA's kda_gate_cumsum applies pre-cumsum:
//   g_dec = -exp(a_log[h]) * softplus(g_raw + dt_bias), softplus(beta=1).
std::vector<float> HostGate(const std::vector<float>& g_raw, const std::vector<float>& a_log,
                            const std::vector<float>& dt_bias, int64_t T, int64_t H, int64_t D) {
  std::vector<float> g(static_cast<size_t>(T) * H * D);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h) {
      const float a = -std::exp(a_log[static_cast<size_t>(h)]);
      for (int64_t d = 0; d < D; ++d) {
        float x = g_raw[static_cast<size_t>((t * H + h) * D + d)];
        if (!dt_bias.empty()) x += dt_bias[static_cast<size_t>(h * D + d)];
        const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
        g[static_cast<size_t>((t * H + h) * D + d)] = a * sp;
      }
    }
  return g;
}

}  // namespace

// ── (a) CPU: chunk op (gate + recurrence) == recurrence fed the same gate ────────
TEST_CASE("kda chunk-prefill: CPU wrapper equals the recurrence fed the fused gate") {
  const int64_t T = 10, H = 4, D = 16;  // tiny (CPU ref path; geometry-agnostic)
  const int64_t proj = H * D;
  auto q = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 1), static_cast<size_t>(T) * H, D);
  auto k = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 2), static_cast<size_t>(T) * H, D);
  auto v = RandF32(static_cast<size_t>(T) * proj, 3);
  auto beta = RandF32(static_cast<size_t>(T) * H, 4, 0.1f, 0.9f);
  auto g_raw = RandF32(static_cast<size_t>(T) * proj, 5, -2.0f, 2.0f);
  auto a_log = RandF32(static_cast<size_t>(H), 6, -1.5f, 0.5f);
  auto dt_bias = RandF32(static_cast<size_t>(proj), 7, -0.5f, 0.5f);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const auto g_dec = HostGate(g_raw, a_log, dt_bias, T, H, D);

  Queue cq = CpuQ();
  std::vector<float> out_ref(static_cast<size_t>(T) * proj, 0.0f), st_ref(H * D * D, 0.0f);
  std::vector<float> out_chk(static_cast<size_t>(T) * proj, 0.0f), st_chk(H * D * D, 0.0f);
  Tensor tqsl = MakeT(const_cast<int32_t*>(qsl), DType::kI32, Cpu(), {2});
  Tensor tq = MakeT(q.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tk = MakeT(k.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tv = MakeT(v.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tb = MakeT(beta.data(), DType::kF32, Cpu(), {T, H});

  Tensor to_ref = MakeT(out_ref.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tg_dec = MakeT(const_cast<float*>(g_dec.data()), DType::kF32, Cpu(), {T, H, D});
  Tensor ts_ref = MakeT(st_ref.data(), DType::kF32, Cpu(), {1, H, D, D});
  vt::KdaGatedDeltaRule(cq, to_ref, tq, tk, tv, tg_dec, tb, ts_ref, tqsl, GdnArgs{scale});

  Tensor to_chk = MakeT(out_chk.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor tg_raw = MakeT(g_raw.data(), DType::kF32, Cpu(), {T, H, D});
  Tensor talog = MakeT(a_log.data(), DType::kF32, Cpu(), {H});
  Tensor tdtb = MakeT(dt_bias.data(), DType::kF32, Cpu(), {proj});
  Tensor ts_chk = MakeT(st_chk.data(), DType::kF32, Cpu(), {1, H, D, D});
  vt::KdaChunkPrefill(cq, to_chk, tq, tk, tv, tg_raw, tb, talog, tdtb, ts_chk, tqsl, GdnArgs{scale});

  size_t bad = 0;
  for (size_t i = 0; i < out_ref.size(); ++i)
    if (std::fabs(out_chk[i] - out_ref[i]) > 1e-5f) ++bad;
  CHECK(bad == 0);  // CPU: identical gate + identical recurrence => bit-for-bit
}

// ── (b) CUDA: the chunk cubins ≈ the recurrence (band); RED with a wrong gate ────
#ifdef VLLM_CPP_CUDA
namespace {
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}
struct QGuard {
  Backend& b;
  Queue q;
  explicit QGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QGuard() { b.DestroyQueue(q); }
  QGuard(const QGuard&) = delete;
  QGuard& operator=(const QGuard&) = delete;
};
class DTensor {
 public:
  DTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
          const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~DTensor() { b_.Free(p_); }
  DTensor(const DTensor&) = delete;
  DTensor& operator=(const DTensor&) = delete;
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

}  // namespace

TEST_CASE("kda chunk-prefill: CUDA chunk cubins match the recurrence within a band") {
  if (!HasCuda()) return;
  // The pinned Kimi KDA geometry (the cubins only fire here): H=32, K=V=128, BT=64.
  const int64_t T = 130, H = 32, D = 128;  // 3 chunks (64,64,2): exercises the partial tail
  const int64_t proj = H * D;
  auto q = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 31), static_cast<size_t>(T) * H, D);
  auto k = L2NormRows(RandF32(static_cast<size_t>(T) * proj, 32), static_cast<size_t>(T) * H, D);
  auto v = RandF32(static_cast<size_t>(T) * proj, 33);
  auto beta = RandF32(static_cast<size_t>(T) * H, 34, 0.1f, 0.9f);
  // MILD decay so the STATE accumulates across the 130 tokens / 3 chunks (exercising
  // chunk_delta_h's inter-chunk recurrence, not just the current-token term) AND so the
  // gate materially affects the output (making the RED case discriminate). Aggressive
  // decay (state fully forgotten each step) collapses the output to a gate-independent
  // current-token term. a_log in [-5,-3] => exp(a_log) tiny; small g_raw => small softplus.
  auto g_raw = RandF32(static_cast<size_t>(T) * proj, 35, -1.0f, 1.0f);
  auto a_log = RandF32(static_cast<size_t>(H), 36, -5.0f, -3.0f);
  auto dt_bias = RandF32(static_cast<size_t>(proj), 37, -0.2f, 0.2f);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const float scale = std::pow(static_cast<float>(D), -0.5f);

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QGuard g(gpu);
  std::vector<float> st_zero(static_cast<size_t>(H) * D * D, 0.0f);

  // Recurrence reference (fed the correct fused gate) — the truth up to reduction order.
  const auto g_dec = HostGate(g_raw, a_log, dt_bias, T, H, D);
  std::vector<float> out_ref(static_cast<size_t>(T) * proj, 0.0f);
  {
    DTensor to(gpu, g.q, DType::kF32, {T, H, D});
    DTensor tq(gpu, g.q, DType::kF32, {T, H, D}, q.data());
    DTensor tk(gpu, g.q, DType::kF32, {T, H, D}, k.data());
    DTensor tv(gpu, g.q, DType::kF32, {T, H, D}, v.data());
    DTensor tg(gpu, g.q, DType::kF32, {T, H, D}, g_dec.data());
    DTensor tb(gpu, g.q, DType::kF32, {T, H}, beta.data());
    DTensor ts(gpu, g.q, DType::kF32, {1, H, D, D}, st_zero.data());
    DTensor tqsl(gpu, g.q, DType::kI32, {2}, qsl);
    vt::KdaGatedDeltaRule(g.q, to.tensor(), tq.tensor(), tk.tensor(), tv.tensor(), tg.tensor(),
                          tb.tensor(), ts.tensor(), tqsl.tensor(), GdnArgs{scale});
    to.Download(g.q, out_ref.data());
  }

  // Chunk op (RAW gate on device): the 6 cubins.
  std::vector<float> out_chk(static_cast<size_t>(T) * proj, 0.0f);
  {
    DTensor to(gpu, g.q, DType::kF32, {T, H, D});
    DTensor tq(gpu, g.q, DType::kF32, {T, H, D}, q.data());
    DTensor tk(gpu, g.q, DType::kF32, {T, H, D}, k.data());
    DTensor tv(gpu, g.q, DType::kF32, {T, H, D}, v.data());
    DTensor tgr(gpu, g.q, DType::kF32, {T, H, D}, g_raw.data());
    DTensor tb(gpu, g.q, DType::kF32, {T, H}, beta.data());
    DTensor talog(gpu, g.q, DType::kF32, {H}, a_log.data());
    DTensor tdtb(gpu, g.q, DType::kF32, {proj}, dt_bias.data());
    DTensor ts(gpu, g.q, DType::kF32, {1, H, D, D}, st_zero.data());
    DTensor tqsl(gpu, g.q, DType::kI32, {2}, qsl);
    vt::KdaChunkPrefill(g.q, to.tensor(), tq.tensor(), tk.tensor(), tv.tensor(), tgr.tensor(),
                        tb.tensor(), talog.tensor(), tdtb.tensor(), ts.tensor(), tqsl.tensor(),
                        GdnArgs{scale});
    to.Download(g.q, out_chk.data());
  }

  // Diagnostics (always printed) + returns mean abs error. The outputs are small
  // (mild decay), so mean-abs — not a relative band — is the stable discriminator:
  // near-zero elements make relative error explode, and a fixed atol either swallows
  // or floods everything. The correct/wrong separation is ~30x in mean abs.
  auto diag = [](const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    double maxe = 0, sume = 0, maxrel = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double e = std::fabs(a[i] - b[i]);
      maxe = std::max(maxe, e);
      sume += e;
      maxrel = std::max(maxrel, e / (std::fabs(b[i]) + 1e-6));
    }
    const double mean = sume / static_cast<double>(a.size());
    std::fprintf(stderr, "[kda-chunk-test] %s: max_abs=%.4g mean_abs=%.4g max_rel=%.4g\n", tag, maxe,
                 mean, maxrel);
    return mean;
  };

  // Chunked-vs-recurrent: all-bf16 chunk path + different reduction order over the
  // accumulated 3-chunk state -> close but NOT bit-exact (§17.5). The RED case proves it.
  const double mean_ok = diag("chunk-vs-recurrent", out_chk, out_ref);
  CHECK(mean_ok < 3e-4);  // measured ~5e-5; the bf16 chunk path tracks the f32 recurrence

  // RED: a reference built with a PERTURBED gate (a_log + 1.0, ~e× the decay) must move
  // the output well past the correct-vs-chunk gap, proving the check has teeth (it is not
  // vacuously loose) — a mis-wired gate / kernel would land here.
  auto a_log_bad = a_log;
  for (auto& x : a_log_bad) x += 1.0f;
  const auto g_dec_bad = HostGate(g_raw, a_log_bad, dt_bias, T, H, D);
  std::vector<float> out_bad(static_cast<size_t>(T) * proj, 0.0f);
  {
    DTensor to(gpu, g.q, DType::kF32, {T, H, D});
    DTensor tq(gpu, g.q, DType::kF32, {T, H, D}, q.data());
    DTensor tk(gpu, g.q, DType::kF32, {T, H, D}, k.data());
    DTensor tv(gpu, g.q, DType::kF32, {T, H, D}, v.data());
    DTensor tg(gpu, g.q, DType::kF32, {T, H, D}, g_dec_bad.data());
    DTensor tb(gpu, g.q, DType::kF32, {T, H}, beta.data());
    DTensor ts(gpu, g.q, DType::kF32, {1, H, D, D}, st_zero.data());
    DTensor tqsl(gpu, g.q, DType::kI32, {2}, qsl);
    vt::KdaGatedDeltaRule(g.q, to.tensor(), tq.tensor(), tk.tensor(), tv.tensor(), tg.tensor(),
                          tb.tensor(), ts.tensor(), tqsl.tensor(), GdnArgs{scale});
    to.Download(g.q, out_bad.data());
  }
  const double mean_bad = diag("chunk-vs-WRONG-gate", out_chk, out_bad);
  CHECK(mean_bad > 1e-3);  // a wrong gate diverges >>3e-4 (measured ~1.7e-3 at +0.5; more at +1.0)
  CHECK(mean_bad > 3.0 * mean_ok);  // and is clearly separated from the correct-gate match
}
#endif  // VLLM_CPP_CUDA
