// Mamba2 silu-gated GROUP RMS norm (vt::RmsNormGatedGroup) — UNIT GATE.
// .agents/specs/mamba2-ssd.md W1, issue #496.
//
// Ported from tests/kernels/mamba/test_mamba_mixer2.py @ pin 555967922
// (vLLM 0.26.0.dev0), preserving its shapes and tolerances: batch_size = 8,
// seq_len = 128, (hidden_size, n_groups) in {(64,1), (64,2), (64,4)},
// atol 5e-3 / rtol 1e-3 (:20-33, :131-137). The op under test mirrors
// `Mixer2RMSNormGated.forward_native`
// (vllm/model_executor/layers/mamba/mamba_mixer2.py:100-149).
//
// HARNESS ADAPTATION (documented, per porting.md). The upstream test is
// `@multi_gpu_test(num_gpus=2)`: it checks that the TP-sharded norm agrees with
// the unsharded one. W1 lands `tp_world_size == 1` only (mamba2-ssd.md §2, §7),
// so what is portable is the unsharded reference the upstream test compares
// AGAINST — `mixer_single_gpu`, built with the TP world size mocked to 1
// (:105-118) — and its shapes and tolerance. That reference is restated here in
// `double`. The TP arm is REFUSED by the op, and a test pins that refusal names
// `extra_groups_for_head_shards`. Upstream's dtype is float16; the vt `out`
// contract is f32/bf16 (`IsOutFloat`, src/vt/ops.cpp), so the reduced-precision
// arm is bf16 and an F32 ARM IS SWEPT ALONGSIDE IT
// ([[bf16-store-absorbs-reduction-order-defects]]).
//
// ─── WHY THIS IS A SIBLING OP, NOT A PARAMETER ────────────────────────────────
// vt::RmsNormGated (landed, GDN/KDA) gates with SIGMOID-or-silu over the WHOLE
// row. This one always SILU-gates and reduces the variance over
// `group_size = hidden / n_groups` slices. Both the activation and the reduction
// extent differ, so it is a sibling (mamba2-ssd.md §0.3). The tests below pin
// BOTH differences: a group-count sensitivity check proves the reduction really
// is per-group (n_groups > 1 must NOT equal n_groups == 1), and the reference
// uses silu, never sigmoid.
//
// TOLERANCES: explicit `torch.testing.assert_close` arithmetic, never
// `doctest::Approx` — its `scale` defaults to 1.0 and floors every comparison at
// ~1.19e-5 absolute ([[doctest-approx-scale-term-floor]]).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormGatedGroupArgs;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
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

void ExpectClose(const char* what, const std::vector<float>& got,
                 const std::vector<double>& want, double atol, double rtol) {
  REQUIRE(got.size() == want.size());
  REQUIRE(!got.empty());
  double worst_slack = -std::numeric_limits<double>::infinity();
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(got[i]);
    const double w = want[i];
    const double slack = std::abs(g - w) - (atol + rtol * std::abs(w));
    if (!std::isfinite(g) || slack > worst_slack) {
      worst_slack = slack;
      worst_i = i;
      if (!std::isfinite(g)) break;
    }
  }
  INFO(what << ": worst element [" << worst_i << "] got=" << got[worst_i]
            << " want=" << want[worst_i] << " |diff|="
            << std::abs(static_cast<double>(got[worst_i]) - want[worst_i])
            << " budget=" << (atol + rtol * std::abs(want[worst_i])));
  CHECK(std::isfinite(static_cast<double>(got[worst_i])));
  CHECK(worst_slack <= 0.0);
}

std::vector<uint8_t> Pack(const std::vector<float>& src, DType dt) {
  std::vector<uint8_t> raw(src.size() * vt::SizeOf(dt));
  for (size_t i = 0; i < src.size(); ++i) {
    if (dt == DType::kF32) {
      std::memcpy(raw.data() + i * 4, &src[i], 4);
    } else {
      const uint16_t v = vt::F32ToBF16(src[i]);
      std::memcpy(raw.data() + i * 2, &v, 2);
    }
  }
  return raw;
}

std::vector<float> Unpack(const std::vector<uint8_t>& raw, size_t n, DType dt) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    if (dt == DType::kF32) {
      std::memcpy(&out[i], raw.data() + i * 4, 4);
    } else {
      uint16_t v;
      std::memcpy(&v, raw.data() + i * 2, 2);
      out[i] = vt::BF16ToF32(v);
    }
  }
  return out;
}

// ─── the `double` reference ──────────────────────────────────────────────────
// `Mixer2RMSNormGated.forward_native` (mamba_mixer2.py:100-149) restated:
//   v      = x * silu(f32(gate))                                        (:114)
//   grouped variance over group_size = hidden / n_groups                (:136-140)
//   out    = weight * (v * rsqrt(var + eps))                            (:149)
// When use_rms_norm is False the whole norm is skipped and the gated value is
// returned as-is (:115-116) — that is the `weight == nullptr` arm of the op.
std::vector<double> GatedGroupNormRef(const std::vector<float>& x,
                                      const std::vector<float>& gate,
                                      const std::vector<float>* weight, int64_t rows,
                                      int64_t hidden, int64_t n_groups, double eps) {
  const int64_t group_size = hidden / n_groups;
  std::vector<double> out(static_cast<size_t>(rows * hidden), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    std::vector<double> v(static_cast<size_t>(hidden));
    for (int64_t j = 0; j < hidden; ++j) {
      const double zv = gate[static_cast<size_t>(r * hidden + j)];
      const double silu = zv / (1.0 + std::exp(-zv));
      v[static_cast<size_t>(j)] = static_cast<double>(x[static_cast<size_t>(r * hidden + j)]) * silu;
    }
    if (weight == nullptr) {
      for (int64_t j = 0; j < hidden; ++j)
        out[static_cast<size_t>(r * hidden + j)] = v[static_cast<size_t>(j)];
      continue;
    }
    for (int64_t g = 0; g < n_groups; ++g) {
      double ss = 0.0;
      for (int64_t j = 0; j < group_size; ++j) {
        const double t = v[static_cast<size_t>(g * group_size + j)];
        ss += t * t;
      }
      const double inv = 1.0 / std::sqrt(ss / static_cast<double>(group_size) + eps);
      for (int64_t j = 0; j < group_size; ++j) {
        const int64_t idx = g * group_size + j;
        out[static_cast<size_t>(r * hidden + idx)] =
            static_cast<double>((*weight)[static_cast<size_t>(idx)]) *
            (v[static_cast<size_t>(idx)] * inv);
      }
    }
  }
  return out;
}

struct NormInputs {
  std::vector<float> x, gate, weight;
};

NormInputs GenerateNorm(int64_t rows, int64_t hidden, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  NormInputs in;
  in.x.resize(static_cast<size_t>(rows * hidden));
  for (auto& v : in.x) v = nd(rng);
  in.gate.resize(static_cast<size_t>(rows * hidden));
  for (auto& v : in.gate) v = nd(rng);
  in.weight.resize(static_cast<size_t>(hidden));  // `torch.rand((hidden_size,))` (:91)
  for (auto& v : in.weight) v = ud(rng);
  return in;
}

std::vector<float> RunNorm(const NormInputs& in, const std::vector<int64_t>& shape,
                           int64_t n_groups, float eps, DType dt, bool use_rms_norm,
                           int64_t tp_world_size = 1) {
  Queue q = CpuQ();
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  std::vector<uint8_t> xb = Pack(in.x, dt);
  std::vector<uint8_t> gb = Pack(in.gate, dt);
  std::vector<uint8_t> ob(n * vt::SizeOf(dt), 0);
  std::vector<float> w = in.weight;

  Tensor xt = MakeT(xb.data(), dt, shape);
  Tensor gt = MakeT(gb.data(), dt, shape);
  Tensor ot = MakeT(ob.data(), dt, shape);
  Tensor wt = MakeT(w.data(), DType::kF32, {shape.back()});

  RmsNormGatedGroupArgs args;
  args.eps = eps;
  args.n_groups = n_groups;
  args.tp_world_size = tp_world_size;
  vt::RmsNormGatedGroup(q, ot, xt, gt, use_rms_norm ? &wt : nullptr, args);
  return Unpack(ob, n, dt);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// (1) The unsharded gated group norm, at upstream's shapes and tolerance.
// batch_size = 8, seq_len = 128, hidden_size = 64, n_groups in {1, 2, 4},
// atol 5e-3 / rtol 1e-3 (test_mamba_mixer2.py:21-33, :131-137).
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 gated group norm matches forward_native") {
  const int64_t batch = 8, seq = 128, hidden = 64;
  const int64_t rows = batch * seq;
  const float eps = 1e-6f;
  const NormInputs in = GenerateNorm(rows, hidden, 0x9A17Eu);
  for (int64_t n_groups : {1, 2, 4}) {
    const std::vector<double> ref =
        GatedGroupNormRef(in.x, in.gate, &in.weight, rows, hidden, n_groups, eps);
    INFO("n_groups=" << n_groups);
    // f32 arm — the one a bf16 store cannot hide a reduction-order defect in.
    ExpectClose("out f32", RunNorm(in, {rows, hidden}, n_groups, eps, DType::kF32, true), ref,
                5e-3, 1e-3);
    // reduced-precision arm (upstream runs float16; the vt out contract is
    // f32/bf16, so this is bf16 at a bf16-appropriate tolerance).
    ExpectClose("out bf16", RunNorm(in, {rows, hidden}, n_groups, eps, DType::kBF16, true), ref,
                5e-2, 1e-2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) The reduction really IS per group. With the same inputs, n_groups > 1 must
// NOT reproduce n_groups == 1: a kernel that reduced over the whole row and
// ignored n_groups would pass (1) for n_groups == 1 and silently for the rest.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 gated group norm reduces per group, not per row") {
  const int64_t rows = 16, hidden = 64;
  const float eps = 1e-6f;
  const NormInputs in = GenerateNorm(rows, hidden, 0x2211u);
  const std::vector<float> g1 = RunNorm(in, {rows, hidden}, 1, eps, DType::kF32, true);
  for (int64_t n_groups : {2, 4, 8}) {
    const std::vector<float> gn = RunNorm(in, {rows, hidden}, n_groups, eps, DType::kF32, true);
    double max_diff = 0.0;
    for (size_t i = 0; i < g1.size(); ++i)
      max_diff = std::max(max_diff, std::abs(static_cast<double>(g1[i]) - gn[i]));
    INFO("n_groups=" << n_groups << " max|diff vs n_groups=1| = " << max_diff);
    CHECK(max_diff > 1e-2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) The gate is SILU, not the sigmoid vt::RmsNormGated uses. `x * silu(z)` and
// `x * sigmoid(z)` differ by the factor z, so a sigmoid kernel cannot pass.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 gated group norm gates with silu, not sigmoid") {
  const int64_t rows = 8, hidden = 64;
  const float eps = 1e-6f;
  const NormInputs in = GenerateNorm(rows, hidden, 0x77u);
  const std::vector<float> got = RunNorm(in, {rows, hidden}, 2, eps, DType::kF32, true);

  // The same computation with a SIGMOID gate (what vt::RmsNormGated would do)
  // — it must NOT match.
  std::vector<double> sig_ref(static_cast<size_t>(rows * hidden));
  {
    const int64_t n_groups = 2, group_size = hidden / n_groups;
    for (int64_t r = 0; r < rows; ++r) {
      std::vector<double> v(static_cast<size_t>(hidden));
      for (int64_t j = 0; j < hidden; ++j) {
        const double z = in.gate[static_cast<size_t>(r * hidden + j)];
        v[static_cast<size_t>(j)] =
            static_cast<double>(in.x[static_cast<size_t>(r * hidden + j)]) /
            (1.0 + std::exp(-z));  // SIGMOID gate
      }
      for (int64_t g = 0; g < n_groups; ++g) {
        double ss = 0.0;
        for (int64_t j = 0; j < group_size; ++j) {
          const double t = v[static_cast<size_t>(g * group_size + j)];
          ss += t * t;
        }
        const double inv = 1.0 / std::sqrt(ss / static_cast<double>(group_size) + eps);
        for (int64_t j = 0; j < group_size; ++j) {
          const int64_t idx = g * group_size + j;
          sig_ref[static_cast<size_t>(r * hidden + idx)] =
              static_cast<double>(in.weight[static_cast<size_t>(idx)]) *
              (v[static_cast<size_t>(idx)] * inv);
        }
      }
    }
  }
  double max_diff = 0.0;
  for (size_t i = 0; i < got.size(); ++i)
    max_diff = std::max(max_diff, std::abs(static_cast<double>(got[i]) - sig_ref[i]));
  INFO("max|silu-gated - sigmoid-gated| = " << max_diff);
  CHECK(max_diff > 1e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) `use_rms_norm = False` — upstream registers NO weight parameter and
// returns just `x * silu(gate)` (mamba_mixer2.py:94-96, :115-116). That is the
// `weight == nullptr` arm.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 gated group norm skips the norm when there is no weight") {
  const int64_t rows = 32, hidden = 64;
  const NormInputs in = GenerateNorm(rows, hidden, 0x4444u);
  const std::vector<double> ref =
      GatedGroupNormRef(in.x, in.gate, nullptr, rows, hidden, 4, 1e-6);
  ExpectClose("gated only, f32", RunNorm(in, {rows, hidden}, 4, 1e-6f, DType::kF32, false), ref,
              5e-3, 1e-3);
  ExpectClose("gated only, bf16", RunNorm(in, {rows, hidden}, 4, 1e-6f, DType::kBF16, false),
              ref, 5e-2, 1e-2);
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) Rank-3 inputs: upstream applies over the LAST dim with arbitrary leading
// dims (`*prefix_dims, hidden_dim`, mamba_mixer2.py:136). A [B,T,Hd] call must
// equal the flattened [B*T,Hd] call element for element.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 gated group norm treats every leading dim as a row") {
  const int64_t b = 4, t = 8, hidden = 64;
  const NormInputs in = GenerateNorm(b * t, hidden, 0x8181u);
  const std::vector<float> flat = RunNorm(in, {b * t, hidden}, 4, 1e-6f, DType::kF32, true);
  const std::vector<float> r3 = RunNorm(in, {b, t, hidden}, 4, 1e-6f, DType::kF32, true);
  REQUIRE(flat.size() == r3.size());
  for (size_t i = 0; i < flat.size(); ++i) CHECK(flat[i] == r3[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) REFUSALS.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("mamba2 gated group norm refuses the arms it does not implement") {
  const int64_t rows = 8, hidden = 64;
  const NormInputs in = GenerateNorm(rows, hidden, 2u);

  SUBCASE("tp_world_size > 1 names extra_groups_for_head_shards") {
    bool threw = false;
    std::string msg;
    try {
      RunNorm(in, {rows, hidden}, 2, 1e-6f, DType::kF32, true, /*tp_world_size=*/2);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    CHECK(threw);
    INFO(msg);
    CHECK(msg.find("extra_groups_for_head_shards") != std::string::npos);
  }

  SUBCASE("n_groups must divide the hidden dim") {
    CHECK_THROWS(RunNorm(in, {rows, hidden}, 7, 1e-6f, DType::kF32, true));
  }
}
