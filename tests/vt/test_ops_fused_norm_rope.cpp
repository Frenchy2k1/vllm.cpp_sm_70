// vllm.cpp original — the RED-first byte-exactness gate for vt::FusedNormRope
// (kFusedNormRope), the Tier-A5 shared MLA norm-rope fold. No upstream mirror.
//
// The fold's WHOLE claim is that ONE FusedNormRope launch is BIT-IDENTICAL to the
// standalone {vt::RmsNorm(latent) ; vt::RopeFromCache(k_pe)} pair the DeepSeek-MLA
// block used to issue. This test proves exactly that: it runs BOTH the composite
// (via the shipped standalone ops) and the fused op over the SAME merged input and
// asserts the two outputs are byte-for-byte equal (== , not Approx). A wrong kernel
// — e.g. one that RMS-reduces the full [off+rot] row (the ds4 per-head form) instead
// of the latent [0,off) only, or ropes the NORMED rather than the RAW pe slice — is
// caught by the exact byte check (the discriminator CHECK at the end asserts the
// full-width-RMS variant genuinely differs, so the test is not vacuously green).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

vt::Device Cpu() { return vt::Device{vt::DeviceType::kCPU, 0}; }

// Deterministic pseudo-random fill (no <random> dependence for reproducibility).
float Pseudo(int64_t i, int64_t salt) {
  const double x = std::sin(static_cast<double>(i) * 12.9898 +
                            static_cast<double>(salt) * 78.233) *
                   43758.5453;
  return static_cast<float>((x - std::floor(x)) * 2.0 - 1.0);  // [-1, 1)
}

}  // namespace

TEST_CASE("fused_norm_rope f32 == standalone {RmsNorm(latent); RopeFromCache(pe)}") {
  constexpr int64_t kT = 5;
  constexpr int64_t kOff = 16;   // kv_lora_rank-shaped latent
  constexpr int kRot = 8;        // qk_rope_head_dim-shaped decoupled pe
  constexpr int64_t kW = kOff + kRot;
  constexpr int64_t kCacheRows = 64;
  const float kEps = 1e-6F;

  for (bool neox : {true, false}) {
    CAPTURE(neox);
    vt::RopeArgs rope{10000.0F, kRot};
    rope.is_neox_style = neox;

    // cos|sin cache built by the PRODUCTION op.
    std::vector<float> cache(static_cast<size_t>(kCacheRows) * kRot, 0.0F);
    {
      std::vector<int64_t> cpos(kCacheRows);
      for (int64_t i = 0; i < kCacheRows; ++i) cpos[static_cast<size_t>(i)] = i;
      vt::Tensor tcp =
          vt::Tensor::Contiguous(cpos.data(), vt::DType::kI64, Cpu(), {kCacheRows});
      vt::Tensor tcache = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(),
                                                 {kCacheRows, kRot});
      vt::Queue cq{Cpu(), nullptr};
      vt::RopeCosSinCache(cq, tcache, tcp, rope);
    }

    // Merged [T, off+rot] input + weight + positions.
    std::vector<float> x(static_cast<size_t>(kT * kW));
    for (size_t i = 0; i < x.size(); ++i) x[i] = Pseudo(static_cast<int64_t>(i), 1);
    std::vector<float> w(kOff);
    for (size_t i = 0; i < w.size(); ++i) w[i] = 0.5F + 0.5F * Pseudo(static_cast<int64_t>(i), 2);
    std::vector<int32_t> pos(kT);
    for (int64_t t = 0; t < kT; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>((t * 7 + 3) % kCacheRows);

    vt::Queue q{Cpu(), nullptr};
    vt::RmsNormArgs nargs{kEps, false};

    // ── GOLDEN composite via the shipped standalone ops ──────────────────────
    // latent: RmsNorm over a CONTIGUOUS copy of x[:, :off].
    std::vector<float> xl(static_cast<size_t>(kT * kOff));
    for (int64_t t = 0; t < kT; ++t)
      std::memcpy(&xl[static_cast<size_t>(t * kOff)], &x[static_cast<size_t>(t * kW)],
                  sizeof(float) * kOff);
    std::vector<float> latent_ref(static_cast<size_t>(kT * kOff));
    {
      vt::Tensor tin = vt::Tensor::Contiguous(xl.data(), vt::DType::kF32, Cpu(), {kT, kOff});
      vt::Tensor tout = vt::Tensor::Contiguous(latent_ref.data(), vt::DType::kF32, Cpu(), {kT, kOff});
      vt::Tensor tw = vt::Tensor::Contiguous(w.data(), vt::DType::kF32, Cpu(), {kOff});
      vt::RmsNorm(q, tout, tin, tw, nargs);
    }
    // pe: RopeFromCache in place over a CONTIGUOUS copy of x[:, off:] as [T,1,rot].
    std::vector<float> pe_ref(static_cast<size_t>(kT * kRot));
    for (int64_t t = 0; t < kT; ++t)
      std::memcpy(&pe_ref[static_cast<size_t>(t * kRot)],
                  &x[static_cast<size_t>(t * kW + kOff)], sizeof(float) * kRot);
    {
      vt::Tensor tpe = vt::Tensor::Contiguous(pe_ref.data(), vt::DType::kF32, Cpu(), {kT, 1, kRot});
      vt::Tensor tp = vt::Tensor::Contiguous(pos.data(), vt::DType::kI32, Cpu(), {kT});
      vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(), {kCacheRows, kRot});
      vt::RopeFromCache(q, tpe, nullptr, tp, tc, rope);
    }

    // ── FUSED op over the merged input ───────────────────────────────────────
    std::vector<float> latent_got(static_cast<size_t>(kT * kOff), 0.0F);
    std::vector<float> pe_got(static_cast<size_t>(kT * kRot), 0.0F);
    {
      vt::Tensor tx = vt::Tensor::Contiguous(x.data(), vt::DType::kF32, Cpu(), {kT, kW});
      vt::Tensor tw = vt::Tensor::Contiguous(w.data(), vt::DType::kF32, Cpu(), {kOff});
      vt::Tensor tlat = vt::Tensor::Contiguous(latent_got.data(), vt::DType::kF32, Cpu(), {kT, kOff});
      vt::Tensor tpe = vt::Tensor::Contiguous(pe_got.data(), vt::DType::kF32, Cpu(), {kT, kRot});
      vt::Tensor tp = vt::Tensor::Contiguous(pos.data(), vt::DType::kI32, Cpu(), {kT});
      vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, Cpu(), {kCacheRows, kRot});
      vt::FusedNormRope(q, tlat, tpe, tx, tw, tp, tc, nargs, rope);
    }

    // BYTE-EXACT (bit-for-bit) — the fold's correctness contract.
    CHECK(std::memcmp(latent_got.data(), latent_ref.data(), latent_got.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(pe_got.data(), pe_ref.data(), pe_got.size() * sizeof(float)) == 0);

    // Discriminator: a full-width RMS over [off+rot] (the WRONG ds4 per-head form)
    // must genuinely differ, so the byte-check above is not vacuously satisfiable.
    std::vector<float> latent_wrong(static_cast<size_t>(kT * kOff));
    for (int64_t t = 0; t < kT; ++t) {
      double ss = 0.0;
      for (int64_t j = 0; j < kW; ++j) { const float v = x[static_cast<size_t>(t * kW + j)]; ss += static_cast<double>(v) * v; }
      const float inv = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(kW)) + kEps);
      for (int64_t j = 0; j < kOff; ++j)
        latent_wrong[static_cast<size_t>(t * kOff + j)] = x[static_cast<size_t>(t * kW + j)] * inv * w[static_cast<size_t>(j)];
    }
    CHECK(std::memcmp(latent_got.data(), latent_wrong.data(), latent_got.size() * sizeof(float)) != 0);
  }
}

TEST_CASE("fused_norm_rope bf16 == standalone {RmsNorm(latent); RopeFromCache(pe)}") {
  constexpr int64_t kT = 4;
  constexpr int64_t kOff = 12;
  constexpr int kRot = 8;
  constexpr int64_t kW = kOff + kRot;
  constexpr int64_t kCacheRows = 48;
  const float kEps = 1e-6F;
  vt::RopeArgs rope{10000.0F, kRot};
  rope.is_neox_style = false;  // DeepSeek-V2/V3 default (GPT-J adjacent pair)

  std::vector<uint16_t> cache(static_cast<size_t>(kCacheRows) * kRot, 0);
  {
    // Build an f32 cache with the production op, then round to bf16.
    std::vector<float> cache_f32(cache.size(), 0.0F);
    std::vector<int64_t> cpos(kCacheRows);
    for (int64_t i = 0; i < kCacheRows; ++i) cpos[static_cast<size_t>(i)] = i;
    vt::Tensor tcp = vt::Tensor::Contiguous(cpos.data(), vt::DType::kI64, Cpu(), {kCacheRows});
    vt::Tensor tcache = vt::Tensor::Contiguous(cache_f32.data(), vt::DType::kF32, Cpu(), {kCacheRows, kRot});
    vt::Queue cq{Cpu(), nullptr};
    vt::RopeCosSinCache(cq, tcache, tcp, rope);
    for (size_t i = 0; i < cache.size(); ++i) cache[i] = vt::F32ToBF16(cache_f32[i]);
  }

  std::vector<uint16_t> x(static_cast<size_t>(kT * kW));
  for (size_t i = 0; i < x.size(); ++i) x[i] = vt::F32ToBF16(Pseudo(static_cast<int64_t>(i), 5));
  std::vector<uint16_t> w(kOff);
  for (size_t i = 0; i < w.size(); ++i) w[i] = vt::F32ToBF16(0.5F + 0.5F * Pseudo(static_cast<int64_t>(i), 6));
  std::vector<int32_t> pos(kT);
  for (int64_t t = 0; t < kT; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>((t * 5 + 1) % kCacheRows);

  vt::Queue q{Cpu(), nullptr};
  vt::RmsNormArgs nargs{kEps, false};

  std::vector<uint16_t> latent_ref(static_cast<size_t>(kT * kOff), 0);
  {
    std::vector<uint16_t> xl(static_cast<size_t>(kT * kOff));
    for (int64_t t = 0; t < kT; ++t)
      std::memcpy(&xl[static_cast<size_t>(t * kOff)], &x[static_cast<size_t>(t * kW)], sizeof(uint16_t) * kOff);
    vt::Tensor tin = vt::Tensor::Contiguous(xl.data(), vt::DType::kBF16, Cpu(), {kT, kOff});
    vt::Tensor tout = vt::Tensor::Contiguous(latent_ref.data(), vt::DType::kBF16, Cpu(), {kT, kOff});
    vt::Tensor tw = vt::Tensor::Contiguous(w.data(), vt::DType::kBF16, Cpu(), {kOff});
    vt::RmsNorm(q, tout, tin, tw, nargs);
  }
  std::vector<uint16_t> pe_ref(static_cast<size_t>(kT * kRot));
  for (int64_t t = 0; t < kT; ++t)
    std::memcpy(&pe_ref[static_cast<size_t>(t * kRot)], &x[static_cast<size_t>(t * kW + kOff)], sizeof(uint16_t) * kRot);
  {
    vt::Tensor tpe = vt::Tensor::Contiguous(pe_ref.data(), vt::DType::kBF16, Cpu(), {kT, 1, kRot});
    vt::Tensor tp = vt::Tensor::Contiguous(pos.data(), vt::DType::kI32, Cpu(), {kT});
    vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kBF16, Cpu(), {kCacheRows, kRot});
    vt::RopeFromCache(q, tpe, nullptr, tp, tc, rope);
  }

  std::vector<uint16_t> latent_got(static_cast<size_t>(kT * kOff), 0);
  std::vector<uint16_t> pe_got(static_cast<size_t>(kT * kRot), 0);
  {
    vt::Tensor tx = vt::Tensor::Contiguous(x.data(), vt::DType::kBF16, Cpu(), {kT, kW});
    vt::Tensor tw = vt::Tensor::Contiguous(w.data(), vt::DType::kBF16, Cpu(), {kOff});
    vt::Tensor tlat = vt::Tensor::Contiguous(latent_got.data(), vt::DType::kBF16, Cpu(), {kT, kOff});
    vt::Tensor tpe = vt::Tensor::Contiguous(pe_got.data(), vt::DType::kBF16, Cpu(), {kT, kRot});
    vt::Tensor tp = vt::Tensor::Contiguous(pos.data(), vt::DType::kI32, Cpu(), {kT});
    vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kBF16, Cpu(), {kCacheRows, kRot});
    vt::FusedNormRope(q, tlat, tpe, tx, tw, tp, tc, nargs, rope);
  }

  CHECK(std::memcmp(latent_got.data(), latent_ref.data(), latent_got.size() * sizeof(uint16_t)) == 0);
  CHECK(std::memcmp(pe_got.data(), pe_ref.data(), pe_got.size() * sizeof(uint16_t)) == 0);
}
