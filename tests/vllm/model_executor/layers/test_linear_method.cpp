// LinearMethod / QuantizationConfig seam — scheme×device selection + bf16 apply.
//
// Ports vLLM's scheme-parameterized create_weights/apply coverage
// (tests/kernels/quantization/**) to the vt::-native seam (work row S4 of
// .agents/specs/accelerator-seam-audit.md): the scheme is chosen ONCE by the
// factory from the checkpoint's populated weights, and the bf16 UnquantizedLinear
// apply runs the exact vt::MatmulBT the inline model path did.
//
// CPU-only (no checkpoint), runs in CI. The NVFP4 numeric path is gated on dgx
// via the paged-engine model tests; here we assert the FACTORY SELECTS the right
// method per scheme (the S4 policy decision) and that the bf16 apply is correct.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h"
#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include <cmath>
#include <random>

namespace {

using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vt::DType;
namespace layers = vllm::layers;

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, uint32_t seed) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = true;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  uint32_t s = seed;
  for (int64_t i = 0; i < numel; ++i) {
    s = s * 1664525u + 1013904223u;
    const float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
    p[i] = vt::F32ToBF16(v);
  }
  return o;
}

// A minimal non-empty W4A16 NVFP4 weight (alpha == 0): enough for the factory to
// select the quantized scheme. Its numeric path is exercised on dgx, not here.
Nvfp4Weight MakeNvfp4W4A16(int64_t N, int64_t K) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.scale2 = 1.0f;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * (K / 2), 0);
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 16;
  w.scale.bytes.resize(static_cast<size_t>(N) * (K / 16), 0);
  return w;
}

// A random MXFP4 W4A16 weight: E2M1 packed [N,K/2] + E8M0 scale [N,K/32], group
// 32, no global, is_mxfp4=true — so the factory + Apply route the MXFP4 keep-quant
// path (Marlin on GPU via BuildMarlinDenseResident).
Nvfp4Weight MakeMxfp4W4A16(int64_t N, int64_t K, uint32_t seed) {
  Nvfp4Weight w;
  w.n = N;
  w.k = K;
  w.group_size = 32;
  w.is_mxfp4 = true;
  w.scale2 = 0.0f;
  w.packed.dtype = DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = N;
  w.packed.shape[1] = K / 2;
  w.packed.bytes.resize(static_cast<size_t>(N) * (K / 2));
  w.scale.dtype = DType::kI8;
  w.scale.rank = 2;
  w.scale.shape[0] = N;
  w.scale.shape[1] = K / 32;
  w.scale.bytes.resize(static_cast<size_t>(N) * (K / 32));
  std::mt19937 rng(seed);
  for (auto& b : w.packed.bytes) b = static_cast<uint8_t>(rng() & 0xFFu);
  for (auto& s : w.scale.bytes) s = static_cast<uint8_t>(118u + (rng() % 15u));
  return w;
}

}  // namespace

#ifdef VT_MARLIN_NVFP4
// The model-facing MXFP4 path END-TO-END: MakeLinearMethod(bf16-empty, mxfp4) ->
// Apply -> MatmulNvfp4W4A16D -> (GPU) MatmulNvfp4MarlinD -> BuildMarlinDenseResident
// -> MoeGroupedGemmNvfp4Marlin. This is the ONE link the op-level unit gate does NOT
// cover (it feeds MANUALLY-built residents), so it isolates a resident-builder bug
// from the kernel. Reference = the INDEPENDENT CPU dequant (DequantMxfp4ToF32 + f32
// matmul). Real Qwen3-8B projection shapes; M=1 (decode) AND M=8 (prefill).
TEST_CASE("linear_method: MXFP4 W4A16 Apply (Marlin BuildMarlinDenseResident) == CPU dequant ref") {
  vt::Backend* gpu = nullptr;
  try {
    gpu = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  // Persist weights in a vector so each shape has a DISTINCT, stable address —
  // the resident cache (MarlinDenseResidentFor) is keyed by weight pointer, and a
  // loop-local reused stack slot would alias residents across shapes (a test
  // artifact, not a model bug: the model's weights are distinct persistent objects).
  const std::vector<std::pair<int64_t, int64_t>> shapes{{4096, 4096}, {12288, 4096}};
  std::vector<Nvfp4Weight> weights;
  for (auto KN : shapes) weights.push_back(MakeMxfp4W4A16(KN.second, KN.first, 2024));
  for (size_t si = 0; si < shapes.size(); ++si) {
    const int64_t K = shapes[si].first, N = shapes[si].second;
    CAPTURE(K);
    CAPTURE(N);
    Nvfp4Weight& w = weights[si];
    OwnedTensor bf16_empty;  // Empty() => factory selects the fp4 method

    std::vector<float> w_f32(static_cast<size_t>(N * K));
    vllm::DequantMxfp4ToF32(reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
                            reinterpret_cast<const uint8_t*>(w.scale.bytes.data()), N, K,
                            w_f32.data());

    for (int64_t M : {int64_t{1}, int64_t{8}}) {
      CAPTURE(M);
      vt::Queue q = gpu->CreateQueue();
      vllm::dense_attn::Dev d{*gpu, q};

      std::vector<uint16_t> act_bf16(static_cast<size_t>(M * K));
      std::mt19937 rng(7 + static_cast<uint32_t>(M));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<float> act_r(static_cast<size_t>(M * K));
      for (size_t i = 0; i < act_bf16.size(); ++i) {
        act_bf16[i] = vt::F32ToBF16(dist(rng));
        act_r[i] = vt::BF16ToF32(act_bf16[i]);
      }
      std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
          float acc = 0.0f;
          for (int64_t k = 0; k < K; ++k)
            acc += act_r[static_cast<size_t>(m * K + k)] * w_f32[static_cast<size_t>(n * K + k)];
          ref[static_cast<size_t>(m * N + n)] = acc;
        }

      vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, act_bf16.data());
      auto method = layers::MakeLinearMethod(bf16_empty, w);
      vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kBF16);
      std::vector<uint16_t> got_bf16(static_cast<size_t>(M * N));
      gpu->Copy(q, got_bf16.data(), out.t().data,
                got_bf16.size() * sizeof(uint16_t));
      gpu->Synchronize(q);
      double max_rel = 0.0, max_abs = 0.0;
      size_t bad = 0;
      for (size_t i = 0; i < got_bf16.size(); ++i) {
        const float g = vt::BF16ToF32(got_bf16[i]);
        const float a = std::fabs(g - ref[i]);
        const float tol = 2e-2f + 2e-2f * std::fabs(ref[i]);
        if (a > tol) ++bad;
        max_abs = std::max(max_abs, static_cast<double>(a));
        max_rel = std::max(max_rel, static_cast<double>(a / (std::fabs(ref[i]) + 1e-6f)));
      }
      MESSAGE("MXFP4 Apply K=" << K << " N=" << N << " M=" << M
              << " bad=" << bad << " max_abs=" << max_abs << " max_rel=" << max_rel);
      CHECK(bad == 0);
      gpu->DestroyQueue(q);
    }
  }
}
#endif  // VT_MARLIN_NVFP4

TEST_CASE("linear_method: factory selects bf16 vs nvfp4-w4a16 by weight presence") {
  OwnedTensor bf16 = MakeBf16({4, 16}, 1);
  Nvfp4Weight empty_fp4;      // Empty() == true
  Nvfp4Weight fp4 = MakeNvfp4W4A16(4, 16);
  REQUIRE(empty_fp4.Empty());
  REQUIRE_FALSE(fp4.Empty());

  // get_quant_method analogue: a bf16 checkpoint => UnquantizedLinearMethod.
  auto m_bf16 = layers::MakeLinearMethod(bf16, empty_fp4);
  CHECK(std::string(m_bf16->Name()) == "bf16-unquantized");

  // An NVFP4-packed checkpoint => the compressed-tensors W4A16 method, chosen
  // ONCE here (not by a per-call IsNvfp4() probe in the model forward).
  auto m_fp4 = layers::MakeLinearMethod(bf16, fp4);
  CHECK(std::string(m_fp4->Name()) == "compressed-tensors-nvfp4-w4a16");
}

TEST_CASE("linear_method: gate_up factory selects scheme by weight presence") {
  OwnedTensor gate_up = MakeBf16({2 * 16, 8}, 2);
  Nvfp4Weight empty;
  Nvfp4Weight gate = MakeNvfp4W4A16(16, 8);
  Nvfp4Weight up = MakeNvfp4W4A16(16, 8);

  auto g_bf16 = layers::MakeMlpGateUpMethod(gate_up, empty, empty, 16);
  CHECK(std::string(g_bf16->Name()) == "bf16-gate-up");

  auto g_fp4 = layers::MakeMlpGateUpMethod(gate_up, gate, up, 16);
  CHECK(std::string(g_fp4->Name()) == "compressed-tensors-nvfp4-w4a16-gate-up");
}

// Tier-A1 fold REUSE PROOF (arch-fusion-fold-plan-2026-07-30 §A1): the SHARED bf16
// gate-up MLP seam (UnquantizedMlpGateUpMethod::Apply) must produce a BYTE-IDENTICAL
// result to the standalone {ResidentWeight; MatmulBT[2I,H]; SiluAndMul} sequence the
// five folded arch MLP blocks (OLMo-2 / Granite / StableLM / qwen3_dflash /
// deepseek_v2) hand-rolled before the fold. Same ops, same order, same device ⇒ the
// comparison is EXACT (raw bf16 bytes), not Approx. This is the CPU composite-golden
// backing the two archs (dflash, deepseek_v2) whose checkpoints are not always on the
// gate box; the three with SACRED goldens (OLMo-2/Granite/StableLM) are additionally
// gated token-exact on dgx.
TEST_CASE("linear_method: fused gate-up seam == standalone MatmulBT+SiluAndMul (byte-exact)") {
  const int64_t M = 3, H = 8, I = 5;
  OwnedTensor gate_up = MakeBf16({2 * I, H}, 11);  // merged [2I, H] raw-NK
  OwnedTensor xw = MakeBf16({M, H}, 13);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, H}, xw.bytes.data());

  // (A) SHARED seam: pick the bf16 arm via the factory (no fp4 present), Apply.
  auto method = layers::MakeMlpGateUpMethod(gate_up, Nvfp4Weight{}, Nvfp4Weight{}, I);
  REQUIRE(std::string(method->Name()) == "bf16-gate-up");
  vllm::dense_attn::DBuf act_fused = method->Apply(d, x.t());

  // (B) STANDALONE reference: the exact op sequence the arch MLP blocks ran.
  vt::Tensor wgu = vllm::dense_attn::ResidentWeight(d, gate_up);  // [2I, H]
  vllm::dense_attn::DBuf gu(d, DType::kBF16, {M, 2 * I});
  vt::MatmulBT(d.q, gu.t(), x.t(), wgu);
  vllm::dense_attn::DBuf act_ref(d, DType::kBF16, {M, I});
  vt::SiluAndMul(d.q, act_ref.t(), gu.t());

  std::vector<uint16_t> got(static_cast<size_t>(M) * I);
  std::vector<uint16_t> ref(static_cast<size_t>(M) * I);
  act_fused.Download(d, got.data());
  act_ref.Download(d, ref.data());
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == ref[i]);  // BYTE-IDENTICAL — the fold changes nothing numerically
}

// Tier-C1 fold REUSE PROOF (arch-fusion-fold-plan-2026-07-30 §C1): the SHARED bf16
// GeGLU gate-up MLP seam (UnquantizedMlpGateUpGeluMethod::Apply) must produce a
// BYTE-IDENTICAL result to the standalone {ResidentWeight; MatmulBT[2I,H]; GeluAndMul}
// sequence the four Gemma-family MLP blocks (Gemma-1/2/3/4) hand-rolled before the
// fold. Same merged [2I,H] operand, same single MatmulBT, same GeluAndMul(tanh)
// epilogue, same device ⇒ EXACT (raw bf16 bytes), not Approx. This is the GeGLU
// sibling of the SwiGLU byte-exact case above; the SACRED gates (Gemma-2 48/48,
// Gemma-4 32/32) prove the same shared method is token-exact end-to-end on the GPU.
TEST_CASE("linear_method: fused GeGLU gate-up seam == standalone MatmulBT+GeluAndMul (byte-exact)") {
  const int64_t M = 3, H = 8, I = 5;
  OwnedTensor gate_up = MakeBf16({2 * I, H}, 17);  // merged [2I, H] raw-NK
  OwnedTensor xw = MakeBf16({M, H}, 19);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};
  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, H}, xw.bytes.data());

  // (A) SHARED seam: the bf16 GeGLU arm (GeluAndMul(tanh) epilogue), Apply.
  layers::UnquantizedMlpGateUpGeluMethod method(&gate_up, I);
  REQUIRE(std::string(method.Name()) == "bf16-gate-up-gelu");
  vllm::dense_attn::DBuf act_fused = method.Apply(d, x.t());

  // (B) STANDALONE reference: the exact op sequence the Gemma MLP blocks ran.
  vt::Tensor wgu = vllm::dense_attn::ResidentWeight(d, gate_up);  // [2I, H]
  vllm::dense_attn::DBuf gu(d, DType::kBF16, {M, 2 * I});
  vt::MatmulBT(d.q, gu.t(), x.t(), wgu);
  vllm::dense_attn::DBuf act_ref(d, DType::kBF16, {M, I});
  vt::GeluAndMul(d.q, act_ref.t(), gu.t());

  std::vector<uint16_t> got(static_cast<size_t>(M) * I);
  std::vector<uint16_t> ref(static_cast<size_t>(M) * I);
  act_fused.Download(d, got.data());
  act_ref.Download(d, ref.data());
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == ref[i]);  // BYTE-IDENTICAL — the fold changes nothing numerically
}

TEST_CASE("linear_method: bf16 UnquantizedLinearMethod apply == reference MatmulBT") {
  const int64_t M = 2, K = 16, N = 4;
  OwnedTensor w = MakeBf16({N, K}, 7);  // raw-NK [N=out, K=in]
  OwnedTensor xw = MakeBf16({M, K}, 9);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{b, q};

  vllm::dense_attn::DBuf x(d, DType::kBF16, {M, K}, xw.bytes.data());
  auto method = layers::MakeLinearMethod(w, Nvfp4Weight{});
  REQUIRE(std::string(method->Name()) == "bf16-unquantized");
  vllm::dense_attn::DBuf out = method->Apply(d, x.t(), DType::kF32);

  std::vector<float> got(static_cast<size_t>(M) * N);
  out.Download(d, got.data());

  const auto* wp = reinterpret_cast<const uint16_t*>(w.bytes.data());
  const auto* xp = reinterpret_cast<const uint16_t*>(xw.bytes.data());
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += vt::BF16ToF32(xp[m * K + k]) * vt::BF16ToF32(wp[n * K + k]);
      CHECK(got[static_cast<size_t>(m) * N + n] == doctest::Approx(acc).epsilon(0.02));
    }
  }
}
