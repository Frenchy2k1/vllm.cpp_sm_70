// CUDA keep-quant GEMM gate (QUANT-GGUF-CIQ-GEMM-CUDA). The kCUDA provider for
// `OpId::kMatmulBTQuant` (src/vt/cuda/cuda_quant_dot.cu) is measured against the
// LANDED CPU keep-quant reference (src/vt/cpu/cpu_quant_gemm.cpp — the oracle,
// itself gated by test_ops_quant_dot.cpp) and against an INDEPENDENT f64
// dequantize-then-dot, on the DeepSeek-V4 encodings IQ2_XXS / IQ3_XXS / Q2_K
// plus the four other Q8_K-family k-quants the CUDA kernel also serves.
//
// The GATE (per the campaign): the CUDA kernel's Q8_K activation quant and the
// whole INTEGER dot are bit-identical to the CPU reference by construction, so
// CUDA-vs-CPU is asserted at a TIGHT NMSE (1e-6, f32 out) — only the per-super-
// block float scale sum is reassociated (warp reduction vs the CPU sequential
// add). CUDA-vs-f64-dequant is asserted at the SAME band test_ops_quant_dot uses
// (5e-4, test-backend-ops.cpp:4277). A wrong codebook index / scale unpack /
// sign would blow both bands (RED-first): the tables live device-side, so a
// transcription slip in cuda_quant_iq_tables.cuh surfaces here as an IQ2/IQ3
// divergence, not silently.
//
// Skips cleanly (returns) when no CUDA backend is present so the CPU CI leg is
// green; it only asserts on a real GB10/CUDA device.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"      // vt::cpu::BlockToFloat
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

// test-backend-ops.cpp:4277 — the MUL_MAT NMSE band the CPU keep-quant is gated
// on; the CUDA kernel must meet the same against the f64 dequant reference.
constexpr double kMaxNmseErr = 5e-4;
// CUDA-vs-CPU-oracle: integer core identical, so only float reassociation drifts.
constexpr double kMaxNmseVsCpu = 1e-6;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
};

// Same table as test_ops_quant_dot.cpp, restricted to the Q8_K family the CUDA
// kernel serves natively (offsets from ggml-common.h, restated independently).
const WeightCase kCases[] = {
    {DType::kIQ2_XXS, 256, 66, 0, -1, "iq2_xxs"},   // DeepSeek-V4 gate/up
    {DType::kIQ3_XXS, 256, 98, 0, -1, "iq3_xxs"},   // DeepSeek-V4 down
    {DType::kQ2_K, 256, 84, 80, 82, "q2_K"},        // DeepSeek-V4 UD-Q2_K_XL
    {DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
};

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = Gpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

}  // namespace

TEST_CASE("CUDA keep-quant GEMM == CPU reference and f64 dequant (Q8_K family)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    // Decode (M=1) through prefill (M=512), plus an odd N that catches a warp/
    // chunking assumption; K = 8 super-blocks (model-ish).
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);

        std::vector<uint8_t> wq =
            RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());

        // --- CPU oracle (the landed keep-quant kernel over host tensors) ------
        std::vector<float> cpu_out(static_cast<size_t>(m * n), 0.0F);
        {
          Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {m, k});
          Tensor bt =
              Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
          bt.dtype = c.dtype;
          Tensor ot =
              Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
          vt::MatmulBTQuant(cq, ot, at, bt);
        }

        // --- CUDA path (device tensors, unified pool) -------------------------
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_w = gpu.Alloc(wq.size());
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        Tensor bt = DevTensor(d_w, c.dtype, {n, k});
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> cuda_out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_w);
        gpu.Free(d_o);

        // --- f64 independent reference: decode weight + f32 activation dot -----
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);

        double num_ref = 0, den_ref = 0, num_cpu = 0, den_cpu = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t jj = 0; jj < n; ++jj) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p)
              ref += static_cast<double>(a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(jj * k + p)]);
            const double got =
                cuda_out[static_cast<size_t>(i * n + jj)];
            const double cpu = cpu_out[static_cast<size_t>(i * n + jj)];
            num_ref += (got - ref) * (got - ref);
            den_ref += ref * ref;
            num_cpu += (got - cpu) * (got - cpu);
            den_cpu += cpu * cpu;
            REQUIRE(std::isfinite(got));
          }
        }
        const double nmse_ref = den_ref > 0 ? num_ref / den_ref : num_ref;
        const double nmse_cpu = den_cpu > 0 ? num_cpu / den_cpu : num_cpu;
        CAPTURE(nmse_ref);
        CAPTURE(nmse_cpu);
        CHECK(nmse_ref <= kMaxNmseErr);   // quantization error vs f64 dequant
        CHECK(nmse_cpu <= kMaxNmseVsCpu);  // matches the CPU oracle (int core exact)
      }
    }
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("CUDA keep-quant GEMM registers the native kCUDA provider") {
  // The registration is what flips the GGUF loader's keep-quant default ON on a
  // CUDA device (GgufQuantComputeAvailable -> OpRegistered(kMatmulBTQuant,kCUDA))
  // so DeepSeek-V4's experts dispatch to the GPU. Present only in a CUDA build.
  if (!HasCuda()) return;
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuant, DeviceType::kCUDA));
}
