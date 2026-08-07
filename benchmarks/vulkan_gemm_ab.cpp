// Vulkan GEMM tactic A/B: cooperative-matrix vs the portable scalar kernel.
// BACKEND-VULKAN work row VK-C.
//
// SAME BINARY, ONE VARIABLE. Both arms run this executable; the only difference
// is VT_VULKAN_COOPMAT, which forces the scalar tactic when set to 0. Comparing
// two builds would confound the kernel with everything else that differs between
// them, which .agents/benchmark-protocol.md rules out.
//
// EACH ARM PROVES WHICH TACTIC IT RAN. `VulkanContext::PipelineExistsFor` is
// checked and printed, because a silent fallback would produce a perfectly
// plausible pair of timings that mean nothing -- the scalar kernel is equally
// correct, so numbers alone cannot tell the arms apart. This is the same failure
// mode the op-provider decline counters exist for, and it already nearly slipped
// through this row's correctness gate.
//
// It also VERIFIES both arms against a host oracle, so a tactic that is fast
// because it is wrong cannot post a good number.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/vulkan/vulkan_context.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

double MedianMs(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  // MatmulBT shape: out[M,N] = a[M,K] @ b[N,K]^T -- the torch Linear layout, which
  // is what a model actually dispatches. Defaults are a decode-ish projection.
  int64_t m = argc > 1 ? std::atoll(argv[1]) : 256;
  int64_t k = argc > 2 ? std::atoll(argv[2]) : 2048;
  int64_t n = argc > 3 ? std::atoll(argv[3]) : 2048;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 30;
  const int warmup = 5;

  if (!vt::vulkan::VulkanDeviceAvailable()) {
    std::fprintf(stderr, "no Vulkan device\n");
    return 2;
  }
  auto& ctx = vt::vulkan::VulkanContext::Get();
  vt::Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  std::printf("device            : %s\n", ctx.device_name().c_str());
  std::printf("coopmat available : %s\n", ctx.coopmat_bf16_f32() ? "YES" : "no");
  std::printf("subgroup size     : %u\n", ctx.subgroup_size());
  const char* lever = std::getenv("VT_VULKAN_COOPMAT");
  std::printf("VT_VULKAN_COOPMAT : %s\n", lever ? lever : "(unset -> enabled)");
  std::printf("shape             : M=%lld K=%lld N=%lld, %d reps\n",
              (long long)m, (long long)k, (long long)n, reps);

  // bf16 operands: the only dtype the coopmat tactic accepts.
  std::vector<uint16_t> ha(m * k), hb(n * k);
  for (int64_t i = 0; i < m * k; ++i) ha[i] = vt::F32ToBF16(0.5f * float((i % 7) - 3));
  for (int64_t i = 0; i < n * k; ++i) hb[i] = vt::F32ToBF16(0.25f * float((i % 5) - 2));

  void* da = vk.Alloc(ha.size() * sizeof(uint16_t));
  void* db = vk.Alloc(hb.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(m * n * sizeof(float)));
  vk.Copy(q, da, ha.data(), ha.size() * sizeof(uint16_t));
  vk.Copy(q, db, hb.data(), hb.size() * sizeof(uint16_t));
  vk.Synchronize(q);

  Tensor ta = Tensor::Contiguous(da, DType::kBF16, d, {m, k});
  Tensor tb = Tensor::Contiguous(db, DType::kBF16, d, {n, k});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, d, {m, n});

  for (int i = 0; i < warmup; ++i) vt::MatmulBT(q, to, ta, tb);
  vk.Synchronize(q);

  // WHICH TACTIC RAN -- printed after the warm-up, so the pipeline has been built.
  const bool coop = ctx.PipelineExistsFor("vt_matmul_coopmat");
  const bool scal = ctx.PipelineExistsFor("vt_matmul");
  std::printf("tactic            : %s%s\n", coop ? "COOPMAT" : "scalar",
              (coop && scal) ? "  (WARNING: both pipelines built)" : "");

  std::vector<double> ms;
  ms.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    vt::MatmulBT(q, to, ta, tb);
    vk.Synchronize(q);
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  // Correctness, so a tactic cannot be fast by being wrong. One spot row against
  // a host dot product in f32.
  std::vector<float> got(m * n);
  vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
  vk.Synchronize(q);
  double worst = 0.0;
  for (int64_t j = 0; j < std::min<int64_t>(n, 64); ++j) {
    double acc = 0.0;
    for (int64_t p = 0; p < k; ++p) {
      acc += double(vt::BF16ToF32(ha[p])) * double(vt::BF16ToF32(hb[j * k + p]));
    }
    worst = std::max(worst, std::abs(acc - double(got[j])) / std::max(1.0, std::abs(acc)));
  }

  const double med = MedianMs(ms);
  const double gflop = 2.0 * double(m) * double(n) * double(k) / 1e9;
  std::printf("median ms         : %.3f\n", med);
  std::printf("GFLOP/s           : %.1f\n", med > 0 ? gflop / (med / 1e3) : 0.0);
  std::printf("worst rel err     : %.3e\n", worst);

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
  return worst < 1e-3 ? 0 : 1;
}
