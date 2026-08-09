// Vulkan GEMM tactic A/B: cooperative-matrix vs the portable scalar kernel.
// BACKEND-VULKAN work row VK-C.
//
// ALSO THE CHEAP REPRODUCER for the lm_head BIMODALITY (VK-G, 2026-08-09).
// `vt_matmul` on the 27B's lm_head (`bt=0 m=1 k=5120 n=248320`, 2.543 GB per
// call) sometimes runs at ~260 ms/call instead of ~12.4 -- a 21x collapse to
// ~10 GB/s -- while `vt_matmul_vec` beside it moves 0.3%. It was chased through
// the 27B at ~3 minutes and ~106 GB of RSS per leg, which is a terrible
// experiment: this program reaches the same kernel on the same byte count in
// seconds. `nn` selects the `[K,N]` orientation the lm_head actually uses, and
// `cycles` REALLOCATES the weight between measurements, which is what separates
// "the slow state is a property of this ALLOCATION" from "of this process" or
// "of the moment".
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
  //   argv: M K N reps [bt|nn] [cycles]
  int64_t m = argc > 1 ? std::atoll(argv[1]) : 256;
  int64_t k = argc > 2 ? std::atoll(argv[2]) : 2048;
  int64_t n = argc > 3 ? std::atoll(argv[3]) : 2048;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 30;
  // `nn` is vt::Matmul with b as [K,N] -- the lm_head's orientation, and the one
  // the GEMV tactic correctly declines.
  const bool nn = argc > 5 && std::strcmp(argv[5], "nn") == 0;
  // Each cycle FREES and REALLOCATES b before measuring again.
  const int cycles = argc > 6 ? std::atoi(argv[6]) : 1;
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
  std::printf("shape             : M=%lld K=%lld N=%lld, %d reps, %s, %d cycle(s)\n",
              (long long)m, (long long)k, (long long)n, reps, nn ? "nn (b is [K,N])" : "bt",
              cycles);

  // bf16 operands: the only dtype the coopmat tactic accepts.
  std::vector<uint16_t> ha(m * k), hb(n * k);
  for (int64_t i = 0; i < m * k; ++i) ha[i] = vt::F32ToBF16(0.5f * float((i % 7) - 3));
  for (int64_t i = 0; i < n * k; ++i) hb[i] = vt::F32ToBF16(0.25f * float((i % 5) - 2));

  void* da = vk.Alloc(ha.size() * sizeof(uint16_t));
  auto* dout = static_cast<float*>(vk.Alloc(m * n * sizeof(float)));
  vk.Copy(q, da, ha.data(), ha.size() * sizeof(uint16_t));
  vk.Synchronize(q);
  Tensor ta = Tensor::Contiguous(da, DType::kBF16, d, {m, k});
  Tensor to = Tensor::Contiguous(dout, DType::kF32, d, {m, n});

  double worst = 0.0;
  for (int c = 0; c < cycles; ++c) {
    // REALLOCATED EVERY CYCLE. If a slow state follows the allocation rather
    // than the process, only this can show it -- measuring the same buffer twice
    // cannot distinguish the two.
    void* db = vk.Alloc(hb.size() * sizeof(uint16_t));
    vk.Copy(q, db, hb.data(), hb.size() * sizeof(uint16_t));
    vk.Synchronize(q);
    // In `nn`, b is [K,N] and vt::Matmul reads b[p*n + j]; in `bt` it is [N,K].
    Tensor tb = nn ? Tensor::Contiguous(db, DType::kBF16, d, {k, n})
                   : Tensor::Contiguous(db, DType::kBF16, d, {n, k});

    for (int i = 0; i < warmup; ++i) {
      if (nn) vt::Matmul(q, to, ta, tb); else vt::MatmulBT(q, to, ta, tb);
    }
    vk.Synchronize(q);

    if (c == 0) {
      // WHICH TACTIC RAN -- after the warm-up, so the pipeline has been built.
      const bool coop = ctx.PipelineExistsFor("vt_matmul_coopmat");
      const bool vec = ctx.PipelineExistsFor("vt_matmul_vec");
      const bool scal = ctx.PipelineExistsFor("vt_matmul");
      std::printf("tactic            : %s%s\n",
                  coop ? "COOPMAT" : (vec ? "GEMV (vt_matmul_vec)" : "scalar (vt_matmul)"),
                  (coop && scal) ? "  (WARNING: both pipelines built)" : "");
      for (const std::string& key : ctx.PipelineKeys()) {
        std::printf("pipeline          : %s\n", key.c_str());
      }
    }

    std::vector<double> ms;
    ms.reserve(reps);
    for (int i = 0; i < reps; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      if (nn) vt::Matmul(q, to, ta, tb); else vt::MatmulBT(q, to, ta, tb);
      vk.Synchronize(q);
      const auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Correctness, so a tactic cannot be fast by being wrong. One spot row
    // against a host dot product.
    std::vector<float> got(m * n);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    for (int64_t j = 0; j < std::min<int64_t>(n, 64); ++j) {
      double acc = 0.0;
      for (int64_t pp = 0; pp < k; ++pp) {
        acc += double(vt::BF16ToF32(ha[pp])) *
               double(vt::BF16ToF32(hb[nn ? (pp * n + j) : (j * k + pp)]));
      }
      worst = std::max(worst, std::abs(acc - double(got[j])) / std::max(1.0, std::abs(acc)));
    }

    const double med = MedianMs(ms);
    std::sort(ms.begin(), ms.end());
    // BYTES MOVED is what this kernel is about, so report GB/s beside GFLOP/s:
    // at M=1 the arithmetic is trivial and the whole cost is streaming b.
    const double gb = double(n) * double(k) * 2.0 / 1e9;
    const double gflop = 2.0 * double(m) * double(n) * double(k) / 1e9;
    std::printf("cycle %3d         : median %8.3f ms  min %8.3f  max %8.3f  "
                "%7.1f GB/s  %7.1f GFLOP/s  buf %p\n",
                c, med, ms.front(), ms.back(), med > 0 ? gb / (med / 1e3) : 0.0,
                med > 0 ? gflop / (med / 1e3) : 0.0, db);
    std::fflush(stdout);
    vk.Free(db);
  }

  std::printf("worst rel err     : %.3e\n", worst);

  vk.Free(da);
  vk.Free(dout);
  vk.DestroyQueue(q);
  return worst < 1e-3 ? 0 : 1;
}
