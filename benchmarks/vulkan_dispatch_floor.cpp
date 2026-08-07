// What does ONE Vulkan dispatch cost when it does almost nothing?
// BACKEND-VULKAN work row VK-A2.
//
// WHY THIS EXISTS. An e2e profile of Qwen3-0.6B showed 2,952 dispatches taking
// ~1,054 ms of a 1,092 ms run, at a steady ~0.357 ms each. That number alone
// cannot say WHY: 0.357 ms of real kernel execution and 0.357 ms of
// submit-plus-fence overhead produce exactly the same wall clock, and they imply
// opposite fixes -- faster kernels versus fewer submissions. Guessing between
// them is how this campaign already published three wrong attributions.
//
// THE SEPARATION. Run the SAME op through the SAME path while varying only the
// element count, across a range wide enough that compute must dominate at the
// top. Two outcomes, and they cannot be confused:
//
//   * time roughly FLAT across the sweep -> the cost is per-dispatch overhead,
//     paid whether or not there is work to do. Batching is the lever.
//   * time roughly PROPORTIONAL to n -> the cost is real compute or bandwidth,
//     and batching would buy nothing.
//
// The crossover -- the size at which a dispatch finally costs more than being
// dispatched at all -- is the actual deliverable, because it says which of a
// model's ops are worth batching and which are already earning their submission.
//
// kAdd is deliberately the cheapest op available: two reads and a write per
// element, no reduction, no shared memory. It is the closest thing to measuring
// the harness rather than the kernel.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
  const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
  const int warmup = 10;

  if (!vt::vulkan::VulkanDeviceAvailable()) {
    std::fprintf(stderr, "no Vulkan device\n");
    return 2;
  }
  auto& ctx = vt::vulkan::VulkanContext::Get();
  vt::Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  std::printf("device   : %s\n", ctx.device_name().c_str());
  std::printf("op       : kAdd (f32), %d reps per size, %d warm-up\n", reps, warmup);
  std::printf("\n");
  std::printf("%12s %12s %12s %12s\n", "elements", "ms/dispatch", "GB/s", "vs floor");
  std::printf("%12s %12s %12s %12s\n", "--------", "-----------", "----", "--------");

  // 256 elements is one workgroup's worth of work -- as close to an empty
  // dispatch as this op can get. 16M is far past the point where a memory-bound
  // elementwise kernel must be limited by bandwidth on any real device.
  const int64_t sizes[] = {256,      1024,     4096,     16384,    65536,
                           262144,   1048576,  4194304,  16777216};

  // Allocated ONCE at the largest size and reused, so allocation cost never
  // lands inside a timed region and every arm dispatches over identical memory.
  const int64_t max_n = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1];
  std::vector<float> host(max_n, 1.5f);
  void* da = vk.Alloc(max_n * sizeof(float));
  void* db = vk.Alloc(max_n * sizeof(float));
  void* dout = vk.Alloc(max_n * sizeof(float));
  vk.Copy(q, da, host.data(), max_n * sizeof(float));
  vk.Copy(q, db, host.data(), max_n * sizeof(float));
  vk.Synchronize(q);

  double floor_ms = 0.0;
  for (int64_t n : sizes) {
    Tensor ta = Tensor::Contiguous(da, DType::kF32, d, {n});
    Tensor tb = Tensor::Contiguous(db, DType::kF32, d, {n});
    Tensor to = Tensor::Contiguous(dout, DType::kF32, d, {n});

    for (int i = 0; i < warmup; ++i) vt::Add(q, to, ta, tb);
    vk.Synchronize(q);

    std::vector<double> ms;
    ms.reserve(reps);
    for (int i = 0; i < reps; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      vt::Add(q, to, ta, tb);
      vk.Synchronize(q);
      const auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const double med = MedianMs(ms);
    if (floor_ms == 0.0) floor_ms = med;
    // Three arrays touched per element: two read, one written.
    const double gbps = med > 0 ? (3.0 * double(n) * sizeof(float)) / (med / 1e3) / 1e9 : 0.0;
    std::printf("%12lld %12.4f %12.2f %11.2fx\n", (long long)n, med, gbps,
                floor_ms > 0 ? med / floor_ms : 0.0);
  }

  std::printf("\n");
  std::printf("READING IT. The sweep spans 65,536x in work. If ms/dispatch is flat\n");
  std::printf("across it, that flat value IS the per-dispatch overhead, and every op\n");
  std::printf("smaller than the crossover is paying it in full regardless of size.\n");

  vk.Free(da);
  vk.Free(db);
  vk.Free(dout);
  vk.DestroyQueue(q);
  return 0;
}
