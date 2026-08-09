#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vt/cuda/argmax_scratch.h"

namespace {
using vt::cuda::ArgmaxScratchKey;
using vt::cuda::ArgmaxScratchOwners;
using vt::cuda::ArgmaxScratchView;

void* FakePtr(std::uintptr_t value) {
  return reinterpret_cast<void*>(value);
}

TEST_CASE("argmax geometric scratch selector is strict and dispatches exactly once") {
  const char* disabled[] = {nullptr, "", "0", "true", "01", "2", " 1", "10", "1x", "1 "};
  for (const char* value : disabled) {
    int incumbent = 0;
    int candidate = 0;
    const int selected = vt::cuda::DispatchArgmaxScratch(
        value,
        [&] {
          ++incumbent;
          return 17;
        },
        [&] {
          ++candidate;
          return 29;
        });
    CHECK(selected == 17);
    CHECK(incumbent == 1);
    CHECK(candidate == 0);
  }
  int incumbent = 0;
  int candidate = 0;
  CHECK(vt::cuda::DispatchArgmaxScratch(
            "1",
            [&] {
              ++incumbent;
              return 17;
            },
            [&] {
              ++candidate;
              return 29;
            }) == 29);
  CHECK(incumbent == 0);
  CHECK(candidate == 1);
}

TEST_CASE("argmax geometric scratch grows transactionally by powers of two") {
  ArgmaxScratchOwners owners;
  const ArgmaxScratchKey key{3, 0x1234};
  std::uintptr_t next = 0x1000;
  std::vector<std::size_t> allocations;
  std::vector<void*> cleanups;
  std::vector<void*> retired;
  std::vector<std::size_t> seen_caps;
  auto allocate = [&](std::size_t bytes) {
    allocations.push_back(bytes);
    next += 0x100;
    return FakePtr(next);
  };
  auto submit = [&](ArgmaxScratchView view) { seen_caps.push_back(view.capacity); };

  for (std::size_t need : {1U, 3U, 5U, 9U, 17U, 32U, 7U, 31U}) {
    owners.Submit(
        key, need, [] { return false; }, allocate, [&](void* p) { cleanups.push_back(p); },
        [&](void* p) { retired.push_back(p); }, submit);
  }
  CHECK(seen_caps == std::vector<std::size_t>{1, 4, 8, 16, 32, 32, 32, 32});
  CHECK(allocations == std::vector<std::size_t>{4, 8, 16, 32, 32, 64, 64, 128, 128, 256});
  CHECK(cleanups.empty());
  CHECK(retired == std::vector<void*>{FakePtr(0x1100), FakePtr(0x1200), FakePtr(0x1300),
                                      FakePtr(0x1400), FakePtr(0x1500), FakePtr(0x1600),
                                      FakePtr(0x1700), FakePtr(0x1800)});
  const auto diag = owners.Diagnostics(key);
  REQUIRE(diag.has_value());
  CHECK(diag->capacity == 32);
  CHECK(diag->active_bytes == 32 * (sizeof(float) + sizeof(std::int64_t)));
  CHECK(diag->retired_bytes == (1 + 4 + 8 + 16) * (sizeof(float) + sizeof(std::int64_t)));
  CHECK(diag->active_bytes + diag->retired_bytes < 2 * diag->active_bytes);
}

TEST_CASE("argmax geometric scratch preserves the published pair on allocation failure") {
  ArgmaxScratchOwners owners;
  const ArgmaxScratchKey key{0, 0x44};
  std::uintptr_t next = 0x2000;
  std::vector<void*> cleanups;
  std::vector<void*> retired;
  int calls = 0;
  auto ok_alloc = [&](std::size_t) {
    next += 0x100;
    return FakePtr(next);
  };
  owners.Submit(
      key, 3, [] { return false; }, ok_alloc, [&](void* p) { cleanups.push_back(p); },
      [&](void* p) { retired.push_back(p); }, [&](ArgmaxScratchView) { ++calls; });
  const auto before = *owners.Diagnostics(key);

  int allocation_number = 0;
  CHECK_THROWS_AS(owners.Submit(
                      key, 5, [] { return false; },
                      [&](std::size_t) -> void* {
                        ++allocation_number;
                        if (allocation_number == 2)
                          throw std::runtime_error("second allocation failed");
                        next += 0x100;
                        return FakePtr(next);
                      },
                      [&](void* p) { cleanups.push_back(p); },
                      [&](void* p) { retired.push_back(p); }, [&](ArgmaxScratchView) { ++calls; }),
                  std::runtime_error);
  const auto after = *owners.Diagnostics(key);
  CHECK(after.values == before.values);
  CHECK(after.indices == before.indices);
  CHECK(after.capacity == before.capacity);
  CHECK(cleanups == std::vector<void*>{FakePtr(0x2300)});
  CHECK(retired.empty());
  CHECK(calls == 1);
}

TEST_CASE("argmax geometric scratch rejects a null first allocation transactionally") {
  ArgmaxScratchOwners owners;
  const ArgmaxScratchKey absent{4, 0x404};
  int launches = 0;
  int retirements = 0;
  CHECK_THROWS_AS(owners.Submit(
                      absent, 3, [] { return false; }, [](std::size_t) -> void* { return nullptr; },
                      [](void*) { FAIL("null first allocation must not need cleanup"); },
                      [&](void*) { ++retirements; }, [&](ArgmaxScratchView) { ++launches; }),
                  std::runtime_error);
  CHECK(owners.OwnerCount() == 0);
  CHECK_FALSE(owners.Diagnostics(absent).has_value());
  CHECK(retirements == 0);
  CHECK(launches == 0);

  const ArgmaxScratchKey existing{4, 0x405};
  std::uintptr_t next = 0x5000;
  owners.Submit(
      existing, 3, [] { return false; },
      [&](std::size_t) {
        next += 0x100;
        return FakePtr(next);
      },
      [](void*) {}, [&](void*) { ++retirements; }, [&](ArgmaxScratchView) { ++launches; });
  const auto before = *owners.Diagnostics(existing);
  CHECK_THROWS_AS(
      owners.Submit(
          existing, 5, [] { return false; }, [](std::size_t) -> void* { return nullptr; },
          [](void*) { FAIL("null first allocation must not need cleanup"); },
          [&](void*) { ++retirements; }, [&](ArgmaxScratchView) { ++launches; }),
      std::runtime_error);
  const auto after = *owners.Diagnostics(existing);
  CHECK(after.values == before.values);
  CHECK(after.indices == before.indices);
  CHECK(after.capacity == before.capacity);
  CHECK(after.growths == before.growths);
  CHECK(retirements == 0);
  CHECK(launches == 1);
}

TEST_CASE("CUDA argmax adapter binds device stream capture allocation and retirement") {
  struct FakeQueue {
    struct {
      int index;
    } device;
  } queue{{7}};
  struct CapabilityLimitedRuntime {
    bool capturing = false;
    std::uintptr_t next = 0x6000;
    std::vector<std::uintptr_t> capture_streams;
    std::vector<std::pair<std::size_t, std::uintptr_t>> allocations;
    std::vector<std::pair<void*, std::uintptr_t>> partial_cleanups;
    std::vector<void*> retirements;

    bool IsCapturing(void* stream) {
      capture_streams.push_back(reinterpret_cast<std::uintptr_t>(stream));
      return capturing;
    }
    void* Allocate(std::size_t bytes, void* stream) {
      allocations.emplace_back(bytes, reinterpret_cast<std::uintptr_t>(stream));
      next += 0x100;
      return FakePtr(next);
    }
    void CleanupPartial(void* p, void* stream) {
      partial_cleanups.emplace_back(p, reinterpret_cast<std::uintptr_t>(stream));
    }
    void Retire(void* p) { retirements.push_back(p); }
  } runtime;

  ArgmaxScratchOwners owners;
  void* const stream = FakePtr(0x777);
  std::vector<ArgmaxScratchView> launches;
  vt::cuda::SubmitCudaArgmaxScratch(owners, queue, stream, 3, runtime,
                                    [&](ArgmaxScratchView view) { launches.push_back(view); });
  vt::cuda::SubmitCudaArgmaxScratch(owners, queue, stream, 5, runtime,
                                    [&](ArgmaxScratchView view) { launches.push_back(view); });
  CHECK(owners.OwnerCount() == 1);
  CHECK(owners.Diagnostics(ArgmaxScratchKey{7, 0x777})->capacity == 8);
  CHECK_FALSE(owners.Diagnostics(ArgmaxScratchKey{0, 0x777}).has_value());
  CHECK_FALSE(owners.Diagnostics(ArgmaxScratchKey{7, 0}).has_value());
  CHECK(runtime.capture_streams == std::vector<std::uintptr_t>{0x777, 0x777});
  CHECK(runtime.allocations == std::vector<std::pair<std::size_t, std::uintptr_t>>{
                                   {16, 0x777}, {32, 0x777}, {32, 0x777}, {64, 0x777}});
  CHECK(runtime.partial_cleanups.empty());
  CHECK(runtime.retirements == std::vector<void*>{FakePtr(0x6100), FakePtr(0x6200)});
  REQUIRE(launches.size() == 2);
  CHECK(launches[0].capacity == 4);
  CHECK(launches[1].capacity == 8);

  CapabilityLimitedRuntime captured;
  captured.capturing = true;
  CHECK_THROWS_AS(vt::cuda::SubmitCudaArgmaxScratch(
                      owners, queue, FakePtr(0x778), 1, captured,
                      [](ArgmaxScratchView) { FAIL("capture miss must not launch"); }),
                  std::runtime_error);
  CHECK(captured.capture_streams == std::vector<std::uintptr_t>{0x778});
  CHECK(captured.allocations.empty());
  CHECK(captured.retirements.empty());
  CHECK_FALSE(owners.Diagnostics(ArgmaxScratchKey{7, 0x778}).has_value());
}

TEST_CASE("CUDA argmax call site binds the capability adapter to graph-safe runtime operations") {
  std::ifstream input(std::string(VLLM_CPP_SOURCE_DIR) + "/src/vt/cuda/cuda_sample.cu");
  REQUIRE(input.good());
  const std::string source((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  const auto runtime_begin = source.find("struct CudaArgmaxScratchRuntime {");
  const auto runtime_end = source.find("\n};", runtime_begin);
  REQUIRE(runtime_begin != std::string::npos);
  REQUIRE(runtime_end != std::string::npos);
  const std::string runtime = source.substr(runtime_begin, runtime_end - runtime_begin);
  CHECK(runtime.find("cudaStreamIsCapturing(stream, &status)") != std::string::npos);
  CHECK(runtime.find("return status != cudaStreamCaptureStatusNone") != std::string::npos);
  CHECK(runtime.find("cudaMallocAsync(&p, bytes, stream)") != std::string::npos);
  CHECK(runtime.find("cudaFreeAsync(p, stream)") != std::string::npos);
  CHECK(runtime.find("void Retire(void* p) const { RetireGraphScratch(p); }") !=
        std::string::npos);

  const auto route = source.find("SubmitCudaArgmaxScratch(", runtime_end);
  REQUIRE(route != std::string::npos);
  const std::string invocation = source.substr(route, 180);
  CHECK(invocation.find("g_argmax_geometric_owners, q, s, need, runtime") != std::string::npos);
}

TEST_CASE("argmax geometric scratch isolates keys and fails capture misses closed") {
  ArgmaxScratchOwners owners;
  std::uintptr_t next = 0x3000;
  int allocations = 0;
  int submits = 0;
  auto allocate = [&](std::size_t) {
    ++allocations;
    next += 0x100;
    return FakePtr(next);
  };
  auto cleanup = [](void*) {};
  auto retire = [](void*) {};
  auto submit = [&](ArgmaxScratchView) { ++submits; };
  const ArgmaxScratchKey a{0, 0x99};
  const ArgmaxScratchKey b{1, 0x99};
  const ArgmaxScratchKey c{0, 0x100};
  owners.Submit(a, 4, [] { return false; }, allocate, cleanup, retire, submit);
  owners.Submit(b, 4, [] { return false; }, allocate, cleanup, retire, submit);
  owners.Submit(c, 4, [] { return false; }, allocate, cleanup, retire, submit);
  CHECK(owners.OwnerCount() == 3);
  CHECK(owners.Diagnostics(a)->values != owners.Diagnostics(b)->values);
  CHECK(owners.Diagnostics(a)->values != owners.Diagnostics(c)->values);

  int capture_queries = 0;
  owners.Submit(
      a, 4,
      [&] {
        ++capture_queries;
        return true;
      },
      allocate, cleanup, retire, submit);
  CHECK(capture_queries == 0);  // a warmed hit is capture-safe
  CHECK_THROWS_AS(owners.Submit(
                      a, 5,
                      [&] {
                        ++capture_queries;
                        return true;
                      },
                      allocate, cleanup, retire, submit),
                  std::runtime_error);
  CHECK(capture_queries == 1);
  CHECK(allocations == 6);
  CHECK(submits == 4);
  CHECK(owners.Diagnostics(a)->capacity == 4);

  CHECK_THROWS_AS(
      owners.Submit(
          ArgmaxScratchKey{7, 0x777}, 1, [] { return true; }, allocate, cleanup, retire, submit),
      std::runtime_error);
  CHECK(owners.OwnerCount() == 3);  // an absent-owner miss changes no state
}

TEST_CASE("argmax geometric scratch submit lock spans the complete launch callback") {
  ArgmaxScratchOwners owners;
  const ArgmaxScratchKey key{0, 0x55};
  std::uintptr_t next = 0x4000;
  auto allocate = [&](std::size_t) {
    next += 0x100;
    return FakePtr(next);
  };
  owners.Submit(
      key, 4, [] { return false; }, allocate, [](void*) {}, [](void*) {}, [](ArgmaxScratchView) {});

  std::mutex mu;
  std::condition_variable cv;
  bool first_between_kernels = false;
  bool release_first = false;
  std::atomic<bool> second_entered{false};
  std::thread first([&] {
    owners.Submit(
        key, 4, [] { return false; }, allocate, [](void*) {}, [](void*) {},
        [&](ArgmaxScratchView) {
          std::unique_lock lock(mu);
          first_between_kernels = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_first; });
        });
  });
  {
    std::unique_lock lock(mu);
    cv.wait(lock, [&] { return first_between_kernels; });
  }
  std::thread second([&] {
    owners.Submit(
        key, 4, [] { return false; }, allocate, [](void*) {}, [](void*) {},
        [&](ArgmaxScratchView) { second_entered = true; });
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK_FALSE(second_entered.load());
  {
    std::lock_guard lock(mu);
    release_first = true;
  }
  cv.notify_all();
  first.join();
  second.join();
  CHECK(second_entered.load());
}

TEST_CASE("argmax geometric scratch rejects capacity and byte overflow") {
  CHECK(vt::cuda::RoundArgmaxScratchCapacity(17) == 32);
  CHECK(vt::cuda::RoundArgmaxScratchCapacity(32) == 32);
  CHECK_THROWS_AS(vt::cuda::RoundArgmaxScratchCapacity(std::numeric_limits<std::size_t>::max()),
                  std::overflow_error);
  ArgmaxScratchOwners owners;
  int allocations = 0;
  CHECK_THROWS_AS(owners.Submit(
                      ArgmaxScratchKey{0, 1}, std::numeric_limits<std::size_t>::max() / 4 + 1,
                      [] { return false; },
                      [&](std::size_t) {
                        ++allocations;
                        return FakePtr(1);
                      },
                      [](void*) {}, [](void*) {}, [](ArgmaxScratchView) {}),
                  std::overflow_error);
  CHECK(allocations == 0);
}
}  // namespace
