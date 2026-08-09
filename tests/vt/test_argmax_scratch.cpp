#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
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
  const char* disabled[] = {nullptr, "", "0", "true", "01", "2", " 1"};
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
  CHECK(retired.size() == 8);  // both blocks from each replaced capacity
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
