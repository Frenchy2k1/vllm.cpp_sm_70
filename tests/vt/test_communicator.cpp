// vllm.cpp original (vt runtime). Scale-out W1 correctness gate for
// vt::Communicator on the CPU in-process multi-rank transport
// (.agents/specs/scale-out-distributed.md §W1). Proves a REAL cross-rank
// reduction: N host threads each drive their own rank's Communicator over a
// shared barrier, and every rank must observe the exact summed / gathered result.
// RED-first: a Communicator whose AllReduce left `data` unchanged would fail the
// sum assertions here (each rank's input differs from the sum).
#include <doctest/doctest.h>

#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

#include "vt/backend.h"
#include "vt/communicator.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vt::Communicator;
using vt::CpuCommGroup;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::ReduceOp;

namespace {

Queue CpuQueue() { return vt::GetBackend(DeviceType::kCPU).CreateQueue(); }

// Run `body(rank)` on one host thread per rank and join. Each thread owns its
// rank's Communicator + buffers, so the only shared state is the group's barrier.
template <typename Body>
void RunRanks(int world, Body body) {
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) threads.emplace_back(body, r);
  for (auto& t : threads) t.join();
}

}  // namespace

TEST_CASE("world_size==1 AllReduce/AllGather are byte-identical no-ops") {
  CpuCommGroup group(1);
  CHECK(group.world_size() == 1);
  Communicator& c = group.Rank(0);
  CHECK(c.rank() == 0);
  CHECK(c.world_size() == 1);
  Queue q = CpuQueue();

  // AllReduce with one rank: data unchanged, bit-for-bit.
  std::vector<float> data = {1.5f, -2.25f, 3.0f, 42.0f};
  const std::vector<float> before = data;
  c.AllReduce(q, data.data(), data.size(), DType::kF32, ReduceOp::kSum);
  CHECK(data == before);

  // AllGather with one rank: recvbuf == sendbuf, bit-for-bit.
  std::vector<float> recv(data.size(), 0.0f);
  c.AllGather(q, data.data(), recv.data(), data.size(), DType::kF32);
  CHECK(recv == before);
}

TEST_CASE("2-rank AllReduce-sum yields the exact sum on every rank (f32)") {
  CpuCommGroup group(2);
  CHECK(group.world_size() == 2);

  // rank 0 = {1,2,3,4}, rank 1 = {10,20,30,40}; sum = {11,22,33,44} on BOTH.
  std::vector<std::vector<float>> bufs = {{1, 2, 3, 4}, {10, 20, 30, 40}};
  const std::vector<float> expected = {11, 22, 33, 44};

  RunRanks(2, [&](int r) {
    Queue q = CpuQueue();
    group.Rank(r).AllReduce(q, bufs[static_cast<size_t>(r)].data(), 4,
                            DType::kF32, ReduceOp::kSum);
  });

  CHECK(bufs[0] == expected);
  CHECK(bufs[1] == expected);
}

TEST_CASE("4-rank AllReduce-sum yields the exact sum on every rank (f32)") {
  constexpr int kWorld = 4;
  constexpr size_t kN = 5;
  CpuCommGroup group(kWorld);

  // rank r contributes {r+1, r+1, ...}. Sum over ranks 0..3 = 1+2+3+4 = 10.
  std::vector<std::vector<float>> bufs(kWorld,
                                       std::vector<float>(kN, 0.0f));
  for (int r = 0; r < kWorld; ++r)
    for (size_t i = 0; i < kN; ++i)
      bufs[static_cast<size_t>(r)][i] = static_cast<float>(r + 1);

  RunRanks(kWorld, [&](int r) {
    Queue q = CpuQueue();
    group.Rank(r).AllReduce(q, bufs[static_cast<size_t>(r)].data(), kN,
                            DType::kF32, ReduceOp::kSum);
  });

  for (int r = 0; r < kWorld; ++r)
    for (size_t i = 0; i < kN; ++i)
      CHECK(bufs[static_cast<size_t>(r)][i] == doctest::Approx(10.0f));
}

TEST_CASE("AllReduce max/min/prod reduce correctly across 4 ranks") {
  constexpr int kWorld = 4;
  auto make = [] { return std::vector<std::vector<int32_t>>{
                       {5, 1, 9, 2}, {3, 8, 0, 2}, {7, 8, 4, 2}, {1, 6, 9, 2}}; };

  SUBCASE("max") {
    auto bufs = make();
    CpuCommGroup group(kWorld);
    RunRanks(kWorld, [&](int r) {
      Queue q = CpuQueue();
      group.Rank(r).AllReduce(q, bufs[static_cast<size_t>(r)].data(), 4,
                              DType::kI32, ReduceOp::kMax);
    });
    const std::vector<int32_t> expected = {7, 8, 9, 2};
    for (int r = 0; r < kWorld; ++r) CHECK(bufs[static_cast<size_t>(r)] == expected);
  }
  SUBCASE("min") {
    auto bufs = make();
    CpuCommGroup group(kWorld);
    RunRanks(kWorld, [&](int r) {
      Queue q = CpuQueue();
      group.Rank(r).AllReduce(q, bufs[static_cast<size_t>(r)].data(), 4,
                              DType::kI32, ReduceOp::kMin);
    });
    const std::vector<int32_t> expected = {1, 1, 0, 2};
    for (int r = 0; r < kWorld; ++r) CHECK(bufs[static_cast<size_t>(r)] == expected);
  }
}

TEST_CASE("AllReduce-sum works for bf16 and i64") {
  SUBCASE("bf16") {
    CpuCommGroup group(2);
    // 1.0 and 2.0 are exact in bf16; sum 3.0 is exact.
    std::vector<std::vector<uint16_t>> bufs = {
        {vt::F32ToBF16(1.0f), vt::F32ToBF16(2.0f)},
        {vt::F32ToBF16(2.0f), vt::F32ToBF16(4.0f)}};
    RunRanks(2, [&](int r) {
      Queue q = CpuQueue();
      group.Rank(r).AllReduce(q, bufs[static_cast<size_t>(r)].data(), 2,
                              DType::kBF16, ReduceOp::kSum);
    });
    for (int r = 0; r < 2; ++r) {
      CHECK(vt::BF16ToF32(bufs[static_cast<size_t>(r)][0]) == doctest::Approx(3.0f));
      CHECK(vt::BF16ToF32(bufs[static_cast<size_t>(r)][1]) == doctest::Approx(6.0f));
    }
  }
  SUBCASE("i64") {
    CpuCommGroup group(3);
    std::vector<std::vector<int64_t>> bufs = {
        {1, 100}, {10, 200}, {100, 300}};
    RunRanks(3, [&](int r) {
      Queue q = CpuQueue();
      group.Rank(r).AllReduce(q, bufs[static_cast<size_t>(r)].data(), 2,
                              DType::kI64, ReduceOp::kSum);
    });
    const std::vector<int64_t> expected = {111, 600};
    for (int r = 0; r < 3; ++r) CHECK(bufs[static_cast<size_t>(r)] == expected);
  }
}

TEST_CASE("2-rank AllGather concatenates every rank's contribution in order") {
  CpuCommGroup group(2);
  std::vector<std::vector<int32_t>> send = {{1, 2, 3}, {4, 5, 6}};
  std::vector<std::vector<int32_t>> recv(2, std::vector<int32_t>(6, -1));

  RunRanks(2, [&](int r) {
    Queue q = CpuQueue();
    group.Rank(r).AllGather(q, send[static_cast<size_t>(r)].data(),
                            recv[static_cast<size_t>(r)].data(), 3, DType::kI32);
  });

  const std::vector<int32_t> expected = {1, 2, 3, 4, 5, 6};
  CHECK(recv[0] == expected);  // rank 0 sees the full concat...
  CHECK(recv[1] == expected);  // ...and so does rank 1.
}

TEST_CASE("4-rank AllGather places rank r's data at slot r") {
  constexpr int kWorld = 4;
  CpuCommGroup group(kWorld);
  std::vector<std::vector<int32_t>> send(kWorld);
  std::vector<std::vector<int32_t>> recv(kWorld, std::vector<int32_t>(kWorld, -1));
  for (int r = 0; r < kWorld; ++r) send[static_cast<size_t>(r)] = {r * 10};

  RunRanks(kWorld, [&](int r) {
    Queue q = CpuQueue();
    group.Rank(r).AllGather(q, send[static_cast<size_t>(r)].data(),
                            recv[static_cast<size_t>(r)].data(), 1, DType::kI32);
  });

  const std::vector<int32_t> expected = {0, 10, 20, 30};
  for (int r = 0; r < kWorld; ++r) CHECK(recv[static_cast<size_t>(r)] == expected);
}

TEST_CASE("Send/Recv rendezvous transfers a buffer between two ranks") {
  CpuCommGroup group(2);
  const std::vector<int32_t> payload = {11, 22, 33, 44};
  std::vector<int32_t> got(4, 0);

  RunRanks(2, [&](int r) {
    Queue q = CpuQueue();
    if (r == 0) {
      group.Rank(0).Send(q, payload.data(), 4, DType::kI32, /*peer=*/1);
    } else {
      group.Rank(1).Recv(q, got.data(), 4, DType::kI32, /*peer=*/0);
    }
  });

  CHECK(got == payload);
}
