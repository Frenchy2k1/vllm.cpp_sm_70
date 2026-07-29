// Scale-out W2 tensor-parallel CORRECTNESS gate (BACKEND-DISTRIBUTED-TP,
// .agents/specs/scale-out-distributed.md §W2). This is the REAL TP gate that
// needs NO GPU: it proves the sharded-matmul + all-reduce wiring computes the
// SAME result as the unsharded (tp=1) forward, using the W1 CPU in-process
// communicator as the transport and the ACTUAL W2 helpers
// (vllm::TpShard / vllm::TpAllReduceSum, tensor_parallel.h).
//
// The algebra it certifies (the whole point of TP):
//   full MLP:  y = relu(x @ W1^T) @ W2^T
//   ColumnParallel W1 (shard rows of I) -> each rank computes its I-columns of
//     the intermediate a with NO communication;
//   RowParallel  W2 (shard cols of I) -> each rank computes a PARTIAL [T,H]
//     product; an ALL-REDUCE(sum) over ranks yields the full y (linear.py:1766).
// So `sharded + all-reduce == full`.
//
// RED-first: the SUBCASE "missing all-reduce yields the WRONG partial output"
// asserts that a rank's pre-all-reduce buffer does NOT equal the full result —
// i.e. dropping the RowParallel all-reduce is observably wrong here.
#include <doctest/doctest.h>

#include <cstdint>
#include <thread>
#include <vector>

#include "vllm/model_executor/models/tensor_parallel.h"
#include "vt/backend.h"
#include "vt/communicator.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::TensorParallel;
using vllm::TpAllReduceSum;
using vllm::TpShard;
using vt::CpuCommGroup;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Queue CpuQueue() { return vt::GetBackend(DeviceType::kCPU).CreateQueue(); }

// y[T,N] += x[T,Kx] @ W[N,K] restricted to input columns [k0,k0+K): the piece a
// RowParallel rank owns (its K-slice of the contraction). W is row-major [N,K].
// Accumulates into y so callers can build a partial or a full product.
void MatmulAcc(std::vector<float>& y, const std::vector<float>& x, int64_t T,
               int64_t Kx, const std::vector<float>& w, int64_t N, int64_t K,
               int64_t xk0) {
  for (int64_t t = 0; t < T; ++t)
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += x[t * Kx + (xk0 + k)] * w[n * K + k];
      y[t * N + n] += acc;
    }
}

Tensor CpuF32(void* data, std::initializer_list<int64_t> shape) {
  return Tensor::Contiguous(data, DType::kF32, vt::Device{DeviceType::kCPU, 0},
                            shape);
}

// The reference (unsharded, tp=1) MLP: y = relu(x @ W1^T) @ W2^T.
//   x  : [T,H]   W1 : [I,H] (intermediate rows)   W2 : [H,I]
std::vector<float> FullMlp(const std::vector<float>& x, int64_t T, int64_t H,
                           int64_t I, const std::vector<float>& w1,
                           const std::vector<float>& w2) {
  std::vector<float> a(static_cast<size_t>(T * I), 0.0f);
  MatmulAcc(a, x, T, H, w1, I, H, 0);  // a = x @ W1^T, [T,I]
  for (float& v : a) v = v > 0.0f ? v : 0.0f;  // relu
  std::vector<float> y(static_cast<size_t>(T * H), 0.0f);
  MatmulAcc(y, a, T, I, w2, H, I, 0);  // y = a @ W2^T, [T,H]
  return y;
}

}  // namespace

TEST_CASE("TP-2 sharded MLP + all-reduce == unsharded tp=1 (the TP gate)") {
  const int64_t T = 3, H = 4, I = 8;  // I divisible by tp=2
  // Deterministic pseudo-random weights/inputs (fixed, portable).
  auto fill = [](std::vector<float>& v, uint32_t seed) {
    for (size_t i = 0; i < v.size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      v[i] = static_cast<float>(static_cast<int32_t>(seed >> 9) % 17 - 8) * 0.25f;
    }
  };
  std::vector<float> x(static_cast<size_t>(T * H));
  std::vector<float> w1(static_cast<size_t>(I * H));  // [I,H]
  std::vector<float> w2(static_cast<size_t>(H * I));  // [H,I]
  fill(x, 1); fill(w1, 2); fill(w2, 3);

  const std::vector<float> y_full = FullMlp(x, T, H, I, w1, w2);

  constexpr int kTp = 2;
  CpuCommGroup group(kTp);

  // Each rank's result buffer, captured for comparison after the threads join.
  std::vector<std::vector<float>> y_rank(kTp);
  std::vector<std::vector<float>> a_shard(kTp);  // ColumnParallel intermediate

  std::vector<std::thread> threads;
  for (int r = 0; r < kTp; ++r) {
    threads.emplace_back([&, r] {
      Queue q = CpuQueue();
      TensorParallel tp{&group.Rank(r)};
      REQUIRE(tp.tp_size() == kTp);

      // ColumnParallel W1: this rank owns intermediate rows [begin,end).
      const vllm::ShardRange cs = TpShard(&tp, I);
      const int64_t Ir = cs.size();
      std::vector<float> w1_r(static_cast<size_t>(Ir * H));
      for (int64_t i = 0; i < Ir; ++i)
        for (int64_t h = 0; h < H; ++h)
          w1_r[i * H + h] = w1[(cs.begin + i) * H + h];
      std::vector<float> a_r(static_cast<size_t>(T * Ir), 0.0f);
      MatmulAcc(a_r, x, T, H, w1_r, Ir, H, 0);  // NO communication
      for (float& v : a_r) v = v > 0.0f ? v : 0.0f;
      a_shard[r] = a_r;

      // RowParallel W2: this rank owns input columns [begin,end) of W2 -> a
      // PARTIAL [T,H] product.
      std::vector<float> w2_r(static_cast<size_t>(H * Ir));
      for (int64_t h = 0; h < H; ++h)
        for (int64_t i = 0; i < Ir; ++i)
          w2_r[h * Ir + i] = w2[h * I + (cs.begin + i)];
      std::vector<float> partial(static_cast<size_t>(T * H), 0.0f);
      MatmulAcc(partial, a_r, T, Ir, w2_r, H, Ir, 0);

      // The RowParallel all-reduce: sum partials -> full y on every rank.
      Tensor pt = CpuF32(partial.data(), {T, H});
      TpAllReduceSum(&tp, q, pt);
      y_rank[r] = partial;
    });
  }
  for (auto& t : threads) t.join();

  // Every rank now holds the FULL result, equal to the unsharded forward.
  for (int r = 0; r < kTp; ++r) {
    REQUIRE(y_rank[r].size() == y_full.size());
    for (size_t i = 0; i < y_full.size(); ++i)
      CHECK(y_rank[r][i] == doctest::Approx(y_full[i]));
  }

  // ColumnParallel sanity: the two rank shards concatenated == the full relu(a).
  std::vector<float> a_full(static_cast<size_t>(T * I), 0.0f);
  MatmulAcc(a_full, x, T, H, w1, I, H, 0);
  for (float& v : a_full) v = v > 0.0f ? v : 0.0f;
  for (int64_t t = 0; t < T; ++t)
    for (int r = 0; r < kTp; ++r) {
      const int64_t Ir = I / kTp;
      for (int64_t i = 0; i < Ir; ++i)
        CHECK(a_shard[r][t * Ir + i] ==
              doctest::Approx(a_full[t * I + r * Ir + i]));
    }

  SUBCASE("RED line: a rank's PARTIAL (pre-all-reduce) is NOT the full output") {
    // Recompute rank 0's partial WITHOUT the all-reduce; it must differ from the
    // full result (otherwise a dropped RowParallel all-reduce would go unnoticed).
    const int64_t Ir = I / kTp;
    std::vector<float> w1_0(static_cast<size_t>(Ir * H));
    for (int64_t i = 0; i < Ir; ++i)
      for (int64_t h = 0; h < H; ++h) w1_0[i * H + h] = w1[i * H + h];
    std::vector<float> a0(static_cast<size_t>(T * Ir), 0.0f);
    MatmulAcc(a0, x, T, H, w1_0, Ir, H, 0);
    for (float& v : a0) v = v > 0.0f ? v : 0.0f;
    std::vector<float> w2_0(static_cast<size_t>(H * Ir));
    for (int64_t h = 0; h < H; ++h)
      for (int64_t i = 0; i < Ir; ++i) w2_0[h * Ir + i] = w2[h * I + i];
    std::vector<float> partial0(static_cast<size_t>(T * H), 0.0f);
    MatmulAcc(partial0, a0, T, Ir, w2_0, H, Ir, 0);

    bool differs = false;
    for (size_t i = 0; i < partial0.size(); ++i)
      if (partial0[i] != doctest::Approx(y_full[i])) differs = true;
    CHECK(differs);
  }
}

TEST_CASE("tp=1 TensorParallel is inert: TpShard is the whole tensor, all-reduce a no-op") {
  TensorParallel tp1;  // null comm => tp_size 1
  CHECK(tp1.tp_size() == 1);
  CHECK(tp1.rank() == 0);
  const vllm::ShardRange r = TpShard(&tp1, 128);
  CHECK(r.begin == 0);
  CHECK(r.end == 128);
  CHECK(r.size() == 128);

  // TpAllReduceSum with a tp=1 group leaves the buffer bit-for-bit unchanged.
  Queue q = CpuQueue();
  std::vector<float> buf = {1.5f, -2.0f, 3.25f, 4.0f};
  const std::vector<float> before = buf;
  Tensor t = CpuF32(buf.data(), {4});
  TpAllReduceSum(&tp1, q, t);
  CHECK(buf == before);

  // A nullptr TensorParallel* (the production single-GPU default) is also inert.
  TpAllReduceSum(nullptr, q, t);
  CHECK(buf == before);
}
