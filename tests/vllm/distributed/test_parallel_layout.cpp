// TP-W1 unit gate (BACKEND-DISTRIBUTED-TP, tensor-parallelism-spike.md §S4).
// The rank-layout group math must equal vLLM's
// `arange(world_size).reshape(external_dp, dp, pp, pcp, tp)` grouping
// (parallel_state.py:1784-1804): the TP group is `all_ranks.view(-1, tp)` and
// tp_rank = global_rank % tp. The ORACLE here is an INDEPENDENT explicit reshape
// (a 5-D index array materialized in the test) — so this cross-checks the impl's
// stride formula, it is not a tautology. Plus the tp=1 inertness the single-GPU
// path relies on, and the thread-local current-rank accessors.
#include <doctest/doctest.h>

#include <array>
#include <thread>
#include <vector>

#include "vllm/distributed/parallel_layout.h"
#include "vllm/model_executor/models/tensor_parallel.h"

using vllm::distributed::AllTpGroups;
using vllm::distributed::AxisGroup;
using vllm::distributed::ParallelAxis;
using vllm::distributed::ParallelDims;
using vllm::distributed::RankGroup;
using vllm::distributed::TpGroup;

namespace {

// Independent oracle: build arange(world).reshape(edp, dp, pp, pcp, tp) as a
// dense 5-D array (row-major, so tp is innermost), then extract the group along
// `axis` through `global_rank` by fixing every OTHER coordinate. This is the
// literal numpy/torch reshape+index vLLM does, computed a different way than the
// impl's closed-form strides.
std::vector<int> ReshapeAxisGroupOracle(const ParallelDims& d, int global_rank,
                                        int axis /*0=edp..4=tp*/,
                                        int& out_local_rank) {
  const std::array<int, 5> dims = {d.external_dp, d.dp, d.pp, d.pcp, d.tp};
  // Row-major strides for [edp, dp, pp, pcp, tp].
  std::array<int, 5> stride{};
  stride[4] = 1;
  for (int a = 3; a >= 0; --a) stride[a] = stride[a + 1] * dims[a + 1];
  // Decompose global_rank into its 5 coordinates.
  std::array<int, 5> coord{};
  for (int a = 0; a < 5; ++a) coord[a] = (global_rank / stride[a]) % dims[a];
  // Group along `axis`: vary that coordinate over [0, dims[axis]).
  std::vector<int> ranks;
  for (int i = 0; i < dims[axis]; ++i) {
    std::array<int, 5> c = coord;
    c[axis] = i;
    int r = 0;
    for (int a = 0; a < 5; ++a) r += c[a] * stride[a];
    ranks.push_back(r);
  }
  out_local_rank = coord[axis];
  return ranks;
}

}  // namespace

TEST_CASE("TP-W1: world_size == product of the parallel degrees") {
  CHECK(ParallelDims{}.world_size() == 1);
  CHECK((ParallelDims{/*tp=*/2}).world_size() == 2);
  CHECK((ParallelDims{/*tp=*/4}).world_size() == 4);
  CHECK((ParallelDims{/*tp=*/2, /*pcp=*/1, /*pp=*/2, /*dp=*/2}).world_size() ==
        8);
}

TEST_CASE("TP-W1: TP-only groups are one contiguous view(-1, tp)") {
  for (int tp : {1, 2, 4, 8}) {
    ParallelDims d;
    d.tp = tp;
    // TP-only ⇒ world == tp ⇒ exactly one TP group = all ranks, in order.
    auto groups = AllTpGroups(d);
    CHECK(groups.size() == 1);
    REQUIRE(static_cast<int>(groups[0].size()) == tp);
    for (int r = 0; r < tp; ++r) {
      CHECK(groups[0][r] == r);
      RankGroup g = TpGroup(d, r);
      CHECK(g.size() == tp);
      CHECK(g.local_rank == r);   // tp_rank = r % tp = r here
      // whole group present, in order
      for (int i = 0; i < tp; ++i) CHECK(g.ranks[i] == i);
    }
  }
}

TEST_CASE("TP-W1: mixed EDP/DP/PP/PCP/TP layout matches the reshape oracle") {
  // A layout exercising every axis and stride: 2*2*2*1*2 = 16 ranks.
  ParallelDims d{/*tp=*/2, /*pcp=*/1, /*pp=*/2, /*dp=*/2, /*external_dp=*/2};
  REQUIRE(d.world_size() == 16);

  const std::array<ParallelAxis, 5> axes = {
      ParallelAxis::kExternalDp, ParallelAxis::kDp, ParallelAxis::kPp,
      ParallelAxis::kPcp, ParallelAxis::kTp};
  for (int r = 0; r < d.world_size(); ++r) {
    for (int a = 0; a < 5; ++a) {
      int oracle_local = -1;
      std::vector<int> oracle = ReshapeAxisGroupOracle(d, r, a, oracle_local);
      RankGroup got = AxisGroup(d, r, axes[a]);
      CHECK(got.ranks == oracle);        // group membership == reshape oracle
      CHECK(got.local_rank == oracle_local);
      CHECK(got.ranks[got.local_rank] == r);  // this rank is at its local index
    }
  }
}

TEST_CASE("TP-W1: TP groups are contiguous tp-chunks; tp_rank = rank % tp") {
  ParallelDims d{/*tp=*/4, /*pcp=*/1, /*pp=*/1, /*dp=*/2};  // 8 ranks, 2 groups
  REQUIRE(d.world_size() == 8);
  auto groups = AllTpGroups(d);
  REQUIRE(groups.size() == 2);
  CHECK(groups[0] == std::vector<int>{0, 1, 2, 3});
  CHECK(groups[1] == std::vector<int>{4, 5, 6, 7});
  for (int r = 0; r < 8; ++r) {
    RankGroup g = TpGroup(d, r);
    CHECK(g.local_rank == r % 4);
    CHECK(g.ranks[0] == (r / 4) * 4);
  }
}

TEST_CASE("TP-W1: tp=1 inertness — singleton group, default accessors") {
  ParallelDims d;  // all degrees 1
  RankGroup g = TpGroup(d, 0);
  CHECK(g.size() == 1);
  CHECK(g.local_rank == 0);
  CHECK(g.ranks == std::vector<int>{0});

  // No handle set ⇒ single-GPU defaults, byte-identical.
  CHECK(vllm::distributed::CurrentTensorParallel() == nullptr);
  CHECK(vllm::distributed::CurrentTpRank() == 0);
  CHECK(vllm::distributed::CurrentTpWorldSize() == 1);
}

TEST_CASE("TP-W1: thread-local current-rank handle is per-thread") {
  // A null comm ⇒ tp_size()==1/rank()==0 (the single-GPU handle). We only need
  // the accessor plumbing here; the reduction transport is gated elsewhere.
  vllm::TensorParallel single;  // comm == nullptr
  vllm::distributed::SetCurrentTensorParallel(&single);
  CHECK(vllm::distributed::CurrentTensorParallel() == &single);
  CHECK(vllm::distributed::CurrentTpRank() == 0);
  CHECK(vllm::distributed::CurrentTpWorldSize() == 1);

  // A sibling thread starts with a CLEAN (null) slot — thread_local isolation,
  // the property the thread-per-rank executor relies on.
  bool sibling_null = false;
  std::thread t([&] {
    sibling_null = (vllm::distributed::CurrentTensorParallel() == nullptr);
  });
  t.join();
  CHECK(sibling_null);

  vllm::distributed::SetCurrentTensorParallel(nullptr);  // restore
}
