// Tensor-parallel rank-layout / group table (BACKEND-DISTRIBUTED-TP TP-W1,
// .agents/specs/tensor-parallelism-spike.md §S4). The GroupCoordinator-analog:
// the pure rank-layout math that turns a flat (rank, world_size) into the TP
// group a rank belongs to, plus thread-local current-rank accessors.
//
// Ground (vLLM @ 555967922):
//   * ParallelConfig degrees + world_size = external_dp * dp * pp * pcp * tp
//     (config/parallel.py:124, :824-828).
//   * initialize_model_parallel lays ranks out as
//     `arange(world_size).reshape(external_dp, dp, pp, pcp, tp)`
//     (parallel_state.py:1718, :1784-1793) — ROW-MAJOR, so TP is the INNERMOST
//     (fastest-varying) axis — and the TP groups are `all_ranks.view(-1, tp)`
//     (:1804): each row is one TP group of `tp` CONTIGUOUS global ranks.
//   * GroupCoordinator BYPASSES every collective at world_size==1
//     (parallel_state.py:638) — mirrored by tp_size()==1 staying a no-op.
//
// This is pure integer arithmetic (no comm, no device): the reshape strides make
// the group along ANY axis mechanical, so PP/DP groups (later bricks) reuse the
// same primitive. TP=1 is byte-identical: the group is the singleton {rank}.
//
// Orchestration note (spec S2a): we run one THREAD per rank in one process, so
// vLLM's process-global get_tp_group()/get_tensor_model_parallel_rank()
// (parallel_state.py:1365,2044) becomes a thread_local handle set by the
// executor per rank-thread. Null on the single-GPU path (tp=1).
#pragma once

#include <vector>

#include "vllm/model_executor/models/tensor_parallel.h"

namespace vllm::distributed {

// Mirror of vLLM ParallelConfig's parallel degrees. `pcp` is
// pipeline-context-parallel; it is carried purely for layout fidelity so the
// reshape strides match upstream even when only TP is used. TP-only ⇒ all other
// degrees are 1 and world_size() == tp.
struct ParallelDims {
  int tp = 1;           // tensor-parallel (innermost reshape axis)
  int pcp = 1;          // pipeline-context-parallel
  int pp = 1;           // pipeline-parallel
  int dp = 1;           // data-parallel
  int external_dp = 1;  // external data-parallel (outermost)

  int world_size() const { return external_dp * dp * pp * pcp * tp; }
};

// A rank's membership in one parallel group: the ordered global ranks that form
// the group, plus this rank's index (local rank) within it. For TP, `size()`
// is the parallel degree and `local_rank` is the tp_rank.
struct RankGroup {
  std::vector<int> ranks;  // global ranks in the group, in group order
  int local_rank = 0;      // this rank's position within `ranks`

  int size() const { return static_cast<int>(ranks.size()); }
};

// The parallel axes of the `arange(world).reshape(external_dp, dp, pp, pcp, tp)`
// layout, outer→inner. TP is innermost (stride 1).
enum class ParallelAxis { kExternalDp, kDp, kPp, kPcp, kTp };

namespace detail {
// (size, stride) of `axis` in the row-major reshape. Stride = product of the
// sizes of every axis INNER to it (tp is innermost, stride 1).
inline void AxisSizeStride(const ParallelDims& d, ParallelAxis axis, int& size,
                           int& stride) {
  switch (axis) {
    case ParallelAxis::kTp: size = d.tp; stride = 1; return;
    case ParallelAxis::kPcp: size = d.pcp; stride = d.tp; return;
    case ParallelAxis::kPp: size = d.pp; stride = d.tp * d.pcp; return;
    case ParallelAxis::kDp: size = d.dp; stride = d.tp * d.pcp * d.pp; return;
    case ParallelAxis::kExternalDp:
      size = d.external_dp;
      stride = d.tp * d.pcp * d.pp * d.dp;
      return;
  }
  size = 1;
  stride = 1;
}
}  // namespace detail

// The group along `axis` containing `global_rank`: the ranks that share every
// OTHER axis index, varying only `axis`. Mirrors initialize_model_parallel
// building each group from the reshaped index array (parallel_state.py:1718).
// For ParallelAxis::kTp this is the contiguous run [(r/tp)*tp, +tp) with
// local_rank = r % tp — exactly all_ranks.view(-1, tp) (:1804).
inline RankGroup AxisGroup(const ParallelDims& d, int global_rank,
                           ParallelAxis axis) {
  int size = 1, stride = 1;
  detail::AxisSizeStride(d, axis, size, stride);
  const int idx = (global_rank / stride) % size;   // this rank's index on axis
  const int base = global_rank - idx * stride;      // first member of the group
  RankGroup g;
  g.ranks.reserve(size);
  for (int i = 0; i < size; ++i) g.ranks.push_back(base + i * stride);
  g.local_rank = idx;
  return g;
}

// The TP group of `global_rank` (the gated case): contiguous `tp` ranks,
// tp_rank = global_rank % tp.
inline RankGroup TpGroup(const ParallelDims& d, int global_rank) {
  return AxisGroup(d, global_rank, ParallelAxis::kTp);
}

// The PP / DP group of `global_rank` (mechanical, for later PP/DP bricks).
inline RankGroup PpGroup(const ParallelDims& d, int global_rank) {
  return AxisGroup(d, global_rank, ParallelAxis::kPp);
}
inline RankGroup DpGroup(const ParallelDims& d, int global_rank) {
  return AxisGroup(d, global_rank, ParallelAxis::kDp);
}

// Every TP group — the full `all_ranks.view(-1, tp)` (parallel_state.py:1804),
// in row-major order over the outer axes. There are world_size/tp of them, each
// a contiguous `tp`-chunk.
inline std::vector<std::vector<int>> AllTpGroups(const ParallelDims& d) {
  std::vector<std::vector<int>> groups;
  const int world = d.world_size();
  const int tp = d.tp;
  for (int base = 0; base < world; base += tp) {
    std::vector<int> g;
    g.reserve(tp);
    for (int i = 0; i < tp; ++i) g.push_back(base + i);
    groups.push_back(std::move(g));
  }
  return groups;
}

// ── Thread-local current-rank TP handle ─────────────────────────────────────
// Our thread-per-rank analog of vLLM's process-global get_tp_group()
// (parallel_state.py:1365). The executor sets this on each rank-thread before
// driving that rank's forward; layers that need get_tensor_model_parallel_rank()
// -style access read it without threading the handle. Null ⇒ tp=1 (single-GPU),
// so CurrentTpRank()==0 and CurrentTpWorldSize()==1 — byte-identical.
namespace detail {
inline const TensorParallel*& CurrentTpSlot() {
  thread_local const TensorParallel* slot = nullptr;
  return slot;
}
}  // namespace detail

inline void SetCurrentTensorParallel(const TensorParallel* tp) {
  detail::CurrentTpSlot() = tp;
}
inline const TensorParallel* CurrentTensorParallel() {
  return detail::CurrentTpSlot();
}
inline int CurrentTpRank() {
  const TensorParallel* tp = detail::CurrentTpSlot();
  return tp != nullptr ? tp->rank() : 0;
}
inline int CurrentTpWorldSize() {
  const TensorParallel* tp = detail::CurrentTpSlot();
  return tp != nullptr ? tp->tp_size() : 1;
}

}  // namespace vllm::distributed
