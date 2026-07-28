// Tensor-parallel wiring for the dense model forward (BACKEND-DISTRIBUTED-TP W2,
// .agents/specs/scale-out-distributed.md §W2). ADDITIVE + default-inert: a
// `TensorParallel*` threads a vt::Communicator through the forward and weight
// loader; when it is null OR tp_size()==1 EVERY helper is a byte-identical no-op
// (the whole-tensor shard, the all-reduce that never runs), so the single-GPU
// path — and every SACRED text-model gate — is untouched.
//
// Ground (vLLM): the RowParallelLinear partial-product all-reduce
// linear.py:1765-1766 (`tensor_model_parallel_all_reduce`, gated by
// reduce_results); the ColumnParallelLinear output-dim shard linear.py:478,569;
// GroupCoordinator BYPASSES every collective at world_size==1
// (parallel_state.py:638) — which is exactly the tp_size()==1 early return here.
#pragma once

#include <cstdint>

#include "vt/communicator.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

// A tensor-parallel group as seen by one rank: the bound Communicator (which
// carries rank()/world_size() + the transport) plus the parallel degree. A null
// `comm` (or a world_size==1 comm) is a tp_size==1 group — the single-GPU engine.
struct TensorParallel {
  vt::Communicator* comm = nullptr;

  int tp_size() const { return comm != nullptr ? comm->world_size() : 1; }
  int rank() const { return comm != nullptr ? comm->rank() : 0; }
};

// A half-open output/input row range [begin, end) for a column/row-parallel
// weight shard. At tp_size==1 it is the WHOLE dimension.
struct ShardRange {
  int64_t begin = 0;
  int64_t end = 0;
  int64_t size() const { return end - begin; }
};

// This rank's contiguous shard of `dim` rows under `tp`. Mirrors vLLM's
// `divide(output_size, tp_size)` even split (linear.py:478); `dim` MUST be
// divisible by tp_size (vLLM asserts the same, utils.py::divide). tp_size==1 ⇒
// [0, dim) — the whole tensor, so the loader memcpy is byte-identical.
inline ShardRange TpShard(const TensorParallel* tp, int64_t dim) {
  const int world = tp != nullptr ? tp->tp_size() : 1;
  if (world == 1) return {0, dim};
  const int64_t per = dim / world;
  const int64_t begin = per * tp->rank();
  return {begin, begin + per};
}

// RowParallel partial-product all-reduce over the TP group (linear.py:1766). The
// sharded matmuls on each rank produce partial sums of the SAME [.,out] tensor;
// this sums them so every rank holds the full result. No-op — not one vt:: op
// enqueued — when `tp` is null or tp_size()==1 (the byte-identical single-GPU
// path, parallel_state.py:638 bypass).
inline void TpAllReduceSum(const TensorParallel* tp, vt::Queue& q,
                           vt::Tensor& t) {
  if (tp == nullptr || tp->tp_size() == 1) return;
  tp->comm->AllReduce(q, t.data, static_cast<size_t>(t.Numel()), t.dtype,
                      vt::ReduceOp::kSum);
}

}  // namespace vllm
