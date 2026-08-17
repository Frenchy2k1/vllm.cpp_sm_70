// vllm.cpp original (vt runtime). Multi-device (TP) phase-3 gate: a REAL
// in-process NCCL collective over the box's discrete GPUs. Calls
// vt_cuda_nccl_group_selfcheck() (in nccl_communicator.cu, VT_NCCL): builds one
// NcclCommunicator per local GPU via ncclCommInitAll, runs an AllReduce +
// AllGather on each device's stream (per-device affinity), and diffs against
// the host truth (allreduce sum == ngpu*(ngpu+1)/2; allgather == ordered
// [1..ngpu]). RED-first: any dropped/incorrect collective shows up as a failed
// KSum / out-of-order gather.
//
// Compiled/linked only when VLLM_CPP_NCCL — a non-NCCL CUDA build has no such
// symbol, so this test target is gated in tests/CMakeLists.txt.
#include <doctest/doctest.h>

extern "C" int vt_cuda_nccl_group_selfcheck(void);
extern "C" int vt_cuda_tp_seam_selfcheck(void);
extern "C" int vt_cuda_loader_slice_selfcheck(void);

TEST_CASE("in-process NCCL collectives across the discrete GPUs (multi-device)") {
  const int rc = vt_cuda_nccl_group_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); multi-GPU NCLL skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("W2 TP seam over the NCCL group (TpShard/TensorParallel row-reduce per rank)") {
  const int rc = vt_cuda_tp_seam_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); TP seam skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("TP loader slice: per-rank shard placed on the addressed GPU, reconstructs the full weight") {
  const int rc = vt_cuda_loader_slice_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); loader slice skipped");
    return;
  }
  CHECK(rc == 0);
}