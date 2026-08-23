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

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime.h>
#endif

extern "C" int vt_cuda_nccl_group_selfcheck(void);
extern "C" int vt_cuda_tp_seam_selfcheck(void);
extern "C" int vt_cuda_loader_slice_selfcheck(void);
extern "C" int vt_cuda_sharded_forward_selfcheck(void);
extern "C" int vt_cuda_dense_mlp_shard_selfcheck(void);
extern "C" void* vt_cuda_tp_acquire(int tp_size);
extern "C" void vt_cuda_tp_release(void* handle);

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

TEST_CASE("engine bridge: acquire a retained NCCL TP group for tp>1, null at tp1") {
  // tp1 must stay the null/no-op path.
  CHECK(vt_cuda_tp_acquire(1) == nullptr);
#ifdef VLLM_CPP_CUDA
  int devs = 0;
  REQUIRE(cudaGetDeviceCount(&devs) == cudaSuccess);
  if (devs < 2) { MESSAGE("single-GPU; bridge skipped"); return; }
  void* h = vt_cuda_tp_acquire(devs);   // a real W=ngpu retained group
  REQUIRE(h != nullptr);
  vt_cuda_tp_release(h);                // and the runner can release it
#endif
}

TEST_CASE("dense-body shard: merged gate/up column-slice + down row-parallel reduce == MLP single-GPU") {
  const int rc = vt_cuda_dense_mlp_shard_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); dense shard skipped");
    return;
  }
  CHECK(rc == 0);
}

TEST_CASE("runner forward: device-sharded GEMM + group all-reduce == single-GPU forward") {
  const int rc = vt_cuda_sharded_forward_selfcheck();
  if (rc == 2) {
    MESSAGE("fewer than 2 CUDA devices (or NCCL unbuilt); runner forward skipped");
    return;
  }
  CHECK(rc == 0);
}