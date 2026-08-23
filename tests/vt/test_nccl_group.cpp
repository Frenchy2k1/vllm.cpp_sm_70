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

#include <cmath>
#include <vector>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime.h>
#endif

extern "C" int vt_cuda_nccl_group_selfcheck(void);
extern "C" int vt_cuda_tp_seam_selfcheck(void);
extern "C" int vt_cuda_loader_slice_selfcheck(void);
extern "C" int vt_cuda_sharded_forward_selfcheck(void);
extern "C" int vt_cuda_dense_mlp_shard_selfcheck(void);
extern "C" int vt_cuda_mlp_shard_run(int O,int H,int I,const float* x,const float* gate,const float* up,const float* down,float* out);
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

TEST_CASE("reusable mlp_shard_run matches the host MLP reference at a fresh shape") {
  constexpr int O=4,H=8,I=16;
  std::vector<float> x(H,1.f), gate(I*H), up(I*H), down(O*I), out(O,0.f);
  for (int k=0;k<H;++k) x[k]=0.5f+(float)(k%3);
  for (int i=0;i<I;++i)for(int k=0;k<H;++k){gate[i*H+k]=0.1f*((i*3+k)%7)+0.05f;up[i*H+k]=0.05f*((i*5+k*2)%9)+0.2f;}
  for (int o=0;o<O;++o)for(int i=0;i<I;++i) down[o*I+i]=0.02f*((o*7+i*3)%11)+0.4f;
  int rc=vt_cuda_mlp_shard_run(O,H,I,x.data(),gate.data(),up.data(),down.data(),out.data());
  if (rc==2){MESSAGE("fewer than 2 GPUs (or NCCL unbuilt); mlp_shard_run skipped");return;}
  CHECK(rc==0);
  for (int o=0;o<O;++o){ double a=0.0;for(int i=0;i<I;++i){float g=0,u=0;for(int k=0;k<H;++k){g+=x[k]*gate[i*H+k];u+=x[k]*up[i*H+k];}float sg=1.f/(1.f+std::exp(-(double)g));a+=(double)((g*sg)*u)*down[o*I+i];} CHECK(out[(size_t)o]==doctest::Approx((float)a));}
}
