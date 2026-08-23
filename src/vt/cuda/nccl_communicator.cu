// NCCL tensor-parallel transport for vt::Communicator on kCUDA
// (BACKEND-DISTRIBUTED-COMM W2, .agents/specs/scale-out-distributed.md §W2/§W3).
//
// Multi-device phase-3 (2026-08-12): this transport is now LIVE and verified on
// the 4xV100 box (VLLM_CPP_NCCL=ON; ncclCommInitAll over the local GPUs). The
// derivative stays a faithful 1:1 port of vLLM's PyNcclCommunicator:
//   * ncclGetUniqueId / ncclCommInitRank ....... pynccl.py:129-139
//   * ncclAllReduce ............................ pynccl.py:166-188
//   * ncclAllGather ............................ pynccl.py:196-215
//   * ncclSend ................................. pynccl.py:305-323
//   * ncclRecv ................................. pynccl.py:332-350
// plus the ncclRedOp/ncclDataType maps. Each collective is stream-ordered on
// the vt::Queue's stream, as pynccl passes `cudaStream` into every op.
//
// The transport is routed through OpProvider/OpId (kAllReduce/kAllGather/kSend/
// kRecv on kCUDA); the CPU in-process group and this NCCL group are two
// providers of the one vt::Communicator seam. Without VT_NCCL the TU realizes
// NO kCUDA collective provider — a CUDA build lacking NCCL leaves

#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vt/communicator.h"
#include "vt/cuda/cuda_comm_group.h"

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vllm/model_executor/models/tensor_parallel.h"

#if defined(VT_NCCL)
#include <atomic>
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace vt {
namespace {

#if defined(VT_NCCL)

void NcclCheck(ncclResult_t r, const char* what) {
  if (r != ncclSuccess) {
    throw std::runtime_error(std::string("NCCL ") + what + ": " +
                             ncclGetErrorString(r));
  }
}

// vt::DType -> ncclDataType_t. Only the dtypes a TP all-reduce/gather moves are
// mapped; block-quantized dtypes never cross a collective.
ncclDataType_t NcclDType(DType dt) {
  switch (dt) {
    case DType::kF32:  return ncclFloat32;
    case DType::kF16:  return ncclFloat16;
    case DType::kBF16: return ncclBfloat16;
    case DType::kI8:   return ncclInt8;
    case DType::kI32:  return ncclInt32;
    case DType::kI64:  return ncclInt64;
    default: throw std::runtime_error("NCCL: unsupported collective dtype");
  }
}

ncclRedOp_t NcclRedOp(ReduceOp op) {
  switch (op) {
    case ReduceOp::kSum: return ncclSum;
    case ReduceOp::kMax: return ncclMax;
    case ReduceOp::kMin: return ncclMin;
    case ReduceOp::kProd: return ncclProd;
  }
  return ncclSum;
}

cudaStream_t Stream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// A process-group over a real NCCL communicator (one per rank; ncclCommInitRank
// by id broadcast, pynccl.py:129-139).
class NcclCommunicator final : public Communicator {
 public:
  NcclCommunicator(ncclComm_t comm, int rank, int world)
      : comm_(comm), rank_(rank), world_(world) {}

  int rank() const override { return rank_; }
  int world_size() const override { return world_; }

  void AllReduce(Queue& q, void* data, size_t count, DType dtype,
                 ReduceOp op) override {
    if (world_ == 1) return;
    NcclCheck(ncclAllReduce(data, data, count, NcclDType(dtype), NcclRedOp(op),
                            comm_, Stream(q)),
              "ncclAllReduce");
  }
  void AllGather(Queue& q, const void* sendbuf, void* recvbuf, size_t count,
                 DType dtype) override {
    if (world_ == 1) return;
    NcclCheck(ncclAllGather(sendbuf, recvbuf, count, NcclDType(dtype), comm_,
                            Stream(q)),
              "ncclAllGather");
  }
  void Send(Queue& q, const void* data, size_t count, DType dtype,
            int peer) override {
    NcclCheck(ncclSend(data, count, NcclDType(dtype), peer, comm_, Stream(q)),
              "ncclSend");
  }
  void Recv(Queue& q, void* data, size_t count, DType dtype,
            int peer) override {
    NcclCheck(ncclRecv(data, count, NcclDType(dtype), peer, comm_, Stream(q)),
              "ncclRecv");
  }

 private:
  ncclComm_t comm_;
  const int rank_;
  const int world_;
};

// kCUDA collective providers: the NcclCommunicator's methods ARE the data
// plane; OpProvider routing forwards to them.
void CudaAllReduce(Communicator& c, Queue& q, void* data, size_t count,
                   DType dtype, ReduceOp op) {
  static_cast<NcclCommunicator&>(c).AllReduce(q, data, count, dtype, op);
}
void CudaAllGather(Communicator& c, Queue& q, const void* sendbuf,
                   void* recvbuf, size_t count, DType dtype) {
  static_cast<NcclCommunicator&>(c).AllGather(q, sendbuf, recvbuf, count, dtype);
}
void CudaSend(Communicator& c, Queue& q, const void* data, size_t count,
              DType dtype, int peer) {
  static_cast<NcclCommunicator&>(c).Send(q, data, count, dtype, peer);
}
void CudaRecv(Communicator& c, Queue& q, void* data, size_t count, DType dtype,
              int peer) {
  static_cast<NcclCommunicator&>(c).Recv(q, data, count, dtype, peer);
}

struct NcclProviderRegistrar {
  NcclProviderRegistrar() {
    RegisterOp(OpId::kAllReduce, DeviceType::kCUDA,
               reinterpret_cast<void*>(&CudaAllReduce));
    RegisterOp(OpId::kAllGather, DeviceType::kCUDA,
               reinterpret_cast<void*>(&CudaAllGather));
    RegisterOp(OpId::kSend, DeviceType::kCUDA,
               reinterpret_cast<void*>(&CudaSend));
    RegisterOp(OpId::kRecv, DeviceType::kCUDA,
               reinterpret_cast<void*>(&CudaRecv));
  }
} nccl_provider_registrar;

// Per-op device affinity for a collective: set the queue's device current for
// the duration, restore the caller's on exit (mirrors CudaDeviceScope).
class NcclDevScope {
 public:
  explicit NcclDevScope(int dev) noexcept : dev_(dev), prev_(0), set_(false) {
    if (cudaGetDevice(&prev_) == cudaSuccess && prev_ != dev_) {
      set_ = (cudaSetDevice(dev_) == cudaSuccess);
    }
  }
  ~NcclDevScope() noexcept { if (set_) (void)cudaSetDevice(prev_); }

 private:
  int dev_;
  int prev_;
  bool set_;
};

// Opaque impl for vt::CudaCommGroup: one NcclCommunicator per local GPU.
struct CudaCommGroupImpl {
  std::vector<ncclComm_t> comm;
  std::vector<std::unique_ptr<NcclCommunicator>> comms;
  int world = 0;
};

#endif  // VT_NCCL

}  // namespace
}  // namespace vt

// ---------------------------------------------------------------------------
// vt::CudaCommGroup — reusable process-local NCCL group (Phase-3 wiring).
// Implemented in this TUU; declared in include/vt/cuda/cuda_comm_group.h.
// ---------------------------------------------------------------------------
#if defined(VT_NCCL)
namespace vt {

CudaCommGroup::CudaCommGroup() : impl_(nullptr), world_(0) {}

CudaCommGroup* CudaCommGroup::Create() {
  int ngpu = 0;
  if (cudaGetDeviceCount(&ngpu) != cudaSuccess || ngpu < 2) return nullptr;
  std::vector<int> devs(ngpu);
  for (int i = 0; i < ngpu; ++i) devs[i] = i;
  std::vector<ncclComm_t> comms(ngpu);
  if (ncclCommInitAll(comms.data(), ngpu, devs.data()) != ncclSuccess)
    return nullptr;

  auto* impl = new CudaCommGroupImpl;
  impl->world = ngpu;
  impl->comm = comms;
  impl->comms.reserve((size_t)ngpu);
  for (int i = 0; i < ngpu; ++i)
    impl->comms.emplace_back(
        std::make_unique<NcclCommunicator>(comms[i], i, ngpu));

  auto* g = new CudaCommGroup();
  g->impl_ = impl;
  g->world_ = ngpu;
  return g;
}

Communicator* CudaCommGroup::Rank(int r) const {
  auto* impl = static_cast<CudaCommGroupImpl*>(impl_);
  if (!impl || r < 0 || r >= impl->world) return nullptr;
  return impl->comms[(size_t)r].get();
}

CudaCommGroup::~CudaCommGroup() {
  auto* impl = static_cast<CudaCommGroupImpl*>(impl_);
  if (impl) {
    for (int i = 0; i < impl->world; ++i) ncclCommDestroy(impl->comm[i]);
    delete impl;
  }
}

}  // namespace vt
#endif  // VT_NCCL

// ---------------------------------------------------------------------------
// Host self-checks (extern "C", mirrored by tests/vt/test_nccl_group.cpp).
// ---------------------------------------------------------------------------
#if defined(VT_NCCL)
namespace vt {
namespace {

extern "C" int vt_cuda_nccl_group_selfcheck(void) {
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  constexpr int kValue = 1000;
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  threads.reserve((size_t)W);
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r, g]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      cudaStream_t st = nullptr;
      float *d = nullptr, *dg = nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&d, sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dg, (size_t)W * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(d); cudaFree(dg); if (st) cudaStreamDestroy(st);
        return;
      }
      // AllReduce kSum: rank r contributes (r+1); result = W(W+1)/2.
      const float mine = static_cast<float>(r + 1);
      cudaMemcpyAsync(d, &mine, sizeof(float), cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      Queue qq;
      qq.device = Device{DeviceType::kCUDA, r};
      qq.handle = st;
      g->Rank(r)->AllReduce(qq, d, 1u,
                            DType::kF32, ReduceOp::kSum);
      float agg = 0.f;
      cudaMemcpyAsync(&agg, d, sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      const float expected = static_cast<float>(W * (W + 1) / 2);
      if (std::fabs(agg - expected) > 1e-6f) {
        fprintf(stderr, "vt nccl self-check: rank %d allreduce %.3f want %.3f\n",
                r, (double)agg, (double)expected);
        bad.store(1, std::memory_order_relaxed);
      }
      // AllGather a FRESH per-rank value (the reduced `d` is now uniform) and
      // require the ordered [1000..1000+W-1] vector.
      const float sendv = static_cast<float>(kValue + r);
      cudaMemcpyAsync(d, &sendv, sizeof(float), cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      Queue qg;
      qg.device = Device{DeviceType::kCUDA, r};
      qg.handle = st;
      g->Rank(r)->AllGather(qg, d, dg, 1u,
                            DType::kF32);
      std::vector<float> got((size_t)W, 0.f);
      cudaMemcpyAsync(got.data(), dg, (size_t)W * sizeof(float),
                      cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (int i = 0; i < W; ++i) {
        if (std::fabs(got[(size_t)i] - (float)(kValue + i)) > 1e-6f) {
          fprintf(stderr, "vt nccl self-check: rank %d allgather[%d]=%.3f want %d\n",
                  r, i, (double)got[(size_t)i], kValue + i);
          bad.store(1, std::memory_order_relaxed);
        }
      }
      cudaFree(d); cudaFree(dg); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

// The engine-seam self-check: the ACTUAL W2 helpers (vllm::TensorParallel +
// TpShard + the row-parallel all-reduce) over the real NCCL group. Each rank
// (thread) verifies TpShard bounds on its TensorParallel, contributes a
// distinct full-row partial on its device, and the group sum must equal the
// unsharded total on every rank.
extern "C" int vt_cuda_tp_seam_selfcheck(void) {
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  constexpr int64_t kDim = 64;   // intermediate dim (divisible by W up to 64)
  constexpr int64_t kOut = 8;    // row-parallel full-row width
  const float total = static_cast<float>(W * (W + 1) / 2);
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  threads.reserve((size_t)W);
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r, g]() {
      if (bad.load(std::memory_order_relaxed)) return;
      vllm::TensorParallel tp{g->Rank(r)};
      if (tp.tp_size() != W || tp.rank() != r) { bad.store(1); return; }
      // ColumnParallel shard bounds on this rank.
      const vllm::ShardRange cs = vllm::TpShard(&tp, kDim);
      if (cs.size() != kDim / W || cs.begin != (kDim / W) * r) { bad.store(1); return; }

      NcclDevScope scope(r);
      cudaStream_t st = nullptr;
      float* d = nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&d, (size_t)kOut * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        if (st) { cudaFree(d); cudaStreamDestroy(st); }
        return;
      }
      // Row-parallel partial: this rank contributes (r+1) to every output col.
      std::vector<float> partial((size_t)kOut, static_cast<float>(r + 1));
      cudaMemcpyAsync(d, partial.data(), (size_t)kOut * sizeof(float),
                      cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      // The row-parallel all-reduce (this is exactly TpAllReduceSum's op).
      Queue qt;
      qt.device = Device{DeviceType::kCUDA, r};
      qt.handle = st;
      tp.comm->AllReduce(qt, d, (size_t)kOut,
                         DType::kF32, ReduceOp::kSum);
      std::vector<float> back((size_t)kOut, 0.f);
      cudaMemcpyAsync(back.data(), d, (size_t)kOut * sizeof(float),
                      cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (int64_t o = 0; o < kOut; ++o) {
        if (std::fabs(back[(size_t)o] - total) > 1e-6f) {
          fprintf(stderr, "vt tp-seam: rank %d out[%lld]=%.3f want %.3f\n",
                  r, (long long)o, (double)back[(size_t)o], (double)total);
          bad.store(1, std::memory_order_relaxed);
        }
      }
      cudaFree(d); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

}  // namespace
}  // namespace vt
#endif  // VT_NCCL