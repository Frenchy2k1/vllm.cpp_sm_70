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
#include <cmath>
#include <cstring>
  #include <cstdio>
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
  // ncclCommInitAll sets the CALLING thread's current device and does not
  // restore it, which would corrupt any later stream/launch the caller makes
  // (invalid-resource-handle). Save + restore so a shard call is a no-op on the
  // caller's device binding (the per-rank collectives run on their own threads).
  int caller_dev = 0;
  const bool saved = cudaGetDevice(&caller_dev) == cudaSuccess;
  std::vector<int> devs(ngpu);
  for (int i = 0; i < ngpu; ++i) devs[i] = i;
  std::vector<ncclComm_t> comms(ngpu);
  const ncclResult_t init_rc =
      ncclCommInitAll(comms.data(), ngpu, devs.data());
  if (saved) (void)cudaSetDevice(caller_dev);
  if (init_rc != ncclSuccess) return nullptr;

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

vt::CudaCommGroup::~CudaCommGroup() {
  auto* impl = static_cast<CudaCommGroupImpl*>(impl_);
  if (impl) {
    for (int i = 0; i < impl->world; ++i) ncclCommDestroy(impl->comm[i]);
    delete impl;
  }
}

}  // namespace vt

// ---------------------------------------------------------------------------
// Engine bridge (retained): a real NCCL-backed group for tp>1, else null (tp1,
// byte-identical path). The runner acquires once at init and holds the opaque
// handle for the model's lifetime, then wraps a rank communicator into a
// vllm::TensorParallel per layer. Release frees the group.
// ---------------------------------------------------------------------------
extern "C" void* vt_cuda_tp_acquire(int tp_size) {
  if (tp_size <= 1) return nullptr;  // tp1: keep the null/tp1 byte-identical path
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return nullptr;
  if (g->world_size() != tp_size) { delete g; return nullptr; }
  return reinterpret_cast<void*>(g);  // retained; caller owns until release
}
extern "C" void vt_cuda_tp_release(void* handle) {
  delete static_cast<vt::CudaCommGroup*>(handle);
}
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

// The LOADER slice (the last rrrollout brick minus the runner): a dense weight
// [K,N] sliced by vllm::TpShard per rank and placed on that rank's device
// exactly as a tp>1 weight loader would (per-shard memcpy onto device r). Each
// rank's returned slice must equal the original's columns [begin,end), and the
// W slices concatenated must reconstruct the full weight. No collective here —
// this is the placement side of TP.
extern "C" int vt_cuda_loader_slice_selfcheck(void) {
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  constexpr int64_t K = 8, N = 64;  // [K,N] row-major; N divisible by W
  const int64_t per = N / W;

  // Full weight: deterministic f32.
  std::vector<float> full((size_t)(K * N));
  for (int64_t i = 0; i < K; ++i)
    for (int64_t j = 0; j < N; ++j)
      full[(size_t)(i * N + j)] = 0.5f + (float)((i * 7 + j * 13) % 11);

  std::atomic<int> bad{0};
  std::vector<std::vector<float>> slices(static_cast<size_t>(W));
  std::vector<std::thread> threads;
  threads.reserve((size_t)W);
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r, g]() {
      if (bad.load(std::memory_order_relaxed)) return;
      vllm::TensorParallel tp{g->Rank(r)};
      const vllm::ShardRange cs = vllm::TpShard(&tp, N);
      if (tp.tp_size() != W || cs.begin != per * r || cs.size() != per) {
        bad.store(1, std::memory_order_relaxed);
        return;
      }
      NcclDevScope scope(r);
      cudaStream_t st = nullptr;
      float* d = nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&d, (size_t)(K * per) * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        if (st) { cudaFree(d); cudaStreamDestroy(st); }
        return;
      }
      // The rank's slice host view: full[…][begin..begin+per).
      std::vector<float> hslice((size_t)(K * per));
      for (int64_t i = 0; i < K; ++i)
        for (int64_t c = 0; c < per; ++c)
          hslice[(size_t)(i * per + c)] = full[(size_t)(i * N + (cs.begin + c))];
      cudaMemcpyAsync(d, hslice.data(), (size_t)(K * per) * sizeof(float),
                      cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      std::vector<float> back((size_t)(K * per), 0.f);
      cudaMemcpyAsync(back.data(), d, (size_t)(K * per) * sizeof(float),
                      cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (size_t i = 0; i < back.size(); ++i) {
        if (std::fabs(back[i] - hslice[i]) > 1e-6f) {
          fprintf(stderr, "vt loader-slice: rank %d elem %zu %.3f want %.3f\n",
                  r, i, (double)back[i], (double)hslice[i]);
          bad.store(1, std::memory_order_relaxed);
        }
      }
      slices[(size_t)r] = std::move(back);
      cudaFree(d); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  // Full reconstruction: cat slices_[r] per row.
  std::vector<float> rebuilt((size_t)(K * N), 0.f);
  for (int r = 0; r < W; ++r)
    for (int64_t i = 0; i < K; ++i)
      for (int64_t c = 0; c < per; ++c)
        rebuilt[(size_t)(i * N + (r * per + c))] = slices[(size_t)r][(size_t)(i * per + c)];
  for (size_t i = 0; i < full.size(); ++i) {
    if (std::fabs(full[i] - rebuilt[i]) > 1e-6f) {
      fprintf(stderr, "vt loader-slice: reconstruction %zu %.3f want %.3f\n",
              i, (double)rebuilt[i], (double)full[i]);
      bad.store(1, std::memory_order_relaxed);
    }
  }
  delete g;
  return bad.load();
}

// Row-by-row matvec over THIS rank's column shards: partial[o] += sum_k(x[k]*W[o][k]).
__global__ void NvKemmRankPartial(const float* x, const float* w_shard,
                                  float* partial, int O, int k_per) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float s = 0.f;
  for (int k = 0; k < k_per; ++k) s += x[k] * w_shard[o * k_per + k];
  partial[o] = s;
}

// RUNNER-FORWARD primitive: y = Wx sharded across the W GPUs (each rank owns a
// K-slice of columns on ITS device), each rank computes its PARTIAL on-device,
// and the group all-reduce sums partials so every rank holds the full y — the
// row-parallel reduce the engine forward runs per layer. Compared to the
// unsharded single-GPU reference.
extern "C" int vt_cuda_sharded_forward_selfcheck(void) {
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  constexpr int64_t O = 8, K = 16;  // [O,K]; W must divide K
  const int64_t k_per = K / W;

  std::vector<float> Wf((size_t)(O * K)), xf((size_t)K);
  for (int64_t k = 0; k < K; ++k) xf[(size_t)k] = 1.0f + (float)(k % 3);
  for (int64_t o = 0; o < O; ++o)
    for (int64_t k = 0; k < K; ++k)
      Wf[(size_t)(o * K + k)] = 0.5f + (float)((o * 5 + k * 3) % 7);

  std::vector<float> yref((size_t)O);
  for (int64_t o = 0; o < O; ++o) {
    double s = 0.0;
    for (int64_t k = 0; k < K; ++k) s += (double)xf[(size_t)k] * (double)Wf[(size_t)(o * K + k)];
    yref[(size_t)o] = (float)s;
  }

  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  threads.reserve((size_t)W);
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r, g]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const int64_t kc = k_per * r;
      cudaStream_t st = nullptr;
      float *dx = nullptr, *dW = nullptr, *dpart = nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dx, (size_t)K * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dW, (size_t)(O * k_per) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dpart, (size_t)O * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dx); cudaFree(dW); cudaFree(dpart); if (st) cudaStreamDestroy(st);
        return;
      }
      std::vector<float> xs((size_t)K, 0.f);
      for (int64_t k = 0; k < K; ++k)
        xs[(size_t)k] = (k >= kc && k < kc + k_per) ? xf[(size_t)k] : 0.f;
      std::vector<float> ws((size_t)(O * k_per));
      for (int64_t o = 0; o < O; ++o)
        for (int64_t k = 0; k < k_per; ++k)
          ws[(size_t)(o * k_per + k)] = Wf[(size_t)(o * K + (kc + k))];
      cudaMemcpyAsync(dx, xs.data(), (size_t)K * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dW, ws.data(), (size_t)(O * k_per) * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      NvKemmRankPartial<<<(unsigned)((O + 31) / 32), 32, 0, st>>>(dx + kc, dW, dpart, (int)O, (int)k_per);
      Queue qr;
      qr.device = Device{DeviceType::kCUDA, r};
      qr.handle = st;
      g->Rank(r)->AllReduce(qr, dpart, (size_t)O, DType::kF32, ReduceOp::kSum);
      std::vector<float> back((size_t)O, 0.f);
      cudaMemcpyAsync(back.data(), dpart, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (int64_t o = 0; o < O; ++o) {
        if (std::fabs(back[(size_t)o] - yref[(size_t)o]) > 1e-5) {
          fprintf(stderr, "vt sharded-forward: rank %d y[%lld]=%.3f want %.3f\n",
                  r, (long long)o, (double)back[(size_t)o], (double)yref[(size_t)o]);
          bad.store(1, std::memory_order_relaxed);
        }
      }
      cudaFree(dx); cudaFree(dW); cudaFree(dpart); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

// ---------------------------------------------------------------------------
// Host NVFP4 decode (the DenseMlpBlock tp>1 shard feed). Mirrors the kernel's
// F8E4M3 + DecodeFp4Byte (E2M1 LUT, bf16-RNE-rounded product) exactly.
// Returns the dequant fp32 [N, K] for one fp4 weight (packed[N,K/2],
// f8-scale[N,K/16], per-row scale2); dk = group count. Deterministic host math.
// ---------------------------------------------------------------------------
static inline float HostF8E4M3ToF32(uint8_t byte) {
  const int sign = (byte >> 7) & 1;
  const int exp = (byte >> 3) & 0xFu;
  const int mant = byte & 0x7u;
  const float s = sign ? -1.0f : 1.0f;
  if (exp == 0xFu && mant == 0x7u) return std::nanf("");
  if (exp == 0u) return s * (static_cast<float>(mant) / 512.0f);
  return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}
// float -> bf16 (RNE) -> float: equals __bfloat162float(__float2bfloat16(v)).
static inline float HostRoundBf16(float v) {
  uint32_t u = 0; std::memcpy(&u, &v, 4);
  // RNE: add 0x7fff + (lsb of low half) then truncate the low 16 bits.
  const uint32_t b = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  float r; std::memcpy(&r, &b, 4); return r;
}
static constexpr float kHostE2M1[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
extern "C" int vt_host_decode_nvfp4_f32(int N, int K, const uint8_t* packed,
                                        const uint8_t* scale8, const float* scale2,
                                        float* out) {
  if (N < 0 || K <= 0 || K == 1) return 1;
  const int halves = K / 2, groups = K / 16;
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < halves; ++j) {
      const uint8_t b = packed[n * halves + j];
      const float gs = HostF8E4M3ToF32(scale8[n * groups + (2*j) / 16]) * scale2[n];
      const int lo = b & 0xFu, hi = b >> 4;
      out[n * K + 2 * j]     = HostRoundBf16(kHostE2M1[lo & 7] * ((lo & 8) ? -1.f : 1.f) * gs);
      out[n * K + 2 * j + 1]  = HostRoundBf16(kHostE2M1[hi & 7] * ((hi & 8) ? -1.f : 1.f) * gs);
    }
  return 0;
}
// ---------------------------------------------------------------------------
// merged gate/up, one thread per output row, row-parallel over the group reduce.
__global__ void DenseMlpRankPartial(const float* x, int H, const float* gate,
                                    const float* up, const float* downfull,
                                    float* out, int per_i, int i0, int I, int O) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  float acc = 0.f;
  for (int q = 0; q < per_i; ++q) {
    const int gi = i0 + q;
    float g = 0.f, u = 0.f;
    for (int k = 0; k < H; ++k) { g += x[k] * gate[gi * H + k]; u += x[k] * up[gi * H + k]; }
    const float sg = 1.f / (1.f + __expf(-g));
    acc += (g * sg) * u * downfull[o * I + gi];
  }
  out[o] = acc;
}

// Batched dense-MLP rank partial: one thread per (t,o); rank owns the
// [i0, i0+per_i) slice of the I intermediate (gate/up column slices, down row
// slice) and writes the per-token partial to out[t*O+o]; the group
// AllReduceSum over the [T*O] partial yields the full result on every rank.
__global__ void DenseMlpRankPartialBatch(const float* x, int T, int H,
                                         const float* gate, const float* up,
                                         const float* downfull, float* out,
                                         int per_i, int i0, int I, int O) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= O) return;
  const int t = blockIdx.y;
  const float* xt = x + (size_t)t * H;
  float acc = 0.f;
  for (int q = 0; q < per_i; ++q) {
    const int gi = i0 + q;
    float g = 0.f, u = 0.f;
    for (int k = 0; k < H; ++k) { g += xt[k] * gate[gi * H + k]; u += xt[k] * up[gi * H + k]; }
    const float sg = 1.f / (1.f + __expf(-g));
    acc += (g * sg) * u * downfull[(size_t)o * I + gi];
  }
  out[(size_t)t * O + o] = acc;
}


// Reusable real-shape dense-MLP shard over the in-process NCCL group. Computes
//   out[o] = sum_i silu(gate_i^T x) * (up_i^T x) * down[o,i]
// with rank r owning I/W of the I intermediate on its device (gate/up column
// slices, down row slice) and the group AllReduceSum over [O] partial. The
// engine DenseMlpBlock tp>1 hook calls this with the layer H/I/O + decoded fp4
// slices. Returns 0 on parity with the on-host reference, 1 mismatch, 2 on
// fewer than 2 GPUs (or NCCL unbuilt; caller keeps its tp1 path).
extern "C" int vt_cuda_mlp_shard_run(int O, int H, int I,
                                     const float* x, const float* gate,
                                     const float* up, const float* down,
                                     float* out) {
  struct CallerDevGuard {
    int dev_;
    bool set_;
    ~CallerDevGuard() { if (set_) (void)cudaSetDevice(dev_); }
  } guard{0, false};
  guard.dev_ = 0;
  guard.set_ = cudaGetDevice(&guard.dev_) == cudaSuccess;
  if (O <= 0 || H <= 0 || I <= 0) return 1;
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  if (I % W != 0) { delete g; return 1; }
  const int64_t per = I / W;

  // host reference.
  std::vector<float> ref((size_t)O, 0.f);
  for (int o = 0; o < O; ++o) {
    double a = 0.0;
    for (int i = 0; i < I; ++i) {
      float gv = 0.f, uv = 0.f;
      for (int k = 0; k < H; ++k) { gv += x[k] * gate[i * (size_t)H + k]; uv += x[k] * up[i * (size_t)H + k]; }
      const float sg = 1.f / (1.f + std::exp(-(double)gv));
      a += (double)((gv * sg) * uv) * down[o * (size_t)I + i];
    }
    ref[o] = (float)a;
  }
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  threads.reserve((size_t)W);
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const int64_t ib = per * r;
      cudaStream_t st = nullptr;
      float *dx=nullptr,*dgu=nullptr,*dd=nullptr,*dout=nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dx, (size_t)H * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dgu, (size_t)(2 * I * H) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dd, (size_t)(O * I) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dout, (size_t)O * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dx); cudaFree(dgu); cudaFree(dd); cudaFree(dout); if (st) cudaStreamDestroy(st);
        return;
      }
      std::vector<float> gu((size_t)(2 * I * H), 0.f);
      for (int64_t i = ib; i < ib + per; ++i)
        for (int k = 0; k < H; ++k) {
          gu[(size_t)(i * H + k)] = gate[i * (size_t)H + k];
          gu[(size_t)(((I + i) * H + k))] = up[i * (size_t)H + k];
        }
      std::vector<float> dmask((size_t)(O * I), 0.f);
      for (int o = 0; o < O; ++o) for (int64_t i = ib; i < ib + per; ++i) dmask[o * I + i] = down[o * I + i];
      cudaMemcpyAsync(dx, x, (size_t)H * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dgu, gu.data(), (size_t)(2 * I * H) * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dd, dmask.data(), (size_t)(O * I) * sizeof(float), cudaMemcpyHostToDevice, st);
      DenseMlpRankPartial<<<(unsigned)((O + 31) / 32), 32, 0, st>>>(
          dx, H, dgu, dgu + (size_t)(I * H), dd, dout, (int)per, (int)ib, (int)I, (int)O);
      Queue q; q.device = Device{DeviceType::kCUDA, r}; q.handle = st;
      g->Rank(r)->AllReduce(q, dout, (size_t)O, DType::kF32, ReduceOp::kSum);
      std::vector<float> back((size_t)O, 0.f);
      cudaMemcpyAsync(back.data(), dout, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      if (r == 0) { for (int o = 0; o < O; ++o) out[o] = back[(size_t)o]; }
      for (int o = 0; o < O; ++o)
        if (std::fabs(back[(size_t)o] - ref[(size_t)o]) >
            5e-4f * std::max(1.0f, std::fabs(ref[(size_t)o])))
          bad.store(1, std::memory_order_relaxed);
      cudaFree(dx); cudaFree(dgu); cudaFree(dd); cudaFree(dout); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

// Batched dense-MLP shard: the same unsharded math as vt_cuda_mlp_shard_run but
// for a [T,H] token batch in ONE group (one NCCL create, one kernel launch per
// rank, one [T*O] AllReduce). x = f32 [T,H], gate/up = f32 [I,H], down = f32
// [O,I], out = f32 [T,O].
//
// verify: 0 = compute-only (production path; the engine's token comparison
// against the tp1 arm is the correctness check). 1 = full parity check
// against a double-precision host reference for every (t,o) (selfcheck path
// only; that reference is O(T*O*I*H) and must not run at 27B shape).
extern "C" int vt_cuda_mlp_shard_run_b(int T, int O, int H, int I, int verify,
                                       const float* x, const float* gate,
                                       const float* up, const float* down,
                                       float* out) {
  struct CallerDevGuard {
    int dev_;
    bool set_;
    ~CallerDevGuard() { if (set_) (void)cudaSetDevice(dev_); }
  } guard{0, false};
  guard.dev_ = 0;
  guard.set_ = cudaGetDevice(&guard.dev_) == cudaSuccess;
  if (T <= 0 || O <= 0 || H <= 0 || I <= 0) return 1;
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  if (I % W != 0) { delete g; return 1; }
  const int64_t per = I / W;

  // Optional double-precision host reference, parallelised over (t,o) rows.
  std::vector<float> ref;
  if (verify) {
    ref.assign((size_t)T * O, 0.f);
    const int ncpu = std::max(1, (int)std::thread::hardware_concurrency());
    std::vector<std::thread> hts;
    for (int c = 0; c < ncpu; ++c)
      hts.emplace_back([&](int c) {
        for (size_t n = (size_t)c; n < (size_t)T * O; n += ncpu) {
          const int t = (int)(n / O), o = (int)(n % O);
          double a = 0.0;
          for (int i = 0; i < I; ++i) {
            double gv = 0.0, uv = 0.0;
            for (int k = 0; k < H; ++k) {
              gv += (double)x[(size_t)t * H + k] * gate[i * (size_t)H + k];
              uv += (double)x[(size_t)t * H + k] * up[i * (size_t)H + k];
            }
            const float sg = 1.f / (1.f + std::exp(-(double)gv));
            a += (double)((gv * sg) * uv) * down[o * (size_t)I + i];
          }
          ref[n] = (float)a;
        }
      }, c);
    for (auto& h : hts) h.join();
  }
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  threads.reserve((size_t)W);
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const int64_t ib = per * r;
      cudaStream_t st = nullptr;
      float *dx=nullptr,*dgu=nullptr,*dd=nullptr,*dout=nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dx, (size_t)T * H * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dgu, (size_t)(2 * I * H) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dd, (size_t)(O * I) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dout, (size_t)T * O * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dx); cudaFree(dgu); cudaFree(dd); cudaFree(dout); if (st) cudaStreamDestroy(st);
        return;
      }
      std::vector<float> gu((size_t)(2 * I * H), 0.f);
      for (int64_t i = ib; i < ib + per; ++i)
        for (int k = 0; k < H; ++k) {
          gu[(size_t)(i * H + k)] = gate[i * (size_t)H + k];
          gu[(size_t)(((I + i) * H + k))] = up[i * (size_t)H + k];
        }
      std::vector<float> dmask((size_t)(O * I), 0.f);
      for (int o = 0; o < O; ++o) for (int64_t i = ib; i < ib + per; ++i) dmask[o * I + i] = down[o * I + i];
      cudaMemcpyAsync(dx, x, (size_t)T * H * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dgu, gu.data(), (size_t)(2 * I * H) * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dd, dmask.data(), (size_t)(O * I) * sizeof(float), cudaMemcpyHostToDevice, st);
      dim3 grid((unsigned)((O + 31) / 32), (unsigned)T);
      DenseMlpRankPartialBatch<<<grid, 32, 0, st>>>(
          dx, T, H, dgu, dgu + (size_t)(I * H), dd, dout, (int)per, (int)ib, (int)I, (int)O);
      Queue q; q.device = Device{DeviceType::kCUDA, r}; q.handle = st;
      g->Rank(r)->AllReduce(q, dout, (size_t)T * O, DType::kF32, ReduceOp::kSum);
      std::vector<float> back((size_t)T * O, 0.f);
      cudaMemcpyAsync(back.data(), dout, (size_t)T * O * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      if (r == 0)
        std::memcpy(out, back.data(), (size_t)T * O * sizeof(float));
      if (verify)
        for (size_t n = 0; n < (size_t)T * O; ++n)
          if (std::fabs(back[n] - ref[n]) > 5e-4f * std::max(1.0f, std::fabs(ref[n])))
            bad.store(1, std::memory_order_relaxed);
      cudaFree(dx); cudaFree(dgu); cudaFree(dd); cudaFree(dout); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

extern "C" int vt_cuda_dense_mlp_shard_selfcheck(void) {
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  constexpr int O = 16, H = 32, I = 32;   // I % W == 0
  const int64_t per = I / W;

  std::vector<float> x(H), gate(I * H), up(I * H), down(O * I);
  for (int k = 0; k < H; ++k) x[k] = 0.5f + (float)(k % 3);
  for (int i = 0; i < I; ++i) for (int k = 0; k < H; ++k) gate[i * H + k] = 0.1f * ((i * 3 + k) % 7) + 0.05f;
  for (int i = 0; i < I; ++i) for (int k = 0; k < H; ++k) up[i * H + k] = 0.05f * ((i * 5 + k * 2) % 9) + 0.2f;
  for (int o = 0; o < O; ++o) for (int i = 0; i < I; ++i) down[o * I + i] = 0.02f * ((o * 7 + i * 3) % 11) + 0.4f;

  std::vector<float> ref(O, 0.f);
  for (int o = 0; o < O; ++o) {
    double a = 0.0;
    for (int i = 0; i < I; ++i) {
      float gu = 0.f;
      for (int k = 0; k < H; ++k) gu += x[k] * gate[i * H + k];
      float uu = 0.f;
      for (int k = 0; k < H; ++k) uu += x[k] * up[i * H + k];
      const float sg = 1.f / (1.f + (float)std::exp(-(double)gu));
      a += (double)((gu * sg) * uu) * down[o * I + i];
    }
    ref[o] = (float)a;
  }

  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const int64_t ib = per * r;
      cudaStream_t st = nullptr;
      float *dx = nullptr, *dgu = nullptr, *dd = nullptr, *dout = nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dx, (size_t)H * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dgu, (size_t)(2 * I * H) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dd, (size_t)(O * I) * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dout, (size_t)O * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dx); cudaFree(dgu); cudaFree(dd); cudaFree(dout); if (st) cudaStreamDestroy(st);
        return;
      }
      std::vector<float> gu((size_t)(2 * I * H), 0.f);
      for (int64_t i = ib; i < ib + per; ++i)
        for (int k = 0; k < H; ++k) {
          gu[(size_t)(i * H + k)] = gate[i * H + k];
          gu[(size_t)(((I + i) * H + k))] = up[i * H + k];
        }
      std::vector<float> dmask((size_t)(O * I), 0.f);
      for (int o = 0; o < O; ++o) for (int64_t i = ib; i < ib + per; ++i) dmask[o * I + i] = down[o * I + i];
      cudaMemcpyAsync(dx, x.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dgu, gu.data(), (size_t)(2 * I * H) * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dd, dmask.data(), (size_t)(O * I) * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      DenseMlpRankPartial<<<(unsigned)((O + 31) / 32), 32, 0, st>>>(
          dx, H, dgu, dgu + (size_t)(I * H), dd, dout, (int)per, (int)ib, (int)I, (int)O);
      Queue q;
      q.device = Device{DeviceType::kCUDA, r};
      q.handle = st;
      g->Rank(r)->AllReduce(q, dout, (size_t)O, DType::kF32, ReduceOp::kSum);
      std::vector<float> back((size_t)O, 0.f);
      cudaMemcpyAsync(back.data(), dout, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (int o = 0; o < O; ++o) {
        if (std::fabs(back[(size_t)o] - ref[(size_t)o]) > 1e-3f) {
          fprintf(stderr, "vt dense-mlp-shard: rank %d out[%d]=%.4f want %.4f\n", r, o,
                  (double)back[(size_t)o], (double)ref[(size_t)o]);
          bad.store(1, std::memory_order_relaxed);
        }
      }
      cudaFree(dx); cudaFree(dgu); cudaFree(dd); cudaFree(dout); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

// Assemble the engine-facing slice for a bf16-represented dense MLP and shard it:
// split the [2I,H] merged gate+up (upper I rows = gate, lower I = up), the
// [H( out),I] down, convert bf16 bit-exact to f32, run the shard. This is the
// assembly the DenseMlpBlock tp>1 branch uses; verified model-free here.
extern "C" int vt_cuda_mlp_shard_run_bf16(int O, int H, int I,
                                          const uint16_t* x16, const uint16_t* gu16,
                                          const uint16_t* down16, float* out) {
  if (O <= 0 || H <= 0 || I <= 0) return 1;
  const auto bf = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; };
  std::vector<float> x((size_t)H), gate((size_t)(I * H)), up((size_t)(I * H)), down((size_t)(O * I));
  for (int i = 0; i < H; ++i) x[(size_t)i] = bf(x16[i]);
  for (int i = 0; i < I; ++i)
    for (int k = 0; k < H; ++k) {
      gate[(size_t)(i * H + k)] = bf(gu16[(size_t)(i * H + k)]);        // rows [0,I)
      up[(size_t)(i * H + k)] = bf(gu16[(size_t)(((I + i) * H + k))]);  // rows [I,2I)
    }
  for (int o = 0; o < O; ++o)
    for (int i = 0; i < I; ++i) down[(size_t)(o * I + i)] = bf(down16[(size_t)(o * I + i)]);
  return vt_cuda_mlp_shard_run(O, H, I, x.data(), gate.data(), up.data(), down.data(), out);
}

// ---------------------------------------------------------------------------
// KV/GQA-shard attention primitive (TP step-2, the attention sibling of the
// dense-MLP shard). Softmax numerator (sum exp(score)*v over kv) and denominator
// (sum exp(score)) are BOTH additive over the kv dimension, so a rank owning a
// contiguous slice of the Hkv kv-heads computes its partial num/den on ITS device
// and the group AllReduceSum reduces both before out = num/den. This is the
// standard cross-rank KV-split attention (TpAllReduce over KV heads instead of
// AllGather), mirroring the runner-attach row-reduce on the o_proj side. Each
// rank holds its slice of k/v; q is the full [T,Hq,D] on every rank (q heads
// are dense; only the KV-intermediate is sharded). Returns 0 on parity with the
// on-host full-kv reference, 1 mismatch, 2 on fewer than 2 GPUs (NCCL unbuilt).
// ---------------------------------------------------------------------------
static void AttnKvHostRef(int T, int Hq, int S, int Hkv, int D, float scale,
                          const float* q, const float* k, const float* v,
                          std::vector<float>& ref, std::vector<float>& den_ref) {
  ref.assign((size_t)T * Hq * D, 0.f);
  den_ref.assign((size_t)T * Hq, 0.f);
  for (int t = 0; t < T; ++t)
    for (int hq = 0; hq < Hq; ++hq) {
      const float* qt = q + ((size_t)t * Hq + hq) * D;
      double den = 0.0;
      for (int s = 0; s < S && s <= t; ++s)             // causal window
        for (int kh = 0; kh < Hkv; ++kh) {
          const float* kp = k + (((size_t)s * Hkv) + kh) * D;
          float sc = 0.f;
          for (int d = 0; d < D; ++d) sc += qt[d] * kp[d];
          const float e = std::exp((double)(sc * scale));
          den += e;
          const float* vp = v + (((size_t)s * Hkv) + kh) * D;
          for (int d = 0; d < D; ++d)
            ref[(size_t)((t * Hq + hq) * D) + d] += (float)(e * (double)vp[d]);
        }
      den_ref[(size_t)(t * Hq + hq)] = (float)den;
    }
  for (size_t i = 0; i < ref.size(); ++i)
    ref[i] /= den_ref[i / D];
}

// Grid.y = T (one query position), block over Hq*D. Each rank owns a contiguous
// `per`-head slice; for each causal position s<=t it accumulates over that slice
// the exp(score*v) partial (dnum) and exp(score) partial (dden). AllReduceSum
// across ranks then reduces both; out = num/den.
__global__ void AttnKvShardPartial(const float* dq, const float* dk,
                                    const float* dv, float* dnum, float* dden,
                                    int Hq, int S, int Hkv, int per, int D,
                                    int hk0, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Hq * D;
  const int t = blockIdx.y;                       // query position == causal bound
  if (idx >= total || t >= S) return;
  const int hq = idx / D, d0 = idx % D;
  const float* qt = dq + ((size_t)t * Hq + hq) * D;
  float acc = 0.f;
  double sum_exp = 0.0;
  for (int s = 0; s <= t; ++s)
    for (int k = 0; k < per; ++k) {
      const int kh = hk0 + k;
      const float* kp = dk + (((size_t)s * Hkv) + kh) * D;
      float sc = 0.f;
      for (int d = 0; d < D; ++d) sc += qt[d] * kp[d];
      const double e = std::exp((double)(sc * scale));
      sum_exp += e;
      const float* vp = dv + (((size_t)s * Hkv) + kh) * D;
      acc += (float)(e * (double)vp[d0]);
    }
  dnum[(size_t)t * Hq * D + idx] = acc;           // partial exp*v
  if (d0 == 0) dden[(size_t)t * Hq + hq] = (float)sum_exp;
}

extern "C" int vt_cuda_attn_kv_shard_run(int T, int Hq, int S, int Hkv, int D,
                                          const float* q, const float* k, const float* v,
                                          float* out) {
  // The NCCL group work can re-bind the CALLING thread's current device (and
  // ncclCommDestroy too); a CUDA op the caller launches right after (e.g.
  // SigmoidGateBf16 via the OpProvider path, which has no device scope) would
  // then hit invalid-resource-handle. Save + restore the whole entry via RAII so
  // the call is a no-op on the caller's thread device.
  struct CallerDevGuard {
    int dev_;
    bool set_;
    ~CallerDevGuard() { if (set_) (void)cudaSetDevice(dev_); }
  } guard{0, false};
  guard.dev_ = 0;
  guard.set_ = cudaGetDevice(&guard.dev_) == cudaSuccess;
  if (T <= 0 || Hq <= 0 || Hkv <= 0 || S <= 0 || D <= 0) return 1;
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  if (Hkv % W != 0) { delete g; return 1; }
  const int per = Hkv / W;
  const float scale = 1.0F / std::sqrt((float)D);
  std::vector<float> ref, den_ref;
  AttnKvHostRef(T, Hq, S, Hkv, D, scale, q, k, v, ref, den_ref);
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const size_t qn = (size_t)T * Hq * D, den_n = (size_t)T * Hq,
                   kn = (size_t)S * Hkv * D;
      cudaStream_t st = nullptr;
      float *dq=nullptr,*dk=nullptr,*dv=nullptr,*dnum=nullptr,*dden=nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dq, qn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dk, kn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dv, kn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dnum, qn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dden, den_n * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dnum); cudaFree(dden);
        if (st) cudaStreamDestroy(st);
        return;
      }
      cudaMemsetAsync(dnum, 0, qn * sizeof(float), st);
      cudaMemsetAsync(dden, 0, den_n * sizeof(float), st);
      cudaMemcpyAsync(dq, q, qn * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dk, k, kn * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dv, v, kn * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      dim3 grid_dim((unsigned)((Hq * D + 31) / 32), (unsigned)T);
      AttnKvShardPartial<<<grid_dim, 32, 0, st>>>(
          dq, dk, dv, dnum, dden, Hq, (int)S, Hkv, per, (int)D, per * r, scale);
      Queue qq; qq.device = Device{DeviceType::kCUDA, r}; qq.handle = st;
      g->Rank(r)->AllReduce(qq, dnum, qn, DType::kF32, ReduceOp::kSum);
      g->Rank(r)->AllReduce(qq, dden, den_n, DType::kF32, ReduceOp::kSum);
      std::vector<float> num(qn), den(den_n);
      cudaMemcpyAsync(num.data(), dnum, qn * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaMemcpyAsync(den.data(), dden, den_n * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (size_t i = 0; i < qn; ++i)
        num[i] /= den[(size_t)(i / D)];
      if (r == 0) std::copy(num.begin(), num.end(), out);
      for (size_t i = 0; i < qn; ++i) {
        if (!(std::fabs(num[i] - ref[i]) <= 1e-3f)) {
          bad.store(1, std::memory_order_relaxed);
        }
      }
      cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dnum); cudaFree(dden);
      cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

// ---------------------------------------------------------------------------
// GQA-exact KV-shard attention primitive (TP step-3, the paged decode path).
//
// UNLIKE vt_cuda_attn_kv_shard_run above (which sums exp(q.k) over ALL kv-heads
// - a gather that only the weak dense 8-token gate tolerates), this primitive
// is standard GQA: each query head hq attends to exactly ONE kv head
// g = hq / (Hq/Hkv). Rank r owns the kv-head slice [per*r, per*r+per). For a
// query head whose g is NOT in the slice, the rank writes 0; the rank that OWNS
// g computes the full single-kv-head softmax over positions s < key_end[t] (no
// sum over kh). AllReduceSum across ranks then gathers the output with NO global
// num/den divide, because each (t,hq) softmax is produced on exactly one rank.
// The result is token-exact to the unsharded GQA (vt::PagedAttention /
// vt::Attention) by construction.
//
// Contiguous [S,Hkv,D] K/V (the caller stages the paged NHD cache to host first).
// key_end[t] = number of keys the query at position t sees (causal: t+1; paged
// decode: the full seq_len). Returns 0 on parity with the host GQA reference,
// 1 mismatch / bad shape, 2 on fewer than 2 GPUs (NCCL unbuilt).
// ---------------------------------------------------------------------------
static void AttnGqaHostRef(int T, int Hq, int Hkv, int D, float scale,
                           const int* key_end, const float* q, const float* k,
                           const float* v, std::vector<float>& ref) {
  ref.assign((size_t)T * Hq * D, 0.f);
  const int qpk = Hq / Hkv;
  for (int t = 0; t < T; ++t)
    for (int hq = 0; hq < Hq; ++hq) {
      const float* qt = q + ((size_t)t * Hq + hq) * D;
      const int g = hq / qpk;          // the ONE kv head this q head attends to
      const int ke = key_end[t];       // keys visible to query t
      double den = 0.0;
      for (int s = 0; s < ke; ++s) {
        const float* kp = k + (((size_t)s * Hkv) + g) * D;
        float sc = 0.f;
        for (int d = 0; d < D; ++d) sc += qt[d] * kp[d];
        den += std::exp((double)(sc * scale));
      }
      for (int s = 0; s < ke; ++s) {
        const float* kp = k + (((size_t)s * Hkv) + g) * D;
        const float* vp = v + (((size_t)s * Hkv) + g) * D;
        float sc = 0.f;
        for (int d = 0; d < D; ++d) sc += qt[d] * kp[d];
        const double e = std::exp((double)(sc * scale));
        for (int d = 0; d < D; ++d)
          ref[(size_t)((t * Hq + hq) * D) + d] += (float)(e * (double)vp[d] / den);
      }
    }
}

// Grid.y = T (one query position), block over Hq*D. Each rank owns a contiguous
// `per`-kv-head slice [hk0, hk0+per); it writes the full single-kv-head softmax
// partial (num = sum_e e*v[d0], den = sum_e e) ONLY for the query heads whose
// kv head g is in that slice, 0 otherwise. AllReduceSum reduces both.
__global__ void AttnGqaShardPartial(const float* dq, const float* dk,
                                    const float* dv, float* dnum, float* dden,
                                    int Hq, int Hkv, int per, int D, int hk0,
                                    float scale, const int* dkey_end) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Hq * D;
  const int t = blockIdx.y;                       // query position
  if (idx >= total || t < 0) return;
  const int hq = idx / D, d0 = idx % D;
  const int g = hq / (Hq / Hkv);                  // owning kv head
  if (g < hk0 || g >= hk0 + per) return;          // not this rank's slice -> 0
  const float* qt = dq + ((size_t)t * Hq + hq) * D;
  const int ke = dkey_end[t];
  float acc = 0.f;
  double sum_exp = 0.0;
  for (int s = 0; s < ke; ++s) {
    const float* kp = dk + (((size_t)s * Hkv) + g) * D;
    float sc = 0.f;
    for (int d = 0; d < D; ++d) sc += qt[d] * kp[d];
    const double e = std::exp((double)(sc * scale));
    sum_exp += e;
    const float* vp = dv + (((size_t)s * Hkv) + g) * D;
    acc += (float)(e * (double)vp[d0]);
  }
  dnum[(size_t)t * Hq * D + idx] = acc;           // partial e*v over its one kv head
  if (d0 == 0) dden[(size_t)t * Hq + hq] = (float)sum_exp;
}

extern "C" int vt_cuda_attn_gqa_shard_run(int T, int Hq, int Hkv, int S, int D,
                                          const int* key_end, const float* q,
                                          const float* k, const float* v,
                                          float* out) {
  // The NCCL group work can re-bind the CALLING thread's current device (and
  // ncclCommDestroy too); save + restore the whole entry via RAII so the call
  // is a no-op on the caller's thread device (the 40bad253 fix).
  struct CallerDevGuard {
    int dev_;
    bool set_;
    ~CallerDevGuard() { if (set_) (void)cudaSetDevice(dev_); }
  } guard{0, false};
  guard.dev_ = 0;
  guard.set_ = cudaGetDevice(&guard.dev_) == cudaSuccess;
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  const int W = g->world_size();
  if (Hkv % W != 0) { delete g; return 1; }
  const int per = Hkv / W;
  const float scale = 1.0F / std::sqrt((float)D);
  std::vector<float> ref;
  AttnGqaHostRef(T, Hq, Hkv, D, scale, key_end, q, k, v, ref);
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const size_t qn = (size_t)T * Hq * D, den_n = (size_t)T * Hq,
                   kn = (size_t)S * Hkv * D, ken = (size_t)T;
      cudaStream_t st = nullptr;
      float *dq=nullptr,*dk=nullptr,*dv=nullptr,*dnum=nullptr,*dden=nullptr;
      int* dke=nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dq, qn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dk, kn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dv, kn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dnum, qn * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dden, den_n * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dke, ken * sizeof(int)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dnum); cudaFree(dden); cudaFree(dke);
        if (st) cudaStreamDestroy(st);
        return;
      }
      cudaMemcpyAsync(dq, q, qn * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dk, k, kn * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dv, v, kn * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dke, key_end, ken * sizeof(int), cudaMemcpyHostToDevice, st);
      cudaMemsetAsync(dnum, 0, qn * sizeof(float), st);
      cudaMemsetAsync(dden, 0, den_n * sizeof(float), st);
      cudaStreamSynchronize(st);
      dim3 grid_dim((unsigned)((Hq * D + 31) / 32), (unsigned)T);
      AttnGqaShardPartial<<<grid_dim, 32, 0, st>>>(
          dq, dk, dv, dnum, dden, Hq, Hkv, per, D, per * r, scale, dke);
      Queue qq; qq.device = Device{DeviceType::kCUDA, r}; qq.handle = st;
      g->Rank(r)->AllReduce(qq, dnum, qn, DType::kF32, ReduceOp::kSum);
      g->Rank(r)->AllReduce(qq, dden, den_n, DType::kF32, ReduceOp::kSum);
      std::vector<float> num(qn), den(den_n);
      cudaMemcpyAsync(num.data(), dnum, qn * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaMemcpyAsync(den.data(), dden, den_n * sizeof(float), cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      for (size_t i = 0; i < qn; ++i)
        num[i] /= den[(size_t)(i / D)];
      if (r == 0) std::copy(num.begin(), num.end(), out);
      for (size_t i = 0; i < qn; ++i) {
        if (!(std::fabs(num[i] - ref[i]) <= 1e-3f)) {
          bad.store(1, std::memory_order_relaxed);
        }
      }
      cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dnum); cudaFree(dden); cudaFree(dke);
      cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

// ---------------------------------------------------------------------------
// lm_head/shard primitive (TP step-2, the logits head). The head is a
// vocab-column-parallel linear: each rank owns a contiguous slice of the vocab
// rows of lm_head [V,H], computes its partial logits [T,per] on ITS device, then
// the group AllGather concatenates the per-rank vocab slices into the FULL
// [T,V] logits on every rank (disjoint columns -> a concatenating gather,
// mirroring vLLM's TensorParallel lm_head AllGather). Returns 0 on parity with
// the full-vocab host reference, 1 mismatch, 2 on <2 GPUs (NCCL unbuilt).
// ---------------------------------------------------------------------------
__global__ void LmHeadShardPartial(const float* dx, const float* dw, float* dlog,
                                    int M, int D, int pervoc) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // [M*pervoc] this rank
  const int total = M * pervoc;
  if (idx >= total) return;
  const int mv = idx % pervoc, m = idx / pervoc;
  const float* xm = dx + (size_t)m * D;
  const float* wr = dw + (size_t)mv * D;  // this rank's vocab-row slice
  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += xm[d] * wr[d];
  dlog[idx] = acc;
}

extern "C" int vt_cuda_lm_head_shard_run(int T, int H, int V,
                                          const float* x, const float* w, float* out) {
  if (T <= 0 || H <= 0 || V <= 0) return 1;
  vt::CudaCommGroup* g = vt::CudaCommGroup::Create();
  if (!g) return 2;
  const int W = g->world_size();
  if (V % W != 0) { delete g; return 1; }
  const int per = V / W;

  // Host reference over the full vocab.
  std::vector<float> ref((size_t)T * V);
  for (int m = 0; m < T; ++m)
    for (int v = 0; v < V; ++v) {
      double acc = 0.0;
      for (int d = 0; d < H; ++d)
        acc += (double)x[(size_t)m * H + d] * (double)w[(size_t)v * H + d];
      ref[(size_t)m * V + v] = (float)acc;
    }

  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int r = 0; r < W; ++r) {
    threads.emplace_back([&, r]() {
      if (bad.load(std::memory_order_relaxed)) return;
      NcclDevScope scope(r);
      const size_t tx = (size_t)T * H, tl = (size_t)T * per, dl = (size_t)per * H;
      cudaStream_t st = nullptr;
      float *dx = nullptr, *dw = nullptr, *dlog = nullptr, *dg = nullptr;
      if (cudaStreamCreate(&st) != cudaSuccess ||
          cudaMalloc(&dx, tx * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dw, dl * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dlog, tl * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&dg, (size_t)T * V * sizeof(float)) != cudaSuccess) {
        bad.store(1, std::memory_order_relaxed);
        cudaFree(dx); cudaFree(dw); cudaFree(dlog); cudaFree(dg);
        if (st) cudaStreamDestroy(st);
        return;
      }
      cudaMemcpyAsync(dx, x, tx * sizeof(float), cudaMemcpyHostToDevice, st);
      cudaMemcpyAsync(dw, w + (size_t)(per * r * H), dl * sizeof(float),
                      cudaMemcpyHostToDevice, st);
      cudaStreamSynchronize(st);
      LmHeadShardPartial<<<(unsigned)((T * per + 31) / 32), 32, 0, st>>>(
          dx, dw, dlog, T, H, per);
      Queue qq; qq.device = Device{DeviceType::kCUDA, r}; qq.handle = st;
      // AllGather the per-rank [T,per] shards into full [T,V] on every rank.
      g->Rank(r)->AllGather(qq, dlog, dg, (size_t)(T * per), DType::kF32);
      // NCCL AllGather returns rank-blocked: rank r's T*per logits at back[r*(T*per)..].
      // Reassemble token-major [T,V] into `out` (rank 0) and compare per local vocab.
      std::vector<float> back((size_t)T * V, 0.f);
      cudaMemcpyAsync(back.data(), dg, (size_t)T * V * sizeof(float),
                      cudaMemcpyDeviceToHost, st);
      cudaStreamSynchronize(st);
      // back (gathered, replicated on every rank) is rank-blocked:
      // back[p*(T*per) + m*per + v_local] for rank p, token m, local vocab v_local.
      // Reassemble to token-major [T,V] (m*V + p*per + v_local) on each rank.
      std::vector<float> full(T * V, 0.f);
      for (int rp = 0; rp < W; ++rp)
        for (int m = 0; m < T; ++m)
          for (int v = 0; v < per; ++v)
            full[(size_t)m * V + (rp * per + v)] =
                back[(size_t)rp * T * per + m * per + v];
      if (r == 0 && out != nullptr) std::copy(full.begin(), full.end(), out);
      for (int m = 0; m < T; ++m)
        for (int v = 0; v < V; ++v) {
          const float got = full[(size_t)m * V + v], want = ref[(size_t)m * V + v];
          if (std::fabs(got - want) > 1e-4f) {
            fprintf(stderr, "vt lm_head_shard: r %d m %d v %d %.3f want %.3f\n",
                    r, m, v, (double)got, (double)want);
            bad.store(1, std::memory_order_relaxed);
          }
        }
      cudaFree(dx); cudaFree(dw); cudaFree(dlog); cudaFree(dg); cudaStreamDestroy(st);
    });
  }
  for (auto& t : threads) t.join();
  delete g;
  return bad.load();
}

}  // namespace
}  // namespace vt
#endif  // VT_NCCL

