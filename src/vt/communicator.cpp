// vllm.cpp original (vt runtime). CPU in-process multi-rank transport for the
// vt::Communicator collective abstraction (scale-out W1,
// .agents/specs/scale-out-distributed.md §W1). Ports the DeviceCommunicatorBase
// contract (base_device_communicator.py:147) onto N in-process ranks sharing one
// barrier + staging + mailbox — a real cross-rank reduction with no GPU, no IPC.
#include "vt/communicator.h"

#include <condition_variable>
#include <cstring>
#include <mutex>

#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/op_provider.h"

namespace vt {
namespace {

// Elementwise reduction of `world` same-length inputs (slots[r]) into `out`,
// accumulating in the storage type. F16/BF16 accumulate in float then round
// back, matching how a real collective would not lose the sub-epsilon terms.
template <typename T>
void ReduceTyped(T* out, const std::vector<const void*>& slots, size_t count,
                 ReduceOp op) {
  const int world = static_cast<int>(slots.size());
  for (size_t i = 0; i < count; ++i) {
    T acc = static_cast<const T*>(slots[0])[i];
    for (int r = 1; r < world; ++r) {
      const T v = static_cast<const T*>(slots[r])[i];
      switch (op) {
        case ReduceOp::kSum:
          acc = static_cast<T>(acc + v);
          break;
        case ReduceOp::kMax:
          acc = v > acc ? v : acc;
          break;
        case ReduceOp::kMin:
          acc = v < acc ? v : acc;
          break;
        case ReduceOp::kProd:
          acc = static_cast<T>(acc * v);
          break;
      }
    }
    out[i] = acc;
  }
}

// F16/BF16 path: convert every rank's element to float, reduce, round back.
template <float (*ToF32)(uint16_t), uint16_t (*FromF32)(float)>
void ReduceHalf(uint16_t* out, const std::vector<const void*>& slots,
                size_t count, ReduceOp op) {
  const int world = static_cast<int>(slots.size());
  for (size_t i = 0; i < count; ++i) {
    float acc = ToF32(static_cast<const uint16_t*>(slots[0])[i]);
    for (int r = 1; r < world; ++r) {
      const float v = ToF32(static_cast<const uint16_t*>(slots[r])[i]);
      switch (op) {
        case ReduceOp::kSum:
          acc += v;
          break;
        case ReduceOp::kMax:
          acc = v > acc ? v : acc;
          break;
        case ReduceOp::kMin:
          acc = v < acc ? v : acc;
          break;
        case ReduceOp::kProd:
          acc *= v;
          break;
      }
    }
    out[i] = FromF32(acc);
  }
}

void ReduceInto(void* out, const std::vector<const void*>& slots, size_t count,
                DType dtype, ReduceOp op) {
  switch (dtype) {
    case DType::kF32:
      ReduceTyped(static_cast<float*>(out), slots, count, op);
      return;
    case DType::kI8:
      ReduceTyped(static_cast<int8_t*>(out), slots, count, op);
      return;
    case DType::kI32:
      ReduceTyped(static_cast<int32_t*>(out), slots, count, op);
      return;
    case DType::kI64:
      ReduceTyped(static_cast<int64_t*>(out), slots, count, op);
      return;
    case DType::kF16:
      ReduceHalf<F16ToF32, F32ToF16>(static_cast<uint16_t*>(out), slots, count,
                                     op);
      return;
    case DType::kBF16:
      ReduceHalf<BF16ToF32, F32ToBF16>(static_cast<uint16_t*>(out), slots,
                                       count, op);
      return;
    default:
      VT_CHECK(false, "Communicator: block-quantized dtype not reducible");
  }
}

}  // namespace

// Shared coordination for one in-process group. All members are guarded by `mu`
// except the staging slots, which distinct ranks write at distinct indices
// (separate memory locations, no data race) with the barrier providing the
// happens-before for the subsequent reads.
struct CpuCommGroup::Shared {
  explicit Shared(int w) : world(w), slots(static_cast<size_t>(w), nullptr) {
    if (world > 1) {
      mailboxes.resize(static_cast<size_t>(world) * static_cast<size_t>(world));
    }
  }

  const int world;

  std::mutex mu;
  std::condition_variable cv;
  int arrived = 0;
  uint64_t generation = 0;

  std::vector<const void*> slots;

  // Rendezvous mailbox for Send/Recv, indexed [src * world + dst].
  struct Mailbox {
    const void* ptr = nullptr;
    size_t count = 0;
    bool full = false;
  };
  std::vector<Mailbox> mailboxes;
  std::condition_variable mb_cv;

  // Generation-counting barrier: returns once all `world` ranks have arrived.
  void Barrier() {
    std::unique_lock<std::mutex> lk(mu);
    const uint64_t gen = generation;
    if (++arrived == world) {
      arrived = 0;
      ++generation;
      cv.notify_all();
    } else {
      cv.wait(lk, [&] { return generation != gen; });
    }
  }
};

namespace {

// One rank's view of the shared group. Holds the group so a rank handle stays
// valid even if the owning CpuCommGroup's rank vector is inspected concurrently.
class CpuRankCommunicator final : public Communicator {
 public:
  CpuRankCommunicator(std::shared_ptr<CpuCommGroup::Shared> shared, int rank)
      : shared_(std::move(shared)), rank_(rank) {}

  int rank() const override { return rank_; }
  int world_size() const override { return shared_->world; }

  // The virtual overrides ROUTE through OpProvider (BACKEND-DISTRIBUTED-COMM W2):
  // the queue's DeviceType selects the transport data plane (kCPU -> the Do*
  // bodies below; kCUDA -> the NCCL provider). Dispatching for world_size==1 is
  // harmless — the Do* bodies short-circuit to the byte-identical no-op before
  // touching the group — and it keeps the single seam every backend plugs into.
  void AllReduce(Queue& q, void* data, size_t count, DType dtype,
                 ReduceOp op) override {
    auto fn = reinterpret_cast<CommAllReduceFn>(
        GetOp(OpId::kAllReduce, q.device.type));
    fn(*this, q, data, count, dtype, op);
  }
  void AllGather(Queue& q, const void* sendbuf, void* recvbuf, size_t count,
                 DType dtype) override {
    auto fn = reinterpret_cast<CommAllGatherFn>(
        GetOp(OpId::kAllGather, q.device.type));
    fn(*this, q, sendbuf, recvbuf, count, dtype);
  }
  void Send(Queue& q, const void* data, size_t count, DType dtype,
            int peer) override {
    auto fn = reinterpret_cast<CommSendFn>(GetOp(OpId::kSend, q.device.type));
    fn(*this, q, data, count, dtype, peer);
  }
  void Recv(Queue& q, void* data, size_t count, DType dtype,
            int peer) override {
    auto fn = reinterpret_cast<CommRecvFn>(GetOp(OpId::kRecv, q.device.type));
    fn(*this, q, data, count, dtype, peer);
  }

  // CPU in-process data planes (registered as the kCPU providers below). These
  // hold the REAL cross-rank reduction/rendezvous — the barrier + staging slots +
  // mailbox in Shared. Public so the free-function providers can forward here.
  void DoAllReduce(Queue& q, void* data, size_t count, DType dtype,
                   ReduceOp op) {
    (void)q;  // CPU backend is synchronous: no stream to order against.
    if (shared_->world == 1) return;  // identity: reduction of one input.
    if (count == 0) return;

    shared_->slots[static_cast<size_t>(rank_)] = data;
    shared_->Barrier();  // all inputs published before any rank reads them.

    // Reduce all ranks' (unmodified) inputs into a private buffer.
    std::vector<uint8_t> tmp(count * SizeOf(dtype));
    ReduceInto(tmp.data(), shared_->slots, count, dtype, op);

    shared_->Barrier();  // every rank has finished reading before any overwrite.
    std::memcpy(data, tmp.data(), tmp.size());
    shared_->Barrier();  // slots reusable by the next collective.
  }

  void DoAllGather(Queue& q, const void* sendbuf, void* recvbuf, size_t count,
                   DType dtype) {
    (void)q;
    const size_t elem = SizeOf(dtype);
    if (shared_->world == 1) {
      std::memcpy(recvbuf, sendbuf, count * elem);  // identity.
      return;
    }
    if (count == 0) return;

    shared_->slots[static_cast<size_t>(rank_)] = sendbuf;
    shared_->Barrier();  // all contributions published.

    auto* out = static_cast<uint8_t*>(recvbuf);
    for (int r = 0; r < shared_->world; ++r) {
      std::memcpy(out + static_cast<size_t>(r) * count * elem,
                  shared_->slots[static_cast<size_t>(r)], count * elem);
    }
    shared_->Barrier();  // all reads done before slots are reused.
  }

  void DoSend(Queue& q, const void* data, size_t count, DType dtype,
              int peer) {
    (void)q;
    if (peer == rank_) return;  // self-send is a no-op (no copy needed).
    const size_t idx =
        static_cast<size_t>(rank_) * static_cast<size_t>(shared_->world) +
        static_cast<size_t>(peer);
    std::unique_lock<std::mutex> lk(shared_->mu);
    auto& box = shared_->mailboxes[idx];
    shared_->mb_cv.wait(lk, [&] { return !box.full; });  // prior msg consumed.
    box.ptr = data;
    box.count = count;
    box.full = true;
    (void)dtype;
    shared_->mb_cv.notify_all();
    shared_->mb_cv.wait(lk, [&] { return !box.full; });  // rendezvous: consumed.
  }

  void DoRecv(Queue& q, void* data, size_t count, DType dtype,
              int peer) {
    (void)q;
    if (peer == rank_) return;
    const size_t idx =
        static_cast<size_t>(peer) * static_cast<size_t>(shared_->world) +
        static_cast<size_t>(rank_);
    std::unique_lock<std::mutex> lk(shared_->mu);
    auto& box = shared_->mailboxes[idx];
    shared_->mb_cv.wait(lk, [&] { return box.full; });  // sender has posted.
    const size_t n = (count < box.count ? count : box.count) * SizeOf(dtype);
    std::memcpy(data, box.ptr, n);
    box.full = false;
    box.ptr = nullptr;
    shared_->mb_cv.notify_all();  // release the sender.
  }

 private:
  std::shared_ptr<CpuCommGroup::Shared> shared_;
  const int rank_;
};

// kCPU collective providers (BACKEND-DISTRIBUTED-COMM W2). Every CPU-bound
// Communicator IS a CpuRankCommunicator, so the provider forwards to the Do*
// body that owns the barrier/mailbox. Registering a faster CPU reduce later is a
// provider swap here — the model forward and the Communicator interface never
// change (the OpProvider selection seam, op_provider.h).
void CpuAllReduce(Communicator& c, Queue& q, void* data, size_t count,
                  DType dtype, ReduceOp op) {
  static_cast<CpuRankCommunicator&>(c).DoAllReduce(q, data, count, dtype, op);
}
void CpuAllGather(Communicator& c, Queue& q, const void* sendbuf, void* recvbuf,
                  size_t count, DType dtype) {
  static_cast<CpuRankCommunicator&>(c).DoAllGather(q, sendbuf, recvbuf, count,
                                                   dtype);
}
void CpuSend(Communicator& c, Queue& q, const void* data, size_t count,
             DType dtype, int peer) {
  static_cast<CpuRankCommunicator&>(c).DoSend(q, data, count, dtype, peer);
}
void CpuRecv(Communicator& c, Queue& q, void* data, size_t count, DType dtype,
             int peer) {
  static_cast<CpuRankCommunicator&>(c).DoRecv(q, data, count, dtype, peer);
}

struct CpuCollectiveRegistrar {
  CpuCollectiveRegistrar() {
    RegisterOp(OpId::kAllReduce, DeviceType::kCPU,
               reinterpret_cast<void*>(&CpuAllReduce));
    RegisterOp(OpId::kAllGather, DeviceType::kCPU,
               reinterpret_cast<void*>(&CpuAllGather));
    RegisterOp(OpId::kSend, DeviceType::kCPU, reinterpret_cast<void*>(&CpuSend));
    RegisterOp(OpId::kRecv, DeviceType::kCPU, reinterpret_cast<void*>(&CpuRecv));
  }
} cpu_collective_registrar;

}  // namespace

CpuCommGroup::CpuCommGroup(int world_size)
    : shared_(std::make_shared<Shared>(world_size)) {
  VT_CHECK(world_size >= 1, "CpuCommGroup: world_size must be >= 1");
  ranks_.reserve(static_cast<size_t>(world_size));
  for (int r = 0; r < world_size; ++r) {
    ranks_.push_back(std::make_unique<CpuRankCommunicator>(shared_, r));
  }
}

CpuCommGroup::~CpuCommGroup() = default;

int CpuCommGroup::world_size() const { return shared_->world; }

Communicator& CpuCommGroup::Rank(int r) {
  VT_CHECK(r >= 0 && r < shared_->world, "CpuCommGroup::Rank: out of range");
  return *ranks_[static_cast<size_t>(r)];
}

}  // namespace vt
