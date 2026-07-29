// NCCL tensor-parallel transport for vt::Communicator on kCUDA
// (BACKEND-DISTRIBUTED-COMM W2, .agents/specs/scale-out-distributed.md §W2/§W3).
//
// DERIVE-AND-SHIP. There is NO >=2-GPU box in this environment, so this transport
// is NOT run and — because it is a CUDA TU with NCCL only under -DVLLM_CPP_NCCL=ON
// — is build-verified only on a CUDA+NCCL host. Every call is ported 1:1 from
// vLLM's PyNcclCommunicator (vllm/distributed/device_communicators/pynccl.py @
// 555967922) so it drops straight onto the same NCCL the reference uses (MIRROR
// policy, memory `mirror-vllm-always-no-asking`):
//   * ncclGetUniqueId / ncclCommInitRank ....... pynccl.py:129-139
//   * ncclAllReduce ............................ pynccl.py:166-188
//   * ncclAllGather ............................ pynccl.py:196-215
//   * ncclSend ................................. pynccl.py:305-323
//   * ncclRecv ................................. pynccl.py:332-350
//   * ncclRedOp_t / ncclDataType_t map ......... pynccl_wrapper.py (ncclRedOpTypeEnum,
//                                                ncclDataTypeEnum)
// Each collective is stream-ordered on the vt::Queue's CUDA stream (q.handle),
// exactly as pynccl passes `cudaStream` into every op (pynccl.py:186 etc).
//
// The transport is routed through OpProvider/OpId (kAllReduce/kAllGather/kSend/
// kRecv on kCUDA) so the model forward and the vt::Communicator interface never
// name NCCL — a backend SUPPLIES the transport, the CPU in-process group and this
// NCCL group being two providers of the one seam.
#include "vt/communicator.h"

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

#if defined(VT_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>

#include <stdexcept>
#include <string>
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

// vt::DType -> ncclDataType_t (pynccl_wrapper.ncclDataTypeEnum). Only the dtypes
// a TP all-reduce/all-gather moves are mapped; block-quantized dtypes never cross
// a collective (the RowParallel partial sums are bf16/f32, linear.py:1766).
ncclDataType_t NcclDType(DType dt) {
  switch (dt) {
    case DType::kF32:  return ncclFloat32;
    case DType::kF16:  return ncclFloat16;
    case DType::kBF16: return ncclBfloat16;
    case DType::kI8:   return ncclInt8;
    case DType::kI32:  return ncclInt32;
    case DType::kI64:  return ncclInt64;
    default:
      throw std::runtime_error("NCCL: unsupported collective dtype");
  }
}

ncclRedOp_t NcclRedOp(ReduceOp op) {
  switch (op) {
    case ReduceOp::kSum:  return ncclSum;
    case ReduceOp::kMax:  return ncclMax;
    case ReduceOp::kMin:  return ncclMin;
    case ReduceOp::kProd: return ncclProd;
  }
  return ncclSum;
}

cudaStream_t Stream(const Queue& q) {
  return static_cast<cudaStream_t>(q.handle);
}

// A process-group over a real NCCL communicator. One per rank; the ncclComm_t is
// built from a unique_id broadcast out-of-band by the launcher (pynccl.py:129-139
// broadcasts the id on the CPU group before every rank calls ncclCommInitRank).
class NcclCommunicator final : public Communicator {
 public:
  NcclCommunicator(ncclComm_t comm, int rank, int world)
      : comm_(comm), rank_(rank), world_(world) {}

  int rank() const override { return rank_; }
  int world_size() const override { return world_; }

  void AllReduce(Queue& q, void* data, size_t count, DType dtype,
                 ReduceOp op) override {
    if (world_ == 1) return;  // GroupCoordinator world_size==1 bypass.
    // In-place all-reduce (sendbuff == recvbuff), pynccl.py:166-188.
    NcclCheck(ncclAllReduce(data, data, count, NcclDType(dtype), NcclRedOp(op),
                            comm_, Stream(q)),
              "ncclAllReduce");
  }

  void AllGather(Queue& q, const void* sendbuf, void* recvbuf, size_t count,
                 DType dtype) override {
    if (world_ == 1) return;
    // recvbuf holds world_*count; rank r lands at r*count (pynccl.py:196-215).
    NcclCheck(ncclAllGather(sendbuf, recvbuf, count, NcclDType(dtype), comm_,
                            Stream(q)),
              "ncclAllGather");
  }

  void Send(Queue& q, const void* data, size_t count, DType dtype,
            int peer) override {
    NcclCheck(ncclSend(data, count, NcclDType(dtype), peer, comm_, Stream(q)),
              "ncclSend");  // pynccl.py:305-323
  }

  void Recv(Queue& q, void* data, size_t count, DType dtype,
            int peer) override {
    NcclCheck(ncclRecv(data, count, NcclDType(dtype), peer, comm_, Stream(q)),
              "ncclRecv");  // pynccl.py:332-350
  }

 private:
  ncclComm_t comm_;
  const int rank_;
  const int world_;
};

// kCUDA collective providers: the NcclCommunicator's own methods ARE the data
// plane, so the OpProvider routing simply forwards to them (the CUDA-bound
// Communicator IS an NcclCommunicator). Symmetric with the kCPU providers in
// communicator.cpp; a build linked against NCCL wins kCUDA here.
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

#endif  // VT_NCCL

}  // namespace
}  // namespace vt

// Without -DVLLM_CPP_NCCL the TU intentionally realizes NO kCUDA collective
// provider: a CUDA build lacking NCCL leaves GetOp(kAllReduce, kCUDA) unresolved,
// so an attempted CUDA tensor-parallel run fails LOUDLY (GetOp throws "no provider
// registered") rather than silently degrading — the honest "transport not built"
// signal. The stub is NAMED here for grep/discoverability.
namespace vt {
namespace nccl_transport_stub {
inline constexpr const char* kName = "nccl (kCUDA collective transport)";
}  // namespace nccl_transport_stub
}  // namespace vt
