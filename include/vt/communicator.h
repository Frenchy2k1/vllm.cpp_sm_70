// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror
// of the C++ shape, but the CONTRACT is ported 1:1 from vLLM's collective seam:
//   * vllm/distributed/device_communicators/base_device_communicator.py:147
//     (DeviceCommunicatorBase) — the backend-plural interface whose
//     all_reduce:215 / all_gather:219 / send:321 / recv:328 this file mirrors.
//   * vllm/distributed/parallel_state.py:358 (GroupCoordinator) — holds a
//     DeviceCommunicatorBase and BYPASSES every collective at world_size == 1
//     (parallel_state.py:638 guard), which is exactly our world_size()==1 no-op
//     that keeps the single-GPU engine byte-identical.
//
// SCOPE (scale-out W1, .agents/specs/scale-out-distributed.md §W1). This is the
// proven ABSTRACTION plus a CPU in-process transport that gates AllReduce /
// AllGather / Send / Recv correctness with NO GPU. The NCCL (kCUDA), multi-Spark
// RoCE, and MLX-ring (kMETAL) transports are W2+ residuals (HW-gated). Routing
// collectives through OpProvider/OpId (kAllReduce, ...) is likewise a W2 residual
// — a direct Communicator method is the cleanest correctness gate for W1, and the
// stream-ordered signature (every method takes a Queue&) already composes with
// compute via the existing Backend::RecordEvent/QueueWaitEvent event machinery
// (include/vt/backend.h:87-104), so the CUDA transport is a drop-in later.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {

// Elementwise reduction for AllReduce, mirroring NCCL's ncclRedOp_t / MLX's
// all_sum|all_max|all_min (scale-out-distributed.md §Leg 3). kSum is the one the
// TP RowParallel/VocabParallel all-reduce needs (linear.py:1766); the others are
// carried so the interface is complete.
enum class ReduceOp : uint8_t { kSum = 0, kMax, kMin, kProd };

// A process-group bound to a device, sibling of vt::Queue (include/vt/device.h:50).
// Port of DeviceCommunicatorBase (base_device_communicator.py:147). Every method
// is stream-ordered (takes a Queue&) so a real transport can order the collective
// against compute through the event machinery; on a synchronous CPU backend the
// queue carries no pending work, so ordering is implicit.
//
// world_size()==1 ⇒ every collective is a no-op / identity copy: this is the
// invariant that leaves the single-GPU path byte-identical (GroupCoordinator's
// world_size==1 bypass, parallel_state.py:638).
class Communicator {
 public:
  virtual ~Communicator() = default;

  virtual int rank() const = 0;
  virtual int world_size() const = 0;

  // In-place all-reduce: on return every rank's `data` (count elements of
  // `dtype`) holds the elementwise `op` reduction across ALL ranks. Mirrors
  // DeviceCommunicatorBase.all_reduce (base_device_communicator.py:215).
  // world_size()==1 ⇒ no-op (data already equals the reduction of itself).
  virtual void AllReduce(Queue& q, void* data, size_t count, DType dtype,
                         ReduceOp op) = 0;

  // All-gather: `sendbuf` holds `count` elements; `recvbuf` must hold
  // world_size()*count, with rank r's contribution landing at
  // recvbuf[r*count ..]. Mirrors DeviceCommunicatorBase.all_gather
  // (base_device_communicator.py:219). world_size()==1 ⇒ memcpy sendbuf→recvbuf
  // (byte-identical identity).
  virtual void AllGather(Queue& q, const void* sendbuf, void* recvbuf,
                         size_t count, DType dtype) = 0;

  // Point-to-point. Send blocks until the matching Recv on `peer` has consumed
  // the buffer (a rendezvous, mirroring GroupCoordinator.send/recv
  // parallel_state.py:1185/1192 — the PP inter-stage transfer). `peer` is a rank.
  virtual void Send(Queue& q, const void* data, size_t count, DType dtype,
                    int peer) = 0;
  virtual void Recv(Queue& q, void* data, size_t count, DType dtype,
                    int peer) = 0;
};

// --- OpProvider-routed collective data planes (BACKEND-DISTRIBUTED-COMM W2) ----
// The collectives are dispatched through OpProvider/OpId (kAllReduce/kAllGather/
// kSend/kRecv, vt/ops.h) keyed on the queue's DeviceType, so a backend SUPPLIES
// the transport: the CPU in-process group registers these on kCPU, an NCCL
// provider registers them on kCUDA (nccl_communicator.cu), MLX-ring on kMETAL.
// The bound Communicator (which owns rank/world + the rendezvous machinery) is
// passed as the first argument; the fn performs the device-specific data
// movement. Mirrors vLLM keying its DeviceCommunicatorBase per device
// (cuda_communicator.py / cpu_communicator.py). Type-erased through the same
// `void*` OpProvider::fn every op uses (RegisterOp / GetOp, include/vt/ops.h).
using CommAllReduceFn = void (*)(Communicator&, Queue&, void*, size_t, DType,
                                 ReduceOp);
using CommAllGatherFn = void (*)(Communicator&, Queue&, const void*, void*,
                                 size_t, DType);
using CommSendFn = void (*)(Communicator&, Queue&, const void*, size_t, DType,
                            int);
using CommRecvFn = void (*)(Communicator&, Queue&, void*, size_t, DType, int);

// In-process multi-rank CPU transport: N ranks live as N Communicator objects in
// ONE process, sharing a barrier + staging buffers + a point-to-point mailbox —
// no IPC. This is the cleanest correctness gate for the collectives (the spec's
// W1 decision, scale-out-distributed.md §Risks: "W1's CPU-loopback provider is
// the decision that lets the abstraction land WITHOUT a GPU"). A real cross-rank
// sum on CPU is the deliverable; ranks are driven by N host threads in the test.
//
// It also sidesteps the W2 blocker: the one-Backend*-per-DeviceType registry
// (backend.cpp:42) + device-0 hardcoding (cuda_backend.cu:297) that a real
// per-device NCCL transport needs are untouched this pass.
class CpuCommGroup {
 public:
  explicit CpuCommGroup(int world_size);
  ~CpuCommGroup();

  CpuCommGroup(const CpuCommGroup&) = delete;
  CpuCommGroup& operator=(const CpuCommGroup&) = delete;

  int world_size() const;

  // The Communicator for rank r (0 <= r < world_size()). Owned by the group;
  // valid for the group's lifetime. Hand rank r's handle to thread r.
  Communicator& Rank(int r);

  // Opaque coordination state (barrier + staging + mailbox), defined in the .cpp.
  // Named publicly only so the .cpp's rank-communicator can hold a shared_ptr to
  // it; the definition — and every field — stays private to the translation unit.
  struct Shared;

 private:
  std::shared_ptr<Shared> shared_;
  std::vector<std::unique_ptr<Communicator>> ranks_;
};

}  // namespace vt
