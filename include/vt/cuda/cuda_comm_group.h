// vllm.cpp original (vt runtime). Multi-device (TP) phase-3: a process-local,
// NCCL-backed group over every GPU on this node — the reusable handle the
// runner passes to each layer's `TensorParallel` (vs. the bare self-check).
//
// One communicator per local GPU (rank == device index, world == GPU count),
// built with ncclCommInitAll inside the anonymous namespace of
// src/vt/cuda/nccl_communicator.cu. Only meaningful when compiled with
// VT_NCCL; without VT_NCCL the factory is not compiled (a CUDA build without
// NCCL should not even reach TensorParallel>1 — it fails loudly in the op
// layer instead).
#pragma once

#include <cstdint>

#include "vt/communicator.h"

namespace vt {

// A process-local NCCL collective group over every discrete GPU. DevNull on
// single-GPU: the caller uses tp_size()/rank() the same way as TensorParallel.
class CudaCommGroup {
 public:
  // Creates one NcclCommunicator per local GPU (world = GPU count). Returns
  // nullptr and clears err if fewer than 2 GPUs or NCCL init fails.
  static CudaCommGroup* Create();

  int world_size() const { return world_; }
  int rank() const { return rank_; }
  // The communicator bound to THIS rank (the one TensorParallel wraps).
  Communicator* Rank(int r) const;

  ~CudaCommGroup();
  CudaCommGroup(const CudaCommGroup&) = delete;
  CudaCommGroup& operator=(const CudaCommGroup&) = delete;

 private:
  CudaCommGroup();
  void* impl_ = nullptr;  // owned by the impl TU (opaque CudaCommGroupImpl)
  int rank_;
  int world_;
};

}  // namespace vt