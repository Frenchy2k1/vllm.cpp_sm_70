// Capture-aware current-device bind for ROCm.
// hipSetDevice is illegal during hipGraph capture; stream launches carry device.
// Outside capture, bind so peer-MoE can leave current device on the expert GPU.
#pragma once

#include <hip/hip_runtime.h>

#include "vt/device.h"

namespace vt::rocm {

inline bool StreamIsCapturing(hipStream_t st) {
  if (st == nullptr) return false;
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(st, &status) != hipSuccess) return false;
  return status != hipStreamCaptureStatusNone;
}

// Bind process current device to q.device when not capturing. No-op if already set
// or if the stream is mid-capture (graph-safe).
inline void EnsureQueueDevice(const Queue& q) {
  const int dev = q.device.index;
  if (dev < 0) return;
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  if (StreamIsCapturing(st)) return;
  int cur = -1;
  if (hipGetDevice(&cur) == hipSuccess && cur == dev) return;
  (void)hipSetDevice(dev);
}

}  // namespace vt::rocm
