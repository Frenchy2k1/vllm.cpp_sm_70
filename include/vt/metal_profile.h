// Metal dispatch profiling — the in-process substitute for a Metal System
// Trace (BACKEND-METAL-MLX, .agents/specs/metal-dispatch-attribution.md).
//
// WHY THIS EXISTS AT ALL. The benchmark protocol requires that a throughput
// claim be backed by an execution trace, not by reading the dispatch code
// (AGENTS.md § "TRACE THE EXECUTION"). On CUDA that is `nsys`. On Apple the
// equivalent is Instruments' Metal System Trace, which needs a full Xcode; the
// project's M4 has Command Line Tools only, so `xctrace` is not available and
// no external profiler can be run there. This collects the same attribution
// from inside the process instead, using timestamps Metal exposes at runtime.
//
// WHAT IT SEPARATES, and why those three phases. Each dispatch is split into:
//   encode_s — Encoder construction through the start of commit: the pipeline
//              lookup and argument binds, i.e. HOST work.
//   wait_s   — wall time inside `commit` + `waitUntilCompleted`.
//   gpu_s    — `MTLCommandBuffer.GPUEndTime - GPUStartTime`, the interval the
//              GPU was actually executing this command buffer.
// `gpu_s / wait_s` is the number that decides where optimisation can pay. Near
// 1.0 means the backend is genuinely compute bound and kernel work is the
// lever. Far below 1.0 means it is submit/synchronisation bound, and no amount
// of kernel tuning can reach the difference — which is exactly what the first
// measurement found (33% to 67% depending on arm).
//
// COST WHEN OFF: one relaxed atomic load per dispatch, next to an operation
// that already blocks on the GPU. Off unless `VT_METAL_PROFILE=1` or
// `SetProfileEnabled(true)`.
//
// THREAD SAFETY: all four entry points are safe to call from any thread.
// Accumulation takes a mutex, which is irrelevant beside a command-buffer round
// trip (measured at ~186 us on an M4).
#pragma once

#include <string>
#include <vector>

namespace vt {
namespace metal {

// One accumulated row. `name` is the MSL kernel function name for a per-kernel
// row, and empty for the process total.
struct ProfileRow {
  std::string name;
  unsigned long long count = 0;
  double encode_s = 0.0;
  double wait_s = 0.0;
  double gpu_s = 0.0;
};

// Is accumulation on? Reflects `VT_METAL_PROFILE=1` at first use, plus any
// later `SetProfileEnabled`.
bool ProfileEnabled();

// Turn accumulation on or off programmatically. Tests use this rather than
// depending on the environment; it does NOT enable the end-of-process report,
// which stays tied to the environment variable so a test cannot spam stderr.
void SetProfileEnabled(bool on);

// Every per-kernel row accumulated so far, plus (as the last element, with an
// empty `name`) the total across all kernels. Empty when nothing was recorded.
std::vector<ProfileRow> GetProfileRows();

// Drop all accumulated rows. Does not change the enabled flag.
void ResetProfile();

// Command buffers COMMITTED since the last ResetProfile. Distinct from the
// dispatch count, and the whole point of `M3c-1`: batching many dispatches into
// one command buffer is visible HERE and nowhere else, because a dispatch count
// alone cannot tell a batched submission from a serialised one.
unsigned long long GetProfileCommits();

}  // namespace metal
}  // namespace vt
