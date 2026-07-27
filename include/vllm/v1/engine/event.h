// Ported from: vllm/v1/engine/__init__.py @ 555967922 (vLLM 0.26.0.dev0)
//   EngineCoreEventType (:150-155) + EngineCoreEvent (:158-176).
//
// A timestamped, per-request engine-core event. The scheduler records these at
// the request QUEUED / SCHEDULED / PREEMPTED sites (all gated on log_stats);
// they ride out on EngineCoreOutput.events, and the frontend's IterationStats
// folds them into the per-request queue / prefill / inference timing intervals
// and the preemption counter (vllm/v1/metrics/stats.py update_from_events /
// update_from_finished_request).
//
// The timestamp is a monotonic clock reading (upstream time.monotonic(); here
// MonotonicSeconds() — the SAME steady clock EngineCoreOutputs.timestamp and
// RequestState timing use, so every derived interval is non-negative). Upstream
// warns these are only comparable within one process; the same holds here.
//
// This is a small standalone header (mirroring the fact that upstream defines
// the type in engine/__init__.py, which request.py imports) so that both
// vllm/v1/request.h (Request.events) and vllm/v1/engine/types.h
// (EngineCoreOutput.events) can use it without a circular include.
#pragma once

#include <optional>

#include "vllm/v1/metrics/stats.h"  // vllm::v1::MonotonicSeconds

namespace vllm::v1 {

// EngineCoreEventType (IntEnum): the type of a request engine-core event. Int
// values are load-bearing upstream (compact msgspec serialization); preserved.
enum class EngineCoreEventType : int {
  kQueued = 1,
  kScheduled = 2,
  kPreempted = 3,
};

// EngineCoreEvent (msgspec.Struct): a timestamped event associated with a
// request. A plain value carrier here.
struct EngineCoreEvent {
  EngineCoreEventType type;
  double timestamp;

  // new_event (:169-176): stamp the current monotonic time when no timestamp is
  // supplied (the QUEUED site), else use the caller's shared step timestamp (the
  // SCHEDULED / PREEMPTED sites reuse the one schedule() reading).
  static EngineCoreEvent new_event(
      EngineCoreEventType event_type,
      std::optional<double> timestamp = std::nullopt) {
    return EngineCoreEvent{event_type,
                           timestamp.has_value() ? *timestamp
                                                 : MonotonicSeconds()};
  }
};

}  // namespace vllm::v1
