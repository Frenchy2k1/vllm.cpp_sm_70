// MiniMax-H3 `/v1/videos` serving surface — request parsing and the job store.
//
// Mirrors vLLM-Omni's two endpoints:
//   POST /v1/videos       -> enqueue, return a job id immediately (async)
//   POST /v1/videos/sync  -> run to completion, return the MP4 in the body
//
// THE PROCESS BOUNDARY (developer-ratified 2026-08-03): the library never spawns
// a process. Generation and muxing are supplied by the caller as a `VideoRunner`
// callback; `examples/` provides one that invokes ffmpeg with the argv built by
// MiniMaxH3BuildMp4MuxArgs. So this header owns the REQUEST CONTRACT and the JOB
// LIFECYCLE, both pure and unit-testable, and nothing else.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace vllm::openai {

// One parsed /v1/videos request. Mirrors the fields vLLM-Omni accepts; anything
// absent falls back to H3's documented defaults via the shape planner.
struct VideoRequest {
  std::string prompt;
  std::string task;            // "" => resolved from the partition + inputs
  double duration_seconds = 0.0;  // <= 0 => per-task default
  int64_t num_frames = 0;         // <= 1 => per-task default
  int64_t height = 0, width = 0;  // <= 0 => aspect-derived default
  int64_t num_inference_steps = 50;
  double flow_shift = 12.0;        // video
  double audio_flow_shift = 3.0;   // audio
  int64_t seed = 0;
  bool has_seed = false;
};

// Parse + validate a request body. Throws (VT_CHECK) with a specific message on
// malformed input rather than silently defaulting, so a bad request is a 400 with
// a reason instead of a surprising generation.
VideoRequest ParseVideoRequest(const std::string& body);

enum class VideoJobStatus { kQueued, kRunning, kSucceeded, kFailed };

const char* VideoJobStatusName(VideoJobStatus status);

struct VideoJob {
  std::string id;
  VideoJobStatus status = VideoJobStatus::kQueued;
  std::string output_path;  // set on success
  std::string error;        // set on failure
};

// A minimal job registry for the async endpoint. Thread-safe: the HTTP worker
// pool touches it from several threads.
class VideoJobStore {
 public:
  // Creates a job in `kQueued` and returns its id.
  std::string Create();
  // Legal transitions only: queued -> running -> {succeeded, failed}. An illegal
  // transition throws rather than corrupting the record.
  void MarkRunning(const std::string& id);
  void MarkSucceeded(const std::string& id, const std::string& output_path);
  void MarkFailed(const std::string& id, const std::string& error);
  // False when the id is unknown (a 404 rather than an empty 200).
  bool Get(const std::string& id, VideoJob* out) const;
  int64_t Size() const;

 private:
  mutable std::mutex mutex_;
  std::map<std::string, VideoJob> jobs_;
  int64_t next_ = 0;
};

// The caller-supplied generation+mux step. Returns the produced .mp4 path, or
// throws to fail the job. `examples/` implements this with ffmpeg.
using VideoRunner = std::function<std::string(const VideoRequest&)>;

// The JSON body for a job-status query (`GET /v1/videos/{id}`).
std::string VideoJobStatusJson(const VideoJob& job);

}  // namespace vllm::openai
