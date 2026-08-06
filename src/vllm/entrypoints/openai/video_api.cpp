// See include/vllm/entrypoints/openai/video_api.h. Request contract + job
// lifecycle only; the library never spawns a process.
#include "vllm/entrypoints/openai/video_api.h"

#include <nlohmann/json.hpp>

#include "vt/dtype.h"

namespace vllm::openai {
namespace {

double ReadNumber(const nlohmann::json& body, const char* key, double fallback) {
  if (!body.contains(key) || body.at(key).is_null()) return fallback;
  VT_CHECK(body.at(key).is_number(), "video request: field must be a number");
  return body.at(key).get<double>();
}

}  // namespace

VideoRequest ParseVideoRequest(const std::string& body) {
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(body);
  } catch (const std::exception&) {
    VT_CHECK(false, "video request: body is not valid JSON");
  }
  VT_CHECK(json.is_object(), "video request: body must be a JSON object");
  VT_CHECK(json.contains("prompt") && json.at("prompt").is_string(),
           "video request: `prompt` is required and must be a string");

  VideoRequest out;
  out.prompt = json.at("prompt").get<std::string>();
  if (json.contains("task") && !json.at("task").is_null()) {
    VT_CHECK(json.at("task").is_string(), "video request: `task` must be a string");
    out.task = json.at("task").get<std::string>();
  }
  // vLLM-Omni carries the generation knobs under `extra_params`; accept them at
  // the top level too so a plain client does not have to nest.
  const nlohmann::json& extra =
      (json.contains("extra_params") && json.at("extra_params").is_object()) ? json.at("extra_params")
                                                                            : json;
  out.duration_seconds = ReadNumber(extra, "duration", 0.0);
  out.num_frames = static_cast<int64_t>(ReadNumber(extra, "num_frames", 0.0));
  out.height = static_cast<int64_t>(ReadNumber(json, "height", 0.0));
  out.width = static_cast<int64_t>(ReadNumber(json, "width", 0.0));
  out.num_inference_steps = static_cast<int64_t>(ReadNumber(extra, "num_inference_steps", 50.0));
  out.flow_shift = ReadNumber(extra, "flow_shift", 12.0);
  out.audio_flow_shift = ReadNumber(extra, "audio_flow_shift", 3.0);
  if (extra.contains("seed") && !extra.at("seed").is_null()) {
    VT_CHECK(extra.at("seed").is_number(), "video request: `seed` must be a number");
    out.seed = extra.at("seed").get<int64_t>();
    out.has_seed = true;
  }

  VT_CHECK(out.num_inference_steps > 0, "video request: `num_inference_steps` must be > 0");
  VT_CHECK(out.flow_shift > 0.0 && out.audio_flow_shift > 0.0,
           "video request: flow shifts must be > 0");
  VT_CHECK(out.duration_seconds >= 0.0, "video request: `duration` must be >= 0");
  VT_CHECK(out.height >= 0 && out.width >= 0, "video request: height/width must be >= 0");
  return out;
}

const char* VideoJobStatusName(VideoJobStatus status) {
  switch (status) {
    case VideoJobStatus::kQueued: return "queued";
    case VideoJobStatus::kRunning: return "running";
    case VideoJobStatus::kSucceeded: return "succeeded";
    case VideoJobStatus::kFailed: return "failed";
  }
  return "unknown";
}

std::string VideoJobStore::Create() {
  std::lock_guard<std::mutex> guard(mutex_);
  VideoJob job;
  job.id = "vid_" + std::to_string(++next_);
  job.status = VideoJobStatus::kQueued;
  jobs_[job.id] = job;
  return job.id;
}

void VideoJobStore::MarkRunning(const std::string& id) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = jobs_.find(id);
  VT_CHECK(it != jobs_.end(), "video job store: unknown job id");
  VT_CHECK(it->second.status == VideoJobStatus::kQueued,
           "video job store: only a queued job may start running");
  it->second.status = VideoJobStatus::kRunning;
}

void VideoJobStore::MarkSucceeded(const std::string& id, const std::string& output_path) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = jobs_.find(id);
  VT_CHECK(it != jobs_.end(), "video job store: unknown job id");
  VT_CHECK(it->second.status == VideoJobStatus::kRunning,
           "video job store: only a running job may succeed");
  VT_CHECK(!output_path.empty(), "video job store: a succeeded job needs an output path");
  it->second.status = VideoJobStatus::kSucceeded;
  it->second.output_path = output_path;
}

void VideoJobStore::MarkFailed(const std::string& id, const std::string& error) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = jobs_.find(id);
  VT_CHECK(it != jobs_.end(), "video job store: unknown job id");
  VT_CHECK(it->second.status == VideoJobStatus::kQueued ||
               it->second.status == VideoJobStatus::kRunning,
           "video job store: only a pending job may fail");
  it->second.status = VideoJobStatus::kFailed;
  it->second.error = error.empty() ? "unspecified error" : error;
}

bool VideoJobStore::Get(const std::string& id, VideoJob* out) const {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto it = jobs_.find(id);
  if (it == jobs_.end()) return false;
  if (out != nullptr) *out = it->second;
  return true;
}

int64_t VideoJobStore::Size() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return static_cast<int64_t>(jobs_.size());
}

std::string VideoJobStatusJson(const VideoJob& job) {
  nlohmann::json out;
  out["id"] = job.id;
  out["status"] = VideoJobStatusName(job.status);
  if (job.status == VideoJobStatus::kSucceeded) out["output_path"] = job.output_path;
  if (job.status == VideoJobStatus::kFailed) out["error"] = job.error;
  return out.dump();
}

}  // namespace vllm::openai
