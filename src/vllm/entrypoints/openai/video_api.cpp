// See include/vllm/entrypoints/openai/video_api.h. Request contract + job
// lifecycle only; the library never spawns a process.
#include "vllm/entrypoints/openai/video_api.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include "vt/dtype.h"

namespace vllm::openai {
namespace {

double ReadNumber(const nlohmann::json& body, const char* key, double fallback) {
  if (!body.contains(key) || body.at(key).is_null()) return fallback;
  VT_CHECK(body.at(key).is_number(), "video request: field must be a number");
  return body.at(key).get<double>();
}

// Present-and-not-null. `false` means "the caller did not specify this", which is
// what the OpenAI-vs-native precedence turns on: an OMITTED native field yields to
// the OpenAI alias, an explicit one (even 0) does not.
bool Has(const nlohmann::json& body, const char* key) {
  return body.contains(key) && !body.at(key).is_null();
}

// A whole non-negative decimal integer, no sign, no spaces, no exponent. Returns
// false on anything else, INCLUDING an empty string or trailing junk ("720p").
bool ParseWholeNumber(const std::string& text, int64_t* out) {
  if (text.empty() || text.size() > 18) return false;
  int64_t value = 0;
  for (const char ch : text) {
    if (std::isdigit(static_cast<unsigned char>(ch)) == 0) return false;
    value = value * 10 + (ch - '0');
  }
  *out = value;
  return true;
}

// OpenAI sends `seconds` as a STRING ("4", "8", "12" in the Sora enum), while our
// native `duration` is a number. Accept both spellings of the VALUE, so a client
// that follows the OpenAI schema literally is not rejected on a type.
double ReadDuration(const nlohmann::json& body, const char* key) {
  const nlohmann::json& value = body.at(key);
  if (value.is_number()) return value.get<double>();
  VT_CHECK(value.is_string(),
           "video request: `seconds` must be a number or a numeric string");
  const std::string text = value.get<std::string>();
  try {
    size_t consumed = 0;
    const double parsed = std::stod(text, &consumed);
    VT_CHECK(consumed == text.size(),
             "video request: `seconds` is not a number: '" + text + "'");
    // std::stod also accepts "inf"/"nan"/hex; a duration must be a real number.
    VT_CHECK(std::isfinite(parsed),
             "video request: `seconds` must be a finite number: '" + text + "'");
    return parsed;
  } catch (const std::invalid_argument&) {
    VT_CHECK(false, "video request: `seconds` is not a number: '" + text + "'");
  } catch (const std::out_of_range&) {
    VT_CHECK(false, "video request: `seconds` is out of range: '" + text + "'");
  }
  return 0.0;  // unreachable; VT_CHECK(false) throws
}

}  // namespace

void ParseVideoSize(const std::string& size, int64_t* width, int64_t* height) {
  const size_t sep = size.find_first_of("xX");
  VT_CHECK(sep != std::string::npos,
           "video request: `size` must be \"<width>x<height>\", got '" + size + "'");
  VT_CHECK(size.find_first_of("xX", sep + 1) == std::string::npos,
           "video request: `size` must carry exactly one 'x', got '" + size + "'");
  int64_t w = 0, h = 0;
  VT_CHECK(ParseWholeNumber(size.substr(0, sep), &w) &&
               ParseWholeNumber(size.substr(sep + 1), &h),
           "video request: `size` must be \"<width>x<height>\" in whole pixels, got '" +
               size + "'");
  VT_CHECK(w > 0 && h > 0,
           "video request: `size` must have a positive width and height, got '" + size + "'");
  *width = w;
  *height = h;
}

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
  if (Has(json, "task")) {
    VT_CHECK(json.at("task").is_string(), "video request: `task` must be a string");
    out.task = json.at("task").get<std::string>();
  }
  // OpenAI `model`. Recorded, not checked here: whether it names something this
  // server serves is the ROUTE's question (only it knows the served names), and
  // its answer is a warning on the job, never a rejection.
  if (Has(json, "model")) {
    VT_CHECK(json.at("model").is_string(), "video request: `model` must be a string");
    out.model = json.at("model").get<std::string>();
    VT_CHECK(!out.model.empty(), "video request: `model` must not be empty");
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

  // ── The OpenAI aliases. Both are VALIDATED whenever present and APPLIED only
  // where the native field was omitted, which is the precedence documented on
  // ParseVideoRequest: a body that parses today keeps its exact meaning, and a
  // body we cannot fully read is a 400 rather than a half-honoured request. ────
  if (Has(json, "size")) {
    VT_CHECK(json.at("size").is_string(),
             "video request: `size` must be a string like \"1280x720\"");
    int64_t size_w = 0, size_h = 0;
    ParseVideoSize(json.at("size").get<std::string>(), &size_w, &size_h);
    if (!Has(json, "width")) out.width = size_w;
    if (!Has(json, "height")) out.height = size_h;
  }
  // `seconds` may nest under extra_params like `duration`, but OpenAI puts it at
  // the top level, so both are looked at — extra_params first, matching how every
  // other knob resolves.
  const nlohmann::json* seconds_owner =
      Has(extra, "seconds") ? &extra : (Has(json, "seconds") ? &json : nullptr);
  if (seconds_owner != nullptr) {
    const double seconds = ReadDuration(*seconds_owner, "seconds");
    VT_CHECK(seconds > 0.0, "video request: `seconds` must be > 0");
    if (!Has(extra, "duration")) out.duration_seconds = seconds;
  }
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

std::string VideoJobStore::Create() { return Create({}, {}); }

std::string VideoJobStore::Create(std::string model, std::string warning) {
  std::lock_guard<std::mutex> guard(mutex_);
  VideoJob job;
  job.id = "vid_" + std::to_string(++next_);
  job.status = VideoJobStatus::kQueued;
  job.model = std::move(model);
  job.warning = std::move(warning);
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
  // The requested model is echoed for the job's whole life, and a mismatch rides
  // along as a `warning` — so "we used a different model than you named" is stated
  // rather than left for the client to notice in the pixels.
  if (!job.model.empty()) out["model"] = job.model;
  if (!job.warning.empty()) out["warning"] = job.warning;
  if (job.status == VideoJobStatus::kSucceeded) out["output_path"] = job.output_path;
  if (job.status == VideoJobStatus::kFailed) out["error"] = job.error;
  return out.dump();
}

}  // namespace vllm::openai
