// MiniMax-H3 request planning: frame/latent shapes, canvas resolution, task
// dispatch, and the rectified-flow time-shifted sigma schedule.
// Port of vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/
// time_request.py (whole file) and pipeline_minimax_h3.py:121-122, 207-222,
// 374-434 (`_align_multiple`, `_reference_image_shape`, `_resolve_task`,
// `_resolve_shape`).
//
// This is the "what does the request actually ask for" layer — the surface that
// decides t2va vs fl2va vs ref2va, how many frames get generated, on what canvas,
// and with which noise schedule. It is pure integer/float logic with no weights,
// so unlike the rest of the H3 chain it is gated EXACTLY against upstream on any
// machine (tests/vllm/models/test_minimax_h3.cpp).
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

// Python's round() is banker's rounding (round-half-to-even); C's round() is
// half-away-from-zero. `_align_multiple` and `_audio_latent_t` both go through
// Python's, so the difference is load-bearing at exact .5 boundaries.
double RoundHalfToEven(double value) {
  const double floor_value = std::floor(value);
  const double diff = value - floor_value;
  if (diff > 0.5) return floor_value + 1.0;
  if (diff < 0.5) return floor_value;
  return (std::fmod(floor_value, 2.0) == 0.0) ? floor_value : floor_value + 1.0;
}

std::string ToLower(const std::string& value) {
  std::string out = value;
  for (char& c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

}  // namespace

// _align_frame_count (time_request.py:5-12).
int64_t MiniMaxH3AlignFrameCount(int64_t frame_count) {
  if (frame_count <= 0) return 1;
  int64_t current = frame_count;
  while (current % 17 != 5) ++current;
  return current;
}

// _video_latent_t (time_request.py:15-18).
int64_t MiniMaxH3VideoLatentT(int64_t frame_count) {
  if (frame_count <= 5) return 2;
  return ((frame_count - 5) / 17) * 5 + 2;
}

// _frame_count_from_video_latent_t (time_request.py:21-26).
int64_t MiniMaxH3FrameCountFromVideoLatentT(int64_t out_t) {
  if (out_t == 1) return 1;
  VT_CHECK(out_t >= 2 && (out_t - 2) % 5 == 0,
           "minimax_h3: video latent T must be 1 or match 5n+2");
  return 17 * ((out_t - 2) / 5) + 5;
}

// _audio_latent_t (time_request.py:29-31): the 40 Hz audio latent boundary.
int64_t MiniMaxH3AudioLatentT(double duration_seconds) {
  return static_cast<int64_t>(RoundHalfToEven(duration_seconds * 40.0));
}

// _align_multiple (pipeline_minimax_h3.py:121-122).
int64_t MiniMaxH3AlignMultiple(double value, int64_t multiple) {
  VT_CHECK(multiple > 0, "minimax_h3: align multiple must be positive");
  const int64_t snapped =
      static_cast<int64_t>(RoundHalfToEven(value / static_cast<double>(multiple))) * multiple;
  return std::max(multiple, snapped);
}

// _reference_image_shape (pipeline_minimax_h3.py:207-222).
std::pair<int64_t, int64_t> MiniMaxH3ReferenceImageShape(int64_t width, int64_t height) {
  VT_CHECK(width > 0 && height > 0, "minimax_h3: reference image dims must be positive");
  VT_CHECK(width <= 4 * height && height <= 4 * width,
           "minimax_h3: reference image aspect ratio must be in [1:4, 4:1]");
  const double scale = static_cast<double>(kMiniMaxH3ReferenceImageShortEdge) /
                       static_cast<double>(std::min(width, height));
  return {MiniMaxH3AlignMultiple(static_cast<double>(width) * scale,
                                 kMiniMaxH3ReferenceImageMultiple),
          MiniMaxH3AlignMultiple(static_cast<double>(height) * scale,
                                 kMiniMaxH3ReferenceImageMultiple)};
}

// _time_shift_sigmas (time_request.py:34-61).
std::vector<double> MiniMaxH3TimeShiftSigmas(int64_t num_steps, double shift_scale) {
  VT_CHECK(shift_scale > 0.0, "minimax_h3: shift_scale must be > 0");
  VT_CHECK(num_steps > 0, "minimax_h3: num_steps must be > 0");

  // torch.linspace(1, 0, n) in FLOAT32 — the schedule is built at f32 upstream,
  // and the shift is applied to those f32 values.
  std::vector<float> base(static_cast<size_t>(num_steps));
  if (num_steps == 1) {
    base[0] = 1.0f;
  } else {
    const double step = 1.0 / static_cast<double>(num_steps - 1);
    for (int64_t i = 0; i < num_steps; ++i) {
      base[static_cast<size_t>(i)] = static_cast<float>(1.0 - static_cast<double>(i) * step);
    }
    // torch.linspace pins both endpoints exactly.
    base[static_cast<size_t>(num_steps - 1)] = 0.0f;
  }

  std::vector<float> shifted(static_cast<size_t>(num_steps));
  const float s = static_cast<float>(shift_scale);
  for (int64_t i = 0; i < num_steps; ++i) {
    const float b = base[static_cast<size_t>(i)];
    shifted[static_cast<size_t>(i)] = s * b / (1.0f + (s - 1.0f) * b);
  }

  // torch.unique_consecutive: collapse RUNS of equal values (not a global unique).
  std::vector<float> collapsed;
  collapsed.reserve(shifted.size());
  for (float value : shifted) {
    if (collapsed.empty() || collapsed.back() != value) collapsed.push_back(value);
  }
  // A one-point request stays exactly one point; otherwise the schedule must end
  // at sigma 0 so the terminal Euler step is the identity.
  if (num_steps > 1 && collapsed.back() > 0.0f) collapsed.push_back(0.0f);

  std::vector<double> out(collapsed.size());
  for (size_t i = 0; i < collapsed.size(); ++i) out[i] = static_cast<double>(collapsed[i]);
  return out;
}

// _resolve_task (pipeline_minimax_h3.py:374-391).
std::string MiniMaxH3ResolveTask(const std::string& requested, const std::string& partition,
                                 bool has_image,
                                 const std::vector<std::string>& supported_tasks) {
  std::string task = requested;
  if (task.empty()) {
    if (partition == "ref2va") {
      task = "ref2va";
    } else if (has_image) {
      task = "fl2va";
    } else {
      task = "t2va";
    }
  }
  task = ToLower(task);
  VT_CHECK(std::find(supported_tasks.begin(), supported_tasks.end(), task) != supported_tasks.end(),
           "minimax_h3: this checkpoint partition does not support the requested task");
  return task;
}

// ---------------------------------------------------------------------------
// Checkpoint PARTITION / task guard (the #77 follow-up)
// ---------------------------------------------------------------------------
namespace {

// "{a, b, c}" for an error message.
std::string JoinBraced(const std::vector<std::string>& items) {
  std::string out = "{";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) out += ", ";
    out += items[i];
  }
  out += "}";
  return out;
}

}  // namespace

// pipeline_minimax_h3.py:279-282 —
//   release = model_index.get("_minimax_h3") or {}
//   self.partition       = str(release.get("partition", ""))
//   self.supported_tasks = frozenset(release.get("tasks") or ())
MiniMaxH3PartitionInfo MiniMaxH3PartitionFromModelIndex(const nlohmann::json& model_index) {
  MiniMaxH3PartitionInfo info;
  info.declared = true;  // a real model_index.json was read; absence of the block
                         // below leaves partition unknown (and the guard refuses).
  if (!model_index.is_object()) return info;
  const auto release_it = model_index.find("_minimax_h3");
  if (release_it == model_index.end() || !release_it->is_object()) return info;
  const nlohmann::json& release = *release_it;
  const auto part_it = release.find("partition");
  if (part_it != release.end() && part_it->is_string()) {
    info.partition = ToLower(part_it->get<std::string>());
  }
  const auto tasks_it = release.find("tasks");
  if (tasks_it != release.end() && tasks_it->is_array()) {
    for (const nlohmann::json& t : *tasks_it) {
      if (t.is_string()) info.supported_tasks.push_back(ToLower(t.get<std::string>()));
    }
  }
  return info;
}

// The recipe's one-server-one-partition split (recipes/MiniMaxAI/MiniMax-H3.md:50-51):
// FL2VA serves t2va + fl2va, Ref2VA serves ref2va. Mirrors the served-task set the
// release model_index.json would have carried, for the community files that strip it.
MiniMaxH3PartitionInfo MiniMaxH3PartitionFromFlag(const std::string& partition) {
  MiniMaxH3PartitionInfo info;
  info.declared = true;
  info.partition = ToLower(partition);
  if (info.partition.empty()) {
    return info;  // declared-but-unknown: the guard refuses every task.
  }
  if (info.partition == "fl2va") {
    info.supported_tasks = {"t2va", "fl2va"};
  } else if (info.partition == "ref2va") {
    info.supported_tasks = {"ref2va"};
  } else {
    VT_CHECK(false,
             "minimax_h3: unknown --partition '" + partition +
                 "' — expected fl2va or ref2va (recipes/MiniMaxAI/MiniMax-H3.md:50-51). "
                 "FL2VA serves t2va+fl2va, Ref2VA serves ref2va.");
  }
  return info;
}

// The request fields are mutually exclusive (packed layout enforces it via
// BuildMiniMaxH3PackedSequenceRef2va vs the keyframe path); ref2va prepends whole
// reference BLOCKS, fl2va pins keyframe FRAMES of the output.
std::string MiniMaxH3TaskOfRequest(const MiniMaxH3T2vaRequest& request) {
  if (!request.ref_blocks.empty()) return "ref2va";
  if (!request.keyframe_frame_indices.empty()) return "fl2va";
  return "t2va";
}

// The raise half of `_resolve_task` (pipeline_minimax_h3.py:387-390):
//   if task not in self.supported_tasks:
//     raise ValueError(f"checkpoint partition {self.partition!r} supports "
//                      f"{sorted(self.supported_tasks)}, got task={task!r}")
// Extended for the stripped community files that carry no served-task set: an
// unknown partition makes EVERY task ambiguous (the FL2VA/Ref2VA DiTs are
// indistinguishable), so it refuses and names the recipe lines instead of guessing.
void MiniMaxH3CheckTaskPartition(const std::string& task, const MiniMaxH3PartitionInfo& info) {
  if (!info.declared) return;  // no checkpoint metadata supplied → guard inactive.
  const std::string t = ToLower(task);
  if (info.supported_tasks.empty()) {
    VT_CHECK(false,
             "minimax_h3: cannot run task '" + t +
                 "' — this checkpoint does not declare its partition. Community "
                 "GGUF/NVFP4 files strip model_index.json `_minimax_h3`, and the "
                 "FL2VA and Ref2VA DiTs are byte-structurally identical (same tensor "
                 "names AND shapes), so the partition cannot be inferred from the "
                 "weights. Pass --partition fl2va|ref2va. One server serves one "
                 "partition: FL2VA serves t2va+fl2va, Ref2VA serves ref2va "
                 "(recipes/MiniMaxAI/MiniMax-H3.md:50-51,289; "
                 "pipeline_minimax_h3.py:374-391).");
  }
  const bool served =
      std::find(info.supported_tasks.begin(), info.supported_tasks.end(), t) !=
      info.supported_tasks.end();
  VT_CHECK(served,
           "minimax_h3: checkpoint partition '" + info.partition + "' supports " +
               JoinBraced(info.supported_tasks) + ", got task='" + t +
               "'. t2va/fl2va require the FL2VA partition, ref2va requires Ref2VA; "
               "one server serves one partition "
               "(recipes/MiniMaxAI/MiniMax-H3.md:50-51,289; "
               "pipeline_minimax_h3.py:387-390).");
}

// _resolve_shape (pipeline_minimax_h3.py:393-434).
MiniMaxH3ShapePlan MiniMaxH3ResolveShape(const std::string& task, double duration_seconds,
                                         int64_t requested_frames, int64_t height, int64_t width,
                                         int64_t image_width, int64_t image_height) {
  const double fps = static_cast<double>(kMiniMaxH3Fps);

  int64_t frames = 0;
  if (duration_seconds > 0.0) {
    frames = static_cast<int64_t>(RoundHalfToEven(duration_seconds * fps));
  } else if (requested_frames > 1) {
    frames = requested_frames;
  } else {
    frames = task == "ref2va" ? kMiniMaxH3DefaultFramesRef2va : kMiniMaxH3DefaultFramesT2va;
  }

  MiniMaxH3ShapePlan plan;
  plan.num_frames = MiniMaxH3AlignFrameCount(frames);

  if (height <= 0 || width <= 0) {
    // fl2va inherits the keyframe's aspect on a 768 short edge; everything else
    // falls back to the shipped 768x1344 canvas.
    if (task == "fl2va" && image_width > 0 && image_height > 0) {
      const double ratio = static_cast<double>(image_width) / static_cast<double>(image_height);
      if (ratio >= 1.0) {
        height = 768;
        width = MiniMaxH3AlignMultiple(768.0 * ratio, 32);
      } else {
        width = 768;
        height = MiniMaxH3AlignMultiple(768.0 / ratio, 32);
      }
    } else {
      height = 768;
      width = 1344;
    }
  }
  // Truncate (not round) onto the 32 grid.
  plan.height = height / 32 * 32;
  plan.width = width / 32 * 32;
  VT_CHECK(plan.height > 0 && plan.width > 0, "minimax_h3: invalid canvas");
  VT_CHECK(plan.width <= 4 * plan.height && plan.height <= 4 * plan.width,
           "minimax_h3: canvas aspect ratio must be in [1:4, 4:1]");

  plan.latent_t = MiniMaxH3VideoLatentT(plan.num_frames);
  plan.audio_t = MiniMaxH3AudioLatentT(static_cast<double>(plan.num_frames) / fps);
  return plan;
}

// ---------------------------------------------------------------------------
// Reference-video input geometry + frame schedule (reference_video.py)
//
// Only the PURE-MATH half of that module is ported here. The rest of it — probe,
// transcode, frame extraction, audio decode — shells out to ffmpeg/soundfile and
// is blocked on the SAME external-dependency decision as `/v1/videos` MP4 muxing:
// one decision unlocks both reference-video INPUT decode and generated-video
// OUTPUT encode. Recorded in .agents/specs/minimax-h3.md section 5.2.
// ---------------------------------------------------------------------------

namespace {

// _nearest_multiple (reference_video.py:24-26): round-half-to-even (Python's
// round()), floored at one multiple.
int64_t NearestMultiple(double value, int64_t multiple) {
  const int64_t snapped =
      static_cast<int64_t>(RoundHalfToEven(value / static_cast<double>(multiple))) * multiple;
  return std::max(multiple, snapped);
}

}  // namespace

// _reference_video_shape (reference_video.py:84-103). Returns {width, height}.
std::pair<int64_t, int64_t> MiniMaxH3ReferenceVideoShape(int64_t width, int64_t height) {
  VT_CHECK(width > 0 && height > 0, "minimax_h3 reference video: dims must be positive");
  const double ratio = static_cast<double>(width) / static_cast<double>(height);
  VT_CHECK(ratio >= 0.25 && ratio <= 4.0,
           "minimax_h3 reference video: aspect ratio must be in [1:4, 4:1]");
  double target_w, target_h;
  if (ratio >= 1.0) {
    target_w = static_cast<double>(kMiniMaxH3RefVideoShortEdge) * ratio;
    target_h = static_cast<double>(kMiniMaxH3RefVideoShortEdge);
  } else {
    target_w = static_cast<double>(kMiniMaxH3RefVideoShortEdge);
    target_h = static_cast<double>(kMiniMaxH3RefVideoShortEdge) / ratio;
  }
  const double area = target_w * target_h;
  if (area > static_cast<double>(kMiniMaxH3RefVideoMaxPixels)) {
    const double scale = std::sqrt(static_cast<double>(kMiniMaxH3RefVideoMaxPixels) / area);
    target_w *= scale;
    target_h *= scale;
  }
  return {NearestMultiple(target_w, kMiniMaxH3RefVideoCanvasMultiple),
          NearestMultiple(target_h, kMiniMaxH3RefVideoCanvasMultiple)};
}

// The frame schedule (reference_video.py:208-220, 253-260): resample 24 FPS down
// to the 2 FPS Qwen video sampling rate, dropping duplicate indices, then average
// timestamps over each temporal patch (padding the tail by REPEATING the last).
MiniMaxH3ReferenceVideoSchedule MiniMaxH3ReferenceVideoFrameSchedule(int64_t frame_count) {
  VT_CHECK(frame_count > 0, "minimax_h3 reference video: frame_count must be positive");
  MiniMaxH3ReferenceVideoSchedule out;
  const double ratio =
      static_cast<double>(kMiniMaxH3Fps) / static_cast<double>(kMiniMaxH3QwenVideoSampleFps);
  double cursor = 0.0;
  while (true) {
    const int64_t frame_index = static_cast<int64_t>(RoundHalfToEven(cursor));
    if (frame_index >= frame_count) break;
    if (out.indices.empty() || frame_index > out.indices.back()) out.indices.push_back(frame_index);
    cursor += ratio;
  }
  VT_CHECK(!out.indices.empty(), "minimax_h3 reference video: no frames sampled");

  std::vector<double> timestamps;
  timestamps.reserve(out.indices.size() + kMiniMaxH3QwenTemporalPatch);
  for (size_t i = 0; i < out.indices.size(); ++i) {
    timestamps.push_back(static_cast<double>(i) /
                         static_cast<double>(kMiniMaxH3QwenVideoSampleFps));
  }
  const int64_t pad = (kMiniMaxH3QwenTemporalPatch -
                       static_cast<int64_t>(timestamps.size()) % kMiniMaxH3QwenTemporalPatch) %
                      kMiniMaxH3QwenTemporalPatch;
  for (int64_t i = 0; i < pad; ++i) timestamps.push_back(timestamps.back());
  for (size_t i = 0; i < timestamps.size(); i += static_cast<size_t>(kMiniMaxH3QwenTemporalPatch)) {
    out.block_timestamps.push_back(
        (timestamps[i] + timestamps[i + static_cast<size_t>(kMiniMaxH3QwenTemporalPatch) - 1]) / 2.0);
  }
  return out;
}

}  // namespace vllm
