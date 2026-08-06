// MiniMax-H3 video output: frame serialization and the MP4 mux command.
//
// ─── WHERE THE PROCESS BOUNDARY SITS, AND WHY ────────────────────────────────
// `/v1/videos` returns MP4 (H.264 + audio). Upstream vLLM-Omni obtains that by
// shelling out to `ffmpeg`. Mirroring upstream is this project's policy, but
// vLLM-Omni is a Python SERVING layer while vllm.cpp is a LIBRARY, and `src/vllm/`
// has no subprocess precedent anywhere — a library that forks a process is an
// architectural commitment (PATH dependence, sandboxing, process lifetime) that
// does not belong here.
//
// So the split is: this file produces the ARTIFACTS (RGB frames as PPM, audio as
// WAV via MiniMaxH3WriteWav) and BUILDS the argv, both as pure data transforms
// with no process spawning; the example/server layer performs the actual
// invocation. That mirrors upstream's behaviour while keeping the library
// dependency-free, and it keeps the interesting logic unit-testable.
//
// RATIFIED by the developer 2026-08-03: "re: ffmpeg invocation, correct -
// let's keep in the examples only". So this boundary is a project decision,
// not an implementation convenience: do NOT add process spawning to src/vllm/.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

// One frame of a [C, T, H, W] float tensor as binary PPM (P6, 8-bit RGB).
// ffmpeg reads PPM natively, so this is the handoff format for the mux step.
// `frames` is the pipeline's decoded output; values are clamped from [-1, 1].
std::string MiniMaxH3WritePpmFrame(const std::vector<float>& frames,
                                   const MiniMaxH3VideoFrameShape& shape, int64_t frame_index) {
  VT_CHECK(shape.channels == 3, "minimax_h3 ppm: expected 3 (RGB) channels");
  VT_CHECK(frame_index >= 0 && frame_index < shape.t, "minimax_h3 ppm: frame index out of range");
  const int64_t plane = shape.h * shape.w;
  VT_CHECK(static_cast<int64_t>(frames.size()) == shape.channels * shape.t * plane,
           "minimax_h3 ppm: frame buffer does not match the shape");

  std::string out = "P6\n" + std::to_string(shape.w) + " " + std::to_string(shape.h) + "\n255\n";
  out.reserve(out.size() + static_cast<size_t>(plane * 3));
  for (int64_t y = 0; y < shape.h; ++y) {
    for (int64_t x = 0; x < shape.w; ++x) {
      for (int64_t c = 0; c < 3; ++c) {
        const size_t idx =
            static_cast<size_t>((c * shape.t + frame_index) * plane + y * shape.w + x);
        // The VAE emits roughly [-1, 1]; map to [0, 255] and clamp.
        const double v = std::min(1.0, std::max(-1.0, static_cast<double>(frames[idx])));
        const int32_t byte = static_cast<int32_t>(std::lround((v + 1.0) * 0.5 * 255.0));
        out.push_back(static_cast<char>(static_cast<uint8_t>(std::min(255, std::max(0, byte)))));
      }
    }
  }
  return out;
}

// The argv the SERVER layer runs to mux frames + audio into MP4. Pure string
// assembly — this function never spawns anything.
//
// Mirrors upstream's container choice: H.264 video (yuv420p so every player
// accepts it) plus AAC audio, `+faststart` so the moov atom leads and the file is
// streamable. `frame_pattern` is a printf-style path (e.g. ".../frame_%06d.ppm").
std::vector<std::string> MiniMaxH3BuildMp4MuxArgs(const MiniMaxH3MuxRequest& request) {
  VT_CHECK(!request.frame_pattern.empty(), "minimax_h3 mux: frame_pattern is required");
  VT_CHECK(!request.output_path.empty(), "minimax_h3 mux: output_path is required");
  VT_CHECK(request.fps > 0, "minimax_h3 mux: fps must be positive");

  std::vector<std::string> argv = {
      "ffmpeg", "-y", "-loglevel", "error",
      // video input: a numbered PPM sequence at the generation frame rate
      "-framerate", std::to_string(request.fps),
      "-i", request.frame_pattern,
  };
  if (!request.audio_path.empty()) {
    argv.insert(argv.end(), {"-i", request.audio_path});
  }
  argv.insert(argv.end(), {"-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                           std::to_string(request.crf)});
  if (!request.audio_path.empty()) {
    argv.insert(argv.end(), {"-c:a", "aac", "-b:a", "192k",
                             // stop at the shorter stream so a half-second of
                             // trailing silence or video never leaks in
                             "-shortest"});
  }
  argv.insert(argv.end(), {"-movflags", "+faststart", request.output_path});
  return argv;
}

}  // namespace vllm
