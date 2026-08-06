// minimax-h3-mux: muxes a MiniMax-H3 clip (PPM frames + WAV) into an MP4 by
// INVOKING ffmpeg.
//
// THIS FILE IS THE RATIFIED HOME OF THE PROCESS SPAWN. The developer's decision
// (2026-08-03): "re: ffmpeg invocation, correct - let's keep in the examples
// only". So the split is deliberate and load-bearing:
//
//   src/vllm/  builds the ARTIFACTS (MiniMaxH3WritePpmFrame, MiniMaxH3WriteWav)
//              and the ARGV (MiniMaxH3BuildMp4MuxArgs) -- and spawns NOTHING.
//   examples/  (here) performs the invocation.
//
// That is also why `/v1/videos` takes a caller-supplied `VideoRunner` callback
// rather than muxing itself: RunFfmpeg below is precisely the piece a server
// embedder plugs into ApiServer::set_video_runner, and it lives outside the
// library on purpose. Keep it that way -- do not move fork/exec into src/vllm/.
//
// Usage:
//   minimax-h3-mux --frames <pattern> --out <out.mp4> [--audio <in.wav>]
//                  [--fps N] [--crf N] [--ffmpeg <path>] [--print-only]
//
//   --frames  printf-style pattern the library's PPM writer filled in,
//             e.g. /tmp/h3/frame_%06d.ppm
//   --audio   omitted => a silent clip
//   --print-only  print the argv and exit WITHOUT spawning (lets the argv be
//                 inspected, diffed or run by hand on a box with no ffmpeg).
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_h3.h"

namespace {

// Run argv to completion and return its exit status. The ONLY process spawn in
// the MiniMax-H3 path, and it is in examples/ by project decision.
int RunFfmpeg(const std::vector<std::string>& args) {
  std::vector<char*> c_args;
  c_args.reserve(args.size() + 1);
  for (const std::string& arg : args) {
    c_args.push_back(const_cast<char*>(arg.c_str()));
  }
  c_args.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    execvp(c_args[0], c_args.data());
    // Only reached if exec failed; _exit (not exit) so the child never runs the
    // parent's atexit handlers or flushes its buffers a second time.
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) throw std::runtime_error("waitpid failed");
  if (WIFSIGNALED(status)) {
    throw std::runtime_error("ffmpeg died on signal " +
                             std::to_string(WTERMSIG(status)));
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string Need(int argc, char** argv, int i, const std::string& flag) {
  if (i >= argc) throw std::runtime_error("missing value for " + flag);
  return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
  vllm::MiniMaxH3MuxRequest request;
  std::string ffmpeg = "ffmpeg";
  bool print_only = false;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--frames") {
        request.frame_pattern = Need(argc, argv, ++i, flag);
      } else if (flag == "--audio") {
        request.audio_path = Need(argc, argv, ++i, flag);
      } else if (flag == "--out") {
        request.output_path = Need(argc, argv, ++i, flag);
      } else if (flag == "--fps") {
        request.fps = std::stoll(Need(argc, argv, ++i, flag));
      } else if (flag == "--crf") {
        request.crf = std::stoll(Need(argc, argv, ++i, flag));
      } else if (flag == "--ffmpeg") {
        ffmpeg = Need(argc, argv, ++i, flag);
      } else if (flag == "--print-only") {
        print_only = true;
      } else {
        throw std::runtime_error("unknown argument: " + flag);
      }
    }
    if (request.frame_pattern.empty() || request.output_path.empty()) {
      std::cerr << "usage: minimax-h3-mux --frames <pattern> --out <out.mp4> "
                   "[--audio <in.wav>] [--fps N] [--crf N] [--ffmpeg <path>] "
                   "[--print-only]\n";
      return 2;
    }

    // The LIBRARY decides the encoding contract (h264/yuv420p + AAC, -shortest,
    // +faststart); this file only runs it.
    std::vector<std::string> args = vllm::MiniMaxH3BuildMp4MuxArgs(request);
    if (!args.empty()) args[0] = ffmpeg;

    for (size_t i = 0; i < args.size(); ++i) {
      std::cout << (i == 0 ? "" : " ") << args[i];
    }
    std::cout << "\n";
    if (print_only) return 0;

    const int status = RunFfmpeg(args);
    if (status == 127) {
      std::cerr << "failed to exec '" << ffmpeg
                << "' — is ffmpeg installed and on PATH?\n";
      return 127;
    }
    if (status != 0) {
      std::cerr << "ffmpeg exited " << status << "\n";
      return status;
    }
    std::cout << "wrote " << request.output_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
