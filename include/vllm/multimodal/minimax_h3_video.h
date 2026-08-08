// MiniMax-H3 video+audio generation seam — the ONE library entry point that
// composes checkpoint loading (DiT GGUF/NVFP4/sharded-bf16, both VAEs, the
// optional H3-Encoder text tower) -> task resolution + the #77 partition guard
// -> conditioning (prompt encode / prompt-embeds file / fl2va keyframes /
// ref2va references) -> the deterministic noise draw -> MiniMaxH3GenerateT2va
// -> artifacts (frame_%06d.ppm + audio.wav) + the ffmpeg argv, and that every
// consumer (C ABI `vllm_video_*`, the OpenAI server's /v1/videos, the
// `minimax-h3-gen` example) drives.
//
// ARCH-ONE-SURFACE ROW 2 (video+audio generation). This file ABSORBS the
// assembly pipeline `examples/minimax_h3_gen/main.cpp` owned privately
// (pre-refactor main.cpp:687-1288 @ fc636c76: loader-arm dispatch, encoder
// conditioning, reference encoding, the splitmix64 noise streams, artifact
// writing, mux-argv assembly) and the twin copy `examples/server/main.cpp`
// carried for /v1/videos (pre-refactor main.cpp:743-1096). Per the ONE SURFACE
// directive the examples keep argv parsing, printing and the PROCESS SPAWN
// only; the capability lives here, reachable by any embedder.
//
// THE PROCESS BOUNDARY (developer-ratified 2026-08-03) stands: this seam
// WRITES artifacts and BUILDS the mux argv (MiniMaxH3BuildMp4MuxArgs) and
// spawns NOTHING — the caller execs `MiniMaxH3VideoResult::mux_argv`.
//
// Upstream mirror shape: vLLM-Omni serves H3 through
// vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py (the pipeline
// object owns checkpoints + does per-request _resolve_task/_resolve_shape);
// this seam is the C++ engine-side equivalent of that pipeline object.
//
// Byte-identity contract: on the committed fold fixture the CPU t2va render is
// byte-identical to the PRE-fold `minimax-h3-gen` binary at fc636c76
// (tests/vllm/models/test_minimax_h3_video_fold.cpp, three-arm gate).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vllm::openai {
struct VideoRequest;  // entrypoints/openai/video_api.h
}

namespace vllm::multimodal {

// ── Load-time parameters (the checkpoint set; the C ABI mirror is
// vllm_video_model_params). Empty string == "not supplied". ─────────────────
struct MiniMaxH3VideoModelParams {
  std::string dit_path;            // GGUF | NVFP4 safetensors | bf16 shard DIR
  std::string encoder_path;        // H3-Encoder GGUF or bf16 shard DIR
  std::string tokenizer_path;      // tokenizer.json (encoder conditioning)
  std::string video_vae_path, video_vae_config_path;
  std::string audio_vae_path, audio_vae_config_path;
  // Fallback conditioning when no encoder is configured: rows of text_dim,
  // little-endian f32 (the pre-fold --prompt-embeds / --video-prompt-embeds).
  std::string prompt_embeds_path;
  // The served checkpoint PARTITION ("fl2va" | "ref2va"). Community GGUF/NVFP4
  // files strip the release model_index.json `_minimax_h3` block and the two
  // DiTs are byte-structurally identical, so it must be DECLARED; empty is
  // declared-but-unknown and the #77 guard refuses every full render
  // (MiniMaxH3PartitionFromFlag / MiniMaxH3CheckTaskPartition).
  std::string partition;
  int32_t device = 0;        // 0 cpu, 1 cuda
  int32_t dequant_bf16 = 0;  // 0 keep-quant, 1 dequant/stream bf16
  // NVFP4 + cuda only: keep the packed FP4 resident and route the quantized
  // projections through the Marlin W4A16 GEMM (the pre-fold --fp4-resident).
  int32_t fp4_resident = 0;
  int64_t encoder_max_layers = 0;  // 0 => all layers
};

// ── Per-generation parameters (the C ABI mirror is vllm_video_params). ──────
struct MiniMaxH3VideoGenParams {
  std::string prompt;  // encoded when the engine has an encoder
  // "" => resolved from the references/partition (upstream _resolve_task):
  // a keyframe => fl2va, any ref2va reference => ref2va, else t2va.
  std::string task;
  double duration_seconds = 0.0;         // <= 0 => per-task default
  int64_t num_frames = 0;                // <= 1 => per-task default
  int64_t height = 0, width = 0;         // <= 0 => aspect-derived default
  int64_t steps = 0;                     // <= 0 => H3 default (50)
  double flow_shift = 0.0;               // <= 0 => H3 default (12.0)
  double audio_flow_shift = 0.0;         // <= 0 => H3 default (3.0)
  uint64_t seed = 0;
  bool has_seed = false;  // false => the pre-fold fixed default streams

  // fl2va KEYFRAMES: binary PPM (P6), as a path or in-memory bytes (exactly
  // one spelling per frame). Pins frame 0 / the last frame OF THE OUTPUT.
  std::string first_frame_path, last_frame_path;
  std::string first_frame_ppm;  // in-memory alternative (server data: URLs)
  double noise_aug = 1.0;       // condition-noise augmentation (1.0 pins)

  // ref2va REFERENCES (exclusive with keyframes, minimax_h3_pipeline.cpp:251):
  std::vector<std::string> ref_image_paths;  // whole reference images (PPM)
  std::string ref_video_dir;                 // DIR of frame_%06d.ppm
  std::string ref_audio_path;                // 16-bit PCM WAV path...
  std::string ref_audio_wav;                 // ...or its bytes

  // Where frame_%06d.ppm + audio.wav land (created if absent). REQUIRED.
  std::string output_dir;
};

// ── One finished generation (the C ABI mirror is vllm_video_result). ────────
struct MiniMaxH3VideoResult {
  std::string frame_dir;   // holds frame_%06d.ppm
  std::string audio_path;  // 16-bit PCM WAV
  int64_t frame_count = 0, width = 0, height = 0;
  int64_t fps = 0, sample_rate = 0;
  // The ffmpeg argv the CALLER may exec to mux <output_dir>/video.mp4
  // (argv[0] is "ffmpeg"; substitute a custom binary before exec'ing).
  std::vector<std::string> mux_argv;
  std::string mux_output_path;  // the -o target mux_argv names
};

// A loaded H3 video checkpoint set, weights staged once, ready to generate.
// Construction throws std::runtime_error naming the problem on any mismatch.
class MiniMaxH3VideoEngine {
 public:
  static std::unique_ptr<MiniMaxH3VideoEngine> Load(const MiniMaxH3VideoModelParams& params);

  MiniMaxH3VideoEngine(MiniMaxH3VideoEngine&&) noexcept;
  MiniMaxH3VideoEngine& operator=(MiniMaxH3VideoEngine&&) noexcept;
  ~MiniMaxH3VideoEngine();

  // True when an encoder tower is loaded (the request PROMPT conditions the
  // render); false => prompt_embeds_path conditioning (or Generate refuses).
  bool has_encoder() const;
  bool has_prompt_embeds() const;

  // Run one blocking generation. Serialized internally (the staged weights are
  // shared state); throws std::runtime_error to fail the request.
  MiniMaxH3VideoResult Generate(const MiniMaxH3VideoGenParams& params);

 private:
  MiniMaxH3VideoEngine();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The ONE mapping from a parsed /v1/videos request onto the seam's params —
// library-owned so the HTTP route and the FFI cannot drift (the pre-fold
// server carried this as a private lambda). `output_dir` is the job directory
// the artifacts land in.
MiniMaxH3VideoGenParams MiniMaxH3VideoGenParamsFromRequest(
    const ::vllm::openai::VideoRequest& request, const std::string& output_dir);

}  // namespace vllm::multimodal
