// server: an OpenAI-compatible HTTP server over the vllm.cpp LLMEngine (M3.1
// Task 4). Loads a supported model (safetensors or GGUF weights + tokenizer + a KV-cache config →
// LLMEngine), constructs the OpenAI serving handlers (chat wired with the real
// chat template via MakeChatTemplatePromptFn(LoadChatTemplateFromConfig(...)))
// and serves /v1/completions, /v1/chat/completions, /v1/models, /health,
// /version.
//
//   server --model <dir> [--host 0.0.0.0] [--port 8000]
//          [--tokenizer-config <tokenizer_config.json>]
//          [--served-model-name <name>]
//          [--block-size N] [--num-blocks N] [--max-model-len N]
//          [--max-num-seqs N] [--max-num-batched-tokens N]
//          [--enable-force-include-usage]
//          [--[no-]enable-prefix-caching]
//          [--scheduling-policy fcfs|priority]
//          [--tool-call-parser <name>|auto|none]
//          [--reasoning-parser <name>|auto|none]
//          [--kv-transfer-config '<json>']
//
// A directory holds config.json, tokenizer.json and supported safetensors
// shards. A supported GGUF file is also accepted and supplies model metadata
// plus embedded vocabulary. If --tokenizer-config is omitted for a directory it
// defaults to <dir>/tokenizer_config.json; when that file has no chat_template
// the chat endpoint falls back to the simple role-join prompt.
//
// NOTE: loading the real 35B checkpoint is a GPU/dgx concern; on the CPU CI box
// this binary is only built + smoke-tested against a synthetic engine (see
// tests/vllm/entrypoints/openai/test_api_server.cpp). The wiring below is the
// same either way.
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>
#ifdef VT_BENCH_PROFILE_CONTROL
#include <atomic>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#endif

#include "vllm/config/kv_transfer.h"
#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include <fstream>
#include "vllm/entrypoints/openai/api_server.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"
#include "vllm/entrypoints/openai/tool_parsers/detect.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/multimodal/parakeet_transcription.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/version.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/kv_offload/kv_connector.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#ifdef VT_BENCH_PROFILE_CONTROL
#include "vt/cuda/cuda_profiler_control.h"
#endif
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

namespace fs = std::filesystem;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;

// Decode a binary PPM (P6) into [3, H, W] floats in [-1, 1] — the layout the H3
// video-VAE encoder takes. It reads from BYTES rather than a path so one decoder
// serves both spellings of `input_reference` (a filesystem path and an inline
// data: URL). PPM is the only still-image container this tree can read: no PNG /
// JPEG codec is vendored, which is the same NAMED residual the chat multimodal
// path carries (see chat_mm.h), not a limitation of this endpoint.
std::vector<float> DecodePpmChw(const std::string& bytes, int64_t* out_h,
                                int64_t* out_w) {
  std::istringstream in(bytes, std::ios::binary);
  std::string magic;
  in >> magic;
  if (magic != "P6") {
    throw std::runtime_error(
        "input_reference: not a binary PPM (P6); no PNG/JPEG codec is vendored, "
        "so a reference image must be supplied as binary PPM");
  }
  auto next_int = [&]() {
    int v = 0;
    while (in >> std::ws, in.peek() == '#') { std::string skip; std::getline(in, skip); }
    in >> v;
    return v;
  };
  const int w = next_int(), h = next_int(), maxv = next_int();
  if (w <= 0 || h <= 0 || maxv <= 0) {
    throw std::runtime_error("input_reference: bad PPM header");
  }
  in.get();  // the single whitespace byte before the payload
  std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
  in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  if (!in) throw std::runtime_error("input_reference: truncated PPM payload");
  std::vector<float> chw(rgb.size());
  const int64_t plane = static_cast<int64_t>(w) * h;
  for (int64_t i = 0; i < plane; ++i) {
    for (int64_t c = 0; c < 3; ++c) {
      chw[static_cast<size_t>(c * plane + i)] =
          static_cast<float>(rgb[static_cast<size_t>(i * 3 + c)]) / (maxv * 0.5f) - 1.0f;
    }
  }
  if (out_h != nullptr) *out_h = h;
  if (out_w != nullptr) *out_w = w;
  return chw;
}

// The bytes behind one reference: the parser hands us either a path or the
// already-decoded payload of a data: URL.
std::string ReadReferenceBytes(const std::string& field, const std::string& path,
                               const std::vector<uint8_t>& inline_bytes) {
  if (!inline_bytes.empty()) return std::string(inline_bytes.begin(), inline_bytes.end());
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(field + ": cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// A ref2va VIDEO reference: DIR/frame_%06d.ppm, the exact layout minimax_h3_gen
// and this server's own muxer WRITE, so one run's frames chain straight into the
// next request. Returns [C, T, H, W] in [-1, 1] (the encoder's clip layout);
// no container demuxer is vendored, which is why this is a frame directory.
std::vector<float> ReadReferenceClipChw(const std::string& dir, int64_t* out_t,
                                        int64_t* out_h, int64_t* out_w) {
  std::vector<float> per_frame;  // frame-major [T][C,H,W]
  int64_t frames = 0, fh = 0, fw = 0;
  for (int64_t k = 0;; ++k) {
    char name[512];
    std::snprintf(name, sizeof(name), "%s/frame_%06lld.ppm", dir.c_str(),
                  static_cast<long long>(k));
    std::ifstream probe(name, std::ios::binary);
    if (!probe) break;
    const std::string bytes((std::istreambuf_iterator<char>(probe)),
                            std::istreambuf_iterator<char>());
    int64_t h = 0, w = 0;
    const std::vector<float> frame = DecodePpmChw(bytes, &h, &w);
    if (frames == 0) { fh = h; fw = w; }
    if (h != fh || w != fw) {
      throw std::runtime_error(
          "metadata.input_reference_video: every frame_%06d.ppm must have the same size");
    }
    per_frame.insert(per_frame.end(), frame.begin(), frame.end());
    ++frames;
  }
  if (frames == 0) {
    throw std::runtime_error("metadata.input_reference_video: no frame_%06d.ppm files in " + dir);
  }
  // [T][C,H,W] -> [C,T,H,W], the causal 3-D encoder's layout.
  std::vector<float> chw(per_frame.size());
  const int64_t plane = fh * fw;
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t k = 0; k < frames; ++k) {
      for (int64_t e = 0; e < plane; ++e) {
        chw[static_cast<size_t>((c * frames + k) * plane + e)] =
            per_frame[static_cast<size_t>(k * 3 * plane + c * plane + e)];
      }
    }
  }
  *out_t = frames;
  *out_h = fh;
  *out_w = fw;
  return chw;
}

struct Args {
  std::string model_dir;
  std::string host = "0.0.0.0";
  int port = 8000;
  std::string tokenizer_config;  // default: <model_dir>/tokenizer_config.json
  std::string served_model_name;  // default: the model dir name
  int block_size = 32;
  int num_blocks = 256;
  int max_model_len = 0;  // 0 => config.max_position_embeddings
  int max_num_seqs = 8;
  int max_num_batched_tokens = 0;  // 0 => per-architecture default.
  // --- MiniMax-H3 video generation (opt-in; absent => /v1/videos is unregistered
  // and the server behaves exactly as before). ---
  std::string video_dit, video_vae, video_vae_config, audio_vae, audio_vae_config;
  std::string video_prompt_embeds, video_workdir = "/tmp/vllm_h3_videos";
  std::string video_encoder, video_tokenizer;
  int video_encoder_max_layers = 50;
  std::string video_ffmpeg = "ffmpeg", video_device = "cuda";
  std::string video_partition;  // served partition (fl2va|ref2va); see the #77 guard
  bool video_keep_quant = false;
  int cuda_profile_graph_replays = 0;  // trace-only diagnostic build seam.
  int cuda_profile_graph_batch = 0;  // 0 => accepted c16 trace contract.
  std::string benchmark_shutdown_fifo;  // paired trace-only control path.
  std::optional<bool> enable_prefix_caching = std::nullopt;
  bool enable_force_include_usage = false;
  // GET /tokenizer_info gate. Mirrors vLLM's --enable-tokenizer-info-endpoint
  // (entrypoints/openai/cli_args.py:140, default False; the route is registered
  // only when serve/tokenize/api_router.py:95 sees the flag). Default off → the
  // route 404s, byte-identical to before.
  bool enable_tokenizer_info_endpoint = false;
  // Dev/admin endpoint gate. Mirrors vLLM's VLLM_SERVER_DEV_MODE env
  // (envs.py:157, default 0): build_app registers the dev/rlhf + dev/cache
  // routers only under `if envs.VLLM_SERVER_DEV_MODE` (api_server.py:238). Off by
  // default → /abort_requests 404s. Enables the /abort_requests production wiring.
  bool enable_server_dev_mode = false;
  // Scheduling policy: "fcfs" (default), "priority" (mirrors vLLM's
  // --scheduling-policy / SchedulerConfig.policy), or "lpm" (SGLang's
  // cache-aware longest-prefix-match admission ordering, ENG-SGLANG-BEHAVIOR-FLAG;
  // opt-in, output-neutral, resolves to fcfs when prefix caching is off).
  // --schedule-policy is accepted as an SGLang-compatible alias.
  std::string scheduling_policy = "fcfs";
  // Jump-forward decoding (ENG-SGLANG-BEHAVIOR-FLAG SW3): tri-state, mirrors the
  // C-ABI vllm_model_params.enable_jump_forward. Unset (default) => OFF unless
  // VT_ENABLE_JUMP_FORWARD is set; --enable-jump-forward forces on,
  // --disable-jump-forward forces off (the env var still overrides). The
  // token-unique forced-run subset only; see .agents/specs/sglang-enablement.md.
  std::optional<bool> enable_jump_forward = std::nullopt;
  // Tool-call / reasoning dialect selection (mirrors vLLM's --tool-call-parser
  // and --reasoning-parser). THE DEFAULTS ARE TODAY'S HARDCODED BEHAVIOUR:
  // "hermes" is exactly what OpenAIServingChat was constructed with before this
  // flag existed, and "none" is the empty reasoning-parser name it passed. An
  // invocation that names neither flag is therefore unchanged, byte for byte.
  // "auto" opts into the chat-template detection the C ABI uses.
  std::string tool_call_parser = "hermes";
  std::string reasoning_parser = "none";
  // vLLM's --kv-transfer-config: the external KV connector selection, as the
  // same JSON object vLLM takes. Empty (default) == no connector == the inert
  // production path. See docs/KV-OFFLOAD.md.
  std::string kv_transfer_config;
  // vLLM's --speculative-config: the speculative-decoding selection, as the same
  // JSON object vLLM takes (e.g. '{"method":"mtp","num_speculative_tokens":1}').
  // Empty (default) == no speculation == the inert production path (SPEC-MTP I5d).
  std::string speculative_config;
};

[[noreturn]] void Usage(const char* argv0, int code) {
  std::cerr
      << "usage: " << argv0
      << " --model <dir> [--host H] [--port P] [--tokenizer-config F]\n"
         "               [--served-model-name N] [--block-size N] "
         "[--num-blocks N] [--max-model-len N]\n"
         "               [--max-num-seqs N] "
         "[--max-num-batched-tokens N]\n"
         "               [--cuda-profile-graph-replays N]\n"
         "               [--cuda-profile-graph-batch N]\n"
         "               [--benchmark-shutdown-fifo F]\n"
         "               [--enable-force-include-usage]\n"
         "               [--enable-tokenizer-info-endpoint]\n"
         "               [--enable-server-dev-mode]\n"
         "               [--[no-]enable-prefix-caching]\n"
         "               [--[no-]enable-radix-attention]\n"
         "               [--scheduling-policy fcfs|priority|lpm]\n"
         "               [--[enable|disable]-jump-forward]\n"
         "               [--tool-call-parser <name>|auto|none]\n"
         "               [--reasoning-parser <name>|auto|none]\n"
         "               [--kv-transfer-config '<json>']\n"
         "               [--speculative-config '<json>']\n";
  std::exit(code);
}

std::string NextArg(int argc, char** argv, int& i, const char* argv0) {
  if (i + 1 >= argc) Usage(argv0, 2);
  return argv[++i];
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") {
      a.model_dir = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--host") {
      a.host = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--port") {
      a.port = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--tokenizer-config") {
      a.tokenizer_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--served-model-name") {
      a.served_model_name = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--block-size") {
      a.block_size = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--num-blocks") {
      a.num_blocks = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-model-len") {
      a.max_model_len = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-num-seqs") {
      a.max_num_seqs = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-num-batched-tokens") {
      a.max_num_batched_tokens = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--cuda-profile-graph-replays") {
      a.cuda_profile_graph_replays =
          std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--cuda-profile-graph-batch") {
      a.cuda_profile_graph_batch =
          std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--benchmark-shutdown-fifo") {
      a.benchmark_shutdown_fifo = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--enable-force-include-usage") {
      a.enable_force_include_usage = true;
    } else if (flag == "--enable-tokenizer-info-endpoint") {
      a.enable_tokenizer_info_endpoint = true;
    } else if (flag == "--video-dit") {
      a.video_dit = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-vae") {
      a.video_vae = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-vae-config") {
      a.video_vae_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--audio-vae") {
      a.audio_vae = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--audio-vae-config") {
      a.audio_vae_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-encoder") {
      a.video_encoder = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-tokenizer") {
      a.video_tokenizer = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-encoder-max-layers") {
      a.video_encoder_max_layers = std::atoi(NextArg(argc, argv, i, argv[0]).c_str());
    } else if (flag == "--video-prompt-embeds") {
      a.video_prompt_embeds = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-workdir") {
      a.video_workdir = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-ffmpeg") {
      a.video_ffmpeg = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-device") {
      a.video_device = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-partition") {
      a.video_partition = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--video-keep-quant") {
      a.video_keep_quant = true;
    } else if (flag == "--enable-server-dev-mode") {
      a.enable_server_dev_mode = true;
    } else if (flag == "--enable-prefix-caching" ||
               flag == "--no-enable-prefix-caching" ||
               flag == "--enable-radix-attention" ||
               flag == "--disable-radix-attention") {
      // --[no-]enable-prefix-caching is vLLM's flag. --enable-radix-attention /
      // --disable-radix-attention are SGLang-compatible ALIASES for the SAME
      // toggle (RadixAttention is fused into our block-hash APC — there is no
      // distinct radix code path; see .agents/specs/sglang-radixattention.md §1).
      // They set the identical tri-state as the vLLM flag; last-wins is rejected
      // (mirrors passing the vLLM flag twice) so a contradictory pair is caught.
      if (a.enable_prefix_caching.has_value()) {
        std::cerr << "server: prefix-caching flag (--[no-]enable-prefix-caching "
                     "/ --[disable|enable]-radix-attention) specified more than "
                     "once\n";
        Usage(argv[0], 2);
      }
      a.enable_prefix_caching =
          flag == "--enable-prefix-caching" || flag == "--enable-radix-attention";
    } else if (flag == "--scheduling-policy" || flag == "--schedule-policy") {
      // --scheduling-policy is vLLM's flag; --schedule-policy is SGLang's name,
      // accepted as an alias. Both take fcfs|priority|lpm.
      a.scheduling_policy = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--enable-jump-forward" ||
               flag == "--disable-jump-forward") {
      // ENG-SGLANG-BEHAVIOR-FLAG SW3: opt into (or force off) jump-forward
      // decoding — the token-unique grammar-speed subset (see
      // .agents/specs/sglang-enablement.md). Absent => the default (OFF unless
      // VT_ENABLE_JUMP_FORWARD is set). The env var, when set, still overrides.
      if (a.enable_jump_forward.has_value()) {
        std::cerr << "server: jump-forward flag "
                     "(--[enable|disable]-jump-forward) specified more than "
                     "once\n";
        Usage(argv[0], 2);
      }
      a.enable_jump_forward = flag == "--enable-jump-forward";
    } else if (flag == "--tool-call-parser") {
      a.tool_call_parser = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--reasoning-parser") {
      a.reasoning_parser = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--kv-transfer-config") {
      a.kv_transfer_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--speculative-config") {
      a.speculative_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "-h" || flag == "--help") {
      Usage(argv[0], 0);
    } else {
      std::cerr << "server: unknown argument '" << flag << "'\n";
      Usage(argv[0], 2);
    }
  }
  if (a.model_dir.empty()) {
    std::cerr << "server: --model <dir> is required\n";
    Usage(argv[0], 2);
  }
  if (a.max_num_seqs <= 0 || a.max_num_batched_tokens < 0 ||
      a.cuda_profile_graph_replays < 0 || a.cuda_profile_graph_batch < 0) {
    std::cerr << "server: scheduler capacities must be positive "
                 "(--max-num-batched-tokens may be 0 for auto)\n";
    Usage(argv[0], 2);
  }
  if ((a.cuda_profile_graph_replays > 0) !=
      !a.benchmark_shutdown_fifo.empty()) {
    std::cerr << "server: --cuda-profile-graph-replays and "
                 "--benchmark-shutdown-fifo must be specified together\n";
    Usage(argv[0], 2);
  }
  if (a.cuda_profile_graph_replays == 0 && a.cuda_profile_graph_batch != 0) {
    std::cerr << "server: --cuda-profile-graph-batch requires "
                 "--cuda-profile-graph-replays\n";
    Usage(argv[0], 2);
  }
  if (a.cuda_profile_graph_replays > 0 && a.cuda_profile_graph_batch == 0) {
    a.cuda_profile_graph_batch = 16;
  }
  if (a.cuda_profile_graph_batch > a.max_num_seqs) {
    std::cerr << "server: --cuda-profile-graph-batch exceeds --max-num-seqs\n";
    Usage(argv[0], 2);
  }
  // Validate a NAMED parser dialect here, before the (multi-GB) model load, so a
  // typo costs a second rather than a full load. "auto" cannot be checked yet —
  // it resolves against the chat template — but detection only ever returns
  // registered names, so it cannot fail later either.
  namespace oai = vllm::entrypoints::openai;
  if (a.tool_call_parser != "auto") {
    (void)oai::ResolveToolParserName(a.tool_call_parser, "");
  }
  if (a.reasoning_parser != "auto") {
    (void)oai::ResolveReasoningParserName(a.reasoning_parser, "");
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);

    const fs::path dir(args.model_dir);
    const std::string config_path = (dir / "config.json").string();
    const std::string tokenizer_path = (dir / "tokenizer.json").string();
    const std::string tokenizer_config_path =
        args.tokenizer_config.empty()
            ? (dir / "tokenizer_config.json").string()
            : args.tokenizer_config;
    const std::string served_model_name =
        args.served_model_name.empty()
            ? (dir.has_filename() ? dir.filename().string()
                                  : dir.parent_path().filename().string())
            : args.served_model_name;

    // ── TASK DISPATCH (ARCH-ONE-SURFACE ROW 1): a model dir whose
    // architectures resolve to a SupportsTranscription-ONLY registration
    // (Parakeet CTC/RNNT/TDT) serves /v1/audio/transcriptions through the ONE
    // library seam — the same ParakeetTranscriber vllm_transcribe drives — and
    // registers NO generate routes (vLLM's task-conditional registration,
    // api_server.py:255-265). Every other model takes the text path below,
    // byte-identical to before. ────────────────────────────────────────────────
    {
      bool transcription_only = false;
      const std::vector<std::string> archs =
          vllm::PeekHfArchitectures(config_path);
      if (!archs.empty()) {
        try {
          transcription_only =
              vllm::ModelRegistry::Resolve(std::span<const std::string>(archs))
                  .info.supports_transcription_only;
        } catch (const std::exception&) {
          transcription_only = false;  // unknown arch: the text path diagnoses
        }
      }
      if (transcription_only) {
        std::cerr << "server: transcription-only model (" << archs[0]
                  << "); serving /v1/audio/transcriptions\n";
        auto transcriber =
            std::make_shared<vllm::multimodal::ParakeetTranscriber>(
                vllm::multimodal::ParakeetTranscriber::FromDir(args.model_dir));
        namespace oai = vllm::entrypoints::openai;
        oai::OpenAIServingModels asr_models(served_model_name);
        oai::ApiServer asr_server(asr_models, vllm::Version());
        asr_server.set_transcriber(
            [transcriber](const uint8_t* wav, size_t n) {
              return transcriber->TranscribeWavBytes(wav, n);
            });
        std::cerr << "server: listening on http://" << args.host << ":"
                  << args.port << "\n";
        if (!asr_server.listen(args.host, args.port)) {
          std::cerr << "server: failed to bind " << args.host << ":"
                    << args.port << "\n";
          return 1;
        }
        return 0;
      }
    }

    // ── Load the model + build the full engine stack via the shared loader
    // (src/vllm/entrypoints/model_loader.cpp) — the same path the C ABI drives.
    // It loads config.json + tokenizer.json + *.safetensors and wires the M1.8
    // LLMEngine over Scheduler + runner + KV + processors. ─────────────────────
    std::cerr << "server: loading model from " << args.model_dir << " (config "
              << config_path << ", tokenizer " << tokenizer_path << ")\n";
    vllm::entrypoints::EngineParams engine_params;
    engine_params.block_size = args.block_size;
    engine_params.num_blocks = args.num_blocks;
    engine_params.max_model_len = args.max_model_len;  // 0 => from config.
    engine_params.max_num_seqs = args.max_num_seqs;
    engine_params.max_num_batched_tokens = args.max_num_batched_tokens;
    engine_params.enable_prefix_caching = args.enable_prefix_caching;
    // Reject an unknown policy string (mirrors upstream SchedulingPolicy(value)).
    engine_params.policy = vllm::SchedulerPolicyFromString(args.scheduling_policy);
    // ENG-SGLANG-BEHAVIOR-FLAG (SW1): `lpm` needs prefix caching to have any
    // cache to match against; with APC explicitly off it degrades to fcfs
    // (the scheduler leaves arrival order intact). Warn once at load so the
    // no-op is visible (mirrors the spec's lpm+cache-off resolution).
    if (engine_params.policy == vllm::SchedulerPolicy::kLPM &&
        args.enable_prefix_caching.has_value() &&
        !args.enable_prefix_caching.value()) {
      std::cerr << "server: --scheduling-policy lpm has no effect with prefix "
                   "caching disabled; falling back to fcfs admission order\n";
    }
    // ENG-SGLANG-BEHAVIOR-FLAG SW3: jump-forward decoding. Unset => the default
    // (env-resolved, OFF); --[enable|disable]-jump-forward forces it, and
    // VT_ENABLE_JUMP_FORWARD still overrides at resolution time.
    engine_params.enable_jump_forward = args.enable_jump_forward;
    // --kv-transfer-config: the external KV connector, mirroring vLLM's own
    // flag and JSON shape. Absent (default) leaves the optional unset, which is
    // the inert no-connector path the server has always run. A malformed
    // document, an unknown key/role, or a connector whose worker half cannot
    // move bytes on this device (the D1 guard, inside LoadedEngine) all throw
    // out of here and are reported at startup by the catch in main.
    if (!args.kv_transfer_config.empty()) {
      vllm::KVTransferConfig kv_cfg =
          vllm::ParseKVTransferConfigJson(args.kv_transfer_config);
      if (kv_cfg.kv_connector.has_value() &&
          !vllm::v1::kv_offload::KVConnectorFactory::IsRegistered(
              *kv_cfg.kv_connector)) {
        std::string msg = "unknown kv_connector \"" + *kv_cfg.kv_connector +
                          "\" (registered connectors: ";
        const std::vector<std::string> names =
            vllm::v1::kv_offload::KVConnectorFactory::RegisteredNames();
        for (size_t n = 0; n < names.size(); ++n) {
          if (n != 0) msg += ", ";
          msg += names[n];
        }
        msg += ")";
        throw std::invalid_argument(msg);
      }
      engine_params.kv_transfer_config = std::move(kv_cfg);
    }
    // --speculative-config: speculative decoding (SPEC-MTP I5d). Absent (default)
    // leaves the optional unset — the byte-identical no-speculation path. The
    // parse validates method/k here; n_predict + the resolved k are finalized in
    // LoadedEngine once the checkpoint's mtp_num_hidden_layers is known. A
    // malformed document or unsupported method throws and is reported at startup.
    if (!args.speculative_config.empty()) {
      engine_params.speculative_config =
          vllm::ParseSpeculativeConfigJson(args.speculative_config);
    }
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                      engine_params);
    std::cerr << "server: prefix caching "
              << (loaded->prefix_caching_enabled() ? "enabled" : "disabled")
              << "\n";
    // W2: the production server uses AsyncLLM over EngineCoreProc's dedicated
    // engine thread. HTTP workers submit independently and stream from their
    // per-request collectors; no server-wide engine mutex remains.
    vllm::v1::AsyncLLM& engine = loaded->async_engine();
    const vllm::tok::Tokenizer& tokenizer = loaded->tokenizer();

    if (args.cuda_profile_graph_replays > 0) {
#ifdef VT_BENCH_PROFILE_CONTROL
      vt::cuda::ConfigureCudaGraphReplayProfiler(
          static_cast<uint32_t>(args.cuda_profile_graph_replays),
          static_cast<uint32_t>(args.cuda_profile_graph_batch));
      std::cerr << "[VT_CUDA_PROFILE] ready pid=" << getpid()
                << " signal=SIGUSR2 target_replays="
                << args.cuda_profile_graph_replays << "\n";
#else
      throw std::invalid_argument(
          "--cuda-profile-graph-replays requires "
          "VLLM_CPP_BENCH_PROFILE_CONTROL=ON");
#endif
    }

    // ── OpenAI serving handlers. The chat handler is wired with the real chat
    // template (Task 3) when tokenizer_config.json carries one; otherwise it
    // keeps the default role-join fallback. ────────────────────────────────
    namespace oai = vllm::entrypoints::openai;
    oai::OpenAIServingModels models(served_model_name);
    oai::OpenAIServingCompletion completion(
        engine, served_model_name, args.enable_force_include_usage);

    oai::ChatPromptFn chat_prompt_fn = oai::DefaultChatPromptFallback;
    // Kept outside the try so the parser resolution below can sniff it when
    // --tool-call-parser/--reasoning-parser are "auto"; empty when the model
    // ships no template (auto then falls back to hermes / disabled).
    std::string chat_template;
    try {
      chat_template =
          vllm::entrypoints::LoadChatTemplateFromConfig(tokenizer_config_path);
      const std::string bos =
          tokenizer.BosId() >= 0 ? tokenizer.Decode({tokenizer.BosId()}) : "";
      const std::string eos =
          tokenizer.EosId() >= 0 ? tokenizer.Decode({tokenizer.EosId()}) : "";
      chat_prompt_fn =
          vllm::entrypoints::MakeChatTemplatePromptFn(chat_template, bos, eos);
      std::cerr << "server: using chat template from " << tokenizer_config_path
                << "\n";
    } catch (const std::exception& e) {
      std::cerr << "server: no chat template (" << e.what()
                << "); falling back to the simple role-join prompt\n";
    }
    // Dialect selection. Defaults reproduce the previously hardcoded pair
    // ("hermes", "") exactly; an unknown name throws std::invalid_argument
    // listing every registered parser and aborts startup, rather than leaving
    // tool/reasoning parsing silently off for the life of the process.
    const std::string tool_parser_name =
        oai::ResolveToolParserName(args.tool_call_parser, chat_template);
    const std::string reasoning_parser_name =
        oai::ResolveReasoningParserName(args.reasoning_parser, chat_template);
    std::cerr << "server: tool-call parser "
              << (tool_parser_name.empty() ? "disabled" : tool_parser_name)
              << ", reasoning parser "
              << (reasoning_parser_name.empty() ? "disabled"
                                                : reasoning_parser_name)
              << "\n";
    oai::OpenAIServingChat chat(engine, served_model_name, chat_prompt_fn,
                                tool_parser_name, reasoning_parser_name,
                                args.enable_force_include_usage);

    // SAMPLE-BEAM (C7): enable use_beam_search on the production AsyncLLM path.
    // Both handlers need the tokenizer (prompt tok + per-beam detok) and the eos
    // id (beam retirement); a use_beam_search request then routes through
    // BeamSearchAsync (online.py) over the async engine. Without this, beam
    // requests reject with "requires an engine and a tokenizer".
    const std::optional<int32_t> beam_eos =
        tokenizer.EosId() >= 0
            ? std::optional<int32_t>(tokenizer.EosId())
            : std::nullopt;
    completion.set_beam_search_tokenizer(&tokenizer, beam_eos);
    chat.set_beam_search_tokenizer(&tokenizer, beam_eos);

    // ── MM-SERVE-E2E: wire the multimodal chat seam for image-capable models.
    // When the model dir carries a preprocessor_config.json the Qwen3-VL image
    // processor loads, we construct the seam body (MakeQwen3VLImageChatFn) so an
    // OpenAI image_url request renders the placeholder marker → tokenizes to the
    // single image_pad id → EXPANDS to N image tokens + mm_features carried onto
    // the engine request. A text-only model (no preprocessor_config.json) leaves
    // the seam UNSET → the chat path is byte-identical. The container-format
    // image codec (PNG/JPEG → RGB) is a NAMED residual: no codec is vendored, so
    // the production codec rejects encoded images with a clear message (the M2c
    // single-sequence gate consumes pre-decoded raw RGB). The mm FORWARD (vision
    // tower + merge + MRoPE/DeepStack on the GPU worker consuming
    // Request.mm_features) is the remaining MM-SERVE-E2E residual — the engine
    // model runner has no mm-forward path yet. Kept alive for the server loop.
    std::unique_ptr<vllm::multimodal::Qwen3VLImageProcessor> mm_image_proc;
    const std::string preprocessor_config_path =
        (dir / "preprocessor_config.json").string();
    if (fs::exists(preprocessor_config_path)) {
      try {
        vllm::multimodal::Qwen3VLProcessorConfig pcfg =
            vllm::multimodal::LoadQwen3VLProcessorConfig(
                preprocessor_config_path, config_path, served_model_name);
        mm_image_proc =
            std::make_unique<vllm::multimodal::Qwen3VLImageProcessor>(pcfg);
        oai::ImageCodecFn codec =
            [](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
          // Raw-RGB passthrough (image/x-raw-rgb): the single-sequence e2e /
          // gate fixture format. A square raw-RGB payload is decoded directly;
          // any container format (PNG/JPEG) is the NAMED codec residual.
          if (media.media_type == "image/x-raw-rgb") {
            const std::size_t n = media.bytes.size();
            const std::size_t px = n / 3;
            const auto side =
                static_cast<int64_t>(std::llround(std::sqrt(
                    static_cast<double>(px))));
            if (side <= 0 || static_cast<std::size_t>(side * side * 3) != n) {
              throw std::runtime_error(
                  "image/x-raw-rgb payload is not a square HxWx3 buffer");
            }
            oai::DecodedImageRgb out;
            out.rgb = media.bytes;
            out.height = side;
            out.width = side;
            return out;
          }
          throw std::runtime_error(
              "multimodal image: container-format decode (PNG/JPEG -> RGB) is a "
              "named MM-SERVE residual; supply raw RGB (image/x-raw-rgb)");
        };
        chat.set_multimodal_chat_fn(oai::MakeQwen3VLImageChatFn(
            *mm_image_proc, tokenizer, chat_prompt_fn, std::move(codec)));
        std::cerr << "server: multimodal image seam wired (Qwen3-VL processor "
                     "from "
                  << preprocessor_config_path << ")\n";
      } catch (const std::exception& e) {
        std::cerr << "server: no multimodal image seam (" << e.what()
                  << "); image requests fall back to the text path\n";
      }
    }

    // Diagnostic opt-out exists only for same-binary attribution. Production
    // defaults to the capacity-derived fixed pool.
    const char* fixed_pool_env = std::getenv("VLLM_CPP_HTTP_FIXED_POOL");
    const auto worker_pool_mode =
        fixed_pool_env != nullptr && std::string(fixed_pool_env) == "0"
            ? oai::ApiServer::HttpWorkerPoolMode::kLegacyDynamic
            : oai::ApiServer::HttpWorkerPoolMode::kCapacityFixed;
    oai::ApiServer server(completion, chat, models, vllm::Version(),
                          static_cast<size_t>(args.max_num_seqs),
                          worker_pool_mode);

    // ── C8 opt-in utility/admin endpoints (SERVE-UTILITY-ENDPOINTS /
    // SERVE-ADMIN). Wire the setters from the LIVE engine + tokenizer through the
    // single shared seam so the production server actually serves /tokenize,
    // /detokenize, /tokenizer_info (flag) and /abort_requests (dev-mode flag),
    // mirroring vLLM 0.26's per-endpoint default gating. /metrics and
    // /reset_prefix_cache stay unwired (no live backing on the AsyncLLM path) —
    // see ConfigureUtilityEndpoints + specs/{utility,admin}-endpoints.md. ────────
    // ── MiniMax-H3 video generation. OPT-IN: with no --video-dit the routes are
    // never registered and the server is byte-identical to before. The runner is a
    // CALLBACK because src/vllm/ must not spawn processes — the ffmpeg invocation
    // is allowed here, in examples/, by the developer's ratified decision. ───────
    struct VideoState {
      vllm::MiniMaxH3GgufDit dit;
      vllm::MiniMaxH3VideoVaeDecoderConfig video_cfg;
      vllm::MiniMaxH3AudioVaeConfig audio_cfg;
      vllm::MiniMaxH3LatentStats video_stats, audio_stats;
      vllm::MiniMaxH3AudioVaeWeights video_weights, audio_weights;
      std::vector<float> prompt_embeds;   // fallback when no encoder is configured
      // Encoder staged ONCE: staging the 32B tower costs ~162 s, so per-request
      // staging would dominate every generation.
      bool has_encoder = false;
      vllm::MiniMaxH3EncoderConfig enc_config;
      vllm::MiniMaxH3EncoderQuantWeights enc_host;
      vllm::MiniMaxH3EncoderDeviceWeights enc_staged;
      std::unique_ptr<vllm::tok::Tokenizer> tokenizer;
      vt::Queue enc_queue{};
      std::string workdir, ffmpeg;
      // The served partition (fl2va|ref2va). Resolved ONCE at load from
      // --video-partition; the per-request guard in MiniMaxH3GenerateT2va refuses a
      // task this partition does not serve (the #77 follow-up).
      vllm::MiniMaxH3PartitionInfo partition_info;
      vt::Device device;
      std::atomic<int64_t> counter{0};
      // The two VAEs' ENCODER halves, for the reference modalities: the video VAE
      // encodes an `input_reference` image and a `metadata.input_reference_video`
      // clip, the audio VAE encodes a `metadata.input_reference_audio` waveform.
      // Loaded LAZILY and ONCE each: a text-to-video server must not pay for
      // weights it never uses, and a server that does use them must not reload
      // per request.
      std::string video_vae_path, audio_vae_path;
      std::mutex encoder_mutex;
      bool video_encoder_loaded = false, audio_encoder_loaded = false;
      vllm::MiniMaxH3AudioVaeWeights video_encoder_weights, audio_encoder_weights;
      vllm::MiniMaxH3EncoderFcn3dConfig video_encoder_cfg;
      vllm::MiniMaxH3AudioVaeEncoderConfig audio_encoder_cfg;
    };
    std::shared_ptr<VideoState> video;
    if (!args.video_dit.empty()) {
      video = std::make_shared<VideoState>();
      std::cerr << "server: loading MiniMax-H3 video checkpoints...\n";
      if (args.video_dit.size() > 5 &&
          args.video_dit.compare(args.video_dit.size() - 5, 5, ".gguf") == 0) {
        const vllm::GgufFile f = vllm::GgufFile::Open(args.video_dit);
        video->dit = vllm::LoadMiniMaxH3DitFromGguf(f, args.video_keep_quant);
      } else {
        const vllm::SafetensorsFile f = vllm::SafetensorsFile::Open(args.video_dit);
        video->dit = vllm::LoadMiniMaxH3DitFromNvfp4(f);
      }
      if (!args.video_vae_config.empty()) {
        std::ifstream in(args.video_vae_config);
        nlohmann::json j;
        in >> j;
        video->video_cfg = vllm::ParseMiniMaxH3VideoVaeDecoderConfig(j, &video->video_stats);
      }
      if (!args.audio_vae_config.empty()) {
        std::ifstream in(args.audio_vae_config);
        nlohmann::json j;
        in >> j;
        video->audio_cfg = vllm::ParseMiniMaxH3AudioVaeConfig(j, &video->audio_stats);
      }
      {
        const vllm::SafetensorsFile f = vllm::SafetensorsFile::Open(args.video_vae);
        video->video_weights = vllm::LoadMiniMaxH3VideoVaeDecoderWeights(f);
      }
      {
        const vllm::SafetensorsFile f = vllm::SafetensorsFile::Open(args.audio_vae);
        video->audio_weights = vllm::LoadMiniMaxH3AudioVaeWeights(f);
      }
      if (!args.video_prompt_embeds.empty()) {
        std::ifstream in(args.video_prompt_embeds, std::ios::binary | std::ios::ate);
        const std::streamsize n = in.tellg();
        in.seekg(0);
        video->prompt_embeds.resize(static_cast<size_t>(n) / sizeof(float));
        in.read(reinterpret_cast<char*>(video->prompt_embeds.data()), n);
      }
      if (!args.video_encoder.empty()) {
        std::cerr << "server: loading MiniMax-H3 encoder (keep-quant)...\n";
        const vllm::GgufFile ef = vllm::GgufFile::Open(args.video_encoder);
        video->enc_host = vllm::LoadMiniMaxH3EncoderFromGguf(ef, args.video_encoder_max_layers);
        video->enc_config = video->enc_host.config;
        // The ComfyUI-style encoder export is WEIGHTS ONLY, so the vocab comes from
        // the checkpoint tokenizer.json unless the GGUF happens to embed one.
        video->tokenizer = std::make_unique<vllm::tok::Tokenizer>(
            args.video_tokenizer.empty()
                ? vllm::tok::Tokenizer::FromGguf(ef)
                : vllm::tok::Tokenizer::FromHfJson(args.video_tokenizer));
        video->enc_queue = vt::Queue{video->device, nullptr};
        if (video->device.type != vt::DeviceType::kCPU) {
          video->enc_queue = vt::GetBackend(video->device.type).CreateQueue();
        }
        std::cerr << "server: staging encoder to device (once)...\n";
        video->enc_staged = vllm::StageMiniMaxH3EncoderWeights(video->enc_queue, video->enc_host);
        video->has_encoder = true;
        std::cerr << "server: encoder ready (layers=" << video->enc_config.num_hidden_layers
                  << ", hidden=" << video->enc_config.hidden_size << ")\n";
      }
      video->workdir = args.video_workdir;
      video->ffmpeg = args.video_ffmpeg;
      video->partition_info = vllm::MiniMaxH3PartitionFromFlag(args.video_partition);
      video->video_vae_path = args.video_vae;
      video->audio_vae_path = args.audio_vae;
      if (args.video_device == "cuda") {
        video->device = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
      }
      std::cerr << "server: /v1/videos on (dit layers=" << video->dit.params.num_layers
                << ", device=" << args.video_device
                << (args.video_keep_quant ? ", keep-quant" : "") << ")\n";
      // HONEST LIMIT, stated at startup rather than buried: turning a PROMPT into
      // conditioning needs the H3-Encoder (a 32B tower + tokenizer), which is not
      // wired here. Until it is, every request is conditioned on the SAME supplied
      // embeddings, so the prompt text does NOT steer the output.
      if (video->has_encoder) {
        std::cerr << "server: /v1/videos conditions on the request PROMPT\n";
      } else if (video->prompt_embeds.empty()) {
        std::cerr << "server: WARNING /v1/videos has neither --video-encoder nor "
                     "--video-prompt-embeds; requests will be REJECTED\n";
      } else {
        std::cerr << "server: WARNING /v1/videos ignores the request PROMPT — pass "
                     "--video-encoder to condition on it\n";
      }

      server.set_video_runner([video](const vllm::openai::VideoRequest& req) -> std::string {
        // REAL text conditioning when an encoder is configured: tokenize the
        // request prompt, gather its rows from the block-quant table, and run the
        // already-staged tower.
        std::vector<float> conditioning;
        if (video->has_encoder) {
          const std::vector<int32_t> ids = video->tokenizer->Encode(req.prompt);
          if (ids.empty()) throw std::runtime_error("the prompt tokenized to nothing");
          const std::vector<float> embeds =
              vllm::MiniMaxH3EncoderEmbedTokens(video->enc_host, ids);
          const int64_t eseq = static_cast<int64_t>(ids.size());
          std::vector<int64_t> epos(static_cast<size_t>(3 * eseq));
          for (int64_t a = 0; a < 3; ++a) {
            for (int64_t s = 0; s < eseq; ++s) epos[static_cast<size_t>(a * eseq + s)] = s;
          }
          vllm::MiniMaxH3EncoderConfig ec = video->enc_config;
          conditioning = vllm::MiniMaxH3EncoderTextForwardDevice(
              video->enc_queue, ec, video->enc_staged, embeds, epos.data(), eseq);
        } else if (!video->prompt_embeds.empty()) {
          conditioning = video->prompt_embeds;
        } else {
          throw std::runtime_error(
              "video generation needs conditioning: start the server with "
              "--video-encoder (to condition on the prompt) or --video-prompt-embeds");
        }
        // OpenAI `input_reference` -> fl2va FIRST-FRAME conditioning.
        //
        // WHY fl2va and not ref2va: OpenAI documents input_reference as the image
        // the generated video STARTS FROM (image-to-video), which is exactly what
        // fl2va expresses — MiniMaxH3EncodeKeyframeCondRows pins frame 0 OF THE
        // OUTPUT to the supplied image. ref2va
        // (MiniMaxH3EncodeReferenceImages) means something else: whole reference
        // images PREPENDED to the sequence as their own blocks, i.e. subject or
        // style guidance that never becomes a frame of the result. Mapping
        // input_reference there would silently change what the API promises. The
        // ref2va modalities OpenAI has no slot for enter through `metadata`
        // (input_reference_video / input_reference_audio) instead; the parser has
        // already refused the combinations the pipeline forbids.
        std::vector<float> reference_chw;
        int64_t reference_h = 0, reference_w = 0;
        if (req.has_input_reference()) {
          reference_chw = DecodePpmChw(
              ReadReferenceBytes("input_reference", req.input_reference_path,
                                 req.input_reference_bytes),
              &reference_h, &reference_w);
        }
        std::vector<float> reference_clip;
        int64_t clip_t = 0, clip_h = 0, clip_w = 0;
        if (req.has_input_reference_video()) {
          reference_clip = ReadReferenceClipChw(req.input_reference_video_dir, &clip_t,
                                                &clip_h, &clip_w);
        }

        const vllm::MiniMaxH3DitParams& p = video->dit.params;
        vllm::MiniMaxH3T2vaRequest r;
        r.partition = video->partition_info;  // #77 guard: MiniMaxH3GenerateT2va
                                              // refuses a task this partition can't serve.
        // With a reference image and no explicit task, the task IS fl2va; a
        // metadata reference means ref2va. The image aspect also drives the
        // default resolution (_resolve_shape).
        const bool has_ref2va =
            req.has_input_reference_video() || req.has_input_reference_audio();
        const std::string task =
            !req.task.empty()
                ? req.task
                : (req.has_input_reference() ? "fl2va" : (has_ref2va ? "ref2va" : "t2va"));
        const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
            task, req.duration_seconds, req.num_frames, req.height, req.width,
            reference_w, reference_h);
        r.latent_t = plan.latent_t;
        r.num_frames = plan.num_frames;
        r.latent_h = plan.height / vllm::kMiniMaxH3VaeRatio;
        r.latent_w = plan.width / vllm::kMiniMaxH3VaeRatio;
        r.audio_t = plan.audio_t;
        r.audio_channel = vllm::kMiniMaxH3AudioChannels;
        r.num_steps = req.num_inference_steps;
        r.video_shift = req.flow_shift;
        r.audio_shift = req.audio_flow_shift;
        r.video_latents_mean = video->video_stats.mean;
        r.video_latents_std = video->video_stats.std_dev;
        r.audio_latents_mean = video->audio_stats.mean;
        r.audio_latents_std = video->audio_stats.std_dev;
        r.text_len = static_cast<int64_t>(conditioning.size()) / p.text_dim;

        // Both VAE encoder halves load at most once, under one lock, whichever
        // reference modality asks for them first.
        auto ensure_video_encoder = [&]() {
          if (video->video_vae_path.empty()) {
            throw std::runtime_error(
                "a video/image reference needs the video VAE ENCODER half: start "
                "the server with --video-vae");
          }
          if (video->video_encoder_loaded) return;
          const vllm::SafetensorsFile vf = vllm::SafetensorsFile::Open(video->video_vae_path);
          video->video_encoder_weights = vllm::LoadMiniMaxH3VideoVaeEncoderWeights(vf);
          video->video_encoder_cfg = vllm::MiniMaxH3EncoderFcn3dConfig{};
          video->video_encoder_cfg.z_channels = 2 * p.latents_dim;  // mean | logvar
          video->video_encoder_loaded = true;
        };
        auto ensure_audio_encoder = [&]() {
          if (video->audio_vae_path.empty()) {
            throw std::runtime_error(
                "metadata.input_reference_audio needs the audio VAE ENCODER half: "
                "start the server with --audio-vae");
          }
          if (video->audio_encoder_loaded) return;
          const vllm::SafetensorsFile af = vllm::SafetensorsFile::Open(video->audio_vae_path);
          video->audio_encoder_weights = vllm::LoadMiniMaxH3AudioVaeEncoderWeights(af);
          video->audio_encoder_cfg = vllm::MiniMaxH3AudioVaeEncoderConfig{};
          video->audio_encoder_cfg.vae_latent_channels = p.audio_latents_dim;
          video->audio_encoder_loaded = true;
        };

        if (req.has_input_reference()) {
          if (reference_h != plan.height || reference_w != plan.width) {
            // No image resampler is vendored, and a mis-sized keyframe would
            // either abort deep in the denoise or pin the wrong latent rows. Say
            // so up front, with the geometry we resolved.
            throw std::runtime_error(
                "input_reference is " + std::to_string(reference_w) + "x" +
                std::to_string(reference_h) + " but this request resolved to " +
                std::to_string(plan.width) + "x" + std::to_string(plan.height) +
                "; supply the reference at the output size (or pass a matching "
                "`size`): no image resampler is vendored");
          }
          std::lock_guard<std::mutex> guard(video->encoder_mutex);
          ensure_video_encoder();
          r.keyframe_frame_indices = {0};  // FIRST frame; upstream also allows {-1}/{0,-1}
          r.imgvid_noise_aug = 1.0;        // pin the frame exactly
          r.keyframe_cond_rows = vllm::MiniMaxH3EncodeKeyframeCondRows(
              video->video_encoder_cfg, video->video_encoder_weights, p, {reference_chw},
              reference_h, reference_w, r.latent_t, r.imgvid_noise_aug, {});
        } else if (has_ref2va) {
          // ── ref2va REFERENCE BLOCKS. Exclusive with the fl2va branch above
          // (minimax_h3_pipeline.cpp:251), which the parser already enforced. ──
          std::lock_guard<std::mutex> guard(video->encoder_mutex);
          std::vector<vllm::MiniMaxH3RefBlock> blocks;
          if (req.has_input_reference_video()) {
            ensure_video_encoder();
            vllm::MiniMaxH3RefBlock block{};
            r.keyframe_cond_rows = vllm::MiniMaxH3EncodeReferenceVideo(
                video->video_encoder_cfg, video->video_encoder_weights, p, reference_clip,
                clip_t, clip_h, clip_w, &block);
            // SILENT by construction: MiniMaxH3EncodeReferenceVideo emits
            // ref_audio_t == 0 because the clip's own soundtrack would need the
            // audio VAE encoder run over it. An audio reference below ATTACHES to
            // this block, which is the layout packed_sequence.py builds.
            blocks.push_back(block);
          }
          if (req.has_input_reference_audio()) {
            ensure_audio_encoder();
            const std::string wav_bytes = ReadReferenceBytes(
                "metadata.input_reference_audio", req.input_reference_audio_path,
                req.input_reference_audio_bytes);
            int64_t samples_per_channel = 0;
            const std::vector<float> waveform = vllm::MiniMaxH3ReadWav(
                wav_bytes, vllm::kMiniMaxH3AudioChannels, vllm::kMiniMaxH3AudioSampleRate,
                &samples_per_channel);
            vllm::MiniMaxH3RefBlock audio_block{};
            r.audio_ref_rows = vllm::MiniMaxH3EncodeReferenceAudio(
                video->audio_encoder_cfg, video->audio_encoder_weights, waveform,
                vllm::kMiniMaxH3AudioChannels, samples_per_channel, video->audio_stats.mean,
                video->audio_stats.std_dev, /*noise_aug=*/1.0, {}, &audio_block);
            if (!blocks.empty() &&
                blocks[0].kind == vllm::MiniMaxH3RefBlock::Kind::kVideoAudio) {
              // The reference video now HAS sound: one kVideoAudio block carries
              // both, so its ref_audio_t must claim exactly the rows just encoded.
              blocks[0].ref_audio_t = audio_block.ref_audio_t;
            } else {
              blocks.push_back(audio_block);
            }
          }
          r.ref_blocks = blocks;
        }

        const int64_t frame_rows =
            (r.latent_h / p.patch_size_h) * (r.latent_w / p.patch_size_w);
        std::vector<float> nv(static_cast<size_t>(r.latent_t * frame_rows *
                                                  p.video_row_width()));
        std::vector<float> na(static_cast<size_t>(r.audio_t * r.audio_channel *
                                                  p.audio_latents_dim));
        uint64_t x = req.has_seed ? static_cast<uint64_t>(req.seed) : 0x5EED1234ULL;
        auto fill = [&x](std::vector<float>& o) {
          for (float& value : o) {
            x += 0x9E3779B97F4A7C15ULL;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            value = static_cast<float>((z >> 11) * 0x1.0p-53 * 2.0 - 1.0);
          }
        };
        fill(nv);
        fill(na);

        const vllm::MiniMaxH3T2vaResult out = vllm::MiniMaxH3GenerateT2va(
            video->device, r, p, video->dit.weights, video->video_cfg, video->video_weights,
            video->audio_cfg, video->audio_weights, conditioning, nv, na,
            vt::DType::kBF16);

        const int64_t id = video->counter.fetch_add(1);
        const std::string dir = video->workdir + "/job" + std::to_string(id);
        if (std::system(("mkdir -p '" + dir + "'").c_str()) != 0) {
          throw std::runtime_error("cannot create " + dir);
        }
        for (int64_t f = 0; f < out.frame_shape.t; ++f) {
          char name[512];
          std::snprintf(name, sizeof(name), "%s/frame_%06lld.ppm", dir.c_str(),
                        static_cast<long long>(f));
          std::ofstream fo(name, std::ios::binary);
          const std::string bytes =
              vllm::MiniMaxH3WritePpmFrame(out.frames, out.frame_shape, f);
          fo.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        const std::string wav = dir + "/audio.wav";
        {
          std::ofstream fo(wav, std::ios::binary);
          const std::string bytes = vllm::MiniMaxH3WriteWav(
              out.waveform, out.audio_channels, out.audio_samples_per_channel,
              out.sample_rate);
          fo.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        vllm::MiniMaxH3MuxRequest mux;
        mux.frame_pattern = dir + "/frame_%06d.ppm";
        mux.audio_path = wav;
        mux.output_path = dir + "/video.mp4";
        std::vector<std::string> argv_mux = vllm::MiniMaxH3BuildMp4MuxArgs(mux);
        if (!argv_mux.empty()) argv_mux[0] = video->ffmpeg;
        std::string cmd;
        for (const std::string& a : argv_mux) cmd += "'" + a + "' ";
        if (std::system(cmd.c_str()) != 0) throw std::runtime_error("ffmpeg failed");
        return mux.output_path;
      });
    }

    oai::UtilityEndpointOptions endpoint_opts;
    endpoint_opts.enable_tokenizer_info_endpoint =
        args.enable_tokenizer_info_endpoint;
    endpoint_opts.enable_server_dev_mode = args.enable_server_dev_mode;
    oai::ConfigureUtilityEndpoints(server, tokenizer, loaded->max_model_len(),
                                   engine, endpoint_opts);
    std::cerr << "server: utility endpoints: /tokenize /detokenize on"
              << (args.enable_tokenizer_info_endpoint ? ", /tokenizer_info on"
                                                      : "")
              << (args.enable_server_dev_mode ? ", /abort_requests on (dev-mode)"
                                              : "")
              << "\n";

    std::cerr << "server: listening on http://" << args.host << ":" << args.port
              << " (model '" << served_model_name << "', HTTP worker pool ";
    if (server.http_worker_count() == 0) {
      std::cerr << "legacy-dynamic";
    } else {
      std::cerr << server.http_worker_count() << " fixed";
    }
    std::cerr << ")\n";

#ifdef VT_BENCH_PROFILE_CONTROL
    std::atomic<bool> benchmark_shutdown_waiter_ready{false};
    std::atomic<bool> benchmark_shutdown_received{false};
    std::atomic<bool> benchmark_shutdown_failed{false};
    std::atomic<bool> benchmark_shutdown_cancelled{false};
    std::thread benchmark_shutdown_thread;
    if (args.cuda_profile_graph_replays > 0) {
      benchmark_shutdown_thread = std::thread([&]() {
        const int shutdown_fd =
            open(args.benchmark_shutdown_fifo.c_str(),
                 O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
        if (shutdown_fd < 0) {
          const int status = errno;
          std::cerr << "[VT_BENCH_SHUTDOWN] failed operation=open status="
                    << status << "\n";
          benchmark_shutdown_failed.store(true, std::memory_order_release);
          return;
        }
        struct stat shutdown_stat {};
        const int stat_status = fstat(shutdown_fd, &shutdown_stat);
        if (stat_status != 0 || !S_ISFIFO(shutdown_stat.st_mode)) {
          const int status = stat_status != 0 ? errno : EINVAL;
          std::cerr << "[VT_BENCH_SHUTDOWN] failed operation=fstat status="
                    << status << "\n";
          close(shutdown_fd);
          benchmark_shutdown_failed.store(true, std::memory_order_release);
          return;
        }
        benchmark_shutdown_waiter_ready.store(true, std::memory_order_release);
        std::cerr << "[VT_BENCH_SHUTDOWN] ready pid=" << getpid()
                  << " control=fifo\n";
        while (!benchmark_shutdown_cancelled.load(std::memory_order_acquire)) {
          char command = '\0';
          const ssize_t bytes = read(shutdown_fd, &command, 1);
          if (bytes == 1) {
            if (command == 'Q') {
              benchmark_shutdown_received.store(true,
                                                 std::memory_order_release);
              std::cerr
                  << "[VT_BENCH_SHUTDOWN] requested control=fifo\n";
              close(shutdown_fd);
              server.stop();
              return;
            }
            std::cerr
                << "[VT_BENCH_SHUTDOWN] failed operation=command status="
                << static_cast<unsigned int>(
                       static_cast<unsigned char>(command))
                << "\n";
            close(shutdown_fd);
            benchmark_shutdown_failed.store(true, std::memory_order_release);
            server.stop();
            return;
          }
          if (bytes < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
              errno != EINTR) {
            const int status = errno;
            std::cerr << "[VT_BENCH_SHUTDOWN] failed operation=read status="
                      << status << "\n";
            close(shutdown_fd);
            benchmark_shutdown_failed.store(true, std::memory_order_release);
            server.stop();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        close(shutdown_fd);
      });
      while (!benchmark_shutdown_waiter_ready.load(std::memory_order_acquire) &&
             !benchmark_shutdown_failed.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (benchmark_shutdown_failed.load(std::memory_order_acquire)) {
        benchmark_shutdown_cancelled.store(true, std::memory_order_release);
        benchmark_shutdown_thread.join();
        return 1;
      }
    }
#endif

    const bool listen_ok = server.listen(args.host, args.port);

#ifdef VT_BENCH_PROFILE_CONTROL
    if (benchmark_shutdown_thread.joinable()) {
      benchmark_shutdown_cancelled.store(true, std::memory_order_release);
      benchmark_shutdown_thread.join();
      if (benchmark_shutdown_received.load(std::memory_order_acquire)) {
        std::cerr << "[VT_BENCH_SHUTDOWN] completed control=fifo\n";
      } else {
        if (!benchmark_shutdown_failed.load(std::memory_order_acquire)) {
          std::cerr
              << "[VT_BENCH_SHUTDOWN] failed operation=cancelled status=0\n";
        }
        if (listen_ok) {
          return 1;
        }
      }
    }
#endif

    if (!listen_ok) {
      std::cerr << "server: failed to bind " << args.host << ":" << args.port
                << "\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "server: fatal: " << e.what() << "\n";
    return 1;
  }
}
