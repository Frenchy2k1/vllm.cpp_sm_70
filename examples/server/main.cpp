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
#include <memory>
#include <optional>
#include <stdexcept>
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
      vt::Device device;
      std::atomic<int64_t> counter{0};
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
        const vllm::MiniMaxH3DitParams& p = video->dit.params;
        vllm::MiniMaxH3T2vaRequest r;
        const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
            req.task.empty() ? "t2va" : req.task, req.duration_seconds, req.num_frames,
            req.height, req.width, 0, 0);
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
