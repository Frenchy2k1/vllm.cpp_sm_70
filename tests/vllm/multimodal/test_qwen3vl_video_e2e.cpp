// M3c — the STRICT end-to-end VIDEO->text token-exact gate on Qwen3-VL-4B.
//
// Runs the FULL pipeline on the committed fixture (video, prompt):
//   fixture video -> C++ Qwen3VLImageProcessor::ProcessVideo (M3c) ->
//                    pixel_values_videos + video_grid_thw (t>1)
//                 -> Qwen3VLVisionForward (M2a tower, per-frame windowed attn)
//                    -> merger | deepstack
//                 -> Qwen3VLGenerateGreedyVideo (video merge mask + per-frame
//                    video MRoPE + DeepStack inject + paged greedy) -> greedy tokens.
// Asserts token-for-token equality with the committed vLLM 0.25.0 golden
// (gen_tokens_i32.bin, gate form STRICT per gen_manifest.json). The expanded model
// input ids are the committed gen_input_ids_i32.bin (vLLM's chat-templated
// prompt_token_ids) — the tokenizer/timestamp text path is out of the e2e numeric
// scope (gated separately in the video-processor unit test's BuildVideoRepl case).
//
// dgx-only: needs CUDA + the cached Qwen/Qwen3-VL-4B-Instruct checkpoint
// (VLLM_QWEN3VL_CKPT, or the default HF cache path). Skipped (not failed) without.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {
namespace fs = std::filesystem;

std::string VidFix() { return std::string(MM_VIDEO_FIXTURE_DIR); }

std::vector<int32_t> ReadI32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<int32_t> v(static_cast<size_t>(n) / sizeof(int32_t));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}
std::vector<uint8_t> ReadU8(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> v(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}
nlohmann::json ReadJson(const std::string& path) {
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  nlohmann::json j;
  f >> j;
  return j;
}

std::string FindCkpt() {
  if (const char* e = std::getenv("VLLM_QWEN3VL_CKPT")) return e;
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& d : fs::directory_iterator(snaps, ec))
    if (fs::exists(d.path() / "config.json", ec)) return d.path().string();
  return "";
}

}  // namespace

TEST_CASE("qwen3vl_e2e_video_token_exact_STRICT_vs_vllm_0_25_0") {
  const std::string ckpt = FindCkpt();
  if (ckpt.empty()) {
    MESSAGE("SKIP: Qwen3-VL-4B checkpoint absent (set VLLM_QWEN3VL_CKPT)");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const std::string in_ids_path = VidFix() + "/gen_input_ids_i32.bin";
  if (!fs::exists(in_ids_path)) {
    MESSAGE("SKIP: gen_input_ids_i32.bin absent (run scripts/mm/m3c_video_oracle_capture.py)");
    return;
  }

  const nlohmann::json manifest = ReadJson(VidFix() + "/manifest.json");
  const auto& meta = manifest.at("video").at("metadata");
  const auto vshape = manifest.at("video").at("shape");
  const int64_t T = vshape[0].get<int64_t>();
  const int64_t H = vshape[1].get<int64_t>();
  const int64_t W = vshape[2].get<int64_t>();
  const std::string raw_file = manifest.at("video").at("raw_file").get<std::string>();
  std::vector<int64_t> frame_indices;
  for (const auto& v : meta.at("frames_indices")) frame_indices.push_back(v.get<int64_t>());
  const double fps = meta.at("fps").get<double>();
  const int32_t video_token_id = manifest.at("config").at("video_token_id").get<int32_t>();
  const int32_t vs_id = manifest.at("config").at("vision_start_token_id").get<int32_t>();
  const int32_t ve_id = manifest.at("config").at("vision_end_token_id").get<int32_t>();

  // --- config + weights -----------------------------------------------------
  const vllm::HfConfig cfg = vllm::LoadHfConfig(ckpt + "/config.json");
  REQUIRE(cfg.hidden_size == 2560);
  REQUIRE(cfg.num_hidden_layers == 36);
  REQUIRE(cfg.head_dim == 128);
  std::vector<vllm::SafetensorsFile> shards;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(ckpt, ec))
    if (e.path().extension() == ".safetensors")
      shards.push_back(vllm::SafetensorsFile::Open(e.path().string()));
  REQUIRE(!shards.empty());
  const vllm::Qwen3VLWeights w = vllm::LoadQwen3VLWeights(shards, cfg);

  // --- M3c: our video processor on the fixture video -> pixel_values_videos ---
  vllm::multimodal::Qwen3VLProcessorConfig pcfg = vllm::multimodal::LoadQwen3VLProcessorConfig(
      ckpt + "/preprocessor_config.json", ckpt + "/config.json", "Qwen/Qwen3-VL-4B-Instruct");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> thwc = ReadU8(VidFix() + "/" + raw_file);
  REQUIRE(thwc.size() == static_cast<size_t>(T * H * W * 3));
  const vllm::multimodal::VideoKwargs vid = proc.ProcessVideo(thwc.data(), T, H, W, frame_indices, fps);
  const std::array<int64_t, 3> grid = vid.video_grid_thw;
  MESSAGE("video_grid_thw = [" << grid[0] << "," << grid[1] << "," << grid[2] << "]  patches="
          << vid.num_patches);
  REQUIRE(grid[0] > 1);

  // --- M2a: vision tower (per-frame windowed attn for video) -----------------
  vllm::multimodal::Qwen3VLVisionConfig vcfg = w.vision_cfg;
  std::vector<float> tower =
      vllm::multimodal::Qwen3VLVisionForward(vid.pixel_values_bf16, grid, w.vision, vcfg, *gpu);
  const int64_t out_hidden = vcfg.out_hidden_size;   // 2560
  const int64_t Ldeep = static_cast<int64_t>(vcfg.deepstack_visual_indexes.size());  // 3
  const int64_t total = out_hidden * (1 + Ldeep);    // 10240
  const int64_t N = static_cast<int64_t>(tower.size()) / total;
  MESSAGE("tower_out rows=" << N << " cols=" << total);
  const int64_t expect_N = grid[0] * (grid[1] / 2) * (grid[2] / 2);
  REQUIRE(N == expect_N);

  // VIDEO-TOWER faithfulness gate: our per-frame windowed-attention tower output
  // vs the dumped vLLM 0.25.0 video tower reference (scripts/mm/
  // m3c_video_tower_ref_dump.py). The M2a image tower gate only covered grid_t==1;
  // this gates the t>1 windowed path. Tolerance = the measured bf16-depth envelope
  // (M2a image tower rel-L2 ~5e-2 over 24 blocks).
  {
    const std::string ref_path = VidFix() + "/tower_out_ref_f32.bin";
    if (fs::exists(ref_path)) {
      std::ifstream rf(ref_path, std::ios::binary);
      rf.seekg(0, std::ios::end);
      const size_t rn = static_cast<size_t>(rf.tellg()) / sizeof(float);
      rf.seekg(0, std::ios::beg);
      std::vector<float> ref(rn);
      rf.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(rn * sizeof(float)));
      REQUIRE(ref.size() == tower.size());
      double num = 0.0, den = 0.0;
      for (size_t i = 0; i < tower.size(); ++i) {
        const double d = static_cast<double>(tower[i]) - static_cast<double>(ref[i]);
        num += d * d;
        den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
      }
      const double relL2 = den > 0 ? std::sqrt(num / den) : 0.0;
      MESSAGE("video tower rel-L2 vs vLLM = " << relL2 << " (image envelope ~5e-2)");
      CHECK(relL2 < 0.1);
    }
  }

  std::vector<float> mm_main(static_cast<size_t>(N * out_hidden));
  std::vector<float> mm_deep(static_cast<size_t>(N * Ldeep * out_hidden));
  for (int64_t i = 0; i < N; ++i) {
    const float* row = tower.data() + static_cast<size_t>(i * total);
    std::copy(row, row + out_hidden, mm_main.begin() + static_cast<size_t>(i * out_hidden));
    std::copy(row + out_hidden, row + total,
              mm_deep.begin() + static_cast<size_t>(i * Ldeep * out_hidden));
  }

  // --- M3c: greedy forward on the committed vLLM chat-templated prompt ids ----
  const std::vector<int32_t> prompt_ids = ReadI32(in_ids_path);
  const std::vector<int32_t> golden = ReadI32(VidFix() + "/gen_tokens_i32.bin");
  const int nsteps = static_cast<int>(golden.size());
  REQUIRE(nsteps > 0);
  MESSAGE("prompt_ids length = " << prompt_ids.size() << "  golden steps = " << nsteps);

  vt::Queue q = gpu->CreateQueue();
  const std::vector<int32_t> gen = vllm::Qwen3VLGenerateGreedyVideo(
      prompt_ids, mm_main, mm_deep, Ldeep, grid, video_token_id, vs_id, ve_id,
      /*eos_token_id=*/151645, w, cfg, q, /*max_new_tokens=*/nsteps);

  REQUIRE(gen.size() == golden.size());
  int matches = 0;
  for (size_t i = 0; i < golden.size(); ++i)
    if (gen[i] == golden[i]) ++matches;
  MESSAGE("STRICT video token-exact: " << matches << "/" << golden.size() << " match golden");
  MESSAGE("ours  [:8] = " << gen[0] << "," << gen[1] << "," << gen[2] << "," << gen[3]
          << "," << gen[4] << "," << gen[5] << "," << gen[6] << "," << gen[7]);
  MESSAGE("golden[:8] = " << golden[0] << "," << golden[1] << "," << golden[2] << ","
          << golden[3] << "," << golden[4] << "," << golden[5] << "," << golden[6] << ","
          << golden[7]);
  for (size_t i = 0; i < golden.size(); ++i) CHECK(gen[i] == golden[i]);
}
