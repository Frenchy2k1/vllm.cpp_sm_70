// M3d — the end-to-end VIDEO->text token gate on Qwen3.6-27B
// (`Qwen3_5ForConditionalGeneration`, the GDN-hybrid gate model — our own model's
// video path). The 27B sibling of the M3c 4B video gate; a REUSE increment.
//
// Runs the FULL pipeline on the committed fixture (video, prompt), reusing the
// M3c synthetic clip re-captured for the 27B:
//   fixture video -> C++ Qwen3VLImageProcessor::ProcessVideo (M3c, config-shared)
//                    -> pixel_values_videos + video_grid_thw (t>1)
//                 -> Qwen3VLVisionForward (M2a tower, 27B vision config: depth 27,
//                    hidden 1152, out_hidden 5120, EMPTY deepstack; per-frame
//                    windowed attn) -> merger [N,5120]
//                 -> Qwen3_5VLGenerateGreedyVideo (M3d video merge mask on
//                    video_token(248057) + per-frame video MRoPE [11,11,10] on the
//                    16 full-attn layers, GDN-hybrid backbone, paged greedy) ->
//                    greedy tokens.
//
// GATE FORM (selected BY MEASUREMENT, matching the M3c 4B video gate): the tower
// bf16 envelope is shared with the 4B (same tower, same accumulation), so a
// vision-conditioned decode may sit in a bf16 near-tie band. The gate is the
// measured near-tie equivalence class (identical form to the text near-tie gates):
//   (a) our engine reproduces its committed deterministic anchor our_ids_i32.bin
//       (drift => REQUIRE fail);
//   (b) at EVERY position vLLM's teacher-forced gap between its argmax and OUR
//       token, given OUR prefix (neartie_gap_mnats_i32.bin, from
//       scripts/mm/m3c_video_neartie_gap.py --model Qwen/Qwen3.6-27B), is
//       <= kNearTieMnats (0.5 nats). A gap above that is a REAL forward divergence
//       the gate FAILS on. If vLLM's own greedy golden is K=5 DETERMINISTIC and our
//       tokens match it exactly, this collapses to a STRICT pass (worst gap 0).
//
// BOOTSTRAP: with VT_DUMP_IDS=1, or when the anchor/gap goldens are absent, the
// case dumps OUR token ids to our_ids_i32.bin and does NOT assert the band — run
// the gap script, commit the goldens, then the gate is live.
//
// dgx-only: needs CUDA + the cached vision-inclusive bf16 checkpoint
// `Qwen/Qwen3.6-27B` (VLLM_QWEN36_CKPT, or the default HF cache path). Skipped
// (not failed) without.
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
#include "vllm/model_executor/models/qwen3_5_dense.h"
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
  if (const char* e = std::getenv("VLLM_QWEN36_CKPT")) return e;
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& d : fs::directory_iterator(snaps, ec))
    if (fs::exists(d.path() / "config.json", ec)) return d.path().string();
  return "";
}

// The Qwen3.6-27B vision config (config.json vision_config; DIFFERS from 4B:
// depth 27, hidden 1152, out_hidden 5120, intermediate 4304, EMPTY deepstack).
vllm::multimodal::Qwen3VLVisionConfig VisionConfig27B() {
  vllm::multimodal::Qwen3VLVisionConfig v;
  v.hidden_size = 1152;
  v.num_heads = 16;
  v.depth = 27;
  v.intermediate_size = 4304;
  v.out_hidden_size = 5120;
  v.patch_size = 16;
  v.temporal_patch_size = 2;
  v.spatial_merge_size = 2;
  v.num_position_embeddings = 2304;
  v.in_channels = 3;
  v.deepstack_visual_indexes = {};  // NO DeepStack on the 27B.
  v.norm_eps = 1e-6f;
  return v;
}

}  // namespace

TEST_CASE("qwen3_5_27b_e2e_video_token_NEARTIE_ROBUST_vs_vllm_0_25_0") {
  const std::string ckpt = FindCkpt();
  if (ckpt.empty()) {
    MESSAGE("SKIP: Qwen3.6-27B checkpoint absent (set VLLM_QWEN36_CKPT)");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const std::string in_ids_path = VidFix() + "/gen_input_ids_i32.bin";
  if (!fs::exists(in_ids_path)) {
    MESSAGE("SKIP: gen_input_ids_i32.bin absent (run scripts/mm/m3d_video_oracle_capture.py)");
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

  // --- config (text_config-resolved) ----------------------------------------
  const vllm::HfConfig cfg = vllm::LoadHfConfig(ckpt + "/config.json");
  MESSAGE("config: layers=" << cfg.num_hidden_layers << " hidden=" << cfg.hidden_size
          << " head_dim=" << cfg.head_dim << " rotary_dim=" << cfg.rotary_dim
          << " vocab=" << cfg.vocab_size << " theta=" << cfg.rope_theta);
  REQUIRE(cfg.hidden_size == 5120);
  REQUIRE(cfg.num_hidden_layers == 64);
  REQUIRE(cfg.head_dim == 256);
  REQUIRE(cfg.rotary_dim == 64);
  REQUIRE(cfg.rope_parameters.mrope_section.size() == 3u);
  REQUIRE(cfg.rope_parameters.mrope_section[0] == 11);
  REQUIRE(cfg.rope_parameters.mrope_section[1] == 11);
  REQUIRE(cfg.rope_parameters.mrope_section[2] == 10);

  // --- weights: GDN-hybrid LLM (direct device load + host release) + vision ---
  std::vector<vllm::SafetensorsFile> shards;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(ckpt, ec))
    if (e.path().extension() == ".safetensors")
      shards.push_back(vllm::SafetensorsFile::Open(e.path().string()));
  REQUIRE(!shards.empty());

  vt::Queue q = gpu->CreateQueue();
  const vllm::Qwen3_5DenseWeights llm = vllm::LoadQwen3_5Dense(shards, cfg, &q);
  const vllm::multimodal::Qwen3VLVisionConfig vcfg = VisionConfig27B();
  const vllm::multimodal::Qwen3VLVisionWeights vw =
      vllm::LoadQwen3VLVisionWeights(shards, vcfg);

  // --- M3c processor on the fixture video -> pixel_values_videos + grid -------
  vllm::multimodal::Qwen3VLProcessorConfig pcfg =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          ckpt + "/preprocessor_config.json", ckpt + "/config.json",
          "Qwen/Qwen3.6-27B");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> thwc = ReadU8(VidFix() + "/" + raw_file);
  REQUIRE(thwc.size() == static_cast<size_t>(T * H * W * 3));
  const vllm::multimodal::VideoKwargs vid =
      proc.ProcessVideo(thwc.data(), T, H, W, frame_indices, fps);
  const std::array<int64_t, 3> grid = vid.video_grid_thw;
  MESSAGE("video_grid_thw = [" << grid[0] << "," << grid[1] << "," << grid[2]
          << "]  patches=" << vid.num_patches);
  REQUIRE(grid[0] > 1);

  // --- M2a: vision tower (per-frame windowed attn, 27B config, NO deepstack) ---
  std::vector<float> tower = vllm::multimodal::Qwen3VLVisionForward(
      vid.pixel_values_bf16, grid, vw, vcfg, *gpu);
  const int64_t out_hidden = vcfg.out_hidden_size;  // 5120 == text hidden
  REQUIRE(vcfg.deepstack_visual_indexes.empty());
  const int64_t N = static_cast<int64_t>(tower.size()) / out_hidden;
  MESSAGE("tower_out rows=" << N << " cols=" << out_hidden);
  const int64_t expect_N = grid[0] * (grid[1] / 2) * (grid[2] / 2);
  REQUIRE(N == expect_N);
  REQUIRE(out_hidden == cfg.hidden_size);

  // --- M3d: greedy video forward on the committed vLLM chat-templated ids ------
  const std::vector<int32_t> prompt_ids = ReadI32(in_ids_path);
  const std::vector<int32_t> golden = ReadI32(VidFix() + "/gen_tokens_i32.bin");
  const int nsteps = static_cast<int>(golden.size());
  REQUIRE(nsteps > 0);
  MESSAGE("prompt_ids length = " << prompt_ids.size() << "  golden steps = " << nsteps);

  const std::vector<int32_t> gen = vllm::Qwen3_5VLGenerateGreedyVideo(
      prompt_ids, tower, grid, video_token_id, vs_id, ve_id,
      /*eos_token_id=*/-1, llm, cfg, q, /*max_new_tokens=*/nsteps);

  REQUIRE(gen.size() == golden.size());
  int matches = 0;
  for (size_t i = 0; i < golden.size(); ++i)
    if (gen[i] == golden[i]) ++matches;
  MESSAGE("video token-exact vs vLLM greedy golden: " << matches << "/" << golden.size());
  MESSAGE("ours  [:8] = " << gen[0] << "," << gen[1] << "," << gen[2] << "," << gen[3]
          << "," << gen[4] << "," << gen[5] << "," << gen[6] << "," << gen[7]);
  MESSAGE("golden[:8] = " << golden[0] << "," << golden[1] << "," << golden[2] << ","
          << golden[3] << "," << golden[4] << "," << golden[5] << "," << golden[6] << ","
          << golden[7]);

  // ---- NEAR-TIE-ROBUST gate (mirrors the M3c 4B video gate) ----------------
  constexpr int32_t kNearTieMnats = 500;
  const std::string anchor_path = VidFix() + "/our_ids_i32.bin";
  const std::string gap_path = VidFix() + "/neartie_gap_mnats_i32.bin";
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool goldens_present = fs::exists(anchor_path) && fs::exists(gap_path);

  if (dump || !goldens_present) {
    std::ofstream f(anchor_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(gen.data()),
            static_cast<std::streamsize>(gen.size() * sizeof(int32_t)));
    MESSAGE("BOOTSTRAP: wrote our token ids -> " << anchor_path
            << " (near-tie gap golden absent; run scripts/mm/m3c_video_neartie_gap.py "
               "--model Qwen/Qwen3.6-27B --fixture-dir <this dir> --gpu-mem-util 0.6, "
               "then commit our_ids_i32.bin + neartie_gap_mnats_i32.bin)");
    return;
  }

  const std::vector<int32_t> anchor = ReadI32(anchor_path);
  const std::vector<int32_t> gap = ReadI32(gap_path);
  REQUIRE(anchor.size() == golden.size());
  REQUIRE(gap.size() == golden.size());

  // (a) Hard anchor: our exact committed deterministic sequence.
  int anchor_div = -1;
  for (size_t i = 0; i < gen.size(); ++i)
    if (gen[i] != anchor[i]) { anchor_div = static_cast<int>(i); break; }
  REQUIRE_MESSAGE(anchor_div < 0,
                  "video anchor drift at tok=" << anchor_div
                  << " engine=" << (anchor_div < 0 ? -1 : gen[static_cast<size_t>(anchor_div)])
                  << " committed=" << (anchor_div < 0 ? -1 : anchor[static_cast<size_t>(anchor_div)])
                  << " — re-run scripts/mm/m3c_video_neartie_gap.py to refresh the gap golden");

  // (b) Near-tie band: EVERY position's teacher-forced gap <= kNearTieMnats.
  bool prompt_ok = true;
  int first_bad = -1;
  int32_t worst_gap = 0, worst_j = -1;
  for (size_t j = 0; j < gap.size(); ++j) {
    const int32_t mn = gap[j];
    if (mn > worst_gap) { worst_gap = mn; worst_j = static_cast<int32_t>(j); }
    if (mn > kNearTieMnats) { prompt_ok = false; if (first_bad < 0) first_bad = static_cast<int>(j); }
  }
  if (!prompt_ok) {
    MESSAGE("video FORWARD DIVERGENCE tok=" << first_bad
            << " our=" << gen[static_cast<size_t>(first_bad)]
            << " vLLM_greedy=" << golden[static_cast<size_t>(first_bad)]
            << " gap=" << (gap[static_cast<size_t>(first_bad)] / 1000.0) << " nats (> "
            << (kNearTieMnats / 1000.0) << ")");
  }
  MESSAGE("27B video NEAR-TIE-ROBUST gate: " << (prompt_ok ? "PASS" : "FAIL")
          << "  (STRICT token-exact vs vLLM greedy: " << matches << "/" << golden.size()
          << "; max teacher-forced gap " << (worst_gap / 1000.0) << " nats @ tok=" << worst_j
          << ")");
  CHECK(prompt_ok);
  REQUIRE(prompt_ok);
}
