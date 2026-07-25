// M3-b — the STRICT end-to-end image->text token-exact gate on Qwen3.6-27B
// (`Qwen3_5ForConditionalGeneration`, the GDN-hybrid gate model — our own model's
// image path).
//
// Runs the FULL pipeline on the committed fixture (image, prompt):
//   fixture image -> C++ Qwen3VLImageProcessor (M1) -> pixel_values + grid
//                 -> Qwen3VLVisionForward (M2a tower, 27B vision config: depth 27,
//                    hidden 1152, out_hidden 5120, EMPTY deepstack) -> merger [N,5120]
//                 -> Qwen3_5VLGenerateGreedy (M3-b forked GDN-hybrid decode:
//                    embed+scatter into image_token(248056) rows, 3-section MRoPE
//                    [11,11,10] interleaved on the 16 full-attn layers, GDN-hybrid
//                    backbone, paged greedy) -> 32 greedy tokens.
// Asserts token-for-token equality with the committed vLLM 0.25.0 golden
// (qwen3_5_27b/gen_tokens_i32.bin, gate form STRICT per gen_manifest.json). The
// expanded model input ids are the committed input_ids_i32.bin (214 tokens, 196
// image at offset 4); the tokenizer text path is out of this test's numeric scope.
//
// dgx-only: needs CUDA + the cached vision-inclusive bf16 checkpoint
// `Qwen/Qwen3.6-27B` (VLLM_QWEN36_CKPT, or the default HF cache path). Skipped
// (not failed) without.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

std::string ImgFix() { return std::string(MM_FIXTURE_DIR); }
std::string GoldFix() { return std::string(GOLD_FIXTURE_DIR); }

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

TEST_CASE("qwen3_5_27b_e2e_image_token_exact_STRICT_vs_vllm_0_25_0") {
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

  // --- config (text_config-resolved) ----------------------------------------
  const vllm::HfConfig cfg = vllm::LoadHfConfig(ckpt + "/config.json");
  MESSAGE("config: layers=" << cfg.num_hidden_layers << " hidden=" << cfg.hidden_size
          << " heads=" << cfg.num_attention_heads << " kv=" << cfg.num_key_value_heads
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

  // --- M1: our processor on the fixture image -> pixel_values + grid ----------
  vllm::multimodal::Qwen3VLProcessorConfig pcfg =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          ckpt + "/preprocessor_config.json", ckpt + "/config.json",
          "Qwen/Qwen3.6-27B");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb =
      ReadU8(ImgFix() + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == 448u * 448u * 3u);
  const vllm::multimodal::ImageKwargs img = proc.ProcessImage(rgb.data(), 448, 448);
  const std::array<int64_t, 3> grid = img.image_grid_thw;
  MESSAGE("grid_thw = [" << grid[0] << "," << grid[1] << "," << grid[2]
          << "]  patches=" << img.num_patches);
  REQUIRE(grid[0] == 1);
  REQUIRE(grid[1] == 28);
  REQUIRE(grid[2] == 28);

  // --- M2a: vision tower -> [N, out_hidden] (no deepstack, so total==out) -----
  std::vector<float> tower =
      vllm::multimodal::Qwen3VLVisionForward(img.pixel_values_bf16, grid, vw, vcfg, *gpu);
  const int64_t out_hidden = vcfg.out_hidden_size;  // 5120 == text hidden
  REQUIRE(vcfg.deepstack_visual_indexes.empty());
  const int64_t N = static_cast<int64_t>(tower.size()) / out_hidden;
  MESSAGE("tower_out rows=" << N << " cols=" << out_hidden);
  REQUIRE(N == 196);
  REQUIRE(out_hidden == cfg.hidden_size);

  // --- M3-b: greedy forward on the committed vLLM prompt_token_ids ------------
  const std::vector<int32_t> prompt_ids = ReadI32(GoldFix() + "/input_ids_i32.bin");
  const std::vector<int32_t> golden = ReadI32(GoldFix() + "/gen_tokens_i32.bin");
  REQUIRE(prompt_ids.size() == 214u);
  REQUIRE(golden.size() == 32u);

  const std::vector<int32_t> gen = vllm::Qwen3_5VLGenerateGreedy(
      prompt_ids, tower, grid, /*image_token_id=*/248056, /*eos_token_id=*/-1,
      llm, cfg, q, /*max_new_tokens=*/32);

  REQUIRE(gen.size() == 32u);
  int matches = 0;
  for (int i = 0; i < 32; ++i)
    if (gen[static_cast<size_t>(i)] == golden[static_cast<size_t>(i)]) ++matches;
  MESSAGE("STRICT image token-exact: " << matches << "/32 match golden");
  MESSAGE("ours  [:8] = " << gen[0] << "," << gen[1] << "," << gen[2] << "," << gen[3]
          << "," << gen[4] << "," << gen[5] << "," << gen[6] << "," << gen[7]);
  MESSAGE("golden[:8] = " << golden[0] << "," << golden[1] << "," << golden[2] << ","
          << golden[3] << "," << golden[4] << "," << golden[5] << "," << golden[6]
          << "," << golden[7]);
  for (int i = 0; i < 32; ++i)
    CHECK(gen[static_cast<size_t>(i)] == golden[static_cast<size_t>(i)]);
}
