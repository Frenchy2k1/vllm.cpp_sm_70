// MM-ENGINE-FORWARD — the STRICT end-to-end image->text token-exact gate on
// Qwen3-VL-4B driven THROUGH the ENGINE registered forward (ModelRegistry::Forward
// → the registered ForwardQwen3VLForConditionalGeneration), NOT the standalone
// Qwen3VLGenerateGreedy driver.
//
// This is gate #1 of the engine-mm-forward integration: it proves the mm-forward
// runs through the registered path and emits the SAME M2c golden tokens the
// standalone driver does (32/32 STRICT). RED-FIRST: before this TU's registration,
// "Qwen3VLForConditionalGeneration" is unregistered → ModelRegistry::Resolve
// throws / ModelRegistry::Forward has no mm path; after, the registered forward is
// token-identical to the standalone driver (both call Qwen3VLForwardStepLastLogits).
//
// Pipeline (identical inputs to test_qwen3vl_e2e):
//   fixture image -> Qwen3VLImageProcessor (M1) -> pixel_values + grid
//                 -> Qwen3VLVisionForward (M2a tower) -> merger | deepstack
//                 -> Qwen3VLGenerateGreedyViaRegistry (per-step ModelRegistry::
//                    Forward with ModelForwardInput.mm) -> 32 greedy tokens.
//
// dgx-only: needs CUDA + the cached Qwen/Qwen3-VL-4B-Instruct checkpoint
// (VLLM_QWEN3VL_CKPT, or the default HF cache path). Skipped (not failed) without.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {
namespace fs = std::filesystem;

std::string TextFix() { return std::string(TEXT_FIXTURE_DIR); }
std::string ImgFix() { return std::string(MM_FIXTURE_DIR); }

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

TEST_CASE("qwen3vl_registry_e2e_image_token_exact_STRICT_via_ModelRegistry_Forward") {
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
  const std::string in_ids_path = TextFix() + "/input_ids_i32.bin";
  if (!fs::exists(in_ids_path)) {
    MESSAGE("SKIP: input_ids_i32.bin absent (run scripts/mm/m2c_e2e_inputs.py)");
    return;
  }

  // The registered arch must resolve (RED-first: unregistered ⇒ this throws).
  const std::vector<std::string> arch = {"Qwen3VLForConditionalGeneration"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
  REQUIRE(reg.architecture == "Qwen3VLForConditionalGeneration");
  REQUIRE(reg.info.supports_multimodal);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->forward != nullptr);

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

  // The registered LoadedModel BORROWS `w` (no multi-GB copy); `w` outlives it and
  // also feeds the host embed/merge inside Qwen3VLGenerateGreedyViaRegistry.
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  REQUIRE(&model->registration() == &reg);

  // --- M1: processor -> pixel_values + grid ---------------------------------
  vllm::multimodal::Qwen3VLProcessorConfig pcfg =
      vllm::multimodal::LoadQwen3VLProcessorConfig(
          ckpt + "/preprocessor_config.json", ckpt + "/config.json",
          "Qwen/Qwen3-VL-4B-Instruct");
  vllm::multimodal::Qwen3VLImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb = ReadU8(ImgFix() + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == 448u * 448u * 3u);
  const vllm::multimodal::ImageKwargs img = proc.ProcessImage(rgb.data(), 448, 448);
  const std::array<int64_t, 3> grid = img.image_grid_thw;
  REQUIRE(grid[0] == 1);
  REQUIRE(grid[1] == 28);
  REQUIRE(grid[2] == 28);

  // --- M2a: vision tower -> split merger | deepstack ------------------------
  vllm::multimodal::Qwen3VLVisionConfig vcfg = w.vision_cfg;
  std::vector<float> tower = vllm::multimodal::Qwen3VLVisionForward(
      img.pixel_values_bf16, grid, w.vision, vcfg, *gpu);
  const int64_t out_hidden = vcfg.out_hidden_size;  // 2560
  const int64_t Ldeep = static_cast<int64_t>(vcfg.deepstack_visual_indexes.size());
  const int64_t total = out_hidden * (1 + Ldeep);  // 10240
  const int64_t N = static_cast<int64_t>(tower.size()) / total;
  REQUIRE(N == 196);

  std::vector<float> mm_main(static_cast<size_t>(N * out_hidden));
  std::vector<float> mm_deep(static_cast<size_t>(N * Ldeep * out_hidden));
  for (int64_t i = 0; i < N; ++i) {
    const float* row = tower.data() + static_cast<size_t>(i * total);
    std::copy(row, row + out_hidden, mm_main.begin() + static_cast<size_t>(i * out_hidden));
    std::copy(row + out_hidden, row + total,
              mm_deep.begin() + static_cast<size_t>(i * Ldeep * out_hidden));
  }

  // --- gate: greedy forward THROUGH ModelRegistry::Forward -------------------
  const std::vector<int32_t> prompt_ids = ReadI32(in_ids_path);
  const std::vector<int32_t> golden = ReadI32(TextFix() + "/gen_tokens_i32.bin");
  REQUIRE(golden.size() == 32u);

  vt::Queue q = gpu->CreateQueue();
  vllm::ModelRegistry::Prepare(*model, cfg, q);  // warm the cos|sin cache
  const std::vector<int32_t> gen = vllm::Qwen3VLGenerateGreedyViaRegistry(
      *model, prompt_ids, mm_main, mm_deep, Ldeep, grid, /*image_token_id=*/151655,
      /*eos_token_id=*/151645, w, cfg, q, /*max_new_tokens=*/32);

  REQUIRE(gen.size() == 32u);
  int matches = 0;
  for (int i = 0; i < 32; ++i)
    if (gen[static_cast<size_t>(i)] == golden[static_cast<size_t>(i)]) ++matches;
  MESSAGE("ENGINE-registered mm-forward STRICT image token-exact: " << matches
          << "/32 match golden");
  for (int i = 0; i < 32; ++i)
    CHECK(gen[static_cast<size_t>(i)] == golden[static_cast<size_t>(i)]);
}
