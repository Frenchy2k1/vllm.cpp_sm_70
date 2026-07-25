// Processor-parity gate (M1 correctness gate). Verifies the C++ Qwen3-VL mm
// input pipeline is BIT/BYTE-identical to the vLLM 0.25.0 oracle fixture
// (captured by scripts/mm/m0_oracle_capture.py): pixel_values (bf16), grid_thw,
// placeholder-expanded prompt ids, and the MultiModalHasher mm-hash.
//
// Golden: tests/vllm/multimodal/fixtures/qwen3vl/{manifest.json,
//   image_rgb_uint8_448x448x3.bin, image_grid_thw_i64.bin, pixel_values_f32.bin}
// Upstream oracle: MULTIMODAL_REGISTRY.create_processor(...).apply for
//   Qwen/Qwen3-VL-4B-Instruct.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vt/dtype.h"

namespace {

std::string FixDir() { return std::string(MM_FIXTURE_DIR); }

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  nlohmann::json j;
  f >> j;
  return j;
}

vllm::multimodal::Qwen3VLProcessorConfig ConfigFromManifest(
    const nlohmann::json& m) {
  vllm::multimodal::Qwen3VLProcessorConfig cfg;
  const auto& c = m.at("config");
  cfg.patch_size = c.at("patch_size").get<int>();
  cfg.temporal_patch_size = c.at("temporal_patch_size").get<int>();
  cfg.merge_size = c.at("merge_size").get<int>();
  cfg.image_mean = c.at("image_mean")[0].get<double>();
  cfg.image_std = c.at("image_std")[0].get<double>();
  cfg.image_token_id = c.at("image_token_id").get<int32_t>();
  cfg.vision_start_token_id = c.at("vision_start_token_id").get<int32_t>();
  cfg.vision_end_token_id = c.at("vision_end_token_id").get<int32_t>();
  cfg.model_id = m.at("model_id").get<std::string>();
  return cfg;
}

}  // namespace

TEST_CASE("qwen3vl-processor-parity: pixel_values + grid + hash + expansion") {
  const std::string dir = FixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ConfigFromManifest(manifest);

  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == static_cast<size_t>(H * W * 3));

  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const auto kw = proc.ProcessImage(rgb.data(), H, W);

  SUBCASE("image_grid_thw exact") {
    const auto g = manifest.at("image_grid_thw").at("values");
    CHECK(kw.image_grid_thw[0] == g[0].get<int64_t>());
    CHECK(kw.image_grid_thw[1] == g[1].get<int64_t>());
    CHECK(kw.image_grid_thw[2] == g[2].get<int64_t>());
  }

  SUBCASE("pixel_values bf16 byte-identical to production golden") {
    // Golden file holds the bf16 production values widened to float32.
    const std::vector<uint8_t> gbytes = ReadBytes(dir + "/pixel_values_f32.bin");
    REQUIRE(gbytes.size() == kw.pixel_values_bf16.size() * sizeof(float));
    size_t mismatches = 0;
    for (size_t i = 0; i < kw.pixel_values_bf16.size(); ++i) {
      float g;
      std::memcpy(&g, gbytes.data() + i * sizeof(float), sizeof(float));
      // Golden values are exactly bf16-representable, so F32ToBF16(golden) is
      // the original bf16 code. Compare bf16 codes -> bit-exact, no type-punning.
      if (kw.pixel_values_bf16[i] != vt::F32ToBF16(g)) ++mismatches;
    }
    CHECK(mismatches == 0);
  }

  SUBCASE("mm-hash byte-identical to MultiModalHasher") {
    const std::string ours = proc.HashImage(rgb.data(), H, W);
    CHECK(ours == manifest.at("mm_hash").get<std::string>());
  }

  SUBCASE("placeholder expansion byte-identical to oracle expanded ids") {
    std::vector<int32_t> pre;
    for (const auto& v : manifest.at("pre_expansion_token_ids"))
      pre.push_back(v.get<int32_t>());
    std::vector<int32_t> golden;
    for (const auto& v : manifest.at("expanded_prompt_token_ids"))
      golden.push_back(v.get<int32_t>());

    std::vector<std::array<int64_t, 3>> grids = {kw.image_grid_thw};
    std::vector<std::array<int, 2>> placeholders;
    const auto expanded = vllm::multimodal::ExpandImagePlaceholders(
        pre, cfg.image_token_id, cfg.merge_size, grids, &placeholders);

    CHECK(expanded == golden);
    REQUIRE(placeholders.size() == 1);
    const int64_t expected_n = (kw.image_grid_thw[0] * kw.image_grid_thw[1] *
                                kw.image_grid_thw[2]) /
                               (cfg.merge_size * cfg.merge_size);
    CHECK(placeholders[0][1] == static_cast<int>(expected_n));
    CHECK(manifest.at("n_image_tokens").get<int>() ==
          static_cast<int>(expected_n));
  }
}
