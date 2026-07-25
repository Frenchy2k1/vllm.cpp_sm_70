// M3c — VIDEO-processor UNIT gate. Verifies the C++ Qwen3-VL VIDEO input pipeline
// is BIT-identical to the vLLM 0.25.0 oracle fixture (scripts/mm/
// m3c_video_oracle_capture.py): pixel_values_videos (bf16), video_grid_thw, the
// per-group timestamps, and the timestamp-interleaved placeholder expansion
// (BuildVideoRepl). RED-first: the video patchify uses the temporal_patch_size=2
// mapping to 2 REAL frames (source frame = grid_t_index*tp + t), NOT the image
// duplicate — a wrong temporal/order mapping fails pixel_values bit-exactness.
//
// Golden: tests/vllm/multimodal/fixtures/qwen3vl_video/{manifest.json,
//   video_rgb_uint8_8x128x128x3.bin, video_grid_thw_i64.bin,
//   pixel_values_videos_f32.bin}
// Upstream oracle: MULTIMODAL_REGISTRY.create_processor(...) for
//   Qwen/Qwen3-VL-4B-Instruct on the fixed synthetic video.
#include <array>
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

std::string FixDir() { return std::string(MM_VIDEO_FIXTURE_DIR); }

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

vllm::multimodal::Qwen3VLProcessorConfig ConfigFromManifest(const nlohmann::json& m) {
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

TEST_CASE("qwen3vl-video-processor-parity: pixel_values_videos + grid + timestamps + expansion") {
  const std::string dir = FixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ConfigFromManifest(manifest);

  const auto vshape = manifest.at("video").at("shape");
  const int64_t T = vshape[0].get<int64_t>();
  const int64_t H = vshape[1].get<int64_t>();
  const int64_t W = vshape[2].get<int64_t>();
  const std::string raw_file = manifest.at("video").at("raw_file").get<std::string>();
  const std::vector<uint8_t> thwc = ReadBytes(dir + "/" + raw_file);
  REQUIRE(thwc.size() == static_cast<size_t>(T * H * W * 3));

  // metadata drives frame_indices + fps (do_sample_frames=false -> identity indices)
  const auto& meta = manifest.at("video").at("metadata");
  std::vector<int64_t> frame_indices;
  for (const auto& v : meta.at("frames_indices")) frame_indices.push_back(v.get<int64_t>());
  const double fps = meta.at("fps").get<double>();

  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const auto kw = proc.ProcessVideo(thwc.data(), T, H, W, frame_indices, fps);

  SUBCASE("video_grid_thw exact") {
    const auto g = manifest.at("video_grid_thw").at("values");
    CHECK(kw.video_grid_thw[0] == g[0].get<int64_t>());
    CHECK(kw.video_grid_thw[1] == g[1].get<int64_t>());
    CHECK(kw.video_grid_thw[2] == g[2].get<int64_t>());
  }

  SUBCASE("timestamps exact vs oracle") {
    const auto ts = manifest.at("timestamps");
    REQUIRE(kw.timestamps.size() == ts.size());
    for (size_t i = 0; i < ts.size(); ++i)
      CHECK(kw.timestamps[i] == doctest::Approx(ts[i].get<double>()).epsilon(1e-9));
  }

  SUBCASE("pixel_values_videos bf16 byte-identical to production golden") {
    const std::vector<uint8_t> gbytes = ReadBytes(dir + "/pixel_values_videos_f32.bin");
    REQUIRE(gbytes.size() == kw.pixel_values_bf16.size() * sizeof(float));
    size_t mismatches = 0;
    for (size_t i = 0; i < kw.pixel_values_bf16.size(); ++i) {
      float g;
      std::memcpy(&g, gbytes.data() + i * sizeof(float), sizeof(float));
      if (kw.pixel_values_bf16[i] != vt::F32ToBF16(g)) ++mismatches;
    }
    MESSAGE("pixel_values_videos mismatches = " << mismatches << " / "
            << kw.pixel_values_bf16.size());
    CHECK(mismatches == 0);
  }

  SUBCASE("BuildVideoRepl reproduces the oracle timestamp-interleaved expansion") {
    std::vector<int32_t> golden;
    for (const auto& v : manifest.at("expanded_prompt_token_ids"))
      golden.push_back(v.get<int32_t>());
    const int32_t vs = cfg.vision_start_token_id;
    const int32_t ve = cfg.vision_end_token_id;
    const int32_t vid = manifest.at("config").at("video_token_id").get<int32_t>();
    const int64_t grid_t = kw.video_grid_thw[0];

    // Extract the per-frame timestamp id lists + counts by walking the golden.
    std::vector<std::vector<int32_t>> ts_ids;
    std::vector<int64_t> tokens_per_frame;
    size_t pos = 0;
    for (int64_t f = 0; f < grid_t; ++f) {
      // vision_start for this frame
      size_t vsi = pos;
      while (vsi < golden.size() && golden[vsi] != vs) ++vsi;
      REQUIRE(vsi < golden.size());
      ts_ids.emplace_back(golden.begin() + pos, golden.begin() + vsi);
      // count video tokens until vision_end
      size_t vi = vsi + 1;
      int64_t cnt = 0;
      while (vi < golden.size() && golden[vi] == vid) { ++cnt; ++vi; }
      REQUIRE(vi < golden.size());
      CHECK(golden[vi] == ve);
      tokens_per_frame.push_back(cnt);
      pos = vi + 1;  // past vision_end
    }
    // Every frame should carry the base tokens-per-frame (16 for the fixture).
    const int64_t expected_per_frame =
        (kw.video_grid_thw[1] * kw.video_grid_thw[2]) /
        (cfg.merge_size * cfg.merge_size);
    for (int64_t c : tokens_per_frame) CHECK(c == expected_per_frame);

    const auto repl = vllm::multimodal::BuildVideoRepl(tokens_per_frame, ts_ids, vs, vid, ve);
    // The video block is golden[0..pos); the tail is the question text.
    std::vector<int32_t> golden_block(golden.begin(), golden.begin() + static_cast<long>(pos));
    CHECK(repl == golden_block);
    CHECK(static_cast<int>(manifest.at("n_video_tokens").get<int>()) ==
          static_cast<int>(grid_t * expected_per_frame));
  }
}
