// Qwen3-VL text-backbone helpers (M2b/M2c). See qwen3_vl_text.h for the port map.
#include "vllm/model_executor/models/qwen3_vl_text.h"

#include <algorithm>
#include <stdexcept>

namespace vllm::multimodal {

std::vector<int32_t> Qwen3VLGetRopeIndex(const std::vector<int32_t>& input_ids,
                                         const std::vector<MmImageSpan>& images_in,
                                         int64_t spatial_merge_size,
                                         int64_t* mrope_position_delta) {
  const int64_t T = static_cast<int64_t>(input_ids.size());
  // Three position rows, filled left-to-right in prompt order.
  std::array<std::vector<int32_t>, 3> rows;
  for (auto& r : rows) r.reserve(static_cast<size_t>(T));

  // Sort image spans by prompt offset (mirror sorted(mm_features, key=offset)).
  std::vector<MmImageSpan> images = images_in;
  std::sort(images.begin(), images.end(),
            [](const MmImageSpan& a, const MmImageSpan& b) { return a.offset < b.offset; });

  int64_t st = 0;
  int64_t last_max = -1;  // running max of emitted positions (-1 => none yet)

  auto append_text = [&](int64_t text_len) {
    if (text_len <= 0) return;
    const int64_t st_idx = last_max + 1;
    for (int64_t i = 0; i < text_len; ++i) {
      const int32_t p = static_cast<int32_t>(st_idx + i);
      rows[0].push_back(p);
      rows[1].push_back(p);
      rows[2].push_back(p);
    }
    last_max = st_idx + text_len - 1;
  };

  for (const MmImageSpan& img : images) {
    const int64_t t = img.grid_thw[0];
    const int64_t h = img.grid_thw[1];
    const int64_t w = img.grid_thw[2];
    if (t != 1) {
      throw std::invalid_argument("Qwen3VLGetRopeIndex: only single-frame images "
                                  "(t==1) supported (video is M3)");
    }
    const int64_t llm_grid_h = h / spatial_merge_size;
    const int64_t llm_grid_w = w / spatial_merge_size;
    const int64_t num = llm_grid_h * llm_grid_w;  // t == 1

    const int64_t text_len = img.offset - st;
    if (text_len < 0) {
      throw std::invalid_argument("Qwen3VLGetRopeIndex: overlapping image spans");
    }
    const int64_t st_idx = last_max + 1;
    append_text(text_len);

    // Image grid positions: np.indices((1,H,W)).reshape(3,-1) + text_len + st_idx.
    // Token j (0..num-1): t_idx=0, h_idx=j/llm_grid_w, w_idx=j%llm_grid_w.
    const int64_t bias = text_len + st_idx;
    int64_t block_max = bias;  // row0 (t) values are all `bias`
    for (int64_t j = 0; j < num; ++j) {
      const int64_t h_idx = j / llm_grid_w;
      const int64_t w_idx = j % llm_grid_w;
      rows[0].push_back(static_cast<int32_t>(bias));
      rows[1].push_back(static_cast<int32_t>(bias + h_idx));
      rows[2].push_back(static_cast<int32_t>(bias + w_idx));
      block_max = std::max(block_max, bias + std::max(h_idx, w_idx));
    }
    last_max = block_max;
    st = img.offset + num;
  }

  // Trailing text after the last image.
  if (st < T) append_text(T - st);

  if (static_cast<int64_t>(rows[0].size()) != T) {
    throw std::invalid_argument("Qwen3VLGetRopeIndex: emitted position count != T "
                                "(image offsets/grids inconsistent with input_ids)");
  }

  std::vector<int32_t> out(static_cast<size_t>(3 * T));
  for (int r = 0; r < 3; ++r)
    std::copy(rows[r].begin(), rows[r].end(), out.begin() + static_cast<size_t>(r) * T);

  if (mrope_position_delta != nullptr) {
    int32_t mx = 0;
    for (int32_t v : out) mx = std::max(mx, v);
    *mrope_position_delta = static_cast<int64_t>(mx) + 1 - T;
  }
  return out;
}

std::vector<int32_t> Qwen3VLGetRopeIndexVideo(
    const std::vector<int32_t>& input_ids, const std::array<int64_t, 3>& video_grid_thw,
    int64_t spatial_merge_size, int32_t vision_start_token_id,
    int32_t video_token_id, int32_t vision_end_token_id,
    int64_t* mrope_position_delta) {
  const int64_t T = static_cast<int64_t>(input_ids.size());
  const int64_t grid_t = video_grid_thw[0];
  const int64_t llm_h = video_grid_thw[1] / spatial_merge_size;
  const int64_t llm_w = video_grid_thw[2] / spatial_merge_size;
  const int64_t expected = llm_h * llm_w;

  std::array<std::vector<int32_t>, 3> rows;
  for (auto& r : rows) r.reserve(static_cast<size_t>(T));
  int64_t st = 0;
  int64_t last_max = -1;  // running max of emitted positions (-1 => none yet)
  bool any = false;

  auto find_from = [&](int32_t tok, int64_t from, int64_t upto) -> int64_t {
    for (int64_t i = from; i < upto; ++i)
      if (input_ids[static_cast<size_t>(i)] == tok) return i;
    return -1;
  };
  auto append_text = [&](int64_t st_idx, int64_t text_len) {
    for (int64_t i = 0; i < text_len; ++i) {
      const int32_t p = static_cast<int32_t>(st_idx + i);
      rows[0].push_back(p); rows[1].push_back(p); rows[2].push_back(p);
    }
    if (text_len > 0) last_max = std::max(last_max, st_idx + text_len - 1);
  };

  int64_t search = 0;
  for (int64_t f = 0; f < grid_t; ++f) {
    const int64_t vs = find_from(vision_start_token_id, search, T);
    if (vs < 0)
      throw std::invalid_argument("Qwen3VLGetRopeIndexVideo: missing vision_start for frame");
    const int64_t ve = find_from(vision_end_token_id, vs, T);
    if (ve < 0)
      throw std::invalid_argument("Qwen3VLGetRopeIndexVideo: missing vision_end for frame");
    const int64_t vid = find_from(video_token_id, vs, ve);
    const int64_t actual = (vid >= 0) ? (ve - vid) : 0;
    const int64_t video_offset = (vid >= 0) ? vid : (vs + 1);
    if (actual == 0) {  // EVS 0-token frame (not exercised by the gate)
      search = ve + 1;
      continue;
    }
    if (actual != expected)
      throw std::invalid_argument("Qwen3VLGetRopeIndexVideo: EVS/lumped frames "
                                  "(actual != expected tokens) not supported by the gate");
    const int64_t text_len = video_offset - st;
    if (text_len < 0)
      throw std::invalid_argument("Qwen3VLGetRopeIndexVideo: negative text_len (ordering)");
    const int64_t st_idx = any ? last_max + 1 : 0;
    append_text(st_idx, text_len);
    const int64_t bias = text_len + st_idx;
    for (int64_t j = 0; j < expected; ++j) {
      const int64_t h_idx = j / llm_w, w_idx = j % llm_w;
      rows[0].push_back(static_cast<int32_t>(bias));
      rows[1].push_back(static_cast<int32_t>(bias + h_idx));
      rows[2].push_back(static_cast<int32_t>(bias + w_idx));
    }
    last_max = std::max(last_max, bias + std::max(llm_h - 1, llm_w - 1));
    st = video_offset + actual;
    search = ve + 1;
    any = true;
  }
  if (st < T) {
    const int64_t st_idx = any ? last_max + 1 : 0;
    append_text(st_idx, T - st);
  }

  if (static_cast<int64_t>(rows[0].size()) != T)
    throw std::invalid_argument("Qwen3VLGetRopeIndexVideo: emitted count != T");

  std::vector<int32_t> out(static_cast<size_t>(3 * T));
  for (int r = 0; r < 3; ++r)
    std::copy(rows[r].begin(), rows[r].end(), out.begin() + static_cast<size_t>(r) * T);
  if (mrope_position_delta != nullptr) {
    int32_t mx = 0;
    for (int32_t v : out) mx = std::max(mx, v);
    *mrope_position_delta = static_cast<int64_t>(mx) + 1 - T;
  }
  return out;
}

void Qwen3VLMergeMultimodal(std::vector<float>& inputs_embeds, int64_t T, int64_t H,
                            const std::vector<float>& mm_embeds,
                            const std::vector<bool>& mask) {
  if (static_cast<int64_t>(inputs_embeds.size()) != T * H)
    throw std::invalid_argument("Qwen3VLMergeMultimodal: inputs_embeds size != T*H");
  if (static_cast<int64_t>(mask.size()) != T)
    throw std::invalid_argument("Qwen3VLMergeMultimodal: mask size != T");
  int64_t k = 0;
  const int64_t N = static_cast<int64_t>(mm_embeds.size()) / (H > 0 ? H : 1);
  for (int64_t t = 0; t < T; ++t) {
    if (!mask[static_cast<size_t>(t)]) continue;
    if (k >= N)
      throw std::invalid_argument("Qwen3VLMergeMultimodal: more visual tokens than "
                                  "mm_embeds rows");
    std::copy(mm_embeds.begin() + static_cast<size_t>(k * H),
              mm_embeds.begin() + static_cast<size_t>((k + 1) * H),
              inputs_embeds.begin() + static_cast<size_t>(t * H));
    ++k;
  }
  if (k != N)
    throw std::invalid_argument("Qwen3VLMergeMultimodal: mm_embeds rows != mask "
                                "true count");
}

std::vector<float> Qwen3VLComputeDeepstack(const std::vector<float>& multiscale,
                                           int64_t N, int64_t L, int64_t H,
                                           const std::vector<bool>& mask, int64_t T) {
  if (static_cast<int64_t>(multiscale.size()) != N * L * H)
    throw std::invalid_argument("Qwen3VLComputeDeepstack: multiscale size != N*L*H");
  if (static_cast<int64_t>(mask.size()) != T)
    throw std::invalid_argument("Qwen3VLComputeDeepstack: mask size != T");
  std::vector<float> out(static_cast<size_t>(L * T * H), 0.0f);  // [L, T, H]
  int64_t k = 0;
  for (int64_t t = 0; t < T; ++t) {
    if (!mask[static_cast<size_t>(t)]) continue;
    if (k >= N)
      throw std::invalid_argument("Qwen3VLComputeDeepstack: more visual tokens than "
                                  "multiscale rows");
    for (int64_t l = 0; l < L; ++l) {
      // out[l, t, :] = multiscale[k, l*H : (l+1)*H]
      const size_t src = static_cast<size_t>(k * (L * H) + l * H);
      const size_t dst = static_cast<size_t>(l * (T * H) + t * H);
      std::copy(multiscale.begin() + src, multiscale.begin() + src + static_cast<size_t>(H),
                out.begin() + dst);
    }
    ++k;
  }
  if (k != N)
    throw std::invalid_argument("Qwen3VLComputeDeepstack: multiscale rows != mask "
                                "true count");
  return out;
}

}  // namespace vllm::multimodal
