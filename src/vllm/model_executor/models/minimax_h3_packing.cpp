// MiniMax-H3 packed-token and packed-sequence layout math.
// Port of vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/
// packed_tokens.py (patchify/unpatchify/audio pack) and packed_sequence.py
// (the fl2va and ref2va packed layouts).
//
// EXACTNESS NOTE. The position grid is FP64 and feeds RoPE directly, so this port
// reproduces upstream's arithmetic ORDER, not merely its formulas:
//   * `NumpyLinspaceNoEndpoint` reproduces numpy.linspace(endpoint=False), which
//     evaluates `i * (delta/num) + start` — NOT `start + i*delta/num`.
//   * `TemporalPositionSpan` reproduces numpy's PAIRWISE summation and
//     `VideoTSpan` reproduces Python's SEQUENTIAL `sum()`. Upstream keeps these
//     two deliberately separate because they diverge in the last ulp from n=16
//     onward (packed_sequence.py:101-113); unifying them would silently shift the
//     fl2va last-keyframe anchor against the ref2va origin.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "vt/dtype.h"

namespace vllm {
namespace {

// packed_sequence.py:37-45
constexpr double kInterp = 32.0;
constexpr int64_t kTGroup = 5;
constexpr double kFramePerToken[kTGroup] = {1.0, 4.0, 4.0, 4.0, 4.0};
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr int64_t kSeqAlign = 64;
constexpr int64_t kPatchH = 2;
constexpr int64_t kPatchW = 2;

// numpy.linspace(start, stop, num, endpoint=False): step = (stop-start)/num and
// y[i] = i*step + start, evaluated in that order (numpy/core/function_base.py).
std::vector<double> NumpyLinspaceNoEndpoint(double start, double stop, int64_t num) {
  VT_CHECK(num > 0, "minimax_h3: linspace num must be positive");
  const double step = (stop - start) / static_cast<double>(num);
  std::vector<double> out(static_cast<size_t>(num));
  for (int64_t i = 0; i < num; ++i) {
    out[static_cast<size_t>(i)] = static_cast<double>(i) * step + start;
  }
  return out;
}

// _axis_from_sqrt_area (packed_sequence.py:258-263).
std::vector<double> AxisFromSqrtArea(int64_t dim, int64_t patch, double sqrt_area) {
  const double ratio = static_cast<double>(dim) / sqrt_area;
  const double left = (1.0 - ratio) * 1.0 / 2.0;
  const double right = left + ratio * 1.0;
  std::vector<double> grid = NumpyLinspaceNoEndpoint(left, right, dim / patch);
  for (double& value : grid) value *= kInterp;
  return grid;
}

// numpy's pairwise summation (numpy/core/src/umath/loops_utils.h pairwise_sum_@TYPE@).
double NumpyPairwiseSum(const double* data, int64_t n) {
  if (n < 8) {
    double res = 0.0;
    for (int64_t i = 0; i < n; ++i) res += data[i];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int i = 0; i < 8; ++i) r[i] = data[i];
    const int64_t body = n - (n % 8);
    for (int64_t i = 8; i < body; i += 8) {
      for (int j = 0; j < 8; ++j) r[j] += data[i + j];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (int64_t i = body; i < n; ++i) res += data[i];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return NumpyPairwiseSum(data, n2) + NumpyPairwiseSum(data + n2, n - n2);
}

// _temporal_position_span (packed_sequence.py:101-113): numpy pairwise sum.
double TemporalPositionSpan(int64_t temporal_length) {
  std::vector<double> spans(static_cast<size_t>(temporal_length), kFrameRescale);
  for (int64_t token_index = 0; token_index < kTGroup; ++token_index) {
    for (int64_t i = token_index; i < temporal_length; i += kTGroup) {
      spans[static_cast<size_t>(i)] *= kFramePerToken[token_index];
    }
  }
  return NumpyPairwiseSum(spans.data(), temporal_length);
}

// _video_t_span (packed_sequence.py:274-277): Python sum(), sequential.
double VideoTSpan(int64_t n) {
  double total = 0.0;
  for (int64_t k = 0; k < n; ++k) total += kFrameRescale * kFramePerToken[k % kTGroup];
  return total;
}

// _video_t_grid (packed_sequence.py:266-271): origin + [0, cumsum(spans[:-1])].
std::vector<double> VideoTGrid(int64_t n, double origin) {
  std::vector<double> grid(static_cast<size_t>(n));
  double running = 0.0;
  for (int64_t k = 0; k < n; ++k) {
    grid[static_cast<size_t>(k)] = origin + running;
    running += kFrameRescale * kFramePerToken[k % kTGroup];
  }
  return grid;
}

// The [h, w] coordinate pairs of one frame, meshgrid(h_grid, w_grid, "ij").
std::vector<double> FrameGrid(const std::vector<double>& h_grid, const std::vector<double>& w_grid) {
  std::vector<double> frame(h_grid.size() * w_grid.size() * 2);
  size_t k = 0;
  for (double h : h_grid) {
    for (double w : w_grid) {
      frame[k++] = h;
      frame[k++] = w;
    }
  }
  return frame;
}

void FillRange(std::vector<int64_t>& out, int64_t start, int64_t stop) {
  for (int64_t i = start; i < stop; ++i) out.push_back(i);
}

int64_t AlignUp(int64_t used) { return ((used + kSeqAlign - 1) / kSeqAlign) * kSeqAlign; }

}  // namespace

// ---------------------------------------------------------------------------
// packed_tokens.py
// ---------------------------------------------------------------------------

// minimax_h3_patchify_video_latent (packed_tokens.py:23-41).
// einsum("nctrhpwq->nthwcrpq") over [B,C,t,pt,h,ph,w,pw].
std::vector<float> MiniMaxH3PatchifyVideoLatent(const std::vector<float>& latent, int64_t batch,
                                                int64_t channels, int64_t full_t, int64_t full_h,
                                                int64_t full_w, int64_t patch_t, int64_t patch_h,
                                                int64_t patch_w) {
  VT_CHECK(patch_t > 0 && patch_h > 0 && patch_w > 0, "minimax_h3 patchify: patch_size must be positive");
  VT_CHECK(full_t % patch_t == 0 && full_h % patch_h == 0 && full_w % patch_w == 0,
           "minimax_h3 patchify: latent dims must be divisible by patch_size");
  const int64_t expected = batch * channels * full_t * full_h * full_w;
  VT_CHECK(static_cast<int64_t>(latent.size()) == expected,
           "minimax_h3 patchify: latent size does not match [B,C,T,H,W]");
  const int64_t t = full_t / patch_t, h = full_h / patch_h, w = full_w / patch_w;
  const int64_t row_width = channels * patch_t * patch_h * patch_w;
  std::vector<float> rows(static_cast<size_t>(batch * t * h * w * row_width));
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t ti = 0; ti < t; ++ti) {
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          const int64_t row = ((n * t + ti) * h + hi) * w + wi;
          int64_t k = 0;
          for (int64_t c = 0; c < channels; ++c) {
            for (int64_t r = 0; r < patch_t; ++r) {
              for (int64_t p = 0; p < patch_h; ++p) {
                for (int64_t q = 0; q < patch_w; ++q) {
                  const int64_t src =
                      (((n * channels + c) * full_t + ti * patch_t + r) * full_h + hi * patch_h + p) *
                          full_w +
                      wi * patch_w + q;
                  rows[static_cast<size_t>(row * row_width + k)] = latent[static_cast<size_t>(src)];
                  ++k;
                }
              }
            }
          }
        }
      }
    }
  }
  return rows;
}

// minimax_h3_unpatchify_video_tokens (packed_tokens.py:44-70).
std::vector<float> MiniMaxH3UnpatchifyVideoTokens(const std::vector<float>& rows, int64_t t,
                                                  int64_t h, int64_t w, int64_t channels,
                                                  int64_t patch_t, int64_t patch_h,
                                                  int64_t patch_w) {
  const int64_t row_width = channels * patch_t * patch_h * patch_w;
  const int64_t rows_per_sample = t * h * w;
  VT_CHECK(row_width > 0 && rows_per_sample > 0, "minimax_h3 unpatchify: degenerate shape");
  VT_CHECK(static_cast<int64_t>(rows.size()) % (rows_per_sample * row_width) == 0,
           "minimax_h3 unpatchify: rows not divisible by t*h*w");
  const int64_t batch = static_cast<int64_t>(rows.size()) / (rows_per_sample * row_width);
  const int64_t full_t = t * patch_t, full_h = h * patch_h, full_w = w * patch_w;
  std::vector<float> latent(static_cast<size_t>(batch * channels * full_t * full_h * full_w));
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t ti = 0; ti < t; ++ti) {
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          const int64_t row = ((n * t + ti) * h + hi) * w + wi;
          int64_t k = 0;
          for (int64_t c = 0; c < channels; ++c) {
            for (int64_t r = 0; r < patch_t; ++r) {
              for (int64_t p = 0; p < patch_h; ++p) {
                for (int64_t q = 0; q < patch_w; ++q) {
                  const int64_t dst =
                      (((n * channels + c) * full_t + ti * patch_t + r) * full_h + hi * patch_h + p) *
                          full_w +
                      wi * patch_w + q;
                  latent[static_cast<size_t>(dst)] = rows[static_cast<size_t>(row * row_width + k)];
                  ++k;
                }
              }
            }
          }
        }
      }
    }
  }
  return latent;
}

// minimax_h3_pack_audio_latent (packed_tokens.py:73-85): permute(0,2,1).reshape.
std::vector<float> MiniMaxH3PackAudioLatent(const std::vector<float>& latent, int64_t audio_channel,
                                            int64_t latent_dim, int64_t steps) {
  VT_CHECK(static_cast<int64_t>(latent.size()) == audio_channel * latent_dim * steps,
           "minimax_h3 audio pack: latent size does not match [C,D,T]");
  std::vector<float> rows(static_cast<size_t>(audio_channel * steps * latent_dim));
  for (int64_t c = 0; c < audio_channel; ++c) {
    for (int64_t s = 0; s < steps; ++s) {
      for (int64_t d = 0; d < latent_dim; ++d) {
        rows[static_cast<size_t>((c * steps + s) * latent_dim + d)] =
            latent[static_cast<size_t>((c * latent_dim + d) * steps + s)];
      }
    }
  }
  return rows;
}

// minimax_h3_unpack_audio_tokens (packed_tokens.py:88-106).
std::vector<float> MiniMaxH3UnpackAudioTokens(const std::vector<float>& rows, int64_t audio_t,
                                              int64_t audio_channel, int64_t latent_dim) {
  VT_CHECK(audio_t > 0 && audio_channel > 0, "minimax_h3 audio unpack: dims must be positive");
  VT_CHECK(audio_t % audio_channel == 0,
           "minimax_h3 audio unpack: audio_t must be divisible by audio_channel");
  VT_CHECK(static_cast<int64_t>(rows.size()) == audio_t * latent_dim,
           "minimax_h3 audio unpack: rows size does not match [audio_t, D]");
  const int64_t steps = audio_t / audio_channel;
  std::vector<float> latent(static_cast<size_t>(audio_channel * latent_dim * steps));
  for (int64_t c = 0; c < audio_channel; ++c) {
    for (int64_t s = 0; s < steps; ++s) {
      for (int64_t d = 0; d < latent_dim; ++d) {
        latent[static_cast<size_t>((c * latent_dim + d) * steps + s)] =
            rows[static_cast<size_t>((c * steps + s) * latent_dim + d)];
      }
    }
  }
  return latent;
}

// ---------------------------------------------------------------------------
// packed_sequence.py — fl2va / t2va
// ---------------------------------------------------------------------------

MiniMaxH3PackedSequence BuildMiniMaxH3PackedSequence(
    int64_t text_len, int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t audio_t,
    int64_t audio_channel, bool include_keyframe_cond,
    const std::vector<int64_t>& keyframe_frame_indices, int64_t frame_count) {
  // _keyframe_cond_frame_indices (packed_sequence.py:48-67): the strict fl2va
  // layout only accepts the first, the last, or first+last as keyframe anchors.
  std::vector<int64_t> cond_frames;
  if (include_keyframe_cond) {
    VT_CHECK(!keyframe_frame_indices.empty(),
             "minimax_h3 fl2va: keyframe cond requires keyframe_frame_indices");
    const bool ok = (keyframe_frame_indices == std::vector<int64_t>{0}) ||
                    (keyframe_frame_indices == std::vector<int64_t>{-1}) ||
                    (keyframe_frame_indices == std::vector<int64_t>{0, -1});
    VT_CHECK(ok, "minimax_h3 fl2va: keyframe_frame_indices must be {0}, {-1} or {0,-1}");
    cond_frames = keyframe_frame_indices;
    VT_CHECK(frame_count > 0, "minimax_h3 fl2va: frame_count required with keyframes");
  } else {
    VT_CHECK(keyframe_frame_indices.empty(),
             "minimax_h3 fl2va: keyframe_frame_indices must be empty without keyframe cond");
  }

  const int64_t ph = latent_h / kPatchH, pw = latent_w / kPatchW;
  const int64_t frame_rows = ph * pw;
  const int64_t cond_rows = static_cast<int64_t>(cond_frames.size()) * frame_rows;
  const int64_t video_rows = latent_t * frame_rows;
  const int64_t audio_rows = audio_t * audio_channel;
  const int64_t used = text_len + cond_rows + audio_rows + video_rows;

  MiniMaxH3PackedSequence out;
  out.seq_len = AlignUp(used);
  const int64_t seq_len = out.seq_len;

  const int64_t text_begin = 0, text_end = text_len;
  const int64_t cond_begin = text_end, cond_end = cond_begin + cond_rows;
  const int64_t audio_begin = cond_end, audio_end = audio_begin + audio_rows;
  const int64_t video_begin = audio_end, video_end = video_begin + video_rows;

  out.input_ids.assign(static_cast<size_t>(seq_len), kMiniMaxH3PadId);
  for (int64_t i = text_begin; i < text_end; ++i) out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3TextId;
  for (int64_t i = cond_begin; i < cond_end; ++i)
    out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3ImgVidCondId;
  for (int64_t i = audio_begin; i < audio_end; ++i)
    out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3AudioId;
  if (audio_rows > 0) out.input_ids[static_cast<size_t>(audio_begin)] = kMiniMaxH3AudioFirstId;
  for (int64_t i = video_begin; i < video_end; ++i)
    out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3VideoId;
  if (video_rows > 0) {
    out.input_ids[static_cast<size_t>(video_begin)] = kMiniMaxH3VideoFirstId;
    out.input_ids[static_cast<size_t>(video_end - 1)] = kMiniMaxH3VideoLastId;
  }

  out.image_mask.assign(static_cast<size_t>(seq_len), 0);
  out.audio_mask.assign(static_cast<size_t>(seq_len), 0);
  for (int64_t i = cond_begin; i < cond_end; ++i) out.image_mask[static_cast<size_t>(i)] = 1;
  for (int64_t i = video_begin; i < video_end; ++i) out.image_mask[static_cast<size_t>(i)] = 1;
  for (int64_t i = audio_begin; i < audio_end; ++i) out.audio_mask[static_cast<size_t>(i)] = 1;

  FillRange(out.img_pos, cond_begin, cond_end);
  FillRange(out.img_pos, video_begin, video_end);
  out.update_mask.assign(out.img_pos.size(), 0);
  for (size_t i = static_cast<size_t>(cond_rows); i < out.update_mask.size(); ++i)
    out.update_mask[i] = 1;
  FillRange(out.audio_pos, audio_begin, audio_end);
  FillRange(out.text_pos, 0, text_len);

  // --- fp64 position grid (packed_sequence.py:181-216) ---
  out.img_position_ids.assign(static_cast<size_t>(seq_len) * 3, 0.0);
  for (int64_t i = 0; i < text_len; ++i) {
    out.img_position_ids[static_cast<size_t>(i) * 3] = static_cast<double>(i);
  }

  const std::vector<double> t_grid = VideoTGrid(latent_t, static_cast<double>(text_len));
  const double sqrt_area = std::sqrt(static_cast<double>(latent_h) * static_cast<double>(latent_w));
  const std::vector<double> h_grid = AxisFromSqrtArea(latent_h, kPatchH, sqrt_area);
  const std::vector<double> w_grid = AxisFromSqrtArea(latent_w, kPatchW, sqrt_area);
  const std::vector<double> frame = FrameGrid(h_grid, w_grid);

  for (size_t block_index = 0; block_index < cond_frames.size(); ++block_index) {
    // _resolve_keyframe_frame_indices (packed_sequence.py:70-98) then the anchor
    // rule at :193-207: only the first and last frames are valid anchors.
    const int64_t semantic = cond_frames[block_index];
    const int64_t resolved = semantic == -1 ? frame_count - 1 : semantic;
    double cond_t = 0.0;
    if (resolved == 0) {
      cond_t = static_cast<double>(text_len);
    } else if (resolved == frame_count - 1) {
      cond_t = static_cast<double>(text_len) + TemporalPositionSpan(latent_t) - kFrameRescale;
    } else {
      VT_CHECK(false, "minimax_h3 fl2va: only first/last keyframe anchors are supported");
    }
    const int64_t begin = cond_begin + static_cast<int64_t>(block_index) * frame_rows;
    for (int64_t r = 0; r < frame_rows; ++r) {
      const size_t base = static_cast<size_t>(begin + r) * 3;
      out.img_position_ids[base] = cond_t;
      out.img_position_ids[base + 1] = frame[static_cast<size_t>(r) * 2];
      out.img_position_ids[base + 2] = frame[static_cast<size_t>(r) * 2 + 1];
    }
  }

  for (int64_t ti = 0; ti < latent_t; ++ti) {
    for (int64_t r = 0; r < frame_rows; ++r) {
      const size_t base = static_cast<size_t>(video_begin + ti * frame_rows + r) * 3;
      out.img_position_ids[base] = t_grid[static_cast<size_t>(ti)];
      out.img_position_ids[base + 1] = frame[static_cast<size_t>(r) * 2];
      out.img_position_ids[base + 2] = frame[static_cast<size_t>(r) * 2 + 1];
    }
  }

  // Audio rows are channel-major over the same t counter and are pinned to the
  // extremes of the w grid (packed_sequence.py:209-216).
  for (int64_t c = 0; c < audio_channel; ++c) {
    for (int64_t s = 0; s < audio_t; ++s) {
      const size_t base = static_cast<size_t>(audio_begin + c * audio_t + s) * 3;
      out.img_position_ids[base] = static_cast<double>(text_len) + static_cast<double>(s);
    }
  }
  for (int64_t i = 0; i < audio_rows; ++i) {
    const size_t base = static_cast<size_t>(audio_begin + i) * 3;
    out.img_position_ids[base + 2] = i < audio_t ? w_grid.front() : w_grid.back();
  }

  out.token_tags.assign(static_cast<size_t>(seq_len), kMiniMaxH3TagPadding);
  for (int64_t i = text_begin; i < text_end; ++i)
    out.token_tags[static_cast<size_t>(i)] = kMiniMaxH3TagText;
  for (int64_t i = audio_begin; i < audio_end; ++i)
    out.token_tags[static_cast<size_t>(i)] = kMiniMaxH3TagAudio;
  for (int64_t pos : out.img_pos) out.token_tags[static_cast<size_t>(pos)] = kMiniMaxH3TagVideo;

  out.cu_seqlens = {0, static_cast<int32_t>(used), static_cast<int32_t>(seq_len)};
  out.document_id.assign(static_cast<size_t>(seq_len), 0);
  for (int64_t i = video_end; i < seq_len; ++i) out.document_id[static_cast<size_t>(i)] = 1;
  return out;
}

// ---------------------------------------------------------------------------
// packed_sequence.py — ref2va block family
// ---------------------------------------------------------------------------

MiniMaxH3PackedSequence BuildMiniMaxH3PackedSequenceRef2va(
    int64_t text_len, int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t audio_t,
    const std::vector<MiniMaxH3RefBlock>& ref_blocks, int64_t audio_channel) {
  using Kind = MiniMaxH3RefBlock::Kind;

  int64_t ref_visual_rows = 0, ref_audio_rows = 0;
  for (const MiniMaxH3RefBlock& block : ref_blocks) {
    if (block.kind == Kind::kImage) {
      VT_CHECK(block.latent_h > 0 && block.latent_w > 0, "minimax_h3 ref2va: image block dims");
      ref_visual_rows += (block.latent_h / kPatchH) * (block.latent_w / kPatchW);
    } else if (block.kind == Kind::kAudio) {
      VT_CHECK(block.ref_audio_t >= 0, "minimax_h3 ref2va: audio block ref_audio_t");
      ref_audio_rows += block.ref_audio_t * audio_channel;
    } else {
      VT_CHECK(block.latent_t > 0 && block.latent_h > 0 && block.latent_w > 0,
               "minimax_h3 ref2va: video block dims");
      ref_audio_rows += block.ref_audio_t * audio_channel;
      ref_visual_rows += block.latent_t * (block.latent_h / kPatchH) * (block.latent_w / kPatchW);
    }
  }

  const int64_t ph = latent_h / kPatchH, pw = latent_w / kPatchW;
  const int64_t frame_rows = ph * pw;
  const int64_t video_rows = latent_t * frame_rows;
  const int64_t audio_rows = audio_t * audio_channel;
  const int64_t used = text_len + ref_visual_rows + ref_audio_rows + audio_rows + video_rows;

  MiniMaxH3PackedSequence out;
  out.seq_len = AlignUp(used);
  const int64_t seq_len = out.seq_len;

  out.input_ids.assign(static_cast<size_t>(seq_len), kMiniMaxH3PadId);
  out.image_mask.assign(static_cast<size_t>(seq_len), 0);
  out.audio_mask.assign(static_cast<size_t>(seq_len), 0);
  out.img_position_ids.assign(static_cast<size_t>(seq_len) * 3, 0.0);
  out.token_tags.assign(static_cast<size_t>(seq_len), kMiniMaxH3TagPadding);

  for (int64_t i = 0; i < text_len; ++i) {
    out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3TextId;
    out.token_tags[static_cast<size_t>(i)] = kMiniMaxH3TagText;
    out.img_position_ids[static_cast<size_t>(i) * 3] = static_cast<double>(i);
  }
  FillRange(out.text_pos, 0, text_len);

  const double target_area = std::sqrt(static_cast<double>(latent_h) * static_cast<double>(latent_w));
  const std::vector<double> h_grid = AxisFromSqrtArea(latent_h, kPatchH, target_area);
  const std::vector<double> w_grid = AxisFromSqrtArea(latent_w, kPatchW, target_area);
  const std::vector<double> target_frame = FrameGrid(h_grid, w_grid);

  // Reference blocks are consumed in request order; each advances the temporal
  // cursor by its own span (packed_sequence.py:416-498).
  std::vector<int64_t> ref_img_pos, ref_audio_pos;
  int64_t cursor = text_len;
  double t_cursor = static_cast<double>(text_len);
  for (const MiniMaxH3RefBlock& block : ref_blocks) {
    if (block.kind == Kind::kImage) {
      const int64_t rows = (block.latent_h / kPatchH) * (block.latent_w / kPatchW);
      const int64_t begin = cursor, end = cursor + rows;
      cursor = end;
      const double area =
          std::sqrt(static_cast<double>(block.latent_h) * static_cast<double>(block.latent_w));
      const std::vector<double> rh = AxisFromSqrtArea(block.latent_h, kPatchH, area);
      const std::vector<double> rw = AxisFromSqrtArea(block.latent_w, kPatchW, area);
      const std::vector<double> ref_frame = FrameGrid(rh, rw);
      for (int64_t i = begin; i < end; ++i) {
        out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3ImgVidCondId;
        out.image_mask[static_cast<size_t>(i)] = 1;
        const size_t base = static_cast<size_t>(i) * 3;
        out.img_position_ids[base] = t_cursor;
        out.img_position_ids[base + 1] = ref_frame[static_cast<size_t>(i - begin) * 2];
        out.img_position_ids[base + 2] = ref_frame[static_cast<size_t>(i - begin) * 2 + 1];
      }
      FillRange(ref_img_pos, begin, end);
      t_cursor += 1.0;
    } else if (block.kind == Kind::kAudio) {
      const int64_t rows = block.ref_audio_t * audio_channel;
      const int64_t begin = cursor, end = cursor + rows;
      cursor = end;
      for (int64_t i = begin; i < end; ++i) {
        out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3AudioRefCondId;
        out.audio_mask[static_cast<size_t>(i)] = 1;
      }
      for (int64_t c = 0; c < audio_channel; ++c) {
        for (int64_t s = 0; s < block.ref_audio_t; ++s) {
          out.img_position_ids[static_cast<size_t>(begin + c * block.ref_audio_t + s) * 3] =
              t_cursor + static_cast<double>(s);
        }
      }
      for (int64_t i = 0; i < rows; ++i) {
        out.img_position_ids[static_cast<size_t>(begin + i) * 3 + 2] =
            i < block.ref_audio_t ? w_grid.front() : w_grid.back();
      }
      FillRange(ref_audio_pos, begin, end);
      t_cursor += static_cast<double>(block.ref_audio_t);
    } else {
      // A video-bearing block packs its audio rows immediately before its video
      // rows; both share a temporal origin and advance by the LONGER span.
      const int64_t block_frame_rows = (block.latent_h / kPatchH) * (block.latent_w / kPatchW);
      const int64_t a_rows = block.ref_audio_t * audio_channel;
      const int64_t v_rows = block.latent_t * block_frame_rows;
      const int64_t a_begin = cursor, a_end = cursor + a_rows;
      const int64_t v_begin = a_end, v_end = a_end + v_rows;
      cursor = v_end;

      const double ref_area =
          std::sqrt(static_cast<double>(block.latent_h) * static_cast<double>(block.latent_w));
      const std::vector<double> rv_h = AxisFromSqrtArea(block.latent_h, kPatchH, ref_area);
      const std::vector<double> rv_w = AxisFromSqrtArea(block.latent_w, kPatchW, ref_area);
      const std::vector<double> rv_frame = FrameGrid(rv_h, rv_w);

      for (int64_t i = a_begin; i < a_end; ++i) {
        out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3AudioRefCondId;
        out.audio_mask[static_cast<size_t>(i)] = 1;
      }
      for (int64_t c = 0; c < audio_channel; ++c) {
        for (int64_t s = 0; s < block.ref_audio_t; ++s) {
          out.img_position_ids[static_cast<size_t>(a_begin + c * block.ref_audio_t + s) * 3] =
              t_cursor + static_cast<double>(s);
        }
      }
      for (int64_t i = 0; i < a_rows; ++i) {
        out.img_position_ids[static_cast<size_t>(a_begin + i) * 3 + 2] =
            i < block.ref_audio_t ? rv_w.front() : rv_w.back();
      }
      FillRange(ref_audio_pos, a_begin, a_end);

      const std::vector<double> rv_t_grid = VideoTGrid(block.latent_t, t_cursor);
      for (int64_t i = v_begin; i < v_end; ++i) {
        out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3ImgVidCondId;
        out.image_mask[static_cast<size_t>(i)] = 1;
      }
      for (int64_t ti = 0; ti < block.latent_t; ++ti) {
        for (int64_t r = 0; r < block_frame_rows; ++r) {
          const size_t base = static_cast<size_t>(v_begin + ti * block_frame_rows + r) * 3;
          out.img_position_ids[base] = rv_t_grid[static_cast<size_t>(ti)];
          out.img_position_ids[base + 1] = rv_frame[static_cast<size_t>(r) * 2];
          out.img_position_ids[base + 2] = rv_frame[static_cast<size_t>(r) * 2 + 1];
        }
      }
      FillRange(ref_img_pos, v_begin, v_end);
      t_cursor += std::max(static_cast<double>(block.ref_audio_t), VideoTSpan(block.latent_t));
    }
  }

  const int64_t audio_begin = cursor, audio_end = cursor + audio_rows;
  const int64_t video_begin = audio_end, video_end = audio_end + video_rows;

  for (int64_t i = audio_begin; i < audio_end; ++i) {
    out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3AudioId;
    out.audio_mask[static_cast<size_t>(i)] = 1;
  }
  if (audio_rows > 0) out.input_ids[static_cast<size_t>(audio_begin)] = kMiniMaxH3AudioFirstId;
  for (int64_t i = video_begin; i < video_end; ++i) {
    out.input_ids[static_cast<size_t>(i)] = kMiniMaxH3VideoId;
    out.image_mask[static_cast<size_t>(i)] = 1;
  }
  if (video_rows > 0) {
    out.input_ids[static_cast<size_t>(video_begin)] = kMiniMaxH3VideoFirstId;
    out.input_ids[static_cast<size_t>(video_end - 1)] = kMiniMaxH3VideoLastId;
  }

  for (int64_t c = 0; c < audio_channel; ++c) {
    for (int64_t s = 0; s < audio_t; ++s) {
      out.img_position_ids[static_cast<size_t>(audio_begin + c * audio_t + s) * 3] =
          t_cursor + static_cast<double>(s);
    }
  }
  for (int64_t i = 0; i < audio_rows; ++i) {
    out.img_position_ids[static_cast<size_t>(audio_begin + i) * 3 + 2] =
        i < audio_t ? w_grid.front() : w_grid.back();
  }

  const std::vector<double> t_grid = VideoTGrid(latent_t, t_cursor);
  for (int64_t ti = 0; ti < latent_t; ++ti) {
    for (int64_t r = 0; r < frame_rows; ++r) {
      const size_t base = static_cast<size_t>(video_begin + ti * frame_rows + r) * 3;
      out.img_position_ids[base] = t_grid[static_cast<size_t>(ti)];
      out.img_position_ids[base + 1] = target_frame[static_cast<size_t>(r) * 2];
      out.img_position_ids[base + 2] = target_frame[static_cast<size_t>(r) * 2 + 1];
    }
  }

  out.img_pos = ref_img_pos;
  FillRange(out.img_pos, video_begin, video_end);
  out.audio_pos = ref_audio_pos;
  FillRange(out.audio_pos, audio_begin, audio_end);

  out.update_mask.assign(out.img_pos.size(), 0);
  for (size_t i = static_cast<size_t>(ref_visual_rows); i < out.update_mask.size(); ++i)
    out.update_mask[i] = 1;
  out.audio_update_mask.assign(out.audio_pos.size(), 0);
  for (size_t i = static_cast<size_t>(ref_audio_rows); i < out.audio_update_mask.size(); ++i)
    out.audio_update_mask[i] = 1;

  for (int64_t pos : ref_audio_pos) out.token_tags[static_cast<size_t>(pos)] = kMiniMaxH3TagAudio;
  for (int64_t i = audio_begin; i < audio_end; ++i)
    out.token_tags[static_cast<size_t>(i)] = kMiniMaxH3TagAudio;
  for (int64_t pos : out.img_pos) out.token_tags[static_cast<size_t>(pos)] = kMiniMaxH3TagVideo;

  out.cu_seqlens = {0, static_cast<int32_t>(used), static_cast<int32_t>(seq_len)};
  out.document_id.assign(static_cast<size_t>(seq_len), 0);
  for (int64_t i = video_end; i < seq_len; ++i) out.document_id[static_cast<size_t>(i)] = 1;
  return out;
}

// ---------------------------------------------------------------------------
// scheduling_minimax_h3_euler_ancestral.py
// ---------------------------------------------------------------------------

std::vector<float> MiniMaxH3RfVToX0(const std::vector<float>& xt, const std::vector<float>& v,
                                    double timestep) {
  VT_CHECK(xt.size() == v.size(), "minimax_h3 rf_v_to_x0: xt and v must match");
  VT_CHECK(timestep >= 0.0 && timestep <= 1.0, "minimax_h3 rf_v_to_x0: timestep must be in [0,1]");
  // x0 = xt + (1 - t) * v, evaluated at the state dtype (f32 here).
  const float sigma = static_cast<float>(1.0 - timestep);
  std::vector<float> out(xt.size());
  for (size_t i = 0; i < xt.size(); ++i) out[i] = xt[i] + sigma * v[i];
  return out;
}

std::vector<float> MiniMaxH3EulerEta0Step(const std::vector<float>& state,
                                          const std::vector<float>& denoised, double sigma_curr,
                                          double sigma_next) {
  VT_CHECK(state.size() == denoised.size(), "minimax_h3 euler step: shapes must match");
  VT_CHECK(sigma_curr >= 0.0 && sigma_next >= 0.0, "minimax_h3 euler step: sigmas must be >= 0");
  if (sigma_curr == 0.0) {
    VT_CHECK(sigma_next == 0.0, "minimax_h3 euler step: sigma_next must be 0 when sigma_curr is 0");
    return state;
  }
  // f32 state keeps compute_dtype == f32 (scheduling:93-100).
  const float ratio = static_cast<float>(sigma_next) / static_cast<float>(sigma_curr);
  std::vector<float> out(state.size());
  for (size_t i = 0; i < state.size(); ++i) out[i] = ratio * state[i] + (1.0f - ratio) * denoised[i];
  return out;
}

// ---------------------------------------------------------------------------
// Condition-noise augmentation (condition_noise.py)
//
// fl2va/ref2va pin their keyframe (and reference-audio) rows to a NOISED anchor
// rather than the clean latent: `out = t*clean + (1-t)*noise` with t the request's
// `noise_aug`. The mix is trivial; what is easy to get wrong is the ROW ACCOUNTING
// around it, which is what these functions own and what the gate pins:
//   * each visual condition draws noise of length
//     `target_latent_t + imgvid_cond_num_frames`, then slices the PREFIX matching
//     its own latent_t (so a shorter condition does NOT get a shorter draw);
//   * every condition restarts the SAME seed, so concatenating and drawing once
//     would be numerically different for multi-reference requests;
//   * rows advance by that condition's own patchified row count.
//
// NOISE IS AN INPUT here, exactly as in the t2va pipeline: upstream draws it from
// a seeded torch CPU generator, and reproducing torch's RNG bit-exactly decides
// WHICH sample you get rather than whether the accounting is right.
// ---------------------------------------------------------------------------

std::vector<float> MiniMaxH3ImgvidCondNoiseAug(const std::vector<float>& clean_rows,
                                               const std::vector<int64_t>& condition_shapes,
                                               int64_t target_latent_t,
                                               int64_t imgvid_cond_num_frames, double noise_aug,
                                               const std::vector<float>& noise_rows) {
  VT_CHECK(noise_aug >= 0.0 && noise_aug <= 1.0,
           "minimax_h3 cond noise: noise_aug must be in [0, 1]");
  if (noise_aug == 1.0) return clean_rows;  // fully clean: the anchor IS the latent
  VT_CHECK(!condition_shapes.empty() && condition_shapes.size() % 3 == 0,
           "minimax_h3 cond noise: condition_shapes must be (t, h, w) triples");
  VT_CHECK(target_latent_t > 0 && imgvid_cond_num_frames > 0,
           "minimax_h3 cond noise: target_latent_t and imgvid_cond_num_frames must be positive");

  int64_t expected_rows = 0;
  const int64_t num_conditions = static_cast<int64_t>(condition_shapes.size()) / 3;
  for (int64_t c = 0; c < num_conditions; ++c) {
    const int64_t t = condition_shapes[static_cast<size_t>(c * 3 + 0)];
    const int64_t h = condition_shapes[static_cast<size_t>(c * 3 + 1)];
    const int64_t w = condition_shapes[static_cast<size_t>(c * 3 + 2)];
    VT_CHECK(t > 0 && h > 0 && w > 0, "minimax_h3 cond noise: condition shape must be positive");
    VT_CHECK(h % 2 == 0 && w % 2 == 0,
             "minimax_h3 cond noise: condition spatial dims must be divisible by 2");
    VT_CHECK(t <= target_latent_t + imgvid_cond_num_frames,
             "minimax_h3 cond noise: condition latent_t exceeds the noise draw length");
    expected_rows += t * (h / 2) * (w / 2);
  }
  VT_CHECK(static_cast<int64_t>(clean_rows.size()) == expected_rows * kMiniMaxH3VideoRowWidth,
           "minimax_h3 cond noise: clean rows do not match the shape-derived row count");
  VT_CHECK(noise_rows.size() == clean_rows.size(),
           "minimax_h3 cond noise: noise rows must match the clean rows");

  const float t_mix = static_cast<float>(noise_aug);
  std::vector<float> out(clean_rows.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = t_mix * clean_rows[i] + (1.0f - t_mix) * noise_rows[i];
  }
  return out;
}

std::vector<float> MiniMaxH3AudioCondNoiseAug(const std::vector<float>& clean_rows,
                                              const std::vector<int64_t>& condition_audio_t,
                                              double noise_aug,
                                              const std::vector<float>& noise_rows) {
  VT_CHECK(noise_aug >= 0.0 && noise_aug <= 1.0,
           "minimax_h3 cond noise: noise_aug must be in [0, 1]");
  if (noise_aug == 1.0) return clean_rows;
  VT_CHECK(!condition_audio_t.empty(), "minimax_h3 cond noise: condition_audio_t must not be empty");
  int64_t expected_rows = 0;
  for (int64_t audio_t : condition_audio_t) {
    VT_CHECK(audio_t > 0, "minimax_h3 cond noise: condition audio length must be positive");
    expected_rows += kMiniMaxH3AudioCondChannels * audio_t;
  }
  VT_CHECK(static_cast<int64_t>(clean_rows.size()) == expected_rows * kMiniMaxH3AudioRowWidth,
           "minimax_h3 cond noise: clean audio rows do not match the derived row count");
  VT_CHECK(noise_rows.size() == clean_rows.size(),
           "minimax_h3 cond noise: noise rows must match the clean rows");

  const float t_mix = static_cast<float>(noise_aug);
  std::vector<float> out(clean_rows.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = t_mix * clean_rows[i] + (1.0f - t_mix) * noise_rows[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Presentation token tags (presentation.py:42-62, 92-129)
//
// The prompt presentation interleaves TEXT spans with VISION blocks, and the AdaLN
// token tags must line up with it. This is the fl2va "vision-span override" the
// denoise loop requires callers to have applied to `token_tags` before the forward.
//
// THE DETAIL THAT MATTERS: a vision block is
// `<|vision_start|> + pad*count + <|vision_end|>`, and the WHOLE block — markers
// included — is tagged VIDEO. Tagging only the pads leaves the two markers as TEXT
// and shifts every AdaLN modulation index after them.
//
// Tokenization stays with the caller (it owns the tokenizer); this takes SPAN
// LENGTHS, which is the part that has to agree with the packed layout.
// ---------------------------------------------------------------------------

int64_t MiniMaxH3VisionBlockTokenLength(int64_t pad_count) {
  VT_CHECK(pad_count > 0, "minimax_h3 presentation: vision pad count must be positive");
  return pad_count + 2;  // the <|vision_start|> / <|vision_end|> markers
}

std::vector<int64_t> MiniMaxH3BuildPresentationTokenTags(
    const std::vector<MiniMaxH3PresentationSpan>& spans) {
  VT_CHECK(!spans.empty(), "minimax_h3 presentation: spans must not be empty");
  std::vector<int64_t> tags;
  for (const MiniMaxH3PresentationSpan& span : spans) {
    VT_CHECK(span.length > 0, "minimax_h3 presentation: span length must be positive");
    const int64_t tag =
        span.kind == MiniMaxH3PresentationSpan::Kind::kVision ? kMiniMaxH3TagVideo : kMiniMaxH3TagText;
    tags.insert(tags.end(), static_cast<size_t>(span.length), tag);
  }
  return tags;
}

}  // namespace vllm
