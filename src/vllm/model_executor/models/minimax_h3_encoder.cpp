// MiniMax-H3 ENCODER text tower — a Qwen3-VL with three H3-specific deltas.
//
// The H3-Encoder produces the `[seq, 5120]` `prompt_embeds` the DiT consumes. It
// is a Qwen3-VL, which this project already ports (qwen3_vl_{text,vision}.cpp), so
// the ARCHITECTURE is reuse; what is H3-specific — and what this file pins down and
// gates — are the three deltas (upstream encoder.py:1-30):
//
//   1. LAYER TRUNCATION. Only the first `MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER`
//      (50) decoder layers are kept: `num_layers = min(config.num_hidden_layers,
//      selected_layer)`.
//   2. UNNORMALIZED OUTPUT. The checkpoint consumes the hidden state straight out
//      of layer 49 — there is NO final RMSNorm, unlike a stock Qwen3-VL text model.
//      Applying one silently shifts every conditioning vector.
//   3. DEEPSTACK. Visual features are ADDED at the visual token positions after
//      each of the first `len(deepstack_visual_embeds)` layers.
//
// The layer itself is the familiar pre-norm block: RMSNorm -> fused-QKV attention
// with per-head q/k RMSNorm and interleaved M-RoPE -> causal GQA SDPA -> o_proj ->
// residual; RMSNorm -> gated-SiLU MLP -> residual.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

void RmsNormRowsEnc(const float* in, const float* weight, float* out, int64_t rows, int64_t width,
                    double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(src[i]) * src[i];
    const double inv = 1.0 / std::sqrt(sum / static_cast<double>(width) + eps);
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] = static_cast<float>(src[i] * inv * weight[i]);
  }
}

float SiluEnc(float x) { return x / (1.0f + std::exp(-x)); }

// y = x @ W^T (no bias; Qwen3-VL text projections are bias-free), W [out, in],
// optionally reading a ROW SLICE of a larger fused weight.
std::vector<float> LinearSlice(const std::vector<float>& x, int64_t rows, int64_t in_features,
                               const std::vector<float>& weight, int64_t row_offset,
                               int64_t out_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      double acc = 0.0;
      const float* w = weight.data() + (row_offset + o) * in_features;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_features + i)]) *
               static_cast<double>(w[i]);
      }
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

// Interleaved M-RoPE (encoder.py:292-300 + 319-329). `positions` is [3, seq]
// (temporal, height, width). Produces cos/sin of [seq, head_dim].
void MiniMaxH3EncoderMrope(const int64_t* positions, int64_t seq, int64_t head_dim,
                           double rope_theta, const std::vector<int64_t>& mrope_section,
                           std::vector<float>* cos_out, std::vector<float>* sin_out) {
  VT_CHECK(head_dim % 2 == 0, "minimax_h3 encoder: head_dim must be even");
  VT_CHECK(mrope_section.size() == 3, "minimax_h3 encoder: mrope_section must have 3 entries");
  const int64_t half = head_dim / 2;

  cos_out->assign(static_cast<size_t>(seq * head_dim), 0.0f);
  sin_out->assign(static_cast<size_t>(seq * head_dim), 0.0f);
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t i = 0; i < half; ++i) {
      // The frequency slot i takes its position from the TEMPORAL axis by
      // default; slots at (1, 4, 7, ...) below 3*section[1] come from HEIGHT and
      // slots at (2, 5, 8, ...) below 3*section[2] come from WIDTH. That is the
      // "chunked [TTT HHH WWW] -> interleaved [THW THW ...]" reshuffle.
      int64_t axis = 0;
      if (i % 3 == 1 && i < 3 * mrope_section[1]) {
        axis = 1;
      } else if (i % 3 == 2 && i < 3 * mrope_section[2]) {
        axis = 2;
      }
      const double inv_freq =
          1.0 / std::pow(rope_theta, static_cast<double>(2 * i) / static_cast<double>(head_dim));
      const double angle = static_cast<double>(positions[axis * seq + s]) * inv_freq;
      const float c = static_cast<float>(std::cos(angle));
      const float sn = static_cast<float>(std::sin(angle));
      // emb = cat(freqs, freqs) => the second half repeats the first.
      (*cos_out)[static_cast<size_t>(s * head_dim + i)] = c;
      (*cos_out)[static_cast<size_t>(s * head_dim + half + i)] = c;
      (*sin_out)[static_cast<size_t>(s * head_dim + i)] = sn;
      (*sin_out)[static_cast<size_t>(s * head_dim + half + i)] = sn;
    }
  }
}

// The H3 layer budget: min(config.num_hidden_layers, selected_layer).
int64_t MiniMaxH3EncoderNumLayers(int64_t config_num_hidden_layers, int64_t selected_layer) {
  VT_CHECK(config_num_hidden_layers > 0 && selected_layer > 0,
           "minimax_h3 encoder: layer counts must be positive");
  return std::min(config_num_hidden_layers, selected_layer);
}

std::vector<float> MiniMaxH3EncoderTextForward(const MiniMaxH3EncoderConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::vector<float>& inputs_embeds,
                                               const int64_t* positions, int64_t seq,
                                               const uint8_t* visual_pos_mask,
                                               const std::vector<std::vector<float>>& deepstack) {
  const int64_t hidden = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t head_dim = config.head_dim;
  const int64_t q_width = heads * head_dim;
  const int64_t kv_width = kv_heads * head_dim;
  VT_CHECK(static_cast<int64_t>(inputs_embeds.size()) == seq * hidden,
           "minimax_h3 encoder: inputs_embeds size does not match [seq, hidden]");
  VT_CHECK(heads % kv_heads == 0, "minimax_h3 encoder: heads must be a multiple of kv_heads");
  const int64_t groups = heads / kv_heads;

  std::vector<float> cos, sin;
  MiniMaxH3EncoderMrope(positions, seq, head_dim, config.rope_theta, config.mrope_section, &cos,
                        &sin);

  const int64_t num_layers =
      MiniMaxH3EncoderNumLayers(config.num_hidden_layers, config.selected_layer);
  std::vector<float> h = inputs_embeds;
  std::vector<float> normed(h.size());

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + ".";
    RmsNormRowsEnc(h.data(), weights.Get(p + "input_layernorm.weight").data(), normed.data(), seq,
                   hidden, config.rms_norm_eps);

    // Fused qkv weight rows are [q_all | k_all | v_all].
    const std::vector<float>& qkv_w = weights.Get(p + "self_attn.qkv_proj.weight");
    std::vector<float> q = LinearSlice(normed, seq, hidden, qkv_w, 0, q_width);
    std::vector<float> k = LinearSlice(normed, seq, hidden, qkv_w, q_width, kv_width);
    const std::vector<float> v = LinearSlice(normed, seq, hidden, qkv_w, q_width + kv_width, kv_width);

    // Per-head q/k RMSNorm, THEN RoPE.
    std::vector<float> qn(q.size()), kn(k.size());
    RmsNormRowsEnc(q.data(), weights.Get(p + "self_attn.q_norm.weight").data(), qn.data(),
                   seq * heads, head_dim, config.rms_norm_eps);
    RmsNormRowsEnc(k.data(), weights.Get(p + "self_attn.k_norm.weight").data(), kn.data(),
                   seq * kv_heads, head_dim, config.rms_norm_eps);
    const int64_t rot_half = head_dim / 2;
    auto apply_rope = [&](std::vector<float>& x, int64_t n_heads) {
      for (int64_t s = 0; s < seq; ++s) {
        const float* c = cos.data() + s * head_dim;
        const float* sn = sin.data() + s * head_dim;
        for (int64_t head = 0; head < n_heads; ++head) {
          float* row = x.data() + (s * n_heads + head) * head_dim;
          for (int64_t i = 0; i < rot_half; ++i) {
            const float lo = row[i], hi = row[i + rot_half];
            row[i] = lo * c[i] - hi * sn[i];
            row[i + rot_half] = hi * c[i + rot_half] + lo * sn[i + rot_half];
          }
        }
      }
    };
    apply_rope(qn, heads);
    apply_rope(kn, kv_heads);

    // CAUSAL GQA attention (is_causal=True upstream).
    std::vector<float> attn(static_cast<size_t>(seq * q_width), 0.0f);
    std::vector<double> probs(static_cast<size_t>(seq));
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int64_t head = 0; head < heads; ++head) {
      const int64_t kv_head = head / groups;
      for (int64_t i = 0; i < seq; ++i) {
        double max_score = -1e30;
        for (int64_t j = 0; j <= i; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(qn[static_cast<size_t>((i * heads + head) * head_dim + d)]) *
                   static_cast<double>(kn[static_cast<size_t>((j * kv_heads + kv_head) * head_dim + d)]);
          }
          probs[static_cast<size_t>(j)] = dot * scale;
          max_score = std::max(max_score, probs[static_cast<size_t>(j)]);
        }
        double denom = 0.0;
        for (int64_t j = 0; j <= i; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(probs[static_cast<size_t>(j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j <= i; ++j) {
            acc += probs[static_cast<size_t>(j)] *
                   static_cast<double>(v[static_cast<size_t>((j * kv_heads + kv_head) * head_dim + d)]);
          }
          attn[static_cast<size_t>((i * heads + head) * head_dim + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }
    const std::vector<float> projected =
        LinearSlice(attn, seq, q_width, weights.Get(p + "self_attn.o_proj.weight"), 0, hidden);
    for (size_t i = 0; i < h.size(); ++i) h[i] += projected[i];

    // gated-SiLU MLP; gate_up_proj rows are [gate | up].
    RmsNormRowsEnc(h.data(), weights.Get(p + "post_attention_layernorm.weight").data(),
                   normed.data(), seq, hidden, config.rms_norm_eps);
    const std::vector<float>& gate_up = weights.Get(p + "mlp.gate_up_proj.weight");
    const std::vector<float> gate = LinearSlice(normed, seq, hidden, gate_up, 0, config.intermediate_size);
    const std::vector<float> up =
        LinearSlice(normed, seq, hidden, gate_up, config.intermediate_size, config.intermediate_size);
    std::vector<float> act(gate.size());
    for (size_t i = 0; i < act.size(); ++i) act[i] = SiluEnc(gate[i]) * up[i];
    const std::vector<float> down =
        LinearSlice(act, seq, config.intermediate_size, weights.Get(p + "mlp.down_proj.weight"), 0,
                    hidden);
    for (size_t i = 0; i < h.size(); ++i) h[i] += down[i];

    // DeepStack: add the visual features into the visual token rows, for the
    // FIRST len(deepstack) layers only (encoder.py:770-779, 792-798).
    if (layer < static_cast<int64_t>(deepstack.size())) {
      VT_CHECK(visual_pos_mask != nullptr,
               "minimax_h3 encoder: deepstack embeds require a visual position mask");
      const std::vector<float>& embeds = deepstack[static_cast<size_t>(layer)];
      int64_t visual_index = 0;
      for (int64_t s = 0; s < seq; ++s) {
        if (!visual_pos_mask[s]) continue;
        for (int64_t i = 0; i < hidden; ++i) {
          h[static_cast<size_t>(s * hidden + i)] +=
              embeds[static_cast<size_t>(visual_index * hidden + i)];
        }
        ++visual_index;
      }
    }
  }
  // NO final norm: the checkpoint consumes the UNNORMALIZED layer-49 state.
  return h;
}

// ---------------------------------------------------------------------------
// Vision tower block (encoder.py:417-481) — the repeated unit of the ViT.
//
// Differs from the TEXT tower in several ways that all matter numerically:
//   * LayerNorm (with bias), not RMSNorm, at eps 1e-6;
//   * the qkv output is reshaped [seq, 3, heads, head_dim] and PERMUTED, so the
//     layout is [q_all | k_all | v_all] per token — not the per-head interleave
//     the video VAE's ViT uses;
//   * rotary is applied in FP32 with cos/sin supplied per token;
//   * attention is NON-CAUSAL and segmented by `cu_seqlens` (one segment per
//     image/frame), so it never crosses a packed-image boundary;
//   * the MLP uses the TANH-approximate GELU (`gelu_pytorch_tanh`), not exact erf.
// ---------------------------------------------------------------------------

namespace {

// nn.LayerNorm over the last dim.
void LayerNormRows(const float* in, const float* weight, const float* bias, float* out,
                   int64_t rows, int64_t width, double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double mean = 0.0;
    for (int64_t i = 0; i < width; ++i) mean += src[i];
    mean /= static_cast<double>(width);
    double var = 0.0;
    for (int64_t i = 0; i < width; ++i) var += (src[i] - mean) * (src[i] - mean);
    var /= static_cast<double>(width);
    const double inv = 1.0 / std::sqrt(var + eps);
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) {
      double value = (src[i] - mean) * inv * weight[i];
      if (bias != nullptr) value += bias[i];
      dst[i] = static_cast<float>(value);
    }
  }
}

// nn.GELU(approximate="tanh").
float GeluTanh(float x) {
  const double xd = x;
  const double inner = 0.7978845608028654 * (xd + 0.044715 * xd * xd * xd);
  return static_cast<float>(0.5 * xd * (1.0 + std::tanh(inner)));
}

// y = x @ W^T + b.
std::vector<float> LinearBias(const std::vector<float>& x, int64_t rows, int64_t in_features,
                              const std::vector<float>& weight, const std::vector<float>* bias,
                              int64_t out_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(o)] : 0.0;
      const float* w = weight.data() + o * in_features;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_features + i)]) *
               static_cast<double>(w[i]);
      }
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

std::vector<float> MiniMaxH3VisionBlockForward(const MiniMaxH3VisionBlockConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::string& prefix,
                                               const std::vector<float>& hidden, int64_t seq,
                                               const float* cos, const float* sin,
                                               const int32_t* cu_seqlens, int64_t num_segments) {
  const int64_t dim = config.hidden_size;
  const int64_t heads = config.num_heads;
  const int64_t head_dim = dim / heads;
  VT_CHECK(dim > 0 && heads > 0 && dim % heads == 0,
           "minimax_h3 vision: hidden_size must divide by num_heads");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == seq * dim,
           "minimax_h3 vision: hidden size does not match [seq, dim]");

  std::vector<float> h = hidden;
  std::vector<float> normed(h.size());

  // --- attention ---
  LayerNormRows(h.data(), weights.Get(prefix + ".norm1.weight").data(),
                weights.Get(prefix + ".norm1.bias").data(), normed.data(), seq, dim, config.eps);
  const std::vector<float> qkv =
      LinearBias(normed, seq, dim, weights.Get(prefix + ".attn.qkv.weight"),
                 &weights.Get(prefix + ".attn.qkv.bias"), 3 * dim);

  // reshape(seq, 3, heads, head_dim).permute(1,0,2,3) => q/k/v each [seq, heads, head_dim].
  std::vector<float> q(static_cast<size_t>(seq * dim));
  std::vector<float> k(static_cast<size_t>(seq * dim));
  std::vector<float> v(static_cast<size_t>(seq * dim));
  for (int64_t s = 0; s < seq; ++s) {
    const float* row = qkv.data() + s * 3 * dim;
    std::copy(row, row + dim, q.begin() + s * dim);
    std::copy(row + dim, row + 2 * dim, k.begin() + s * dim);
    std::copy(row + 2 * dim, row + 3 * dim, v.begin() + s * dim);
  }

  // Rotary in FP32, cos/sin shared across heads (encoder.py:404-415).
  const int64_t half = head_dim / 2;
  for (int64_t s = 0; s < seq; ++s) {
    const float* c = cos + s * head_dim;
    const float* sn = sin + s * head_dim;
    for (int64_t head = 0; head < heads; ++head) {
      for (std::vector<float>* target : {&q, &k}) {
        float* row = target->data() + (s * heads + head) * head_dim;
        for (int64_t i = 0; i < half; ++i) {
          const double lo = row[i], hi = row[i + half];
          row[i] = static_cast<float>(lo * c[i] - hi * sn[i]);
          row[i + half] = static_cast<float>(hi * c[i + half] + lo * sn[i + half]);
        }
      }
    }
  }

  // Non-causal attention, segmented by cu_seqlens.
  std::vector<float> attn(static_cast<size_t>(seq * dim), 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<double> probs;
  for (int64_t seg = 0; seg < num_segments; ++seg) {
    const int64_t begin = cu_seqlens[seg];
    const int64_t end = cu_seqlens[seg + 1];
    const int64_t len = end - begin;
    if (len <= 0) continue;
    probs.resize(static_cast<size_t>(len));
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t i = begin; i < end; ++i) {
        double max_score = -1e30;
        for (int64_t j = begin; j < end; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(q[static_cast<size_t>((i * heads + head) * head_dim + d)]) *
                   static_cast<double>(k[static_cast<size_t>((j * heads + head) * head_dim + d)]);
          }
          probs[static_cast<size_t>(j - begin)] = dot * scale;
          max_score = std::max(max_score, probs[static_cast<size_t>(j - begin)]);
        }
        double denom = 0.0;
        for (int64_t j = 0; j < len; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(probs[static_cast<size_t>(j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < len; ++j) {
            acc += probs[static_cast<size_t>(j)] *
                   static_cast<double>(
                       v[static_cast<size_t>(((begin + j) * heads + head) * head_dim + d)]);
          }
          attn[static_cast<size_t>((i * heads + head) * head_dim + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }
  }
  const std::vector<float> projected =
      LinearBias(attn, seq, dim, weights.Get(prefix + ".attn.proj.weight"),
                 &weights.Get(prefix + ".attn.proj.bias"), dim);
  for (size_t i = 0; i < h.size(); ++i) h[i] += projected[i];

  // --- MLP with tanh-approximate GELU ---
  LayerNormRows(h.data(), weights.Get(prefix + ".norm2.weight").data(),
                weights.Get(prefix + ".norm2.bias").data(), normed.data(), seq, dim, config.eps);
  std::vector<float> mid =
      LinearBias(normed, seq, dim, weights.Get(prefix + ".mlp.linear_fc1.weight"),
                 &weights.Get(prefix + ".mlp.linear_fc1.bias"), config.intermediate_size);
  for (float& value : mid) value = GeluTanh(value);
  const std::vector<float> down =
      LinearBias(mid, seq, config.intermediate_size, weights.Get(prefix + ".mlp.linear_fc2.weight"),
                 &weights.Get(prefix + ".mlp.linear_fc2.bias"), dim);
  for (size_t i = 0; i < h.size(); ++i) h[i] += down[i];
  return h;
}

// ---------------------------------------------------------------------------
// Vision tower surround (encoder.py:483-600) — patch embed, interpolated
// position embedding, 2D rotary, the block stack, and the patch mergers.
// ---------------------------------------------------------------------------

// The learned position grid is BILINEARLY resampled to each image's patch grid,
// then permuted into spatial-merge order (encoder.py:544-585). `num_grid_per_side`
// is sqrt(num_position_embeddings).
std::vector<float> MiniMaxH3VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_table,
                                                      int64_t num_grid_per_side, int64_t dim,
                                                      const std::vector<int64_t>& grid_thw,
                                                      int64_t merge_size) {
  std::vector<float> out;
  for (size_t g = 0; g + 2 < grid_thw.size() + 1 && g * 3 + 2 < grid_thw.size(); ++g) {
    const int64_t t = grid_thw[g * 3 + 0];
    const int64_t h = grid_thw[g * 3 + 1];
    const int64_t w = grid_thw[g * 3 + 2];

    // torch.linspace(0, n-1, k): with k == 1 torch returns just the START.
    auto linspace = [&](int64_t count) {
      std::vector<double> v(static_cast<size_t>(count));
      if (count == 1) {
        v[0] = 0.0;
        return v;
      }
      const double step = static_cast<double>(num_grid_per_side - 1) / static_cast<double>(count - 1);
      for (int64_t i = 0; i < count; ++i) v[static_cast<size_t>(i)] = static_cast<double>(i) * step;
      return v;
    };
    const std::vector<double> h_idxs = linspace(h);
    const std::vector<double> w_idxs = linspace(w);

    // Bilinear over the four surrounding grid points; `.int()` TRUNCATES.
    std::vector<float> plane(static_cast<size_t>(h * w * dim), 0.0f);
    for (int64_t i = 0; i < h; ++i) {
      const int64_t hf = static_cast<int64_t>(h_idxs[static_cast<size_t>(i)]);
      const int64_t hc = std::min<int64_t>(hf + 1, num_grid_per_side - 1);
      const double dh = h_idxs[static_cast<size_t>(i)] - static_cast<double>(hf);
      for (int64_t j = 0; j < w; ++j) {
        const int64_t wf = static_cast<int64_t>(w_idxs[static_cast<size_t>(j)]);
        const int64_t wc = std::min<int64_t>(wf + 1, num_grid_per_side - 1);
        const double dw = w_idxs[static_cast<size_t>(j)] - static_cast<double>(wf);
        const int64_t idx[4] = {hf * num_grid_per_side + wf, hf * num_grid_per_side + wc,
                                hc * num_grid_per_side + wf, hc * num_grid_per_side + wc};
        const double wt[4] = {(1.0 - dh) * (1.0 - dw), (1.0 - dh) * dw, dh * (1.0 - dw), dh * dw};
        float* dst = plane.data() + (i * w + j) * dim;
        for (int64_t c = 0; c < dim; ++c) {
          double acc = 0.0;
          for (int k = 0; k < 4; ++k) {
            acc += static_cast<double>(
                       pos_embed_table[static_cast<size_t>(idx[k] * dim + c)]) * wt[k];
          }
          dst[c] = static_cast<float>(acc);
        }
      }
    }

    // repeat over t, then view(t, h/m, m, w/m, m, -1).permute(0,1,3,2,4,5).flatten
    const int64_t mh = h / merge_size, mw = w / merge_size;
    for (int64_t frame = 0; frame < t; ++frame) {
      for (int64_t bh = 0; bh < mh; ++bh) {
        for (int64_t bw = 0; bw < mw; ++bw) {
          for (int64_t ih = 0; ih < merge_size; ++ih) {
            for (int64_t iw = 0; iw < merge_size; ++iw) {
              const int64_t row = (bh * merge_size + ih) * w + (bw * merge_size + iw);
              const float* src = plane.data() + row * dim;
              out.insert(out.end(), src, src + dim);
            }
          }
        }
      }
    }
  }
  return out;
}

// 2D rotary position ids in spatial-merge order (encoder.py:510-537), then
// freq_table[pos_ids].flatten(1) -> [tokens, 2 * (head_dim/2 / 2)].
std::vector<float> MiniMaxH3VisionRotary(const std::vector<int64_t>& grid_thw, int64_t merge_size,
                                         int64_t rotary_dim, double theta) {
  int64_t max_hw = 0;
  for (size_t g = 0; g * 3 + 2 < grid_thw.size(); ++g) {
    max_hw = std::max({max_hw, grid_thw[g * 3 + 1], grid_thw[g * 3 + 2]});
  }
  const int64_t freqs = rotary_dim / 2;
  std::vector<double> inv(static_cast<size_t>(freqs));
  for (int64_t i = 0; i < freqs; ++i) {
    inv[static_cast<size_t>(i)] =
        1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(rotary_dim));
  }
  // freq_table[p][f] = p * inv[f]
  auto table = [&](int64_t pos, int64_t f) { return static_cast<double>(pos) * inv[static_cast<size_t>(f)]; };

  std::vector<float> out;
  for (size_t g = 0; g * 3 + 2 < grid_thw.size(); ++g) {
    const int64_t t = grid_thw[g * 3 + 0];
    const int64_t h = grid_thw[g * 3 + 1];
    const int64_t w = grid_thw[g * 3 + 2];
    const int64_t mh = h / merge_size, mw = w / merge_size;
    std::vector<float> one;
    for (int64_t bh = 0; bh < mh; ++bh) {
      for (int64_t bw = 0; bw < mw; ++bw) {
        for (int64_t ih = 0; ih < merge_size; ++ih) {
          for (int64_t iw = 0; iw < merge_size; ++iw) {
            const int64_t row = bh * merge_size + ih;
            const int64_t col = bw * merge_size + iw;
            for (int64_t f = 0; f < freqs; ++f) one.push_back(static_cast<float>(table(row, f)));
            for (int64_t f = 0; f < freqs; ++f) one.push_back(static_cast<float>(table(col, f)));
          }
        }
      }
    }
    for (int64_t frame = 0; frame < t; ++frame) out.insert(out.end(), one.begin(), one.end());
  }
  (void)max_hw;  // the table is evaluated lazily; max_hw only bounds it upstream
  return out;
}

namespace {

// PatchMerger (encoder.py:372-386): LayerNorm -> fc1 -> exact-erf GELU -> fc2.
// `use_postshuffle_norm` decides whether the norm sees the pre- or post-shuffle
// width, which is why the DeepStack mergers and the final merger differ.
std::vector<float> PatchMerger(const MiniMaxH3AudioVaeWeights& weights, const std::string& prefix,
                               const std::vector<float>& x, int64_t rows, int64_t dim,
                               int64_t merged_width, int64_t out_hidden, bool postshuffle,
                               double eps) {
  const int64_t groups = rows / (merged_width / dim);
  std::vector<float> normed(x.size());
  if (postshuffle) {
    LayerNormRows(x.data(), weights.Get(prefix + ".norm.weight").data(),
                  weights.Get(prefix + ".norm.bias").data(), normed.data(), groups, merged_width,
                  eps);
  } else {
    LayerNormRows(x.data(), weights.Get(prefix + ".norm.weight").data(),
                  weights.Get(prefix + ".norm.bias").data(), normed.data(), rows, dim, eps);
  }
  std::vector<float> mid =
      LinearBias(normed, groups, merged_width, weights.Get(prefix + ".linear_fc1.weight"),
                 &weights.Get(prefix + ".linear_fc1.bias"), merged_width);
  for (float& value : mid) {
    // nn.GELU() default = EXACT erf, unlike the vision MLP's tanh approximation.
    value = static_cast<float>(0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0))));
  }
  return LinearBias(mid, groups, merged_width, weights.Get(prefix + ".linear_fc2.weight"),
                    &weights.Get(prefix + ".linear_fc2.bias"), out_hidden);
}

}  // namespace

MiniMaxH3VisionTowerResult MiniMaxH3VisionTowerForward(
    const MiniMaxH3VisionTowerConfig& config, const MiniMaxH3AudioVaeWeights& weights,
    const std::vector<float>& patches, const std::vector<int64_t>& grid_thw) {
  const int64_t dim = config.block.hidden_size;
  const int64_t merge = config.spatial_merge_size;
  const int64_t patch_elems =
      config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
  int64_t tokens = 0;
  for (size_t g = 0; g * 3 + 2 < grid_thw.size(); ++g) {
    tokens += grid_thw[g * 3] * grid_thw[g * 3 + 1] * grid_thw[g * 3 + 2];
  }
  VT_CHECK(static_cast<int64_t>(patches.size()) == tokens * patch_elems,
           "minimax_h3 vision: patch input does not match the grid");

  // patch_embed: a Conv3d whose kernel EQUALS its stride is a linear map over the
  // flattened patch (encoder.py:337-353).
  std::vector<float> h = LinearBias(patches, tokens, patch_elems,
                                    weights.Get("patch_embed.proj.weight"),
                                    &weights.Get("patch_embed.proj.bias"), dim);

  const int64_t num_grid_per_side =
      static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(config.num_position_embeddings))));
  VT_CHECK(num_grid_per_side * num_grid_per_side == config.num_position_embeddings,
           "minimax_h3 vision: num_position_embeddings must be a perfect square");
  const std::vector<float> pos = MiniMaxH3VisionPosEmbedInterpolate(
      weights.Get("pos_embed.weight"), num_grid_per_side, dim, grid_thw, merge);
  VT_CHECK(pos.size() == h.size(), "minimax_h3 vision: position embedding size mismatch");
  for (size_t i = 0; i < h.size(); ++i) h[i] += pos[i];

  const int64_t head_dim = dim / config.block.num_heads;
  const std::vector<float> freqs =
      MiniMaxH3VisionRotary(grid_thw, merge, head_dim / 2, config.rope_theta);
  // emb = cat(freqs, freqs); cos/sin over [tokens, head_dim].
  const int64_t rot = head_dim / 2;
  std::vector<float> cos(static_cast<size_t>(tokens * head_dim));
  std::vector<float> sin(static_cast<size_t>(tokens * head_dim));
  for (int64_t s = 0; s < tokens; ++s) {
    for (int64_t i = 0; i < rot; ++i) {
      const double angle = freqs[static_cast<size_t>(s * rot + i)];
      cos[static_cast<size_t>(s * head_dim + i)] = static_cast<float>(std::cos(angle));
      cos[static_cast<size_t>(s * head_dim + rot + i)] = static_cast<float>(std::cos(angle));
      sin[static_cast<size_t>(s * head_dim + i)] = static_cast<float>(std::sin(angle));
      sin[static_cast<size_t>(s * head_dim + rot + i)] = static_cast<float>(std::sin(angle));
    }
  }

  // cu_seqlens: one segment per FRAME (h*w repeated t times), cumulative.
  std::vector<int32_t> cu = {0};
  for (size_t g = 0; g * 3 + 2 < grid_thw.size(); ++g) {
    const int64_t frame_tokens = grid_thw[g * 3 + 1] * grid_thw[g * 3 + 2];
    for (int64_t f = 0; f < grid_thw[g * 3]; ++f) cu.push_back(cu.back() + static_cast<int32_t>(frame_tokens));
  }

  MiniMaxH3VisionTowerResult result;
  const int64_t merged_width = dim * merge * merge;
  for (int64_t layer = 0; layer < config.depth; ++layer) {
    h = MiniMaxH3VisionBlockForward(config.block, weights,
                                    "blocks." + std::to_string(layer), h, tokens, cos.data(),
                                    sin.data(), cu.data(), static_cast<int64_t>(cu.size()) - 1);
    for (size_t d = 0; d < config.deepstack_visual_indexes.size(); ++d) {
      if (config.deepstack_visual_indexes[d] != layer) continue;
      result.deepstack.push_back(PatchMerger(
          weights, "deepstack_merger_list." + std::to_string(d), h, tokens, dim, merged_width,
          config.out_hidden_size, /*postshuffle=*/true, config.block.eps));
    }
  }
  result.merged = PatchMerger(weights, "merger", h, tokens, dim, merged_width,
                              config.out_hidden_size, /*postshuffle=*/false, config.block.eps);
  return result;
}

}  // namespace vllm
