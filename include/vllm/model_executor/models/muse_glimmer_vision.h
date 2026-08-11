// Muse Glimmer perception encoder (`MuseGlimmerVisionEncoder` +
// `MuseGlimmerVisionAdapter`) — the W3 standalone tower forward.
//
// ─── OFF-PIN HONESTY (up front) ──────────────────────────────────────────────
// Muse Glimmer does not exist at the parity pin `555967922`, and not on vLLM
// `main` either: the only upstream implementation is the still-open PR
// vllm#51655 at head `075d645af`. Every `file:line` below points at that BRANCH
// HEAD (recorded as porting-inventory §9 deviation 16). The pinned oracle cannot
// load this model, so there is NO throughput denominator and NO speed axis is
// claimable here. W3 establishes the tower's PER-STAGE NUMERICS against a
// reference derived from the upstream Python formulas
// (`scripts/mm/muse_glimmer_vision_ref.py`); image and video end-to-end
// correctness are W4/W5 and are NOT established by this file.
//
// ─── WHAT THIS IS A PORT OF (file:line @ vllm#51655 head 075d645af) ──────────
//   OURS                                      <-  UPSTREAM (muse_glimmer.py)
//   MuseGlimmerVisionPatchify                 <-  :902-935  (_patchify)
//   MuseGlimmerVisionPosEmbedInterpolate      <-  :761-820  (_get_pos_emb)
//   MuseGlimmerVisionRopeCosSin               <-  :741-759  (_make_2d_rope)
//   MuseGlimmerVisionSparsePermutation        <-  :844-867  (_get_sparse_permutation)
//   MuseGlimmerVisionPixelShuffle             <-  :822-842  (_pixel_shuffle_downsample)
//   MuseGlimmerVisionForward                  <-  :937-1034 (Encoder.forward)
//                                                 + :651-689 (Block), :555-638
//                                                 (Attention), :641-648 (MLP)
//   MuseGlimmerVisionAdapterForward           <-  :1036-1044 (VisionAdapter)
//
// ─── THE FOUR SILENT TRAPS ───────────────────────────────────────────────────
// Each of these produces plausible-but-wrong image understanding rather than an
// error, so each is gated by a mutation in tests/vllm/models/
// test_muse_glimmer_vision.cpp rather than by inspection:
//   1. `conv1_linear` is a Linear over the PATCHIFIED input, not a Conv2d
//      (:696, :710). The patch vector layout is (t, c, ph, pw).
//   2. The positional table is bilinearly interpolated with a HALF-PIXEL
//      convention — `(i + 0.5) * (table/grid) - 0.5` — and per-corner validity
//      masking, so samples off either end of the table contribute nothing
//      (:761-820). Sampling at cell centres instead is silent.
//   3. 2D RoPE concatenates WIDTH FIRST: `freqs = cat([freq_w, freq_h])`, over
//      1-BASED positions (:741-759). Transposing w/h keeps every shape and norm.
//   4. Pixel shuffle groups merge^2 spatial neighbours AND transposes the group
//      into HIDDEN-major order (:822-842). Dropping the transpose preserves the
//      shape and the multiset of values.
//
// ln_pre / ln_post and the per-block ln_1 / ln_2 are plain `nn.LayerNorm` —
// weight AND bias, mean-subtracting (:714, :732, :660, :666) — NOT the RMSNorm
// the text tower uses. The vision MLP is `c_fc` -> GELU(erf) -> `c_proj`, both
// WITH bias (:641-648); the adapter is `gelu(c_proj(gelu(c_fc(x))))`, both
// bias-free (:1036-1044).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm::multimodal {

// The perception-encoder geometry. Mirrors `MuseGlimmerVisionParams`
// (include/vllm/model_executor/models/muse_glimmer.h, the W0 config parse) at
// the tower's own seam, exactly as `Qwen3VLVisionConfig` mirrors the Qwen3-VL
// vision config: the tower is unit-gateable without an HfConfig or a
// checkpoint. Wiring the parsed params into this struct is W4's job, along with
// the loader — W3 deliberately owns no checkpoint path. Defaults are the
// released `meta-models/Muse-Glimmer-30B` scale.
struct MuseGlimmerVisionConfig {
  int64_t hidden_size = 1536;
  int64_t num_attention_heads = 16;  // head_dim 96
  int64_t num_hidden_layers = 50;
  int64_t intermediate_size = 8960;
  int64_t patch_size = 14;
  int64_t patch_temporal = 2;
  int64_t merge_kernel_size = 2;
  int64_t pos_emb_height = 32;
  int64_t pos_emb_width = 32;
  int64_t output_dim = 6144;   // MUST equal hidden_size * merge_kernel_size^2
  int64_t adapter_dim = 4096;
  float layer_norm_eps = 1e-5f;
  // One entry per layer: "full_attention" attends over the whole image;
  // anything else ("window_attention") attends within one
  // pos_emb_height x pos_emb_width block (:1017-1021 selects on this).
  std::vector<std::string> layer_types;
  // Production model dtype. bf16 is what the checkpoint ships and what the
  // tower runs in; f32 exists so the per-stage gate can pin the arithmetic
  // without a bf16 rounding envelope in the way.
  vt::DType compute_dtype = vt::DType::kBF16;

  int64_t head_dim() const { return hidden_size / num_attention_heads; }
  int64_t merge_unit() const { return merge_kernel_size * merge_kernel_size; }
  // patch_temporal * 3 * patch_size^2 — the conv1_linear input width (:696).
  int64_t patch_dim() const { return patch_temporal * 3 * patch_size * patch_size; }
  bool has_window_layers() const;
};

// Host-side row-major f32 weights, in torch's storage layout (a Linear weight is
// [out, in]). Names mirror the upstream module tree.
struct MuseGlimmerVisionBlockWeights {
  std::vector<float> ln_1_w, ln_1_b;      // [hidden]
  std::vector<float> ln_2_w, ln_2_b;      // [hidden]
  std::vector<float> qkv_w, qkv_b;        // [3*hidden, hidden], [3*hidden]
  std::vector<float> o_w, o_b;            // [hidden, hidden], [hidden]
  std::vector<float> c_fc_w, c_fc_b;      // [inter, hidden], [inter]
  std::vector<float> c_proj_w, c_proj_b;  // [hidden, inter], [hidden]
};

struct MuseGlimmerVisionWeights {
  std::vector<float> conv1_w;             // [hidden, patch_dim]  (bias=False)
  std::vector<float> pos_emb;             // [pos_emb_height*pos_emb_width, hidden]
  std::vector<float> ln_pre_w, ln_pre_b;  // [hidden]
  std::vector<MuseGlimmerVisionBlockWeights> blocks;  // num_hidden_layers
  std::vector<float> ln_post_w, ln_post_b;            // [hidden]
};

// Both projections are bias-free (:1039-1040).
struct MuseGlimmerVisionAdapterWeights {
  std::vector<float> c_fc_w;    // [adapter_dim, output_dim]
  std::vector<float> c_proj_w;  // [adapter_dim, adapter_dim]
};

// One image (or one packed video clip). `pixels` is host f32 [channels, height,
// width] row-major. `channels` is 3 (a still image, broadcast across the
// temporal patch) or patch_temporal*3 (frames stacked on the channel axis) —
// upstream rejects anything else (:930-934). height and width must each divide
// patch_size * merge_kernel_size (:955-960).
struct MuseGlimmerVisionImage {
  std::vector<float> pixels;
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;
};

// Per-stage intermediates, host f32, filled only when a capture is passed.
// Production callers pass nullptr and pay nothing.
struct MuseGlimmerVisionCapture {
  std::vector<std::vector<float>> patchified;  // per image [tokens, patch_dim]
  std::vector<std::vector<float>> pos_embeds;  // per image [tokens, hidden]
  std::vector<std::vector<float>> ln_pre_out;  // per image [tokens, hidden]
  std::vector<float> block0_out;               // [total_tokens, hidden] (permuted)
};

// --- host precomputes (each individually gateable) ---------------------------

// _patchify (:902-935) -> [grid_h*grid_w, patch_dim], patch layout (t, c, ph, pw).
std::vector<float> MuseGlimmerVisionPatchify(const MuseGlimmerVisionImage& image,
                                             const MuseGlimmerVisionConfig& cfg);

// _get_pos_emb (:761-820) -> [grid_h*grid_w, hidden]. Bilinear resample of the
// learned [pos_emb_height*pos_emb_width, hidden] table onto the actual grid,
// with the half-pixel convention and per-corner validity masking.
std::vector<float> MuseGlimmerVisionPosEmbedInterpolate(const std::vector<float>& pos_emb,
                                                        int64_t grid_height,
                                                        int64_t grid_width,
                                                        const MuseGlimmerVisionConfig& cfg);

// _make_2d_rope (:741-759) -> cos/sin, each [grid_h*grid_w, head_dim/2], laid
// out WIDTH frequencies first then HEIGHT.
void MuseGlimmerVisionRopeCosSin(int64_t grid_height, int64_t grid_width,
                                 const MuseGlimmerVisionConfig& cfg,
                                 std::vector<float>* cos, std::vector<float>* sin);

// _get_sparse_permutation (:844-867). `permutation` is the surviving token order
// (block-major over pos_emb_height x pos_emb_width blocks of a -1-padded grid);
// `seq_lens` is the per-block count of surviving tokens, which becomes the
// windowed-attention segmentation.
void MuseGlimmerVisionSparsePermutation(int64_t grid_height, int64_t grid_width,
                                        const MuseGlimmerVisionConfig& cfg,
                                        std::vector<int32_t>* permutation,
                                        std::vector<int32_t>* seq_lens);

// _pixel_shuffle_downsample (:822-842). `hidden` is [grid_h*grid_w, dim];
// returns [(grid_h/merge)*(grid_w/merge), dim*merge^2], hidden-major within each
// merged group.
std::vector<float> MuseGlimmerVisionPixelShuffle(const std::vector<float>& hidden,
                                                 int64_t grid_height, int64_t grid_width,
                                                 int64_t dim,
                                                 const MuseGlimmerVisionConfig& cfg);

// --- the tower ---------------------------------------------------------------

// MuseGlimmerVisionEncoder.forward (:937-1034) over a batch of images. Images
// are patch-embedded and normalized independently, then CONCATENATED for the
// block stack: full-attention layers attend within one image, window layers
// within one pos_emb block (:998-1021). Returns host f32
// [sum_i tokens_i/merge^2, hidden*merge^2] — the pixel-shuffled features, in
// image order.
std::vector<float> MuseGlimmerVisionForward(const std::vector<MuseGlimmerVisionImage>& images,
                                            const MuseGlimmerVisionWeights& weights,
                                            const MuseGlimmerVisionConfig& cfg,
                                            vt::Backend& backend,
                                            MuseGlimmerVisionCapture* capture = nullptr);

// MuseGlimmerVisionAdapter.forward (:1036-1044): gelu(c_proj(gelu(c_fc(x)))).
// `features` is [num_tokens, output_dim] host f32 (the encoder's output);
// returns [num_tokens, adapter_dim].
std::vector<float> MuseGlimmerVisionAdapterForward(
    const std::vector<float>& features, int64_t num_tokens,
    const MuseGlimmerVisionAdapterWeights& weights, const MuseGlimmerVisionConfig& cfg,
    vt::Backend& backend);

}  // namespace vllm::multimodal
