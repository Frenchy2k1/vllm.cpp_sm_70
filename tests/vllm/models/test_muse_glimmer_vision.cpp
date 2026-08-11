// Muse Glimmer perception encoder (W3) — per-stage numeric gate.
//
// Compares every stage of `src/vllm/model_executor/models/muse_glimmer_vision.cpp`
// against the reference in `muse_glimmer_vision_goldens.inc`, which
// `scripts/mm/muse_glimmer_vision_ref.py` derives in torch from a VERBATIM
// transcription of vllm PR #51655 head `075d645af`,
// `vllm/model_executor/models/muse_glimmer.py:555-1044`.
//
// HONESTY. Muse Glimmer does not exist at the parity pin `555967922`; the pinned
// oracle cannot load it. This gate therefore establishes the TOWER'S PER-STAGE
// NUMERICS against a reference we computed — it does NOT establish image or
// video end-to-end correctness (W4/W5, which need the checkpoint), and it
// licenses NO speed claim of any kind. See `.agents/specs/muse-glimmer.md` §0.
//
// The synthetic weights are an explicit LCG reproduced bit-exactly on both
// sides, so no weight blob is committed — only the reference OUTPUTS. Every
// tolerance below is stated against the compute dtype actually used: the host
// precomputes (patchify, positional interpolation, 2D RoPE, the window
// permutation, pixel shuffle) are f32 and gated near machine epsilon; the
// device tower is gated in f32 and again in the production bf16.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/muse_glimmer_vision.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include "muse_glimmer_vision_goldens.inc"

namespace {

using vllm::multimodal::MuseGlimmerVisionAdapterWeights;
using vllm::multimodal::MuseGlimmerVisionBlockWeights;
using vllm::multimodal::MuseGlimmerVisionCapture;
using vllm::multimodal::MuseGlimmerVisionConfig;
using vllm::multimodal::MuseGlimmerVisionImage;
using vllm::multimodal::MuseGlimmerVisionWeights;

// The generator's LCG (scripts/mm/muse_glimmer_vision_ref.py:lcg): the integer
// recurrence and the double-precision mapping are identical, so both sides hold
// bit-identical f32 weights without shipping a weight blob.
std::vector<float> Lcg(uint32_t seed, size_t n, double scale) {
  std::vector<float> out(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const double u = static_cast<double>(s >> 8) / 16777216.0;
    out[i] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

std::vector<float> LcgPlusOne(uint32_t seed, size_t n, double scale) {
  std::vector<float> v = Lcg(seed, n, scale);
  for (float& x : v) x += 1.0f;
  return v;
}

MuseGlimmerVisionConfig FixtureConfig() {
  MuseGlimmerVisionConfig cfg;
  cfg.hidden_size = 32;
  cfg.num_attention_heads = 4;  // head_dim 8 -> spatial_dim 4, 2 freqs per axis
  cfg.num_hidden_layers = 3;
  cfg.intermediate_size = 48;
  cfg.patch_size = 2;
  cfg.patch_temporal = 2;
  cfg.merge_kernel_size = 2;
  cfg.pos_emb_height = 4;  // window block = 4x4 = 16 tokens
  cfg.pos_emb_width = 4;
  cfg.output_dim = 128;  // hidden * merge^2
  cfg.adapter_dim = 16;
  cfg.layer_norm_eps = 1e-5f;
  cfg.layer_types = {"window_attention", "full_attention", "window_attention"};
  return cfg;
}

MuseGlimmerVisionWeights FixtureWeights(const MuseGlimmerVisionConfig& cfg) {
  const size_t h = static_cast<size_t>(cfg.hidden_size);
  const size_t i = static_cast<size_t>(cfg.intermediate_size);
  MuseGlimmerVisionWeights w;
  w.conv1_w = Lcg(1001, h * static_cast<size_t>(cfg.patch_dim()), 0.1);
  w.pos_emb = Lcg(1002,
                  static_cast<size_t>(cfg.pos_emb_height * cfg.pos_emb_width) * h, 0.5);
  w.ln_pre_w = LcgPlusOne(1003, h, 0.3);
  w.ln_pre_b = Lcg(1004, h, 0.1);
  w.ln_post_w = LcgPlusOne(1005, h, 0.3);
  w.ln_post_b = Lcg(1006, h, 0.1);
  w.blocks.resize(static_cast<size_t>(cfg.num_hidden_layers));
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    const uint32_t s = static_cast<uint32_t>(2000 + 100 * l);
    MuseGlimmerVisionBlockWeights& b = w.blocks[static_cast<size_t>(l)];
    b.ln_1_w = LcgPlusOne(s + 1, h, 0.3);
    b.ln_1_b = Lcg(s + 2, h, 0.1);
    b.ln_2_w = LcgPlusOne(s + 3, h, 0.3);
    b.ln_2_b = Lcg(s + 4, h, 0.1);
    b.qkv_w = Lcg(s + 5, 3 * h * h, 0.1);
    b.qkv_b = Lcg(s + 6, 3 * h, 0.1);
    b.o_w = Lcg(s + 7, h * h, 0.1);
    b.o_b = Lcg(s + 8, h, 0.1);
    b.c_fc_w = Lcg(s + 9, i * h, 0.1);
    b.c_fc_b = Lcg(s + 10, i, 0.1);
    b.c_proj_w = Lcg(s + 11, h * i, 0.1);
    b.c_proj_b = Lcg(s + 12, h, 0.1);
  }
  return w;
}

MuseGlimmerVisionAdapterWeights FixtureAdapter(const MuseGlimmerVisionConfig& cfg) {
  MuseGlimmerVisionAdapterWeights a;
  a.c_fc_w = Lcg(9001, static_cast<size_t>(cfg.adapter_dim * cfg.output_dim), 0.1);
  a.c_proj_w = Lcg(9002, static_cast<size_t>(cfg.adapter_dim * cfg.adapter_dim), 0.1);
  return a;
}

// IMAGES in the generator: [3,12,12] (grid 6x6; pos-emb interpolation runs off
// BOTH ends of the 4x4 learned grid, and the 8x8 window padding yields blocks of
// 16/8/8/4 valid tokens) and [6,8,8] (grid 4x4; the patch_temporal*3 branch of
// _patchify, exactly one full window).
std::vector<MuseGlimmerVisionImage> FixtureImages() {
  std::vector<MuseGlimmerVisionImage> imgs(2);
  imgs[0].channels = 3;
  imgs[0].height = 12;
  imgs[0].width = 12;
  imgs[0].pixels = Lcg(5001, 3 * 12 * 12, 1.0);
  imgs[1].channels = 6;
  imgs[1].height = 8;
  imgs[1].width = 8;
  imgs[1].pixels = Lcg(5002, 6 * 8 * 8, 1.0);
  return imgs;
}

struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
};

template <typename T>
Err Compare(const std::vector<float>& got, const T* ref, size_t n) {
  REQUIRE(got.size() == n);
  double num = 0.0, den = 0.0, mx = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    mx = std::max(mx, std::abs(d));
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

std::string Fmt(const Err& e) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "rel_l2=%.3e max_abs=%.3e", e.rel_l2, e.max_abs);
  return std::string(buf);
}

}  // namespace

// --- STAGE 1: patchify (muse_glimmer.py:902-935) -----------------------------
// conv1_linear is a Linear over PATCHIFIED input, not a conv (:696,:710), so the
// patch vector layout (t, c, ph, pw) is load-bearing: a transposed layout still
// produces a full-rank embedding and a plausible-looking image understanding.
TEST_CASE("muse_glimmer_vision_patchify_matches_upstream") {
  const MuseGlimmerVisionConfig cfg = FixtureConfig();
  const std::vector<MuseGlimmerVisionImage> imgs = FixtureImages();

  const std::vector<float> p0 = vllm::multimodal::MuseGlimmerVisionPatchify(imgs[0], cfg);
  const Err e0 = Compare(p0, muse_glimmer_vision_ref::kPatchify0,
                         std::size(muse_glimmer_vision_ref::kPatchify0));
  MESSAGE("patchify 3ch: ", Fmt(e0));
  CHECK_MESSAGE(e0.max_abs == 0.0, "3-channel patchify (temporal broadcast): ", Fmt(e0));

  const std::vector<float> p1 = vllm::multimodal::MuseGlimmerVisionPatchify(imgs[1], cfg);
  const Err e1 = Compare(p1, muse_glimmer_vision_ref::kPatchify1,
                         std::size(muse_glimmer_vision_ref::kPatchify1));
  MESSAGE("patchify 6ch: ", Fmt(e1));
  CHECK_MESSAGE(e1.max_abs == 0.0, "6-channel patchify (per-frame stack): ", Fmt(e1));
}

// --- STAGE 2: positional-embedding bilinear interpolation (:761-820) ---------
// The `+0.5 ... -0.5` half-pixel convention AND the per-corner validity masking
// are both silent when wrong: sampling at cell centres instead degrades image
// understanding without any shape or range error. The 6x6 grid deliberately
// samples OFF BOTH ENDS of the 4x4 learned grid so the masking is exercised.
TEST_CASE("muse_glimmer_vision_pos_emb_interp_matches_upstream") {
  const MuseGlimmerVisionConfig cfg = FixtureConfig();
  const MuseGlimmerVisionWeights w = FixtureWeights(cfg);

  const std::vector<float> g0 =
      vllm::multimodal::MuseGlimmerVisionPosEmbedInterpolate(w.pos_emb, 6, 6, cfg);
  const Err e0 = Compare(g0, muse_glimmer_vision_ref::kPosEmb0,
                         std::size(muse_glimmer_vision_ref::kPosEmb0));
  MESSAGE("pos-emb interp 6x6: ", Fmt(e0));
  CHECK_MESSAGE(e0.max_abs < 1e-6, "6x6 interp from a 4x4 table: ", Fmt(e0));

  // 4x4 -> 4x4 is the identity sample (h_grid == arange), so the interpolation
  // must reproduce the learned table exactly.
  const std::vector<float> g1 =
      vllm::multimodal::MuseGlimmerVisionPosEmbedInterpolate(w.pos_emb, 4, 4, cfg);
  const Err e1 = Compare(g1, muse_glimmer_vision_ref::kPosEmb1,
                         std::size(muse_glimmer_vision_ref::kPosEmb1));
  CHECK_MESSAGE(e1.max_abs < 1e-6, "4x4 identity interp: ", Fmt(e1));
  const Err eid = Compare(g1, w.pos_emb.data(), w.pos_emb.size());
  CHECK_MESSAGE(eid.max_abs < 1e-6, "identity interp != learned table: ", Fmt(eid));
}

// --- STAGE 3: 2D RoPE (:741-759) --------------------------------------------
// `freqs = cat([freq_w, freq_h])` — WIDTH FIRST. Transposing w/h keeps every
// shape, norm and range intact and silently rotates the image 90 degrees in
// position space. Positions are also 1-BASED (`arange(1, grid+1)`).
TEST_CASE("muse_glimmer_vision_2d_rope_matches_upstream") {
  const MuseGlimmerVisionConfig cfg = FixtureConfig();
  std::vector<float> cos, sin;
  vllm::multimodal::MuseGlimmerVisionRopeCosSin(6, 6, cfg, &cos, &sin);
  const Err ec = Compare(cos, muse_glimmer_vision_ref::kRopeCos0,
                         std::size(muse_glimmer_vision_ref::kRopeCos0));
  const Err es = Compare(sin, muse_glimmer_vision_ref::kRopeSin0,
                         std::size(muse_glimmer_vision_ref::kRopeSin0));
  MESSAGE("rope cos: ", Fmt(ec));
  CHECK_MESSAGE(ec.max_abs < 1e-6, "rope cos: ", Fmt(ec));
  MESSAGE("rope sin: ", Fmt(es));
  CHECK_MESSAGE(es.max_abs < 1e-6, "rope sin: ", Fmt(es));

  // A non-square grid pins the w/h ORDER independently of the goldens: on a
  // 2x3 grid the first half of row 0 must carry the WIDTH frequency (w=1) and
  // the second half the HEIGHT frequency (h=1) — identical here — while row 1
  // (h=1,w=2) must differ from the transposed row 3 (h=2,w=1).
  std::vector<float> c2, s2;
  vllm::multimodal::MuseGlimmerVisionRopeCosSin(2, 3, cfg, &c2, &s2);
  const int64_t half = cfg.head_dim() / 2;   // 4 columns: [w0 w1 h0 h1]
  const int64_t nfreq = half / 2;            // 2 frequencies per axis
  REQUIRE(static_cast<int64_t>(s2.size()) == 6 * half);
  // token 1 = (h=1, w=2); token 3 = (h=2, w=1): width-first means s2[1][0..nfreq)
  // is sin(2*inv_freq) and s2[3][0..nfreq) is sin(1*inv_freq).
  CHECK(std::abs(s2[static_cast<size_t>(1 * half)] -
                 static_cast<float>(std::sin(2.0))) < 1e-6f);
  CHECK(std::abs(s2[static_cast<size_t>(3 * half)] -
                 static_cast<float>(std::sin(1.0))) < 1e-6f);
  // and the HEIGHT half is the mirror image of that.
  CHECK(std::abs(s2[static_cast<size_t>(1 * half + nfreq)] -
                 static_cast<float>(std::sin(1.0))) < 1e-6f);
  CHECK(std::abs(s2[static_cast<size_t>(3 * half + nfreq)] -
                 static_cast<float>(std::sin(2.0))) < 1e-6f);
}

// --- STAGE 4: block-windowed permutation (:844-867) --------------------------
// Blocks of pos_emb_height x pos_emb_width over a -1-PADDED grid; the per-block
// count of surviving (non-padded) entries becomes the attention seq_len.
TEST_CASE("muse_glimmer_vision_window_permutation_matches_upstream") {
  const MuseGlimmerVisionConfig cfg = FixtureConfig();
  std::vector<int32_t> perm, seq_lens;
  vllm::multimodal::MuseGlimmerVisionSparsePermutation(6, 6, cfg, &perm, &seq_lens);
  REQUIRE(perm.size() == std::size(muse_glimmer_vision_ref::kSparsePerm0));
  for (size_t i = 0; i < perm.size(); ++i)
    CHECK(perm[i] == muse_glimmer_vision_ref::kSparsePerm0[i]);
  // 6x6 padded to 8x8 -> four 4x4 blocks with 16 / 8 / 8 / 4 valid tokens.
  REQUIRE(seq_lens.size() == 4);
  CHECK(seq_lens[0] == 16);
  CHECK(seq_lens[1] == 8);
  CHECK(seq_lens[2] == 8);
  CHECK(seq_lens[3] == 4);
  int64_t total = 0;
  for (int32_t n : seq_lens) total += n;
  CHECK(total == 36);

  std::vector<int32_t> perm1, seq1;
  vllm::multimodal::MuseGlimmerVisionSparsePermutation(4, 4, cfg, &perm1, &seq1);
  REQUIRE(seq1.size() == 1);
  CHECK(seq1[0] == 16);
  for (size_t i = 0; i < perm1.size(); ++i) CHECK(perm1[i] == static_cast<int32_t>(i));
}

// --- STAGE 5: pixel-shuffle downsample (:822-842) ----------------------------
// The merge permutation groups merge^2 spatial neighbours AND the channel
// transpose makes the output HIDDEN-major within each group. Dropping the
// transpose keeps the shape and every value, only reordered — the projector
// still runs and the model still emits fluent text.
TEST_CASE("muse_glimmer_vision_pixel_shuffle_matches_upstream") {
  const MuseGlimmerVisionConfig cfg = FixtureConfig();
  // ramp[i, d] = i * 4 + d over a 6x6 grid with hidden 4 (generator: `ramp`).
  std::vector<float> ramp(36 * 4);
  for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<float>(i);
  const std::vector<float> got =
      vllm::multimodal::MuseGlimmerVisionPixelShuffle(ramp, 6, 6, 4, cfg);
  const Err e = Compare(got, muse_glimmer_vision_ref::kPixelShuffleRamp,
                        std::size(muse_glimmer_vision_ref::kPixelShuffleRamp));
  MESSAGE("pixel shuffle: ", Fmt(e));
  CHECK_MESSAGE(e.max_abs == 0.0, "pixel shuffle: ", Fmt(e));
}

// --- STAGE 6+7: the whole tower and the adapter ------------------------------
// Runs the encoder on both fixture images at once (upstream concatenates and
// carries per-image cu_seqlens, :998-1010) and gates ln_pre, block 0 and the
// final pixel-shuffled features, then the adapter (:1036-1044).
TEST_CASE("muse_glimmer_vision_tower_matches_upstream_f32") {
  vt::Backend* cpu = vt::TryGetBackend(vt::DeviceType::kCPU);
  REQUIRE(cpu != nullptr);
  MuseGlimmerVisionConfig cfg = FixtureConfig();
  cfg.compute_dtype = vt::DType::kF32;
  const MuseGlimmerVisionWeights w = FixtureWeights(cfg);

  MuseGlimmerVisionCapture cap;
  const std::vector<float> tower =
      vllm::multimodal::MuseGlimmerVisionForward(FixtureImages(), w, cfg, *cpu, &cap);

  REQUIRE(cap.ln_pre_out.size() == 2);
  const Err ep0 = Compare(cap.ln_pre_out[0], muse_glimmer_vision_ref::kLnPre0,
                          std::size(muse_glimmer_vision_ref::kLnPre0));
  MESSAGE("ln_pre image 0 (f32): ", Fmt(ep0));
  CHECK_MESSAGE(ep0.rel_l2 < 1e-6, "ln_pre image 0: ", Fmt(ep0));
  const Err ep1 = Compare(cap.ln_pre_out[1], muse_glimmer_vision_ref::kLnPre1,
                          std::size(muse_glimmer_vision_ref::kLnPre1));
  MESSAGE("ln_pre image 1 (f32): ", Fmt(ep1));
  CHECK_MESSAGE(ep1.rel_l2 < 1e-6, "ln_pre image 1: ", Fmt(ep1));

  const Err eb = Compare(cap.block0_out, muse_glimmer_vision_ref::kBlock0,
                         std::size(muse_glimmer_vision_ref::kBlock0));
  MESSAGE("block 0 (f32): ", Fmt(eb));
  CHECK_MESSAGE(eb.rel_l2 < 1e-6, "block 0 (window attention): ", Fmt(eb));

  const Err et = Compare(tower, muse_glimmer_vision_ref::kTowerOut,
                         std::size(muse_glimmer_vision_ref::kTowerOut));
  MESSAGE("tower output (f32): ", Fmt(et));
  CHECK_MESSAGE(et.rel_l2 < 1e-6, "tower output: ", Fmt(et));

  const std::vector<float> adapted = vllm::multimodal::MuseGlimmerVisionAdapterForward(
      tower, 13, FixtureAdapter(cfg), cfg, *cpu);
  const Err ea = Compare(adapted, muse_glimmer_vision_ref::kAdapterOut,
                         std::size(muse_glimmer_vision_ref::kAdapterOut));
  MESSAGE("adapter output (f32): ", Fmt(ea));
  CHECK_MESSAGE(ea.rel_l2 < 1e-6, "adapter output: ", Fmt(ea));
}

// The production dtype. bf16 keeps ~3 decimal digits, so the envelope here is
// stated as bf16 depth over a 3-block tower — NOT loosened to hide a defect:
// every mutation in the spec's table breaks the f32 case above outright.
TEST_CASE("muse_glimmer_vision_tower_bf16_within_envelope") {
  vt::Backend* cpu = vt::TryGetBackend(vt::DeviceType::kCPU);
  REQUIRE(cpu != nullptr);
  MuseGlimmerVisionConfig cfg = FixtureConfig();  // compute_dtype defaults to bf16
  REQUIRE(cfg.compute_dtype == vt::DType::kBF16);
  const MuseGlimmerVisionWeights w = FixtureWeights(cfg);
  const std::vector<float> tower =
      vllm::multimodal::MuseGlimmerVisionForward(FixtureImages(), w, cfg, *cpu, nullptr);
  const Err et = Compare(tower, muse_glimmer_vision_ref::kTowerOut,
                         std::size(muse_glimmer_vision_ref::kTowerOut));
  MESSAGE("tower output (bf16): ", Fmt(et));
  CHECK_MESSAGE(et.rel_l2 < 2e-2, "bf16 tower output: ", Fmt(et));
}
