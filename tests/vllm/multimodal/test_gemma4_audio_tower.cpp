// G3 — Gemma-4 USM-Conformer AUDIO tower per-stage faithfulness gate. Runs the
// host-f32 Gemma4AudioForward (src/vllm/model_executor/models/gemma4_audio.cpp) on
// the committed golden log-mel `input_features` [250,128] and asserts every stage
// matches the dumped transformers-5.13.1 Gemma4AudioModel reference
// (scripts/mm/g3_audio_tower_ref.py) f32-vs-f32 (the USM-Conformer MATH is the
// sole parity variable — a wrong chunk/rel-shift/GLU/softcap drives a stage far
// past its band):
//   subsample_out, position_embeddings, block0, block_mid, block_last,
//   output_proj (== last_hidden_state), projected (embed_audio == merge input).
//
// Weights (fp32, ~90 MiB, NOT committed) are dumped via
// scripts/mm/g3_audio_tower_ref.py (WEIGHT_OUT); point VLLM_GEMMA4_AUDIO_WEIGHTS at
// the dir. Refs default to the committed golden audio_refs dir. No GPU / no CUDA —
// pure host f32. Without the weight dir the gate is skipped, not failed.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/gemma4_audio.h"

namespace {

using vllm::multimodal::AClip;
using vllm::multimodal::ClipLinear;
using vllm::multimodal::Gemma4AudioCapture;
using vllm::multimodal::Gemma4AudioConfig;
using vllm::multimodal::Gemma4AudioLayerWeights;
using vllm::multimodal::Gemma4AudioWeights;

std::string RefDir() { return std::string(GEMMA4_AUDIO_REF_DIR); }

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

float Scalar(const std::string& dir, const std::string& name) {
  auto v = ReadF32(dir + "/" + name + ".bin");
  REQUIRE(v.size() == 1);
  return v[0];
}

struct Err {
  double rel_l2 = 0.0, max_abs = 0.0;
};
Err Compare(const std::vector<float>& got, const std::vector<float>& ref) {
  REQUIRE(got.size() == ref.size());
  double num = 0.0, den = 0.0, mx = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    mx = std::max(mx, std::abs(d));
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

ClipLinear LoadClip(const std::string& d, const std::string& p) {
  ClipLinear cl;
  cl.w = ReadF32(d + "/" + p + ".linear.weight.bin");
  cl.clip.in_min = Scalar(d, p + ".input_min");
  cl.clip.in_max = Scalar(d, p + ".input_max");
  cl.clip.out_min = Scalar(d, p + ".output_min");
  cl.clip.out_max = Scalar(d, p + ".output_max");
  return cl;
}

std::vector<float> W(const std::string& d, const std::string& name) {
  return ReadF32(d + "/" + name + ".bin");
}

}  // namespace

TEST_CASE("gemma4_audio_usm_conformer_faithful_vs_transformers_5_13_1") {
  const char* wdir_c = std::getenv("VLLM_GEMMA4_AUDIO_WEIGHTS");
  if (wdir_c == nullptr) {
    MESSAGE("SKIP: set VLLM_GEMMA4_AUDIO_WEIGHTS to the dumped audio tower weight dir "
            "(scripts/mm/g3_audio_tower_ref.py) to run the G3 tower gate");
    return;
  }
  const std::string wd = wdir_c;
  const std::string R = RefDir();
  Gemma4AudioConfig cfg;  // E4B audio_config defaults (see header/manifest).

  // --- weights ----------------------------------------------------------------
  Gemma4AudioWeights w;
  w.sub0_conv = W(wd, "subsample_conv_projection.layer0.conv.weight");
  w.sub0_norm = W(wd, "subsample_conv_projection.layer0.norm.weight");
  w.sub1_conv = W(wd, "subsample_conv_projection.layer1.conv.weight");
  w.sub1_norm = W(wd, "subsample_conv_projection.layer1.norm.weight");
  w.input_proj = W(wd, "subsample_conv_projection.input_proj_linear.weight");
  w.output_proj_w = W(wd, "output_proj.weight");
  w.output_proj_b = W(wd, "output_proj.bias");
  w.embed_proj = W(wd, "embed_audio.embedding_projection.weight");
  w.layers.resize(static_cast<size_t>(cfg.num_layers));
  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "layers." + std::to_string(l);
    Gemma4AudioLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.ff1.ffw1 = LoadClip(wd, p + ".feed_forward1.ffw_layer_1");
    lw.ff1.ffw2 = LoadClip(wd, p + ".feed_forward1.ffw_layer_2");
    lw.ff1.pre_ln = W(wd, p + ".feed_forward1.pre_layer_norm.weight");
    lw.ff1.post_ln = W(wd, p + ".feed_forward1.post_layer_norm.weight");
    lw.ff2.ffw1 = LoadClip(wd, p + ".feed_forward2.ffw_layer_1");
    lw.ff2.ffw2 = LoadClip(wd, p + ".feed_forward2.ffw_layer_2");
    lw.ff2.pre_ln = W(wd, p + ".feed_forward2.pre_layer_norm.weight");
    lw.ff2.post_ln = W(wd, p + ".feed_forward2.post_layer_norm.weight");
    lw.attn.q_proj = LoadClip(wd, p + ".self_attn.q_proj");
    lw.attn.k_proj = LoadClip(wd, p + ".self_attn.k_proj");
    lw.attn.v_proj = LoadClip(wd, p + ".self_attn.v_proj");
    lw.attn.post = LoadClip(wd, p + ".self_attn.post");
    lw.attn.relative_k_proj = W(wd, p + ".self_attn.relative_k_proj.weight");
    lw.attn.per_dim_scale = W(wd, p + ".self_attn.per_dim_scale");
    lw.lconv.linear_start = LoadClip(wd, p + ".lconv1d.linear_start");
    lw.lconv.linear_end = LoadClip(wd, p + ".lconv1d.linear_end");
    lw.lconv.depthwise = W(wd, p + ".lconv1d.depthwise_conv1d.weight");
    lw.lconv.pre_ln = W(wd, p + ".lconv1d.pre_layer_norm.weight");
    lw.lconv.conv_norm = W(wd, p + ".lconv1d.conv_norm.weight");
    lw.norm_pre_attn = W(wd, p + ".norm_pre_attn.weight");
    lw.norm_post_attn = W(wd, p + ".norm_post_attn.weight");
    lw.norm_out = W(wd, p + ".norm_out.weight");
  }

  // --- input (committed golden) -----------------------------------------------
  std::vector<float> feats = ReadF32(R + "/input_features_f32.bin");  // [250,128]
  const int64_t T = static_cast<int64_t>(feats.size()) / cfg.feature_size;
  REQUIRE(T * cfg.feature_size == static_cast<int64_t>(feats.size()));
  std::ifstream mf(R + "/input_features_mask_i32.bin", std::ios::binary);
  mf.seekg(0, std::ios::end);
  const std::streamoff mn = mf.tellg();
  mf.seekg(0, std::ios::beg);
  std::vector<int32_t> mask(static_cast<size_t>(mn) / sizeof(int32_t));
  mf.read(reinterpret_cast<char*>(mask.data()), mn);
  REQUIRE(static_cast<int64_t>(mask.size()) == T);

  // --- run tower --------------------------------------------------------------
  Gemma4AudioCapture cap;
  std::vector<float> projected =
      vllm::multimodal::Gemma4AudioForward(feats, T, mask, w, cfg, &cap);

  // --- references + per-stage gates (f32-vs-f32; tight bands, RED-first) -------
  Err e_sub = Compare(cap.subsample_out, ReadF32(R + "/subsample_out_f32.bin"));
  INFO("subsample relL2=", e_sub.rel_l2, " maxabs=", e_sub.max_abs);
  CHECK(e_sub.rel_l2 < 2e-4);

  Err e_pos = Compare(cap.position_embeddings, ReadF32(R + "/position_embeddings_f32.bin"));
  INFO("posemb relL2=", e_pos.rel_l2, " maxabs=", e_pos.max_abs);
  CHECK(e_pos.rel_l2 < 1e-5);

  Err e_b0 = Compare(cap.block0, ReadF32(R + "/block0_f32.bin"));
  INFO("block0 relL2=", e_b0.rel_l2, " maxabs=", e_b0.max_abs);
  CHECK(e_b0.rel_l2 < 5e-4);

  Err e_bm = Compare(cap.block_mid, ReadF32(R + "/block_mid_f32.bin"));
  INFO("block_mid relL2=", e_bm.rel_l2, " maxabs=", e_bm.max_abs);
  CHECK(e_bm.rel_l2 < 1e-3);

  Err e_bl = Compare(cap.block_last, ReadF32(R + "/block_last_f32.bin"));
  INFO("block_last relL2=", e_bl.rel_l2, " maxabs=", e_bl.max_abs);
  CHECK(e_bl.rel_l2 < 2e-3);

  Err e_op = Compare(cap.output_proj, ReadF32(R + "/output_proj_f32.bin"));
  INFO("output_proj relL2=", e_op.rel_l2, " maxabs=", e_op.max_abs);
  CHECK(e_op.rel_l2 < 2e-3);

  Err e_pr = Compare(projected, ReadF32(R + "/projected_f32.bin"));
  INFO("projected relL2=", e_pr.rel_l2, " maxabs=", e_pr.max_abs);
  CHECK(e_pr.rel_l2 < 3e-3);
}
