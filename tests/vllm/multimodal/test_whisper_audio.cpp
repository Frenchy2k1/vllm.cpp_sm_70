// A2 — Whisper-class AUDIO encoder-tower faithfulness gate. Runs the C++
// WhisperAudioEncoderForward (src/vllm/model_executor/models/whisper_audio.cpp) on
// the committed A1 log-mel `input_features` [80,3000] golden and asserts every
// stage matches the dumped transformers-5.13.1 WhisperEncoder reference
// (scripts/mm/a2_audio_encoder_ref.py) within the measured bf16-depth envelope:
//   post_conv, post_pos, block0_out, final_ln_out (the [1500,768] encoder output).
// This is the A2 milestone: the encoder tower proven faithful in ISOLATION (the
// audio->text e2e is A3 on Voxtral-Mini-3B over the LANDED Mistral backbone).
//
// GPU + weights required: the encoder.* weights are dumped (f32, ~130 MiB, NOT
// committed) via scripts/mm/a2_audio_encoder_weight_dump.py; point
// VLLM_WHISPER_ENC_WEIGHTS at the dir. embed_positions is loaded from the COMMITTED
// golden fixture. Without the weight dir (or without CUDA) the gate is skipped, not
// failed.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/whisper_audio.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::multimodal::WhisperAudioEncoderCapture;
using vllm::multimodal::WhisperAudioEncoderConfig;
using vllm::multimodal::WhisperAudioEncoderWeights;
using vllm::multimodal::WhisperEncoderLayerWeights;

std::string AudioFix() { return std::string(MM_AUDIO_FIXTURE_DIR); }

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

struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
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

std::vector<float> W(const std::string& dir, const std::string& name) {
  return ReadF32(dir + "/" + name + ".bin");
}

}  // namespace

TEST_CASE("whisper_audio_encoder_faithful_vs_transformers_5_13_1") {
  const char* wdir_c = std::getenv("VLLM_WHISPER_ENC_WEIGHTS");
  if (wdir_c == nullptr) {
    MESSAGE("SKIP: set VLLM_WHISPER_ENC_WEIGHTS to the dumped encoder.* weight dir "
            "(scripts/mm/a2_audio_encoder_weight_dump.py) to run the A2 tower gate");
    return;
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    MESSAGE("SKIP: no CUDA backend");
    return;
  }
  const std::string wdir = wdir_c;
  const std::string F = AudioFix();

  WhisperAudioEncoderConfig cfg;  // whisper-small defaults (see header).

  // --- weights ----------------------------------------------------------------
  WhisperAudioEncoderWeights w;
  w.conv1_w = W(wdir, "conv1.weight");
  w.conv1_b = W(wdir, "conv1.bias");
  w.conv2_w = W(wdir, "conv2.weight");
  w.conv2_b = W(wdir, "conv2.bias");
  // embed_positions is a fixed sinusoid — loaded from the COMMITTED golden.
  w.embed_positions_w = ReadF32(F + "/enc_embed_positions_f32.bin");
  w.final_ln_w = W(wdir, "layer_norm.weight");
  w.final_ln_b = W(wdir, "layer_norm.bias");
  w.layers.resize(static_cast<size_t>(cfg.num_layers));
  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "layers." + std::to_string(l);
    WhisperEncoderLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.attn_ln_w = W(wdir, p + ".self_attn_layer_norm.weight");
    lw.attn_ln_b = W(wdir, p + ".self_attn_layer_norm.bias");
    lw.final_ln_w = W(wdir, p + ".final_layer_norm.weight");
    lw.final_ln_b = W(wdir, p + ".final_layer_norm.bias");
    lw.q_w = W(wdir, p + ".self_attn.q_proj.weight");
    lw.q_b = W(wdir, p + ".self_attn.q_proj.bias");
    lw.k_w = W(wdir, p + ".self_attn.k_proj.weight");  // NO bias in Whisper
    lw.v_w = W(wdir, p + ".self_attn.v_proj.weight");
    lw.v_b = W(wdir, p + ".self_attn.v_proj.bias");
    lw.out_w = W(wdir, p + ".self_attn.out_proj.weight");
    lw.out_b = W(wdir, p + ".self_attn.out_proj.bias");
    lw.fc1_w = W(wdir, p + ".fc1.weight");
    lw.fc1_b = W(wdir, p + ".fc1.bias");
    lw.fc2_w = W(wdir, p + ".fc2.weight");
    lw.fc2_b = W(wdir, p + ".fc2.bias");
  }

  // --- input (committed A1 log-mel golden) ------------------------------------
  std::vector<float> feats = ReadF32(F + "/input_features_f32.bin");  // [80,3000]
  REQUIRE(feats.size() == static_cast<size_t>(cfg.num_mel_bins * cfg.n_frames));

  // --- run tower with capture -------------------------------------------------
  WhisperAudioEncoderCapture cap;
  std::vector<float> out =
      vllm::multimodal::WhisperAudioEncoderForward(feats, w, cfg, *gpu, &cap);

  // --- references -------------------------------------------------------------
  auto ref_post_conv = ReadF32(F + "/enc_post_conv_f32.bin");
  auto ref_post_pos = ReadF32(F + "/enc_post_pos_f32.bin");
  auto ref_block0 = ReadF32(F + "/enc_block0_f32.bin");
  auto ref_final = ReadF32(F + "/enc_final_ln_f32.bin");

  // --- Tolerance model (RCA, measured on GB10 sm_121a — see the A2 ledger) -----
  // The conv frontend + first encoder block match the transformers bf16 reference
  // TIGHTLY (proving the conv im2col+MatmulBT and the pre-norm block are correct);
  // the final-LN output diverges only by SMOOTH bf16 accumulation across the 12
  // encoder layers between two independent bf16 kernel stacks (our vt ops vs
  // transformers' bf16 conv/linear/SDPA), the same ~0.25%/layer envelope the M2a
  // vision tower measured. The bounds below are that measured envelope (measured x
  // ~1.25 margin); a wrong conv stride / missing sinusoid / GELU-tanh-vs-erf /
  // skipped final-LN drives each stage FAR past its band (RED, see the ledger).
  Err e_conv = Compare(cap.post_conv, ref_post_conv);
  INFO("post_conv relL2=", e_conv.rel_l2, " maxabs=", e_conv.max_abs);
  CHECK(e_conv.rel_l2 < 8e-3);

  Err e_pos = Compare(cap.post_pos, ref_post_pos);
  INFO("post_pos relL2=", e_pos.rel_l2, " maxabs=", e_pos.max_abs);
  CHECK(e_pos.rel_l2 < 8e-3);

  Err e_b0 = Compare(cap.block0_out, ref_block0);
  INFO("block0 relL2=", e_b0.rel_l2, " maxabs=", e_b0.max_abs);
  CHECK(e_b0.rel_l2 < 1.5e-2);

  Err e_final = Compare(out, ref_final);
  INFO("final_ln relL2=", e_final.rel_l2, " maxabs=", e_final.max_abs);
  CHECK(e_final.rel_l2 < 5e-2);
  // the returned tensor IS the final-LN output.
  Err e_cap = Compare(cap.final_ln_out, ref_final);
  CHECK(e_cap.rel_l2 == doctest::Approx(e_final.rel_l2));
}
