// P5: `ParakeetForRNNT` / `ParakeetForTDT` END-TO-END gate against a
// HuggingFace ORACLE.
//
// Oracle: transformers `main`
//   transformers/models/parakeet/modeling_parakeet.py
//     ParakeetRNNTDecoder:831, ParakeetRNNTJointNetwork:879, ParakeetForRNNT:922,
//     ParakeetTDTJointNetwork:1035, ParakeetForTDT:1052
//   transformers/models/parakeet/generation_parakeet.py
//     ParakeetRNNTDecoderCache:23, ParakeetRNNTGenerationMixin:125,
//     ParakeetTDTGenerationMixin:271
//
// **THE CORRECTION THIS ROW RESTS ON.** P4 recorded the transducer as having "NO
// upstream in either vLLM or HF transformers", and therefore as a product call
// rather than mirror work. That was measured against the LOCALLY INSTALLED
// transformers 5.3.0, which ships only `ParakeetForCTC`. It is FALSE of current
// upstream, which implements the whole stack at the lines above. The provenance
// check below asserts the dump did NOT come from 5.3.0, so the fixture cannot
// silently regress to the version that motivated the wrong claim.
//
// **HONESTY NOTE: what this gate is and is not.** The oracle is a SMALL,
// SEEDED, RANDOMLY-INITIALISED model, dumped by
// scripts/mm/p5_parakeet_transducer_oracle_dump.py. Random weights exercise the
// module MATH exactly as pretrained ones would, but they make the decoded TEXT
// meaningless, so this gate asserts the DECODER and JOINT tensors (rel-L2) and
// the EMITTED SEQUENCE and PER-STEP DURATIONS (exact) and claims NOTHING about a
// transcript. The pretrained arm is `parakeet_transducer_pretrained_checkpoint`
// below: it gates our ids against a real `generate()` run recorded by
// scripts/mm/p5_parakeet_transducer_reference.py, and is SKIPPED (not silently
// passed) when the checkpoint is absent.
//
// **TRACED, not read (T0).** `decoder_trace` in the manifest is a forward hook
// over a full `generate()` run: every decoder call took input_ids of shape
// [1, 1]. That is asserted here, because the whole greedy loop is built on it:
// one token per step, the LSTM never re-running its prefix.
//
// The fixture is also asserted to actually WALK the loop: `branch_coverage`
// records blank emissions, non-blank emissions, tokens that advance the frame
// (the RNN-T `max_symbols_per_step` guard) and tokens held at one frame. A
// regeneration that degenerated to a single branch FAILS here instead of
// quietly gating nothing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vllm/model_executor/models/parakeet_transducer.h"
#include "vt/backend.h"

namespace {

using vllm::multimodal::LoadParakeetTransducer;
using vllm::multimodal::ParakeetEncoderConfig;
using vllm::multimodal::ParakeetEncoderForward;
using vllm::multimodal::ParakeetEncoderLayerWeights;
using vllm::multimodal::ParakeetEncoderWeights;
using vllm::multimodal::ParakeetForTransducerForward;
using vllm::multimodal::ParakeetForTransducerWeights;
using vllm::multimodal::ParakeetLstmCell;
using vllm::multimodal::ParakeetLstmLayerWeights;
using vllm::multimodal::ParakeetRNNTDecoderState;
using vllm::multimodal::ParakeetRNNTDecoderStep;
using vllm::multimodal::ParakeetSubsamplingWeights;
using vllm::multimodal::ParakeetTransducerConfig;
using vllm::multimodal::ParakeetTransducerGreedyDecode;
using vllm::multimodal::ParakeetTransducerJoint;
using vllm::multimodal::ParakeetTransducerOutput;

std::string Fix() { return std::string(PARAKEET_TRANSDUCER_FIXTURE_DIR); }

template <typename T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<T> v(static_cast<size_t>(n) / sizeof(T));
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
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    if (std::abs(d) > mx) mx = std::abs(d);
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

nlohmann::json Manifest() {
  std::ifstream f(Fix() + "/manifest.json");
  REQUIRE_MESSAGE(f.good(), "cannot open the parakeet transducer fixture manifest");
  nlohmann::json m;
  f >> m;
  return m;
}

ParakeetTransducerConfig ConfigFromManifest(const nlohmann::json& model) {
  const nlohmann::json& c = model.at("config");
  ParakeetTransducerConfig cfg;
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.decoder_hidden_size = c.at("decoder_hidden_size").get<int64_t>();
  cfg.num_decoder_layers = c.at("num_decoder_layers").get<int64_t>();
  cfg.hidden_act = c.at("hidden_act").get<std::string>();
  cfg.max_symbols_per_step = c.at("max_symbols_per_step").get<int64_t>();
  cfg.blank_token_id = c.at("blank_token_id").get<int32_t>();
  cfg.pad_token_id = c.at("pad_token_id").get<int32_t>();
  cfg.durations = c.at("durations").get<std::vector<int64_t>>();
  cfg.decoder_start_token_id =
      model.at("generation_config").at("decoder_start_token_id").get<int32_t>();
  return cfg;
}

ParakeetEncoderConfig EncoderConfigFromManifest(const nlohmann::json& model) {
  const nlohmann::json& c = model.at("encoder_config");
  ParakeetEncoderConfig cfg;
  cfg.hidden_size = c.at("hidden_size").get<int64_t>();
  cfg.num_hidden_layers = c.at("num_hidden_layers").get<int64_t>();
  cfg.num_attention_heads = c.at("num_attention_heads").get<int64_t>();
  cfg.num_key_value_heads = c.at("num_key_value_heads").get<int64_t>();
  cfg.intermediate_size = c.at("intermediate_size").get<int64_t>();
  cfg.hidden_act = c.at("hidden_act").get<std::string>();
  cfg.attention_bias = c.at("attention_bias").get<bool>();
  cfg.convolution_bias = c.at("convolution_bias").get<bool>();
  cfg.conv_kernel_size = c.at("conv_kernel_size").get<int64_t>();
  cfg.subsampling_factor = c.at("subsampling_factor").get<int64_t>();
  cfg.subsampling_conv_channels = c.at("subsampling_conv_channels").get<int64_t>();
  cfg.num_mel_bins = c.at("num_mel_bins").get<int64_t>();
  cfg.subsampling_conv_kernel_size = c.at("subsampling_conv_kernel_size").get<int64_t>();
  cfg.subsampling_conv_stride = c.at("subsampling_conv_stride").get<int64_t>();
  cfg.max_position_embeddings = c.at("max_position_embeddings").get<int64_t>();
  cfg.scale_input = c.at("scale_input").get<bool>();
  return cfg;
}

// The transducer half of the dumped `state_dict()`, under the key map
// src/vllm/model_executor/models/parakeet_weights.cpp documents.
void LoadTransducerFixtureWeights(const std::string& prefix,
                                  const ParakeetTransducerConfig& cfg,
                                  ParakeetForTransducerWeights* w) {
  auto W = [&](const std::string& name) {
    return ReadBin<float>(Fix() + "/" + prefix + "weights/" + name + ".bin");
  };
  w->encoder_projector_w = W("encoder_projector.weight");
  w->encoder_projector_b = W("encoder_projector.bias");
  w->decoder.embedding = W("decoder.embedding.weight");
  for (int64_t l = 0; l < cfg.num_decoder_layers; ++l) {
    const std::string s = std::to_string(l);
    ParakeetLstmLayerWeights lw;
    lw.weight_ih = W("decoder.lstm.weight_ih_l" + s);
    lw.weight_hh = W("decoder.lstm.weight_hh_l" + s);
    lw.bias_ih = W("decoder.lstm.bias_ih_l" + s);
    lw.bias_hh = W("decoder.lstm.bias_hh_l" + s);
    w->decoder.lstm.push_back(std::move(lw));
  }
  w->decoder.projector_w = W("decoder.decoder_projector.weight");
  w->decoder.projector_b = W("decoder.decoder_projector.bias");
  w->joint_head_w = W("joint.head.weight");
  w->joint_head_b = W("joint.head.bias");
}

// The ENCODER half, same key map as the P4 fixture loader.
ParakeetEncoderWeights LoadEncoderFixtureWeights(const std::string& prefix,
                                                 const ParakeetEncoderConfig& cfg) {
  auto W = [&](const std::string& name) {
    return ReadBin<float>(Fix() + "/" + prefix + "encoder_weights/encoder." + name + ".bin");
  };
  ParakeetEncoderWeights w;
  ParakeetSubsamplingWeights& sub = w.subsampling;
  sub.conv0_w = W("subsampling.layers.0.weight");
  sub.conv0_b = W("subsampling.layers.0.bias");
  for (int64_t i = 0; i + 1 < cfg.num_subsampling_layers(); ++i) {
    ParakeetSubsamplingWeights::Stage stage;
    const std::string dw = "subsampling.layers." + std::to_string(2 + 3 * i);
    const std::string pw = "subsampling.layers." + std::to_string(3 + 3 * i);
    stage.depthwise_w = W(dw + ".weight");
    stage.depthwise_b = W(dw + ".bias");
    stage.pointwise_w = W(pw + ".weight");
    stage.pointwise_b = W(pw + ".bias");
    sub.stages.push_back(std::move(stage));
  }
  sub.linear_w = W("subsampling.linear.weight");
  sub.linear_b = W("subsampling.linear.bias");
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    ParakeetEncoderLayerWeights lw;
    lw.feed_forward1.linear1_w = W(p + "feed_forward1.linear1.weight");
    lw.feed_forward1.linear1_b = W(p + "feed_forward1.linear1.bias");
    lw.feed_forward1.linear2_w = W(p + "feed_forward1.linear2.weight");
    lw.feed_forward1.linear2_b = W(p + "feed_forward1.linear2.bias");
    lw.feed_forward2.linear1_w = W(p + "feed_forward2.linear1.weight");
    lw.feed_forward2.linear1_b = W(p + "feed_forward2.linear1.bias");
    lw.feed_forward2.linear2_w = W(p + "feed_forward2.linear2.weight");
    lw.feed_forward2.linear2_b = W(p + "feed_forward2.linear2.bias");
    lw.self_attn.q_w = W(p + "self_attn.q_proj.weight");
    lw.self_attn.q_b = W(p + "self_attn.q_proj.bias");
    lw.self_attn.k_w = W(p + "self_attn.k_proj.weight");
    lw.self_attn.k_b = W(p + "self_attn.k_proj.bias");
    lw.self_attn.v_w = W(p + "self_attn.v_proj.weight");
    lw.self_attn.v_b = W(p + "self_attn.v_proj.bias");
    lw.self_attn.o_w = W(p + "self_attn.o_proj.weight");
    lw.self_attn.o_b = W(p + "self_attn.o_proj.bias");
    lw.self_attn.relative_k_w = W(p + "self_attn.relative_k_proj.weight");
    lw.self_attn.bias_u = W(p + "self_attn.bias_u");
    lw.self_attn.bias_v = W(p + "self_attn.bias_v");
    lw.conv.pointwise1_w = W(p + "conv.pointwise_conv1.weight");
    lw.conv.pointwise1_b = W(p + "conv.pointwise_conv1.bias");
    lw.conv.depthwise_w = W(p + "conv.depthwise_conv.weight");
    lw.conv.depthwise_b = W(p + "conv.depthwise_conv.bias");
    lw.conv.norm_w = W(p + "conv.norm.weight");
    lw.conv.norm_b = W(p + "conv.norm.bias");
    lw.conv.norm_running_mean = W(p + "conv.norm.running_mean");
    lw.conv.norm_running_var = W(p + "conv.norm.running_var");
    lw.conv.pointwise2_w = W(p + "conv.pointwise_conv2.weight");
    lw.conv.pointwise2_b = W(p + "conv.pointwise_conv2.bias");
    lw.norm_feed_forward1_w = W(p + "norm_feed_forward1.weight");
    lw.norm_feed_forward1_b = W(p + "norm_feed_forward1.bias");
    lw.norm_self_att_w = W(p + "norm_self_att.weight");
    lw.norm_self_att_b = W(p + "norm_self_att.bias");
    lw.norm_conv_w = W(p + "norm_conv.weight");
    lw.norm_conv_b = W(p + "norm_conv.bias");
    lw.norm_feed_forward2_w = W(p + "norm_feed_forward2.weight");
    lw.norm_feed_forward2_b = W(p + "norm_feed_forward2.bias");
    lw.norm_out_w = W(p + "norm_out.weight");
    lw.norm_out_b = W(p + "norm_out.bias");
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// An INDEPENDENT scalar LSTM step, written straight from the torch `nn.LSTM`
// definition rather than from the port: the four gates are sliced out into
// separate vectors first and the recurrence is applied afterwards, which is a
// different loop structure and a different accumulation order from
// ParakeetLstmCell's fused row walk. Agreement is therefore evidence, not a
// tautology.
void ReferenceLstmCell(const std::vector<float>& x, const ParakeetLstmLayerWeights& w,
                       int64_t in, int64_t H, std::vector<float>* h,
                       std::vector<float>* c) {
  auto matvec = [](const std::vector<float>& m, const std::vector<float>& v, int64_t rows,
                   int64_t cols, int64_t row_offset) {
    std::vector<double> out(static_cast<size_t>(rows), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int64_t k = 0; k < cols; ++k) {
        acc += static_cast<double>(m[static_cast<size_t>(row_offset + r) * cols + k]) *
               static_cast<double>(v[static_cast<size_t>(k)]);
      }
      out[static_cast<size_t>(r)] = acc;
    }
    return out;
  };

  std::vector<std::vector<double>> gate(4);
  for (int64_t g = 0; g < 4; ++g) {
    std::vector<double> from_x = matvec(w.weight_ih, x, H, in, g * H);
    std::vector<double> from_h = matvec(w.weight_hh, *h, H, H, g * H);
    gate[static_cast<size_t>(g)].resize(static_cast<size_t>(H));
    for (int64_t j = 0; j < H; ++j) {
      double v = from_x[static_cast<size_t>(j)] + from_h[static_cast<size_t>(j)];
      if (!w.bias_ih.empty()) {
        v += static_cast<double>(w.bias_ih[static_cast<size_t>(g * H + j)]);
        v += static_cast<double>(w.bias_hh[static_cast<size_t>(g * H + j)]);
      }
      gate[static_cast<size_t>(g)][static_cast<size_t>(j)] = v;
    }
  }
  auto sig = [](double z) { return 1.0 / (1.0 + std::exp(-z)); };
  for (int64_t j = 0; j < H; ++j) {
    const size_t u = static_cast<size_t>(j);
    const double cn = sig(gate[1][u]) * static_cast<double>((*c)[u]) +
                      sig(gate[0][u]) * std::tanh(gate[2][u]);
    (*c)[u] = static_cast<float>(cn);
    (*h)[u] = static_cast<float>(sig(gate[3][u]) * std::tanh(cn));
  }
}

}  // namespace

// The LSTM cell against the independent in-test reference, over several shapes
// and several sequential steps (so the recurrence, not just one step, is gated),
// with and without biases.
TEST_CASE("parakeet_lstm_cell_matches_scalar_reference") {
  std::mt19937 rng(20260807);
  std::normal_distribution<float> nd(0.0f, 0.5f);

  for (const auto& shape : std::vector<std::pair<int64_t, int64_t>>{
           {1, 1}, {3, 5}, {8, 8}, {7, 4}, {4, 13}}) {
    const int64_t in = shape.first;
    const int64_t H = shape.second;
    for (bool with_bias : {true, false}) {
      CAPTURE(in);
      CAPTURE(H);
      CAPTURE(with_bias);
      ParakeetLstmLayerWeights w;
      w.weight_ih.resize(static_cast<size_t>(4 * H * in));
      w.weight_hh.resize(static_cast<size_t>(4 * H * H));
      for (float& v : w.weight_ih) v = nd(rng);
      for (float& v : w.weight_hh) v = nd(rng);
      if (with_bias) {
        w.bias_ih.resize(static_cast<size_t>(4 * H));
        w.bias_hh.resize(static_cast<size_t>(4 * H));
        for (float& v : w.bias_ih) v = nd(rng);
        for (float& v : w.bias_hh) v = nd(rng);
      }

      std::vector<float> h(static_cast<size_t>(H), 0.0f), c(static_cast<size_t>(H), 0.0f);
      std::vector<float> rh = h, rc = c;
      for (int step = 0; step < 6; ++step) {
        std::vector<float> x(static_cast<size_t>(in));
        for (float& v : x) v = nd(rng);
        ParakeetLstmCell(x, w, in, H, &h, &c);
        ReferenceLstmCell(x, w, in, H, &rh, &rc);
        const Err eh = Compare(h, rh);
        const Err ec = Compare(c, rc);
        CHECK(eh.max_abs < 1e-5);
        CHECK(ec.max_abs < 1e-5);
      }
    }
  }
}

TEST_CASE("parakeet_transducer_matches_hf_oracle") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const nlohmann::json m = Manifest();
  CHECK(m.at("provenance").at("pretrained").get<bool>() == false);
  // The dump MUST NOT come from the transformers release whose missing
  // transducer classes produced the stale "no upstream" claim P4 recorded.
  const std::string tfv = m.at("provenance").at("transformers").get<std::string>();
  CHECK(tfv != "5.3.0");
  MESSAGE("oracle transformers ", tfv);

  for (const std::string kind : {"rnnt", "tdt"}) {
    CAPTURE(kind);
    const nlohmann::json& mm = m.at("models").at(kind);
    CHECK(mm.at("attn_implementation").get<std::string>() == "sdpa");
    CHECK(mm.at("architecture").get<std::string>() ==
          (kind == "rnnt" ? "ParakeetForRNNT" : "ParakeetForTDT"));

    // TRACED: every decoder call in the real generate() run took ONE token.
    for (const auto& step : mm.at("decoder_trace")) {
      const auto shape = step.at("input_ids_shape").get<std::vector<int64_t>>();
      REQUIRE(shape.size() == 2);
      CHECK(shape[0] == 1);
      CHECK(shape[1] == 1);
    }
    // The fixture must actually walk the loop, not just one branch of it.
    const nlohmann::json& cov = mm.at("branch_coverage");
    CHECK(cov.at("blank_emissions").get<int>() > 0);
    CHECK(cov.at("token_emissions").get<int>() > 0);
    CHECK(cov.at("tokens_held_at_frame").get<int>() > 0);
    if (kind == "rnnt") CHECK(cov.at("frame_advancing_tokens").get<int>() > 0);

    const ParakeetTransducerConfig cfg = ConfigFromManifest(mm);
    const ParakeetEncoderConfig enc_cfg = EncoderConfigFromManifest(mm);
    CHECK(cfg.is_tdt() == (kind == "tdt"));

    const std::string prefix = kind + "_";
    ParakeetForTransducerWeights w;
    LoadTransducerFixtureWeights(prefix, cfg, &w);

    const int64_t D = cfg.decoder_hidden_size;
    const int64_t frames = mm.at("encoder_frames").get<int64_t>();
    const std::vector<float> ref_pooled = ReadBin<float>(Fix() + "/" + prefix + "encoder_projected.bin");
    REQUIRE(static_cast<int64_t>(ref_pooled.size()) == frames * D);

    // --- the prediction network, over a FIXED token walk. The walk starts with
    // the blank and repeats a token, so it drives the cache's blank fast path
    // (gen:851-855) and an ordinary LSTM update in the same case.
    const auto walk = mm.at("walk").get<std::vector<int32_t>>();
    const std::vector<float> ref_dec = ReadBin<float>(Fix() + "/" + prefix + "decoder_steps.bin");
    const std::vector<float> ref_joint = ReadBin<float>(Fix() + "/" + prefix + "joint_steps.bin");
    REQUIRE(static_cast<int64_t>(ref_dec.size()) == static_cast<int64_t>(walk.size()) * D);

    ParakeetRNNTDecoderState state;
    for (size_t i = 0; i < walk.size(); ++i) {
      CAPTURE(i);
      ParakeetRNNTDecoderStep(walk[i], w.decoder, cfg, &state);
      const std::vector<float> ref_step(
          ref_dec.begin() + static_cast<ptrdiff_t>(i * static_cast<size_t>(D)),
          ref_dec.begin() + static_cast<ptrdiff_t>((i + 1) * static_cast<size_t>(D)));
      const Err e = Compare(state.output, ref_step);
      MESSAGE(kind, " decoder step ", i, ": rel_l2=", e.rel_l2, " max_abs=", e.max_abs);
      CHECK(e.rel_l2 < 1e-5);

      // The joint at the matching encoder frame, the same pairing the dump used.
      const int64_t frame = std::min<int64_t>(static_cast<int64_t>(i), frames - 1);
      const std::vector<float> got_joint = ParakeetTransducerJoint(
          &ref_pooled[static_cast<size_t>(frame) * D], state.output, w, cfg, cpu);
      const int64_t jn = cfg.joint_output_size();
      const std::vector<float> ref_j(
          ref_joint.begin() + static_cast<ptrdiff_t>(i * static_cast<size_t>(jn)),
          ref_joint.begin() + static_cast<ptrdiff_t>((i + 1) * static_cast<size_t>(jn)));
      const Err ej = Compare(got_joint, ref_j);
      MESSAGE(kind, " joint step ", i, ": rel_l2=", ej.rel_l2, " max_abs=", ej.max_abs);
      CHECK(ej.rel_l2 < 1e-5);
    }

    // --- the greedy decode, from the ORACLE's own projected encoder output, so
    // any mismatch is the loop's and not the encoder's.
    const auto ref_seq = mm.at("sequences").get<std::vector<int32_t>>();
    const auto ref_dur = mm.at("step_durations").get<std::vector<int32_t>>();
    const ParakeetTransducerOutput got =
        ParakeetTransducerGreedyDecode(ref_pooled, frames, frames, w, cfg, cpu);
    CHECK(got.sequences == ref_seq);
    CHECK(got.durations == ref_dur);

    // token_ids is `sequences` minus the start token and every blank: no
    // collapse, since a transducer does not group repeats.
    std::vector<int32_t> expect_tokens;
    for (size_t i = 1; i < ref_seq.size(); ++i) {
      if (ref_seq[i] != cfg.blank_token_id) expect_tokens.push_back(ref_seq[i]);
    }
    CHECK(got.token_ids == expect_tokens);

    // --- the WHOLE model from mel features, which additionally gates the
    // encoder_projector orientation and the encoder wiring.
    w.encoder = LoadEncoderFixtureWeights(prefix, enc_cfg);
    const int64_t in_frames = mm.at("frames").get<int64_t>();
    const std::vector<float> features = ReadBin<float>(Fix() + "/" + prefix + "input_features.bin");
    REQUIRE(static_cast<int64_t>(features.size()) == in_frames * enc_cfg.num_mel_bins);

    int64_t valid_rows = 0;
    const std::vector<float> hidden =
        ParakeetEncoderForward(features, in_frames, mm.at("valid_frames").get<int64_t>(),
                               w.encoder, enc_cfg, cpu, &valid_rows, nullptr);
    CHECK(static_cast<int64_t>(hidden.size()) == frames * enc_cfg.hidden_size);

    const ParakeetTransducerOutput e2e = ParakeetForTransducerForward(
        features, in_frames, mm.at("valid_frames").get<int64_t>(), w, enc_cfg, cfg, cpu);
    CHECK(e2e.encoder_frames == frames);
    CHECK(e2e.sequences == ref_seq);
    CHECK(e2e.durations == ref_dur);
  }
}

// The PRETRAINED arm. Set both:
//   VLLM_PARAKEET_TRANSDUCER_CKPT: an HF-format ParakeetForRNNT/ParakeetForTDT
//     directory (config.json + model.safetensors), e.g. a snapshot of
//     nvidia/parakeet-rnnt-0.6b or nvidia/parakeet-tdt-0.6b-v3;
//   VLLM_PARAKEET_TRANSDUCER_REF : the output directory of
//     scripts/mm/p5_parakeet_transducer_reference.py for that checkpoint and
//     clip (reference.json + input_features.bin).
// The gate is the EMITTED TOKEN IDS, exactly, against what the real HF
// `generate()` produced on the same clip. Features come from the reference dump
// so this case isolates the model from the log-mel front end, which has its own
// gate in tests/vllm/multimodal/test_parakeet_audio_processor.cpp.
// SKIPPED, not passed, when either variable is unset.
TEST_CASE("parakeet_transducer_pretrained_checkpoint") {
  const char* ckpt = std::getenv("VLLM_PARAKEET_TRANSDUCER_CKPT");
  const char* refdir = std::getenv("VLLM_PARAKEET_TRANSDUCER_REF");
  if (ckpt == nullptr || refdir == nullptr) {
    MESSAGE(
        "SKIP: set VLLM_PARAKEET_TRANSDUCER_CKPT and VLLM_PARAKEET_TRANSDUCER_REF "
        "to gate against a real HF generate() run");
    return;
  }
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);

  nlohmann::json ref;
  {
    std::ifstream f(std::string(refdir) + "/reference.json");
    REQUIRE_MESSAGE(f.good(), "cannot open reference.json in ", refdir);
    f >> ref;
  }
  CHECK(ref.at("provenance").at("transformers").get<std::string>() != "5.3.0");

  ParakeetEncoderConfig enc_cfg;
  ParakeetTransducerConfig cfg;
  const ParakeetForTransducerWeights w = LoadParakeetTransducer(ckpt, &enc_cfg, &cfg);
  CHECK(static_cast<int64_t>(w.encoder.layers.size()) == enc_cfg.num_hidden_layers);
  CHECK(cfg.vocab_size == ref.at("config").at("vocab_size").get<int64_t>());
  CHECK(cfg.blank_token_id == ref.at("config").at("blank_token_id").get<int32_t>());
  CHECK(cfg.is_tdt() == !ref.at("config").at("durations").get<std::vector<int64_t>>().empty());

  const auto shape = ref.at("shapes").at("input_features").get<std::vector<int64_t>>();
  REQUIRE(shape.size() == 3);
  const int64_t in_frames = shape[1];
  const std::vector<float> features =
      ReadBin<float>(std::string(refdir) + "/input_features.bin");
  REQUIRE(static_cast<int64_t>(features.size()) == in_frames * shape[2]);

  const ParakeetTransducerOutput got = ParakeetForTransducerForward(
      features, in_frames, ref.at("valid_frames").get<int64_t>(), w, enc_cfg, cfg, cpu);
  CHECK(got.sequences == ref.at("sequences").get<std::vector<int32_t>>());
  CHECK(got.token_ids == ref.at("emitted_tokens").get<std::vector<int32_t>>());
}
