// P4 — `ParakeetForCTC` END-TO-END gate against a HuggingFace ORACLE.
//
// Oracle: transformers 5.3.0 `ParakeetForCTC`
// (transformers/models/parakeet/modeling_parakeet.py:675) — which IS what vLLM
// runs, since vllm/model_executor/models/parakeet.py:14,61 imports and
// instantiates `transformers.ParakeetEncoder` rather than implementing one. The
// CTC collapse is `ParakeetTokenizer._decode`
// (transformers/models/parakeet/tokenization_parakeet.py:28-49).
//
// **HONESTY NOTE — what this gate is and is not.** The oracle is a SMALL,
// SEEDED, RANDOMLY-INITIALISED `ParakeetForCTC`, dumped by
// scripts/mm/p4_parakeet_oracle_dump.py, not a pretrained checkpoint. No
// `nvidia/parakeet-*` weights were downloaded for this row: AGENTS.md's safe
// defaults forbid pulling large assets without the developer saying so, and the
// smallest HF-format CTC checkpoint (nvidia/parakeet-ctc-0.6b) is a 2.4 GB
// `model.safetensors`. Random weights exercise the module MATH exactly as
// pretrained ones would — every stage, every mask, every bias — but they make
// the decoded TEXT meaningless, so this gate asserts LOGITS (rel-L2) and TOKEN
// IDS (exact) and deliberately claims NOTHING about a transcript. The pretrained
// arm is `parakeet_ctc_pretrained_checkpoint`, below: it is wired, one env var
// away, and SKIPPED (not silently passed) when the checkpoint is absent.
//
// The traced attention path for the dump is `sdpa` (recorded in the manifest,
// per T0 "trace the execution, not just the code"). It returns exactly ZERO for
// a fully-masked query row, which is what vt::AttentionRelPos documents and what
// the forward reproduces; the padded second batch row gates that.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vt/backend.h"

namespace {

using vllm::multimodal::LoadParakeetForCTC;
using vllm::multimodal::ParakeetCtcGreedyCollapse;
using vllm::multimodal::ParakeetEncoderCapture;
using vllm::multimodal::ParakeetEncoderConfig;
using vllm::multimodal::ParakeetEncoderForward;
using vllm::multimodal::ParakeetEncoderLayerWeights;
using vllm::multimodal::ParakeetForCTCForward;
using vllm::multimodal::ParakeetForCTCWeights;
using vllm::multimodal::ParakeetSubsamplingOutputLength;
using vllm::multimodal::ParakeetSubsamplingWeights;

std::string Fix() { return std::string(PARAKEET_FIXTURE_DIR); }

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

std::vector<float> W(const std::string& name) {
  return ReadBin<float>(Fix() + "/weights/" + name + ".bin");
}

struct Err {
  double rel_l2 = 0.0;
  double max_abs = 0.0;
};

// Compare only the first `rows` rows of a [*, cols] pair (so a case can gate the
// valid prefix separately from the padded tail).
Err CompareRows(const std::vector<float>& got, const std::vector<float>& ref,
                int64_t offset, int64_t rows, int64_t cols) {
  double num = 0.0, den = 0.0, mx = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t cidx = 0; cidx < cols; ++cidx) {
      const size_t gi = static_cast<size_t>(r) * cols + cidx;
      const size_t ri = static_cast<size_t>(offset + r) * cols + cidx;
      REQUIRE(gi < got.size());
      REQUIRE(ri < ref.size());
      const double d = static_cast<double>(got[gi]) - static_cast<double>(ref[ri]);
      num += d * d;
      den += static_cast<double>(ref[ri]) * static_cast<double>(ref[ri]);
      if (std::abs(d) > mx) mx = std::abs(d);
    }
  }
  return Err{std::sqrt(num / (den + 1e-30)), mx};
}

ParakeetEncoderConfig ConfigFromManifest(const nlohmann::json& m) {
  const nlohmann::json& c = m.at("config");
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
  cfg.subsampling_conv_kernel_size =
      c.at("subsampling_conv_kernel_size").get<int64_t>();
  cfg.subsampling_conv_stride = c.at("subsampling_conv_stride").get<int64_t>();
  cfg.max_position_embeddings = c.at("max_position_embeddings").get<int64_t>();
  cfg.scale_input = c.at("scale_input").get<bool>();
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.pad_token_id = c.at("pad_token_id").get<int32_t>();
  return cfg;
}

// Reads the dumped `state_dict()` into the weight structs. The key map is the
// one src/vllm/model_executor/models/parakeet_weights.cpp documents; loading it
// here from loose .bin files keeps the fixture free of a safetensors container.
ParakeetForCTCWeights LoadFixtureWeights(const ParakeetEncoderConfig& cfg) {
  ParakeetForCTCWeights w;
  ParakeetSubsamplingWeights& sub = w.encoder.subsampling;
  sub.conv0_w = W("encoder.subsampling.layers.0.weight");
  sub.conv0_b = W("encoder.subsampling.layers.0.bias");
  for (int64_t i = 0; i + 1 < cfg.num_subsampling_layers(); ++i) {
    ParakeetSubsamplingWeights::Stage stage;
    const std::string dw = "encoder.subsampling.layers." + std::to_string(2 + 3 * i);
    const std::string pw = "encoder.subsampling.layers." + std::to_string(3 + 3 * i);
    stage.depthwise_w = W(dw + ".weight");
    stage.depthwise_b = W(dw + ".bias");
    stage.pointwise_w = W(pw + ".weight");
    stage.pointwise_b = W(pw + ".bias");
    sub.stages.push_back(std::move(stage));
  }
  sub.linear_w = W("encoder.subsampling.linear.weight");
  sub.linear_b = W("encoder.subsampling.linear.bias");

  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = "encoder.layers." + std::to_string(l) + ".";
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
    w.encoder.layers.push_back(std::move(lw));
  }
  w.ctc_head_w = W("ctc_head.weight");
  w.ctc_head_b = W("ctc_head.bias");
  return w;
}

nlohmann::json Manifest() {
  std::ifstream f(Fix() + "/manifest.json");
  REQUIRE_MESSAGE(f.good(), "cannot open the parakeet fixture manifest");
  nlohmann::json m;
  f >> m;
  return m;
}

}  // namespace

TEST_CASE("parakeet_ctc_matches_hf_oracle_stage_by_stage") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const nlohmann::json m = Manifest();
  // The dump must have come from the path we mirror, not from `eager`.
  CHECK(m.at("provenance").at("attn_implementation").get<std::string>() == "sdpa");
  CHECK(m.at("provenance").at("transformers").get<std::string>() == "5.3.0");
  CHECK(m.at("provenance").at("pretrained").get<bool>() == false);

  const ParakeetEncoderConfig cfg = ConfigFromManifest(m);
  const ParakeetForCTCWeights w = LoadFixtureWeights(cfg);

  const int64_t batch = m.at("batch").get<int64_t>();
  const std::vector<int64_t> lengths = m.at("input_lengths").get<std::vector<int64_t>>();
  const int64_t frames = m.at("shapes").at("input_features").at(1).get<int64_t>();
  const int64_t out_frames = m.at("shapes").at("logits").at(1).get<int64_t>();

  const std::vector<float> features = ReadBin<float>(Fix() + "/input_features.bin");
  const std::vector<float> ref_sub = ReadBin<float>(Fix() + "/subsampling_out.bin");
  const std::vector<float> ref_pos = ReadBin<float>(Fix() + "/pos_embed.bin");
  const std::vector<float> ref_ff1 = ReadBin<float>(Fix() + "/l0_ff1.bin");
  const std::vector<float> ref_attn = ReadBin<float>(Fix() + "/l0_attn.bin");
  const std::vector<float> ref_conv = ReadBin<float>(Fix() + "/l0_conv.bin");
  const std::vector<float> ref_b0 = ReadBin<float>(Fix() + "/block0_out.bin");
  const std::vector<float> ref_b1 = ReadBin<float>(Fix() + "/block1_out.bin");
  const std::vector<float> ref_hidden = ReadBin<float>(Fix() + "/last_hidden_state.bin");
  const std::vector<float> ref_logits = ReadBin<float>(Fix() + "/logits.bin");
  const std::vector<int32_t> ref_ids = ReadBin<int32_t>(Fix() + "/greedy_ids.bin");
  const auto ref_collapsed =
      m.at("collapsed_ids").get<std::vector<std::vector<int32_t>>>();

  const int64_t hidden = cfg.hidden_size;
  for (int64_t item = 0; item < batch; ++item) {
    CAPTURE(item);
    const std::vector<float> clip(
        features.begin() + static_cast<ptrdiff_t>(item * frames * cfg.num_mel_bins),
        features.begin() + static_cast<ptrdiff_t>((item + 1) * frames * cfg.num_mel_bins));

    ParakeetEncoderCapture cap;
    int64_t valid_rows = 0;
    const std::vector<float> got_hidden =
        ParakeetEncoderForward(clip, frames, lengths[static_cast<size_t>(item)],
                               w.encoder, cfg, cpu, &valid_rows, &cap);
    REQUIRE(static_cast<int64_t>(got_hidden.size()) == out_frames * hidden);
    CHECK(valid_rows ==
          ParakeetSubsamplingOutputLength(lengths[static_cast<size_t>(item)], cfg));

    struct Stage {
      const char* name;
      const std::vector<float>* got;
      const std::vector<float>* ref;
      int64_t cols;
    };
    const std::vector<Stage> stages = {
        {"subsampling", &cap.subsampling_out, &ref_sub, hidden},
        {"layer0_ff1", &cap.layer0_ff1, &ref_ff1, hidden},
        {"layer0_attn", &cap.layer0_attn, &ref_attn, hidden},
        {"layer0_conv", &cap.layer0_conv, &ref_conv, hidden},
        {"block0", &cap.block0_out, &ref_b0, hidden},
        {"block1", &got_hidden, &ref_b1, hidden},
        {"last_hidden_state", &got_hidden, &ref_hidden, hidden},
    };
    for (const Stage& s : stages) {
      // The FULL padded extent, not just the valid prefix: the traced sdpa path
      // makes even the padded rows well defined, so anything less would be a
      // weaker gate than the oracle supports.
      const Err e = CompareRows(*s.got, *s.ref, item * out_frames, out_frames, s.cols);
      char note[192];
      std::snprintf(note, sizeof(note), "item %lld stage %s: rel_l2=%.3e max_abs=%.3e",
                    static_cast<long long>(item), s.name, e.rel_l2, e.max_abs);
      MESSAGE(note);
      CHECK(e.rel_l2 < 1e-5);
    }

    // The relative-position table is batch-independent upstream (it depends only
    // on the sequence length), so every item sees the same rows.
    const Err pe = CompareRows(cap.pos_embed, ref_pos, item * (2 * out_frames - 1),
                               2 * out_frames - 1, hidden);
    MESSAGE("item ", item, " pos_embed: rel_l2=", pe.rel_l2, " max_abs=", pe.max_abs);
    CHECK(pe.max_abs < 1e-6);

    const auto out = ParakeetForCTCForward(clip, frames,
                                           lengths[static_cast<size_t>(item)], w, cfg, cpu);
    CHECK(out.num_output_frames == out_frames);
    const Err le = CompareRows(out.logits, ref_logits, item * out_frames, out_frames,
                               cfg.vocab_size);
    MESSAGE("item ", item, " logits: rel_l2=", le.rel_l2, " max_abs=", le.max_abs);
    CHECK(le.rel_l2 < 1e-5);

    // The DISCRETE outputs are exact, not approximate: greedy argmax ids
    // (:796-801) and the collapse (tokenization_parakeet.py:38-42).
    const std::vector<int32_t> expect_ids(
        ref_ids.begin() + static_cast<ptrdiff_t>(item * out_frames),
        ref_ids.begin() + static_cast<ptrdiff_t>((item + 1) * out_frames));
    CHECK(out.greedy_ids == expect_ids);
    CHECK(out.token_ids == ref_collapsed[static_cast<size_t>(item)]);
    CHECK(ParakeetCtcGreedyCollapse(expect_ids, cfg.pad_token_id) ==
          ref_collapsed[static_cast<size_t>(item)]);
  }
}

// The pretrained arm. Point VLLM_PARAKEET_CKPT at a directory holding an
// HF-format `ParakeetForCTC` (config.json + model.safetensors, e.g.
// nvidia/parakeet-ctc-0.6b) to exercise the real loader end to end. SKIPPED, not
// passed, when the variable is unset — no pretrained transcript is claimed by
// this row (see the honesty note at the top of this file).
TEST_CASE("parakeet_ctc_pretrained_checkpoint") {
  const char* dir = std::getenv("VLLM_PARAKEET_CKPT");
  if (dir == nullptr) {
    MESSAGE(
        "SKIP: set VLLM_PARAKEET_CKPT to an HF-format ParakeetForCTC directory "
        "to run the pretrained-checkpoint arm");
    return;
  }
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  ParakeetEncoderConfig cfg;
  const ParakeetForCTCWeights w = LoadParakeetForCTC(dir, &cfg);
  CHECK(static_cast<int64_t>(w.encoder.layers.size()) == cfg.num_hidden_layers);
  CHECK(static_cast<int64_t>(w.ctc_head_b.size()) == cfg.vocab_size);

  // One second of silence is enough to prove the whole path runs on real weights
  // and produces well-formed output; a transcript claim needs audio and a
  // tokenizer, which this row does not own.
  const int64_t frames = 101;
  const std::vector<float> features(
      static_cast<size_t>(frames * cfg.num_mel_bins), 0.0f);
  const auto out = ParakeetForCTCForward(features, frames, frames, w, cfg, cpu);
  CHECK(out.num_output_frames == ParakeetSubsamplingOutputLength(frames, cfg));
  CHECK(static_cast<int64_t>(out.logits.size()) ==
        out.num_output_frames * cfg.vocab_size);
  for (int32_t id : out.token_ids) {
    CHECK(id >= 0);
    CHECK(id < cfg.vocab_size);
    CHECK(id != cfg.pad_token_id);
  }
}
