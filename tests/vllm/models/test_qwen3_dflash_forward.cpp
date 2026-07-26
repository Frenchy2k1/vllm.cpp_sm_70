// vllm.cpp original. DFlash draft model forward + fc + attn-mode-resolution unit
// tests (SPEC-DFLASH D2, DF-DRAFT-MODEL). Ported semantics: qwen3_dflash.py
// @ 555967922 (DFlashQwen3Model.forward :621-640, combine_hidden_states :750-770,
// _resolve_layer_attention :86-146). These run the CONTEXT-FREE block forward on
// synthetic weights and pin the D2 load-bearing invariants:
//   (1) the forward runs and returns finite [T, draft_vocab] logits;
//   (2) RED — flipping the FULL layer to causal CHANGES the logits (the
//       non-causal in-block mask is load-bearing end-to-end);
//   (3) per-request BLOCK isolation via cu_seqlens (a block's logits do not depend
//       on another block's tokens);
//   (4) the fc aux-combine matches an independent reference, and RED — reversing
//       the tap column order changes the output (wrong tap order is caught);
//   (5) _resolve_layer_attention maps 4xSWA + 1xfull correctly.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using namespace vllm;

namespace {
vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// Small deterministic bf16 weight: value(i) = amp * sin(seed + 0.7*i).
OwnedTensor MkBf16(const std::vector<int64_t>& shape, double seed, double amp, bool nk) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  t.nk = nk;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return t;
}

struct Dims {
  int64_t H = 4, Hq = 2, Hkv = 1, Dh = 2, I = 6, vocab = 8, layers = 2, taps = 2;
};

HfConfig MakeConfig(const Dims& dm) {
  HfConfig c;
  c.hidden_size = dm.H;
  c.num_attention_heads = dm.Hq;
  c.num_key_value_heads = dm.Hkv;
  c.head_dim = dm.Dh;
  c.rotary_dim = dm.Dh;  // full rope over head_dim
  c.rope_theta = 10000.0;
  c.intermediate_size = dm.I;
  c.vocab_size = dm.vocab;
  c.num_hidden_layers = dm.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 64;  // >> block, so the SWA layer is causal-over-block
  c.layer_types = {"sliding_attention", "full_attention"};
  c.raw = nlohmann::json::object();
  c.raw["dflash_config"] = {{"mask_token_id", 7}};
  return c;
}

Qwen3DFlashWeights MakeWeights(const Dims& dm) {
  Qwen3DFlashWeights w;
  w.num_taps = dm.taps;
  w.mask_token_id = 7;
  w.draft_vocab_size = dm.vocab;
  const int64_t qdim = dm.Hq * dm.Dh, kdim = dm.Hkv * dm.Dh;
  w.embed_tokens = MkBf16({dm.vocab, dm.H}, 0.1, 0.3, false);
  w.fc = MkBf16({dm.H, dm.H * dm.taps}, 0.2, 0.2, true);
  w.hidden_norm = MkBf16({dm.H}, 0.3, 0.5, false);
  w.final_norm = MkBf16({dm.H}, 0.4, 0.5, false);
  w.lm_head = MkBf16({dm.vocab, dm.H}, 0.5, 0.3, true);
  const std::vector<Qwen3DFlashLayerAttnMode> modes = {{true, 64}, {false, 0}};
  for (int64_t l = 0; l < dm.layers; ++l) {
    Qwen3DFlashLayerWeights lw;
    const double s = 1.0 + static_cast<double>(l);
    lw.input_layernorm = MkBf16({dm.H}, s + 0.01, 0.5, false);
    lw.post_attention_layernorm = MkBf16({dm.H}, s + 0.02, 0.5, false);
    lw.qkv_proj = MkBf16({qdim + 2 * kdim, dm.H}, s + 0.03, 0.25, true);
    lw.o_proj = MkBf16({dm.H, qdim}, s + 0.04, 0.25, true);
    lw.q_norm = MkBf16({dm.Dh}, s + 0.05, 0.5, false);
    lw.k_norm = MkBf16({dm.Dh}, s + 0.06, 0.5, false);
    lw.gate_up_proj = MkBf16({2 * dm.I, dm.H}, s + 0.07, 0.2, true);
    lw.down_proj = MkBf16({dm.H, dm.I}, s + 0.08, 0.2, true);
    lw.attn_mode = modes[static_cast<size_t>(l)];
    w.layers.push_back(std::move(lw));
  }
  return w;
}
}  // namespace

TEST_CASE("qwen3_dflash: context-free block forward runs and returns finite logits") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  // One (1+k) block of 3 tokens: anchor + 2 mask tokens.
  std::vector<int32_t> ids = {2, 7, 7};
  std::vector<int32_t> pos = {0, 1, 2};
  std::vector<int32_t> cu = {0, 3};
  std::vector<float> logits =
      Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, cfg, q);
  REQUIRE(logits.size() == static_cast<size_t>(3 * dm.vocab));
  for (float v : logits) CHECK(std::isfinite(v));
}

TEST_CASE("qwen3_dflash RED: full-layer causal flip changes the block logits") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights base = MakeWeights(dm);
  vt::Queue q = Cpu();
  std::vector<int32_t> ids = {2, 7, 7};
  std::vector<int32_t> pos = {0, 1, 2};
  std::vector<int32_t> cu = {0, 3};
  std::vector<float> nc = Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, base, cfg, q);

  // Corrupt: force the FULL-attention layer (index 1) to be CAUSAL — the exact
  // bug the D2 gate must catch (a causal-instead-of-non-causal full layer).
  Qwen3DFlashWeights bad = MakeWeights(dm);
  bad.layers[1].attn_mode.causal = true;
  std::vector<float> c = Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, bad, cfg, q);

  double maxdiff = 0.0;
  for (size_t i = 0; i < nc.size(); ++i)
    maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(nc[i] - c[i])));
  // The non-causal mask is load-bearing end-to-end: flipping the full layer to
  // causal shifts the logits by a real, well-above-noise amount (measured ~1e-2
  // on these synthetic weights); a byte-identical rerun would give exactly 0.
  CHECK(maxdiff > 1e-3);
}

TEST_CASE("qwen3_dflash: per-request BLOCK isolation via cu_seqlens") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  // Two identical blocks: their per-block logits must be identical.
  std::vector<int32_t> ids2 = {2, 7, 7, 2, 7, 7};
  std::vector<int32_t> pos2 = {0, 1, 2, 0, 1, 2};
  std::vector<int32_t> cu2 = {0, 3, 6};
  std::vector<float> l2 = Qwen3DFlashModel::ForwardBlockLogits(ids2, pos2, cu2, w, cfg, q);
  for (int64_t r = 0; r < 3; ++r)
    for (int64_t j = 0; j < dm.vocab; ++j)
      CHECK(l2[static_cast<size_t>(r * dm.vocab + j)] ==
            doctest::Approx(l2[static_cast<size_t>((r + 3) * dm.vocab + j)]));

  // Changing block 1's anchor token must NOT change block 0's logits.
  std::vector<int32_t> ids3 = {2, 7, 7, 5, 7, 7};
  std::vector<float> l3 = Qwen3DFlashModel::ForwardBlockLogits(ids3, pos2, cu2, w, cfg, q);
  for (int64_t r = 0; r < 3; ++r)
    for (int64_t j = 0; j < dm.vocab; ++j)
      CHECK(l3[static_cast<size_t>(r * dm.vocab + j)] ==
            doctest::Approx(l2[static_cast<size_t>(r * dm.vocab + j)]));
}

TEST_CASE("qwen3_dflash fc: combine matches reference, RED on reversed tap order") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t T = 2, Fin = dm.H * dm.taps;
  std::vector<float> aux(static_cast<size_t>(T * Fin));
  for (size_t i = 0; i < aux.size(); ++i) aux[i] = 0.2f * std::sin(0.9 + 0.3 * static_cast<double>(i));
  std::vector<float> comb = Qwen3DFlashModel::CombineAuxFeatures(aux, T, w, cfg, q);
  REQUIRE(comb.size() == static_cast<size_t>(T * dm.H));

  // Independent reference: bf16(aux) @ bf16(fc)^T, fc is [H, H*taps] nk.
  const auto* fcp = reinterpret_cast<const uint16_t*>(w.fc.bytes.data());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t o = 0; o < dm.H; ++o) {
      float acc = 0.0f;
      for (int64_t k = 0; k < Fin; ++k)
        acc += vt::BF16ToF32(vt::F32ToBF16(aux[static_cast<size_t>(t * Fin + k)])) *
               vt::BF16ToF32(fcp[o * Fin + k]);
      // bf16 GEMM accumulation differs slightly; envelope check.
      CHECK(comb[static_cast<size_t>(t * dm.H + o)] == doctest::Approx(acc).epsilon(0.05));
    }

  // RED: reverse the two H-wide tap blocks in aux -> different combined feature.
  std::vector<float> aux_rev(aux.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t k = 0; k < Fin; ++k) {
      const int64_t half = dm.H;
      const int64_t src = (k < half) ? (k + half) : (k - half);
      aux_rev[static_cast<size_t>(t * Fin + k)] = aux[static_cast<size_t>(t * Fin + src)];
    }
  std::vector<float> comb_rev = Qwen3DFlashModel::CombineAuxFeatures(aux_rev, T, w, cfg, q);
  double maxdiff = 0.0;
  for (size_t i = 0; i < comb.size(); ++i)
    maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(comb[i] - comb_rev[i])));
  CHECK(maxdiff > 1e-3);  // wrong tap concat order is caught
}

TEST_CASE("qwen3_dflash: _resolve_layer_attention 4xSWA + 1xfull") {
  HfConfig c;
  c.num_hidden_layers = 5;
  c.head_dim = 128;
  c.sliding_window = 2048;
  c.layer_types = {"sliding_attention", "sliding_attention", "sliding_attention",
                   "sliding_attention", "full_attention"};
  c.raw = nlohmann::json::object();
  std::vector<Qwen3DFlashLayerAttnMode> modes = ResolveQwen3DFlashAttnModes(c);
  REQUIRE(modes.size() == 5);
  for (int i = 0; i < 4; ++i) {
    CHECK(modes[static_cast<size_t>(i)].causal);              // SWA -> causal
    CHECK(modes[static_cast<size_t>(i)].sliding_window == 2048);
  }
  CHECK_FALSE(modes[4].causal);           // full -> NON-causal (the new primitive)
  CHECK(modes[4].sliding_window == 0);
}
