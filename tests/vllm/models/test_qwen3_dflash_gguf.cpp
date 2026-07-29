// SPEC-DFLASH-GGUF GD1-GD2: the DFlash draft read out of a `dflash`-arch GGUF.
//
// Gated on VLLM_DFLASH_GGUF_MODEL pointing at one (e.g. the published
// Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp Q4_K_M). Absent => skip, so
// CI stays asset-free.
//
// This gate exists for the two conventions that NO shape or name check can
// catch, both established by reading llama.cpp's converter rather than by
// inspecting values (a value-distribution check was run first and was
// ambiguous - draft and trunk norms both cluster near ~1):
//
//   1. `dflash.target_layers` is written +1-OFFSET
//      (DFlashModel.set_gguf_parameters: `[i + 1 for i in target_layer_ids]`),
//      so the rebuilt ids must be decremented. Get this wrong and num_taps is
//      still right, every tap is still a valid layer index, every shape still
//      matches - and the draft silently reads the wrong target layers.
//   2. Norms are stored RAW. `DFlashModel` extends `Qwen3Model`, NOT
//      `Qwen3NextModel` where the `(w + 1)` shift lives, so unlike the Qwen3.5
//      trunk and the MTP head these must NOT be un-shifted.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dflash_gguf.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

const char* DflashGgufPath() { return std::getenv("VLLM_DFLASH_GGUF_MODEL"); }

}  // namespace

TEST_CASE("dflash gguf: config undoes the +1 target-layer offset") {
  const char* path = DflashGgufPath();
  if (path == nullptr) return;  // asset-gated

  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig c = vllm::MakeDflashGgufConfig(g);

  REQUIRE(c.raw.contains("dflash_config"));
  const auto& dcfg = c.raw.at("dflash_config");
  REQUIRE(dcfg.contains("target_layer_ids"));
  const std::vector<int64_t> ids =
      dcfg.at("target_layer_ids").get<std::vector<int64_t>>();
  REQUIRE(!ids.empty());

  // Every rebuilt id must be EXACTLY one less than the stored KV. Reading the
  // KV back here rather than hard-coding values keeps the gate valid for any
  // dflash draft, not just the 27B one it was written against.
  const vllm::GgufValue* tl = g.FindKv("dflash.target_layers");
  REQUIRE(tl != nullptr);
  const auto& stored = std::get<vllm::GgufArray>(tl->v).elems;
  REQUIRE(stored.size() == ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    int64_t raw = 0;
    switch (stored[i].TypeId()) {
      case vllm::kGgufI32: raw = std::get<int32_t>(stored[i].v); break;
      case vllm::kGgufU32: raw = std::get<uint32_t>(stored[i].v); break;
      case vllm::kGgufI64: raw = std::get<int64_t>(stored[i].v); break;
      default: raw = -1; break;
    }
    REQUIRE(raw >= 1);
    CHECK(ids[i] == raw - 1);
  }

  // block_size and the mask token are the other two config-only carriers.
  CHECK(c.raw.at("block_size").get<int64_t>() > 0);
  CHECK(dcfg.at("mask_token_id").get<int64_t>() > 0);

  // vocab_size stays 0 on purpose: the draft carries no vocab key and shares the
  // target's lm_head, so the caller fills it. Asserting it is 0 pins the
  // contract - a loader that "helpfully" invented one would break that sharing.
  CHECK(c.vocab_size == 0);

  // layer_types come from the sliding-window pattern and must cover every block.
  CHECK(static_cast<int64_t>(c.layer_types.size()) == c.num_hidden_layers);
}

TEST_CASE("dflash gguf: weights load with the raw-norm convention") {
  const char* path = DflashGgufPath();
  if (path == nullptr) return;  // asset-gated

  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig c = vllm::MakeDflashGgufConfig(g);
  const auto& dcfg = c.raw.at("dflash_config");
  const int64_t num_taps =
      static_cast<int64_t>(dcfg.at("target_layer_ids").size());
  const int32_t mask_id = dcfg.at("mask_token_id").get<int32_t>();

  const vllm::Qwen3DFlashWeights w =
      vllm::LoadQwen3DFlashFromGguf(g, c, num_taps, mask_id);

  const int64_t H = c.hidden_size;
  const int64_t I = c.intermediate_size;
  const int64_t Dh = c.head_dim;

  // fc fuses the num_taps target hidden states: [H, H*num_taps], raw-NK. GGUF
  // already stores torch [N, K], so this is the verbatim orientation - and the
  // nk FLAG must be set, which is the exact defect that cost SPEC-MTP-GGUF a
  // debug cycle (shape right, flag wrong, fails only inside the forward).
  REQUIRE(w.fc.rank == 2);
  CHECK(w.fc.shape[0] == H);
  CHECK(w.fc.shape[1] == H * num_taps);
  CHECK(w.fc.nk);
  CHECK(w.num_taps == num_taps);
  CHECK(w.mask_token_id == mask_id);

  REQUIRE(w.hidden_norm.rank == 1);
  CHECK(w.hidden_norm.shape[0] == H);
  REQUIRE(w.final_norm.rank == 1);
  CHECK(w.final_norm.shape[0] == H);

  // The draft shares the target's embedding and lm_head, so the GGUF carries
  // neither and this loader must leave both EMPTY for the caller to fill.
  CHECK(w.embed_tokens.Empty());
  CHECK(w.lm_head.Empty());

  REQUIRE(static_cast<int64_t>(w.layers.size()) == c.num_hidden_layers);
  const auto& l0 = w.layers[0];
  // qkv is the row-concatenation of the three separate GGUF tensors.
  const int64_t qkv_rows =
      c.num_attention_heads * Dh + 2 * c.num_key_value_heads * Dh;
  REQUIRE(l0.qkv_proj.rank == 2);
  CHECK(l0.qkv_proj.shape[0] == qkv_rows);
  CHECK(l0.qkv_proj.shape[1] == H);
  CHECK(l0.qkv_proj.nk);
  // gate_up is likewise gate|up concatenated.
  REQUIRE(l0.gate_up_proj.rank == 2);
  CHECK(l0.gate_up_proj.shape[0] == 2 * I);
  CHECK(l0.gate_up_proj.shape[1] == H);
  REQUIRE(l0.o_proj.rank == 2);
  CHECK(l0.o_proj.shape[0] == H);
  REQUIRE(l0.down_proj.rank == 2);
  CHECK(l0.down_proj.shape[0] == H);
  CHECK(l0.q_norm.shape[0] == Dh);
  CHECK(l0.k_norm.shape[0] == Dh);
  CHECK(l0.input_layernorm.shape[0] == H);
  CHECK(l0.post_attention_layernorm.shape[0] == H);
}

// ── Gate 2: CROSS-FORMAT DRAFT EQUIVALENCE, in weight space ────────────────
//
// The spike wrote this as "the SAME draft, loaded from safetensors and from an
// F16 GGUF, produces bit-identical Qwen3DFlashWeights ... the strongest cheap
// gate and needs no GPU", then recorded it `NOT APPLICABLE` on the belief that
// the only published GGUF of this draft was Q4_K_M, which cannot be bit-equal
// to bf16 by construction. That belief was wrong:
// `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` also publishes
// `Qwen3.6-27B-DFlash-BF16.gguf` (ggml type 30) beside the quantized ladder, so
// the gate IS available and is enabled here.
//
// It is the load-time half of the axis-A bar and it is worth more than a shape
// check: an UNQUANTIZED GGUF exercises the SAME MapName / dequant-to-bf16 /
// LoadQwen3DFlash path a Q4_K_M draft does, so byte equality against the
// safetensors sibling rules out every STRUCTURAL way the GGUF arm could differ
// - wrong tensor mapped, wrong orientation, a stray norm shift, a mis-scaled
// tensor - and leaves the quantization error itself as the only thing a
// quantized draft can change. Without this, an acceptance-rate difference
// between the two containers has two candidate causes; with it, one.
//
// Asset-gated on BOTH `VLLM_DFLASH_GGUF_BF16_MODEL` (an UNQUANTIZED dflash
// GGUF) and `VLLM_DFLASH_ST_DIR` (the z-lab safetensors checkpoint dir).
TEST_CASE("dflash gguf: an unquantized GGUF draft is byte-identical to the safetensors draft") {
  const char* gguf_path = std::getenv("VLLM_DFLASH_GGUF_BF16_MODEL");
  const char* st_dir = std::getenv("VLLM_DFLASH_ST_DIR");
  if (gguf_path == nullptr || st_dir == nullptr) return;  // asset-gated

  vllm::GgufFile g = vllm::GgufFile::Open(gguf_path);
  const vllm::HfConfig c = vllm::MakeDflashGgufConfig(g);
  const auto& dcfg = c.raw.at("dflash_config");
  const int64_t num_taps =
      static_cast<int64_t>(dcfg.at("target_layer_ids").size());
  const int32_t mask_id = dcfg.at("mask_token_id").get<int32_t>();

  std::vector<std::string> shard_paths;
  for (const auto& e : std::filesystem::directory_iterator(st_dir)) {
    if (e.path().extension() == ".safetensors")
      shard_paths.push_back(e.path().string());
  }
  REQUIRE_MESSAGE(!shard_paths.empty(),
                  "VLLM_DFLASH_ST_DIR has no *.safetensors: " << st_dir);
  std::sort(shard_paths.begin(), shard_paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(shard_paths.size());
  for (const std::string& p : shard_paths)
    shards.push_back(vllm::SafetensorsFile::Open(p));

  // ONE config for both arms on purpose: this case is about the WEIGHTS. The
  // config half is pinned by the first case in this file plus the real
  // config.json values it is derived from.
  const vllm::Qwen3DFlashWeights a =
      vllm::LoadQwen3DFlashFromGguf(g, c, num_taps, mask_id);
  const vllm::Qwen3DFlashWeights b =
      vllm::LoadQwen3DFlash(shards, c, num_taps, mask_id);

  const auto same = [](const char* what, const vllm::OwnedTensor& x,
                       const vllm::OwnedTensor& y) {
    REQUIRE_MESSAGE(x.rank == y.rank, what << ": rank");
    for (int i = 0; i < x.rank; ++i)
      REQUIRE_MESSAGE(x.shape[i] == y.shape[i], what << ": shape[" << i << "]");
    CHECK_MESSAGE(x.dtype == y.dtype, what << ": dtype");
    CHECK_MESSAGE(x.nk == y.nk, what << ": nk");
    REQUIRE_MESSAGE(x.bytes.size() == y.bytes.size(), what << ": nbytes");
    // Report the FIRST differing byte rather than a bare false: a structural
    // fault lands at a predictable offset, quantization noise does not.
    size_t first_diff = x.bytes.size();
    size_t ndiff = 0;
    for (size_t i = 0; i < x.bytes.size(); ++i) {
      if (x.bytes.data()[i] != y.bytes.data()[i]) {
        if (first_diff == x.bytes.size()) first_diff = i;
        ++ndiff;
      }
    }
    CHECK_MESSAGE(ndiff == 0, what << ": " << ndiff << " of " << x.bytes.size()
                                   << " bytes differ, first at " << first_diff);
  };

  same("fc", a.fc, b.fc);
  same("hidden_norm", a.hidden_norm, b.hidden_norm);
  same("final_norm", a.final_norm, b.final_norm);
  same("mask_embedding", a.mask_embedding, b.mask_embedding);
  CHECK(a.num_taps == b.num_taps);
  CHECK(a.mask_token_id == b.mask_token_id);
  // Both arms leave the SHARED head to the target; neither may invent one.
  CHECK(a.embed_tokens.Empty());
  CHECK(b.embed_tokens.Empty());
  CHECK(a.lm_head.Empty());
  CHECK(b.lm_head.Empty());

  REQUIRE(a.layers.size() == b.layers.size());
  for (size_t i = 0; i < a.layers.size(); ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    const auto& la = a.layers[i];
    const auto& lb = b.layers[i];
    CHECK_MESSAGE(la.attn_mode.causal == lb.attn_mode.causal, p << "attn_mode.causal");
    CHECK_MESSAGE(la.attn_mode.sliding_window == lb.attn_mode.sliding_window,
                  p << "attn_mode.sliding_window");
    same((p + "input_layernorm").c_str(), la.input_layernorm, lb.input_layernorm);
    same((p + "post_attention_layernorm").c_str(), la.post_attention_layernorm,
         lb.post_attention_layernorm);
    same((p + "qkv_proj").c_str(), la.qkv_proj, lb.qkv_proj);
    same((p + "o_proj").c_str(), la.o_proj, lb.o_proj);
    same((p + "q_norm").c_str(), la.q_norm, lb.q_norm);
    same((p + "k_norm").c_str(), la.k_norm, lb.k_norm);
    same((p + "gate_up_proj").c_str(), la.gate_up_proj, lb.gate_up_proj);
    same((p + "down_proj").c_str(), la.down_proj, lb.down_proj);
  }
}
