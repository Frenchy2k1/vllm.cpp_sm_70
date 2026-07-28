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

#include <cstdlib>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
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
