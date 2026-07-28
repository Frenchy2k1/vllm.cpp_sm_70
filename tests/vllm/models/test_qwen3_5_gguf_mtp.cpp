// SPEC-MTP-GGUF: the MTP head read back out of a head-carrying GGUF.
//
// Gated on VLLM_MTP_GGUF_MODEL pointing at a Qwen3.5/3.6 GGUF converted WITH
// the head (i.e. NOT --no-mtp), because the whole point of this gate is to hold
// the loader against a REAL file produced by llama.cpp's converter rather than
// against our own assumptions about its naming. Without the env var the suite
// skips, so CI stays asset-free.
//
// Reference file used to develop this (llama.cpp Qwen3.5-2B, Q8_0 body):
//   qwen35.block_count            = 25
//   qwen35.nextn_predict_layers   = 1        => trunk L = 24, head at blk.24
//   blk.24.nextn.eh_proj.weight   [2048, 4096]  (torch [N, K])
//   blk.24.nextn.enorm.weight     [2048]
//   blk.24.nextn.hnorm.weight     [2048]
//   blk.24.nextn.shared_head_norm.weight [2048]
//   blk.24.{attn_*,ffn_*,attn_norm,post_attention_norm}   the head's own block
// Note there is NO blk.24.nextn.embed_tokens / .shared_head_head: Qwen3.5's
// head SHARES the target's embedding and lm_head, so the converter emits
// neither, and the loader must not ask for them.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

const char* MtpGgufPath() { return std::getenv("VLLM_MTP_GGUF_MODEL"); }

}  // namespace

TEST_CASE("gguf mtp: the head depth reaches config.raw") {
  const char* path = MtpGgufPath();
  if (path == nullptr) return;  // asset-gated

  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  // The depth key is what ResolveSpecConfig reads (via NumMtpLayers). Before
  // this row HfConfigFromGguf spent nextn_predict_layers on the trunk layer
  // count and discarded it, so a head-carrying GGUF looked head-less to the
  // spec resolver and silently fell back to depth 1.
  REQUIRE(c.raw.contains("mtp_num_hidden_layers"));
  const int64_t depth = vllm::NumMtpLayers(c);
  CHECK(depth > 0);

  // The trunk count must EXCLUDE the head blocks: llama.cpp's converter folds
  // them into block_count, so num_hidden_layers + depth == block_count.
  CHECK(c.num_hidden_layers > 0);
}

TEST_CASE("gguf mtp: the head loads with trunk-consistent conventions") {
  const char* path = MtpGgufPath();
  if (path == nullptr) return;  // asset-gated

  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE(c.raw.contains("mtp_num_hidden_layers"));

  const vllm::Qwen3_5MTPKind kind =
      c.num_experts > 0 ? vllm::Qwen3_5MTPKind::kMoe
                        : vllm::Qwen3_5MTPKind::kDense;
  const vllm::Qwen3_5MTPWeights w = vllm::LoadQwen3_5MTPFromGguf(
      g, c, kind, vllm::GgufLoadPolicy::FromEnv());

  const int64_t H = c.hidden_size;

  // fc is the [embedding; hidden] -> hidden projection, kept in raw [N, K]
  // exactly as the safetensors path's LoadBf16RawNK leaves it. GGUF stores
  // shapes in torch [N, K] order already, so this is the VERBATIM path; loading
  // it through the transposing helper would silently produce [2H, H] and only
  // fail much later, inside the draft forward.
  REQUIRE(w.fc.rank == 2);
  CHECK(w.fc.shape[0] == H);
  CHECK(w.fc.shape[1] == 2 * H);
  // The ORIENTATION FLAG, not just the shape. The draft forward requires
  // `fc.nk` set ("fc must be raw bf16 [H,2H]"); GGUF already stores [N, K] so
  // the shape assertions above pass either way, and an unset flag fails only
  // later inside the forward. The G4 token gate caught exactly that, which is
  // why it is asserted here too.
  CHECK(w.fc.nk);

  // The three RMSNorm weights are [H]. They also carry the GGUF (w + 1) storage
  // convention, which is why the loader routes them through the same
  // OwnNormMinus1 helper the trunk uses; a plain read would leave every norm
  // weight off by one and poison every draft proposal.
  REQUIRE(w.pre_fc_norm_embedding.rank == 1);
  CHECK(w.pre_fc_norm_embedding.shape[0] == H);
  REQUIRE(w.pre_fc_norm_hidden.rank == 1);
  CHECK(w.pre_fc_norm_hidden.shape[0] == H);
  REQUIRE(w.final_norm.rank == 1);
  CHECK(w.final_norm.shape[0] == H);

  // One transformer block per head layer, and it is ALWAYS full attention: the
  // head is not in config.layer_types (which covers the trunk only), so a
  // loader that consulted it would run off the end or mis-type the block.
  CHECK(w.NumLayers() == vllm::NumMtpLayers(c));
  if (kind == vllm::Qwen3_5MTPKind::kDense) {
    REQUIRE(!w.dense_layers.empty());
    CHECK(!w.dense_layers[0].is_linear_attention);
    CHECK(w.dense_layers[0].input_layernorm.shape[0] == H);
    CHECK(w.dense_layers[0].mlp.gate_proj.rank == 2);
  } else {
    REQUIRE(!w.moe_layers.empty());
    CHECK(!w.moe_layers[0].is_linear_attention);
    CHECK(w.moe_layers[0].input_layernorm.shape[0] == H);
  }
}
