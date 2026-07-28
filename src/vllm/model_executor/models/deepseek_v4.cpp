// DeepSeek-V4-Flash forward — SKELETON (W1/W2). The full forward composes the
// NEW-primitive stack that W3-W8 port; NONE of it exists yet, so both entrypoints
// VT_CHECK(false, ...) — a forward LOUDLY reports the pending brick instead of
// returning a silent wrong answer. This TU exists so the reuse-wiring is captured
// in one place and the additive skeleton BUILDS clean.
//
// ─── REUSE-WIRING PLAN (what the real forward will compose, W3-W8) ───────────
// Per DeepseekV4DecoderLayer (nvidia/model.py:794-957), for each of the
// `hc_mult` residual streams [T, hc_mult, H]:
//   1. MHC pre-mix (Sinkhorn, hc_attn_fn/base/scale) — NEW (W5), swallows attn_norm.
//   2. 512-wide MLA block (448 NoPE + 64 RoPE): q via wq_a->q_norm->wq_b,
//      kv via wkv->kv_norm, dual-theta mscale-disabled RoPE (common/rope.py),
//      grouped OUTPUT-LoRA wo_a(bmm)->wo_b, per-head attention sink, SWA=128,
//      fp8_ds_mla latent KV. REUSES mla::ForwardMlaAttentionBlock
//      (src/vllm/model_executor/layers/attention/mla_attention.cpp) +
//      vt::MlaDecodeAttention/MlaPrefillAttention; ADDS the geometry/sink/SWA/
//      output-LoRA/DSA seams — W3 (dense-first) + W4 (DSA indexer/compressor).
//   3. MHC post-mix + ffn pre-mix (hc_ffn_*) — NEW (W5), swallows ffn_norm.
//   4. MoE: sqrtsoftplus router + noaux_tc e_score_correction_bias (or `tid2eid`
//      hash on layers 0-2) + clamped SwiGLU shared expert + 256-expert NVFP4
//      FusedMoE fallback. REUSES the DeepSeek-V2 MoE block structure
//      (src/vllm/model_executor/models/deepseek_v2.cpp) + the NVFP4 grouped GEMM
//      (src/vt/cuda/cuda_matmul_nvfp4_sm100.cu); ADDS scoring/hash/clamp — W6.
// Final: MHC head collapse (hc_head_*) -> norm -> untied head -> logits — W7.
// Grounding: vllm/models/deepseek_v4/nvidia/model.py:1080-1148 (DeepseekV4Model).
#include "vllm/model_executor/models/deepseek_v4.h"

#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

namespace {
constexpr const char* kPending =
    "DeepseekV4 forward is not yet implemented — the NEW-primitive stack "
    "(Manifold Hyper-Connections, DSA Lightning-Indexer + Compressor, 512-wide "
    "MLA with grouped output-LoRA, sqrtsoftplus/hash MoE) is the named W3-W8 "
    "residual (see .agents/specs/deepseek-v4-flash.md §5). This TU is W1/W2 "
    "scaffolding: registry stub + config parse + checkpoint-verified loader map.";
}  // namespace

std::vector<float> DeepseekV4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false, kPending);
  return {};
}

ForwardLogits DeepseekV4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false, kPending);
  return {};
}

}  // namespace vllm
