// Kimi K3 forward — REFUSE-by-name SKELETON (W5). The scaled 93-layer KDA/MLA
// hybrid + 896-expert MoE forward composes primitives that are NOT wired in this
// lane, so both entrypoints VT_CHECK(false, ...) — a forward LOUDLY reports the
// pending brick instead of returning a silent wrong answer, exactly like
// deepseek_v4.cpp. This TU captures the reuse-wiring plan in one place and BUILDS
// clean on CPU.
//
// ─── REUSE-WIRING PLAN (what the real forward composes; NOT implemented here) ──
// Per KimiDecoderLayer (kimi_linear.py:288-378), for each of the 93 layers:
//   1. input_layernorm (fused add+RMSNorm residual, kimi_linear.py:361-365).
//   2. self_attn dispatch by is_kda_layer (kimi_linear.py:304-326):
//      - KDA layer (69 of 93): KimiGatedDeltaNetAttention
//        (kimi_gdn_linear_attn.py:85). REUSES our landed GDN state/conv/chunked-
//        delta/WY machinery (src/vt/cuda/cuda_gdn.cu, gdn_attn.cpp). The KDA
//        DELTA — per-channel [H,D] low-rank decay (f_a_proj/f_b_proj), the
//        sigmoid-gated output norm (FusedRMSNormGated), the 3 separate q/k/v short
//        convs — is NOT-YET-BUILDABLE here; it is the KDA kernel campaign scoped on
//        the Kimi-Linear row (`MODEL-TEXT-kimi-linear-*`, `CLAIM-MLA-DEEPSEEK`).
//      - MLA layer (24 of 93): KimiMLAAttention (kimi_linear.py:180) at the exact
//        DeepSeek-V3 geometry (kv_lora 512 / q_lora 1536 / nope 128 / rope 64).
//        REUSES our landed DeepSeek-MLA block
//        (src/vllm/model_executor/models/deepseek_v2.cpp,
//        src/vllm/model_executor/layers/attention/mla_attention.cpp).
//   3. post_attention_layernorm (kimi_linear.py:376).
//   4. mlp dispatch (kimi_linear.py:328-347): MoE (KimiMoE, :104) with the
//      DeepSeek-style noaux_tc sigmoid router + shared experts, scaled to 896
//      experts / top-16 / 2 shared — REUSES our DeepSeek-style MoE grouped GEMM
//      (src/vt/cuda/cuda_moe.cu) + the Qwen3.6-35B GDN-hybrid-MoE decoder skeleton
//      (src/vllm/model_executor/models/qwen3_5_moe.cpp) as the structural twin; OR
//      dense KimiMLP for the first_k_dense_replace layers.
// Final: norm -> untied lm_head -> logits (kimi_linear.py:457, 635-639).
// Two K3 blockers stay NOT-YET-BUILDABLE: MXFP4 weights (shared DeepSeek-V4 row)
// and the MoonViT-V2 vision merge (W7). Grounding: kimi_linear.py:426-458, 587-598.
#include "vllm/model_executor/models/kimi_k3.h"

#include <cstdint>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

namespace {
constexpr const char* kPending =
    "KimiK3 forward is not yet implemented — the scaled 93-layer KDA/MLA hybrid + "
    "896-expert MoE forward is a REFUSE-by-name skeleton (W5). Its NEW primitives "
    "are NOT-YET-BUILDABLE in this lane: the KDA kernel delta (Kimi-Linear row), "
    "MXFP4 compressed-tensors (shared DeepSeek-V4 MXFP4 row), and the MoonViT-V2 "
    "vision tower (W7). Also, KimiK3ForConditionalGeneration is beyond the pinned "
    "oracle (555967922) so there is no on-box e2e golden. See "
    ".agents/specs/kimi-k3.md §5.";
}  // namespace

std::vector<float> KimiK3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiK3Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices,
    const vllm::TensorParallel* tp) {
  (void)tp;  // REFUSE-by-name skeleton: no forward body to thread tp into
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

ForwardLogits KimiK3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiK3Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices,
    const vllm::TensorParallel* tp) {
  (void)tp;  // REFUSE-by-name skeleton: no forward body to thread tp into
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
