// Kimi-Linear DEVICE forward — REFUSE-by-name (the born-on-the-runner W6/W7 lane).
// The CPU reference forward (`KimiLinearModel::Forward`, the host `!gather_logits`
// path) is now REAL and lives in kimi_linear_forward.cpp — it composes the whole
// 27-layer KDA/NoPE-MLA hybrid + 256-expert MoE from the landed host primitives so
// the ONLY remaining correctness step is the e2e SACRED token golden on GB10
// (spike §8). What stays REFUSE-by-name HERE is `ForwardDevice`: the DEFAULT
// `gather_logits` production/runner forward (the KDA device kernel W3, the absorbed
// MLA decode W4, the grouped-MoE slabs W5, the het-KV runner wiring W6) is not yet
// wired, so it VT_CHECK(false, ...) — a runner forward LOUDLY reports the pending
// brick instead of returning a silent wrong answer (exactly like deepseek_v4 /
// kimi_k3's device stubs), and the model-matrix row stays SPIKE. This TU also
// captures the device reuse-wiring plan in one place and BUILDS clean on CPU.
//
// ─── REUSE-WIRING PLAN (what the real forward composes; NOT implemented here) ──
// Per KimiDecoderLayer (kimi_linear.py:288-378), for each of the 27 layers:
//   1. input_layernorm (fused add+RMSNorm residual, kimi_linear.py:361-365) via the
//      shared vt::FusedChain glue seam.
//   2. self_attn dispatch by is_kda_layer (kimi_linear.py:304-326):
//      - KDA layer (20 of 27): KimiGatedDeltaNetAttention
//        (kimi_gdn_linear_attn.py:85). REUSES our landed GDN state/conv/chunked-
//        delta/WY machinery (src/vt/cuda/cuda_gdn.cu, gdn_attn.cpp) + the KDA host
//        refs (kimi_kda.{h,cpp}, task #173) as the device-kernel oracle. NET-NEW
//        device kernel = spike W3.
//      - MLA layer (7 of 27): KimiMLAAttention (kimi_linear.py:180) at the NoPE
//        Kimi geometry (kv_lora 512 / qk_nope 128 / qk_rope 64 / v 128, q_lora
//        null, rotary_emb=None). REUSES our landed DeepSeek-MLA block
//        (src/vllm/model_executor/layers/attention/mla_attention.cpp) with RoPE
//        SKIPPED (spike W4 NoPE branch).
//   3. post_attention_layernorm (kimi_linear.py:376).
//   4. mlp dispatch (kimi_linear.py:328-347): MoE (KimiMoE, :104) with the
//      DeepSeek-style sigmoid noaux_tc router + e_score_correction_bias + 1 shared
//      expert, 256 experts / top-8 / routed_scaling 2.446 — REUSES deepseek_v2.cpp
//      RunMoeBlock + the merged-GEMM MergedGemmGroup seam (spike W5); OR dense
//      KimiMLP for layer 0 (first_k_dense_replace=1).
// Final: norm -> untied lm_head -> logits (kimi_linear.py:457, 635-639), routed
// through ModelRegistry::Forward (born-on-the-runner, spike W6). Grounding:
// kimi_linear.py:426-458.
#include "vllm/model_executor/models/kimi_linear.h"

#include <cstdint>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

namespace {
constexpr const char* kDevicePending =
    "KimiLinear DEVICE forward (the default gather_logits runner path) is not yet "
    "wired — the born-on-the-runner device forward is the named residual: the KDA "
    "device kernel (W3), the absorbed MLA decode (W4), the grouped-MoE slabs (W5), "
    "and the het-KV runner wiring (W6). The CPU reference forward "
    "(KimiLinearModel::Forward) IS implemented (kimi_linear_forward.cpp); the e2e "
    "SACRED token golden on GB10 is the only remaining correctness step "
    "(W0/W7). See .agents/specs/kimi-linear.md §5/§8.";
}  // namespace

ForwardLogits KimiLinearModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false, kDevicePending);
  return {};
}

}  // namespace vllm
