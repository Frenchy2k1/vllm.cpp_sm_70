// Laguna-S-2.1 forward — W3/W4 ASSEMBLY (STUB this increment). The
// `VT_CHECK(false, ...)` stub will be replaced by a REAL `LagunaModel::Forward`
// that COMPOSES the ~85-90% reuse + the three genuinely-NEW small host ops. This
// TU documents the exact composition (per-layer, cited to the reuse source) so
// the W3 port is mechanical.
//
// ─── PER-LAYER COMPOSITION (LagunaDecoderLayer, laguna.py) ────────────────────
//
//   x_res = x
//   h = RMSNorm(x, input_norm)                            [shared vt::RmsNorm]
//
//   // (1) VARIABLE-Q-HEAD GQA attention. num_heads is PER-LAYER
//   //     (48 global / 72 sliding); KV heads 8, head_dim 128 both.
//   //     Riskiest bit — extends the Gemma-4 heterogeneous-per-layer-KV runner
//   //     wiring (gemma4 G1b, task #148) to per-layer Q-head COUNT + GQA group.
//   q = q_proj(h)  [num_heads_l*128]   k = k_proj(h)  v = v_proj(h)  [8*128]
//
//   // (2) DUAL PER-LAYER RoPE, selected by layer_types[l]:
//   //     full_attention -> YaRN cache (theta 500000, factor 128, mscale 1.4852)
//   //       over the PARTIAL first 64 of 128 dims (partial_rotary 0.5); the last
//   //       64 dims pass through unrotated.
//   //       [olmo2_weights.cpp:198-217 BuildOlmo3YarnCache + phi partial-rope]
//   //     sliding_attention -> plain RoPE (theta 10000, full 128-dim).
//   q, k = ApplyRope(q, k, IsGlobalLayer(l) ? yarn_full_64 : plain_slide_128)
//
//   // (3) INTERLEAVED sliding-window(512) causal mask on sliding layers; full
//   //     causal on global layers (1:3 pattern).  [gemma3.cpp is_sliding path]
//   attn = GQA_FA2(q, k, v, mask = IsGlobalLayer(l) ? causal : sliding512)
//
//   // (4) NEW: per-head SOFTPLUS attention OUTPUT gate.
//   //     gate = softplus(g_proj(h).float()).type_as(attn)   [num_heads_l]
//   //     attn = attn.view(num_heads_l, 128) * gate[:, None]  (broadcast over dh)
//   //     [Softplusf @ gemma4_audio.cpp:20; ds4 per-head q-gate is the cousin]
//   o = o_proj(attn.flatten())
//   x = x_res + o
//
//   // (5) FFN: layer 0 dense SwiGLU MLP (mlp_only_layers=[0]); layers 1..47 MoE.
//   x_res2 = x
//   h2 = RMSNorm(x, post_attn_norm)
//   if IsDenseLayer(l):
//     f = down_proj(SiluAndMul(gate_up_proj(h2)))          [shared SwiGLU seam]
//   else:
//     // UNGROUPED sigmoid noaux_tc router + shared expert + routed_scaling 2.5.
//     //   scores = sigmoid(router(h2))                      [ds2 scoring_func]
//     //   biased = scores + e_score_correction_bias         [ds2 noaux bias]
//     //   idx = topk(biased, 10)  (UNGROUPED — DROP ds2's group step; the NEW bit)
//     //   w = scores[idx]; if norm_topk_prob: w /= sum(w)
//     //   routed = sum_i w_i * expert_i(h2)                 [grouped-expert GEMM]
//     //   f = routed * moe_routed_scaling_factor(2.5) + shared_expert(h2)
//     //   [deepseek_v2.cpp:340-365 RunMoeBlock — reuse MINUS the group step]
//   x = x_res2 + f
//
//   // final: RMSNorm(x, norm) -> lm_head (untied) -> logits
//
// EVERY heavy op above is LANDED. The only NEW compute is the per-head softplus
// gate (step 4, ~30-line host op) + the ungrouped topk (step 5, a config peel of
// the ds2 grouped router). NO new kernel. See the reuse map in laguna.h +
// `.agents/specs/laguna-s21-scope-2026-07-30.md` §4.
#include "vllm/model_executor/models/laguna.h"

#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/dtype.h"

namespace vllm {

std::vector<float> LagunaModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  (void)token_ids;
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  (void)weights;
  (void)config;
  (void)queue;
  (void)logits_indices;
  VT_CHECK(false,
           "laguna: LagunaModel::Forward is a W3/W4 residual. The per-layer "
           "composition (variable-Q-head GQA + dual per-layer RoPE + sliding-"
           "window mask + per-head softplus out-gate; dense MLP L0; ungrouped "
           "sigmoid-noaux MoE L1..47) is documented in laguna.cpp; the reuse is "
           "LANDED and the strict dual-oracle gate (llama.cpp-Q4_K token-exact + "
           "vLLM-NVFP4 near-tie) is W4. See "
           ".agents/specs/laguna-s21-w1w2-2026-07-30.md.");
  return {};
}

ForwardLogits LagunaModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  // The device path reuses the host composition until the W3 device assembly
  // lands (mirrors ds4's ForwardDevice-after-Forward staging).
  return HostLogits(
      LagunaModel::Forward(token_ids, positions, attn_meta, attn_kv, weights,
                           config, queue, logits_indices),
      weights.params.vocab_size);
}

}  // namespace vllm
