// DeepSeek-V4-Flash config resolution + checkpoint name-map (W1/W2 scaffolding).
// The header carries the full `file:line`-on-both-sides port map; this TU
// implements `ParseDeepseekV4Params` (config-descent, unit-testable) and the
// `LoadDeepseekV4ForCausalLMWeights` ACCOUNTING pass over the checkpoint's
// verified name-map. Heavy tensor MATERIALIZATION (FP8-block MLA linears, NVFP4
// grouped experts, MHC/DSA towers) is the named W2b/W3-W6 residual.
//
// ─── NAME MAP VERIFIED vs the REAL header (2026-07-28, HTTP range, NO download) ─
// `nvidia/DeepSeek-V4-Flash-NVFP4` model.safetensors.index.json (135,235 tensors,
// 46 shards, total_size 168,266,793,544 B = 156.7 GiB) + shard-2 header. The
// checkpoint uses a FLAT `layers.N.` prefix (no `model.`/mm wrapper); vLLM's
// `_make_deepseek_v4_weights_mapper` (nvidia/model.py:1316) re-prefixes to
// `model.layers.` and fuses wq_a|wkv / compressor.wkv|wgate + w1|w3 -> gate_up.
// We keep the checkpoint's own flat names. VERIFIED dtypes/shapes (layer 0):
//
//   MODEL LEVEL
//     embed.weight               BF16 [129280,4096]      head.weight BF16 [129280,4096]
//     norm.weight                BF16 [4096]
//     hc_head_{base,fn,scale}    F32  (final MHC head collapse, nvidia/model.py:1137)
//
//   PER LAYER N (0..42) — FLAT `layers.N.`
//     attn_norm.weight / ffn_norm.weight            BF16 [4096]
//     hc_attn_{base[24],fn[24,16384],scale[3]}      F32   (MHC: 24=(2+hc_mult)*hc_mult,
//     hc_ffn_{base,fn,scale}                        F32    16384=hc_mult*H) — NEW, no eager ref
//     -- 512-wide MLA (attention.py), FP8-block E4M3 weight + E8M0 [.,.] block scale --
//     attn.wq_a.{weight[1024,4096],scale[8,32]}     q down-proj (q_lora_rank=1024)
//     attn.wq_b.{weight[32768,1024],scale}          q up-proj (64 heads * 512 head_dim)
//     attn.wkv.{weight[512,4096],scale[4,32]}       kv down (fused w/ wq_a upstream)
//     attn.wo_a.{weight[8192,4096],scale}           OUTPUT-LoRA down (o_groups*o_lora=8*1024) NEW
//     attn.wo_b.{weight[4096,8192],scale}           OUTPUT-LoRA up
//     attn.q_norm.weight[1024] / attn.kv_norm.weight[512]   BF16 RMSNorm
//     attn.attn_sink[64]                            F32   per-head attention sink — NEW
//     -- DSA compressor, layers where compress_ratio!=0 (41 layers) --   NEW
//     attn.compressor.{ape, norm.weight, wgate.weight, wkv.weight}
//     -- DSA Lightning-Indexer, layers where compress_ratio==4 (21 layers) --  NEW
//     attn.indexer.compressor.{ape,norm.weight,wgate.weight,wkv.weight}
//     attn.indexer.weights_proj.weight / attn.indexer.wq_b.{weight,scale}
//     -- MoE (nvidia/model.py:512) --
//     ffn.gate.weight[256,4096]  BF16
//     ffn.gate.tid2eid           HASH layers 0,1,2 only (num_hash_layers=3) — NEW
//     ffn.gate.bias              non-hash layers (noaux_tc e_score_correction_bias)
//     ffn.shared_experts.w{1,2,3}.{weight,scale}    FP8-block E4M3 + E8M0 (NOT NVFP4)
//     ffn.experts.E.w{1,2,3}.{weight, weight_scale, weight_scale_2, input_scale}
//                                                   NVFP4: U8-packed [I,H/2] + group-16
//                                                   E4M3 weight_scale + F32 scalar scale_2/input
//   MTP TAIL (`mtp.*`, num_nextn_predict_layers=1) — SKIPPED by the loader,
//   exactly as vLLM's AutoWeightsLoader(skip_substrs=["mtp."]) (nvidia/model.py:1474).
//
// ─── HW-FIT REVERSAL (recorded) ─────────────────────────────────────────────
// total_size = 156.7 GiB. Only the 256 routed experts are NVFP4 (4-bit); the MLA
// + shared-expert linears are FP8 block (`exclude_modules: *.attn.*,
// *.ffn.shared_experts.*, head, mtp.*`), and NVFP4 carries a double scale
// (weight_scale + weight_scale_2) + input_scale per weight. So this "NVFP4"
// checkpoint is ~the same size as the native fp4 (148.7 GiB) and does NOT fit ONE
// GB10's 119 GiB unified pool. The W0 spike's "~83 GiB fits" was a bad estimate.
// W1 (single-GB10 oracle run) is MEMORY-INFEASIBLE — needs multi-node TP / offload.
#include "vllm/model_executor/models/deepseek_v4.h"

#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// --- raw config.json readers (DeepSeek V4 keys are not typed on HfConfig) ---
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_string()) ? f->get<std::string>() : fallback;
}

}  // namespace

DeepseekV4Params ParseDeepseekV4Params(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  DeepseekV4Params p;

  // --- shared geometry (typed fields fall back to raw) ---
  p.hidden_size = config.hidden_size > 0 ? config.hidden_size
                                         : RawInt(raw, "hidden_size", 0);
  p.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(raw, "num_hidden_layers", 0);
  p.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(raw, "vocab_size", 0);
  p.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : RawInt(raw, "num_attention_heads", 0);
  p.num_key_value_heads = RawInt(raw, "num_key_value_heads", 1);
  p.rms_norm_eps = static_cast<float>(RawDouble(raw, "rms_norm_eps", 1e-6));
  p.tie_word_embeddings = RawBool(raw, "tie_word_embeddings", false);
  p.max_position_embeddings = RawInt(raw, "max_position_embeddings", 0);
  p.num_nextn_predict_layers = RawInt(raw, "num_nextn_predict_layers", 0);

  // --- 512-wide MLA geometry (NEW) ---
  p.head_dim =
      config.head_dim > 0 ? config.head_dim : RawInt(raw, "head_dim", 0);
  p.qk_rope_head_dim = RawInt(raw, "qk_rope_head_dim", 64);
  p.q_lora_rank = RawInt(raw, "q_lora_rank", 0);
  p.o_lora_rank = RawInt(raw, "o_lora_rank", 0);
  p.o_groups = RawInt(raw, "o_groups", 0);
  p.sliding_window = RawInt(raw, "sliding_window", 0);
  p.rope_theta = RawDouble(raw, "rope_theta", 10000.0);
  p.compress_rope_theta = RawDouble(raw, "compress_rope_theta", 160000.0);

  // --- MoE ---
  p.n_routed_experts = RawInt(raw, "n_routed_experts", 0);
  p.num_experts_per_tok = RawInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = RawInt(raw, "moe_intermediate_size", 0);
  p.n_shared_experts = RawInt(raw, "n_shared_experts", 0);
  p.norm_topk_prob = RawBool(raw, "norm_topk_prob", true);
  p.routed_scaling_factor = RawDouble(raw, "routed_scaling_factor", 1.0);
  p.swiglu_limit = RawDouble(raw, "swiglu_limit", 0.0);
  p.scoring_func = RawString(raw, "scoring_func", "sqrtsoftplus");
  p.topk_method = RawString(raw, "topk_method", "noaux_tc");
  p.num_hash_layers = RawInt(raw, "num_hash_layers", 0);
  p.expert_dtype = RawString(raw, "expert_dtype", "fp4");

  // --- MHC ---
  p.hc_mult = RawInt(raw, "hc_mult", 0);
  p.hc_sinkhorn_iters = RawInt(raw, "hc_sinkhorn_iters", 0);
  p.hc_eps = RawDouble(raw, "hc_eps", 1e-6);

  // --- DSA ---
  p.index_head_dim = RawInt(raw, "index_head_dim", 0);
  p.index_n_heads = RawInt(raw, "index_n_heads", 0);
  p.index_topk = RawInt(raw, "index_topk", 0);
  if (const nlohmann::json* cr = Field(raw, "compress_ratios");
      cr != nullptr && cr->is_array()) {
    for (const auto& v : *cr)
      p.compress_ratios.push_back(v.is_number() ? v.get<int64_t>() : 0);
  }

  // --- validation (throw with a precise message on anything unrepresentable) ---
  VT_CHECK(p.hidden_size > 0, "deepseek-v4: hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0,
           "deepseek-v4: num_hidden_layers must be positive");
  VT_CHECK(p.n_routed_experts > 0,
           "deepseek-v4: n_routed_experts must be positive (this is a MoE arch)");
  VT_CHECK(p.hc_mult > 0,
           "deepseek-v4: hc_mult must be positive — Manifold Hyper-Connections "
           "are structural to V4 (config.raw.hc_mult missing?)");
  VT_CHECK(p.head_dim == 512,
           "deepseek-v4: only the 512-wide MLA geometry (448 NoPE + 64 RoPE) is "
           "scoped; got head_dim=" + std::to_string(p.head_dim));
  VT_CHECK(p.scoring_func == "sqrtsoftplus",
           "deepseek-v4: only scoring_func='sqrtsoftplus' is scoped; got '" +
               p.scoring_func + "'");
  VT_CHECK(p.expert_dtype == "fp4" || p.expert_dtype == "fp8",
           "deepseek-v4: expert_dtype must be 'fp4' (NVFP4/MXFP4) or 'fp8'; got '" +
               p.expert_dtype + "'");
  return p;
}

void ParseDeepseekV4Config(const HfConfig& config) {
  // The resolve itself IS the validation (throws on every unsupported field).
  (void)ParseDeepseekV4Params(config);
}

DeepseekV4Weights LoadDeepseekV4ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  const DeepseekV4Params p = ParseDeepseekV4Params(config);

  // The full checkpoint name set (for the W2 accounting pass).
  std::unordered_set<std::string> have;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) have.insert(name);

  int64_t accounted = 0;
  const auto require = [&](const std::string& name) {
    VT_CHECK(have.count(name) != 0,
             "deepseek-v4 loader: expected checkpoint tensor missing: " + name);
    ++accounted;
  };

  // NVFP4 experts carry a double weight-scale (weight_scale + weight_scale_2) +
  // an input_scale; FP8-block experts carry a single `.scale`. Verified: the
  // `expert_dtype=fp4` NVFP4 vehicle uses the former.
  const std::vector<std::string> expert_suffixes =
      (p.expert_dtype == "fp4")
          ? std::vector<std::string>{".weight", ".weight_scale",
                                     ".weight_scale_2", ".input_scale"}
          : std::vector<std::string>{".weight", ".scale"};

  // --- model level ---
  require("embed.weight");
  require("norm.weight");
  if (!p.tie_word_embeddings) require("head.weight");
  for (const char* s : {"hc_head_base", "hc_head_fn", "hc_head_scale"}) require(s);

  // --- per layer (flat `layers.N.`) ---
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    require(b + "attn_norm.weight");
    require(b + "ffn_norm.weight");
    for (const char* h : {"hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                          "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"})
      require(b + h);

    // 512-wide MLA (FP8-block weight + block scale).
    const std::string a = b + "attn.";
    for (const char* w : {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}) {
      require(a + w + ".weight");
      require(a + w + ".scale");
    }
    require(a + "q_norm.weight");
    require(a + "kv_norm.weight");
    require(a + "attn_sink");

    // DSA compressor (compress_ratio != 0) + Lightning-Indexer (== 4).
    if (p.has_compressor(l)) {
      for (const char* c : {"ape", "norm.weight", "wgate.weight", "wkv.weight"})
        require(a + "compressor." + c);
    }
    if (p.has_indexer(l)) {
      for (const char* c : {"ape", "norm.weight", "wgate.weight", "wkv.weight"})
        require(a + "indexer.compressor." + c);
      require(a + "indexer.weights_proj.weight");
      require(a + "indexer.wq_b.weight");
      require(a + "indexer.wq_b.scale");
    }

    // MoE gate: hash layers carry `tid2eid` (no bias); others carry the
    // noaux_tc `bias` (e_score_correction_bias).
    const std::string f = b + "ffn.";
    require(f + "gate.weight");
    if (p.is_hash_layer(l))
      require(f + "gate.tid2eid");
    else
      require(f + "gate.bias");

    // Shared expert (FP8-block).
    for (const char* w : {"w1", "w2", "w3"}) {
      require(f + "shared_experts." + w + ".weight");
      require(f + "shared_experts." + w + ".scale");
    }

    // 256 routed experts (NVFP4).
    for (int64_t e = 0; e < p.n_routed_experts; ++e) {
      const std::string ep = f + "experts." + std::to_string(e) + ".";
      for (const char* w : {"w1", "w2", "w3"})
        for (const std::string& suf : expert_suffixes) require(ep + w + suf);
    }
  }

  // TODO(W2b): materialize the accounted towers into device OwnedTensors —
  //   * FP8-block MLA linears (wq_a/wq_b/wkv/wo_a/wo_b) + E8M0 block scales:
  //     reuse the fp8 block loaders + cuda_scaled_mm_c3x_sm100.
  //   * NVFP4 grouped experts (U8 w13/w2 + group-16 E4M3 weight_scale + F32
  //     weight_scale_2/input_scale): reuse src/vt/cuda/cuda_matmul_nvfp4_sm100.cu
  //     + the nvfp4 tactics for the FusedMoE-fallback path (MegaMoE is SM100-only,
  //     nvidia/model.py:307 — GB10 uses the fallback).
  //   * MHC mixing matrices (hc_*_{base,fn,scale}) + DSA indexer/compressor
  //     towers: NEW primitives, ported at W3-W6 (see deepseek-v4-flash.md §5).

  DeepseekV4Weights w;
  w.params = p;
  w.accounted_tensors = accounted;
  return w;
}

}  // namespace vllm
