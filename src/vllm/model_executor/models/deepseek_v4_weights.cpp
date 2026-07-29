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

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
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

// ════════════════════════════════════════════════════════════════════════════
// W2b — the `deepseek4` GGUF keep-quant TOWER materialization.
//
// Wires the LANDED `blk.N.*` -> V4 name map (scripts/check-dsv4-gguf-namemap.py,
// EXACT 1328/1328 coverage) + the LANDED keep-quant vec_dot (IQ2_XXS/IQ3_XXS/Q2_K,
// CIQ) into the DeepseekV4 weight towers, so `unsloth/DeepSeek-V4-Flash-GGUF
// UD-IQ2_XXS` (~91 GiB, the only single-GB10-fitting build) LOADS. MW/SEW roles
// KEEP their ~2-3-bit blocks COMPRESSED (OwnGgufQuantBlocks, the memory enabler —
// dequant-to-bf16 would need ~316 GiB and OOM-reboot the unified pool); the small
// V/ET/HASH tensors dequant to f32/bf16 exactly as our other GGUF loaders do
// (qwen3_5_gguf_weights.cpp is the structural mirror). The load ALSO dequants the
// tiny-config CPU composition tower (`host`) so a loaded model can Forward.
//
// HONEST 3-state: the tiny-synthetic load->forward here is DERIVED + BUILD-VERIFIED
// (structural, tiny shape, test_deepseek_v4_gguf_load.cpp). The REAL 91 GiB
// checkpoint load + generate stays W8-final (operational: download + DGX) — the
// name-map coverage vs the real 1328-tensor manifest is gated separately
// (check-dsv4-gguf-namemap.py rc=0).
namespace {

// --- GGUF KV metadata readers (mirror qwen3_5_gguf_weights.cpp KvInt/KvFloat) ---
int64_t GgKvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("deepseek-v4 gguf: key " + key + " is not an integer");
  }
}
double GgKvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(GgKvInt(v, key));
}
int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "deepseek-v4 gguf: missing metadata key " + key);
  return GgKvInt(*v, key);
}
int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? GgKvInt(*v, key) : dflt;
}
double OptFloat(const GgufFile& g, const std::string& key, double dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? GgKvFloat(*v, key) : dflt;
}

std::string Blk(int64_t l, const std::string& suffix) {
  return "blk." + std::to_string(l) + "." + suffix;
}

bool HasGgufTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors())
    if (t.name == name) return true;
  return false;
}

// --- host-tower dequant bridge (torch [out,in] row-major f32) ----------------
// Reads the RAW ggml bytes (Q8_0/IQ2_XXS/IQ3_XXS/F32 are self-contained — no
// sidecar scale), independent of the keep-quant OwnedTensor residency.
std::vector<float> DqRowF32(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo& t = g.Get(name);
  int64_t numel = 1;
  for (int64_t d : t.shape) numel *= d;
  return DequantGgufRowToF32(t.ggml_type, t.data, numel);
}

// --- tower materializers (route once, account, keep-quant or expand) ---------
OwnedTensor MakeBf16Owned(const std::vector<uint16_t>& dq,
                          const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  VT_CHECK(static_cast<int64_t>(dq.size()) == n, "deepseek-v4 gguf: bf16 size mismatch");
  o.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  std::memcpy(o.bytes.data(), dq.data(), o.bytes.size());
  o.nk = nk;
  return o;
}
OwnedTensor MakeF32Owned(const std::vector<float>& dq,
                         const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = vt::DType::kF32;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  VT_CHECK(static_cast<int64_t>(dq.size()) == n, "deepseek-v4 gguf: f32 size mismatch");
  o.bytes.resize(static_cast<size_t>(n) * sizeof(float));
  std::memcpy(o.bytes.data(), dq.data(), o.bytes.size());
  return o;
}

// A per-load routing context: the file, the residency policy, and the set of
// consumed tensor names (the totality/accounting contract — every routed tensor
// is recorded, and a leftover unroutes-to-fail at the end).
struct V4GgufCtx {
  const GgufFile& g;
  const GgufLoadPolicy& pol;
  std::unordered_set<std::string> consumed;

  const GgufTensorInfo& Take(const std::string& name) {
    const GgufTensorInfo& t = g.Get(name);  // throws (unmapped/missing -> FAIL)
    consumed.insert(name);
    return t;
  }
  // 2-D [out,in] matmul weight: keep its blocks (MW keep-quant) else expand bf16,
  // both in the file's own [N,K] order (nk=true) — GGUF disk order IS MatmulBT.
  OwnedTensor Mw(const std::string& name) {
    const GgufTensorInfo& t = Take(name);
    VT_CHECK(t.shape.size() == 2, "deepseek-v4 gguf: expected 2-D MW " + name);
    const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
    if (r != GgufResidency::kExpandBf16) {
      return OwnGgufQuantBlocks(t, t.shape[0], t.shape[1]);
    }
    return MakeBf16Owned(DequantGgufRowToBf16(t.ggml_type, t.data, t.shape[0] * t.shape[1]),
                         {t.shape[0], t.shape[1]}, /*nk=*/true);
  }
  // Stacked [E,out,in] expert weight: KEEP the whole block slab COMPRESSED (each
  // expert = out whole rows = whole blocks, so E*out rows is one contiguous keep),
  // else expand to bf16 [E*out,in].
  OwnedTensor Sew(const std::string& name, int64_t experts) {
    const GgufTensorInfo& t = Take(name);
    VT_CHECK(t.shape.size() == 3 && t.shape[0] == experts,
             "deepseek-v4 gguf: expected [E,out,in] expert tensor " + name);
    const int64_t rows = t.shape[0] * t.shape[1];  // E*out
    const int64_t k = t.shape[2];                  // in
    const GgufResidency r = pol.Route(t, GgufTensorRole::kStackedExpertWeight);
    if (r != GgufResidency::kExpandBf16) {
      return OwnGgufQuantBlocks(t, rows, k, /*row_offset=*/0);
    }
    return MakeBf16Owned(DequantGgufRowToBf16(t.ggml_type, t.data, rows * k),
                         {rows, k}, /*nk=*/true);
  }
  // A value/table tensor whose bytes are rewritten (norm/bias/scale/sink/table/
  // embed) — NEVER keep-quant. Asserts the policy agrees (totality) then dequants
  // to f32 in the file's torch shape.
  OwnedTensor Vec(const std::string& name, GgufTensorRole role) {
    const GgufTensorInfo& t = Take(name);
    VT_CHECK(pol.Route(t, role) == GgufResidency::kExpandBf16,
             std::string("deepseek-v4 gguf: a ") + Name(role) +
                 " tensor must not keep quant blocks: " + name);
    return MakeF32Owned(DqRowF32(g, name), t.shape);
  }
};

// Fill the tiny-config CPU composition host field for one flat weight.
std::vector<float> HostVec(const GgufFile& g, const std::string& name) {
  return DqRowF32(g, name);
}

}  // namespace

DeepseekV4Params DeepseekV4ParamsFromGguf(const GgufFile& g) {
  const GgufValue* arch = g.FindKv("general.architecture");
  VT_CHECK(arch != nullptr && arch->TypeId() == kGgufString,
           "deepseek-v4 gguf: general.architecture must be a string");
  VT_CHECK(std::get<std::string>(arch->v) == "deepseek4",
           "deepseek-v4 gguf: expected general.architecture 'deepseek4', got '" +
               std::get<std::string>(arch->v) + "'");
  const std::string p = "deepseek4.";

  DeepseekV4Params d;
  d.hidden_size = ReqInt(g, p + "embedding_length");
  d.num_hidden_layers = ReqInt(g, p + "block_count");
  d.num_attention_heads = ReqInt(g, p + "attention.head_count");
  d.num_key_value_heads = OptInt(g, p + "attention.head_count_kv", 1);
  d.head_dim = ReqInt(g, p + "attention.key_length");
  d.qk_rope_head_dim = OptInt(g, p + "rope.dimension_count", 64);
  d.q_lora_rank = ReqInt(g, p + "attention.q_lora_rank");
  d.o_lora_rank = ReqInt(g, p + "attention.output_lora_rank");
  d.o_groups = ReqInt(g, p + "attention.output_group_count");
  d.sliding_window = OptInt(g, p + "attention.sliding_window", 0);
  d.rope_theta = OptFloat(g, p + "rope.freq_base", 10000.0);
  d.compress_rope_theta = OptFloat(g, p + "attention.compress_rope_freq_base", 160000.0);
  d.rms_norm_eps =
      static_cast<float>(OptFloat(g, p + "attention.layer_norm_rms_epsilon", 1e-6));
  d.max_position_embeddings = OptInt(g, p + "context_length", 0);
  d.num_nextn_predict_layers = OptInt(g, p + "nextn_predict_layers", 0);

  // MoE.
  d.n_routed_experts = ReqInt(g, p + "expert_count");
  d.num_experts_per_tok = ReqInt(g, p + "expert_used_count");
  d.n_shared_experts = OptInt(g, p + "expert_shared_count", 0);
  d.moe_intermediate_size = ReqInt(g, p + "expert_feed_forward_length");
  d.num_hash_layers = OptInt(g, p + "hash_layer_count", 0);
  d.routed_scaling_factor = OptFloat(g, p + "expert_weights_scale", 1.0);
  d.swiglu_limit = OptFloat(g, p + "swiglu_clamp", 0.0);
  d.norm_topk_prob = OptInt(g, p + "norm_topk_prob", 1) != 0;
  // The `deepseek4` arch is fixed sqrtsoftplus/noaux_tc (expert_gating_func==4);
  // the forward selects those unconditionally, so we do not re-validate the enum.
  d.scoring_func = "sqrtsoftplus";
  d.topk_method = "noaux_tc";
  d.expert_dtype = "fp4";

  // MHC.
  d.hc_mult = ReqInt(g, p + "hyper_connection.count");
  d.hc_sinkhorn_iters = OptInt(g, p + "hyper_connection.sinkhorn_iterations", 20);
  d.hc_eps = OptFloat(g, p + "hyper_connection.epsilon", 1e-6);

  // DSA.
  d.index_head_dim = OptInt(g, p + "attention.indexer.key_length", 0);
  d.index_n_heads = OptInt(g, p + "attention.indexer.head_count", 0);
  d.index_topk = OptInt(g, p + "attention.indexer.top_k", 0);
  if (const GgufValue* cr = g.FindKv(p + "attention.compress_ratios");
      cr != nullptr && cr->TypeId() == kGgufArray) {
    for (const GgufValue& e : std::get<GgufArray>(cr->v).elems)
      d.compress_ratios.push_back(GgKvInt(e, "compress_ratios"));
  }

  // vocab: prefer the token_embd leading (out) dim, else the kv.
  const GgufValue* vk = g.FindKv(p + "vocab_size");
  d.vocab_size = vk != nullptr ? GgKvInt(*vk, p + "vocab_size")
                               : g.Get("token_embd.weight").shape[0];

  // Minimal self-consistency (the GGUF is the source of geometry truth — the
  // strict head_dim==512/sqrtsoftplus assertions live in ParseDeepseekV4Params
  // for the safetensors path; a tiny synthetic GGUF uses a small head_dim).
  VT_CHECK(d.hidden_size > 0 && d.num_hidden_layers > 0 && d.head_dim > 0,
           "deepseek-v4 gguf: degenerate geometry");
  VT_CHECK(d.n_routed_experts > 0, "deepseek-v4 gguf: n_routed_experts must be > 0");
  VT_CHECK(d.hc_mult > 0, "deepseek-v4 gguf: hc_mult must be > 0 (MHC is structural)");
  VT_CHECK(static_cast<int64_t>(d.compress_ratios.size()) == d.num_hidden_layers,
           "deepseek-v4 gguf: compress_ratios length must equal block_count");
  return d;
}

HfConfig DeepseekV4HfConfigFromGguf(const GgufFile& g) {
  // The GGUF is the source of geometry truth; resolve it ONCE (throws on a
  // non-deepseek4 arch or a missing required key) and republish it.
  const DeepseekV4Params p = DeepseekV4ParamsFromGguf(g);

  HfConfig c;
  // Keep llama.cpp's GGUF family key in model_type, but map `architectures` onto
  // the registered vLLM model class so ModelRegistry::Resolve routes a deepseek4
  // file into the DeepSeek-V4 factory — the same trick HfConfigFromGguf uses to
  // map the qwen35* keys onto the Qwen3.5 wrappers.
  c.model_type = "deepseek4";
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = p.hidden_size;
  c.num_hidden_layers = p.num_hidden_layers;
  c.vocab_size = p.vocab_size;
  c.num_attention_heads = p.num_attention_heads;
  c.num_key_value_heads = p.num_key_value_heads;
  c.head_dim = p.head_dim;
  c.rms_norm_eps = p.rms_norm_eps;
  c.rope_theta = p.rope_theta;
  c.max_position_embeddings = p.max_position_embeddings;
  c.torch_dtype = "bfloat16";

  // Republish the DeepSeek-V4 scalars the registry parse hook
  // (ParseDeepseekV4Config -> ParseDeepseekV4Params) reads from config.raw, so
  // that hook validates the SAME geometry this GGUF describes. The weight loader
  // itself re-derives from the GGUF KV (LoadDeepseekV4FromGguf ignores config).
  nlohmann::json& raw = c.raw = nlohmann::json::object();
  raw["hidden_size"] = p.hidden_size;
  raw["num_hidden_layers"] = p.num_hidden_layers;
  raw["vocab_size"] = p.vocab_size;
  raw["num_attention_heads"] = p.num_attention_heads;
  raw["num_key_value_heads"] = p.num_key_value_heads;
  raw["head_dim"] = p.head_dim;
  raw["qk_rope_head_dim"] = p.qk_rope_head_dim;
  raw["q_lora_rank"] = p.q_lora_rank;
  raw["o_lora_rank"] = p.o_lora_rank;
  raw["o_groups"] = p.o_groups;
  raw["sliding_window"] = p.sliding_window;
  raw["rope_theta"] = p.rope_theta;
  raw["compress_rope_theta"] = p.compress_rope_theta;
  raw["rms_norm_eps"] = p.rms_norm_eps;
  raw["max_position_embeddings"] = p.max_position_embeddings;
  raw["num_nextn_predict_layers"] = p.num_nextn_predict_layers;
  raw["n_routed_experts"] = p.n_routed_experts;
  raw["num_experts_per_tok"] = p.num_experts_per_tok;
  raw["moe_intermediate_size"] = p.moe_intermediate_size;
  raw["n_shared_experts"] = p.n_shared_experts;
  raw["norm_topk_prob"] = p.norm_topk_prob;
  raw["routed_scaling_factor"] = p.routed_scaling_factor;
  raw["swiglu_limit"] = p.swiglu_limit;
  raw["scoring_func"] = p.scoring_func;
  raw["topk_method"] = p.topk_method;
  raw["num_hash_layers"] = p.num_hash_layers;
  raw["expert_dtype"] = p.expert_dtype;
  raw["hc_mult"] = p.hc_mult;
  raw["hc_sinkhorn_iters"] = p.hc_sinkhorn_iters;
  raw["hc_eps"] = p.hc_eps;
  raw["index_head_dim"] = p.index_head_dim;
  raw["index_n_heads"] = p.index_n_heads;
  raw["index_topk"] = p.index_topk;
  raw["compress_ratios"] = p.compress_ratios;
  return c;
}

DeepseekV4Weights LoadDeepseekV4FromGguf(const GgufFile& g, const HfConfig& config,
                                        const GgufLoadPolicy* policy) {
  (void)config;  // params resolved from the GGUF KV (self-describing vehicle)
  const GgufLoadPolicy env = GgufLoadPolicy::FromEnv();
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env;
  const DeepseekV4Params p = DeepseekV4ParamsFromGguf(g);

  DeepseekV4Weights w;
  w.params = p;
  V4GgufCtx ctx{g, pol, {}};

  const int64_t H = p.hidden_size;
  const int64_t nh = p.num_attention_heads;
  const int64_t hd = p.head_dim;
  const int64_t qlr = p.q_lora_rank;
  const int64_t ne = p.n_routed_experts;
  const int64_t topk = p.num_experts_per_tok;
  const int64_t mi = p.moe_intermediate_size;
  const int64_t hc = p.hc_mult;
  const int64_t inh = p.index_n_heads;
  const int64_t ihd = p.index_head_dim;
  const bool tied = !HasGgufTensor(g, "output.weight");

  // ── model-level tower slots ─────────────────────────────────────────────
  DeepseekV4GgufWeights& tw = w.gguf;
  tw.embed = ctx.Vec("token_embd.weight", GgufTensorRole::kEmbeddingTable);
  tw.lm_head = tied ? ctx.Vec("token_embd.weight", GgufTensorRole::kEmbeddingTable)
                    : ctx.Mw("output.weight");
  tw.final_norm = ctx.Vec("output_norm.weight", GgufTensorRole::kVector);
  tw.hc_head_base = ctx.Vec("output_hc_base.weight", GgufTensorRole::kVector);
  tw.hc_head_fn = ctx.Vec("output_hc_fn.weight", GgufTensorRole::kVector);
  tw.hc_head_scale = ctx.Vec("output_hc_scale.weight", GgufTensorRole::kVector);

  // ── host composition tower (dequant bridge) ─────────────────────────────
  DeepseekV4HostWeights& hw = w.host;
  hw.embed = HostVec(g, "token_embd.weight");
  hw.lm_head = tied ? hw.embed : HostVec(g, "output.weight");
  hw.final_norm_weight = HostVec(g, "output_norm.weight");
  hw.hc_head_fn = HostVec(g, "output_hc_fn.weight");
  hw.hc_head_base = HostVec(g, "output_hc_base.weight");
  const std::vector<float> hhs = HostVec(g, "output_hc_scale.weight");
  hw.hc_head_scale = hhs.empty() ? 0.0f : hhs[0];
  hw.layers.resize(static_cast<size_t>(p.num_hidden_layers));

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4GgufLayerWeights lw;
    DeepseekV4LayerHostWeights& hl = hw.layers[static_cast<size_t>(l)];
    lw.is_hash = p.is_hash_layer(l);
    lw.has_compressor = p.has_compressor(l);
    lw.has_indexer = p.has_indexer(l);

    // 512-wide MLA (MW keep-quant) + norms/sink (V).
    lw.wq_a = ctx.Mw(Blk(l, "attn_q_a.weight"));
    lw.wq_b = ctx.Mw(Blk(l, "attn_q_b.weight"));
    lw.wkv = ctx.Mw(Blk(l, "attn_kv.weight"));
    lw.wo_a = ctx.Mw(Blk(l, "attn_output_a.weight"));
    lw.wo_b = ctx.Mw(Blk(l, "attn_output_b.weight"));
    lw.attn_norm = ctx.Vec(Blk(l, "attn_norm.weight"), GgufTensorRole::kVector);
    lw.q_a_norm = ctx.Vec(Blk(l, "attn_q_a_norm.weight"), GgufTensorRole::kVector);
    lw.kv_a_norm = ctx.Vec(Blk(l, "attn_kv_a_norm.weight"), GgufTensorRole::kVector);
    lw.attn_sink = ctx.Vec(Blk(l, "attn_sinks.weight"), GgufTensorRole::kVector);
    lw.ffn_norm = ctx.Vec(Blk(l, "ffn_norm.weight"), GgufTensorRole::kVector);

    // MHC per-layer mixing (V).
    lw.hc_attn_base = ctx.Vec(Blk(l, "hc_attn_base.weight"), GgufTensorRole::kVector);
    lw.hc_attn_fn = ctx.Vec(Blk(l, "hc_attn_fn.weight"), GgufTensorRole::kVector);
    lw.hc_attn_scale = ctx.Vec(Blk(l, "hc_attn_scale.weight"), GgufTensorRole::kVector);
    lw.hc_ffn_base = ctx.Vec(Blk(l, "hc_ffn_base.weight"), GgufTensorRole::kVector);
    lw.hc_ffn_fn = ctx.Vec(Blk(l, "hc_ffn_fn.weight"), GgufTensorRole::kVector);
    lw.hc_ffn_scale = ctx.Vec(Blk(l, "hc_ffn_scale.weight"), GgufTensorRole::kVector);

    // MoE: router gate (MW), 256 routed experts + shared (SEW/MW keep-quant),
    // the hash `tid2eid` (hash layers) or the noaux_tc `exp_probs_b` bias.
    lw.moe_gate = ctx.Mw(Blk(l, "ffn_gate_inp.weight"));
    lw.moe_gate_exps = ctx.Sew(Blk(l, "ffn_gate_exps.weight"), ne);
    lw.moe_up_exps = ctx.Sew(Blk(l, "ffn_up_exps.weight"), ne);
    lw.moe_down_exps = ctx.Sew(Blk(l, "ffn_down_exps.weight"), ne);
    lw.shared_gate = ctx.Mw(Blk(l, "ffn_gate_shexp.weight"));
    lw.shared_up = ctx.Mw(Blk(l, "ffn_up_shexp.weight"));
    lw.shared_down = ctx.Mw(Blk(l, "ffn_down_shexp.weight"));
    if (lw.is_hash) {
      lw.tid2eid = ctx.Vec(Blk(l, "ffn_gate_tid2eid.weight"), GgufTensorRole::kVector);
    } else {
      lw.e_score_bias = ctx.Vec(Blk(l, "exp_probs_b.bias"), GgufTensorRole::kVector);
    }

    // DSA compressor (compress_ratio != 0) + Lightning-Indexer (== 4).
    if (lw.has_compressor) {
      lw.comp_ape = ctx.Vec(Blk(l, "attn_compressor_ape.weight"), GgufTensorRole::kVector);
      lw.comp_wgate = ctx.Mw(Blk(l, "attn_compressor_gate.weight"));
      lw.comp_wkv = ctx.Mw(Blk(l, "attn_compressor_kv.weight"));
      lw.comp_norm = ctx.Vec(Blk(l, "attn_compressor_norm.weight"), GgufTensorRole::kVector);
    }
    if (lw.has_indexer) {
      lw.idx_wq_b = ctx.Mw(Blk(l, "indexer.attn_q_b.weight"));
      lw.idx_proj = ctx.Vec(Blk(l, "indexer.proj.weight"), GgufTensorRole::kVector);
      lw.idx_comp_ape =
          ctx.Vec(Blk(l, "indexer_compressor_ape.weight"), GgufTensorRole::kVector);
      lw.idx_comp_wgate = ctx.Mw(Blk(l, "indexer_compressor_gate.weight"));
      lw.idx_comp_wkv = ctx.Mw(Blk(l, "indexer_compressor_kv.weight"));
      lw.idx_comp_norm =
          ctx.Vec(Blk(l, "indexer_compressor_norm.weight"), GgufTensorRole::kVector);
    }

    // ── host bridge for THIS layer (dequant the slots the CPU forward reads) ──
    hl.attn_norm_weight = HostVec(g, Blk(l, "attn_norm.weight"));
    hl.ffn_norm_weight = HostVec(g, Blk(l, "ffn_norm.weight"));
    hl.hc_attn_fn = HostVec(g, Blk(l, "hc_attn_fn.weight"));
    hl.hc_attn_base = HostVec(g, Blk(l, "hc_attn_base.weight"));
    hl.hc_attn_scale = HostVec(g, Blk(l, "hc_attn_scale.weight"));
    hl.hc_ffn_fn = HostVec(g, Blk(l, "hc_ffn_fn.weight"));
    hl.hc_ffn_base = HostVec(g, Blk(l, "hc_ffn_base.weight"));
    hl.hc_ffn_scale = HostVec(g, Blk(l, "hc_ffn_scale.weight"));
    hl.wq_a = HostVec(g, Blk(l, "attn_q_a.weight"));
    hl.q_norm_weight = HostVec(g, Blk(l, "attn_q_a_norm.weight"));
    hl.wq_b = HostVec(g, Blk(l, "attn_q_b.weight"));
    hl.wkv = HostVec(g, Blk(l, "attn_kv.weight"));
    hl.kv_norm_weight = HostVec(g, Blk(l, "attn_kv_a_norm.weight"));
    hl.attn_sink = HostVec(g, Blk(l, "attn_sinks.weight"));
    hl.wo_a = HostVec(g, Blk(l, "attn_output_a.weight"));
    hl.wo_b = HostVec(g, Blk(l, "attn_output_b.weight"));
    hl.gate_weight = HostVec(g, Blk(l, "ffn_gate_inp.weight"));
    if (lw.is_hash) {
      const std::vector<float> t2e = HostVec(g, Blk(l, "ffn_gate_tid2eid.weight"));
      hl.tid2eid.resize(t2e.size());
      for (size_t i = 0; i < t2e.size(); ++i)
        hl.tid2eid[i] = static_cast<int32_t>(std::lround(t2e[i]));
    } else {
      hl.gate_bias = HostVec(g, Blk(l, "exp_probs_b.bias"));
    }
    hl.shared_w1 = HostVec(g, Blk(l, "ffn_gate_shexp.weight"));
    hl.shared_w3 = HostVec(g, Blk(l, "ffn_up_shexp.weight"));
    hl.shared_w2 = HostVec(g, Blk(l, "ffn_down_shexp.weight"));
    hl.exp_w1 = HostVec(g, Blk(l, "ffn_gate_exps.weight"));
    hl.exp_w3 = HostVec(g, Blk(l, "ffn_up_exps.weight"));
    hl.exp_w2 = HostVec(g, Blk(l, "ffn_down_exps.weight"));
    if (lw.has_compressor) {
      hl.comp_wgate = HostVec(g, Blk(l, "attn_compressor_gate.weight"));
      hl.comp_ape = HostVec(g, Blk(l, "attn_compressor_ape.weight"));
      hl.comp_norm_weight = HostVec(g, Blk(l, "attn_compressor_norm.weight"));
    }
    if (lw.has_indexer) {
      // Tiny-config STRUCTURAL bridge (documented divergence, deepseek_v4.cpp):
      // the host indexer uses a simplified q/k/proj triple — idx_wq <- the
      // indexer query proj, idx_wproj <- the weights_proj, idx_wk <- the
      // indexer compressor kv (the real model derives the key from the
      // compressed latent). The full geometry is the W7-device / W8 seam.
      hl.idx_wq = HostVec(g, Blk(l, "indexer.attn_q_b.weight"));
      hl.idx_wproj = HostVec(g, Blk(l, "indexer.proj.weight"));
      hl.idx_wk = HostVec(g, Blk(l, "indexer_compressor_kv.weight"));
    }

    tw.layers.push_back(std::move(lw));
  }

  // ── the ACCOUNTING gate: every file tensor must have been routed exactly once
  //    (none unmapped, none leftover) — the C++ half of the name-map contract
  //    (scripts/check-dsv4-gguf-namemap.py gates the real 1328-tensor manifest). ─
  for (const GgufTensorInfo& t : g.Tensors()) {
    VT_CHECK(ctx.consumed.count(t.name) != 0,
             "deepseek-v4 gguf loader: LEFTOVER tensor not covered by the blk.N.* "
             "name map: " + t.name);
  }
  w.accounted_tensors = static_cast<int64_t>(ctx.consumed.size());
  w.has_gguf_weights = true;
  w.has_host_weights = true;
  // Silence unused in a non-indexer/degenerate tiny config.
  (void)nh; (void)hd; (void)qlr; (void)topk; (void)mi; (void)hc; (void)inh; (void)ihd; (void)H;
  return w;
}

}  // namespace vllm
