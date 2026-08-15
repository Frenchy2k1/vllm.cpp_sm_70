// Nemotron-H (`NemotronHForCausalLM`) registry TU — the ADDITIVE
// self-registration seam for the W3 structural bring-up (#517,
// .agents/specs/nemotron-h-model.md §4 W3). Follows the
// kimi_linear_registry.cpp / deepseek_v2_registry.cpp seam exactly: a NEW
// translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook, the
// HETEROGENEOUS KV-cache spec (a full-attention group over the 6 GQA layers +
// a Mamba2 recurrent-state group over the 23 mamba layers), the LoadedModel
// subclass and the factory.
//
// W3 registers the arch so it RESOLVES, parses its config and enumerates its
// checkpoint. It does NOT forward: `ForwardNemotronHForCausalLM` REFUSES BY
// NAME (VT_CHECK(false), exactly like kimi_linear / deepseek_v4 / kimi_k3), so
// the TU builds and the structure is unit-testable while a forward LOUDLY
// reports the pending brick instead of returning a silent wrong answer. The
// GGUF arm refuses by name too — it is OWED (spec §5b W7), and a silent
// dequantization to a supported path is exactly what a token gate cannot see.
// The model-matrix row stays INVENTORIED until W4-W6 land.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py `_ModelInfo` for NemotronH (registry.py:179 ->
// models/nemotron_h.py::NemotronHForCausalLM): text generation, HYBRID (23
// Mamba2 layers ⇒ a recurrent-state KV group), not multimodal. Upstream's class
// carries `HasInnerState` + `IsHybrid` (nemotron_h.py:700-712); our ModelInfo is
// a consumed subset whose only `has_inner_state` reader short-circuits on
// `is_hybrid`, so this follows the established hybrid-recurrent registration
// convention (kQwen3_5Info, kKimiLinearInfo).
inline constexpr ModelInfo kNemotronHInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class NemotronHLoadedModel final : public LoadedModel {
 public:
  NemotronHLoadedModel(const ModelRegistration& registration,
                       NemotronHParams params)
      : LoadedModel(registration), params_(std::move(params)) {}
  const NemotronHParams& params() const { return params_; }
  // W4 ports the forward MECHANISM (nemotron_h.cpp). The weight LOAD that fills
  // this — the 18487 enumerated tensors, NVFP4 W4A16 g16 experts and FP8 W8A8
  // mamba projections included — is still owed, so `materialized` stays false on
  // the checkpoint path and `NemotronHForward` refuses by name. A direct caller
  // (the unit gate) constructs the weights itself and reaches the same forward.
  NemotronHHostWeights& weights() { return weights_; }
  const NemotronHHostWeights& weights() const { return weights_; }
  // What the load did, in numbers. Kept on the model rather than returned by
  // value because the load happens inside the type-erased registry factory and
  // a gate has no other way to reach it; `NemotronHLoadReportOf` is the
  // accessor.
  NemotronHLoadReport& report() { return report_; }
  const NemotronHLoadReport& report() const { return report_; }

 private:
  NemotronHParams params_;
  NemotronHHostWeights weights_;
  NemotronHLoadReport report_;
};

std::unique_ptr<LoadedModel> LoadNemotronHForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    // AGENTS.md makes GGUF k-quants a standing requirement, not a per-model
    // choice; it is OWED here (spec §5b, W7) and refused BY NAME rather than
    // routed to a supported path behind the caller's back.
    throw std::runtime_error(
        "Model architecture NemotronHForCausalLM does not support GGUF weights "
        "yet: the GGUF k-quant/i-quant arm is not ported (see "
        ".agents/specs/nemotron-h-model.md §5b W7)");
  }
  // The config descent IS the validation, and it refuses by name on anything
  // this bring-up cannot represent.
  NemotronHParams params = ParseNemotronHParams(config);
  auto model =
      std::make_unique<NemotronHLoadedModel>(registration, params);
  if (source.safetensors == nullptr) {
    throw std::runtime_error(
        "Model architecture NemotronHForCausalLM: the safetensors source "
        "carries no shards");
  }
  // MATERIALIZE. The 18487 enumerated tensors are read into the host weights in
  // the memory format the checkpoint SHIPS them in — NVFP4 W4A16 g16 experts and
  // lm_head, FP8 W8A8 static mamba projections, bf16 everything else — and the
  // MTP tower is deferred by name (W5). See nemotron_h_loader.h.
  model->weights() = LoadNemotronHHostWeights(
      *source.safetensors, params, ResolveNemotronHModelDType(config),
      &model->report());
  return model;
}

void PrepareNemotronHForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardNemotronHForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  // ★ G-SAFE (#810, .agents/specs/nemotron-h-abi-e2e.md §0) — THE SAFETY
  // INTERLOCK. Do not remove or weaken it without landing the device/paged
  // forward it guards.
  //
  // Before #810 A1, `GPUModelRunner::initialize_kv_cache` REFUSED to build a
  // NemotronH engine at all: it rebuilt the recurrent half of the KV allocation
  // from Qwen3.5's `linear_*` config fields and cross-checked the model's own
  // MambaSpec against that reconstruction. A1 makes that allocation
  // spec-driven, so `vllm_engine_load` now SUCCEEDS and a scheduler step
  // reaches this forward.
  //
  // This forward is the HOST REFERENCE. It consumes exactly three of
  // `ModelForwardInput`'s eighteen fields — `token_ids`, `logits_indices`,
  // `queue` — and ignores `attn_kv`, `gdn_state`, `gdn_meta`,
  // `gdn_state_slots`, `num_reqs` and `positions`.
  // `NemotronHAttentionMixer` (nemotron_h.cpp:585-630) recomputes Q/K/V over
  // the whole sequence on every call and pages nothing, and the recurrent state
  // is rebuilt from scratch each step. A server past the old refusal would
  // therefore decode step 2 onward with FRESH recurrent state and NO KV, and
  // treat a multi-request batch as one concatenated causal sequence: fluent
  // output, wrong tokens, no error. That is strictly worse than the loud
  // failure A1 removes, which is why A1 does not land without this guard.
  //
  // Structurally the inverse of the predicate
  // `ForwardKimiLinearForCausalLM` already uses to select its paged fold
  // (kimi_linear_registry.cpp:99-102, `!input.attn_kv.empty() &&
  // !input.gdn_state.empty()`): there the paged caches SELECT the paged path;
  // here their presence means the caller expects a path that does not exist.
  //
  // The guard runs BEFORE `ModelAs` deliberately. It reads only `input` and
  // never touches `model`, so #775's guarantee — no member call before the
  // dynamic type is established — is untouched; and putting it first is what
  // makes it reachable from a test without fabricating a look-alike
  // `NemotronHLoadedModel`, which is exactly the stub #784 removed. Order is
  // not part of the G-SAFE requirement; being gated is.
  //
  // NARROWED, NEVER DELETED: A2 (the device/paged forward) drops the `attn_kv`
  // / `gdn_state` clauses when it consumes them, and A2b drops `num_reqs` when
  // batching lands.
  VT_CHECK(
      input.attn_kv.empty() && input.gdn_state.empty() && input.num_reqs <= 1,
      "Model architecture NemotronHForCausalLM: the PAGED/BATCHED decode path "
      "is not ported (issue #810, .agents/specs/nemotron-h-abi-e2e.md A2). "
      "This forward is the host reference: it recomputes K/V over the whole "
      "sequence every step, carries no recurrent state between steps, and "
      "treats token_ids as ONE causal sequence -- so running it against the "
      "runner's paged KV / recurrent state, or against a multi-request batch, "
      "would return plausible WRONG tokens instead of failing. Refusing by "
      "name until the device/paged forward lands.");
  // #775: CHECKED, not `static_cast`. A bare downcast down this hierarchy is a
  // promise the compiler is entitled to act on, so on a model that is not
  // really a `NemotronHLoadedModel` every `nh.` member call below is undefined
  // behaviour — and it happens on the way to a refusal that throws anyway,
  // which is what kept it invisible outside the sanitizer lane. `ModelAs`
  // establishes the dynamic type first and refuses by name instead.
  auto& nh = ModelAs<NemotronHLoadedModel>(model, "NemotronHForCausalLM");
  // W4: the hybrid layer loop, the Mamba2 mixer wiring, the 6 attention layers
  // and the MoE layers are ported (nemotron_h.cpp) and reached HERE, through the
  // shared `ModelRegistry::Forward` seam — never through a parallel entry point.
  // `NemotronHForward` refuses BY NAME when the host weights are not
  // materialized, which is the state every checkpoint load leaves them in until
  // the weight loader lands (spec §5b); that refusal names the missing piece
  // instead of returning a silent zero forward.
  return HostLogits(NemotronHForward(nh.weights(), nh.params(), input.token_ids,
                                     input.logits_indices, input.queue),
                    nh.params().vocab_size);
}

const ModelFactory kNemotronHFactory{
    .parse_config = &ParseNemotronHConfig,
    .load_weights = &LoadNemotronHForCausalLM,
    .prepare = &PrepareNemotronHForCausalLM,
    .forward = &ForwardNemotronHForCausalLM,
    .make_kv_cache = &MakeNemotronHKVCache,
    .is_dense_model = false,
};

}  // namespace

const NemotronHLoadReport& NemotronHLoadReportOf(const LoadedModel& model) {
  const auto* nh = dynamic_cast<const NemotronHLoadedModel*>(&model);
  if (nh == nullptr) {
    throw std::runtime_error(
        "NemotronHLoadReportOf: this LoadedModel is not a NemotronH model");
  }
  return nh->report();
}

v1::KVCacheConfig MakeNemotronHKVCache(const HfConfig& config, int block_size,
                                       int num_blocks) {
  const NemotronHParams p = ParseNemotronHParams(config);

  // Both state tensors' dtypes come from `_mamba_state_dtype`
  // (mamba_utils.py:93-106): the CONVOLUTION state follows the cache dtype
  // (the model dtype, bf16, unless overridden), and the TEMPORAL/SSM state is
  // resolved INDEPENDENTLY from `mamba_ssm_cache_dtype` — "float32" on this
  // checkpoint, so the two differ. Collapsing the SSM state to the activation
  // dtype is a silent precision loss a token gate can absorb. See
  // NemotronHSsmCacheDType for why the shared qwen3_5 resolver is NOT the right
  // reader here (it is keyed on a different config spelling).
  const vt::DType conv_dtype = vt::DType::kBF16;
  const vt::DType ssm_dtype = NemotronHSsmCacheDType(p, conv_dtype);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;

  // (1) the 6 GQA full-attention layers. `FullAttentionSpec` sizes the paged
  //     K+V page; the fp8 KV scheme the checkpoint ships (k_scale/v_scale) is a
  //     W4/W6 storage decision and is deliberately NOT selected here.
  std::vector<std::string> attn_layers;
  for (int64_t i : p.LayerIndices(NemotronHBlock::kAttention)) {
    attn_layers.push_back("backbone.layers." + std::to_string(i) + ".mixer");
  }
  kv.kv_cache_groups.emplace_back(
      std::move(attn_layers),
      std::make_shared<v1::FullAttentionSpec>(
          block_size, static_cast<int>(p.num_key_value_heads),
          static_cast<int>(p.head_dim), v1::ResolveKvCacheDType()));

  // (2) the 23 Mamba2 layers. mamba2_state_shape (mamba_utils.py:173-198) at
  //     tp_world_size 1:
  //       conv  = (conv_dim, conv_kernel - 1 + num_spec)
  //       state = (num_heads, head_dim, state_size)
  //     with conv_dim = mamba_num_heads*mamba_head_dim + 2*n_groups*state_size
  //     = 4096 + 2*8*128 = 6144, confirmed on disk by the released
  //     `mixer.conv1d.weight` BF16 [6144, 1, 4].
  //
  //     LAYOUT NOTE: upstream's DEFAULT conv layout is "SD" = (state_len, dim)
  //     (mamba_utils.py:27-48, `VLLM_SSM_CONV_STATE_LAYOUT` unset ⇒ "SD"),
  //     while our local convention across qwen3_5_common.cpp:85 and
  //     kimi_linear_registry.cpp:156 is (dim, state_len). The BYTES are
  //     identical — same product, same page size — and this follows the local
  //     convention so the shared runner/manager code sees one orientation. The
  //     discrepancy is recorded here rather than left for W4 to rediscover.
  //
  //     num_spec is 0: speculative decoding widens the conv row to
  //     (K-1)+k taps, and the MTP head is W5.
  std::vector<std::string> mamba_layers;
  for (int64_t i : p.LayerIndices(NemotronHBlock::kMamba)) {
    mamba_layers.push_back("backbone.layers." + std::to_string(i) + ".mixer");
  }
  kv.kv_cache_groups.emplace_back(
      std::move(mamba_layers),
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {p.conv_dim(), p.conv_kernel - 1},
              {p.mamba_num_heads, p.mamba_head_dim, p.ssm_state_size}},
          std::vector<vt::DType>{conv_dtype, ssm_dtype}));
  return kv;
}

REGISTER_VLLM_MODEL(nemotron_h, "NemotronHForCausalLM", kNemotronHFactory,
                    kNemotronHInfo)

}  // namespace vllm
