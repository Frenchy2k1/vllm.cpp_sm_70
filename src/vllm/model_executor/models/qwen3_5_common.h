// Shared Qwen3.5/3.6 registry-glue helpers used by BOTH per-variant registry
// TUs (qwen3_5_dense.cpp, qwen3_5_moe.cpp). Not installed.
//
// Holds the registry-facing bits the dense and MoE variants share verbatim: the
// per-family _ModelInfo capability record, the config hook, the KV-cache spec
// builder (identical for both variants), the borrowed-weights tag, and the
// host-logits carrier helper. The heavy per-variant forward machinery
// (Qwen3_5Model::/Qwen3_5DenseModel::) stays in qwen3_5.cpp.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"

namespace vllm {

// Tag selecting the borrowing (non-owning) LoadedModel constructor overloads in
// each variant TU (the synthetic in-memory Borrow* adapters).
struct BorrowedWeightsTag {};

// registry.py _ModelInfo for both Qwen3.6 variants (dense + MoE): text
// generation, hybrid attention, multimodal-capable wrappers.
inline constexpr ModelInfo kQwen3_5Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

// Per-family config hook. LoadHfConfig/HfConfigFromGguf already materialize the
// consumed Qwen fields; this explicit hook is where a family adds normalization
// or validation without changing the registry/runner contract.
void ParseQwen3_5Config(const HfConfig& config);

// KV-cache spec builder — identical for the dense and MoE variants (both are the
// same hybrid full-attention + GDN/Mamba layout). This is the `KVCacheSpecBuilder`
// registered in the model factory, so its signature is fixed; it delegates with
// num_spec == 0 (no speculative decoding), the production default.
v1::KVCacheConfig MakeQwen3_5KVCache(const HfConfig& config, int block_size,
                                     int num_blocks);

// Speculative-decoding-aware variant (SPEC-MTP I4). `num_spec` is the resolved
// `num_speculative_tokens` (k); 0 reproduces MakeQwen3_5KVCache exactly. Mirrors
// vllm/model_executor/models/qwen3_5.py:524-543 (`num_spec` from the
// SpeculativeConfig) plus mamba_utils.py:213-234 and mamba/abstract.py:55-59:
//
//   * conv_state row grows to `conv_kernel_size - 1 + num_spec` taps
//     (mamba_utils.py:226) — the sliding window vt::CausalConv1dSpecUpdate
//     advances by the ACCEPTED count. The conv needs no extra SLOTS.
//   * MambaSpec::num_speculative_blocks = num_spec (abstract.py:55-59), which
//     makes MambaManager allocate k+1 state blocks per request
//     (single_type_kv_cache_manager.py:1206-1215) — the k+1 SSM snapshot slots
//     the spec kernel writes, one per draft position.
//
// MEMORY: one SSM slot is Hv*Dv*Dk*4B per GDN layer, so k=1 DOUBLES the GDN SSM
// state per in-flight request (27B: ~144 MiB/request/slot over 48 GDN layers;
// 35B: ~60 MiB over 30). This is inherent to vLLM's scheme; we mirror it.
v1::KVCacheConfig MakeQwen3_5KVCacheSpec(const HfConfig& config, int block_size,
                                         int num_blocks, int num_spec);

// Wraps a host logits vector into a ForwardLogits (rows inferred from vocab).
ForwardLogits HostLogits(std::vector<float>&& host, int64_t vocab);

}  // namespace vllm
