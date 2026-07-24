// Shared Qwen3.5/3.6 registry-glue helper definitions (see qwen3_5_common.h).
// Extracted verbatim (behavior-preserving) from the former model_registry.cpp
// monolith so the dense and MoE variant TUs share one copy.
#include "vllm/model_executor/models/qwen3_5_common.h"

#include <cstdlib>
#include <optional>

#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_internal.h"  // ResolveMambaSsmCacheDType
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {

void ParseQwen3_5Config(const HfConfig& config) {
  // LoadHfConfig/HfConfigFromGguf already materialize the consumed Qwen fields.
  // This explicit per-family hook is where a family adds normalization or
  // validation without changing the registry/runner contract.
  (void)config;
}

ForwardLogits HostLogits(std::vector<float>&& host, int64_t vocab) {
  ForwardLogits logits;
  logits.vocab = vocab;
  logits.rows = vocab > 0 ? static_cast<int64_t>(host.size()) / vocab : 0;
  logits.host = std::move(host);
  return logits;
}

v1::KVCacheConfig MakeQwen3_5KVCache(const HfConfig& config, int block_size,
                                     int num_blocks) {
  return MakeQwen3_5KVCacheSpec(config, block_size, num_blocks, /*num_spec=*/0);
}

v1::KVCacheConfig MakeQwen3_5KVCacheSpec(const HfConfig& config, int block_size,
                                         int num_blocks, int num_spec) {
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);
  const int num_value_heads = static_cast<int>(config.linear_num_value_heads);
  const int value_head_dim = static_cast<int>(config.linear_value_head_dim);
  const int key_head_dim = static_cast<int>(config.linear_key_head_dim);
  const int conv_kernel = static_cast<int>(config.linear_conv_kernel_dim);
  const int key_dim =
      static_cast<int>(config.linear_num_key_heads) * key_head_dim;
  const int value_dim = num_value_heads * value_head_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  // Diagnostic state-storage overrides belong to planning, not allocation:
  // the MambaSpec must describe the exact bytes the runner will consume.
  vt::DType conv_dtype = vt::DType::kBF16;
  vt::DType ssm_dtype =
      detail::ResolveMambaSsmCacheDType(config, conv_dtype);
  if (const char* state_dtype = std::getenv("VT_GDN_STATE_BF16")) {
    if (state_dtype[0] == '0') {
      conv_dtype = vt::DType::kF32;
      ssm_dtype = vt::DType::kF32;
    } else if (state_dtype[0] == '1') {
      conv_dtype = vt::DType::kBF16;
      ssm_dtype = vt::DType::kBF16;
    }
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      // The spec is the SINGLE source of truth for the paged-KV storage dtype
      // and layout: the runner sizes the buffer from spec->page_size_bytes()
      // and builds its cache view from the spec's fields (MLA campaign W1).
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      // SPEC-MTP I4: with k speculative tokens the conv row widens to
      // (K-1)+k taps (mamba_utils.py:226 `conv_kernel_size - 1 + num_spec`) so
      // the sliding window can be rewound to the accepted count, and
      // num_speculative_blocks = k gives MambaManager the k+1 SSM snapshot slots
      // per request (mamba/abstract.py:55-59). num_spec == 0 is the production
      // default and reproduces the pre-I4 spec byte for byte.
      std::make_shared<v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{
              {conv_dim, conv_kernel - 1 + num_spec},
              {num_value_heads, value_head_dim, key_head_dim}},
          std::vector<vt::DType>{conv_dtype, ssm_dtype},
          /*page_size_padded=*/std::nullopt,
          /*mamba_cache_mode=*/"none",
          /*num_speculative_blocks=*/num_spec));
  return kv;
}

}  // namespace vllm
