// The paged KV-cache STORAGE dtype, resolved in ONE place.
//
// No single upstream twin: vLLM carries the KV storage dtype on the cache
// config (`CacheConfig.cache_dtype` -> `AttentionSpec.dtype`, consumed by
// `kv_cache_interface.py:380-398`). We mirror the SHAPE of that contract — the
// KV cache SPEC is the single source of truth for the storage dtype, and the
// allocator sizes buffers from `spec->page_size_bytes()` — while keeping our
// own `VT_KV_CACHE_F32` same-binary A/B as the thing that picks the value.
//
// DEFAULT: bf16 (vLLM's bf16 flash_attn KV store — halves KV memory vs f32).
// `VT_KV_CACHE_F32=1` selects f32 for the A/B. Zero bytes are +0.0f in both.
//
// Every producer of an attention KV-cache spec (the model KV-cache factories
// and the runner tests) MUST build its spec with this dtype, because the runner
// now derives BOTH the allocation size and the cache view from the spec.
#ifndef VLLM_V1_KV_CACHE_DTYPE_H_
#define VLLM_V1_KV_CACHE_DTYPE_H_

#include <cstdlib>
#include <string>
#include <string_view>

#include "vt/dtype.h"
#include "vt/fp8_kv.h"

namespace vllm::v1 {

inline vt::DType ResolveKvCacheDType() {
  const char* kv_f32_env = std::getenv("VT_KV_CACHE_F32");
  return (kv_f32_env != nullptr && kv_f32_env[0] == '1') ? vt::DType::kF32
                                                         : vt::DType::kBF16;
}

// Resolution of vLLM's `CacheConfig.cache_dtype` string (config/cache.py:19-36,76)
// into what our KV cache + ops need: the STORAGE dtype the allocator sizes blocks
// from, plus the fp8 read/write interpretation. Mirrors the `CacheDType` Literal
// and `is_quantized_kv_cache` (utils/torch_utils.py) contract.
//
// KV-FP8 W1 owns the fp8-e4m3 family here; the other CacheDType members
// (fp8_e5m2 compute, fp8_inc, fp8_ds_mla, turboquant_*, *_per_token_head, nvfp4)
// are owned by later rows (KV-NVFP4-TURBO, fp8_ds_mla) and are parsed to their
// selector but refused by the CPU compute path until those bricks land.
struct ResolvedCacheDType {
  bool is_fp8 = false;  // quantized fp8 KV (cache pages are 1-byte fp8 / kI8)
  vt::DType storage = vt::DType::kBF16;  // block-allocation dtype
  vt::Fp8KVCacheDataType fp8_kind = vt::Fp8KVCacheDataType::kAuto;
};

// mirror vllm.utils.torch_utils.is_quantized_kv_cache: anything other than
// "auto" (i.e. a non-model storage dtype) is a quantized KV cache. The explicit
// float aliases ("float16"/"bfloat16") are NOT quantized.
inline bool IsQuantizedKvCache(std::string_view cache_dtype) {
  return cache_dtype != "auto" && cache_dtype != "float16" && cache_dtype != "bfloat16";
}

// Parse a CacheDType string. `model_dtype` is the resolved model storage dtype
// used for the "auto" path (config/cache.py:76 "If auto, use model data type").
inline ResolvedCacheDType ParseCacheDType(std::string_view cache_dtype, vt::DType model_dtype) {
  ResolvedCacheDType r;
  if (cache_dtype == "auto") {
    r.storage = model_dtype;
    r.fp8_kind = vt::Fp8KVCacheDataType::kAuto;
    return r;
  }
  if (cache_dtype == "float16") {
    r.storage = vt::DType::kF16;
    return r;
  }
  if (cache_dtype == "bfloat16") {
    r.storage = vt::DType::kBF16;
    return r;
  }
  // fp8 == fp8_e4m3 (config/cache.py:76 "CUDA 11.8+ supports fp8 (=fp8_e4m3)").
  if (cache_dtype == "fp8" || cache_dtype == "fp8_e4m3") {
    r.is_fp8 = true;
    r.storage = vt::DType::kI8;  // 1-byte fp8 storage
    r.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
    return r;
  }
  if (cache_dtype == "fp8_e5m2") {
    r.is_fp8 = true;
    r.storage = vt::DType::kI8;
    r.fp8_kind = vt::Fp8KVCacheDataType::kFp8E5M2;
    return r;
  }
  VT_CHECK(false, std::string("cache_dtype '") + std::string(cache_dtype) +
                      "' is not implemented in KV-FP8 W1 (fp8/fp8_e4m3/fp8_e5m2, "
                      "auto/float16/bfloat16 only; fp8_inc/fp8_ds_mla/turboquant_*/"
                      "*_per_token_head/nvfp4 are owned by later rows)");
  return r;
}

}  // namespace vllm::v1

#endif  // VLLM_V1_KV_CACHE_DTYPE_H_
