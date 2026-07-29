// fp8 KV-cache storage codec + dtype selector (KV-FP8 W1).
//
// vt-layer home for the fp8-e4m3fn <-> f32 conversions the paged KV cache uses,
// plus the `Fp8KVCacheDataType` selector that mirrors vLLM's kernel-level
// `Fp8KVCacheDataType` enum (csrc/attention/dtype_fp8.cuh:9-13 @ 555967922) and
// the per-tensor k/v scale contract.
//
// The vt runtime deliberately does NOT depend on the vllm:: model layer
// (see src/vt/cpu/cpu_ops.cpp:408-410), so the e4m3 math is re-stated here as
// the canonical vt-layer copy. It is BIT-IDENTICAL to the landed codecs
//   vllm::F8E4M3ToF32   (src/vllm/model_executor/model_loader/nvfp4_dequant.cpp:11)
//   vllm::F32ToF8E4M3   (src/vllm/model_executor/layers/quantization/
//                        compressed_tensors/nvfp4_emulation.cpp:14)
// and to the in-file copies in cpu_ops.cpp (Fp8ToF32 / F32ToFp8), which a later
// cleanup can consolidate onto these (recorded in .agents/specs/fp8-kv-cache.md).
//
// SCALE CONVENTION (mirror vLLM quant_utils.cuh:296-300 "Convention of the
// scale"): the stored fp8 byte is Quantize(HP / scale); the read dequant is
// Dequant(fp8) * scale. So StoreKvFp8 divides by the scale, LoadKvFp8 multiplies.
#ifndef VT_FP8_KV_H_
#define VT_FP8_KV_H_

#include <cmath>
#include <cstdint>
#include <limits>

namespace vt {

// The KV-cache storage interpretation of a 1-byte (DType::kI8) cache page.
// Mirrors vLLM's Fp8KVCacheDataType (dtype_fp8.cuh:9-13): the cache tensor's
// element type is a raw byte and the *interpretation* travels as this enum,
// exactly as vLLM passes CACHE_T=uint8_t + the KV_DTYPE template param.
enum class Fp8KVCacheDataType : uint8_t {
  kAuto = 0,   // not fp8: the cache holds the model float dtype (no scale)
  kFp8E4M3,    // IEEE fp8-e4m3fn (bias 7, finite, NaN at 0x7F/0xFF)
  kFp8E5M2,    // IEEE fp8-e5m2 (bias 15, has inf/nan) — CPU compute is a later brick
};

// IEEE fp8-e4m3fn byte -> f32. Bit-matches vllm::F8E4M3ToF32.
inline float F8E4M3ToF32(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0F : 1.0F;
  if (exp == 0xFU && mant == 0x7U) return std::numeric_limits<float>::quiet_NaN();
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0F / 512.0F));
  const float mantissa = 1.0F + static_cast<float>(mant) * (1.0F / 8.0F);
  return sm * std::ldexp(mantissa, static_cast<int>(exp) - 7);
}

// f32 -> fp8-e4m3fn byte, round-to-nearest-even, SATURATING to +/-448 (no inf;
// NaN only at 0x7F/0xFF). Bit-matches vllm::F32ToF8E4M3.
inline uint8_t F32ToF8E4M3(float f) {
  constexpr float kFp8Max = 448.0F;
  if (std::isnan(f)) return 0x7FU;
  const uint8_t sign = std::signbit(f) ? 0x80U : 0x00U;
  const float a = std::fabs(f);
  if (!std::isfinite(a) || a >= kFp8Max) return static_cast<uint8_t>(sign | 0x7EU);
  if (a == 0.0F) return sign;
  int e2 = 0;
  const float frac = std::frexp(a, &e2);
  int exp_field = (e2 - 1) + 7;
  if (exp_field <= 0) {
    const double qd = static_cast<double>(a) * 512.0;
    const int qi = static_cast<int>(std::nearbyint(qd));
    if (qi <= 0) return sign;
    if (qi < 8) return static_cast<uint8_t>(sign | static_cast<uint8_t>(qi));
    return static_cast<uint8_t>(sign | (1U << 3));
  }
  const double sig = static_cast<double>(frac) * 2.0;
  int mi = static_cast<int>(std::nearbyint(sig * 8.0));
  if (mi == 16) {
    mi = 8;
    exp_field += 1;
  }
  const int mant = mi - 8;
  if (exp_field > 15 || (exp_field == 15 && mant >= 7)) {
    return static_cast<uint8_t>(sign | 0x7EU);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint8_t>(exp_field) << 3) |
                              static_cast<uint8_t>(mant));
}

// Store: fp8 byte = Quantize(hp / scale). scale must be > 0 (per-tensor k/v
// scale). Mirrors CopyWithScaleOp's fp8 branch (cache_kernels.cu:246-250) via
// fp8::scaled_convert (quant_utils.cuh scale convention).
inline uint8_t StoreKvFp8E4M3(float hp, float scale) {
  return F32ToF8E4M3(hp / scale);
}

// Read: hp = Dequant(fp8) * scale. Mirrors scaled_vec_conversion<float,uint8_t>
// (quant_utils.cuh:302-308): fp8 -> float, then * scale.
inline float LoadKvFp8E4M3(uint8_t byte, float scale) { return F8E4M3ToF32(byte) * scale; }

}  // namespace vt

#endif  // VT_FP8_KV_H_
