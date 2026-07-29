// vllm.cpp original GGUF-format dequant loader (porting-inventory.md §9
// deviation, like the safetensors reader / gguf_reader.{h,cpp}). No upstream
// vLLM mirror: this is the ggml GGUF quant block format, ported byte-for-byte
// from llama.cpp @ 237ad9b961f009ae19ac29dbce4cd0c1251f94b3:
//
//   ggml/src/ggml-common.h  — block_q4_0, block_q8_0, block_q3_K, block_q4_K,
//                             block_q5_K, block_q6_K struct layouts
//   ggml/src/ggml-quants.c  — dequantize_row_q4_0 / _q8_0 / _q3_K / _q4_K /
//                             _q5_K / _q6_K + get_scale_min_k4 (the 6-bit
//                             super-block scale/min unpack)
//
// The K-quant super-block is 256 elements = 8 sub-blocks of 32 (Q4_K/Q5_K) or
// 16 blocks of 16 (Q3_K/Q6_K); the packed 6-bit scales/mins are the subtle
// part (get_scale_min_k4 for Q4_K/Q5_K, the kmask aux shuffle for Q3_K, plain
// int8 for Q6_K). A wrong bit-unpack yields garbage weights.
//
// NVFP4 (ggml type 40) is a fork/toolchain extension rather than mainline ggml,
// and it is the ONE encoding here whose blocks are not self-contained: the
// tensor also needs a `<stem>.scale` f32 sidecar TENSOR. Its numerics are the
// same NVFP4 the safetensors path already implements
// (nvfp4_dequant.h) — only the container differs. Layout evidence and the
// container comparison: .agents/specs/gguf-nvfp4-notes.md Sec 5.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {

// True when the encoding's blocks are NOT self-contained: the tensor value
// needs a per-tensor scalar that lives OUTSIDE the block bytes, in a GGUF
// sidecar tensor. NVFP4 (40) is the only such encoding we read — see
// .agents/specs/gguf-nvfp4-notes.md Sec 5. Every K-quant, Q4_0/Q8_0, F16/BF16
// and F32 is self-contained and returns false.
//
// This exists so the omission cannot be silent: a caller that has not resolved
// the sidecar must be told, not handed values that are off by a per-tensor
// factor of ~1e4 yet look perfectly finite.
bool GgmlTypeNeedsGlobalScale(uint32_t ggml_type);

// Dequantize `numel` elements from the packed GGUF block bytes at `data` to
// f32. `ggml_type` is the ggml type id (see enum ggml_type / GgufValueType).
// `numel` MUST be a multiple of the type's block_elems (GgmlTraits(type)) —
// GGUF rows always are. Supported types: F32(0), F16(1), Q4_0(2), Q8_0(8),
// Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14), BF16(30) and, through the 4-argument
// overload below, NVFP4(40). Any other id (e.g. IQ2_S(22)/IQ4_XS(23)) throws
// std::runtime_error("unsupported ggml type N (Task 2/i-quant)").
//
// These 3-argument forms THROW for an encoding with
// GgmlTypeNeedsGlobalScale(ggml_type): they have no scale to apply and will not
// guess 1.0.
//
// Reads exactly numel/block_elems * block_bytes bytes from `data`; the caller
// (the GGUF loader) must have validated the tensor span (gguf_reader does).
std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel);

// Same as above but returns bf16 bit patterns (dequant to f32, then
// vt::F32ToBF16 round-to-nearest-even). The Qwen3.6 loader (Task 2) targets
// bf16 OwnedTensors, matching the safetensors path.
std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel);

// The same two entry points, plus the per-tensor scalar the container keeps
// outside the blocks. `global_scale` is the value of the GGUF `<stem>.scale`
// sidecar tensor, applied as a MULTIPLIER: it is the reciprocal of the
// compressed-tensors `weight_global_scale` DIVISOR, i.e. the same
// `weight_scale_2` convention nvfp4_dequant.h already uses (measured
// bit-identical on the real Qwen3.6 exports; see the spec). A stacked expert
// tensor's sidecar holds ONE scalar PER EXPERT, so the caller dequantizes one
// expert slab per call.
//
// `global_scale` must be finite and > 0 for a scale-carrying encoding, and must
// be exactly 1.0 for a self-contained one — handing a scale to an encoding that
// cannot apply it is a caller bug and throws rather than being ignored.
std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel, float global_scale);
std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel,
                                           float global_scale);

// --- NVFP4 (type 40) REPACK — the `C` column's whole mechanism ---------------
//
// Rearrange `rows` x `k` NVFP4 elements from the ggml type-40 container into the
// two operand streams every NVFP4 GEMM in this tree already consumes:
//
//   out_packed [rows, k/2]   two e2m1 nibbles per byte, TORCH-PAIRWISE
//                            (element 2i low, 2i+1 high)
//   out_scale  [rows, k/16]  one IEEE fp8-e4m3fn byte per 16-element group,
//                            LINEAR (row-major, not swizzled)
//
// This is a PURE BYTE PERMUTATION: no value is decoded, rounded or rescaled, and
// the per-tensor `<stem>.scale` sidecar is NOT applied here (it travels on as
// `Nvfp4Weight::scale2`, exactly as `weight_scale_2` does on the safetensors
// side). The output is therefore BIT-IDENTICAL to the `weight_packed` /
// `weight_scale` tensors of the compressed-tensors export of the same
// quantization run — measured, zero differing bytes, on five real Qwen3.6-27B
// projections; see .agents/specs/gguf-nvfp4-native-compute.md Sec B. That
// identity is what lets a GGUF weight enter the already-gated fp4 kernels
// without re-litigating any numerics.
//
// The two containers differ EXACTLY in the nibble order (ggml packs a 16-element
// sub-block as byte j = element j low / element j+8 high; torch packs it
// pairwise) and in whether the scales are interleaved with the nibbles. Getting
// that permutation wrong preserves the value histogram, so it produces finite,
// plausible logits — hence the byte-identity gate rather than a tolerance.
//
// `k` must be a multiple of 64 (the ggml block). `src` holds
// rows * k/64 * 36 bytes; the caller (gguf_reader) has validated the span.
void RepackGgufNvfp4Rows(const uint8_t* src, int64_t rows, int64_t k,
                         uint8_t* out_packed, uint8_t* out_scale);

}  // namespace vllm
