// vllm.cpp original GGUF-format dequant loader (porting-inventory.md §9
// deviation). Ported byte-for-byte from llama.cpp @
// 237ad9b961f009ae19ac29dbce4cd0c1251f94b3:
//   ggml/src/ggml-common.h  (block_q4_0/q8_0/q3_K/q4_K/q5_K/q6_K layouts)
//   ggml/src/ggml-quants.c  (dequantize_row_* + get_scale_min_k4).
// The BLOCK decoders themselves now live one layer down in vt
// (src/vt/cpu/cpu_quant_dequant.cpp) and are shared with the compute-in-quant
// GEMM; this file keeps the GGUF-facing validation, the unquantized types, and
// the ggml-type -> vt::DType routing. See gguf_dequant.h for the format
// citation.
// NVFP4 (type 40) decodes HERE rather than in vt, because it reuses the
// already-gated safetensors NVFP4 numerics (nvfp4_dequant.h) and has no vt
// block dtype / vec_dot: only its CONTAINER differs from the safetensors form.
#include "vllm/model_executor/model_loader/gguf_dequant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace vllm {
namespace {

// Read a little-endian ggml_half (f16) at byte pointer `p` and widen to f32.
// (Aligned load is not guaranteed for mmap'd block bytes, so memcpy.)
float ReadF16(const uint8_t* p) {
  uint16_t h = 0;
  std::memcpy(&h, p, sizeof(h));
  return vt::F16ToF32(h);
}

// One 4-bit e2m1 nibble -> f32, with the SAME LUT and sign rule the
// safetensors NVFP4 path uses (nvfp4_dequant.h `kE2M1Lut`). Shared on purpose:
// the two containers hold the same nibbles, so their decoders must not drift.
inline float E2M1ToF32(uint8_t nibble) {
  return kE2M1Lut[nibble & 0x7U] * ((nibble & 0x8U) != 0 ? -1.0F : 1.0F);
}

}  // namespace

bool GgmlTypeNeedsGlobalScale(uint32_t ggml_type) { return ggml_type == 40; }

std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel, float global_scale) {
  VT_CHECK(data != nullptr, "gguf dequant: data is null");
  VT_CHECK(numel >= 0, "gguf dequant: negative numel");
  if (GgmlTypeNeedsGlobalScale(ggml_type)) {
    VT_CHECK(std::isfinite(global_scale) && global_scale > 0.0F,
             "gguf dequant: NVFP4 needs the finite, positive per-tensor "
             "<stem>.scale sidecar value");
  } else {
    // Refuse to DROP a scale the caller passed. Every other encoding here is
    // self-contained, so a non-1.0 scale means the caller has confused two
    // tensors and would otherwise be handed silently wrong weights.
    VT_CHECK(global_scale == 1.0F,
             std::string("gguf dequant: ggml type ")
                 .append(std::to_string(ggml_type))
                 .append(" has self-contained blocks and cannot apply a global "
                         "scale")
                 .c_str());
  }

  const GgmlTypeTraits& traits = GgmlTraits(ggml_type);
  const int64_t block_elems = traits.block_elems;
  VT_CHECK(numel % block_elems == 0,
           std::string("gguf dequant: numel not a multiple of block_elems for ")
               .append(traits.name).c_str());

  std::vector<float> out(static_cast<size_t>(numel));
  float* y = out.data();

  // Guard: our hardcoded per-type block byte stride must equal the reader's
  // traits. A mismatch means a port bug in one place or the other.
  auto check_bytes = [&](int64_t bytes) {
    VT_CHECK(traits.block_bytes == bytes,
             std::string("gguf dequant: block_bytes mismatch vs traits for ")
                 .append(traits.name).c_str());
  };

  switch (ggml_type) {
    case 0:  // F32
      check_bytes(4);
      std::memcpy(y, data, static_cast<size_t>(numel) * sizeof(float));
      break;
    case 1:  // F16 (ggml fp16 = IEEE half; unquantized tensors in mixed files)
      check_bytes(2);
      for (int64_t i = 0; i < numel; ++i) y[i] = ReadF16(data + i * 2);
      break;
    case 30: {  // BF16 (upper 16 bits of f32)
      check_bytes(2);
      for (int64_t i = 0; i < numel; ++i) {
        uint16_t b;
        std::memcpy(&b, data + i * 2, sizeof(b));
        uint32_t u = static_cast<uint32_t>(b) << 16;
        std::memcpy(&y[i], &u, sizeof(u));
      }
      break;
    }
    case 2:    // Q4_0
    case 8:    // Q8_0
    case 11:   // Q3_K
    case 12:   // Q4_K
    case 13:   // Q5_K
    case 14: {  // Q6_K
      // The block decoders moved to vt (src/vt/cpu/cpu_quant_dequant.cpp) so
      // the loader oracle and the compute-in-quant GEMM's generic fallback
      // share ONE implementation. The code is byte-identical to what lived
      // here, so numerics are unchanged; this test suite gates that.
      vt::DType dtype = vt::DType::kF32;
      VT_CHECK(vt::BlockDTypeFromGgmlTypeId(ggml_type, &dtype),
               "gguf dequant: no vt block dtype for this ggml type");
      check_bytes(vt::BlockBytes(dtype));
      VT_CHECK(vt::BlockElems(dtype) == block_elems,
               "gguf dequant: vt block_elems disagrees with reader traits");
      vt::cpu::BlockToFloat(dtype)(data, y, numel);
      break;
    }
    case 40: {  // NVFP4
      // block_nvfp4 = uint8 d[4] then uint8 qs[32], 64 elements in 36 bytes.
      // d[s] is an IEEE fp8-e4m3fn scale for 16-element sub-block s. Sub-block
      // s owns qs[s*8 .. s*8+8); within it, byte j holds element j in the LOW
      // nibble and element j+8 in the HIGH nibble (the ggml split-half packing,
      // NOT the torch pairwise packing the safetensors container uses).
      //
      // Layout established by BYTE-COMPARING the real Qwen3.6-27B-NVFP4 GGUF
      // against the compressed-tensors export of the same quantization run:
      // the scale bytes are identical and the nibbles are exactly this
      // permutation. .agents/specs/gguf-nvfp4-notes.md Sec 5.
      //
      // The arithmetic and its ORDER mirror DequantNvfp4ToBf16: the group scale
      // is formed first (fp8 x global), then multiplied into the e2m1 value, so
      // the two containers agree bit for bit.
      check_bytes(36);
      VT_CHECK(block_elems == 64, "gguf dequant: NVFP4 block_elems must be 64");
      const int64_t nblocks = numel / 64;
      for (int64_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = data + b * 36;
        const uint8_t* qs = blk + 4;
        float* yb = y + b * 64;
        for (int s = 0; s < 4; ++s) {
          const float group_scale = F8E4M3ToF32(blk[s]) * global_scale;
          const uint8_t* sub = qs + s * 8;
          float* ys = yb + s * kNvfp4GroupSize;
          for (int j = 0; j < 8; ++j) {
            const uint8_t byte = sub[j];
            ys[j] = E2M1ToF32(byte & 0x0FU) * group_scale;
            ys[j + 8] = E2M1ToF32(byte >> 4) * group_scale;
          }
        }
      }
      break;
    }
    default:
      throw std::runtime_error("gguf dequant: unsupported ggml type " +
                               std::to_string(ggml_type) +
                               " (" + traits.name + ") (Task 2/i-quant)");
  }
  return out;
}

std::vector<float> DequantGgufRowToF32(uint32_t ggml_type, const uint8_t* data,
                                       int64_t numel) {
  VT_CHECK(!GgmlTypeNeedsGlobalScale(ggml_type),
           std::string("gguf dequant: ggml type ")
               .append(std::to_string(ggml_type))
               .append(" blocks are not self-contained; dequant it through the "
                       "4-argument overload with its <stem>.scale sidecar")
               .c_str());
  return DequantGgufRowToF32(ggml_type, data, numel, 1.0F);
}

std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel,
                                           float global_scale) {
  const std::vector<float> f32 =
      DequantGgufRowToF32(ggml_type, data, numel, global_scale);
  std::vector<uint16_t> out(f32.size());
  for (size_t i = 0; i < f32.size(); ++i) out[i] = vt::F32ToBF16(f32[i]);
  return out;
}

std::vector<uint16_t> DequantGgufRowToBf16(uint32_t ggml_type,
                                           const uint8_t* data, int64_t numel) {
  VT_CHECK(!GgmlTypeNeedsGlobalScale(ggml_type),
           std::string("gguf dequant: ggml type ")
               .append(std::to_string(ggml_type))
               .append(" blocks are not self-contained; dequant it through the "
                       "4-argument overload with its <stem>.scale sidecar")
               .c_str());
  return DequantGgufRowToBf16(ggml_type, data, numel, 1.0F);
}

}  // namespace vllm
