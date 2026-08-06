// CPU half of the MiniMax-H3 DiT device-forward glue table (vt::OpId::kMiniMaxH3).
//
// Three small ops the shared vt:: surface does not cover; see
// include/vllm/model_executor/models/minimax_h3_device.h for why the table is this
// short (everything else in the DiT forward reuses tuned shared ops).
//
// Each kernel is a 1:1 transcription of the host reference it stands in for
// (minimax_h3.cpp: ModulateScaleShift / ModulateGate / Silu), in the same
// arithmetic order. The CUDA sibling lives in src/vt/cuda/cuda_minimax_h3.cu.
//
// DTYPE: arithmetic is ALWAYS f32; only the load/store width varies. A bf16 stream
// therefore rounds on STORE, which is exactly upstream's cast point -- no separate
// rounding pass, and no way for the fused store to drift from it.
//
// Registering this on kCPU (Laguna's equivalent table is CUDA-only) is what lets
// the whole device-forward code path be covered by CPU CI, so a GPU is needed to
// gate the KERNELS, not the port's structure.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vllm/model_executor/models/minimax_h3_device.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vt::cpu {
namespace {

// bf16 is stored as the high 16 bits of the f32 pattern; round-to-nearest-even on
// store, matching torch's `.to(bfloat16)` and minimax_h3.cpp's RoundBf16.
inline float LoadBf16(const void* p, int64_t i) {
  const uint32_t bits = static_cast<uint32_t>(static_cast<const uint16_t*>(p)[i]) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline void StoreBf16(void* p, int64_t i, float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u) {  // inf/nan pass through
    bits &= 0xFFFF0000u;
  } else {
    const uint32_t lsb = (bits >> 16) & 1u;
    bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
  }
  static_cast<uint16_t*>(p)[i] = static_cast<uint16_t>(bits >> 16);
}

inline float Load(const void* p, int64_t i, bool bf16) {
  return bf16 ? LoadBf16(p, i) : static_cast<const float*>(p)[i];
}

inline void Store(void* p, int64_t i, float v, bool bf16) {
  if (bf16) {
    StoreBf16(p, i, v);
  } else {
    static_cast<float*>(p)[i] = v;
  }
}

// _modulate_scale_shift (minimax_h3_transformer.py:183-192).
void MiniMaxH3ModulateScaleShift(Queue&, void* x, const void* shift, const void* scale,
                                 const int32_t* idx, int64_t rows, int64_t width,
                                 int64_t src_stride, DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t row = idx[r];
    for (int64_t i = 0; i < width; ++i) {
      const float v = Load(x, r * width + i, bf16) * (1.0f + Load(scale, row * src_stride + i, bf16)) +
                      Load(shift, row * src_stride + i, bf16);
      Store(x, r * width + i, v, bf16);
    }
  }
}

// _modulate_gate (minimax_h3_transformer.py:195-204).
void MiniMaxH3ModulateGate(Queue&, void* residual, const void* gate, const void* other,
                           const int32_t* idx, int64_t rows, int64_t width, int64_t src_stride,
                           DType dtype) {
  const bool bf16 = dtype == DType::kBF16;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t row = idx[r];
    for (int64_t i = 0; i < width; ++i) {
      const float v = Load(residual, r * width + i, bf16) +
                      Load(gate, row * src_stride + i, bf16) * Load(other, r * width + i, bf16);
      Store(residual, r * width + i, v, bf16);
    }
  }
}

// Matches minimax_h3.cpp's Silu EXACTLY (x / (1 + exp(-x))); the algebraically
// equivalent x * sigmoid(x) is NOT bit-identical, and this path is gated against
// the host reference.
void MiniMaxH3Silu(Queue&, float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

const vllm::minimax_h3::MiniMaxH3DeviceKernels kKernels{
    &MiniMaxH3ModulateScaleShift,
    &MiniMaxH3ModulateGate,
    &MiniMaxH3Silu,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMiniMaxH3, DeviceType::kCPU,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
