// Architecture-specific NVFP4 bodies split from cuda_matmul_nvfp4.cu for the
// W1 heterogeneous fat binary. This TU is gencode'd only for sm_120a/sm_121a;
// the portable NVFP4 kernels remain in the common TU for all release SMs.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "vt/cuda/cuda_arch_tactics.h"
#include "vt/cuda/cuda_device_caps.h"
#include "vt/cuda/cuda_nvfp4_sm12x.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: nvfp4 sm12x: ") + what +
                             ": " + cudaGetErrorString(err));
  }
}

bool NativeFp4MmaEnabled() {
  const char* value = std::getenv("VT_NVFP4_FP4_NATIVE");
  return value != nullptr && value[0] == '1';
}

bool FusedFp4VectorEnabled() {
  const char* value = std::getenv("VT_FP4_FUSED_VEC");
  return value != nullptr && value[0] == '1';
}

bool PointerAligned(const void* pointer, uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

__device__ __forceinline__ float F8E4M3ToF32Dev(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1u;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFu;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7u;
  const float sign_mul = sign ? -1.0f : 1.0f;
  if (exp == 0xFu && mant == 0x7u) return CUDART_NAN_F;
  if (exp == 0u) return sign_mul * (static_cast<float>(mant) * (1.0f / 512.0f));
  return sign_mul *
         ldexpf(1.0f + static_cast<float>(mant) * (1.0f / 8.0f),
                static_cast<int>(exp) - 7);
}

__device__ __forceinline__ uint8_t F32ToFp8Dev(float value) {
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float ReciprocalApproximateFtz(float value) {
  float reciprocal;
  asm volatile("rcp.approx.ftz.f32 %0, %1;"
               : "=f"(reciprocal)
               : "f"(value));
  return reciprocal;
}

__device__ __forceinline__ int64_t CutlassScaleOffset(
    int64_t row, int64_t column, int64_t padded_cols) {
  const int64_t m_tile = row / 128;
  const int64_t outer_m = row % 32;
  const int64_t inner_m = (row % 128) / 32;
  const int64_t k_tile = column / 4;
  const int64_t inner_k = column % 4;
  return ((((m_tile * (padded_cols / 4) + k_tile) * 32 + outer_m) * 4 +
            inner_m) *
               4 +
           inner_k);
}

struct alignas(32) PackedBf16x16 {
  uint32_t words[8];
};

struct PackedFp4x16 {
  uint32_t lo;
  uint32_t hi;
};

__device__ __forceinline__ uint32_t CanonicalizeFp4NegativeZero(
    uint32_t packed) {
  constexpr uint32_t kMagnitude = 0x77777777u;
  constexpr uint32_t kSign = 0x88888888u;
  const uint32_t magnitude = packed & kMagnitude;
  const uint32_t nonzero_sign =
      ((magnitude << 1) | (magnitude << 2) | (magnitude << 3)) & kSign;
  return packed & (kMagnitude | nonzero_sign);
}

__device__ __forceinline__ void LoadBf16x16Cg(
    PackedBf16x16& value, const __nv_bfloat16* pointer) {
  asm volatile(
      "ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
      : "=r"(value.words[0]), "=r"(value.words[1]),
        "=r"(value.words[2]), "=r"(value.words[3]),
        "=r"(value.words[4]), "=r"(value.words[5]),
        "=r"(value.words[6]), "=r"(value.words[7])
      : "l"(pointer));
}

__device__ __forceinline__ __nv_bfloat162& Bf16Pair(
    PackedBf16x16& value, int index) {
  return reinterpret_cast<__nv_bfloat162*>(value.words)[index];
}

__device__ __forceinline__ PackedFp4x16 PackFp4x16(float2 (&values)[8]) {
  PackedFp4x16 packed;
  asm volatile(
      "{\n"
      ".reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n"
      ".reg .b8 b4; .reg .b8 b5; .reg .b8 b6; .reg .b8 b7;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n"
      "}\n"
      : "=r"(packed.lo), "=r"(packed.hi)
      : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x),
        "f"(values[1].y), "f"(values[2].x), "f"(values[2].y),
        "f"(values[3].x), "f"(values[3].y), "f"(values[4].x),
        "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
        "f"(values[6].x), "f"(values[6].y), "f"(values[7].x),
        "f"(values[7].y));
  packed.lo = CanonicalizeFp4NegativeZero(packed.lo);
  packed.hi = CanonicalizeFp4NegativeZero(packed.hi);
  return packed;
}

__global__ __launch_bounds__(512) void SiluAndMulFp4QuantPackedBf16Kernel(
    uint8_t* __restrict__ packed, uint8_t* __restrict__ scale,
    const __nv_bfloat16* __restrict__ gate_up, float input_global_scale,
    int32_t m_rows, int32_t i_dim, int32_t scale_cols, bool approx_recip) {
  const int32_t groups = i_dim / 16;
  const int32_t group =
      static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
  if (group >= groups) return;
  for (int32_t row = static_cast<int32_t>(blockIdx.x); row < m_rows;
       row += static_cast<int32_t>(gridDim.x)) {
    const int64_t input_row_base = static_cast<int64_t>(row) * 2 * i_dim;
    const int64_t group_base = static_cast<int64_t>(group) * 16;
    PackedBf16x16 gate;
    PackedBf16x16 up;
    LoadBf16x16Cg(gate, gate_up + input_row_base + group_base);
    LoadBf16x16Cg(up, gate_up + input_row_base + i_dim + group_base);
    PackedBf16x16 activation;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
      const float2 gate_pair = __bfloat1622float2(Bf16Pair(gate, pair));
      const float2 up_pair = __bfloat1622float2(Bf16Pair(up, pair));
      const float lo =
          (gate_pair.x / (1.0f + expf(-gate_pair.x))) * up_pair.x;
      const float hi =
          (gate_pair.y / (1.0f + expf(-gate_pair.y))) * up_pair.y;
      Bf16Pair(activation, pair) = __floats2bfloat162_rn(lo, hi);
    }
    __nv_bfloat162 local_max = __habs2(Bf16Pair(activation, 0));
#pragma unroll
    for (int pair = 1; pair < 8; ++pair) {
      local_max = __hmax2(local_max, __habs2(Bf16Pair(activation, pair)));
    }
    const float2 max_pair = __bfloat1622float2(local_max);
    const float vmax = fmaxf(max_pair.x, max_pair.y);
    const float inverse_six =
        approx_recip ? ReciprocalApproximateFtz(6.0f) : (1.0f / 6.0f);
    float sf = input_global_scale * (vmax * inverse_six);
    sf = fminf(fmaxf(sf, -448.0f), 448.0f);
    const uint8_t sf8 = F32ToFp8Dev(sf);
    scale[CutlassScaleOffset(row, group, scale_cols)] = sf8;
    const float sf_value = F8E4M3ToF32Dev(sf8);
    float output_scale = 0.0f;
    if (sf_value != 0.0f) {
      output_scale =
          approx_recip
              ? ReciprocalApproximateFtz(
                    sf_value * ReciprocalApproximateFtz(input_global_scale))
              : input_global_scale / sf_value;
    }
    float2 values[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
      values[pair] = __bfloat1622float2(Bf16Pair(activation, pair));
      values[pair].x *= output_scale;
      values[pair].y *= output_scale;
    }
    const PackedFp4x16 fp4 = PackFp4x16(values);
    const uint64_t packed64 =
        (static_cast<uint64_t>(fp4.hi) << 32) | fp4.lo;
    const int64_t output_byte =
        (static_cast<int64_t>(row) * i_dim + group_base) / 2;
    *reinterpret_cast<uint64_t*>(packed + output_byte) = packed64;
  }
}

int PackedFusedFp4ResidentBlocks() {
  static const int blocks = [] {
    int device = 0;
    int multiprocessors = 0;
    int blocks_per_multiprocessor = 0;
    Check(cudaGetDevice(&device), "packed producer get device");
    Check(cudaDeviceGetAttribute(&multiprocessors,
                                 cudaDevAttrMultiProcessorCount, device),
          "packed producer multiprocessor count");
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks_per_multiprocessor,
              SiluAndMulFp4QuantPackedBf16Kernel, 512, 0),
          "packed producer occupancy");
    return std::max(1, multiprocessors * blocks_per_multiprocessor);
  }();
  return blocks;
}

__device__ __forceinline__ uint8_t GetNib(
    const uint8_t* pointer, int64_t row, int64_t column, int64_t k) {
  const uint8_t byte = pointer[row * (k / 2) + column / 2];
  return (column & 1) ? static_cast<uint8_t>(byte >> 4)
                      : static_cast<uint8_t>(byte & 0xFu);
}

__device__ inline void Store(float* pointer, int64_t index, float value) {
  pointer[index] = value;
}
__device__ inline void Store(__nv_bfloat16* pointer, int64_t index,
                             float value) {
  pointer[index] = __float2bfloat16(value);
}

template <typename Tout>
__global__ void MatmulNvfp4Fp4Native(
    Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
    const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
    int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  const int lane = static_cast<int>(threadIdx.x);
  const int g = lane / 4;
  const int t = lane % 4;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * 8;
  const int64_t groups = k_dim / 16;
  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  for (int64_t k0 = 0; k0 < k_dim; k0 += 64) {
    const int64_t row_a = m0 + g;
    const int64_t row_a8 = row_a + 8;
    const int64_t row_b = n0 + g;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t ka = k0 + t * 8 + j;
      const int64_t kb = k0 + 32 + t * 8 + j;
      if (row_a < m_rows) {
        if (ka < k_dim) {
          a0 |= static_cast<uint32_t>(GetNib(a_packed, row_a, ka, k_dim))
                << (4 * j);
        }
        if (kb < k_dim) {
          a2 |= static_cast<uint32_t>(GetNib(a_packed, row_a, kb, k_dim))
                << (4 * j);
        }
      }
      if (row_a8 < m_rows) {
        if (ka < k_dim) {
          a1 |= static_cast<uint32_t>(GetNib(a_packed, row_a8, ka, k_dim))
                << (4 * j);
        }
        if (kb < k_dim) {
          a3 |= static_cast<uint32_t>(GetNib(a_packed, row_a8, kb, k_dim))
                << (4 * j);
        }
      }
      if (row_b < n_cols) {
        if (ka < k_dim) {
          b0 |= static_cast<uint32_t>(GetNib(b_packed, row_b, ka, k_dim))
                << (4 * j);
        }
        if (kb < k_dim) {
          b1 |= static_cast<uint32_t>(GetNib(b_packed, row_b, kb, k_dim))
                << (4 * j);
        }
      }
    }
    uint32_t scale_a = 0x38383838u;
    uint32_t scale_b = 0x38383838u;
    const int64_t scale_row_a =
        (t == 0) ? (m0 + g) : (t == 1 ? (m0 + g + 8) : -1);
    if (scale_row_a >= 0 && scale_row_a < m_rows) {
      uint32_t value = 0;
#pragma unroll
      for (int block = 0; block < 4; ++block) {
        const int64_t group = k0 / 16 + block;
        const uint8_t scale =
            group < groups ? a_scale[scale_row_a * groups + group] : 0x38u;
        value |= static_cast<uint32_t>(scale) << (8 * block);
      }
      scale_a = value;
    }
    if (t == 0 && row_b < n_cols) {
      uint32_t value = 0;
#pragma unroll
      for (int block = 0; block < 4; ++block) {
        const int64_t group = k0 / 16 + block;
        const uint8_t scale =
            group < groups ? b_scale[row_b * groups + group] : 0x38u;
        value |= static_cast<uint32_t>(scale) << (8 * block);
      }
      scale_b = value;
    }
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {%15, %16}, %17, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(scale_a),
          "n"(0), "n"(0), "r"(scale_b), "n"(0), "n"(0));
  }
  const int64_t row = m0 + g;
  const int64_t row8 = row + 8;
  const int64_t col0 = n0 + t * 2;
  const int64_t col1 = col0 + 1;
  if (row < m_rows && col0 < n_cols) {
    Store(out, row * n_cols + col0, alpha * d0);
  }
  if (row < m_rows && col1 < n_cols) {
    Store(out, row * n_cols + col1, alpha * d1);
  }
  if (row8 < m_rows && col0 < n_cols) {
    Store(out, row8 * n_cols + col0, alpha * d2);
  }
  if (row8 < m_rows && col1 < n_cols) {
    Store(out, row8 * n_cols + col1, alpha * d3);
  }
}

bool Sm12xFp4MmaSupports(const DeviceCaps& caps) {
  return caps.valid && caps.sm_major == 12 && NativeFp4MmaEnabled();
}

bool Sm12xFp4MmaLaunch(const DeviceCaps&, void* args_value) {
  const auto& args = *static_cast<const Nvfp4Fp4MmaArgs*>(args_value);
  auto* stream = static_cast<cudaStream_t>(args.stream);
  const dim3 grid(static_cast<unsigned>((args.n + 7) / 8),
                  static_cast<unsigned>((args.m + 15) / 16));
  switch (args.out_dtype) {
    case DType::kF32:
      MatmulNvfp4Fp4Native<float><<<grid, 32, 0, stream>>>(
          static_cast<float*>(args.out), args.a_packed, args.a_scale,
          args.b_packed, args.b_scale, args.alpha, args.m, args.n, args.k);
      break;
    case DType::kBF16:
      MatmulNvfp4Fp4Native<__nv_bfloat16><<<grid, 32, 0, stream>>>(
          static_cast<__nv_bfloat16*>(args.out), args.a_packed, args.a_scale,
          args.b_packed, args.b_scale, args.alpha, args.m, args.n, args.k);
      break;
    default:
      return false;
  }
  Check(cudaGetLastError(), "native fp4 MMA launch");
  return true;
}

struct Fp4MmaTacticRegistrar {
  Fp4MmaTacticRegistrar() {
    RegisterArchTactic(
        TacticFamily::kNvfp4Fp4Mma,
        ArchTactic{"nvfp4-fp4-mma/sm12x", &Sm12xFp4MmaSupports,
                   &Sm12xFp4MmaLaunch});
  }
} fp4_mma_tactic_registrar;

}  // namespace

bool TryLaunchSiluAndMulFp4QuantPackedSm12x(
    void* stream_value, uint8_t* packed, uint8_t* scale, const void* gate_up,
    float input_global_scale, int64_t m_rows, int64_t i_dim,
    int64_t scale_cols, int64_t scale_numel, bool approx_recip) {
  const DeviceCaps& caps = GetDeviceCaps();
  const bool eligible =
      FusedFp4VectorEnabled() && caps.valid && caps.sm_major == 12 &&
      m_rows <= std::numeric_limits<int32_t>::max() &&
      i_dim <= std::numeric_limits<int32_t>::max() &&
      scale_cols <= std::numeric_limits<int32_t>::max() &&
      PointerAligned(gate_up, 32) && PointerAligned(packed, 8);
  if (!eligible) return false;
  auto* stream = static_cast<cudaStream_t>(stream_value);
  Check(cudaMemsetAsync(scale, 0, static_cast<size_t>(scale_numel), stream),
        "packed producer zero scale");
  const int groups = static_cast<int>(i_dim / 16);
  const int block = std::min(groups, 512);
  const int grid_y = (groups + block - 1) / block;
  const int grid_x = std::min(
      static_cast<int>(m_rows),
      std::max(1, PackedFusedFp4ResidentBlocks() / grid_y));
  SiluAndMulFp4QuantPackedBf16Kernel<<<dim3(grid_x, grid_y), block, 0,
                                       stream>>>(
      packed, scale, static_cast<const __nv_bfloat16*>(gate_up),
      input_global_scale, static_cast<int32_t>(m_rows),
      static_cast<int32_t>(i_dim), static_cast<int32_t>(scale_cols),
      approx_recip);
  Check(cudaGetLastError(), "packed producer launch");
  return true;
}

}  // namespace vt::cuda
