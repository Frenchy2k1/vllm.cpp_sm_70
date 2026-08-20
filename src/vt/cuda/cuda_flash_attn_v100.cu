// SPDX-License-Identifier: BSD-3-Clause
// sm70 FA2 fragment-core build gate (Phase 2, brick E).
//
// Vendors and compiles the *Volta WMMA fragment abstraction* from 1Cat-vLLM's
// flash-attention-v100 (BSD-3-Clause, copyright (c) 2025 D.Skryabin — see
// third_party/flash_attn_v100/COPYING-flash-attn-v100 — the torch-free device
// layer that FA2-on-Volta builds on: `volta::fragment<...>` fragments + loads +
// MMA, sm_70 only by construction, `#error` above 7.0). The full paged
// attention kernels (cuda_sm70_flash_attn.cu) use the same vendored core; this
// TU only instantiates the fragment abstraction so a build fails if the
// abstraction breaks for Volta.
//
// This cell only ever compiles for sm_70 (feature `sm70-fa2-v1`); the TU's
// content is a compile-time instantiations of the vendored core so a build
// fail if the fragment abstraction breaks for Volta.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../../third_party/flash_attn_v100/include/fused_mma.h"  // volta::fragment<...>
#include "../../../third_party/flash_attn_v100/kernel/flash_v100_traits.cuh"

namespace vt::cuda {

// Compile-time instantiation of the fragment core (never launched; any
// reference here makes the abstraction observable to the build).
static_assert(sizeof(volta::fragment<volta::matrix_a, 16, 16, 16, half,
                                        volta::row_major>) ==
                  sizeof(uint32_t) * 8,
              "fa2-v100: fragment<matrix_a,16,16,16> layout regressed");
static_assert(sizeof(volta::fragment<volta::accumulator, 16, 16, 16, float>) ==
                  sizeof(float) * 8,
              "fa2-v100: accumulator fragment layout regressed");

// A device function that forces the fragment arithmetic into the sm_70 pass
// (it is never launched — registers a symbol so the core files' device code
// participates in the compile, not just the headers).
__global__ void Fa2V100ObjectiveFragment(const half* __restrict__ a,
                                         const half* __restrict__ b,
                                         float* __restrict__ out) {
  (void)a;
  (void)b;
  volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major> fa;
  const half2 h2 = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
  *reinterpret_cast<half2*>(&fa.x[0]) = h2;
  out[0] = 0.f;
  (void)fa;
}
}  // namespace vt::cuda