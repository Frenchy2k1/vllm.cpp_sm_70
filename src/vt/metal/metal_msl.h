// Metal backend — the embedded Metal Shading Language source (BACKEND-METAL-MLX,
// W0 skeleton). vllm.cpp original: vLLM has no Metal kernels to mirror, so the
// PER-ELEMENT MATH of every kernel here is ported 1:1 from our own CPU reference
// kernels (which are themselves the vLLM-parity goldens), and only the DISPATCH
// SHAPE is ported from llama.cpp's Metal backend.
//
// Math ported FROM (line-for-line correspondence noted per kernel below):
//   src/vt/cpu/cpu_ops.cpp:225-250  RmsNormKernel        -> vt_rms_norm
//   src/vt/cpu/cpu_ops.cpp:252-264  SiluAndMulKernel     -> vt_silu_and_mul
//   src/vt/cpu/cpu_ops.cpp:1436-1451 CastBf16/CastF32Kernel -> vt_cast
//   src/vt/cpu/cpu_ops.cpp:1649-1695 FusedChainInterpKernel -> vt_fused_chain
//   src/vt/cpu/cpu_layernorm.cpp:49-73 LayerNormKernel   -> vt_layer_norm
//   src/vt/cpu/cpu_layernorm.cpp:75-85 ReluKernel        -> vt_relu
//   src/vt/cpu/cpu_layernorm.cpp:87-99 AddKernel         -> vt_add
//   src/vt/cpu/cpu_ops.cpp MatmulKernel/MatmulBTKernel  -> vt_matmul
//   src/vt/dtype.cpp:224-233        BF16<->F32 codec     -> vt_bf16_to_f32 /
//                                                           vt_f32_to_bf16
//
// Dispatch shape ported FROM llama.cpp `ggml/src/ggml-metal/` @ 237ad9b96:
//   * flat elementwise ops dispatch one thread per element
//     (`ggml_metal_op_bin` / `kernel_add`);
//   * row-reducing ops dispatch ONE THREADGROUP PER ROW with a threadgroup-memory
//     tree reduction (`ggml_metal_op_norm` / `kernel_rms_norm`).
//
// NUMERICS. The reduction ORDER necessarily differs from the CPU reference: the
// CPU tier's reproducibility comes from a fixed SEQUENTIAL accumulation
// (src/vt/cpu/cpu_quant_dot.cpp:22-28, deliberate) and no GPU tree reduction
// preserves it. Per the spike § Gates the bar is therefore NMSE <= 5e-4 for
// reducing ops, NOT bit-exactness. Non-reducing, non-arithmetic paths (Copy /
// Memset / a same-dtype cast) ARE bit-exact and are gated as such.
// The library is compiled with MTLMathModeSafe (see metal_context.mm) so `exp`,
// `sqrt` and the arithmetic below keep IEEE semantics rather than the Metal
// default fast-math relaxations.
#ifndef VT_METAL_METAL_MSL_H_
#define VT_METAL_METAL_MSL_H_

namespace vt::metal {

// clang-format off
inline constexpr const char* kMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Storage dtype codes. These mirror the three FLOAT entries of vt::DType and are
// translated host-side by DtypeCode() in metal_ops.mm — the shader never sees
// vt::DType itself.
#define VT_DT_F32  0u
#define VT_DT_F16  1u
#define VT_DT_BF16 2u

// --- bf16 codec, bit-identical to src/vt/dtype.cpp:224-233 -------------------
// BF16ToF32 is a pure 16-bit left shift; F32ToBF16 is round-to-nearest-EVEN with
// the NaN-quieting special case. Reproduced exactly so a bf16 round-trip through
// a Metal kernel rounds the same way the CPU reference does.
inline float vt_bf16_to_f32(ushort b) {
  return as_type<float>(uint(b) << 16);
}
inline ushort vt_f32_to_bf16(float f) {
  uint u = as_type<uint>(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
    return ushort((u >> 16) | 0x0040u);   // nan: keep quiet, truncate
  }
  uint rounding = 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
  return ushort((u + rounding) >> 16);
}

// --- dtype-erased element access, mirroring cpu_ops.cpp:27-43 LoadF32/StoreF32.
// Reduced-width outputs round ON STORE, exactly once, like the CPU reference.
inline float vt_load(device const uchar* base, uint dt, ulong idx) {
  if (dt == VT_DT_F32)  { return ((device const float*)base)[idx]; }
  if (dt == VT_DT_F16)  { return float(((device const half*)base)[idx]); }
  return vt_bf16_to_f32(((device const ushort*)base)[idx]);
}
inline void vt_store(device uchar* base, uint dt, ulong idx, float v) {
  if (dt == VT_DT_F32)  { ((device float*)base)[idx] = v; return; }
  if (dt == VT_DT_F16)  { ((device half*)base)[idx] = half(v); return; }
  ((device ushort*)base)[idx] = vt_f32_to_bf16(v);
}

// silu/sigmoid in f32, matching cpu_ops.cpp:1646 FSigmoid and the `gate / (1 +
// exp(-gate))` spelling of SiluAndMulKernel.
inline float vt_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Threadgroup tree reduction over `tg` lanes. `tg` is always a power of two
// (host-side ChooseThreadgroupSize guarantees it), so the halving loop is exact.
//
// The LEADING barrier makes this safe to call MORE THAN ONCE per kernel, which
// vt_layer_norm does (mean, then variance) and vt_fused_chain can do (one per
// kRmsNorm step). Without it there is a real race: every thread reads smem[0] at
// the end of one call, and a thread that races ahead into the next call would
// overwrite smem[0] while a slower thread is still reading the previous result.
// It is NOT redundant with the trailing barrier inside the loop, which only
// orders the reduction itself.
inline float vt_tg_sum(threadgroup float* smem, uint tid, uint tg, float v) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  smem[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg / 2u; s > 0u; s >>= 1u) {
    if (tid < s) { smem[tid] += smem[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return smem[0];
}

// Same tree shape, `max` instead of `+`. Used by the paged-attention softmax to
// find the running row maximum. Same leading-barrier rule, same reason: it is
// called once per key CHUNK inside a loop, so consecutive calls must not race.
inline float vt_tg_max(threadgroup float* smem, uint tid, uint tg, float v) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  smem[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tg / 2u; s > 0u; s >>= 1u) {
    if (tid < s) { smem[tid] = max(smem[tid], smem[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return smem[0];
}

// ===========================================================================
// Flat elementwise kernels — one thread per output element.
// ===========================================================================

struct VtElemParams {
  uint  n;        // total elements
  uint  d;        // last-dim extent (for the rank-1 bias broadcast)
  uint  a_dt;
  uint  b_dt;
  uint  out_dt;
  uint  bcast;    // 1 => b is rank-1 [d] and is indexed by (i % d)
};

// cpu_layernorm.cpp:87-99 AddKernel, including the nn.Linear bias row-broadcast.
kernel void vt_add(device const uchar* a   [[buffer(0)]],
                   device const uchar* b   [[buffer(1)]],
                   device uchar*       out [[buffer(2)]],
                   constant VtElemParams& p [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) { return; }
  ulong bi = p.bcast != 0u ? ulong(gid % p.d) : ulong(gid);
  vt_store(out, p.out_dt, ulong(gid), vt_load(a, p.a_dt, ulong(gid)) + vt_load(b, p.b_dt, bi));
}

// cpu_layernorm.cpp:75-85 ReluKernel.
kernel void vt_relu(device const uchar* x   [[buffer(0)]],
                    device uchar*       out [[buffer(1)]],
                    constant VtElemParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) { return; }
  float v = vt_load(x, p.a_dt, ulong(gid));
  vt_store(out, p.out_dt, ulong(gid), v > 0.0f ? v : 0.0f);
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — both are the identity
// through LoadF32/StoreF32, i.e. a dtype-converting copy. When src and dst dtype
// are equal this is a bit-exact copy (gated as such).
kernel void vt_cast(device const uchar* x   [[buffer(0)]],
                    device uchar*       out [[buffer(1)]],
                    constant VtElemParams& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) { return; }
  vt_store(out, p.out_dt, ulong(gid), vt_load(x, p.a_dt, ulong(gid)));
}

struct VtSiluMulParams {
  uint t;
  uint d;        // HALF the input inner dim: x is [t, 2*d], out is [t, d]
  uint x_dt;
  uint out_dt;
};

// cpu_ops.cpp:252-264 SiluAndMulKernel: out[i,j] = silu(x[i,j]) * x[i,d+j].
kernel void vt_silu_and_mul(device const uchar* x   [[buffer(0)]],
                            device uchar*       out [[buffer(1)]],
                            constant VtSiluMulParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= p.t * p.d) { return; }
  uint i = gid / p.d;
  uint j = gid % p.d;
  ulong base = ulong(i) * ulong(2u * p.d);
  float gate = vt_load(x, p.x_dt, base + ulong(j));
  float up   = vt_load(x, p.x_dt, base + ulong(p.d) + ulong(j));
  vt_store(out, p.out_dt, ulong(i) * ulong(p.d) + ulong(j), gate * vt_sigmoid(gate) * up);
}

// ===========================================================================
// Row-reducing kernels — one THREADGROUP per row (llama.cpp kernel_rms_norm
// shape). VT_TG_MAX matches the M4's maxTotalThreadsPerThreadgroup (1024); the
// threadgroup scratch is 4 KiB, well inside the 32 KiB limit.
// ===========================================================================

#define VT_TG_MAX 1024

struct VtRmsParams {
  uint  t;
  uint  h;
  uint  x_dt;
  uint  w_dt;
  uint  out_dt;
  uint  res_dt;
  uint  has_res;
  uint  gemma;
  uint  tg;
  float eps;
};

// cpu_ops.cpp:225-250 RmsNormKernel. The residual idiom is reproduced EXACTLY,
// including the deliberate store-then-RE-READ that makes a bf16 residual stream
// bit-faithful (cpu_ops.cpp:235-237): add in f32, round into the residual's own
// dtype, then read the ROUNDED value back for both the variance and the scale.
kernel void vt_rms_norm(device const uchar* x   [[buffer(0)]],
                        device const uchar* w   [[buffer(1)]],
                        device uchar*       out [[buffer(2)]],
                        device uchar*       res [[buffer(3)]],
                        constant VtRmsParams& p [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  ulong rbase = ulong(row) * ulong(p.h);

  float partial = 0.0f;
  for (uint j = tid; j < p.h; j += p.tg) {
    float v = vt_load(x, p.x_dt, rbase + ulong(j));
    if (p.has_res != 0u) {
      v += vt_load(res, p.res_dt, rbase + ulong(j));
      vt_store(res, p.res_dt, rbase + ulong(j), v);
      v = vt_load(res, p.res_dt, rbase + ulong(j));
    }
    partial += v * v;
  }
  if (p.has_res != 0u) { threadgroup_barrier(mem_flags::mem_device); }
  float sumsq = vt_tg_sum(smem, tid, p.tg, partial);
  float inv = 1.0f / sqrt(sumsq / float(p.h) + p.eps);

  for (uint j = tid; j < p.h; j += p.tg) {
    float v = p.has_res != 0u ? vt_load(res, p.res_dt, rbase + ulong(j))
                              : vt_load(x, p.x_dt, rbase + ulong(j));
    float wj = vt_load(w, p.w_dt, ulong(j));
    if (p.gemma != 0u) { wj += 1.0f; }
    vt_store(out, p.out_dt, rbase + ulong(j), v * inv * wj);
  }
}

struct VtLayerNormParams {
  uint  rows;
  uint  d;
  uint  x_dt;
  uint  w_dt;
  uint  b_dt;
  uint  out_dt;
  uint  has_w;
  uint  has_b;
  uint  tg;
  float eps;
};

// cpu_layernorm.cpp:49-73 LayerNormKernel — the two-pass numerically stable form
// (mean, then squared deviations ABOUT that mean), BIASED (1/N) variance, f32
// accumulation, one rounding on store.
kernel void vt_layer_norm(device const uchar* x   [[buffer(0)]],
                          device const uchar* w   [[buffer(1)]],
                          device const uchar* b   [[buffer(2)]],
                          device uchar*       out [[buffer(3)]],
                          constant VtLayerNormParams& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  ulong base = ulong(row) * ulong(p.d);

  float psum = 0.0f;
  for (uint i = tid; i < p.d; i += p.tg) { psum += vt_load(x, p.x_dt, base + ulong(i)); }
  float mean = vt_tg_sum(smem, tid, p.tg, psum) / float(p.d);

  float psq = 0.0f;
  for (uint i = tid; i < p.d; i += p.tg) {
    float dv = vt_load(x, p.x_dt, base + ulong(i)) - mean;
    psq += dv * dv;
  }
  float rstd = 1.0f / sqrt(vt_tg_sum(smem, tid, p.tg, psq) / float(p.d) + p.eps);

  for (uint i = tid; i < p.d; i += p.tg) {
    float v = (vt_load(x, p.x_dt, base + ulong(i)) - mean) * rstd;
    if (p.has_w != 0u) { v *= vt_load(w, p.w_dt, ulong(i)); }
    if (p.has_b != 0u) { v += vt_load(b, p.b_dt, ulong(i)); }
    vt_store(out, p.out_dt, base + ulong(i), v);
  }
}

// ===========================================================================
// The Tier-1 FusedChain interpreter — the ONE registration that inherits the
// whole portable fusion catalog for this backend (include/vt/fused_recipe.h).
// Structural mirror of cpu_ops.cpp:1649-1695 FusedChainInterpKernel, including
// its canonical operand indexing (cpu_ops.cpp:1621-1643 FusedLoad/FusedStore):
//   0 = x [t,h]   1 = weight [h]   2 = residual [t,h]   3 = out [t,h]
// with 2 and 3 the only WRITABLE slots. Opcodes mirror vt::FOp; only the
// Tier-1-able subset {kAdd,kMul,kSilu,kSigmoid,kRmsNorm} can reach here (the
// device-agnostic Tier-0 composite in src/vt/ops.cpp handles everything else, so
// this backend inherits those recipes too without a second kernel).
// ===========================================================================

#define VT_FOP_ADD      0u
#define VT_FOP_MUL      1u
#define VT_FOP_SILU     2u
#define VT_FOP_SIGMOID  3u
#define VT_FOP_RMSNORM  4u

struct VtFStep {
  uint op;
  uint out;
  uint in0;
  uint in1;
  uint gemma;
  uint pad;
};

// ===========================================================================
// Dense GEMM — the NATIVE MSL provider for kMatmul and kMatmulBT.
//
// This is the DEFAULT on Metal. It exists so the MLX provider
// (src/vt/metal/metal_mlx_provider.mm) is a CONFIGURATION rather than the only
// way to get a GEMM: the provider seam's whole premise is that two providers of
// one op coexist and can be A/B'd against each other, which requires ours to be
// there. It also gives the correctness gate its middle column
// (MLX vs native MSL vs the CPU oracle).
//
// Math ported FROM src/vt/cpu/cpu_ops.cpp MatmulKernel / MatmulBTKernel: f32
// accumulation over K regardless of storage dtype, one rounding on store.
// Dispatch shape is the classic threadgroup-tiled GEMM (llama.cpp
// ggml-metal `kernel_mul_mm` uses the same 2-D tile-per-threadgroup structure
// over simdgroup_matrix; we stay on plain threadgroup memory because the W0
// build compiles MSL at RUNTIME with no offline `metal`, and a portable tile
// loop is what the CPU-reference math transcribes to directly).
//
// ORIENTATION. One kernel serves both ops via `bt`:
//   bt == 0 (kMatmul):   B is [K,N] row-major, element (kk,col) at kk*N + col
//   bt == 1 (kMatmulBT): B is [N,K] row-major, element (col,kk) at col*K + kk
// `lda` is the ACTIVATION's row stride in elements — kMatmulBT explicitly admits
// a row-strided activation (src/vt/ops.cpp MatmulBT: `a.stride[0] >= a.shape[1]`),
// so it cannot be assumed equal to K.
#define VT_GEMM_TILE 16u

struct VtGemmParams {
  uint m;
  uint n;
  uint k;
  uint lda;     // activation row stride, in ELEMENTS
  uint a_dt;
  uint b_dt;
  uint out_dt;
  uint bt;      // 0 => b is [K,N]; 1 => b is [N,K] (the torch Linear orientation)
};

kernel void vt_matmul(device const uchar* a   [[buffer(0)]],
                      device const uchar* b   [[buffer(1)]],
                      device uchar*       out [[buffer(2)]],
                      constant VtGemmParams& p [[buffer(3)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint2 lid  [[thread_position_in_threadgroup]]) {
  threadgroup float as[VT_GEMM_TILE][VT_GEMM_TILE];
  threadgroup float bs[VT_GEMM_TILE][VT_GEMM_TILE];

  const uint row = tgid.y * VT_GEMM_TILE + lid.y;
  const uint col = tgid.x * VT_GEMM_TILE + lid.x;
  const uint ntiles = (p.k + VT_GEMM_TILE - 1u) / VT_GEMM_TILE;

  float acc = 0.0f;
  for (uint t = 0u; t < ntiles; ++t) {
    const uint ka = t * VT_GEMM_TILE + lid.x;
    as[lid.y][lid.x] =
        (row < p.m && ka < p.k) ? vt_load(a, p.a_dt, ulong(row) * ulong(p.lda) + ulong(ka)) : 0.0f;

    const uint kb = t * VT_GEMM_TILE + lid.y;
    float bv = 0.0f;
    if (col < p.n && kb < p.k) {
      bv = (p.bt != 0u) ? vt_load(b, p.b_dt, ulong(col) * ulong(p.k) + ulong(kb))
                        : vt_load(b, p.b_dt, ulong(kb) * ulong(p.n) + ulong(col));
    }
    bs[lid.y][lid.x] = bv;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < VT_GEMM_TILE; ++i) { acc += as[lid.y][i] * bs[i][lid.x]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < p.m && col < p.n) {
    vt_store(out, p.out_dt, ulong(row) * ulong(p.n) + ulong(col), acc);
  }
}

// --- M3d: the DECODE GEMV fast path (kMatmulBT, m == 1) --------------------
// Shape-class profiling of a 128-token generation found 21,464 of 21,632
// matmuls are m=1 and ALL of them BT; only 168 are prefill GEMMs. The tile
// kernel above is the wrong shape for that: with m=1 only one of its
// VT_GEMM_TILE threadgroup ROWS holds a valid activation, so 15/16 of every
// threadgroup is wasted, and each thread re-walks the K loop with two
// threadgroup barriers per tile.
//
// BT is what makes the fast path worth writing: with B laid out [N,K], row
// `col` is CONTIGUOUS over k, so one simdgroup can stream it fully coalesced.
// One simdgroup owns one output column; each lane accumulates a strided slice
// of K and `simd_sum` reduces across the lanes. That is a tree reduction where
// the CPU reference is sequential, so the bar is NMSE (<= 5e-4), never
// bit-exactness, exactly as for the tile GEMM and paged attention.
//
// The `col >= p.n` early-out is SIMD-UNIFORM (col depends on the simdgroup
// index, not the lane), so every lane of a given simdgroup takes the same
// branch and `simd_sum` is never reached with a partial simdgroup. That is a
// correctness precondition of this kernel, not an incidental detail.
#define VT_GEMV_SGS 8u  // simdgroups per threadgroup => 8 output columns each

kernel void vt_matmul_bt_gemv(device const uchar* a   [[buffer(0)]],
                              device const uchar* b   [[buffer(1)]],
                              device uchar*       out [[buffer(2)]],
                              constant VtGemmParams& p [[buffer(3)]],
                              uint tgid  [[threadgroup_position_in_grid]],
                              uint sgitg [[simdgroup_index_in_threadgroup]],
                              uint tiisg [[thread_index_in_simdgroup]],
                              uint sgsize [[threads_per_simdgroup]]) {
  const uint col = tgid * VT_GEMV_SGS + sgitg;
  if (col >= p.n) return;

  // m == 1, so the activation row is row 0 and `lda` cannot contribute.
  const ulong brow = ulong(col) * ulong(p.k);
  float acc = 0.0f;

  // TYPED, HOISTED LOAD PATH. `vt_load` costs two branches per element, and this
  // loop runs K times per lane over the whole weight matrix: at 21,464 dispatches
  // it is ~50% of decode GPU time. Specialising on the dtype pair OUTSIDE the
  // loop removes the branches and lets the compiler widen the loads; the lane
  // stride is unchanged, so consecutive lanes still read consecutive addresses
  // and the access stays fully coalesced.
  //
  // bf16xbf16 is the case that matters (it is what the gate model runs); every
  // other combination keeps the generic path, so no dtype loses correctness for
  // one gaining speed.
  // VECTORISED bf16 path. The typed scalar loop below it already removed
  // vt_load's per-element branches, but each lane still issued ONE 2-byte load
  // per element. Reading a `ushort4` instead issues one 8-byte load per lane, so
  // a simdgroup fetches 256 contiguous bytes per iteration instead of 64. This
  // is the same change that took the tile GEMM's staging from 45% to ~80% of
  // MLX's kernel, applied to the op that dominates DECODE.
  //
  // Alignment: the vector cast needs the element index to be a multiple of 4.
  // `brow = col * k`, so `k % 4 == 0` carries every row start, and the loop index
  // is a multiple of 4 by construction. Anything else falls to the scalar paths.
  if (p.a_dt == VT_DT_BF16 && p.b_dt == VT_DT_BF16 && (p.k & 3u) == 0u) {
    device const ushort4* a4 = (device const ushort4*)a;
    device const ushort4* b4 = (device const ushort4*)((device const ushort*)b + brow);
    const uint n4 = p.k >> 2u;
    // Four accumulators, one per vector lane: a single `acc` would serialise the
    // FMA chain in a loop that is latency- rather than throughput-bound.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint i = tiisg; i < n4; i += sgsize) {
      const ushort4 av = a4[i];
      const ushort4 bv = b4[i];
      a0 += vt_bf16_to_f32(av.x) * vt_bf16_to_f32(bv.x);
      a1 += vt_bf16_to_f32(av.y) * vt_bf16_to_f32(bv.y);
      a2 += vt_bf16_to_f32(av.z) * vt_bf16_to_f32(bv.z);
      a3 += vt_bf16_to_f32(av.w) * vt_bf16_to_f32(bv.w);
    }
    acc = (a0 + a1) + (a2 + a3);
  } else if (p.a_dt == VT_DT_BF16 && p.b_dt == VT_DT_BF16) {
    device const ushort* au = (device const ushort*)a;
    device const ushort* bu = (device const ushort*)b + brow;
    uint kk = tiisg;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint step = sgsize * 4u;
    for (; kk + sgsize * 3u < p.k; kk += step) {
      a0 += vt_bf16_to_f32(au[kk]) * vt_bf16_to_f32(bu[kk]);
      a1 += vt_bf16_to_f32(au[kk + sgsize]) * vt_bf16_to_f32(bu[kk + sgsize]);
      a2 += vt_bf16_to_f32(au[kk + sgsize * 2u]) * vt_bf16_to_f32(bu[kk + sgsize * 2u]);
      a3 += vt_bf16_to_f32(au[kk + sgsize * 3u]) * vt_bf16_to_f32(bu[kk + sgsize * 3u]);
    }
    for (; kk < p.k; kk += sgsize) {
      a0 += vt_bf16_to_f32(au[kk]) * vt_bf16_to_f32(bu[kk]);
    }
    acc = (a0 + a1) + (a2 + a3);
  } else {
    for (uint kk = tiisg; kk < p.k; kk += sgsize) {
      acc += vt_load(a, p.a_dt, ulong(kk)) * vt_load(b, p.b_dt, brow + ulong(kk));
    }
  }
  acc = simd_sum(acc);
  if (tiisg == 0u) {
    vt_store(out, p.out_dt, ulong(col), acc);
  }
}

// --- 2-D blocked simdgroup-matrix GEMM (m > 1) -------------------------------
// Replaces the 16x16 scalar tile loop for every m > 1. Attribution put PREFILL
// at 33.7% of GPU time from only 168 dispatches and ~8.2x behind MLX-LM, and the
// small-m dead-end identified the missing property exactly: A REUSE ACROSS
// COLUMNS, which a one-column-per-simdgroup kernel cannot have and 2-D blocking
// does.
//
// SHAPE. A threadgroup owns a 32x32 output tile and holds 4 simdgroups in a 2x2
// grid, so each simdgroup owns 16x16 = a 2x2 block of simdgroup_float8x8
// accumulators. The K loop advances in steps of 8.
//
// WHY STAGE THROUGH THREADGROUP MEMORY. `simdgroup_load` needs a TYPED pointer,
// but our operands carry a runtime dtype code (f32/f16/bf16) and the output does
// too. Staging each tile into threadgroup `float` lets ONE kernel serve every
// dtype combination, which is the same reason the scalar kernels use vt_load.
// It also gives the reuse: each staged A tile is consumed by all 32 columns of
// the tile, and each staged B tile by all 32 rows.
//
// Accumulators are stored back through threadgroup memory for the same reason:
// `simdgroup_store` cannot write a runtime-dtype output buffer directly.
//
// Threadgroup memory: 32*8 + 8*32 + 32*32 floats = 6 KB, well inside the 32 KB
// limit, so occupancy is not the constraint here.
#define VT_MM_BM 64u
#define VT_MM_BN 64u
#define VT_MM_BK 16u
#define VT_MM_SGS 8u  // 2x4 simdgroups => 256 threads

kernel void vt_matmul_bt_mm(device const uchar* a   [[buffer(0)]],
                            device const uchar* b   [[buffer(1)]],
                            device uchar*       out [[buffer(2)]],
                            constant VtGemmParams& p [[buffer(3)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint  tid  [[thread_index_in_threadgroup]],
                            uint  sgitg [[simdgroup_index_in_threadgroup]]) {
  // FLAT, with the stride stated once and shared by the writes and the
  // simdgroup_load calls. A 2-D threadgroup array lets the compiler pick its own
  // row pitch for bank-conflict avoidance; `simdgroup_load(..., elements_per_row)`
  // then walks a stride the writes never used. That disagreement is invisible at
  // BM=32 and corrupts BM=64 (rows written at half spacing) — see the OPEN LEAD
  // fingerprint in .agents/specs/metal-dispatch-attribution.md.
  threadgroup float sa[VT_MM_BM * VT_MM_BK];
  threadgroup float sb[VT_MM_BK * VT_MM_BN];
  threadgroup float sc[VT_MM_BM * VT_MM_BN];

  const uint row0 = tgid.y * VT_MM_BM;
  const uint col0 = tgid.x * VT_MM_BN;

  simdgroup_float8x8 acc[4][2];
  for (uint i = 0u; i < 4u; ++i)
    for (uint j = 0u; j < 2u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const uint sg_r = sgitg / 4u;  // 0..1 : 2 row groups of 32
  const uint sg_c = sgitg % 4u;  // 0..3 : 4 column groups of 16
  const uint nthreads = VT_MM_SGS * 32u;

  for (uint k0 = 0u; k0 < p.k; k0 += VT_MM_BK) {
    // Stage A[BM x BK] and B[BK x BN]. 256 elements each, 128 threads => 2 apiece.
    // Out-of-range lanes stage ZERO so the accumulate stays correct at the edges
    // without any special-casing in the inner loop.
    // VECTORISED STAGING. Every operand element passes through here once per
    // tile, and the generic path below pays two branches per element inside
    // vt_load plus a scalar threadgroup write. MLX's steel loader instead copies
    // a whole vector per instruction
    // (steel/gemm/loader.h: *(threadgroup ReadVector*)dst = *(const device
    // ReadVector*)src), which is the single structural difference left between
    // its GEMM and ours after tile width, barriers, staging latency and mma
    // precision were each measured and excluded.
    //
    // GUARDS. The vector cast needs 8-byte alignment, so it is taken only when
    // both operands are bf16 AND the indices are multiples of 4: `kk` steps by 4
    // within a BK=8 tile, k0 by BK, and `p.k % 4 == 0` / `p.lda % 4 == 0` carry
    // the row starts. Anything else keeps the generic path, so no shape loses
    // correctness for another gaining speed. The interior tile rows are also
    // fully in range whenever `row0 + BM <= m` and `col0 + BN <= n`, which is
    // what lets the fast path skip the per-element bounds test entirely.
    const bool vec_ok = p.a_dt == VT_DT_BF16 && p.b_dt == VT_DT_BF16 &&
                        (p.k & 3u) == 0u && (p.lda & 3u) == 0u && p.bt != 0u &&
                        (row0 + VT_MM_BM) <= p.m && (col0 + VT_MM_BN) <= p.n &&
                        (k0 + VT_MM_BK) <= p.k;
    if (vec_ok) {
      device const ushort* au = (device const ushort*)a;
      device const ushort* bu = (device const ushort*)b;
      for (uint e = tid * 4u; e < VT_MM_BM * VT_MM_BK; e += nthreads * 4u) {
        const uint r = e / VT_MM_BK, kk = e % VT_MM_BK;
        const ulong idx = ulong(row0 + r) * ulong(p.lda) + ulong(k0 + kk);
        const ushort4 v = *((device const ushort4*)(au + idx));
        threadgroup float* d = &sa[r * VT_MM_BK + kk];
        d[0] = vt_bf16_to_f32(v.x); d[1] = vt_bf16_to_f32(v.y);
        d[2] = vt_bf16_to_f32(v.z); d[3] = vt_bf16_to_f32(v.w);
      }
      // B tile, vectorised ALONG K. BT means row `gc` is contiguous over k, so a
      // vector load runs along k for one column and scatters four results down
      // the tile's k dimension. At BK=16 this is (BK/4)*BN = 256 loads over 256
      // threads, exactly one each; at BK=8 it was 128 and half the threadgroup
      // idled, which is what made this a shape-dependent wash before.
      for (uint e = tid; e < (VT_MM_BK / 4u) * VT_MM_BN; e += nthreads) {
        const uint kq = e / VT_MM_BN, c = e % VT_MM_BN;
        const uint kk = kq * 4u;
        const ulong idx = ulong(col0 + c) * ulong(p.k) + ulong(k0 + kk);
        const ushort4 v = *((device const ushort4*)(bu + idx));
        sb[(kk + 0u) * VT_MM_BN + c] = vt_bf16_to_f32(v.x);
        sb[(kk + 1u) * VT_MM_BN + c] = vt_bf16_to_f32(v.y);
        sb[(kk + 2u) * VT_MM_BN + c] = vt_bf16_to_f32(v.z);
        sb[(kk + 3u) * VT_MM_BN + c] = vt_bf16_to_f32(v.w);
      }
    } else {
    for (uint e = tid; e < VT_MM_BM * VT_MM_BK; e += nthreads) {
      const uint r = e / VT_MM_BK, kk = e % VT_MM_BK;
      const uint gr = row0 + r, gk = k0 + kk;
      sa[r * VT_MM_BK + kk] = (gr < p.m && gk < p.k)
                      ? vt_load(a, p.a_dt, ulong(gr) * ulong(p.lda) + ulong(gk))
                      : 0.0f;
    }
    for (uint e = tid; e < VT_MM_BK * VT_MM_BN; e += nthreads) {
      const uint kk = e / VT_MM_BN, c = e % VT_MM_BN;
      const uint gc = col0 + c, gk = k0 + kk;
      float v = 0.0f;
      if (gc < p.n && gk < p.k) {
        v = (p.bt != 0u) ? vt_load(b, p.b_dt, ulong(gc) * ulong(p.k) + ulong(gk))
                         : vt_load(b, p.b_dt, ulong(gk) * ulong(p.n) + ulong(gc));
      }
      sb[kk * VT_MM_BN + c] = v;
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4 row blocks x 2 column blocks per simdgroup, over the FLAT tiles: the
    // stride is VT_MM_BK / VT_MM_BN in both the address arithmetic and the
    // elements_per_row argument, so the two cannot disagree.
    // BK=16 is consumed as two 8-deep fragments; the simdgroup matrices are 8x8.
    for (uint kk = 0u; kk < VT_MM_BK; kk += 8u) {
      simdgroup_float8x8 ma[4], mb[2];
      for (uint i = 0u; i < 4u; ++i) {
        simdgroup_load(ma[i], &sa[(sg_r * 32u + i * 8u) * VT_MM_BK + kk], VT_MM_BK);
      }
      for (uint j = 0u; j < 2u; ++j) {
        simdgroup_load(mb[j], &sb[kk * VT_MM_BN + sg_c * 16u + j * 8u], VT_MM_BN);
      }
      for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 2u; ++j)
          simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Land the accumulators in threadgroup memory, then write out with vt_store so
  // the runtime output dtype is honoured.
  for (uint i = 0u; i < 4u; ++i)
    for (uint j = 0u; j < 2u; ++j)
      simdgroup_store(acc[i][j],
                      &sc[(sg_r * 32u + i * 8u) * VT_MM_BN + sg_c * 16u + j * 8u],
                      VT_MM_BN);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint e = tid; e < VT_MM_BM * VT_MM_BN; e += nthreads) {
    const uint r = e / VT_MM_BN, c = e % VT_MM_BN;
    const uint gr = row0 + r, gc = col0 + c;
    if (gr < p.m && gc < p.n) {
      vt_store(out, p.out_dt, ulong(gr) * ulong(p.n) + ulong(gc), sc[r * VT_MM_BN + c]);
    }
  }
}

struct VtFcParams {
  uint  t;
  uint  h;
  uint  nsteps;
  uint  x_dt;
  uint  w_dt;
  uint  res_dt;
  uint  out_dt;
  uint  tg;
  float eps;
};

inline float vt_fc_load(uint idx, ulong rbase, uint j,
                        device const uchar* x, uint x_dt,
                        device const uchar* w, uint w_dt,
                        device const uchar* res, uint res_dt,
                        device const uchar* out, uint out_dt) {
  if (idx == 0u) { return vt_load(x, x_dt, rbase + ulong(j)); }
  if (idx == 1u) { return vt_load(w, w_dt, ulong(j)); }
  if (idx == 2u) { return vt_load(res, res_dt, rbase + ulong(j)); }
  return vt_load(out, out_dt, rbase + ulong(j));
}

inline void vt_fc_store(uint idx, ulong rbase, uint j, float v,
                        device uchar* res, uint res_dt,
                        device uchar* out, uint out_dt) {
  if (idx == 2u) { vt_store(res, res_dt, rbase + ulong(j), v); return; }
  vt_store(out, out_dt, rbase + ulong(j), v);
}

kernel void vt_fused_chain(device const uchar*  x     [[buffer(0)]],
                           device const uchar*  w     [[buffer(1)]],
                           device uchar*        res   [[buffer(2)]],
                           device uchar*        out   [[buffer(3)]],
                           device const VtFStep* steps [[buffer(4)]],
                           constant VtFcParams& p     [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  ulong rbase = ulong(row) * ulong(p.h);

  for (uint s = 0u; s < p.nsteps; ++s) {
    VtFStep st = steps[s];
    if (st.op == VT_FOP_ADD || st.op == VT_FOP_MUL) {
      for (uint j = tid; j < p.h; j += p.tg) {
        float a = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        float b = vt_fc_load(st.in1, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        vt_fc_store(st.out, rbase, j, st.op == VT_FOP_ADD ? a + b : a * b,
                    res, p.res_dt, out, p.out_dt);
      }
    } else if (st.op == VT_FOP_SILU || st.op == VT_FOP_SIGMOID) {
      for (uint j = tid; j < p.h; j += p.tg) {
        float a = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        float sg = vt_sigmoid(a);
        vt_fc_store(st.out, rbase, j, st.op == VT_FOP_SILU ? a * sg : sg,
                    res, p.res_dt, out, p.out_dt);
      }
    } else {  // VT_FOP_RMSNORM (the host validated reduce == kMeanSquare)
      float partial = 0.0f;
      for (uint j = tid; j < p.h; j += p.tg) {
        float v = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        partial += v * v;   // f32 variance accumulation
      }
      float inv = 1.0f / sqrt(vt_tg_sum(smem, tid, p.tg, partial) / float(p.h) + p.eps);
      for (uint j = tid; j < p.h; j += p.tg) {
        float v = vt_fc_load(st.in0, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        float wj = vt_fc_load(st.in1, rbase, j, x, p.x_dt, w, p.w_dt, res, p.res_dt, out, p.out_dt);
        if (st.gemma != 0u) { wj += 1.0f; }
        vt_fc_store(st.out, rbase, j, v * inv * wj, res, p.res_dt, out, p.out_dt);
      }
    }
    // A later step may read what this step wrote (the interpreter is a chain),
    // and a step's own two phases race across lanes, so every step boundary is a
    // device-memory barrier within the row's threadgroup.
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

// ===========================================================================
// M3a — the five kernels OPT-125m (`OPTForCausalLM`) needs beyond the W0 set.
// Math ported 1:1 from our own CPU reference kernels, which are the vLLM-parity
// goldens:
//   src/vt/cpu/cpu_ops.cpp:531-543        EmbeddingKernel       -> vt_embedding
//   src/vt/cpu/cpu_ops.cpp:1529-1543      QkvSplitKernel        -> vt_qkv_split
//   src/vt/cpu/cpu_cache.cpp:33-72        ReshapeAndCacheKernel -> vt_reshape_and_cache
//   src/vt/cpu/cpu_paged_attn.cpp:51-131  PagedAttentionKernel  -> vt_paged_attention
//   src/vt/cpu/cpu_sample.cpp:40-57       GreedyArgmaxKernel    -> vt_greedy_argmax
// ===========================================================================

struct VtEmbedParams {
  uint rows;     // T
  uint h;        // embedding width
  uint vocab;    // table rows, for the bounds check
  uint id_i64;   // 1 => ids are i64, 0 => i32
  uint tab_dt;
  uint out_dt;
};

// cpu_ops.cpp:531-543 EmbeddingKernel — a pure row gather through LoadF32/
// StoreF32. One thread per output ELEMENT. An out-of-range id cannot throw from
// a shader, so the row is written as zeros and the host-side wrapper's own
// validation remains the place the error surfaces (the CPU reference VT_CHECKs;
// vt::Embedding in src/vt/ops.cpp validates shapes, and OPT's ids come from the
// tokenizer, so an out-of-range id is a corrupted-checkpoint case, not a
// reachable input).
kernel void vt_embedding(device const uchar* table [[buffer(0)]],
                         device const uchar* ids   [[buffer(1)]],
                         device uchar*       out   [[buffer(2)]],
                         constant VtEmbedParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows * p.h) { return; }
  const uint i = gid / p.h;
  const uint j = gid % p.h;
  long id = p.id_i64 != 0u ? ((device const long*)ids)[i] : long(((device const int*)ids)[i]);
  if (id < 0 || id >= long(p.vocab)) {
    vt_store(out, p.out_dt, ulong(gid), 0.0f);
    return;
  }
  vt_store(out, p.out_dt, ulong(gid), vt_load(table, p.tab_dt, ulong(id) * ulong(p.h) + ulong(j)));
}

struct VtQkvSplitParams {
  uint t;
  uint q_dim;
  uint k_dim;
  uint v_dim;
  uint in_dt;
  uint q_dt;
  uint k_dt;
  uint v_dt;
};

// cpu_ops.cpp:1529-1543 QkvSplitKernel — the merged [T, q+k+v] row cut into three
// DENSE per-shard buffers. One thread per merged element; the column decides
// which output it belongs to.
kernel void vt_qkv_split(device const uchar* qkv [[buffer(0)]],
                         device uchar*       q   [[buffer(1)]],
                         device uchar*       k   [[buffer(2)]],
                         device uchar*       v   [[buffer(3)]],
                         constant VtQkvSplitParams& p [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint total = p.q_dim + p.k_dim + p.v_dim;
  if (gid >= p.t * total) { return; }
  const uint i = gid / total;
  const uint c = gid % total;
  const float val = vt_load(qkv, p.in_dt, ulong(gid));
  if (c < p.q_dim) {
    vt_store(q, p.q_dt, ulong(i) * ulong(p.q_dim) + ulong(c), val);
  } else if (c < p.q_dim + p.k_dim) {
    vt_store(k, p.k_dt, ulong(i) * ulong(p.k_dim) + ulong(c - p.q_dim), val);
  } else {
    vt_store(v, p.v_dt, ulong(i) * ulong(p.v_dim) + ulong(c - p.q_dim - p.k_dim), val);
  }
}

// 8-byte members FIRST, then an even count of 4-byte members: neither MSL nor the
// host struct in metal_ops.mm can insert interior padding, so the two layouts
// coincide by construction (metal_ops.mm static_asserts the offsets).
struct VtCacheParams {
  ulong k_blk_stride; // all strides in ELEMENTS, from the tensors themselves
  ulong k_pg_stride;
  ulong v_blk_stride;
  ulong v_pg_stride;
  ulong k_tok_stride;
  ulong v_tok_stride;
  uint  num_slots;
  uint  n_elems;      // num_kv_heads * head_size — one token's NHD page
  uint  block_size;
  uint  esz;          // element size in BYTES (2 = bf16/f16, 4 = f32)
};

// cpu_cache.cpp:33-72 ReshapeAndCacheKernel. Upstream (and our CPU reference) is
// a `memcpy` per token, so this copies RAW ELEMENTS rather than routing through
// LoadF32/StoreF32 — the cache write is BIT-EXACT on every backend, which is what
// the gate claims for pure copy/layout ops. `slot < 0` is the padded-token skip
// the upstream kernel documents.
kernel void vt_reshape_and_cache(device const uchar* k    [[buffer(0)]],
                                 device const uchar* v    [[buffer(1)]],
                                 device uchar*       kc   [[buffer(2)]],
                                 device uchar*       vc   [[buffer(3)]],
                                 device const long*  slots [[buffer(4)]],
                                 constant VtCacheParams& p [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num_slots * p.n_elems) { return; }
  const uint t = gid / p.n_elems;
  const uint e = gid % p.n_elems;
  const long slot = slots[t];
  if (slot < 0) { return; }
  const ulong block = ulong(slot) / ulong(p.block_size);
  const ulong off = ulong(slot) % ulong(p.block_size);
  const ulong kd = block * p.k_blk_stride + off * p.k_pg_stride + ulong(e);
  const ulong vd = block * p.v_blk_stride + off * p.v_pg_stride + ulong(e);
  const ulong ks = ulong(t) * p.k_tok_stride + ulong(e);
  const ulong vs = ulong(t) * p.v_tok_stride + ulong(e);
  if (p.esz == 4u) {
    ((device uint*)kc)[kd] = ((device const uint*)k)[ks];
    ((device uint*)vc)[vd] = ((device const uint*)v)[vs];
  } else {
    ((device ushort*)kc)[kd] = ((device const ushort*)k)[ks];
    ((device ushort*)vc)[vd] = ((device const ushort*)v)[vs];
  }
}

// Paged causal/full GQA attention over the NHD paged cache. Tile widths: the key
// CHUNK bounds the threadgroup score buffer, VT_PA_MAXD bounds the accumulator.
// Both are asserted host-side, so an unsupported head_size is a loud vt:: error
// rather than a silent wrong answer.
#define VT_PA_CHUNK 256u
#define VT_PA_MAXD  256u

// Same layout discipline as VtCacheParams: 8-byte members first, then 4-byte.
struct VtPagedAttnParams {
  ulong kc_blk; ulong kc_pg; ulong kc_hd;
  ulong vc_blk; ulong vc_pg; ulong vc_hd;
  uint  num_reqs;
  uint  hq;
  uint  d;
  uint  qpk;         // q-heads per kv-head (the GQA ratio)
  uint  block_size;
  uint  causal;
  uint  tg;
  int   window_left;
  int   window_right;
  int   bt_row;
  int   bt_col;
  uint  q_dt; uint kc_dt; uint vc_dt; uint out_dt;
  float scale;
};

// cpu_paged_attn.cpp:51-131 PagedAttentionKernel. The CPU reference is a THREE-
// PASS max-subtracted softmax that materializes the whole score row; on a GPU
// that row can be arbitrarily long, so this is the algebraically identical
// ONLINE (flash) form: keys are consumed in chunks of VT_PA_CHUNK with a running
// (max, denominator) pair and a rescaled accumulator. Same f32 accumulation, same
// single rounding on store. The reduction ORDER differs from the CPU reference by
// construction — per the spike § Gates the bar for reducing ops is NMSE <= 5e-4,
// not bit-exactness, and no bit-exactness is claimed here.
//
// One THREADGROUP per (query token, q-head). The owning request is found by
// scanning query_start_loc in-kernel rather than uploading a per-token request
// index: num_reqs is tiny, the scan is uniform across the threadgroup, and it
// keeps the op's device-side inputs exactly the ones the vt:: signature already
// passes.
kernel void vt_paged_attention(device const uchar* q     [[buffer(0)]],
                               device const uchar* kc    [[buffer(1)]],
                               device const uchar* vc    [[buffer(2)]],
                               device const int*   btab  [[buffer(3)]],
                               device const int*   slens [[buffer(4)]],
                               device const int*   qsl   [[buffer(5)]],
                               device uchar*       out   [[buffer(6)]],
                               constant VtPagedAttnParams& p [[buffer(7)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               // MSL requires every position attribute in one
                               // signature to have the SAME dimensionality, so
                               // the thread index is uint2 with an unused .y even
                               // though the threadgroup is 1-D.
                               uint2 tid2 [[thread_position_in_threadgroup]],
                               uint tiisg [[thread_index_in_simdgroup]],
                               uint sgitg [[simdgroup_index_in_threadgroup]],
                               uint sgsize [[threads_per_simdgroup]]) {
  const uint tid = tid2.x;
  threadgroup float smem[VT_TG_MAX];
  threadgroup float scores[VT_PA_CHUNK];
  // Per-key V base addresses, computed ONCE per chunk. Without this every thread
  // re-reads `btab` from DEVICE memory and recomputes the same address for every
  // key it accumulates, i.e. jc * tg block-table loads per chunk where jc would
  // do. The block table is small but it is device memory in the inner loop.
  threadgroup ulong vbases[VT_PA_CHUNK];
  threadgroup float acc[VT_PA_MAXD];
  threadgroup float sh_m;
  threadgroup float sh_l;

  const uint h = tgid.x;
  const uint t = tgid.y;

  // Locate the request owning global query index `t`: qsl[r] <= t < qsl[r+1].
  // UNIFORM across the threadgroup, so the early-outs below never split a
  // barrier.
  int r = -1;
  for (uint i = 0u; i < p.num_reqs; ++i) {
    if (int(t) >= qsl[i] && int(t) < qsl[i + 1u]) { r = int(i); break; }
  }
  if (r < 0) { return; }
  const int q0 = qsl[r];
  const int query_len = qsl[r + 1] - q0;
  const int seqlen = slens[r];
  const int context = seqlen - query_len;   // past positions before this chunk
  const int pos = context + (int(t) - q0);  // absolute position of this token

  int jmin = p.window_left >= 0 ? max(0, pos - p.window_left) : 0;
  int jmax = p.causal != 0u ? pos : seqlen - 1;
  if (p.window_right >= 0) { jmax = min(jmax, pos + p.window_right); }
  jmax = min(jmax, seqlen - 1);
  if (jmax < jmin) { return; }  // cpu_paged_attn.cpp `continue`: out left untouched

  const uint g = h / p.qpk;
  const ulong qoff = (ulong(t) * ulong(p.hq) + ulong(h)) * ulong(p.d);

  for (uint e = tid; e < p.d; e += p.tg) { acc[e] = 0.0f; }
  if (tid == 0u) { sh_m = -INFINITY; sh_l = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j0 = jmin; j0 <= jmax; j0 += int(VT_PA_CHUNK)) {
    const int jc = min(int(VT_PA_CHUNK), jmax - j0 + 1);

    // Scores for this chunk (scaled), ONE SIMDGROUP PER KEY.
    //
    // The original form gave one THREAD the whole d-length dot product. Adjacent
    // threads then read entirely different K rows (kbase varies per key), so the
    // loads never coalesced and this kernel ran at roughly 4% of the M4's memory
    // bandwidth while holding 24% of GPU time. Splitting `d` across a
    // simdgroup's lanes makes neighbouring lanes read neighbouring elements of
    // the SAME K row, which is the coalesced direction, and lets each lane take a
    // 4-wide vector load when the dtype and alignment allow.
    //
    // `simd_sum` requires every lane to participate, so the loop bound is the
    // SIMDGROUP index and the tail is handled by having idle simdgroups still
    // execute the reduction; `jc` is uniform across the threadgroup, so no lane
    // diverges on the loop itself.
    const uint n_sg = p.tg / sgsize;
    for (int idx = int(sgitg); idx < jc; idx += int(n_sg)) {
      const int j = j0 + idx;
      const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
      const int off = j % int(p.block_size);
      const ulong kbase = ulong(blk) * p.kc_blk + ulong(off) * p.kc_pg + ulong(g) * p.kc_hd;
      float part = 0.0f;
      if (p.q_dt == VT_DT_BF16 && p.kc_dt == VT_DT_BF16 && (p.d & 3u) == 0u &&
          ((qoff | kbase) & 3ul) == 0ul) {
        device const ushort4* q4 = (device const ushort4*)((device const ushort*)q + qoff);
        device const ushort4* k4 = (device const ushort4*)((device const ushort*)kc + kbase);
        const uint d4 = p.d >> 2u;
        for (uint e = tiisg; e < d4; e += sgsize) {
          const ushort4 qv = q4[e];
          const ushort4 kv = k4[e];
          part += vt_bf16_to_f32(qv.x) * vt_bf16_to_f32(kv.x);
          part += vt_bf16_to_f32(qv.y) * vt_bf16_to_f32(kv.y);
          part += vt_bf16_to_f32(qv.z) * vt_bf16_to_f32(kv.z);
          part += vt_bf16_to_f32(qv.w) * vt_bf16_to_f32(kv.w);
        }
      } else {
        for (uint e = tiisg; e < p.d; e += sgsize) {
          part += vt_load(q, p.q_dt, qoff + ulong(e)) * vt_load(kc, p.kc_dt, kbase + ulong(e));
        }
      }
      const float dot = simd_sum(part);
      if (tiisg == 0u) { scores[idx] = dot * p.scale; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Running max, then rescale the accumulator and the denominator.
    float pm = -INFINITY;
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) { pm = max(pm, scores[idx]); }
    const float mchunk = vt_tg_max(smem, tid, p.tg, pm);
    const float mold = sh_m;
    const float mnew = max(mold, mchunk);
    const float corr = mold == -INFINITY ? 0.0f : exp(mold - mnew);
    for (uint e = tid; e < p.d; e += p.tg) { acc[e] *= corr; }

    for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
      scores[idx] = exp(scores[idx] - mnew);
    }
    float ps = 0.0f;
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) { ps += scores[idx]; }
    // vt_tg_sum's LEADING barrier is what makes every thread's exp() above
    // visible to the V accumulation below, which reads the WHOLE chunk.
    const float lchunk = vt_tg_sum(smem, tid, p.tg, ps);

    // Resolve each key's V base address ONCE for the whole threadgroup.
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
      const int j = j0 + idx;
      const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
      const int off = j % int(p.block_size);
      vbases[idx] = ulong(blk) * p.vc_blk + ulong(off) * p.vc_pg + ulong(g) * p.vc_hd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted V accumulation, parallel over the head dimension so each thread
    // owns its own acc[] slots. Adjacent threads read adjacent elements of the
    // same V row, which is already the coalesced direction; the win here is
    // dropping the redundant block-table traffic above.
    for (uint e = tid; e < p.d; e += p.tg) {
      float s = 0.0f;
      for (int idx = 0; idx < jc; ++idx) {
        s += scores[idx] * vt_load(vc, p.vc_dt, vbases[idx] + ulong(e));
      }
      acc[e] += s;
    }
    if (tid == 0u) { sh_l = sh_l * corr + lchunk; sh_m = mnew; }
    // Orders this chunk's acc/scores reads before the next chunk overwrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / sh_l;  // every valid window has >= 1 key
  for (uint e = tid; e < p.d; e += p.tg) {
    vt_store(out, p.out_dt, qoff + ulong(e), acc[e] * inv);
  }
}

struct VtArgmaxParams {
  uint n;
  uint v;
  uint tg;
};

// cpu_sample.cpp:40-57 GreedyArgmaxKernel. Upstream `greedy_sample` is
// argmax(dim=-1) and our CPU reference uses a STRICT `>` so the FIRST (lowest-
// index) maximum wins, bit-exact vs torch.argmax. A tree reduction must
// reproduce that tie rule explicitly, so the combine is "greater value, or equal
// value at a lower index" — which makes the result independent of the reduction
// order and therefore BIT-EXACT vs the CPU reference, not merely close.
kernel void vt_greedy_argmax(device const float* logits [[buffer(0)]],
                             device long*        ids    [[buffer(1)]],
                             constant VtArgmaxParams& p [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float bv[VT_TG_MAX];
  threadgroup uint  bi[VT_TG_MAX];
  const ulong base = ulong(row) * ulong(p.v);

  float best_v = -INFINITY;
  uint best_i = 0xFFFFFFFFu;
  for (uint j = tid; j < p.v; j += p.tg) {
    const float x = logits[base + ulong(j)];
    if (x > best_v || (x == best_v && j < best_i)) { best_v = x; best_i = j; }
  }
  bv[tid] = best_v;
  bi[tid] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = p.tg / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      const float ov = bv[tid + s];
      const uint oi = bi[tid + s];
      if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) { ids[row] = long(bi[0] == 0xFFFFFFFFu ? 0u : bi[0]); }
}

// ===========================================================================
// RoPE (NeoX) — the Qwen3-dense default (deterministic) rotation, work row M3b.
// ===========================================================================
// cpu_ops.cpp:636-665 RopeRotateHead / RopeNeoxKernel. In-place rotation of the
// leading `rot` dims of every (token, head) row of q [T,Hq,D] and k [T,Hk,D],
// NeoX split (pair `i` with `i+half`). One thread per (row, pair), where a row is
// one (token, head) in q (rows [0, T*Hq)) then one in k (rows [T*Hq, T*Hq+T*Hk)).
// Non-reducing and per-element, so no thread aliases another's pair.
//
// NUMERICS. The CPU/CUDA reference computes freq/angle and cos/sin in DOUBLE and
// casts to f32; Metal has no double, so this uses f32 `pow`/`precise::cos`/
// `precise::sin`. The rotation arithmetic (x*c - y*s / x*s + y*c) is otherwise the
// identical f32 form with a single bf16 rounding on store. Per the spike § Gates
// the bar is therefore NMSE <= 5e-4 vs the CPU oracle, NOT bit-exactness — the
// same posture as vt_paged_attention. For the short-prompt greedy gate the angles
// stay small enough that the bf16-rounded result matches the reference at all but
// bf16 near-ties (measured on the M4).
struct VtRopeParams {
  uint  t;        // tokens
  uint  hq;       // query heads
  uint  hk;       // key heads
  uint  d;        // head_dim
  uint  rot;      // rotary_dim (even, <= d)
  uint  rhalf;    // rot / 2
  uint  q_dt;
  uint  k_dt;
  uint  pos_i64;  // 1 => positions are i64, else i32
  float base;
};

kernel void vt_rope_neox(device uchar* q             [[buffer(0)]],
                         device uchar* k             [[buffer(1)]],
                         device const uchar* positions [[buffer(2)]],
                         constant VtRopeParams& p    [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint pair = gid % p.rhalf;
  const uint row  = gid / p.rhalf;
  const uint qrows = p.t * p.hq;
  device uchar* buf;
  uint dt;
  uint token;
  ulong head_off;
  if (row < qrows) {
    buf = q; dt = p.q_dt; token = row / p.hq; head_off = ulong(row) * p.d;
  } else {
    const uint r2 = row - qrows;
    buf = k; dt = p.k_dt; token = r2 / p.hk; head_off = ulong(r2) * p.d;
  }
  const long pos = p.pos_i64
      ? ((device const long*)positions)[token]
      : (long)((device const int*)positions)[token];
  const float freq  = pow(p.base, -2.0f * float(pair) / float(p.rot));
  const float angle = float(pos) * freq;
  const float c = precise::cos(angle);
  const float s = precise::sin(angle);
  const float x = vt_load(buf, dt, head_off + pair);
  const float y = vt_load(buf, dt, head_off + pair + p.rhalf);
  vt_store(buf, dt, head_off + pair,          x * c - y * s);
  vt_store(buf, dt, head_off + pair + p.rhalf, x * s + y * c);
}

// cpu_ops.cpp:751-768 RopeCosSinCacheKernel. Fill cos_sin[T, rot]: cols [0,half)
// = cos, [half,rot) = sin. One thread per (token, pair). Same f32-transcendental
// deviation and NMSE bar as vt_rope_neox above. (Built once per step by the dense
// attention preamble; consumed only by the opt-in RopeFromCache path, so on the
// deterministic default path its output is unused — but the op must exist or the
// engine's GetOp throws.)
struct VtRopeCacheParams {
  uint  t;
  uint  rot;
  uint  rhalf;
  uint  out_dt;
  uint  pos_i64;
  float base;
};

kernel void vt_rope_cos_sin_cache(device uchar* cos_sin          [[buffer(0)]],
                                  device const uchar* positions  [[buffer(1)]],
                                  constant VtRopeCacheParams& p  [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  const uint pair  = gid % p.rhalf;
  const uint token = gid / p.rhalf;
  const long pos = p.pos_i64
      ? ((device const long*)positions)[token]
      : (long)((device const int*)positions)[token];
  const float freq  = pow(p.base, -2.0f * float(pair) / float(p.rot));
  const float angle = float(pos) * freq;
  vt_store(cos_sin, p.out_dt, ulong(token) * p.rot + pair,           precise::cos(angle));
  vt_store(cos_sin, p.out_dt, ulong(token) * p.rot + p.rhalf + pair, precise::sin(angle));
}

// ===========================================================================
// RoPE from a precomputed cos|sin cache — the Qwen3-dense DEFAULT rotation
// (VT_QWEN3_ROPE_CACHE defaults ON), work row M3b.
// ===========================================================================
// cpu_ops.cpp:690-742 RopeFromCacheKernel. Rotate the leading `rot` dims of every
// (token, head) row of q [T,Hq,D] and k [T,Hk,D] using cos|sin READ from a
// cos_sin[P, rot] cache (cols [0,half)=cos, [half,rot)=sin), indexed by
// positions[token]. STRIDE-DRIVEN (q/k need only a unit-stride innermost dim).
// One thread per (row, pair). No transcendentals in the kernel — the c/s are the
// SAME cached values the CPU reference reads, and the rotation (x*c - y*s /
// x*s + y*c) is the identical f32 arithmetic with a single bf16 rounding on store,
// so this is bit-exact to the CPU oracle (a non-reducing per-element op). This is
// the op the correctness gate actually exercises; kRopeNeox above serves the
// VT_QWEN3_ROPE_CACHE=0 opt-out. Rank-1 positions only (MRoPE is guarded off in
// the host wrapper — Qwen3-dense never uses it).
struct VtRopeApplyParams {
  ulong q_s0;    // q element stride over tokens
  ulong q_s1;    // q element stride over heads
  ulong k_s0;
  ulong k_s1;
  uint  t;
  uint  hq;
  uint  hk;
  uint  rot;
  uint  rhalf;
  uint  is_neox;
  uint  q_dt;
  uint  k_dt;
  uint  cache_dt;
  uint  pos_i64;
  uint  has_k;
  uint  pad;
};

kernel void vt_rope_from_cache(device uchar* q               [[buffer(0)]],
                               device uchar* k               [[buffer(1)]],
                               device const uchar* positions [[buffer(2)]],
                               device const uchar* cache     [[buffer(3)]],
                               constant VtRopeApplyParams& p [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
  const uint pair = gid % p.rhalf;
  const uint row  = gid / p.rhalf;
  const uint qrows = p.t * p.hq;
  device uchar* buf;
  uint dt;
  ulong s0, s1;
  uint token, head;
  if (row < qrows) {
    buf = q; dt = p.q_dt; s0 = p.q_s0; s1 = p.q_s1; token = row / p.hq; head = row % p.hq;
  } else {
    if (p.has_k == 0u) return;
    const uint r2 = row - qrows;
    buf = k; dt = p.k_dt; s0 = p.k_s0; s1 = p.k_s1; token = r2 / p.hk; head = r2 % p.hk;
  }
  const long pos = p.pos_i64
      ? ((device const long*)positions)[token]
      : (long)((device const int*)positions)[token];
  const ulong coff = ulong(pos) * p.rot;
  const float c = vt_load(cache, p.cache_dt, coff + pair);
  const float s = vt_load(cache, p.cache_dt, coff + p.rhalf + pair);
  const uint first  = p.is_neox ? pair : pair * 2u;
  const uint second = p.is_neox ? pair + p.rhalf : pair * 2u + 1u;
  const ulong off = ulong(token) * s0 + ulong(head) * s1;
  const float x = vt_load(buf, dt, off + first);
  const float y = vt_load(buf, dt, off + second);
  vt_store(buf, dt, off + first,  x * c - y * s);
  vt_store(buf, dt, off + second, x * s + y * c);
}
)MSL";
// clang-format on

}  // namespace vt::metal

#endif  // VT_METAL_METAL_MSL_H_
