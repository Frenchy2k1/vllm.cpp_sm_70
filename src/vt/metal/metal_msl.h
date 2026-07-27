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
  // Epilogue scratch, ONE 8x8 fragment PER SIMDGROUP rather than the whole
  // BM x BN tile. A simdgroup only ever reads back the fragment it just stored,
  // so nothing crosses simdgroups and a simdgroup_barrier suffices where the
  // full tile needed a threadgroup barrier. At 64x64 that is 2 KB instead of
  // 16 KB, taking the kernel from 24 KB of threadgroup memory to 10 KB.
  //
  // It buys less than the footprint suggests (GEMM 653 -> 642 ms, ~1.7%), and
  // that is the finding: this kernel is NOT occupancy-limited, so freeing 14 KB
  // barely moves it. Kept because it is faster, simpler and drops a barrier.
  threadgroup float sc[VT_MM_SGS * 64u];

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

  // Land each accumulator fragment in this SIMDGROUP's own 8x8 scratch and write
  // it out with vt_store, so the runtime output dtype is honoured. Private per
  // simdgroup means no threadgroup barrier here: simdgroup_barrier orders the
  // store against the read-back, and the next fragment against the read.
  threadgroup float* mysc = &sc[sgitg * 64u];
  const uint lane = tid % 32u;
  for (uint i = 0u; i < 4u; ++i) {
    for (uint j = 0u; j < 2u; ++j) {
      simdgroup_store(acc[i][j], mysc, 8u);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint br = row0 + sg_r * 32u + i * 8u;
      const uint bc = col0 + sg_c * 16u + j * 8u;
      for (uint e = lane; e < 64u; e += 32u) {
        const uint gr = br + e / 8u, gc = bc + e % 8u;
        if (gr < p.m && gc < p.n) {
          vt_store(out, p.out_dt, ulong(gr) * ulong(p.n) + ulong(gc), mysc[e]);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
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
// How many key-groups the V accumulation may split into. The threadgroup is
// sized at 2*d, so the d-parallel V loop left half the threads idle; splitting
// the chunk's KEYS across `tg/d` groups puts them all to work. Capped so the
// partial-accumulator tile stays small (4 * 256 floats = 4 KB).
#define VT_PA_VGROUPS 4u
// Query tokens per threadgroup. With one threadgroup per (head, query token),
// PREFILL re-reads the entire K/V range once per query: ~57 GB for a 512-token
// prompt on this model, which is why prefill attention measured ~175 GFLOP/s
// against MLX's ~3000. Serving QTILE queries from one threadgroup reuses each
// K/V element QTILE times, and the working set per chunk is small enough to stay
// in cache. Decode has one query, so QTILE-1 slots simply idle there.
#define VT_PA_QTILE 4u
// ushort4 registers per lane for the K row held across the tile's queries.
// d <= VT_PA_MAXD (256) gives d4 <= 64, i.e. 2 per lane at simdgroup width 32.
#define VT_PA_KREG 8u

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
// The body is shared by two SPECIALISED pipelines, `QT` being the number of
// query tokens one threadgroup serves. It is a template parameter and not a
// runtime value because both properties that make each pipeline fast are decided
// at compile time: the per-query loops must unroll to keep qjmin/qjmax/qok/s in
// REGISTERS (with a runtime bound MSL spills them to thread-private memory, and
// `s[u]` sits in the innermost V loop — measured at +42ms on prefill attention),
// and the threadgroup arrays must be sized for the tile so DECODE keeps its
// original, smaller allocation. One kernel serving both loses either way: a
// runtime bound costs prefill the spills, a fixed bound of 4 makes decode
// compute and mask three dead slots.
template <uint QT>
inline void vt_paged_attention_impl(device const uchar* q,
                                    device const uchar* kc,
                                    device const uchar* vc,
                                    device const int*   btab,
                                    device const int*   slens,
                                    device const int*   qsl,
                                    device uchar*       out,
                                    constant VtPagedAttnParams& p,
                                    // Sized REFERENCES, not pointers: bare
                                    // threadgroup pointers cost decode ~17% here
                                    // because the compiler can no longer prove
                                    // they do not alias and reloads across every
                                    // barrier.
                                    threadgroup float  (&smem)[VT_TG_MAX],
                                    threadgroup float  (&scores)[QT * VT_PA_CHUNK],
                                    threadgroup ulong  (&vbases)[VT_PA_CHUNK],
                                    threadgroup float  (&pacc)[VT_PA_VGROUPS * VT_PA_MAXD],
                                    threadgroup float  (&acc)[QT * VT_PA_MAXD],
                                    threadgroup float  (&sh_m)[QT],
                                    threadgroup float  (&sh_l)[QT],
                                    uint2 tgid,
                                    uint tid,
                                    uint tiisg,
                                    uint sgitg,
                                    uint sgsize) {
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
  const int loc = int(t) - q0;              // this token's index within the request

  // PREFILL ONLY: take QT consecutive queries of the SAME request so
  // the chunk loop loads each K row and each V element once and reuses it
  // across the tile.
  //
  // With one threadgroup per query token, prefill re-streamed the entire K/V
  // range once per query: ~57 GB for a 512-token prompt on Qwen3-1.7B, which is
  // why prefill attention measured ~175 GFLOP/s against MLX's ~3000 and held 33%
  // of prefill GPU time. Decode keeps one threadgroup per query — it has one
  // query per request, so tiling there would divide the grid by QTILE and cost
  // far more parallelism than the (nonexistent) reuse is worth.
  const int qt = query_len > 1 ? int(QT) : 1;
  if ((loc % qt) != 0) { return; }  // non-leaders idle; uniform, no barrier split
  const int nq = min(qt, query_len - loc);

  const uint g = h / p.qpk;

  // Per-query windows. A query whose window is empty is left UNTOUCHED in `out`,
  // matching cpu_paged_attn.cpp's `continue`.
  int qjmin[QT];
  int qjmax[QT];
  bool qok[QT];
  int jlo = 0x7fffffff;
  int jhi = -1;
  for (int u = 0; u < int(QT); ++u) {
    // Slots past the tile's end are marked invalid rather than skipped, so every
    // loop below can keep its compile-time bound: they score to -INFINITY, get
    // their score row zeroed by the running-max branch, and are not stored.
    if (u >= nq) { qjmin[u] = 0; qjmax[u] = -1; qok[u] = false; continue; }
    const int pos = context + loc + u;
    int a = p.window_left >= 0 ? max(0, pos - p.window_left) : 0;
    int b = p.causal != 0u ? pos : seqlen - 1;
    if (p.window_right >= 0) { b = min(b, pos + p.window_right); }
    b = min(b, seqlen - 1);
    qjmin[u] = a;
    qjmax[u] = b;
    qok[u] = (b >= a);
    if (qok[u]) { jlo = min(jlo, a); jhi = max(jhi, b); }
  }
  if (jhi < jlo) { return; }  // no query in the tile has any key

  for (int u = 0; u < int(QT); ++u) {
    for (uint e = tid; e < p.d; e += p.tg) { acc[u * int(VT_PA_MAXD) + int(e)] = 0.0f; }
  }
  if (tid == 0u) {
    for (int u = 0; u < int(QT); ++u) { sh_m[u] = -INFINITY; sh_l[u] = 0.0f; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j0 = jlo; j0 <= jhi; j0 += int(VT_PA_CHUNK)) {
    const int jc = min(int(VT_PA_CHUNK), jhi - j0 + 1);

    // Scores for this chunk (scaled), ONE SIMDGROUP PER KEY, with the K row held
    // in REGISTERS across all `nq` queries of the tile.
    //
    // The original form gave one THREAD the whole d-length dot product. Adjacent
    // threads then read entirely different K rows (kbase varies per key), so the
    // loads never coalesced and this kernel ran at roughly 4% of the M4's memory
    // bandwidth. Splitting `d` across a simdgroup's lanes makes neighbouring
    // lanes read neighbouring elements of the SAME K row, which is the coalesced
    // direction, and lets each lane take a 4-wide vector load. Hoisting that row
    // into `kreg` then serves the whole tile from registers: K device traffic
    // drops by a factor of `nq`.
    //
    // `simd_sum` requires every lane to participate, so the loop bound is the
    // SIMDGROUP index; `jc` and `nq` are uniform, so no lane diverges.
    const uint n_sg = p.tg / sgsize;
    const uint d4 = p.d >> 2u;
    for (int idx = int(sgitg); idx < jc; idx += int(n_sg)) {
      const int j = j0 + idx;
      const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
      const int off = j % int(p.block_size);
      const ulong kbase = ulong(blk) * p.kc_blk + ulong(off) * p.kc_pg + ulong(g) * p.kc_hd;
      // The register path must cover the WHOLE row, else the staged and the
      // re-read halves would disagree; d <= VT_PA_MAXD makes this always true at
      // simdgroup width 32, and the bound keeps it honest if either changes.
      const bool fast = (p.q_dt == VT_DT_BF16 && p.kc_dt == VT_DT_BF16 &&
                         (p.d & 3u) == 0u && (kbase & 3ul) == 0ul &&
                         d4 <= VT_PA_KREG * sgsize);
      ushort4 kreg[VT_PA_KREG];
      if (fast) {
        device const ushort4* k4 = (device const ushort4*)((device const ushort*)kc + kbase);
        uint c = 0u;
        for (uint e = tiisg; e < d4; e += sgsize) { kreg[c++] = k4[e]; }
      }
      for (int u = 0; u < int(QT); ++u) {
        // Clamped: an invalid slot re-reads the tile's last valid row instead of
        // running off the end of Q. Its score is masked to -INFINITY below.
        const uint tu = t + uint(min(u, nq - 1));
        const ulong qoff = (ulong(tu) * ulong(p.hq) + ulong(h)) * ulong(p.d);
        float part = 0.0f;
        if (fast && (qoff & 3ul) == 0ul) {
          device const ushort4* q4 = (device const ushort4*)((device const ushort*)q + qoff);
          uint c = 0u;
          for (uint e = tiisg; e < d4; e += sgsize) {
            const ushort4 qv = q4[e];
            const ushort4 kv = kreg[c++];
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
        if (tiisg == 0u) {
          // Each query of the tile has its own causal/window bound, so the tile
          // walks the UNION of their key ranges and masks per query here.
          const bool inwin = qok[u] && j >= qjmin[u] && j <= qjmax[u];
          scores[u * int(VT_PA_CHUNK) + idx] = inwin ? dot * p.scale : -INFINITY;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Running max, then rescale the accumulator and the denominator — per query.
    for (int u = 0; u < int(QT); ++u) {
      float pm = -INFINITY;
      for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
        pm = max(pm, scores[u * int(VT_PA_CHUNK) + idx]);
      }
      const float mchunk = vt_tg_max(smem, tid, p.tg, pm);
      // UNIFORM (vt_tg_max broadcasts smem[0]), so this branch never splits a
      // barrier. No key of this chunk lies in this query's window — zero the row
      // so the V accumulation below, which is shared across the tile, adds
      // nothing for it rather than propagating -INFINITY.
      if (mchunk == -INFINITY) {
        for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
          scores[u * int(VT_PA_CHUNK) + idx] = 0.0f;
        }
        continue;
      }
      const float mold = sh_m[u];
      const float mnew = max(mold, mchunk);
      const float corr = mold == -INFINITY ? 0.0f : exp(mold - mnew);
      for (uint e = tid; e < p.d; e += p.tg) { acc[u * int(VT_PA_MAXD) + int(e)] *= corr; }

      // Masked keys hold -INFINITY, so exp() sends them to exactly 0.
      for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
        scores[u * int(VT_PA_CHUNK) + idx] = exp(scores[u * int(VT_PA_CHUNK) + idx] - mnew);
      }
      float ps = 0.0f;
      for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
        ps += scores[u * int(VT_PA_CHUNK) + idx];
      }
      // vt_tg_sum's LEADING barrier is what makes every thread's exp() above
      // visible to the V accumulation below, which reads the WHOLE chunk.
      const float lchunk = vt_tg_sum(smem, tid, p.tg, ps);
      if (tid == 0u) { sh_l[u] = sh_l[u] * corr + lchunk; sh_m[u] = mnew; }
    }

    // Resolve each key's V base address ONCE for the whole threadgroup — and now
    // for the whole tile.
    for (int idx = int(tid); idx < jc; idx += int(p.tg)) {
      const int j = j0 + idx;
      const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
      const int off = j % int(p.block_size);
      vbases[idx] = ulong(blk) * p.vc_blk + ulong(off) * p.vc_pg + ulong(g) * p.vc_hd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted V accumulation over BOTH the head dimension and the chunk's keys.
    // Parallelising over `d` alone used only `d` threads, so at the 2*d
    // threadgroup width half the group idled; measuring 4*d was WORSE than the
    // baseline for exactly that reason. Each key-group walks a strided slice of
    // the keys into its own partial, and the partials are summed once.
    //
    // Each V element is now loaded ONCE and applied to all `nq` queries from a
    // register, so V device traffic drops by the same factor as K's. Adjacent
    // threads still read adjacent elements of the same V row.
    const uint ngrp = min(max(p.tg / p.d, 1u), VT_PA_VGROUPS);
    const uint grp = min(tid / p.d, ngrp - 1u);
    const uint ge = tid % p.d;
    float s[QT];
    for (uint u = 0u; u < QT; ++u) { s[u] = 0.0f; }
    if (tid < ngrp * p.d) {
      for (int idx = int(grp); idx < jc; idx += int(ngrp)) {
        const float v = vt_load(vc, p.vc_dt, vbases[idx] + ulong(ge));
        // Compile-time bound: keeps `s` in registers. Invalid slots hold 0.
        for (uint u = 0u; u < QT; ++u) {
          s[u] += scores[u * VT_PA_CHUNK + uint(idx)] * v;
        }
      }
    }
    // `pacc` is reused per query, so each round needs its own pair of barriers.
    for (int u = 0; u < int(QT); ++u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < ngrp * p.d) { pacc[grp * VT_PA_MAXD + ge] = s[u]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = tid; e < p.d; e += p.tg) {
        float tot = 0.0f;
        for (uint gg = 0u; gg < ngrp; ++gg) { tot += pacc[gg * VT_PA_MAXD + e]; }
        acc[u * int(VT_PA_MAXD) + int(e)] += tot;
      }
    }
    // Orders this chunk's acc/scores reads before the next chunk overwrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // sh_l is written by thread 0 only.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int u = 0; u < int(QT); ++u) {
    if (!qok[u]) { continue; }  // empty window / past tile end: leave `out` alone
    const float inv = 1.0f / sh_l[u];  // every valid window has >= 1 key
    const ulong qoff = (ulong(t + uint(u)) * ulong(p.hq) + ulong(h)) * ulong(p.d);
    for (uint e = tid; e < p.d; e += p.tg) {
      vt_store(out, p.out_dt, qoff + ulong(e), acc[u * int(VT_PA_MAXD) + int(e)] * inv);
    }
  }
}


// DECODE — the reference implementation, restored verbatim.
//
// It is NOT routed through vt_paged_attention_impl. Sharing one body with the
// tiled kernel was measured three ways (runtime tile bound, compile-time bound,
// sized threadgroup references) and every one cost decode 15-17%: the shared
// form stages K through registers in two passes where decode wants one fused
// pass, and carries per-query barriers decode does not need. Decode runs 128x
// more often than prefill here, so that trade is heavily negative even though
// the tiling makes prefill attention 20% faster.
//
// Any change to the online-softmax algorithm must be made in BOTH this kernel
// and vt_paged_attention_impl below; test_ops_paged_attn covers both paths
// because PagedAttentionKernel picks by query length.
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
  // Per-key-group partial V accumulators, reduced into `acc` once per chunk.
  threadgroup float pacc[VT_PA_VGROUPS * VT_PA_MAXD];
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

    // Weighted V accumulation over BOTH the head dimension and the chunk's keys.
    // Parallelising over `d` alone used only `d` threads, so at the 2*d
    // threadgroup width half the group idled; measuring 4*d was WORSE than the
    // baseline for exactly that reason. Each key-group now walks a strided slice
    // of the keys into its own partial, and the partials are summed once.
    // Adjacent threads still read adjacent elements of the same V row, so the
    // coalescing that made this loop cheap is preserved.
    const uint ngrp = min(max(p.tg / p.d, 1u), VT_PA_VGROUPS);
    const uint grp = min(tid / p.d, ngrp - 1u);
    const uint ge = tid % p.d;
    if (tid < ngrp * p.d) {
      float s = 0.0f;
      for (int idx = int(grp); idx < jc; idx += int(ngrp)) {
        s += scores[idx] * vt_load(vc, p.vc_dt, vbases[idx] + ulong(ge));
      }
      pacc[grp * VT_PA_MAXD + ge] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.d; e += p.tg) {
      float tot = 0.0f;
      for (uint gg = 0u; gg < ngrp; ++gg) { tot += pacc[gg * VT_PA_MAXD + e]; }
      acc[e] += tot;
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


// Prefill: VT_PA_QTILE query tokens of the same request per threadgroup, so each
// K row is held in registers and each V element loaded once across the tile.
kernel void vt_paged_attention_tiled(device const uchar* q     [[buffer(0)]],
                                     device const uchar* kc    [[buffer(1)]],
                                     device const uchar* vc    [[buffer(2)]],
                                     device const int*   btab  [[buffer(3)]],
                                     device const int*   slens [[buffer(4)]],
                                     device const int*   qsl   [[buffer(5)]],
                                     device uchar*       out   [[buffer(6)]],
                                     constant VtPagedAttnParams& p [[buffer(7)]],
                                     uint2 tgid [[threadgroup_position_in_grid]],
                                     uint2 tid2 [[thread_position_in_threadgroup]],
                                     uint tiisg [[thread_index_in_simdgroup]],
                                     uint sgitg [[simdgroup_index_in_threadgroup]],
                                     uint sgsize [[threads_per_simdgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  threadgroup float scores[VT_PA_QTILE * VT_PA_CHUNK];
  threadgroup ulong vbases[VT_PA_CHUNK];
  threadgroup float pacc[VT_PA_VGROUPS * VT_PA_MAXD];
  threadgroup float acc[VT_PA_QTILE * VT_PA_MAXD];
  threadgroup float sh_m[VT_PA_QTILE];
  threadgroup float sh_l[VT_PA_QTILE];
  vt_paged_attention_impl<VT_PA_QTILE>(q, kc, vc, btab, slens, qsl, out, p, smem, scores,
                                       vbases, pacc, acc, sh_m, sh_l, tgid, tid2.x, tiisg,
                                       sgitg, sgsize);
}


// ===========================================================================
// PREFILL attention on the MATRIX UNITS.
//
// vt_paged_attention scores with simd_sum dot products and accumulates V with
// scalar FMAs, so it never issues a simdgroup_multiply_accumulate. Measured at
// 108 GFLOP/s against the GEMM's ~2250 on the same device in the same forward:
// 21x slower PER FLOP, purely because the matrix units are idle.
//
// This is the flash-attention form: S = Q@K^T and O += P@V both by mma, with the
// online softmax between them. The one MSL-specific subtlety is the running
// rescale of O. There is no elementwise operation on a simdgroup_matrix, so the
// per-row correction is applied as diag(corr) @ O — an ordinary mma against a
// diagonal matrix built in threadgroup memory.
//
// Tiling: BQ=32 queries x BK=16 keys, 8 simdgroups mapped as 4 row blocks x 2
// column halves. That gives each simdgroup exactly ONE S tile (S is 4x2 tiles of
// 8x8) and EIGHT O tiles (O is 4x16), so O stays in registers across the whole
// key loop and only the correction touches it.
#define VT_PAM_BQ   32u
#define VT_PAM_BK   16u
#define VT_PAM_MAXD 128u

kernel void vt_paged_attention_mma(device const uchar* q     [[buffer(0)]],
                                   device const uchar* kc    [[buffer(1)]],
                                   device const uchar* vc    [[buffer(2)]],
                                   device const int*   btab  [[buffer(3)]],
                                   device const int*   slens [[buffer(4)]],
                                   device const int*   qsl   [[buffer(5)]],
                                   device uchar*       out   [[buffer(6)]],
                                   constant VtPagedAttnParams& p [[buffer(7)]],
                                   uint2 tgid [[threadgroup_position_in_grid]],
                                   uint2 tid2 [[thread_position_in_threadgroup]],
                                   uint tiisg [[thread_index_in_simdgroup]],
                                   uint sgitg [[simdgroup_index_in_threadgroup]],
                                   uint sgsize [[threads_per_simdgroup]]) {
  threadgroup bfloat sq[VT_PAM_BQ * VT_PAM_MAXD];   // 8 KB
  threadgroup bfloat sk[VT_PAM_BK * VT_PAM_MAXD];   // 4 KB
  // V and P stay F32. Q and K are bf16 in the checkpoint so staging them as
  // bfloat is lossless, but the exp'd probabilities are COMPUTED here and the
  // scalar kernel accumulated V against them in f32. Rounding P to bf16 cost
  // enough precision to move a greedy token at prompt[0] tok=5 (engine 15344 vs
  // anchor 96251) — the near-tie gate catches what the op-level NMSE bar does not.
  threadgroup float  sv[VT_PAM_BK * VT_PAM_MAXD];   // 8 KB
  threadgroup float  sp[VT_PAM_BQ * VT_PAM_BK];     // 2 KB — exp'd scores for P@V
  threadgroup float  ss[VT_PAM_BQ * VT_PAM_BK];     // 2 KB — raw scores
  threadgroup float  sdiag[4u * 64u];               // 1 KB — per-row-block diag(corr)
  threadgroup float  smax[VT_PAM_BQ];
  threadgroup float  slsum[VT_PAM_BQ];
  threadgroup float  sout[8u * 64u];                // 2 KB — epilogue, per simdgroup

  const uint tid = tid2.x;
  const uint h = tgid.x;
  const uint t = tgid.y;

  int r = -1;
  for (uint i = 0u; i < p.num_reqs; ++i) {
    if (int(t) >= qsl[i] && int(t) < qsl[i + 1u]) { r = int(i); break; }
  }
  if (r < 0) { return; }
  const int q0 = qsl[r];
  const int query_len = qsl[r + 1] - q0;
  const int seqlen = slens[r];
  const int context = seqlen - query_len;
  const int loc = int(t) - q0;

  // One threadgroup per BQ queries OF THE SAME REQUEST; the rest idle. Keeping
  // tiles inside a request is what lets the whole tile share one block table.
  if ((loc % int(VT_PAM_BQ)) != 0) { return; }
  const int nq = min(int(VT_PAM_BQ), query_len - loc);

  const uint g = h / p.qpk;
  const uint rb = sgitg % 4u;   // row block: queries [rb*8, rb*8+8)
  const uint ch = sgitg / 4u;   // column half: dims [ch*64, ch*64+64)
  const uint lane = tid % sgsize;

  // Union of the tile's causal windows; per-row masking happens in the softmax.
  int jlo = 0x7fffffff;
  int jhi = -1;
  for (int u = 0; u < nq; ++u) {
    const int pos = context + loc + u;
    int a = p.window_left >= 0 ? max(0, pos - p.window_left) : 0;
    int b = p.causal != 0u ? pos : seqlen - 1;
    if (p.window_right >= 0) { b = min(b, pos + p.window_right); }
    b = min(b, seqlen - 1);
    if (b >= a) { jlo = min(jlo, a); jhi = max(jhi, b); }
  }
  if (jhi < jlo) { return; }

  // Stage Q once for the whole key loop.
  for (uint e = tid; e < VT_PAM_BQ * p.d; e += p.tg) {
    const uint rr = e / p.d, dd = e % p.d;
    if (int(rr) < nq) {
      const ulong qoff = (ulong(uint(int(t) + int(rr))) * ulong(p.hq) + ulong(h)) * ulong(p.d);
      sq[rr * VT_PAM_MAXD + dd] = bfloat(vt_load(q, p.q_dt, qoff + ulong(dd)));
    } else {
      sq[rr * VT_PAM_MAXD + dd] = bfloat(0.0f);
    }
  }
  if (tid < VT_PAM_BQ) { smax[tid] = -INFINITY; slsum[tid] = 0.0f; }

  simdgroup_float8x8 acc[8];
  for (uint c = 0u; c < 8u; ++c) { acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j0 = jlo; j0 <= jhi; j0 += int(VT_PAM_BK)) {
    const int jc = min(int(VT_PAM_BK), jhi - j0 + 1);

    // Gather this key block out of the paged cache into contiguous tiles.
    for (uint e = tid; e < VT_PAM_BK * p.d; e += p.tg) {
      const uint kk = e / p.d, dd = e % p.d;
      if (int(kk) < jc) {
        const int j = j0 + int(kk);
        const int blk = btab[r * p.bt_row + (j / int(p.block_size)) * p.bt_col];
        const int off = j % int(p.block_size);
        const ulong kbase = ulong(blk) * p.kc_blk + ulong(off) * p.kc_pg + ulong(g) * p.kc_hd;
        const ulong vbase = ulong(blk) * p.vc_blk + ulong(off) * p.vc_pg + ulong(g) * p.vc_hd;
        sk[kk * VT_PAM_MAXD + dd] = bfloat(vt_load(kc, p.kc_dt, kbase + ulong(dd)));
        sv[kk * VT_PAM_MAXD + dd] = vt_load(vc, p.vc_dt, vbase + ulong(dd));
      } else {
        // Padding keys score to -INFINITY below and carry zero V, so a partial
        // block needs no special case in the mma.
        sk[kk * VT_PAM_MAXD + dd] = bfloat(0.0f);
        sv[kk * VT_PAM_MAXD + dd] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T, one 8x8 tile per simdgroup, accumulated over d in 8-deep
    // steps. K is loaded TRANSPOSED so the [8 keys][8 d] block presents as the
    // [8 d][8 keys] operand the mma wants.
    simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (ch * 8u < uint(VT_PAM_BK)) {
      for (uint dd = 0u; dd < p.d; dd += 8u) {
        simdgroup_matrix<bfloat, 8, 8> qf, kf;
        simdgroup_load(qf, &sq[rb * 8u * VT_PAM_MAXD + dd], VT_PAM_MAXD);
        simdgroup_load(kf, &sk[ch * 8u * VT_PAM_MAXD + dd], VT_PAM_MAXD, 0, true);
        simdgroup_multiply_accumulate(sacc, qf, kf, sacc);
      }
      simdgroup_store(sacc, &ss[rb * 8u * VT_PAM_BK + ch * 8u], VT_PAM_BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, one thread per query row. Cheap next to the mma, and it
    // keeps the running max/sum in plain scalars where they are easy to reason
    // about.
    for (uint e = tid; e < 4u * 64u; e += p.tg) { sdiag[e] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < VT_PAM_BQ) {
      const uint rr = tid;
      float corr = 1.0f;
      if (int(rr) < nq) {
        const int pos = context + loc + int(rr);
        int a = p.window_left >= 0 ? max(0, pos - p.window_left) : 0;
        int b = p.causal != 0u ? pos : seqlen - 1;
        if (p.window_right >= 0) { b = min(b, pos + p.window_right); }
        b = min(b, seqlen - 1);
        float mc = -INFINITY;
        for (int idx = 0; idx < jc; ++idx) {
          const int j = j0 + idx;
          const bool inwin = (b >= a) && j >= a && j <= b;
          const float s = inwin ? ss[rr * VT_PAM_BK + uint(idx)] * p.scale : -INFINITY;
          ss[rr * VT_PAM_BK + uint(idx)] = s;
          mc = max(mc, s);
        }
        if (mc > -INFINITY) {
          const float mold = smax[rr];
          const float mnew = max(mold, mc);
          corr = mold == -INFINITY ? 0.0f : exp(mold - mnew);
          float ls = 0.0f;
          for (int idx = 0; idx < jc; ++idx) {
            const float e0 = exp(ss[rr * VT_PAM_BK + uint(idx)] - mnew);
            sp[rr * VT_PAM_BK + uint(idx)] = e0;
            ls += e0;
          }
          smax[rr] = mnew;
          slsum[rr] = slsum[rr] * corr + ls;
        } else {
          corr = 1.0f;  // nothing in window this block: O and l stay as they are
          for (int idx = 0; idx < jc; ++idx) { sp[rr * VT_PAM_BK + uint(idx)] = 0.0f; }
        }
      } else {
        for (int idx = 0; idx < jc; ++idx) { sp[rr * VT_PAM_BK + uint(idx)] = 0.0f; }
      }
      for (int idx = jc; idx < int(VT_PAM_BK); ++idx) {
        sp[rr * VT_PAM_BK + uint(idx)] = 0.0f;
      }
      // diag(corr) for this row, in its row block's 8x8.
      sdiag[(rr / 8u) * 64u + (rr % 8u) * 8u + (rr % 8u)] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O <- diag(corr) @ O, then O += P @ V. Both mma; the diagonal multiply is
    // how the per-row rescale reaches a register-resident simdgroup_matrix.
    simdgroup_float8x8 dg;
    simdgroup_load(dg, &sdiag[rb * 64u], 8u);
    for (uint c = 0u; c < 8u; ++c) {
      simdgroup_float8x8 tmp;
      simdgroup_multiply(tmp, dg, acc[c]);
      acc[c] = tmp;
    }
    for (uint kk = 0u; kk < VT_PAM_BK; kk += 8u) {
      simdgroup_float8x8 pf;
      simdgroup_load(pf, &sp[rb * 8u * VT_PAM_BK + kk], VT_PAM_BK);
      for (uint c = 0u; c < 8u; ++c) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, &sv[kk * VT_PAM_MAXD + ch * 64u + c * 8u], VT_PAM_MAXD);
        simdgroup_multiply_accumulate(acc[c], pf, vf, acc[c]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Epilogue: one 8x8 fragment at a time through this simdgroup's own scratch,
  // so no threadgroup barrier is needed (see vt_matmul_bt_mm).
  threadgroup float* mysc = &sout[sgitg * 64u];
  for (uint c = 0u; c < 8u; ++c) {
    simdgroup_store(acc[c], mysc, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64u; e += sgsize) {
      const uint rr = rb * 8u + e / 8u;
      const uint col = ch * 64u + c * 8u + e % 8u;
      if (int(rr) < nq && col < p.d && slsum[rr] > 0.0f) {
        const ulong qoff = (ulong(uint(int(t) + int(rr))) * ulong(p.hq) + ulong(h)) * ulong(p.d);
        vt_store(out, p.out_dt, qoff + ulong(col), mysc[e] / slsum[rr]);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}


// Bandwidth probe (test-only, opt-in). Reads `n_chunks` chunks of `chunk_f4`
// float4s each, advancing `stride_f4` float4s between chunks. stride == chunk is
// a contiguous stream; stride > chunk reproduces the paged KV cache's access
// pattern, where one kv head's slot data is `d` elements every
// `num_kv_heads * d`. This exists to settle whether decode attention's measured
// 29 GB/s is a LAYOUT effect or something else, without changing the cache.
struct VtBwParams { uint n_chunks; uint chunk_f4; uint stride_f4; uint tg; uint chunks_per_tg; uint pad; };

kernel void vt_bw_probe(device const float4* src [[buffer(0)]],
                        device float*        out [[buffer(1)]],
                        constant VtBwParams& p   [[buffer(2)]],
                        uint2 tgid [[threadgroup_position_in_grid]],
                        uint2 tid2 [[thread_position_in_threadgroup]]) {
  threadgroup float smem[VT_TG_MAX];
  const uint tid = tid2.x;
  // Each threadgroup walks MANY chunks. One threadgroup per chunk would size the
  // group by chunk_f4 (16 threads) and measure launch overhead rather than
  // bandwidth — which is exactly what the first version of this probe did.
  const uint c0 = tgid.x * p.chunks_per_tg;
  float acc = 0.0f;
  for (uint i = tid; i < p.chunks_per_tg * p.chunk_f4; i += p.tg) {
    const uint c = c0 + i / p.chunk_f4;
    const uint e = i % p.chunk_f4;
    if (c < p.n_chunks) {
      const float4 v = src[ulong(c) * ulong(p.stride_f4) + ulong(e)];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  const float s = vt_tg_sum(smem, tid, p.tg, acc);
  if (tid == 0u) { out[tgid.x] = s; }
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
