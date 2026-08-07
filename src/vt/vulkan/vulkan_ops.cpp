// Vulkan backend — op kernels: descriptor binding + dispatch of the committed
// SPIR-V in vulkan_spirv.h, plus the `RegisterOp` table entries. BACKEND-VULKAN,
// W0 skeleton. Self-registering TU, copying the `src/vt/cpu/cpu_ops.cpp`
// Registrar idiom exactly, so adding this backend edited NO existing kernel file.
//
// WHAT THIS TU COVERS (deliberately a SEAM PROOF, not a model):
//   kAdd, kRelu, kSiluAndMul, kCastBf16, kCastF32, kLayerNorm, kRmsNorm and the
//   single kFusedChain registration that inherits the portable fusion catalog.
// That set spans every structural class the seam has to get right: flat
// elementwise, a rank-1 broadcast, a dtype-converting copy, TWO different row
// reductions, an optional in-place residual stream, and the recipe interpreter.
// It matches the Metal skeleton's set exactly, so the two backends are directly
// comparable through tests/vt/test_backend_cross_device.cpp.
//
// WHAT IS STILL STUBBED: everything else. `kMatmul`/`kMatmulBT`,
// `kPagedAttention`, `kReshapeAndCache`, the whole quant tier and the sampler
// ops are NOT registered, so `vt::GetOp` throws its normal "no kernel for op N
// on device type 3" for them (src/vt/ops.cpp:104-111 — a partial backend is a
// supported, tested state). NO MODEL RUNS ON THIS BACKEND.
//
// BINDING MODEL: every tensor operand occupies TWO consecutive descriptor
// bindings onto the SAME VkBuffer — a uint32_t view and a uint16_t view — and
// its BYTE OFFSET travels in the push constants. See
// src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL for why.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vulkan_buffers.h"
#include "vulkan_context.h"
#include "vt/ops.h"

namespace vt::vulkan {
namespace {

// Storage dtype -> the shader-side code (vt_common.glsl VT_DT_*).
uint32_t DtypeCode(DType d) {
  switch (d) {
    case DType::kF32: return 0;
    case DType::kF16: return 1;
    case DType::kBF16: return 2;
    default: break;
  }
  VT_CHECK(false, "vulkan: unsupported storage dtype (f32/f16/bf16 only in the W0 skeleton)");
  return 0;
}

// Collects the (buffer, byte-offset) pairs for a dispatch. Each Add() appends
// the SAME buffer twice — bindings 2k and 2k+1, the u32 and u16 views — and
// returns the byte offset for the push-constant block.
class Binder {
 public:
  uint32_t Add(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    buffers_.push_back(r.buffer);
    // f32 access indexes a uint32_t[] view, so a f32 operand's byte offset must
    // be 4-byte aligned; 16-bit access only needs 2. Tensor storage always
    // satisfies this (allocations are 64-byte aligned and views advance by whole
    // elements), but a violation would silently read shifted data.
    VT_CHECK(t.dtype != DType::kF32 || r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    VT_CHECK(r.offset % 2 == 0,
             std::string("vulkan: ") + what + " has an odd byte offset");
    return r.offset;
  }
  // A raw buffer bound ONCE (no 16-bit view): the fused-chain step list.
  void AddRaw(void* buffer) { buffers_.push_back(buffer); }

  // A tensor bound through the uint32_t view ONLY, for shaders that declare a
  // single binding per operand because the operand is integer (embedding ids,
  // sampler token ids) or f32-by-contract (logits). Binding the unused 16-bit
  // view would need the shader to declare it too, and a descriptor a shader does
  // not declare must not be written.
  uint32_t AddU32Only(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    VT_CHECK(r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    return r.offset;
  }

  const void* const* data() const { return buffers_.data(); }
  uint32_t count() const { return static_cast<uint32_t>(buffers_.size()); }

 private:
  std::vector<const void*> buffers_;
};

// ---- Host mirrors of the shaders' push-constant blocks. Field order and types
// must match the GLSL declarations EXACTLY. GLSL `uint`/`float` are 4-byte with
// 4-byte alignment and every block below is a run of 4-byte scalars, so the std430
// push-constant layout coincides with the C++ layout with no padding surprises.
struct AddParams {
  uint32_t n, d, a_dt, b_dt, out_dt, bcast, a_off, b_off, out_off;
};
struct UnaryParams {
  uint32_t n, a_dt, out_dt, a_off, out_off;
};
// vt_cast carries its dtype pair in specialization constants instead, so its
// push block is only the shape and the two offsets.
struct CastParams {
  uint32_t n, a_off, out_off;
};
struct MatmulParams {
  uint32_t m, n, k, a_off, b_off, out_off;
};
struct EmbeddingParams {
  uint32_t t, h, table_off, ids_off, out_off;
};
struct ArgmaxParams {
  uint32_t n, v, logits_off, out_off;
};
struct QkvSplitParams {
  uint32_t tokens, q_dim, k_dim, v_dim, src_off, q_off, k_off, v_off;
};
struct RopeFromCacheParams {
  uint32_t tokens, half_dim, rotary_dim, hq, hk;
  uint32_t q_s0, q_s1, k_s0, k_s1;
  uint32_t q_off, k_off, c_off, p_off;
};
struct ReshapeAndCacheParams {
  uint32_t num_slots, n_elems, block_size;
  uint32_t k_blk, k_pg, v_blk, v_pg;
  uint32_t k_tok, v_tok;
  uint32_t k_off, v_off, kc_off, vc_off, sm_off;
};
struct PagedAttnParams {
  uint32_t total_q, hq, d, block_size, qpk, num_reqs;
  uint32_t causal;
  int32_t window_left, window_right;
  uint32_t kc_blk, kc_pg, kc_hd;
  uint32_t vc_blk, vc_pg, vc_hd;
  uint32_t bt_row, bt_col;
  uint32_t q_off, k_off, v_off, out_off;
  uint32_t bt_off, sl_off, qsl_off;
  float scale;
  float softcap;
};
struct SiluMulParams {
  uint32_t t, d, x_dt, out_dt, x_off, out_off;
};
struct RmsParams {
  uint32_t t, h, x_dt, w_dt, out_dt, res_dt, has_res, gemma, x_off, w_off, out_off, res_off;
  float eps;
};
struct LayerNormParams {
  uint32_t rows, d, x_dt, w_dt, b_dt, out_dt, has_w, has_b, x_off, w_off, b_off, out_off;
  float eps;
};
struct FcParams {
  uint32_t t, h, nsteps, x_dt, w_dt, res_dt, out_dt, x_off, w_off, res_off, out_off;
  float eps;
};

// Vulkan only GUARANTEES 128 bytes of push-constant space (maxPushConstantsSize);
// staying inside it is what keeps this backend portable without a probe.
static_assert(sizeof(RmsParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(LayerNormParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(FcParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(PagedAttnParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");

template <typename P>
void Go(const char* name, const Binder& b, const P& p, uint32_t groups,
        const uint32_t* spec = nullptr, uint32_t spec_count = 0) {
  VulkanContext::Get().Dispatch(name, b.data(), b.count(), &p, sizeof(P), groups, spec,
                                spec_count);
}

// ---------------------------------------------------------------------------
// Kernels. Every argument was already validated by the vt:: wrapper in
// src/vt/ops.cpp before GetOp dispatched here, so these only translate.
// ---------------------------------------------------------------------------

// cpu_layernorm.cpp:87-99 AddKernel.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  const bool bcast = b.rank == 1 && a.rank != 1;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "add: a");
  const uint32_t b_off = bind.Add(b, "add: b");
  const uint32_t out_off = bind.Add(out, "add: out");
  AddParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(d),
              DtypeCode(a.dtype),      DtypeCode(b.dtype),
              DtypeCode(out.dtype),    bcast ? 1u : 0u,
              a_off,                   b_off,
              out_off};
  Go("vt_add", bind, p, FlatGroupCount(n));
}

// cpu_layernorm.cpp:75-85 ReluKernel.
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  Binder bind;
  const uint32_t x_off = bind.Add(x, "relu: x");
  const uint32_t out_off = bind.Add(out, "relu: out");
  UnaryParams p{static_cast<uint32_t>(n), DtypeCode(x.dtype), DtypeCode(out.dtype), x_off,
                out_off};
  Go("vt_relu", bind, p, FlatGroupCount(n));
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — one shader serves both
// (the CPU pair is likewise the same LoadF32/StoreF32 body twice).
void CastKernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  Binder bind;
  const uint32_t in_off = bind.Add(in, "cast: in");
  const uint32_t out_off = bind.Add(out, "cast: out");
  // The dtype pair rides SPECIALIZATION CONSTANTS rather than push constants, so
  // the per-element dtype branch is folded away at pipeline creation and each
  // (src, dst) pair is its own cached pipeline. Ascending constantID order, which
  // is what GetPipeline binds against the module's declared SpecIds.
  const uint32_t spec[2] = {DtypeCode(in.dtype), DtypeCode(out.dtype)};
  CastParams p{static_cast<uint32_t>(n), in_off, out_off};
  Go("vt_cast", bind, p, FlatGroupCount(n), spec, 2);
}

// cpu_ops.cpp:252-264 SiluAndMulKernel.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "silu_and_mul: x");
  const uint32_t out_off = bind.Add(out, "silu_and_mul: out");
  SiluMulParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(d), DtypeCode(x.dtype),
                  DtypeCode(out.dtype), x_off, out_off};
  Go("vt_silu_and_mul", bind, p, FlatGroupCount(t * d));
}

// cpu_ops.cpp:225-250 RmsNormKernel. One workgroup per token row.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm: x");
  const uint32_t w_off = bind.Add(w, "rmsnorm: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm: out");
  // Bindings 6/7 are always written: a descriptor a shader statically uses must
  // be valid even on the code path that never reads it. With has_res == 0 they
  // alias `out` and are dead.
  const uint32_t res_off =
      residual != nullptr ? bind.Add(*residual, "rmsnorm: residual") : bind.Add(out, "rmsnorm: out");
  RmsParams p{static_cast<uint32_t>(t),
              static_cast<uint32_t>(h),
              DtypeCode(x.dtype),
              DtypeCode(w.dtype),
              DtypeCode(out.dtype),
              residual != nullptr ? DtypeCode(residual->dtype) : 0u,
              residual != nullptr ? 1u : 0u,
              args.gemma ? 1u : 0u,
              x_off,
              w_off,
              out_off,
              res_off,
              args.eps};
  Go("vt_rms_norm", bind, p, static_cast<uint32_t>(t));
}

// cpu_layernorm.cpp:49-73 LayerNormKernel.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "layer_norm: x");
  const uint32_t w_off =
      weight != nullptr ? bind.Add(*weight, "layer_norm: weight") : bind.Add(x, "layer_norm: x");
  const uint32_t b_off =
      bias != nullptr ? bind.Add(*bias, "layer_norm: bias") : bind.Add(x, "layer_norm: x");
  const uint32_t out_off = bind.Add(out, "layer_norm: out");
  LayerNormParams p{static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(d),
                    DtypeCode(x.dtype),
                    weight != nullptr ? DtypeCode(weight->dtype) : 0u,
                    bias != nullptr ? DtypeCode(bias->dtype) : 0u,
                    DtypeCode(out.dtype),
                    weight != nullptr ? 1u : 0u,
                    bias != nullptr ? 1u : 0u,
                    x_off,
                    w_off,
                    b_off,
                    out_off,
                    args.eps};
  Go("vt_layer_norm", bind, p, static_cast<uint32_t>(rows));
}

// cpu_ops.cpp:1649-1702 FusedChainInterpKernel — the Tier-1 interpreter. ONE
// registration; every Tier-1-able recipe in include/vt/recipes.h realizes
// through it, and every non-Tier-1 recipe realizes through the device-agnostic
// Tier-0 composite in src/vt/ops.cpp, which re-enters this backend's standalone
// ops. That is the whole "2 lines -> all 10 recipes" property the spike claims.
void FusedChainKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  VT_CHECK(r.n >= 1 && r.n <= kMaxFusedSteps, "vulkan fused_chain: bad step count");

  // Words per step, matching VT_STEP_WORDS in vt_fused_chain.comp.
  constexpr uint32_t kStepWords = 5;
  std::vector<uint32_t> steps(static_cast<size_t>(r.n) * kStepWords, 0u);
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    uint32_t op = 0;
    switch (st.op) {
      case FOp::kAdd: op = 0; break;
      case FOp::kMul: op = 1; break;
      case FOp::kSilu: op = 2; break;
      case FOp::kSigmoid: op = 3; break;
      case FOp::kRmsNorm:
        // Mirrors the CPU interpreter's assertion (cpu_ops.cpp:1674): the shader
        // hard-codes the mean-square reduction, so any other kind must not reach it.
        VT_CHECK(st.reduce == FReduce::kMeanSquare,
                 "vulkan fused_chain: rmsnorm needs kMeanSquare");
        op = 4;
        break;
      default:
        VT_CHECK(false, "vulkan fused_chain: non-Tier-1 opcode reached the interpreter");
    }
    // Canonical operand indices (cpu_ops.cpp:1621-1643): 0=x 1=weight 2=residual
    // 3=out, with 2 and 3 the only writable slots.
    VT_CHECK(st.out == 2 || st.out == 3, "vulkan fused_chain: step writes a read-only operand");
    VT_CHECK(st.in[0] <= 3 && st.in[1] <= 3, "vulkan fused_chain: operand index out of range");
    VT_CHECK(residual != nullptr || (st.out != 2 && st.in[0] != 2 && st.in[1] != 2),
             "vulkan fused_chain: recipe touches the residual slot but none was bound");
    const size_t base = static_cast<size_t>(s) * kStepWords;
    steps[base + 0] = op;
    steps[base + 1] = st.out;
    steps[base + 2] = st.in[0];
    steps[base + 3] = st.in[1];
    steps[base + 4] = st.gemma ? 1u : 0u;
  }

  VulkanContext& ctx = VulkanContext::Get();
  const size_t step_bytes = steps.size() * sizeof(uint32_t);
  VT_CHECK(step_bytes <= VulkanContext::kScratchBytes,
           "vulkan fused_chain: step list exceeds the scratch buffer");
  std::memcpy(ctx.ScratchData(), steps.data(), step_bytes);

  Binder bind;
  const uint32_t x_off = bind.Add(x, "fused_chain: x");
  const uint32_t w_off = bind.Add(weight, "fused_chain: weight");
  const uint32_t res_off = residual != nullptr ? bind.Add(*residual, "fused_chain: residual")
                                               : bind.Add(out, "fused_chain: out");
  const uint32_t out_off = bind.Add(out, "fused_chain: out");
  bind.AddRaw(ctx.ScratchBuffer());
  FcParams p{static_cast<uint32_t>(t),
             static_cast<uint32_t>(h),
             static_cast<uint32_t>(r.n),
             DtypeCode(x.dtype),
             DtypeCode(weight.dtype),
             residual != nullptr ? DtypeCode(residual->dtype) : 0u,
             DtypeCode(out.dtype),
             x_off,
             w_off,
             res_off,
             out_off,
             eps};
  Go("vt_fused_chain", bind, p, static_cast<uint32_t>(t));
}

// cpu_ops.cpp:187-260 MatmulChunked / MatmulKernel / MatmulBTKernel. One
// invocation per OUTPUT ELEMENT with the whole K reduction on it, which is what
// the CPU kernel does too (it deliberately never splits a K reduction across
// threads), so the accumulation ORDER matches rather than merely the tolerance.
//
// The naive body is the portable correctness tier on purpose; the tiled and
// cooperative-matrix ports (llama.cpp mul_mm.comp / mul_mm_cm2.comp) are VK-C,
// which needs exactly this as its same-device A/B reference.
// TACTIC SELECTION (VK-C). Every condition below is a HARD requirement of the
// cooperative-matrix path, not a heuristic, and failing any one of them means the
// scalar kernel -- which is always correct -- runs instead:
//
//   * the device reports the EXACT configuration the committed coopmat SPIR-V is
//     written to (16x16x16, bf16/bf16/f32/f32, SUBGROUP). Vulkan matches
//     configurations exactly, so "close enough" does not exist;
//   * subgroup size is 32, because the shader's workgroup is a literal 32 (see
//     the shader for why the size cannot travel as a specialization constant at
//     this target);
//   * BOTH operands are bf16. Every configuration GB10 reports takes
//     bf16/f16/int8 inputs, so f32 operands can never use this path -- a hardware
//     constraint, not a policy;
//   * K is a multiple of 16. A ragged K tail cannot be masked inside a
//     cooperative-matrix load, and silently truncating it would drop terms from
//     the dot product. Ragged M and N are fine: the shader bounds-checks its
//     store.
//
// MEASURED: GB10 satisfies all four; llvmpipe -- the only Vulkan device CI can
// reach -- fails the first, so CI exercises the scalar tactic and this selection
// returning false is the property CI can actually gate.
bool CoopMatMatmulUsable(const Tensor& a, const Tensor& b, int64_t k, int64_t m, int64_t n) {
  // VT_VULKAN_COOPMAT=0 forces the scalar tactic. This exists for ONE reason: a
  // same-binary A/B. Comparing the two tactics across two builds would confound
  // the kernel with everything else that differs between them, and the project's
  // benchmark protocol wants the arms to differ in exactly one thing. Default is
  // ON -- absent or any value other than "0" leaves selection to the capability
  // probe, so production behaviour is unchanged by the lever's existence.
  static const bool kDisabled = [] {
    const char* v = std::getenv("VT_VULKAN_COOPMAT");
    return v != nullptr && std::strcmp(v, "0") == 0;
  }();
  if (kDisabled) return false;

  const VulkanContext& ctx = VulkanContext::Get();
  return ctx.coopmat_bf16_f32() && ctx.subgroup_size() == 32 &&
         a.dtype == DType::kBF16 && b.dtype == DType::kBF16 && k % 16 == 0 &&
         // M AND N MUST ALSO BE WHOLE TILES. `coopMatLoad` reads a FULL 16x16
         // tile with no masking, so a partial tile reads past the end of the
         // operand -- and the store being bounds-checked does not save it,
         // because the fault happens on the LOAD. MEASURED: lm_head at M=1
         // (single decode token) read 15 rows (~30 KB) past a small activation
         // buffer, faulted the GPU, and the fence NEVER SIGNALLED -- an infinite
         // vkWaitForFences, which presents as a hang, not as an error.
         //
         // The original correctness gate used M=20, N=12 precisely to exercise
         // ragged shapes and PASSED, because there the out-of-bounds read stayed
         // inside the allocation and its garbage rows were discarded by the
         // bounds-checked store. Raggedness alone was not enough; the read has to
         // leave the allocation to fault.
         m % 16 == 0 && n % 16 == 0;
}

// GEMV TACTIC SELECTION (VK-F). Same shape of contract as the coopmat predicate
// above -- every requirement is a hard one, and failing any of them runs the
// always-correct scalar kernel instead.
//
// The problem this solves is COALESCING, measured: vt_matmul was ~55% of all GPU
// time in an e2e decode run. It puts one invocation on each output element and
// loops K there, so for MatmulBT lane j reads b[j*k + q] and adjacent lanes land
// k*2 bytes apart -- each pulling its own cache line to use 2 bytes of it. The
// GEMV shader instead gives each output element a workgroup whose lanes stride K,
// so adjacent lanes read adjacent addresses.
//
//   * MatmulBT ONLY. In the other orientation vt_matmul reads b[q*n + j], which
//     is ALREADY coalesced across lanes; the GEMV shape would make that strided
//     and strictly worse. This is not a universally better kernel and the
//     predicate does not pretend otherwise.
//   * m == 1, the decode shape. One workgroup per output element is the right
//     trade only when there are few of them: at prefill m*n workgroups would each
//     do k/128 multiplies, and prefill is the coopmat tactic's job anyway.
//   * k >= the workgroup width, so the strided loop actually has work for every
//     lane. Below that most lanes contribute a zero partial and the reduction
//     costs more than the loop saves.
//
// ACCUMULATION ORDER: the K reduction becomes a tree, so this tactic does NOT
// share the CPU's accumulation order -- it sits in the NMSE tier alongside
// coopmat. That is why it is gated on a token-exactness run and not on an NMSE
// bound alone.
bool GemvMatmulUsable(bool bt, int64_t k, int64_t m) {
  // VT_VULKAN_GEMV=0 forces the scalar tactic, for the same single reason the
  // coopmat lever exists: a same-binary A/B, so the arms differ in exactly one
  // thing. Default ON.
  static const bool kDisabled = [] {
    const char* v = std::getenv("VT_VULKAN_GEMV");
    return v != nullptr && std::strcmp(v, "0") == 0;
  }();
  if (kDisabled) return false;
  if (!bt || m != 1) return false;
  return k >= static_cast<int64_t>(kWorkgroupSize);
}

template <bool kBT>
void MatmulGeneric(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1];
  const int64_t n = kBT ? b.shape[0] : b.shape[1];
  if (m == 0 || n == 0) return;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "matmul: a");
  const uint32_t b_off = bind.Add(b, "matmul: b");
  const uint32_t out_off = bind.Add(out, "matmul: out");

  if (CoopMatMatmulUsable(a, b, k, m, n)) {
    // One workgroup (= one subgroup) per 16x16 OUTPUT TILE. Deliberately not
    // FlatGroupCount, which divides an element count by the workgroup size: here
    // the whole subgroup cooperates on one tile.
    const uint32_t tiles =
        static_cast<uint32_t>(((m + 15) / 16) * ((n + 15) / 16));
    const uint32_t spec[2] = {kBT ? 1u : 0u, DtypeCode(out.dtype)};
    MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                   static_cast<uint32_t>(k), a_off, b_off, out_off};
    Go("vt_matmul_coopmat", bind, p, tiles, spec, 2);
    return;
  }

  if (GemvMatmulUsable(kBT, k, m)) {
    // ONE WORKGROUP PER OUTPUT ELEMENT -- not FlatGroupCount, which would divide
    // the element count by the workgroup size and put the whole K reduction back
    // on a single lane. The workgroup cooperates on one element.
    const uint32_t groups = static_cast<uint32_t>(m * n);
    const uint32_t spec[3] = {DtypeCode(a.dtype), DtypeCode(b.dtype),
                              DtypeCode(out.dtype)};
    MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                   static_cast<uint32_t>(k), a_off, b_off, out_off};
    Go("vt_matmul_vec", bind, p, groups, spec, 3);
    return;
  }

  // Scalar tactic: the portable reference, and the only one whose accumulation
  // ORDER matches the CPU kernel's.
  // Ascending constantID order: a dtype, b dtype, out dtype, orientation.
  const uint32_t spec[4] = {DtypeCode(a.dtype), DtypeCode(b.dtype), DtypeCode(out.dtype),
                            kBT ? 1u : 0u};
  MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                 a_off, b_off, out_off};
  Go("vt_matmul", bind, p, FlatGroupCount(m * n), spec, 4);
}

// cpu_ops.cpp:661-672 EmbeddingKernel. One output ELEMENT per invocation.
// The id dtype (i32 vs i64) is a specialization constant rather than a
// per-element branch; see the shader for why only the low 32 bits are read.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1];
  if (t == 0 || h == 0) return;
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "vulkan embedding: ids must be i32 or i64");
  Binder bind;
  const uint32_t table_off = bind.Add(table, "embedding: table");
  const uint32_t ids_off = bind.Add(ids, "embedding: ids");
  const uint32_t out_off = bind.Add(out, "embedding: out");
  const uint32_t spec[3] = {DtypeCode(table.dtype), DtypeCode(out.dtype),
                            ids.dtype == DType::kI64 ? 1u : 0u};
  EmbeddingParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(h), table_off, ids_off,
                    out_off};
  Go("vt_embedding", bind, p, FlatGroupCount(t * h), spec, 3);
}

// cpu_sample.cpp:40-56 GreedyArgmaxKernel. ONE INVOCATION PER ROW, because the
// tie-break (strict `>`, so the first maximum wins) is part of the token-exact
// contract and a tree reduction would have to carry the index and break ties
// toward the lower one at every merge. Rows are few at decode; the vocabulary
// scan is the slow axis and is deliberately left for a later change.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  VT_CHECK(logits.dtype == DType::kF32, "vulkan greedy argmax: logits must be f32");
  VT_CHECK(token_ids.dtype == DType::kI64, "vulkan greedy argmax: token_ids must be i64");
  Binder bind;
  const uint32_t logits_off = bind.AddU32Only(logits, "argmax: logits");
  const uint32_t out_off = bind.AddU32Only(token_ids, "argmax: token_ids");
  ArgmaxParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(v), logits_off, out_off};
  // ONE WORKGROUP PER ROW, matching vt_rms_norm's convention -- the shader
  // tree-reduces the vocabulary across the workgroup's lanes. NOT
  // FlatGroupCount(n), which would allot one INVOCATION per row and leave the
  // vocabulary scan serial; at decode n is 1, so that dispatched a single lane
  // and measured 10.03 ms per call.
  Go("vt_greedy_argmax", bind, p, static_cast<uint32_t>(n));
}

// cpu_paged_attn.cpp:52-171 PagedAttentionKernel. ONE WORKGROUP per (query
// token, query head), lanes splitting the head dimension; see the shader for why
// the CPU's three passes become one online-softmax recurrence (its `probs` array
// is one float per key in the window, which a shader cannot allocate).
//
// This is the only kernel in the backend with NO llama.cpp counterpart to port
// from: its Vulkan backend has no paged KV anywhere. The block-table indirection
// and windowing come from the CPU kernel above, the online-softmax skeleton from
// flash_attn.comp's shape.
void PagedAttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];

  // PER-CALL REFUSAL, not a silent regression. An fp8 KV cache stores 1-byte
  // pages that must be dequantised as Dequant(fp8) * k_scale|v_scale before the
  // f32 softmax (cpu_paged_attn.cpp:79-93). This shader reads f32/f16/bf16 only,
  // so rather than throw -- which would REMOVE a capability the portable
  // reference tier already provides -- it declines through the provider seam and
  // forwards to the next provider down, which is exactly what GetOpFallback is
  // for (op_provider.h:94-100: per-call refusal belongs in the kernel, because
  // GetOp has no shape or dtype to inspect).
  if (args.kv_cache_dtype != vt::Fp8KVCacheDataType::kAuto) {
    auto next = reinterpret_cast<PagedAttentionFn>(
        GetOpFallback(OpId::kPagedAttention, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc, args);
    return;
  }

  if (total_q == 0 || hq == 0 || d == 0) return;
  // The shader keeps its accumulator in VT_PA_ACC_MAX slots per lane, one per
  // head-dim element the lane owns. Asserted rather than trusted: overflowing it
  // would write past a local array.
  VT_CHECK(d <= 8 * static_cast<int64_t>(kWorkgroupSize),
           "vulkan paged attention: head dim " + std::to_string(d) +
               " exceeds the per-lane accumulator (8 * workgroup)");
  VT_CHECK(num_kv_heads > 0 && hq % num_kv_heads == 0,
           "vulkan paged attention: query heads must be a multiple of kv heads");

  Binder bind;
  const uint32_t q_off = bind.Add(query, "paged_attn: query");
  const uint32_t k_off = bind.Add(k_cache, "paged_attn: k_cache");
  const uint32_t v_off = bind.Add(v_cache, "paged_attn: v_cache");
  const uint32_t out_off = bind.Add(out, "paged_attn: out");
  const uint32_t bt_off = bind.AddU32Only(block_table, "paged_attn: block_table");
  const uint32_t sl_off = bind.AddU32Only(seq_lens, "paged_attn: seq_lens");
  const uint32_t qsl_off = bind.AddU32Only(query_start_loc, "paged_attn: query_start_loc");

  const int64_t wl = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t wr = args.window_size.has_value() ? args.window_size->right : -1;

  const uint32_t spec[4] = {DtypeCode(query.dtype), DtypeCode(k_cache.dtype),
                            DtypeCode(v_cache.dtype), DtypeCode(out.dtype)};
  PagedAttnParams p{static_cast<uint32_t>(total_q),
                    static_cast<uint32_t>(hq),
                    static_cast<uint32_t>(d),
                    static_cast<uint32_t>(block_size),
                    static_cast<uint32_t>(hq / num_kv_heads),
                    static_cast<uint32_t>(num_reqs),
                    args.causal ? 1u : 0u,
                    static_cast<int32_t>(wl),
                    static_cast<int32_t>(wr),
                    static_cast<uint32_t>(k_cache.stride[0]),
                    static_cast<uint32_t>(k_cache.stride[1]),
                    static_cast<uint32_t>(k_cache.stride[2]),
                    static_cast<uint32_t>(v_cache.stride[0]),
                    static_cast<uint32_t>(v_cache.stride[1]),
                    static_cast<uint32_t>(v_cache.stride[2]),
                    static_cast<uint32_t>(block_table.stride[0]),
                    static_cast<uint32_t>(block_table.stride[1]),
                    q_off,
                    k_off,
                    v_off,
                    out_off,
                    bt_off,
                    sl_off,
                    qsl_off,
                    args.scale,
                    args.logits_soft_cap};
  // One workgroup per (token, head) -- NOT FlatGroupCount, which divides by the
  // workgroup size; here the whole workgroup cooperates on one output row.
  Go("vt_paged_attn", bind, p, static_cast<uint32_t>(total_q * hq), spec, 4);
}

// cpu_cache.cpp:33-72 ReshapeAndCacheKernel. Pure BYTE MOVEMENT -- the CPU
// kernel is two memcpys per token and converts nothing -- so the dtype selects
// only the storage WIDTH to copy at, and the gate for it is bit-exactness.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems = k_cache.shape[2] * k_cache.shape[3];  // one token's page
  if (num_slots == 0 || n_elems == 0) return;
  VT_CHECK(slot_mapping.dtype == DType::kI64,
           "vulkan reshape_and_cache: slot_mapping must be i64");
  VT_CHECK(k.dtype == k_cache.dtype && v.dtype == v_cache.dtype,
           "vulkan reshape_and_cache: source and cache dtypes must match (this op "
           "moves bytes and converts nothing)");

  Binder bind;
  const uint32_t k_off = bind.Add(k, "reshape_and_cache: k");
  const uint32_t v_off = bind.Add(v, "reshape_and_cache: v");
  const uint32_t kc_off = bind.Add(k_cache, "reshape_and_cache: k_cache");
  const uint32_t vc_off = bind.Add(v_cache, "reshape_and_cache: v_cache");
  const uint32_t sm_off = bind.AddU32Only(slot_mapping, "reshape_and_cache: slot_mapping");

  const uint32_t spec[1] = {k.dtype == DType::kF32 ? 0u : 1u};
  ReshapeAndCacheParams p{static_cast<uint32_t>(num_slots),
                          static_cast<uint32_t>(n_elems),
                          static_cast<uint32_t>(block_size),
                          static_cast<uint32_t>(k_cache.stride[0]),
                          static_cast<uint32_t>(k_cache.stride[1]),
                          static_cast<uint32_t>(v_cache.stride[0]),
                          static_cast<uint32_t>(v_cache.stride[1]),
                          static_cast<uint32_t>(k.stride[0]),
                          static_cast<uint32_t>(v.stride[0]),
                          k_off,
                          v_off,
                          kc_off,
                          vc_off,
                          sm_off};
  Go("vt_reshape_and_cache", bind, p, FlatGroupCount(num_slots * n_elems), spec, 1);
}

// vt::RopeFromCache — the APPLY half of vLLM's rotary split.
// Upstream: rotary_embedding/base.py:160-252, common.py:145-185 @ e24d1b24fe96;
// our reference is cpu_ops.cpp RopeFromCacheKernel (:751-802).
//
// vLLM's RotaryEmbedding builds cos_sin_cache once in __init__ and the forward
// only applies it, so kRopeCosSinCache (the table, built in double) stays on the
// portable tier and this native kernel is the per-token apply. See the shader for
// why that boundary is also the right one numerically.
void RopeFromCacheKernel(Queue& queue, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  // MROPE DECLINES rather than throws. Multimodal RoPE selects a different
  // position AXIS per pair (cpu_ops.cpp:769-771 via MropeAxisForPair, mirroring
  // vLLM mrope.py), which this shader does not implement -- and throwing would
  // REMOVE a capability the portable reference tier already provides. Forwarded
  // through the provider seam, the same per-call refusal fp8 KV uses.
  if (positions.rank == 2) {
    auto next = reinterpret_cast<RopeFromCacheFn>(
        GetOpFallback(OpId::kRopeFromCache, DeviceType::kVULKAN, kNativeProviderName));
    next(queue, qs, ks, positions, cache, args);
    return;
  }

  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  if (tokens == 0 || half == 0 || (hq + hk) == 0) return;
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "vulkan rope_from_cache: positions must be i32 or i64");

  Binder bind;
  const uint32_t q_off = bind.Add(qs, "rope_from_cache: q");
  // Bindings 2/3 are declared by the shader whether or not k exists, and a
  // descriptor a shader statically uses must be valid even on the path that never
  // reads it -- so with hk == 0 they alias q and are dead. Same arrangement the
  // rmsnorm kernel already uses for its optional residual.
  const uint32_t k_off = ks != nullptr ? bind.Add(*ks, "rope_from_cache: k")
                                       : bind.Add(qs, "rope_from_cache: q");
  const uint32_t c_off = bind.Add(cache, "rope_from_cache: cos_sin_cache");
  const uint32_t p_off = bind.AddU32Only(positions, "rope_from_cache: positions");

  const uint32_t spec[5] = {DtypeCode(qs.dtype),
                            ks != nullptr ? DtypeCode(ks->dtype) : DtypeCode(qs.dtype),
                            DtypeCode(cache.dtype),
                            args.is_neox_style ? 1u : 0u,
                            positions.dtype == DType::kI64 ? 1u : 0u};
  RopeFromCacheParams p{static_cast<uint32_t>(tokens),
                        static_cast<uint32_t>(half),
                        static_cast<uint32_t>(args.rotary_dim),
                        static_cast<uint32_t>(hq),
                        static_cast<uint32_t>(hk),
                        static_cast<uint32_t>(qs.stride[0]),
                        static_cast<uint32_t>(qs.stride[1]),
                        static_cast<uint32_t>(ks != nullptr ? ks->stride[0] : 0),
                        static_cast<uint32_t>(ks != nullptr ? ks->stride[1] : 0),
                        q_off,
                        k_off,
                        c_off,
                        p_off};
  Go("vt_rope_from_cache", bind, p, FlatGroupCount(tokens * half * (hq + hk)), spec, 5);
}

// cpu_ops.cpp:2162-2176 QkvSplitKernel. Mirrors vLLM's QKVParallelLinear output
// split (qkv.split([q_size, kv_size, kv_size], dim=-1)); the three widths are
// independent because under GQA k and v are narrower than q. One invocation per
// OUTPUT element across all three destinations, so this is one dispatch.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  const int64_t t = qkv.shape[0];
  if (t == 0) return;
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  VT_CHECK(q_out.dtype == k_out.dtype && k_out.dtype == v_out.dtype,
           "vulkan qkv_split: the three destinations must share a dtype");
  Binder bind;
  const uint32_t src_off = bind.Add(qkv, "qkv_split: qkv");
  const uint32_t q_off = bind.Add(q_out, "qkv_split: q");
  const uint32_t k_off = bind.Add(k_out, "qkv_split: k");
  const uint32_t v_off = bind.Add(v_out, "qkv_split: v");
  const uint32_t spec[2] = {DtypeCode(qkv.dtype), DtypeCode(q_out.dtype)};
  QkvSplitParams p{static_cast<uint32_t>(t),     static_cast<uint32_t>(q_dim),
                   static_cast<uint32_t>(k_dim), static_cast<uint32_t>(v_dim),
                   src_off,                      q_off,
                   k_off,                        v_off};
  Go("vt_qkv_split", bind, p, FlatGroupCount(t * (q_dim + k_dim + v_dim)), spec, 2);
}

struct Registrar {
  Registrar() {
    // Same guard as the backend registrar: a Vulkan-enabled build on a host with
    // no loader or no conformant device registers nothing, so GetOp throws its
    // normal not-registered error.
    if (!VulkanContext::Available()) return;
    // static_cast against the ops.h aliases ties every kernel signature to the
    // registration contract at COMPILE time (the cpu_ops.cpp idiom).
    RegisterOp(OpId::kQkvSplit, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    RegisterOp(OpId::kMatmul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulGeneric<false>)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulGeneric<true>)));
    RegisterOp(OpId::kAdd, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastKernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::vulkan
