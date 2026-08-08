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
// SINCE THEN this TU has grown the dense path (both GEMM orientations plus the
// decode GEMV and coopmat tactics), the attention block (paged attention, the KV
// write, the QKV split, the rotary apply), the two ends of the model (embedding
// and greedy argmax), and — this row, BACKEND-VULKAN-GDN — six of the GDN /
// conv1d glue ops that Qwen3.6-27B needs.
//
// WHAT IS STILL NOT REGISTERED: the quant tier, MoE, the sampler beyond greedy
// argmax, the GDN recurrences themselves, and the ops listed in the
// BACKEND-VULKAN-GDN block comment further down. None of them THROW any more:
// since the portable reference tier landed, a missed GetOp on this
// unified-memory device installs the CPU kernel as a priority -1000 provider, so
// an unregistered op is CORRECT AND SLOW rather than fatal.
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

  // A tensor bound through the uint32_t view ONLY, for BYTE-granular reads: an
  // i8 operand (GDN's has_initial_state) may legitimately start at any byte, so
  // AddU32Only's 4-byte assertion would reject a valid tensor. The shader
  // recovers the byte with a shift, which is exact because every buffer is bound
  // WHOLE at offset 0 and the returned offset is therefore a plain byte address
  // into it (vt_common.glsl § STORAGE MODEL). Deliberately NOT VK_KHR_8bit_storage:
  // this backend does not probe for it, and requiring it would narrow the set of
  // devices the backend registers on for the sake of one boolean array.
  uint32_t AddByteView(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
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
// --- The GDN / conv1d family (BACKEND-VULKAN-GDN). Same rule as above: field
// order and types must match the GLSL push-constant blocks EXACTLY.
struct SigmoidGateParams {
  uint32_t n, a_off, g_off, o_off;
};
struct RmsNormGatedParams {
  uint32_t rows, d, x_dt, z_dt, w_dt, out_dt, sigmoid_gate, gate_group, gate_outer;
  uint32_t x_off, z_off, w_off, out_off;
  float eps;
};
struct GdnStateGatherParams {
  uint32_t rows, work_row, work_inner, cache_inner, cache_row, n_cache_rows;
  uint32_t work_dt, cache_dt, his_mode;
  uint32_t work_off, cache_off, idx_off, his_off;
};
struct GdnStateScatterParams {
  uint32_t rows, work_row, work_inner, cache_inner, cache_row, n_cache_rows;
  uint32_t work_dt, cache_dt;
  uint32_t cache_off, work_off, idx_off;
};
struct ConvUpdateParams {
  uint32_t batch, c_dim, k, width, state_len, x_rs, n_state_rows;
  uint32_t has_bias, has_idx, silu;
  uint32_t out_dt, x_dt, w_dt, bias_dt;
  uint32_t out_off, x_off, w_off, bias_off, st_off, idx_off;
};
struct GdnPostConvParams {
  uint32_t t, hk, dk, hv, dv, key_dim, value_dim, conv_dim, a_rs, b_rs;
  uint32_t conv_off, q_off, k_off, v_off, a_off, b_off;
  uint32_t g_off, beta_off, alog_off, dtb_off;
  float eps;
};
// ONE block for BOTH recurrences: vt_gdn_prefill and vt_gdn_decode share their
// step body through an include, so they must also share their push layout. The
// prefill shader ignores has_idx / n_state_rows (they are the decode state-row
// indirection) rather than each op carrying a block that drifts from the other.
struct GdnRecurrenceParams {
  uint32_t hk, dk, hv, dv, nv, ratio, has_idx, n_state_rows;
  uint32_t q_off, k_off, v_off, out_off, g_off, beta_off, state_off, meta_off;
  float scale;
};

// Vulkan only GUARANTEES 128 bytes of push-constant space (maxPushConstantsSize);
// staying inside it is what keeps this backend portable without a probe.
static_assert(sizeof(RmsParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(LayerNormParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(FcParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(PagedAttnParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(ConvUpdateParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
// The widest block in the backend at 84 bytes: the fused post-conv carries ten
// operand offsets. If it ever needs an eleventh, the step list has to move to the
// scratch buffer the way vt_fused_chain's does.
static_assert(sizeof(GdnPostConvParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(RmsNormGatedParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(GdnStateGatherParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(GdnRecurrenceParams) <= 128,
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
    // VT_VULKAN_GEMV_UNROLL=1 forces the un-unrolled body, for the same-binary A/B.
    static const uint32_t kUnroll = [] {
      const char* v = std::getenv("VT_VULKAN_GEMV_UNROLL");
      return (v != nullptr && std::strcmp(v, "1") == 0) ? 1u : 4u;
    }();
    const uint32_t spec[4] = {DtypeCode(a.dtype), DtypeCode(b.dtype),
                              DtypeCode(out.dtype), kUnroll};
    MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                   static_cast<uint32_t>(k), a_off, b_off, out_off};
    Go("vt_matmul_vec", bind, p, groups, spec, 4);
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

// ===========================================================================
// The GDN / conv1d family (BACKEND-VULKAN-GDN). Qwen3.6-27B is a GDN hybrid, so
// before this row every one of these ops fell to the PORTABLE CPU REFERENCE TIER
// on Vulkan — correct, and running on the host against shared memory.
//
// The two RECURRENCES themselves (kGdnPrefill / kGdnDecode) landed in the
// follow-up row BACKEND-VULKAN-GDN-CORE and are at the bottom of this section.
//
// WHAT IS DELIBERATELY NOT HERE, so a later row does not have to re-derive it:
//   * kRopeCosSinCache — the rotary TABLE BUILD, which constructs its angles in
//     `double` (cpu_ops.cpp RopeCosSinCacheKernel). GLSL has no f64 here and
//     emulating it would be a numerics divergence in the one place vLLM itself
//     keeps the work off the device (its RotaryEmbedding builds the cache once in
//     __init__). Leaving it on the host MIRRORS upstream; "implementing" it would
//     be a regression, and the assertion in tests/vt/test_vulkan_backend.cpp says
//     so out loud.
//   * kCausalConv1dFwd — the PREFILL conv. It is the same arithmetic as the
//     update below but its state write-back reads the OLD state row while other
//     tokens of the same sequence are still reading it, so it needs either a
//     per-(sequence, channel) serial invocation over the whole token range or a
//     buffered old row; that is a different dispatch shape, not a wider push
//     block, and it is left for a follow-up rather than guessed at here.
// ===========================================================================

// cpu_ops.cpp:2272-2279 SigmoidGateBf16Kernel. Flat, one invocation per element.
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  Binder bind;
  const uint32_t a_off = bind.Add(attn, "sigmoid_gate_bf16: attn");
  const uint32_t g_off = bind.Add(gate, "sigmoid_gate_bf16: gate");
  const uint32_t o_off = bind.Add(out, "sigmoid_gate_bf16: out");
  // Only the attention operand varies; `gate` is f32 and `out` is bf16 by the op
  // contract (src/vt/ops.cpp:3327-3334) and are compile-time constants in the
  // shader, so a violation fails the host VT_CHECK rather than silently working.
  const uint32_t spec[1] = {DtypeCode(attn.dtype)};
  SigmoidGateParams p{static_cast<uint32_t>(n), a_off, g_off, o_off};
  Go("vt_sigmoid_gate_bf16", bind, p, FlatGroupCount(n), spec, 1);
}

// cpu_ops.cpp:1210-1235 RmsNormGatedKernel. ONE WORKGROUP PER ROW (the workgroup
// tree-reduces the mean square), not FlatGroupCount — same convention as
// vt_rms_norm.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& w, const RmsNormGatedArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  if (rows == 0 || d == 0) return;
  // The rank-3 gate is a padded-row [T,Hv,D] view of the merged qkvz z slice;
  // rank-2 degenerates to group 1 with outer stride d (cpu_ops.cpp:1218-1219).
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm_gated: x");
  const uint32_t z_off = bind.Add(gate, "rmsnorm_gated: gate");
  const uint32_t w_off = bind.Add(w, "rmsnorm_gated: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm_gated: out");
  RmsNormGatedParams p{static_cast<uint32_t>(rows),
                       static_cast<uint32_t>(d),
                       DtypeCode(x.dtype),
                       DtypeCode(gate.dtype),
                       DtypeCode(w.dtype),
                       DtypeCode(out.dtype),
                       args.sigmoid_gate ? 1u : 0u,
                       static_cast<uint32_t>(gate_group),
                       static_cast<uint32_t>(gate_outer),
                       x_off,
                       z_off,
                       w_off,
                       out_off,
                       args.eps};
  Go("vt_rms_norm_gated", bind, p, static_cast<uint32_t>(rows));
}

// Shared geometry of the two state-cache ops (cpu_ops.cpp:1676-1682 and
// :1723-1727 compute it identically). `mid` is the channels/heads per row and
// `cache_row` the row's PHYSICAL width, which exceeds work_row when the conv
// state has been widened for spec-decode rollback.
struct GdnStateGeom {
  int64_t rows, work_row, work_inner, cache_inner, cache_row;
};
GdnStateGeom GdnStateGeometry(const Tensor& working, const Tensor& cache,
                              const Tensor& state_idx) {
  GdnStateGeom g{};
  g.rows = state_idx.shape[0];
  if (g.rows == 0) return g;
  g.work_inner = working.shape[working.rank - 1];
  g.cache_inner = cache.shape[cache.rank - 1];
  g.work_row = working.Numel() / g.rows;
  const int64_t mid = g.work_inner == 0 ? 0 : g.work_row / g.work_inner;
  g.cache_row = mid * g.cache_inner;
  return g;
}

// cpu_ops.cpp:1666-1707 GdnStateGatherKernel.
void GdnStateGatherKernel(Queue&, Tensor& working, const Tensor& cache,
                          const Tensor& state_idx, const Tensor* has_initial_state) {
  const GdnStateGeom g = GdnStateGeometry(working, cache, state_idx);
  if (g.rows == 0 || g.work_row == 0) return;
  Binder bind;
  const uint32_t work_off = bind.Add(working, "gdn_state_gather: working");
  const uint32_t cache_off = bind.Add(cache, "gdn_state_gather: cache");
  const uint32_t idx_off = bind.AddU32Only(state_idx, "gdn_state_gather: state_idx");
  // Binding 5 is always written: a descriptor a shader statically uses must be
  // valid even on the path that never reads it. With his_mode == 0 it aliases
  // state_idx and is dead — the same arrangement vt_rms_norm uses for its
  // optional residual.
  const uint32_t his_off =
      has_initial_state != nullptr
          ? bind.AddByteView(*has_initial_state, "gdn_state_gather: has_initial_state")
          : bind.AddByteView(state_idx, "gdn_state_gather: state_idx");
  uint32_t his_mode = 0;
  if (has_initial_state != nullptr) {
    his_mode = has_initial_state->dtype == DType::kI8 ? 1u : 2u;
  }
  GdnStateGatherParams p{static_cast<uint32_t>(g.rows),
                         static_cast<uint32_t>(g.work_row),
                         static_cast<uint32_t>(g.work_inner),
                         static_cast<uint32_t>(g.cache_inner),
                         static_cast<uint32_t>(g.cache_row),
                         static_cast<uint32_t>(cache.shape[0]),
                         DtypeCode(working.dtype),
                         DtypeCode(cache.dtype),
                         his_mode,
                         work_off,
                         cache_off,
                         idx_off,
                         his_off};
  Go("vt_gdn_state_gather", bind, p, FlatGroupCount(g.rows * g.work_row));
}

// cpu_ops.cpp:1709-1745 GdnStateScatterKernel.
void GdnStateScatterKernel(Queue&, Tensor& cache, const Tensor& working,
                           const Tensor& state_idx) {
  const GdnStateGeom g = GdnStateGeometry(working, cache, state_idx);
  if (g.rows == 0 || g.work_row == 0) return;
  Binder bind;
  const uint32_t cache_off = bind.Add(cache, "gdn_state_scatter: cache");
  const uint32_t work_off = bind.Add(working, "gdn_state_scatter: working");
  const uint32_t idx_off = bind.AddU32Only(state_idx, "gdn_state_scatter: state_idx");
  GdnStateScatterParams p{static_cast<uint32_t>(g.rows),
                          static_cast<uint32_t>(g.work_row),
                          static_cast<uint32_t>(g.work_inner),
                          static_cast<uint32_t>(g.cache_inner),
                          static_cast<uint32_t>(g.cache_row),
                          static_cast<uint32_t>(cache.shape[0]),
                          DtypeCode(working.dtype),
                          DtypeCode(cache.dtype),
                          cache_off,
                          work_off,
                          idx_off};
  Go("vt_gdn_state_scatter", bind, p, FlatGroupCount(g.rows * g.work_row));
}

// cpu_ops.cpp:1081-1127 CausalConv1dUpdateKernel. One invocation per
// (token, channel) — the CPU kernel's own row-chunking unit, and what makes the
// read-old-then-roll safe with no barrier.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  if (batch == 0 || c_dim == 0) return;
  // bf16 conv_state is a CUDA-only extension of the contract
  // (src/vt/ops.cpp CheckConvCommon), so on this backend the state is f32 and the
  // shader reads it through the 32-bit view directly.
  VT_CHECK(conv_state.dtype == DType::kF32,
           "vulkan causal_conv1d_update: conv_state must be f32");
  Binder bind;
  const uint32_t out_off = bind.Add(out, "causal_conv1d_update: out");
  const uint32_t x_off = bind.Add(x, "causal_conv1d_update: x");
  const uint32_t w_off = bind.Add(w, "causal_conv1d_update: weight");
  // Bindings 6/7 alias the weight when there is no bias; see the note above.
  const uint32_t bias_off = bias != nullptr ? bind.Add(*bias, "causal_conv1d_update: bias")
                                            : bind.Add(w, "causal_conv1d_update: weight");
  const uint32_t st_off = bind.AddU32Only(conv_state, "causal_conv1d_update: conv_state");
  const uint32_t idx_off =
      conv_state_indices != nullptr
          ? bind.AddU32Only(*conv_state_indices, "causal_conv1d_update: conv_state_indices")
          : bind.AddU32Only(conv_state, "causal_conv1d_update: conv_state");
  ConvUpdateParams p{static_cast<uint32_t>(batch),
                     static_cast<uint32_t>(c_dim),
                     static_cast<uint32_t>(k),
                     static_cast<uint32_t>(k - 1),
                     static_cast<uint32_t>(conv_state.shape[2]),
                     static_cast<uint32_t>(x.stride[0]),
                     static_cast<uint32_t>(conv_state.shape[0]),
                     bias != nullptr ? 1u : 0u,
                     conv_state_indices != nullptr ? 1u : 0u,
                     args.silu_activation ? 1u : 0u,
                     DtypeCode(out.dtype),
                     DtypeCode(x.dtype),
                     DtypeCode(w.dtype),
                     bias != nullptr ? DtypeCode(bias->dtype) : DtypeCode(w.dtype),
                     out_off,
                     x_off,
                     w_off,
                     bias_off,
                     st_off,
                     idx_off};
  Go("vt_causal_conv1d_update", bind, p, FlatGroupCount(batch * c_dim));
}

// cpu_ops.cpp:2337-2417 GdnPostConvKernel. ONE WORKGROUP PER (token, head slot)
// over Hk + Hv slots — upstream's own (L, H+HV) grid — not FlatGroupCount: the
// q/k slots tree-reduce an L2 norm across the workgroup's lanes.
void GdnPostConvKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                       Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                       const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                       const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0 || hk + hv == 0) return;
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  Binder bind;
  const uint32_t conv_off = bind.Add(conv, "gdn_post_conv: conv");
  const uint32_t q_off = bind.Add(q_out, "gdn_post_conv: q_out");
  const uint32_t k_off = bind.Add(k_out, "gdn_post_conv: k_out");
  const uint32_t v_off = bind.Add(v_out, "gdn_post_conv: v_out");
  const uint32_t a_off = bind.Add(araw, "gdn_post_conv: araw");
  const uint32_t b_off = bind.Add(braw, "gdn_post_conv: braw");
  // f32 BY CONTRACT (src/vt/ops.cpp:3459-3463), so one binding each rather than a
  // dtype-erased pair whose 16-bit half could never be taken.
  const uint32_t g_off = bind.AddU32Only(g_out, "gdn_post_conv: g_out");
  const uint32_t beta_off = bind.AddU32Only(beta_out, "gdn_post_conv: beta_out");
  const uint32_t alog_off = bind.AddU32Only(a_log, "gdn_post_conv: a_log");
  const uint32_t dtb_off = bind.AddU32Only(dt_bias, "gdn_post_conv: dt_bias");
  // Ascending constantID order: conv dtype, the shared q/k/v dtype, the shared
  // araw/braw dtype.
  const uint32_t spec[3] = {DtypeCode(conv.dtype), DtypeCode(q_out.dtype),
                            DtypeCode(araw.dtype)};
  GdnPostConvParams p{static_cast<uint32_t>(t),
                      static_cast<uint32_t>(hk),
                      static_cast<uint32_t>(dk),
                      static_cast<uint32_t>(hv),
                      static_cast<uint32_t>(dv),
                      static_cast<uint32_t>(key_dim),
                      static_cast<uint32_t>(value_dim),
                      static_cast<uint32_t>(2 * key_dim + value_dim),
                      static_cast<uint32_t>(araw.stride[0]),
                      static_cast<uint32_t>(braw.stride[0]),
                      conv_off,
                      q_off,
                      k_off,
                      v_off,
                      a_off,
                      b_off,
                      g_off,
                      beta_off,
                      alog_off,
                      dtb_off,
                      args.eps};
  Go("vt_gdn_post_conv", bind, p, static_cast<uint32_t>(t * (hk + hv)), spec, 3);
}

// ---------------------------------------------------------------------------
// The two GATED-DELTA RECURRENCES (BACKEND-VULKAN-GDN-CORE). These are not glue:
// they ARE Qwen3.6's linear-attention core, and with the glue above already
// native they were the whole of what the model still ran on the host — a 512
// token prompt spent ~280 s in kGdnPrefill on the reference tier.
//
// The shaders (src/vt/vulkan/shaders/vt_gdn_prefill.comp, vt_gdn_decode.comp and
// the shared vt_gdn_recurrence.glsl) carry the port provenance and the tile
// geometry. What the HOST has to get right is only the grid and the decline.
// ---------------------------------------------------------------------------

// Must equal VT_GDN_BV / VT_GDN_MAX_DK in vt_gdn_recurrence.glsl. Duplicated
// rather than shared because the shader constants are in GLSL and the SPIR-V is
// committed, so nothing can compute one from the other — the gate in
// tests/vt/test_vulkan_backend.cpp exercises a Dv that is NOT a multiple of the
// tile so a drift shows up as wrong numbers there rather than in a model run.
constexpr int64_t kGdnTileRows = 16;
constexpr int64_t kGdnMaxDk = 128;

// Shared grid + binding setup for the two recurrences. Returns false when this
// backend cannot serve the shape and the caller must decline to the next
// provider.
bool GdnRecurrenceCommon(const Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         const Tensor& state, Binder& bind, GdnRecurrenceParams& p,
                         uint32_t spec[2], float scale) {
  const int64_t hv = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  // PER-CALL REFUSAL rather than a throw, the same seam vt_paged_attn uses for an
  // fp8 KV cache: declining forwards to the portable reference tier, which is
  // correct for every shape, instead of REMOVING a capability the backend already
  // had. Two reasons to decline.
  //   * Dk beyond the shared tile's compile-time extent. The tile has to be sized
  //     at compile time against Vulkan's GUARANTEED 16 KB of shared memory, and
  //     the real gate dim is 128.
  //   * q/k/v disagreeing on dtype. One specialization constant covers the three
  //     (they come out of one GdnPostConv dispatch and always agree), and CUDA
  //     asserts the same thing (cuda_gdn.cu:2577).
  if (dk > kGdnMaxDk || dk <= 0) return false;
  if (k.dtype != q_in.dtype || v.dtype != q_in.dtype) return false;
  const uint32_t q_off = bind.Add(q_in, "gdn recurrence: q");
  const uint32_t k_off = bind.Add(k, "gdn recurrence: k");
  const uint32_t v_off = bind.Add(v, "gdn recurrence: v");
  const uint32_t out_off = bind.Add(out, "gdn recurrence: out");
  // g, beta and the state are f32 by the op contract on this device
  // (src/vt/ops.cpp:1629-1643 — a compressed state is CUDA-only), so one binding
  // each rather than a dtype-erased pair whose 16-bit half could never be taken.
  const uint32_t g_off = bind.AddU32Only(g, "gdn recurrence: g");
  const uint32_t beta_off = bind.AddU32Only(beta, "gdn recurrence: beta");
  const uint32_t state_off = bind.AddU32Only(state, "gdn recurrence: state");
  spec[0] = DtypeCode(q_in.dtype);
  spec[1] = DtypeCode(out.dtype);
  p.hk = static_cast<uint32_t>(hk);
  p.dk = static_cast<uint32_t>(dk);
  p.hv = static_cast<uint32_t>(hv);
  p.dv = static_cast<uint32_t>(dv);
  p.nv = static_cast<uint32_t>((dv + kGdnTileRows - 1) / kGdnTileRows);
  p.ratio = static_cast<uint32_t>(hv / hk);
  p.has_idx = 0;
  p.n_state_rows = static_cast<uint32_t>(state.shape[0]);
  p.q_off = q_off;
  p.k_off = k_off;
  p.v_off = v_off;
  p.out_off = out_off;
  p.g_off = g_off;
  p.beta_off = beta_off;
  p.state_off = state_off;
  p.meta_off = 0;
  p.scale = scale;
  return true;
}

// cpu_ops.cpp:1331-1366 GdnPrefillKernel. ONE WORKGROUP PER
// (sequence, value-head, value-tile) — the CPU kernel's own (SEQUENCE,
// VALUE-HEAD) chunking plus the value-row tile our CUDA kernel already uses as
// its grid.x (cuda_gdn.cu:2421). NOT FlatGroupCount: the whole workgroup
// cooperates on one tile, and the sequence stays sequential inside it.
void GdnPrefillKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                      const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                      const Tensor& query_start_loc, const GdnArgs& args) {
  const int64_t n = state.shape[0], hv = state.shape[1], dv = state.shape[2];
  if (n == 0 || hv == 0 || dv == 0) return;
  Binder bind;
  GdnRecurrenceParams p{};
  uint32_t spec[2] = {0, 0};
  if (!GdnRecurrenceCommon(out, q_in, k, v, g, beta, state, bind, p, spec, args.scale)) {
    auto next = reinterpret_cast<GdnPrefillFn>(
        GetOpFallback(OpId::kGdnPrefill, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, q_in, k, v, g, beta, state, query_start_loc, args);
    return;
  }
  p.meta_off = bind.AddU32Only(query_start_loc, "gdn_prefill: query_start_loc");
  Go("vt_gdn_prefill", bind, p, static_cast<uint32_t>(n * hv * p.nv), spec, 2);
}

// cpu_ops.cpp:1368-1396 GdnDecodeKernel, one step per batch token. Same grid with
// the sequence axis replaced by the batch — cuda_gdn.cu:2513's
// (NV, n*Hv) flattened.
void GdnDecodeKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                     const Tensor* state_idx, const GdnArgs& args) {
  const int64_t batch = q_in.shape[0], hv = state.shape[1], dv = state.shape[2];
  if (batch == 0 || hv == 0 || dv == 0) return;
  Binder bind;
  GdnRecurrenceParams p{};
  uint32_t spec[2] = {0, 0};
  if (!GdnRecurrenceCommon(out, q_in, k, v, g, beta, state, bind, p, spec, args.scale)) {
    auto next = reinterpret_cast<GdnDecodeFn>(
        GetOpFallback(OpId::kGdnDecode, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, q_in, k, v, g, beta, state, state_idx, args);
    return;
  }
  // Binding 11 is always written: a descriptor a shader statically uses must be
  // valid even on the path that never reads it. With has_idx == 0 it aliases the
  // state buffer and is dead.
  if (state_idx != nullptr) {
    p.has_idx = 1;
    p.meta_off = bind.AddU32Only(*state_idx, "gdn_decode: state_idx");
  } else {
    p.meta_off = bind.AddU32Only(state, "gdn_decode: state");
  }
  Go("vt_gdn_decode", bind, p, static_cast<uint32_t>(batch * hv * p.nv), spec, 2);
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
    // BACKEND-VULKAN-GDN: the GDN glue family. kCausalConv1dFwd (the prefill
    // conv) and kRopeCosSinCache (the double-precision rotary table, deliberately
    // host-side) stay on the portable reference tier; see the block comment above
    // these kernels.
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnStateGatherFn>(&GdnStateGatherKernel)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnStateScatterFn>(&GdnStateScatterKernel)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    RegisterOp(OpId::kGdnPostConv, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    // BACKEND-VULKAN-GDN-CORE: the two recurrences.
    RegisterOp(OpId::kGdnPrefill, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::vulkan
