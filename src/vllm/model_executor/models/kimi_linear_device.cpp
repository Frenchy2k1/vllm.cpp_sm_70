// Kimi-Linear W7 — the real DBuf-RESIDENT device COMPUTE forward. This TU is the
// device-compute lane the W6 SEAM (kimi_linear.cpp) documented as its plan: a
// pooled-DBuf forward that routes the whole 27-layer KDA/NoPE-MLA + 256-expert-MoE
// hybrid through the SHARED vt:: device ops, gated on CPU against the W2 host f32
// reference (kimi_linear_forward.cpp). Because the CPU backend executes the SAME
// vt:: dispatch (a pooled DBuf is a device buffer on CPU, ResidentWeight aliases
// the host weight bytes on CPU), a tight CPU match to the reference is a REAL proof
// of the residual-stream / vt-op / MoE-routing / on-device-logits WIRING the GPU
// will run — only the GPU numerics (bf16 activations, the GDN Triton-AOT cubins,
// the paged het-KV, the grouped-MoE slabs) + the e2e SACRED golden stay pending
// (box down). Honest labeling: NOT a DONE claim for the GPU-unverified device path.
//
// ─── DEVICE vs HOST-FALLBACK, precisely (each op CPU+CUDA-registered) ──────────
// ON DEVICE (genuine vt:: dispatch on pooled DBufs, f32 activations to match the
// f32 reference tightly):
//   embed                vt::Embedding
//   add+RMSNorm glue      vt::FusedChain(kFusedAddRmsNormStd)  (the fusion seam)
//   every projection      vt::MatmulBT (host weights are torch [out,in] = [N,K])
//   KDA short convs       vt::CausalConv1dFwd (silu, fresh zero conv-state)
//   KDA q/k L2-norm       vt::L2Norm
//   KDA output gated-norm vt::RmsNormGated (sigmoid gate)
//   MoE router            vt::MoeRouterTopK (sigmoid noaux_tc, group 1/1, bias, scale)
//   SwiGLU activation     vt::MoeSiluMul (dense MLP + each expert + shared expert)
//   MoE weighted combine  vt::MoeCombine (+ shared term)
//   lm_head               vt::MatmulBT
// HOST-FALLBACK ISLANDS (the W7-speed residuals — no portable device op yet):
//   (1) KDA per-k-channel gated-delta RECURRENCE + its decay gate g =
//       -exp(A_log)*softplus(f_b(f_a(x))+dt_bias) + beta=sigmoid(b_proj). vt::Gdn
//       Decode/GdnPrefill carry only a per-HEAD scalar decay g[T,Hv] (ops.h), so
//       they CANNOT express KDA's per-channel g[T,H,D]; the recurrence is computed
//       on host from device-resident q/k/v/g1/beta and uploaded (kimi_kda refs).
//   (2) NoPE-MLA attention CORE (causal scores/softmax/weighted-V). The device
//       path is mla::ForwardMlaAttentionBlock over the runner's PAGED het-KV cache
//       + the load-time W_UK/W_UV absorption + TritonMLAImpl — the born-on-runner
//       residual. This seam keeps every MLA projection + kv_a_layernorm ON DEVICE
//       and computes only the softmax core on host (identical materialized-MHA math
//       as the W2 reference, NoPE so no RoPE).
//
// Grounding: the reuse-wiring plan authored in kimi_linear.cpp; the per-op numerics
// in kimi_linear_forward.cpp (the W2 reference) + kimi_kda.{h,cpp}. Mirrors the
// deepseek_v2.cpp device forward structure (ForwardBody/RunLayer/MoeBlock/DenseMlp
// residual-stream + WrapDeviceLogits).
#include "vllm/model_executor/models/kimi_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // dense_attn::{Dev,DBuf,MakeTensor}
#include "vllm/model_executor/models/device_pool.h"        // Pool()
#include "vllm/model_executor/models/kimi_kda.h"
#include "vllm/platforms/interface.h"                       // platforms::GetPlatform (is_cpu)
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // vt::kFusedAddRmsNormStd

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::DType;
using vt::Tensor;

// VT_FUSED_CHAIN_ADOPT gate, mirroring dense_attn::FusedChainAdoptEnabled (a
// local copy so this TU need not pull the heavy dense_attn_block.h). Default ON:
// the residual add+RMSNorm goes through the vt::FusedChain catalog seam; =0 falls
// back to the bit-identical residual RmsNorm overload for an A/B.
bool FusedGlue() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ─── STRICT-path numerics knobs (W7-speed lane) ────────────────────────────────
// The correctness vehicle is MORE precise than vLLM (f32 residual stream + f64 host
// islands), which flips near-tie argmaxes where vLLM's deterministic bf16 top-1 has
// a small margin (spec §13 root cause). These two knobs mirror vLLM's bf16 compute
// regime so the near-ties become token-exact. Both default OFF → the f32 vehicle is
// byte-identical (the CPU tiny-config gate stays 13/13·656); flip ON only for the
// full-model on-box token gate vs the STRICT golden.
//
// (1) VT_KIMI_BF16_RESIDUAL — carry the residual stream in bf16 like vLLM. vLLM's
// fused_add_rms_norm stores `residual` as bf16 and `hidden_states`/block-outputs are
// bf16, while it computes the RMSNorm variance over the f32 pre-store sum
// (layernorm.py / fused_add_rms_norm.cu). We keep f32 STORAGE (so the two host-fallback
// islands still consume f32) but round the VALUE to bf16 precision at exactly vLLM's
// rounding points: the embed output, each block output before it re-enters the add,
// and the residual AFTER each add (so the norm still sees the f32 sum — byte-matching
// vLLM's fused-add-rms-norm order). 54 bf16 roundings across 27 layers × 2 norms that
// vLLM does and our f32 vehicle does not.
bool Bf16Residual() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_BF16_RESIDUAL");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (2) VT_KIMI_BF16_ISLANDS — round the host-fallback island INPUTS (KDA recurrence
// q/k/v/g1/beta, MLA softmax q/kv/kpe) to bf16 precision before the f64 recurrence.
// vLLM feeds bf16 activations into the GDN Triton-AOT / FA2 kernels; our islands
// download f32 (bf16-precision projection outputs stored to f32) and recompute in f64.
// Rounding the inputs to bf16 moves the island toward vLLM's kernel precision without a
// new device kernel (the true fix — the device GDN per-channel-decay recurrence + the
// paged FA2 MLA — is the named W7-speed residual, spec §13).
bool Bf16Islands() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_BF16_ISLANDS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Round an f32 value to bf16 precision (round-to-nearest-even), matching torch/vLLM's
// bf16 cast (and vt::CastBf16). Truncate-with-RNE-bias on the top 16 bits; qNaN-safe.
inline float ToBf16Rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    u |= 0x00400000u;  // qNaN
  } else {
    u += 0x00007fffu + ((u >> 16) & 1u);  // RNE rounding bias
  }
  u &= 0xffff0000u;
  float r;
  std::memcpy(&r, &u, sizeof(r));
  return r;
}
inline void RoundHostBf16(std::vector<float>& v) {
  if (!Bf16Islands()) return;
  for (float& x : v) x = ToBf16Rne(x);
}

// (3) VT_KIMI_ISLAND_F32ACC — compute the host-fallback island recurrence/softmax in
// f32 accumulation (not f64), matching vLLM's GDN Triton / FA2 kernels (bf16 I/O, f32
// accumulation). Our island defaults to f64 (MORE precise than vLLM); rounding each
// accumulation step to f32 mirrors the device kernel's rounding. Combined with
// VT_KIMI_BF16_ISLANDS (bf16 I/O), this is the closest host approximation of vLLM's
// actual kernel numerics without a new device kernel (the named W7-speed residual).
bool IslandF32Acc() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_ISLAND_F32ACC");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (4) VT_KIMI_DEVICE_KDA — run the KDA per-k-channel gated-delta RECURRENCE through the
// device op vt::KdaGatedDeltaRule (the net-new per-channel-decay GDN kernel, cuda_gdn.cu
// KdaScanKernel) instead of the f64 host recompute. This is the principled path to STRICT
// AND the speed lever (spec §14): the recurrence runs vLLM's actual f32-on-bf16 arithmetic
// (FLA fused_recurrent_gated_delta_rule_fwd_kernel IS_KDA=True) on device rather than a
// host f64 recompute that is MORE precise than vLLM and coin-flips near-ties. The decay
// gate `g = -exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` and beta = sigmoid(b) stay host
// (elementwise, numerically stable — the numerically-sensitive object is the recurrence).
// Default OFF (parity-enabler: flip ON only with the token gate green). Independent of the
// bf16-precision knobs (BF16_ISLANDS still rounds the gate inputs when both are set).
bool DeviceKda() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_KDA");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (4b) VT_KIMI_DEVICE_KDA_CHUNK — process the PROMPT-length KDA with the CHUNKED
// prefill kernel family (vt::KdaChunkPrefill: the vendored FLA Triton-AOT cubins
// kda_gate_cumsum -> kkt -> solve_tril -> recompute_w_u -> chunk_delta_h ->
// chunk_gla_o) instead of the RECURRENT form, exactly as vLLM
// (kimi_gdn_linear_attn.py:141 chunk_kda_with_fused_gate; decode stays recurrent).
// This is the spec §15 STRICT residual (c): vLLM processes the prompt chunked, we
// still recur — a different reduction ORDER coin-flips the p7 near-tie. Requires
// VT_KIMI_DEVICE_KDA=1 (the recurrence is the T==1 / fallback path). Default OFF
// (parity-enabler; flip ON only with the token gate green). The chunk op fuses the
// gate on-device (raw g1 + a_log + dt_bias), so unlike the recurrent branch nothing
// is host-rounded — the chunk kernels carry vLLM's exact bf16 path.
bool DeviceKdaChunk() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_KDA_CHUNK");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (5) VT_KIMI_DEVICE_MLA — run the NoPE-MLA attention CORE (causal scores/softmax/
// weighted-V) through the shared device op vt::Attention instead of the f64 host
// softmax island. This is the MLA twin of VT_KIMI_DEVICE_KDA (spec §14/§15 residual
// (d)): the host island runs an f64 softmax that is MORE precise than vLLM's FA2 and
// coin-flips near-ties, whereas vt::Attention runs the f32 max-subtracted online-
// softmax accumulation vLLM's kernels use (cpu_ops AttentionKernel / cuda_ops). The
// MLA attention has asymmetric head dims (qk = qk_nope+qk_rope = 192, v = 128), which
// vt::Attention (single head-dim D for q/k/v) does not express directly, so the value
// is PADDED to the qk head-dim with zeros (weighted-sum over the zero tail = 0) and the
// out[:, :, :v] slice is the true attention core — bit-exact to the unpadded math since
// softmax weights depend only on q·k. Requires VT_KIMI_DEVICE_COMPUTE=1. Independent of
// the bf16-precision knobs (BF16_ISLANDS still rounds the q/kv/kpe inputs when both are
// set — applied on device before the attention).
//
// ★ MEASURED NEGATIVE (2026-08-07, GB10 full 48.9B gate, spec §16) — kept as a
// documented-negative A/B knob, DEFAULT OFF. On the device-KDA best config
// (VT_KIMI_DEVICE_KDA=1), adding VT_KIMI_DEVICE_MLA REGRESSES 122→109/128 AND slows
// 4.24→3.89 tok/s: (1) vt::Attention's f32 online max-subtracted softmax is NOT vLLM's
// FA2 reduction ORDER — it is a DIFFERENT approximation, so it coin-flips near-ties (it
// BREAKS p3 16/16→3/16 into the `163586×` repeat loop while p7 stays diverged), the same
// §14 plateau class; (2) the per-(t,h) key/value build copies + the 192-dim pad-V waste
// ADD overhead to the O(n²) recompute path. The principled MLA-half STRICT lever is
// vLLM's ACTUAL FA2 via paged mla::ForwardMlaAttentionBlock (residual d, coupled with
// paged-incremental decode e), NOT this softmax approximation — this negative is the
// measurement that proves the approximation is not enough.
bool DeviceMla() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_MLA");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// Round a running f64 accumulator to f32 precision when the knob is on (identity else).
inline double AccR(double x) {
  static const bool f32 = IslandF32Acc();
  return f32 ? static_cast<double>(static_cast<float>(x)) : x;
}

// In-place round an f32 device buffer to bf16 precision (f32→bf16→f32, on-device, no
// download). The VALUE becomes bf16-exact so the RMSNorm variance and the next residual
// add see the same bf16 numbers vLLM does; the STORAGE stays f32 (the islands read f32).
void RoundDevBf16(const Dev& d, DBuf& x) {
  Tensor xt = x.t();
  std::vector<int64_t> shape(xt.shape, xt.shape + xt.rank);
  DBuf b(d, DType::kBF16, shape);
  vt::CastBf16(d.q, b.t(), xt);
  Tensor out = x.t();
  vt::CastF32(d.q, out, b.t());
}

// Device-resident weight view. On CPU this ALIASES the host f32 bytes exactly as
// dense_attn::ResidentWeight does for a CPU device (host-pointer aliasing is a CPU
// property); the CUDA staging over materialized OwnedTensors is the born-on-runner
// residual (Kimi's device weights are not materialized as OwnedTensors yet — see
// kimi_linear.h). The device-compute lane is CPU-reachable in this brick.
inline Tensor WF32(const Dev& d, const std::vector<float>& v,
                   const std::vector<int64_t>& shape) {
  return MakeTensor(const_cast<float*>(v.data()), DType::kF32, d.q.device, shape);
}

// ─── bf16-RESIDENT weight view + cast-GEMM (§13) ───────────────────────────────
// Device-resident bf16 weight view over an OwnedTensor (mirror laguna.cpp:125-139
// LagunaResidentBf16W / dense_attn::ResidentWeight). CPU: alias the host bf16 bytes
// (host-pointer aliasing is a CPU property). CUDA: return the d_dev copy — pre-staged
// at load (StageKimiResidentBf16 + ReleaseHost, the pool-math path) or, if absent,
// uploaded ONCE here (cudaMalloc + one H2D, byte-exact — no ATS penalty).
inline Tensor ResidentBf16W(const Dev& d, const OwnedTensor& w,
                            const std::vector<int64_t>& shape) {
  if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu())
    return MakeTensor(const_cast<uint8_t*>(w.bytes.data()), w.dtype, d.q.device, shape);
  if (!w.d_dev) {
    VT_CHECK(w.HasHostBytes(),
             "kimi resident: bf16 weight host bytes released before device staging");
    const size_t nb = w.bytes.size();
    void* p = d.b.Alloc(nb);
    d.b.Copy(d.q, p, w.bytes.data(), nb);
    vt::Backend* bk = &d.b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return MakeTensor(w.d_dev.get(), w.dtype, d.q.device, shape);
}

// bf16 cast-GEMM: out[T,N] f32 = cast(act[T,K] -> bf16) . w_bf16[N,K]^T (mirror
// laguna.cpp:1939-1946 GemmBf16Into). The (bf16,bf16)->f32 MatmulBT IS vLLM's
// projection numerics (cuda_matmul.cu:3); the residual stream stays f32.
void GemmBf16(const Dev& d, Tensor& out, const Tensor& act, const OwnedTensor& w,
              int64_t N, int64_t K) {
  const int64_t T = act.shape[0];
  DBuf ab(d, DType::kBF16, {T, K});
  vt::CastBf16(d.q, ab.t(), act);
  Tensor wt = ResidentBf16W(d, w, {N, K});
  vt::MatmulBT(d.q, out, ab.t(), wt);
}

// Fused residual add + standard RMSNorm: res += x; out = rmsnorm(res) * w. The
// canonical add+RMSNorm glue seam (identical residual accumulation to the W2
// reference's single-`h` stream — see the deepseek_v2.cpp RunLayer derivation).
void AddRmsNorm(const Dev& d, DBuf& out, const Tensor& x, const Tensor& w, DBuf& res,
                float eps) {
  if (FusedGlue()) {
    vt::FusedChain(d.q, out.t(), x, w, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, out.t(), x, w, vt::RmsNormArgs{eps, false}, &res.t());
  }
}

// silu(gate@x) * (up@x) -> down@(...) — a gated SwiGLU MLP via the shared vt:: ops
// on separate gate/up/down host weights (torch [out,in]). MoeSiluMul (not
// SiluAndMul) is used so the merged-GEMM checker is not tripped: the fused
// MlpGateUp merged-GEMM arm needs a fused gate_up weight (a loader residual), so
// the CPU device gate uses the separate-GEMM + MoeSiluMul equivalent.
DBuf SwiGluDevice(const Dev& d, const std::vector<float>& gate,
                  const std::vector<float>& up, const std::vector<float>& down,
                  const Tensor& dh, int64_t H, int64_t I, int64_t T) {
  DBuf dg(d, DType::kF32, {T, I});
  vt::MatmulBT(d.q, dg.t(), dh, WF32(d, gate, {I, H}));
  DBuf du(d, DType::kF32, {T, I});
  vt::MatmulBT(d.q, du.t(), dh, WF32(d, up, {I, H}));
  DBuf da(d, DType::kF32, {T, I});
  vt::MoeSiluMul(d.q, da.t(), dg.t(), du.t());
  DBuf out(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, out.t(), da.t(), WF32(d, down, {H, I}));
  return out;
}

// One depthwise causal short conv (silu), fresh zero conv-state (single sequence).
// Mirrors the qwen3_5.cpp GDN conv call (qwen3_5.cpp:3032-3040): f32 conv_state
// [1,C,K-1] + query_start_loc {0,T} + has_initial_state {0}. weight is [C,K].
DBuf ConvSilu(const Dev& d, const Tensor& x, const std::vector<float>& weight,
              int64_t T, int64_t C, int64_t K) {
  DBuf out(d, DType::kF32, {T, C});
  DBuf state(d, DType::kF32, {1, C, K - 1});
  state.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  vt::CausalConv1dFwd(d.q, out.t(), x, WF32(d, weight, {C, K}), nullptr, state.t(),
                      dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
  return out;
}

// ── HOST-FALLBACK ISLAND: KDA decay gate + per-k-channel gated-delta RECURRENCE.
// vt::GdnDecode carries only a per-HEAD scalar decay (ops.h g/beta[T,Hv]); KDA's
// decay is per-k-channel, so the recurrence is computed on host from the device-
// resident q_n/k_n/v/g1/beta via the landed kimi_kda refs + the reference recurrence
// (kimi_linear_forward.cpp:142-183), then uploaded. THE W7-speed residual. Shared by
// the f32 (KdaLayerDevice) and bf16 (KdaLayerDeviceBf16) paths — the recurrence itself
// runs the IDENTICAL host code on both; only the GEMMs feeding it differ (f32 alias vs
// bf16 cast-GEMM), so extracting it keeps the two paths byte-identical here.
DBuf KdaRecurrenceIsland(const Dev& d, DBuf& qn, DBuf& kn, DBuf& vc, DBuf& g1,
                         DBuf& braw, const std::vector<float>& a_log,
                         const std::vector<float>& dt_bias, const KimiLinearParams& p,
                         int64_t T) {
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;

  // ── DEVICE RECURRENCE (VT_KIMI_DEVICE_KDA): run the per-k-channel gated-delta
  // recurrence on device via vt::KdaGatedDeltaRule (KdaScanKernel), vLLM's actual
  // f32-on-bf16 arithmetic, instead of the host f64 recompute below. q_n/k_n/v are
  // ALREADY device-resident; only the elementwise gate (KdaDecayGate) + beta = sigmoid(b)
  // are computed on host and uploaded (small, numerically stable). Fresh zero state,
  // single sequence, qsl=[0,T] — the stateless full-sequence recurrence the island needs.
  // ── CHUNK-PREFILL (VT_KIMI_DEVICE_KDA_CHUNK): route the whole prompt through the
  // chunked FLA Triton-AOT cubins (vt::KdaChunkPrefill) — vLLM's ACTUAL prefill path
  // — instead of the recurrence. The op fuses the gate on-device from the RAW g1
  // projection + a_log + dt_bias (no host gate compute, no bf16 island rounding); only
  // beta = sigmoid(braw) is the tiny host elementwise. T==1 (a single token) keeps the
  // recurrence below (the op itself also falls back for T==1). Spec §17.
  if (DeviceKda() && DeviceKdaChunk() && T > 1) {
    std::vector<float> hbraw(static_cast<size_t>(T) * nh), hbeta(static_cast<size_t>(T) * nh);
    braw.Download(d, hbraw.data());
    for (size_t i = 0; i < hbeta.size(); ++i) hbeta[i] = static_cast<float>(Sigmoid(hbraw[i]));
    DBuf dbeta(d, DType::kF32, {T, nh}, hbeta.data());
    DBuf da_log(d, DType::kF32, {nh}, a_log.data());
    DBuf ddt(d, DType::kF32, {static_cast<int64_t>(dt_bias.size())},
             dt_bias.empty() ? nullptr : dt_bias.data());
    DBuf dstate(d, DType::kF32, {1, nh, hd, hd});
    dstate.Zero(d);
    DBuf dcore(d, DType::kF32, {T, proj});
    const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
    DBuf dqsl(d, DType::kI32, {2}, qsl);
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor vc3 = MakeTensor(vc.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor gr3 = MakeTensor(g1.ptr(), DType::kF32, d.q.device, {T, nh, hd});  // RAW gate proj
    Tensor out3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));
    vt::KdaChunkPrefill(d.q, out3, qn3, kn3, vc3, gr3, dbeta.t(), da_log.t(), ddt.t(), dstate.t(),
                        dqsl.t(), vt::GdnArgs{scale});
    return dcore;
  }

  if (DeviceKda()) {
    std::vector<float> dhg1(static_cast<size_t>(T) * proj),
        dhbraw(static_cast<size_t>(T) * nh);
    g1.Download(d, dhg1.data());
    braw.Download(d, dhbraw.data());
    RoundHostBf16(dhg1);   // honor BF16_ISLANDS on the gate inputs
    RoundHostBf16(dhbraw);
    const std::vector<float> gch =
        kimi_kda::KdaDecayGate(dhg1, a_log, dt_bias, T, nh, hd);  // [T,nh,hd] per-channel
    std::vector<float> hbeta(static_cast<size_t>(T) * nh);
    for (size_t i = 0; i < hbeta.size(); ++i)
      hbeta[i] = static_cast<float>(Sigmoid(dhbraw[i]));
    DBuf dg(d, DType::kF32, {T, nh, hd}, gch.data());
    DBuf dbeta(d, DType::kF32, {T, nh}, hbeta.data());
    DBuf dstate(d, DType::kF32, {1, nh, hd, hd});
    dstate.Zero(d);
    DBuf dcore(d, DType::kF32, {T, proj});
    const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
    DBuf dqsl(d, DType::kI32, {2}, qsl);
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor vc3 = MakeTensor(vc.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor out3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));
    vt::KdaGatedDeltaRule(d.q, out3, qn3, kn3, vc3, dg.t(), dbeta.t(), dstate.t(), dqsl.t(),
                          vt::GdnArgs{scale});
    return dcore;
  }

  std::vector<float> hqn(static_cast<size_t>(T) * proj), hkn(hqn.size()),
      hv(hqn.size()), hg1(hqn.size()), hbraw(static_cast<size_t>(T) * nh);
  qn.Download(d, hqn.data());
  kn.Download(d, hkn.data());
  vc.Download(d, hv.data());
  g1.Download(d, hg1.data());
  braw.Download(d, hbraw.data());
  // VT_KIMI_BF16_ISLANDS: feed bf16-precision inputs to the recurrence (like vLLM's
  // GDN kernel), keeping the f64 accumulation. No-op when the knob is off.
  RoundHostBf16(hqn);
  RoundHostBf16(hkn);
  RoundHostBf16(hv);
  RoundHostBf16(hg1);
  RoundHostBf16(hbraw);

  const std::vector<float> g =
      kimi_kda::KdaDecayGate(hg1, a_log, dt_bias, T, nh, hd);  // [T,nh,hd]
  const double scale = std::pow(static_cast<double>(hd), -0.5);
  std::vector<double> S(static_cast<size_t>(nh) * hd * hd, 0.0);
  std::vector<float> core(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<double> u(static_cast<size_t>(hd));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < nh; ++h) {
      const int64_t base = t * proj + h * hd;
      const float* qnp = &hqn[static_cast<size_t>(base)];
      const float* knp = &hkn[static_cast<size_t>(base)];
      const float* vvp = &hv[static_cast<size_t>(base)];
      const float* gh = &g[static_cast<size_t>(base)];
      const double b = Sigmoid(hbraw[static_cast<size_t>(t * nh + h)]);
      double* Sp = &S[static_cast<size_t>(h) * hd * hd];
      for (int64_t vd = 0; vd < hd; ++vd) {
        double* Sr = &Sp[vd * hd];
        for (int64_t k = 0; k < hd; ++k) Sr[k] = AccR(Sr[k] * std::exp(static_cast<double>(gh[k])));
      }
      for (int64_t vd = 0; vd < hd; ++vd) {
        const double* Sr = &Sp[vd * hd];
        double pred = 0.0;
        for (int64_t k = 0; k < hd; ++k) pred = AccR(pred + Sr[k] * knp[k]);
        u[static_cast<size_t>(vd)] = AccR((static_cast<double>(vvp[vd]) - pred) * b);
      }
      for (int64_t vd = 0; vd < hd; ++vd) {
        double* Sr = &Sp[vd * hd];
        const double uv = u[static_cast<size_t>(vd)];
        for (int64_t k = 0; k < hd; ++k) Sr[k] = AccR(Sr[k] + uv * knp[k]);
      }
      float* cr = &core[static_cast<size_t>(base)];
      for (int64_t vd = 0; vd < hd; ++vd) {
        const double* Sr = &Sp[vd * hd];
        double o = 0.0;
        for (int64_t k = 0; k < hd; ++k)
          o = AccR(o + Sr[k] * (static_cast<double>(qnp[k]) * scale));
        cr[vd] = static_cast<float>(o);  // f32 output (bf16-rounding the output MEASURED -14, reverted)
      }
    }
  }
  return DBuf(d, DType::kF32, {T, proj}, core.data());  // upload back to device
}

// ── DEVICE MLA attention CORE (VT_KIMI_DEVICE_MLA): the NoPE causal softmax over the
// per-head k_nope|k_pe / v, run through the shared device op vt::Attention (f32 online
// softmax — vLLM's FA2 regime) instead of the f64 host recompute. MLA has asymmetric
// head dims (qk = qk_nope+qk_rope, v = v_head_dim), which vt::Attention (one head-dim D
// for q/k/v) does not express, so the value is PADDED to qk with zeros: the weighted sum
// over the zero tail is 0, so out[:, :, :vh] is the exact attention core (softmax weights
// depend only on q·k, which is unaffected by the padded v). q is already laid out per head
// as [q_nope(qn) | q_pe(qr)] so dq views directly as [T,nah,qk]; key is built per (t,h) as
// [k_nope(qn) | k_pe(qr, SHARED across heads)] and value as [v(vh) | 0]. Scale = qk^-0.5,
// matching the host island / kimi_linear_forward.cpp:223. Returns [T, nah*v_head_dim].
DBuf MlaAttnCoreDevice(const Dev& d, DBuf& dq, DBuf& dkv, DBuf& dkpe,
                       const KimiLinearParams& p, int64_t T) {
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (qn + vh);

  DBuf key(d, DType::kF32, {T, nah, qk});
  DBuf val(d, DType::kF32, {T, nah, qk});
  val.Zero(d);  // pad-V: the [vh, qk) tail stays 0
  {
    const size_t qkb = static_cast<size_t>(qk) * sizeof(float);
    const size_t qnb = static_cast<size_t>(qn) * sizeof(float);
    const size_t qrb = static_cast<size_t>(qr) * sizeof(float);
    const size_t vhb = static_cast<size_t>(vh) * sizeof(float);
    const char* kv = static_cast<const char*>(dkv.ptr());
    const char* kpe = static_cast<const char*>(dkpe.ptr());
    char* kp = static_cast<char*>(key.ptr());
    char* vp = static_cast<char*>(val.ptr());
    for (int64_t t = 0; t < T; ++t) {
      const char* kpe_t = kpe + static_cast<size_t>(t) * qrb;
      for (int64_t h = 0; h < nah; ++h) {
        const char* src =
            kv + (static_cast<size_t>(t) * kvw + static_cast<size_t>(h) * (qn + vh)) *
                     sizeof(float);
        char* kdst = kp + (static_cast<size_t>(t) * nah + h) * qkb;
        char* vdst = vp + (static_cast<size_t>(t) * nah + h) * qkb;
        d.b.Copy(d.q, kdst, src, qnb);          // k_nope[qn]
        d.b.Copy(d.q, kdst + qnb, kpe_t, qrb);  // k_pe[qr] (shared across heads)
        d.b.Copy(d.q, vdst, src + qnb, vhb);    // v[vh]; the [vh,qk) tail stays 0
      }
    }
  }
  // VT_KIMI_BF16_ISLANDS: round the attention inputs to bf16 precision on device
  // (matching the host island's RoundHostBf16), before the f32 softmax.
  if (Bf16Islands()) {
    RoundDevBf16(d, dq);
    RoundDevBf16(d, key);
    RoundDevBf16(d, val);
  }
  Tensor query = MakeTensor(dq.ptr(), DType::kF32, d.q.device, {T, nah, qk});
  DBuf attn(d, DType::kF32, {T, nah, qk});
  const float scale = static_cast<float>(std::pow(static_cast<double>(qk), -0.5));
  vt::Attention(d.q, attn.t(), query, key.t(), val.t(), vt::AttentionArgs{scale, true});

  // slice out[:, :, :vh] -> [T, nah*vh] (the pad-V tail is 0 by construction).
  DBuf out(d, DType::kF32, {T, nah * vh});
  {
    const size_t qkb = static_cast<size_t>(qk) * sizeof(float);
    const size_t vhb = static_cast<size_t>(vh) * sizeof(float);
    const char* ap = static_cast<const char*>(attn.ptr());
    char* op = static_cast<char*>(out.ptr());
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < nah; ++h)
        d.b.Copy(d.q, op + (static_cast<size_t>(t) * nah + h) * vhb,
                 ap + (static_cast<size_t>(t) * nah + h) * qkb, vhb);
  }
  return out;
}

// ── HOST-FALLBACK ISLAND: the materialized-MHA attention CORE (causal softmax over
// the per-head k_nope|k_pe / v, NoPE so no RoPE). Identical math to kimi_linear_
// forward.cpp:223-258; shared by the f32 and bf16 MLA paths (only the projections
// feeding dq/dkv/dkpe differ). Returns [T, nah*v_head_dim]. When VT_KIMI_DEVICE_MLA is
// set, the attention core runs on device via vt::Attention (MlaAttnCoreDevice) instead.
DBuf MlaSoftmaxIsland(const Dev& d, DBuf& dq, DBuf& dkv, DBuf& dkpe,
                      const KimiLinearParams& p, int64_t T) {
  if (DeviceMla()) return MlaAttnCoreDevice(d, dq, dkv, dkpe, p, T);
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (qn + vh);
  std::vector<float> hq(static_cast<size_t>(T) * nah * qk),
      hkv(static_cast<size_t>(T) * kvw), hkpe(static_cast<size_t>(T) * qr);
  dq.Download(d, hq.data());
  dkv.Download(d, hkv.data());
  dkpe.Download(d, hkpe.data());
  // VT_KIMI_BF16_ISLANDS: bf16-precision inputs to the softmax core (like vLLM's FA2).
  RoundHostBf16(hq);
  RoundHostBf16(hkv);
  RoundHostBf16(hkpe);
  const double scale = std::pow(static_cast<double>(qk), -0.5);
  std::vector<float> out(static_cast<size_t>(T) * nah * vh, 0.0f);
  std::vector<double> sc(static_cast<size_t>(T));
  for (int64_t h = 0; h < nah; ++h) {
    for (int64_t t = 0; t < T; ++t) {
      const float* q_nope = &hq[static_cast<size_t>(t * nah * qk + h * qk)];
      const float* q_pe = q_nope + qn;
      double mx = -INFINITY;
      for (int64_t s = 0; s <= t; ++s) {
        const float* k_nope = &hkv[static_cast<size_t>(s * kvw + h * (qn + vh))];
        const float* kpe = &hkpe[static_cast<size_t>(s * qr)];
        double dot = 0.0;
        for (int64_t dd = 0; dd < qn; ++dd)
          dot = AccR(dot + static_cast<double>(q_nope[dd]) * k_nope[dd]);
        for (int64_t dd = 0; dd < qr; ++dd)
          dot = AccR(dot + static_cast<double>(q_pe[dd]) * kpe[dd]);
        dot = AccR(dot * scale);
        sc[static_cast<size_t>(s)] = dot;
        mx = std::max(mx, dot);
      }
      double sum = 0.0;
      for (int64_t s = 0; s <= t; ++s) {
        const double e = AccR(std::exp(sc[static_cast<size_t>(s)] - mx));
        sc[static_cast<size_t>(s)] = e;
        sum = AccR(sum + e);
      }
      float* ot = &out[static_cast<size_t>(t * nah * vh + h * vh)];
      for (int64_t dd = 0; dd < vh; ++dd) {
        double acc = 0.0;
        for (int64_t s = 0; s <= t; ++s) {
          const float* vs = &hkv[static_cast<size_t>(s * kvw + h * (qn + vh) + qn)];
          acc = AccR(acc + (sc[static_cast<size_t>(s)] / sum) * static_cast<double>(vs[dd]));
        }
        ot[dd] = static_cast<float>(acc);  // f32 output (bf16-rounding the output MEASURED -14, reverted)
      }
    }
  }
  return DBuf(d, DType::kF32, {T, nah * vh}, out.data());  // upload
}

// ─── (1) KDA linear-attention layer (device + one host-fallback island) ───────
// Grounding kimi_linear_forward.cpp:101-190 (the W2 reference this must match).
DBuf KdaLayerDevice(const Dev& d, const KdaLayerHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;

  // q/k/v projections -> silu short convs (ON DEVICE).
  DBuf rq(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, rq.t(), dh, WF32(d, w.q_proj, {proj, H}));
  DBuf rk(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, rk.t(), dh, WF32(d, w.k_proj, {proj, H}));
  DBuf rv(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, rv.t(), dh, WF32(d, w.v_proj, {proj, H}));
  DBuf qc = ConvSilu(d, rq.t(), w.q_conv, T, proj, K);
  DBuf kc = ConvSilu(d, rk.t(), w.k_conv, T, proj, K);
  DBuf vc = ConvSilu(d, rv.t(), w.v_conv, T, proj, K);  // v (no L2-norm)

  // per-head q/k L2-norm over head_dim (ON DEVICE) — view [T,proj] as [T*nh,hd].
  DBuf qn(d, DType::kF32, {T, proj});
  DBuf kn(d, DType::kF32, {T, proj});
  {
    Tensor qc3 = MakeTensor(qc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kc3 = MakeTensor(kc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    vt::L2Norm(d.q, qn3, qc3, vt::L2NormArgs{1e-6f});
    vt::L2Norm(d.q, kn3, kc3, vt::L2NormArgs{1e-6f});
  }

  // beta_raw = b_proj(x); low-rank decay g1 = f_b(f_a(x)); gate g2 = g_b(g_a(x))
  // (ON DEVICE). The sigmoid(beta_raw) + the decay gate g live in the island.
  DBuf braw(d, DType::kF32, {T, nh});
  vt::MatmulBT(d.q, braw.t(), dh, WF32(d, w.b_proj, {nh, H}));
  DBuf fa(d, DType::kF32, {T, hd});
  vt::MatmulBT(d.q, fa.t(), dh, WF32(d, w.f_a_proj, {hd, H}));
  DBuf g1(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, g1.t(), fa.t(), WF32(d, w.f_b_proj, {proj, hd}));
  DBuf ga(d, DType::kF32, {T, hd});
  vt::MatmulBT(d.q, ga.t(), dh, WF32(d, w.g_a_proj, {hd, H}));
  DBuf g2(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, g2.t(), ga.t(), WF32(d, w.g_b_proj, {proj, hd}));

  // HOST-FALLBACK ISLAND: KDA decay gate + per-k-channel gated-delta recurrence
  // (shared KdaRecurrenceIsland — see its definition above). THE W7-speed residual.
  DBuf dcore = KdaRecurrenceIsland(d, qn, kn, vc, g1, braw, w.a_log, w.dt_bias, p, T);

  // sigmoid-gated output RMSNorm then o_proj (ON DEVICE).
  DBuf dcn(d, DType::kF32, {T, proj});
  {
    Tensor x3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor g3 = MakeTensor(g2.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor o3 = MakeTensor(dcn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    vt::RmsNormGated(d.q, o3, x3, g3, WF32(d, w.o_norm, {hd}),
                     vt::RmsNormGatedArgs{p.rms_norm_eps, /*sigmoid_gate=*/true});
  }
  DBuf out(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, out.t(), dcn.t(), WF32(d, w.o_proj, {H, proj}));
  return out;
}

// ─── (2) NoPE-MLA full-attention layer (device projections + host attn core) ──
// Grounding kimi_linear_forward.cpp:193-260.
DBuf MlaLayerDevice(const Dev& d, const MlaLayerHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t kvw = nah * (qn + vh);

  // q_proj, kv_a projections (ON DEVICE).
  DBuf dq(d, DType::kF32, {T, nah * qk});
  vt::MatmulBT(d.q, dq.t(), dh, WF32(d, w.q_proj, {nah * qk, H}));
  DBuf dlat(d, DType::kF32, {T, L + qr});
  vt::MatmulBT(d.q, dlat.t(), dh, WF32(d, w.kv_a_proj_with_mqa, {L + qr, H}));

  // Split latent -> kv_c[T,L] (normed), k_pe[T,qr] (shared, not normed) via device
  // row column-slice copies.
  DBuf dkvc(d, DType::kF32, {T, L});
  DBuf dkpe(d, DType::kF32, {T, qr});
  {
    const size_t rl = static_cast<size_t>(L + qr) * sizeof(float);
    const char* src = static_cast<const char*>(dlat.ptr());
    char* pc = static_cast<char*>(dkvc.ptr());
    char* pp = static_cast<char*>(dkpe.ptr());
    for (int64_t t = 0; t < T; ++t) {
      d.b.Copy(d.q, pc + static_cast<size_t>(t) * L * sizeof(float),
               src + static_cast<size_t>(t) * rl, static_cast<size_t>(L) * sizeof(float));
      d.b.Copy(d.q, pp + static_cast<size_t>(t) * qr * sizeof(float),
               src + static_cast<size_t>(t) * rl + static_cast<size_t>(L) * sizeof(float),
               static_cast<size_t>(qr) * sizeof(float));
    }
  }
  // kv_a_layernorm (ON DEVICE, non-residual) then kv_b_proj (ON DEVICE).
  DBuf dkvcn(d, DType::kF32, {T, L});
  vt::RmsNorm(d.q, dkvcn.t(), dkvc.t(), WF32(d, w.kv_a_layernorm, {L}),
              vt::RmsNormArgs{p.rms_norm_eps, false});
  DBuf dkv(d, DType::kF32, {T, kvw});
  vt::MatmulBT(d.q, dkv.t(), dkvcn.t(), WF32(d, w.kv_b_proj, {kvw, L}));

  // HOST-FALLBACK ISLAND: the materialized-MHA attention core (causal softmax, NoPE)
  // via the shared MlaSoftmaxIsland (see its definition above). The device path is
  // mla::ForwardMlaAttentionBlock over the runner's paged KV — the born-on-runner
  // residual. Identical math to kimi_linear_forward.cpp:223-258.
  DBuf dout = MlaSoftmaxIsland(d, dq, dkv, dkpe, p, T);
  DBuf attn(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, attn.t(), dout.t(), WF32(d, w.o_proj, {H, nah * vh}));
  return attn;
}

// ─── (3) sigmoid noaux_tc MoE block (device router+experts+combine) ───────────
// Grounding kimi_linear_forward.cpp:263-350 + deepseek_v2.cpp:331-472 MoeBlock.
DBuf MoeBlockDevice(const Dev& d, const MoeHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t k = p.num_experts_per_token;
  const int64_t I = p.moe_intermediate_size;

  // router: logits = gate(x) then grouped sigmoid top-k (ON DEVICE).
  DBuf dlog(d, DType::kF32, {T, E});
  vt::MatmulBT(d.q, dlog.t(), dh, WF32(d, w.gate, {E, H}));
  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(k);
  args.renormalize = p.moe_renormalize;
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(p.num_expert_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = static_cast<float>(p.routed_scaling_factor);
  DBuf dtw(d, DType::kF32, {T, k});
  DBuf dtid(d, DType::kI32, {T, k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.empty())
    bias = std::make_unique<Tensor>(WF32(d, w.e_score_correction_bias, {E}));
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // routed experts — the CPU-lane per-expert gather / SwiGLU / scatter (the grouped
  // CUDA GEMM is CUDA-only; deepseek_v2.cpp:414-455 REFERENCE path, f32 + MatmulBT).
  DBuf expert_out(d, DType::kF32, {T, k, H});
  expert_out.Zero(d);
  std::vector<int32_t> ids(static_cast<size_t>(T) * k);
  dtid.Download(d, ids.data());
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < k; ++j)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t * k + j)])].push_back({t, j});
  const size_t row_bytes = static_cast<size_t>(H) * sizeof(float);
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    DBuf xg(d, DType::kF32, {n, H});
    for (int64_t r = 0; r < n; ++r)
      d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
               static_cast<const char*>(dh.data) +
                   static_cast<size_t>(list[static_cast<size_t>(r)].first) * row_bytes,
               row_bytes);
    const MlpHostWeights& ex = w.experts[static_cast<size_t>(e)];
    DBuf y = SwiGluDevice(d, ex.gate_proj, ex.up_proj, ex.down_proj, xg.t(), H, I, n);
    for (int64_t r = 0; r < n; ++r) {
      const auto& tj = list[static_cast<size_t>(r)];
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(tj.first * k + tj.second) * row_bytes,
               static_cast<const char*>(y.ptr()) + static_cast<size_t>(r) * row_bytes,
               row_bytes);
    }
  }

  // shared expert (always, added to the routed sum) + weighted combine (ON DEVICE).
  DBuf out(d, DType::kF32, {T, H});
  if (w.has_shared) {
    const int64_t shared_i = I * p.num_shared_experts;
    DBuf shared = SwiGluDevice(d, w.shared.gate_proj, w.shared.up_proj,
                               w.shared.down_proj, dh, H, shared_i, T);
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared.t());
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

// ─── (4) dense layer-0 SwiGLU MLP (ON DEVICE) ─────────────────────────────────
DBuf DenseMlpDevice(const Dev& d, const MlpHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  return SwiGluDevice(d, w.gate_proj, w.up_proj, w.down_proj, dh, p.hidden_size,
                      p.intermediate_size, T);
}

// Wrap [rows,vocab] f32 device logits as a DEVICE-RESIDENT ForwardLogits — verbatim
// the kimi_linear.cpp / deepseek_v2.cpp:633 seam (return the pooled block to the
// shared DevicePool via the shared_ptr deleter; expose the [rows,vocab] view).
ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  const size_t alloc = dlogits.alloc_bytes();
  void* pp = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(pp, [alloc](void* q) { Pool().Put(alloc, q); });
  return fl;
}

// The whole device-compute forward over a single token sequence — the pre-norm
// residual stream (kimi_linear_forward.cpp HostForwardSeq / deepseek_v2.cpp
// ForwardBody). Returns the DEVICE-RESIDENT [rows,vocab] f32 logits DBuf.
DBuf DeviceForwardBody(const Dev& d, const KimiLinearWeights& weights,
                       const std::vector<int32_t>& token_ids,
                       const std::vector<int32_t>& logits_indices) {
  const KimiLinearHostWeights& host = weights.host;
  const KimiLinearParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const float eps = p.rms_norm_eps;
  VT_CHECK(T > 0, "KimiLinear device compute: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "KimiLinear device compute: host layer count != num_hidden_layers");

  // embed -> residual-stream delta (ON DEVICE).
  DBuf hidden(d, DType::kF32, {T, H});
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor htab = WF32(d, host.embed_tokens, {V, H});
    Tensor hh = hidden.t();
    vt::Embedding(d.q, hh, htab, dids.t());
  }
  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);
  Tensor hcur = hidden.t();
  std::shared_ptr<void> hold;

  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerHostWeights& lw = host.layers[static_cast<size_t>(l)];
    DBuf dhn(d, DType::kF32, {T, H});
    AddRmsNorm(d, dhn, hcur, WF32(d, lw.input_layernorm, {H}), res, eps);
    DBuf attn = lw.is_kda ? KdaLayerDevice(d, lw.kda, dhn.t(), p, T)
                          : MlaLayerDevice(d, lw.mla, dhn.t(), p, T);
    DBuf dh2(d, DType::kF32, {T, H});
    AddRmsNorm(d, dh2, attn.t(), WF32(d, lw.post_attention_layernorm, {H}), res, eps);
    DBuf mlp = lw.is_moe ? MoeBlockDevice(d, lw.moe, dh2.t(), p, T)
                         : DenseMlpDevice(d, lw.dense, dh2.t(), p, T);
    auto* held = new DBuf(std::move(mlp));
    hcur = held->t();
    hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  DBuf dnorm(d, DType::kF32, {T, H});
  AddRmsNorm(d, dnorm, hcur, WF32(d, host.final_norm, {H}), res, eps);

  // logits_indices gather-before-lm_head, in REQUEST order (mirrors the reference's
  // `want` construction — gather whenever indices are given).
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kF32,
               logits_indices.empty()
                   ? std::vector<int64_t>{1, 1}
                   : std::vector<int64_t>{static_cast<int64_t>(logits_indices.size()), H});
  if (!logits_indices.empty()) {
    const size_t rb = static_cast<size_t>(H) * sizeof(float);
    char* dp = static_cast<char*>(dgather.ptr());
    const char* sp = static_cast<const char*>(dnorm.ptr());
    for (size_t i = 0; i < logits_indices.size(); ++i) {
      const int32_t idx = logits_indices[i];
      VT_CHECK(idx >= 0 && idx < T,
               "KimiLinear device compute: logits index out of range");
      d.b.Copy(d.q, dp + i * rb, sp + static_cast<size_t>(idx) * rb, rb);
    }
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  const bool tied = p.tie_word_embeddings || host.lm_head.empty();
  Tensor lm = tied ? WF32(d, host.embed_tokens, {V, H}) : WF32(d, host.lm_head, {V, H});
  DBuf logits(d, DType::kF32, {n_out, V});
  vt::MatmulBT(d.q, logits.t(), src, lm);
  return logits;
}

// ═══ bf16-RESIDENT device COMPUTE (§13) — the FULL-model device forward ═════════
// Byte-for-byte the f32 structure above, with each of the ~20 projection GEMMs
// swapped from WF32+MatmulBT (f32 host alias) to GemmBf16 over the bf16-resident
// OwnedTensor (cast the f32 activation to bf16, MatmulBT bf16xbf16->f32 — vLLM's own
// projection numerics). The two host-fallback islands (KdaRecurrenceIsland /
// MlaSoftmaxIsland) and every small-vector op (short convs, norms, L2Norm,
// RmsNormGated, router topk, weighted combine, SwiGLU activation) are UNCHANGED (f32
// activations; the tiny vectors alias host f32 via WF32). The residual stream stays
// f32. Same call graph as the f32 path, so the tiny-config gate proves the exact
// wiring the full model runs.
DBuf SwiGluDeviceBf16(const Dev& d, const OwnedTensor& gate, const OwnedTensor& up,
                      const OwnedTensor& down, const Tensor& dh, int64_t H, int64_t I,
                      int64_t T) {
  DBuf dg(d, DType::kF32, {T, I});
  GemmBf16(d, dg.t(), dh, gate, I, H);
  DBuf du(d, DType::kF32, {T, I});
  GemmBf16(d, du.t(), dh, up, I, H);
  DBuf da(d, DType::kF32, {T, I});
  vt::MoeSiluMul(d.q, da.t(), dg.t(), du.t());
  DBuf out(d, DType::kF32, {T, H});
  GemmBf16(d, out.t(), da.t(), down, H, I);
  return out;
}

DBuf KdaLayerDeviceBf16(const Dev& d, const KdaResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;

  // q/k/v projections -> silu short convs.
  DBuf rq(d, DType::kF32, {T, proj});
  GemmBf16(d, rq.t(), dh, w.q_proj, proj, H);
  DBuf rk(d, DType::kF32, {T, proj});
  GemmBf16(d, rk.t(), dh, w.k_proj, proj, H);
  DBuf rv(d, DType::kF32, {T, proj});
  GemmBf16(d, rv.t(), dh, w.v_proj, proj, H);
  DBuf qc = ConvSilu(d, rq.t(), w.q_conv, T, proj, K);
  DBuf kc = ConvSilu(d, rk.t(), w.k_conv, T, proj, K);
  DBuf vc = ConvSilu(d, rv.t(), w.v_conv, T, proj, K);  // v (no L2-norm)

  // per-head q/k L2-norm over head_dim — view [T,proj] as [T*nh,hd].
  DBuf qn(d, DType::kF32, {T, proj});
  DBuf kn(d, DType::kF32, {T, proj});
  {
    Tensor qc3 = MakeTensor(qc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kc3 = MakeTensor(kc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    vt::L2Norm(d.q, qn3, qc3, vt::L2NormArgs{1e-6f});
    vt::L2Norm(d.q, kn3, kc3, vt::L2NormArgs{1e-6f});
  }

  DBuf braw(d, DType::kF32, {T, nh});
  GemmBf16(d, braw.t(), dh, w.b_proj, nh, H);
  DBuf fa(d, DType::kF32, {T, hd});
  GemmBf16(d, fa.t(), dh, w.f_a_proj, hd, H);
  DBuf g1(d, DType::kF32, {T, proj});
  GemmBf16(d, g1.t(), fa.t(), w.f_b_proj, proj, hd);
  DBuf ga(d, DType::kF32, {T, hd});
  GemmBf16(d, ga.t(), dh, w.g_a_proj, hd, H);
  DBuf g2(d, DType::kF32, {T, proj});
  GemmBf16(d, g2.t(), ga.t(), w.g_b_proj, proj, hd);

  // HOST-FALLBACK ISLAND: KDA decay gate + per-k-channel gated-delta recurrence.
  DBuf dcore = KdaRecurrenceIsland(d, qn, kn, vc, g1, braw, w.a_log, w.dt_bias, p, T);

  DBuf dcn(d, DType::kF32, {T, proj});
  {
    Tensor x3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor g3 = MakeTensor(g2.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor o3 = MakeTensor(dcn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    vt::RmsNormGated(d.q, o3, x3, g3, WF32(d, w.o_norm, {hd}),
                     vt::RmsNormGatedArgs{p.rms_norm_eps, /*sigmoid_gate=*/true});
  }
  DBuf out(d, DType::kF32, {T, H});
  GemmBf16(d, out.t(), dcn.t(), w.o_proj, H, proj);
  return out;
}

DBuf MlaLayerDeviceBf16(const Dev& d, const MlaResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t kvw = nah * (qn + vh);

  DBuf dq(d, DType::kF32, {T, nah * qk});
  GemmBf16(d, dq.t(), dh, w.q_proj, nah * qk, H);
  DBuf dlat(d, DType::kF32, {T, L + qr});
  GemmBf16(d, dlat.t(), dh, w.kv_a_proj_with_mqa, L + qr, H);

  DBuf dkvc(d, DType::kF32, {T, L});
  DBuf dkpe(d, DType::kF32, {T, qr});
  {
    const size_t rl = static_cast<size_t>(L + qr) * sizeof(float);
    const char* src = static_cast<const char*>(dlat.ptr());
    char* pc = static_cast<char*>(dkvc.ptr());
    char* pp = static_cast<char*>(dkpe.ptr());
    for (int64_t t = 0; t < T; ++t) {
      d.b.Copy(d.q, pc + static_cast<size_t>(t) * L * sizeof(float),
               src + static_cast<size_t>(t) * rl, static_cast<size_t>(L) * sizeof(float));
      d.b.Copy(d.q, pp + static_cast<size_t>(t) * qr * sizeof(float),
               src + static_cast<size_t>(t) * rl + static_cast<size_t>(L) * sizeof(float),
               static_cast<size_t>(qr) * sizeof(float));
    }
  }
  DBuf dkvcn(d, DType::kF32, {T, L});
  vt::RmsNorm(d.q, dkvcn.t(), dkvc.t(), WF32(d, w.kv_a_layernorm, {L}),
              vt::RmsNormArgs{p.rms_norm_eps, false});
  DBuf dkv(d, DType::kF32, {T, kvw});
  GemmBf16(d, dkv.t(), dkvcn.t(), w.kv_b_proj, kvw, L);

  DBuf dout = MlaSoftmaxIsland(d, dq, dkv, dkpe, p, T);
  DBuf attn(d, DType::kF32, {T, H});
  GemmBf16(d, attn.t(), dout.t(), w.o_proj, H, nah * vh);
  return attn;
}

DBuf MoeBlockDeviceBf16(const Dev& d, const MoeResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t k = p.num_experts_per_token;
  const int64_t I = p.moe_intermediate_size;

  // router: logits = gate(x) (bf16, like vLLM) then grouped sigmoid top-k.
  DBuf dlog(d, DType::kF32, {T, E});
  GemmBf16(d, dlog.t(), dh, w.gate, E, H);
  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(k);
  args.renormalize = p.moe_renormalize;
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(p.num_expert_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = static_cast<float>(p.routed_scaling_factor);
  DBuf dtw(d, DType::kF32, {T, k});
  DBuf dtid(d, DType::kI32, {T, k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.empty())
    bias = std::make_unique<Tensor>(WF32(d, w.e_score_correction_bias, {E}));
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // routed experts — per-expert gather / SwiGLU / scatter (bf16 GEMMs).
  DBuf expert_out(d, DType::kF32, {T, k, H});
  expert_out.Zero(d);
  std::vector<int32_t> ids(static_cast<size_t>(T) * k);
  dtid.Download(d, ids.data());
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < k; ++j)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t * k + j)])].push_back({t, j});
  const size_t row_bytes = static_cast<size_t>(H) * sizeof(float);
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    DBuf xg(d, DType::kF32, {n, H});
    for (int64_t r = 0; r < n; ++r)
      d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
               static_cast<const char*>(dh.data) +
                   static_cast<size_t>(list[static_cast<size_t>(r)].first) * row_bytes,
               row_bytes);
    const MlpResidentWeights& ex = w.experts[static_cast<size_t>(e)];
    DBuf y = SwiGluDeviceBf16(d, ex.gate_proj, ex.up_proj, ex.down_proj, xg.t(), H, I, n);
    for (int64_t r = 0; r < n; ++r) {
      const auto& tj = list[static_cast<size_t>(r)];
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(tj.first * k + tj.second) * row_bytes,
               static_cast<const char*>(y.ptr()) + static_cast<size_t>(r) * row_bytes,
               row_bytes);
    }
  }

  // shared expert (always) + weighted combine.
  DBuf out(d, DType::kF32, {T, H});
  if (w.has_shared) {
    const int64_t shared_i = I * p.num_shared_experts;
    DBuf shared = SwiGluDeviceBf16(d, w.shared.gate_proj, w.shared.up_proj,
                                   w.shared.down_proj, dh, H, shared_i, T);
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared.t());
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

DBuf DenseMlpDeviceBf16(const Dev& d, const MlpResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  return SwiGluDeviceBf16(d, w.gate_proj, w.up_proj, w.down_proj, dh, p.hidden_size,
                          p.intermediate_size, T);
}

DBuf DeviceForwardBodyBf16(const Dev& d, const KimiLinearWeights& weights,
                           const std::vector<int32_t>& token_ids,
                           const std::vector<int32_t>& logits_indices) {
  const KimiLinearResidentWeights& rw = weights.resident;
  const KimiLinearParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const float eps = p.rms_norm_eps;
  VT_CHECK(T > 0, "KimiLinear bf16 device compute: empty token sequence");
  VT_CHECK(static_cast<int64_t>(rw.layers.size()) == L,
           "KimiLinear bf16 device compute: resident layer count != num_hidden_layers");

  // embed (bf16 table -> f32 out) -> residual stream.
  DBuf hidden(d, DType::kF32, {T, H});
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor htab = ResidentBf16W(d, rw.embed_tokens, {V, H});
    Tensor hh = hidden.t();
    vt::Embedding(d.q, hh, htab, dids.t());
  }
  const bool bf16_res = Bf16Residual();
  if (bf16_res) RoundDevBf16(d, hidden);  // vLLM embed output is bf16
  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);
  Tensor hcur = hidden.t();
  std::shared_ptr<void> hold;

  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerResidentWeights& lw = rw.layers[static_cast<size_t>(l)];
    DBuf dhn(d, DType::kF32, {T, H});
    AddRmsNorm(d, dhn, hcur, WF32(d, lw.input_layernorm, {H}), res, eps);
    if (bf16_res) RoundDevBf16(d, res);  // vLLM stores residual bf16 (variance saw f32 sum)
    DBuf attn = lw.is_kda ? KdaLayerDeviceBf16(d, lw.kda, dhn.t(), p, T)
                          : MlaLayerDeviceBf16(d, lw.mla, dhn.t(), p, T);
    if (bf16_res) RoundDevBf16(d, attn);  // vLLM attn_output is bf16
    DBuf dh2(d, DType::kF32, {T, H});
    AddRmsNorm(d, dh2, attn.t(), WF32(d, lw.post_attention_layernorm, {H}), res, eps);
    if (bf16_res) RoundDevBf16(d, res);
    DBuf mlp = lw.is_moe ? MoeBlockDeviceBf16(d, lw.moe, dh2.t(), p, T)
                         : DenseMlpDeviceBf16(d, lw.dense, dh2.t(), p, T);
    if (bf16_res) RoundDevBf16(d, mlp);  // vLLM mlp output is bf16
    auto* held = new DBuf(std::move(mlp));
    hcur = held->t();
    hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  DBuf dnorm(d, DType::kF32, {T, H});
  AddRmsNorm(d, dnorm, hcur, WF32(d, rw.final_norm, {H}), res, eps);

  // logits_indices gather-before-lm_head, in REQUEST order.
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kF32,
               logits_indices.empty()
                   ? std::vector<int64_t>{1, 1}
                   : std::vector<int64_t>{static_cast<int64_t>(logits_indices.size()), H});
  if (!logits_indices.empty()) {
    const size_t rb = static_cast<size_t>(H) * sizeof(float);
    char* dp = static_cast<char*>(dgather.ptr());
    const char* sp = static_cast<const char*>(dnorm.ptr());
    for (size_t i = 0; i < logits_indices.size(); ++i) {
      const int32_t idx = logits_indices[i];
      VT_CHECK(idx >= 0 && idx < T,
               "KimiLinear bf16 device compute: logits index out of range");
      d.b.Copy(d.q, dp + i * rb, sp + static_cast<size_t>(idx) * rb, rb);
    }
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  const bool tied = p.tie_word_embeddings || rw.lm_head.Empty();
  const OwnedTensor& lm = tied ? rw.embed_tokens : rw.lm_head;
  DBuf logits(d, DType::kF32, {n_out, V});
  GemmBf16(d, logits.t(), src, lm, V, H);
  return logits;
}

}  // namespace

// ─── per-op device wrappers (host-in / host-out) — the per-op CPU gates ────────
std::vector<float> KimiKdaLayerForwardDevice(const KdaLayerHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = KdaLayerDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

std::vector<float> KimiNoPEMlaLayerForwardDevice(const MlaLayerHostWeights& w,
                                                 const std::vector<float>& hidden_normed,
                                                 const KimiLinearParams& p,
                                                 int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = MlaLayerDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

// Device MLA attention CORE only (pad-V + vt::Attention), host-in / host-out — the
// dedicated RED-first CPU gate for the VT_KIMI_DEVICE_MLA wiring, independent of the
// env flag. q [T,nah*qk], kv [T,nah*(qn+vh)], kpe [T,qr] -> [T,nah*vh].
std::vector<float> KimiMlaAttnCoreDevice(const std::vector<float>& q_host,
                                         const std::vector<float>& kv_host,
                                         const std::vector<float>& kpe_host,
                                         const KimiLinearParams& p, int64_t num_tokens,
                                         vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t nah = p.num_attention_heads;
  const int64_t qk = p.qk_nope_head_dim + p.qk_rope_head_dim;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (p.qk_nope_head_dim + vh);
  const int64_t qr = p.qk_rope_head_dim;
  DBuf dq(d, DType::kF32, {num_tokens, nah * qk}, q_host.data());
  DBuf dkv(d, DType::kF32, {num_tokens, kvw}, kv_host.data());
  DBuf dkpe(d, DType::kF32, {num_tokens, qr}, kpe_host.data());
  DBuf out = MlaAttnCoreDevice(d, dq, dkv, dkpe, p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * nah * vh);
  out.Download(d, h.data());
  return h;
}

std::vector<float> KimiMoeBlockForwardDevice(const MoeHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = MoeBlockDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

std::vector<float> KimiDenseMlpForwardDevice(const MlpHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = DenseMlpDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

bool KimiDeviceComputeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_COMPUTE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

ForwardLogits KimiLinearModel::ForwardDeviceCompute(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  // §13: the FULL model loads as bf16-resident (host f32 would OOM the pool), so
  // route through the bf16 device forward whenever resident weights are present; the
  // f32 host path stays for the tiny-config unit gate.
  const bool bf16 = weights.resident.resident;
  VT_CHECK(bf16 || weights.host.materialized,
           "KimiLinear device compute: neither bf16-resident (§13, "
           "LoadKimiLinearResidentBf16Weights) nor host-float (LoadKimiLinearFor"
           "CausalLMWeights) weights are populated. The device compute reads one or "
           "the other; the full model MUST use the bf16-resident path (183 GiB f32 "
           "OOMs the 119 GiB unified pool).");
  // The device compute manages a fresh single-sequence context (NoPE, causal); the
  // runner's paged het-KV / positions are consumed by the born-on-runner residual
  // (the paged incremental decode), not this seam.
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = bf16 ? DeviceForwardBodyBf16(d, weights, token_ids, logits_indices)
                      : DeviceForwardBody(d, weights, token_ids, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

}  // namespace vllm
