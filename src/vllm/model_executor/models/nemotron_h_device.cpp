// Nemotron-H (`NemotronHForCausalLM`) — A2-R, the DEVICE arm
// ([spec](../../../../.agents/specs/nemotron-h-abi-e2e.md), issue #810; parent
// row #517).
//
// ─── WHAT RUNS WHERE, WHICH IS THE WHOLE SCOPE OF THIS FILE ─────────────────
//
//   DEVICE  the embedding lookup, all 52 layer norms + norm_f, and the 6 GQA
//           attention blocks. The residual stream is device-resident for the
//           entire forward.
//   HOST    the 23 Mamba2 blocks, the 23 MoE blocks, and lm_head.
//
// The line is drawn by the MEMORY FORMAT THE CHECKPOINT SHIPS, not by taste.
// The released checkpoint is `quant_algo: MIXED_PRECISION` and only 216 of its
// tensors are plain bf16 (gated: test_nemotron_h_loader.cpp:254). Everything
// this file puts on the device is from that bf16 population. Everything it
// leaves on the host is not:
//
//   * the 23 Mamba2 blocks are entered through `mixer.in_proj`, which is FP8
//     W8A8 static. The block is NOT splittable — `in_proj` produces the fused
//     zxbcdt that the conv and the scan both consume — so it moves as a unit,
//     after the shared FP8 W8A8 linear seam is extracted out of qwen3_5.cpp
//     (`ResidentFp8` :1456, `MatmulFp8CutlassD` :1495) into a real shared
//     header. That extraction is issue #940 and is deliberately NOT done here:
//     its own gate is Qwen3.5 byte-identity, which has no business landing
//     inside a NemotronH row.
//   * the 23 MoE blocks' 5888 routed + 23 shared expert projections and
//     `lm_head` are NVFP4 W4A16 g16 — 30.19e9 parameters, 15.8 GiB packed and
//     56.2 GiB dequantized to bf16. There is NO device NVFP4->bf16 dequant
//     kernel in vt (the only standalone device dequant is
//     `vt::DequantFp8ChannelBf16`, fused_ops.h:31, ROCm-only and per-channel
//     FP8), so putting them on the device in bf16 would mean a host dequant
//     plus a 56.2 GiB upload. Both gate hosts are unified-memory, so that is a
//     REBOOT rather than an OOM. nemotron_h_loader.h:36-46 rejected the same
//     design for the load; this file does not re-open it.
//
// So 46 of 52 layers still compute on the host, and each costs one download of
// the normed hidden and one upload of the mixer output. THAT BOUNCE IS
// SCAFFOLD, NOT ARCHITECTURE: every later unit deletes one pair of it. It is
// also why this file makes NO SPEED CLAIM OF ANY KIND — it is slower than the
// host reference, and that is expected and irrelevant to what it gates.
//
// ─── WHY NOT `dense_attn::AttnBlock` (issue #941) ──────────────────────────
//
// The spec's §2 names `dense_attn::AttnBlock` as this model's seam. That is
// wrong, and routing through it would reintroduce the exact defect #810 just
// removed from the runner — a shared function reading HF-config fields this
// architecture does not ship. Three measured reasons:
//
//   1. It takes `Qwen3DenseAttnWeights` (dense_attn_block.h:335), not this
//      model's separate q/k/v/o.
//   2. It reads `cfg.rms_norm_eps`. NemotronH's config ships
//      `layer_norm_epsilon` and `norm_eps` and NO `rms_norm_eps`, which
//      `hf_config.cpp:551` defaults to **0.0** — a silent eps=0 normalization.
//   3. Its default path calls `vt::RopeNeox` unconditionally
//      (dense_attn_block.h:496). NemotronH has no positional embedding at all
//      (`kNemotronHAttentionHasNoRope`; nemotron_h.py:473-486 @ 555967922).
//
// The tree's own idiom for exactly this case is a MODEL-LOCAL block —
// `granite.cpp:84 GraniteAttnBlock`, `gemma.cpp:42`, `gemma2.cpp:123`,
// `gemma3.cpp:108`, `glm4.cpp:80`, `gemma4.cpp:206`, none of them allowlisted.
// `NemotronHAttnBlock` below follows it and documents its deltas.
//
// ─── G-SAFE IS UNTOUCHED ────────────────────────────────────────────────────
//
// Nothing in this file consumes `attn_kv`, `gdn_state`, `gdn_meta`,
// `gdn_state_slots` or `num_reqs`. This arm is NON-PAGED and SINGLE-REQUEST: it
// recomputes Q/K/V over the whole sequence every call, exactly as the host
// reference does. So it does not create the capability the interlock at
// `nemotron_h_registry.cpp:161-170` guards, and all three of that interlock's
// clauses stay exactly as they are. A2-P narrows them; A2-R does not.
#include "vllm/model_executor/models/nemotron_h_forward.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// The SHARED device glue: Dev, DBuf, MakeTensor, Reshape and — the reason this
// header rather than dense_device_glue.h — `ResidentWeight`, the lazy
// upload-once seam this row converts NemotronH's dense weights onto.
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kFusedAddRmsNormStd

namespace vllm {
namespace {

using vt::DType;
using vt::Queue;
using vt::Tensor;

// Same reuse the other model-local blocks take (granite.cpp:52): Dev / DBuf /
// ResidentWeight / MakeTensor / Reshape verbatim, and NOT AttnBlock.
using namespace dense_attn;

int64_t NumelOf(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// VT_FUSED_CHAIN_ADOPT, read exactly as the host arm reads it
// (nemotron_h.cpp:55). This is NOT an incidental duplicate: the whole point of
// A2-R's gate is that the two arms compose the IDENTICAL vt:: op sequence and
// differ only in which backend runs it. A device arm that hand-called
// `vt::RmsNorm(..., &residual)` while the host arm routed the same chain
// through `vt::FusedChain` would be comparing two different compositions and
// calling the result an equivalence — and `scripts/check-fusion-consistency.py`
// refuses it outright (AGENTS.md, "Route model fusion through `vt::FusedChain`").
//
// PRECISELY WHAT IS AND IS NOT GUARANTEED. These are TWO file-local
// function-local statics with byte-identical predicates — this one and
// `nemotron_h.cpp:56` — each latching on its own first call. They read the same
// variable with the same default, so within one process they resolve the same
// way and the equivalence gate below compares like with like. What is NOT true,
// and an earlier draft of this comment claimed, is that they "can never
// straddle": a caller that changed `VT_FUSED_CHAIN_ADOPT` between the two first
// calls would latch two different answers. Nothing in this tree does that, and
// the duplication is the tree's idiom for this env read (qwen3_5.cpp:1699), but
// the property is a convention rather than an impossibility. Hoisting the
// predicate into one shared reader would make it an impossibility, and that is
// a tree-wide change, not this row's.
bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// The residual-add + RMSNorm preamble, through the declared fusion recipe.
// `residual` is updated in place (res += x) and `out` receives the norm, which
// is the contract `kFusedAddRmsNormStd` encodes and the Tier-0 composite
// dispatches to the same `vt::RmsNorm(..., &residual)` primitive, so the two
// branches are bit-identical (tests/vt/test_ops_fused_chain.cpp).
void AddRmsNorm(Dev d, Tensor& out, Tensor& x, const Tensor& w, Tensor& residual,
                const vt::RmsNormArgs& nargs, double eps) {
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, out, x, w, &residual, vt::kFusedAddRmsNormStd,
                   static_cast<float>(eps));
  } else {
    vt::RmsNorm(d.q, out, x, w, nargs, &residual);
  }
}

// Pack a host f32 vector into `dt` and upload it. The staging buffer is kept
// alive across an explicit Synchronize: `DBuf`'s constructor issues an ASYNC
// `cudaMemcpyAsync` (dense_device_glue.h:126, cuda_backend.cu:90) and does not
// wait, so a staging buffer that died at the end of this function would be a
// use-after-free that pageable-memory semantics happen to hide most of the
// time. This arm records no throughput number, so an explicit sync per upload
// costs nothing it is measuring.
DBuf UploadAs(Dev d, const std::vector<float>& v, DType dt,
              const std::vector<int64_t>& shape) {
  const int64_t n = NumelOf(shape);
  VT_CHECK(static_cast<int64_t>(v.size()) == n,
           "NemotronH device: upload element count does not match the shape");
  std::vector<uint8_t> staging(static_cast<size_t>(n) * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(staging.data(), v.data(), staging.size());
  } else {
    auto* dst = reinterpret_cast<uint16_t*>(staging.data());
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::F32ToBF16(v[static_cast<size_t>(i)]);
  }
  DBuf b(d, dt, shape, staging.data());
  d.b.Synchronize(d.q);
  return b;
}

// Download a device buffer and widen it to f32, the comparison currency every
// host-side entry point in nemotron_h_forward.h already speaks.
std::vector<float> DownloadF32(Dev d, DBuf& b, DType dt, int64_t n) {
  std::vector<uint8_t> staging(static_cast<size_t>(n) * vt::SizeOf(dt));
  b.Download(d, staging.data());  // Copy + Synchronize (dense_device_glue.h:158)
  std::vector<float> out(static_cast<size_t>(n));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), staging.data(), staging.size());
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(staging.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  }
  return out;
}

// Refuse by name when a weight the device arm is about to upload is absent or
// mis-shaped. Mirrors the host arm's `RequireWeight` (nemotron_h.cpp:180) —
// same three properties, so a defect that would refuse on the host refuses here
// too rather than reaching a kernel with a null pointer.
void RequireDeviceWeight(const OwnedTensor& w, const char* what, DType want,
                         const std::vector<int64_t>& shape) {
  VT_CHECK(!w.Empty(),
           std::string("NemotronH device forward: weight '") + what +
               "' is not materialized");
  VT_CHECK(w.dtype == want, std::string("NemotronH device forward: weight '") + what +
                                "' has the wrong dtype for this arm");
  VT_CHECK(w.rank == static_cast<int>(shape.size()) &&
               std::equal(shape.begin(), shape.end(), w.shape),
           std::string("NemotronH device forward: weight '") + what +
               "' has the wrong shape");
}

// ─── the model-local attention block ────────────────────────────────────────
//
// One NemotronH GQA self-attention block (nemotron_h.py:415-486 @ 555967922).
// `normed` is the already-normed hidden [T,H] in `adt` on the device; returns
// the o_proj output [T,H] in `adt` on the device.
//
// DELTAS vs `dense_attn::AttnBlock`, each one measured, not assumed:
//   (a) NO POSITIONAL EMBEDDING. `nemotron_h.py` contains zero occurrences of
//       `rope`/`rotary`/`Rotary`; `.forward` (:473-486) is qkv -> split -> attn
//       -> o_proj with no rotation step. `kNemotronHAttentionHasNoRope` states
//       it and the mutation gate proves the property is armed NUMERICALLY —
//       applying a rotation here changes no tensor shape and need not move a
//       token on a short prompt.
//   (b) SEPARATE q/k/v, not a merged qkv owner. The checkpoint ships
//       `q_proj`/`k_proj`/`v_proj` as three tensors (upstream fuses them at
//       load through its stacked-params mapping); `EnumerateNemotronHTensors`
//       already claims them separately, so there is nothing to merge here.
//   (c) NO per-head q/k RMSNorm. `NemotronHAttention.__init__` (:415-471)
//       builds `qkv_proj`, `o_proj` and `Attention` and nothing else.
//   (d) NO biases — `attention_bias=false` on this checkpoint and the loader
//       ships no q/k/v/o bias tensor.
//   (e) NON-PAGED. No `PagedKvCache`, no `ReshapeAndCache`, no slot mapping: a
//       dense causal `vt::Attention` over the whole [T,·], which is exactly
//       what the host reference does (nemotron_h.cpp:615-626). That is what
//       keeps this arm inside G-SAFE, and it is A2-P that makes it paged.
//   (f) Every geometry and epsilon comes from `NemotronHParams`, never from
//       `HfConfig` — the defect class of #810 and #941.
DBuf NemotronHAttnBlock(Dev d, const NemotronHAttentionWeights& w,
                        const NemotronHParams& params, const Tensor& normed,
                        int64_t T, DType adt) {
  const int64_t H = params.hidden_size;
  const int64_t Hq = params.num_attention_heads;
  const int64_t Hkv = params.num_key_value_heads;
  const int64_t Dh = params.head_dim;
  const int64_t qdim = params.q_proj_out_features();
  const int64_t kvdim = params.kv_proj_out_features();

  // The same two refusals the host arm raises, in the same order, so an
  // unsupported checkpoint fails identically on both arms.
  VT_CHECK(!params.attention_bias,
           "NemotronH device forward: attention_bias is not ported (the "
           "checkpoint has attention_bias=false and ships no q/k/v/o bias)");
  VT_CHECK(!params.sliding_window.has_value(),
           "NemotronH device forward: per-layer sliding_window is not ported "
           "(this checkpoint ships sliding_window=null)");

  RequireDeviceWeight(w.q_proj, "mixer.q_proj", adt, {qdim, H});
  RequireDeviceWeight(w.k_proj, "mixer.k_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.v_proj, "mixer.v_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.o_proj, "mixer.o_proj", adt, {H, qdim});

  // `nk` IS CONSUMED HERE, exactly as the host arm consumes it
  // (nemotron_h.cpp:306). All four projections below go through `vt::MatmulBT`,
  // which reads `b` as [N=out, K=in] — the raw torch-Linear orientation
  // `nk = true` names. A weight recorded as [K, N] is a transposed GEMM operand
  // with the same shape and the same byte count, so nothing else here would
  // notice it.
  VT_CHECK(w.q_proj.nk && w.k_proj.nk && w.v_proj.nk && w.o_proj.nk,
           "NemotronH device forward: an attention projection is not in the "
           "[out, in] torch-Linear orientation vt::MatmulBT consumes");

  // THE RESIDENCY SEAM. Each of the four uploads ONCE, on the first step, and
  // every later step reuses the same device allocation. This is the shared
  // `dense_attn::ResidentWeight` (dense_attn_block.h:178) and not a local
  // equivalent: a `d_dev` bolted onto NemotronHOwned would have been the
  // parallel path AGENTS.md forbids, even though it is the smaller diff.
  Tensor wq = ResidentWeight(d, w.q_proj);
  Tensor wk = ResidentWeight(d, w.k_proj);
  Tensor wv = ResidentWeight(d, w.v_proj);
  Tensor wo = ResidentWeight(d, w.o_proj);

  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kvdim});
  DBuf v(d, adt, {T, kvdim});
  vt::MatmulBT(d.q, q.t(), normed, wq);
  vt::MatmulBT(d.q, k.t(), normed, wk);
  vt::MatmulBT(d.q, v.t(), normed, wv);

  // (a) NO RoPE HERE. This gap is the port, not an omission.

  DBuf attn(d, adt, {T, Hq, Dh});
  {
    Tensor qt = Reshape(q.t(), {T, Hq, Dh});
    Tensor kt = Reshape(k.t(), {T, Hkv, Dh});
    Tensor vt_ = Reshape(v.t(), {T, Hkv, Dh});
    vt::AttentionArgs args;
    // `self.scaling = self.head_dim**-0.5` (nemotron_h.py:440) — the SAME
    // expression the host arm evaluates (nemotron_h.cpp:622), in f64 before the
    // narrowing, so the two arms feed `vt::Attention` bit-identical scales.
    args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
    args.causal = true;
    vt::Attention(d.q, attn.t(), qt, kt, vt_, args);
  }

  DBuf out(d, adt, {T, H});
  {
    Tensor at = Reshape(attn.t(), {T, qdim});
    vt::MatmulBT(d.q, out.t(), at, wo);
  }
  return out;
}

}  // namespace

// ─── the per-block equivalence seam ─────────────────────────────────────────

std::vector<float> NemotronHAttnBlockHostIO(const NemotronHAttentionWeights& w,
                                            const NemotronHParams& params,
                                            const std::vector<float>& hidden_normed,
                                            int64_t num_tokens, DType act_dtype,
                                            Queue& dev_queue) {
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  VT_CHECK(T > 0, "NemotronH device attention: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH device attention: hidden size mismatch");
  VT_CHECK(act_dtype == DType::kBF16 || act_dtype == DType::kF32,
           "NemotronH device attention: the model dtype must be bf16 or f32");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device attention: this is the DEVICE arm and requires a "
           "non-CPU queue; the host reference is NemotronHAttentionMixer");

  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  // Round the input through `act_dtype` on the way in, exactly as the host arm
  // does with `PackF32` (nemotron_h.cpp:610). Feeding the device f32 values the
  // host arm would have rounded first is the kind of "more precise" deviation
  // that makes an equivalence gate quietly meaningless.
  DBuf x = UploadAs(d, hidden_normed, act_dtype, {T, H});
  DBuf out = NemotronHAttnBlock(d, w, params, x.t(), T, act_dtype);
  return DownloadF32(d, out, act_dtype, T * H);
}

// ─── the hybrid forward ─────────────────────────────────────────────────────

std::vector<float> NemotronHDeviceForward(const NemotronHHostWeights& host,
                                          const NemotronHParams& params,
                                          const std::vector<int32_t>& token_ids,
                                          const std::vector<int32_t>& logits_indices,
                                          Queue& dev_queue, Queue& host_queue,
                                          NemotronHTrace* trace) {
  VT_CHECK(host.materialized,
           "NemotronH device forward: host weights are not materialized");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device forward: `dev_queue` must be a non-CPU queue");
  VT_CHECK(host_queue.device.type == vt::DeviceType::kCPU,
           "NemotronH device forward: `host_queue` must be a CPU queue — it is "
           "what the 23 Mamba2 blocks, the 23 MoE blocks and lm_head run on");

  const DType adt = host.act_dtype;
  VT_CHECK(adt == DType::kBF16 || adt == DType::kF32,
           "NemotronH device forward: the model dtype must be bf16 or f32");
  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  const int64_t L = params.num_hidden_layers();
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "NemotronH device forward: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "NemotronH device forward: host layer count != layers_block_type length");

  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};

  // --- the embedding lookup, on the device, off the resident table.
  RequireDeviceWeight(host.embeddings, "backbone.embeddings.weight", adt, {V, H});
  RequireDeviceWeight(host.norm_f, "backbone.norm_f.weight", adt, {H});
  DBuf residual(d, adt, {T, H});
  {
    std::vector<int32_t> ids = token_ids;
    for (int32_t id : ids) {
      VT_CHECK(id >= 0 && id < V, "NemotronH device forward: token id out of range");
    }
    DBuf it(d, DType::kI32, {T}, ids.data());
    d.b.Synchronize(d.q);  // `ids` is a local; see UploadAs for why this waits.
    Tensor tab = ResidentWeight(d, host.embeddings);
    vt::Embedding(d.q, residual.t(), tab, it.t());
  }

  vt::RmsNormArgs nargs;
  nargs.eps = static_cast<float>(params.layer_norm_epsilon);
  nargs.gemma = false;

  if (trace != nullptr && trace->capture) {
    trace->normed.assign(static_cast<size_t>(L), {});
    trace->mixer.assign(static_cast<size_t>(L), {});
    trace->hidden.assign(static_cast<size_t>(L), {});
  }

  // The single-branch pre-norm stream (nemotron_h.py:625-640). Layer 0 sees
  // `residual is None`, so the embedding IS the residual and the norm is
  // un-fused; every later layer folds the previous mixer output into the
  // residual inside the norm. Identical control flow to the host arm
  // (nemotron_h.cpp:872-...), which is what makes the two comparable.
  DBuf carry(d, adt, {T, H});
  for (int64_t l = 0; l < L; ++l) {
    const NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(l)];
    VT_CHECK(lw.block == params.layers_block_type[static_cast<size_t>(l)],
             "NemotronH device forward: host layer block kind disagrees with "
             "layers_block_type");
    RequireDeviceWeight(lw.norm, "layer norm", adt, {H});

    DBuf normed(d, adt, {T, H});
    {
      Tensor wt = ResidentWeight(d, lw.norm);
      if (l == 0) {
        // `residual is None` (nemotron_h.py:627-631): the embedding IS the
        // residual, so there is no add to fuse and this is not a fusion site.
        vt::RmsNorm(d.q, normed.t(), residual.t(), wt, nargs, nullptr);
      } else {
        Tensor rt = residual.t();
        Tensor xt = carry.t();
        Tensor ot = normed.t();
        AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
      }
    }

    // The 6 attention layers stay on the device end to end. The other 46
    // bounce: download the normed hidden, run the HOST mixer on `host_queue`,
    // upload the result. One helper, one place, so the scaffold is visible and
    // deletable rather than scattered through the loop.
    std::vector<float> nvec;
    const bool needs_host = lw.block != NemotronHBlock::kAttention;
    if (needs_host || (trace != nullptr && trace->capture)) {
      nvec = DownloadF32(d, normed, adt, T * H);
    }

    // Assigned STRAIGHT INTO `carry`, which is the only thing that reads the
    // mixer output. A `DBuf mixer_out(d, adt, {T, H})` declared here and
    // move-assigned over on both branches was one dead device allocation per
    // layer per step; the previous `carry` block still returns to the pool at
    // exactly the same statement it did before, after every enqueue for this
    // layer, so the lifetimes are unchanged. No speed claim is made or implied.
    std::vector<float> mvec;
    if (lw.block == NemotronHBlock::kAttention) {
      carry = NemotronHAttnBlock(d, lw.attn, params, normed.t(), T, adt);
      if (trace != nullptr && trace->capture) {
        mvec = DownloadF32(d, carry, adt, T * H);
      }
    } else {
      switch (lw.block) {
        case NemotronHBlock::kMamba:
          mvec = NemotronHMamba2Mixer(lw.mamba, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kMoe:
          mvec = NemotronHMoeMixer(lw.moe, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kMlp:
          mvec = NemotronHMlpMixer(lw.mlp, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kAttention:
          break;  // handled above
      }
      carry = UploadAs(d, mvec, adt, {T, H});
    }

    if (trace != nullptr && trace->capture) {
      trace->normed[static_cast<size_t>(l)] = std::move(nvec);
      // The residual AFTER this layer is what the next norm folds `carry` into.
      std::vector<float> h = DownloadF32(d, residual, adt, T * H);
      for (size_t i = 0; i < h.size(); ++i) h[i] += mvec[i];
      trace->mixer[static_cast<size_t>(l)] = std::move(mvec);
      trace->hidden[static_cast<size_t>(l)] = std::move(h);
    }
  }

  // `hidden_states, _ = self.norm_f(hidden_states, residual)` (nemotron_h.py:641).
  DBuf final_normed(d, adt, {T, H});
  {
    Tensor wt = ResidentWeight(d, host.norm_f);
    Tensor rt = residual.t();
    Tensor xt = carry.t();
    Tensor ot = final_normed.t();
    AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
  }
  const std::vector<float> fvec = DownloadF32(d, final_normed, adt, T * H);
  if (trace != nullptr && trace->capture) trace->final_normed = fvec;

  // --- lm_head, on the HOST: it is NVFP4 W4A16 g16 on the released
  // checkpoint. Both arms therefore end in the IDENTICAL host projection, which
  // is what makes A2-R's token gate attributable: a token difference can only
  // have come from the 6 device attention blocks and the device residual
  // stream, never from the output projection.
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) want[static_cast<size_t>(i)] = i;
  } else {
    for (int32_t idx : logits_indices) {
      VT_CHECK(idx >= 0 && idx < T,
               "NemotronH device forward: logits index out of range");
      want.push_back(idx);
    }
  }
  std::vector<float> gathered(want.size() * static_cast<size_t>(H));
  for (size_t r = 0; r < want.size(); ++r) {
    std::memcpy(gathered.data() + r * static_cast<size_t>(H),
                fvec.data() + static_cast<size_t>(want[r]) * static_cast<size_t>(H),
                static_cast<size_t>(H) * sizeof(float));
  }
  return NemotronHHostLmHead(host, params, gathered,
                             static_cast<int64_t>(want.size()), host_queue);
}

}  // namespace vllm
