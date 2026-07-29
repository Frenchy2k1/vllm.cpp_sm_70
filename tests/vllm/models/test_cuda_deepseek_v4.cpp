// DeepSeek-V4-Flash W7-device — per-kernel CUDA gate + the ForwardDevice
// composition gate. Each of the four NEW V4 op families' CUDA kernels
// (src/vt/cuda/cuda_deepseek_v4.cu, dispatched through the vt OpProvider seam) is
// run at a SMALL synthetic shape and compared against its LANDED portable HOST
// reference (the oracle): BIT-EXACT where the math is integer/exact (top-k
// selection, hash route ids), NMSE / near-tie for fp reductions (Sinkhorn,
// softmax pool, sqrtsoftplus — device expf/sqrtf differ from host by ULPs). The
// device cases SKIP on a CPU-only build (no CUDA backend registered).
//
// HONEST SCOPE (mirrors W3-W7): this is the runtime evidence the kernels match
// the references at small shape on a real GPU — NOT a real-checkpoint token gate
// (the fixed-config 167B does not fit ONE GB10; that is the W8 residual). See
// .agents/specs/deepseek-v4-flash.md §W7.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_compressor.h"
#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/deepseek_v4_moe.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/backend.h"

namespace dv4 = vllm::deepseek_v4;
using vllm::DeepseekV4HostWeights;
using vllm::DeepseekV4LayerHostWeights;
using vllm::DeepseekV4Params;

namespace {

bool HasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return vllm::deepseek_v4::V4DeviceKernelsAvailable();
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct QueueGuard {
  vt::Backend& b;
  vt::Queue q;
  explicit QueueGuard(vt::Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

struct Rng {
  uint32_t s = 0x9E3779B9u;
  float next(float lo, float hi) {
    s = s * 1664525u + 1013904223u;
    const float u = static_cast<float>(s >> 8) / 16777216.0f;
    return lo + u * (hi - lo);
  }
};
std::vector<float> Rand(Rng& r, int64_t n, float lo = -1.0f, float hi = 1.0f) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& e : v) e = r.next(lo, hi);
  return v;
}

// Relative L2 over two equal-length buffers.
double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(a[i]) * a[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}
float MaxAbs(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

constexpr double kTol = 1e-4;  // near-tie for fp reductions (device vs host transcendentals)

}  // namespace

// ===========================================================================
// (1) MHC family
// ===========================================================================
TEST_CASE("W7-device MHC Sinkhorn: CUDA vs host reference (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t hc = 4, iters = 5;
  const float eps = 1e-6f;
  Rng r;
  const auto logits = Rand(r, hc * hc, -2.0f, 2.0f);
  const auto ref = dv4::MhcSinkhorn(logits, hc, iters, eps);
  const auto got = dv4::MhcDevice()->sinkhorn(g.q, logits, hc, iters, eps);
  REQUIRE(got.size() == ref.size());
  CHECK(RelL2(got, ref) < kTol);
  // Doubly-stochastic property (col sums == 1 at eps~0) holds on the device too.
  for (int64_t k = 0; k < hc; ++k) {
    float c = 0.0f;
    for (int64_t j = 0; j < hc; ++j) c += got[j * hc + k];
    CHECK(c == doctest::Approx(1.0f).epsilon(1e-3));
  }
}

TEST_CASE("W7-device MhcPre: CUDA vs host reference (all four outputs near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t hc = 4, hidden = 8, iters = 5;
  const int64_t hc3 = (2 + hc) * hc, hcH = hc * hidden;
  Rng r;
  const auto residual = Rand(r, hc * hidden, -1.0f, 1.0f);
  const auto fn = Rand(r, hc3 * hcH, -0.3f, 0.3f);
  const auto scale = Rand(r, 3, -0.5f, 0.5f);
  const auto base = Rand(r, hc3, -0.3f, 0.3f);
  const auto nw = Rand(r, hidden, 0.9f, 1.1f);
  const float eps = 1e-6f;
  const auto ref = dv4::MhcPre(residual, fn, scale, base, hc, hidden, eps, eps, eps, 2.0f,
                               iters, nw, eps);
  const auto got = dv4::MhcDevice()->pre(g.q, residual, fn, scale, base, hc, hidden, eps, eps,
                                         eps, 2.0f, iters, nw, eps);
  CHECK(RelL2(got.pre_mix, ref.pre_mix) < kTol);
  CHECK(RelL2(got.post_mix, ref.post_mix) < kTol);
  CHECK(RelL2(got.comb_mix, ref.comb_mix) < kTol);
  CHECK(RelL2(got.layer_input, ref.layer_input) < kTol);
}

TEST_CASE("W7-device MhcPost + HcHeadCollapse: CUDA vs host reference (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t hc = 4, hidden = 8;
  Rng r;
  const auto x = Rand(r, hidden), residual = Rand(r, hc * hidden);
  const auto post_mix = Rand(r, hc, 0.0f, 2.0f), comb = Rand(r, hc * hc, 0.0f, 1.0f);
  const auto post_ref = dv4::MhcPost(x, residual, post_mix, comb, hc, hidden);
  const auto post_got = dv4::MhcDevice()->post(g.q, x, residual, post_mix, comb, hc, hidden);
  CHECK(RelL2(post_got, post_ref) < kTol);

  const auto xh = Rand(r, hc * hidden), fn = Rand(r, hc * hc * hidden, -0.3f, 0.3f);
  const auto base = Rand(r, hc, -0.3f, 0.3f);
  const auto head_ref = dv4::HcHeadCollapse(xh, fn, 0.5f, base, hc, hidden, 1e-6f, 1e-6f);
  const auto head_got = dv4::MhcDevice()->head(g.q, xh, fn, 0.5f, base, hc, hidden, 1e-6f, 1e-6f);
  CHECK(RelL2(head_got, head_ref) < kTol);
}

// ===========================================================================
// (2) DSA family
// ===========================================================================
TEST_CASE("W7-device DSA weight-fold + MQA logits: CUDA vs host (near-tie, -inf exact)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t T = 3, inh = 2, ihd = 4, nk = 3;
  Rng r;
  const auto wp = Rand(r, T * inh, -1.0f, 1.0f);
  const auto fold_ref = dv4::DsaIndexerWeightFold(wp, T, inh, ihd);
  const auto fold_got = dv4::DsaDevice()->weight_fold(g.q, wp, T, inh, ihd);
  CHECK(RelL2(fold_got, fold_ref) < kTol);

  const auto q = Rand(r, T * inh * ihd), k = Rand(r, nk * ihd);
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = t + 1; }  // causal
  const auto lref = dv4::DsaIndexerLogits(q, k, fold_ref, ws, we, T, nk, inh, ihd);
  const auto lgot = dv4::DsaDevice()->logits(g.q, q, k, fold_ref, ws, we, T, nk, inh, ihd);
  REQUIRE(lgot.size() == lref.size());
  for (size_t i = 0; i < lref.size(); ++i) {
    if (std::isinf(lref[i])) {
      CHECK(std::isinf(lgot[i]));  // out-of-window keys are -inf on BOTH
    } else {
      CHECK(lgot[i] == doctest::Approx(lref[i]).epsilon(1e-4));
    }
  }
}

TEST_CASE("W7-device DSA top-k select: CUDA matches host ids BIT-EXACT") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const int64_t T = 4, nk = 5, topk = 3;
  Rng r;
  const auto logits = Rand(r, T * nk, -3.0f, 3.0f);  // distinct random -> no ties
  std::vector<int64_t> ws(T), we(T);
  for (int64_t t = 0; t < T; ++t) { ws[t] = 0; we[t] = nk; }  // full window, n>topk
  const auto ref = dv4::DsaTopkSelect(logits, ws, we, T, nk, topk);
  const auto got = dv4::DsaDevice()->topk(g.q, logits, ws, we, T, nk, topk);
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == ref[i]);  // selection is EXACT

  // Short-context (n<=topk) -> all candidates ascending, -1 pad; also exact.
  std::vector<int64_t> ws2(T), we2(T);
  for (int64_t t = 0; t < T; ++t) { ws2[t] = 0; we2[t] = t + 1; }  // t+1 <= topk for t<3
  const auto sref = dv4::DsaTopkSelect(logits, ws2, we2, T, nk, topk);
  const auto sgot = dv4::DsaDevice()->topk(g.q, logits, ws2, we2, T, nk, topk);
  for (size_t i = 0; i < sref.size(); ++i) CHECK(sgot[i] == sref[i]);
}

TEST_CASE("W7-device attention-sink softmax + grouped output-LoRA: CUDA vs host (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const auto scores = Rand(r, 5, -2.0f, 2.0f);
  const auto sref = dv4::SoftmaxWithSink(scores, 0.5f);
  const auto sgot = dv4::DsaDevice()->softmax_sink(g.q, scores, 0.5f);
  CHECK(RelL2(sgot, sref) < kTol);
  // The sink removes probability mass: Σ prob < 1 on BOTH (property, not just tie).
  float sum = 0.0f;
  for (float p : sgot) sum += p;
  CHECK(sum < 1.0f);

  const int64_t T = 2, nh = 2, hd = 6, ng = 2, olr = 4, H = 8;
  const int64_t ipg = nh * hd / ng;
  const auto o = Rand(r, T * nh * hd), wa = Rand(r, ng * olr * ipg, -0.3f, 0.3f);
  const auto wb = Rand(r, H * ng * olr, -0.3f, 0.3f);
  const auto oref = dv4::GroupedOutputLora(o, wa, wb, T, nh, hd, ng, olr, H);
  const auto ogot = dv4::DsaDevice()->grouped_olora(g.q, o, wa, wb, T, nh, hd, ng, olr, H);
  CHECK(RelL2(ogot, oref) < kTol);
}

// ===========================================================================
// (3) Compressor family
// ===========================================================================
TEST_CASE("W7-device compressor save-APE + pool-norm: CUDA vs host (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t T = 3, width = 6, cr = 2;
  const auto score = Rand(r, T * width), ape = Rand(r, cr * width, -0.5f, 0.5f);
  std::vector<int64_t> pos = {0, 1, 2};
  const auto aref = dv4::CompressorSaveScoreApe(score, ape, pos, T, width, cr);
  const auto agot = dv4::CompressorDevice()->save_score_ape(g.q, score, ape, pos, T, width, cr);
  CHECK(RelL2(agot, aref) < kTol);

  const int64_t window = 2, hd = 6;
  const auto kv = Rand(r, window * hd), sc = Rand(r, window * hd);
  const auto rms = Rand(r, hd, 0.9f, 1.1f);
  for (auto valid : std::vector<std::vector<uint8_t>>{{1, 1}, {0, 1}}) {
    const auto pref = dv4::CompressorPoolNorm(kv, sc, valid, rms, 1e-6f, window, hd);
    const auto pgot = dv4::CompressorDevice()->pool_norm(g.q, kv, sc, valid, rms, 1e-6f, window, hd);
    CHECK(RelL2(pgot, pref) < kTol);
  }
}

TEST_CASE("W7-device fp8_ds_mla KV encode+decode round-trip: CUDA vs host (fp8 granularity)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const dv4::Fp8DsMlaLayout L = dv4::MakeFp8DsMlaLayout(/*nope=*/8, /*rope=*/4, /*qblk=*/4);
  const auto head = Rand(r, L.nope_head_dim + L.rope_head_dim, -2.0f, 2.0f);

  const auto htok = dv4::Fp8DsMlaEncodeToken(head, L);
  const auto dtok = dv4::CompressorDevice()->encode(g.q, head, L);
  // Decode BOTH tokens on host + device; the reconstructed latent agrees within
  // fp8 e4m3 granularity (3 mantissa bits -> ~0.05 relative, the W4 host bound).
  const auto hdec_h = dv4::Fp8DsMlaDecodeToken(htok, L);           // host token, host decode
  const auto ddec_d = dv4::CompressorDevice()->decode(g.q, dtok, L);  // device token, device decode
  CHECK(MaxAbs(ddec_d, hdec_h) < 0.05f * 2.0f);
  // Device decode of the HOST token vs host decode: e4m3 bit layout is standard,
  // so this is near-exact (isolates the decode path from the encode rounding).
  const auto hdec_on_dev = dv4::CompressorDevice()->decode(g.q, htok, L);
  CHECK(RelL2(hdec_on_dev, hdec_h) < 1e-3);
  // The RoPE tail is bf16-verbatim on both -> exact bit pattern round-trip.
  for (int64_t j = 0; j < L.rope_head_dim; ++j)
    CHECK(dtok.rope_bf16[j] == htok.rope_bf16[j]);
}

// ===========================================================================
// (4) MoE family
// ===========================================================================
TEST_CASE("W7-device sqrtsoftplus + clamped SwiGLU: CUDA vs host (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const auto x = Rand(r, 16, -4.0f, 4.0f);
  std::vector<float> spref(x.size());
  for (size_t i = 0; i < x.size(); ++i) spref[i] = dv4::SqrtSoftplus(x[i]);
  const auto spgot = dv4::MoeDevice()->sqrtsoftplus(g.q, x);
  CHECK(RelL2(spgot, spref) < kTol);

  // Clamped SwiGLU with values that exercise BOTH the gate max-clamp and the
  // up two-sided clamp (asymmetry load-bearing).
  const int64_t d = 6;
  std::vector<float> gate_up = {-5.0f, 3.0f, 12.0f, 0.5f, -1.0f, 8.0f,   // gate
                                -20.0f, 2.0f, 15.0f, -0.5f, 1.0f, -11.0f};  // up
  const auto cref = dv4::ClampedSwiGLU(gate_up, d, 10.0f, 1.0f, 0.0f);
  const auto cgot = dv4::MoeDevice()->clamped_swiglu(g.q, gate_up, d, 10.0f, 1.0f, 0.0f);
  CHECK(RelL2(cgot, cref) < kTol);
}

TEST_CASE("W7-device sqrtsoftplus/hash router: CUDA ids BIT-EXACT, weights near-tie") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  Rng r;
  const int64_t T = 4, E = 8, topk = 3, vocab = 12;
  const auto gating = Rand(r, T * E, -3.0f, 3.0f);
  const auto bias = Rand(r, E, -0.3f, 0.3f);

  // (a) learned top-k with the noaux_tc bias (selection biased, weights unbiased).
  {
    const auto ref = dv4::SqrtSoftplusRouteTopk(gating, T, E, topk, bias, true, 1.5f, {}, {}, vocab);
    const auto got = dv4::MoeDevice()->route(g.q, gating, T, E, topk, bias, true, 1.5f, {}, {}, vocab);
    REQUIRE(got.topk_ids.size() == ref.topk_ids.size());
    for (size_t i = 0; i < ref.topk_ids.size(); ++i) CHECK(got.topk_ids[i] == ref.topk_ids[i]);
    CHECK(RelL2(got.topk_weights, ref.topk_weights) < kTol);
  }
  // (b) hash route: tid2eid lookup bypasses top-k; weights from UNBIASED scores.
  {
    std::vector<int64_t> in_tokens = {3, 7, 1, 9};
    std::vector<int32_t> tid2eid(static_cast<size_t>(vocab * topk));
    for (int64_t tok = 0; tok < vocab; ++tok)
      for (int64_t j = 0; j < topk; ++j)
        tid2eid[static_cast<size_t>(tok * topk + j)] = static_cast<int32_t>((tok * 5 + j) % E);
    const auto ref = dv4::SqrtSoftplusRouteTopk(gating, T, E, topk, {}, true, 1.5f, in_tokens, tid2eid, vocab);
    const auto got = dv4::MoeDevice()->route(g.q, gating, T, E, topk, {}, true, 1.5f, in_tokens, tid2eid, vocab);
    for (size_t i = 0; i < ref.topk_ids.size(); ++i) CHECK(got.topk_ids[i] == ref.topk_ids[i]);
    CHECK(RelL2(got.topk_weights, ref.topk_weights) < kTol);
  }
}

// ===========================================================================
// ForwardDevice composition gate: device forward == host forward (near-tie), at
// the tiny structural config (the four families all routed through CUDA).
// ===========================================================================
namespace {
DeepseekV4Params TinyParams() {
  DeepseekV4Params p;
  p.hidden_size = 8;
  p.num_hidden_layers = 4;
  p.vocab_size = 12;
  p.num_attention_heads = 2;
  p.num_key_value_heads = 1;
  p.rms_norm_eps = 1e-6f;
  p.max_position_embeddings = 4096;
  p.head_dim = 6;
  p.qk_rope_head_dim = 2;
  p.q_lora_rank = 4;
  p.o_lora_rank = 4;
  p.o_groups = 2;
  p.sliding_window = 128;
  p.rope_theta = 10000.0;
  p.compress_rope_theta = 160000.0;
  p.n_routed_experts = 4;
  p.num_experts_per_tok = 2;
  p.moe_intermediate_size = 6;
  p.n_shared_experts = 1;
  p.norm_topk_prob = true;
  p.routed_scaling_factor = 1.5;
  p.swiglu_limit = 10.0;
  p.scoring_func = "sqrtsoftplus";
  p.num_hash_layers = 2;
  p.expert_dtype = "fp4";
  p.hc_mult = 4;
  p.hc_sinkhorn_iters = 5;
  p.hc_eps = 1e-6;
  p.index_head_dim = 4;
  p.index_n_heads = 2;
  p.index_topk = 3;
  p.compress_ratios = {0, 4, 2, 4};
  return p;
}

DeepseekV4HostWeights TinyWeights(const DeepseekV4Params& p) {
  Rng rng;
  const int64_t H = p.hidden_size, V = p.vocab_size, hc = p.hc_mult;
  const int64_t nh = p.num_attention_heads, hd = p.head_dim, qlr = p.q_lora_rank;
  const int64_t og = p.o_groups, olr = p.o_lora_rank;
  const int64_t in_per_group = nh * hd / og;
  const int64_t ne = p.n_routed_experts, topk = p.num_experts_per_tok, mi = p.moe_intermediate_size;
  const int64_t inh = p.index_n_heads, ihd = p.index_head_dim;
  const int64_t hc3 = (2 + hc) * hc, hcH = hc * H;
  auto rnd = [&](int64_t n, float sc) { return Rand(rng, n, -sc, sc); };
  auto normw = [&](int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& e : v) e = 1.0f + rng.next(-0.1f, 0.1f);
    return v;
  };
  DeepseekV4HostWeights hw;
  hw.embed = rnd(V * H, 0.8f);
  hw.lm_head = rnd(V * H, 0.5f);
  hw.final_norm_weight = normw(H);
  hw.hc_head_fn = rnd(hc * hcH, 0.2f);
  hw.hc_head_base = rnd(hc, 0.2f);
  hw.hc_head_scale = 0.5f;
  hw.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(l)];
    L.attn_norm_weight = normw(H);
    L.ffn_norm_weight = normw(H);
    L.hc_attn_fn = rnd(hc3 * hcH, 0.2f);
    L.hc_attn_base = rnd(hc3, 0.2f);
    L.hc_attn_scale = rnd(3, 0.5f);
    L.hc_ffn_fn = rnd(hc3 * hcH, 0.2f);
    L.hc_ffn_base = rnd(hc3, 0.2f);
    L.hc_ffn_scale = rnd(3, 0.5f);
    L.wq_a = rnd(qlr * H, 0.3f);
    L.q_norm_weight = normw(qlr);
    L.wq_b = rnd((nh * hd) * qlr, 0.3f);
    L.wkv = rnd(hd * H, 0.3f);
    L.kv_norm_weight = normw(hd);
    L.attn_sink = {0.7f, -0.4f};
    L.wo_a = rnd(og * olr * in_per_group, 0.3f);
    L.wo_b = rnd(H * (og * olr), 0.3f);
    if (p.has_indexer(l)) {
      L.idx_wq = rnd((inh * ihd) * H, 0.3f);
      L.idx_wk = rnd(ihd * H, 0.3f);
      L.idx_wproj = rnd(inh * H, 0.3f);
    }
    if (p.has_compressor(l)) {
      const int64_t cr = p.compress_ratio(l);
      L.comp_wgate = rnd(hd * H, 0.3f);
      L.comp_ape = rnd(cr * hd, 0.2f);
      L.comp_norm_weight = normw(hd);
    }
    L.gate_weight = rnd(ne * H, 0.4f);
    if (p.is_hash_layer(l)) {
      L.tid2eid.assign(static_cast<size_t>(V * topk), 0);
      for (int64_t tok = 0; tok < V; ++tok)
        for (int64_t j = 0; j < topk; ++j)
          L.tid2eid[static_cast<size_t>(tok * topk + j)] =
              static_cast<int32_t>((tok * 7 + j * 3 + 1) % ne);
    } else {
      L.gate_bias = rnd(ne, 0.3f);
    }
    L.shared_w1 = rnd(mi * H, 0.3f);
    L.shared_w3 = rnd(mi * H, 0.3f);
    L.shared_w2 = rnd(H * mi, 0.3f);
    L.exp_w1 = rnd(ne * mi * H, 0.3f);
    L.exp_w3 = rnd(ne * mi * H, 0.3f);
    L.exp_w2 = rnd(ne * H * mi, 0.3f);
  }
  return hw;
}
const std::vector<int32_t> kTokens = {3, 7, 1, 9, 4};
const std::vector<int32_t> kPositions = {0, 1, 2, 3, 4};
}  // namespace

TEST_CASE("W7-device ForwardDevice ASSEMBLES: device forward == host forward (near-tie)") {
  if (!HasCuda()) { MESSAGE("no CUDA; skip"); return; }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  QueueGuard g(gpu);
  const DeepseekV4Params p = TinyParams();
  vllm::DeepseekV4Weights w;
  w.params = p;
  w.host = TinyWeights(p);
  w.has_host_weights = true;

  const std::vector<float> host =
      vllm::DeepseekV4ForwardHost(w.host, p, kTokens, kPositions, {}, vllm::V4Miswire::kNone, nullptr);
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  const vllm::ForwardLogits dev =
      vllm::DeepseekV4Model::ForwardDevice(kTokens, kPositions, attn_meta, attn_kv, w, g.q, {});

  REQUIRE(dev.host.size() == host.size());
  CHECK(dev.rows == static_cast<int64_t>(kTokens.size()));
  CHECK(dev.vocab == p.vocab_size);
  for (float v : dev.host) CHECK(std::isfinite(v));
  // The whole 4-family device composition matches the host oracle within the
  // accumulated near-tie envelope (device expf/sqrtf/rsqrt vs host, over 4 layers).
  CHECK(RelL2(dev.host, host) < 2e-3);
}
