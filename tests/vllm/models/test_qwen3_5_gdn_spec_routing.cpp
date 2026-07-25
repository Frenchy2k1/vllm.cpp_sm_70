// SPEC-MTP I5a — GDN layer spec routing (GdnBlockPaged spec branch).
//
// Proves that GdnBlockPaged's num_spec_decodes>0 branch (mirror of
// vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1329-1576 @
// e24d1b24) routes a PURE spec batch through vt::CausalConv1dSpecUpdate +
// vt::GdnSpecDecode with the correct per-request slots / cu_seqlens /
// num_accepted, BIT-FOR-BIT against the shipped non-spec decode path.
//
// THE REFERENCE (the "I4 ops applied directly" chain). A speculative step over a
// single request with num_accepted==1 processes its 1+k query tokens
// sequentially from the request's initial state, snapshotting each timestep.
// That is EXACTLY a token-by-token run of the shipped non-spec decode path
// (vt::GdnDecode + vt::CausalConv1dUpdate — the ops I4 proved the spec kernels
// reduce to at T==1/accepted==1) carrying the state forward. So one spec call
// over T tokens must equal T single-token decode calls over the same weights and
// the same initial state, row for row. A mis-wired split / slot select / conv
// window / merge diverges immediately. Real gate-checkpoint GDN dims (Hv 48/32,
// Dk==Dv 128, conv width 4). On CPU the projection GEMM is batch-invariant so the
// match is BIT-EXACT; the CUDA build additionally proves the spec ops execute
// on-device (asserted within a tight bf16-ULP band, since the M=1 vs M=T
// projection GEMM may retile — the spec routing itself is exact).

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::GdnLayerWeights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed, float lo, float hi) {
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(lo + u * (hi - lo));
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed, float lo = -0.08f,
                      float hi = 0.08f) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i), lo, hi));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i), lo, hi);
  }
  return t;
}

struct GdnDims {
  int64_t hk, hv, dk, dv, kw;
  const char* name;
};

// Minimal dense (num_experts==0) GDN config at the real gate-checkpoint GDN dims.
HfConfig MakeConfig(const GdnDims& g, int64_t H) {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.hidden_size = H;
  c.num_hidden_layers = 1;
  c.num_experts = 0;
  c.linear_num_key_heads = g.hk;
  c.linear_num_value_heads = g.hv;
  c.linear_key_head_dim = g.dk;
  c.linear_value_head_dim = g.dv;
  c.linear_conv_kernel_dim = g.kw;
  c.rms_norm_eps = 1e-6;
  return c;
}

GdnLayerWeights MakeGdnWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hv = c.linear_num_value_heads;
  const int64_t key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
  const int64_t value_dim = Hv * c.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t Kw = c.linear_conv_kernel_dim;
  GdnLayerWeights w;
  w.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, 10);
  w.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, 20);
  w.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, 30);
  w.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, 40);
  w.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, 50);
  // A_log / dt_bias positive-ish so g = -exp(A_log)*softplus(...) is a sane decay.
  w.a_log = MakeOwned(DType::kF32, {Hv}, 60, 0.1f, 1.0f);
  w.dt_bias = MakeOwned(DType::kF32, {Hv}, 70, -0.5f, 0.5f);
  w.norm_weight = MakeOwned(DType::kBF16, {c.linear_value_head_dim}, 80, 0.5f, 1.5f);
  w.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, 90);
  return w;
}

// One fresh single-token decode step at state slot 0.
GDNAttentionMetadata DecodeMeta() {
  GDNAttentionMetadata g;
  g.num_decodes = 1;
  g.num_decode_tokens = 1;
  g.num_actual_tokens = 1;
  g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  return g;
}

// One PURE spec request over T tokens (k = T-1) writing snapshots into slots
// 0..T-1, reading its initial state from slot 0 (num_accepted == 1).
GDNAttentionMetadata SpecMeta(int64_t T) {
  GDNAttentionMetadata g;
  g.num_spec_decodes = 1;
  g.num_spec_decode_tokens = static_cast<int>(T);
  g.num_actual_tokens = static_cast<int>(T);
  g.spec_state_indices_num_cols = static_cast<int>(T);  // num_spec + 1
  std::vector<int32_t> ssi(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) ssi[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  g.spec_state_indices_tensor = ssi;
  g.spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.spec_sequence_masks = std::vector<uint8_t>{1};
  g.spec_token_indx = ssi;  // identity gather over the pure batch
  g.num_accepted_tokens = std::vector<int32_t>{1};
  return g;
}

vt::Queue Q(vt::DeviceType dev) { return vt::Queue{vt::Device{dev, 0}, nullptr}; }

void RunSpecRoutingCase(vt::DeviceType dev, const GdnDims& g, bool bit_exact) {
  const int64_t H = 128;
  const int64_t T = 4;              // 1 + k, k = 3 draft tokens
  const HfConfig c = MakeConfig(g, H);
  const GdnLayerWeights w = MakeGdnWeights(c);
  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;

  // Shared inputs and initial state.
  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(1000 + i, -1.0f, 1.0f);
  std::vector<float> ssm0(static_cast<size_t>(ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(2000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(conv_dim * (Kw - 1)));
  for (size_t i = 0; i < conv0.size(); ++i) conv0[i] = RandV(3000 + i, -1.0f, 1.0f);

  // ── Reference: T single-token decode calls, carrying state (narrow conv). ──
  std::vector<float> ref_out;
  {
    const int64_t slots = 1;
    std::vector<float> ssm(static_cast<size_t>(slots * ssm_row));
    std::memcpy(ssm.data(), ssm0.data(), ssm0.size() * sizeof(float));
    std::vector<float> conv(static_cast<size_t>(slots * conv_dim * (Kw - 1)));
    std::memcpy(conv.data(), conv0.data(), conv0.size() * sizeof(float));
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> h1(h.begin() + static_cast<std::ptrdiff_t>(t * H),
                            h.begin() + static_cast<std::ptrdiff_t>((t + 1) * H));
      GDNAttentionMetadata dm = DecodeMeta();
      std::vector<float> row = vllm::GdnBlockPagedForTest(Q(dev), w, c, h1, dm, ssm, conv,
                                                          slots, Kw - 1, /*T=*/1);
      ref_out.insert(ref_out.end(), row.begin(), row.end());
    }
  }

  // ── Spec: one call over T tokens, num_accepted == 1, widened conv state. ──
  std::vector<float> spec_out;
  {
    const int64_t slots = T;                 // one slot per timestep snapshot
    const int64_t conv_len = (Kw - 1) + (T - 1);  // widened spec row
    std::vector<float> ssm(static_cast<size_t>(slots * ssm_row), 0.0f);
    // Initial state lives in slot 0 (num_accepted-1 == 0).
    std::memcpy(ssm.data(), ssm0.data(), ssm0.size() * sizeof(float));
    std::vector<float> conv(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
    // Slot 0's first K-1 taps are the initial conv window; the rest are unread.
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv[static_cast<size_t>(ch * conv_len + j)] =
            conv0[static_cast<size_t>(ch * (Kw - 1) + j)];
    GDNAttentionMetadata sm = SpecMeta(T);
    spec_out = vllm::GdnBlockPagedForTest(Q(dev), w, c, h, sm, ssm, conv, slots, conv_len, T);
  }

  REQUIRE(spec_out.size() == ref_out.size());
  if (bit_exact) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < spec_out.size(); ++i)
      if (std::memcmp(&spec_out[i], &ref_out[i], sizeof(float)) != 0) {
        if (bad == 0) first = i;
        ++bad;
      }
    CAPTURE(g.name);
    CAPTURE(bad);
    CAPTURE(first);
    if (bad != 0) {
      CAPTURE(spec_out[first]);
      CAPTURE(ref_out[first]);
    }
    CHECK(bad == 0);
  } else {
    float maxabs = 0.0f;
    for (size_t i = 0; i < spec_out.size(); ++i)
      maxabs = std::max(maxabs, std::fabs(spec_out[i] - ref_out[i]));
    CAPTURE(g.name);
    CAPTURE(maxabs);
    // Tight band: only the M=1-vs-M=T projection GEMM retile differs; the spec
    // conv/recurrence routing is exact. A broken split would blow past this.
    CHECK(maxabs < 0.05f);
  }
}

// ── MIXED spec+non-spec batch metadata: one spec request (2 tokens, k=1, state
// slots 0,1) followed by one PREFILL request (Tp tokens, state slot 2). Mirrors
// the GDN metadata builder's mixed output (gdn_attn.cpp:218-333). ──
GDNAttentionMetadata MixedMeta(int Tp) {
  const int Ts = 2;  // 1 + k, k = 1
  GDNAttentionMetadata g;
  g.num_spec_decodes = 1;
  g.num_spec_decode_tokens = Ts;
  g.num_prefills = 1;
  g.num_prefill_tokens = Tp;
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = Ts + Tp;
  g.spec_state_indices_num_cols = 2;
  g.spec_state_indices_tensor = std::vector<int32_t>{0, 1};
  g.spec_query_start_loc = std::vector<int32_t>{0, Ts};
  g.spec_sequence_masks = std::vector<uint8_t>{1, 0};
  g.spec_token_indx = std::vector<int32_t>{0, 1};
  std::vector<int32_t> nst;
  for (int t = Ts; t < Ts + Tp; ++t) nst.push_back(t);
  g.non_spec_token_indx = nst;
  g.num_accepted_tokens = std::vector<int32_t>{1};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{2};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, Tp};
  g.has_initial_state = std::vector<uint8_t>{0};
  g.prefill_state_indices = std::vector<int32_t>{2};
  g.prefill_query_start_loc = std::vector<int32_t>{0, Tp};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  return g;
}

// One fresh PURE prefill request over Tp tokens at state slot `slot`.
GDNAttentionMetadata PrefillMeta(int Tp, int slot) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = Tp;
  g.num_actual_tokens = Tp;
  g.non_spec_state_indices_tensor = std::vector<int32_t>{slot};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, Tp};
  g.has_initial_state = std::vector<uint8_t>{0};
  g.prefill_state_indices = std::vector<int32_t>{slot};
  g.prefill_query_start_loc = std::vector<int32_t>{0, Tp};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  return g;
}

// THE MIXED-BATCH PROOF (model-independent). The mixed spec+non-spec batch
// processes the spec request and the prefill request over DISJOINT state slots,
// so its per-row output MUST equal the spec request run as a PURE spec batch
// (rows 0..1) followed by the prefill request run as a PURE prefill batch
// (rows 2..). A mis-wired index_select split, a wrong sub-batch metadata feed,
// or a mis-indexed index_copy merge diverges immediately. This is independent
// of any model-level bf16 batch-nondeterminism (the e2e c>1 confound): the
// CPU projection GEMM is row-invariant, so the split/merge is BIT-EXACT.
void RunMixedRoutingCase(vt::DeviceType dev, const GdnDims& g, bool bit_exact) {
  setenv("VT_GDN_INDEXED_STATE_IO", "1", 1);  // mixed needs widened indexed IO
  const int64_t H = 128;
  const int Ts = 2, Tp = 3, T = Ts + Tp;
  const HfConfig c = MakeConfig(g, H);
  const GdnLayerWeights w = MakeGdnWeights(c);
  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;
  const int64_t conv_len = (Kw - 1) + 1;  // widened spec row (k = 1)
  const int64_t slots = 3;

  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(5000 + i, -1.0f, 1.0f);
  // Initial state: slot 0 = the spec request's initial (read at num_accepted-1);
  // slot 2 = the prefill request (fresh — zeroed by has_initial_state==0).
  std::vector<float> ssm0(static_cast<size_t>(slots * ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(6000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
  for (int64_t s = 0; s < slots; ++s)
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv0[static_cast<size_t>((s * conv_dim + ch) * conv_len + j)] =
            RandV(7000 + (s * conv_dim + ch) * (Kw - 1) + j, -1.0f, 1.0f);

  // ── Mixed run. ──
  std::vector<float> ssm_m = ssm0, conv_m = conv0;
  const std::vector<float> mixed_out = vllm::GdnBlockPagedForTest(
      Q(dev), w, c, h, MixedMeta(Tp), ssm_m, conv_m, slots, conv_len, T);

  // ── Reference: spec rows via the PURE spec path (slot 0 initial). ──
  std::vector<float> ssm_s = ssm0, conv_s = conv0;
  std::vector<float> h_spec(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(Ts * H));
  const std::vector<float> spec_ref = vllm::GdnBlockPagedForTest(
      Q(dev), w, c, h_spec, SpecMeta(Ts), ssm_s, conv_s, slots, conv_len, Ts);

  // ── Reference: prefill rows via the PURE prefill path (fresh slot 2). ──
  std::vector<float> ssm_p = ssm0, conv_p = conv0;
  std::vector<float> h_pf(h.begin() + static_cast<std::ptrdiff_t>(Ts * H), h.end());
  const std::vector<float> pf_ref = vllm::GdnBlockPagedForTest(
      Q(dev), w, c, h_pf, PrefillMeta(Tp, 2), ssm_p, conv_p, slots, conv_len, Tp);

  REQUIRE(static_cast<int64_t>(mixed_out.size()) == T * H);
  REQUIRE(static_cast<int64_t>(spec_ref.size()) == Ts * H);
  REQUIRE(static_cast<int64_t>(pf_ref.size()) == Tp * H);
  std::vector<float> ref;
  ref.insert(ref.end(), spec_ref.begin(), spec_ref.end());
  ref.insert(ref.end(), pf_ref.begin(), pf_ref.end());

  if (bit_exact) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < mixed_out.size(); ++i)
      if (std::memcmp(&mixed_out[i], &ref[i], sizeof(float)) != 0) {
        if (bad == 0) first = i;
        ++bad;
      }
    CAPTURE(g.name);
    CAPTURE(bad);
    CAPTURE(first);
    if (bad != 0) { CAPTURE(mixed_out[first]); CAPTURE(ref[first]); }
    CHECK(bad == 0);
  } else {
    float maxabs = 0.0f;
    for (size_t i = 0; i < mixed_out.size(); ++i)
      maxabs = std::max(maxabs, std::fabs(mixed_out[i] - ref[i]));
    CAPTURE(g.name);
    CAPTURE(maxabs);
    CHECK(maxabs < 0.05f);
  }
}

constexpr GdnDims kGate27B{16, 48, 128, 128, 4, "27B (Hv=48)"};
constexpr GdnDims kGate35B{16, 32, 128, 128, 4, "35B (Hv=32)"};

}  // namespace

TEST_CASE("GDN spec routing (CPU): pure spec batch == token-sequential decode chain") {
  RunSpecRoutingCase(vt::DeviceType::kCPU, kGate27B, /*bit_exact=*/true);
  RunSpecRoutingCase(vt::DeviceType::kCPU, kGate35B, /*bit_exact=*/true);
}

TEST_CASE("GDN MIXED spec+prefill routing (CPU): mixed batch == pure spec + pure prefill") {
  RunMixedRoutingCase(vt::DeviceType::kCPU, kGate27B, /*bit_exact=*/true);
  RunMixedRoutingCase(vt::DeviceType::kCPU, kGate35B, /*bit_exact=*/true);
}

#ifdef VLLM_CPP_CUDA
TEST_CASE("GDN spec routing (CUDA): spec branch runs on-device and matches decode chain") {
  vt::GetBackend(vt::DeviceType::kCUDA);  // skip cleanly if no device
  RunSpecRoutingCase(vt::DeviceType::kCUDA, kGate27B, /*bit_exact=*/false);
  RunSpecRoutingCase(vt::DeviceType::kCUDA, kGate35B, /*bit_exact=*/false);
}

TEST_CASE("GDN MIXED spec+prefill routing (CUDA): mixed batch matches pure spec + prefill") {
  vt::GetBackend(vt::DeviceType::kCUDA);
  RunMixedRoutingCase(vt::DeviceType::kCUDA, kGate27B, /*bit_exact=*/false);
  RunMixedRoutingCase(vt::DeviceType::kCUDA, kGate35B, /*bit_exact=*/false);
}
#endif
