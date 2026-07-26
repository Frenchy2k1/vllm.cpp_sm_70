// vllm.cpp original. DFlash D3 (DF-DRAFT-KV-PREP) CPU gate: prepare_dflash_inputs
// bit-exactness, context-KV precompute numerics, and the context-aware draft
// forward. Ported semantics: qwen3_dflash.py @ 555967922
// (precompute_and_store_context_kv :548-619, _project_context_kv :505-534,
// _normalize_context_k :536-546) and dflash/speculator.py @ 555967922
// (_prepare_dflash_inputs_kernel :472-618, prepare_dflash_inputs :621-687).
//
// These are the deterministic RED-first CPU gates that pin the load-bearing D3
// invariants BEFORE the dgx numerical parity vs the dumped vLLM reference
// (scripts/spec/d3_dflash_kvprep_ref.py):
//   (1) prepare_dflash_inputs is INTEGER bit-exact vs a hand-computed reference
//       (ids/positions/slots/context-slots/sample-maps/seq_lens/padding), and RED
//       — perturbing num_rejected shifts valid_ctx_end -> the query positions move
//       (the rejected-token exclusion is load-bearing);
//   (2) the context-KV precompute matches an independent f32 envelope reference on
//       V, and RED — the hidden_norm weight (K and V), the k_norm weight (K only),
//       and a context position (K only, via RoPE) are each load-bearing;
//   (3) the context-aware forward DEGENERATES to the D2 context-free forward when
//       the context is empty (consistency), and context is load-bearing when
//       present (the block proposal changes); per-request block isolation holds.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using namespace vllm;

namespace {
vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

OwnedTensor MkBf16(const std::vector<int64_t>& shape, double seed, double amp, bool nk) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  t.nk = nk;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return t;
}

struct Dims {
  int64_t H = 4, Hq = 2, Hkv = 1, Dh = 2, I = 6, vocab = 8, layers = 2, taps = 2;
};

HfConfig MakeConfig(const Dims& dm) {
  HfConfig c;
  c.hidden_size = dm.H;
  c.num_attention_heads = dm.Hq;
  c.num_key_value_heads = dm.Hkv;
  c.head_dim = dm.Dh;
  c.rotary_dim = dm.Dh;
  c.rope_theta = 10000.0;
  c.intermediate_size = dm.I;
  c.vocab_size = dm.vocab;
  c.num_hidden_layers = dm.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 64;
  c.layer_types = {"sliding_attention", "full_attention"};
  c.raw = nlohmann::json::object();
  c.raw["dflash_config"] = {{"mask_token_id", 7}};
  return c;
}

Qwen3DFlashWeights MakeWeights(const Dims& dm) {
  Qwen3DFlashWeights w;
  w.num_taps = dm.taps;
  w.mask_token_id = 7;
  w.draft_vocab_size = dm.vocab;
  const int64_t qdim = dm.Hq * dm.Dh, kdim = dm.Hkv * dm.Dh;
  w.embed_tokens = MkBf16({dm.vocab, dm.H}, 0.1, 0.3, false);
  w.fc = MkBf16({dm.H, dm.H * dm.taps}, 0.2, 0.2, true);
  w.hidden_norm = MkBf16({dm.H}, 0.3, 0.5, false);
  w.final_norm = MkBf16({dm.H}, 0.4, 0.5, false);
  w.lm_head = MkBf16({dm.vocab, dm.H}, 0.5, 0.3, true);
  const std::vector<Qwen3DFlashLayerAttnMode> modes = {{true, 64}, {false, 0}};
  for (int64_t l = 0; l < dm.layers; ++l) {
    Qwen3DFlashLayerWeights lw;
    const double s = 1.0 + static_cast<double>(l);
    lw.input_layernorm = MkBf16({dm.H}, s + 0.01, 0.5, false);
    lw.post_attention_layernorm = MkBf16({dm.H}, s + 0.02, 0.5, false);
    lw.qkv_proj = MkBf16({qdim + 2 * kdim, dm.H}, s + 0.03, 0.25, true);
    lw.o_proj = MkBf16({dm.H, qdim}, s + 0.04, 0.25, true);
    lw.q_norm = MkBf16({dm.Dh}, s + 0.05, 0.5, false);
    lw.k_norm = MkBf16({dm.Dh}, s + 0.06, 0.5, false);
    lw.gate_up_proj = MkBf16({2 * dm.I, dm.H}, s + 0.07, 0.2, true);
    lw.down_proj = MkBf16({dm.H, dm.I}, s + 0.08, 0.2, true);
    lw.attn_mode = modes[static_cast<size_t>(l)];
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// A concrete 2-request batch with distinct context lengths, rejection on req1,
// and a chunked-prefill (num_sampled==0) bonus on req1.
DflashPrepareBatch MakeBatch() {
  DflashPrepareBatch b;
  b.target_query_start_loc = {0, 3, 5};            // req0 ctx=3, req1 ctx=2
  b.target_positions = {0, 1, 2, 0, 1};            // int64
  b.idx_mapping = {0, 1};
  b.last_sampled = {42, 99, 0};                    // indexed by req_state_idx
  b.next_prefill_tokens = {0, 88, 0};
  b.num_sampled = {1, 0};                          // req1 chunked-prefill -> next_prefill
  b.num_rejected = {0, 1};                         // req1 rejects 1 -> valid_ctx_end=4
  b.block_table = {10, 11, 12, 13, 20, 21, 22, 23, 0, 0, 0, 0};
  b.block_table_stride = 4;
  b.block_size = 4;
  b.parallel_drafting_token_id = 7;
  b.num_query_per_req = 3;                          // 1 + k, k=2
  b.num_speculative_steps = 2;
  b.max_num_reqs = 3;
  b.max_num_tokens = 12;
  b.max_model_len = 100;
  b.sample_from_anchor = false;
  return b;
}
}  // namespace

TEST_CASE("dflash prepare_dflash_inputs: integer bit-exact vs hand reference") {
  DflashPrepareOutputs o = PrepareDflashInputs(MakeBatch());
  // Query ids: [bonus0=42, mask, mask, bonus1=88, mask, mask].
  CHECK(o.input_ids == std::vector<int32_t>({42, 7, 7, 88, 7, 7}));
  // Query positions: req0 last_valid_pos=2 -> [3,4,5]; req1 last_valid_pos=0 -> [1,2,3].
  CHECK(o.query_positions == std::vector<int64_t>({3, 4, 5, 1, 2, 3}));
  // Context positions (global, length 5): req0 [0,1,2], req1 [0,1].
  CHECK(o.context_positions == std::vector<int64_t>({0, 1, 2, 0, 1}));
  // Context slots: req0 blocks[0]=10 -> 40,41,42; req1 blocks[0]=20 -> 80,81.
  CHECK(o.context_slot_mapping == std::vector<int64_t>({40, 41, 42, 80, 81}));
  // Query slots (active [0,6)) then PAD_SLOT_ID(-1) to max_num_tokens=12.
  // req0: pos3->blk0(10)slot43, pos4->blk1(11)slot44, pos5->blk1(11)slot45.
  // req1: pos1->blk0(20)slot81, pos2->slot82, pos3->slot83.
  CHECK(o.query_slot_mapping ==
        std::vector<int64_t>({43, 44, 45, 81, 82, 83, -1, -1, -1, -1, -1, -1}));
  // query_start_loc [0,3,6,6] (padded to max_num_reqs+1); seq_lens [6,4,0].
  CHECK(o.query_start_loc == std::vector<int32_t>({0, 3, 6, 6}));
  CHECK(o.seq_lens == std::vector<int32_t>({6, 4, 0}));
  // Sample maps: 2 samples/req at offsets 1,2; padded to max_num_reqs*nspec=6.
  CHECK(o.sample_indices == std::vector<int64_t>({1, 2, 4, 5, 0, 0}));
  CHECK(o.sample_pos == std::vector<int64_t>({4, 5, 2, 3, 0, 0}));
  CHECK(o.sample_idx_mapping == std::vector<int32_t>({0, 0, 1, 1, -1, -1}));
}

TEST_CASE("dflash prepare_dflash_inputs RED: num_rejected shifts valid_ctx_end") {
  DflashPrepareBatch b = MakeBatch();
  b.num_rejected = {0, 0};  // req1 no longer rejects -> last_valid_pos=pos[1]=1
  DflashPrepareOutputs o = PrepareDflashInputs(b);
  // req1 block now anchors at last_valid_pos+1 = 2 -> query positions [2,3,4]
  // (was [1,2,3]); the rejected-token exclusion is load-bearing.
  CHECK(o.query_positions == std::vector<int64_t>({3, 4, 5, 2, 3, 4}));
  CHECK(o.seq_lens == std::vector<int32_t>({6, 5, 0}));
}

TEST_CASE("dflash context-KV precompute: V matches f32 envelope reference") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t C = 3;
  std::vector<float> ctx(static_cast<size_t>(C * dm.H));
  for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = 0.4f * std::sin(1.3 + 0.5 * static_cast<double>(i));
  std::vector<int32_t> pos = {0, 1, 2};
  Qwen3DFlashModel::ContextKV ckv = Qwen3DFlashModel::PrecomputeContextKV(ctx, pos, w, cfg, q);
  REQUIRE(ckv.num_ctx == C);
  REQUIRE(ckv.k.size() == static_cast<size_t>(dm.layers));
  REQUIRE(ckv.v[0].size() == static_cast<size_t>(C * dm.Hkv * dm.Dh));
  for (float v : ckv.k[0]) CHECK(std::isfinite(v));
  for (float v : ckv.v[0]) CHECK(std::isfinite(v));

  // Independent reference for layer 0 V = matmul(bf16(rmsnorm(ctx, hidden_norm)), wv).
  const int64_t H = dm.H, kdim = dm.Hkv * dm.Dh, qdim = dm.Hq * dm.Dh;
  const float eps = 1e-6f;
  const auto* hn = reinterpret_cast<const uint16_t*>(w.hidden_norm.bytes.data());
  const auto* qkv = reinterpret_cast<const uint16_t*>(w.layers[0].qkv_proj.bytes.data());
  for (int64_t r = 0; r < C; ++r) {
    // normed[j] = ctx[r,j] / sqrt(mean(ctx^2)+eps) * hidden_norm[j], rounded bf16.
    float ms = 0.0f;
    for (int64_t j = 0; j < H; ++j) ms += ctx[static_cast<size_t>(r * H + j)] * ctx[static_cast<size_t>(r * H + j)];
    ms = ms / static_cast<float>(H);
    const float inv = 1.0f / std::sqrt(ms + eps);
    std::vector<float> normed(static_cast<size_t>(H));
    for (int64_t j = 0; j < H; ++j)
      normed[static_cast<size_t>(j)] = vt::BF16ToF32(vt::F32ToBF16(
          ctx[static_cast<size_t>(r * H + j)] * inv * vt::BF16ToF32(hn[j])));
    // V rows of qkv_proj = rows [qdim+kdim, qdim+2*kdim).
    for (int64_t o = 0; o < kdim; ++o) {
      float acc = 0.0f;
      const int64_t row = qdim + kdim + o;
      for (int64_t j = 0; j < H; ++j)
        acc += normed[static_cast<size_t>(j)] * vt::BF16ToF32(qkv[row * H + j]);
      CHECK(ckv.v[0][static_cast<size_t>(r * kdim + o)] == doctest::Approx(acc).epsilon(0.05));
    }
  }
}

TEST_CASE("dflash context-KV precompute RED: hidden_norm/k_norm/position are load-bearing") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  vt::Queue q = Cpu();
  const int64_t C = 3;
  std::vector<float> ctx(static_cast<size_t>(C * dm.H));
  for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = 0.4f * std::sin(1.3 + 0.5 * static_cast<double>(i));
  std::vector<int32_t> pos = {0, 1, 2};

  Qwen3DFlashWeights base = MakeWeights(dm);
  Qwen3DFlashModel::ContextKV b = Qwen3DFlashModel::PrecomputeContextKV(ctx, pos, base, cfg, q);

  auto maxdiff = [](const std::vector<float>& a, const std::vector<float>& c) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i] - c[i])));
    return m;
  };

  // (a) hidden_norm perturbed -> BOTH K and V change (it feeds the shared proj).
  Qwen3DFlashWeights wh = MakeWeights(dm);
  wh.hidden_norm = MkBf16({dm.H}, 0.9, 0.5, false);
  Qwen3DFlashModel::ContextKV h = Qwen3DFlashModel::PrecomputeContextKV(ctx, pos, wh, cfg, q);
  CHECK(maxdiff(b.k[0], h.k[0]) > 1e-3);
  CHECK(maxdiff(b.v[0], h.v[0]) > 1e-3);

  // (b) k_norm perturbed -> K changes, V unchanged (k_norm only normalizes K).
  Qwen3DFlashWeights wk = MakeWeights(dm);
  wk.layers[0].k_norm = MkBf16({dm.Dh}, 0.9, 0.5, false);
  Qwen3DFlashModel::ContextKV kk = Qwen3DFlashModel::PrecomputeContextKV(ctx, pos, wk, cfg, q);
  CHECK(maxdiff(b.k[0], kk.k[0]) > 1e-3);
  CHECK(maxdiff(b.v[0], kk.v[0]) == doctest::Approx(0.0));

  // (c) a context position perturbed -> K changes via RoPE, V unchanged.
  std::vector<int32_t> pos2 = {0, 5, 2};
  Qwen3DFlashModel::ContextKV pk = Qwen3DFlashModel::PrecomputeContextKV(ctx, pos2, base, cfg, q);
  CHECK(maxdiff(b.k[0], pk.k[0]) > 1e-3);
  CHECK(maxdiff(b.v[0], pk.v[0]) == doctest::Approx(0.0));
}

TEST_CASE("dflash context forward: empty context degenerates to the D2 context-free forward") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  std::vector<int32_t> ids = {2, 7, 7};
  std::vector<int32_t> pos = {0, 1, 2};
  std::vector<int32_t> cu = {0, 3};
  std::vector<float> ref = Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, cfg, q);

  std::vector<float> empty_ctx;              // C = 0
  std::vector<int32_t> empty_pos;
  std::vector<int32_t> ctx_cu = {0, 0};
  std::vector<float> got = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      empty_ctx, empty_pos, ctx_cu, ids, pos, cu, w, cfg, q);
  REQUIRE(got.size() == ref.size());
  for (size_t i = 0; i < ref.size(); ++i) CHECK(got[i] == doctest::Approx(ref[i]).epsilon(1e-4));
}

TEST_CASE("dflash context forward: context is load-bearing and blocks stay isolated") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  // One request: 2 context tokens (positions 0,1) then a 3-token block (2,3,4).
  std::vector<float> ctx(static_cast<size_t>(2 * dm.H));
  for (size_t i = 0; i < ctx.size(); ++i) ctx[i] = 0.3f * std::sin(0.7 + 0.4 * static_cast<double>(i));
  std::vector<int32_t> ctx_pos = {0, 1};
  std::vector<int32_t> ctx_cu = {0, 2};
  std::vector<int32_t> ids = {2, 7, 7};
  std::vector<int32_t> pos = {2, 3, 4};
  std::vector<int32_t> cu = {0, 3};
  std::vector<float> with_ctx = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx, ctx_pos, ctx_cu, ids, pos, cu, w, cfg, q);
  std::vector<float> no_ctx = Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, cfg, q);
  REQUIRE(with_ctx.size() == no_ctx.size());
  for (float v : with_ctx) CHECK(std::isfinite(v));
  double m = 0.0;
  for (size_t i = 0; i < with_ctx.size(); ++i)
    m = std::max(m, std::fabs(static_cast<double>(with_ctx[i] - no_ctx[i])));
  CHECK(m > 1e-3);  // context participates in the block attention

  // Two identical (context, block) requests -> identical per-request block logits.
  std::vector<float> ctx2(ctx);
  ctx2.insert(ctx2.end(), ctx.begin(), ctx.end());
  std::vector<int32_t> ctx_pos2 = {0, 1, 0, 1};
  std::vector<int32_t> ctx_cu2 = {0, 2, 4};
  std::vector<int32_t> ids2 = {2, 7, 7, 2, 7, 7};
  std::vector<int32_t> pos2 = {2, 3, 4, 2, 3, 4};
  std::vector<int32_t> cu2 = {0, 3, 6};
  std::vector<float> l2 = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx2, ctx_pos2, ctx_cu2, ids2, pos2, cu2, w, cfg, q);
  for (int64_t r = 0; r < 3; ++r)
    for (int64_t j = 0; j < dm.vocab; ++j)
      CHECK(l2[static_cast<size_t>(r * dm.vocab + j)] ==
            doctest::Approx(l2[static_cast<size_t>((r + 3) * dm.vocab + j)]));
}
