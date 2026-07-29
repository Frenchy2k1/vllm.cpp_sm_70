// DeepSeek-V4-Flash W2b — the `deepseek4` GGUF keep-quant TOWER materialization.
//
// Builds a TINY SYNTHETIC `deepseek4` GGUF (gguf_builder.h) with the REAL tensor
// NAMING convention (`blk.N.*`, the check-dsv4-gguf-namemap.py map) + a REAL
// keep-quant ggml type (Q8_0) at tiny dims, then asserts the loader
// (LoadDeepseekV4FromGguf):
//   (a) ACCOUNTS for EVERY GGUF tensor into a non-empty tower slot — none
//       unmapped (a missing required tensor throws), none leftover (a tensor the
//       map does not cover throws);
//   (b) KEEPS the MW/SEW weights COMPRESSED (block dtype resident, smaller than
//       the dequant-to-f32 image) while the small V/ET tensors dequant;
//   (c) load -> Forward works END-TO-END at tiny shape (finite, deterministic
//       logits over the assembled W7 CPU composition).
// RED-first by construction: a missing tensor, a leftover tensor, or forcing
// dequant-instead-of-keep each changes the outcome.
//
// HONEST 3-state: this tiny-synthetic load->forward is DERIVED + BUILD-VERIFIED.
// The real 91 GiB `UD-IQ2_XXS` load + generate stays W8-final (download + DGX);
// the name-map coverage vs the real 1328-tensor manifest is gated separately by
// scripts/check-dsv4-gguf-namemap.py. The IQ2_XXS/IQ3_XXS/Q2_K keep-quant vec_dot
// is gated by tests/vt/test_ops_quant_dot.cpp — the tower WIRING here is
// quant-type-agnostic (it routes by role via HasQuantDotKernel), so Q8_0 (a real
// keep-quant type, block 32, easy to synthesize with meaningful values) exercises
// the identical routing/residency/accounting path the i-quants take.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vt/dtype.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::I32ArrayKv;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

// ── tiny geometry (Q8_0 block = 32, so every keep-quant K is a multiple of 32) ──
struct Dims {
  int64_t H = 32, vocab = 16;
  int64_t nh = 2, head_dim = 32, rope = 8;   // nope = 24
  int64_t qlr = 32, olr = 32, o_groups = 2;  // wo_a [o_groups*olr, H], wo_b [H, o_groups*olr]
  int64_t E = 4, used = 2, mi = 32;
  int64_t hc = 2, sinkhorn = 3;
  int64_t inh = 2, ihd = 32, index_topk = 3;
  int64_t n_layer = 4, n_hash = 2;
  std::vector<int32_t> compress_ratios = {0, 4, 2, 4};  // idx {1,3}, comp {1,2,3}
};

int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Little-endian f32 bytes for `n` values from fill(idx).
template <typename F>
std::string F32Data(int64_t n, F fill) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = fill(i);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
  }
  return s;
}

// Encode a torch [out,in] (in % 32 == 0) row-major float image into Q8_0 blocks
// (block_q8_0 = { f16 d; int8 qs[32] } = 34 B), the ggml storage order.
template <typename F>
std::string Q8Data(int64_t out, int64_t in, F fill) {
  VT_CHECK(in % 32 == 0, "Q8Data: in must be a multiple of 32");
  std::string s;
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t b = 0; b < in / 32; ++b) {
      float amax = 0.0f;
      float x[32];
      for (int j = 0; j < 32; ++j) {
        x[j] = fill(o * in + b * 32 + j);
        amax = std::max(amax, std::fabs(x[j]));
      }
      const float d = amax / 127.0f;
      const uint16_t dh = vt::F32ToF16(d);
      s.push_back(static_cast<char>(dh & 0xff));
      s.push_back(static_cast<char>((dh >> 8) & 0xff));
      for (int j = 0; j < 32; ++j) {
        int q = d > 0.0f ? static_cast<int>(std::lround(x[j] / d)) : 0;
        q = std::max(-127, std::min(127, q));
        s.push_back(static_cast<char>(static_cast<int8_t>(q)));
      }
    }
  }
  return s;
}

std::vector<uint64_t> GgmlDims(const std::vector<int64_t>& torch_shape) {
  std::vector<uint64_t> d;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    d.push_back(static_cast<uint64_t>(*it));
  return d;
}

// A smooth, nonzero-amax fill so Q8_0 has a real scale and the forward is finite.
float WFill(int64_t i) { return 0.05f * static_cast<float>((i % 13) - 6); }

template <typename F>
void AddF32(GgufModelBuilder& b, const std::string& name,
            const std::vector<int64_t>& torch_shape, F fill) {
  b.AddTensor(name, GgmlDims(torch_shape), /*F32=*/0, F32Data(Prod(torch_shape), fill));
}
void AddQ8(GgufModelBuilder& b, const std::string& name,
           const std::vector<int64_t>& torch_shape) {
  const int64_t out = torch_shape.size() == 3 ? torch_shape[0] * torch_shape[1]
                                              : torch_shape[0];
  const int64_t in = torch_shape.back();
  b.AddTensor(name, GgmlDims(torch_shape), /*Q8_0=*/8, Q8Data(out, in, WFill));
}

std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// Build the tiny synthetic deepseek4 GGUF. `drop` omits one tensor name (RED:
// unaccounted -> throw); `extra` adds a bogus tensor (RED: leftover -> throw).
std::string BuildGguf(const Dims& d, const std::string& drop = "",
                      const std::string& extra = "") {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "deepseek4"));
  const std::string p = "deepseek4.";
  b.AddKv(U32Kv(p + "embedding_length", d.H));
  b.AddKv(U32Kv(p + "block_count", d.n_layer));
  b.AddKv(U32Kv(p + "attention.head_count", d.nh));
  b.AddKv(U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(U32Kv(p + "attention.key_length", d.head_dim));
  b.AddKv(U32Kv(p + "rope.dimension_count", d.rope));
  b.AddKv(U32Kv(p + "attention.q_lora_rank", d.qlr));
  b.AddKv(U32Kv(p + "attention.output_lora_rank", d.olr));
  b.AddKv(U32Kv(p + "attention.output_group_count", d.o_groups));
  b.AddKv(U32Kv(p + "attention.sliding_window", 4));
  b.AddKv(F32Kv(p + "rope.freq_base", 10000.0f));
  b.AddKv(F32Kv(p + "attention.compress_rope_freq_base", 160000.0f));
  b.AddKv(F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(U32Kv(p + "expert_count", d.E));
  b.AddKv(U32Kv(p + "expert_used_count", d.used));
  b.AddKv(U32Kv(p + "expert_shared_count", 1));
  b.AddKv(U32Kv(p + "expert_feed_forward_length", d.mi));
  b.AddKv(U32Kv(p + "hash_layer_count", d.n_hash));
  b.AddKv(F32Kv(p + "swiglu_clamp", 10.0f));
  b.AddKv(U32Kv(p + "hyper_connection.count", d.hc));
  b.AddKv(U32Kv(p + "hyper_connection.sinkhorn_iterations", d.sinkhorn));
  b.AddKv(F32Kv(p + "hyper_connection.epsilon", 1e-6f));
  b.AddKv(U32Kv(p + "attention.indexer.head_count", d.inh));
  b.AddKv(U32Kv(p + "attention.indexer.key_length", d.ihd));
  b.AddKv(U32Kv(p + "attention.indexer.top_k", d.index_topk));
  b.AddKv(I32ArrayKv(p + "attention.compress_ratios", d.compress_ratios));

  const int64_t hcf = (2 + d.hc) * d.hc;  // MHC fn out rows

  auto add = [&](const std::string& name, bool q8,
                 const std::vector<int64_t>& shape) {
    if (name == drop) return;
    if (q8) {
      AddQ8(b, name, shape);
    } else {
      AddF32(b, name, shape, WFill);
    }
  };

  // model level.
  add("token_embd.weight", false, {d.vocab, d.H});
  add("output.weight", true, {d.vocab, d.H});
  add("output_norm.weight", false, {d.H});
  add("output_hc_base.weight", false, {d.hc});
  add("output_hc_fn.weight", false, {d.hc, d.hc * d.H});
  add("output_hc_scale.weight", false, {1});

  for (int64_t l = 0; l < d.n_layer; ++l) {
    const int64_t cr = d.compress_ratios[static_cast<size_t>(l)];
    const bool has_comp = cr != 0;
    const bool has_idx = cr == 4;
    const bool is_hash = l < d.n_hash;
    // MLA (Q8_0 MW) + norms/sink (F32 V).
    add(Blk(l, "attn_q_a.weight"), true, {d.qlr, d.H});
    add(Blk(l, "attn_q_b.weight"), true, {d.nh * d.head_dim, d.qlr});
    add(Blk(l, "attn_kv.weight"), true, {d.head_dim, d.H});
    add(Blk(l, "attn_output_a.weight"), true, {d.o_groups * d.olr, d.nh * d.head_dim / d.o_groups});
    add(Blk(l, "attn_output_b.weight"), true, {d.H, d.o_groups * d.olr});
    add(Blk(l, "attn_norm.weight"), false, {d.H});
    add(Blk(l, "attn_q_a_norm.weight"), false, {d.qlr});
    add(Blk(l, "attn_kv_a_norm.weight"), false, {d.head_dim});
    add(Blk(l, "attn_sinks.weight"), false, {d.nh});
    add(Blk(l, "ffn_norm.weight"), false, {d.H});
    // MHC per-layer mixing (F32 V).
    add(Blk(l, "hc_attn_base.weight"), false, {hcf});
    add(Blk(l, "hc_attn_fn.weight"), false, {hcf, d.hc * d.H});
    add(Blk(l, "hc_attn_scale.weight"), false, {3});
    add(Blk(l, "hc_ffn_base.weight"), false, {hcf});
    add(Blk(l, "hc_ffn_fn.weight"), false, {hcf, d.hc * d.H});
    add(Blk(l, "hc_ffn_scale.weight"), false, {3});
    // MoE.
    add(Blk(l, "ffn_gate_inp.weight"), true, {d.E, d.H});
    add(Blk(l, "ffn_gate_exps.weight"), true, {d.E, d.mi, d.H});
    add(Blk(l, "ffn_up_exps.weight"), true, {d.E, d.mi, d.H});
    add(Blk(l, "ffn_down_exps.weight"), true, {d.E, d.H, d.mi});
    add(Blk(l, "ffn_gate_shexp.weight"), true, {d.mi, d.H});
    add(Blk(l, "ffn_up_shexp.weight"), true, {d.mi, d.H});
    add(Blk(l, "ffn_down_shexp.weight"), true, {d.H, d.mi});
    if (is_hash) {
      // tid2eid holds expert ids in [0, E); F32 (rounded to int at load).
      const std::string nm = Blk(l, "ffn_gate_tid2eid.weight");
      if (nm != drop)
        AddF32(b, nm, {d.vocab, d.used},
               [&](int64_t i) { return static_cast<float>(i % d.E); });
    } else {
      add(Blk(l, "exp_probs_b.bias"), false, {d.E});
    }
    // DSA compressor (compress_ratio != 0) + indexer (== 4).
    if (has_comp) {
      add(Blk(l, "attn_compressor_ape.weight"), false, {cr, d.head_dim});
      add(Blk(l, "attn_compressor_gate.weight"), true, {d.head_dim, d.H});
      add(Blk(l, "attn_compressor_kv.weight"), true, {d.head_dim, d.H});
      add(Blk(l, "attn_compressor_norm.weight"), false, {d.head_dim});
    }
    if (has_idx) {
      add(Blk(l, "indexer.attn_q_b.weight"), true, {d.inh * d.ihd, d.H});
      add(Blk(l, "indexer.proj.weight"), false, {d.inh, d.H});
      add(Blk(l, "indexer_compressor_ape.weight"), false, {cr, d.ihd});
      add(Blk(l, "indexer_compressor_gate.weight"), true, {d.ihd, d.H});
      add(Blk(l, "indexer_compressor_kv.weight"), true, {d.ihd, d.H});
      add(Blk(l, "indexer_compressor_norm.weight"), false, {d.ihd});
    }
  }
  if (!extra.empty()) AddF32(b, extra, {d.H}, WFill);
  return b.Build();
}

// keep-quant policy (deterministic, independent of the platform probe).
vllm::GgufLoadPolicy KeepPolicy() {
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  return pol;
}
const vllm::GgufLoadPolicy kExpandAll;  // all defaults -> dequant everything

}  // namespace

TEST_CASE("DeepseekV4ParamsFromGguf resolves the deepseek4 hparams") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::DeepseekV4Params p = vllm::DeepseekV4ParamsFromGguf(g);
  CHECK(p.hidden_size == 32);
  CHECK(p.num_hidden_layers == 4);
  CHECK(p.head_dim == 32);
  CHECK(p.q_lora_rank == 32);
  CHECK(p.o_groups == 2);
  CHECK(p.n_routed_experts == 4);
  CHECK(p.num_experts_per_tok == 2);
  CHECK(p.hc_mult == 2);
  CHECK(p.num_hash_layers == 2);
  CHECK(p.index_topk == 3);
  CHECK(p.vocab_size == 16);
  REQUIRE(p.compress_ratios.size() == 4);
  CHECK(p.compress_ratios[1] == 4);
  CHECK(p.has_indexer(1));
  CHECK(p.has_indexer(3));
  CHECK_FALSE(p.has_indexer(0));
  CHECK(p.has_compressor(2));
  CHECK(p.is_hash_layer(1));
  CHECK_FALSE(p.is_hash_layer(2));
}

TEST_CASE("LoadDeepseekV4FromGguf: accounts for every tensor into a tower slot") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol);

  // Total = 6 top-level + Σ layers (layer0 24, layer1 34, layer2 28, layer3 34)
  // = 126, and the loader consumed EXACTLY the file's tensor set (no leftover).
  CHECK(w.accounted_tensors == 126);
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  CHECK(w.has_gguf_weights);
  CHECK(w.has_host_weights);
  REQUIRE(w.gguf.layers.size() == 4);
  // Per-layer topology mapped through: layer1 has both indexer + compressor +
  // hash; layer2 gated + compressor, no indexer; layer0 hash, no comp/idx.
  CHECK(w.gguf.layers[0].is_hash);
  CHECK_FALSE(w.gguf.layers[0].has_compressor);
  CHECK(w.gguf.layers[1].has_indexer);
  CHECK(w.gguf.layers[1].has_compressor);
  CHECK_FALSE(w.gguf.layers[2].has_indexer);
  CHECK(w.gguf.layers[2].has_compressor);
  // Non-empty slots (materialized, not left as empty OwnedTensors).
  CHECK_FALSE(w.gguf.embed.Empty());
  CHECK_FALSE(w.gguf.lm_head.Empty());
  CHECK_FALSE(w.gguf.layers[0].wq_a.Empty());
  CHECK_FALSE(w.gguf.layers[0].moe_down_exps.Empty());
  CHECK_FALSE(w.gguf.layers[1].idx_wq_b.Empty());
  CHECK_FALSE(w.gguf.layers[2].comp_wgate.Empty());
}

TEST_CASE("LoadDeepseekV4FromGguf: keep-quant blocks stay COMPRESSED") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &keep);

  // The 256-expert down proj (SEW) keeps its Q8_0 blocks: block dtype resident,
  // shape [E*out, in], nk = true, and byte size = compressed (34 B / 32 elems),
  // strictly smaller than the dequant-to-f32 image (4 B / elem).
  const vllm::OwnedTensor& dn = w.gguf.layers[3].moe_down_exps;
  CHECK(dn.dtype == vt::DType::kQ8_0);
  CHECK(vt::IsBlockQuant(dn.dtype));
  CHECK(dn.nk);
  const int64_t rows = d.E * d.H;   // E*out
  const int64_t k = d.mi;           // in
  CHECK(dn.shape[0] == rows);
  CHECK(dn.shape[1] == k);
  const int64_t elems = rows * k;
  CHECK(static_cast<int64_t>(dn.bytes.size()) == rows * (k / 32) * 34);
  CHECK(static_cast<int64_t>(dn.bytes.size()) < elems * 4);  // < dequant-f32
  // An MLA linear (MW) also keeps quant; a norm (V) is dequant f32.
  CHECK(w.gguf.layers[0].wq_a.dtype == vt::DType::kQ8_0);
  CHECK(w.gguf.lm_head.dtype == vt::DType::kQ8_0);
  CHECK(w.gguf.layers[0].attn_norm.dtype == vt::DType::kF32);
  CHECK(w.gguf.layers[0].attn_sink.dtype == vt::DType::kF32);

  // RED-first: dequant-instead-of-keep (an expand policy) leaves the SAME weight
  // UNCOMPRESSED (bf16), proving keep-quant is what compresses it.
  const vllm::DeepseekV4Weights we =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &kExpandAll);
  CHECK(we.gguf.layers[3].moe_down_exps.dtype == vt::DType::kBF16);
  CHECK(static_cast<int64_t>(we.gguf.layers[3].moe_down_exps.bytes.size()) == elems * 2);
}

TEST_CASE("LoadDeepseekV4FromGguf: load -> Forward runs end-to-end at tiny shape") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol);

  const std::vector<int32_t> tokens = {1, 5, 9};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<float> logits =
      vllm::DeepseekV4ForwardHost(w.host, w.params, tokens, positions);

  REQUIRE(logits.size() == tokens.size() * static_cast<size_t>(d.vocab));
  for (float v : logits) CHECK(std::isfinite(v));
  // Deterministic: the same loaded tower re-runs bit-identically.
  const std::vector<float> again =
      vllm::DeepseekV4ForwardHost(w.host, w.params, tokens, positions);
  REQUIRE(again.size() == logits.size());
  for (size_t i = 0; i < logits.size(); ++i) CHECK(again[i] == logits[i]);
}

// ── entrypoint wiring (W8-final deliverable #1): a `deepseek4` GGUF must route
//    through the top-level GGUF dispatch onto the registered DeepseekV4ForCausalLM
//    model class, so the CLI/server recognizes the arch (it was qwen-only). ─────
TEST_CASE("DeepseekV4HfConfigFromGguf: deepseek4 GGUF routes to DeepseekV4ForCausalLM") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::DeepseekV4HfConfigFromGguf(g);

  // model_type keeps llama.cpp's GGUF family key; architectures maps onto the
  // registered vLLM model class the top-level dispatch resolves.
  CHECK(c.model_type == "deepseek4");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "DeepseekV4ForCausalLM");
  // Typed geometry republished from the GGUF KV.
  CHECK(c.hidden_size == 32);
  CHECK(c.num_hidden_layers == 4);
  CHECK(c.head_dim == 32);
  CHECK(c.num_attention_heads == 2);
  CHECK(c.vocab_size == 16);
  // The scalars the registry parse hook (ParseDeepseekV4Config) reads from raw.
  CHECK(c.raw.at("hc_mult").get<int64_t>() == 2);
  CHECK(c.raw.at("n_routed_experts").get<int64_t>() == 4);
  CHECK(c.raw.at("num_experts_per_tok").get<int64_t>() == 2);
  CHECK(c.raw.at("scoring_func").get<std::string>() == "sqrtsoftplus");
  CHECK(c.raw.at("expert_dtype").get<std::string>() == "fp4");
  REQUIRE(c.raw.at("compress_ratios").is_array());
  CHECK(c.raw.at("compress_ratios").size() == 4);

  // The registry resolves the mapped architecture to the DeepSeek-V4 factory —
  // i.e. the top-level GGUF dispatch now recognizes a deepseek4 file (the qwen-
  // only HfConfigFromGguf would throw "unexpected architecture 'deepseek4'").
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(c);
  CHECK(reg.architecture == "DeepseekV4ForCausalLM");
  REQUIRE(reg.factory != nullptr);
  CHECK_FALSE(reg.factory->is_dense_model);
  CHECK(reg.factory->load_weights != nullptr);
}

TEST_CASE("LoadDeepseekV4FromGguf: RED-first — unmapped/leftover tensors throw") {
  Dims d;
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  // (a) a MISSING required tensor -> the load throws (unaccounted route).
  {
    TempFile f(BuildGguf(d, /*drop=*/Blk(2, "attn_kv.weight")));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS(vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol));
  }
  // (b) a LEFTOVER tensor the blk.N.* map does not cover -> the load throws.
  {
    TempFile f(BuildGguf(d, /*drop=*/"", /*extra=*/"blk.0.bogus_unmapped.weight"));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS(vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol));
  }
}
