// vllm.cpp original. DFlash D3 (DF-DRAFT-KV-PREP) numeric-parity gate vs the REAL
// loaded vLLM DFlash draft (SPEC-DFLASH D3 — the decisive GPU gate the dev-box CPU
// session could not run). Reference: scripts/spec/d3_dflash_kvprep_ref.py dumps,
// from the REAL loaded vLLM draft (DFlashQwen3ForCausalLM @ 555967922, via
// collective_rpc into the worker):
//   * prepare_ref.json : the output of vLLM's ACTUAL Triton _prepare_dflash_inputs
//     _kernel (launched directly on GPU for a fixed single-request contiguous
//     prefill), cross-checked bit-exact vs a line-for-line numpy replica. INTEGER,
//     so OUR host PrepareDflashInputs must match BIT-exact.
//   * ctxkv_ref.json   : the precomputed context K (normed+RoPE'd) / V (raw) for
//     every draft attention layer, from vLLM's OWN _project_context_kv +
//     _normalize_context_k + the fused RoPE. Envelope-exact (bf16 GEMM/f32) vs OUR
//     PrecomputeContextKV.
//   * propose_ref.json : the (1+k) block proposal computed by vLLM's OWN submodules
//     with the precomputed context K/V prepended to each layer's attention. STRICT-
//     or-ratified-near-tie on the k proposed ids (near-tie-distributional-gate).
//
// This test loads the SAME z-lab/Qwen3.6-27B-DFlash draft weights + the target's
// SHARED embed_tokens/lm_head, runs OUR Qwen3DFlashModel::PrecomputeContextKV /
// ForwardBlockLogitsWithContext + PrepareDflashInputs on the identical fixtures, and
// gates: prepare INTEGER bit-exact, context-KV per-layer rel-L2 within envelope,
// block proposal STRICT/near-tie. GPU + checkpoint + fixture gated: SKIPs loudly
// when any is absent (CPU/CI dev box). The deterministic RED proofs live in the CPU
// unit test tests/vllm/v1/spec_decode/test_dflash_kvprep.cpp.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hf_snapshot.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace vllm;

namespace {

bool HasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

vt::Queue GpuQ() {
  static vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  static vt::Queue q = b.CreateQueue();
  return q;
}

// GATE-PIN-UNPINNED-SNAPSHOTS (#471): the private `SnapDir` that used to live
// here resolved `unsloth/Qwen3.6-27B-NVFP4` by readdir order, on a host caching
// both @890bdef7 (NVFP4) and @ccdaab7e (the same repo re-quantized to FP8). It is
// DELETED, not kept beside the pinned call. See parity::HfSnapshot.

std::vector<std::string> Shards(const std::string& dir) {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec))
    if (e.path().extension() == ".safetensors") out.push_back(e.path().string());
  return out;
}

OwnedTensor LoadTargetBf16(const std::vector<SafetensorsFile>& shards, const std::string& name,
                           bool nk) {
  for (const SafetensorsFile& s : shards) {
    for (const std::string& n : s.Names()) {
      if (n != name) continue;
      const StTensor& t = s.Get(name);
      REQUIRE(t.dtype == "BF16");
      OwnedTensor out;
      out.dtype = vt::DType::kBF16;
      out.rank = static_cast<int>(t.shape.size());
      out.nk = nk;
      for (int i = 0; i < out.rank; ++i) out.shape[i] = t.shape[static_cast<size_t>(i)];
      out.bytes.resize(t.nbytes);
      std::memcpy(out.bytes.data(), t.data, t.nbytes);
      return out;
    }
  }
  return OwnedTensor{};
}

json LoadJson(const fs::path& p) {
  std::ifstream f(p);
  REQUIRE_MESSAGE(f.good(), "missing fixture " << p.string());
  json j;
  f >> j;
  return j;
}

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / std::sqrt(std::max(den, 1e-12));
}

HfConfig MakeConfig(const json& c) {
  HfConfig cfg;
  cfg.hidden_size = c["hidden_size"];
  cfg.num_attention_heads = c["num_attention_heads"];
  cfg.num_key_value_heads = c["num_key_value_heads"];
  cfg.head_dim = c["head_dim"];
  cfg.rotary_dim = c["head_dim"];
  cfg.rope_theta = c["rope_theta"];
  cfg.intermediate_size = c["intermediate_size"];
  cfg.vocab_size = c["vocab_size"];
  cfg.num_hidden_layers = c["num_hidden_layers"];
  cfg.rms_norm_eps = c["rms_norm_eps"];
  cfg.sliding_window = static_cast<int64_t>(c["sliding_window"]);
  cfg.layer_types = c["layer_types"].get<std::vector<std::string>>();
  cfg.raw = json::object();
  cfg.raw["dflash_config"] = {{"mask_token_id", c["mask_token_id"]},
                              {"target_layer_ids", c["target_layer_ids"]}};
  return cfg;
}

template <typename T>
std::vector<T> IntVec(const json& j) {
  std::vector<T> out;
  out.reserve(j.size());
  for (const auto& e : j) out.push_back(static_cast<T>(e.get<int64_t>()));
  return out;
}

}  // namespace

TEST_CASE("qwen3_dflash D3 context-KV + prepare + proposal parity vs the real vLLM draft") {
  // Resolve FIRST, then probe CUDA. The pinned revisions are reported on every
  // box, including the CPU dev box that can never run the body -- a skip banner
  // that says only "no CUDA backend" hides which checkpoint the gate would have
  // demanded, and makes the pin unobservable exactly where it is cheapest to
  // check.
  const std::string draft_dir = parity::Qwen27DFlashDraftSnapshot();
  const std::string target_dir = parity::Qwen27NvfP4Snapshot();
  const bool has_cuda = HasCuda();
  const fs::path fdir = fs::path(PARITY_GOLDENS_DIR) / "dflash_27b_kvprep";
  if (!has_cuda || draft_dir.empty() || target_dir.empty() ||
      !fs::exists(fdir / "ctxkv_ref.json")) {
    MESSAGE("SKIP (dev box): DFlash D3 kvprep parity needs CUDA (got: "
            << std::string(has_cuda ? "yes" : "NO") << "), "
               "z-lab/Qwen3.6-27B-DFlash @"
            << std::string(parity::kQwen27DFlashDraftRevision)
            << " (got: " << (draft_dir.empty() ? "ABSENT" : draft_dir) << "), "
            << "unsloth/Qwen3.6-27B-NVFP4 @" << std::string(parity::kQwen27NvfP4Revision)
            << " (got: " << (target_dir.empty() ? "ABSENT" : target_dir) << "), "
            << "and the D3 fixtures. A cache holding a DIFFERENT revision of "
               "either repo skips rather than being substituted (#471).");
    return;
  }

  const json cfg_j = LoadJson(fdir / "config.json");
  const HfConfig cfg = MakeConfig(cfg_j);
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int64_t nlayers = cfg.num_hidden_layers;
  const int64_t num_taps = cfg_j["num_taps"];
  const int32_t mask_id = cfg_j["mask_token_id"];

  // ---- Load the draft (layer weights + fc + norms; embed/lm_head shared).
  std::vector<SafetensorsFile> dshards;
  for (const std::string& p : Shards(draft_dir)) dshards.push_back(SafetensorsFile::Open(p));
  Qwen3DFlashWeights w = LoadQwen3DFlash(dshards, cfg, num_taps, mask_id);
  std::vector<SafetensorsFile> tshards;
  for (const std::string& p : Shards(target_dir)) tshards.push_back(SafetensorsFile::Open(p));
  w.embed_tokens = LoadTargetBf16(tshards, "model.language_model.embed_tokens.weight", false);
  w.lm_head = LoadTargetBf16(tshards, "lm_head.weight", true);
  REQUIRE_FALSE(w.embed_tokens.Empty());
  REQUIRE_FALSE(w.lm_head.Empty());
  w.draft_vocab_size = w.lm_head.shape[0];

  vt::Queue q = GpuQ();

  // ======================================================================
  // GATE 1 — prepare_dflash_inputs INTEGER bit-exact vs the REAL Triton kernel.
  // ======================================================================
  {
    const json pr = LoadJson(fdir / "prepare_ref.json");
    CHECK_MESSAGE(pr["kernel_matches_numpy"].get<bool>(),
                  "ref: numpy replica diverged from the real Triton kernel");
    const json& k = pr["kernel"];
    const int32_t ctx_len = pr["ctx_len"];
    const int32_t block = pr["block"];        // num_query_per_req
    const int32_t kvbs = pr["kv_block_size"];
    const int32_t nspec = block - 1;
    const auto block_table = IntVec<int32_t>(pr["block_table"]);
    const int32_t stride = static_cast<int32_t>(block_table.size());

    DflashPrepareBatch b;
    b.target_query_start_loc = {0, ctx_len};
    b.target_positions.resize(static_cast<size_t>(ctx_len));
    for (int32_t i = 0; i < ctx_len; ++i) b.target_positions[static_cast<size_t>(i)] = i;
    b.idx_mapping = {0};
    b.last_sampled = {static_cast<int32_t>(pr["anchor_token_id"])};
    b.next_prefill_tokens = {0};
    b.num_sampled = {1};
    b.num_rejected = {0};
    b.block_table = block_table;
    b.block_table_stride = stride;
    b.block_size = kvbs;
    b.parallel_drafting_token_id = mask_id;
    b.num_query_per_req = block;
    b.num_speculative_steps = nspec;
    b.max_num_reqs = pr["max_num_reqs"];
    b.max_num_tokens = pr["max_num_tokens"];
    b.max_model_len = pr["max_model_len"];
    b.sample_from_anchor = false;

    DflashPrepareOutputs o = PrepareDflashInputs(b);
    // Active-region bit-exact comparisons against vLLM's real kernel output.
    CHECK(o.input_ids == IntVec<int32_t>(k["input_ids"]));
    CHECK(o.query_positions == IntVec<int64_t>(k["query_positions"]));
    CHECK(o.query_start_loc == IntVec<int32_t>(k["query_start_loc"]));
    CHECK(o.context_positions == IntVec<int64_t>(k["context_positions"]));
    CHECK(o.context_slot_mapping == IntVec<int64_t>(k["context_slot_mapping"]));
    // query_slot_mapping: kernel buffer == max_num_tokens; compare full.
    CHECK(o.query_slot_mapping == IntVec<int64_t>(k["query_slot_mapping"]));
    CHECK(static_cast<int32_t>(o.seq_lens[0]) == k["seq_lens"][0].get<int32_t>());
    // sample maps: kernel buffer == max_num_reqs*nspec; compare full.
    CHECK(o.sample_indices == IntVec<int64_t>(k["sample_indices"]));
    CHECK(o.sample_pos == IntVec<int64_t>(k["sample_pos"]));
    CHECK(o.sample_idx_mapping == IntVec<int32_t>(k["sample_idx_mapping"]));
    MESSAGE("prepare_dflash_inputs: INTEGER bit-exact vs vLLM's real Triton kernel");
  }

  // ======================================================================
  // GATE 2 — context-KV precompute per-layer rel-L2 within the bf16 envelope.
  // ======================================================================
  const json ck = LoadJson(fdir / "ctxkv_ref.json");
  const int64_t C = ck["ctx_len"];
  REQUIRE(ck["num_layers"].get<int64_t>() == nlayers);
  const auto context_states = ck["context_states"].get<std::vector<float>>();
  const auto context_positions = IntVec<int32_t>(ck["context_positions"]);
  const auto ref_k = ck["k"].get<std::vector<float>>();
  const auto ref_v = ck["v"].get<std::vector<float>>();
  const int64_t layer_stride = C * Hkv * Dh;
  REQUIRE(static_cast<int64_t>(ref_k.size()) == nlayers * layer_stride);
  {
    Qwen3DFlashModel::ContextKV ckv =
        Qwen3DFlashModel::PrecomputeContextKV(context_states, context_positions, w, cfg, q);
    REQUIRE(ckv.num_ctx == C);
    REQUIRE(static_cast<int64_t>(ckv.k.size()) == nlayers);
    double worst_k = 0.0, worst_v = 0.0;
    for (int64_t l = 0; l < nlayers; ++l) {
      std::vector<float> rk(ref_k.begin() + static_cast<long>(l * layer_stride),
                            ref_k.begin() + static_cast<long>((l + 1) * layer_stride));
      std::vector<float> rv(ref_v.begin() + static_cast<long>(l * layer_stride),
                            ref_v.begin() + static_cast<long>((l + 1) * layer_stride));
      const double kr = RelL2(ckv.k[static_cast<size_t>(l)], rk);
      const double vr = RelL2(ckv.v[static_cast<size_t>(l)], rv);
      MESSAGE("ctxkv layer " << l << " K rel-L2 = " << kr << "  V rel-L2 = " << vr);
      worst_k = std::max(worst_k, kr);
      worst_v = std::max(worst_v, vr);
    }
    MESSAGE("ctxkv worst K rel-L2 = " << worst_k << "  worst V rel-L2 = " << worst_v);
    CHECK(worst_v < 0.05);  // V: single bf16 GEMM
    CHECK(worst_k < 0.06);  // K: bf16 GEMM + k-norm + NeoX RoPE
  }

  // ======================================================================
  // GATE 3 — block proposal STRICT-or-ratified-near-tie vs the real draft.
  // ======================================================================
  {
    const json pp = LoadJson(fdir / "propose_ref.json");
    const auto block_input_ids = IntVec<int32_t>(pp["input_ids"]);
    const auto block_positions = IntVec<int32_t>(pp["block_positions"]);
    const int64_t Tq = static_cast<int64_t>(block_input_ids.size());
    std::vector<int32_t> ctx_cu = {0, static_cast<int32_t>(C)};
    std::vector<int32_t> cu = {0, static_cast<int32_t>(Tq)};

    std::vector<std::vector<float>> per_layer;
    std::vector<float> final_h;
    std::vector<float> logits = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
        context_states, context_positions, ctx_cu, block_input_ids, block_positions, cu, w, cfg, q,
        &per_layer, &final_h);

    const auto ref_layers = pp["per_layer_hidden"];
    REQUIRE(per_layer.size() == ref_layers.size());
    double worst_layer = 0.0;
    for (size_t l = 0; l < per_layer.size(); ++l)
      worst_layer =
          std::max(worst_layer, RelL2(per_layer[l], ref_layers[l].get<std::vector<float>>()));
    const double final_rel = RelL2(final_h, pp["final_hidden"].get<std::vector<float>>());
    MESSAGE("propose worst layer-hidden rel-L2 = " << worst_layer
                                                   << "  final rel-L2 = " << final_rel);
    // Sanity envelope: OUR internal per-layer context-KV diverges from vLLM's fused
    // ctxkv in bf16, so the context-aware forward accrues more than the D2 context-
    // free forward; a ROOT divergence (wrong context/mask) would be >> this.
    CHECK(worst_layer < 0.12);
    CHECK(final_rel < 0.12);

    // Proposed-id gate: near-tie-distributional form (user-ratified). STRICT top-1
    // where vLLM's head is deterministic (gap > TIE), tied-cluster membership at a
    // bf16 near-tie. kTie=0.125 as calibrated for the real 27B draft head (D2 gate).
    const double kTie = 0.125;
    const int64_t V = pp["draft_vocab"];
    REQUIRE(static_cast<int64_t>(logits.size()) == Tq * V);
    const auto topk_ids = pp["topk_ids"];
    const auto topk_vals = pp["topk_vals"];
    int strict = 0, near_tie = 0;
    for (int64_t t = 0; t < Tq; ++t) {
      int64_t best = 0;
      float bv = logits[static_cast<size_t>(t * V)];
      for (int64_t j = 1; j < V; ++j)
        if (logits[static_cast<size_t>(t * V + j)] > bv) {
          bv = logits[static_cast<size_t>(t * V + j)];
          best = j;
        }
      const int64_t ref_top1 = topk_ids[t][0];
      const double v0 = topk_vals[t][0];
      const double gap = v0 - static_cast<double>(topk_vals[t][1]);
      if (gap > kTie) {
        ++strict;
        CHECK_MESSAGE(best == ref_top1, "row " << t << " (deterministic gap " << gap << ") argmax "
                                               << best << " != vLLM " << ref_top1);
      } else {
        ++near_tie;
        bool in_cluster = false;
        for (size_t r = 0; r < topk_ids[t].size(); ++r)
          if (static_cast<int64_t>(topk_ids[t][r]) == best &&
              v0 - static_cast<double>(topk_vals[t][r]) <= kTie)
            in_cluster = true;
        CHECK_MESSAGE(in_cluster, "row " << t << " (near-tie gap " << gap << ") argmax " << best
                                         << " not in vLLM's tied top-k cluster");
      }
    }
    MESSAGE("proposal parity: " << strict << " deterministic STRICT-matched, " << near_tie
                                << " bf16-near-tie cluster-matched (Tq=" << Tq << ")");
  }
}
