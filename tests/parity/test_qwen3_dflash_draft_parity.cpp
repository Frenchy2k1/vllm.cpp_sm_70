// vllm.cpp original. DFlash draft-forward PARITY gate vs the dumped vLLM DFlash
// draft reference (SPEC-DFLASH D2, DF-DRAFT-MODEL — the decisive D2 GPU gate the
// dev-box CPU session could not run). Reference: scripts/spec/d2_dflash_draft_ref.py
// dumps, from the REAL loaded vLLM draft (DFlashQwen3ForCausalLM @ 555967922, via
// collective_rpc into the worker): the fc combine over a fixed synthetic aux, and
// the CONTEXT-FREE (1+k) mask-block forward (per-layer hidden + final hidden +
// per-row top-k ids/vals) built from vLLM's OWN loaded submodules with the paged
// attention core replaced by the documented in-block attention (full=bidirectional,
// SWA=causal-in-window) — exactly what vt::DFlashBlockAttention computes in
// isolation from D3's context-KV precompute.
//
// This test loads the SAME z-lab/Qwen3.6-27B-DFlash checkpoint (draft weights) +
// the target's SHARED embed_tokens/lm_head (the draft ckpt omits them), runs OUR
// Qwen3DFlashModel::CombineAuxFeatures + ForwardBlockLogits on the identical mask
// block, and gates per-stage rel-L2 within the bf16 (f32-softmax) envelope + STRICT
// top-1 proposed-id identity. GPU + checkpoint + fixture gated: SKIPs loudly when
// any is absent (CPU/CI dev box). The RED proofs (causal-flip / reversed-tap) live
// in the deterministic CPU unit test test_qwen3_dflash_forward.cpp.
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
// here took the first `directory_iterator` entry under `<repo>/snapshots/`, so
// which of `unsloth/Qwen3.6-27B-NVFP4`'s two cached revisions this gate measured
// was a property of readdir order. It is DELETED rather than left beside the
// pinned call -- a helper that can resolve a checkpoint without a revision is
// the defect, and one kept "just in case" is how the defect comes back.
// Resolution now goes through parity::HfSnapshot, which skips rather than
// substitutes.

std::vector<std::string> Shards(const std::string& dir) {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec))
    if (e.path().extension() == ".safetensors") out.push_back(e.path().string());
  return out;
}

// Copy a named BF16 tensor out of a checkpoint's shards into an OwnedTensor.
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

// rel-L2 = ||a-b|| / max(||b||, eps).
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

// Build the 27B DFlash draft HfConfig from the dumped fixture config (mirrors the
// CPU unit test's MakeConfig with the real z-lab dims).
HfConfig MakeConfig(const json& c) {
  HfConfig cfg;
  cfg.hidden_size = c["hidden_size"];
  cfg.num_attention_heads = c["num_attention_heads"];
  cfg.num_key_value_heads = c["num_key_value_heads"];
  cfg.head_dim = c["head_dim"];
  cfg.rotary_dim = c["head_dim"];  // full rotary
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

}  // namespace

TEST_CASE("qwen3_dflash draft-forward parity vs the dumped vLLM DFlash draft") {
  // Resolve FIRST, then probe CUDA. The pinned revisions are reported on every
  // box, including the CPU dev box that can never run the body -- a skip banner
  // that says only "no CUDA backend" hides which checkpoint the gate would have
  // demanded, and makes the pin unobservable exactly where it is cheapest to
  // check.
  const std::string draft_dir = parity::Qwen27DFlashDraftSnapshot();
  const std::string target_dir = parity::Qwen27NvfP4Snapshot();
  const bool has_cuda = HasCuda();
  const fs::path fdir = fs::path(PARITY_GOLDENS_DIR) / "dflash_27b_draft";
  if (!has_cuda || draft_dir.empty() || target_dir.empty() ||
      !fs::exists(fdir / "block_ref.json")) {
    // Name the revisions, not just the repos. A skip that says only "checkpoint
    // absent" cannot be told apart from a skip caused by holding the WRONG
    // revision of a repo that is very much present -- which is now a real and
    // intended outcome, and the reader has to be able to see it.
    MESSAGE("SKIP (dev box): DFlash draft parity needs CUDA (got: "
            << std::string(has_cuda ? "yes" : "NO") << "), z-lab/Qwen3.6-27B-DFlash @"
            << std::string(parity::kQwen27DFlashDraftRevision)
            << " (got: " << (draft_dir.empty() ? "ABSENT" : draft_dir) << "), "
            << "unsloth/Qwen3.6-27B-NVFP4 @" << std::string(parity::kQwen27NvfP4Revision)
            << " (got: " << (target_dir.empty() ? "ABSENT" : target_dir) << "), "
            << "and the dumped fixtures. A cache holding a DIFFERENT revision of "
               "either repo skips rather than being substituted (#471).");
    return;
  }

  const json cfg_j = LoadJson(fdir / "config.json");
  const HfConfig cfg = MakeConfig(cfg_j);
  const int64_t H = cfg.hidden_size;
  const int64_t num_taps = cfg_j["num_taps"];
  const int32_t mask_id = cfg_j["mask_token_id"];

  // Load the draft (layer weights + fc + norms; embed/lm_head empty — shared).
  std::vector<SafetensorsFile> dshards;
  for (const std::string& p : Shards(draft_dir)) dshards.push_back(SafetensorsFile::Open(p));
  Qwen3DFlashWeights w = LoadQwen3DFlash(dshards, cfg, num_taps, mask_id);
  CHECK(w.embed_tokens.Empty());  // draft ckpt omits it (loader tolerance)
  CHECK(w.lm_head.Empty());

  // Supply the target's SHARED embed_tokens + lm_head.
  std::vector<SafetensorsFile> tshards;
  for (const std::string& p : Shards(target_dir)) tshards.push_back(SafetensorsFile::Open(p));
  w.embed_tokens = LoadTargetBf16(tshards, "model.language_model.embed_tokens.weight", false);
  w.lm_head = LoadTargetBf16(tshards, "lm_head.weight", true);
  REQUIRE_FALSE(w.embed_tokens.Empty());
  REQUIRE_FALSE(w.lm_head.Empty());
  w.draft_vocab_size = w.lm_head.shape[0];

  vt::Queue q = GpuQ();

  // ---- fc parity: OUR CombineAuxFeatures vs vLLM's combine_hidden_states.
  const json fc = LoadJson(fdir / "fc_ref.json");
  const int64_t T = fc["T"];
  const int64_t Fin = H * num_taps;
  std::vector<float> aux(static_cast<size_t>(T * Fin));
  for (size_t i = 0; i < aux.size(); ++i)
    aux[i] = 0.2f * std::sin(0.3 * static_cast<double>(i));  // == the ref's aux
  std::vector<float> comb = Qwen3DFlashModel::CombineAuxFeatures(aux, T, w, cfg, q);
  const double fc_rel = RelL2(comb, fc["comb"].get<std::vector<float>>());
  MESSAGE("fc combine rel-L2 vs vLLM = " << fc_rel);
  CHECK(fc_rel < 0.02);  // bf16 GEMM envelope

  // ---- block forward parity: per-layer + final hidden + top-1 proposed id.
  const json blk = LoadJson(fdir / "block_ref.json");
  const auto input_ids = blk["input_ids"].get<std::vector<int32_t>>();
  const auto positions = blk["positions"].get<std::vector<int32_t>>();
  const auto cu = blk["cu_seqlens"].get<std::vector<int32_t>>();
  std::vector<std::vector<float>> per_layer;
  std::vector<float> final_h;
  std::vector<float> logits =
      Qwen3DFlashModel::ForwardBlockLogits(input_ids, positions, cu, w, cfg, q, &per_layer, &final_h);

  const auto ref_layers = blk["per_layer_hidden"];
  REQUIRE(per_layer.size() == ref_layers.size());
  double worst_layer = 0.0;
  for (size_t l = 0; l < per_layer.size(); ++l) {
    const double r = RelL2(per_layer[l], ref_layers[l].get<std::vector<float>>());
    MESSAGE("layer " << l << " hidden rel-L2 = " << r);
    worst_layer = std::max(worst_layer, r);
  }
  const double final_rel = RelL2(final_h, blk["final_hidden"].get<std::vector<float>>());
  MESSAGE("final hidden rel-L2 = " << final_rel);
  CHECK(worst_layer < 0.05);  // deep bf16 hidden envelope (f32 attn softmax)
  CHECK(final_rel < 0.05);

  // Proposed-id gate — the near-tie distributional form (user-ratified, see
  // [[near-tie-distributional-gate]]): STRICT top-1 identity where vLLM's head is
  // DETERMINISTIC (top1-top2 logit gap > TIE), and tied-CLUSTER membership where
  // vLLM's own head sits at a bf16 near-tie (gap <= TIE). At draft_vocab magnitude
  // ~8, one bf16 ULP is 0.0625, so TIE=0.125 (two ULPs) is the justified band; a
  // real divergence (our top-1 outside vLLM's top-k, or hidden > envelope above)
  // still FAILS. This mirrors vLLM-DFlash-ON's own greedy non-determinism at k=16
  // near-ties measured at D0.
  const double kTie = 0.125;
  const int64_t V = blk["draft_vocab"];
  REQUIRE(static_cast<int64_t>(logits.size()) == T * V);
  const auto topk_ids = blk["topk_ids"];
  const auto topk_vals = blk["topk_vals"];
  int strict = 0, near_tie = 0;
  for (int64_t t = 0; t < T; ++t) {
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
    if (gap > kTie) {  // deterministic head -> STRICT
      ++strict;
      CHECK_MESSAGE(best == ref_top1,
                    "row " << t << " (deterministic gap " << gap << ") argmax " << best
                           << " != vLLM " << ref_top1);
    } else {  // bf16 near-tie -> our pick must be in vLLM's tied cluster
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
  MESSAGE("proposed-id parity: " << strict << " deterministic rows STRICT-matched, " << near_tie
                                 << " bf16-near-tie rows cluster-matched (T=" << T << ")");
}
