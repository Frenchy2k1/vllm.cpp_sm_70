// Muse Glimmer TEXT TOWER on the REAL `meta-models/Muse-Glimmer-30B` weights.
//
// ─── WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT ─────────────────────────────
//
// ESTABLISHES (when the checkpoint env vars are set and the fixtures are
// present): our loader accepts the REAL 1436-tensor bf16 checkpoint, the real
// geometry parses, the text tower forwards on real data producing finite,
// correctly shaped, NON-DEGENERATE logits, and those logits agree with a
// standalone torch transcription of vllm#51655 head `075d645af` run on the
// IDENTICAL bytes — same argmax at every position, small numeric residual.
//
// DOES NOT ESTABLISH — and this is the point of the file, so it is said first:
//
//   * NOT token-exact vs the model's own reference runtime. There is no
//     runnable Muse Glimmer reference on this machine. Released `transformers`
//     does not register `model_type: muse_glimmer` (the checkpoint declares
//     `transformers_version 5.15.0.dev0`; 5.3.0 raises
//     `ValueError: ... does not recognize this architecture`), the checkpoint
//     ships NO remote-code modelling file, and the parity pin `555967922` has
//     no `muse_glimmer` at all. The comparison here is against a SECOND
//     TRANSCRIPTION of the same upstream python
//     (`scripts/mm/muse_glimmer_text_ref.py`), not against Meta's runtime. Two
//     transcriptions agreeing rules out a large class of porting defects; it
//     cannot rule out a shared misreading of #51655.
//   * NOTHING about speed, on any axis. The pinned oracle cannot load this
//     model, so there is no denominator (specs/muse-glimmer.md §0).
//   * NOTHING about the perception encoder. This is the text tower only.
//
// ─── HOW TO RUN ──────────────────────────────────────────────────────────────
// Every case SKIPs cleanly when its env var is unset, so CI never depends on a
// 59.55 GB NAS asset.
//
//   # reduced model built from REAL tensors (first K layers, ~9.3 GB) — the
//   # fixture directory is produced by:
//   #   scripts/mm/muse_glimmer_text_ref.py --ckpt <30b> --out <dir>
//   #       --layers 4 --emit-weights
//   VLLM_MUSE_REF_DIR=<dir> ./test_muse_glimmer_real_weights
//
//   # the FULL 52-layer tower. Loads ~55.7 GB of bf16 into host RAM; only run
//   # it on a box with the headroom. The reference directory comes from the
//   # same script with `--layers 0` (which streams one layer at a time and
//   # peaks around 7 GB, so it fits where the C++ load may not).
//   VLLM_MUSE_FULL_REF_DIR=<dir> VLLM_MUSE_FULL_CKPT=<30b>
//       ./test_muse_glimmer_real_weights
#include "vllm/model_executor/models/muse_glimmer.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::HfConfig;
using vllm::MuseGlimmerModel;
using vllm::MuseGlimmerParams;
using vllm::MuseGlimmerWeights;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;

namespace {

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

// The reference bundle written by scripts/mm/muse_glimmer_text_ref.py.
struct Reference {
  std::string ref_dir;
  std::string ckpt_dir;
  nlohmann::json meta;
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<float> logits;  // [T, V] row-major, post-softcap
  int64_t T = 0;
  int64_t V = 0;
};

// Loads the bundle, or returns false when the env var is unset / the fixture is
// missing. A MISSING fixture is a skip, but a MALFORMED one is a hard failure —
// silently passing on a truncated logits file would be the worst outcome here.
bool LoadReference(const char* dir_env, const char* ckpt_env, Reference* out) {
  const char* dir = Env(dir_env);
  if (dir == nullptr) return false;
  const std::filesystem::path d(dir);
  const std::filesystem::path meta_path = d / "ref.json";
  const std::filesystem::path logits_path = d / "ref_logits.f32";
  if (!std::filesystem::exists(meta_path)) return false;

  out->ref_dir = d.string();
  const char* ckpt = Env(ckpt_env);
  out->ckpt_dir = ckpt != nullptr ? std::string(ckpt) : d.string();

  std::ifstream mf(meta_path);
  REQUIRE_MESSAGE(mf.good(), "cannot open " << meta_path.string());
  mf >> out->meta;

  for (const auto& t : out->meta.at("token_ids")) out->tokens.push_back(t.get<int32_t>());
  for (const auto& p : out->meta.at("positions")) out->positions.push_back(p.get<int32_t>());
  out->T = out->meta.at("logits_shape").at(0).get<int64_t>();
  out->V = out->meta.at("logits_shape").at(1).get<int64_t>();
  REQUIRE(out->T == static_cast<int64_t>(out->tokens.size()));

  std::ifstream lf(logits_path, std::ios::binary);
  REQUIRE_MESSAGE(lf.good(), "cannot open " << logits_path.string());
  const size_t want = static_cast<size_t>(out->T * out->V);
  out->logits.resize(want);
  lf.read(reinterpret_cast<char*>(out->logits.data()),
          static_cast<std::streamsize>(want * sizeof(float)));
  REQUIRE_MESSAGE(static_cast<size_t>(lf.gcount()) == want * sizeof(float),
                  "truncated " << logits_path.string());
  return true;
}

// Every `*.safetensors` in the checkpoint directory, in sorted (shard) order.
std::vector<std::string> ShardPaths(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".safetensors")
      paths.push_back(e.path().string());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const MuseGlimmerParams& p, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = p.text.num_key_value_heads, Dh = p.text.head_dim;
    for (int64_t l = 0; l < p.text.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh),
                       0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

int64_t ArgMax(const float* row, int64_t n) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

// The whole real-weight body, shared by the reduced and full-depth cases.
void RunRealWeightGate(const Reference& ref) {
  const std::string cfg_path = (std::filesystem::path(ref.ckpt_dir) / "config.json").string();
  const HfConfig cfg = vllm::LoadHfConfig(cfg_path);
  const MuseGlimmerParams params = vllm::ParseMuseGlimmerParams(cfg);

  // The geometry the reference ran must be the geometry we are about to run.
  // Without this a stale fixture would be compared against a different model
  // and the numeric check would be meaningless.
  REQUIRE(params.text.num_hidden_layers == ref.meta.at("num_hidden_layers").get<int64_t>());
  REQUIRE(params.text.vocab_size == ref.V);
  REQUIRE(params.text.hidden_size == ref.meta.at("hidden_size").get<int64_t>());
  CHECK(params.text.scale_query_by ==
        doctest::Approx(ref.meta.at("scale_query_by").get<double>()));
  {
    std::vector<int64_t> want;
    for (const auto& v : ref.meta.at("no_rope_layers")) want.push_back(v.get<int64_t>());
    CHECK(params.text.no_rope_layers == want);
  }

  const std::vector<std::string> paths = ShardPaths(ref.ckpt_dir);
  REQUIRE_MESSAGE(!paths.empty(), "no .safetensors under " << ref.ckpt_dir);
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) shards.push_back(vllm::SafetensorsFile::Open(p));

  const MuseGlimmerWeights w =
      vllm::LoadMuseGlimmerForConditionalGenerationWeights(shards, cfg);
  REQUIRE(w.text_loaded);
  REQUIRE(w.layers.size() == static_cast<size_t>(params.text.num_hidden_layers));
  MESSAGE("real checkpoint: accounted " << w.accounted_tensors << " / enumerated "
                                        << w.enumerated_tensors << " tensors over "
                                        << paths.size() << " shard(s)");
  CHECK(w.accounted_tensors <= w.enumerated_tensors);

  // The TEXT tower's enumerated names must ALL be present under the same
  // normalization the loader uses. This is the part this file's forward depends
  // on, so it is asserted directly rather than inferred from the aggregate
  // counter — and it is asserted for BOTH fixtures, text-only and multimodal.
  {
    MuseGlimmerParams text_only = params;
    text_only.vision.present = false;
    std::vector<std::string> normalized;
    for (const vllm::SafetensorsFile& s : shards) {
      for (const std::string& raw : s.Names()) {
        std::string canonical;
        if (vllm::NormalizeMuseGlimmerWeightName(raw, &canonical))
          normalized.push_back(canonical);
      }
    }
    std::sort(normalized.begin(), normalized.end());
    int64_t missing = 0;
    std::string first_missing;
    for (const std::string& want : vllm::EnumerateMuseGlimmerTensors(text_only)) {
      if (!std::binary_search(normalized.begin(), normalized.end(), want)) {
        ++missing;
        if (first_missing.empty()) first_missing = want;
      }
    }
    CHECK_MESSAGE(missing == 0, "text-tower names absent from the checkpoint: "
                                    << missing << ", first: " << first_missing);
  }

  // The accounting must close on EVERY checkpoint, multimodal included. This
  // assertion was demoted to a MESSAGE while the vision enumeration was short by
  // 50 (a merged `attn.qkv_proj` the checkpoint never ships, plus the missing
  // vision attention biases); the W4 enumeration correction fixed both, and
  // test_muse_glimmer_wiring now proves the released 30B's 1436 tensors are
  // accounted 1436/1436. A conditional assertion is a disarmed one, so it is
  // armed again — the same guarantee runs on the synthetic multimodal checkpoint
  // in test_muse_glimmer_wiring ("the perception encoder loads, q|k|v merged in
  // order"), which is what makes this reachable without the NAS.
  CHECK(w.accounted_tensors == w.enumerated_tensors);
  CHECK(w.embed_tokens.shape[0] == params.text.vocab_size);
  CHECK(w.embed_tokens.shape[1] == params.text.hidden_size);
  CHECK(w.lm_head.shape[0] == params.text.hidden_size);  // Matmul-B [in,out]
  CHECK(w.lm_head.shape[1] == params.text.vocab_size);

  CachePool pool(w.params, /*num_blocks=*/4, /*block_size=*/16);
  const CommonAttentionMetadata am = PrefillMeta(ref.T, 16);
  vt::Queue q = Qcpu();
  const std::vector<float> got =
      MuseGlimmerModel::Forward(ref.tokens, ref.positions, am, pool.attn_kv, w, q);

  REQUIRE(got.size() == static_cast<size_t>(ref.T * ref.V));

  // ── 1. structural + coherence, independent of the reference ──
  double absmax = 0.0;
  for (float x : got) {
    REQUIRE(std::isfinite(x));
    absmax = std::max(absmax, std::abs(static_cast<double>(x)));
  }
  // The final soft-cap bounds every logit by `final_logit_softcapping`.
  CHECK(absmax <= params.text.final_logit_softcapping + 1e-3);
  CHECK(absmax > 1e-3);  // not an all-zero forward

  std::vector<int64_t> ours(static_cast<size_t>(ref.T));
  std::vector<int64_t> theirs(static_cast<size_t>(ref.T));
  for (int64_t t = 0; t < ref.T; ++t) {
    ours[static_cast<size_t>(t)] = ArgMax(&got[static_cast<size_t>(t * ref.V)], ref.V);
    theirs[static_cast<size_t>(t)] =
        ArgMax(&ref.logits[static_cast<size_t>(t * ref.V)], ref.V);
  }
  {
    std::string line;
    for (int64_t id : ours) line += std::to_string(id) + " ";
    MESSAGE("ours   argmax: " << line);
    line.clear();
    for (int64_t id : theirs) line += std::to_string(id) + " ";
    MESSAGE("torch  argmax: " << line);
  }
  // Degeneracy check: a tower that has lost its weights (or normalizes them
  // away) still produces finite, in-range logits — it just emits the SAME token
  // everywhere. That failure mode is invisible to a finiteness assert, so it
  // gets its own one. Only meaningful for T > 1.
  if (ref.T > 1) {
    const bool all_same =
        std::all_of(ours.begin(), ours.end(), [&](int64_t v) { return v == ours[0]; });
    const bool ref_all_same = std::all_of(theirs.begin(), theirs.end(),
                                          [&](int64_t v) { return v == theirs[0]; });
    // Only assert non-degeneracy when the reference itself is non-degenerate:
    // a genuinely degenerate reference (e.g. a 1-layer truncation) would make
    // this a false alarm rather than a finding.
    if (!ref_all_same) CHECK_FALSE(all_same);
  }

  // ── 2. agreement with the torch transcription on the identical bytes ──
  double max_abs = 0.0;
  double sum_sq = 0.0;
  double ref_sum_sq = 0.0;
  double dot = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double a = got[i], b = ref.logits[i];
    max_abs = std::max(max_abs, std::abs(a - b));
    sum_sq += a * a;
    ref_sum_sq += b * b;
    dot += a * b;
  }
  const double cosine = dot / (std::sqrt(sum_sq) * std::sqrt(ref_sum_sq));
  MESSAGE("max|ours-torch| = " << max_abs << "   cosine = " << cosine
                               << "   |logits|max = " << absmax);

  // The argmax is the load-bearing check: it is what a token-exact gate would
  // compare, and it is insensitive to the fp32-vs-bf16 accumulation-order
  // differences that separate two honest implementations.
  CHECK(ours == theirs);
  // And the ids the reference itself recorded, so a corrupted logits blob
  // cannot make the comparison vacuously agree with itself.
  {
    std::vector<int64_t> recorded;
    for (const auto& v : ref.meta.at("argmax")) recorded.push_back(v.get<int64_t>());
    CHECK(theirs == recorded);
  }

  // Numeric bounds, calibrated against MEASURED numbers rather than taste.
  // Both sides consume the SAME bf16 bytes; the residual is accumulation order
  // (the torch reference runs every matmul in fp32 with its own blocking, our
  // CPU path uses vt's), not model disagreement.
  //
  // Observed on the reduced 4-layer real-weight model, prompt "The capital of
  // France is" (5 tokens):
  //     honest run                       max|diff| 0.089   cosine 0.999981
  //     MUTANT: query pre-scale divided
  //             by sqrt(head_dim) again  max|diff| 12.14    cosine 0.8453
  //     MUTANT: weightless QK-norm
  //             dropped from q           max|diff| 1.49     cosine 0.9953
  // Both mutants were generated by editing the reference script and rerunning
  // it against the identical weights; both FAILED this test, which is why the
  // bounds sit where they do — 1.0 is ~11x the honest residual and below the
  // weaker mutant, and 0.999 separates 0.999981 from 0.9953.
  //
  // The QK-norm mutant is the reason the numeric bound is not decorative: it
  // left the argmax stream UNCHANGED at this depth and prompt, so the token
  // comparison alone would have passed it. A gate that only compared tokens
  // here would have had a hole.
  const double tol = Env("VLLM_MUSE_TOL") != nullptr
                         ? std::atof(Env("VLLM_MUSE_TOL"))
                         : 0.05 * params.text.final_logit_softcapping;
  CHECK(max_abs <= tol);
  CHECK(cosine > 0.999);
}

}  // namespace

TEST_CASE("muse_glimmer real weights: reduced-depth tower vs the torch transcription") {
  Reference ref;
  if (!LoadReference("VLLM_MUSE_REF_DIR", "VLLM_MUSE_CKPT", &ref)) {
    MESSAGE(
        "SKIP: set VLLM_MUSE_REF_DIR to a directory produced by "
        "scripts/mm/muse_glimmer_text_ref.py --emit-weights");
    return;
  }
  MESSAGE("reduced fixture: " << ref.ref_dir << " (depth "
                              << ref.meta.at("num_hidden_layers").get<int64_t>() << "/"
                              << ref.meta.at("full_depth").get<int64_t>() << ")");
  RunRealWeightGate(ref);
}

// The committed record of the full-depth reference result. It exists because
// the fixture it summarizes is ~4 MB of logits over a 59.55 GB checkpoint and
// neither is committable; this file is, so a regenerated reference that
// disagrees with what was recorded is LOUD rather than silently accepted.
nlohmann::json LoadGolden() {
  const std::filesystem::path p =
      std::filesystem::path(VLLM_MUSE_GOLDEN_DIR) / "muse_glimmer_real_weights_golden.json";
  std::ifstream f(p);
  REQUIRE_MESSAGE(f.good(), "cannot open committed golden " << p.string());
  nlohmann::json j;
  f >> j;
  return j;
}

TEST_CASE("muse_glimmer real weights: the committed full-depth record is self-consistent") {
  // Runs everywhere, with no checkpoint: it guards the record itself.
  const nlohmann::json g = LoadGolden();
  CHECK(g.at("num_hidden_layers").get<int64_t>() == 52);
  CHECK(g.at("vocab_size").get<int64_t>() == 202048);
  CHECK(g.at("hidden_size").get<int64_t>() == 6656);
  CHECK(g.at("token_ids").size() == g.at("argmax").size());
  CHECK(g.at("argmax").size() == g.at("argmax_text").size());
  // Every recorded argmax id is in range, and the stream is NOT degenerate —
  // a record of "the model emitted the same token five times" would be a
  // finding, not a golden.
  std::vector<int64_t> ids;
  for (const auto& v : g.at("argmax")) ids.push_back(v.get<int64_t>());
  for (int64_t id : ids) CHECK((id >= 0 && id < g.at("vocab_size").get<int64_t>()));
  CHECK_FALSE(std::all_of(ids.begin(), ids.end(),
                          [&](int64_t v) { return v == ids.front(); }));
  // The recorded top-1 must actually lead its runner-up, otherwise comparing
  // tokens at this prompt would be a coin flip and the whole record is unsafe
  // to gate on.
  const auto vals = g.at("last_position_top2_values");
  CHECK(vals.at(0).get<double>() - vals.at(1).get<double>() > 1.0);
  CHECK(g.at("last_position_top2_ids").at(0).get<int64_t>() == ids.back());
  // The soft-cap really is the bound the recorded logits respect.
  CHECK(g.at("logit_absmax").get<double>() <=
        g.at("final_logit_softcapping").get<double>());
}

TEST_CASE("muse_glimmer real weights: FULL 52-layer tower vs the torch transcription") {
  Reference ref;
  if (!LoadReference("VLLM_MUSE_FULL_REF_DIR", "VLLM_MUSE_FULL_CKPT", &ref)) {
    MESSAGE(
        "SKIP: set VLLM_MUSE_FULL_REF_DIR (and VLLM_MUSE_FULL_CKPT) to run the "
        "full-depth gate; it loads ~55.7 GB of bf16 into host RAM");
    return;
  }
  REQUIRE(ref.meta.at("num_hidden_layers").get<int64_t>() ==
          ref.meta.at("full_depth").get<int64_t>());
  MESSAGE("full fixture: " << ref.ref_dir << " over checkpoint " << ref.ckpt_dir);

  // The supplied fixture must be the one this repo recorded: same prompt, same
  // token ids, same reference argmax. A fixture regenerated from a different
  // prompt, a different checkpoint revision, or a drifted reference script
  // fails HERE, before its numbers are used to bless anything.
  const nlohmann::json g = LoadGolden();
  CHECK(ref.meta.at("prompt").get<std::string>() == g.at("prompt").get<std::string>());
  CHECK(ref.meta.at("token_ids") == g.at("token_ids"));
  CHECK(ref.meta.at("argmax") == g.at("argmax"));
  CHECK(ref.meta.at("num_hidden_layers") == g.at("num_hidden_layers"));

  RunRealWeightGate(ref);
}
