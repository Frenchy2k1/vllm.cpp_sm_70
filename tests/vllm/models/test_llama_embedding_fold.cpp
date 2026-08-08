// ARCH-ONE-SURFACE fold ROW 6 gate — embeddings through the ONE surface.
//
// The pooling lane's cosine gate (tests/vllm/v1/worker/gpu/pool/
// test_pooling_runner.cpp) is STRUCTURAL: it drives the PoolingRunner over a
// synthetic hidden buffer, BYPASSING the registry. This test is the fold's
// correctness anchor: the SAME correctness statement (LAST-token pool + L2
// normalize, checked against an independent DOUBLE-PRECISION reference) now
// runs THROUGH the registry/runner path, on the committed tiny `LlamaModel`
// fixture (scripts/mm/llama_embed_fixture_gen.py, the #121 committed-fixture
// precedent), with a REGISTRY-PATH IDENTITY arm: the full-engine embedding
// (LoadedEngine -> scheduler -> GPUModelRunner::pool_tokens) must be
// IDENTICAL to the direct ModelRegistry::Forward + PoolingRunner path.
//
// Upstream mirror grounding:
//   registry.py:230  "LlamaModel": ("llama", "LlamaForCausalLM") in
//                    _EMBEDDING_MODELS
//   adapters.py:230  as_embedding_model — backbone forward, NO lm_head,
//                    DispatchPooler.for_embedding (LAST, interfaces_base.py:160)
//   model_runner.py:368-369, 1586-1607 — PoolingRunner built iff pooling model;
//                    pool replaces sample
//   pooling_runner.py:29-42 — gather at logits_indices + normalize; is_valid
//   scheduler.py:1718-1721 — pooling stops as soon as there is output
//   config/vllm.py:1068-1073 — async scheduling OFF for pooling models
//
// HONEST RESIDUAL: the fixture weights are synthetic (deterministic seed). A
// REAL embedding checkpoint (the e5-mistral class) through this fold — the
// `vllm.LLM(task="embed").encode` oracle cosine — is the NAMED residual; no
// cosine-vs-oracle number is fabricated here.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/worker/gpu/pool/pooling_runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::ModelRegistry;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;

std::string FixtureDir() { return std::string(LLAMA_EMBED_FIXTURE_DIR); }

// The fixture's prompt: 5 in-vocab token ids (the tokenizer.json maps
// "the quick brown fox sat" onto these exact ids).
const std::vector<int32_t> kPromptIds = {0, 1, 2, 3, 9};

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

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

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(
          static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
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

double L2(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * x;
  return std::sqrt(s);
}

double Cosine(const std::vector<float>& a, const std::vector<double>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += b[i] * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

// The DIRECT registry path: ModelRegistry::Load + ModelRegistry::Forward (the
// registered pooling forward -> [1, H] last-token hidden) + the landed
// PoolingRunner over the model-owned pooler. Returns {pooled, raw_hidden}.
std::pair<std::vector<float>, std::vector<float>> DirectRegistryEmbed() {
  const std::string dir = FixtureDir();
  HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(dir + "/model.safetensors"));
  std::unique_ptr<vllm::LoadedModel> model =
      ModelRegistry::Load(config, vllm::ModelSource::FromSafetensors(shards));
  REQUIRE(model != nullptr);
  REQUIRE(model->registration().info.is_pooling_model);
  REQUIRE(model->pooler() != nullptr);

  const int64_t T = static_cast<int64_t>(kPromptIds.size());
  std::vector<int32_t> positions;
  for (int64_t t = 0; t < T; ++t) positions.push_back(static_cast<int32_t>(t));
  CachePool pool(config, /*num_blocks=*/4, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const GDNAttentionMetadata gm{};  // dense: unused
  std::vector<vllm::GdnStateCache> gdn_state;
  vt::Queue q = Q();
  // gather at the LAST token — upstream pooling_runner.py:36
  // `hidden_states[input_batch.logits_indices]`.
  const std::vector<int32_t> logits_indices = {static_cast<int32_t>(T - 1)};
  ModelForwardInput in{kPromptIds, positions, am,     gm,
                       pool.attn_kv, gdn_state, config, q,
                       logits_indices};
  in.num_reqs = 1;
  in.gather_logits = true;
  vllm::ForwardLogits fl = ModelRegistry::Forward(*model, in);

  // The pooling forward returns a HOST [1, H] hidden carrier (vocab == H).
  REQUIRE(!fl.on_device());
  REQUIRE(fl.rows == 1);
  REQUIRE(fl.vocab == config.hidden_size);
  std::vector<float> hidden = fl.host;

  // The landed PoolingRunner over the model's own DispatchPooler.
  vllm::PoolingRunner runner(*model->pooler());
  vllm::PoolingMetadata md;
  md.pooling_cursor.first_token_indices = {0};
  md.pooling_cursor.last_token_indices = {0};
  md.pooling_cursor.prompt_lens = {1};
  md.pooling_cursor.seq_lens = {1};
  md.pooling_cursor.num_scheduled_tokens = {1};
  vllm::PoolingParams pp;
  pp.task = vllm::PoolingTask::kEmbed;
  pp.use_activation = true;
  md.pooling_params = {pp};
  md.tasks = {vllm::PoolingTask::kEmbed};
  vt::Tensor rows = vt::Tensor::Contiguous(
      hidden.data(), vt::DType::kF32, vt::Device{vt::DeviceType::kCPU, 0},
      {1, config.hidden_size});
  vllm::PoolerOutput out = runner.Pool(rows, md);
  REQUIRE(out.size() == 1u);
  return {out[0], hidden};
}

// The FULL ENGINE path: LoadedEngine::FromModelDir -> LLMEngine::embed ->
// scheduler -> GPUModelRunner::pool_tokens (the ONE path vllm_embed and
// /v1/embeddings drive).
std::vector<float> EngineEmbed(int max_num_batched_tokens = 0) {
  vllm::entrypoints::EngineParams params;
  params.max_model_len = 64;
  if (max_num_batched_tokens > 0)
    params.max_num_batched_tokens = max_num_batched_tokens;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(FixtureDir(), params);
  REQUIRE(loaded != nullptr);
  CHECK(loaded->is_pooling_model());
  // config/vllm.py:1068-1073 mirror: async scheduling resolves OFF for a
  // pooling model (the landed ResolveAsyncScheduling arm, now WIRED).
  CHECK_FALSE(loaded->async_scheduling_enabled());

  vllm::RequestOutput ro = loaded->engine().embed(kPromptIds);
  REQUIRE(ro.finished);
  REQUIRE(ro.pooling_output.has_value());
  return *ro.pooling_output;
}

}  // namespace

TEST_CASE(
    "llama embedding fold: registry resolves LlamaModel as a pooling model") {
  const std::vector<std::string> archs{"LlamaModel"};
  const vllm::ModelRegistration& reg =
      ModelRegistry::Resolve(std::span<const std::string>(archs));
  CHECK(reg.architecture == "LlamaModel");
  CHECK(reg.info.is_pooling_model);
  CHECK_FALSE(reg.info.is_text_generation_model);
}

TEST_CASE(
    "llama embedding fold: direct registry path matches the double-precision "
    "LAST+normalize reference (the lane's cosine gate, registry-anchored)") {
  auto [pooled, hidden] = DirectRegistryEmbed();
  REQUIRE(pooled.size() == hidden.size());

  // Independent double-precision reference over the SAME hidden row: L2
  // normalize (pooling_runner.py:38 F.normalize; the pooling lane's reference).
  std::vector<double> ref(hidden.size());
  double n = 0.0;
  for (size_t i = 0; i < hidden.size(); ++i) {
    ref[i] = static_cast<double>(hidden[i]);
    n += ref[i] * ref[i];
  }
  n = std::sqrt(n);
  REQUIRE(n > 0.0);
  for (double& x : ref) x /= n;

  CHECK(L2(pooled) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(Cosine(pooled, ref) == doctest::Approx(1.0).epsilon(1e-6));
  for (size_t i = 0; i < pooled.size(); ++i) {
    CAPTURE(i);
    CHECK(pooled[i] == doctest::Approx(ref[i]).epsilon(1e-5));
  }
}

TEST_CASE(
    "llama embedding fold: the FULL ENGINE path is identical to the direct "
    "registry path (the registry-path identity arm)") {
  auto [direct, hidden] = DirectRegistryEmbed();
  (void)hidden;
  const std::vector<float> engine = EngineEmbed();
  REQUIRE(engine.size() == direct.size());
  CHECK(L2(engine) == doctest::Approx(1.0).epsilon(1e-6));
  for (size_t i = 0; i < engine.size(); ++i) {
    CAPTURE(i);
    // Same op sequence (registry forward, gather-at-last, normalize) on both
    // paths — identical to fp32 round-off.
    CHECK(engine[i] == doctest::Approx(direct[i]).epsilon(1e-5));
  }
}

TEST_CASE(
    "llama embedding fold: chunked prefill pools only the FULLY prefilled "
    "prompt (is_valid gating) and matches the unchunked vector") {
  // max_num_batched_tokens=2 forces the 5-token prompt through 3 prefill
  // chunks; the runner reports nullopt for the partial chunks (the
  // seq_lens == prompt_len validity predicate, pooling_runner.py:40-41) and
  // the scheduler keeps the request running until the LAST chunk pools it.
  const std::vector<float> chunked = EngineEmbed(/*max_num_batched_tokens=*/2);
  const std::vector<float> whole = EngineEmbed();
  REQUIRE(chunked.size() == whole.size());
  for (size_t i = 0; i < chunked.size(); ++i) {
    CAPTURE(i);
    CHECK(chunked[i] == doctest::Approx(whole[i]).epsilon(1e-5));
  }
}
