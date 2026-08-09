// DSpark draft model — the Markov transition head and the draft->target vocab
// map (SPEC-DSPARK W2). See the header for scope and the exact upstream anchors.
//
// Ported from vllm/model_executor/models/qwen3_dspark.py @ 555967922:
// DSparkMarkovHead.embed :61-63, .bias :65-67, map_draft_to_target :137-141,
// the draft_vocab_size default :99-100. The parallel backbone is the INHERITED
// DFlash model and is not re-implemented here.
#include "vllm/model_executor/models/qwen3_dspark.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Tensor;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight

}  // namespace

std::vector<float> Qwen3DSparkModel::MarkovEmbed(const std::vector<int32_t>& prev_token_ids,
                                                 const Qwen3DSparkWeights& weights,
                                                 vt::Queue& queue) {
  VT_CHECK(!prev_token_ids.empty(), "qwen3_dspark markov_embed: no previous tokens");
  VT_CHECK(weights.markov_rank > 0 && weights.vocab_size > 0,
           "qwen3_dspark markov_embed: dims unresolved (call ResolveDsparkDims)");
  VT_CHECK(!weights.markov_w1.Empty(), "qwen3_dspark markov_embed: markov_w1 missing");
  for (int32_t id : prev_token_ids) {
    VT_CHECK(id >= 0 && static_cast<int64_t>(id) < weights.vocab_size,
             "qwen3_dspark markov_embed: previous token id out of target vocab");
  }
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t B = static_cast<int64_t>(prev_token_ids.size());
  const int64_t R = weights.markov_rank;

  // markov_w1 is a VocabParallelEmbedding: a plain [vocab, r] row gather, the
  // same op the backbone uses for embed_tokens.
  DBuf embed(d, DType::kBF16, {B, R});
  {
    Tensor table = ResidentWeight(d, weights.markov_w1, {weights.vocab_size, R});
    DBuf ids(d, DType::kI32, {B}, prev_token_ids.data());
    vt::Embedding(d.q, embed.t(), table, ids.t());
  }
  DBuf embed32(d, DType::kF32, {B, R});
  vt::CastF32(d.q, embed32.t(), embed.t());
  std::vector<float> out(static_cast<size_t>(B) * R);
  embed32.Download(d, out.data());
  return out;
}

std::vector<float> Qwen3DSparkModel::MarkovBias(const std::vector<float>& markov_embed,
                                                int64_t num_rows,
                                                const Qwen3DSparkWeights& weights,
                                                vt::Queue& queue) {
  const int64_t R = weights.markov_rank;
  const int64_t V = weights.draft_vocab_size;
  VT_CHECK(num_rows > 0 && R > 0 && V > 0, "qwen3_dspark markov_bias: bad dims");
  VT_CHECK(static_cast<int64_t>(markov_embed.size()) == num_rows * R,
           "qwen3_dspark markov_bias: markov_embed must be [B, markov_rank]");
  VT_CHECK(!weights.markov_w2.Empty(), "qwen3_dspark markov_bias: markov_w2 missing");
  Dev d{vt::GetBackend(queue.device.type), queue};

  // logits_processor(markov_w2, markov_embed): the same [B,r] x [V,r]^T MatmulBT
  // the lm_head runs, scaled by logit_scale (1.0 in every shipped checkpoint).
  DBuf e32(d, DType::kF32, {num_rows, R}, markov_embed.data());
  DBuf eb(d, DType::kBF16, {num_rows, R});
  vt::CastBf16(d.q, eb.t(), e32.t());
  Tensor w2 = ResidentWeight(d, weights.markov_w2, {V, R});
  DBuf bias(d, DType::kF32, {num_rows, V});
  vt::MatmulBT(d.q, bias.t(), eb.t(), w2);
  std::vector<float> out(static_cast<size_t>(num_rows) * V);
  bias.Download(d, out.data());
  if (weights.logit_scale != 1.0f) {
    for (float& v : out) v *= weights.logit_scale;
  }
  return out;
}

std::vector<float> Qwen3DSparkModel::MarkovBiasForTokens(
    const std::vector<int32_t>& prev_token_ids, const Qwen3DSparkWeights& weights,
    vt::Queue& queue) {
  return MarkovBias(MarkovEmbed(prev_token_ids, weights, queue),
                    static_cast<int64_t>(prev_token_ids.size()), weights, queue);
}

int32_t Qwen3DSparkModel::MapDraftToTarget(int32_t draft_id,
                                           const Qwen3DSparkWeights& weights) {
  // Upstream: `draft_ids + self.draft_id_to_target_id[draft_ids]` when the table
  // exists, else the identity (qwen3_dspark.py:137-141). The table holds the
  // OFFSET, so the target id is draft_id + table[draft_id].
  if (weights.draft_id_to_target_id.empty()) {
    return draft_id;
  }
  VT_CHECK(draft_id >= 0 &&
               static_cast<size_t>(draft_id) < weights.draft_id_to_target_id.size(),
           "qwen3_dspark map_draft_to_target: draft id out of the draft vocab");
  return draft_id + weights.draft_id_to_target_id[static_cast<size_t>(draft_id)];
}

bool Qwen3DSparkModel::IsSpeculatorsDsparkConfig(const nlohmann::json& doc) {
  // base.py:52-56: a speculators checkpoint is identified by
  // `speculators_model_type`; we only claim the dspark one.
  return doc.is_object() && doc.contains("speculators_model_type") &&
         doc.at("speculators_model_type").is_string() &&
         doc.at("speculators_model_type").get<std::string>() == "dspark";
}

nlohmann::json Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(
    const nlohmann::json& doc) {
  VT_CHECK(IsSpeculatorsDsparkConfig(doc),
           "qwen3_dspark: not a speculators-format DSpark config "
           "(speculators_model_type != \"dspark\")");
  // validate_speculators_config (base.py:90-95).
  VT_CHECK(doc.contains("transformer_layer_config") &&
               doc.at("transformer_layer_config").is_object(),
           "qwen3_dspark: a speculators config must provide an object "
           "\"transformer_layer_config\"");

  // extract_transformers_pre_trained_config (base.py:60-64): start from the
  // transformer layer config, then apply the algorithm updater IN PLACE.
  nlohmann::json out = doc.at("transformer_layer_config");

  // update_dspark (algos.py:152-178).
  out["architectures"] = nlohmann::json::array({"Qwen3DSparkModel"});
  // Default FALSE on this path (the anchor is a bonus token, the DFlash 1+N
  // fill-in layout), algos.py:157-159.
  out["sample_from_anchor"] =
      doc.contains("sample_from_anchor") && doc.at("sample_from_anchor").is_boolean()
          ? doc.at("sample_from_anchor").get<bool>()
          : false;

  VT_CHECK(doc.contains("aux_hidden_state_layer_ids") &&
               doc.at("aux_hidden_state_layer_ids").is_array(),
           "qwen3_dspark: a speculators DSpark config must provide "
           "\"aux_hidden_state_layer_ids\"");
  const nlohmann::json& aux = doc.at("aux_hidden_state_layer_ids");
  out["eagle_aux_hidden_state_layer_ids"] = aux;
  // "DSpark indexes target layers as aux_id - 1" (algos.py:164-165).
  nlohmann::json target_layer_ids = nlohmann::json::array();
  for (const auto& id : aux) {
    VT_CHECK(id.is_number_integer(),
             "qwen3_dspark: aux_hidden_state_layer_ids must be integers");
    target_layer_ids.push_back(id.get<int64_t>() - 1);
  }
  out["target_layer_ids"] = target_layer_ids;

  // The copied DSpark keys (algos.py:167-178) — only when present and non-null.
  for (const char* key : {"draft_vocab_size", "target_hidden_size", "mask_token_id",
                          "markov_rank", "markov_head_type", "block_size",
                          "enable_confidence_head", "confidence_head_with_markov"}) {
    if (doc.contains(key) && !doc.at(key).is_null()) {
      out[key] = doc.at(key);
    }
  }
  return out;
}

int Qwen3DSparkModel::SpeculatorsNumSpeculativeTokens(const nlohmann::json& doc) {
  // build_vllm_speculative_config (base.py:113-136).
  VT_CHECK(doc.is_object() && doc.contains("speculators_config") &&
               doc.at("speculators_config").is_object(),
           "qwen3_dspark: speculators config missing \"speculators_config\"");
  const nlohmann::json& spec = doc.at("speculators_config");
  VT_CHECK(spec.contains("proposal_methods") &&
               spec.at("proposal_methods").is_array() &&
               !spec.at("proposal_methods").empty(),
           "qwen3_dspark: no proposal methods found in speculators config");
  const nlohmann::json& first = spec.at("proposal_methods").at(0);
  VT_CHECK(first.is_object() && first.contains("speculative_tokens") &&
               first.at("speculative_tokens").is_number_integer(),
           "qwen3_dspark: missing \"speculative_tokens\" in the proposal method");
  const int k = first.at("speculative_tokens").get<int>();
  VT_CHECK(k > 0, "qwen3_dspark: speculative_tokens must be positive");
  return k;
}

void Qwen3DSparkModel::ResolveDsparkDims(const HfConfig& config,
                                         Qwen3DSparkWeights& weights) {
  VT_CHECK(config.raw.is_object(), "qwen3_dspark: draft config has no raw object");
  // markov_rank is REQUIRED: upstream reads config.markov_rank unguarded
  // (qwen3_dspark.py:86), so a config without it is not a DSpark draft.
  VT_CHECK(config.raw.contains("markov_rank") &&
               config.raw.at("markov_rank").is_number_integer(),
           "qwen3_dspark: the draft config has no integer \"markov_rank\" — not a "
           "DSpark draft checkpoint");
  weights.markov_rank = config.raw.at("markov_rank").get<int64_t>();
  VT_CHECK(weights.markov_rank > 0, "qwen3_dspark: markov_rank must be positive");

  weights.vocab_size = config.vocab_size;
  VT_CHECK(weights.vocab_size > 0, "qwen3_dspark: the draft config has no vocab_size");
  // draft_vocab_size defaults to vocab_size (qwen3_dspark.py:99-100).
  weights.draft_vocab_size = weights.vocab_size;
  if (config.raw.contains("draft_vocab_size") &&
      config.raw.at("draft_vocab_size").is_number_integer()) {
    weights.draft_vocab_size = config.raw.at("draft_vocab_size").get<int64_t>();
    VT_CHECK(weights.draft_vocab_size > 0 &&
                 weights.draft_vocab_size <= weights.vocab_size,
             "qwen3_dspark: draft_vocab_size must be in (0, vocab_size]");
  }
  weights.backbone.draft_vocab_size = weights.draft_vocab_size;

  weights.logit_scale = 1.0f;  // qwen3_dspark.py:110 getattr(config, "logit_scale", 1.0)
  if (config.raw.contains("logit_scale") && config.raw.at("logit_scale").is_number()) {
    weights.logit_scale = config.raw.at("logit_scale").get<float>();
  }
}

}  // namespace vllm
