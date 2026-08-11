// Ported from: vllm/v1/engine/logprobs.py @ 555967922 (vLLM 0.26.0.dev0)
// See include/vllm/v1/engine/logprobs.h for scope + deviations.
#include "vllm/v1/engine/logprobs.h"

#include <cstddef>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm::v1 {

namespace {

// convert_ids_list_to_tokens (detokenizer_utils): decode each token id to its
// string. Per-id Decode({id}) mirrors the `tokenizer.decode([token_id])` path
// _get_decoded_token falls back to (the U+FFFD byte-fallback stitching is not
// ported — see the header). Returns None strings when detokenization is off.
std::vector<std::optional<std::string>> DecodeRow(const tok::Tokenizer* tok,
                                                  const int32_t* ids,
                                                  std::size_t width) {
  std::vector<std::optional<std::string>> out(width);
  if (tok == nullptr) return out;  // NONES: decoded_token stays None
  for (std::size_t j = 0; j < width; ++j) {
    out[j] = tok->Decode({ids[j]});
  }
  return out;
}

}  // namespace

LogprobsProcessor LogprobsProcessor::FromNewRequest(
    const tok::Tokenizer* tokenizer, const SamplingParams& sampling_params) {
  LogprobsProcessor lp;
  lp.tokenizer_ = tokenizer;
  // logprobs.py:52 reads the `num_logprobs` PROPERTY, not the raw `logprobs`
  // field: for a generative-scoring request it is len(logprob_token_ids), which
  // is the row width the sampler hands back (sampling_params.py:724-729).
  lp.num_logprobs_ = sampling_params.num_logprobs();
  lp.num_prompt_logprobs_ = sampling_params.prompt_logprobs;
  // cumulative_logprob = None if num_logprobs is None else 0.0 (:54).
  if (lp.num_logprobs_.has_value()) {
    lp.cumulative_logprob_ = 0.0;
    lp.logprobs_ = SampleLogprobs{};  // create_sample_logprobs (non-flat)
  }
  if (lp.num_prompt_logprobs_.has_value()) {
    // create_prompt_logprobs: first prompt token's logprob is None (:162-167).
    PromptLogprobs pl;
    pl.push_back(std::nullopt);
    lp.prompt_logprobs_ = std::move(pl);
  }
  return lp;
}

void LogprobsProcessor::UpdateSampleLogprobs(const LogprobsTensors& lists) {
  // _update_sample_logprobs (:69-119). Outer loop over positions (>1 only under
  // spec decode); each row is [sampled | top-k], sampled logprob first.
  const int width = lists.num_tokens_per_position;
  // A plain zero-width guard: nothing to append. It never screened the
  // num_logprobs==-1 raw-vocab shape (that one sets width == vocab), and no
  // longer needs to — since #231 the input batch widens `-1` at admission, so
  // that shape is unreachable from a live request.
  if (width <= 0) return;
  for (int pos = 0; pos < lists.num_positions; ++pos) {
    const std::size_t base = static_cast<std::size_t>(pos) *
                             static_cast<std::size_t>(width);
    const int32_t* id_row = &lists.logprob_token_ids[base];
    const float* lp_row = &lists.logprobs[base];
    const int rank = lists.selected_token_ranks[static_cast<std::size_t>(pos)];

    std::vector<int32_t> token_ids(id_row, id_row + width);
    std::vector<float> logprobs(lp_row, lp_row + width);
    std::vector<std::optional<std::string>> decoded =
        DecodeRow(tokenizer_, id_row, static_cast<std::size_t>(width));

    // Sampler puts the sampled logprob in first (:107-109).
    *cumulative_logprob_ += logprobs[0];
    AppendLogprobsForNextPosition(*logprobs_, token_ids, logprobs, decoded, rank,
                                  *num_logprobs_);
  }
}

void LogprobsProcessor::UpdatePromptLogprobs(const LogprobsTensors& tensors) {
  // _update_prompt_logprobs (:121-187). Each row is one prompt position's
  // [prompt token | top-k]; ranks[pos] is the prompt token's rank.
  const int width = tensors.num_tokens_per_position;
  if (width <= 0) return;
  for (int pos = 0; pos < tensors.num_positions; ++pos) {
    const std::size_t base = static_cast<std::size_t>(pos) *
                             static_cast<std::size_t>(width);
    const int32_t* id_row = &tensors.logprob_token_ids[base];
    const float* lp_row = &tensors.logprobs[base];
    const int rank = tensors.selected_token_ranks[static_cast<std::size_t>(pos)];

    std::vector<int32_t> token_ids(id_row, id_row + width);
    std::vector<float> logprobs(lp_row, lp_row + width);
    std::vector<std::optional<std::string>> decoded =
        DecodeRow(tokenizer_, id_row, static_cast<std::size_t>(width));

    // Reuse the sample append into a scratch SampleLogprobs, then move.
    SampleLogprobs scratch;
    AppendLogprobsForNextPosition(scratch, token_ids, logprobs, decoded, rank,
                                  *num_prompt_logprobs_);
    prompt_logprobs_->push_back(std::move(scratch.front()));
  }
}

std::optional<PromptLogprobs> LogprobsProcessor::pop_prompt_logprobs() {
  // pop_prompt_logprobs (:189-206): return + clear (DELTA semantics).
  if (!prompt_logprobs_.has_value()) return std::nullopt;
  std::optional<PromptLogprobs> plp = std::move(prompt_logprobs_);
  // "if plp: self.prompt_logprobs = []" — clear only when non-empty.
  if (plp->empty()) {
    prompt_logprobs_ = std::move(*plp);
    return prompt_logprobs_;
  }
  prompt_logprobs_ = PromptLogprobs{};
  return plp;
}

void LogprobsProcessor::update_from_output(const EngineCoreOutput& output) {
  if (output.new_logprobs.has_value()) {
    UpdateSampleLogprobs(*output.new_logprobs);
  }
  if (output.new_prompt_logprobs_tensors.has_value()) {
    UpdatePromptLogprobs(*output.new_prompt_logprobs_tensors);
  }
}

}  // namespace vllm::v1
