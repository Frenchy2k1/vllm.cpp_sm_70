// Ported from: vllm/v1/engine/logprobs.py @ 555967922 (vLLM 0.26.0.dev0)
//
// LogprobsProcessor: the per-request incremental logprobs accumulator the
// OutputProcessor owns. It consumes each step's EngineCoreOutput (new_logprobs
// for the decode tokens, new_prompt_logprobs_tensors during prefill),
// detokenizes each token id, and appends one {token_id -> Logprob} dict per
// position to the request's SampleLogprobs / PromptLogprobs. Mirrors upstream
// LogprobsProcessor.from_new_request / _update_sample_logprobs /
// _update_prompt_logprobs / pop_prompt_logprobs / update_from_output.
//
// DEVIATIONS, recorded:
//   - Detokenization uses our Tokenizer::Decode({id}) per token id (the
//     `tokenizer.decode([token_id])` path). The upstream U+FFFD byte-fallback
//     correction (_verify_tokens / _correct_decoded_token, which stitches a
//     multi-byte UTF-8 char split across byte-fallback tokens using preceding
//     sampled-token context) is a decode-quality refinement for
//     byte-fallback tokenizers and is NOT ported here; it is exercised only on
//     tokens ending in U+FFFD. Recorded as a follow-on (the logprob VALUES,
//     ranks and token ids — the parity-load-bearing fields — are exact).
//   - We port the non-flat (list[dict]) container only (see vllm/logprobs.h).
#ifndef VLLM_V1_ENGINE_LOGPROBS_H_
#define VLLM_V1_ENGINE_LOGPROBS_H_

#include <optional>

#include "vllm/logprobs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/types.h"

namespace vllm::tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h
}

namespace vllm::v1 {

// LogprobsProcessor (@dataclass, vllm/v1/engine/logprobs.py:29-67). `tokenizer`
// may be nullptr (detokenization disabled => Logprob.decoded_token stays None).
class LogprobsProcessor {
 public:
  // from_new_request (:42-67): engage the sample / prompt accumulators per the
  // request's sampling params. num_logprobs / num_prompt_logprobs use our `-1`
  // ("all") sentinel exactly as SamplingParams.
  static LogprobsProcessor FromNewRequest(const tok::Tokenizer* tokenizer,
                                          const SamplingParams& sampling_params);

  // update_from_output (:348-352): apply this step's sample + prompt logprobs.
  void update_from_output(const EngineCoreOutput& output);

  // pop_prompt_logprobs (:189-206): return + forget the accumulated prompt
  // logprobs (DELTA semantics — emitted once at end of prefill). None when
  // prompt logprobs are disabled.
  std::optional<PromptLogprobs> pop_prompt_logprobs();

  // Accessors (upstream reads the fields directly in make_request_output).
  const std::optional<SampleLogprobs>& logprobs() const { return logprobs_; }
  const std::optional<PromptLogprobs>& prompt_logprobs() const {
    return prompt_logprobs_;
  }
  std::optional<double> cumulative_logprob() const { return cumulative_logprob_; }

 private:
  void UpdateSampleLogprobs(const LogprobsTensors& logprobs_lists);
  void UpdatePromptLogprobs(const LogprobsTensors& prompt_logprobs_tensors);

  const tok::Tokenizer* tokenizer_ = nullptr;
  std::optional<SampleLogprobs> logprobs_;
  std::optional<PromptLogprobs> prompt_logprobs_;
  std::optional<double> cumulative_logprob_;
  std::optional<int> num_logprobs_;
  std::optional<int> num_prompt_logprobs_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_ENGINE_LOGPROBS_H_
