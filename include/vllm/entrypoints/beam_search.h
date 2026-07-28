// Ported from: vllm/entrypoints/generate/beam_search/utils.py @ 555967922
// (BeamSearchSequence, BeamSearchOutput, BeamSearchInstance,
// get_beam_search_score, create_sort_beams_key_function) +
// vllm/entrypoints/generate/beam_search/offline.py @ 555967922
// (BeamSearchOfflineMixin.beam_search / _beam_search_step — the per-step
// expand/score/select/EOS loop) + vllm/sampling_params.py:1114 (BeamSearchParams).
//
// Scope (ROAD-V1-C7 `SAMPLE-BEAM`): beam search is an OUTER loop over the engine,
// NOT a core-sampler param. Each step runs ONE decode per active beam with
// logprobs=2*beam_width, expands each beam to those next tokens (scored by
// cumulative logprob), keeps the top-beam_width by the length-penalty score,
// retires EOS-terminated beams into `completed`, and after max_tokens (or once all
// beams complete) returns the top-beam_width completed beams as multiple outputs.
// This mirrors vLLM's algorithm EXACTLY (deterministic ⇒ token-exact gate).
//
// The algorithm splits into a MODEL-FREE core and a thin engine driver so the
// scoring + selection + EOS + length-penalty (the correctness content) is gated on
// a hand-specified logprob table with no model:
//   - get_beam_search_score / SortBeamsKey / BeamSearchStep — pure, model-free.
//   - BeamSearch(LLMEngine&, ...) — the driver: sources each step's per-beam
//     next-token logprob dict from the engine (one max_tokens=1 request per beam)
//     and calls BeamSearchStep, then does the final sort/select.
//
// DEVIATIONS vs upstream (recorded, use OUR names):
//   - BeamSearchSequence drops orig_prompt/lora_request/stop_reason (encoder-
//     decoder + LoRA + structured-output are out of this T0/T1 scope). `tokens`
//     still carries the FULL sequence (prompt + generated) exactly as upstream, so
//     get_beam_search_score's seq_len (which INCLUDES the prompt) matches.
//   - `eos_token_id` is std::optional<int> (upstream `int | None`); a nullopt eos
//     never matches a token (get_beam_search_score's seq_len is never decremented,
//     and no beam is ever retired for EOS — the `ignore_eos`/no-eos path).
//   - The structured-output (grammar bitmask) beam-sampling-params branch and the
//     multi-instance batch loop are NOT ported (single prompt per BeamSearch call;
//     grammar-constrained beam search is a named residual).
//   - Text decode is optional: BeamSearch fills `text` only when a tokenizer is
//     supplied (upstream always decodes since it owns the tokenizer).
#ifndef VLLM_ENTRYPOINTS_BEAM_SEARCH_H_
#define VLLM_ENTRYPOINTS_BEAM_SEARCH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/logprobs.h"  // LogprobsOnePosition / SampleLogprobs

namespace vllm {

namespace tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h
}  // namespace tok

namespace v1 {
class LLMEngine;  // vllm/v1/engine/llm_engine.h
}  // namespace v1

// BeamSearchParams (sampling_params.py:1114). Beam-search parameters for text
// generation. Field names + defaults mirror upstream 1:1 (the T0/T1 subset;
// include_stop_str_in_output / structured_outputs are not wired here).
struct BeamSearchParams {
  // Number of beams kept per step and returned (upstream: beam_width).
  int beam_width;
  // Maximum number of decode steps (upstream: max_tokens).
  int max_tokens;
  // Keep generating past EOS instead of retiring the beam (upstream: ignore_eos).
  bool ignore_eos = false;
  // Sampling temperature for the per-beam step; 0.0 (the default) => greedy, the
  // deterministic beam-search regime (upstream: temperature).
  double temperature = 0.0;
  // Length-penalty exponent for get_beam_search_score (upstream: length_penalty).
  double length_penalty = 1.0;
};

// BeamSearchSequence (utils.py:18): a beam — its FULL token sequence (prompt +
// generated), the per-position sample-logprob dicts, and the cumulative logprob.
struct BeamSearchSequence {
  // The full token ids (prompt + generated). len is load-bearing for scoring.
  std::vector<int32_t> tokens;
  // One {token_id -> Logprob} dict per generated token (utils.py:30 `logprobs`).
  SampleLogprobs logprobs;
  // Cumulative log probability of the generated tokens (utils.py:32).
  double cum_logprob = 0.0;
  // Detokenized full text; filled only when a tokenizer is supplied (utils.py:33).
  std::optional<std::string> text;
  // "stop" (EOS) / "length" — set on the returned beams (utils.py:34).
  std::optional<std::string> finish_reason;
};

// BeamSearchOutput (utils.py:102): the beam_width best beams for one prompt.
struct BeamSearchOutput {
  std::vector<BeamSearchSequence> sequences;
};

// BeamSearchInstance (utils.py:112): the per-prompt beam-search state — the active
// `beams` (seeded with one beam == the prompt) and the `completed` beams that hit
// EOS. BeamSearchStep advances it one decode step.
struct BeamSearchInstance {
  std::vector<BeamSearchSequence> beams;
  std::vector<BeamSearchSequence> completed;

  // Seed with a single beam carrying the prompt tokens (utils.py:125-133).
  explicit BeamSearchInstance(const std::vector<int32_t>& prompt_tokens) {
    BeamSearchSequence seed;
    seed.tokens = prompt_tokens;
    beams.push_back(std::move(seed));
  }
};

// get_beam_search_score (utils.py:137-153): cumulative logprob normalized by the
// length penalty. seq_len is len(tokens) (INCLUDING the prompt), minus one when
// the last token is EOS. length_penalty defaults to 1.0. Adapted from HF
// transformers generation/beam_search.py:938.
//   score = cumulative_logprob / (seq_len ** length_penalty)
double get_beam_search_score(const std::vector<int32_t>& tokens,
                             double cumulative_logprob,
                             std::optional<int> eos_token_id,
                             double length_penalty = 1.0);

// sort_beams_key (utils.py:156-162): the beam's get_beam_search_score. Beams are
// sorted by this DESCENDING (reverse=True) with a STABLE sort so ties keep input
// order, exactly as Python's `sorted(..., reverse=True)` (Timsort is stable).
double SortBeamsKey(const BeamSearchSequence& beam,
                    std::optional<int> eos_token_id, double length_penalty);

// BeamSearchStep (offline.py:193-327, the non-structured-output path): run one
// token step. For each active beam, `per_beam_logprobs[i]` is that beam's
// next-token {token_id -> Logprob} dict (the model's top-2*beam_width for this
// step), or nullptr if the beam produced no logprobs (sequence completed / aborted
// — offline.py:297-303). Each (token_id, logprob) forms a candidate continuation
// with cum_logprob += logprob; EOS candidates (token_id == eos && !ignore_eos) go
// to `instance.completed`, the rest are sorted by the length-penalty score and the
// top-beam_width become the new `instance.beams` (offline.py:291-325).
//
// Returns true if the step exhausted all beams (offline.py:219-220 all_beams == 0
// ⇒ stop). `per_beam_logprobs.size()` must equal `instance.beams.size()`.
bool BeamSearchStep(BeamSearchInstance& instance,
                    const std::vector<const LogprobsOnePosition*>& per_beam_logprobs,
                    std::optional<int> eos_token_id, bool ignore_eos,
                    int beam_width, double length_penalty);

// BeamSearch (offline.py:58-191 BeamSearchOfflineMixin.beam_search, single-prompt):
// drive beam search over `engine` for one pre-tokenized prompt. Each step issues
// one max_tokens=1 request per active beam (logprobs=2*beam_width, the beam
// temperature) to read the next-token logprob dict, calls BeamSearchStep, and
// stops at max_tokens or once all beams complete. Finally appends the remaining
// active beams to `completed`, sorts by the length-penalty score DESCENDING, and
// returns the top-beam_width beams (finish_reason "stop" for EOS-retired beams,
// else "length"). `tokenizer` (optional) fills each returned beam's `text`.
BeamSearchOutput BeamSearch(v1::LLMEngine& engine,
                            const std::vector<int32_t>& prompt_tokens,
                            const BeamSearchParams& params,
                            std::optional<int> eos_token_id,
                            const tok::Tokenizer* tokenizer = nullptr);

}  // namespace vllm

#endif  // VLLM_ENTRYPOINTS_BEAM_SEARCH_H_
