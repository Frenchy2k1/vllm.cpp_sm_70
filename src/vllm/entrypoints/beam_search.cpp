// Ported from: vllm/entrypoints/generate/beam_search/{utils,offline}.py @ 555967922.
// See include/vllm/entrypoints/beam_search.h for scope, the split into a model-free
// core (scoring/selection) + the engine driver, and the recorded deviations.
#include "vllm/entrypoints/beam_search.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/llm_engine.h"

namespace vllm {

double get_beam_search_score(const std::vector<int32_t>& tokens,
                             double cumulative_logprob,
                             std::optional<int> eos_token_id,
                             double length_penalty) {
  // utils.py:149-153. seq_len INCLUDES the prompt (tokens starts at the prompt
  // ids); drop one when the last token is EOS.
  double seq_len = static_cast<double>(tokens.size());
  if (!tokens.empty() && eos_token_id.has_value() &&
      tokens.back() == *eos_token_id) {
    seq_len -= 1.0;
  }
  return cumulative_logprob / std::pow(seq_len, length_penalty);
}

double SortBeamsKey(const BeamSearchSequence& beam,
                    std::optional<int> eos_token_id, double length_penalty) {
  // utils.py:156-162 (create_sort_beams_key_function's closure).
  return get_beam_search_score(beam.tokens, beam.cum_logprob, eos_token_id,
                               length_penalty);
}

namespace {

// std::stable_sort with a strict-DESCENDING comparator reproduces Python's
// `sorted(key=..., reverse=True)` EXACTLY: Timsort is stable and `reverse=True`
// preserves the input order of equal-key elements (it does NOT flip ties). Using
// `a > b` (never `>=`) keeps the comparator a valid strict weak ordering so equal
// keys are "equivalent" and stable_sort leaves them in input order.
void SortBeamsDescending(std::vector<BeamSearchSequence>& beams,
                         std::optional<int> eos_token_id,
                         double length_penalty) {
  std::stable_sort(beams.begin(), beams.end(),
                   [&](const BeamSearchSequence& a, const BeamSearchSequence& b) {
                     return SortBeamsKey(a, eos_token_id, length_penalty) >
                            SortBeamsKey(b, eos_token_id, length_penalty);
                   });
}

}  // namespace

bool BeamSearchStep(
    BeamSearchInstance& instance,
    const std::vector<const LogprobsOnePosition*>& per_beam_logprobs,
    std::optional<int> eos_token_id, bool ignore_eos, int beam_width,
    double length_penalty) {
  // offline.py:219-220: no active beams => the search is exhausted.
  if (instance.beams.empty()) return true;

  // offline.py:291-319: expand every active beam by each of its next-token
  // logprob entries; EOS candidates retire to `completed`, the rest compete.
  std::vector<BeamSearchSequence> new_beams;
  for (std::size_t i = 0; i < instance.beams.size(); ++i) {
    const BeamSearchSequence& current = instance.beams[i];
    const LogprobsOnePosition* logprobs = per_beam_logprobs[i];
    // offline.py:297-303: a null dict means the sequence completed (max-model-len
    // / abortion) and produces no continuations.
    if (logprobs == nullptr) continue;

    // Iterate the dict in insertion order (offline.py:305 `logprobs.items()`;
    // our LogprobsOnePosition preserves it via `order`).
    for (int32_t token_id : logprobs->order) {
      const Logprob* lp = logprobs->find(token_id);
      BeamSearchSequence candidate;
      candidate.tokens = current.tokens;
      candidate.tokens.push_back(token_id);
      candidate.logprobs = current.logprobs;
      candidate.logprobs.push_back(*logprobs);
      candidate.cum_logprob = current.cum_logprob + lp->logprob;

      if (eos_token_id.has_value() && token_id == *eos_token_id && !ignore_eos) {
        // offline.py:316-317: EOS beam finishes.
        instance.completed.push_back(std::move(candidate));
      } else {
        // offline.py:318-319.
        new_beams.push_back(std::move(candidate));
      }
    }
  }

  // offline.py:320-325: keep the top-beam_width continuations by score.
  SortBeamsDescending(new_beams, eos_token_id, length_penalty);
  if (static_cast<int>(new_beams.size()) > beam_width) {
    new_beams.resize(static_cast<std::size_t>(beam_width));
  }
  instance.beams = std::move(new_beams);

  // offline.py:327: not exhausted (a step that produced beams; the next call
  // returns true if they all completed and beams is now empty).
  return false;
}

BeamSearchOutput BeamSearch(v1::LLMEngine& engine,
                            const std::vector<int32_t>& prompt_tokens,
                            const BeamSearchParams& params,
                            std::optional<int> eos_token_id,
                            const tok::Tokenizer* tokenizer) {
  // offline.py:80-88.
  const int beam_width = params.beam_width;
  const int max_tokens = params.max_tokens;

  // offline.py:118-123: per-beam step params — logprobs=2*beam_width, one token,
  // the beam temperature (0 => greedy, the deterministic regime).
  auto MakeStepParams = [&]() {
    SamplingParams sp;
    sp.logprobs = 2 * beam_width;
    sp.max_tokens = 1;
    sp.temperature = params.temperature;
    // Non-cumulative aggregation is irrelevant for a 1-token request; keep the
    // default cumulative output_kind so generate() returns the finished output.
    sp.PostInit();
    return sp;
  };

  // offline.py:124-138: seed one instance with a single beam == the prompt.
  BeamSearchInstance instance(prompt_tokens);

  // offline.py:160-173: run at most max_tokens steps.
  std::int64_t req_counter = 0;
  for (int step = 0; step < max_tokens; ++step) {
    if (instance.beams.empty()) break;

    // offline.py:266-303: one decode step per active beam; read its single
    // generated position's logprob dict. We OWN the SampleLogprobs storage so the
    // dict pointers stay valid across BeamSearchStep.
    std::vector<std::optional<LogprobsOnePosition>> step_dicts;
    step_dicts.reserve(instance.beams.size());
    for (const BeamSearchSequence& beam : instance.beams) {
      const std::string req_id =
          "beam-" + std::to_string(step) + "-" + std::to_string(req_counter++);
      RequestOutput out =
          engine.generate(beam.tokens, MakeStepParams(), req_id);
      if (!out.outputs.empty() && out.outputs[0].logprobs.has_value() &&
          !out.outputs[0].logprobs->empty()) {
        // The single generated position's dict (offline.py:303 logprobs[0]).
        step_dicts.emplace_back((*out.outputs[0].logprobs)[0]);
      } else {
        step_dicts.emplace_back(std::nullopt);
      }
    }
    std::vector<const LogprobsOnePosition*> per_beam_logprobs;
    per_beam_logprobs.reserve(step_dicts.size());
    for (const auto& d : step_dicts) {
      per_beam_logprobs.push_back(d.has_value() ? &(*d) : nullptr);
    }

    const bool stop = BeamSearchStep(instance, per_beam_logprobs, eos_token_id,
                                     params.ignore_eos, beam_width,
                                     params.length_penalty);
    if (stop) break;
  }

  // offline.py:178-189: fold the surviving active beams into completed, sort by
  // score DESCENDING, take the top-beam_width, decode text.
  for (BeamSearchSequence& beam : instance.beams) {
    instance.completed.push_back(std::move(beam));
  }
  instance.beams.clear();
  SortBeamsDescending(instance.completed, eos_token_id, params.length_penalty);

  BeamSearchOutput output;
  const int keep =
      std::min<int>(beam_width, static_cast<int>(instance.completed.size()));
  output.sequences.reserve(static_cast<std::size_t>(keep));
  for (int i = 0; i < keep; ++i) {
    BeamSearchSequence beam = std::move(instance.completed[static_cast<std::size_t>(i)]);
    // finish_reason: "stop" when the beam ends on EOS, else "length" (mirrors the
    // completion-output finish reasons; upstream leaves it None on the offline
    // path, we set it so the multi-output aggregation carries a reason).
    if (eos_token_id.has_value() && !beam.tokens.empty() &&
        beam.tokens.back() == *eos_token_id) {
      beam.finish_reason = "stop";
    } else {
      beam.finish_reason = "length";
    }
    if (tokenizer != nullptr) {
      beam.text = tokenizer->Decode(beam.tokens);
    }
    output.sequences.push_back(std::move(beam));
  }
  return output;
}

}  // namespace vllm
