// Ported from: vllm/reasoning/muse_glimmer_reasoning_parser.py @ 075d645af
// (vLLM PR #51655 head — deliberately NOT the parity pin; Muse Glimmer does not
// exist at pin 555967922. See .agents/specs/muse-glimmer.md §0 and
// .agents/porting-inventory.md §9 deviation 16.)
//
// Muse Glimmer emits chain-of-thought in ATEM channel-scoped messages:
//
//   <|start|>assistant to=self<|message|>...reasoning...<|eom|>
//   <|start|>assistant to=<tool>.<fn><|message|><atem:function_calls>...<|eom|>
//   <|start|>assistant to=user<|message|>...final answer...<|eot|>
//
// A turn may hold several `to=self` blocks interleaved with tool calls. Because
// the framing markers are not guaranteed to be single vocab tokens on every
// checkpoint's tokenizer, upstream works on the DECODED TEXT with regexes rather
// than the single start/end-token base class — which is exactly the shape our
// text-only ReasoningParser seam already has.
//
// DEVIATIONS from the upstream file. Each is forced by a seam difference; the
// FIRST one IS observable by a client and is an open gap, the rest are not:
//
//  1. `adjust_request` (upstream:117, forces skip_special_tokens=False) is
//     DROPPED, and the claim this comment used to make — "our detokenizer does not
//     strip the ATEM markers in the first place" — is FALSE.
//     `skip_special_tokens` is declared `= true` at protocol.h:240 (completion)
//     and :461 (chat) and honoured at v1/engine/detokenizer.cpp:68, and the
//     released checkpoint marks the framing markers special: `<|eom|>` 200007,
//     `<|eot|>` 200008, `<|start|>` 200022, `<|message|>` 200023, each
//     `"special": true` in `tokenizer.json` and listed under
//     `extra_special_tokens` in `tokenizer_config.json`. At server defaults the
//     framing this parser keys on is therefore GONE before it runs, which is
//     exactly the reason upstream forces the flag off.
//
//     The seam has no `adjust_request` DISPATCH SITE at all — `KimiK2ToolParser
//     ::adjust_request` (kimi_k2.cpp:87) has no callers either — so this is a
//     pre-existing seam gap rather than something this port chose to drop. It is
//     recorded as an OPEN GAP, not a no-op: channel scoping does not work at
//     server defaults. The unit gates below pass because they feed the parser
//     framed strings directly. Tracked in .agents/specs/muse-glimmer.md §6.7 and
//     docs/FEATURES.md; NOT fixed here.
//  2. `is_reasoning_end(input_ids)` / `is_reasoning_end_streaming` /
//     `extract_content_ids` are token-ID methods upstream; this seam's
//     `is_reasoning_end` is TEXT-based (reasoning_parsers/abstract.h documents
//     the deviation for the whole family). Upstream itself decodes the ids and
//     then runs the identical text rule, so the behaviour is the same rule.
//  3. `get_streaming_fallback_content` (upstream:211) is called by
//     `DelegatingParser.finalize_generation`, which this seam has no analogue of.
//     Dropped.
//  4. THE HANDOFF IS INCREMENTAL, NOT ONE-SHOT. Upstream fires the
//     reasoning→tool handoff exactly once (upstream:293-300) because
//     `DelegatingParser.parse_delta` then flips phase and never consults the
//     reasoning parser again — the tool parser owns every later delta directly.
//     Our seam (entrypoints/openai/serving_chat.cpp `ShapeChatDelta`) calls the
//     reasoning parser on EVERY delta and derives the tool parser's
//     previous_text/current_text from `extract_reasoning(previous_text).content`.
//     So this port keeps forwarding the tool channel, and forwards exactly the
//     span `extract_reasoning()` returns as content (upstream:244-247), which is
//     what makes the two consistent. The stream the tool parser sees is
//     byte-identical to upstream's; only the number of DeltaMessages carrying it
//     differs. Upstream's `_tool_handoff_done` bool is therefore replaced by an
//     emitted-prefix cursor.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class MuseGlimmerReasoningParser final : public ReasoningParser {
 public:
  MuseGlimmerReasoningParser() = default;

  // muse_glimmer_reasoning_parser.py:224 (extract_reasoning).
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // muse_glimmer_reasoning_parser.py:257 (extract_reasoning_streaming).
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // muse_glimmer_reasoning_parser.py:129 (is_reasoning_end) — TEXT form
  // (deviation 2). True once a real TOOL channel carrying ATEM has opened; a
  // `to=user` answer is NOT a reason to leave the reasoning phase, and an
  // `<atem:invoke>` merely echoed inside the CoT never flips it.
  bool is_reasoning_end(const std::string& text) const override;

  // muse_glimmer_reasoning_parser.py:158 (_tool_channel_remainder). Text from
  // the first TOOL-channel header onward (framing included), with reasoning
  // spans — closed and unterminated — removed first. Exposed because the ported
  // tests assert on the channel scoping directly.
  static std::string tool_channel_remainder(const std::string& text);

 private:
  // Cursors over what was ACTUALLY emitted (upstream:114-116). Diffing a freshly
  // reclassified `previous_text` is unsafe: a classified body legitimately
  // SHRINKS when a partial header becomes recognisable, and diffing against the
  // shrunken value re-emits text that already went out.
  std::string emitted_reasoning_;
  std::string emitted_content_;
  // Deviation 4: the prefix of the tool channel already handed to the tool
  // parser (upstream's one-shot `_tool_handoff_done` flag).
  std::string emitted_handoff_;
};

}  // namespace vllm::entrypoints::openai
