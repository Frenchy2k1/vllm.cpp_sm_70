// Ported from: vllm/reasoning/muse_glimmer_reasoning_parser.py @ 075d645af
// (vLLM PR #51655 head). See muse_glimmer.h for the deviation list.
#include "vllm/entrypoints/openai/reasoning_parsers/muse_glimmer.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace vllm::entrypoints::openai {

namespace {

// muse_glimmer_reasoning_parser.py:25-29.
const char* const kEom = "<|eom|>";
const char* const kEot = "<|eot|>";
const char* const kFunctionCallsOpen = "<atem:function_calls>";
const char* const kInvokeOpen = "<atem:invoke";
const char* const kReasoningOpen = "to=self<|message|>";
const char* const kAssistantTurnOpen = "<|start|>assistant";

// muse_glimmer_reasoning_parser.py:33 (_HEADER_PAT). Python's re.DOTALL `.` is
// spelled `[\s\S]` in the ECMAScript grammar std::regex uses; every other atom
// is identical.
const char* const kHeaderPat = R"re(to=[^\s<]+<\|message\|>)re";

// muse_glimmer_reasoning_parser.py:32 (_CHANNEL_HEADER_RE) — group 1 is the
// recipient: `self` (reasoning), `user` (final answer) or `<tool>[.<fn>]`.
const std::regex& ChannelHeaderRe() {
  static const std::regex re(R"re(to=([^\s<]+)<\|message\|>)re");
  return re;
}

// :35 (_COLLAPSE_RE) — collapse the gap between reasoning blocks so multiple
// to=self spans join.
const std::regex& CollapseRe() {
  static const std::regex re(
      R"re(<\|eom\|>(?:(?!to=self<\|message\|>)[\s\S])*?to=self<\|message\|>)re");
  return re;
}

// :38 (_REASONING_RE).
const std::regex& ReasoningRe() {
  static const std::regex re(R"re(to=self<\|message\|>([\s\S]*?)<\|eom\|>)re");
  return re;
}

// :39 (_CONTENT_RE).
const std::regex& ContentRe() {
  static const std::regex re(
      R"re(to=user<\|message\|>([\s\S]*?)(?=<\|eot\|>|<\|eom\|>|$))re");
  return re;
}

// :43 (_STRIP_REASONING_RE) — a CLOSED reasoning span.
const std::regex& StripReasoningRe() {
  static const std::regex re(
      R"re((?:<\|start\|>assistant\s*)?to=self<\|message\|>[\s\S]*?<\|eom\|>)re");
  return re;
}

// :57 (_STRIP_OPEN_REASONING_RE) — an UNTERMINATED trailing reasoning span. It
// MUST stop at the next channel header rather than running to end-of-text: the
// model sometimes leaves the analysis channel without emitting <|eom|>, and an
// unbounded version would swallow the real tool call with it (is_reasoning_end
// would then never fire and the whole generation would be dropped).
const std::regex& StripOpenReasoningRe() {
  static const std::regex re(
      std::string(
          R"re((?:<\|start\|>assistant\s*)?to=self<\|message\|>(?:(?!<\|eom\|>)(?!)re") +
      kHeaderPat + R"re()[\s\S])*(?=)re" + kHeaderPat + R"re(|$))re");
  return re;
}

// :63 (_OPEN_REASONING_RE) — same bound, but capturing the partial body.
const std::regex& OpenReasoningRe() {
  static const std::regex re(
      std::string(R"re(to=self<\|message\|>((?:(?!<\|eom\|>)(?!)re") + kHeaderPat +
      R"re()[\s\S])*)(?=)re" + kHeaderPat + R"re(|$))re");
  return re;
}

// :73 (_OPEN_TAIL_HEADER_RE) — a trailing fragment that could still grow into a
// channel header (" t", " to", " to=", " to=skill").
const std::regex& OpenTailHeaderRe() {
  static const std::regex re(R"re([\s](?:t|to|to=[^\s<]*)$)re");
  return re;
}

// :69 (_HOLDBACK_MARKERS) — markers whose PREFIX could appear at the tail of an
// OPEN (still-streaming) body.
const std::vector<std::string>& HoldbackMarkers() {
  static const std::vector<std::string> m = {kEom, kEot, "<|start|>",
                                             "<|message|>"};
  return m;
}

bool Contains(const std::string& s, const char* sub) {
  return s.find(sub) != std::string::npos;
}

bool HasAtem(const std::string& s) {
  return Contains(s, kFunctionCallsOpen) || Contains(s, kInvokeOpen);
}

// :74 (_current_assistant_turn). is_reasoning_end is evaluated on the PROMPT
// text too, and a Muse Glimmer prompt legitimately contains ATEM markers
// (render_tool_defs writes a literal <atem:function_calls> example into the
// system message, and prior assistant turns may carry real tool calls).
// Anchoring on the last channel-open keeps prompt text from deciding the phase.
std::string CurrentAssistantTurn(const std::string& text) {
  const std::size_t idx = text.rfind(kAssistantTurnOpen);
  if (idx == std::string::npos) return text;
  return text.substr(idx + std::string(kAssistantTurnOpen).size());
}

// :84 (_trim_open_body). Iterated to a fixpoint because the two cases compose:
// " to=skill<" needs the partial-marker trim (`<`) before the partial-header
// trim can see " to=skill". Trimming only once leaks the recipient name.
std::string TrimOpenBody(std::string body) {
  while (true) {
    std::string trimmed = body;
    for (const std::string& marker : HoldbackMarkers()) {
      bool cut = false;
      const std::size_t kmax = std::min(marker.size() - 1, trimmed.size());
      for (std::size_t k = kmax; k >= 1; --k) {
        if (trimmed.compare(trimmed.size() - k, k, marker, 0, k) == 0) {
          trimmed.erase(trimmed.size() - k);
          cut = true;
          break;
        }
      }
      if (cut) break;  // Python for/else: only the FIRST matching marker cuts.
    }
    std::smatch m;
    if (std::regex_search(trimmed, m, OpenTailHeaderRe())) {
      trimmed.erase(static_cast<std::size_t>(m.position(0)));
    }
    if (trimmed == body) return body;
    body = std::move(trimmed);
  }
}

// Python's `pattern.search(text, pos)`: leftmost match at or after `pos`, with
// absolute offsets. (std::regex has no pos overload; iterators supply it.)
bool SearchFrom(const std::string& s, std::size_t pos, const std::regex& re,
                std::smatch& m, std::size_t* start, std::size_t* end) {
  if (pos > s.size()) return false;
  const auto begin = s.cbegin() + static_cast<std::ptrdiff_t>(pos);
  if (!std::regex_search(begin, s.cend(), m, re)) return false;
  *start = pos + static_cast<std::size_t>(m.position(0));
  *end = *start + static_cast<std::size_t>(m.length(0));
  return true;
}

// :173 (_classify_bodies). Split `text` into (reasoning_body, content_body),
// channel-aware. Framing markers and tool channels contribute nothing — the
// tool parser owns those. A body ends at <|eom|> / <|eot|>, at the next channel
// header, or at end-of-text (an OPEN body, which is held back).
void ClassifyBodies(const std::string& text, std::string* reasoning,
                    std::string* content) {
  reasoning->clear();
  content->clear();
  const std::string eom = kEom;
  const std::string eot = kEot;
  const std::size_t n = text.size();
  std::size_t pos = 0;
  while (pos < n) {
    std::smatch m;
    std::size_t hstart = 0, hend = 0;
    if (!SearchFrom(text, pos, ChannelHeaderRe(), m, &hstart, &hend)) break;
    const std::string recipient = m[1].str();
    const std::size_t body_start = hend;

    const std::size_t eom_pos = text.find(eom, body_start);
    const std::size_t eot_pos = text.find(eot, body_start);
    std::vector<std::size_t> terminators;
    if (eom_pos != std::string::npos) terminators.push_back(eom_pos);
    if (eot_pos != std::string::npos) terminators.push_back(eot_pos);
    std::smatch nm;
    std::size_t nstart = 0, nend = 0;
    if (SearchFrom(text, body_start, ChannelHeaderRe(), nm, &nstart, &nend)) {
      terminators.push_back(nstart);
    }
    const bool any = !terminators.empty();
    const std::size_t body_end =
        any ? *std::min_element(terminators.begin(), terminators.end()) : n;
    std::string body = text.substr(body_start, body_end - body_start);
    if (!any) body = TrimOpenBody(std::move(body));

    if (recipient == "self") {
      *reasoning += body;
    } else if (recipient == "user") {
      // Never surface tool XML echoed into a user channel.
      if (!HasAtem(body)) *content += body;
    }

    if (any && (body_end == eom_pos || body_end == eot_pos)) {
      pos = body_end + (body_end == eom_pos ? eom.size() : eot.size());
    } else {
      pos = body_end;
    }
  }
}

// :244-245 — the remainder after BOTH reasoning strips. This is exactly the span
// extract_reasoning() forwards as content when the turn carries ATEM, and (per
// deviation 4) exactly the span the streaming handoff feeds the tool parser.
std::string StripReasoningSpans(const std::string& text) {
  const std::string closed =
      std::regex_replace(text, StripReasoningRe(), std::string());
  return std::regex_replace(closed, StripOpenReasoningRe(), std::string());
}

}  // namespace

// :152 (_scoped_turn) + :158 (_tool_channel_remainder). The remainder must start
// AT the `to=<name><|message|>` header: handing over the text after the header
// loses the recipient, and the tool parser then sees a bare `<|message|>`,
// classifies it as the content channel, and leaks the ATEM markup.
std::string MuseGlimmerReasoningParser::tool_channel_remainder(
    const std::string& text) {
  const std::string scoped = StripReasoningSpans(CurrentAssistantTurn(text));
  for (auto it = std::sregex_iterator(scoped.begin(), scoped.end(),
                                      ChannelHeaderRe());
       it != std::sregex_iterator(); ++it) {
    const std::string recipient = (*it)[1].str();
    if (recipient != "self" && recipient != "user") {
      return scoped.substr(static_cast<std::size_t>(it->position(0)));
    }
  }
  return std::string();
}

// :129 (is_reasoning_end), text form. Both closed and unterminated reasoning
// spans are stripped before the check, so an <atem:invoke> the model merely
// echoes inside its CoT never flips the phase; and a `to=user` answer is NOT a
// reason to leave reasoning, since this parser surfaces that content itself.
bool MuseGlimmerReasoningParser::is_reasoning_end(const std::string& text) const {
  return HasAtem(tool_channel_remainder(text));
}

// :224 (extract_reasoning).
ExtractedReasoning MuseGlimmerReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string collapsed =
      std::regex_replace(model_output, CollapseRe(), std::string("\n"));

  std::vector<std::string> matches;
  for (auto it = std::sregex_iterator(collapsed.begin(), collapsed.end(),
                                      ReasoningRe());
       it != std::sregex_iterator(); ++it) {
    matches.push_back((*it)[1].str());
  }
  std::optional<std::string> reasoning;
  if (!matches.empty()) {
    std::string joined = matches[0];
    for (std::size_t i = 1; i < matches.size(); ++i) joined += "\n" + matches[i];
    reasoning = joined;
  }

  // Truncation fallback: generation stopped inside a to=self block, so there is
  // no closing <|eom|>. Bounded at the next channel header so a real tool call
  // that follows a header-less channel switch is not absorbed into reasoning.
  std::smatch open_match;
  if (std::regex_search(model_output, open_match, OpenReasoningRe()) &&
      open_match[1].length() > 0) {
    const std::string partial = open_match[1].str();
    reasoning = reasoning.has_value() ? (*reasoning + "\n" + partial) : partial;
  }

  // Content is everything that is not a reasoning block. In a reasoning +
  // tool-call turn there is no to=user answer, but the tool channels MUST be
  // forwarded — the serving layer runs the tool parser on this returned
  // `content`, not on the original model_output.
  const std::string remainder = StripReasoningSpans(model_output);
  if (HasAtem(remainder)) {
    ExtractedReasoning out;
    out.reasoning = reasoning;
    if (!remainder.empty()) out.content = remainder;
    return out;
  }

  ExtractedReasoning out;
  out.reasoning = reasoning;
  std::smatch content_match;
  if (std::regex_search(model_output, content_match, ContentRe())) {
    if (content_match[1].length() > 0) out.content = content_match[1].str();
  } else if (Contains(model_output, kReasoningOpen)) {
    out.content = std::nullopt;
  } else {
    if (!model_output.empty()) out.content = model_output;
    out.reasoning = std::nullopt;
  }
  return out;
}

// :257 (extract_reasoning_streaming). Classifies the full `current_text` and
// emits only what has not been emitted yet, so no framing token is ever
// surfaced and a delta straddling a channel boundary only contributes the
// portion inside a real body.
std::optional<DeltaMessage>
MuseGlimmerReasoningParser::extract_reasoning_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/,
    const ChatCompletionRequest& /*request*/) {
  std::string curr_reason;
  std::string curr_content;
  ClassifyBodies(current_text, &curr_reason, &curr_content);

  std::string reasoning_delta;
  if (curr_reason.size() > emitted_reasoning_.size() &&
      curr_reason.compare(0, emitted_reasoning_.size(), emitted_reasoning_) ==
          0) {
    reasoning_delta = curr_reason.substr(emitted_reasoning_.size());
    emitted_reasoning_ = curr_reason;
  }
  std::string content_delta;
  if (curr_content.size() > emitted_content_.size() &&
      curr_content.compare(0, emitted_content_.size(), emitted_content_) == 0) {
    content_delta = curr_content.substr(emitted_content_.size());
    emitted_content_ = curr_content;
  }

  // Hand the tool channel to the tool parser, starting at its header. This must
  // not fire before the channel actually contains ATEM: emitting on the bare
  // `to=<name><|message|>` header would deliver the header to the client as
  // visible content. Deviation 4 (see muse_glimmer.h): the forward is
  // INCREMENTAL and uses the same span extract_reasoning() returns, because
  // ShapeChatDelta re-derives the tool parser's previous_text from that call.
  std::string handoff;
  const std::string forward = StripReasoningSpans(current_text);
  if (HasAtem(forward) && forward.size() > emitted_handoff_.size() &&
      forward.compare(0, emitted_handoff_.size(), emitted_handoff_) == 0) {
    handoff = forward.substr(emitted_handoff_.size());
    emitted_handoff_ = forward;
  }

  if (!handoff.empty()) {
    DeltaMessage m;
    if (!reasoning_delta.empty()) m.reasoning = reasoning_delta;
    m.content = handoff;
    return m;
  }
  if (!reasoning_delta.empty() && !content_delta.empty()) {
    DeltaMessage m;
    m.reasoning = reasoning_delta;
    m.content = content_delta;
    return m;
  }
  if (!reasoning_delta.empty()) {
    DeltaMessage m;
    m.reasoning = reasoning_delta;
    return m;
  }
  if (!content_delta.empty()) {
    DeltaMessage m;
    m.content = content_delta;
    return m;
  }
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
