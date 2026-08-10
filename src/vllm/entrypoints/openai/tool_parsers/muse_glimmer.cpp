// Ported from: vllm/tool_parsers/muse_glimmer_tool_parser.py @ 075d645af
// (vLLM PR #51655 head). See muse_glimmer.h for the deviation list.
#include "vllm/entrypoints/openai/tool_parsers/muse_glimmer.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using nlohmann::ordered_json;

// muse_glimmer_tool_parser.py:76 (_MSG_HEADER_RE). An assistant message header;
// all three parts are optional except the <|message|> terminator:
//   "<|start|>assistant to=get_weather<|message|>"  — after an <|eom|> boundary
//   " to=self<|message|>"                           — first message of a turn
//                                                     (the prompt already ended
//                                                     with "<|start|>assistant")
//   "<|message|>"                                   — bare recipient
const std::regex& MsgHeaderRe() {
  static const std::regex re(
      R"re((?:<\|start\|>\s*assistant)?[^\S\n]*(?:to=([A-Za-z0-9_.\-]+))?<\|message\|>)re");
  return re;
}

// :79 (_MSG_END_RE).
const std::regex& MsgEndRe() {
  static const std::regex re(R"re(<\|eom\|>|<\|eot\|>)re");
  return re;
}

// :93 (_OPEN_TAIL_TO_RE) — a trailing " to=NAME" that could still grow into a
// bare message header.
const std::regex& OpenTailToRe() {
  static const std::regex re(R"re([^\S\n]+to=[A-Za-z0-9_.\-]*$)re");
  return re;
}

// :96-101 — the ATEM extraction regexes, unchanged from MUSE_GLIMMER_RESPONSE_SCHEMA.
// (Python re.DOTALL `.` -> `[\s\S]`; the custom raw-string delimiter is required
// because these patterns contain the byte sequence )" .)
const std::regex& InvokeRe() {
  static const std::regex re(R"re(<atem:invoke\b[\s\S]*?</atem:invoke>)re");
  return re;
}
const std::regex& NameRe() {
  static const std::regex re(R"re(<atem:invoke\b[^>]*?\bname="([^"]+)")re");
  return re;
}
const std::regex& ParamRe() {
  static const std::regex re(
      R"re(<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>([\s\S]*?)</atem:parameter>)re");
  return re;
}

const char* const kFunctionCallsOpen = "<atem:function_calls>";
const char* const kInvokeOpen = "<atem:invoke";
// :82-83 — recipients whose bodies are NOT tool calls.
const char* const kReasoningRecipient = "self";
const char* const kUserRecipient = "user";

// :88 (_STRUCTURAL_MARKERS) — must never reach the client.
const std::vector<std::string>& StructuralMarkers() {
  static const std::vector<std::string> m = {"<|eom|>", "<|eot|>", "<|start|>",
                                             "<|message|>"};
  return m;
}
std::size_t MaxMarkerLen() {
  static const std::size_t n = [] {
    std::size_t v = 0;
    for (const std::string& m : StructuralMarkers()) v = std::max(v, m.size());
    return v;
  }();
  return n;
}

bool Contains(const std::string& s, const char* sub) {
  return s.find(sub) != std::string::npos;
}
bool HasAtem(const std::string& s) {
  return Contains(s, kFunctionCallsOpen) || Contains(s, kInvokeOpen);
}

// Python's `pattern.search(text, pos)`: leftmost match at or after `pos`.
bool SearchFrom(const std::string& s, std::size_t pos, const std::regex& re,
                std::smatch& m, std::size_t* start, std::size_t* end) {
  if (pos > s.size()) return false;
  const auto begin = s.cbegin() + static_cast<std::ptrdiff_t>(pos);
  if (!std::regex_search(begin, s.cend(), m, re)) return false;
  *start = pos + static_cast<std::size_t>(m.position(0));
  *end = *start + static_cast<std::size_t>(m.length(0));
  return true;
}

// :116 (_iter_messages) yield shape: (recipient, body, closed).
struct Message {
  bool has_recipient = false;
  std::string recipient;
  std::string body;
  bool closed = false;
};

// :116 (_iter_messages). Segment `text` into assistant messages. A message is
// ALSO terminated by the start of the NEXT header: without that, a reasoning
// block whose <|eom|> is missing (truncation, or a chunk dropped at the
// reasoning -> tool transition) would absorb the tool-call message that follows
// it and the call would be lost — the same defect the subtractive regexes have.
std::vector<Message> IterMessages(const std::string& text) {
  std::vector<Message> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::smatch hm;
    std::size_t hstart = 0, hend = 0;
    if (!SearchFrom(text, pos, MsgHeaderRe(), hm, &hstart, &hend)) return out;
    const bool has_rcpt = hm[1].matched;
    const std::string rcpt = has_rcpt ? hm[1].str() : std::string();
    const std::size_t body_start = hend;

    std::smatch em;
    std::size_t estart = 0, eend = 0;
    const bool has_end = SearchFrom(text, body_start, MsgEndRe(), em, &estart, &eend);
    std::smatch nm;
    std::size_t nstart = 0, nend = 0;
    const bool has_next =
        SearchFrom(text, body_start, MsgHeaderRe(), nm, &nstart, &nend);

    std::size_t body_end = has_end ? estart : text.size();
    bool closed = has_end;
    std::size_t next_pos;
    if (has_next && nstart < body_end) {
      body_end = nstart;
      closed = false;
      next_pos = nstart;
    } else {
      next_pos = has_end ? eend : text.size();
    }
    std::string body = text.substr(body_start, body_end - body_start);
    // A body can never legitimately contain <|start|>. Seeing one means the next
    // header is only partially generated (its <|message|> has not arrived), so
    // the regex could not recognise it yet. Cut there, otherwise the streamed
    // body would grow to include the next header and then shrink back.
    const std::size_t start_tok = body.find("<|start|>");
    if (start_tok != std::string::npos) {
      body = body.substr(0, start_tok);
      closed = false;
    }
    Message msg;
    msg.has_recipient = has_rcpt;
    msg.recipient = rcpt;
    msg.body = std::move(body);
    msg.closed = closed;
    out.push_back(std::move(msg));
    pos = next_pos;
  }
  return out;
}

// :159 (_trailing_partial_marker_len).
std::size_t TrailingPartialMarkerLen(const std::string& text) {
  const std::size_t max_overlap = std::min(text.size(), MaxMarkerLen() - 1);
  for (std::size_t overlap = max_overlap; overlap >= 1; --overlap) {
    const std::string suffix = text.substr(text.size() - overlap);
    for (const std::string& marker : StructuralMarkers()) {
      if (marker.size() >= suffix.size() &&
          marker.compare(0, suffix.size(), suffix) == 0) {
        return overlap;
      }
    }
  }
  return 0;
}

// :169 (_safe_open_body). Hold back anything that could still turn out to be
// structural, so the emitted prefix only ever grows. Chunks under speculative
// decoding are large enough that markers routinely straddle them.
std::string SafeOpenBody(const std::string& body) {
  std::smatch m;
  if (std::regex_search(body, m, OpenTailToRe())) {
    return body.substr(0, static_cast<std::size_t>(m.position(0)));
  }
  const std::size_t partial = TrailingPartialMarkerLen(body);
  return partial ? body.substr(0, body.size() - partial) : body;
}

// :105 (_decode_value). JSON-decode when possible, else keep the raw string
// (the schema's `x-parser: json` with `allow_non_json: True`).
//
// NOTE: Python's json.loads also accepts the bare literals NaN / Infinity /
// -Infinity, which nlohmann rejects; those decode to the raw STRING here. No
// upstream case exercises them and a non-finite tool argument is not
// serializable back out as JSON anyway.
ordered_json DecodeValue(const std::string& raw) {
  try {
    return ordered_json::parse(raw);
  } catch (...) {
    return ordered_json(raw);
  }
}

// :269 (_visible_channels). The *_open flags say whether that channel's LAST
// message is still being generated; only then must the caller hold back a
// partial structural marker. Tracking them PER CHANNEL matters: a closed
// reasoning block whose text happens to end in `<` would otherwise stay
// permanently truncated while a later content message is open.
void VisibleChannels(const std::string& text, std::string* content,
                     std::string* reasoning, bool* content_open,
                     bool* reasoning_open) {
  content->clear();
  reasoning->clear();
  *content_open = false;
  *reasoning_open = false;
  for (const Message& m : IterMessages(text)) {
    if (m.has_recipient && m.recipient == kReasoningRecipient) {
      *reasoning += m.body;
      *reasoning_open = !m.closed;
    } else if (!m.has_recipient || m.recipient == kUserRecipient) {
      *content += m.body;
      *content_open = !m.closed;
    }
  }
}

// :299 (_registered_names).
std::set<std::string> RegisteredNames(const ChatCompletionRequest& request) {
  std::set<std::string> names;
  if (!request.tools.has_value()) return names;
  for (const ChatCompletionToolsParam& t : *request.tools) {
    if (!t.function.name.empty()) names.insert(t.function.name);
  }
  return names;
}

// The segment after the LAST '.' (the "trailing segment" / leaf).
std::string Leaf(const std::string& name) {
  const std::size_t dot = name.rfind('.');
  return dot == std::string::npos ? name : name.substr(dot + 1);
}

// :365 (_extract_content). The user-facing body, or the raw text when unframed.
std::optional<std::string> ExtractContent(const std::string& text) {
  std::string content, reasoning;
  bool c_open = false, r_open = false;
  VisibleChannels(text, &content, &reasoning, &c_open, &r_open);
  if (!content.empty()) return content;
  // No framing at all -> the whole thing is plain content.
  std::smatch dummy;
  if (reasoning.empty() && !std::regex_search(text, dummy, MsgHeaderRe())) {
    if (!text.empty()) return text;
    return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace

// :313 (_normalize_name), plus the leaf rule the upstream TEST demands. See the
// long note on the declaration in muse_glimmer.h.
std::string MuseGlimmerToolParser::normalize_name(
    const std::string& emitted, const std::set<std::string>& registered) {
  if (registered.empty() || registered.count(emitted) > 0) return emitted;

  // Upstream:330 — collapse the doubled form the shipped chat template induces
  // for a bare-registered tool (`get_weather` -> recipient `get_weather.*` ->
  // the model emits `get_weather.get_weather`). Head and tail are identical and
  // the collapsed name is registered, so this is unambiguous.
  const std::size_t dot = emitted.find('.');
  if (dot != std::string::npos) {
    const std::string head = emitted.substr(0, dot);
    const std::string tail = emitted.substr(dot + 1);
    if (head == tail && registered.count(head) > 0) return head;
  }

  // test_muse_glimmer_toolname_normalize.py::test_trailing_segment_unambiguous —
  // bind on the trailing segment, but ONLY onto a registered BARE name and ONLY
  // when exactly one registered tool has that leaf. Restricting to bare names is
  // what preserves upstream's stated safety invariant (:324) that an emitted
  // `weather.get` must not be rewritten onto a registered `calendar.get`.
  const std::string leaf = Leaf(emitted);
  if (leaf != emitted) {
    const std::string* bare_match = nullptr;
    std::size_t hits = 0;
    for (const std::string& r : registered) {
      if (Leaf(r) != leaf) continue;
      ++hits;
      if (r.find('.') == std::string::npos) bare_match = &r;
    }
    // Two independent guards, each with its own case in the suite:
    //   hits == 1        — the leaf must identify ONE registered tool. With
    //                      {get_weather, ns.get_weather} both registered, an
    //                      emitted `foo.get_weather` is ambiguous, so it stands.
    //   bare_match       — the target must be BARE. This is what keeps an
    //                      emitted `weather.get` off a registered `calendar.get`
    //                      (upstream:324), the case upstream's own docstring
    //                      says leaf matching must never dispatch.
    if (hits == 1 && bare_match != nullptr) return *bare_match;
  }
  // Upstream logs a warning here; this seam has no logger (deviation 2).
  return emitted;
}

// :241 (_tool_channel_text). Bodies of the messages addressed to a TOOL, joined
// with "\n". Falls back to the whole text when no message header is present at
// all — that means the framing never reached us, and scanning everything is
// strictly better than returning nothing.
std::string MuseGlimmerToolParser::tool_channel_text(const std::string& text) {
  std::vector<std::string> bodies;
  for (const Message& m : IterMessages(text)) {
    if (m.has_recipient && m.recipient != kReasoningRecipient &&
        m.recipient != kUserRecipient) {
      bodies.push_back(m.body);
    }
  }
  if (!bodies.empty()) {
    std::string joined = bodies[0];
    for (std::size_t i = 1; i < bodies.size(); ++i) joined += "\n" + bodies[i];
    return joined;
  }
  std::smatch dummy;
  if (!std::regex_search(text, dummy, MsgHeaderRe()) && HasAtem(text)) {
    // Upstream warns "ATEM markup with no channel framing; is
    // skip_special_tokens enabled upstream?" (deviation 2: no logger here).
    return text;
  }
  return std::string();
}

namespace {

// :340 (_parse_tool_calls).
std::vector<ToolCall> ParseToolCalls(const std::string& text,
                                     const std::set<std::string>& registered) {
  const std::string scoped = MuseGlimmerToolParser::tool_channel_text(text);
  std::vector<ToolCall> tool_calls;
  for (auto it = std::sregex_iterator(scoped.begin(), scoped.end(), InvokeRe());
       it != std::sregex_iterator(); ++it) {
    const std::string invoke = it->str();
    std::smatch nm;
    if (!std::regex_search(invoke, nm, NameRe())) continue;
    const std::string name =
        MuseGlimmerToolParser::normalize_name(nm[1].str(), registered);
    ordered_json args = ordered_json::object();
    for (auto pit = std::sregex_iterator(invoke.begin(), invoke.end(), ParamRe());
         pit != std::sregex_iterator(); ++pit) {
      args[(*pit)[1].str()] = DecodeValue((*pit)[2].str());
    }
    ToolCall tc;
    tc.id = make_tool_call_id();  // upstream ToolCall.id default_factory
    tc.type = "function";
    tc.function.name = name;
    tc.function.arguments = args.dump();
    tool_calls.push_back(std::move(tc));
  }
  return tool_calls;
}

}  // namespace

// :378 (extract_tool_calls).
ExtractedToolCallInformation MuseGlimmerToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  if (!HasAtem(model_output)) {
    return ExtractedToolCallInformation{false, {}, ExtractContent(model_output)};
  }
  try {
    const std::set<std::string> registered = RegisteredNames(request);
    std::vector<ToolCall> tool_calls = ParseToolCalls(model_output, registered);
    if (tool_calls.empty()) {
      // A tool block was opened but no COMPLETE <atem:invoke>...</atem:invoke>
      // parsed — typically a truncated call (finish_reason='length'/abort).
      // Upstream logs this because silently returning "no tool call" is
      // indistinguishable from the model choosing not to call one.
      return ExtractedToolCallInformation{false, {},
                                          ExtractContent(model_output)};
    }
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    info.content = ExtractContent(model_output);
    return info;
  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

// :422 (extract_tool_calls_streaming). Incremental ATEM streaming for tool calls
// AND content. Once reasoning has ended this parser owns every delta — anything
// it does not emit, INCLUDING the to=user final answer, never reaches the
// client — so content is emitted here, not left to the reasoning parser.
//
// Tool calls surface only when an <atem:invoke> block becomes COMPLETE: the XML
// is opaque until closed and Muse Glimmer parameters are not incremental JSON,
// so there is nothing meaningful to stream before then.
std::optional<DeltaMessage> MuseGlimmerToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& /*delta_text*/, const ChatCompletionRequest& request) {
  if (previous_text.empty()) {
    // First delta of the tool phase (the serving layer resets previous_text to
    // "" when it hands the stream over). Reset the cursors.
    streamed_content_len_ = 0;
    streamed_reasoning_len_ = 0;
    emitted_tool_calls_ = 0;
  }

  try {
    const std::set<std::string> registered = RegisteredNames(request);
    const std::vector<ToolCall> calls = ParseToolCalls(current_text, registered);

    std::string content, reasoning;
    bool content_open = false, reasoning_open = false;
    VisibleChannels(current_text, &content, &reasoning, &content_open,
                    &reasoning_open);

    // Deviation 4 (muse_glimmer.h): the unframed-content fallback the
    // non-streaming _extract_content already has. Our seam's reasoning parser
    // classifies the framing away before this parser sees the text, so without
    // this a to=user answer would be dropped on the floor.
    if (content.empty() && reasoning.empty() && !HasAtem(current_text)) {
      std::smatch dummy;
      if (!std::regex_search(current_text, dummy, MsgHeaderRe())) {
        content = current_text;
        content_open = true;
      }
    }

    // Trim the tail of a channel that is still growing, so the emitted prefix
    // never shrinks between deltas.
    if (content_open) content = SafeOpenBody(content);
    if (reasoning_open) reasoning = SafeOpenBody(reasoning);

    const std::string content_delta =
        content.size() > streamed_content_len_
            ? content.substr(streamed_content_len_)
            : std::string();
    const std::string reasoning_delta =
        reasoning.size() > streamed_reasoning_len_
            ? reasoning.substr(streamed_reasoning_len_)
            : std::string();

    std::vector<DeltaToolCall> tool_deltas;
    for (std::size_t i = emitted_tool_calls_; i < calls.size(); ++i) {
      DeltaToolCall d;
      d.index = static_cast<int>(i);
      d.type = "function";
      d.id = make_tool_call_id();
      d.function.name = calls[i].function.name;
      d.function.arguments = calls[i].function.arguments;
      tool_deltas.push_back(std::move(d));
    }

    if (content_delta.empty() && reasoning_delta.empty() && tool_deltas.empty()) {
      return std::nullopt;
    }

    streamed_content_len_ = content.size();
    streamed_reasoning_len_ = reasoning.size();
    emitted_tool_calls_ = calls.size();

    DeltaMessage message;
    if (!content_delta.empty()) message.content = content_delta;
    if (!reasoning_delta.empty()) message.reasoning = reasoning_delta;
    if (!tool_deltas.empty()) message.tool_calls = std::move(tool_deltas);
    return message;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
