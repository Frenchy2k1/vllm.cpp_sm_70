// Ported from: vllm/parser/kimi_k2.py @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/kimi_k2.h"

#include <regex>

namespace vllm::parser {

namespace {

bool is_py_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && is_py_space(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && is_py_space(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// kimi_k2.py:49 _TOOL_ID_RE = re.compile(r"(?P<id>.+:\d+)"), used with .match
// (anchored at start).
const std::regex& tool_id_re() {
  static const std::regex re(R"(^(.+:\d+))");
  return re;
}

}  // namespace

std::pair<std::optional<std::string>, std::optional<std::string>>
KimiK2Parser::extract_tool_id_and_name(
    const std::optional<std::string>& header) {
  if (!header) return {std::nullopt, std::nullopt};
  std::string h = strip(*header);
  std::smatch m;
  if (!std::regex_search(h, m, tool_id_re()))
    return {std::nullopt, std::nullopt};
  std::string tool_id = strip(m[1].str());
  std::string before_colon = tool_id.substr(0, tool_id.find(':'));
  const std::string prefix = "functions.";
  std::string tool_name = before_colon;
  if (before_colon.rfind(prefix, 0) == 0)
    tool_name = before_colon.substr(prefix.size());
  return {tool_id, tool_name};
}

void KimiK2Parser::emit_name_delta(
    int idx, std::vector<engine::oai::DeltaToolCall>& deltas,
    const std::optional<std::string>& name) {
  auto [tool_id, tool_name] = extract_tool_id_and_name(name);
  if (!tool_name || tool_name->empty()) {
    if (idx >= 0 && idx < static_cast<int>(tool_slots_.size()))
      tool_slots_[static_cast<std::size_t>(idx)].name = "";
    return;
  }
  tool_slots_[static_cast<std::size_t>(idx)].id = tool_id.value_or("");
  engine::ParserEngine::emit_name_delta(idx, deltas, tool_name);
}

void KimiK2Parser::handle_tool_end(
    const engine::SemanticEvent& event,
    std::vector<engine::oai::DeltaToolCall>& deltas) {
  const int idx = event.tool_index;
  if (idx >= 0 && idx < static_cast<int>(tool_slots_.size()) &&
      !tool_slots_[static_cast<std::size_t>(idx)].name_sent) {
    auto [tool_id, tool_name] =
        extract_tool_id_and_name(tool_slots_[static_cast<std::size_t>(idx)].name);
    if (tool_name && !tool_name->empty()) {
      tool_slots_[static_cast<std::size_t>(idx)].id = tool_id.value_or("");
      tool_slots_[static_cast<std::size_t>(idx)].name = *tool_name;
    }
  }
  engine::ParserEngine::handle_tool_end(event, deltas);
}

void KimiK2Parser::handle_arg_chunk(
    const engine::SemanticEvent& event,
    std::vector<engine::oai::DeltaToolCall>& deltas) {
  const int idx = event.tool_index;
  const bool name_sent_before =
      idx >= 0 && idx < static_cast<int>(tool_slots_.size()) &&
      tool_slots_[static_cast<std::size_t>(idx)].name_sent;
  engine::ParserEngine::handle_arg_chunk(event, deltas);
  if (!event.value.empty() && !name_sent_before && idx >= 0 &&
      idx < static_cast<int>(tool_slots_.size()) &&
      tool_slots_[static_cast<std::size_t>(idx)].name_sent) {
    engine::oai::DeltaToolCall d;
    d.index = idx;
    d.function.arguments = event.value;
    deltas.push_back(std::move(d));
  }
}

std::string KimiK2Parser::extract_args_json(const std::string& raw_args,
                                            const std::string& /*func_name*/) {
  std::string s = strip(raw_args);
  return s.empty() ? "{}" : s;
}

}  // namespace vllm::parser
