// Ported from: vllm/parser/glm47_moe.py @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/glm47_moe.h"

#include <cstddef>

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

}  // namespace

void Glm47MoeParser::emit_name_delta(
    int idx, std::vector<engine::oai::DeltaToolCall>& deltas,
    const std::optional<std::string>& name) {
  // glm47_moe.py:198: `if name is not None: name = name.strip()`.
  std::optional<std::string> stripped = name;
  if (stripped) stripped = strip(*stripped);
  engine::ParserEngine::emit_name_delta(idx, deltas, stripped);
}

void Glm47MoeParser::handle_tool_end(
    const engine::SemanticEvent& event,
    std::vector<engine::oai::DeltaToolCall>& deltas) {
  // glm47_moe.py:203-207: strip the slot name, then the base end handling.
  const int idx = event.tool_index;
  if (idx >= 0 && idx < static_cast<int>(tool_slots_.size())) {
    engine::ToolCallSlot& slot = tool_slots_[static_cast<std::size_t>(idx)];
    slot.name = strip(slot.name);
  }
  engine::ParserEngine::handle_tool_end(event, deltas);
}

}  // namespace vllm::parser
