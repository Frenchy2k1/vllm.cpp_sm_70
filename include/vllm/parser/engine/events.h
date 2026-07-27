// Ported from: vllm/parser/engine/events.py @ 555967922 (vLLM 0.26.0.dev0)
//
// Semantic event types emitted by the unified streaming parser engine. Each
// model tool-call / reasoning format is declared as a ParserEngineConfig
// (terminals + a transition state machine); the shared StreamingParserEngine
// turns (delta_text, delta_token_ids) pairs into a stream of SemanticEvents.
#pragma once

#include <string>

namespace vllm::parser::engine {

// Ported from: vllm/parser/engine/events.py:11 (EventType).
enum class EventType {
  TEXT_CHUNK,
  REASONING_START,
  REASONING_CHUNK,
  REASONING_END,
  TOOL_CALL_START,
  TOOL_NAME,
  ARG_VALUE_CHUNK,
  TOOL_CALL_END,
};

// Ported from: vllm/parser/engine/events.py:22 (SemanticEvent).
struct SemanticEvent {
  EventType type;
  std::string value;
  int tool_index = -1;

  SemanticEvent(EventType t, std::string v = "", int idx = -1)
      : type(t), value(std::move(v)), tool_index(idx) {}

  bool operator==(const SemanticEvent& o) const {
    return type == o.type && value == o.value && tool_index == o.tool_index;
  }
};

inline const char* event_type_name(EventType t) {
  switch (t) {
    case EventType::TEXT_CHUNK: return "TEXT_CHUNK";
    case EventType::REASONING_START: return "REASONING_START";
    case EventType::REASONING_CHUNK: return "REASONING_CHUNK";
    case EventType::REASONING_END: return "REASONING_END";
    case EventType::TOOL_CALL_START: return "TOOL_CALL_START";
    case EventType::TOOL_NAME: return "TOOL_NAME";
    case EventType::ARG_VALUE_CHUNK: return "ARG_VALUE_CHUNK";
    case EventType::TOOL_CALL_END: return "TOOL_CALL_END";
  }
  return "?";
}

}  // namespace vllm::parser::engine
