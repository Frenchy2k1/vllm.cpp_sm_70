// Ported from: vllm/parser/engine/parser_engine_config.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// Declarative description of a model's tool-call / reasoning format. The engine
// feeds terminals from the incremental lexer into the transition table and
// emits the corresponding semantic events. Content tokens (text between
// terminals) are classified by the current state via content_events.
//
// SCOPE NOTE: this struct carries only the fields the StreamingParserEngine
// actually reads (terminals, token_id_terminals, transitions, content_events,
// initial_state, tool_args_json, preserve_tokens). The remaining
// ParserEngineConfig fields upstream (arg_converter, stream_arg_deltas,
// strip_*/validate_tool_names/drop_whitespace_only_content_before_tools,
// arg_structural_chars) are consumed by the higher-level ParserEngine assembly
// layer (parser_engine.py) that turns SemanticEvents into DeltaMessage/
// ExtractedToolCallInformation, a tracked residual (see
// .agents/specs/streaming-parser-engine.md).
#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/parser/engine/events.h"

namespace vllm::parser::engine {

// Ported from: parser_engine_config.py:28 (ParserState).
enum class ParserState {
  CONTENT,
  REASONING,
  MESSAGE_HEADER,
  TOOL_PREAMBLE,
  TOOL_NAME,
  TOOL_ARGS,
  TOOL_BETWEEN,
};

// Ported from: parser_engine_config.py:38 (Transition).
struct Transition {
  ParserState next_state;
  std::vector<EventType> events;
  bool skip_in_token_id_mode = false;

  Transition() : next_state(ParserState::CONTENT) {}
  Transition(ParserState ns, std::vector<EventType> ev = {},
             bool skip = false)
      : next_state(ns), events(std::move(ev)), skip_in_token_id_mode(skip) {}
};

using TransitionKey = std::pair<ParserState, std::string>;

// Ported from: parser_engine_config.py:45 (ParserEngineConfig).
struct ParserEngineConfig {
  std::string name;

  // terminal_name -> literal string
  std::map<std::string, std::string> terminals;

  // terminal_name -> token text (matched by token ID)
  std::map<std::string, std::string> token_id_terminals;

  // (state, terminal_name) -> transition
  std::map<TransitionKey, Transition> transitions;

  // Per-state content classification. Defaults mirror upstream
  // parser_engine_config.py:66.
  std::map<ParserState, EventType> content_events{
      {ParserState::CONTENT, EventType::TEXT_CHUNK},
      {ParserState::REASONING, EventType::REASONING_CHUNK},
      {ParserState::TOOL_NAME, EventType::TOOL_NAME},
      {ParserState::TOOL_ARGS, EventType::ARG_VALUE_CHUNK},
  };

  ParserState initial_state = ParserState::CONTENT;

  bool tool_args_json = true;

  // Special tokens exempt from auto-drop but not state-machine terminals.
  std::set<std::string> preserve_tokens;
};

}  // namespace vllm::parser::engine
