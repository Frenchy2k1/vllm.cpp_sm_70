// Ported from: vllm/parser/engine/parser_engine_config.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// Declarative description of a model's tool-call / reasoning format. The engine
// feeds terminals from the incremental lexer into the transition table and
// emits the corresponding semantic events. Content tokens (text between
// terminals) are classified by the current state via content_events.
//
// SCOPE NOTE: the StreamingParserEngine reads only the streaming-relevant
// fields (terminals, token_id_terminals, transitions, content_events,
// initial_state, tool_args_json, preserve_tokens). The remaining
// ParserEngineConfig fields upstream (arg_converter, stream_arg_deltas,
// strip_*/validate_tool_names/drop_whitespace_only_content_before_tools,
// arg_structural_chars) are consumed by the higher-level ParserEngine assembly
// layer (ROAD-V1-C8 TOOLS-STREAMING-PARSER-ASSEMBLY, parser_engine.py) that
// turns SemanticEvents into DeltaMessage/ExtractedToolCallInformation. As of
// the assembly port those fields are carried here too (defaults mirror
// parser_engine_config.py:76-97); see .agents/specs/parser-assembly-c8.md.
#pragma once

#include <functional>
#include <map>
#include <optional>
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

  // ── Assembly-layer fields (parser_engine_config.py:76-97) ─────────────
  // Consumed by ParserEngine (parser_engine.{h,cpp}), NOT the streaming
  // engine. Optional-empty arg_converter == upstream `None`.

  // arg_converter(raw_args, partial) -> JSON args string. When set (e.g. the
  // qwen3 <parameter=...> XML converter), ARG_VALUE_CHUNK bodies are converted
  // to JSON before streaming/extraction. None => args stream/extract verbatim.
  using ArgConverter = std::function<std::string(const std::string& raw_args,
                                                 bool partial)>;
  ArgConverter arg_converter;

  // parser_engine_config.py:78 stream_arg_deltas.
  bool stream_arg_deltas = true;

  bool tool_args_json = true;

  // parser_engine_config.py:82 arg_structural_chars. When set, an arg delta with
  // NONE of these characters is skipped by the converter path (nullopt => off).
  std::optional<std::set<char>> arg_structural_chars;

  // Special tokens exempt from auto-drop but not state-machine terminals.
  std::set<std::string> preserve_tokens;

  // parser_engine_config.py:88 strip_trailing_reasoning_whitespace.
  bool strip_trailing_reasoning_whitespace = true;

  // parser_engine_config.py:91 drop_whitespace_only_content_before_tools.
  bool drop_whitespace_only_content_before_tools = true;

  // parser_engine_config.py:94 strip_content_whitespace_with_tools.
  bool strip_content_whitespace_with_tools = true;

  // parser_engine_config.py:97 validate_tool_names.
  bool validate_tool_names = false;
};

}  // namespace vllm::parser::engine
