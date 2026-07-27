// Ported from: vllm/parser/engine/streaming_parser_engine.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// Streaming parser engine that orchestrates token-ID scanning, incremental
// lexing, and state-machine-driven semantic event emission. Consumes
// (delta_text, delta_token_ids) pairs and produces a stream of SemanticEvents.
// Create one per request (it is stateful).
#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vllm/parser/engine/events.h"
#include "vllm/parser/engine/incremental_lexer.h"
#include "vllm/parser/engine/parser_engine_config.h"
#include "vllm/parser/engine/token_id_scanner.h"

namespace vllm::parser::engine {

class StreamingParserEngine {
 public:
  StreamingParserEngine(
      ParserEngineConfig config, const EngineTokenizer* tokenizer,
      std::optional<ParserState> initial_state = std::nullopt,
      const std::map<std::string, int>* vocab = nullptr);

  void reset(std::optional<ParserState> initial_state = std::nullopt);

  std::vector<SemanticEvent> feed(const std::string& delta_text,
                                  const std::vector<int>& delta_token_ids);
  std::vector<SemanticEvent> finish();
  std::vector<SemanticEvent> parse_complete(const std::string& text);

  ParserState state() const { return state_; }
  int tool_index() const { return tool_index_; }
  bool skip_tool_parsing = false;

 private:
  // Precomputed token-ID/lexer setup (mirrors the __init__ body ordering so
  // the scanner and lexer members can be built in the initializer list).
  struct Prep {
    std::map<int, std::string> resolved_token_ids;
    LexerShape lexer_shape;
    bool has_drops = false;
  };
  static Prep prepare(const ParserEngineConfig& config,
                      const EngineTokenizer* tokenizer,
                      const std::map<std::string, int>* vocab);
  StreamingParserEngine(ParserEngineConfig config,
                        const EngineTokenizer* tokenizer,
                        std::optional<ParserState> initial_state, Prep prep);

  void reset_args_state();
  std::vector<SemanticEvent> process_scanner_items(
      const std::vector<LexerInput>& items);
  std::vector<SemanticEvent> process_lex_tokens(
      const std::vector<LexToken>& tokens);
  std::vector<SemanticEvent> on_terminal(const std::string& terminal,
                                         const std::string& value);
  std::vector<SemanticEvent> emit_for_state(const std::string& text);
  std::vector<SemanticEvent> on_content(const std::string& text);
  std::vector<SemanticEvent> apply_transition(const Transition& transition,
                                              const std::string& value);
  std::vector<SemanticEvent> feed_args_text(const std::string& text);
  std::vector<SemanticEvent> feed_args_char(char ch);
  std::vector<SemanticEvent> flush_safe_args();

  ParserEngineConfig config_;
  std::map<int, std::string> resolved_token_ids_;
  bool has_drops_ = false;
  std::set<std::string> token_id_terminal_names_;
  std::set<std::string> tool_terminals_;

  TokenIDScanner scanner_;
  IncrementalLexer lexer_;

  ParserState state_;
  int tool_index_ = -1;
  bool ever_had_token_ids_ = false;

  std::string args_buffer_;
  std::size_t args_safe_end_ = 0;
  int args_brace_depth_ = 0;
  bool args_in_string_ = false;
  bool args_escape_next_ = false;
};

}  // namespace vllm::parser::engine
