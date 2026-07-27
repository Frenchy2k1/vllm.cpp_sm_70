// Ported from: vllm/parser/engine/streaming_parser_engine.py @ 555967922.
#include "vllm/parser/engine/streaming_parser_engine.h"

#include <optional>
#include <utility>

namespace vllm::parser::engine {

namespace {

const std::set<ParserState>& tool_states() {
  static const std::set<ParserState> s = {
      ParserState::TOOL_PREAMBLE, ParserState::TOOL_NAME, ParserState::TOOL_ARGS,
      ParserState::TOOL_BETWEEN};
  return s;
}

std::optional<EventType> content_event_for(const ParserEngineConfig& cfg,
                                           ParserState state) {
  auto it = cfg.content_events.find(state);
  if (it == cfg.content_events.end()) return std::nullopt;
  return it->second;
}

// streaming_parser_engine.py:39 (_build_drop_info). Returns nullopt when the
// tokenizer exposes no special-token table (mirrors the AttributeError path) or
// there is nothing to drop.
struct DropInfo {
  LexerShape lexer_shape;
  std::map<int, std::string> extra_token_ids;
};

std::optional<DropInfo> build_drop_info(const ParserEngineConfig& config,
                                        const EngineTokenizer* tokenizer) {
  std::vector<std::string> special_tokens;
  std::vector<int> special_ids;
  if (!tokenizer->all_special(special_tokens, special_ids)) return std::nullopt;
  if (special_tokens.empty()) return std::nullopt;

  std::set<std::string> configured;
  for (const auto& [name, text] : config.token_id_terminals) configured.insert(text);
  for (const auto& [name, text] : config.terminals) configured.insert(text);
  for (const auto& t : config.preserve_tokens) configured.insert(t);

  std::map<int, std::string> extra_token_ids;
  std::set<std::string> drop_texts;
  std::size_t n = std::min(special_tokens.size(), special_ids.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (!configured.count(special_tokens[i])) {
      extra_token_ids[special_ids[i]] = kDropTerminal;
      drop_texts.insert(special_tokens[i]);
    }
  }
  if (drop_texts.empty()) return std::nullopt;

  std::vector<TerminalDef> all_defs = terminals_from_literals(config.terminals);
  for (const auto& text : drop_texts) {
    all_defs.push_back(TerminalDef{kDropTerminal, text, true});
  }
  return DropInfo{LexerShape(all_defs), std::move(extra_token_ids)};
}

}  // namespace

// streaming_parser_engine.py:112 (__init__) — precompute step.
StreamingParserEngine::Prep StreamingParserEngine::prepare(
    const ParserEngineConfig& config, const EngineTokenizer* tokenizer,
    const std::map<std::string, int>* vocab) {
  Prep p;
  if (tokenizer != nullptr) {
    const std::map<std::string, int>& v =
        vocab != nullptr ? *vocab : tokenizer->get_vocab();
    if (!config.token_id_terminals.empty()) {
      for (const auto& [terminal_name, token_text] : config.token_id_terminals) {
        auto it = v.find(token_text);
        if (it != v.end()) p.resolved_token_ids[it->second] = terminal_name;
      }
    }
  }

  p.lexer_shape = LexerShape(terminals_from_literals(config.terminals));

  if (tokenizer != nullptr) {
    auto drop = build_drop_info(config, tokenizer);
    if (drop) {
      for (const auto& [tid, name] : drop->extra_token_ids) {
        p.resolved_token_ids[tid] = name;
      }
      p.lexer_shape = drop->lexer_shape;
      p.has_drops = true;
    }
  }
  return p;
}

StreamingParserEngine::StreamingParserEngine(
    ParserEngineConfig config, const EngineTokenizer* tokenizer,
    std::optional<ParserState> initial_state,
    const std::map<std::string, int>* vocab)
    : StreamingParserEngine(config, tokenizer, initial_state,
                            prepare(config, tokenizer, vocab)) {}

StreamingParserEngine::StreamingParserEngine(ParserEngineConfig config,
                                             const EngineTokenizer* tokenizer,
                                             std::optional<ParserState> initial_state,
                                             Prep prep)
    : config_(std::move(config)),
      resolved_token_ids_(std::move(prep.resolved_token_ids)),
      has_drops_(prep.has_drops),
      scanner_(resolved_token_ids_, tokenizer),
      lexer_(prep.lexer_shape) {
  for (const auto& [tid, name] : resolved_token_ids_) {
    token_id_terminal_names_.insert(name);
  }
  const auto& ts = tool_states();
  for (const auto& [key, tr] : config_.transitions) {
    if (ts.count(tr.next_state) || ts.count(key.first)) {
      tool_terminals_.insert(key.second);
    }
  }
  reset(initial_state);
}

void StreamingParserEngine::reset_args_state() {
  args_buffer_.clear();
  args_safe_end_ = 0;
  args_brace_depth_ = 0;
  args_in_string_ = false;
  args_escape_next_ = false;
}

// streaming_parser_engine.py:170 (reset). Preserves skip_tool_parsing.
void StreamingParserEngine::reset(std::optional<ParserState> initial_state) {
  state_ = initial_state ? *initial_state : config_.initial_state;
  tool_index_ = -1;
  ever_had_token_ids_ = false;
  scanner_.reset();
  lexer_.reset();
  reset_args_state();
}

// streaming_parser_engine.py:190 (feed).
std::vector<SemanticEvent> StreamingParserEngine::feed(
    const std::string& delta_text, const std::vector<int>& delta_token_ids) {
  if (!delta_token_ids.empty()) ever_had_token_ids_ = true;

  // Fast path: plain content with no special tokens / terminal-starting chars.
  if (!delta_text.empty() && lexer_.buffer().empty() && !scanner_.has_deferred()) {
    bool disjoint = true;
    for (char ch : delta_text) {
      if (lexer_.literal_first_chars().count(ch)) {
        disjoint = false;
        break;
      }
    }
    if (disjoint) {
      bool has_special = false;
      for (int tid : delta_token_ids) {
        if (resolved_token_ids_.count(tid)) {
          has_special = true;
          break;
        }
      }
      if (!has_special) return emit_for_state(delta_text);
    }
  }

  std::vector<LexerInput> scanner_items = scanner_.scan(delta_text, delta_token_ids);

  if (scanner_items.size() == 1 && scanner_items[0].is_text()) {
    std::vector<LexToken> lex_tokens = lexer_.feed(scanner_items[0].text);
    if (lex_tokens.size() == 1 && lex_tokens[0].terminal == kContentTerminal) {
      return emit_for_state(lex_tokens[0].value);
    }
    return process_lex_tokens(lex_tokens);
  }

  return process_scanner_items(scanner_items);
}

// streaming_parser_engine.py:225 (_process_scanner_items).
std::vector<SemanticEvent> StreamingParserEngine::process_scanner_items(
    const std::vector<LexerInput>& items) {
  std::vector<SemanticEvent> events;
  for (const auto& item : items) {
    if (item.is_terminal()) {
      auto flushed = process_lex_tokens(lexer_.flush());
      events.insert(events.end(), flushed.begin(), flushed.end());
      auto e = on_terminal(item.terminal, item.text);
      events.insert(events.end(), e.begin(), e.end());
    } else {
      auto fed = process_lex_tokens(lexer_.feed(item.text));
      events.insert(events.end(), fed.begin(), fed.end());
    }
  }
  return events;
}

// streaming_parser_engine.py:237 (finish).
std::vector<SemanticEvent> StreamingParserEngine::finish() {
  std::vector<SemanticEvent> events = process_scanner_items(scanner_.flush_pending());

  auto flushed = process_lex_tokens(lexer_.flush());
  events.insert(events.end(), flushed.begin(), flushed.end());

  if (!args_buffer_.empty()) {
    events.emplace_back(EventType::ARG_VALUE_CHUNK, args_buffer_, tool_index_);
    args_buffer_.clear();
    args_safe_end_ = 0;
  }

  if (state_ == ParserState::TOOL_PREAMBLE || state_ == ParserState::TOOL_ARGS ||
      state_ == ParserState::TOOL_NAME || state_ == ParserState::TOOL_BETWEEN) {
    if (tool_index_ >= 0) {
      events.emplace_back(EventType::TOOL_CALL_END, "", tool_index_);
    }
    state_ = ParserState::CONTENT;
  } else if (state_ == ParserState::REASONING) {
    events.emplace_back(EventType::REASONING_END, "", tool_index_);
    state_ = ParserState::CONTENT;
  } else if (state_ == ParserState::MESSAGE_HEADER) {
    state_ = ParserState::CONTENT;
  }

  return events;
}

// streaming_parser_engine.py:277 (parse_complete).
std::vector<SemanticEvent> StreamingParserEngine::parse_complete(
    const std::string& text) {
  std::vector<int> token_ids;
  std::vector<SemanticEvent> events = feed(text, token_ids);
  auto fin = finish();
  events.insert(events.end(), fin.begin(), fin.end());
  return events;
}

// streaming_parser_engine.py:283 (_process_lex_tokens).
std::vector<SemanticEvent> StreamingParserEngine::process_lex_tokens(
    const std::vector<LexToken>& tokens) {
  std::vector<SemanticEvent> events;
  const bool strict = ever_had_token_ids_;
  for (const auto& tok : tokens) {
    bool as_content =
        (tok.terminal == kContentTerminal) ||
        (strict && token_id_terminal_names_.count(tok.terminal) > 0);
    if (as_content) {
      auto e = on_content(tok.value);
      events.insert(events.end(), e.begin(), e.end());
    } else {
      auto e = on_terminal(tok.terminal, tok.value);
      events.insert(events.end(), e.begin(), e.end());
    }
  }
  return events;
}

// streaming_parser_engine.py:302 (_on_terminal).
std::vector<SemanticEvent> StreamingParserEngine::on_terminal(
    const std::string& terminal, const std::string& value) {
  auto it = config_.transitions.find({state_, terminal});
  if (it == config_.transitions.end()) {
    if (has_drops_ && terminal == kDropTerminal) return {};
    return emit_for_state(value);
  }
  const Transition& transition = it->second;

  if (skip_tool_parsing && tool_terminals_.count(terminal)) {
    if (state_ == ParserState::MESSAGE_HEADER) {
      state_ = ParserState::CONTENT;
      return {SemanticEvent(EventType::TEXT_CHUNK, value, tool_index_)};
    }
    bool has_reasoning_end = false;
    for (EventType e : transition.events) {
      if (e == EventType::REASONING_END) {
        has_reasoning_end = true;
        break;
      }
    }
    if (has_reasoning_end) {
      state_ = ParserState::CONTENT;
      return {SemanticEvent(EventType::REASONING_END, value, tool_index_),
              SemanticEvent(EventType::TEXT_CHUNK, value, tool_index_)};
    }
    auto ct = content_event_for(config_, state_);
    if (ct) return {SemanticEvent(*ct, value, tool_index_)};
    return {};
  }

  if (transition.skip_in_token_id_mode && ever_had_token_ids_) {
    return emit_for_state(value);
  }

  return apply_transition(transition, value);
}

// streaming_parser_engine.py:347 (_emit_for_state).
std::vector<SemanticEvent> StreamingParserEngine::emit_for_state(
    const std::string& text) {
  if (state_ == ParserState::TOOL_ARGS) {
    if (config_.tool_args_json) return feed_args_text(text);
    return {SemanticEvent(EventType::ARG_VALUE_CHUNK, text, tool_index_)};
  }
  auto ct = content_event_for(config_, state_);
  if (ct) return {SemanticEvent(*ct, text, tool_index_)};
  return {};
}

// streaming_parser_engine.py:363 (_on_content).
std::vector<SemanticEvent> StreamingParserEngine::on_content(
    const std::string& text) {
  if (text.empty()) return {};
  return emit_for_state(text);
}

// streaming_parser_engine.py:368 (_apply_transition).
std::vector<SemanticEvent> StreamingParserEngine::apply_transition(
    const Transition& transition, const std::string& value) {
  std::vector<SemanticEvent> events;

  if (state_ == ParserState::TOOL_ARGS &&
      transition.next_state != ParserState::TOOL_ARGS && !args_buffer_.empty()) {
    events.emplace_back(EventType::ARG_VALUE_CHUNK, args_buffer_, tool_index_);
    args_buffer_.clear();
  }

  state_ = transition.next_state;

  for (EventType event_type : transition.events) {
    if (event_type == EventType::TOOL_CALL_START) tool_index_ += 1;
    events.emplace_back(event_type, value, tool_index_);
  }

  if (state_ == ParserState::TOOL_ARGS) {
    args_brace_depth_ = 0;
    args_in_string_ = false;
    args_escape_next_ = false;
    args_safe_end_ = 0;
  }

  return events;
}

// streaming_parser_engine.py:410 (_feed_args_text).
std::vector<SemanticEvent> StreamingParserEngine::feed_args_text(
    const std::string& text) {
  std::vector<SemanticEvent> events;
  for (char ch : text) {
    auto r = feed_args_char(ch);
    events.insert(events.end(), r.begin(), r.end());
  }
  return events;
}

// streaming_parser_engine.py:422 (_feed_args_char).
std::vector<SemanticEvent> StreamingParserEngine::feed_args_char(char ch) {
  args_buffer_.push_back(ch);

  if (args_escape_next_) {
    args_escape_next_ = false;
    args_safe_end_ = args_buffer_.size();
    return flush_safe_args();
  }

  if (args_in_string_) {
    if (ch == '\\') {
      args_escape_next_ = true;
    } else if (ch == '"') {
      args_in_string_ = false;
    }
    args_safe_end_ = args_buffer_.size();
    return flush_safe_args();
  }

  if (ch == '"') {
    args_in_string_ = true;
    args_safe_end_ = args_buffer_.size();
    return flush_safe_args();
  }

  if (ch == '{' || ch == '[') {
    args_brace_depth_ += 1;
    args_safe_end_ = args_buffer_.size();
    return flush_safe_args();
  }

  if (ch == '}' || ch == ']') {
    if (args_brace_depth_ > 0) args_brace_depth_ -= 1;
    if (args_brace_depth_ == 0) return {};
    args_safe_end_ = args_buffer_.size();
    return flush_safe_args();
  }

  args_safe_end_ = args_buffer_.size();
  return flush_safe_args();
}

// streaming_parser_engine.py:459 (_flush_safe_args).
std::vector<SemanticEvent> StreamingParserEngine::flush_safe_args() {
  if (args_safe_end_ == 0) return {};
  std::string to_emit = args_buffer_.substr(0, args_safe_end_);
  args_buffer_ = args_buffer_.substr(args_safe_end_);
  args_safe_end_ = 0;
  return {SemanticEvent(EventType::ARG_VALUE_CHUNK, to_emit, tool_index_)};
}

}  // namespace vllm::parser::engine
