// Ported from: vllm/parser/engine/incremental_lexer.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// Incremental text lexer that converts text chunks into terminal tokens, with
// prefix-match buffering for ambiguous boundaries (e.g. "<tool_" held until it
// either completes "<tool_call>" or flushes as content).
//
// PORT SCOPE: literal terminals only. Every engine config registered by vLLM
// 0.26 (qwen3, kimi_k2, gemma4, minimax_m2, deepseek_v4/v32, seed_oss,
// glm47_moe, nemotron_v3, inkling) builds its terminals via
// terminals_from_literals (all literals). Non-literal (regex) terminals are a
// tracked residual. Byte-wise iteration is equivalent to upstream's
// character-wise iteration because every literal's first character is ASCII
// (< 0x80), so a UTF-8 lead/continuation byte can never spuriously match a
// terminal first-char and content boundaries always fall on char boundaries.
#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vllm::parser::engine {

inline constexpr const char* kContentTerminal = "__CONTENT__";

// Ported from: incremental_lexer.py:15 (TerminalDef). Literal-only subset.
struct TerminalDef {
  std::string name;
  std::string literal;
  bool is_literal = true;
};

// Ported from: incremental_lexer.py:23 (LexToken).
struct LexToken {
  std::string terminal;
  std::string value;
};

// Ported from: incremental_lexer.py:29 (LexerShape). Immutable precomputed data
// derived from terminal definitions; shared across lexer instances.
class LexerShape {
 public:
  LexerShape() = default;
  explicit LexerShape(const std::vector<TerminalDef>& terminals);

  // (literal, name) pairs, sorted literals-first then longest-first.
  std::vector<std::pair<std::string, std::string>> literal_strings;
  std::size_t max_literal_len = 0;
  std::set<char> literal_first_chars;
  bool has_only_literals = true;
  std::set<std::string> prefix_set;
  std::map<char, std::vector<std::pair<std::string, std::string>>>
      literals_by_first;
};

// Ported from: incremental_lexer.py:80 (IncrementalLexer).
class IncrementalLexer {
 public:
  explicit IncrementalLexer(LexerShape shape,
                            std::string content_terminal = kContentTerminal);

  void reset();
  std::vector<LexToken> feed(const std::string& text);
  std::vector<LexToken> flush();

  const std::string& buffer() const { return buffer_; }
  const std::set<char>& literal_first_chars() const {
    return shape_.literal_first_chars;
  }

 private:
  std::vector<LexToken> drain(bool final);
  std::size_t find_content_boundary() const;

  LexerShape shape_;
  std::string content_terminal_;
  std::string buffer_;
};

// Ported from: incremental_lexer.py:214 (terminals_from_literals).
std::vector<TerminalDef> terminals_from_literals(
    const std::map<std::string, std::string>& literals);

}  // namespace vllm::parser::engine
