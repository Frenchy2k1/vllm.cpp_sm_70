// Ported from: vllm/parser/engine/token_id_scanner.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// Scans delta token IDs for special tokens and splits the stream into
// pre-lexed terminals and plain text chunks. Handles detokenizer hold-back
// text and deferral of terminals whose text has not yet arrived.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace vllm::parser::engine {

inline constexpr const char* kDropTerminal = "__DROP__";

// Abstract tokenizer seam the scanner needs: single-token decode + vocab, and
// (optionally) the special-token table used to build drop info. Return false
// from all_special to mirror upstream's AttributeError/NotImplementedError
// fallback (drop machinery disabled).
class EngineTokenizer {
 public:
  virtual ~EngineTokenizer() = default;
  virtual std::string decode(int token_id) const = 0;
  virtual const std::map<std::string, int>& get_vocab() const = 0;
  virtual bool all_special(std::vector<std::string>& tokens,
                           std::vector<int>& ids) const {
    (void)tokens;
    (void)ids;
    return false;
  }
};

// Ported from: token_id_scanner.py:14 (TextChunk) + :19 (PreLexedTerminal),
// unified as a tagged item.
struct LexerInput {
  enum class Kind { Text, Terminal } kind;
  std::string text;       // TextChunk.text OR PreLexedTerminal.text
  std::string terminal;   // PreLexedTerminal.terminal
  int token_id = -1;      // PreLexedTerminal.token_id

  static LexerInput Chunk(std::string t) {
    return LexerInput{Kind::Text, std::move(t), "", -1};
  }
  static LexerInput Terminal_(std::string term, int tid, std::string t) {
    return LexerInput{Kind::Terminal, std::move(t), std::move(term), tid};
  }
  bool is_text() const { return kind == Kind::Text; }
  bool is_terminal() const { return kind == Kind::Terminal; }
};

// Ported from: token_id_scanner.py:29 (TokenIDScanner).
class TokenIDScanner {
 public:
  TokenIDScanner(std::map<int, std::string> token_id_to_terminal,
                 const EngineTokenizer* tokenizer);

  void reset();
  std::vector<LexerInput> scan(const std::string& delta_text,
                               const std::vector<int>& delta_token_ids);
  std::vector<LexerInput> flush_pending();

  bool has_deferred() const { return !deferred_terminals_.empty(); }

 private:
  std::string decode_token(int token_id);
  std::pair<std::vector<LexerInput>, std::string> resolve_deferred(
      const std::string& delta_text);
  std::vector<LexerInput> recover_holdback_text(const std::string& delta_text,
                                                std::vector<LexerInput> results);
  std::string join_decoded_text(const std::vector<LexerInput>& results) const;
  std::vector<LexerInput> rebuild_from_anchors(const std::string& delta_text,
                                               std::vector<LexerInput>& results);

  std::map<int, std::string> token_id_to_terminal_;
  const EngineTokenizer* tokenizer_;
  std::map<int, std::string> token_text_cache_;
  std::vector<LexerInput> deferred_terminals_;  // all Kind::Terminal
  std::string deferred_post_text_;
};

}  // namespace vllm::parser::engine
