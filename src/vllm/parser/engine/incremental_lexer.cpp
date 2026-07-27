// Ported from: vllm/parser/engine/incremental_lexer.py @ 555967922.
#include "vllm/parser/engine/incremental_lexer.h"

#include <algorithm>

namespace vllm::parser::engine {

// incremental_lexer.py:46 (LexerShape.__init__).
LexerShape::LexerShape(const std::vector<TerminalDef>& terminals_in) {
  std::vector<TerminalDef> terms = terminals_in;
  // Sort key: (not is_literal, -len(pattern.pattern)). All our terminals are
  // literals with no regex-special chars in the ordering-relevant positions;
  // ordering only tie-breaks equal-length literals, which never both match, so
  // sorting by actual literal length (descending) is behavior-equivalent.
  std::stable_sort(terms.begin(), terms.end(),
                   [](const TerminalDef& a, const TerminalDef& b) {
                     if (a.is_literal != b.is_literal) return a.is_literal;
                     return a.literal.size() > b.literal.size();
                   });

  for (const auto& t : terms) {
    if (t.is_literal) {
      literal_strings.emplace_back(t.literal, t.name);
    }
    if (!t.is_literal) has_only_literals = false;
  }

  for (const auto& [lit, name] : literal_strings) {
    max_literal_len = std::max(max_literal_len, lit.size());
    if (!lit.empty()) literal_first_chars.insert(lit[0]);
  }

  for (const auto& [lit, name] : literal_strings) {
    for (std::size_t i = 1; i < lit.size(); ++i) {
      prefix_set.insert(lit.substr(0, i));
    }
  }

  for (const auto& [lit, name] : literal_strings) {
    if (!lit.empty()) literals_by_first[lit[0]].emplace_back(lit, name);
  }
}

IncrementalLexer::IncrementalLexer(LexerShape shape, std::string content_terminal)
    : shape_(std::move(shape)), content_terminal_(std::move(content_terminal)) {}

void IncrementalLexer::reset() { buffer_.clear(); }

// incremental_lexer.py:118 (feed).
std::vector<LexToken> IncrementalLexer::feed(const std::string& text) {
  if (buffer_.empty() && shape_.has_only_literals &&
      !shape_.literal_first_chars.empty()) {
    bool has_first = false;
    for (char ch : text) {
      if (shape_.literal_first_chars.count(ch)) {
        has_first = true;
        break;
      }
    }
    if (!has_first) {
      return {LexToken{content_terminal_, text}};
    }
  }
  buffer_ += text;
  return drain(false);
}

// incremental_lexer.py:128 (flush).
std::vector<LexToken> IncrementalLexer::flush() {
  std::vector<LexToken> tokens;
  if (!buffer_.empty()) {
    auto drained = drain(true);
    tokens.insert(tokens.end(), drained.begin(), drained.end());
  }
  if (!buffer_.empty()) {
    tokens.push_back(LexToken{content_terminal_, buffer_});
    buffer_.clear();
  }
  return tokens;
}

// incremental_lexer.py:137 (_drain).
std::vector<LexToken> IncrementalLexer::drain(bool final) {
  std::vector<LexToken> tokens;
  const auto& first_chars = shape_.literal_first_chars;
  const bool only_literals = shape_.has_only_literals;

  while (!buffer_.empty()) {
    if (only_literals && !first_chars.empty()) {
      bool has_potential = false;
      for (char ch : buffer_) {
        if (first_chars.count(ch)) {
          has_potential = true;
          break;
        }
      }
      if (!has_potential) {
        tokens.push_back(LexToken{content_terminal_, buffer_});
        buffer_.clear();
        break;
      }
    }

    const std::string* best_name = nullptr;
    const std::string* best_lit = nullptr;
    std::size_t best_len = 0;
    bool have_best = false;

    char first = buffer_[0];
    auto it = shape_.literals_by_first.find(first);
    if (it != shape_.literals_by_first.end()) {
      for (const auto& [lit, name] : it->second) {
        if (buffer_.compare(0, lit.size(), lit) == 0 &&
            (!have_best || lit.size() > best_len)) {
          best_name = &name;
          best_lit = &lit;
          best_len = lit.size();
          have_best = true;
        }
      }
    }

    // If the current buffer is both a complete literal and the prefix of a
    // longer literal, wait for the next chunk.
    if (shape_.prefix_set.count(buffer_) && !final) {
      if (have_best) {
        bool longer_match = false;
        if (it != shape_.literals_by_first.end()) {
          for (const auto& [lit, name] : it->second) {
            if (lit.size() > best_len &&
                lit.compare(0, buffer_.size(), buffer_) == 0) {
              longer_match = true;
              break;
            }
          }
        }
        if (!longer_match) {
          tokens.push_back(LexToken{*best_name, *best_lit});
          buffer_.erase(0, best_len);
          continue;
        }
        break;
      } else {
        break;
      }
    }

    if (have_best) {
      tokens.push_back(LexToken{*best_name, *best_lit});
      buffer_.erase(0, best_len);
    } else {
      std::size_t content_end = find_content_boundary();
      if (content_end > 0) {
        tokens.push_back(LexToken{content_terminal_, buffer_.substr(0, content_end)});
        buffer_.erase(0, content_end);
      } else {
        tokens.push_back(LexToken{content_terminal_, buffer_.substr(0, 1)});
        buffer_.erase(0, 1);
      }
    }
  }

  return tokens;
}

// incremental_lexer.py:199 (_find_content_boundary).
std::size_t IncrementalLexer::find_content_boundary() const {
  const std::string& buf = buffer_;
  std::size_t n = buf.size();
  const auto& first_chars = shape_.literal_first_chars;
  for (std::size_t i = 1; i < n; ++i) {
    if (!first_chars.count(buf[i])) continue;
    std::size_t remaining = n - i;
    for (const auto& [lit, name] : shape_.literal_strings) {
      std::size_t check_len = std::min(remaining, lit.size());
      if (buf.compare(i, check_len, lit, 0, check_len) == 0) {
        return i;
      }
    }
  }
  return n;
}

// incremental_lexer.py:214 (terminals_from_literals). Preserves insertion order.
std::vector<TerminalDef> terminals_from_literals(
    const std::map<std::string, std::string>& literals) {
  std::vector<TerminalDef> out;
  out.reserve(literals.size());
  for (const auto& [name, lit] : literals) {
    out.push_back(TerminalDef{name, lit, true});
  }
  return out;
}

}  // namespace vllm::parser::engine
