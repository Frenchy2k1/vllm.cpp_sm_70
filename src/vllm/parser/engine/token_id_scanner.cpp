// Ported from: vllm/parser/engine/token_id_scanner.py @ 555967922.
#include "vllm/parser/engine/token_id_scanner.h"

#include <string>

namespace vllm::parser::engine {

TokenIDScanner::TokenIDScanner(std::map<int, std::string> token_id_to_terminal,
                               const EngineTokenizer* tokenizer)
    : token_id_to_terminal_(std::move(token_id_to_terminal)),
      tokenizer_(tokenizer) {}

// token_id_scanner.py:54 (reset) — preserves the token text cache.
void TokenIDScanner::reset() {
  deferred_terminals_.clear();
  deferred_post_text_.clear();
}

std::string TokenIDScanner::decode_token(int token_id) {
  auto it = token_text_cache_.find(token_id);
  if (it != token_text_cache_.end()) return it->second;
  std::string t = tokenizer_ ? tokenizer_->decode(token_id) : std::string();
  token_text_cache_[token_id] = t;
  return t;
}

// token_id_scanner.py:66 (scan).
std::vector<LexerInput> TokenIDScanner::scan(
    const std::string& delta_text, const std::vector<int>& delta_token_ids) {
  std::vector<LexerInput> prefix_items;
  std::string effective_text = delta_text;

  if (!deferred_terminals_.empty()) {
    auto [items, remaining] = resolve_deferred(delta_text);
    prefix_items = std::move(items);
    effective_text = std::move(remaining);
  }

  if (token_id_to_terminal_.empty()) {
    if (!effective_text.empty()) {
      prefix_items.push_back(LexerInput::Chunk(effective_text));
    }
    return prefix_items;
  }

  bool has_special = false;
  for (int tid : delta_token_ids) {
    if (token_id_to_terminal_.count(tid)) {
      has_special = true;
      break;
    }
  }

  if (!has_special) {
    if (!effective_text.empty()) {
      if (prefix_items.empty()) {
        return {LexerInput::Chunk(effective_text)};
      }
      prefix_items.push_back(LexerInput::Chunk(effective_text));
    }
    return prefix_items;
  }

  std::vector<std::string> token_texts;
  token_texts.reserve(delta_token_ids.size());
  for (int tid : delta_token_ids) token_texts.push_back(decode_token(tid));

  std::vector<LexerInput> results;
  std::string text_accum;

  for (std::size_t idx = 0; idx < delta_token_ids.size(); ++idx) {
    int tid = delta_token_ids[idx];
    auto it = token_id_to_terminal_.find(tid);
    if (it != token_id_to_terminal_.end()) {
      if (!text_accum.empty()) {
        results.push_back(LexerInput::Chunk(text_accum));
        text_accum.clear();
      }
      results.push_back(LexerInput::Terminal_(it->second, tid, token_texts[idx]));
    } else {
      text_accum += token_texts[idx];
    }
  }
  if (!text_accum.empty()) {
    results.push_back(LexerInput::Chunk(text_accum));
  }

  if (!effective_text.empty()) {
    results = recover_holdback_text(effective_text, std::move(results));
  } else {
    // No detokenizer text to validate against — defer PreLexedTerminals so the
    // state machine doesn't transition before preceding text arrives.
    std::vector<LexerInput> kept;
    for (auto& r : results) {
      if (r.is_terminal()) {
        deferred_terminals_.push_back(r);
      }
    }
    results.clear();
  }

  std::vector<LexerInput> out = std::move(prefix_items);
  out.insert(out.end(), results.begin(), results.end());
  return out;
}

// token_id_scanner.py:134 (flush_pending).
std::vector<LexerInput> TokenIDScanner::flush_pending() {
  if (deferred_terminals_.empty() && deferred_post_text_.empty()) return {};
  std::vector<LexerInput> results;
  if (!deferred_post_text_.empty()) {
    results.push_back(LexerInput::Chunk(deferred_post_text_));
    deferred_post_text_.clear();
  }
  results.insert(results.end(), deferred_terminals_.begin(),
                 deferred_terminals_.end());
  deferred_terminals_.clear();
  return results;
}

// token_id_scanner.py:145 (_resolve_deferred).
std::pair<std::vector<LexerInput>, std::string> TokenIDScanner::resolve_deferred(
    const std::string& delta_text) {
  std::vector<LexerInput> deferred = std::move(deferred_terminals_);
  deferred_terminals_.clear();

  std::vector<LexerInput> results;
  std::string remaining = delta_text;

  if (!deferred_post_text_.empty()) {
    remaining = deferred_post_text_ + remaining;
    deferred_post_text_.clear();
  }

  for (auto& terminal : deferred) {
    std::size_t pos = remaining.find(terminal.text);
    if (pos != std::string::npos && pos > 0) {
      results.push_back(LexerInput::Chunk(remaining.substr(0, pos)));
      results.push_back(terminal);
      remaining = remaining.substr(pos + terminal.text.size());
    } else if (pos == 0) {
      results.push_back(terminal);
      remaining = remaining.substr(terminal.text.size());
    } else {
      if (!remaining.empty()) {
        deferred_post_text_ += remaining;
        remaining.clear();
      }
      deferred_terminals_.push_back(terminal);
    }
  }

  return {std::move(results), std::move(remaining)};
}

// token_id_scanner.py:193 (_recover_holdback_text).
std::vector<LexerInput> TokenIDScanner::recover_holdback_text(
    const std::string& delta_text, std::vector<LexerInput> results) {
  if (results.empty()) {
    return {LexerInput::Chunk(delta_text)};
  }

  std::string reconstructed = join_decoded_text(results);

  if (reconstructed.empty()) {
    std::vector<LexerInput> out;
    out.push_back(LexerInput::Chunk(delta_text));
    out.insert(out.end(), results.begin(), results.end());
    return out;
  }

  std::size_t pos = delta_text.find(reconstructed);
  if (pos != std::string::npos && pos > 0) {
    std::vector<LexerInput> out;
    out.push_back(LexerInput::Chunk(delta_text.substr(0, pos)));
    out.insert(out.end(), results.begin(), results.end());
    return out;
  }
  if (pos == 0) {
    return results;
  }

  return rebuild_from_anchors(delta_text, results);
}

// token_id_scanner.py:223 (_join_decoded_text).
std::string TokenIDScanner::join_decoded_text(
    const std::vector<LexerInput>& results) const {
  std::string parts;
  for (const auto& item : results) parts += item.text;
  return parts;
}

// token_id_scanner.py:231 (_rebuild_from_anchors).
std::vector<LexerInput> TokenIDScanner::rebuild_from_anchors(
    const std::string& delta_text, std::vector<LexerInput>& results) {
  std::vector<LexerInput> anchors;
  for (const auto& item : results) {
    if (item.is_terminal()) anchors.push_back(item);
  }
  if (anchors.empty()) {
    return {LexerInput::Chunk(delta_text)};
  }

  // Resolve positions right-to-left with rfind, each match bounded to end
  // within search_end (Python delta_text.rfind(text, 0, search_end)).
  std::vector<long> positions(anchors.size(), -1);
  std::size_t search_end = delta_text.size();
  for (std::size_t k = anchors.size(); k-- > 0;) {
    const std::string& needle = anchors[k].text;
    if (search_end >= needle.size()) {
      std::size_t start = search_end - needle.size();
      std::size_t pos = delta_text.rfind(needle, start);
      if (pos != std::string::npos) {
        positions[k] = static_cast<long>(pos);
        search_end = pos;
      }
    }
  }

  std::vector<LexerInput> new_results;
  std::size_t consumed = 0;
  for (std::size_t k = 0; k < anchors.size(); ++k) {
    long pos = positions[k];
    if (pos >= 0 && static_cast<std::size_t>(pos) >= consumed) {
      if (static_cast<std::size_t>(pos) > consumed) {
        new_results.push_back(
            LexerInput::Chunk(delta_text.substr(consumed, pos - consumed)));
      }
      new_results.push_back(anchors[k]);
      consumed = static_cast<std::size_t>(pos) + anchors[k].text.size();
    } else {
      bool has_later_valid = false;
      for (std::size_t j = k + 1; j < positions.size(); ++j) {
        if (positions[j] >= 0) {
          has_later_valid = true;
          break;
        }
      }
      if (!has_later_valid && consumed < delta_text.size() &&
          anchors[k].terminal != kDropTerminal) {
        deferred_post_text_ += delta_text.substr(consumed);
        consumed = delta_text.size();
      }
      deferred_terminals_.push_back(anchors[k]);
    }
  }
  if (consumed < delta_text.size()) {
    new_results.push_back(LexerInput::Chunk(delta_text.substr(consumed)));
  }
  return new_results;
}

}  // namespace vllm::parser::engine
