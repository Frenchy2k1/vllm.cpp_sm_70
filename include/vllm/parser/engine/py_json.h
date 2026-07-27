// Ported helper for: json.dumps(obj, ensure_ascii=False) as used by
// vllm/parser/engine/parser_engine.py and vllm/parser/qwen3.py @ 555967922
// (vLLM 0.26.0.dev0).
//
// nlohmann's compact dump() emits no separators ("{\"a\":\"b\"}"), but CPython's
// json.dumps default separators are (", ", ": ") — a space after every comma AND
// colon. The parser-assembly gate compares the exact serialized argument string
// byte-for-byte against the upstream oracle, so we must reproduce that spacing.
// This inline dumper walks an nlohmann::ordered_json (insertion-ordered, mirroring
// Python dict order) and formats scalars via nlohmann (whose string escaping and
// int/bool/null forms already match json.dumps for the ASCII/UTF-8 values the
// converters produce), while emitting Python's object/array separators itself.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace vllm::parser::engine {

inline void python_json_dump_into(const nlohmann::ordered_json& j,
                                   std::string& out) {
  if (j.is_object()) {
    out += '{';
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!first) out += ", ";
      first = false;
      out += nlohmann::json(it.key()).dump();  // escaped key literal
      out += ": ";
      python_json_dump_into(it.value(), out);
    }
    out += '}';
  } else if (j.is_array()) {
    out += '[';
    bool first = true;
    for (const auto& e : j) {
      if (!first) out += ", ";
      first = false;
      python_json_dump_into(e, out);
    }
    out += ']';
  } else {
    // Scalars: string (escaped), number, bool, null — nlohmann matches
    // json.dumps for the values the parser converters emit.
    out += j.dump();
  }
}

// json.dumps(j, ensure_ascii=False) with default (", ", ": ") separators.
inline std::string python_json_dumps(const nlohmann::ordered_json& j) {
  std::string out;
  python_json_dump_into(j, out);
  return out;
}

}  // namespace vllm::parser::engine
