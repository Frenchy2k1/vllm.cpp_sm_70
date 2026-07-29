// Ported (SEMANTICS) from: xgrammar cpp/json_schema_converter.cc @
// mlc-ai/xgrammar a32ac892676d2eedc0327416105b9b06edfb94b2. See
// include/vllm/v1/structured_output/xgrammar_json_schema.h for the contract,
// the parity rationale (key order + any_whitespace + basic_* rules), and the W1
// supported subset / deviations. Emits an EBNF/GBNF grammar string the native
// GbnfParser (backend_native.cpp) accepts; the native push-down FSM + token-byte
// trie then produce the per-step token bitmask (xgrammar's algorithm, reused).
#include "vllm/v1/structured_output/xgrammar_json_schema.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::v1 {
namespace {

// ordered_json preserves object insertion (declaration) order — THE xgrammar
// key-order parity point (vs nlohmann::json's lexicographic sort used by the
// native json_schema_to_gbnf path).
using ojson = nlohmann::ordered_json;

// A GBNF string literal matching the exact bytes `s` (xgrammar emits key/const
// literals verbatim). Escapes what GbnfParser's ParseStringLiteral needs.
std::string GbnfLiteral(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  out += "\"";
  return out;
}

// The fixed xgrammar `basic_*` rule set + the `ws` whitespace rule. Emitted
// VERBATIM as json_schema_converter.cc's kBasicRules (so free-form JSON values
// constrain identically). `ws` realizes compile_json_schema(any_whitespace=...):
// ON => `[ \n\t]*` between every structural token; OFF => empty at W1 (the
// strict compact separators are a W2 residual, per the header).
std::string BasicRules(bool any_whitespace) {
  std::string rules = R"EBNF(basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= "\"" | [^"\\\x00-\x1F] basic_string_sub | "\\" basic_escape basic_string_sub
basic_string ::= "\"" basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_integer ::= "0" | "-"? [1-9] [0-9]*
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_array ::= "[" ws basic_any (ws "," ws basic_any)* ws "]" | "[" ws "]"
basic_object ::= "{" ws basic_string ws ":" ws basic_any (ws "," ws basic_string ws ":" ws basic_any)* ws "}" | "{" ws "}"
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
)EBNF";
  rules += any_whitespace ? "ws ::= [ \\n\\t]*\n" : "ws ::= \"\"\n";
  return rules;
}

// Walks a (parsed) JSON schema, emitting named sub-rules for objects/arrays and
// returning an EXPRESSION (a rule ref or an inline `( ... )` group) for the root
// and each nested schema. Mirrors json_schema_converter.cc's recursion; W1
// subset per the header (throws on anything outside it).
class Converter {
 public:
  std::string Convert(const ojson& schema) {
    const std::string root_expr = ExprFor(schema);
    std::string out = BasicRules(any_whitespace_was_);
    for (const std::string& r : rules_) out += r;
    out += "root ::= " + root_expr + "\n";
    return out;
  }

  explicit Converter(bool any_whitespace)
      : any_whitespace_was_(any_whitespace) {}

 private:
  std::string Fresh() { return "gen_" + std::to_string(counter_++); }

  void AddRule(const std::string& name, const std::string& body) {
    rules_.push_back(name + " ::= " + body + "\n");
  }

  // The expression for a schema value.
  std::string ExprFor(const ojson& schema) {
    // Boolean schema `true` (or {}) => any JSON value.
    if (schema.is_boolean()) {
      if (schema.get<bool>()) return "basic_any";
      throw std::runtime_error(
          "xgrammar json: a `false` schema matches nothing (unsupported)");
    }
    if (!schema.is_object()) {
      throw std::runtime_error("xgrammar json: schema must be an object or bool");
    }

    // const / enum take precedence (a literal set).
    if (schema.contains("const")) {
      return GbnfLiteral(schema["const"].dump());
    }
    if (schema.contains("enum")) {
      const ojson& e = schema["enum"];
      if (!e.is_array() || e.empty()) {
        throw std::runtime_error("xgrammar json: `enum` must be a non-empty array");
      }
      std::string body = "(";
      bool first = true;
      for (const ojson& v : e) {
        if (!first) body += " | ";
        first = false;
        body += GbnfLiteral(v.dump());
      }
      body += ")";
      return body;
    }

    // anyOf / oneOf => alternation of the sub-schemas.
    for (const char* key : {"anyOf", "oneOf"}) {
      if (schema.contains(key)) {
        const ojson& arr = schema[key];
        if (!arr.is_array() || arr.empty()) {
          throw std::runtime_error(
              std::string("xgrammar json: `") + key +
              "` must be a non-empty array");
        }
        std::string body = "(";
        bool first = true;
        for (const ojson& sub : arr) {
          if (!first) body += " | ";
          first = false;
          body += ExprFor(sub);
        }
        body += ")";
        return body;
      }
    }

    // Unsupported combinators — throw rather than mis-constrain (W1).
    for (const char* key : {"$ref", "allOf", "not"}) {
      if (schema.contains(key)) {
        throw std::runtime_error(std::string("xgrammar json: `") + key +
                                 "` is unsupported at W1");
      }
    }

    if (!schema.contains("type")) {
      // No type and no combinator/const/enum => any JSON value.
      return "basic_any";
    }

    const ojson& type = schema["type"];
    if (type.is_string()) {
      return ExprForType(type.get<std::string>(), schema);
    }
    if (type.is_array()) {
      // A union of per-type grammars.
      std::string body = "(";
      bool first = true;
      for (const ojson& t : type) {
        if (!first) body += " | ";
        first = false;
        body += ExprForType(t.get<std::string>(), schema);
      }
      body += ")";
      return body;
    }
    throw std::runtime_error("xgrammar json: `type` must be a string or array");
  }

  std::string ExprForType(const std::string& type, const ojson& schema) {
    if (type == "string") return "basic_string";
    if (type == "integer") return "basic_integer";
    if (type == "number") return "basic_number";
    if (type == "boolean") return "basic_boolean";
    if (type == "null") return "basic_null";
    if (type == "array") return ArrayRule(schema);
    if (type == "object") return ObjectRule(schema);
    throw std::runtime_error("xgrammar json: unsupported type '" + type + "'");
  }

  std::string ArrayRule(const ojson& schema) {
    if (!schema.contains("items")) return "basic_array";
    const std::string elem = ExprFor(schema["items"]);
    const std::string name = Fresh();
    // xgrammar-faithful: elements separated by ws "," ws; empty array allowed.
    const std::string body = "\"[\" ws " + elem + " (ws \",\" ws " + elem +
                             ")* ws \"]\" | \"[\" ws \"]\"";
    AddRule(name, body);
    return name;
  }

  std::string ObjectRule(const ojson& schema) {
    if (!schema.contains("properties") || !schema["properties"].is_object() ||
        schema["properties"].empty()) {
      return "basic_object";
    }
    const ojson& props = schema["properties"];
    std::unordered_set<std::string> required;
    if (schema.contains("required") && schema["required"].is_array()) {
      for (const ojson& r : schema["required"]) {
        required.insert(r.get<std::string>());
      }
    }
    // W1: every declared property must be required (declaration order). Optional
    // properties (the comma/permutation emission) are a W2 residual.
    std::string body = "\"{\" ws ";
    bool first = true;
    for (auto it = props.begin(); it != props.end(); ++it) {
      const std::string& key = it.key();
      if (required.find(key) == required.end()) {
        throw std::runtime_error(
            "xgrammar json: optional property '" + key +
            "' is unsupported at W1 (all properties must be `required`)");
      }
      if (!first) body += " ws \",\" ws ";
      first = false;
      body += GbnfLiteral(ojson(key).dump());  // the quoted key literal
      body += " ws \":\" ws ";
      body += ExprFor(it.value());
    }
    body += " ws \"}\"";
    const std::string name = Fresh();
    AddRule(name, body);
    return name;
  }

  bool any_whitespace_was_;
  int counter_ = 0;
  std::vector<std::string> rules_;
};

}  // namespace

std::string XgrammarJsonSchemaToEbnf(const std::string& raw_schema_text,
                                     bool any_whitespace) {
  ojson schema;
  try {
    schema = ojson::parse(raw_schema_text);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("xgrammar json: schema is not valid JSON: ") + e.what());
  }
  return Converter(any_whitespace).Convert(schema);
}

std::string XgrammarJsonObjectEbnf(bool any_whitespace) {
  // {"type": "object"} => any JSON object (backend_xgrammar.py:85-88).
  return BasicRules(any_whitespace) + "root ::= basic_object\n";
}

}  // namespace vllm::v1
