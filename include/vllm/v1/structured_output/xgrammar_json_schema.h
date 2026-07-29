// Ported (SEMANTICS) from: xgrammar cpp/json_schema_converter.cc @
// mlc-ai/xgrammar a32ac892676d2eedc0327416105b9b06edfb94b2 — the JSON-Schema ->
// EBNF converter vLLM's default `auto`/`xgrammar` structured-output backend uses
// (via xgr.GrammarCompiler.compile_json_schema, backend_xgrammar.py:82). See
// .agents/specs/xgrammar-backend.md and porting-inventory §9 for the decision to
// mirror xgrammar's algorithm PORTABLY rather than vendor its C++ library.
//
// WHY a SECOND JSON->grammar lowering (vs json_schema_to_gbnf.h): xgrammar and
// our native lowering DIVERGE exactly where the feature-gap analysis flags a
// parity miss (vllm-feature-gap-analysis.md, TOOLS-XGRAMMAR row):
//   - KEY ORDER: xgrammar emits object properties in the schema's DECLARATION
//     order; our native path parses into nlohmann::json which SORTS keys
//     lexicographically (json_schema_to_gbnf.h DEVIATIONS note). This converter
//     parses with nlohmann::ordered_json so declaration order is preserved,
//     matching xgrammar 1:1.
//   - WHITESPACE: xgrammar's `any_whitespace` inserts a `[ \n\t]*` rule between
//     every structural token (compile_json_schema(any_whitespace=...),
//     backend_xgrammar.py:83; the flag is `not disable_any_whitespace`, the vLLM
//     structured_outputs_config default => any-whitespace ON).
//   - The `basic_*` rule set (basic_number/basic_string/basic_boolean/basic_null/
//     basic_array/basic_object/basic_any + basic_escape/basic_string_sub) is
//     emitted VERBATIM as xgrammar's json_schema_converter.cc does, so exotic /
//     free-form JSON values constrain identically.
//
// OUTPUT: an EBNF/GBNF grammar STRING with a `root` rule, in the dialect the
// native GbnfParser (backend_native.cpp) already accepts (literals, char classes
// incl. negated + `\xHH` ranges, `* + ? {m,n}`, alternation, groups, rule refs).
// The native push-down FSM matcher + token-byte trie THEN produce the per-step
// token bitmask — that matcher IS xgrammar's algorithm (grammar -> pushdown
// automaton -> per-step token bitmask), so the xgrammar backend reuses it and
// this file supplies only the xgrammar-faithful FRONT-END.
//
// SUPPORTED SUBSET (W1 — correctness-grade; throws loudly on anything outside it
// so the emitted grammar NEVER accepts schema-invalid JSON):
//   - type: object / array / string / number / integer / boolean / null; `type`
//     as an array => a union alternation of the per-type grammars;
//   - object `properties` where every declared key is `required` (declaration
//     order preserved), nested recursively; a property-less object => basic_object;
//   - array `items` (homogeneous), item-less array => basic_array;
//   - `enum` (alternation of the compact-serialized values), `const` (the single
//     compact-serialized value);
//   - `anyOf` / `oneOf` => alternation of the sub-schemas.
//
// W1 DEVIATIONS / LIMITS (recorded; grammar stays a strict schema subset):
//   - OPTIONAL object properties (a `properties` key not in `required`) THROW at
//     W1 — xgrammar's comma/optional-permutation emission is W2 (see the spec's
//     Work breakdown). This is the "throw rather than mis-constrain" contract.
//   - `disable_any_whitespace` mode collapses `ws` to the empty match at W1 (the
//     strict compact-indent separators xgrammar emits are a W2 residual). The
//     default (any-whitespace ON) is fully faithful.
//   - string `pattern`/`format`, numeric bounds/`multipleOf`, array length,
//     `$ref`/`allOf`/`not`, `additionalProperties` schemas: not constrained /
//     throw (mirrors the native path's documented subset).
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace vllm::v1 {

// Lower a JSON Schema (parsed as nlohmann::json; internally re-read as
// ordered_json off `raw_schema_text` when key order matters) into an xgrammar-
// faithful EBNF grammar STRING with a `root` rule matching exactly the JSON
// conforming to the schema. `any_whitespace` mirrors xgrammar's
// compile_json_schema(any_whitespace=...) (vLLM default: true). Throws
// std::runtime_error on an unsupported construct (never emits a grammar that
// would accept schema-invalid JSON for the supported subset).
//
// `raw_schema_text` is the ORIGINAL schema JSON text (as stored by
// get_structured_output_key). It is re-parsed with nlohmann::ordered_json so
// object `properties` keep their DECLARATION order (the xgrammar parity point).
// Pass the same string the backend received.
std::string XgrammarJsonSchemaToEbnf(const std::string& raw_schema_text,
                                     bool any_whitespace);

// The permissive any-JSON grammar xgrammar compiles for `{"type": "object"}` /
// response_format `json_object` (backend_xgrammar.py:85-88): a `root` matching
// any well-formed JSON object. `any_whitespace` as above.
std::string XgrammarJsonObjectEbnf(bool any_whitespace);

}  // namespace vllm::v1
