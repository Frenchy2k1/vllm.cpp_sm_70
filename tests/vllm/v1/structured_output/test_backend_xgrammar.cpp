// Tests for the xgrammar structured-output backend (backend_xgrammar.{h,cpp} +
// xgrammar_json_schema.{h,cpp}) — the W1 brick of row TOOLS-XGRAMMAR. Ports the
// intent of upstream's JSON-schema structured-output cases
// (tests/v1/structured_output/ + tests/entrypoints/llm/test_struct_output_generate.py:214)
// to a MODEL-FREE unit gate: given a schema + a partial token sequence, the
// per-step bitmask allows EXACTLY the grammar-valid next tokens.
//
// THE PARITY POINT (vllm-feature-gap-analysis.md, TOOLS-XGRAMMAR): xgrammar emits
// object properties in DECLARATION order; the native path sorts keys
// lexicographically. The RED-first proof lives in-tree: on the SAME schema the
// native backend admits the sorted-first key's byte after `{"`, which the
// xgrammar (declaration-order) grammar FORBIDS — a key-sorting compiler produces
// a mask allowing an invalid-for-xgrammar token. This test pins the xgrammar
// backend to declaration order and asserts the two diverge exactly there.
//
// The tokenizer is a REAL byte-level BPE fixture (built via Tokenizer::FromHfJson
// like test_backend_native.cpp) so token->raw-bytes decoding is real.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/backend_xgrammar.h"
#include "vllm/v1/structured_output/xgrammar_json_schema.h"

using nlohmann::json;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::MakeStructuredOutputBackendFactory;
using vllm::v1::NativeStructuredOutputBackend;
using vllm::v1::ResolveStructuredOutputBackend;
using vllm::v1::StructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;
using vllm::v1::XgrammarJsonSchemaToEbnf;
using vllm::v1::XgrammarStructuredOutputBackend;

namespace {

using Ids = std::vector<int32_t>;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_xgrammar_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Token ids used across the JSON cases.
enum Tok : int32_t {
  kLBrace = 0,
  kRBrace = 1,
  kLBrack = 2,
  kRBrack = 3,
  kQuote = 4,
  kColon = 5,
  kComma = 6,
  kSpace = 7,
  kN = 8,
  kA = 9,
  kM = 10,
  kE = 11,
  kG = 12,
  kZero = 13,
  kOne = 14,
  kFive = 15,
  kT = 16,
  kR = 17,
  kU = 18,
  kF = 19,
  kL = 20,
  kS = 21,
  kX = 22,
  kDash = 23,
  kEos = 30,
};
constexpr int kJunk = 40;

// A real byte-level BPE tokenizer whose vocab covers JSON punctuation + the
// letters/digits the schemas need. Space is the byte-level-mapped 0x20 ("Ġ").
Tokenizer BuildFixture() {
  json vocab = {
      {"{", kLBrace}, {"}", kRBrace}, {"[", kLBrack}, {"]", kRBrack},
      {"\"", kQuote}, {":", kColon},  {",", kComma},  {"n", kN},
      {"a", kA},      {"m", kM},      {"e", kE},      {"g", kG},
      {"0", kZero},   {"1", kOne},    {"5", kFive},   {"t", kT},
      {"r", kR},      {"u", kU},      {"f", kF},      {"l", kL},
      {"s", kS},      {"x", kX},      {"-", kDash}};
  // Space 0x20 is NOT printable-identity under the byte-level alphabet.
  vocab[MapBytesToUnicode(" ")] = kSpace;
  // Junk tokens qN: never a valid JSON-structural prefix (the perf property).
  for (int i = 0; i < kJunk; ++i) {
    vocab["q" + std::to_string(i)] = 100 + i;
  }

  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kEos}, {"content", "<eos>"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

int VocabSize() { return static_cast<int>(Fixture().VocabSize()); }

// The gate schema: two REQUIRED properties in NON-alphabetical declaration order
// (name, age). Sorted order would be (age, name) — the parity divergence.
const char* kSchema =
    R"({"type":"object",)"
    R"("properties":{"name":{"type":"string"},"age":{"type":"integer"}},)"
    R"("required":["name","age"]})";

// Whether token id `t` is allowed by a freshly filled single-row bitmask for the
// grammar's current state.
bool Allowed(StructuredOutputGrammar& g, int32_t t) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
  return (word & (1 << (t & 31))) != 0;
}

}  // namespace

// (1) The seam: the xgrammar backend compiles a JSON schema and produces a
// per-step bitmask allowing EXACTLY the grammar-valid next tokens.
TEST_CASE("xgrammar JSON: bitmask allows exactly the grammar-valid next tokens") {
  XgrammarStructuredOutputBackend backend(Fixture(), VocabSize(),
                                          /*disable_any_whitespace=*/false,
                                          Ids{kEos});
  auto g = backend.compile_grammar(StructuredOutputOptions::kJson, kSchema);

  // At the start ONLY `{` may open the object (ws is between tokens, not before).
  CHECK(Allowed(*g, kLBrace));
  CHECK_FALSE(Allowed(*g, kLBrack));   // `[` — not an object
  CHECK_FALSE(Allowed(*g, kQuote));    // `"` — a key can't precede `{`
  CHECK_FALSE(Allowed(*g, kN));        // a bare letter
  CHECK_FALSE(Allowed(*g, kSpace));    // no leading whitespace
  CHECK_FALSE(Allowed(*g, kEos));      // not a complete value yet

  // After `{`: any-whitespace is ON, so either a whitespace token OR the opening
  // `"` of the first key is valid — nothing else.
  REQUIRE(g->accept_tokens("r", Ids{kLBrace}));
  CHECK(Allowed(*g, kSpace));
  CHECK(Allowed(*g, kQuote));
  CHECK_FALSE(Allowed(*g, kN));
  CHECK_FALSE(Allowed(*g, kLBrace));
  CHECK_FALSE(Allowed(*g, kRBrace));   // the object is not empty (name required)
}

// (2) THE KEY-ORDER PARITY + RED-first proof. After `{"` the xgrammar
// (declaration-order) grammar requires the FIRST declared key `name` => `n`
// allowed, `a` forbidden. The native backend sorts keys (age first) => it admits
// `a` and forbids `n`: a key-sorting compiler's mask allows a token xgrammar
// forbids. The two backends diverge EXACTLY at this byte.
TEST_CASE("xgrammar JSON: object keys keep DECLARATION order (vs native sort)") {
  const Ids open{kLBrace, kQuote};  // `{` then the opening key quote

  // xgrammar: declaration order (name, age).
  XgrammarStructuredOutputBackend xg(Fixture(), VocabSize(), false, Ids{kEos});
  auto gx = xg.compile_grammar(StructuredOutputOptions::kJson, kSchema);
  REQUIRE(gx->accept_tokens("r", open));
  CHECK(Allowed(*gx, kN));         // `name` first
  CHECK_FALSE(Allowed(*gx, kA));   // `age` is NOT the first key

  // native: lexicographic sort (age, name) — the RED-first witness that a
  // key-sorting compiler admits `a`, which the xgrammar grammar forbids.
  NativeStructuredOutputBackend nat(Fixture(), VocabSize(), Ids{kEos});
  auto gn = nat.compile_grammar(StructuredOutputOptions::kJson, kSchema);
  REQUIRE(gn->accept_tokens("r", open));
  CHECK(Allowed(*gn, kA));         // native emits `age` first
  CHECK_FALSE(Allowed(*gn, kN));
}

// (3) The any_whitespace flag: disable_any_whitespace collapses the `ws` rule, so
// after `{` NO whitespace token is allowed — only the opening key quote.
TEST_CASE("xgrammar JSON: disable_any_whitespace forbids inter-token whitespace") {
  XgrammarStructuredOutputBackend backend(Fixture(), VocabSize(),
                                          /*disable_any_whitespace=*/true,
                                          Ids{kEos});
  auto g = backend.compile_grammar(StructuredOutputOptions::kJson, kSchema);
  REQUIRE(g->accept_tokens("r", Ids{kLBrace}));
  CHECK(Allowed(*g, kQuote));
  CHECK_FALSE(Allowed(*g, kSpace));  // no flexible whitespace
}

// (4) json_object: {"type":"object"} => any JSON object. At the start only `{`.
TEST_CASE("xgrammar json_object: any object, opens with `{`") {
  XgrammarStructuredOutputBackend backend(Fixture(), VocabSize(), false,
                                          Ids{kEos});
  auto g = backend.compile_grammar(StructuredOutputOptions::kJsonObject, "");
  CHECK(Allowed(*g, kLBrace));
  CHECK_FALSE(Allowed(*g, kLBrack));
  CHECK_FALSE(Allowed(*g, kQuote));
}

// (5) The converter emits the xgrammar-faithful basic_* + ws rules and preserves
// property declaration order in the EBNF text itself.
TEST_CASE("xgrammar converter: EBNF carries basic_* rules and declaration order") {
  const std::string ebnf =
      XgrammarJsonSchemaToEbnf(kSchema, /*any_whitespace=*/true);
  CHECK(ebnf.find("basic_string_sub") != std::string::npos);
  CHECK(ebnf.find("basic_integer") != std::string::npos);
  CHECK(ebnf.find("ws ::= [ \\n\\t]*") != std::string::npos);
  // `name` appears before `age` in the emitted root/property rules.
  const std::size_t name_at = ebnf.find("name");
  const std::size_t age_at = ebnf.find("age");
  REQUIRE(name_at != std::string::npos);
  REQUIRE(age_at != std::string::npos);
  CHECK(name_at < age_at);

  // any_whitespace OFF => the `ws` rule collapses to the empty match.
  const std::string ebnf_off =
      XgrammarJsonSchemaToEbnf(kSchema, /*any_whitespace=*/false);
  CHECK(ebnf_off.find("ws ::= \"\"") != std::string::npos);
}

// (6) Backend selection mirrors vLLM: `auto` -> xgrammar; the factory builds the
// xgrammar backend, which constrains a JSON schema (native would sort keys).
TEST_CASE("xgrammar selection: `auto` resolves to xgrammar and constrains JSON") {
  CHECK(ResolveStructuredOutputBackend("auto") == "xgrammar");
  CHECK(ResolveStructuredOutputBackend("") == "xgrammar");
  CHECK(ResolveStructuredOutputBackend("xgrammar") == "xgrammar");
  CHECK(ResolveStructuredOutputBackend("guidance") == "native");
  CHECK_THROWS(ResolveStructuredOutputBackend("nonsense"));

  auto factory = MakeStructuredOutputBackendFactory("auto", Fixture(),
                                                    VocabSize(), false, Ids{kEos});
  std::unique_ptr<StructuredOutputBackend> b = factory();
  auto g = b->compile_grammar(StructuredOutputOptions::kJson, kSchema);
  REQUIRE(g->accept_tokens("r", Ids{kLBrace, kQuote}));
  CHECK(Allowed(*g, kN));        // declaration order => xgrammar was selected
  CHECK_FALSE(Allowed(*g, kA));
}
