// SGLANG-DISTINCT SW3 — jump-forward decoding, the SAFE (token-unique) subset.
// Gate: generation WITH jump-forward is BYTE-IDENTICAL (same token IDs) to
// generation WITHOUT it (pure per-token constrained decode) — jump-forward is a
// SPEED optimization, not a behavior change. Also gates: (a) the jump actually
// FIRED (fewer model steps over a forced span), (b) inert when no forced span
// exists, (c) RED-first — a deliberately-WRONG jump that emits a canonical
// re-tokenization of a boundary-ambiguous forced byte run produces DIFFERENT
// tokens than per-token decode, and our SAFE driver refuses to jump that span.
//
// Reuses the real byte-level BPE fixture shape from test_backend_native.cpp so
// token->raw-bytes decoding + the trie are real. A tiny reference decoder stands
// in for the model: a fixed per-token preference over the vocab; the constrained
// sampler masks to grammar-valid tokens and takes the argmax by preference — so
// when >=2 tokens are valid the "model" chooses, exactly the case jump-forward
// must not silently override.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/jump_forward.h"

using nlohmann::json;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::DrainForcedTokens;
using vllm::v1::NativeGrammar;
using vllm::v1::NativeStructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;

namespace {

using Ids = std::vector<int32_t>;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_jump_forward_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

constexpr int32_t kEos = 40;
constexpr int kJunk = 50;

// A byte-level BPE fixture. ASCII single/merged tokens hand-picked for the
// grammars below: "!"(31) has no "!!"-superstring so a run of '!' is
// token-unique (jumpable); "a"(10)/"b"(14)/"ab"(27) make "ab" a
// boundary-AMBIGUOUS byte run (["a","b"] vs ["ab"]) — NOT token-unique.
Tokenizer BuildFixture() {
  json vocab = {
      {"y", 0},   {"e", 1},   {"s", 2},   {"n", 3},    {"o", 4},
      {"ye", 5},  {"es", 6},  {"yes", 7}, {"no", 8},   {"c", 9},
      {"a", 10},  {"t", 11},  {"d", 12},  {"g", 13},   {"b", 14},
      {"i", 15},  {"r", 16},  {"cat", 17},{"dog", 18}, {"bird", 19},
      {"x", 20},  {"z", 21},  {"0", 22},  {"1", 23},   {"5", 24},
      {"9", 25},  {"12", 26}, {"ab", 27}, {"!", 31}};
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

std::unique_ptr<NativeStructuredOutputBackend> MakeBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
}

std::unique_ptr<NativeGrammar> Compile(NativeStructuredOutputBackend& backend,
                                       StructuredOutputOptions type,
                                       const std::string& spec) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(type, spec);
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

// The set of grammar-valid token ids at the current state (via fill_bitmask).
Ids AllowedTokens(NativeGrammar& g) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  Ids out;
  for (int32_t t = 0; t < VocabSize(); ++t) {
    const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
    if ((word & (1 << (t & 31))) != 0) out.push_back(t);
  }
  return out;
}

// The "model": a preference score per token id. The constrained sampler picks
// the allowed token with the highest score (ties -> smallest id). This models
// argmax over the masked logits and is INDEPENDENT of grammar (a real model).
struct MockModel {
  std::vector<int> score;  // indexed by token id
  explicit MockModel(int vocab) : score(vocab, 0) {}
  int32_t argmax(const Ids& allowed) const {
    int32_t best = -1;
    int best_score = 0;
    for (const int32_t t : allowed) {
      const int s = score[static_cast<std::size_t>(t)];
      if (best < 0 || s > best_score) {
        best = t;
        best_score = s;
      }
    }
    return best;
  }
};

struct DecodeResult {
  Ids tokens;
  int model_steps = 0;
};

// A minimal reference decode loop. jump=true drains the token-unique forced run
// (no model step) before each model step; jump=false is pure per-token
// constrained decode.
DecodeResult Decode(NativeGrammar& g, const MockModel& model, bool jump,
                    int max_len) {
  DecodeResult r;
  const std::string rid = "r";
  while (static_cast<int>(r.tokens.size()) < max_len) {
    if (jump) {
      const int budget = max_len - static_cast<int>(r.tokens.size());
      Ids forced;
      DrainForcedTokens(g, rid, forced, /*enabled=*/true, budget);
      for (const int32_t t : forced) r.tokens.push_back(t);
    }
    if (g.is_terminated()) break;
    const Ids allowed = AllowedTokens(g);
    if (allowed.empty()) break;
    const int32_t tok = model.argmax(allowed);
    if (tok < 0) break;
    ++r.model_steps;
    const bool ok = g.accept_tokens(rid, {tok});
    REQUIRE(ok);
    r.tokens.push_back(tok);
    if (tok == kEos) break;
  }
  return r;
}

}  // namespace

// ── forced_token: the detection hook ────────────────────────────────────────
TEST_CASE("forced_token: unique non-accepting token is forced; ambiguous/none not") {
  auto backend = MakeBackend();

  SUBCASE("token-unique run '!!!' is forced at every step") {
    auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                     R"(root ::= "!" "!" "!")");
    CHECK(g->forced_token() == std::optional<int32_t>(31));  // "!"
    REQUIRE(g->accept_tokens("r", {31}));
    CHECK(g->forced_token() == std::optional<int32_t>(31));
    REQUIRE(g->accept_tokens("r", {31}));
    CHECK(g->forced_token() == std::optional<int32_t>(31));
    REQUIRE(g->accept_tokens("r", {31}));
    // Fully matched -> accepting -> EOS is the alternative -> NOT forced.
    CHECK_FALSE(g->forced_token().has_value());
  }

  SUBCASE("boundary-ambiguous byte run 'ab' is NOT forced (2 tokenizations)") {
    auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                     R"(root ::= "ab")");
    // Byte run "ab" is deterministic, but "a"(10) and "ab"(27) are both valid
    // -> forced_token nullopt -> safe subset falls back to normal decode.
    CHECK_FALSE(g->forced_token().has_value());
    // After committing "a", only "b"(14) is valid and the state is
    // non-accepting -> the TAIL is token-unique/forced.
    REQUIRE(g->accept_tokens("r", {10}));
    CHECK(g->forced_token() == std::optional<int32_t>(14));  // "b"
  }

  SUBCASE("multi-choice state is not forced") {
    auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                     R"(root ::= "yes" | "no")");
    CHECK_FALSE(g->forced_token().has_value());  // y/ye/yes/n/no all valid
  }
}

// ── (1) OUTPUT-IDENTITY + jump FIRED over a forced span ─────────────────────
TEST_CASE("jump-forward: forced span is byte-identical to per-token decode, fewer steps") {
  auto backend = MakeBackend();
  MockModel model(VocabSize());
  // Prefer digit "5"(24) so the free tail is deterministic across both runs.
  model.score[24] = 100;
  model.score[22] = 10;  // "0"

  // 3 forced '!' then a free single digit [0-9].
  const char* kSpec = R"(root ::= "!" "!" "!" [0-9])";

  auto g_ref = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const DecodeResult ref = Decode(*g_ref, model, /*jump=*/false, 16);

  auto g_jf = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const DecodeResult jf = Decode(*g_jf, model, /*jump=*/true, 16);

  // BYTE-IDENTICAL token IDs — jump-forward changed nothing but the step count.
  CHECK(jf.tokens == ref.tokens);
  CHECK(ref.tokens == Ids{31, 31, 31, 24});  // "!","!","!","5"

  // The jump FIRED: per-token spent 4 model steps; jump spent 1 (the free digit
  // only) — the 3 forced '!' were emitted with zero model steps.
  CHECK(ref.model_steps == 4);
  CHECK(jf.model_steps == 1);
  CHECK(jf.model_steps < ref.model_steps);
}

// ── (2) INERT when there is no forced span ──────────────────────────────────
TEST_CASE("jump-forward: inert (identical steps + tokens) when no forced span") {
  auto backend = MakeBackend();
  MockModel model(VocabSize());
  model.score[7] = 100;  // "yes"

  const char* kSpec = R"(root ::= "yes" | "no")";  // start = 5-way choice

  auto g_ref = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const DecodeResult ref = Decode(*g_ref, model, /*jump=*/false, 16);

  auto g_jf = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const DecodeResult jf = Decode(*g_jf, model, /*jump=*/true, 16);

  CHECK(jf.tokens == ref.tokens);
  CHECK(ref.tokens == Ids{7});          // model chose "yes"
  CHECK(jf.model_steps == ref.model_steps);  // no step elided => inert
  CHECK(jf.model_steps == 1);
}

// ── default-off: DrainForcedTokens does nothing, grammar untouched ──────────
TEST_CASE("jump-forward: disabled drains nothing and leaves the grammar untouched") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "!" "!" "!")");
  Ids out;
  const int n = DrainForcedTokens(*g, "r", out, /*enabled=*/false);
  CHECK(n == 0);
  CHECK(out.empty());
  // The forced run is still fully available -> the grammar state was untouched.
  CHECK(g->forced_token() == std::optional<int32_t>(31));
}

// ── (3) RED-first: a WRONG jump (canonical re-tokenization of a boundary-
//        ambiguous forced byte run, no rollback) CHANGES the tokens; the SAFE
//        driver refuses to jump it and stays byte-identical. ──────────────────
namespace {

// A DELIBERATELY-WRONG jump modelling the re-tokenization hazard: greedily emit
// the LONGEST-matching grammar-valid token at each step (SGLang-style canonical
// re-tokenization favours the longest piece), blind to the fact that the model
// would choose a different (shorter) tokenization — and with NO boundary
// rollback. This is exactly what the safe subset must NOT do.
Ids BadJumpLongestMatch(NativeGrammar& g, const Tokenizer& tok) {
  Ids emitted;
  const std::string rid = "bad";
  for (;;) {
    const Ids allowed = AllowedTokens(g);
    int32_t best = -1;
    std::size_t best_len = 0;
    for (const int32_t t : allowed) {
      if (t == kEos) continue;
      const std::size_t len = tok.TokenText(t).size();
      if (best < 0 || len > best_len) {
        best = t;
        best_len = len;
      }
    }
    if (best < 0) break;
    if (!g.accept_tokens(rid, {best})) break;
    emitted.push_back(best);
    if (g.is_terminated()) break;  // only the forced head, then stop.
  }
  return emitted;
}

}  // namespace

TEST_CASE("jump-forward RED: naive longest-match re-tokenization changes tokens; safe subset does not") {
  auto backend = MakeBackend();
  MockModel model(VocabSize());
  // The model PREFERS the short piece "a"(10) over the merged "ab"(27).
  model.score[10] = 100;  // "a"
  model.score[27] = 1;    // "ab"
  model.score[14] = 50;   // "b"

  const char* kSpec = R"(root ::= "ab")";

  // Per-token constrained decode: step1 allowed {a,ab} -> argmax "a"; step2
  // allowed {b} -> "b". Tokens [10,14], 2 model steps.
  auto g_ref = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const DecodeResult ref = Decode(*g_ref, model, /*jump=*/false, 16);
  CHECK(ref.tokens == Ids{10, 14});
  CHECK(ref.model_steps == 2);

  // WRONG jump: greedy longest-match re-tokenizes "ab" as the single token "ab"
  // (27) — a DIFFERENT token sequence than per-token decode would emit. RED.
  auto g_bad = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const Ids bad = BadJumpLongestMatch(*g_bad, Fixture());
  CHECK(bad == Ids{27});          // ["ab"]
  CHECK(bad != ref.tokens);       // RED: a naive jump CHANGES the output tokens.

  // SAFE jump (our driver): refuses the ambiguous head (forced_token nullopt at
  // start), lets the model pick "a", then jumps the now-unique tail "b". Tokens
  // are byte-identical to per-token decode AND a step is still elided.
  auto g_safe = Compile(*backend, StructuredOutputOptions::kGrammar, kSpec);
  const DecodeResult safe = Decode(*g_safe, model, /*jump=*/true, 16);
  CHECK(safe.tokens == ref.tokens);   // [10,14] — byte-identical, no token change
  CHECK(safe.model_steps == 1);       // fired: "b" jumped
  CHECK(safe.model_steps < ref.model_steps);
}
