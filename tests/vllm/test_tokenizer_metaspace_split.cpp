// Metaspace `split: true` + the Metaspace DECODER in vllm::Tokenizer —
// ARCH-ONE-SURFACE ROW 1 (Parakeet ASR fold), W1(a).
//
// Ported from HF tokenizers 0.22:
//   - pre_tokenizers/metaspace.rs `Metaspace::pre_tokenize` + its test module
//     (`basic`, `multiple_spaces`): with `split: true` the normalized string is
//     split at every replacement occurrence with SplitDelimiterBehavior::
//     MergedWithNext (the ▁ starts a NEW pretoken, attached to what follows),
//     so BPE merges can never cross a ▁ boundary.
//   - decoders (metaspace.rs `decode_chain` + the decoder test): inside the
//     FIRST token every replacement char is DROPPED (when prepend_scheme !=
//     "never"), inside every later token it becomes ONE space; tokens are
//     concatenated. No ByteFallback / Fuse / Strip — the bare Metaspace decoder
//     is NOT the Mistral/Gemma Sequence chain.
//
// Why now: every published Parakeet checkpoint ships `{"type": "Metaspace",
// "replacement": "▁", "prepend_scheme": "always", "split": true}` as BOTH
// pre_tokenizer and decoder. `Tokenizer::FromHfJson` used to refuse split=true
// ("no golden in scope"), which is exactly why examples/parakeet_transcribe
// grew a private LoadVocab + DecodeIds pair. This suite is the golden that
// guard asked for; the reference arm below replicates the pre-refactor
// DecodeIds byte for byte and pins the library equal to it on the committed
// pre-refactor transcript goldens.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/tokenizer/tokenizer.h"

namespace {

using vllm::tok::Tokenizer;

// Write `body` to a fresh temp tokenizer.json and load it.
Tokenizer FromJson(const std::string& body) {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("metaspace_split_tok_" + std::to_string(counter++) + ".json"))
          .string();
  {
    std::ofstream out(path, std::ios::binary);
    out << body;
  }
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

// A minimal SentencePiece-family tokenizer.json. `split_flag` goes into the
// Metaspace pre_tokenizer; `decoder` (when non-empty) is spliced verbatim.
std::string MakeJson(const std::string& vocab, const std::string& merges,
                     bool split_flag, const std::string& decoder,
                     const std::string& prepend = "always") {
  std::string s = "{";
  s += "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\","
       "\"prepend_scheme\":\"" + prepend + "\",\"split\":";
  s += split_flag ? "true" : "false";
  s += "},";
  if (!decoder.empty()) s += "\"decoder\":" + decoder + ",";
  s += "\"model\":{\"type\":\"BPE\",\"vocab\":" + vocab +
       ",\"merges\":" + merges + "}}";
  return s;
}

const std::string kMetaspaceDecoder =
    "{\"type\":\"Metaspace\",\"replacement\":\"▁\","
    "\"prepend_scheme\":\"always\",\"split\":true}";

}  // namespace

TEST_CASE("metaspace split=true is accepted (the old guard is gone)") {
  // Pre-change this threw `Metaspace split=true unsupported (no golden in
  // scope)`; the Parakeet fold implements the rule instead of refusing.
  const Tokenizer tok = FromJson(MakeJson(
      R"({"▁":0,"a":1,"▁a":2})", "[]", /*split=*/true, ""));
  CHECK(tok.IsSentencePiece());
}

TEST_CASE("split=true pre-splits at the replacement: merges cannot cross") {
  // tokenizers 0.22 pre_tokenizers/metaspace.rs `basic`, expressed at the
  // encode observable: "a a" normalizes to ▁a▁a; MergedWithNext yields the
  // pretokens [▁a][▁a], so the ▁a+▁a merge can NEVER apply under split=true
  // and MUST apply under split=false. Same file otherwise.
  const std::string vocab =
      R"({"▁":0,"a":1,"▁a":2,"▁a▁a":3})";
  const std::string merges =
      R"([["▁","a"],["▁a","▁a"]])";
  const Tokenizer split_true =
      FromJson(MakeJson(vocab, merges, /*split=*/true, ""));
  CHECK(split_true.Encode("a a") == std::vector<int32_t>{2, 2});
  const Tokenizer split_false =
      FromJson(MakeJson(vocab, merges, /*split=*/false, ""));
  CHECK(split_false.Encode("a a") == std::vector<int32_t>{3});
}

TEST_CASE("split=true keeps consecutive spaces as separate ▁ pretokens") {
  // tokenizers 0.22 metaspace.rs `multiple_spaces`: "a  a" -> ▁a ▁ ▁a.
  const std::string vocab =
      R"({"▁":0,"a":1,"▁a":2,"▁▁":3})";
  const std::string merges =
      R"([["▁","a"],["▁","▁"]])";
  const Tokenizer tok = FromJson(MakeJson(vocab, merges, /*split=*/true, ""));
  // Under split=true the ▁+▁ merge cannot apply either (each ▁ starts its own
  // pretoken), so the middle space stays a lone ▁ piece.
  CHECK(tok.Encode("a  a") == std::vector<int32_t>{2, 0, 2});
}

TEST_CASE("Metaspace decoder: first piece drops ▁, later pieces map ▁ to space") {
  // tokenizers 0.22 decoders test: decode_chain(["▁Hey", "▁friend!"]) ==
  // ["Hey", " friend!"], i.e. "Hey friend!" concatenated.
  const std::string vocab =
      R"({"▁Hey":0,"▁friend!":1,"▁▁x":2})";
  const Tokenizer tok =
      FromJson(MakeJson(vocab, "[]", /*split=*/true, kMetaspaceDecoder));
  CHECK(tok.Decode({0, 1}) == "Hey friend!");
  CHECK(tok.Decode({1}) == "friend!");
  // EVERY replacement inside the first piece is dropped, not just a leading
  // one — that is the flat_map rule, and it is what distinguishes the
  // Metaspace decoder from the Sequence chain's Strip(1 leading space).
  CHECK(tok.Decode({2}) == "x");
  CHECK(tok.Decode({0, 2}) == "Hey  x");
  CHECK(tok.Decode({}).empty());
}

TEST_CASE("Metaspace decoder with prepend_scheme=never keeps the space") {
  const std::string never_decoder =
      "{\"type\":\"Metaspace\",\"replacement\":\"▁\","
      "\"prepend_scheme\":\"never\",\"split\":true}";
  const std::string vocab = R"({"▁Hey":0,"▁friend!":1})";
  const Tokenizer tok = FromJson(
      MakeJson(vocab, "[]", /*split=*/true, never_decoder, "never"));
  CHECK(tok.Decode({0, 1}) == " Hey friend!");
}

TEST_CASE("no Metaspace decoder node: the Sequence chain is unchanged") {
  // Regression pin for Mistral/Gemma: a file WITHOUT a Metaspace decoder keeps
  // the Sequence decoder (Replace -> ByteFallback -> Fuse -> Strip ONE leading
  // space). "▁▁x" is the distinguishing probe: Sequence gives " x" (two spaces,
  // one stripped), the Metaspace decoder gives "x" (both dropped in piece 0).
  const std::string vocab = R"({"▁Hey":0,"▁▁x":1})";
  const Tokenizer tok = FromJson(MakeJson(vocab, "[]", /*split=*/false, ""));
  CHECK(tok.Decode({1}) == " x");
  CHECK(tok.Decode({0}) == "Hey");
}

// ── the Parakeet fixture: the library now decodes what the example decoded ──

#include <nlohmann/json.hpp>

namespace {

// The PRE-refactor examples/parakeet_transcribe reference, replicated byte for
// byte (main.cpp:104-159 @ f98e1e48, LoadVocab + DecodeIds): id -> piece from
// model.vocab + added_tokens, then the Metaspace decode_chain rule applied by
// hand — exactly the code the split=true guard forced the example to carry.
std::map<int32_t, std::string> RefLoadVocabJson(const std::string& dir) {
  std::map<int32_t, std::string> vocab;
  std::ifstream f(dir + "/tokenizer.json", std::ios::binary);
  REQUIRE(f.good());
  nlohmann::json doc;
  f >> doc;
  const auto model = doc.find("model");
  if (model != doc.end()) {
    const auto v = model->find("vocab");
    if (v != model->end() && v->is_object()) {
      for (auto it = v->begin(); it != v->end(); ++it) {
        vocab[it.value().get<int32_t>()] = it.key();
      }
    }
  }
  const auto added = doc.find("added_tokens");
  if (added != doc.end() && added->is_array()) {
    for (const auto& t : *added) {
      vocab[t.at("id").get<int32_t>()] = t.at("content").get<std::string>();
    }
  }
  return vocab;
}

std::string RefDecodeIds(const std::vector<int32_t>& ids,
                         const std::map<int32_t, std::string>& vocab) {
  static const std::string kReplacement = "\xe2\x96\x81";  // U+2581
  std::string text;
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto it = vocab.find(ids[i]);
    if (it == vocab.end()) continue;
    const std::string& piece = it->second;
    for (size_t p = 0; p < piece.size();) {
      if (piece.compare(p, kReplacement.size(), kReplacement) == 0) {
        if (i != 0) text.push_back(' ');
        p += kReplacement.size();
      } else {
        text.push_back(piece[p]);
        ++p;
      }
    }
  }
  return text;
}

std::string FixtureDir() { return std::string(PARAKEET_E2E_FIXTURE_DIR); }

}  // namespace

TEST_CASE("parakeet fixture tokenizer: Tokenizer::Decode == pre-refactor DecodeIds") {
  for (const char* head : {"ctc", "rnnt"}) {
    const std::string dir = FixtureDir() + "/" + head;
    const Tokenizer tok = Tokenizer::FromHfJson(dir + "/tokenizer.json");
    const std::map<int32_t, std::string> ref_vocab = RefLoadVocabJson(dir);
    // The committed pre-refactor golden id sequences, plus probes that hit the
    // first-piece rule and the added <blank> token.
    const std::vector<std::vector<int32_t>> cases = {
        {3, 4, 3},                                              // golden_ctc ids
        {5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6},  // rnnt
        {0, 3, 6},  // ▁the at ▁on — first-piece drop + later-piece space
        {6},        // lone ▁on as the first piece
        {7},        // the added <blank> decodes literally
        {},
    };
    for (const auto& ids : cases) {
      CHECK_MESSAGE(tok.Decode(ids) == RefDecodeIds(ids, ref_vocab),
                    "head=", head, " n_ids=", ids.size());
    }
    // The two committed transcript goldens, verbatim.
    if (std::string(head) == "ctc") {
      CHECK(tok.Decode({3, 4, 3}) == "atheat");
    } else {
      CHECK(tok.Decode({5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5,
                        6, 6, 6, 6, 6, 6, 6, 6}) ==
            "sss on on onssssss on on on on on on on on");
    }
  }
}
