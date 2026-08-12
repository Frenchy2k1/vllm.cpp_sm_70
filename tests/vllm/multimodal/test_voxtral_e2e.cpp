// A3 — e2e AUDIO->TEXT gate on Voxtral-Mini-3B. The FIRST audio understanding in
// the tree: the FULL C++ pipeline (A1 log-mel -> A2 Whisper-class encoder at
// Voxtral's config -> AudioLanguageAdapter projector -> masked-scatter merge into
// the LANDED Mistral decoder -> greedy) must reproduce the vLLM 0.25.0 greedy
// golden token-for-token (gate form STRICT — measured K=5 self-deterministic).
//
// Provenance: vllm/model_executor/models/voxtral.py @ e24d1b24. Golden captured by
// scripts/mm/a3_voxtral_oracle_capture.py (vLLM 0.25.0 + mistral_common 1.11.5,
// load_format=mistral) — see tests/vllm/multimodal/fixtures/voxtral_audio/.
//
// GPU + weights required: point VLLM_VOXTRAL_SAFETENSORS at the downloaded
// consolidated.safetensors (mistral format, ~8.8 GiB, NOT committed). Fixtures
// (WAV, log-mel/mel/sinusoid goldens, prompt ids, golden tokens) ARE committed.
// Without the weights (or without CUDA) the gate is SKIPPED — and a skip EXITS 77,
// never 0, so it can never be read as a pass. See SkipGate below (issue #463).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "nlohmann/json.hpp"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/voxtral.h"
#include "vllm/multimodal/audio_processor.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::HfConfig;
using vllm::SafetensorsFile;
using vllm::VoxtralWeights;

std::string Fix() { return std::string(MM_VOXTRAL_FIXTURE_DIR); }

// A gate that CANNOT RUN must never report success.
//
// Returning early out of a doctest TEST_CASE prints `assertions: 0 | 0 passed |
// 0 failed` followed by `Status: SUCCESS!` and exits 0 — indistinguishable, in a
// log or in a `&&` chain, from a gate that loaded 8.8 GiB of weights and matched
// the oracle. multimodal-speed.md §17.7 recorded that trap; issue #463 files it.
//
// The fix: exit with CTest's SKIP_RETURN_CODE (77, registered by
// `vllm_cpp_add_test` in tests/CMakeLists.txt). `ctest` then reports the test as
// **Skipped**, not Passed; a shell chain stops at the non-zero status; and doctest
// never gets to print a SUCCESS banner for a run that asserted nothing. Exiting is
// deliberate rather than a `FAIL_CHECK`: a CPU-only or weightless box legitimately
// cannot run this gate, and "skipped" is the true result there — what was wrong
// was reporting it as "passed".
[[noreturn]] void SkipGate(const char* why) {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** test_voxtral_e2e: %s\n\n",
               why);
  std::fflush(stderr);
  std::exit(77);
}

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num / (den + 1e-30));
}

// Mistral/Llama text HfConfig for Voxtral-Mini-3B (params.json text side).
HfConfig VoxtralTextConfig() {
  HfConfig c;
  c.model_type = "llama";
  c.hidden_size = 3072;
  c.num_hidden_layers = 30;
  c.num_attention_heads = 32;
  c.num_key_value_heads = 8;
  c.head_dim = 128;
  c.rotary_dim = 128;
  c.intermediate_size = 8192;
  c.vocab_size = 131072;
  c.rms_norm_eps = 1e-5;
  c.rope_theta = 100000000.0;  // 1e8
  c.rope_parameters.rope_type = "default";
  return c;
}

}  // namespace

TEST_CASE("voxtral_audio_to_text_e2e_strict_vs_vllm_0_25_0") {
  const char* stf = std::getenv("VLLM_VOXTRAL_SAFETENSORS");
  if (stf == nullptr) {
    SkipGate("set VLLM_VOXTRAL_SAFETENSORS to Voxtral consolidated.safetensors");
  }
  vt::Backend* gpu = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (gpu == nullptr) {
    SkipGate("no CUDA backend");
  }

  // ---- TEXT-ONLY isolation path (decoder-only, no audio/merge) ----
  if (std::getenv("VLLM_VOXTRAL_TEXTONLY") != nullptr) {
    nlohmann::json tj;
    { std::ifstream f(Fix() + "/voxtral_textonly.json"); f >> tj; }
    std::vector<int32_t> tp = tj["prompt_ids"].get<std::vector<int32_t>>();
    std::vector<int32_t> tg = tj["output_token_ids"].get<std::vector<int32_t>>();
    SafetensorsFile st = SafetensorsFile::Open(stf);
    std::vector<float> ep = ReadF32(Fix() + "/voxtral_embed_positions_f32.bin");
    VoxtralWeights w = vllm::LoadVoxtralWeights(st, ep, VoxtralTextConfig());
    vt::Queue q = gpu->CreateQueue();
    std::vector<int32_t> g = vllm::VoxtralGenerateGreedy(
        tp, {}, /*audio_token_id=*/-999, /*eos=*/2, w, VoxtralTextConfig(), q,
        static_cast<int>(tg.size()));
    gpu->DestroyQueue(q);
    int m = 0;
    for (size_t i = 0; i < std::min(g.size(), tg.size()); ++i)
      if (g[i] == tg[i]) ++m;
    MESSAGE("TEXTONLY match ", m, "/", tg.size(), "  got[0..3]=", g.size() > 0 ? g[0] : -1,
            ",", g.size() > 1 ? g[1] : -1, ",", g.size() > 2 ? g[2] : -1);
    CHECK(m == static_cast<int>(tg.size()));
    return;
  }

  // ---- fixtures + golden ----
  nlohmann::json man;
  { std::ifstream f(Fix() + "/voxtral_manifest.json"); f >> man; }
  nlohmann::json gold;
  { std::ifstream f(Fix() + "/voxtral_golden.json"); f >> gold; }
  REQUIRE(gold["gate_form"] == "STRICT");
  const int32_t audio_token_id = man["audio_token_id"].get<int32_t>();
  std::vector<int32_t> prompt_ids = man["prompt_ids"].get<std::vector<int32_t>>();
  std::vector<int32_t> golden_tokens = gold["output_token_ids"].get<std::vector<int32_t>>();
  const int max_new = static_cast<int>(golden_tokens.size());

  // ---- STEP 1: C++ log-mel (A1 processor at Voxtral config) ----
  vllm::multimodal::AudioProcessorConfig acfg;
  acfg.n_fft = 400;
  acfg.hop_length = 160;
  acfg.n_mels = 128;
  acfg.sampling_rate = 16000;
  acfg.chunk_length_s = 30;
  acfg.max_source_positions = 1500;
  acfg.audio_placeholder_id = audio_token_id;
  acfg.model_id = "mistralai/Voxtral-Mini-3B-2507";
  std::vector<float> mel = ReadF32(Fix() + "/voxtral_mel_filters_f32.bin");  // [201,128]
  vllm::multimodal::WhisperAudioProcessor proc(acfg, mel);

  std::vector<uint8_t> wav = ReadBytes(Fix() + "/voxtral_input_16k_mono.wav");
  vllm::multimodal::DecodedAudio dec = vllm::multimodal::DecodeWavPcm16Mono(wav.data(), wav.size());
  vllm::multimodal::AudioKwargs feat =
      proc.ProcessWaveform(dec.samples.data(), static_cast<int64_t>(dec.samples.size()),
                           dec.sampling_rate);
  // log-mel parity vs the oracle input_features (sub-check; A1 methodology).
  std::vector<float> feat_ref = ReadF32(Fix() + "/voxtral_input_features_f32.bin");
  const double mel_rel = RelL2(feat.input_features, feat_ref);
  MESSAGE("log-mel rel-L2 vs oracle: ", mel_rel);
  CHECK(mel_rel < 2e-4);

  // ---- STEP 2: A2 encoder tower at Voxtral config ----
  VoxtralWeights weights;
  {
    SafetensorsFile st = SafetensorsFile::Open(stf);
    std::vector<float> embed_pos = ReadF32(Fix() + "/voxtral_embed_positions_f32.bin");
    weights = vllm::LoadVoxtralWeights(st, embed_pos, VoxtralTextConfig());
  }
  std::vector<float> enc =
      vllm::multimodal::WhisperAudioEncoderForward(feat.input_features, weights.encoder,
                                                   weights.encoder_cfg, *gpu, nullptr);
  // enc = [1500, 1280].
  REQUIRE(static_cast<int64_t>(enc.size()) ==
          weights.encoder_cfg.max_source_positions * weights.encoder_cfg.d_model);

  const char* dbg = std::getenv("VLLM_VOXTRAL_DEBUG_DIR");
  if (dbg != nullptr) {
    std::vector<float> enc_ref = ReadF32(std::string(dbg) + "/dbg_encoder_out.bin");
    MESSAGE("encoder_out rel-L2 vs vLLM: ", RelL2(enc, enc_ref));
  }

  // ---- STEP 3: projector (downsample-concat + adapter) -> [375, 3072] ----
  std::vector<float> aud = vllm::VoxtralProjectAudio(enc, weights, *gpu);
  if (dbg != nullptr) {
    std::vector<float> aud_ref = ReadF32(std::string(dbg) + "/dbg_audio_embeds.bin");
    MESSAGE("audio_embeds rel-L2 vs vLLM: ", RelL2(aud, aud_ref));
  }
  const int64_t n_aud = static_cast<int64_t>(aud.size()) / VoxtralTextConfig().hidden_size;
  MESSAGE("audio embeds rows: ", n_aud);
  REQUIRE(n_aud == man["num_audio_tokens"].get<int64_t>());

  // DEBUG: optionally substitute vLLM's exact audio embeddings to isolate the
  // text decoder from the encoder/projector.
  if (std::getenv("VLLM_VOXTRAL_USE_REF_AUDIO") != nullptr && dbg != nullptr) {
    aud = ReadF32(std::string(dbg) + "/dbg_audio_embeds.bin");
    MESSAGE("USING vLLM reference audio embeds for decode");
  }

  // ---- STEP 4: merge + Mistral greedy ----
  vt::Queue q = gpu->CreateQueue();
  std::vector<int32_t> got = vllm::VoxtralGenerateGreedy(
      prompt_ids, aud, audio_token_id, /*eos=*/2, weights, VoxtralTextConfig(), q, max_new);
  gpu->DestroyQueue(q);

  // Dump our tokens (near-tie-robust gate input) when requested.
  if (const char* op = std::getenv("VLLM_VOXTRAL_OUT_TOKENS")) {
    std::ofstream of(op);
    for (size_t i = 0; i < got.size(); ++i) of << (i ? "," : "") << got[i];
  }

  // ---- GATE (near-tie DISTRIBUTIONAL, user-ratified [[near-tie-distributional-gate]]) ----
  //
  // WHY distributional (not byte-exact to one branch): Voxtral text decode now runs
  // through the FA2 varlen split-KV kernel (LaunchDecodeVarlenFA2Bf16), routed since
  // block_size is rounded UP to a multiple of 16 (voxtral.cpp; multimodal-speed.md
  // §11-12 — this is the audio decode-speed win: TPOT 59.4->38.2 ms/tok, BEATS vLLM).
  // vLLM's OWN greedy at bf16 is NON-DETERMINISTIC at exact logit ties, and this
  // decode has TWO such ties inside 48 tokens: pos 18 (2-way, gap 0.000: FA2 tok
  // 24466 vs golden 1584, IDENTICAL logprob) and pos 33 (4-way, gap 0.000). A 1-ULP
  // reduction-order difference decides each. The FA2 f32 reduction takes the OTHER
  // side of the pos-18 tie than the vLLM greedy golden did, after which the equally
  // valid divergent context yields a different continuation. BOTH sequences are vLLM
  // greedy: teacher-forcing vLLM 0.25.0 on the FA2 sequence gives 0 divergent
  // positions, worst gap 0.0000 nats, RESULT PASS (scripts/mm/a3_voxtral_neartie_gate.py
  // -> voxtral_neartie.json). So pinning a byte-match to ONE arbitrary branch (the old
  // scalar-kernel golden) over-constrains a true tie; the correct bar is the ratified
  // distributional PASS: every produced token is within vLLM's near-tie band (<= 0.5
  // nats of its argmax), the strict prefix is token-exact vs vLLM greedy up to the
  // FIRST genuine tie, and the sequence teacher-forces PASS. This gate is
  // KERNEL-INDEPENDENT: both the scalar and the FA2 branch PASS it.
  //
  // voxtral_neartie.json::our_tokens is regenerated to the SHIPPING kernel's sequence
  // (FA2) purely as a determinism anchor; the pass/fail VERDICT is the teacher-force
  // PASS below (result==PASS + zero divergent + worst_gap<=0.5), NOT the byte identity.
  //
  // WHAT THIS PROCESS CAN AND CANNOT CHECK — read before trusting the printed line.
  // The teacher-force runs OFFLINE, in Python, against the live oracle
  // (scripts/mm/a3_voxtral_neartie_gate.py). Its verdict is COMMITTED into
  // voxtral_neartie.json. This test does not have an oracle in-process and does not
  // re-run it. So `result` / `n_divergent` / `over_band_failures` / `worst_gap_nats`
  // below are **FIXTURE PROVENANCE for `our_tokens`** — they record which sequence was
  // validated and how — and they are CONSTANTS with respect to `got`. Asserting them
  // pins the fixture (a regenerated near-tie file that no longer PASSES cannot be
  // committed silently); it says nothing about this run's output. The only checks here
  // that discriminate on `got` are `got.size()`, `strict_prefix >= 18` and
  // `repro == 48`: `repro` is what ties `got` to the teacher-force-validated sequence,
  // and therefore what carries the correctness claim into this process.
  // Reviewed and corrected on PR #439 (finding F7): the earlier "BINDING CORRECTNESS"
  // label on the fixture constants overstated what they do — the FA-2 arm prints
  // `divergent=0 worst_gap_nats=0` while FAILING, precisely because those two came out
  // of the file rather than out of `got`.
  nlohmann::json nt;
  { std::ifstream f(Fix() + "/voxtral_neartie.json"); f >> nt; }
  std::vector<int32_t> nt_tokens = nt["our_tokens"].get<std::vector<int32_t>>();
  // Fixture-provenance constants (see above): loaded from the committed JSON, NOT
  // computed from `got`.
  const double fixture_worst_gap = nt["worst_gap_nats"].get<double>();
  const int fixture_n_divergent = nt["n_divergent"].get<int>();
  const size_t fixture_over_band = nt["over_band_failures"].size();

  int strict_prefix = 0;
  for (size_t i = 0; i < std::min(got.size(), golden_tokens.size()); ++i) {
    if (got[i] != golden_tokens[i]) break;
    ++strict_prefix;
  }
  int repro = 0;
  for (size_t i = 0; i < std::min(got.size(), nt_tokens.size()); ++i)
    if (got[i] == nt_tokens[i]) ++repro;
  MESSAGE("STRICT prefix vs vLLM greedy [THIS RUN]: ", strict_prefix, "/",
          golden_tokens.size(), " (exact up to the first genuine bf16 tie; FA2 branch: "
          "pos 18)");
  MESSAGE("reproduces the teacher-force-validated sequence [THIS RUN]: ", repro, "/",
          nt_tokens.size());
  MESSAGE("fixture provenance for that sequence [FROM voxtral_neartie.json, NOT from "
          "this run]: result=", nt["result"].get<std::string>(),
          " divergent=", fixture_n_divergent, " worst_gap_nats=", fixture_worst_gap,
          " over-band(>0.5nats)=", fixture_over_band);

  // ---- checks that discriminate on THIS RUN's output ----
  // Length: exactly the golden's token count.
  CHECK(static_cast<int>(got.size()) == static_cast<int>(golden_tokens.size()));

  // Strict prefix: token-exact vs vLLM greedy up to the first genuine bf16 exact tie.
  // The FA2 kernel takes the other side of the pos-18 2-way exact tie (gap 0.000) than
  // the greedy golden, so its exact-match prefix is 18 (the scalar kernel's was 33 at
  // the pos-33 4-way tie; both branches are teacher-force-valid). We require the
  // pipeline to reproduce vLLM greedy EXACTLY up to that first tie.
  CHECK(strict_prefix >= 18);

  // THE BINDING CORRECTNESS CHECK in this process: `got` IS, position for position,
  // the sequence the offline teacher-force validated against the oracle. This is what
  // carries the distributional verdict into the build — combined with the fixture
  // assertions below, which pin what that verdict was.
  CHECK(repro == static_cast<int>(nt_tokens.size()));

  // ---- fixture-provenance assertions (CONSTANTS w.r.t. `got`) ----
  // These pin the committed near-tie file to a PASSING teacher-force, so a
  // regenerated fixture that no longer passes cannot slip in under the `repro` check
  // above. They are not, and must not be read as, a measurement of this run.
  CHECK(nt["result"].get<std::string>() == "PASS");
  CHECK(fixture_n_divergent == 0);  // the validated sequence IS vLLM's argmax throughout
  CHECK(fixture_over_band == 0);    // no divergence exceeded the 0.5-nat near-tie band
  CHECK(fixture_worst_gap <= 0.5);
}
