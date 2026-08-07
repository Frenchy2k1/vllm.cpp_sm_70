// Parakeet ASR end to end — a THIN CLIENT of the public C ABI (include/vllm.h)
// and NOTHING else, per the ONE SURFACE directive (ARCH-ONE-SURFACE ROW 1).
//
// The pre-fold version of this example owned the whole pipeline privately
// (its own WAV reader, the model_type head dispatch, a LoadVocab + DecodeIds
// pair working around the tokenizer's Metaspace split=true refusal). All of
// that now lives in the library behind vllm_engine_load + vllm_transcribe —
// the SAME entry points any embedder gets — and this file keeps exactly what
// an example may own: argv parsing, timing, and printing. The transcript is
// byte-identical to the pre-fold binary (gated by
// tests/vllm/models/test_parakeet_transcription_fold.cpp and the committed
// goldens in tests/vllm/models/fixtures/parakeet_e2e).
//
//   parakeet-transcribe <hf-parakeet-dir> <audio.wav> [reps]
//
// `<hf-parakeet-dir>` is any HF-format Parakeet checkpoint (CTC / RNN-T /
// TDT — the head comes from config.json, resolved inside the library). Token
// ids always go to stdout, diffable against HF's own `generate()`; when the
// checkpoint ships a tokenizer.json the decoded TEXT is printed too. The
// optional `reps` repeats the TRANSCRIPTION call so the checkpoint load stays
// out of the timed region (the load happens once, in vllm_engine_load).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "vllm.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <hf-parakeet-dir> <audio.wav> [reps]\n",
                 argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  const std::string wav = argv[2];
  const int reps = (argc > 3) ? std::atoi(argv[3]) : 1;

  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = ckpt.c_str();
  vllm_engine* engine = nullptr;
  if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    std::fprintf(stderr, "load failed: %s\n", vllm_last_error());
    return 1;
  }

  vllm_transcription_params tp = vllm_transcription_params_default();
  tp.audio_path = wav.c_str();

  vllm_transcription out;
  double best_ms = 0.0;
  vllm_status st = VLLM_OK;
  for (int r = 0; r < reps; ++r) {
    if (r > 0) vllm_transcription_free(&out);
    const auto t0 = std::chrono::steady_clock::now();
    st = vllm_transcribe(engine, &tp, &out);
    const auto t1 = std::chrono::steady_clock::now();
    if (st != VLLM_OK) break;
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (r == 0 || ms < best_ms) best_ms = ms;
    if (reps > 1) std::fprintf(stderr, "transcribe rep %d: %.1f ms\n", r, ms);
  }
  if (st != VLLM_OK) {
    std::fprintf(stderr, "transcribe failed: %s\n", vllm_last_error());
    vllm_engine_free(engine);
    return 1;
  }
  if (reps > 1) std::fprintf(stderr, "transcribe best: %.1f ms\n", best_ms);

  for (int32_t i = 0; i < out.n_token_ids; ++i) {
    std::printf("%s%d", i ? " " : "", out.token_ids[i]);
  }
  std::printf("\n");
  if (out.has_text) {
    std::printf("%s\n", out.text);
  } else {
    std::fprintf(stderr, "no tokenizer.json in the checkpoint: ids only\n");
  }

  vllm_transcription_free(&out);
  vllm_engine_free(engine);
  return 0;
}
