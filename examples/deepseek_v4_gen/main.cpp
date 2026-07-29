// deepseek-v4-gen — a minimal greedy generation driver for the single-Spark
// DeepSeek-V4-Flash keep-quant GGUF vehicle (antirez/ds4 q2-imatrix).
//
// V4's keep-quant forward (`DeepseekV4ForwardGguf`) is a STATELESS full-sequence
// recompute (it ignores the KV cache), so greedy decode is a manual loop: feed
// the growing token list each step, take the last-row logits, argmax, append.
// This is the honest single-Spark run vehicle (the paged incremental engine does
// not apply to a stateless recompute forward). Expert GEMMs run keep-quant on the
// CPU tier (no CUDA keep-quant vec_dot exists) — SLOW for a 158 B model, but a
// REAL baseline; peak resident is asserted to stay ~keep-quant (not f32-expanded).
//
//   deepseek-v4-gen --model <file.gguf> [--prompt "..."] [--token-ids 1,2,3]
//                   [--max-tokens N] [--load-only]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace {

// Peak resident set size in GiB, read from /proc/self/status VmHWM (high-water).
double PeakResidentGiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str() + 6, "%ld", &kb);
      return static_cast<double>(kb) / (1024.0 * 1024.0);
    }
  }
  return 0.0;
}

double CurResidentGiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str() + 6, "%ld", &kb);
      return static_cast<double>(kb) / (1024.0 * 1024.0);
    }
  }
  return 0.0;
}

int64_t KvInt(const vllm::GgufFile& g, const char* key, int64_t dflt) {
  const vllm::GgufValue* v = g.FindKv(key);
  if (v == nullptr) return dflt;
  switch (v->TypeId()) {
    case vllm::kGgufU32: return std::get<uint32_t>(v->v);
    case vllm::kGgufI32: return std::get<int32_t>(v->v);
    case 10: return static_cast<int64_t>(std::get<uint64_t>(v->v));  // u64
    case 11: return std::get<int64_t>(v->v);                          // i64
    default: return dflt;
  }
}

int ArgmaxLastRow(const std::vector<float>& logits, int64_t vocab) {
  const int64_t rows = static_cast<int64_t>(logits.size()) / vocab;
  const float* row = logits.data() + (rows - 1) * vocab;
  int best = 0;
  float bv = row[0];
  for (int64_t i = 1; i < vocab; ++i)
    if (row[i] > bv) { bv = row[i]; best = static_cast<int>(i); }
  return best;
}

std::vector<int32_t> ParseIds(const std::string& s) {
  std::vector<int32_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt = "The capital of France is", token_ids_arg;
  int max_tokens = 16;
  bool load_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--prompt") prompt = next();
    else if (a == "--token-ids") token_ids_arg = next();
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--load-only") load_only = true;
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (model.empty()) { std::fprintf(stderr, "usage: --model <file.gguf> [--prompt ...] [--token-ids ...] [--max-tokens N] [--load-only]\n"); return 2; }

  std::fprintf(stderr, "[gen] opening %s\n", model.c_str());
  const auto t_open0 = std::chrono::steady_clock::now();
  const vllm::GgufFile g = vllm::GgufFile::Open(model);
  const int64_t eos = KvInt(g, "tokenizer.ggml.eos_token_id", 1);

  // KEEP-QUANT load (the memory enabler): the routed experts stay ~2-3-bit blocks.
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  std::fprintf(stderr, "[gen] loading keep-quant tower (RSS before %.1f GiB)...\n", CurResidentGiB());
  const auto t_load0 = std::chrono::steady_clock::now();
  const vllm::DeepseekV4Weights w = vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol);
  const auto t_load1 = std::chrono::steady_clock::now();
  const double load_s = std::chrono::duration<double>(t_load1 - t_load0).count();
  const double open_s = std::chrono::duration<double>(t_load0 - t_open0).count();
  const int64_t vocab = w.params.vocab_size;
  std::fprintf(stderr,
      "[gen] LOADED: layers=%lld experts=%lld vocab=%lld has_gguf=%d | open %.1fs load %.1fs | RSS %.1f GiB PEAK %.1f GiB\n",
      (long long)w.params.num_hidden_layers, (long long)w.params.n_routed_experts,
      (long long)vocab, (int)w.has_gguf_weights, open_s, load_s, CurResidentGiB(), PeakResidentGiB());
  if (!w.has_gguf_weights) { std::fprintf(stderr, "[gen] ERROR: no keep-quant tower\n"); return 1; }
  if (load_only) { std::fprintf(stderr, "[gen] --load-only done. PEAK %.2f GiB\n", PeakResidentGiB()); return 0; }

  // Prompt -> token ids. Prefer injected ids (exact cross-check); else our encode.
  std::vector<int32_t> tokens;
  if (!token_ids_arg.empty()) {
    tokens = ParseIds(token_ids_arg);
    std::fprintf(stderr, "[gen] using %zu injected token ids\n", tokens.size());
  } else {
    const vllm::tok::Tokenizer tkz = vllm::tok::Tokenizer::FromGguf(g);
    tokens = tkz.Encode(prompt);
    std::fprintf(stderr, "[gen] encoded prompt (%zu ids): ", tokens.size());
    for (int32_t t : tokens) std::fprintf(stderr, "%d ", t);
    std::fprintf(stderr, "\n");
  }
  const size_t n_prompt = tokens.size();

  // Greedy loop: full recompute each step (stateless keep-quant forward).
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<int32_t> generated;
  double prefill_s = 0.0, decode_s = 0.0;
  for (int step = 0; step < max_tokens; ++step) {
    std::vector<int32_t> positions(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) positions[i] = static_cast<int32_t>(i);
    const std::vector<int32_t> logits_idx = {static_cast<int32_t>(tokens.size() - 1)};
    const auto s0 = std::chrono::steady_clock::now();
    const std::vector<float> logits = vllm::DeepseekV4ForwardGguf(w, q, tokens, positions, logits_idx);
    const auto s1 = std::chrono::steady_clock::now();
    const double step_s = std::chrono::duration<double>(s1 - s0).count();
    const int next = ArgmaxLastRow(logits, vocab);
    if (step == 0) prefill_s = step_s; else decode_s += step_s;
    std::fprintf(stderr, "[gen] step %d (ctx=%zu): next=%d  %.2fs  (RSS %.1f GiB)\n",
                 step, tokens.size(), next, step_s, CurResidentGiB());
    generated.push_back(next);
    tokens.push_back(next);
    if (next == eos) { std::fprintf(stderr, "[gen] EOS\n"); break; }
  }

  // Decode (vocab-based, exact) + report.
  std::string text;
  try {
    const vllm::tok::Tokenizer tkz = vllm::tok::Tokenizer::FromGguf(g);
    text = tkz.Decode(generated);
  } catch (const std::exception& e) { text = std::string("<decode failed: ") + e.what() + ">"; }

  std::printf("\n===== DEEPSEEK-V4 GENERATION =====\n");
  std::printf("prompt: %s\n", prompt.c_str());
  std::printf("generated ids:");
  for (int32_t t : generated) std::printf(" %d", t);
  std::printf("\ngenerated text: %s\n", text.c_str());
  const int n_dec = static_cast<int>(generated.size()) - 1;  // decode steps (after prefill)
  std::printf("prompt_tokens=%zu gen_tokens=%zu\n", n_prompt, generated.size());
  std::printf("prefill: %.2fs (%.3f tok/s over %zu ctx)\n", prefill_s,
              prefill_s > 0 ? n_prompt / prefill_s : 0.0, n_prompt);
  std::printf("decode:  %.2fs  TPOT %.2fs/tok  (%.4f tok/s) over %d steps\n",
              decode_s, n_dec > 0 ? decode_s / n_dec : 0.0,
              (n_dec > 0 && decode_s > 0) ? n_dec / decode_s : 0.0, n_dec);
  std::printf("PEAK RESIDENT: %.2f GiB\n", PeakResidentGiB());
  std::fflush(stdout);
  return 0;
}
