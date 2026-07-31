// laguna-gen — a minimal greedy generation driver for the single-GB10
// Laguna-S-2.1 keep-quant GGUF vehicle (unsloth UD-Q4_K_XL, 3 shards).
//
// Laguna's keep-quant forward (`LagunaForwardGguf`) is a STATELESS full-sequence
// recompute (it ignores the KV cache), so greedy decode is a manual loop: feed the
// growing token list each step, take the last-row logits, argmax, append. Mirrors
// examples/deepseek_v4_gen. The UD-Q4_K_XL model ships as 3 shards (shard-1 =
// header/KV only), so this opens ALL shards and passes them to the multi-shard
// keep-quant loader (LoadLagunaFromGgufShards), which routes each tensor to the
// shard that holds it. Expert/attention GEMMs run keep-quant on the CPU tier by
// default (--gpu routes the block-quant GEMMs to the GB10 via kMatmulBTQuant off
// the unified-memory mmap'd blocks — no device copy of the ~69 GiB tower).
//
//   laguna-gen --model <shard-1.gguf> [--prompt "..."] [--token-ids 1,2,3]
//              [--max-tokens N] [--load-only] [--gpu]
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
#include "vllm/model_executor/models/laguna.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vt/backend.h"  // vt::GetBackend / CreateQueue (--gpu: GEMMs on the GB10)
#include "vt/device.h"

namespace {

double CurResidentGiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str() + 6, "%ld", &kb);
      return static_cast<double>(kb) / (1024.0 * 1024.0);
    }
  return 0.0;
}
double PeakResidentGiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmHWM:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str() + 6, "%ld", &kb);
      return static_cast<double>(kb) / (1024.0 * 1024.0);
    }
  return 0.0;
}

int64_t KvInt(const vllm::GgufFile& g, const char* key, int64_t dflt) {
  const vllm::GgufValue* v = g.FindKv(key);
  if (v == nullptr) return dflt;
  switch (v->TypeId()) {
    case vllm::kGgufU8: return std::get<uint8_t>(v->v);
    case vllm::kGgufI8: return std::get<int8_t>(v->v);
    case vllm::kGgufU16: return std::get<uint16_t>(v->v);   // split.count is U16
    case vllm::kGgufI16: return std::get<int16_t>(v->v);
    case vllm::kGgufU32: return std::get<uint32_t>(v->v);
    case vllm::kGgufI32: return std::get<int32_t>(v->v);
    case vllm::kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v->v));
    case vllm::kGgufI64: return std::get<int64_t>(v->v);
    case vllm::kGgufBool: return std::get<bool>(v->v) ? 1 : 0;
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

// Derive the split-sibling shard paths from the shard-1 path + a split count.
// The llama.cpp split naming is "<stem>-NNNNN-of-MMMMM.gguf"; replace the 5-digit
// shard index before "-of-". Returns {path} for a single-file (non-split) model.
std::vector<std::string> ShardPaths(const std::string& shard1, int64_t count) {
  if (count <= 1) return {shard1};
  const std::string tag = "-of-";
  const size_t of = shard1.rfind(tag);
  std::vector<std::string> out;
  if (of == std::string::npos || of < 5) { out.push_back(shard1); return out; }
  const std::string prefix = shard1.substr(0, of - 5);  // up to the NNNNN index
  const std::string suffix = shard1.substr(of);         // "-of-MMMMM.gguf"
  for (int64_t s = 1; s <= count; ++s) {
    char idx[24];
    std::snprintf(idx, sizeof(idx), "%05d", static_cast<int>(s));
    out.push_back(prefix + idx + suffix);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt = "The capital of France is", token_ids_arg;
  int max_tokens = 24;
  bool load_only = false, use_gpu = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--prompt") prompt = next();
    else if (a == "--token-ids") token_ids_arg = next();
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--load-only") load_only = true;
    else if (a == "--gpu") use_gpu = true;
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (model.empty()) {
    std::fprintf(stderr, "usage: --model <shard-1.gguf> [--prompt ...] "
                         "[--token-ids ...] [--max-tokens N] [--load-only] [--gpu]\n");
    return 2;
  }

  // Open shard-1 (metadata) first to read the split count, then all shards.
  std::fprintf(stderr, "[gen] opening %s\n", model.c_str());
  vllm::GgufFile meta = vllm::GgufFile::Open(model);
  const int64_t split_count = KvInt(meta, "split.count", 1);
  const std::vector<std::string> paths = ShardPaths(model, split_count);
  std::vector<vllm::GgufFile> shards;
  shards.reserve(paths.size());
  shards.push_back(std::move(meta));  // shard-1 stays index 0 (has the KV)
  for (size_t s = 1; s < paths.size(); ++s) {
    std::fprintf(stderr, "[gen] opening shard %zu: %s\n", s + 1, paths[s].c_str());
    shards.push_back(vllm::GgufFile::Open(paths[s]));
  }
  std::vector<const vllm::GgufFile*> shard_ptrs;
  for (const vllm::GgufFile& s : shards) shard_ptrs.push_back(&s);

  const int64_t eos = KvInt(shards[0], "tokenizer.ggml.eos_token_id", 2);
  const int64_t eot = KvInt(shards[0], "tokenizer.ggml.eot_token_id", 24);

  // KEEP-QUANT load (the memory enabler): the routed experts stay Q4_K/Q5_K blocks;
  // mmap residency borrows them in place (no ~69 GiB copy).
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  pol.mmap_residency = true;
  std::fprintf(stderr, "[gen] loading keep-quant tower (RSS before %.1f GiB)...\n",
               CurResidentGiB());
  const auto t0 = std::chrono::steady_clock::now();
  const vllm::LagunaWeights w = vllm::LoadLagunaFromGgufShards(shard_ptrs, &pol);
  const auto t1 = std::chrono::steady_clock::now();
  const int64_t vocab = w.params.vocab_size;
  std::fprintf(stderr,
      "[gen] LOADED: layers=%lld experts=%lld vocab=%lld has_gguf=%d | load %.1fs | "
      "RSS %.1f GiB PEAK %.1f GiB\n",
      (long long)w.params.num_hidden_layers, (long long)w.params.num_experts,
      (long long)vocab, (int)w.has_gguf_weights,
      std::chrono::duration<double>(t1 - t0).count(), CurResidentGiB(),
      PeakResidentGiB());
  if (!w.has_gguf_weights) { std::fprintf(stderr, "[gen] ERROR: no keep-quant tower\n"); return 1; }
  if (load_only) { std::fprintf(stderr, "[gen] --load-only done.\n"); return 0; }

  // Prompt -> token ids. Prefer injected ids (exact cross-check); else our encode.
  std::vector<int32_t> tokens;
  if (!token_ids_arg.empty()) {
    tokens = ParseIds(token_ids_arg);
    std::fprintf(stderr, "[gen] using %zu injected token ids\n", tokens.size());
  } else {
    const vllm::tok::Tokenizer tkz = vllm::tok::Tokenizer::FromGguf(shards[0]);
    tokens = tkz.Encode(prompt);
    if (KvInt(shards[0], "tokenizer.ggml.add_bos_token", 0) != 0) {
      const int64_t bos = KvInt(shards[0], "tokenizer.ggml.bos_token_id", 2);
      tokens.insert(tokens.begin(), static_cast<int32_t>(bos));
    }
    std::fprintf(stderr, "[gen] encoded prompt (%zu ids):", tokens.size());
    for (int32_t t : tokens) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");
  }
  const size_t n_prompt = tokens.size();

  vt::Backend* gpu_backend = nullptr;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  if (use_gpu) {
    gpu_backend = &vt::GetBackend(vt::DeviceType::kCUDA);
    q = gpu_backend->CreateQueue();
    std::fprintf(stderr, "[gen] GPU keep-quant GEMMs on CUDA device %d\n", q.device.index);
  } else {
    std::fprintf(stderr, "[gen] CPU keep-quant queue\n");
  }

  std::vector<int32_t> generated;
  double prefill_s = 0.0, decode_s = 0.0;
  for (int step = 0; step < max_tokens; ++step) {
    std::vector<int32_t> positions(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) positions[i] = static_cast<int32_t>(i);
    const std::vector<int32_t> logits_idx = {static_cast<int32_t>(tokens.size() - 1)};
    const auto s0 = std::chrono::steady_clock::now();
    const std::vector<float> logits =
        vllm::LagunaForwardGguf(w, q, tokens, positions, logits_idx);
    const auto s1 = std::chrono::steady_clock::now();
    const double step_s = std::chrono::duration<double>(s1 - s0).count();
    const int next = ArgmaxLastRow(logits, vocab);
    if (step == 0) prefill_s = step_s; else decode_s += step_s;
    std::fprintf(stderr, "[gen] step %d (ctx=%zu): next=%d  %.2fs  (RSS %.1f GiB)\n",
                 step, tokens.size(), next, step_s, CurResidentGiB());
    generated.push_back(next);
    tokens.push_back(next);
    if (next == eos || next == eot) { std::fprintf(stderr, "[gen] EOS/EOT\n"); break; }
  }

  std::string text;
  try {
    const vllm::tok::Tokenizer tkz = vllm::tok::Tokenizer::FromGguf(shards[0]);
    text = tkz.Decode(generated);
  } catch (const std::exception& e) { text = std::string("<decode failed: ") + e.what() + ">"; }

  std::printf("\n===== LAGUNA-S-2.1 GENERATION =====\n");
  std::printf("prompt: %s\n", prompt.c_str());
  std::printf("generated ids:");
  for (int32_t t : generated) std::printf(" %d", t);
  std::printf("\ngenerated text: %s\n", text.c_str());
  const int n_dec = static_cast<int>(generated.size()) - 1;
  std::printf("prompt_tokens=%zu gen_tokens=%zu\n", n_prompt, generated.size());
  std::printf("prefill: %.2fs  decode: %.2fs  TPOT %.2fs/tok over %d steps\n",
              prefill_s, decode_s, n_dec > 0 ? decode_s / n_dec : 0.0, n_dec);
  std::printf("PEAK RESIDENT: %.2f GiB\n", PeakResidentGiB());
  std::fflush(stdout);
  if (gpu_backend != nullptr) gpu_backend->DestroyQueue(q);
  return 0;
}
