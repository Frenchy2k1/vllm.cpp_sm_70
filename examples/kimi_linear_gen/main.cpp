// kimi-linear-gen — THIN PUBLIC-ABI CLIENT (ONE SURFACE / ROW 7, §20.3 B4).
//
// The Kimi-Linear greedy token battery against the STRICT oracle golden, driven
// ENTIRELY through the flat C ABI (include/vllm.h): vllm_engine_load builds the
// full engine (the bf16-resident §13 loader + the shared paged runner the fold
// landed — KDA state in the MambaSpec group, NoPE-MLA latent in the paged MLA
// group), and vllm_complete_tokens (ABI v13) generates from the golden's
// pre-tokenized prompts. The former private harness (bf16-resident loader +
// KimiDecodeCache incremental decode driven through internal headers) is gone:
// the fast paged-incremental decode IS the engine's production path now, so
// this example is exactly what an embedder with only libvllm + vllm.h can do.
//
//   kimi-linear-gen --model <hf-snapshot-dir> [--golden <dir>] [--steps N]
//                   [--prompts M] [--load-only]
//
// The golden dir layout matches the §12 capture: p{i}_prompt.i32 (raw LE int32
// token ids) + greedy_ids.npy ([P,T] <i4/<i8 C-order).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm.h"

namespace fs = std::filesystem;

namespace {

// Read a raw little-endian int32 file (the golden p{i}_prompt.i32 prompts).
std::vector<int32_t> ReadI32File(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<int32_t> out;
  int32_t v = 0;
  while (f.read(reinterpret_cast<char*>(&v), 4)) out.push_back(v);
  return out;
}

// Minimal .npy reader for the [P,T] int golden ids (dtype <i4 or <i8, C-order).
struct NpyInts {
  std::vector<int64_t> shape;
  std::vector<int64_t> data;  // flattened, C-order
};
NpyInts ReadNpyInts(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not a .npy file: " + path);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint16_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 2);  // v1.0 little-endian header length
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  const bool i8 = header.find("<i8") != std::string::npos;
  const bool i4 = header.find("<i4") != std::string::npos;
  if (!i8 && !i4) throw std::runtime_error("npy dtype not <i4/<i8: " + header);
  NpyInts out;
  const size_t sp = header.find("'shape':");
  const size_t lp = header.find('(', sp);
  const size_t rp = header.find(')', lp);
  std::string dims = header.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < dims.size();) {
    while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) ++i;
    if (i >= dims.size()) break;
    size_t j = i;
    while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') ++j;
    if (j > i) out.shape.push_back(std::atoll(dims.substr(i, j - i).c_str()));
    i = j + 1;
  }
  int64_t n = 1;
  for (int64_t d : out.shape) n *= d;
  out.data.resize(static_cast<size_t>(n));
  for (int64_t k = 0; k < n; ++k) {
    if (i8) {
      int64_t v = 0;
      f.read(reinterpret_cast<char*>(&v), 8);
      out.data[static_cast<size_t>(k)] = v;
    } else {
      int32_t v = 0;
      f.read(reinterpret_cast<char*>(&v), 4);
      out.data[static_cast<size_t>(k)] = v;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, golden;
  int steps = 16, prompts = 8;
  // The golden battery is short (prompt ~40 + 16 continuations); a bounded
  // max_model_len keeps the engine's per-request token tables sized for the
  // battery rather than the checkpoint's 1M context. Override with
  // --max-model-len for longer runs.
  int max_model_len = 4096;
  bool load_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--golden") golden = next();
    else if (a == "--steps") steps = std::atoi(next());
    else if (a == "--prompts") prompts = std::atoi(next());
    else if (a == "--max-model-len") max_model_len = std::atoi(next());
    else if (a == "--load-only") load_only = true;
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (model.empty()) {
    std::fprintf(stderr,
                 "usage: --model <hf-snapshot-dir> [--golden <dir>] [--steps N] "
                 "[--prompts M] [--load-only]\n");
    return 2;
  }

  std::fprintf(stderr, "[kimi] libvllm %s (ABI %d, header %d)\n", vllm_version(),
               vllm_abi_version(), VLLM_ABI_VERSION);

  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = model.c_str();
  mp.max_model_len = max_model_len;
  vllm_engine* eng = nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  const vllm_status lst = vllm_engine_load(&mp, &eng);
  const auto t1 = std::chrono::steady_clock::now();
  if (lst != VLLM_OK) {
    std::fprintf(stderr, "[kimi] engine load FAILED: %s\n", vllm_last_error());
    return 1;
  }
  std::fprintf(stderr, "[kimi] engine loaded in %.1fs\n",
               std::chrono::duration<double>(t1 - t0).count());
  if (load_only) {
    vllm_engine_free(eng);
    return 0;
  }

  NpyInts gold;
  const bool have_golden = !golden.empty();
  if (have_golden) {
    gold = ReadNpyInts((fs::path(golden) / "greedy_ids.npy").string());
    std::fprintf(stderr, "[kimi] golden greedy_ids shape [%lld,%lld]\n",
                 (long long)(gold.shape.size() > 0 ? gold.shape[0] : 0),
                 (long long)(gold.shape.size() > 1 ? gold.shape[1] : 0));
  }

  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.0f;  // greedy
  sp.max_tokens = steps;
  sp.ignore_eos = 1;

  int total = 0, matched = 0;
  double first_prompt_full_s = 0.0, first_prompt_one_s = 0.0;
  for (int pi = 0; pi < prompts; ++pi) {
    std::vector<int32_t> prompt;
    if (have_golden) {
      const std::string pp =
          (fs::path(golden) / ("p" + std::to_string(pi) + "_prompt.i32")).string();
      try { prompt = ReadI32File(pp); } catch (const std::exception& e) {
        std::fprintf(stderr, "[kimi] prompt %d: %s — skip\n", pi, e.what());
        continue;
      }
    }
    if (prompt.empty()) { std::fprintf(stderr, "[kimi] prompt %d empty, skip\n", pi); continue; }

    std::vector<int32_t> gen(static_cast<size_t>(steps), 0);
    int32_t n_gen = 0;
    const auto ts = std::chrono::steady_clock::now();
    const vllm_status st = vllm_complete_tokens(
        eng, prompt.data(), static_cast<int32_t>(prompt.size()), &sp, gen.data(),
        static_cast<int32_t>(gen.size()), &n_gen, nullptr);
    const auto te = std::chrono::steady_clock::now();
    if (st != VLLM_OK) {
      std::fprintf(stderr, "[kimi] prompt %d FAILED: %s\n", pi, vllm_last_error());
      vllm_engine_free(eng);
      return 1;
    }
    if (pi == 0) {
      first_prompt_full_s = std::chrono::duration<double>(te - ts).count();
      // Two-length diff for the steady decode rate: re-run the same prompt at
      // max_tokens=1 and subtract (prefix caching is off by default on the
      // hybrid archs, so both runs pay the same prefill).
      vllm_sampling_params sp1 = sp;
      sp1.max_tokens = 1;
      int32_t one_tok = 0;
      int32_t n_one = 0;
      const auto u0 = std::chrono::steady_clock::now();
      if (vllm_complete_tokens(eng, prompt.data(),
                               static_cast<int32_t>(prompt.size()), &sp1, &one_tok,
                               1, &n_one, nullptr) == VLLM_OK) {
        const auto u1 = std::chrono::steady_clock::now();
        first_prompt_one_s = std::chrono::duration<double>(u1 - u0).count();
      }
    }

    if (have_golden && pi < (gold.shape.empty() ? 0 : static_cast<int>(gold.shape[0]))) {
      const int64_t T = gold.shape.size() > 1 ? gold.shape[1] : 0;
      int row_match = 0;
      const int n = static_cast<int>(std::min<int64_t>(T, n_gen));
      std::string got, exp;
      for (int t = 0; t < n; ++t) {
        const int64_t g = gold.data[static_cast<size_t>(pi) * T + t];
        const int32_t o = gen[static_cast<size_t>(t)];
        got += std::to_string(o) + (t + 1 < n ? "," : "");
        exp += std::to_string(g) + (t + 1 < n ? "," : "");
        ++total;
        if (static_cast<int64_t>(o) == g) { ++matched; ++row_match; }
      }
      std::fprintf(stderr, "[kimi] prompt %d: %d/%d tokens match golden\n", pi,
                   row_match, n);
      if (row_match != n)
        std::fprintf(stderr, "        got: %s\n        exp: %s\n", got.c_str(),
                     exp.c_str());
    } else {
      std::string got;
      for (int t = 0; t < n_gen; ++t)
        got += std::to_string(gen[static_cast<size_t>(t)]) + ",";
      std::fprintf(stderr, "[kimi] prompt %d gen: %s\n", pi, got.c_str());
    }
  }

  if (have_golden)
    std::fprintf(stderr, "\n[kimi] TOKEN MATCH: %d/%d  (%s)\n", matched, total,
                 matched == total ? "STRICT PASS" : "DIVERGENCE");
  if (first_prompt_full_s > 0.0 && first_prompt_one_s > 0.0 && steps > 1) {
    const double steady = (first_prompt_full_s - first_prompt_one_s) /
                          static_cast<double>(steps - 1);
    std::fprintf(stderr,
                 "[kimi] p0 wall %.3fs (N=%d) vs %.3fs (N=1) => steady %.3f "
                 "s/tok (%.2f tok/s, two-length diff)\n",
                 first_prompt_full_s, steps, first_prompt_one_s, steady,
                 steady > 0 ? 1.0 / steady : 0.0);
  }
  vllm_engine_free(eng);
  return (have_golden && matched != total) ? 1 : 0;
}
