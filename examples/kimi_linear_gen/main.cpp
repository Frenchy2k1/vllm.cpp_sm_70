// kimi-linear-gen — the §13 full-model e2e harness for Kimi-Linear-48B-A3B on one
// GB10. It loads the bf16-RESIDENT weights (LoadKimiLinearResidentBf16Weights — NEVER
// the 183 GiB f32 MaterializeHost; the pool-math path stages each large matmul weight
// to cudaMalloc'd d_dev and ReleaseHost's its host mirror, so the LOAD peak holds only
// one tensor's host bytes on top of the growing ~91.5 GiB device residency) and greedy-
// decodes the §12 8-prompt battery x N tokens through the bf16 ForwardDeviceCompute
// (VT_KIMI_DEVICE_COMPUTE arm), comparing token-exact to the STRICT oracle golden
// (tests/parity/goldens/kimi_linear_greedy/greedy_ids.npy).
//
// GB10 load recipe (context-first + shard-release, mirror examples/laguna_gen/
// main.cpp:185-237): create the CUDA context BEFORE loading weights (the driver's
// reservation would OOM against the page-cache pressure of a post-load reservation),
// then drop the mmap'd shards after the memcpy loader.
//
//   kimi-linear-gen --model <hf-snapshot-dir> --golden <golden-dir>
//                   [--gpu] [--steps N] [--prompts M] [--load-only]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kimi_linear.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace fs = std::filesystem;

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

std::vector<vllm::SafetensorsFile> OpenSafetensorsDir(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ".safetensors")
      paths.push_back(e.path().string());
  if (paths.empty()) throw std::runtime_error("no *.safetensors shards in " + dir);
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) shards.push_back(vllm::SafetensorsFile::Open(p));
  return shards;
}

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
  // parse the shape tuple "'shape': (P, T)" (or "(P,)" / "(N,)").
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
  bool use_gpu = false, load_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--golden") golden = next();
    else if (a == "--steps") steps = std::atoi(next());
    else if (a == "--prompts") prompts = std::atoi(next());
    else if (a == "--gpu") use_gpu = true;
    else if (a == "--load-only") load_only = true;
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (model.empty()) {
    std::fprintf(stderr,
                 "usage: --model <hf-snapshot-dir> [--golden <dir>] [--gpu] "
                 "[--steps N] [--prompts M] [--load-only]\n");
    return 2;
  }

  // Create the CUDA context BEFORE loading weights (GB10 load recipe).
  vt::Backend* gpu_backend = nullptr;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  if (use_gpu) {
    gpu_backend = &vt::GetBackend(vt::DeviceType::kCUDA);
    q = gpu_backend->CreateQueue();
    std::fprintf(stderr, "[kimi] GPU context reserved on CUDA device %d (before load)\n",
                 q.device.index);
  } else {
    std::fprintf(stderr, "[kimi] CPU queue (won't fit the full model — smoke only)\n");
  }

  const std::string config_path = (fs::path(model) / "config.json").string();
  std::fprintf(stderr, "[kimi] config %s\n", config_path.c_str());
  const vllm::HfConfig config = vllm::LoadHfConfig(config_path);
  std::vector<vllm::SafetensorsFile> shards = OpenSafetensorsDir(model);
  std::fprintf(stderr,
               "[kimi] %zu safetensors shard(s); loading bf16-resident tower "
               "(RSS before %.1f GiB)...\n",
               shards.size(), CurResidentGiB());

  const auto t0 = std::chrono::steady_clock::now();
  vllm::KimiLinearWeights w = vllm::LoadKimiLinearResidentBf16Weights(
      shards, config, use_gpu ? &q : nullptr);
  const auto t1 = std::chrono::steady_clock::now();
  std::fprintf(stderr,
               "[kimi] LOADED bf16-resident: layers=%lld experts=%lld vocab=%lld | "
               "load %.1fs | RSS %.1f GiB PEAK %.1f GiB\n",
               (long long)w.params.num_hidden_layers, (long long)w.params.num_experts,
               (long long)w.params.vocab_size,
               std::chrono::duration<double>(t1 - t0).count(), CurResidentGiB(),
               PeakResidentGiB());

  // Drop the mmap'd shards (the loader copied every tensor into owned buffers).
  const size_t n_shards = shards.size();
  shards.clear();
  shards.shrink_to_fit();
  std::fprintf(stderr, "[kimi] released %zu mmap'd shard(s); RSS %.1f GiB\n", n_shards,
               CurResidentGiB());
  if (load_only) return 0;

  // Greedy decode + compare to the STRICT golden.
  const int64_t V = w.params.vocab_size;
  vllm::v1::CommonAttentionMetadata attn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  vt::Backend& be = vt::GetBackend(q.device.type);

  NpyInts gold;
  bool have_golden = !golden.empty();
  if (have_golden) {
    gold = ReadNpyInts((fs::path(golden) / "greedy_ids.npy").string());
    std::fprintf(stderr, "[kimi] golden greedy_ids shape [%lld,%lld]\n",
                 (long long)(gold.shape.size() > 0 ? gold.shape[0] : 0),
                 (long long)(gold.shape.size() > 1 ? gold.shape[1] : 0));
  }

  int total = 0, matched = 0;
  double first_tok_s = 0.0;
  int steady_steps = 0;
  double steady_s = 0.0;
  for (int pi = 0; pi < prompts; ++pi) {
    std::vector<int32_t> prompt;
    if (have_golden) {
      const std::string pp =
          (fs::path(golden) / ("p" + std::to_string(pi) + "_prompt.i32")).string();
      prompt = ReadI32File(pp);
    }
    if (prompt.empty()) { std::fprintf(stderr, "[kimi] prompt %d empty, skip\n", pi); continue; }

    std::vector<int32_t> seq = prompt;
    std::vector<int32_t> gen;
    for (int s = 0; s < steps; ++s) {
      std::vector<int32_t> positions(seq.size());
      for (size_t t = 0; t < seq.size(); ++t) positions[t] = static_cast<int32_t>(t);
      const std::vector<int32_t> li = {static_cast<int32_t>(seq.size() - 1)};
      const auto ts = std::chrono::steady_clock::now();
      vllm::ForwardLogits fl = vllm::KimiLinearModel::ForwardDeviceCompute(
          seq, positions, attn_meta, attn_kv, w, q, li);
      // download the single logit row + argmax on host.
      std::vector<float> row(static_cast<size_t>(V));
      be.Copy(q, row.data(), fl.device_tensor.data, row.size() * sizeof(float));
      be.Synchronize(q);
      const auto te = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(te - ts).count();
      if (pi == 0 && s == 0) first_tok_s = dt;
      else { steady_s += dt; ++steady_steps; }
      int best = 0;
      float bv = row[0];
      for (int64_t o = 1; o < V; ++o)
        if (row[static_cast<size_t>(o)] > bv) { bv = row[static_cast<size_t>(o)]; best = static_cast<int>(o); }
      gen.push_back(best);
      seq.push_back(best);
    }

    // compare to golden row pi.
    if (have_golden && pi < (gold.shape.empty() ? 0 : static_cast<int>(gold.shape[0]))) {
      const int64_t T = gold.shape.size() > 1 ? gold.shape[1] : 0;
      int row_match = 0;
      const int n = static_cast<int>(std::min<int64_t>(T, steps));
      std::string got, exp;
      for (int t = 0; t < n; ++t) {
        const int64_t g = gold.data[static_cast<size_t>(pi) * T + t];
        const int32_t o = gen[static_cast<size_t>(t)];
        got += std::to_string(o) + (t + 1 < n ? "," : "");
        exp += std::to_string(g) + (t + 1 < n ? "," : "");
        ++total;
        if (static_cast<int64_t>(o) == g) { ++matched; ++row_match; }
      }
      std::fprintf(stderr, "[kimi] prompt %d: %d/%d tokens match golden\n", pi, row_match, n);
      if (row_match != n) {
        std::fprintf(stderr, "        got: %s\n        exp: %s\n", got.c_str(), exp.c_str());
      }
    } else {
      std::string got;
      for (int t = 0; t < steps; ++t) got += std::to_string(gen[static_cast<size_t>(t)]) + ",";
      std::fprintf(stderr, "[kimi] prompt %d gen: %s\n", pi, got.c_str());
    }
  }

  if (have_golden)
    std::fprintf(stderr, "\n[kimi] TOKEN MATCH: %d/%d  (%s)\n", matched, total,
                 matched == total ? "STRICT PASS" : "DIVERGENCE");
  if (steady_steps > 0)
    std::fprintf(stderr, "[kimi] first-step %.3fs | steady %.3f s/step (%.2f tok/s) over %d steps\n",
                 first_tok_s, steady_s / steady_steps, steady_steps / steady_s, steady_steps);
  std::fprintf(stderr, "[kimi] RSS %.1f GiB PEAK %.1f GiB\n", CurResidentGiB(),
               PeakResidentGiB());
  return (have_golden && matched != total) ? 1 : 0;
}
