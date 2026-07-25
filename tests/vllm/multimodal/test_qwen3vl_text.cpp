// M2b/M2c — Qwen3-VL text-backbone unit gates vs the dumped vLLM 0.25.0
// reference (scripts/mm/m2b_text_ref_dump.py, fixtures qwen3vl_text/). Four
// RED-first contracts, all CPU (no GPU / no weights):
//   1. get_rope_index — MRoPE 3-D positions [3,T] BIT-exact.
//   2. 3-section MRoPE application — vt::RopeFromCache (positions [3,T] +
//      mrope_section=[24,20,20] interleaved) vs MRotaryEmbedding.forward_native,
//      within a bf16 envelope; a WRONG config (interleaved off) must diverge.
//   3. DeepStack scatter — _compute_deepstack_embeds [L,T,H] BIT-exact.
//   4. embed-merge — _merge_multimodal_embeddings masked scatter [T,H] BIT-exact.
//
// Grounds: qwen3_vl.py _get_mrope_input_positions(:2567), _compute_deepstack_embeds
// (:2761); utils.py _merge_multimodal_embeddings(:524); mrope.py MRotaryEmbedding.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vllm::multimodal::MmImageSpan;
using vllm::multimodal::Qwen3VLComputeDeepstack;
using vllm::multimodal::Qwen3VLGetRopeIndex;
using vllm::multimodal::Qwen3VLMergeMultimodal;

// Fixture geometry (scripts/mm/m2b_text_ref_dump.py; qwen3vl/manifest.json).
constexpr int64_t kT = 204;       // expanded prompt length
constexpr int64_t kN = 196;       // merged visual tokens
constexpr int64_t kL = 3;         // deepstack levels
constexpr int64_t kHt = 16;       // reduced hidden for scatter fixtures
constexpr int64_t kDh = 128;      // real rotary_dim
constexpr int64_t kHqT = 4;       // reduced q-heads for the rotary fixture
constexpr int64_t kHkvT = 2;      // reduced kv-heads
constexpr int32_t kImageTok = 151655;
constexpr int32_t kVStart = 151652;
constexpr int32_t kVEnd = 151653;
constexpr int64_t kMerge = 2;

std::string Fix() { return std::string(TEXT_FIXTURE_DIR); }

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

std::vector<int32_t> ReadI32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open: ", path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<int32_t> v(static_cast<size_t>(n) / sizeof(int32_t));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

double RelL2(const std::vector<float>& got, const std::vector<float>& ref) {
  REQUIRE(got.size() == ref.size());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return std::sqrt(num / (den + 1e-30));
}

// The fixed fixture: image at offset 1, grid (1,28,28), N=196 image tokens.
std::vector<int32_t> FixtureInputIds() {
  std::vector<int32_t> ids(static_cast<size_t>(kT), 0);
  ids[0] = kVStart;
  for (int64_t i = 1; i <= kN; ++i) ids[static_cast<size_t>(i)] = kImageTok;
  ids[static_cast<size_t>(1 + kN)] = kVEnd;
  // trailing text ("What is in this image?") — values irrelevant to get_rope_index.
  const int32_t txt[6] = {3838, 374, 304, 419, 2168, 30};
  for (int j = 0; j < 6; ++j) ids[static_cast<size_t>(2 + kN + j)] = txt[j];
  return ids;
}

std::vector<bool> FixtureMask() {
  std::vector<bool> m(static_cast<size_t>(kT), false);
  for (int64_t i = 1; i <= kN; ++i) m[static_cast<size_t>(i)] = true;
  return m;
}

}  // namespace

TEST_CASE("qwen3vl_get_rope_index_bit_exact_vs_vllm") {
  const std::vector<int32_t> ids = FixtureInputIds();
  std::vector<MmImageSpan> images = {{/*offset=*/1, /*grid_thw=*/{1, 28, 28}}};
  int64_t delta = 0;
  std::vector<int32_t> pos = Qwen3VLGetRopeIndex(ids, images, kMerge, &delta);

  const std::vector<int32_t> ref = ReadI32(Fix() + "/rope_index_positions_i32.bin");
  REQUIRE(pos.size() == ref.size());
  REQUIRE(pos.size() == static_cast<size_t>(3 * kT));
  int64_t mismatches = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (pos[i] != ref[i]) ++mismatches;
  CHECK(mismatches == 0);
  CHECK(delta == -182);  // max(21)+1 - 204

  // RED guard: sequential (text-only) positions would differ massively.
  int64_t seq_mismatch = 0;
  for (int r = 0; r < 3; ++r)
    for (int64_t t = 0; t < kT; ++t)
      if (ref[static_cast<size_t>(r * kT + t)] != static_cast<int32_t>(t)) ++seq_mismatch;
  CHECK(seq_mismatch > 0);  // the reference is genuinely NOT sequential
}

TEST_CASE("qwen3vl_mrope_3section_application_vs_vllm") {
  // q [T, HqT, Dh], k [T, HkvT, Dh] (bf16-rounded, widened to f32), positions
  // [3,T] i32, cos_sin cache [Pmax, Dh] (bf16->f32).
  std::vector<float> q = ReadF32(Fix() + "/rotary_q_in_f32.bin");
  std::vector<float> k = ReadF32(Fix() + "/rotary_k_in_f32.bin");
  const std::vector<float> q_ref = ReadF32(Fix() + "/rotary_q_out_f32.bin");
  const std::vector<float> k_ref = ReadF32(Fix() + "/rotary_k_out_f32.bin");
  std::vector<float> cache = ReadF32(Fix() + "/rope_cos_sin_cache_f32.bin");
  std::vector<int32_t> pos = ReadI32(Fix() + "/rope_index_positions_i32.bin");
  const int64_t cache_rows = static_cast<int64_t>(cache.size()) / kDh;
  REQUIRE(q.size() == static_cast<size_t>(kT * kHqT * kDh));
  REQUIRE(k.size() == static_cast<size_t>(kT * kHkvT * kDh));

  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue queue{cpu, nullptr};

  vt::RopeArgs args{/*base=*/5000000.0F, /*rotary_dim=*/static_cast<int>(kDh)};
  args.is_neox_style = true;
  args.mrope_interleaved = true;
  args.mrope_section = {24, 20, 20};

  auto run = [&](std::vector<float> qq, std::vector<float> kk,
                 const vt::RopeArgs& a) {
    vt::Tensor tq = vt::Tensor::Contiguous(qq.data(), vt::DType::kF32, cpu, {kT, kHqT, kDh});
    vt::Tensor tk = vt::Tensor::Contiguous(kk.data(), vt::DType::kF32, cpu, {kT, kHkvT, kDh});
    vt::Tensor tp = vt::Tensor::Contiguous(pos.data(), vt::DType::kI32, cpu, {3, kT});
    vt::Tensor tc = vt::Tensor::Contiguous(cache.data(), vt::DType::kF32, cpu, {cache_rows, kDh});
    vt::RopeFromCache(queue, tq, &tk, tp, tc, a);
    return std::pair<std::vector<float>, std::vector<float>>{std::move(qq), std::move(kk)};
  };

  auto [q_got, k_got] = run(q, k, args);
  const double q_err = RelL2(q_got, q_ref);
  const double k_err = RelL2(k_got, k_ref);
  MESSAGE("MRoPE q rel-L2=", q_err, "  k rel-L2=", k_err);
  // bf16 envelope: our CPU rope is f32-precise, vLLM's reference is bf16-rounded.
  CHECK(q_err < 6e-3);
  CHECK(k_err < 6e-3);

  // RED guard: WRONG mrope layout (contiguous split instead of interleaved) must
  // diverge far outside the bf16 band — proves the gate discriminates the exact
  // section selection Qwen3-VL uses.
  vt::RopeArgs wrong = args;
  wrong.mrope_interleaved = false;
  auto [q_wrong, k_wrong] = run(q, k, wrong);
  CHECK(RelL2(q_wrong, q_ref) > 5e-2);
}

TEST_CASE("qwen3vl_deepstack_scatter_bit_exact_vs_vllm") {
  const std::vector<float> multiscale = ReadF32(Fix() + "/deepstack_multiscale_f32.bin");
  const std::vector<float> ref = ReadF32(Fix() + "/deepstack_out_f32.bin");
  REQUIRE(multiscale.size() == static_cast<size_t>(kN * kL * kHt));
  REQUIRE(ref.size() == static_cast<size_t>(kL * kT * kHt));

  std::vector<float> out =
      Qwen3VLComputeDeepstack(multiscale, kN, kL, kHt, FixtureMask(), kT);
  REQUIRE(out.size() == ref.size());
  CHECK(RelL2(out, ref) == doctest::Approx(0.0));  // bit-exact scatter
  // spot-check: non-visual positions (token 0 = vision_start) are zero.
  for (int64_t l = 0; l < kL; ++l)
    for (int64_t c = 0; c < kHt; ++c)
      CHECK(out[static_cast<size_t>(l * kT * kHt + 0 * kHt + c)] == 0.0F);
}

TEST_CASE("qwen3vl_embed_merge_bit_exact_vs_vllm") {
  std::vector<float> merged = ReadF32(Fix() + "/merge_text_f32.bin");   // [T,Ht]
  const std::vector<float> main = ReadF32(Fix() + "/merge_main_f32.bin");  // [N,Ht]
  const std::vector<float> ref = ReadF32(Fix() + "/merge_out_f32.bin");
  REQUIRE(merged.size() == static_cast<size_t>(kT * kHt));
  REQUIRE(main.size() == static_cast<size_t>(kN * kHt));

  Qwen3VLMergeMultimodal(merged, kT, kHt, main, FixtureMask());
  CHECK(RelL2(merged, ref) == doctest::Approx(0.0));  // bit-exact masked scatter

  // RED guard: an UNMERGED text tensor differs (proves the scatter did work).
  const std::vector<float> text0 = ReadF32(Fix() + "/merge_text_f32.bin");
  CHECK(RelL2(text0, ref) > 1e-3);
}
