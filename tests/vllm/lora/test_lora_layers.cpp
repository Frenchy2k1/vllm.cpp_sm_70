// LoRA-wrapped layer family — packed adapters, merged qkv/gate_up slices, the
// tensor-parallel slice_lora_a/slice_lora_b rules, embedding and logits LoRA.
//
// PORTED FROM (${VLLM_SOURCE} @ 555967922 / vLLM 0.26.0.dev0):
//   tests/lora/test_layers.py:116-142   get_random_id_to_index
//   tests/lora/test_layers.py:145-199   populate_loras (pack + per-slice optimize)
//   tests/lora/test_layers.py:202-244   create_random_inputs
//   tests/lora/test_layers.py:269-364   test_embeddings
//   tests/lora/test_layers.py:372-485   test_lm_head_logits_processor
//   tests/lora/test_layers.py:490-510   test_lm_head_logits_processor_invalid_vocab_size
//   tests/lora/test_layers.py:516-620   test_linear_replicated
//   tests/lora/test_layers.py:629-750   test_linear_parallel (row/column x fully_shard)
//   tests/lora/test_layers.py:762-917   test_column_parallel_packed (repeats 1/2/3)
//   tests/lora/test_layers.py:924-1035  test_merged_column_parallel_variable_slice
//                                       (num_slices 3/5, UNEQUAL widths)
//   tests/lora/utils.py:29-46           DummyLoRAManager.init_random_lora
//                                       (rank 8, alpha 1 => scaling 1/8, uniform [0,1))
//
// Every ported case keeps upstream's second `create_random_inputs` call with
// `active_lora_ids=[0]` (`:340`, `:472`, `:596`, `:894`, `:1010`): after the
// reset the batch is re-drawn as BASE-model tokens, whose slot index is -1
// (utils.py:104-110). That fixture is what exercises the `-1` SKIP path, so
// dropping it would leave the project's own ratified deviation untested. The
// FIRST batch of each case additionally carries base tokens alongside the
// adapters — upstream's `active_lora_ids` is `lora_dict.keys()` there, but the
// expected-result loop already skips `lora_id <= 0`, and a mixed batch with
// LIVE adapter weights is the only arm where a wrongly-applied `-1` delta is
// visible at all.
//
// HARNESS ADAPTATIONS (documented per the port discipline):
//  * Upstream builds a real `ReplicatedLinear`/`QKVParallelLinear` and compares
//    `lora_linear(x)` against `linear(x) + delta`. Our wrapped layer is a DELTA
//    APPLIER over a base-linear output (the base GEMM stays behind
//    `LinearMethodBase`, exactly like `_apply_lora_to_output` receives an
//    already-computed `output`), so the base output is drawn directly instead of
//    being recomputed from a base weight. The delta — everything the layer owns —
//    is checked against an independent double-precision per-adapter reference.
//  * `torch.rand` under `set_random_seed(i)` becomes a deterministic LCG; the
//    role (reproducible uniform [0,1) inputs across NUM_RANDOM_SEEDS=2 seeds) is
//    preserved.
//  * Tolerances are upstream's `TOLERANCES[torch.float32] = (5e-3, 5e-3)`
//    (test_layers.py:53-57) applied as torch.testing.assert_close's
//    |a-b| <= atol + rtol*|b|. Our portable path stores float32, so the float32
//    row is the applicable one.
//  * The tensor-parallel cases run the slicing rules directly at tp_size 2/4
//    rather than under a real process group: `slice_lora_a`/`slice_lora_b` are
//    pure functions of (tp_rank, tp_size) and upstream's own tests only ever
//    exercise them at tp_size 1.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/lora/layers.h"
#include "vllm/lora/lora_weights.h"
#include "vllm/lora/punica.h"

using vllm::lora::ColumnParallelLinearWithLoRA;
using vllm::lora::ColumnParallelLinearWithShardedLoRA;
using vllm::lora::LogitsProcessorWithLoRA;
using vllm::lora::LoRALayerWeights;
using vllm::lora::LoRAMat;
using vllm::lora::MatList;
using vllm::lora::MergedColumnParallelLinearWithLoRA;
using vllm::lora::MergedColumnParallelLinearVariableSliceWithLoRA;
using vllm::lora::MergedColumnParallelLinearWithShardedLoRA;
using vllm::lora::MergedQKVParallelLinearWithLoRA;
using vllm::lora::MergedQKVParallelLinearWithShardedLoRA;
using vllm::lora::PackedLoRALayerWeights;
using vllm::lora::QKVParallelLinearWithLoRA;
using vllm::lora::ReplicatedLinearWithLoRA;
using vllm::lora::RowParallelLinearWithLoRA;
using vllm::lora::RowParallelLinearWithShardedLoRA;
using vllm::lora::VocabParallelEmbeddingWithLoRA;

namespace {

// ---------------------------------------------------------------------------
// Harness

// Deterministic uniform [0,1). Stands in for torch.rand under set_random_seed.
class Rng {
 public:
  explicit Rng(uint32_t seed) : s_(seed * 2654435761u + 12345u) {}
  float Next() {
    s_ = s_ * 1664525u + 1013904223u;
    return static_cast<float>(s_ >> 8) / 16777216.0f;
  }
  std::vector<float> Vec(int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& e : v) e = Next();
    return v;
  }
  int64_t Int(int64_t lo, int64_t hi) {  // [lo, hi)
    return lo + static_cast<int64_t>(Next() * static_cast<float>(hi - lo));
  }

 private:
  uint32_t s_;
};

LoRAMat RandMat(Rng& r, int64_t rows, int64_t cols) {
  LoRAMat m;
  m.rows = rows;
  m.cols = cols;
  m.data = r.Vec(rows * cols);
  return m;
}

// TOLERANCES[torch.float32] (test_layers.py:53-57) under torch.testing's
// |got - ref| <= atol + rtol * |ref|.
constexpr double kRtol = 5e-3;
constexpr double kAtol = 5e-3;

// One CHECK for the whole tensor (a per-element CHECK would emit millions of
// assertions); reports the worst violation so a failure is diagnosable.
void CheckAllClose(const std::vector<float>& got, const std::vector<double>& ref) {
  REQUIRE(got.size() == ref.size());
  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double slack =
        std::abs(static_cast<double>(got[i]) - ref[i]) - (kAtol + kRtol * std::abs(ref[i]));
    if (slack > worst) {
      worst = slack;
      worst_i = i;
    }
  }
  if (worst > 0.0) {
    MESSAGE("worst element " << worst_i << ": got " << got[worst_i] << " ref "
                             << ref[worst_i]);
  }
  CHECK(worst <= 0.0);
}

// get_random_id_to_index (test_layers.py:116-142). Slot -> lora id (1-based),
// 0 == free slot (upstream's None).
std::vector<int> GetRandomIdToIndex(int num_loras, int num_slots, Rng& rng) {
  REQUIRE(num_loras <= num_slots);
  std::vector<int> slots(static_cast<size_t>(num_slots), 0);
  // Fisher-Yates over the slot indices == torch.randperm(num_slots)[:num_loras].
  std::vector<int> perm(static_cast<size_t>(num_slots));
  for (int i = 0; i < num_slots; ++i) perm[static_cast<size_t>(i)] = i;
  for (int i = num_slots - 1; i > 0; --i) {
    const int j = static_cast<int>(rng.Int(0, i + 1));
    std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
  }
  for (int lora_id = 1; lora_id <= num_loras; ++lora_id) {
    slots[static_cast<size_t>(perm[static_cast<size_t>(lora_id - 1)])] = lora_id;
  }
  return slots;
}

// One adapter as populate_loras builds it: `repeats` sub-LoRAs, each already
// optimize()d so its scaling is folded into lora_b (test_layers.py:174-186).
struct AdapterGroup {
  std::vector<LoRALayerWeights> subloras;
  MatList a;  // per sub-module lora_a  [rank, in]
  MatList b;  // per sub-module lora_b  [out_i, rank]
};

// DummyLoRAManager.init_random_lora (utils.py:29-46) + the populate_loras row
// slice + optimize(). `weight_rows`/`weight_cols` are the base layer weight
// shape upstream passes as `layer_weights`.
AdapterGroup MakeAdapterGroup(Rng& rng, int64_t weight_rows, int64_t weight_cols,
                              int repeats, int rank = 8) {
  AdapterGroup g;
  const int64_t sublora_len = weight_rows / repeats;
  for (int i = 0; i < repeats; ++i) {
    LoRAMat a = RandMat(rng, rank, weight_cols);
    LoRAMat full_b = RandMat(rng, weight_rows, rank);
    LoRAMat b;
    b.rows = sublora_len;
    b.cols = rank;
    b.data.assign(full_b.data.begin() + static_cast<long>(sublora_len * i * rank),
                  full_b.data.begin() + static_cast<long>(sublora_len * (i + 1) * rank));
    LoRALayerWeights w("fake_" + std::to_string(i), rank, /*alpha=*/1, a.data, b.data,
                       static_cast<int>(weight_cols), static_cast<int>(sublora_len));
    w.Optimize();  // scaling 1/rank folded into lora_b => scaling == 1
    g.subloras.push_back(w);
  }
  for (const auto& w : g.subloras) {
    LoRAMat a;
    a.rows = w.rank;
    a.cols = w.input_dim;
    a.data = w.lora_a;
    LoRAMat b;
    b.rows = w.output_dim;
    b.cols = w.rank;
    b.data = w.lora_b;
    g.a.push_back(a);
    g.b.push_back(b);
  }
  return g;
}

// create_random_inputs (test_layers.py:202-244): num_inputs rows, each assigned
// a random active lora id; index_mapping repeats the id per token.
struct Batch {
  std::vector<float> x;         // [T, input_size]
  std::vector<int32_t> slots;   // [T] slot index, -1 == no adapter
  std::vector<int> lora_ids;    // [T] lora id (0 == base model)
  int64_t T = 0;
};

Batch MakeBatch(Rng& rng, const std::vector<int>& active_lora_ids,
                const std::vector<int>& id_to_index, int64_t num_inputs,
                int64_t input_size) {
  Batch batch;
  batch.T = num_inputs;
  batch.x = rng.Vec(num_inputs * input_size);
  for (int64_t t = 0; t < num_inputs; ++t) {
    const int lora_id =
        active_lora_ids[static_cast<size_t>(rng.Int(0, static_cast<int64_t>(active_lora_ids.size())))];
    batch.lora_ids.push_back(lora_id);
    int32_t slot = -1;  // convert_mapping: id <= 0 -> -1 (utils.py:104-110)
    for (size_t s = 0; s < id_to_index.size(); ++s) {
      if (lora_id > 0 && id_to_index[s] == lora_id) slot = static_cast<int32_t>(s);
    }
    batch.slots.push_back(slot);
  }
  return batch;
}

// `active_lora_ids` for the first arm: every loaded adapter PLUS the base model
// (id 0 -> slot -1), so the `-1` SKIP path runs against live weights.
std::vector<int> WithBaseModel(std::vector<int> active_ids) {
  active_ids.push_back(0);
  return active_ids;
}

// The reset arm's `create_random_inputs(active_lora_ids=[0], ...)`: a fresh
// batch of base-model tokens, every slot -1.
Batch MakeBaseBatch(Rng& rng, const std::vector<int>& id_to_index,
                    int64_t num_inputs, int64_t input_size) {
  Batch batch = MakeBatch(rng, {0}, id_to_index, num_inputs, input_size);
  for (int32_t slot : batch.slots) REQUIRE(slot == -1);
  return batch;
}

// Independent double reference for one token/slice:
//   y[off : off+out_i] += x[t] @ a_i^T @ b_i^T * scaling
void RefAddSlice(std::vector<double>& ref, int64_t t, int64_t y_width, int64_t offset,
                 const std::vector<float>& x, int64_t in_dim, const LoRAMat& a,
                 const LoRAMat& b, double scaling) {
  std::vector<double> shrunk(static_cast<size_t>(a.rows), 0.0);
  for (int64_t r = 0; r < a.rows; ++r) {
    double acc = 0.0;
    for (int64_t i = 0; i < in_dim; ++i) {
      acc += static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]) *
             static_cast<double>(a.data[static_cast<size_t>(r * a.cols + i)]);
    }
    shrunk[static_cast<size_t>(r)] = acc * scaling;
  }
  for (int64_t o = 0; o < b.rows; ++o) {
    double acc = 0.0;
    for (int64_t r = 0; r < b.cols; ++r) {
      acc += shrunk[static_cast<size_t>(r)] *
             static_cast<double>(b.data[static_cast<size_t>(o * b.cols + r)]);
    }
    ref[static_cast<size_t>(t * y_width + offset + o)] += acc;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// PackedLoRALayerWeights (lora_weights.py:99-282)

TEST_CASE("PackedLoRALayerWeights::Pack folds scaling and keeps per-slice weights") {
  // lora_weights.py:126-152 — pack() optimize()s every sub-LoRA, takes rank and
  // module_name from the first present one, and sets every present slice's
  // scaling to 1 (absent slices stay absent).
  LoRALayerWeights q("qkv_proj", /*rank=*/2, /*alpha=*/4, /*a=*/{1.0f, 2.0f, 3.0f, 4.0f},
                     /*b=*/{1.0f, 1.0f}, /*in=*/2, /*out=*/1);  // scaling 2.0
  LoRALayerWeights v("qkv_proj", /*rank=*/2, /*alpha=*/2, /*a=*/{5.0f, 6.0f, 7.0f, 8.0f},
                     /*b=*/{2.0f, 3.0f}, /*in=*/2, /*out=*/1);  // scaling 1.0

  std::vector<std::optional<LoRALayerWeights>> loras = {q, std::nullopt, v};
  PackedLoRALayerWeights packed = PackedLoRALayerWeights::Pack(loras);

  CHECK(packed.is_packed());
  CHECK(packed.rank == 2);
  CHECK(packed.module_name == "qkv_proj");
  REQUIRE(packed.subloras.size() == 3);
  CHECK(packed.subloras[0].has_value());
  CHECK_FALSE(packed.subloras[1].has_value());  // the "None" sub-module
  CHECK(packed.subloras[2].has_value());

  // q's scaling (alpha/rank = 2) folded into lora_b; scaling now 1.
  CHECK(packed.subloras[0]->scaling == doctest::Approx(1.0));
  CHECK(packed.subloras[0]->lora_b[0] == doctest::Approx(2.0f));
  CHECK(packed.subloras[0]->lora_b[1] == doctest::Approx(2.0f));
  // v's scaling was already 1 => optimize() is a no-op.
  CHECK(packed.subloras[2]->lora_b[0] == doctest::Approx(2.0f));
  CHECK(packed.subloras[2]->lora_b[1] == doctest::Approx(3.0f));

  // Packed::Optimize (lora_weights.py:263-270) is idempotent after Pack.
  auto before = packed.subloras[0]->lora_b;
  packed.Optimize();
  CHECK(packed.subloras[0]->lora_b == before);
}

// ---------------------------------------------------------------------------
// test_linear_replicated (test_layers.py:516-620)

TEST_CASE("test_linear_replicated: batched apply then reset to base") {
  constexpr int64_t kInput = 512;
  constexpr int64_t kOutput = 512;
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int num_loras : {1, 2, 4}) {
    for (int seed = 0; seed < 2; ++seed) {  // NUM_RANDOM_SEEDS
      Rng rng(static_cast<uint32_t>(seed * 31 + num_loras));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      ReplicatedLinearWithLoRA layer(kInput, kOutput);
      layer.CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/false);
      REQUIRE(layer.n_slices() == 1);
      REQUIRE(layer.output_slices().size() == 1);

      std::vector<AdapterGroup> groups;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, kOutput, kInput, /*repeats=*/1);
        layer.SetLora(static_cast<int64_t>(slot), g.a, g.b);
        groups.push_back(g);
        active_ids.push_back(id_to_index[slot]);
      }

      Batch batch = MakeBatch(rng, WithBaseModel(active_ids), id_to_index,
                              32 * num_loras, kInput);
      std::vector<float> base = rng.Vec(batch.T * kOutput);
      std::vector<float> y = base;
      layer.ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < batch.T; ++t) {
        const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
        if (lora_id <= 0) continue;
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == lora_id) gi = k;
        }
        const AdapterGroup& g = groups[gi];
        RefAddSlice(ref, t, kOutput, 0, batch.x, kInput, g.a[0], g.b[0],
                    g.subloras[0].scaling);
      }
      CheckAllClose(y, ref);

      // "Check that resetting the lora weights succeeds" (test_layers.py:596-620),
      // re-drawn with active_lora_ids=[0]: every slot is -1.
      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
      Batch base_batch = MakeBaseBatch(rng, id_to_index, 32 * num_loras, kInput);
      std::vector<float> base2 = rng.Vec(base_batch.T * kOutput);
      std::vector<float> y2 = base2;
      layer.ApplyLoraToOutput(y2.data(), base_batch.x.data(), base_batch.T,
                              base_batch.slots.data());
      CHECK(y2 == base2);
    }
  }
}

// ---------------------------------------------------------------------------
// test_linear_parallel (test_layers.py:629-750)

TEST_CASE("test_linear_parallel: row and column orientation, fully_shard on/off") {
  constexpr int64_t kSize = 512;  // upstream 4096; the shape structure is what matters
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (const std::string orientation : {"row", "column"}) {
    for (bool fully_shard : {false, true}) {
      for (int num_loras : {1, 2, 4}) {
        Rng rng(static_cast<uint32_t>(num_loras * 7 + (fully_shard ? 3 : 0) +
                                      (orientation == "row" ? 100u : 200u)));
        auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

        std::unique_ptr<vllm::lora::BaseLinearLayerWithLoRA> layer;
        if (orientation == "row") {
          if (fully_shard) {
            layer = std::make_unique<RowParallelLinearWithShardedLoRA>(kSize, kSize, 1, 0);
          } else {
            layer = std::make_unique<RowParallelLinearWithLoRA>(kSize, kSize, 1, 0);
          }
        } else {
          if (fully_shard) {
            layer = std::make_unique<ColumnParallelLinearWithShardedLoRA>(kSize, kSize, 1, 0);
          } else {
            layer = std::make_unique<ColumnParallelLinearWithLoRA>(kSize, kSize, 1, 0);
          }
        }
        layer->CreateLoraWeights(kMaxLoras, kMaxRank, fully_shard);
        // test_layers.py:677-682 — n_slices == len(a_stacked) == len(b_stacked) == 1
        REQUIRE(layer->n_slices() == 1);

        std::vector<AdapterGroup> groups;
        std::vector<int> active_ids;
        for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
          if (id_to_index[slot] == 0) continue;
          AdapterGroup g = MakeAdapterGroup(rng, kSize, kSize, /*repeats=*/1);
          layer->SetLora(static_cast<int64_t>(slot), g.a, g.b);
          groups.push_back(g);
          active_ids.push_back(id_to_index[slot]);
        }

        Batch batch = MakeBatch(rng, WithBaseModel(active_ids), id_to_index,
                                32 * num_loras, kSize);
        std::vector<float> base = rng.Vec(batch.T * kSize);
        std::vector<float> y = base;
        layer->ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

        std::vector<double> ref(base.begin(), base.end());
        for (int64_t t = 0; t < batch.T; ++t) {
          const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
          if (lora_id <= 0) continue;
          size_t gi = 0;
          for (size_t k = 0; k < active_ids.size(); ++k) {
            if (active_ids[k] == lora_id) gi = k;
          }
          RefAddSlice(ref, t, kSize, 0, batch.x, kSize, groups[gi].a[0], groups[gi].b[0],
                      groups[gi].subloras[0].scaling);
        }
        CheckAllClose(y, ref);

        // test_layers.py:735-750 — reset, then a fresh active_lora_ids=[0] batch.
        for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer->ResetLora(slot);
        Batch base_batch = MakeBaseBatch(rng, id_to_index, 32 * num_loras, kSize);
        std::vector<float> base2 = rng.Vec(base_batch.T * kSize);
        std::vector<float> y2 = base2;
        layer->ApplyLoraToOutput(y2.data(), base_batch.x.data(), base_batch.T,
                                 base_batch.slots.data());
        CHECK(y2 == base2);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// test_column_parallel_packed (test_layers.py:762-917)

TEST_CASE("test_column_parallel_packed: repeats 1 (qkv), 2 (gate_up), 3 (merged qkv)") {
  constexpr int64_t kInput = 512;
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;
  // QKVParallelLinear(4096, head_size=64, total_num_heads=32) scaled down: the
  // three slices are q/k/v with head_size * num_heads each.
  constexpr int64_t kHeadSize = 16;
  constexpr int64_t kNumHeads = 8;

  for (int repeats : {1, 2, 3}) {
    for (int num_loras : {1, 2, 4}) {
      Rng rng(static_cast<uint32_t>(repeats * 101 + num_loras));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      std::unique_ptr<vllm::lora::BaseLinearLayerWithLoRA> layer;
      int64_t total_out = 0;
      if (repeats == 2) {
        // MergedColumnParallelLinear(input, [out] * 2) -> gate_up_proj.
        layer = std::make_unique<MergedColumnParallelLinearWithLoRA>(
            kInput, std::vector<int64_t>{kInput, kInput}, 1, 0);
        total_out = 2 * kInput;
      } else if (repeats == 3) {
        layer = std::make_unique<MergedQKVParallelLinearWithLoRA>(
            kInput, kHeadSize, kNumHeads, kNumHeads, kNumHeads, kNumHeads, 1, 0);
        total_out = 3 * kHeadSize * kNumHeads;
      } else {
        layer = std::make_unique<QKVParallelLinearWithLoRA>(
            kInput, kHeadSize, kNumHeads, kNumHeads, kNumHeads, kNumHeads, 1, 0);
        total_out = 3 * kHeadSize * kNumHeads;
      }
      layer->CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/false);
      REQUIRE(layer->n_slices() == repeats);
      REQUIRE(static_cast<int>(layer->output_slices().size()) == repeats);

      std::vector<AdapterGroup> groups;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, total_out, kInput, repeats);
        layer->SetLora(static_cast<int64_t>(slot), g.a, g.b);
        groups.push_back(g);
        active_ids.push_back(id_to_index[slot]);
      }

      Batch batch = MakeBatch(rng, WithBaseModel(active_ids), id_to_index,
                              32 * num_loras, kInput);
      std::vector<float> base = rng.Vec(batch.T * total_out);
      std::vector<float> y = base;
      layer->ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

      // test_layers.py:889-897 — each sublora writes its own output window.
      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < batch.T; ++t) {
        const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
        if (lora_id <= 0) continue;
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == lora_id) gi = k;
        }
        const AdapterGroup& g = groups[gi];
        for (int i = 0; i < repeats; ++i) {
          const int64_t offset = g.b[static_cast<size_t>(i)].rows * i;
          RefAddSlice(ref, t, total_out, offset, batch.x, kInput, g.a[static_cast<size_t>(i)],
                      g.b[static_cast<size_t>(i)],
                      g.subloras[static_cast<size_t>(i)].scaling);
        }
      }
      CheckAllClose(y, ref);

      // test_layers.py:894-917 — reset, then a fresh active_lora_ids=[0] batch.
      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer->ResetLora(slot);
      Batch base_batch = MakeBaseBatch(rng, id_to_index, 32 * num_loras, kInput);
      std::vector<float> base2 = rng.Vec(base_batch.T * total_out);
      std::vector<float> y2 = base2;
      layer->ApplyLoraToOutput(y2.data(), base_batch.x.data(), base_batch.T,
                               base_batch.slots.data());
      CHECK(y2 == base2);
    }
  }
}

// ---------------------------------------------------------------------------
// Tensor-parallel slicing (column_parallel_linear.py / row_parallel_linear.py)

TEST_CASE("slice_lora_a / slice_lora_b under tensor parallel") {
  constexpr int64_t kRank = 4;

  SUBCASE("RowParallelLinearWithLoRA slices lora_a along the input dim") {
    // row_parallel_linear.py:32-37 — a[:, rank*shard : (rank+1)*shard].
    Rng rng(1);
    const int64_t tp = 4, in_per_partition = 8;
    RowParallelLinearWithLoRA layer(in_per_partition, /*output_size=*/6, tp, /*tp_rank=*/2);
    LoRAMat a = RandMat(rng, kRank, in_per_partition * tp);
    MatList sliced = layer.SliceLoraA({a});
    REQUIRE(sliced.size() == 1);
    CHECK(sliced[0].rows == kRank);
    CHECK(sliced[0].cols == in_per_partition);
    for (int64_t r = 0; r < kRank; ++r) {
      for (int64_t c = 0; c < in_per_partition; ++c) {
        CHECK(sliced[0].At(r, c) == a.At(r, 2 * in_per_partition + c));
      }
    }
    // slice_lora_b is the identity for row-parallel (row_parallel_linear.py:39-40).
    LoRAMat b = RandMat(rng, 6, kRank);
    CHECK(layer.SliceLoraB({b})[0].data == b.data);
  }

  SUBCASE("ColumnParallelLinearWithLoRA slices lora_b along the output dim") {
    // column_parallel_linear.py:125-129.
    Rng rng(2);
    const int64_t tp = 4, out_per_partition = 5;
    ColumnParallelLinearWithLoRA layer(/*input_size=*/8, out_per_partition, tp, 3);
    LoRAMat b = RandMat(rng, out_per_partition * tp, kRank);
    MatList sliced = layer.SliceLoraB({b});
    CHECK(sliced[0].rows == out_per_partition);
    for (int64_t r = 0; r < out_per_partition; ++r) {
      for (int64_t c = 0; c < kRank; ++c) {
        CHECK(sliced[0].At(r, c) == b.At(3 * out_per_partition + r, c));
      }
    }
    // slice_lora_a is the identity (column_parallel_linear.py:104-105).
    LoRAMat a = RandMat(rng, kRank, 8);
    CHECK(layer.SliceLoraA({a})[0].data == a.data);
  }

  SUBCASE("ColumnParallelLinearWithLoRA takes both halves for a merged base layer") {
    // column_parallel_linear.py:110-122 — is_merged_col_linear concatenates the
    // rank's shard of the left half and of the right half.
    Rng rng(3);
    const int64_t tp = 2, out_per_partition = 6;  // shard_size = 3 per half
    ColumnParallelLinearWithLoRA layer(/*input_size=*/8, out_per_partition, tp, /*tp_rank=*/1,
                                       /*is_merged_col_linear=*/true);
    LoRAMat b = RandMat(rng, out_per_partition * tp, kRank);  // 12 rows: two halves of 6
    MatList sliced = layer.SliceLoraB({b});
    REQUIRE(sliced[0].rows == out_per_partition);
    const int64_t shard = out_per_partition / 2;  // 3
    for (int64_t r = 0; r < shard; ++r) {
      for (int64_t c = 0; c < kRank; ++c) {
        CHECK(sliced[0].At(r, c) == b.At(1 * shard + r, c));
        CHECK(sliced[0].At(shard + r, c) == b.At(6 + 1 * shard + r, c));
      }
    }
  }

  SUBCASE("QKVParallelLinearWithLoRA concatenates the q, k and v shards") {
    // column_parallel_linear.py:399-420.
    Rng rng(4);
    const int64_t tp = 2, head_size = 4, total_heads = 8, total_kv_heads = 8;
    const int64_t heads = total_heads / tp, kv_heads = total_kv_heads / tp;
    QKVParallelLinearWithLoRA layer(/*input_size=*/8, head_size, total_heads, heads,
                                    total_kv_heads, kv_heads, tp, /*tp_rank=*/1);
    const int64_t q_total = total_heads * head_size;      // 32
    const int64_t kv_total = total_kv_heads * head_size;  // 32
    LoRAMat b = RandMat(rng, q_total + 2 * kv_total, kRank);
    MatList sliced = layer.SliceLoraB({b});
    const int64_t q_shard = heads * head_size;      // 16
    const int64_t kv_shard = kv_heads * head_size;  // 16
    REQUIRE(sliced[0].rows == q_shard + 2 * kv_shard);
    for (int64_t r = 0; r < q_shard; ++r) {
      CHECK(sliced[0].At(r, 0) == b.At(q_shard + r, 0));
    }
    for (int64_t r = 0; r < kv_shard; ++r) {
      CHECK(sliced[0].At(q_shard + r, 0) == b.At(q_total + kv_shard + r, 0));
      CHECK(sliced[0].At(q_shard + kv_shard + r, 0) ==
            b.At(q_total + kv_total + kv_shard + r, 0));
    }
  }

  SUBCASE("MergedColumnParallelLinearWithLoRA slices every sub-module") {
    // column_parallel_linear.py:253-264.
    Rng rng(5);
    const int64_t tp = 2;
    MergedColumnParallelLinearWithLoRA layer(/*input_size=*/8,
                                             std::vector<int64_t>{8, 8}, tp, /*tp_rank=*/1);
    LoRAMat b0 = RandMat(rng, 8, kRank);
    LoRAMat b1 = RandMat(rng, 8, kRank);
    MatList sliced = layer.SliceLoraB({b0, b1});
    REQUIRE(sliced.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
      CHECK(sliced[i].rows == 4);
      const LoRAMat& src = i == 0 ? b0 : b1;
      for (int64_t r = 0; r < 4; ++r) {
        CHECK(sliced[i].At(r, 0) == src.At(4 + r, 0));
      }
    }
  }

  SUBCASE("Fully-sharded (S-LoRA) variants shard lora_a along the rank dim") {
    // column_parallel_linear.py:516-520 (column), :553-563 (merged column),
    // :632-649 (merged qkv); row_parallel_linear.py:111-116 (b, not a).
    Rng rng(6);
    const int64_t tp = 2, max_rank = 8;
    ColumnParallelLinearWithShardedLoRA col(/*input_size=*/8, /*output_size=*/8, tp, 1);
    col.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    CHECK(col.lora_a_rows() == max_rank / tp);
    LoRAMat a = RandMat(rng, max_rank, 8);
    MatList sliced = col.SliceLoraA({a});
    CHECK(sliced[0].rows == max_rank / tp);
    for (int64_t r = 0; r < max_rank / tp; ++r) {
      CHECK(sliced[0].At(r, 0) == a.At(max_rank / tp + r, 0));
    }

    RowParallelLinearWithShardedLoRA row(/*input_size=*/8, /*output_size=*/8, tp, 1);
    row.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    // row_parallel_linear.py:120-125 — lora_b is sharded along the output dim.
    LoRAMat b = RandMat(rng, 8, max_rank);
    MatList rsliced = row.SliceLoraB({b});
    CHECK(rsliced[0].rows == 4);
    for (int64_t r = 0; r < 4; ++r) CHECK(rsliced[0].At(r, 0) == b.At(4 + r, 0));

    MergedColumnParallelLinearWithShardedLoRA mcol(
        /*input_size=*/8, std::vector<int64_t>{8, 8}, tp, 1);
    mcol.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    LoRAMat a0 = RandMat(rng, max_rank, 8);
    LoRAMat a1 = RandMat(rng, max_rank, 8);
    MatList msliced = mcol.SliceLoraA({a0, a1});
    REQUIRE(msliced.size() == 2);
    CHECK(msliced[0].rows == max_rank / tp);
    CHECK(msliced[1].rows == max_rank / tp);

    MergedQKVParallelLinearWithShardedLoRA mqkv(/*input_size=*/8, /*head_size=*/4,
                                                /*total_num_heads=*/8, /*num_heads=*/4,
                                                /*total_num_kv_heads=*/8, /*num_kv_heads=*/4,
                                                tp, 1);
    mqkv.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    LoRAMat qa = RandMat(rng, max_rank, 8);
    MatList qsliced = mqkv.SliceLoraA({qa, qa, qa});
    REQUIRE(qsliced.size() == 3);
    for (const auto& m : qsliced) CHECK(m.rows == max_rank / tp);
  }
}

// ---------------------------------------------------------------------------
// test_embeddings (test_layers.py:269-364)

TEST_CASE("test_embeddings: lora_a is an embedding lookup, lora_b an expand") {
  constexpr int64_t kEmbedDim = 64;  // upstream 256
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int64_t vocab_size : {512, 32000}) {  // upstream also 64000/128000
    for (int num_loras : {1, 2, 4}) {
      Rng rng(static_cast<uint32_t>(vocab_size + num_loras));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      VocabParallelEmbeddingWithLoRA layer(vocab_size, kEmbedDim);
      layer.CreateLoraWeights(kMaxLoras, kMaxRank);

      // populate_loras(layer_weights=embedding.weight.T) => lora_a [rank, vocab],
      // lora_b [embedding_dim, rank] (test_layers.py:303-307).
      std::vector<LoRAMat> a_mats, b_mats;
      std::vector<double> scalings;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, kEmbedDim, vocab_size, /*repeats=*/1);
        layer.SetLora(static_cast<int64_t>(slot), g.a[0], g.b[0]);
        a_mats.push_back(g.a[0]);
        b_mats.push_back(g.b[0]);
        scalings.push_back(g.subloras[0].scaling);
        active_ids.push_back(id_to_index[slot]);
      }

      const int64_t T = 64;
      // active_lora_ids = the loaded adapters PLUS the base model (id 0),
      // whose slot is -1: the gather pins it to slot 0 and the expand skips
      // it, so its embedding row must come back untouched.
      const std::vector<int> batch_ids = WithBaseModel(active_ids);
      std::vector<int32_t> tokens(static_cast<size_t>(T));
      std::vector<int32_t> slots(static_cast<size_t>(T));
      std::vector<int> lora_ids(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t) {
        tokens[static_cast<size_t>(t)] = static_cast<int32_t>(rng.Int(1, vocab_size));
        const int lora_id =
            batch_ids[static_cast<size_t>(rng.Int(0, static_cast<int64_t>(batch_ids.size())))];
        lora_ids[static_cast<size_t>(t)] = lora_id;
        int32_t slot = -1;
        for (size_t s = 0; s < id_to_index.size(); ++s) {
          if (id_to_index[s] == lora_id) slot = static_cast<int32_t>(s);
        }
        slots[static_cast<size_t>(t)] = slot;
      }

      std::vector<float> base = rng.Vec(T * kEmbedDim);
      std::vector<float> y = base;
      layer.ApplyLoraToOutput(y.data(), tokens.data(), T, slots.data());

      // Reference (test_layers.py:325-332):
      //   after_a = F.embedding(input_, lora_a.T);  result += after_a @ lora_b.T
      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < T; ++t) {
        if (lora_ids[static_cast<size_t>(t)] <= 0) continue;  // base token: no delta
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == lora_ids[static_cast<size_t>(t)]) gi = k;
        }
        const LoRAMat& a = a_mats[gi];
        const LoRAMat& b = b_mats[gi];
        const int64_t tok = tokens[static_cast<size_t>(t)];
        for (int64_t o = 0; o < kEmbedDim; ++o) {
          double acc = 0.0;
          for (int64_t r = 0; r < a.rows; ++r) {
            acc += static_cast<double>(a.At(r, tok)) *
                   static_cast<double>(b.At(o, r)) * scalings[gi];
          }
          ref[static_cast<size_t>(t * kEmbedDim + o)] += acc;
        }
      }
      CheckAllClose(y, ref);

      // test_layers.py:334-364 — reset, then a fresh active_lora_ids=[0]
      // batch: every slot -1.
      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
      std::vector<int32_t> base_slots(static_cast<size_t>(T), -1);
      std::vector<float> base2 = rng.Vec(T * kEmbedDim);
      std::vector<float> y2 = base2;
      layer.ApplyLoraToOutput(y2.data(), tokens.data(), T, base_slots.data());
      CHECK(y2 == base2);
    }
  }
}

// ---------------------------------------------------------------------------
// test_lm_head_logits_processor (test_layers.py:372-510)

TEST_CASE("test_lm_head_logits_processor: sampler-indexed shrink/expand onto logits") {
  constexpr int64_t kHidden = 128;  // upstream 1024
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int64_t vocab_size : {6400, 25856}) {  // upstream 64000/256512/258048
    for (int num_loras : {1, 2, 4}) {
      Rng rng(static_cast<uint32_t>(vocab_size + num_loras * 3));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      LogitsProcessorWithLoRA layer(vocab_size, kHidden);
      layer.CreateLoraWeights(kMaxLoras, kMaxRank);

      std::vector<LoRAMat> a_mats, b_mats;
      std::vector<double> scalings;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, vocab_size, kHidden, /*repeats=*/1);
        layer.SetLora(static_cast<int64_t>(slot), g.a[0], g.b[0]);
        a_mats.push_back(g.a[0]);
        b_mats.push_back(g.b[0]);
        scalings.push_back(g.subloras[0].scaling);
        active_ids.push_back(id_to_index[slot]);
      }

      Batch batch = MakeBatch(rng, WithBaseModel(active_ids), id_to_index,
                              8 * num_loras, kHidden);
      std::vector<float> base = rng.Vec(batch.T * vocab_size);
      std::vector<float> y = base;
      layer.ApplyLoraToLogits(y.data(), vocab_size, batch.x.data(), batch.T,
                              batch.slots.data());

      // result += input_ @ lora_a.T @ lora_b.T * lora.scaling (test_layers.py:446).
      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < batch.T; ++t) {
        if (batch.lora_ids[static_cast<size_t>(t)] <= 0) continue;  // base request
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == batch.lora_ids[static_cast<size_t>(t)]) gi = k;
        }
        RefAddSlice(ref, t, vocab_size, 0, batch.x, kHidden, a_mats[gi], b_mats[gi],
                    scalings[gi]);
      }
      CheckAllClose(y, ref);

      // test_layers.py:462-485 — reset, then a fresh active_lora_ids=[0] batch.
      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
      Batch base_batch = MakeBaseBatch(rng, id_to_index, 8 * num_loras, kHidden);
      std::vector<float> base2 = rng.Vec(base_batch.T * vocab_size);
      std::vector<float> y2 = base2;
      layer.ApplyLoraToLogits(y2.data(), vocab_size, base_batch.x.data(),
                              base_batch.T, base_batch.slots.data());
      CHECK(y2 == base2);
    }
  }
}

TEST_CASE("test_lm_head_logits_processor_invalid_vocab_size") {
  // logits_processor.py:90-92 — "When using LoRA, vocab size must be <= 258048".
  for (int64_t vocab_size : {258049, 300000}) {
    LogitsProcessorWithLoRA layer(vocab_size, /*hidden_size=*/128);
    CHECK_THROWS_WITH_AS(layer.CreateLoraWeights(/*max_loras=*/8, /*max_lora_rank=*/8),
                         "When using LoRA, vocab size must be <= 258048",
                         std::invalid_argument);
  }
  // 258048 itself is legal (the guard is strictly greater-than).
  LogitsProcessorWithLoRA ok(258048, /*hidden_size=*/8);
  CHECK_NOTHROW(ok.CreateLoraWeights(/*max_loras=*/1, /*max_lora_rank=*/1));
}

// ---------------------------------------------------------------------------
// test_merged_column_parallel_variable_slice (test_layers.py:924-1035)

TEST_CASE("test_merged_column_parallel_variable_slice: one fused checkpoint tensor "
          "split across UNEQUAL slice widths") {
  // The only upstream case whose output windows differ in width, and the only
  // one that drives MergedColumnParallelLinearVariableSliceWithLoRA. Every
  // other ported case has equal slices, which makes the expand's per-slice
  // offset walk (punica_cpu.py:225-236) indistinguishable from a constant
  // stride.
  constexpr int64_t kInput = 256;  // upstream 4096
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int num_slices : {3, 5}) {
    for (int num_loras : {1, 2, 4}) {
      for (int seed = 0; seed < 2; ++seed) {  // NUM_RANDOM_SEEDS
        Rng rng(static_cast<uint32_t>(num_slices * 977 + num_loras * 13 + seed));
        auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

        // `output_sizes = [1024 + i * 256 for i in range(num_slices)]` at 1/8
        // scale: strictly increasing, so no two windows share a width.
        std::vector<int64_t> output_sizes;
        int64_t total_output = 0;
        for (int i = 0; i < num_slices; ++i) {
          output_sizes.push_back(128 + i * 32);
          total_output += output_sizes.back();
        }

        MergedColumnParallelLinearVariableSliceWithLoRA layer(
            kInput, output_sizes, /*tp_size=*/1, /*tp_rank=*/0);
        layer.CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/false);
        REQUIRE(layer.n_slices() == num_slices);
        REQUIRE(layer.output_slices() == output_sizes);

        // `lora_a = torch.rand(8, 4096)`, `lora_b = torch.rand(total_output, 8)`
        // — one fused tensor each, no per-slice split and no optimize() call,
        // so the scaling is 1 (test_layers.py:963-970).
        std::vector<LoRAMat> a_mats, b_mats;
        std::vector<int> active_ids;
        for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
          if (id_to_index[slot] == 0) continue;
          LoRAMat a = RandMat(rng, kMaxRank, kInput);
          LoRAMat b = RandMat(rng, total_output, kMaxRank);
          layer.SetLora(static_cast<int64_t>(slot), {a}, {b});
          a_mats.push_back(a);
          b_mats.push_back(b);
          active_ids.push_back(id_to_index[slot]);
        }

        Batch batch = MakeBatch(rng, WithBaseModel(active_ids), id_to_index,
                                32 * num_loras, kInput);
        std::vector<float> base = rng.Vec(batch.T * total_output);
        std::vector<float> y = base;
        layer.ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

        // test_layers.py:991-1001 — `torch.split(lora_b, output_sizes, dim=0)`,
        // each piece writing its own window with the offset walking by ITS size.
        std::vector<double> ref(base.begin(), base.end());
        for (int64_t t = 0; t < batch.T; ++t) {
          const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
          if (lora_id <= 0) continue;
          size_t gi = 0;
          for (size_t k = 0; k < active_ids.size(); ++k) {
            if (active_ids[k] == lora_id) gi = k;
          }
          int64_t offset = 0;
          for (int64_t sz : output_sizes) {
            LoRAMat piece;
            piece.rows = sz;
            piece.cols = kMaxRank;
            piece.data.assign(
                b_mats[gi].data.begin() + static_cast<long>(offset * kMaxRank),
                b_mats[gi].data.begin() + static_cast<long>((offset + sz) * kMaxRank));
            RefAddSlice(ref, t, total_output, offset, batch.x, kInput, a_mats[gi],
                        piece, /*scaling=*/1.0);
            offset += sz;
          }
        }
        CheckAllClose(y, ref);

        // test_layers.py:1003-1035 — reset, then a fresh active_lora_ids=[0] batch.
        for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
        Batch base_batch = MakeBaseBatch(rng, id_to_index, 32 * num_loras, kInput);
        std::vector<float> base2 = rng.Vec(base_batch.T * total_output);
        std::vector<float> y2 = base2;
        layer.ApplyLoraToOutput(y2.data(), base_batch.x.data(), base_batch.T,
                                base_batch.slots.data());
        CHECK(y2 == base2);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The slot copy's stride (base_linear.py:179-184)

TEST_CASE("an adapter whose rank is below max_lora_rank keeps the slot's stride") {
  // `stacked[index, 0, :a.shape[0], :a.shape[1]].copy_(a)` writes into the
  // LEADING sub-block of a slot whose row stride belongs to the LAYER. Upstream's
  // populate_loras always builds rank == max_lora_rank, so both strides coincide
  // in every ported case and nothing there pins which one the copy uses.
  constexpr int64_t kInput = 64;
  constexpr int64_t kOutput = 48;
  constexpr int64_t kMaxLoras = 4;
  constexpr int64_t kMaxRank = 8;
  constexpr int kRank = 3;  // < kMaxRank, and not a divisor of it

  Rng rng(4242);
  ReplicatedLinearWithLoRA layer(kInput, kOutput);
  layer.CreateLoraWeights(kMaxLoras, kMaxRank);
  AdapterGroup g = MakeAdapterGroup(rng, kOutput, kInput, /*repeats=*/1, kRank);
  const int64_t kSlot = 2;
  layer.SetLora(kSlot, g.a, g.b);

  // lora_b_stacked slot: [kOutput, kMaxRank], the adapter's kRank columns
  // followed by zero padding.
  const std::vector<float>& b_stacked = layer.lora_b_stacked(0);
  for (int64_t o = 0; o < kOutput; ++o) {
    for (int64_t r = 0; r < kMaxRank; ++r) {
      const float got =
          b_stacked[static_cast<size_t>(kSlot * kOutput * kMaxRank + o * kMaxRank + r)];
      const float want = r < kRank ? g.b[0].At(o, r) : 0.0f;
      REQUIRE(got == want);
    }
  }
  // lora_a_stacked slot: [kMaxRank, kInput], rows kRank.. left zero.
  const std::vector<float>& a_stacked = layer.lora_a_stacked(0);
  for (int64_t r = 0; r < kMaxRank; ++r) {
    for (int64_t c = 0; c < kInput; ++c) {
      const float got =
          a_stacked[static_cast<size_t>(kSlot * kMaxRank * kInput + r * kInput + c)];
      const float want = r < kRank ? g.a[0].At(r, c) : 0.0f;
      REQUIRE(got == want);
    }
  }

  // ... and the apply that reads those slots still matches the reference.
  const int64_t T = 12;
  std::vector<float> x = rng.Vec(T * kInput);
  std::vector<float> base = rng.Vec(T * kOutput);
  std::vector<int32_t> slots(static_cast<size_t>(T), static_cast<int32_t>(kSlot));
  slots[0] = -1;  // one base token
  std::vector<float> y = base;
  layer.ApplyLoraToOutput(y.data(), x.data(), T, slots.data());
  std::vector<double> ref(base.begin(), base.end());
  for (int64_t t = 1; t < T; ++t) {
    RefAddSlice(ref, t, kOutput, 0, x, kInput, g.a[0], g.b[0], g.subloras[0].scaling);
  }
  CheckAllClose(y, ref);
}

// ---------------------------------------------------------------------------
// SetLora's tp_size > 1 branch (base_linear.py:175-177)

TEST_CASE("SetLora slices lora_a with slice_lora_a and lora_b with slice_lora_b") {
  // The branch only runs at tp_size > 1, and every apply case above builds at
  // tp_size == 1 where both rules are the identity — so which tensor goes
  // through which rule is unobservable there.
  constexpr int64_t kRank = 4;
  constexpr int64_t kMaxLoras = 2;
  const int64_t kSlot = 1;

  SUBCASE("row-parallel cuts lora_a along the input dim and keeps lora_b whole") {
    Rng rng(77);
    const int64_t in_per = 8, tp = 2, out = 6, tp_rank = 1;
    RowParallelLinearWithLoRA layer(in_per, out, tp, tp_rank);
    layer.CreateLoraWeights(kMaxLoras, kRank);
    LoRAMat a = RandMat(rng, kRank, in_per * tp);
    LoRAMat b = RandMat(rng, out, kRank);
    layer.SetLora(kSlot, {a}, {b});

    const std::vector<float>& a_stacked = layer.lora_a_stacked(0);
    for (int64_t r = 0; r < kRank; ++r) {
      for (int64_t c = 0; c < in_per; ++c) {
        REQUIRE(a_stacked[static_cast<size_t>(kSlot * kRank * in_per + r * in_per + c)] ==
                a.At(r, tp_rank * in_per + c));
      }
    }
    const std::vector<float>& b_stacked = layer.lora_b_stacked(0);
    for (int64_t o = 0; o < out; ++o) {
      for (int64_t r = 0; r < kRank; ++r) {
        REQUIRE(b_stacked[static_cast<size_t>(kSlot * out * kRank + o * kRank + r)] ==
                b.At(o, r));
      }
    }
  }

  SUBCASE("column-parallel cuts lora_b along the output dim and keeps lora_a whole") {
    Rng rng(78);
    const int64_t in = 8, tp = 2, out_per = 6, tp_rank = 1;
    ColumnParallelLinearWithLoRA layer(in, out_per, tp, tp_rank);
    layer.CreateLoraWeights(kMaxLoras, kRank);
    LoRAMat a = RandMat(rng, kRank, in);
    LoRAMat b = RandMat(rng, out_per * tp, kRank);
    layer.SetLora(kSlot, {a}, {b});

    const std::vector<float>& a_stacked = layer.lora_a_stacked(0);
    for (int64_t r = 0; r < kRank; ++r) {
      for (int64_t c = 0; c < in; ++c) {
        REQUIRE(a_stacked[static_cast<size_t>(kSlot * kRank * in + r * in + c)] ==
                a.At(r, c));
      }
    }
    const std::vector<float>& b_stacked = layer.lora_b_stacked(0);
    for (int64_t o = 0; o < out_per; ++o) {
      for (int64_t r = 0; r < kRank; ++r) {
        REQUIRE(b_stacked[static_cast<size_t>(kSlot * out_per * kRank + o * kRank + r)] ==
                b.At(tp_rank * out_per + o, r));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The deferral REFUSES (file header; _mcp_apply, column_parallel_linear.py:24-82)

TEST_CASE("the fully-sharded apply refuses at tp_size > 1 instead of returning a "
          "partial delta") {
  constexpr int64_t kIn = 16;
  constexpr int64_t kOut = 8;
  constexpr int64_t kMaxRank = 8;
  constexpr int64_t kMaxLoras = 2;
  const int64_t T = 4;
  Rng rng(909);

  std::vector<float> x = rng.Vec(T * kIn);
  std::vector<int32_t> idx(static_cast<size_t>(T), 0);

  SUBCASE("column-parallel, tp_size 2") {
    ColumnParallelLinearWithShardedLoRA layer(kIn, kOut, /*tp_size=*/2, /*tp_rank=*/0);
    layer.CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/true);
    layer.SetLora(0, {RandMat(rng, kMaxRank, kIn)}, {RandMat(rng, kOut * 2, kMaxRank)});
    std::vector<float> y(static_cast<size_t>(T * kOut), 0.0f);
    CHECK_THROWS_AS(layer.ApplyLoraToOutput(y.data(), x.data(), T, idx.data()),
                    std::logic_error);
  }

  SUBCASE("row-parallel, tp_size 2") {
    RowParallelLinearWithShardedLoRA layer(kIn, kOut, /*tp_size=*/2, /*tp_rank=*/0);
    layer.CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/true);
    layer.SetLora(0, {RandMat(rng, kMaxRank, kIn * 2)}, {RandMat(rng, kOut, kMaxRank)});
    std::vector<float> y(static_cast<size_t>(T * kOut / 2), 0.0f);
    CHECK_THROWS_AS(layer.ApplyLoraToOutput(y.data(), x.data(), T, idx.data()),
                    std::logic_error);
  }

  SUBCASE("tp_size 1 is upstream-equivalent and still applies") {
    ColumnParallelLinearWithShardedLoRA layer(kIn, kOut, /*tp_size=*/1, /*tp_rank=*/0);
    layer.CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/true);
    layer.SetLora(0, {RandMat(rng, kMaxRank, kIn)}, {RandMat(rng, kOut, kMaxRank)});
    std::vector<float> y(static_cast<size_t>(T * kOut), 0.0f);
    CHECK_NOTHROW(layer.ApplyLoraToOutput(y.data(), x.data(), T, idx.data()));
    bool any_nonzero = false;
    for (float v : y) any_nonzero = any_nonzero || v != 0.0f;
    CHECK(any_nonzero);
  }
}

TEST_CASE("the class and fully_sharded_loras must pair as can_replace_layer does") {
  // layers/utils.py:76-101 — upstream cannot construct either mismatch.
  ColumnParallelLinearWithShardedLoRA sharded(/*input_size=*/8, /*output_size=*/8,
                                              /*tp_size=*/2, /*tp_rank=*/0);
  CHECK_THROWS_AS(sharded.CreateLoraWeights(2, 8, /*fully_sharded_loras=*/false),
                  std::logic_error);

  ColumnParallelLinearWithLoRA plain(/*input_size=*/8, /*output_size=*/8,
                                     /*tp_size=*/2, /*tp_rank=*/0);
  CHECK_THROWS_AS(plain.CreateLoraWeights(2, 8, /*fully_sharded_loras=*/true),
                  std::logic_error);

  RowParallelLinearWithShardedLoRA row(8, 8, 2, 0);
  CHECK_THROWS_AS(row.CreateLoraWeights(2, 8, false), std::logic_error);

  // ReplicatedLinearWithLoRA carries NEITHER decorator, so both are legal.
  ReplicatedLinearWithLoRA replicated(8, 8);
  CHECK_NOTHROW(replicated.CreateLoraWeights(2, 8, /*fully_sharded_loras=*/false));
  CHECK_NOTHROW(replicated.CreateLoraWeights(2, 8, /*fully_sharded_loras=*/true));
}

TEST_CASE("the slicing primitives refuse a window outside the incoming adapter") {
  // An adapter shaped for a different tp_size lands here; torch basic indexing
  // raises IndexError, and reading past the tensor would fabricate a delta.
  Rng rng(31337);
  RowParallelLinearWithLoRA row(/*input_size_per_partition=*/8, /*output_size=*/6,
                                /*tp_size=*/4, /*tp_rank=*/3);
  LoRAMat too_narrow = RandMat(rng, 4, 8);  // a full lora_a would be 4 x 32
  CHECK_THROWS_AS(row.SliceLoraA({too_narrow}), std::invalid_argument);

  ColumnParallelLinearWithLoRA col(/*input_size=*/8, /*output_size_per_partition=*/6,
                                   /*tp_size=*/4, /*tp_rank=*/3);
  LoRAMat too_short = RandMat(rng, 6, 4);  // a full lora_b would be 24 x 4
  CHECK_THROWS_AS(col.SliceLoraB({too_short}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// The logits expand's third bound (lora_ops.py:42)

TEST_CASE("ApplyLoraToLogits clamps to the narrower of the adapter vocab and the "
          "logits width") {
  // "LoRA adapter and model may add different amounts of padding to output"
  // (lora_ops.py:41): common_len = min(outputs.shape[1], output_tensor.shape[1]).
  // `y` is the caller's already-gathered logits buffer, so its width bounds the
  // write independently of the adapter's vocab.
  constexpr int64_t kVocab = 64;
  constexpr int64_t kHidden = 16;
  constexpr int64_t kMaxRank = 4;
  constexpr int64_t kMaxLoras = 2;
  constexpr int64_t kWidth = 32;  // the lm_head pads LESS than the adapter
  const int64_t T = 5;
  Rng rng(5150);

  LogitsProcessorWithLoRA layer(kVocab, kHidden);
  layer.CreateLoraWeights(kMaxLoras, kMaxRank);
  LoRAMat a = RandMat(rng, kMaxRank, kHidden);
  LoRAMat b = RandMat(rng, kVocab, kMaxRank);
  layer.SetLora(1, a, b);

  std::vector<float> x = rng.Vec(T * kHidden);
  std::vector<float> base = rng.Vec(T * kWidth);
  std::vector<int32_t> idx(static_cast<size_t>(T), 1);
  idx[0] = -1;  // one base request
  std::vector<float> y = base;
  layer.ApplyLoraToLogits(y.data(), kWidth, x.data(), T, idx.data());

  // Only the first kWidth vocab entries may be touched, and nothing past the
  // row: with the clamp absent this writes kVocab floats into a kWidth row.
  LoRAMat b_head;
  b_head.rows = kWidth;
  b_head.cols = kMaxRank;
  b_head.data.assign(b.data.begin(), b.data.begin() + static_cast<long>(kWidth * kMaxRank));
  std::vector<double> ref(base.begin(), base.end());
  for (int64_t t = 1; t < T; ++t) {
    RefAddSlice(ref, t, kWidth, 0, x, kHidden, a, b_head, /*scaling=*/1.0);
  }
  CheckAllClose(y, ref);
}

// ---------------------------------------------------------------------------
// convert_mapping's two index arrays (vocal_parallel_embedding.py:99-121)

TEST_CASE("the embedding gather pins a base-model token to slot 0") {
  // `_embeddings_indices[1]` is `slot * (vocab + extra_vocab)` where the slot
  // is 0, not -1, for a base token (utils.py:114-115,126-129); the expand's
  // `token_lora_indices` is -1 there and writes nothing. The gathered row is
  // therefore dead, but it must still land INSIDE the table — a slot the caller
  // never allocated would index past it.
  CHECK(VocabParallelEmbeddingWithLoRA::GatherSlot(-1) == 0);
  CHECK(VocabParallelEmbeddingWithLoRA::GatherSlot(-7) == 0);
  CHECK(VocabParallelEmbeddingWithLoRA::GatherSlot(0) == 0);
  CHECK(VocabParallelEmbeddingWithLoRA::GatherSlot(3) == 3);
}
