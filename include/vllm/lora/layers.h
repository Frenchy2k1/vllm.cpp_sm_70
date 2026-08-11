// LoRA-wrapped layers — the stacked adapter slots a base layer owns, the
// tensor-parallel slicing rules that decide which rows/columns of an incoming
// adapter this rank keeps, and the batched delta apply.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922 /
// vLLM 0.26.0.dev0):
//   vllm/lora/layers/base_linear.py:70-238           BaseLinearLayerWithLoRA
//   vllm/lora/layers/replicated_linear.py            ReplicatedLinearWithLoRA
//   vllm/lora/layers/column_parallel_linear.py:85-746 column / merged-column /
//                                                    qkv / merged-qkv + the
//                                                    fully-sharded (S-LoRA)
//                                                    slice_lora_a overrides
//   vllm/lora/layers/row_parallel_linear.py:22-177   row parallel (+ sharded)
//   vllm/lora/layers/vocal_parallel_embedding.py     VocabParallelEmbeddingWithLoRA
//   vllm/lora/layers/logits_processor.py             LogitsProcessorWithLoRA
//
// A LoRA-wrapped layer is a DELTA APPLIER: the base linear runs first
// (`_get_quant_method().apply(...)`, base_linear.py:207) and the layer adds the
// low-rank delta onto that output (`_apply_lora_to_output`, `:215-238`). We
// mirror exactly that split — `ApplyLoraToOutput` takes the base output in
// place — so the base GEMM keeps living behind `LinearMethodBase` and the LoRA
// path stays additive.
//
// STACKED SLOT LAYOUT (base_linear.py:129-151, with the "1" layer dim collapsed
// as in .agents/specs/lora-adapter.md; there is one adapter slot per layer):
//   lora_a_stacked[s] : [max_loras, lora_a_rows, input_size ] row-major
//   lora_b_stacked[s] : [max_loras, output_slices[s], max_rank] row-major
// `lora_a_rows` is `max_rank`, except on the fully-sharded (S-LoRA) path where
// it is `max_rank / tp_size` (base_linear.py:112-116).
//
// NOT PORTED HERE, with reason:
//  * The fully-sharded APPLY path (`_mcp_apply`, column_parallel_linear.py:24-82,
//    and RowParallelLinearWithShardedLoRA.apply, row_parallel_linear.py:118-159)
//    is *defined* by `tensor_model_parallel_all_gather` / `all_reduce` between
//    the shrink and the expand. vllm.cpp's TP seam does not yet expose those
//    collectives (the TP row is mid-flight). The sharded classes therefore carry
//    their SLICING rules (pure functions of tp_rank/tp_size, and the part W2
//    owns) and inherit the unsharded apply, which is exactly the upstream
//    behaviour at tp_size == 1 — the shrink's `max_rank / tp_size` rows ARE the
//    whole rank there and the all-gather is the identity.
//    At tp_size > 1 the inherited apply does NOT reduce to upstream: the shrink
//    fills only `max_rank / tp_size` of the buffer (row-parallel: lora_b holds
//    only this rank's output shard) while the expand consumes the full rank, so
//    it would return a PARTIAL delta that looks plausible. `ApplyLoraToOutput`
//    therefore THROWS `std::logic_error` for a fully-sharded layer at
//    tp_size > 1 rather than returning a number no gate could catch. A deferral
//    refuses; it does not approximate.
//  * `FusedMoEWithLoRA` (fused_moe.py) and the `pack_moe` stacks — W7.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/lora/lora_weights.h"

namespace vllm {
namespace lora {

// A row-major dense adapter tensor as it arrives from a checkpoint: lora_a is
// [rank, input_dim], lora_b is [output_dim, rank]. Upstream passes torch
// tensors and slices them with basic indexing; this is the portable equivalent.
struct LoRAMat {
  std::vector<float> data;
  int64_t rows = 0;
  int64_t cols = 0;

  float At(int64_t r, int64_t c) const {
    return data[static_cast<size_t>(r * cols + c)];
  }
  bool empty() const { return rows == 0 || cols == 0; }
};

// Upstream's `list[torch.Tensor | None]`: one entry per packed sub-module, an
// EMPTY LoRAMat standing for `None` ("this sub-module has no LoRA").
using MatList = std::vector<LoRAMat>;

// BaseLinearLayerWithLoRA (base_linear.py:70). Owns `max_loras` stacked adapter
// slots across `n_slices` output windows.
class BaseLinearLayerWithLoRA {
 public:
  virtual ~BaseLinearLayerWithLoRA() = default;

  BaseLinearLayerWithLoRA(const BaseLinearLayerWithLoRA&) = delete;
  BaseLinearLayerWithLoRA& operator=(const BaseLinearLayerWithLoRA&) = delete;

  // Which of upstream's two can_replace_layer decorators gates this class
  // (layers/utils.py:76-101). Upstream can never CONSTRUCT a sharded variant
  // without `fully_sharded_loras`, nor a plain parallel variant with it —
  // `_fully_sharded_can_replace` / `_not_fully_sharded_can_replace` decide
  // which class wraps the base layer from the same LoRAConfig that later
  // shapes the stacked slots. Our API takes the flag as an argument instead,
  // so nothing binds the two; `CreateLoraWeights` enforces the same pairing.
  // `kEither` is ReplicatedLinearWithLoRA, which carries neither decorator.
  enum class ShardingGate { kEither, kNotFullySharded, kFullySharded };
  virtual ShardingGate sharding_gate() const { return ShardingGate::kEither; }

  // create_lora_weights (base_linear.py:100-151): allocate the zeroed stacked
  // slots. `fully_sharded_loras` shrinks lora_a's rank rows for column-like
  // layers and lora_b's output rows for row-parallel ones (`:111-125`).
  // Throws `std::logic_error` when the flag contradicts `sharding_gate()`.
  void CreateLoraWeights(int64_t max_loras, int64_t max_lora_rank,
                         bool fully_sharded_loras = false);

  // reset_lora (base_linear.py:153-156): zero slot `index` in every slice.
  void ResetLora(int64_t index);

  // set_lora (base_linear.py:158-184 / column_parallel_linear.py:302-328):
  // reset, apply the TP slicing when tp_size > 1, then copy each present slice
  // into the leading sub-block of its stacked slot (the rest stays zero-padded).
  virtual void SetLora(int64_t index, const MatList& lora_a,
                       const MatList& lora_b);

  // _apply_lora_to_output (base_linear.py:215-238) -> add_lora_linear with
  // scale 1.0 (`:227-229`; each slot's scaling was folded into lora_b by the
  // adapter's optimize()). `y` is [T, output_size] and already holds the base
  // linear output; `indices[t] < 0` means "no adapter for this token".
  //
  // THROWS `std::logic_error` for a fully-sharded layer at tp_size > 1: that
  // apply is `_mcp_apply` / RowParallelLinearWithShardedLoRA.apply, which is
  // defined by the rank-dim collective we do not have. See the file header.
  void ApplyLoraToOutput(float* y, const float* x, int64_t T,
                         const int32_t* indices) const;

  // slice_lora_a / slice_lora_b — the per-rank slicing rules. The base is the
  // identity for both (a ReplicatedLinear keeps everything).
  virtual MatList SliceLoraA(const MatList& lora_a) const { return lora_a; }
  virtual MatList SliceLoraB(const MatList& lora_b) const { return lora_b; }

  int64_t n_slices() const { return n_slices_; }
  // base_linear.py:151 — the widths the expand actually writes.
  const std::vector<int64_t>& output_slices() const { return lora_b_rows_; }
  int64_t output_size() const;
  int64_t input_size() const { return input_size_; }
  int64_t max_rank() const { return max_rank_; }
  int64_t lora_a_rows() const { return lora_a_rows_; }
  int64_t max_loras() const { return max_loras_; }
  const std::vector<float>& lora_a_stacked(int64_t slice) const {
    return lora_a_stacked_[static_cast<size_t>(slice)];
  }
  const std::vector<float>& lora_b_stacked(int64_t slice) const {
    return lora_b_stacked_[static_cast<size_t>(slice)];
  }

 protected:
  // Which base layer this wraps — the isinstance ladder of
  // base_linear.py:107-127, which decides what `fully_sharded_loras` shrinks.
  enum class Kind { kReplicated, kColumn, kRow };

  BaseLinearLayerWithLoRA(Kind kind, int64_t input_size,
                          std::vector<int64_t> output_slices, int64_t tp_size,
                          int64_t tp_rank)
      : kind_(kind),
        input_size_(input_size),
        output_slices_(std::move(output_slices)),
        n_slices_(static_cast<int64_t>(output_slices_.size())),
        tp_size_(tp_size),
        tp_rank_(tp_rank) {}

  // Row window [start, start+count) of `m` — the building block of every
  // slicing rule. Torch basic indexing raises IndexError on an out-of-range
  // window; these throw `std::invalid_argument` rather than reading past the
  // tensor, which is what an adapter shaped for a different tp_size does.
  static LoRAMat RowSlice(const LoRAMat& m, int64_t start, int64_t count);
  // Column window [start, start+count) of `m` (row-parallel slices lora_a's
  // input dim, row_parallel_linear.py:32-37). Bounds-checked like RowSlice.
  static LoRAMat ColSlice(const LoRAMat& m, int64_t start, int64_t count);
  // Vertical concatenation (torch.cat(dim=0), column_parallel_linear.py:122).
  static LoRAMat ConcatRows(const std::vector<LoRAMat>& parts);

  Kind kind_;
  int64_t input_size_ = 0;
  // The layer's own output windows, as the base layer defines them.
  std::vector<int64_t> output_slices_;
  // The widths lora_b_stacked was actually created with — upstream's
  // `self.output_slices = (self.lora_b_stacked[0].shape[2],)`
  // (base_linear.py:151). Equal to `output_slices_` everywhere except a
  // fully-sharded row-parallel layer, where lora_b is sharded output-wise.
  std::vector<int64_t> lora_b_rows_;
  int64_t n_slices_ = 1;
  int64_t tp_size_ = 1;
  int64_t tp_rank_ = 0;

  int64_t max_loras_ = 0;
  int64_t max_rank_ = 0;
  int64_t lora_a_rows_ = 0;
  bool fully_sharded_ = false;
  std::vector<std::vector<float>> lora_a_stacked_;
  std::vector<std::vector<float>> lora_b_stacked_;
};

// ReplicatedLinearWithLoRA (replicated_linear.py). No sharding, one slice.
class ReplicatedLinearWithLoRA : public BaseLinearLayerWithLoRA {
 public:
  ReplicatedLinearWithLoRA(int64_t input_size, int64_t output_size)
      : BaseLinearLayerWithLoRA(Kind::kReplicated, input_size, {output_size},
                                /*tp_size=*/1, /*tp_rank=*/0) {}
};

// ColumnParallelLinearWithLoRA (column_parallel_linear.py:85-181). One slice;
// lora_b is sharded along the output dim. `is_merged_col_linear` covers a
// MergedColumnParallelLinear base whose checkpoint ships ONE lora_b for both
// halves (`:110-122`).
class ColumnParallelLinearWithLoRA : public BaseLinearLayerWithLoRA {
 public:
  ColumnParallelLinearWithLoRA(int64_t input_size,
                               int64_t output_size_per_partition,
                               int64_t tp_size, int64_t tp_rank,
                               bool is_merged_col_linear = false)
      : BaseLinearLayerWithLoRA(Kind::kColumn, input_size,
                                {output_size_per_partition}, tp_size, tp_rank),
        is_merged_col_linear_(is_merged_col_linear) {}

  MatList SliceLoraB(const MatList& lora_b) const override;
  // _not_fully_sharded_can_replace (layers/utils.py:76-87).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kNotFullySharded;
  }

 protected:
  bool is_merged_col_linear_ = false;
};

// MergedColumnParallelLinearWithLoRA (column_parallel_linear.py:184-368) —
// gate_up_proj: N sub-modules packed into one linear, one adapter each.
// `output_sizes` is the UNSHARDED per-sub-module size; each slice's own window
// is output_sizes[i] / tp_size (`:200-205`).
class MergedColumnParallelLinearWithLoRA : public BaseLinearLayerWithLoRA {
 public:
  MergedColumnParallelLinearWithLoRA(int64_t input_size,
                                     std::vector<int64_t> output_sizes,
                                     int64_t tp_size, int64_t tp_rank);

  // slice_lora_b (column_parallel_linear.py:253-264): each sub-module keeps the
  // window its own shard id selects.
  MatList SliceLoraB(const MatList& lora_b) const override;

  // set_lora (column_parallel_linear.py:302-328) with the packed-group expansion
  // of `:266-300` when the adapter ships fewer groups than the layer has slices.
  void SetLora(int64_t index, const MatList& lora_a,
               const MatList& lora_b) override;

  // expand_packed_lora (column_parallel_linear.py:266-300): split a lora_b that
  // covers several consecutive output windows and replicate its lora_a.
  void ExpandPackedLora(MatList& lora_a, MatList& lora_b) const;

  // _not_fully_sharded_can_replace (layers/utils.py:76-87).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kNotFullySharded;
  }

  const std::vector<int64_t>& output_sizes() const { return output_sizes_; }

 protected:
  MergedColumnParallelLinearWithLoRA(Kind kind, int64_t input_size,
                                     std::vector<int64_t> output_slices,
                                     std::vector<int64_t> output_sizes,
                                     std::vector<int64_t> output_ids,
                                     int64_t tp_size, int64_t tp_rank)
      : BaseLinearLayerWithLoRA(kind, input_size, std::move(output_slices),
                                tp_size, tp_rank),
        output_sizes_(std::move(output_sizes)),
        output_ids_(std::move(output_ids)) {}

  std::vector<int64_t> output_sizes_;
  std::vector<int64_t> output_ids_;
};

// MergedColumnParallelLinearVariableSliceWithLoRA
// (column_parallel_linear.py:674-746) — a checkpoint with ONE fused lora_b for
// a 3+-slice layer: replicate lora_a and split lora_b by `output_sizes`.
class MergedColumnParallelLinearVariableSliceWithLoRA
    : public MergedColumnParallelLinearWithLoRA {
 public:
  using MergedColumnParallelLinearWithLoRA::MergedColumnParallelLinearWithLoRA;

  void SetLora(int64_t index, const MatList& lora_a,
               const MatList& lora_b) override;
};

// QKVParallelLinearWithLoRA (column_parallel_linear.py:371-434) — a qkv_proj
// with ONE adapter covering q, k and v; TP has to cut three windows out of the
// single lora_b and re-concatenate them.
class QKVParallelLinearWithLoRA : public BaseLinearLayerWithLoRA {
 public:
  QKVParallelLinearWithLoRA(int64_t input_size, int64_t head_size,
                            int64_t total_num_heads, int64_t num_heads,
                            int64_t total_num_kv_heads, int64_t num_kv_heads,
                            int64_t tp_size, int64_t tp_rank,
                            int64_t num_kv_head_replicas = 1);

  MatList SliceLoraB(const MatList& lora_b) const override;
  // _not_fully_sharded_can_replace (layers/utils.py:76-87).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kNotFullySharded;
  }

 protected:
  int64_t q_proj_total_size_ = 0;
  int64_t q_proj_shard_size_ = 0;
  int64_t kv_proj_total_size_ = 0;
  int64_t kv_proj_shard_size_ = 0;
  int64_t num_kv_head_replicas_ = 1;
};

// MergedQKVParallelLinearWithLoRA (column_parallel_linear.py:437-496) — q, k
// and v each carry their OWN adapter, so the layer has three slices whose
// widths differ (q may be wider than k/v).
class MergedQKVParallelLinearWithLoRA : public MergedColumnParallelLinearWithLoRA {
 public:
  MergedQKVParallelLinearWithLoRA(int64_t input_size, int64_t head_size,
                                  int64_t total_num_heads, int64_t num_heads,
                                  int64_t total_num_kv_heads,
                                  int64_t num_kv_heads, int64_t tp_size,
                                  int64_t tp_rank,
                                  int64_t num_kv_head_replicas = 1);
};

// RowParallelLinearWithLoRA (row_parallel_linear.py:22-93). lora_a is sharded
// along the INPUT dim; lora_b is kept whole.
class RowParallelLinearWithLoRA : public BaseLinearLayerWithLoRA {
 public:
  RowParallelLinearWithLoRA(int64_t input_size_per_partition,
                            int64_t output_size, int64_t tp_size,
                            int64_t tp_rank)
      : BaseLinearLayerWithLoRA(Kind::kRow, input_size_per_partition,
                                {output_size}, tp_size, tp_rank) {}

  MatList SliceLoraA(const MatList& lora_a) const override;
  // _not_fully_sharded_can_replace (layers/utils.py:76-87).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kNotFullySharded;
  }
};

// --- Fully-sharded (S-LoRA) variants -----------------------------------------
// Y. Sheng et al., S-LoRA (arXiv:2311.03285): the low-rank factors are sharded
// too, so a rank-dim collective sits between the shrink and the expand. Only
// the SLICING is ported here; see the file header for why the apply is not.

// column_parallel_linear.py:504-542.
class ColumnParallelLinearWithShardedLoRA : public ColumnParallelLinearWithLoRA {
 public:
  using ColumnParallelLinearWithLoRA::ColumnParallelLinearWithLoRA;
  MatList SliceLoraA(const MatList& lora_a) const override;
  // _fully_sharded_can_replace (layers/utils.py:90-101).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kFullySharded;
  }
};

// column_parallel_linear.py:545-585.
class MergedColumnParallelLinearWithShardedLoRA
    : public MergedColumnParallelLinearWithLoRA {
 public:
  using MergedColumnParallelLinearWithLoRA::MergedColumnParallelLinearWithLoRA;
  MatList SliceLoraA(const MatList& lora_a) const override;
  // _fully_sharded_can_replace (layers/utils.py:90-101).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kFullySharded;
  }
};

// column_parallel_linear.py:588-621.
class QKVParallelLinearWithShardedLoRA : public QKVParallelLinearWithLoRA {
 public:
  using QKVParallelLinearWithLoRA::QKVParallelLinearWithLoRA;
  MatList SliceLoraA(const MatList& lora_a) const override;
  // _fully_sharded_can_replace (layers/utils.py:90-101).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kFullySharded;
  }
};

// column_parallel_linear.py:624-671.
class MergedQKVParallelLinearWithShardedLoRA
    : public MergedQKVParallelLinearWithLoRA {
 public:
  using MergedQKVParallelLinearWithLoRA::MergedQKVParallelLinearWithLoRA;
  MatList SliceLoraA(const MatList& lora_a) const override;
  // _fully_sharded_can_replace (layers/utils.py:90-101).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kFullySharded;
  }
};

// row_parallel_linear.py:101-177.
class RowParallelLinearWithShardedLoRA : public RowParallelLinearWithLoRA {
 public:
  using RowParallelLinearWithLoRA::RowParallelLinearWithLoRA;
  MatList SliceLoraB(const MatList& lora_b) const override;
  // _fully_sharded_can_replace (layers/utils.py:90-101).
  ShardingGate sharding_gate() const override {
    return ShardingGate::kFullySharded;
  }
};

// VocabParallelEmbeddingWithLoRA (vocal_parallel_embedding.py:17-140). The
// embedding LoRA has no shrink GEMM: lora_a is itself an embedding table, so
// the "shrink" is a row gather and only the expand is a GEMM.
//
//   lora_a_stacked : [max_loras, org_vocab_size, max_rank]   (lora_a TRANSPOSED
//                    on the way in, `:86-91`)
//   lora_b_stacked : [max_loras, embedding_dim, max_rank]
class VocabParallelEmbeddingWithLoRA {
 public:
  VocabParallelEmbeddingWithLoRA(int64_t org_vocab_size, int64_t embedding_dim)
      : org_vocab_size_(org_vocab_size), embedding_dim_(embedding_dim) {}

  void CreateLoraWeights(int64_t max_loras, int64_t max_lora_rank);
  void ResetLora(int64_t index);
  // set_lora (vocal_parallel_embedding.py:77-94). `lora_a` is [rank,
  // org_vocab_size] and is stored transposed; `lora_b` is [embedding_dim, rank].
  void SetLora(int64_t index, const LoRAMat& lora_a, const LoRAMat& lora_b);

  // forward (vocal_parallel_embedding.py:96-126): gather each token's lora_a
  // row, then expand onto the base embedding output `y` ([T, embedding_dim]).
  //
  // Upstream reads TWO index arrays produced by convert_mapping: the gather
  // uses `_embeddings_indices[1]` = slot * (vocab + extra_vocab) — which is 0,
  // not -1, for a base-model token (utils.py:114-115,126-129) — while the
  // expand uses `token_lora_indices`, which IS -1 there and therefore writes
  // nothing. Both derive from the same per-token slot, so we take that one
  // array and reproduce the derivation (`max(slot, 0)` for the gather).
  // Building it from a LoRAMapping is convert_mapping's job (W3).
  void ApplyLoraToOutput(float* y, const int32_t* token_ids, int64_t T,
                         const int32_t* indices) const;

  // The gather half of that derivation, as a pure function so it is testable
  // on its own: `_embeddings_indices[1]` is `slot * (vocab + extra_vocab)`
  // where `slot` is `indices_list[0]`, which convert_mapping builds as
  // "0 if the token has no adapter" (utils.py:114-115,126-129) — NOT the -1
  // the expand's `token_lora_indices` carries. The gathered row of a base
  // token is dead work upstream (the expand skips it) and equally dead here;
  // it is pinned to slot 0 so it can never index outside the table.
  static int64_t GatherSlot(int32_t index) { return index < 0 ? 0 : index; }

  int64_t org_vocab_size() const { return org_vocab_size_; }
  int64_t embedding_dim() const { return embedding_dim_; }
  int64_t max_rank() const { return max_rank_; }

 private:
  int64_t org_vocab_size_ = 0;
  int64_t embedding_dim_ = 0;
  int64_t max_loras_ = 0;
  int64_t max_rank_ = 0;
  std::vector<float> lora_a_stacked_;  // [max_loras, org_vocab_size, max_rank]
  std::vector<float> lora_b_stacked_;  // [max_loras, embedding_dim, max_rank]
};

// LogitsProcessorWithLoRA (logits_processor.py:20-208). The LoRA delta on the
// lm_head output, indexed by the SAMPLER indices (one per request).
class LogitsProcessorWithLoRA {
 public:
  // `sharded_to_full_mapping` (logits_processor.py:30-32) may be empty for "no
  // reindexing", exactly like upstream's None.
  LogitsProcessorWithLoRA(int64_t vocab_size, int64_t hidden_size,
                          std::vector<int32_t> sharded_to_full_mapping = {})
      : vocab_size_(vocab_size),
        hidden_size_(hidden_size),
        sharded_to_full_mapping_(std::move(sharded_to_full_mapping)) {}

  // create_lora_weights (logits_processor.py:84-119). Throws
  // std::invalid_argument for vocab_size > 258048, mirroring upstream's
  // ValueError (`:90-92`) — the stacked lora_b would not fit the kernel's
  // addressing.
  void CreateLoraWeights(int64_t max_loras, int64_t max_lora_rank);
  void ResetLora(int64_t index);
  // set_lora (logits_processor.py:125-139): lora_a [rank, hidden_size],
  // lora_b [vocab_size, rank].
  void SetLora(int64_t index, const LoRAMat& lora_a, const LoRAMat& lora_b);

  // _get_logits (logits_processor.py:141-194), LoRA half: add the delta onto
  // the already-gathered logits. `logits` is [T, logits_width]; the width may
  // exceed vocab_size because the lm_head pads (upstream truncates afterwards,
  // `:193`).
  void ApplyLoraToLogits(float* logits, int64_t logits_width,
                         const float* hidden_states, int64_t T,
                         const int32_t* sampler_indices) const;

  // The sharded->full reindex of logits_processor.py:166-183, as a pure
  // function so it is testable without a process group.
  std::vector<float> ReindexShardedToFull(const std::vector<float>& logits,
                                          int64_t T, int64_t width) const;

  int64_t vocab_size() const { return vocab_size_; }
  int64_t hidden_size() const { return hidden_size_; }
  int64_t max_rank() const { return max_rank_; }

 private:
  int64_t vocab_size_ = 0;
  int64_t hidden_size_ = 0;
  int64_t max_loras_ = 0;
  int64_t max_rank_ = 0;
  std::vector<int32_t> sharded_to_full_mapping_;
  std::vector<float> lora_a_stacked_;  // [max_loras, max_rank, hidden_size]
  std::vector<float> lora_b_stacked_;  // [max_loras, vocab_size, max_rank]
};

}  // namespace lora
}  // namespace vllm
