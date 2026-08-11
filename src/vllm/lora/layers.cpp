// LoRA-wrapped layers — implementation.
//
// UPSTREAM (ported FROM; ${VLLM_SOURCE} @ 555967922 / vLLM 0.26.0.dev0):
//   vllm/lora/layers/base_linear.py:100-238
//   vllm/lora/layers/column_parallel_linear.py:85-746
//   vllm/lora/layers/row_parallel_linear.py:22-177
//   vllm/lora/layers/vocal_parallel_embedding.py:24-126
//   vllm/lora/layers/logits_processor.py:84-194
//   vllm/lora/lora_weights.py:126-270           PackedLoRALayerWeights
#include "vllm/lora/layers.h"

#include <algorithm>
#include <cstddef>

#include "vllm/lora/punica.h"

namespace vllm {
namespace lora {

namespace {

// `divide` (vllm/distributed/utils.py) — an exact division that refuses a
// ragged split, matching upstream's assert.
int64_t Divide(int64_t numerator, int64_t denominator) {
  if (denominator <= 0 || numerator % denominator != 0) {
    throw std::invalid_argument("LoRA: " + std::to_string(numerator) +
                                " is not divisible by " +
                                std::to_string(denominator));
  }
  return numerator / denominator;
}

// Copy `src` into the leading [0,src.rows) x [0,src.cols) sub-block of a
// row-major [rows, cols] slot, leaving the zero padding alone. This is
// upstream's `stacked[index, 0, :a.shape[0], :a.shape[1]].copy_(a)`
// (base_linear.py:179-184).
void CopyIntoSlot(float* slot, int64_t slot_cols, const LoRAMat& src) {
  for (int64_t r = 0; r < src.rows; ++r) {
    float* dst = slot + r * slot_cols;
    const float* s = src.data.data() + r * src.cols;
    for (int64_t c = 0; c < src.cols; ++c) dst[c] = s[c];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// PackedLoRALayerWeights (lora_weights.py:126-270)

PackedLoRALayerWeights PackedLoRALayerWeights::Pack(
    const std::vector<std::optional<LoRALayerWeights>>& loras) {
  // lora_weights.py:134 — `next(lora for lora in loras if lora is not None)`
  // raises StopIteration when every sub-module is absent.
  const LoRALayerWeights* first = nullptr;
  for (const auto& l : loras) {
    if (l.has_value()) {
      first = &l.value();
      break;
    }
  }
  if (first == nullptr) {
    throw std::invalid_argument(
        "PackedLoRALayerWeights::Pack: no sub-module carries a LoRA");
  }

  PackedLoRALayerWeights obj;
  obj.rank = first->rank;
  obj.module_name = first->module_name;
  obj.subloras = loras;
  // `:135-138` — optimize() every present sub-LoRA, which is what makes
  // `:147-150`'s "scaling = 1 for every present slice" true.
  for (auto& l : obj.subloras) {
    if (l.has_value()) l->Optimize();
  }
  return obj;
}

PackedLoRALayerWeights& PackedLoRALayerWeights::Optimize() {
  // `:263-270` — skip absent slices and slices already at scaling == 1.
  for (auto& l : subloras) {
    if (l.has_value()) l->Optimize();
  }
  return *this;
}

// ---------------------------------------------------------------------------
// BaseLinearLayerWithLoRA

int64_t BaseLinearLayerWithLoRA::output_size() const {
  int64_t total = 0;
  for (int64_t o : output_slices_) total += o;
  return total;
}

LoRAMat BaseLinearLayerWithLoRA::RowSlice(const LoRAMat& m, int64_t start,
                                          int64_t count) {
  LoRAMat out;
  if (m.empty()) return out;
  // Torch basic indexing raises IndexError here; an adapter shaped for another
  // tp_size (or a sharded class created without `fully_sharded_loras`, which
  // upstream's _fully_sharded_can_replace makes unconstructible) lands exactly
  // on this window, and reading past the tensor would fabricate a delta.
  if (start < 0 || count < 0 || start + count > m.rows) {
    throw std::invalid_argument(
        "LoRA RowSlice: window [" + std::to_string(start) + ", " +
        std::to_string(start + count) + ") is outside a " +
        std::to_string(m.rows) + "-row tensor");
  }
  out.rows = count;
  out.cols = m.cols;
  out.data.assign(m.data.begin() + static_cast<std::ptrdiff_t>(start * m.cols),
                  m.data.begin() +
                      static_cast<std::ptrdiff_t>((start + count) * m.cols));
  return out;
}

LoRAMat BaseLinearLayerWithLoRA::ColSlice(const LoRAMat& m, int64_t start,
                                          int64_t count) {
  LoRAMat out;
  if (m.empty()) return out;
  if (start < 0 || count < 0 || start + count > m.cols) {
    throw std::invalid_argument(
        "LoRA ColSlice: window [" + std::to_string(start) + ", " +
        std::to_string(start + count) + ") is outside a " +
        std::to_string(m.cols) + "-column tensor");
  }
  out.rows = m.rows;
  out.cols = count;
  out.data.resize(static_cast<size_t>(m.rows * count));
  for (int64_t r = 0; r < m.rows; ++r) {
    for (int64_t c = 0; c < count; ++c) {
      out.data[static_cast<size_t>(r * count + c)] = m.At(r, start + c);
    }
  }
  return out;
}

LoRAMat BaseLinearLayerWithLoRA::ConcatRows(const std::vector<LoRAMat>& parts) {
  LoRAMat out;
  for (const LoRAMat& p : parts) {
    if (p.empty()) continue;
    out.cols = p.cols;
    out.rows += p.rows;
    out.data.insert(out.data.end(), p.data.begin(), p.data.end());
  }
  return out;
}

void BaseLinearLayerWithLoRA::CreateLoraWeights(int64_t max_loras,
                                                int64_t max_lora_rank,
                                                bool fully_sharded_loras) {
  // layers/utils.py:76-101 — the class and the flag are ONE decision upstream:
  // `_fully_sharded_can_replace` only lets a sharded variant wrap a base layer
  // when `fully_sharded_loras` is set, and `_not_fully_sharded_can_replace`
  // only lets a plain parallel variant wrap one when it is not. Our
  // `fully_sharded_loras` is an argument, so nothing else forbids the two
  // impossible pairings — and both are silently wrong rather than loud: a
  // sharded class without the flag slices `max_rank`-row windows out of a
  // `max_rank`-row tensor at every rank but 0, and a plain class with it keeps
  // full-rank slots the sharded slicing never fills.
  switch (sharding_gate()) {
    case ShardingGate::kEither:
      break;
    case ShardingGate::kNotFullySharded:
      if (fully_sharded_loras) {
        throw std::logic_error(
            "LoRA CreateLoraWeights: this layer is the NOT-fully-sharded "
            "variant (_not_fully_sharded_can_replace, layers/utils.py:76); "
            "fully_sharded_loras selects the ...WithShardedLoRA class instead");
      }
      break;
    case ShardingGate::kFullySharded:
      if (!fully_sharded_loras) {
        throw std::logic_error(
            "LoRA CreateLoraWeights: this layer is a fully-sharded (S-LoRA) "
            "variant (_fully_sharded_can_replace, layers/utils.py:90) and is "
            "unreachable without fully_sharded_loras");
      }
      break;
  }

  // base_linear.py:106-151. The isinstance ladder decides what
  // `fully_sharded_loras` shrinks: a column-like layer shards lora_a's rank
  // rows, a row-parallel one shards lora_b's output rows.
  max_loras_ = max_loras;
  max_rank_ = max_lora_rank;
  fully_sharded_ = fully_sharded_loras;

  lora_a_rows_ = max_lora_rank;
  lora_b_rows_ = output_slices_;
  switch (kind_) {
    case Kind::kReplicated:
      break;
    case Kind::kColumn:
      if (fully_sharded_loras) lora_a_rows_ = Divide(max_lora_rank, tp_size_);
      break;
    case Kind::kRow:
      if (fully_sharded_loras) {
        for (int64_t& r : lora_b_rows_) r = Divide(r, tp_size_);
      }
      break;
  }

  lora_a_stacked_.assign(
      static_cast<size_t>(n_slices_),
      std::vector<float>(static_cast<size_t>(max_loras * lora_a_rows_ * input_size_),
                         0.0f));
  lora_b_stacked_.clear();
  for (int64_t s = 0; s < n_slices_; ++s) {
    lora_b_stacked_.emplace_back(
        static_cast<size_t>(max_loras * lora_b_rows_[static_cast<size_t>(s)] * max_rank_),
        0.0f);
  }
}

void BaseLinearLayerWithLoRA::ResetLora(int64_t index) {
  // base_linear.py:153-156.
  for (int64_t s = 0; s < n_slices_; ++s) {
    const size_t si = static_cast<size_t>(s);
    const int64_t a_stride = lora_a_rows_ * input_size_;
    const int64_t b_stride = lora_b_rows_[si] * max_rank_;
    std::fill_n(lora_a_stacked_[si].begin() + static_cast<std::ptrdiff_t>(index * a_stride),
                static_cast<size_t>(a_stride), 0.0f);
    std::fill_n(lora_b_stacked_[si].begin() + static_cast<std::ptrdiff_t>(index * b_stride),
                static_cast<size_t>(b_stride), 0.0f);
  }
}

void BaseLinearLayerWithLoRA::SetLora(int64_t index, const MatList& lora_a,
                                      const MatList& lora_b) {
  // base_linear.py:158-184 for the single-slice layers and
  // column_parallel_linear.py:302-328 for the merged ones. The two differ only
  // in that the merged path tolerates an absent (None) sub-module, so one loop
  // covers both; the single-slice contract (`:168-172`, exactly one tensor) is
  // kept as an explicit check.
  if (static_cast<int64_t>(lora_a.size()) != n_slices_ ||
      static_cast<int64_t>(lora_b.size()) != n_slices_) {
    throw std::invalid_argument(
        "LoRA SetLora: expected " + std::to_string(n_slices_) +
        " slice(s), got " + std::to_string(lora_a.size()) + "/" +
        std::to_string(lora_b.size()));
  }
  ResetLora(index);

  MatList a = lora_a;
  MatList b = lora_b;
  if (tp_size_ > 1) {  // base_linear.py:175-177
    a = SliceLoraA(a);
    b = SliceLoraB(b);
  }

  for (int64_t s = 0; s < n_slices_; ++s) {
    const size_t si = static_cast<size_t>(s);
    if (!a[si].empty()) {
      CopyIntoSlot(lora_a_stacked_[si].data() + index * lora_a_rows_ * input_size_,
                   input_size_, a[si]);
    }
    if (!b[si].empty()) {
      CopyIntoSlot(
          lora_b_stacked_[si].data() + index * lora_b_rows_[si] * max_rank_,
          max_rank_, b[si]);
    }
  }
}

void BaseLinearLayerWithLoRA::ApplyLoraToOutput(float* y, const float* x,
                                                int64_t T,
                                                const int32_t* indices) const {
  // A fully-sharded layer's apply is NOT this function: it is `_mcp_apply`
  // (column_parallel_linear.py:24-82) or
  // RowParallelLinearWithShardedLoRA.apply (row_parallel_linear.py:118-159),
  // both of which are defined by a collective between the shrink and the
  // expand. At tp_size == 1 that collective is the identity and the two
  // coincide, which is why the sharded classes inherit this apply; at
  // tp_size > 1 they do not, and running it anyway would add a PARTIAL delta
  // (the shrink fills max_rank/tp_size of a max_rank buffer; row-parallel's
  // lora_b holds only this rank's output shard). Refuse instead.
  if (fully_sharded_ && tp_size_ > 1) {
    throw std::logic_error(
        "LoRA ApplyLoraToOutput: the fully-sharded (S-LoRA) apply needs the "
        "rank-dim all-gather / all-reduce (_mcp_apply, "
        "column_parallel_linear.py:24-82) that vllm.cpp's TP seam does not "
        "expose yet; only the slicing rules are ported at tp_size > 1");
  }

  // base_linear.py:227-229 — add_lora_linear(output, x, a, b, 1.0,
  // output_slices). The scale is 1.0 because each adapter's scaling was folded
  // into lora_b by optimize() before the slot copy.
  std::vector<const float*> a_ptrs;
  std::vector<const float*> b_ptrs;
  a_ptrs.reserve(lora_a_stacked_.size());
  b_ptrs.reserve(lora_b_stacked_.size());
  for (const auto& v : lora_a_stacked_) a_ptrs.push_back(v.data());
  for (const auto& v : lora_b_stacked_) b_ptrs.push_back(v.data());
  AddLoraLinear(y, x, T, input_size_, a_ptrs, b_ptrs, lora_b_rows_, max_loras_,
                lora_a_rows_, max_rank_, indices, /*scale=*/1.0);
}

// ---------------------------------------------------------------------------
// ColumnParallelLinearWithLoRA (column_parallel_linear.py:104-130)

MatList ColumnParallelLinearWithLoRA::SliceLoraB(const MatList& lora_b) const {
  const int64_t out = output_slices_[0];
  if (is_merged_col_linear_) {
    // `:110-122` — the checkpoint ships one lora_b covering BOTH halves; this
    // rank keeps its shard of each half and concatenates them.
    const int64_t shard_size = out / 2;
    const int64_t offset = lora_b[0].rows / 2;
    LoRAMat left = RowSlice(lora_b[0], tp_rank_ * shard_size, shard_size);
    LoRAMat right = RowSlice(lora_b[0], offset + tp_rank_ * shard_size, shard_size);
    return {ConcatRows({left, right})};
  }
  // `:125-129` — a plain column-parallel layer keeps one contiguous window.
  return {RowSlice(lora_b[0], tp_rank_ * out, out)};
}

// ---------------------------------------------------------------------------
// MergedColumnParallelLinearWithLoRA

MergedColumnParallelLinearWithLoRA::MergedColumnParallelLinearWithLoRA(
    int64_t input_size, std::vector<int64_t> output_sizes, int64_t tp_size,
    int64_t tp_rank)
    : BaseLinearLayerWithLoRA(Kind::kColumn, input_size, {}, tp_size, tp_rank),
      output_sizes_(std::move(output_sizes)) {
  // column_parallel_linear.py:198-205 — output_sizes is UNSHARDED, so each
  // slice's own window is output_sizes[i] / tp_size, and every slice reads the
  // same shard id (this rank).
  for (int64_t size : output_sizes_) output_slices_.push_back(Divide(size, tp_size));
  n_slices_ = static_cast<int64_t>(output_slices_.size());
  output_ids_.assign(static_cast<size_t>(n_slices_), tp_rank);
}

MatList MergedColumnParallelLinearWithLoRA::SliceLoraB(
    const MatList& lora_b) const {
  // column_parallel_linear.py:253-264 — absent sub-modules stay absent.
  MatList sliced(static_cast<size_t>(n_slices_));
  for (int64_t i = 0; i < n_slices_; ++i) {
    const size_t ii = static_cast<size_t>(i);
    if (lora_b[ii].empty()) continue;
    const int64_t shard_size = output_slices_[ii];
    sliced[ii] = RowSlice(lora_b[ii], shard_size * output_ids_[ii], shard_size);
  }
  return sliced;
}

void MergedColumnParallelLinearWithLoRA::ExpandPackedLora(MatList& lora_a,
                                                          MatList& lora_b) const {
  // column_parallel_linear.py:266-300 — an adapter group whose lora_b covers
  // several consecutive output windows (e.g. in_proj_qkv over q+k+v) is split
  // by output_sizes, and its lora_a is replicated once per window.
  MatList expanded_a;
  MatList expanded_b;
  int64_t start_idx = 0;
  for (size_t g = 0; g < lora_b.size(); ++g) {
    const int64_t b_rows = lora_b[g].rows;
    int64_t cu_rows = 0;
    int64_t covered = 0;
    for (int64_t i = start_idx; i < n_slices_; ++i) {
      cu_rows += output_sizes_[static_cast<size_t>(i)];
      if (cu_rows == b_rows) {
        covered = i - start_idx + 1;
        break;
      }
    }
    if (covered == 0) {
      throw std::invalid_argument(
          "LoRA: cannot split lora_b with " + std::to_string(b_rows) +
          " rows into " + std::to_string(n_slices_) + " slices starting at " +
          std::to_string(start_idx));
    }
    int64_t start = 0;
    for (int64_t j = 0; j < covered; ++j) {
      const int64_t size = output_sizes_[static_cast<size_t>(start_idx + j)];
      expanded_b.push_back(RowSlice(lora_b[g], start, size));
      expanded_a.push_back(lora_a[g]);
      start += size;
    }
    start_idx += covered;
  }
  lora_a = expanded_a;
  lora_b = expanded_b;
}

void MergedColumnParallelLinearWithLoRA::SetLora(int64_t index,
                                                 const MatList& lora_a,
                                                 const MatList& lora_b) {
  // column_parallel_linear.py:310-314 — expand the packed groups first when the
  // adapter ships fewer of them than the layer has slices.
  if (static_cast<int64_t>(lora_b.size()) != n_slices_) {
    MatList a = lora_a;
    MatList b = lora_b;
    ExpandPackedLora(a, b);
    BaseLinearLayerWithLoRA::SetLora(index, a, b);
    return;
  }
  BaseLinearLayerWithLoRA::SetLora(index, lora_a, lora_b);
}

void MergedColumnParallelLinearVariableSliceWithLoRA::SetLora(
    int64_t index, const MatList& lora_a, const MatList& lora_b) {
  // column_parallel_linear.py:718-746 — a single fused checkpoint tensor:
  // replicate lora_a per slice and cut lora_b by output_sizes.
  if (lora_a.size() == 1 && lora_b.size() == 1 &&
      static_cast<int64_t>(lora_b[0].rows) != lora_b_rows_[0]) {
    MatList a(static_cast<size_t>(n_slices_), lora_a[0]);
    MatList b;
    int64_t start = 0;
    for (int64_t size : output_sizes_) {
      b.push_back(RowSlice(lora_b[0], start, size));
      start += size;
    }
    MergedColumnParallelLinearWithLoRA::SetLora(index, a, b);
    return;
  }
  MergedColumnParallelLinearWithLoRA::SetLora(index, lora_a, lora_b);
}

// ---------------------------------------------------------------------------
// QKVParallelLinearWithLoRA (column_parallel_linear.py:384-420)

QKVParallelLinearWithLoRA::QKVParallelLinearWithLoRA(
    int64_t input_size, int64_t head_size, int64_t total_num_heads,
    int64_t num_heads, int64_t total_num_kv_heads, int64_t num_kv_heads,
    int64_t tp_size, int64_t tp_rank, int64_t num_kv_head_replicas)
    : BaseLinearLayerWithLoRA(
          Kind::kColumn, input_size,
          {num_heads * head_size + 2 * num_kv_heads * head_size}, tp_size,
          tp_rank),
      q_proj_total_size_(total_num_heads * head_size),
      q_proj_shard_size_(num_heads * head_size),
      kv_proj_total_size_(total_num_kv_heads * head_size),
      kv_proj_shard_size_(num_kv_heads * head_size),
      num_kv_head_replicas_(num_kv_head_replicas) {}

MatList QKVParallelLinearWithLoRA::SliceLoraB(const MatList& lora_b) const {
  // `:399-420` — one lora_b covers q|k|v; cut this rank's q shard, then its k
  // shard at the k offset, then its v shard, and concatenate.
  const int64_t q_shard_id = tp_rank_;
  const int64_t kv_shard_id = tp_rank_ / num_kv_head_replicas_;
  LoRAMat q = RowSlice(lora_b[0], q_proj_shard_size_ * q_shard_id, q_proj_shard_size_);
  const int64_t k_offset = q_proj_total_size_;
  LoRAMat k = RowSlice(lora_b[0], k_offset + kv_proj_shard_size_ * kv_shard_id,
                       kv_proj_shard_size_);
  const int64_t v_offset = k_offset + kv_proj_total_size_;
  LoRAMat v = RowSlice(lora_b[0], v_offset + kv_proj_shard_size_ * kv_shard_id,
                       kv_proj_shard_size_);
  return {ConcatRows({q, k, v})};
}

// ---------------------------------------------------------------------------
// MergedQKVParallelLinearWithLoRA (column_parallel_linear.py:448-469)

MergedQKVParallelLinearWithLoRA::MergedQKVParallelLinearWithLoRA(
    int64_t input_size, int64_t head_size, int64_t total_num_heads,
    int64_t num_heads, int64_t total_num_kv_heads, int64_t num_kv_heads,
    int64_t tp_size, int64_t tp_rank, int64_t num_kv_head_replicas)
    : MergedColumnParallelLinearWithLoRA(
          Kind::kColumn, input_size,
          /*output_slices=*/
          {num_heads * head_size, num_kv_heads * head_size,
           num_kv_heads * head_size},
          /*output_sizes=*/
          {total_num_heads * head_size, total_num_kv_heads * head_size,
           total_num_kv_heads * head_size},
          /*output_ids=*/
          {tp_rank, tp_rank / num_kv_head_replicas,
           tp_rank / num_kv_head_replicas},
          tp_size, tp_rank) {
  // `:451` — three slices, q possibly wider than k/v.
  n_slices_ = 3;
}

// ---------------------------------------------------------------------------
// RowParallelLinearWithLoRA (row_parallel_linear.py:32-40)

MatList RowParallelLinearWithLoRA::SliceLoraA(const MatList& lora_a) const {
  return {ColSlice(lora_a[0], tp_rank_ * input_size_, input_size_)};
}

// ---------------------------------------------------------------------------
// Fully-sharded (S-LoRA) slicing

MatList ColumnParallelLinearWithShardedLoRA::SliceLoraA(
    const MatList& lora_a) const {
  // column_parallel_linear.py:516-520 — shard along the RANK dim.
  return {RowSlice(lora_a[0], tp_rank_ * lora_a_rows_, lora_a_rows_)};
}

MatList MergedColumnParallelLinearWithShardedLoRA::SliceLoraA(
    const MatList& lora_a) const {
  // `:553-563`.
  MatList out(lora_a.size());
  for (size_t i = 0; i < lora_a.size(); ++i) {
    if (lora_a[i].empty()) continue;
    out[i] = RowSlice(lora_a[i], tp_rank_ * lora_a_rows_, lora_a_rows_);
  }
  return out;
}

MatList QKVParallelLinearWithShardedLoRA::SliceLoraA(
    const MatList& lora_a) const {
  // `:596-600`.
  return {RowSlice(lora_a[0], tp_rank_ * lora_a_rows_, lora_a_rows_)};
}

MatList MergedQKVParallelLinearWithShardedLoRA::SliceLoraA(
    const MatList& lora_a) const {
  // `:632-649` — three sub-LoRAs, any of which may be absent.
  MatList out(lora_a.size());
  for (size_t i = 0; i < lora_a.size(); ++i) {
    if (lora_a[i].empty()) continue;
    out[i] = RowSlice(lora_a[i], tp_rank_ * lora_a_rows_, lora_a_rows_);
  }
  return out;
}

MatList RowParallelLinearWithShardedLoRA::SliceLoraB(
    const MatList& lora_b) const {
  // row_parallel_linear.py:111-116 — shard along the OUTPUT dim; the shard size
  // is the width lora_b_stacked was created with.
  const int64_t shard_size = lora_b_rows_[0];
  return {RowSlice(lora_b[0], tp_rank_ * shard_size, shard_size)};
}

// ---------------------------------------------------------------------------
// VocabParallelEmbeddingWithLoRA (vocal_parallel_embedding.py:24-126)

void VocabParallelEmbeddingWithLoRA::CreateLoraWeights(int64_t max_loras,
                                                       int64_t max_lora_rank) {
  // `:49-71`. The added-vocab bookkeeping (`:30-47`) belongs with the adapter
  // loader that can produce an embeddings_tensor (W4).
  max_loras_ = max_loras;
  max_rank_ = max_lora_rank;
  lora_a_stacked_.assign(
      static_cast<size_t>(max_loras * org_vocab_size_ * max_rank_), 0.0f);
  lora_b_stacked_.assign(
      static_cast<size_t>(max_loras * embedding_dim_ * max_rank_), 0.0f);
}

void VocabParallelEmbeddingWithLoRA::ResetLora(int64_t index) {
  // `:73-75`.
  const int64_t a_stride = org_vocab_size_ * max_rank_;
  const int64_t b_stride = embedding_dim_ * max_rank_;
  std::fill_n(lora_a_stacked_.begin() + static_cast<std::ptrdiff_t>(index * a_stride),
              static_cast<size_t>(a_stride), 0.0f);
  std::fill_n(lora_b_stacked_.begin() + static_cast<std::ptrdiff_t>(index * b_stride),
              static_cast<size_t>(b_stride), 0.0f);
}

void VocabParallelEmbeddingWithLoRA::SetLora(int64_t index,
                                             const LoRAMat& lora_a,
                                             const LoRAMat& lora_b) {
  // `:77-94`. "self.lora_a_stacked is row-major, and lora_a is col-major, so we
  // need transpose here" — lora_a arrives as [rank, org_vocab_size] and is
  // stored as [org_vocab_size, rank].
  ResetLora(index);
  float* a = lora_a_stacked_.data() + index * org_vocab_size_ * max_rank_;
  for (int64_t v = 0; v < lora_a.cols; ++v) {
    for (int64_t r = 0; r < lora_a.rows; ++r) {
      a[v * max_rank_ + r] = lora_a.At(r, v);
    }
  }
  float* b = lora_b_stacked_.data() + index * embedding_dim_ * max_rank_;
  for (int64_t o = 0; o < lora_b.rows; ++o) {
    for (int64_t r = 0; r < lora_b.cols; ++r) {
      b[o * max_rank_ + r] = lora_b.At(o, r);
    }
  }
}

void VocabParallelEmbeddingWithLoRA::ApplyLoraToOutput(
    float* y, const int32_t* token_ids, int64_t T,
    const int32_t* indices) const {
  // `:99-121`. F.embedding(x + embeddings_indices[1], lora_a_stacked_2d) is a
  // gather of row `slot * org_vocab_size + token_id`; a base-model token uses
  // slot 0 there (utils.py:114) but is skipped by the expand, which reads the
  // per-token indices (-1 for base).
  std::vector<float> gathered(static_cast<size_t>(T * max_rank_), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t slot = GatherSlot(indices[t]);
    const int64_t row = slot * org_vocab_size_ + token_ids[t];
    const float* src = lora_a_stacked_.data() + row * max_rank_;
    float* dst = gathered.data() + t * max_rank_;
    for (int64_t r = 0; r < max_rank_; ++r) dst[r] = src[r];
  }
  AddLoraEmbedding(y, gathered.data(), T, max_rank_, lora_b_stacked_.data(),
                   max_loras_, embedding_dim_, indices, /*add_inputs=*/true);
}

// ---------------------------------------------------------------------------
// LogitsProcessorWithLoRA (logits_processor.py:84-194)

void LogitsProcessorWithLoRA::CreateLoraWeights(int64_t max_loras,
                                                int64_t max_lora_rank) {
  // `:90-92` — the guard is the reason a LoRA model is capped at 258048 tokens.
  if (vocab_size_ > 258048) {
    throw std::invalid_argument("When using LoRA, vocab size must be <= 258048");
  }
  max_loras_ = max_loras;
  max_rank_ = max_lora_rank;
  lora_a_stacked_.assign(static_cast<size_t>(max_loras * max_rank_ * hidden_size_),
                         0.0f);
  lora_b_stacked_.assign(static_cast<size_t>(max_loras * vocab_size_ * max_rank_),
                         0.0f);
}

void LogitsProcessorWithLoRA::ResetLora(int64_t index) {
  // `:121-123`.
  const int64_t a_stride = max_rank_ * hidden_size_;
  const int64_t b_stride = vocab_size_ * max_rank_;
  std::fill_n(lora_a_stacked_.begin() + static_cast<std::ptrdiff_t>(index * a_stride),
              static_cast<size_t>(a_stride), 0.0f);
  std::fill_n(lora_b_stacked_.begin() + static_cast<std::ptrdiff_t>(index * b_stride),
              static_cast<size_t>(b_stride), 0.0f);
}

void LogitsProcessorWithLoRA::SetLora(int64_t index, const LoRAMat& lora_a,
                                      const LoRAMat& lora_b) {
  // `:125-139`.
  ResetLora(index);
  float* a = lora_a_stacked_.data() + index * max_rank_ * hidden_size_;
  for (int64_t r = 0; r < lora_a.rows; ++r) {
    for (int64_t c = 0; c < lora_a.cols; ++c) a[r * hidden_size_ + c] = lora_a.At(r, c);
  }
  float* b = lora_b_stacked_.data() + index * vocab_size_ * max_rank_;
  for (int64_t o = 0; o < lora_b.rows; ++o) {
    for (int64_t r = 0; r < lora_b.cols; ++r) b[o * max_rank_ + r] = lora_b.At(o, r);
  }
}

void LogitsProcessorWithLoRA::ApplyLoraToLogits(
    float* logits, int64_t logits_width, const float* hidden_states, int64_t T,
    const int32_t* sampler_indices) const {
  // `:185-187` — add_lora_logits(logits, hidden_states, a, b, 1.0).
  AddLoraLogits(logits, logits_width, hidden_states, T, hidden_size_, vocab_size_,
                lora_a_stacked_.data(), lora_b_stacked_.data(), max_loras_,
                max_rank_, sampler_indices, /*scale=*/1.0);
}

std::vector<float> LogitsProcessorWithLoRA::ReindexShardedToFull(
    const std::vector<float>& logits, int64_t T, int64_t width) const {
  // `:166-183` — logits = logits[:, sharded_to_full_mapping].
  if (sharded_to_full_mapping_.empty()) return logits;
  std::vector<float> out(static_cast<size_t>(T) * sharded_to_full_mapping_.size());
  const int64_t n = static_cast<int64_t>(sharded_to_full_mapping_.size());
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t c = 0; c < n; ++c) {
      out[static_cast<size_t>(t * n + c)] =
          logits[static_cast<size_t>(t * width + sharded_to_full_mapping_[static_cast<size_t>(c)])];
    }
  }
  return out;
}

}  // namespace lora
}  // namespace vllm
