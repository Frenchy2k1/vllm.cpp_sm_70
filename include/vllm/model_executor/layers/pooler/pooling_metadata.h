// Ported from:
//   vllm/v1/pool/metadata.py:13-33 (PoolingCursor, is_partial_prefill)
//   vllm/v1/pool/metadata.py:47-71 (PoolingMetadata shell)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W1 brick (CLAIM-POOLING): the minimal pooling metadata the sequence
// pooling methods (CLS/LAST/MEAN) actually read. Upstream `PoolingCursor`
// carries GPU index tensors plus CPU length tensors; on our synchronous CPU
// path the "..._gpu" index tensors are host int64 vectors of flat token
// indices into the packed [num_tokens, hidden] hidden-state buffer.
//
// W2 (CLAIM-POOLING) ADDS the `pooling_params` and `tasks` fields the pooler
// HEADS (matryoshka dims / per-request activation flags) and the DispatchPooler
// task routing read. Still deferred (recorded so the shell is honest):
// `prompt_token_ids{,_cpu}` and `pooling_states` (chunked-prefill ALL pooling),
// consumed by the tokwise methods (StepPool token-id filtering) and chunked
// prefill — the W5 residual.
#pragma once

#include <cstdint>
#include <vector>

#include <string>

#include "vllm/model_executor/layers/pooler/pooling_params.h"

namespace vllm {

// Mirror of vllm/v1/pool/metadata.py:13 PoolingCursor. Index vectors are flat
// token offsets into the packed hidden-state buffer; the "_cpu" length vectors
// mirror the identically named upstream CPU tensors.
struct PoolingCursor {
  std::vector<int64_t> first_token_indices;   // first_token_indices_gpu
  std::vector<int64_t> last_token_indices;    // last_token_indices_gpu
  std::vector<int64_t> prompt_lens;           // prompt_lens_cpu
  std::vector<int64_t> seq_lens;              // seq_lens_cpu
  std::vector<int64_t> num_scheduled_tokens;  // num_scheduled_tokens_cpu

  int64_t num_seqs() const {
    return static_cast<int64_t>(prompt_lens.size());
  }

  // vllm/v1/pool/metadata.py:31 — `not torch.all(prompt_lens == num_scheduled)`.
  // True when any sequence has more prompt tokens than were scheduled this step
  // (a partial prefill chunk), which CLS/MEAN pooling reject.
  bool is_partial_prefill() const {
    return prompt_lens != num_scheduled_tokens;
  }

  // vllm/v1/pool/metadata.py:38 __getitem__ — the contiguous seq subrange, used
  // by DispatchPooler to hand each task's group to its sub-pooler. Token indices
  // stay ABSOLUTE (see the DispatchPooler deviation note): because our
  // sub-poolers index the full [num_tokens, hidden] buffer directly, no
  // token-offset shift is needed.
  PoolingCursor Slice(int64_t seq_offset, int64_t count) const {
    auto sub = [seq_offset, count](const std::vector<int64_t>& v) {
      return std::vector<int64_t>(v.begin() + seq_offset,
                                  v.begin() + seq_offset + count);
    };
    PoolingCursor out;
    out.first_token_indices = sub(first_token_indices);
    out.last_token_indices = sub(last_token_indices);
    out.prompt_lens = sub(prompt_lens);
    out.seq_lens = sub(seq_lens);
    out.num_scheduled_tokens = sub(num_scheduled_tokens);
    return out;
  }
};

// Mirror of vllm/v1/pool/metadata.py:47 PoolingMetadata. W1 carried only the
// cursor (the seqwise methods' single dependency); W2 adds the per-request
// `pooling_params` (the heads read matryoshka dims / activation flags) and the
// `tasks` list (DispatchPooler routes on it). `prompt_token_ids{,_cpu}` and
// `pooling_states` remain the W5 (tokwise / chunked-prefill) deferral.
struct PoolingMetadata {
  PoolingCursor pooling_cursor;
  // vllm/v1/pool/metadata.py:52 pooling_params — one per sequence, aligned with
  // the cursor rows.
  std::vector<PoolingParams> pooling_params;
  // vllm/v1/pool/metadata.py — the resolved per-sequence task (DispatchPooler
  // groupby key). Aligned with the cursor rows.
  std::vector<PoolingTask> tasks;

  // vllm/v1/pool/metadata.py:38 __getitem__ — the contiguous seq subrange.
  PoolingMetadata Slice(int64_t seq_offset, int64_t count) const {
    PoolingMetadata out;
    out.pooling_cursor = pooling_cursor.Slice(seq_offset, count);
    if (!pooling_params.empty()) {
      out.pooling_params.assign(pooling_params.begin() + seq_offset,
                                pooling_params.begin() + seq_offset + count);
    }
    if (!tasks.empty()) {
      out.tasks.assign(tasks.begin() + seq_offset,
                       tasks.begin() + seq_offset + count);
    }
    return out;
  }
};

}  // namespace vllm
