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
// Deferred to W2+ (not needed by the seqwise methods, recorded so the shell is
// honest): `pooling_params`, `prompt_token_ids{,_cpu}`, `pooling_states`
// (chunked-prefill ALL pooling), and `tasks`. These are consumed by the pooler
// HEADS (matryoshka dims / per-request activation flags), the tokwise methods
// (StepPool token-id filtering), and the DispatchPooler task routing, all of
// which land with the runner brick.
#pragma once

#include <cstdint>
#include <vector>

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
};

// Mirror of vllm/v1/pool/metadata.py:47 PoolingMetadata, reduced to the single
// field the sequence pooling methods read on the CPU path. Heads/tokwise fields
// are the W2+ deferral documented above.
struct PoolingMetadata {
  PoolingCursor pooling_cursor;
};

}  // namespace vllm
