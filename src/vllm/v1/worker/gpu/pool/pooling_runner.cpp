// Ported from:
//   vllm/v1/worker/gpu/pool/pooling_runner.py:22-42
// @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/v1/worker/gpu/pool/pooling_runner.h"

#include <algorithm>

namespace vllm {

// pooling_runner.py:22 get_supported_tasks.
std::vector<PoolingTask> PoolingRunner::GetSupportedTasks(const Pooler& pooler) {
  const std::set<PoolingTask> supported = pooler.GetSupportedTasks();
  return std::vector<PoolingTask>(supported.begin(), supported.end());
}

// pooling_runner.py:41 — is_valid = (input_batch.seq_lens == prompt_len). On our
// synchronous CPU path both live in the cursor as host vectors.
std::vector<uint8_t> PoolingRunner::ComputeValid(
    const PoolingMetadata& metadata) const {
  const PoolingCursor& cursor = metadata.pooling_cursor;
  const int64_t n = cursor.num_seqs();
  std::vector<uint8_t> valid(static_cast<size_t>(n), 0);
  for (int64_t i = 0; i < n; ++i) {
    valid[i] = (cursor.seq_lens[i] == cursor.prompt_lens[i]) ? 1 : 0;
  }
  return valid;
}

}  // namespace vllm
