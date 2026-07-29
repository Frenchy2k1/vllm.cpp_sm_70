// Ported from:
//   vllm/v1/worker/gpu/pool/pooling_runner.py:18-46 (PoolingRunner)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W3 brick (CLAIM-POOLING): the pooling RUNNER path — where the generation
// runner would SAMPLE a token, the pooling runner applies the model's `Pooler`
// to the last hidden state and returns the POOLED DATA (an embedding vector / a
// classification logit row). This is the non-generative counterpart of the
// token sampler.
//
// GENERALIZATION DEVIATION (recorded): upstream `PoolingRunner.pool` is
// hardcoded to LAST-token + L2-normalize ("Currently ... only supports the LAST
// pooling task on decoder-only models", woosuk NOTE). We route through the
// model's `Pooler` (the general bert.py / *EmbeddingModel path — `model.pooler(
// hidden_states, pooling_metadata)`), which is strictly more capable (CLS/MEAN/
// LAST, embed/classify, DispatchPooler task routing) and mirrors how upstream
// pooling MODELS actually pool. The hardcoded LAST path is the degenerate case
// of an `EmbeddingPoolerHead(activation=PoolerNormalize)` over `LastPool`.
//
// Deferred (documented): the InputBatch -> PoolingMetadata construction (the
// `logits_indices` gather, GPU->CPU cursor build) is the full model-runner
// integration and rides the endpoint brick (W4); this runner consumes a
// hidden-state buffer + a PoolingMetadata, the reusable seam a concrete pooling
// model's forward feeds.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/layers/pooler/poolers.h"
#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vt/tensor.h"

namespace vllm {

// pooling_runner.py:18 PoolingRunner.
class PoolingRunner {
 public:
  explicit PoolingRunner(const Pooler& pooler) : pooler_(&pooler) {}

  // pooling_runner.py:22 get_supported_tasks. Upstream asserts "embed" is
  // supported and returns ["embed"] (its LAST-only limitation); our general
  // path returns the model pooler's actual supported task set.
  static std::vector<PoolingTask> GetSupportedTasks(const Pooler& pooler);

  // pooling_runner.py:29 pool — apply the model's pooler to the hidden states.
  // `hidden_states` is the packed [num_tokens, hidden] last-hidden-state buffer;
  // the metadata's cursor names each sequence's first/last token. Returns the
  // pooled per-sequence vectors (the runner's "output" in place of sampled
  // tokens).
  PoolerOutput Pool(const vt::Tensor& hidden_states,
                    const PoolingMetadata& metadata) const {
    return pooler_->Forward(hidden_states, metadata);
  }

  // pooling_runner.py:41 is_valid — a pooled request is valid once its whole
  // prompt has been seen (seq_lens == prompt_len). One flag per sequence.
  std::vector<uint8_t> ComputeValid(const PoolingMetadata& metadata) const;

 private:
  const Pooler* pooler_;
};

}  // namespace vllm
