// Ported from:
//   vllm/pooling_params.py:35-70 (PoolingParams — the per-request API params)
//   vllm/tasks.py:10-18 (PoolingTask taxonomy)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING): the pooling task enum and the per-request
// pooling parameters the HEADS read — the matryoshka `dimensions` slice and the
// `use_activation` flag, plus the resolved `task`. Upstream `PoolingParams`
// additionally carries the tokwise step_tag_id / returned_token_ids and the
// internal late-interaction / plugin fields; those ride the tokwise (W5) and
// endpoint (W4) bricks and are the documented deferral.
//
// This is the lowest layer of the W2 pooler headers (no PooledData / tensor
// dependency), so `common.h`, `pooling_metadata.h`, and the heads can all build
// on it without an include cycle.
//
// LOCATION DEVIATION: upstream lives at the top-level `vllm/pooling_params.py`
// and `vllm/tasks.py`; we co-locate under `pooler/` so the whole W2 pooler
// composite is one additive directory. Semantics are faithful.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace vllm {

// vllm/tasks.py:10 PoolingTask (the Literal set). token_embed/token_classify and
// the plugin/combined tasks are carried for parity; the tokwise poolers that
// consume them are the W5 residual (spec §Work breakdown).
enum class PoolingTask {
  kEmbed,
  kClassify,
  kTokenEmbed,
  kTokenClassify,
  kPlugin,
  kEmbedAndTokenClassify,
};

// vllm/tasks.py:10 — the string spelling used on the wire and in DispatchPooler
// task maps.
inline std::string ToString(PoolingTask task) {
  switch (task) {
    case PoolingTask::kEmbed:
      return "embed";
    case PoolingTask::kClassify:
      return "classify";
    case PoolingTask::kTokenEmbed:
      return "token_embed";
    case PoolingTask::kTokenClassify:
      return "token_classify";
    case PoolingTask::kPlugin:
      return "plugin";
    case PoolingTask::kEmbedAndTokenClassify:
      return "embed&token_classify";
  }
  throw std::invalid_argument("Unknown pooling task");
}

inline PoolingTask PoolingTaskFromString(const std::string& task) {
  if (task == "embed") return PoolingTask::kEmbed;
  if (task == "classify") return PoolingTask::kClassify;
  if (task == "token_embed") return PoolingTask::kTokenEmbed;
  if (task == "token_classify") return PoolingTask::kTokenClassify;
  if (task == "plugin") return PoolingTask::kPlugin;
  if (task == "embed&token_classify") return PoolingTask::kEmbedAndTokenClassify;
  throw std::invalid_argument("Unknown pooling task: '" + task + "'");
}

// vllm/model_executor/layers/pooler/common.py:18 PoolingParamsUpdate — the
// merge-flag a pooler advertises so the engine knows to feed prompt token ids.
// `requires_token_ids` is set by the tokwise StepPool (W5); the sequence methods
// leave it false. Placed here (the lowest pooler header) so the seqwise method
// base can return it without an include cycle.
struct PoolingParamsUpdate {
  bool requires_token_ids = false;

  // common.py:23 __or__ — union of the flags.
  PoolingParamsUpdate operator|(const PoolingParamsUpdate& other) const {
    return PoolingParamsUpdate{requires_token_ids || other.requires_token_ids};
  }
};

// vllm/pooling_params.py:35 PoolingParams (the fields the seqwise heads read).
struct PoolingParams {
  // pooling_params.py:50 — None uses the pooler default (True in most cases);
  // the head treats an unset flag as "off" (torch `if None` is falsy).
  std::optional<bool> use_activation;
  // pooling_params.py:56 — matryoshka output dimension (truncate embeddings).
  std::optional<int64_t> dimensions;
  // pooling_params.py:64 — the resolved task (embed/classify/...).
  std::optional<PoolingTask> task;

  // Convenience: the head applies the activation when the flag is truthy.
  // Mirrors upstream `if flags[0]` / `... if f else vecs`.
  bool activation_enabled() const {
    return use_activation.has_value() && *use_activation;
  }
};

}  // namespace vllm
