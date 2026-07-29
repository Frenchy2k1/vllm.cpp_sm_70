// Ported from:
//   vllm/config/pooler.py:16,30-90 (SequencePoolingType, PoolerConfig,
//     get_seq_pooling_type)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING): the model-level pooler configuration that the
// `pooler_for_embed` / `pooler_for_classify` factories read — the sequence
// pooling type, the classifier affine calibration (logit_mean/logit_sigma), the
// default activation flag, and the matryoshka default dimension.
//
// LOCATION DEVIATION: upstream lives at `vllm/config/pooler.py`; we co-locate it
// under `pooler/` so the whole W2 composite is one additive directory, and reuse
// the `SequencePoolingType` enum already defined by the W1 methods brick rather
// than redeclaring it. Semantics are faithful. The pydantic `pooling_type ->
// seq_pooling_type` auto-population, the chunked-processing / max-embed-len
// fields, and the tokwise `tok_pooling_type` are deferred (documented) to the
// endpoint / tokwise bricks.
#pragma once

#include <optional>
#include <string>

#include "vllm/model_executor/layers/pooler/methods.h"  // SequencePoolingType

namespace vllm {

// vllm/config/pooler.py:30 PoolerConfig (the fields the seqwise factories read).
struct PoolerConfig {
  // config/pooler.py:49 seq_pooling_type. Unset falls back to the model
  // default resolved by get_seq_pooling_type() below.
  std::optional<SequencePoolingType> seq_pooling_type;
  // config/pooler.py:60 use_activation (pooler default when a request leaves it
  // unset). None => the pooler's own default.
  std::optional<bool> use_activation;
  // config/pooler.py:66 dimensions (default matryoshka slice).
  std::optional<int64_t> dimensions;
  // config/pooler.py logit_mean / logit_sigma — the ClassifierPoolerHead affine
  // score calibration `(logit - mean) / sigma`.
  std::optional<double> logit_mean;
  std::optional<double> logit_sigma;

  // config/pooler.py get_seq_pooling_type(): the configured type, or the
  // supplied default when unset. Upstream raises when neither is available; the
  // factory passes the model's architectural default.
  SequencePoolingType GetSeqPoolingType(SequencePoolingType fallback) const {
    return seq_pooling_type.value_or(fallback);
  }
};

}  // namespace vllm
