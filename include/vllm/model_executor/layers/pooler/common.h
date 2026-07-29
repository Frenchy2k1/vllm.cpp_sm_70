// Ported from:
//   vllm/model_executor/layers/pooler/common.py:12-30
//     (ProjectorFn, ClassifierFn, ActivationFn, PoolingParamsUpdate)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W2 brick (CLAIM-POOLING): the shared pooler vocabulary — the pooled-output
// container, the projector/classifier callable seams, and the
// PoolingParamsUpdate flag. These are the types the pooler HEADS,
// `SequencePooler`, and `DispatchPooler` compose over. The PoolingTask enum lives
// in the lower-level `pooling_params.h`.
#pragma once

#include <functional>

#include "vllm/model_executor/layers/pooler/methods.h"         // PooledData
#include "vllm/model_executor/layers/pooler/pooling_params.h"  // PoolingTask

namespace vllm {

// The pooled result: one vector per sequence (mirror of upstream
// `torch.Tensor | list[torch.Tensor]`). We always materialise the per-sequence
// list form; the head's uniform-vs-list branching is a torch dtype/return-shape
// nuance with no numeric effect (recorded deviation, see heads.h).
using PoolerOutput = std::vector<std::vector<float>>;

// common.py:12-14 — the linear/projector/activation callable seams. The
// projector and classifier map a packed [batch, in] PooledData to a packed
// [batch, out] PooledData (upstream applies an nn.Linear over the whole batch;
// a row-wise or whole-batch matmul is value-identical).
using ProjectorFn = std::function<PooledData(const PooledData&)>;
using ClassifierFn = std::function<PooledData(const PooledData&)>;

// (PoolingParamsUpdate lives in pooling_params.h — the lowest pooler header — so
// the seqwise method base can return it without an include cycle.)

}  // namespace vllm
