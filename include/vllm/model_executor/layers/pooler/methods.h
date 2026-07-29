// Ported from:
//   vllm/model_executor/layers/pooler/seqwise/methods.py:35-121
//     (SequencePoolingMethod, CLSPool, LastPool, MeanPool, get_seq_pooling_method)
//   vllm/config/pooler.py:16 (SequencePoolingType = "CLS"|"LAST"|"MEAN")
// @ 555967922 (vLLM 0.26.0.dev0).
//
// CPU W1 brick (CLAIM-POOLING): the sequence-pooling OP — extract or aggregate
// the per-sequence pooled vector from a packed [num_tokens, hidden] hidden-state
// buffer, keyed by a PoolingCursor. This is the non-generative-runner primitive
// that turns hidden states into an embedding/logit row instead of a sampled
// token.
#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/pooler/pooling_metadata.h"
#include "vt/tensor.h"

namespace vllm {

// vllm/config/pooler.py:16 SequencePoolingType.
enum class SequencePoolingType { kCLS, kLast, kMean };

// Owning result of a sequence pooling op: a row-major [rows, cols] buffer, one
// pooled vector per sequence. Upstream returns a torch.Tensor [batch, hidden];
// MeanPool always yields float32 (it upcasts), CLS/LAST preserve the input
// values (here always materialised as float32 on the CPU path).
struct PooledData {
  std::vector<float> data;
  int64_t rows = 0;
  int64_t cols = 0;

  float At(int64_t r, int64_t c) const { return data[r * cols + c]; }
};

// vllm/model_executor/layers/pooler/seqwise/methods.py:22 SequencePoolingMethod.
// hidden_states is a CPU rank-2 tensor [num_tokens, hidden] of kF32/kF16/kBF16
// (upstream forwards whatever dtype the model produced; we read all three and
// materialise float32).
class SequencePoolingMethod {
 public:
  virtual ~SequencePoolingMethod() = default;
  virtual PooledData Forward(const vt::Tensor& hidden_states,
                             const PoolingMetadata& metadata) const = 0;

  // methods.py:22 get_supported_tasks — every sequence method serves the four
  // seqwise/tokwise-compatible tasks; the head narrows this (SequencePooler
  // intersects method ∩ head). W2 addition (defaulted, non-breaking for W1).
  virtual std::set<PoolingTask> GetSupportedTasks() const {
    return {PoolingTask::kTokenEmbed, PoolingTask::kTokenClassify,
            PoolingTask::kEmbed, PoolingTask::kClassify};
  }

  // methods.py:24 get_pooling_updates — the sequence methods advertise no
  // token-id requirement (StepPool, which does, is the W5 tokwise brick).
  virtual PoolingParamsUpdate GetPoolingUpdates(PoolingTask /*task*/) const {
    return PoolingParamsUpdate{};
  }
};

// methods.py:36 CLSPool — the first token of each sequence. Rejects partial
// prefill (the first token may not be in this chunk).
class CLSPool : public SequencePoolingMethod {
 public:
  PooledData Forward(const vt::Tensor& hidden_states,
                     const PoolingMetadata& metadata) const override;
};

// methods.py:49 LastPool — the last scheduled token of each sequence. Partial
// prefill IS allowed (returns the last token seen so far).
class LastPool : public SequencePoolingMethod {
 public:
  PooledData Forward(const vt::Tensor& hidden_states,
                     const PoolingMetadata& metadata) const override;
};

// methods.py:60 MeanPool — mean over each sequence's tokens, accumulated in
// float32 (upstream upcasts to float32 before summing). Rejects partial prefill.
class MeanPool : public SequencePoolingMethod {
 public:
  PooledData Forward(const vt::Tensor& hidden_states,
                     const PoolingMetadata& metadata) const override;
};

// methods.py:109 get_seq_pooling_method.
std::unique_ptr<SequencePoolingMethod> GetSeqPoolingMethod(
    SequencePoolingType pooling_type);
std::unique_ptr<SequencePoolingMethod> GetSeqPoolingMethod(
    const std::string& pooling_type);

}  // namespace vllm
