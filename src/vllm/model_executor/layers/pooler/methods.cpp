// Ported from:
//   vllm/model_executor/layers/pooler/seqwise/methods.py:35-121
// @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/model_executor/layers/pooler/methods.h"

#include <stdexcept>

#include "vt/dtype.h"

namespace vllm {

namespace {

// Read hidden_states[row, col] as float32, honouring strides and dtype. Upstream
// forwards whatever dtype the model emitted; MeanPool upcasts to float32
// (methods.py:99 `.to(dtype=torch.float32)`), CLS/LAST keep the value — on the
// CPU path every pooled value is materialised as float32.
float ReadF32(const vt::Tensor& t, int64_t row, int64_t col) {
  const int64_t offset = row * t.stride[0] + col * t.stride[1];
  switch (t.dtype) {
    case vt::DType::kF32:
      return t.Ptr<float>()[offset];
    case vt::DType::kF16:
      return vt::F16ToF32(t.Ptr<uint16_t>()[offset]);
    case vt::DType::kBF16:
      return vt::BF16ToF32(t.Ptr<uint16_t>()[offset]);
    default:
      throw std::runtime_error(
          "pooler: hidden_states dtype must be f32/f16/bf16");
  }
}

int64_t HiddenSize(const vt::Tensor& hidden_states) {
  if (hidden_states.rank != 2) {
    throw std::runtime_error(
        "pooler: hidden_states must be rank-2 [num_tokens, hidden]");
  }
  return hidden_states.shape[1];
}

// Gather one full token row into the pooled buffer at output row `out_row`.
void CopyToken(const vt::Tensor& hidden_states, int64_t token_index,
               PooledData& out, int64_t out_row) {
  for (int64_t c = 0; c < out.cols; ++c) {
    out.data[out_row * out.cols + c] = ReadF32(hidden_states, token_index, c);
  }
}

}  // namespace

// methods.py:36 CLSPool.forward
PooledData CLSPool::Forward(const vt::Tensor& hidden_states,
                            const PoolingMetadata& metadata) const {
  const PoolingCursor& cursor = metadata.pooling_cursor;
  if (cursor.is_partial_prefill()) {
    throw std::runtime_error(
        "partial prefill is not supported with CLS pooling");
  }
  const int64_t hidden = HiddenSize(hidden_states);
  const int64_t num_seqs = cursor.num_seqs();
  PooledData out{std::vector<float>(static_cast<size_t>(num_seqs * hidden)),
                 num_seqs, hidden};
  for (int64_t i = 0; i < num_seqs; ++i) {
    CopyToken(hidden_states, cursor.first_token_indices[i], out, i);
  }
  return out;
}

// methods.py:49 LastPool.forward
PooledData LastPool::Forward(const vt::Tensor& hidden_states,
                             const PoolingMetadata& metadata) const {
  const PoolingCursor& cursor = metadata.pooling_cursor;
  const int64_t hidden = HiddenSize(hidden_states);
  const int64_t num_seqs = cursor.num_seqs();
  PooledData out{std::vector<float>(static_cast<size_t>(num_seqs * hidden)),
                 num_seqs, hidden};
  for (int64_t i = 0; i < num_seqs; ++i) {
    CopyToken(hidden_states, cursor.last_token_indices[i], out, i);
  }
  return out;
}

// methods.py:60 MeanPool.forward
PooledData MeanPool::Forward(const vt::Tensor& hidden_states,
                             const PoolingMetadata& metadata) const {
  const PoolingCursor& cursor = metadata.pooling_cursor;
  if (cursor.is_partial_prefill()) {
    throw std::runtime_error(
        "partial prefill is not supported with MEAN pooling");
  }
  const int64_t hidden = HiddenSize(hidden_states);
  const int64_t num_seqs = cursor.num_seqs();
  // methods.py:71 early return for an empty batch: shape (0, hidden), float32.
  PooledData out{std::vector<float>(static_cast<size_t>(num_seqs * hidden)),
                 num_seqs, hidden};
  for (int64_t i = 0; i < num_seqs; ++i) {
    const int64_t start = cursor.first_token_indices[i];
    const int64_t count = cursor.prompt_lens[i];
    // methods.py:83-104 — accumulate in float32, then divide by the prompt len.
    for (int64_t c = 0; c < hidden; ++c) {
      float acc = 0.0f;
      for (int64_t t = 0; t < count; ++t) {
        acc += ReadF32(hidden_states, start + t, c);
      }
      out.data[i * hidden + c] = acc / static_cast<float>(count);
    }
  }
  return out;
}

// methods.py:109 get_seq_pooling_method
std::unique_ptr<SequencePoolingMethod> GetSeqPoolingMethod(
    SequencePoolingType pooling_type) {
  switch (pooling_type) {
    case SequencePoolingType::kCLS:
      return std::make_unique<CLSPool>();
    case SequencePoolingType::kLast:
      return std::make_unique<LastPool>();
    case SequencePoolingType::kMean:
      return std::make_unique<MeanPool>();
  }
  throw std::invalid_argument("Unknown sequence pooling type");
}

std::unique_ptr<SequencePoolingMethod> GetSeqPoolingMethod(
    const std::string& pooling_type) {
  if (pooling_type == "CLS") return std::make_unique<CLSPool>();
  if (pooling_type == "LAST") return std::make_unique<LastPool>();
  if (pooling_type == "MEAN") return std::make_unique<MeanPool>();
  throw std::invalid_argument("Unknown sequence pooling type: '" +
                              pooling_type + "'");
}

}  // namespace vllm
