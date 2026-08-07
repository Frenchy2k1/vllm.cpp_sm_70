// Parakeet RNN-T / TDT transducer heads: the forward and the greedy decode.
// Upstream citations, the recorded correction and the traced evidence are in
// include/vllm/model_executor/models/parakeet_transducer.h; line numbers below
// are `transformers/models/parakeet/modeling_parakeet.py` unless prefixed with
// `gen:`, which means `generation_parakeet.py`.
//
// Op routing. The two GEMMs that carry real work go through `vt::MatmulBT`: the
// encoder projector (:930, one [T, hidden] x [D, hidden] over the whole clip)
// and the joint head (:885/:1044, a [1, D] x [joint_out, D] per emitted step,
// with the weight uploaded ONCE for the whole decode rather than per step). The
// LSTM cell stays on the host on purpose: at decoder_hidden_size 640 a step is
// two ~1.6 MFLOP GEMVs, so marshalling its four weight matrices across the op
// boundary every step would cost more than the arithmetic it replaces, and this
// row's gate is faithfulness (the family's speed gate is GB10-only, spike
// § Gates, and is not claimed here). Same contract as the P4 encoder.
#include "vllm/model_executor/models/parakeet_transducer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/ops.h"

namespace vllm::multimodal {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// Same RAII device buffer the encoder forward uses.
struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;

  Buf(Backend& backend, Queue& q, const std::vector<int64_t>& shape,
      const void* host = nullptr)
      : b(backend) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(DType::kF32);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    t.data = p;
    t.dtype = DType::kF32;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    if (host != nullptr) b.Copy(q, p, host, bytes);
  }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;

  Tensor& tensor() { return t; }
  std::vector<float> Download(Queue& q) {
    std::vector<float> out(bytes / sizeof(float));
    b.Copy(q, out.data(), p, bytes);
    b.Synchronize(q);
    return out;
  }
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

std::unique_ptr<Buf> Up(Backend& b, Queue& q, const std::vector<float>& host,
                        const std::vector<int64_t>& shape) {
  if (static_cast<int64_t>(host.size()) != Numel(shape)) {
    throw std::runtime_error("parakeet: tensor size mismatch on upload");
  }
  return std::make_unique<Buf>(b, q, shape, host.data());
}

// torch `nn.Linear` over [M, K] with weight [N, K].
std::vector<float> Linear(Backend& b, Queue& q, const std::vector<float>& x, int64_t M,
                          int64_t K, const std::vector<float>& w,
                          const std::vector<float>& bias, int64_t N) {
  auto xb = Up(b, q, x, {M, K});
  auto wb = Up(b, q, w, {N, K});
  Buf out(b, q, {M, N});
  vt::MatmulBT(q, out.tensor(), xb->tensor(), wb->tensor());
  if (!bias.empty()) {
    auto bb = Up(b, q, bias, {N});
    vt::Add(q, out.tensor(), out.tensor(), bb->tensor());
  }
  return out.Download(q);
}

// `ACT2FN[config.hidden_act]` for the joint (:884). Only relu is ported, which
// is what every published transducer checkpoint sets (:170).
void ApplyJointActivation(std::vector<float>* v, const std::string& act) {
  if (act != "relu") {
    throw std::runtime_error("parakeet: unsupported joint hidden_act '" + act +
                             "' (only relu is ported)");
  }
  for (float& x : *v) x = x > 0.0f ? x : 0.0f;
}

// `ParakeetRNNTJointNetwork.forward` (:888-894) with the head weight resident
// across the whole decode. `head` is [joint_out, D]; one step is a [1, D] GEMV.
class JointRunner {
 public:
  JointRunner(Backend& backend, const ParakeetForTransducerWeights& w,
              const ParakeetTransducerConfig& cfg)
      : b_(backend), q_(backend.CreateQueue()), cfg_(cfg), d_(cfg.decoder_hidden_size),
        n_(cfg.joint_output_size()),
        head_(Up(backend, q_, w.joint_head_w, {n_, d_})),
        bias_(w.joint_head_b.empty() ? nullptr
                                     : Up(backend, q_, w.joint_head_b, {n_})) {}
  ~JointRunner() { b_.DestroyQueue(q_); }
  JointRunner(const JointRunner&) = delete;
  JointRunner& operator=(const JointRunner&) = delete;

  std::vector<float> Run(const float* encoder_frame,
                         const std::vector<float>& decoder_output) {
    // :893: the sum happens BEFORE the activation, and both operands are
    // already in the decoder's width (the encoder side was projected at :949).
    std::vector<float> joined(static_cast<size_t>(d_));
    for (int64_t i = 0; i < d_; ++i) {
      joined[static_cast<size_t>(i)] =
          encoder_frame[i] + decoder_output[static_cast<size_t>(i)];
    }
    ApplyJointActivation(&joined, cfg_.hidden_act);
    auto xb = Up(b_, q_, joined, {1, d_});
    Buf out(b_, q_, {1, n_});
    vt::MatmulBT(q_, out.tensor(), xb->tensor(), head_->tensor());
    if (bias_ != nullptr) vt::Add(q_, out.tensor(), out.tensor(), bias_->tensor());
    return out.Download(q_);
  }

 private:
  Backend& b_;
  Queue q_;
  const ParakeetTransducerConfig& cfg_;
  int64_t d_;
  int64_t n_;
  std::unique_ptr<Buf> head_;
  std::unique_ptr<Buf> bias_;
};

// `torch.argmax` over a half-open column range. torch returns the FIRST maximal
// index, so the comparison must be STRICT.
int64_t ArgMax(const std::vector<float>& v, int64_t begin, int64_t end) {
  int64_t best = begin;
  for (int64_t i = begin + 1; i < end; ++i) {
    if (v[static_cast<size_t>(i)] > v[static_cast<size_t>(best)]) best = i;
  }
  return best;
}

}  // namespace

void ParakeetLstmCell(const std::vector<float>& x, const ParakeetLstmLayerWeights& w,
                      int64_t input_size, int64_t hidden_size, std::vector<float>* h,
                      std::vector<float>* c) {
  const int64_t H = hidden_size;
  if (static_cast<int64_t>(x.size()) != input_size ||
      static_cast<int64_t>(h->size()) != H || static_cast<int64_t>(c->size()) != H ||
      static_cast<int64_t>(w.weight_ih.size()) != 4 * H * input_size ||
      static_cast<int64_t>(w.weight_hh.size()) != 4 * H * H) {
    throw std::runtime_error("parakeet: LSTM cell shape mismatch");
  }
  const bool has_bias = !w.bias_ih.empty();

  // gates[g*H + j] for g in {i, f, g, o}: torch's row packing order.
  std::vector<double> gates(static_cast<size_t>(4 * H), 0.0);
  for (int64_t r = 0; r < 4 * H; ++r) {
    double acc = 0.0;
    const float* wi = &w.weight_ih[static_cast<size_t>(r) * input_size];
    for (int64_t k = 0; k < input_size; ++k) acc += static_cast<double>(wi[k]) * x[static_cast<size_t>(k)];
    const float* wh = &w.weight_hh[static_cast<size_t>(r) * H];
    for (int64_t k = 0; k < H; ++k) acc += static_cast<double>(wh[k]) * (*h)[static_cast<size_t>(k)];
    // torch applies BOTH bias vectors (`b_ih + b_hh`), a cuDNN-compat
    // redundancy; dropping either would silently halve the bias.
    if (has_bias) {
      acc += static_cast<double>(w.bias_ih[static_cast<size_t>(r)]) +
             static_cast<double>(w.bias_hh[static_cast<size_t>(r)]);
    }
    gates[static_cast<size_t>(r)] = acc;
  }

  auto sigmoid = [](double z) { return 1.0 / (1.0 + std::exp(-z)); };
  for (int64_t j = 0; j < H; ++j) {
    const double i_g = sigmoid(gates[static_cast<size_t>(j)]);
    const double f_g = sigmoid(gates[static_cast<size_t>(H + j)]);
    const double g_g = std::tanh(gates[static_cast<size_t>(2 * H + j)]);
    const double o_g = sigmoid(gates[static_cast<size_t>(3 * H + j)]);
    const double cn = f_g * static_cast<double>((*c)[static_cast<size_t>(j)]) + i_g * g_g;
    (*c)[static_cast<size_t>(j)] = static_cast<float>(cn);
    (*h)[static_cast<size_t>(j)] = static_cast<float>(o_g * std::tanh(cn));
  }
}

void ParakeetRNNTDecoderStep(int32_t input_id, const ParakeetRNNTDecoderWeights& w,
                             const ParakeetTransducerConfig& cfg,
                             ParakeetRNNTDecoderState* state) {
  const int64_t H = cfg.decoder_hidden_size;
  const int64_t layers = cfg.num_decoder_layers;

  // gen:851-855: the BLANK FAST PATH. Once the cache is initialized, a blank
  // input returns the cached output and advances NOTHING: no LSTM step, no
  // hidden/cell update. This is what makes a blank a pure encoder-frame advance.
  if (state->initialized && input_id == cfg.blank_token_id) return;

  if (!state->initialized) {
    // gen:31-58 lazy_initialization: zeros for (h, c) AND for `cache.cache`.
    state->hidden.assign(static_cast<size_t>(layers * H), 0.0f);
    state->cell.assign(static_cast<size_t>(layers * H), 0.0f);
    state->output.assign(static_cast<size_t>(H), 0.0f);
    state->initialized = true;
  }
  if (input_id < 0 || input_id >= cfg.vocab_size) {
    throw std::runtime_error("parakeet: decoder input id " + std::to_string(input_id) +
                             " is outside the vocabulary");
  }

  // :857 self.embedding(input_ids)
  std::vector<float> x(
      w.embedding.begin() + static_cast<ptrdiff_t>(input_id) * H,
      w.embedding.begin() + static_cast<ptrdiff_t>(input_id + 1) * H);

  // :868 the stacked LSTM: layer k consumes layer k-1's output at this step.
  for (int64_t l = 0; l < layers; ++l) {
    std::vector<float> h(state->hidden.begin() + static_cast<ptrdiff_t>(l * H),
                         state->hidden.begin() + static_cast<ptrdiff_t>((l + 1) * H));
    std::vector<float> c(state->cell.begin() + static_cast<ptrdiff_t>(l * H),
                         state->cell.begin() + static_cast<ptrdiff_t>((l + 1) * H));
    ParakeetLstmCell(x, w.lstm[static_cast<size_t>(l)], static_cast<int64_t>(x.size()), H,
                     &h, &c);
    std::copy(h.begin(), h.end(), state->hidden.begin() + static_cast<ptrdiff_t>(l * H));
    std::copy(c.begin(), c.end(), state->cell.begin() + static_cast<ptrdiff_t>(l * H));
    x = h;
  }

  // :869 self.decoder_projector(lstm_output): a plain [H, H] Linear, kept on
  // the host with the cell for the reason in the file header.
  std::vector<float> out(static_cast<size_t>(H), 0.0f);
  for (int64_t r = 0; r < H; ++r) {
    double acc = w.projector_b.empty() ? 0.0
                                       : static_cast<double>(w.projector_b[static_cast<size_t>(r)]);
    const float* row = &w.projector_w[static_cast<size_t>(r) * H];
    for (int64_t k = 0; k < H; ++k) acc += static_cast<double>(row[k]) * x[static_cast<size_t>(k)];
    out[static_cast<size_t>(r)] = static_cast<float>(acc);
  }
  state->output = std::move(out);  // gen:73 cache.cache.copy_(decoder_output)
}

std::vector<float> ParakeetTransducerJoint(const float* encoder_frame,
                                           const std::vector<float>& decoder_output,
                                           const ParakeetForTransducerWeights& w,
                                           const ParakeetTransducerConfig& cfg,
                                           Backend& backend) {
  JointRunner runner(backend, w, cfg);
  return runner.Run(encoder_frame, decoder_output);
}

ParakeetTransducerOutput ParakeetTransducerGreedyDecode(
    const std::vector<float>& encoder_projected, int64_t frames, int64_t valid_frames,
    const ParakeetForTransducerWeights& w, const ParakeetTransducerConfig& cfg,
    Backend& backend) {
  const int64_t D = cfg.decoder_hidden_size;
  if (static_cast<int64_t>(encoder_projected.size()) != frames * D) {
    throw std::runtime_error("parakeet: encoder_projected size mismatch");
  }
  valid_frames = std::min(std::max<int64_t>(valid_frames, 0), frames);

  ParakeetTransducerOutput out;
  out.encoder_frames = frames;
  out.valid_encoder_frames = valid_frames;
  if (frames == 0 || valid_frames == 0) return out;

  JointRunner joint(backend, w, cfg);
  ParakeetRNNTDecoderState state;

  // gen:222-226 the frame pointer starts at 0; `generate()` seeds `sequences`
  // with `decoder_start_token_id` and gen:259-262 pads `durations` to match.
  const int32_t start = cfg.start_token();
  out.sequences.push_back(start);
  out.durations.push_back(0);

  int64_t t = 0;
  int64_t symbols_at_frame = 0;
  int32_t last = start;
  // gen:177-181: `max_length = max_symbols_per_step * encoder_seq_len`, and
  // `MaxLengthCriteria` measures the WHOLE sequence, start token included.
  const int64_t max_length = cfg.max_symbols_per_step * frames;

  while (true) {
    ParakeetRNNTDecoderStep(last, w.decoder, cfg, &state);

    // gen:243-246: the gather is CLAMPED to the last frame, so the step that
    // exhausts the encoder still has a well-defined row to joint against.
    const int64_t frame = std::min(t, frames - 1);
    const std::vector<float> logits =
        joint.Run(&encoder_projected[static_cast<size_t>(frame) * D], state.output);

    // Token selection over the VOCABULARY columns only: see the header for why
    // that is exactly what the published TDT `suppress_tokens` does, and why it
    // is vacuous for RNN-T.
    const int32_t token = static_cast<int32_t>(ArgMax(logits, 0, cfg.vocab_size));
    out.sequences.push_back(token);

    int64_t advance = 0;
    if (cfg.is_tdt()) {
      // gen:288 the duration head's argmax indexes `config.durations`.
      const int64_t didx =
          ArgMax(logits, cfg.vocab_size, cfg.joint_output_size()) - cfg.vocab_size;
      advance = cfg.durations[static_cast<size_t>(didx)];
      // gen:291-292: a blank with a zero duration would stall the walk, so it
      // is forced to one frame. Non-blank zero durations are left alone: that is
      // how TDT emits several symbols at one frame.
      if (token == cfg.blank_token_id && advance == 0) advance = 1;
    } else {
      // gen:148-157 the RNN-T rule.
      const bool blank = token == cfg.blank_token_id;
      const int64_t symbols = blank ? 0 : symbols_at_frame + 1;
      const bool force = symbols >= cfg.max_symbols_per_step;
      symbols_at_frame = (blank || force) ? 0 : symbols;
      advance = (blank || force) ? 1 : 0;
    }
    t += advance;
    out.durations.push_back(static_cast<int32_t>(advance));
    last = token;

    // gen:161 / gen:113-122 EncoderExhaustedCriteria, then `EosTokenCriteria`
    // and `MaxLengthCriteria`: all evaluated AFTER the token is appended.
    if (t >= valid_frames) break;
    if (std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), token) !=
        cfg.eos_token_ids.end()) {
      break;
    }
    if (static_cast<int64_t>(out.sequences.size()) >= max_length) break;
  }

  // processing_parakeet.py:128-130: a transducer does NOT collapse repeats
  // (group_tokens is false for rnnt/tdt); only the blank is dropped. The start
  // token is skipped explicitly rather than relying on it being the blank.
  for (size_t i = 1; i < out.sequences.size(); ++i) {
    if (out.sequences[i] != cfg.blank_token_id) out.token_ids.push_back(out.sequences[i]);
  }
  return out;
}

ParakeetTransducerOutput ParakeetForTransducerForward(
    const std::vector<float>& input_features, int64_t num_frames, int64_t valid_frames,
    const ParakeetForTransducerWeights& w, const ParakeetEncoderConfig& enc_cfg,
    const ParakeetTransducerConfig& cfg, Backend& backend) {
  // :944-948 the encoder.
  int64_t valid_rows = 0;
  const std::vector<float> hidden = ParakeetEncoderForward(
      input_features, num_frames, valid_frames, w.encoder, enc_cfg, backend, &valid_rows,
      nullptr);
  const int64_t rows =
      enc_cfg.hidden_size == 0 ? 0 : static_cast<int64_t>(hidden.size()) / enc_cfg.hidden_size;

  // :949 encoder_outputs.pooler_output = self.encoder_projector(last_hidden_state)
  Queue q = backend.CreateQueue();
  std::vector<float> projected;
  try {
    projected = Linear(backend, q, hidden, rows, enc_cfg.hidden_size, w.encoder_projector_w,
                       w.encoder_projector_b, cfg.decoder_hidden_size);
  } catch (...) {
    backend.DestroyQueue(q);
    throw;
  }
  backend.DestroyQueue(q);

  return ParakeetTransducerGreedyDecode(projected, rows, valid_rows, w, cfg, backend);
}

}  // namespace vllm::multimodal
