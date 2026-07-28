// vllm.cpp original (no upstream 1:1 file). A single host-registered custom
// logits-processor callback, the C++ carrier of the C-ABI `vllm_logits_processor`
// (include/vllm.h, ABI v8) and the internal analogue of vLLM's per-request
// `SamplingParams.logits_processors` (vllm/sampling_params.py) / SGLang's
// `CustomLogitProcessor` (python/sglang/srt/sampling/custom_logit_processor.py:24).
//
// DESIGN (recorded deviation from vLLM's plugin object graph): vLLM builds a
// `LogitsProcessors` manager of `LogitsProcessor` subclasses (interface.py::
// LogitsProcessor.apply) that each mutate the whole batch logits tensor; SGLang
// takes a per-request Python callable. We are a vLLM port exposed through a C ABI,
// so we mirror vLLM's APPLICATION POINT + ORDERING (the non-argmax-invariant
// logits-processor stage, after allowed_token_ids / bad_words / min_tokens /
// logit_bias, before penalties — sampler.py:399) but carry a single per-request C
// function pointer instead of the Python plugin graph. The callback receives the
// request's generated token ids so far + a MUTABLE view of that request's logits
// row and edits the row in place. Absent (fn == nullptr) => the sampler path is
// byte-identical to a build with no processor.
#ifndef VLLM_LOGITS_PROCESSOR_CALLBACK_H_
#define VLLM_LOGITS_PROCESSOR_CALLBACK_H_

#include <cstdint>

namespace vllm {

// Host callback invoked once per decode step for a request, BEFORE sampling.
//   - token_ids / n_token_ids: the request's generated output token ids so far
//     (n_token_ids == 0 on the first decode step). Borrowed; valid only for the
//     duration of the call.
//   - logits / vocab_size: a mutable f32 view of THIS request's logits row
//     [vocab_size]. Edit in place (bias / mask / force tokens); the edited row is
//     what the sampler then samples from.
//   - user_data: the opaque pointer registered alongside the callback.
// The callback is C code and MUST NOT throw across the ABI boundary.
using LogitsProcessorFn = void (*)(const int32_t* token_ids, int32_t n_token_ids,
                                   float* logits, int32_t vocab_size,
                                   void* user_data);

// A registered custom logits processor (function pointer + opaque user data).
// A default-constructed value (fn == nullptr) means "no processor".
struct LogitsProcessorCallback {
  LogitsProcessorFn fn = nullptr;
  void* user_data = nullptr;
};

}  // namespace vllm

#endif  // VLLM_LOGITS_PROCESSOR_CALLBACK_H_
