// Parakeet (`ParakeetForCTC` / `ParakeetForRNNT` / `ParakeetForTDT`) registry
// TU — the ADDITIVE self-registration seam for the ARCH-ONE-SURFACE ROW 1
// audio-transcription fold. Follows the kimi_k3_registry.cpp refuse-stub
// precedent exactly: a NEW translation unit with REGISTER_VLLM_MODEL lines and
// ZERO edit to any shared array.
//
// WHY these registrations exist: so an HF Parakeet checkpoint directory
// RESOLVES from config.json `architectures` and the entrypoints can dispatch
// BY TASK — the ModelInfo carries the SupportsTranscription mirror
// (vllm/model_executor/models/interfaces.py:1110-1118, with
// `supports_transcription_only` true: these archs have NO text-generation
// path), which LoadedEngine::FromModelDir turns into a clean refusal pointing
// at vllm_transcribe / /v1/audio/transcriptions. The transcription work itself
// runs through vllm::multimodal::ParakeetTranscriber
// (include/vllm/multimodal/parakeet_transcription.h), NOT through
// ModelRegistry::Forward — an encoder+greedy-decode pipeline has no
// logits-per-step contract to satisfy.
//
// BEYOND-PIN BREADTH, recorded: the pinned vLLM registers NO standalone
// Parakeet architecture — upstream's Parakeet is the audio-encoder COMPONENT
// of NemotronH_Nano_VL_V2 (vllm/model_executor/models/registry.py:511-513),
// wrapping transformers' encoder (parakeet.py:37,62). The standalone
// CTC/RNNT/TDT classes we resolve here are transformers-`main` classes
// (modeling_parakeet.py: ParakeetForCTC:675, ParakeetForRNNT:922,
// ParakeetForTDT:1052) — grounded per file:line in parakeet_encoder.h /
// parakeet_transducer.h and marked beyond-pin in .agents/model-matrix.md.
//
// A registered Parakeet must NOT crash the text-generation paths: every
// factory hook below refuses with the same actionable message instead of
// VT_CHECK-aborting, and the KV spec is a never-exercised placeholder (the
// kimi_k3 stub precedent) so resolution-time plumbing that sizes caches
// cannot trip on a null factory field.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

[[noreturn]] void RefuseTextGeneration() {
  throw std::runtime_error(
      "Parakeet architectures support transcription only "
      "(SupportsTranscription, no text-generation path): drive the checkpoint "
      "through vllm_transcribe on the C ABI or the server's "
      "/v1/audio/transcriptions, not the text-generation entry points");
}

// registry.py _ModelInfo mirror: NOT a text-generation model; ASR-capable and
// ASR-ONLY (interfaces.py:1116-1118).
inline constexpr ModelInfo kParakeetInfo{
    .is_text_generation_model = false,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = true,
    .supports_transcription_only = true,
    .score_type = "bi-encoder",
};

void ParseParakeetConfig(const HfConfig& config) {
  // Nothing to validate at resolution time: the transcription seam's own
  // loader (LoadParakeetForCTC / LoadParakeetTransducer) parses config.json
  // and fails loudly per missing/misshaped field.
  (void)config;
}

std::unique_ptr<LoadedModel> LoadParakeetRefused(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  (void)registration;
  (void)config;
  (void)source;
  RefuseTextGeneration();
}

void PrepareParakeetRefused(LoadedModel& model, const HfConfig& config,
                            vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
  RefuseTextGeneration();
}

ForwardLogits ForwardParakeetRefused(LoadedModel& model,
                                     const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // VT_CHECK(false, ...) — the registry's REFUSE-stub contract the
  // runner-routing gate recognizes (classify_body REFUSE, the kimi_k3/
  // deepseek_v4 precedent). It THROWS a clean std::runtime_error
  // (vt/dtype.h:11), never aborts, so the refusal stays actionable.
  VT_CHECK(false,
           "Parakeet architectures support transcription only "
           "(SupportsTranscription, no text-generation path): drive the "
           "checkpoint through vllm_transcribe on the C ABI or the server's "
           "/v1/audio/transcriptions, not the text-generation entry points");
  return {};
}

v1::KVCacheConfig MakeParakeetKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // PLACEHOLDER, never exercised: the encoder has no KV cache at all (no
  // autoregressive attention), and the refuse-by-task gate in
  // LoadedEngine::FromModelDir fires before any cache is built. One minimal
  // full-attention group so a caller that sizes specs eagerly cannot crash.
  (void)config;
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"encoder"},
      std::make_shared<v1::FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                              /*head_size=*/64,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

const ModelFactory kParakeetFactory{
    .parse_config = &ParseParakeetConfig,
    .load_weights = &LoadParakeetRefused,
    .prepare = &PrepareParakeetRefused,
    .forward = &ForwardParakeetRefused,
    .make_kv_cache = &MakeParakeetKVCache,
    .is_dense_model = false,
};

}  // namespace

REGISTER_VLLM_MODEL(parakeet_ctc, "ParakeetForCTC", kParakeetFactory,
                    kParakeetInfo)
REGISTER_VLLM_MODEL(parakeet_rnnt, "ParakeetForRNNT", kParakeetFactory,
                    kParakeetInfo)
REGISTER_VLLM_MODEL(parakeet_tdt, "ParakeetForTDT", kParakeetFactory,
                    kParakeetInfo)

}  // namespace vllm
