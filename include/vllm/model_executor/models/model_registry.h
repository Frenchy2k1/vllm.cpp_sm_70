// Ported from: vllm/model_executor/models/registry.py
//               @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c
// (_ModelInfo:746-796, _ModelRegistry:998-1082,
//  resolve_model_cls:1244-1296, global registry:1396-1404).
//
// Python-only lazy imports, subprocess inspection/cache, dynamic Transformers,
// terratorch, and out-of-tree runtime loading intentionally have no C++ runtime
// analogue. The ordered lookup, capability metadata, implemented-model factory
// hooks, and unsupported-architecture messages mirror the pinned registry.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class GgufFile;
class SafetensorsFile;
struct ForwardLogits;
struct GdnStateCache;
struct PagedKvCache;
struct Qwen3_5DenseWeights;
struct Qwen3_5MoeWeights;
// SPEC-MTP I5d-pre: the checkpoint-owned MTP draft weights, the drafter's
// hidden-state carrier, and the draft model wrapper (all defined in
// qwen3_5_mtp.h). Forward-declared here so the type-erased LoadedModel can
// expose a typed path to the draft (BuildMtpDraft) and a hidden-state tap
// out-field WITHOUT pulling the MTP header — or any concrete weight type — into
// this shared registry seam. Inert unless speculative decoding is configured.
struct Qwen3_5MTPWeights;
struct Qwen3_5MTPHiddenStates;
struct Qwen3_5AuxTaps;
class Qwen3_5MTPModel;

namespace v1 {
struct CommonAttentionMetadata;
struct GDNAttentionMetadata;
}  // namespace v1

// Consumed subset of upstream _ModelInfo. Extend this POD as task rows land;
// these fields are enough to choose generation vs pooling/scoring and hybrid vs
// full-attention construction without importing a Python model class.
struct ModelInfo {
  bool is_text_generation_model = false;
  bool is_pooling_model = false;
  bool is_hybrid = false;
  bool has_inner_state = false;
  bool supports_multimodal = false;
  std::string_view score_type = "bi-encoder";
};

// Type-erased checkpoint source passed to a registration's weight loader.
struct ModelSource {
  enum class Kind { kSafetensors, kGguf };

  static ModelSource FromSafetensors(
      const std::vector<SafetensorsFile>& shards,
      vt::Queue* load_queue = nullptr);
  // Shares ownership of the mmap'd shards so a loader may keep them alive past
  // the load (e.g. the Qwen3.6-35B MoE deferred-expert streaming that
  // materializes routed experts per layer during PrepareMarlinResident). The
  // borrowing `safetensors` pointer is set to the owned vector.
  static ModelSource FromSafetensorsOwned(
      std::shared_ptr<const std::vector<SafetensorsFile>> shards,
      vt::Queue* load_queue = nullptr);
  static ModelSource FromGguf(const GgufFile& gguf);

  Kind kind = Kind::kSafetensors;
  const std::vector<SafetensorsFile>* safetensors = nullptr;
  // Non-null only for FromSafetensorsOwned: shared ownership a loader can retain.
  std::shared_ptr<const std::vector<SafetensorsFile>> safetensors_owned;
  const GgufFile* gguf = nullptr;
  // Non-owning queue selected by the entrypoint. Dense plain-BF16 loaders may
  // stage completed layers here and the runner then reuses the same queue.
  vt::Queue* load_queue = nullptr;
};

struct ModelFactory;
struct ModelRegistration;

// Opaque, lifetime-owning (or test-only borrowing) concrete model. The runner
// sees only this base and dispatches through its registration's function table.
class LoadedModel {
 public:
  virtual ~LoadedModel();

  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
  LoadedModel(LoadedModel&&) = delete;
  LoadedModel& operator=(LoadedModel&&) = delete;

  const ModelRegistration& registration() const { return registration_; }
  // Runtime capability rather than architecture metadata: GGUF/synthetic
  // instances of a W4A4-capable family may contain only BF16 weights.
  virtual bool uses_nvfp4_w4a4() const { return false; }

  // ── SPEC-MTP I5d-pre: typed access to the MTP draft, without breaking the
  //    type-erasure of this base. Only the concrete Qwen3.5 dense/MoE
  //    LoadedModel (which owns the target Qwen3_5DenseWeights/Qwen3_5MoeWeights)
  //    can retain the loaded `mtp.*` weights and construct the draft that shares
  //    the target's embed_tokens/lm_head (qwen3_5_mtp.h:61-66). Every default
  //    below is the non-MTP behavior: no support, and BuildMtpDraft yields null.
  //    All of this is unreachable on the production default path (no
  //    SpeculativeConfig ⇒ FromModelDir never attaches MTP weights), so the
  //    engine is byte-identical when speculative decoding is off. ──────────────

  // Whether this concrete model can hold MTP draft weights and build the draft.
  virtual bool supports_mtp_draft() const { return false; }

  // Retain the checkpoint's loaded `mtp.*` draft weights (the 15/19 BF16 tensors
  // LoadQwen3_5MTP produced) inside the concrete model so the draft can be built
  // on demand with a stable lifetime. Default: unsupported (throws), because a
  // non-MTP architecture has nowhere valid to put them.
  virtual void AttachMtpDraftWeights(Qwen3_5MTPWeights weights);

  // Construct the Qwen3_5MTPModel draft from the retained MTP weights + this
  // model's own target weights. Returns null when no MTP weights were attached
  // or the architecture is not an MTP target. The returned draft borrows this
  // model's weights/config, so it must not outlive this LoadedModel.
  virtual std::unique_ptr<Qwen3_5MTPModel> BuildMtpDraft(
      const HfConfig& config) const;

 protected:
  explicit LoadedModel(const ModelRegistration& registration)
      : registration_(registration) {}

 private:
  const ModelRegistration& registration_;
};

// One MRV2 forward invocation. References stay valid for the duration of the
// registered forward hook; model-specific decode-graph state lives in the
// concrete LoadedModel rather than leaking concrete weight types into runner.
struct ModelForwardInput {
  const std::vector<int32_t>& token_ids;
  const std::vector<int32_t>& positions;
  const v1::CommonAttentionMetadata& attn_meta;
  const v1::GDNAttentionMetadata& gdn_meta;
  std::vector<PagedKvCache>& attn_kv;
  std::vector<GdnStateCache>& gdn_state;
  const HfConfig& config;
  vt::Queue& queue;
  const std::vector<int32_t>& logits_indices;
  int num_reqs = 0;
  int64_t gdn_state_slots = 0;
  bool pure_decode = false;
  bool gather_logits = true;
  // SPEC-MTP I5d-pre hidden-state tap. When non-null (only the spec verify
  // forward sets it, I5d), the Qwen3.5 dense/MoE forward routes to
  // Qwen3_5{,Dense}Model::ForwardDeviceTap, which returns byte-identical logits
  // AND moves the full [num_actual_tokens, H] post-final-norm hidden into
  // `*hidden_tap` for the MTP drafter. When null (every spec-off run) the
  // forward takes the current ModelRegistry::Forward path and is byte-identical.
  // Models other than Qwen3.5 ignore this field.
  Qwen3_5MTPHiddenStates* hidden_tap = nullptr;
  // SPEC-DFLASH D1 (DF-AUX-TAPS) multi-layer aux hidden taps. When non-null (only
  // the DFlash spec verify forward sets it, D4), the Qwen3.5 dense/MoE forward
  // routes to Qwen3_5{,Dense}Model::ForwardDeviceMultiTap, which returns byte-
  // identical logits AND captures (hidden + residual) at each configured
  // target_layer_id boundary into `aux_tap->tensor` = [T, H×taps] for the DFlash
  // drafter. When null (every non-DFlash run) the forward takes the current path
  // and is byte-identical. Mutually exclusive with hidden_tap. Models other than
  // Qwen3.5 ignore this field.
  Qwen3_5AuxTaps* aux_tap = nullptr;
};

using ModelConfigHook = void (*)(const HfConfig& config);
using ModelWeightLoader = std::unique_ptr<LoadedModel> (*)(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source);
using ModelPrepareFn = void (*)(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue);
using ModelForwardFn = ForwardLogits (*)(LoadedModel& model,
                                         const ModelForwardInput& input);
using KVCacheSpecBuilder = v1::KVCacheConfig (*)(const HfConfig& config,
                                                 int block_size,
                                                 int num_blocks);

// Per-family plug-in seam. load_weights contains that family's on-disk name map
// and construction path; forward remains type-erased over LoadedModel.
struct ModelFactory {
  ModelConfigHook parse_config = nullptr;
  ModelWeightLoader load_weights = nullptr;
  ModelPrepareFn prepare = nullptr;
  ModelForwardFn forward = nullptr;
  KVCacheSpecBuilder make_kv_cache = nullptr;
  // Preserves the already-gated per-arch scheduler default. This is execution
  // policy, not an upstream _ModelInfo capability.
  bool is_dense_model = false;
};

struct ModelRegistration {
  std::string_view architecture;
  const ModelFactory* factory = nullptr;
  ModelInfo info;
};

struct UnsupportedModelInfo {
  std::string_view architecture;
  std::string_view detail;  // previous version or plugin URL
};

// Self-registration seam (mirrors REGISTER_VLLM_MODEL in
// vllm/model_executor/models/registry.py:682-693 assembling `_VLLM_MODELS`).
// Each architecture registers itself from its OWN translation unit via a static
// `ModelRegistrar`, exactly like the RegisterOp/RegisterBackend/RegisterPlatform
// static-init idiom (src/vt/ops.cpp, src/vt/backend.cpp,
// src/vllm/platforms/platform.cpp). Adding a model = a new TU with one
// REGISTER_VLLM_MODEL line — ZERO edit to a shared array.
//
// The passed ModelRegistration is copied into the process-global registry (its
// string_view/factory members point at TU-static data with static lifetime, so
// the copy stays valid). Registration order across TUs is unspecified under C++
// static init, so the registry imposes a stable canonical sort by architecture
// name on first query (see model_registry.cpp) — resolution semantics (first
// config-architecture match) are order-independent and unchanged.
void RegisterModel(const ModelRegistration& registration);

// Internal adapter helper: returns the registry entry for an implemented
// architecture (used by the synthetic in-memory Make/Borrow adapters below).
const ModelRegistration& RegistrationFor(std::string_view architecture);

// Static-init helper whose constructor performs the self-registration; used
// only through the REGISTER_VLLM_MODEL macro.
struct ModelRegistrar {
  explicit ModelRegistrar(const ModelRegistration& registration) {
    RegisterModel(registration);
  }
};

// Registers one architecture's factory from its own TU. Place at namespace
// scope inside `namespace vllm { ... }`; `unique_tag` is any TU-unique token.
#define REGISTER_VLLM_MODEL(unique_tag, architecture_name, factory_ref,   \
                            info_val)                                      \
  namespace {                                                             \
  const ::vllm::ModelRegistrar vllm_model_registrar_##unique_tag(         \
      ::vllm::ModelRegistration{(architecture_name), &(factory_ref),      \
                                (info_val)});                             \
  } /* namespace */

class ModelRegistry {
 public:
  // Mirrors get_supported_archs() and resolve_model_cls(): registrations are
  // declaration-ordered and Resolve returns the first architecture-list match.
  static std::span<const ModelRegistration> Registrations();
  static std::vector<std::string_view> SupportedArchs();
  static const ModelRegistration& Resolve(
      std::span<const std::string> architectures);
  static const ModelRegistration& Resolve(const HfConfig& config);

  // Mirrors _raise_for_unsupported exactly. The explicit-supported overload is
  // used by the subset-oracle parity test and covers the upstream
  // registered-but-inspection-failed branch.
  [[noreturn]] static void RaiseForUnsupported(
      std::span<const std::string> architectures);
  [[noreturn]] static void RaiseForUnsupported(
      std::span<const std::string> architectures,
      std::span<const std::string_view> supported_architectures);

  static std::span<const UnsupportedModelInfo> PreviouslySupportedModels();
  static std::span<const UnsupportedModelInfo> OutOfTreeSupportedModels();

  // Type-erased construction and live runner hooks.
  static std::unique_ptr<LoadedModel> Load(const HfConfig& config,
                                           const ModelSource& source);
  static void Prepare(LoadedModel& model, const HfConfig& config,
                      vt::Queue& queue);
  static ForwardLogits Forward(LoadedModel& model,
                               const ModelForwardInput& input);
  static v1::KVCacheConfig MakeKVCache(const LoadedModel& model,
                                       const HfConfig& config, int block_size,
                                       int num_blocks);
  static bool IsDenseModel(const LoadedModel& model);
};

// Compatibility adapters for synthetic in-memory Qwen tests and callers that
// already own weights. Production disk loading uses ModelRegistry::Load.
std::unique_ptr<LoadedModel> MakeQwen3_5MoeLoadedModel(
    Qwen3_5MoeWeights weights);
std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights);
std::unique_ptr<LoadedModel> BorrowQwen3_5MoeLoadedModel(
    const Qwen3_5MoeWeights& weights);
std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights);

}  // namespace vllm
