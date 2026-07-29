// Ported from: vllm/plugins/__init__.py @ 555967922 (vLLM 0.26.0.dev0)
//   DEFAULT_PLUGINS_GROUP "vllm.general_plugins" (:18),
//   load_plugins_by_group (:36-74) — discovery + the VLLM_PLUGINS allowlist,
//   load_general_plugins (:77-90) — idempotent load-once + invoke each register().
//
// C++ adaptation (recorded in porting-inventory.md §9, ENG-PLUGIN-SYSTEM):
// vLLM discovers plugins through Python `importlib.metadata` entry points under
// the group `vllm.general_plugins`; each entry point resolves to a `register()`
// function that `load_general_plugins()` calls. Pure C++20 has no Python entry
// points, so the faithful analog is a process-global REGISTRATION registry that
// an OUT-OF-CORE translation unit or shared object populates at load time
// (static-init, or `dlopen` of a plugin .so whose C-ABI entry symbol runs the
// same registration), exactly like the existing `REGISTER_VLLM_MODEL` /
// `RegisterPlatform` static-init seams. `LoadGeneralPlugins()` then invokes each
// registered callback, honoring the same `VLLM_PLUGINS` allowlist, the same
// load-once idempotence, and the same per-plugin failure isolation as upstream.
//
// A general plugin's callback registers its out-of-tree contribution through the
// EXISTING public seams WITHOUT editing engine core: `vllm::RegisterModel(...)`
// (or the `REGISTER_VLLM_MODEL` macro) for a model factory, `RegisterPlatform`
// for a platform, the quant registry for a quant method. This header adds only
// the discovery + orchestration layer those seams were missing.
#pragma once

#include <string_view>

namespace vllm::plugins {

// The default plugin group name, mirroring DEFAULT_PLUGINS_GROUP
// (plugins/__init__.py:18). Kept as the documented contract string even though
// C++ registration is not entry-point-group-scoped: a real dlopen loader keys
// its manifest on this same name.
inline constexpr std::string_view kGeneralPluginsGroup = "vllm.general_plugins";

// The C-ABI entry-symbol name a real shared-object plugin exports so a future
// `dlopen`-based loader (W2 residual) can resolve + run its registration:
//     extern "C" void vllm_plugin_register(void);
// The body calls `vllm::RegisterModel` / `vllm::plugins::RegisterGeneralPlugin`
// / `RegisterPlatform` — the same in-process seams used here — so the static and
// dynamic paths converge on one registration mechanism.
inline constexpr std::string_view kPluginCAbiRegisterSymbol =
    "vllm_plugin_register";

// A general plugin's registration callback. Mirrors the `register()` function an
// upstream `vllm.general_plugins` entry point resolves to (e.g.
// tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/__init__.py:register).
using GeneralPluginFn = void (*)();

// Register an out-of-core general plugin's callback under `name`. Called at plugin
// load time (static-init via REGISTER_VLLM_GENERAL_PLUGIN, or from a dlopen'd
// plugin's C-ABI entry). The callback itself is NOT run here — it runs when
// `LoadGeneralPlugins()` is invoked, mirroring upstream discovery (registration)
// being separate from `load_general_plugins()` (invocation). `name` must have
// static lifetime (a string literal); it is stored as a view.
void RegisterGeneralPlugin(std::string_view name, GeneralPluginFn fn);

// Mirror of load_general_plugins (plugins/__init__.py:77-90). Invokes each
// registered general-plugin callback exactly once per process:
//   * IDEMPOTENT — a `plugins_loaded` latch makes a second call a no-op, so it is
//     safe to call from every engine construction path (upstream calls it from
//     worker_base, engine core, arg_utils, and the model registry).
//   * ALLOWLIST — honors the `VLLM_PLUGINS` env var exactly like
//     load_plugins_by_group: unset ⇒ load all; set ⇒ load only the comma-listed
//     names (note `VLLM_PLUGINS=""` parses to the single empty name and matches
//     nothing, so it disables every plugin).
//   * FAILURE ISOLATION — a callback that throws is logged to stderr and SKIPPED;
//     it never aborts the load or the engine (mirror of `logger.exception`).
void LoadGeneralPlugins();

// Whether LoadGeneralPlugins has already run in this process (the `plugins_loaded`
// module global, plugins/__init__.py:33). Exposed for the unit gate.
bool PluginsLoaded();

// Number of general-plugin callbacks that have actually been INVOKED across all
// LoadGeneralPlugins calls in this process (allowlist-gated, failure-isolated).
// Exposed so the idempotence + allowlist gates can assert exact invocation
// counts. Not part of upstream (upstream has no such counter); test-support only.
int InvokedCountForTesting();

// Test-only: clear the load-once latch so a different `VLLM_PLUGINS` allowlist
// can be exercised within one process. Does NOT unregister anything a prior load
// already installed (the model/platform/quant registries only ever grow), so
// scenarios must be ordered monotonically (allowlist-blocked before allowlisted).
void ResetLoadedForTesting();

// Static-init helper whose constructor self-registers a general plugin. Used only
// through REGISTER_VLLM_GENERAL_PLUGIN, mirroring `ModelRegistrar`.
struct GeneralPluginRegistrar {
  GeneralPluginRegistrar(std::string_view name, GeneralPluginFn fn) {
    RegisterGeneralPlugin(name, fn);
  }
};

}  // namespace vllm::plugins

// Registers one out-of-core general plugin from its OWN translation unit / shared
// object, with ZERO edit to engine core — the extensibility seam. `plugin_name`
// is the allowlist key (matched against VLLM_PLUGINS); `fn` is the `register()`
// callback that installs the plugin's model/platform/quant contribution when
// LoadGeneralPlugins runs. `unique_tag` is any TU-unique token. Place at
// namespace scope.
#define REGISTER_VLLM_GENERAL_PLUGIN(unique_tag, plugin_name, fn)          \
  namespace {                                                              \
  const ::vllm::plugins::GeneralPluginRegistrar                           \
      vllm_general_plugin_registrar_##unique_tag((plugin_name), (fn));    \
  } /* namespace */
