// Ported from: vllm/plugins/__init__.py @ 555967922 (vLLM 0.26.0.dev0)
//   load_plugins_by_group (:36-74), load_general_plugins (:77-90),
//   the plugins_loaded module latch (:33), the VLLM_PLUGINS allowlist
//   (envs.py:1104-1108: unset ⇒ None ⇒ load all; else split(",")).
//
// This is the GENERIC, contribution-agnostic plugin orchestration layer: the
// registration list, the load-once latch, the VLLM_PLUGINS allowlist gate, and
// the per-plugin failure isolation. The actual out-of-core CONTRIBUTIONS (a model
// factory, a platform, a quant method) are installed by each plugin's callback
// through the existing public registries (vllm::RegisterModel, RegisterPlatform,
// the quant registry) — this TU never names them, exactly as upstream
// load_general_plugins "only needs to execute the loaded functions".
#include "vllm/plugins/plugins.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vllm::plugins {
namespace {

struct RegisteredPlugin {
  std::string_view name;
  GeneralPluginFn fn;
};

// Process-global registration list, populated at plugin load time (static-init
// via REGISTER_VLLM_GENERAL_PLUGIN, or a dlopen'd plugin's C-ABI entry). Meyers
// singleton: constructed on the first RegisterGeneralPlugin call, safely before
// any registrar runs.
std::vector<RegisteredPlugin>& PluginStorage() {
  static std::vector<RegisteredPlugin> storage;
  return storage;
}

// The plugins_loaded module global (plugins/__init__.py:33) — one process loads
// plugins once.
bool& LoadedLatch() {
  static bool loaded = false;
  return loaded;
}

int& InvokedCounter() {
  static int invoked = 0;
  return invoked;
}

// Mirror of the VLLM_PLUGINS env lambda (envs.py:1104-1108): the var UNSET ⇒
// nullopt ⇒ every plugin loads; SET ⇒ the comma-split allowlist (note the empty
// string parses to the single-element list `{""}`, which matches no real plugin
// name — the "disable all plugins" posture the offline OOT test relies on).
std::optional<std::vector<std::string>> ParseVllmPluginsAllowlist() {
  const char* raw = std::getenv("VLLM_PLUGINS");
  if (raw == nullptr) return std::nullopt;

  std::vector<std::string> names;
  std::string_view view(raw);
  size_t start = 0;
  while (true) {
    const size_t comma = view.find(',', start);
    if (comma == std::string_view::npos) {
      names.emplace_back(view.substr(start));
      break;
    }
    names.emplace_back(view.substr(start, comma - start));
    start = comma + 1;
  }
  return names;
}

}  // namespace

void RegisterGeneralPlugin(std::string_view name, GeneralPluginFn fn) {
  if (fn == nullptr) return;
  PluginStorage().push_back(RegisteredPlugin{name, fn});
}

void LoadGeneralPlugins() {
  // Idempotence latch (plugins/__init__.py:82-85): set BEFORE invoking so a
  // re-entrant call from inside a plugin still no-ops.
  bool& loaded = LoadedLatch();
  if (loaded) return;
  loaded = true;

  const std::optional<std::vector<std::string>> allowlist =
      ParseVllmPluginsAllowlist();

  for (const RegisteredPlugin& plugin : PluginStorage()) {
    if (allowlist.has_value()) {
      const bool allowed =
          std::find(allowlist->begin(), allowlist->end(),
                    std::string(plugin.name)) != allowlist->end();
      if (!allowed) continue;
    }

    // Per-plugin failure isolation (plugins/__init__.py:68-72,
    // logger.exception): a throwing plugin is logged + skipped, never fatal.
    try {
      plugin.fn();
      ++InvokedCounter();
    } catch (const std::exception& e) {
      std::cerr << "vllm.cpp: Failed to load plugin " << plugin.name << ": "
                << e.what() << std::endl;
    } catch (...) {
      std::cerr << "vllm.cpp: Failed to load plugin " << plugin.name
                << ": unknown error" << std::endl;
    }
  }
}

bool PluginsLoaded() { return LoadedLatch(); }

int InvokedCountForTesting() { return InvokedCounter(); }

void ResetLoadedForTesting() { LoadedLatch() = false; }

}  // namespace vllm::plugins
