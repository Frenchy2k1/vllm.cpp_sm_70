# Plugin system — out-of-core registration (`ENG-PLUGIN-SYSTEM`, `CLAIM-PLUGIN-SYSTEM`)

W0 spike for the plugin-system RECORDS-GAP named in
[vllm-feature-gap-analysis.md](vllm-feature-gap-analysis.md) (row "Plugin system
(general / io_processor / platform / endpoint plugins)", anchor
`vllm/plugins/__init__.py:18,77`). The analysis flagged this as a `RECORDS-GAP`
(no stable row anywhere) that "directly serves the extensibility-first priority
(additive HW/models/endpoints)" and recommended creating `ENG-PLUGIN-SYSTEM`.
This spec CREATES that row, inventories vLLM's plugin surface, fixes the faithful
C++ mechanism, names the tests, and lays out the W-breakdown. Pinned oracle
`/home/mudler/_git/vllm` @ `555967922` (vLLM 0.26.0.dev0); every anchor below is
real `file:line` at that pin.

## Scope

- **Row.** New engine row `ENG-PLUGIN-SYSTEM` (this change), `INVENTORIED`→
  `ACTIVE`. No model/quant/platform row is created — the plugin system is the
  DISCOVERY + ORCHESTRATION layer over the existing registration seams, not a new
  model or backend.
- **In (W1, this change).** The `vllm::plugins::LoadGeneralPlugins()` hook + the
  out-of-core general-plugin registration seam (`RegisterGeneralPlugin` +
  the `REGISTER_VLLM_GENERAL_PLUGIN` macro), mirroring vLLM's
  `load_general_plugins` load-once idempotence, the `VLLM_PLUGINS` allowlist, and
  per-plugin failure isolation. Proven by an OUT-OF-CORE test plugin TU that
  registers a toy model factory through the public `vllm::RegisterModel` seam,
  unit-gated RED-first (the toy arch resolves ONLY after LoadGeneralPlugins runs
  it, and NOT when the allowlist blocks it).
- **Out (W2+, named residuals).** Real shared-object loading (`dlopen` of a
  plugin `.so` + resolving the documented C-ABI `vllm_plugin_register` entry
  symbol); the CLI `--load-plugins` / library wiring that calls
  `LoadGeneralPlugins()` from the engine construction paths; the platform-plugin
  kind (`vllm.platform_plugins`, loaded during `current_platform` resolution) and
  the quant-method plugin kind; the io_processor / stat_logger / endpoint plugin
  groups. Each is a named brick in the Work breakdown.

## Upstream chain

### `vllm/plugins/__init__.py` — the plugin loader

- `:18` `DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"` — the default entry-point
  group loaded in ALL processes (process0, engine-core, workers).
- `:19-30` the other groups: `vllm.io_processor_plugins` (process0 only),
  `vllm.platform_plugins` (loaded when `current_platform` first resolves),
  `vllm.stat_logger_plugins` (async serve, process0), `vllm.endpoint_plugins`
  (API front-end only, opt-in for network-surface safety).
- `:33` `plugins_loaded = False` — the module-global load-once latch.
- `:36-74` `load_plugins_by_group(group)` — discovers entry points in a group,
  applies the `VLLM_PLUGINS` allowlist (`:40,56,64`: `None` ⇒ load all; a set ⇒
  only named names), and, critically, `:68-72` wraps each `plugin.load()` in
  `try/except` + `logger.exception("Failed to load plugin %s")` — a broken plugin
  is logged and skipped, never fatal.
- `:77-90` `load_general_plugins()` — `:82-85` the idempotence latch (returns
  early if already loaded, sets the latch BEFORE loading), then `:87-90` loads
  the general group and EXECUTES each resolved function ("general plugins, we
  only need to execute the loaded functions"). The docstring `:78-80` warns
  plugins may be loaded multiple times across processes and MUST be safe to load
  repeatedly.
- `envs.py:1104-1108` `VLLM_PLUGINS` — unset ⇒ `None` (load all); set ⇒
  `os.environ["VLLM_PLUGINS"].split(",")` (note `""` ⇒ `[""]`, matching nothing).

### The registration a general plugin calls — out-of-tree model registration

- `vllm/model_executor/models/registry.py:1039-1083`
  `_ModelRegistry.register_model(model_arch, model_cls)` — the public seam a
  plugin's `register()` calls to add an out-of-tree architecture (direct class or
  lazy `"<module>:<class>"` string). `:1036` `get_supported_archs()`.
- `vllm/__init__.py` re-exports `ModelRegistry`; the dummy plugin
  (`tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/__init__.py`)
  `register()` calls `ModelRegistry.register_model("MyOPTForCausalLM", ...)`
  guarded by `if "MyOPTForCausalLM" not in ModelRegistry.get_supported_archs()`.
- `setup.py` of that plugin declares
  `entry_points={"vllm.general_plugins": ["register_dummy_model = vllm_add_dummy_model:register"]}`
  — the discovery record `load_plugins_by_group` reads.

### Where vLLM invokes the load hook (the engine seams)

- `vllm/v1/worker/worker_base.py:245-247`, `vllm/v1/engine/core.py:115-117`,
  `vllm/engine/arg_utils.py:782-784`, `vllm/model_executor/models/registry.py:1475-1477`
  all call `load_general_plugins()` — idempotently — from construction paths, so
  plugins are installed before the registry is queried.

### The platform-plugin kind (W2 residual reference)

- `vllm/platforms/__init__.py:202-244` `builtin_platform_plugins` + `:211`
  `resolve_current_platform_cls_qualname()` calls `load_plugins_by_group(
  PLATFORM_PLUGINS_GROUP)` and chains OOT platform plugins with the built-ins —
  the analog for our `RegisterPlatform` seam (`src/vllm/platforms/platform.cpp`).

## Our baseline

vllm.cpp ALREADY has the out-of-core registration seams a plugin would call; it
lacked only the discovery/orchestration layer:

- Model factories self-register from their own TU via `REGISTER_VLLM_MODEL`
  (`include/vllm/model_executor/models/model_registry.h:288`) into a
  process-global registry (`src/vllm/model_executor/models/model_registry.cpp:143`
  `RegisterModel`). `ModelRegistry::Resolve` (`:217`) throws
  `RaiseForUnsupported` (`:245`) for an unknown arch.
- Platforms self-register via `RegisterPlatform`
  (`src/vllm/platforms/platform.cpp:52`); quant schemes and vt ops/backends use
  the same static-init `Register*` idiom (`src/vt/ops.cpp`, `src/vt/backend.cpp`).

The gap: nothing let an OUT-OF-CORE unit (a plugin) run its registration at a
controlled load point with an allowlist + failure isolation, the way vLLM's
entry-point discovery + `load_general_plugins` does. `ENG-PLUGIN-SYSTEM` adds
exactly that layer, reusing the existing seams unchanged.

## Port map

C++ mechanism (recorded in [porting-inventory.md](../porting-inventory.md) §9 as
a legitimate mechanical divergence — pure C++20 has no Python entry points):

| vLLM (Python) | vllm.cpp (C++) |
|---|---|
| `vllm.general_plugins` entry-point group discovery | process-global registration list populated at plugin LOAD time (static-init via `REGISTER_VLLM_GENERAL_PLUGIN`, or a `dlopen`'d plugin's C-ABI entry) |
| a `register()` function per entry point | `GeneralPluginFn` callback + `RegisterGeneralPlugin(name, fn)` |
| `load_general_plugins()` load-once + execute | `vllm::plugins::LoadGeneralPlugins()` (idempotent latch, allowlist, failure isolation) |
| `VLLM_PLUGINS` allowlist (`None`⇒all; split(",")) | `getenv("VLLM_PLUGINS")` parsed identically (unset⇒all; `""`⇒`{""}`⇒none) |
| `ModelRegistry.register_model(arch, cls)` | existing `vllm::RegisterModel(...)` (unchanged) |
| entry-point manifest string | documented C-ABI entry symbol `vllm_plugin_register` for a future `dlopen` loader |

Files (all additive; ZERO edit to engine core):

- NEW `include/vllm/plugins/plugins.h` — `LoadGeneralPlugins`,
  `RegisterGeneralPlugin`, `PluginsLoaded`, `kGeneralPluginsGroup`,
  `kPluginCAbiRegisterSymbol`, the `REGISTER_VLLM_GENERAL_PLUGIN` macro, and the
  test-support latch reset + invocation counter.
- NEW `src/vllm/plugins/plugins.cpp` — the registration list, the `plugins_loaded`
  latch, the `VLLM_PLUGINS` allowlist parse, and per-plugin `try/catch` isolation.
- NEW `tests/vllm/plugins/toy_model_plugin.cpp` — the OUT-OF-CORE plugin fixture:
  a toy model factory registered via `vllm::RegisterModel`, published via
  `REGISTER_VLLM_GENERAL_PLUGIN`, plus a deliberately-throwing plugin for the
  isolation gate. Compiled ONLY into the test executable (never the library), so
  it never pollutes the 28-arch registry the other suites count.
- NEW `tests/vllm/plugins/test_plugin_system.cpp` — the RED-first unit gate.
- `CMakeLists.txt` (+1 source line), `tests/CMakeLists.txt` (+1 test line).

## Tests to port

- `tests/plugins_tests/test_oot_registration_offline.py:14-25` `test_plugin`
  (`VLLM_PLUGINS=""` ⇒ the OOT arch is NOT registered ⇒ resolution raises "are
  not supported for now") → ported as Phase 1 of `test_plugin_system.cpp`.
- `tests/plugins_tests/test_oot_registration_offline.py:27-45`
  `test_oot_registration_text_generation` (`VLLM_PLUGINS` names the plugin ⇒ the
  OOT arch resolves) → ported as Phase 2 (the RED→GREEN transition).
- `vllm/plugins/__init__.py:82-85` idempotence + `:78-80` "safe to load multiple
  times" → Phase 3. `:68-72` failure isolation → Phase 4.
- SKIPPED-with-reason (no C++ vehicle yet, named W2/W3 residuals):
  `test_platform_plugins.py` (platform-plugin kind), `test_oot_registration_online.py`
  (needs the serving/LLM e2e path + a real checkpoint), the io_processor /
  endpoint / stat_logger plugin tests. Recorded here, not stubbed into the suite.

## Gates

- **Correctness (W1):** `test_plugin_system` — RED-first proven inline (toy arch
  throws "are not supported for now" before load and under `VLLM_PLUGINS=""`;
  resolves to the plugin factory after LoadGeneralPlugins runs it), allowlist
  gate, idempotence (callback runs once per load; no duplicate arch across
  reloads), failure isolation (a throwing plugin is skipped, the good arch
  survives, LoadGeneralPlugins never throws). Result: **1 case / 29 assertions,
  PASS**, CPU `-Werror` clean.
- **Performance:** NOT APPLICABLE (a registration/discovery layer runs once at
  startup; no hot-path or oracle-throughput axis). Recorded NOT-APPLICABLE in
  docs/BENCHMARKS.md.
- **No oracle correctness gate** at W1: the mechanism is a C++ registration layer
  with no Python entry-point equivalent to diff token-for-token; the ported
  offline OOT test IS the behavioral oracle (matched posture, not a golden).

## Dependencies

None to implement W1 (reuses the existing `RegisterModel` seam). Downstream: real
`dlopen` loading depends on a platform `dlopen`/`LoadLibrary` shim; the
platform-plugin kind rides `RegisterPlatform`; the quant-plugin kind rides the
quant registry; endpoint plugins depend on the serving router seam.

## Work breakdown

1. **W0 (this change).** This spike + create the `ENG-PLUGIN-SYSTEM` row.
2. **W1 (this change).** `LoadGeneralPlugins` + the general-plugin registration
   seam + the toy-model out-of-core proof, unit-gated RED-first. **DONE.**
3. **W2.** Real shared-object loading: a `LoadPluginLibrary(path)` that `dlopen`s
   a plugin `.so` and calls the documented `vllm_plugin_register` C-ABI entry;
   port `test_platform_plugins.py` once a dummy `.so` fixture exists.
4. **W3.** Engine/CLI wiring: call `LoadGeneralPlugins()` from the engine
   construction paths (mirror worker_base / engine core / arg_utils) + a
   `--load-plugins` CLI flag; port `test_oot_registration_online.py`.
5. **W4.** The platform-plugin kind (`vllm.platform_plugins` analog over
   `RegisterPlatform`) + the quant-method plugin kind.
6. **W5.** The io_processor / stat_logger / endpoint plugin groups (network-
   surface opt-in posture for endpoint plugins).

## Risks/decisions

- **Python-entrypoints → C++ registration is a mechanical divergence, recorded
  honestly in porting-inventory §9.** C++ has no `importlib.metadata`; the
  faithful analog is the static-init/`dlopen` registration idiom the project
  already uses (`REGISTER_VLLM_MODEL`, `RegisterPlatform`). We mirror the
  BEHAVIOR (load-once, allowlist, failure isolation, register-then-invoke), not
  the transport.
- **The toy plugin must NOT be in the library.** Registering a toy arch at
  library static-init would bump the 28-arch count the model-registry suites
  assert. It lives in the test executable only — which is also the faithful
  model (a plugin is out-of-core by definition).
- **Idempotence + monotonic registries.** `LoadGeneralPlugins` is load-once; a
  test-only latch reset lets one process exercise multiple `VLLM_PLUGINS`
  postures. The model/platform/quant registries only grow, so scenarios are
  ordered allowlist-blocked → allowlisted (never requiring an unregister).
- **No counted-row inflation beyond the one real row.** Exactly one new engine
  row (`ENG-PLUGIN-SYSTEM`); the `check-agent-record.py` `ENGINE_ROWS` constant
  bumps 126→127 with dated rationale, never to force a state transition.
