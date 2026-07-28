# SGLang-behavior enablement — API / ABI / server surface

**CLAIM-SGLANG-ABI-DOCS** (2026-07-28). Side-quest: make the SGLang-alike
behaviors first-class, DOCUMENTED knobs exposed from BOTH the C++ API and the
C ABI (not env-var / internal-only), with user-facing docs. CPU-only, exact-gate,
strict default-inertness (every knob defaults to today's behavior ⇒
byte-identical). User-facing companion: [`docs/SGLANG-COMPAT.md`](../../docs/SGLANG-COMPAT.md).

This spec is the engineering record: which SGLang behavior maps to which knob on
each surface, grounded in the code that was added, plus the honest residuals.

> **RECONCILED to ABI v10 (2026-07-28).** This claim originally proposed a v9 that
> appended an int `scheduler_policy` (enum ordinal) AND `enable_jump_forward`. A
> concurrent session shipped ABI **v9** first with an overlapping engine-config
> growth, including a **string** `scheduling_policy` ("fcfs"/"priority"/"lpm")
> that already exposes scheduler policy incl. LPM. To avoid a duplicate knob, the
> int `scheduler_policy` was **DROPPED entirely**; only the genuinely-new
> `enable_jump_forward` is kept, appended as ABI **v10** at the END of
> `vllm_model_params` (after the concurrent session's v9 fields). LPM is now
> reached through the concurrent session's `scheduling_policy = "lpm"` string
> field. `vllm_abi_version()` returns **10**.

## The four behaviors and their surfaces

| Behavior (sglang-matrix row) | C++ API (`EngineParams` / `SamplingParams`) | C ABI (`vllm_model_params` / `vllm_sampling_params`) | Server flag | Env |
|---|---|---|---|---|
| RadixAttention / APC (`SGLANG-KV-RADIX`) | `enable_prefix_caching` (`std::optional<bool>`) | `enable_prefix_caching` int tri-state (ABI **v7**) | `--[no-]enable-prefix-caching`, alias `--[enable\|disable]-radix-attention` | — |
| LPM cache-aware scheduling (`SGLANG-SCHED-LPM`, SW1) | `policy` (`SchedulerPolicy::kLPM`) | `scheduling_policy` **string** `"fcfs"`/`"priority"`/`"lpm"` (ABI **v9**, concurrent session) | `--scheduling-policy lpm`, alias `--schedule-policy lpm` | — |
| Jump-forward decoding (`SGLANG-CONSTRAIN-JUMP`, SW3) | `enable_jump_forward` (`std::optional<bool>`) | `enable_jump_forward` int tri-state (ABI **v10**) | `--[enable\|disable]-jump-forward` | `VT_ENABLE_JUMP_FORWARD` (override) |
| Custom logits processors (`SGLANG-SAMPLING-CUSTOM`) | sampler callback stage | `vllm_sampling_params.logits_processor` + `logits_processor_user_data` (ABI **v8**) | — (per-request programmatic hook) | — |

## ABI v10 field (added this claim)

A single field appended at the END of `vllm_model_params`, after the concurrent
session's v9 fields (`max_num_batched_tokens` / `scheduling_policy` /
`kv_transfer_config`), with `0`=default semantics, so a v9 caller that zero-fills
the growth is byte-identical:

| Field | Type | Values | Default | Maps to |
|---|---|---|---|---|
| `enable_jump_forward` | `int32_t` | `0`=default (env-resolved, OFF), `1`=on, `2`=off | `0` | `EngineParams::enable_jump_forward` |

Out-of-range ⇒ `VLLM_ERR_INVALID_ARGUMENT`, rejected before any load work
(mirrors the ABI v7 `enable_prefix_caching` tri-state validation exactly).
Scheduler policy (incl. LPM) is the concurrent session's v9 **string** field
`scheduling_policy` — there is no separate int knob. `vllm_abi_version()` returns
**10**.

## Grounding (file:line — the mirrored `enable_prefix_caching` pattern)

The ABI v7 `enable_prefix_caching` thread was mirrored end to end for
jump-forward; LPM plumbing is the concurrent session's `scheduling_policy` string:

- ABI struct field + doc block — `include/vllm.h` (`enable_prefix_caching`
  `:178` → new `enable_jump_forward` appended after the v9
  `kv_transfer_config`); `VLLM_ABI_VERSION` 9→10, new v10 doc paragraph.
- default-struct init — `src/capi/vllm_c.cpp`
  `vllm_model_params_default()` (`p.enable_jump_forward = 0`).
- load-time tri-state translation — `src/capi/vllm_c.cpp` `vllm_engine_load`:
  the `enable_jump_forward` switch is folded INLINE (0/1/2 → `ep.enable_jump_forward`
  nullopt/true/false; out-of-range ⇒ `VLLM_ERR_INVALID_ARGUMENT`), mirroring the
  `enable_prefix_caching` switch directly above it. (No shared helper: it is a
  single-field translation, so the 266f5568 `ApplyAbiSchedulerAndJumpForward`
  helper + its `engine_handle.h` decl were dropped along with the int
  scheduler-policy half.) `scheduling_policy` is translated by the concurrent
  session via `vllm::SchedulerPolicyFromString(params->scheduling_policy)`.
- C++ config — `include/vllm/entrypoints/model_loader.h` `EngineParams::policy`
  (pre-existing) + new `EngineParams::enable_jump_forward` (`std::optional<bool>`).
- engine resolution — `src/vllm/entrypoints/model_loader.cpp` ctor:
  `jump_forward_enabled_(vllm::v1::JumpForwardEnabled(params.enable_jump_forward))`;
  the policy already threads through the scheduler config from `params.policy`.
- observable accessors (for the gate) — `LoadedEngine::jump_forward_enabled()`
  (`model_loader.h`), `Scheduler::policy()` (`include/vllm/v1/core/sched/scheduler.h`,
  re-added by this reconciliation — the concurrent session did not add it).
- jump-forward config-aware resolution — `src/vllm/v1/structured_output/jump_forward.cpp`
  `JumpForwardEnabled(std::optional<bool> configured)` (env override wins when set,
  else the config field, default OFF); the no-arg overload = `JumpForwardEnabled(nullopt)`.
- server flags — `examples/server/main.cpp`: `--scheduling-policy`/`--schedule-policy`
  (pre-existing), new `--[enable|disable]-jump-forward` (double-specify rejected).

## Env-var precedence (jump-forward)

`VT_ENABLE_JUMP_FORWARD`, WHEN SET, overrides the C-ABI / C++ / server field
(`1`/`true`/`TRUE`/`on` ⇒ on, anything else ⇒ off). When UNSET, the field decides
(`std::nullopt` / ABI `0` ⇒ OFF). This mirrors the house `VT_ASYNC_SCHED`
convention (`AsyncSchedulingEnabled`). The env var is retained as the rollback /
override path.

## Honest landed-vs-residual

- **LPM (SW1):** admission-order parity, output-neutral; degrades to `fcfs` with
  APC off. Exposed on all three surfaces — the C-ABI half is the concurrent
  session's `scheduling_policy = "lpm"` string field (this reconciliation does NOT
  re-add an int knob for it).
- **Jump-forward (SW3):** the **token-unique subset only** is landed and
  output-neutral. The general byte-forced-but-multi-tokenizable span (SGLang's
  re-tokenize + boundary rollback) is a named residual, and the production
  scheduler splice (KV recompute for jumped tokens) is not yet wired — so
  jump-forward stays DEFAULT-OFF. SGLang itself removed its jump-forward scheduler
  wiring upstream (commit 935cda944b). Now exposed on all three surfaces (was
  env-var only): C++ `EngineParams::enable_jump_forward`, C-ABI (ABI v10), server.
- **RadixAttention/APC (v7)** and **custom logits processors (v8):** already
  exposed on the C ABI; documented here for completeness, unchanged.
- **SW2 throughput lever:** NOT-APPLICABLE to our APC (blocks cached at allocation
  time; the redundant same-step prefill SGLang avoids never occurs here).

## Gate (CPU, exact, RED-first)

- ABI e2e — `tests/capi/test_capi.cpp`: two ABI v10 jump-forward cases. (1)
  `enable_jump_forward` defaults to `0`, the tri-state validates (valid `0/1/2` ⇒
  reach load, out-of-range ⇒ `VLLM_ERR_INVALID_ARGUMENT`); (2) through
  `EngineParams` (the same field `vllm_engine_load` translates onto) a built
  `LoadedEngine` with `enable_jump_forward=true` has `jump_forward_enabled()==true`,
  and `nullopt`/`false` are inert. RED-first: on base main the field / accessor /
  `JumpForwardEnabled(optional<bool>)` overload do not exist (compile error);
  default (all-zero) is byte-identical to ABI v9. LPM via ABI is covered by the
  concurrent session's `scheduling_policy` string cases (not duplicated here).
- Inertness — full `test_capi` (33/33), `test_scheduler` (36/36),
  `test_scheduler_lpm` (6/6), `test_jump_forward` (5/5), `test_scheduler_config`
  (13/13) all green; default (all-zero) config byte-identical, and the concurrent
  session's v9 fields (`scheduling_policy`/`max_num_batched_tokens`/
  `kv_transfer_config`) still work.
- Server — `--[enable|disable]-jump-forward` parses; double-specify rejected;
  `--scheduling-policy lpm` parses. Clean `-Werror` build incl. `server`, 0 warnings.
