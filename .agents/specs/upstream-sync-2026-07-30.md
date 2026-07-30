# Upstream sync ranked queue — 2026-07-30 (`555967922..e04a30a77`, 198 commits)

Pin `555967922` (vLLM 0.26.0.dev0) → target `e04a30a77` (2026-07-30). Only the
subset touching our mirrored surface (`src/vllm/**`, `include/vllm/**`,
`src/vt/**`) is ranked. **1 landed this cycle (rank 0); the pin is NOT advanced.**
Report: [sync/2026-07-30-e04a30a.md](../sync/2026-07-30-e04a30a.md).

## Ranked work-list (portable + high-value first)

| rank | vllm# | subject | tier | our file(s) | CPU-buildable? | est size | status |
|---|---|---|---|---|---|---|---|
| 0 | 49754 | expose stream_interval req param | T1 | `sampling_params.*`, `v1/engine/output_processor.cpp`, `entrypoints/openai/protocol.*` | yes | S | **DONE (this cycle)** |
| 1 | 48245 | fix num_output_placeholders preemption underflow | T1-corr | `v1/core/sched/scheduler.cpp`, `async_scheduler.cpp`, `v1/request.*` | yes | L | **SKIPPED (incr 2) — unported prerequisite machinery.** The 1:1 fix rebuilds the async stale-output path around `num_in_flight_tokens` + `num_stale_output_tokens`/`drop_stale_output` + `reset_prefix_cache(reset_running)` force-preempt + `skipped_waiting` + the V2-runner PP concurrent-batch pipeline — NONE ported here (our `async_tokens_to_discard` is never set; no PP batch-queue window). The underflow cannot occur. REQUEUED as a scoped port: land the `num_in_flight_tokens`+force-preempt machinery FIRST, then this fix. |
| 2 | 50297 | fix P/D preemption race condition | T1-corr | `v1/core/sched/scheduler.cpp` | yes (logic) | M | **SKIPPED (incr 2) — depends on rank 1 + KV P/D.** Builds on #48245's `drop_stale_output` and adds `requires_kv_delivery` sourced from the KV-connector producer/consumer role; our scheduler has no connector role in the preempt path and P/D KV transport is N-A by sync policy. REQUEUED behind rank 1 + KV-connector P/D. |
| 3 | 49343 | eagle draft max position embeddings | T1-corr | `config/speculative.h` (queue mapping to `v1/spec_decode/**` was imprecise — the fix is config-level) | yes | S | **PARTIAL-DONE (incr 2).** Ported the pure clamp helper `SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings` (1:1 with `_maybe_override_draft_max_position_embeddings`) + 3 unit tests, `test_speculative_draft_max_position_embeddings` 4/4 CPU (RED-first). No live call site yet (eagle/eagle3 draft loading deferred, EAGLE3 T2); helper is wired in when the eagle loader lands. The 2 model-integration tests SKIPPED (need HF ModelConfig). The bundled `llm_base_proposer.py` block_size determinism half is N-A (no ported eagle proposer). |
| 4 | 49750 | RMSNorm uncontiguous support (1.2–3.1x) | T0-perf | `src/vt/**/rms_norm*` (CUDA + CPU ref) | structure-yes / verify GPU | M | QUEUED — DERIVED+BUILD-VERIFY on CPU ref, runtime perf gate on GB10 |
| 5 | 48391 | RMSNorm batch-invariance (pin block size) | T0-det | `src/vt/**/rms_norm*` | structure-yes / verify GPU | S | QUEUED — determinism; pairs with rank 4 |
| 6 | 49483 | compressed-tensors prioritize fused-name match | T1-corr | `model_executor/layers/quantization/compressed_tensors/**` | yes | S | QUEUED — target-matching order; confirm our matcher |
| 7 | 48589 | extra_config layer-name suffix matching | T1 | quant config | yes | S | QUEUED |
| 8 | 49134 | reject contradictory custom-op directives | T3/T1 | `config/**` | yes | S | QUEUED — config validation |
| 9 | 50210 | Qwen3.5 text-only dense + MoE | T1 | `model_executor/models/qwen3_5*` | yes | L | QUEUED — diff vs our existing qwen3_5 for behavior drift |
| 10 | 48912 | enable EVS for Qwen3.5 | T1 | `qwen3_5*` + multimodal | yes | M | QUEUED — video EVS (efficient-video-sampling) |
| 11 | 47750 | VidCom2 video token pruning | T2 | multimodal | yes | M | INVENTORY — new feature, not ported |

## Already-correct / no-op (recorded, no port)
- **49030** video temporal padding — our `qwen3vl_processor.cpp:279` ceil formula
  `((n+tp-1)/tp)*tp` already equals the fixed `n+(-n%tp)`; affected upstream
  models (qwen2_vl, glm4_1v, keye, kanana_v, llava_onevision2, mimo_v2_omni) are
  not ported. No action.

## N-A bucket (≈150 commits — reason)
ROCm/AITER/Quark/gfx, XPU/Intel, Power/s390x, Rubin/SM107, NIXL/MoRI-IO/KV-
connector P/D transport, Elastic-EP/DeepEP, Rust-frontend/gRPC, CI/buildkite/
docs, libtorch_stable, torch.compile/Inductor plumbing, Kimi-K3 Python/Rust
frontend + NVIDIA CuTe-DSL kernels. All target subsystems/platforms the project
does not gate (upstream-sync.md Rules; discipline.md deviations §9).

## Pin-advance gate
Advance `555967922 → e04a30a77` only after ranks 1–5 land AND the T0 RMSNorm
runtime gate re-runs on GB10 (gate models Qwen3.6-27B/35B token-exact + speed).
Until then the pin holds; this doc is the carry-over work-list.
