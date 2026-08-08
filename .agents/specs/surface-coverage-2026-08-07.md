# Surface-coverage audit — 2026-08-07 (`ARCH-ONE-SURFACE` supporting audit)

The systemic defect: model capabilities keep landing in a per-model CLI instead of the
shared model runner / engine / OpenAI server / C-ABI. This audit grounds every cell in
code (`file:line`) and ships the guard that prevents recurrence
(`scripts/check-surface-coverage.py`, two axes). It is the code-grounded input to the
ONE SURFACE program (`.agents/specs/one-surface-abi.md`, row `ARCH-ONE-SURFACE`): its
"Order of work" item 3 (the recurrence guard) is delivered here.

The four surfaces, and the public boundary the guard draws:

- **Registry** — `REGISTER_VLLM_MODEL(sym,"Arch",factory,info)`; `ModelRegistry::Forward`
  (`include/vllm/model_executor/models/model_registry.h:334`) is the ONE generation entry
  point the server + C-ABI both drive. 30 archs registered (+ `check-supported-models.py`).
- **Shared runner fast path** — device-resident logits + on-GPU sampler + paged bf16-KV
  attention (`scripts/check-runner-routing-consistency.py` axes a/b/c).
- **Server** — `ApiServer::register_routes` (`src/vllm/entrypoints/openai/api_server.cpp:713`).
- **C-ABI** — `include/vllm.h`, the ONLY installed header (`CMakeLists.txt:1757`
  `install(FILES include/vllm.h)`; `vllm_shared` PUBLIC-includes exactly it and version-
  scripts to export only `vllm_*`, `CMakeLists.txt:1712-1730`). The public boundary =
  `include/vllm.h`. Everything under `include/vllm/**`, `vt/**`, `src/**` is internal.

## Summary — worst offenders

| Rank | Gap | Registered | Servable | C-ABI | Where the capability actually lives |
|---|---|---|---|---|---|
| 1 | **MiniMax-H3 video+audio generation** | **CLOSED (ROW 2)**: still off-registry (a diffusion lane, not a text arch) but library-owned end to end | **`/v1/videos` through the library seam** (`MiniMaxH3VideoEngine` + `MiniMaxH3VideoGenParamsFromRequest`; the server keeps only flag plumbing + the ffmpeg exec) | **`vllm_video_*` (ABI v12)** | library seam `vllm::multimodal::MiniMaxH3VideoEngine`; both examples are clean ABI clients |
| 2 | **Laguna fast decode** | yes, but forward is a stub | no (stub `VT_CHECK`s non-bf16) | no | `examples/laguna_gen` (keep-quant GGUF + NVFP4 W4A4) |
| 3 | **DeepSeek-V4 fast decode** | yes, but forward is a W3 stub | no (stub) | no | `examples/deepseek_v4_gen` (keep-quant GGUF) |
| 4 | **Audio transcription** | **CLOSED (ROW 1)**: Parakeet CTC/RNNT/TDT registered (transcription-only; Whisper/Voxtral still off-registry) | **live `/v1/audio/transcriptions`** (task-conditional; the run_batch line stays a residual) | **`vllm_transcribe` (ABI v11)** | library seam `ParakeetTranscriber`; example is a clean ABI client |
| 5 | **Kimi-Linear incremental decode** | yes (recompute forward IS shared) | recompute only | no | `examples/kimi_linear_gen` (§18/§19 paged-incremental + resident loader) |
| 6 | **Embeddings / pooling** | NO (all `is_pooling_model=false`) | no (`/v1/embeddings` = residual) | no | engine-side pooler exists (`ENG-POOLER-SEQ`), never invoked live |
| 7 | **Multimodal input over HTTP/ABI** | 5 archs `supports_multimodal` | image seam only; tower not run in engine step | no (text-only chat) | `chat_mm.cpp` seam; towers test-only |

21 of 30 registered text archs are fully on-framework (registry + runner + server + ABI):
no gap. The defect is concentrated in the seven rows above.

## Coverage matrix

### A. On-framework text archs (no gap)

`Cohere`, `DeepseekV2`, `Gemma/2/3`, `Glm4`, `Glm4MoeLite`, `Granite`, `InternLM2/3`,
`Llama`, `MiniCPM/3`, `Mistral`, `Olmo2/3`, `OPT`, `Phi/3`, `Qwen3`, `Qwen3Moe`,
`StableLm` — registered, decode is born on the runner (device-resident logits, on-GPU
sampler; clean on `check-runner-routing-consistency.py`), served over `/v1/chat` +
`/v1/completions`, and reachable through `vllm_complete`/`vllm_chat`. `KimiK3` and
`Cohere` are registered scaffolds (forward refuses / W0), not gaps in this audit's sense.

### B. Registered, but the FAST decode is CLI-only (the "registered ≠ served" trap)

| Arch | Registered forward | CLI-only fast path (file:line) | Server | C-ABI |
|---|---|---|---|---|
| `LagunaForCausalLM` | reference stub — `VT_CHECK(false)` on non-bf16 (`laguna.cpp:156`; registry note `laguna_registry.cpp:10-18`) | `LagunaForwardGguf(Cached)` (`laguna.cpp:1549,2713`), Marlin residents (`laguna.cpp:1293`), fp4-shared (`laguna_shared_fp4.cpp:121`); driver `examples/laguna_gen/main.cpp` | no | no |
| `DeepseekV4ForCausalLM` | W3-W8 stub — forward `VT_CHECK`s, paged KV is a placeholder "never exercised" (`deepseek_v4_registry.cpp:22,128-145`) | `DeepseekV4ForwardGguf(Cached)` + `DeepseekV4KvCache` (`deepseek_v4.cpp:2280,2303`); driver `examples/deepseek_v4_gen/main.cpp:124,199` | no | no |
| `KimiLinearForCausalLM` | recompute forward IS shared (`ForwardDevice`→`ForwardDeviceCompute`, `kimi_linear.cpp:104`) | §18/§19 paged-incremental decode `ForwardPrefillIncremental`/`ForwardDecodeStepIncremental`/`KimiDecodeCache` + resident loader `LoadKimiLinearResidentBf16Weights` (`kimi_linear_device.cpp:1631`, `kimi_linear_weights.cpp:531`); driver `examples/kimi_linear_gen/main.cpp` | recompute only | no |

All three drivers run a PRIVATE host-argmax greedy loop, not the on-GPU sampler.

### C. Off-registry capability lanes

| Lane | Registered | Code (file:line) | Server | C-ABI | Driver |
|---|---|---|---|---|---|
| MiniMax-H3 video+audio GEN | NO (diffusion lane; **ROW 2** made it library-owned without a registry entry) | `minimax_h3*.cpp` (~22 TUs) + the **`minimax_h3_video.cpp` seam** (ROW 2), vt op `kMiniMaxH3` | `/v1/videos` via `set_video_runner`, the runner now a thin exec wrapper over the LIBRARY seam (`MiniMaxH3VideoGenParamsFromRequest` -> `Generate` -> exec argv) | **`vllm_video_engine_load` / `vllm_video_generate` / `vllm_video_result_free` / `vllm_video_mux_argv` (ABI v12)** | `examples/minimax_h3_gen`, `examples/minimax_h3_mux` = thin `vllm.h` clients |
| Parakeet/FastConformer ASR | **YES (ROW 1)**: ParakeetForCTC/RNNT/TDT, `parakeet_registry.cpp` (SupportsTranscription-only; text paths refuse by task) | `parakeet_transcription.cpp` seam composes encoder/transducer/audio-processor; the example's private `ReadWav16BitMono`/`LoadVocab`/`DecodeIds` are DELETED (`vllm::Tokenizer` now decodes Metaspace split=true) | **`/v1/audio/transcriptions`** (task-conditional) | **`vllm_transcribe` (ABI v11)** | `examples/parakeet_transcribe` = thin `vllm.h` client |
| Voxtral audio->text | NO (`VoxtralForConditionalGeneration` unregistered) | `voxtral.cpp` (`vllm::multimodal`) | NO (`/v1/audio/transcriptions` = `run_batch.cpp:188` residual) | NO | tests-only reachability |
| Whisper audio encoder | NO | `whisper_audio.cpp:174` | NO | NO | tests-only callers |
| Pooling / embeddings | NO (`is_pooling_model=false` in all 27) | `layers/pooler/*.cpp`, `pool/pooling_runner` (`ENG-POOLER-SEQ`) | NO (`/v1/embeddings` = residual) | NO | `PoolingRunner` test-only |
| Multimodal INPUT | 5 archs `supports_multimodal` (Gemma4, KimiK3, Qwen3VL, Qwen3.5/-Moe) | towers `qwen3_vl_vision.cpp:374`, `gemma4_vision.cpp:170`; **Gemma-4 AUDIO USM tower is STANDALONE** (Gemma-4 text+image route via `ModelRegistry::Forward`, audio does NOT) | image seam only, raw-RGB, no stream, **tower not run in live step** (`chat_mm.cpp`; runner never consumes `mm_features`) | NO (text-only; `vllm_c.cpp` sets no mm seam) | — |
| MTP / DFlash / ngram speculators | NO (sub-config / draft checkpoint; EAGLE unwired) | `spec_decode/{mtp,dflash}/speculator.cpp`, `ngram_proposer.cpp` | via `speculative_config` | via `speculative_config` (`vllm.h:172`) | — |

Speculators are the one non-text lane already reachable through BOTH server and ABI (a
JSON sub-config, not a separate surface) — the model for how the others should land.

## Example include-graph audit (guard axis 1)

Public boundary = `#include "vllm.h"` only. 14 example units; `examples/cli` (vllm-cli,
links `vllm::shared`, `#include "vllm.h"` only, `cli/main.cpp:16`) is the sole clean ABI
client. **ROW 1 UPDATE (2026-08-07): `parakeet_transcribe` is the SECOND clean ABI
client** — the Parakeet fold rewrote it against `vllm.h` + `vllm::shared` only, and the
ratchet fell 12 -> 11. **ROW 2 UPDATE (2026-08-08): `minimax_h3_gen` and
`minimax_h3_mux` are the THIRD and FOURTH clean ABI clients** (the video fold, ABI
v12 `vllm_video_*`), and the ratchet fell 11 -> 9. **ROW 7 UPDATE (2026-08-07):
`kimi_linear_gen` is the FIFTH clean ABI client** — the Kimi-Linear paged-runner fold
made the fast paged-incremental decode the ENGINE's production path, grew
`vllm_complete_tokens` (ABI v13, pre-tokenized completion returning generated token
ids) and rewrote the example against `vllm.h` + `vllm::shared` only; the ratchet fell
9 -> 8. **RATCHET EXCEPTION (2026-08-08): 8 -> 9 — conscious operator exception** —
external contribution #65 (`examples/cpu_kernel_bench`, the RPi5/Cortex-A76 PMU
kernel-bench harness) predates the guard; dev-tool class, fold row
`ARCH-ONE-SURFACE` (the same op-bench ABI fold as `quant_gemm_bench`); the ratchet
re-shrinks when the bench folds. The remaining 9 reach `include/vllm/**` / `vt/**`
and are transition-tracked in `scripts/example-abi-allowlist.txt`:

- Capability drivers: `deepseek_v4_gen`, `laguna_gen`, `server`.
- Dev/diagnostic (internal-by-nature, folded for consistency): `bench` (via
  `bench_core.h`), `tokenize`, `dump_container`, `dequant_nvfp4`, `quant_gemm_bench`,
  `cpu_kernel_bench`.
- Out of the gated `examples/` tree: `benchmarks/vulkan_gemm_ab.cpp` (Vulkan A/B harness).

**Policy (developer-directed 2026-08-07): no permanent exemptions.** Every allowlist entry
— drivers AND dev/diagnostic tools — is a transition-tracker pointing at a fold row; the
guard fails on any internal include not tracked, and a shrink-only ratchet
(`MAX_INTERNAL_REACHING`, 9 since the 2026-08-08 #65 operator exception; 8 since ROW 7; 9 since ROW 2; 11 since ROW 1) means the count can only fall as folds land, never grow to
admit a new violation (the sole recorded exception: the dated #65 note above). The public header set is DERIVED from the CMake install rules
(exactly `include/vllm.h` today), not hardcoded. The guard catches BOTH breach vectors: a
`#include "vllm/..."|"vt/..."|"src/..."` AND a CMake `-I` grant into the internal tree
(`target_include_directories(<target> ... ${CMAKE_SOURCE_DIR}/src)` — `quant-gemm-bench`,
`examples/CMakeLists.txt:51`, so a bare `#include` can't sneak in). `examples/cli` is the
target-state exemplar: `#include "vllm.h"` only, links `vllm::shared`.

**Why the existing registry gate does NOT cover this.** `check-supported-models.py:22-24`
deliberately tolerates "non-registered lanes (standalone audio / diffusion forwards …)" —
the exact class Parakeet/Voxtral/H3 occupy — so it can bind the FEATURES arch table to the
registry but can NOT enforce that a capability is embedder-reachable. That is why this is a
separate, first-of-its-kind gate: no checker scanned example includes or ABI reachability
before it.

## C-ABI capability coverage (guard axis 2)

Bound to `include/vllm.h` by the marked `abi-capability-table` in `docs/FEATURES.md`.
The ABI is text-generation-complete (completion, chat, async, structured output, tool +
reasoning parsers, speculative config, custom logits processor — 7 `reachable` rows). 4
`embedder-unreachable` rows, each tracked in `scripts/abi-capability-allowlist.txt`
against `ARCH-ONE-SURFACE`: embeddings/pooling, audio transcription, video+audio
generation, multimodal input.

**Severity note — the ABI happy path is itself untested.** `vllm_engine_load` is never
CI-gated on a REAL model load: `tests/capi/test_capi.cpp` covers only the bad-path error
contract (null args, bad paths → `VLLM_ERR_*`). So even the `reachable` text-gen rows are
verified by the CLI/engine tests, not through the ABI on a live model. A real-model ABI
smoke test (a tiny checkpoint through `vllm_engine_load` + `vllm_complete`) is owed under
`ARCH-ONE-SURFACE`, so "reachable" means reachable-and-exercised, not merely declared.

## Ranked fold plan

Per gap the order is fixed (ONE SURFACE): **(1) grow the public C-ABI entry point →
(2) rewrite the example as an ABI client → (3) delete the parallel implementation.** All
lanes are leaves of `ARCH-ONE-SURFACE` (do not open parallel rows).

| # | Fold | Grow ABI (new `vllm.h` surface) | Then rewrite / delete | Effort | Depends on |
|---|---|---|---|---|---|
| 1 | Video+audio gen | **DONE (ROW 2, 2026-08-08)**: `vllm_video_engine_load`/`vllm_video_generate`/`vllm_video_result_free` + `vllm_video_mux_argv` (ABI v12); `/v1/videos` routes through the SAME `MiniMaxH3VideoEngine` seam (job/status/content stay `VideoJobStore`-served; the runner is now a thin exec wrapper the example injects, because the SPAWN stays in examples/ — the ratified ffmpeg boundary) | **DONE**: `minimax_h3_gen` + `minimax_h3_mux` rewritten as `vllm.h` clients (frames+WAV byte-identical to the pre-fold binary on the fold fixture); server driver glue deleted (54 -> 7 H3 refs, all seam type names) | L | H3 loaders; ffmpeg-mux boundary (ratified in `examples/`) |
| 2 | Laguna fast decode | make the registered `LagunaForCausalLM` forward route the keep-quant/NVFP4 device path (retire the stub); load keep-quant GGUF/NVFP4 dirs through `vllm_engine_load` | rewrite `laguna_gen`; delete `LagunaForwardGguf*` | M | keep-quant load in the engine loader |
| 3 | DeepSeek-V4 fast decode | same as (2) for `DeepseekV4ForCausalLM`; real MLA paged KV (retire the W3 stub) | rewrite `deepseek_v4_gen`; delete `DeepseekV4ForwardGguf*` | M | MLA paged-KV topology |
| 4 | Audio transcription | **DONE (ROW 1, 2026-08-07)**: `vllm_transcribe` (ABI v11) + live `/v1/audio/transcriptions`; ParakeetForCTC/RNNT/TDT registered (SupportsTranscription mirror, refuse-by-task) | **DONE**: `parakeet_transcribe` rewritten as a `vllm.h` client (byte-identical transcript goldens); route live, task-conditional | M | encoder→text seam (LANDED: `ParakeetTranscriber`) |
| 5 | Kimi-Linear incremental | expose the incremental decode path through the runner/engine (the recompute forward already routes) | rewrite `kimi_linear_gen` | S–M | `KimiDecodeCache` on the runner |
| 6 | Embeddings/pooling | `vllm_embed`/pooling entry point + live `/v1/embeddings`; register a pooling arch (`is_pooling_model=true`); invoke `PoolingRunner` in the step | — | M | pooler live-wiring |
| 7 | Multimodal input | multimodal-content entry point on `vllm_chat`; run the vision/audio tower in the engine step (`mm_features`→`ModelForwardInput.mm`) | wire `chat_mm` seam into the ABI | L | `MM-SERVE-E2E` engine mm-forward residual |
| 8 | Device-selection knob | **DONE (ROW 8, 2026-08-08, `row/DEVICE-KNOB`; leakage follow-up PR #139)**: `vllm_model_params.device` (ABI v14: 0=auto/1=cpu/2=cuda, the vLLM `DeviceConfig.device` names, device.py:13) → `EngineParams::device` → `SelectQueue`; the stable public name resolves through `FindPlatformByName` and its registered `DeviceType` is propagated without a shared CUDA literal; explicit cpu never probes, explicit ABSENT cuda fails LOUD before model I/O; DSR 32 / `kcuda=0` | **DONE**: both thin clients consume the field; zero value byte-identical (auto probe) | S | mirror vLLM `--device`/`DeviceConfig` |
| 9 | Voxtral + audio chat seam | register `VoxtralForConditionalGeneration` + fold `VoxtralGenerateGreedy` into the registry forward; audio-capable chat fn + an engine consumer for `AudioKwargs` mm_features | rewrite tests→clients | M | mirror upstream `voxtral.py:309`, `SupportsTranscription` |
| 10 | Gemma-4 audio e2e | bf16 device audio forward + audio→text merge (residual `gemma4_audio.h:41`); USM log-mel front end | fold into the registered mm forward | M | `MM-SERVE-E2E` |
| 11 | Tokenizer/bench ABI + real-load gate | `vllm_tokenize`/`vllm_detokenize`; token-id/count fields on the stream callback for bench; **gate `vllm_engine_load` on a REAL tiny checkpoint at least once** (today bad-path only, `test_capi.cpp:474`) | rewrite `tokenize`/`bench` as clients | S-M | `/tokenize` route exists (`api_server.cpp:432`) |

The AUTHORITATIVE ranked fold plan (10 rows with vLLM mirror shapes + `file:line` for every
step) is the ONE SURFACE audit synthesis feeding `ARCH-ONE-SURFACE`; remediation leaves are
already tracked (Kimi-Linear runner fold task #279, Parakeet ASR task #280). This spec's
table is the surface-coverage slice of it, grouped by the seven code-grounded gaps above.

## FEATURES overclaim corrections made

- `DeepseekV4ForCausalLM` / `LagunaForCausalLM` registered-arch rows: their speed cells
  ("beats ds4 1.144x" / "vLLM parity+ 1.03x, default on") now name that the number is via
  the `deepseek-v4-gen` / `laguna-gen` CLI and the registered engine forward is a stub —
  so a reader does not read it as server/ABI-served (`ARCH-ONE-SURFACE` fold).
- Added the `abi-capability-table` so the C-ABI's text-only scope and its 4 open
  capability gaps are stated and gated, not implied.
- Rows found ALREADY honest (left unchanged): "Multimodal over the OpenAI server ☐",
  "Multimodal over HTTP: architecturally blocked", "Embedding/pooling endpoints ◐ engine
  only", "`/v1/videos` ✅" (the server genuinely serves it, opt-in, via the runner).

## Spike contract

**Scope.** In: the recurrence guard (both axes) + the code-grounded matrix + the ranked
fold plan as `ARCH-ONE-SURFACE` leaves + FEATURES corrections. Out: implementing any
fold (that is `ARCH-ONE-SURFACE`'s own work); no CUDA build (CPU-only session).

**Upstream chain.** vLLM ships no C-ABI (deviation, `porting-inventory.md §9`); the
mirror target is llama.cpp's `llama.h` handle surface. Server routes mirror
`vllm/entrypoints/openai/*/api_router.py`.

**Our baseline.** `include/vllm.h` (ABI v10, text-gen), `api_server.cpp:713` routes,
`ModelRegistry::Forward`, the two runner-routing allowlists, `check-supported-models.py`.

**Port map / harness map.** New: `scripts/check-surface-coverage.py`,
`scripts/example-abi-allowlist.txt`, `scripts/abi-capability-allowlist.txt`,
`tests/scripts/test_check_surface_coverage.py`; `docs/FEATURES.md` abi-capability-table;
wired into `scripts/agent-preflight.sh` + `.github/workflows/ci.yml`.

**Tests to port.** `tests/scripts/test_check_surface_coverage.py` — 46 unit + mutation
cases (both axes: uncovered/stale/malformed/reachable-missing-symbol/unreachable-not-
allowlisted/stale-capability/unknown-status/comment-only-symbol/quoted+angle-include/
CMake-grant + shipped-tree-green + SUBPROCESS enforcement seams that run the checker binary
on mutated fixtures so main()'s grant-merge, ratchet and capability_errors wiring reds).

**Gates.** `python3 scripts/check-surface-coverage.py` == 0 and the mutation suite == 0,
both in preflight and CI; no CUDA gate owed (correctness/tracking change).

**Dependencies.** `ARCH-ONE-SURFACE` / `.agents/specs/one-surface-abi.md` (the fold
program this feeds); PR #89 (Parakeet, the first example the guard caught).

**Risks / decisions.** The include-boundary lint treats `include/vllm/**` as internal
(there is no curated public C++ API today — only `vllm.h` is installed); if a curated
public C++ header set is later declared in CMake, extend `EXPECTED_PUBLIC` + the pin check.
Diagnostic tools are allowlisted (transition-trackers), not permanently exempted; the
consistency fold is lowest priority.

DESIGN DECISION (axis 2 binding is advertised→real, one-directional). The capability table
binds each ADVERTISED capability to a real `vllm.h` symbol (or a tracked gap); it does NOT
pin which capabilities MUST be advertised. So DELETING a `reachable` row passes silently —
that removes an advertisement, not a reachability guarantee, and FEATURES' own completeness
is a separate concern (`check-public-doc-tables.py`). Adding an aspirational row, changing a
symbol to a missing one, or dropping the whole table all still red. Guarding reachable-row
REMOVAL would mean pinning the advertised set, which is maintenance churn out of proportion
to the risk; if that risk ever bites, red a `reachable` removal unless its symbol also left
`vllm.h`.

META-GAP (Finding 7, not fixed here — systemic). That the two axes RUN in
`scripts/agent-preflight.sh` + CI is enforced by CONVENTION (the roster lists), not by a
gate — the same as every peer checker (`check-fusion-consistency`, `check-runner-routing`,
…). A meta-guard that a checker is actually wired belongs to a separate row, not this one.

**Work breakdown (non-overlapping).** Each fold-plan row above is one schedulable
`ARCH-ONE-SURFACE` leaf; the guard's allowlist entries shrink one at a time as each lands.
