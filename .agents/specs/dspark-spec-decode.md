# DSpark speculative decoding — spike spec (`SPEC-DSPARK`)

| Field | Value |
|---|---|
| Row | `SPEC-DSPARK` (engine-matrix), feature-matrix §8 "DSpark (semi-autoregressive block drafter)" |
| Scope | Port DSpark semi-autoregressive block speculative decoding for the Qwen3 and Gemma4 draft families onto the landed `SPEC-DFLASH` lane: config/method resolution, the Markov logit-bias head and draft model, native + Speculators checkpoint loading, the anchor-as-first-prediction query layout, sequential Markov draft sampling (greedy then probabilistic), runner/one-surface wiring, and the token-exact + acceptance + speed gates. **Excluded:** DeepSeek-V4 DSpark (`models/deepseek_v4/*/dspark.py`, HW-blocked), the confidence head (upstream does not wire it), TLI heterogeneous-vocabulary drafting (`SPEC-TLI`), and DSV4 sparse-MLA noncausal attention. |
| Upstream chain | `config/speculative.py:62,310,706-709,869-887,934-964,1004-1027,1333` → `v1/core/sched/scheduler.py:261-265` → `v1/worker/gpu/spec_decode/__init__.py:17-18` → `v1/worker/gpu/spec_decode/dspark/speculator.py:37,76,100,151` → `dspark/utils.py:15` → `model_executor/models/qwen3_dspark.py:36,70,95,132-147,149-185` (+ `gemma4_dspark.py:134,182`) → `transformers_utils/configs/speculators/algos.py:120-165`, all @ `555967922` |
| Roadmap | `ROAD-V1-C3` spec-decode named tail (with `SPEC-TLI`) |
| Role / claim | helper, branch `row/SPEC-DSPARK` |
| Base | `bc6e3d7216523c40fca75c47fec7c5777d04d64c` (origin/main, 2026-08-09) |
| Parity pin | vLLM `555967922` (0.26.0.dev0) at `$VLLM_SOURCE` |
| Supersedes | [dspark-speculator-note.md](dspark-speculator-note.md) — the 5-line 2026-08-08 grounding rider (kept; this is its promised full scope) |
| Our baseline | The landed `SPEC-DFLASH` lane (`CLAIM-DFLASH-D14`, `DONE`): `include/vllm/v1/worker/gpu/spec_decode/dflash/speculator.h` + `src/.../dflash/speculator.cpp`, `include/vllm/model_executor/models/qwen3_dflash.h` (+ `_weights.cpp`, `_gguf.cpp`), the runner branch `src/vllm/v1/worker/gpu/runner.cpp:1183,1760-1761,1874-1891`, `include/vllm/config/speculative.h` (`method` already accepts the string `"dspark"` in `use_eagle()` `:203-204`; `PrepareDflashInputs.sample_from_anchor` exists at `qwen3_dflash.h:320`, always `false`). Detail in §2. |
| Port map | §2 (the six-item A–F delta table: Markov head, sequential sampling, anchor layout, `d2t` reduced vocab, config resolution, Speculators translation) and §4 (which file each lands in). |
| Tests to port | §5 — `tests/v1/e2e/spec_decode/test_spec_decode.py:1489-1530` (`dspark_config` + `test_dspark_correctness_and_acceptance_rate`) and `:291-350` (Gemma4), `tests/models/registry.py:1467-1476`; `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` checked in SKIPPED (DSV4 sparse MLA, out of scope). |
| Gates | §5 — token-exact (or ratified near-tie) vs the pinned oracle on the same target+draft+k, spec-OFF byte-identical, acceptance rate/length at or above the upstream band, our DSpark-ON at or above vLLM DSpark-ON on every throughput axis, and `scripts/check-surface-coverage.py` green. |
| Dependencies | Landed: `SPEC-DFLASH` (`DONE`), `SPEC-REJECTION` verify half, `SPEC-GDN-SEGMENTS`. External, PENDING developer authority: checkpoint downloads (2.79-8.80 GB), dgx GPU time, push/draft-PR. Blocking unknown: R1, whether the pinned oracle runs DSpark at all. |
| Work breakdown | §4 — W1 config, W2 Markov head + draft model, W3 loader (native + Speculators), W4 speculator (anchor layout + sequential sampling), W5 runner + one-surface, W6 gates. W1-W4 are CPU-gateable. |
| Risks/decisions | §6 — R1 oracle runnability (V2 runner), R2 Speculators format is a new subsystem, R3 the community 27B checkpoint's `attn_output_gate`, R4 the `k >= dspark_block_size` garbling trap, R5 sequential sampling vs CUDA-graph capture, R6 greedy before probabilistic, R7 GB10 host-RAM pressure. |
| Status | SPIKE — no code claimed yet |
| Goal (developer, 2026-08-09) | a FULL DSpark implementation in vllm.cpp, mirrored from vLLM |

## 0. Verdict

DSpark is reachable on top of the landed DFlash lane, and it is a **small
additive delta, not a new mechanism**. Upstream's entire DSpark surface is
**1613 lines across 5 files**, and 3 of those files are `class X(DFlashY)`
subclasses:

| Upstream file @ `555967922` | lines | relation to what we already ship |
|---|---:|---|
| `v1/worker/gpu/spec_decode/dspark/speculator.py` | 169 | `DSparkSpeculator(DFlashSpeculator)` — 2 overrides |
| `v1/worker/gpu/spec_decode/dspark/utils.py` | 74 | draft build/alias, ~= our `LoadQwen3DFlash` seam |
| `model_executor/models/qwen3_dspark.py` | 185 | `Qwen3DSparkModel(DFlashQwen3Model)` + Markov head |
| `model_executor/models/gemma4_dspark.py` | 312 | same head over the Gemma4 draft (second target family) |
| `models/deepseek_v4/nvidia/dspark.py` | (DSV4) | **OUT OF SCOPE** — DeepSeek-V4 is HW-blocked (2× Spark) |

Everything DSpark needs that is *hard* — the non-causal in-block attention
primitive, the multi-tap aux feature combine (`fc(cat(aux))`), the context-KV
precompute, the separate-draft-checkpoint loader, the `--speculative-config`
plumbing, the verify/propose runner loop — **is already landed and gated** under
`SPEC-DFLASH` (`CLAIM-DFLASH-D14`, speed gate met). DSpark adds exactly three
things on top (§2).

Draft checkpoints exist for **both gate models** and for a 4B lane that mirrors
the upstream test one-for-one (§3), so this is gateable on-box without the
DeepSeek-V4 hardware blocker.

## 1. What DSpark is (upstream mechanism)

DSpark is **semi-autoregressive block drafting**: DFlash's one-parallel-pass
block draft, plus a cheap sequential pass that re-introduces intra-block
dependency.

Anchors (`555967922`):

1. **Speculator** — `v1/worker/gpu/spec_decode/dspark/speculator.py:37`
   `DSparkSpeculator(DFlashSpeculator)`; overrides `load_draft_model` `:76`,
   adds `_sample_sequential` `:100`, overrides `_generate_draft` `:151`.
   Class docstring `:5-24` states the two differences verbatim.
2. **Draft build** — `dspark/utils.py:15` `load_dspark_model`: same non-causal
   backend resolution as DFlash (`dflash_has_any_non_causal`), same
   embed_tokens / lm_head aliasing to the target (`_should_share`), no PP.
3. **Draft model** — `models/qwen3_dspark.py:36` `DSparkMarkovHead`
   (`markov_w1: [vocab_size, markov_rank]`, `markov_w2: [draft_vocab_size,
   markov_rank]`), `:70` `Qwen3DSparkModel(DFlashQwen3Model)`, `:95`
   `Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM)` with
   `compute_draft_logits` `:132`, `map_draft_to_target` `:137`,
   `markov_embed`/`markov_bias` `:143-147`, `load_weights` `:149-185`.
4. **Config** — `config/speculative.py:62` (`DSparkModelTypes`), `:310`
   (method literal), `:706-709` (target_model_config required; DSV4 ships the
   draft inside the target), `:869-887` (method auto-detect: name contains
   `dspark` OR architectures contain `Qwen3DSparkModel`/`Gemma4DSparkModel`),
   `:934-961` (DSV4 / Gemma4 config normalization), `:963-964`
   (`parallel_drafting = True`), `:1004-1027` (**`num_speculative_tokens >=
   dspark_block_size` or output is garbled, not merely worse**), `:1333`
   `use_dspark()`.
5. **Scheduler lookahead** — `v1/core/sched/scheduler.py:261-265`:
   `num_lookahead_tokens = num_spec_tokens` for DSpark, versus DFlash's
   `num_spec_tokens + 1` `:256-260`. This is the single scheduler-visible
   consequence of anchor-as-first-prediction.
6. **Runner routing** — `v1/worker/gpu/model_runner.py:208` (aux hidden states
   for `eagle3|dflash|dspark`); `spec_decode/__init__.py:17-18` (factory);
   `spec_decode/utils.py:54-75` `get_parallel_drafting_token_id` (mask-token
   resolution order incl. `dspark_noise_token_id`);
   `spec_decode/eagle/eagle3_utils.py:48-50` (`dspark_target_layer_ids` → aux
   layer ids, `+1` indexing).
7. **V2-runner-only** — `config/vllm.py:560-568` forces the V2 runner for
   `method == "dspark"`; `:2168-2177` lists DFlash/DSpark as "parallel drafting
   natively in V2 via their own speculators". (Informational for us: we have one
   runner. See `.agents/vllm-v1-v2.md`.)
8. **Speculators-format translation** — `transformers_utils/configs/speculators/
   algos.py:133-165` `update_dspark`: maps `aux_hidden_state_layer_ids` →
   `eagle_aux_hidden_state_layer_ids` + `target_layer_ids = [i-1]`, sets
   `architectures = ["Qwen3DSparkModel"]`, carries `sample_from_anchor`
   (**default False in this path**), `draft_vocab_size`, `mask_token_id`,
   `markov_rank`, `block_size`, and maps `sliding_window_non_causal` →
   `dflash_config.causal` (`:120-131`, the shared DFlash branch).

### The draft step, precisely

Per request, per decode step:

```
queries      = N = num_speculative_tokens            (sample_from_anchor=True)
             = 1 + N                                 (sample_from_anchor=False, DFlash layout)
q[0]         = anchor  (the last verified/bonus token id)
q[1..]       = noise/mask token id  (mask_token_id | dspark_noise_token_id)

parallel:    head_hidden = DFlash block forward over [context_kv ; queries]
sequential:  prev = anchor_id                        (TARGET vocab)
             for i in 0..N-1:
                 bias_i   = markov_w2( markov_w1[prev] )        # [draft_vocab]
                 logits_i = draft_lm_head(head_hidden[i]) + bias_i
                 tok_i    = argmax(logits_i)          (greedy)  -> map_draft_to_target
                          | gumbel_sample(...)        (probabilistic, target vocab
                                                       after d2t scatter, key pos Q-1)
                 draft[i] = tok_i ; prev = tok_i
```

`sample_pos = query_pos + 1` in the anchor path (standard next-token), against
DFlash's masks-sit-at-the-predicted-position layout; the probabilistic branch
passes `sample_pos - 1` as the Gumbel key because the target verifies a draft
token with its *predecessor's* key (`speculator.py:135-137`).

## 2. Delta versus our landed DFlash lane

Ours today (`SPEC-DFLASH`, `DONE`): `include/vllm/v1/worker/gpu/spec_decode/
dflash/speculator.h` + `src/.../dflash/speculator.cpp` (`DflashProposeBlock`,
`SampleDflashBlockDrafts`), `include/vllm/model_executor/models/qwen3_dflash.h`
(+ `_weights.cpp`, `_gguf.cpp`), runner branch
`src/vllm/v1/worker/gpu/runner.cpp:1183,1760-1761,1874-1891`
(`propose_drafts_dflash`, `set_dflash_draft`, `dflash_tap_layer_ids_`),
`include/vllm/config/speculative.h` (`method` already carries the string
`"dspark"` in `use_eagle()` `:203-204`, and `PrepareDflashInputs` already has a
`sample_from_anchor` field, `qwen3_dflash.h:320`, hardcoded `false`).

| # | New surface | Where it lands | Why it is new |
|---|---|---|---|
| **A** | **Markov head** — `markov_w1 [V, r]` embedding gather + `markov_w2 [V_draft, r]` GEMV, added to the base draft logits | new `qwen3_dspark.{h,cpp}` (+ weights) | no analogue in DFlash; `r = 256` in every shipped ckpt |
| **B** | **Sequential N-step sample loop** with the running `prev` token, replacing DFlash's single parallel argmax | new `spec_decode/dspark/speculator.{h,cpp}` | DFlash samples all block positions independently; DSpark's step *i* depends on step *i−1* |
| **C** | **Anchor-as-first-prediction layout** (`N` queries, `sample_pos = q+1`) | `qwen3_dflash.h` `PrepareDflashInputs.sample_from_anchor` (field already present, currently dead — always `false`) | our only exercised path is `sample_from_anchor=false`. **Already correct, verified**: `NumLookaheadTokens()` (`speculative.h:220-229`) branches on `use_dflash()` (`== "dflash"` only) for `k+1` and falls through to `use_eagle()` — which already lists `"dspark"` — for `k`, matching `scheduler.py:261-265`. No change needed; add the pinning test |
| **D** | **Reduced draft vocab (`d2t`)** — `draft_vocab_size=32000` in both RedHatAI speculators ckpts; greedy remaps ids, probabilistic scatters logits into target columns | draft model + sampler | our DFlash drafts are full-vocab; `draft_id_to_target_id` is unloaded today |
| **E** | **Config**: method `dspark`, `dspark_block_size` floor, `dspark_noise_token_id`, `dspark_target_layer_ids`, `n_predict = block_size` | `include/vllm/config/speculative.h` + `src/vllm/config/speculative.cpp` | method string exists; the resolution rules do not |
| **F** | **Speculators-format config translation** (`algos.py:133`) | new; we have **zero** `speculators` handling today (grep: no hits in `src/`,`include/`) | needed for the RedHatAI gate-model drafts (§3) |

Explicitly **reused unchanged**: the non-causal block attention primitive
(`vt::DFlashBlockAttention`), `CombineAuxFeatures`/`fc`, context-KV precompute,
`ForwardBlockLogitsWithContext`, the multi-tap aux plumbing, the draft-checkpoint
loader shape, the `--speculative-config` JSON parse, the C-ABI/CLI/server
surface, and the verify/rejection loop.

## 3. Checkpoints (HF API, fetched 2026-08-09)

| Checkpoint | Target | Format | Size | Block | `sample_from_anchor` | draft vocab | Why it matters |
|---|---|---|---:|---:|---|---|---|
| `deepseek-ai/dspark_qwen3_4b_block7` | `Qwen/Qwen3-4B-FP8` | native | 2.79 GB | 7 | absent → **True** | full (151936) | **the upstream test's own pair** (`test_dspark_correctness_and_acceptance_rate`, `large_gpu_mark(min_gb=24)`) — smallest honest lane |
| `deepseek-ai/dspark_qwen3_8b_block7` | Qwen3-8B | native | 4.74 GB | 7 | absent → True | full | upstream's default `model` (`speculative.py:875`) |
| `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` | **our 35B gate model** | speculators | 1.90 GB | 8 | **True** | **32000 + d2t** | binding gate-model lane; 5× SWA layers, hidden 2048 |
| `satgeze/Qwen3.6-27B-DSpark` | **our 27B gate model** | native | 8.80 GB (+3.73 GB GGUF) | 15 | absent → True | full | 27B lane; community ckpt, `attn_output_gate: true` (**risk R3**) |
| `RedHatAI/gemma-4-31B-it-speculator.dspark` | Gemma4-31B | speculators | 8.39 GB | 8 | **False** (1+N layout) | 32000 + d2t | second target family; exercises the DFlash-layout branch |
| `deepseek-ai/dspark_gemma4_12b_block7` | gemma-4-12B-it | native | — | 7 | — | — | the upstream Gemma4 e2e test's pair |

Non-goals: `deepseek-ai/DeepSeek-V4-*-DSpark` (draft ships inside the target;
DSV4 is HW-blocked, see [mla-deepseek-campaign.md](mla-deepseek-campaign.md)),
Kimi-K3/GLM-5.2/MiniMax-M3 DSpark drafts (targets not in scope for this row).

Every shipped config sets `enable_confidence_head: true`, and upstream
**deliberately does not wire it** (`qwen3_dspark.py:165-167`: "confidence_head is
not wired into inference yet; skip its weights"). We mirror that: skip, do not
invent.

## 4. Slice plan

Each slice: red test first, focused gate, full gate, fresh review, then the next.
CPU-runnable through W3; W4+ needs the GPU box.

| Slice | Content | Gate | HW |
|---|---|---|---|
| **W1 config** | accept `"dspark"` in `ParseSpeculativeConfigJson` (+ require the `model` key, as dflash does), `ResolveDspark` (method resolution by name/architectures, `n_predict = block_size`, the `k >= dspark_block_size` hard error), noise-token resolution order, `parallel_drafting`; pin `NumLookaheadTokens() == k` | new `tests/vllm/config/test_speculative_dspark.cpp`. **RED today**: `speculative.cpp:44-49` rejects every method outside `mtp\|dflash\|ngram\|draft_model`, so `{"method":"dspark",…}` throws | CPU |
| **W2 Markov head + draft model** | `Qwen3DSparkModel` = our `Qwen3DFlashModel` backbone + `DSparkMarkovHead`; `compute_draft_logits`, `markov_embed`/`markov_bias`, `map_draft_to_target`, `d2t` | op-level golden vs a dumped HF Markov head (`tools/parity/`), rel-L2 band as in D2 | CPU (+GPU port) |
| **W3 loader** | native key spelling (`model.markov_head.markov_w{1,2}`, `d2t`→`draft_id_to_target_id`, skip `t2d`/`mask_embedding`/`confidence_head`) **and** the speculators translation (`algos.py:133`) incl. `transformer_layer_config` unwrap | loader unit test on the real ckpt tensor list; the D-lane lesson: assert the exact on-disk spelling | CPU |
| **W4 speculator** | anchor layout (`N` queries, `sample_pos=q+1`) + `_sample_sequential`; greedy first, probabilistic (Gumbel + d2t scatter) second | `SampleDsparkBlockDrafts` unit gate on a synthetic fixture (RED: parallel-sample ≠ sequential when the Markov bias is non-zero) | CPU |
| **W5 runner + surface** | `propose_drafts_dspark`, `set_dspark_draft`, aux taps from `dspark_target_layer_ids`, `--speculative-config '{"method":"dspark",...}'` through CLI/server/C-ABI (**POL-ONE-SURFACE**: `include/vllm.h` already carries `speculative_config`; no new ABI symbol expected — confirm with `scripts/check-surface-coverage.py`) | engine e2e; spec-OFF byte-identical (the DFlash inertness discipline) | GPU |
| **W6 gates** | token-exactness vs the pinned oracle running the SAME draft; acceptance rate/length; the on-par-or-above speed A/B vs vLLM `--speculative-config dspark` | §5 | GPU (dgx, flock) |

Order of lanes: **4B first** (upstream's own test pair, 24 GB class, native
format, full vocab, greedy path) → **35B gate model** (speculators format +
reduced vocab + `sample_from_anchor=True`) → **27B** → Gemma4 (`1+N` layout).

## 5. Tests to port (POL-PORT-TESTS) and gates

Upstream test surface at the pin (there is no unit test for the DSpark
speculator; upstream covers it e2e):

| Upstream | Ours |
|---|---|
| `tests/v1/e2e/spec_decode/test_spec_decode.py:1489-1530` `dspark_config` + `test_dspark_correctness_and_acceptance_rate` (Qwen3-4B-FP8 + `dspark_qwen3_4b_block7`, k=7, probabilistic, GSM8K acc ≥ 0.801·0.9, acceptance_rate ≥ 0.428·0.9, acceptance_len ≥ 3.994·0.9) | `tests/parity/test_qwen3_dspark_spec_decode.cpp` (e2e) + the acceptance-rate floor; we additionally gate greedy token-exactness vs the oracle (stricter, per [gates.md](../verification.md)) |
| `tests/v1/e2e/spec_decode/test_spec_decode.py:291-350` `test_gemma4_dspark_correctness_and_acceptance_rate` | deferred to the Gemma4 lane; checked in SKIPPED with the reason until then |
| `tests/models/registry.py:1467-1476` (`Qwen3DSparkModel`, `Gemma4DSparkModel`, `DSparkDraftModel`) | model-registry rows + `docs/FEATURES.md` arch table (CI-bound, 33 archs today) |
| `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` | **OUT OF SCOPE** — DSV4 sparse MLA, already checked in SKIPPED under the MLA campaign |

Binding acceptance for `DONE` (house gate, [verification.md](../verification.md)):

1. **Token-exact** greedy output versus the pinned vLLM oracle running the same
   target+draft+k, and versus our own spec-OFF decode, at c1 — or the ratified
   near-tie distributional form where the oracle's own greedy is non-deterministic.
2. **Spec-OFF byte-identical**: no DSpark code on the non-speculative path.
3. **Acceptance rate/length** at or above the upstream reference band.
4. **Speed**: our DSpark-ON ≥ vLLM's DSpark-ON on every throughput axis, ≤ on
   latency/memory, on an idle box, ≥2 reps, `nsys` both sides with the same tool
   if a kernel claim is made.
5. **One surface**: reachable through `include/vllm.h` + CLI + server, examples
   as thin clients; `scripts/check-surface-coverage.py` green.

## 6. Risks and open questions

- **R1 — oracle runnability.** Before any implementation slice, prove the pinned
  oracle actually *runs* `Qwen/Qwen3-4B-FP8` + `dspark_qwen3_4b_block7` and emits
  a greedy golden (memory: "gateability = model RUNS, not config constructs";
  DSpark forces the **V2 runner**, which our oracle recipes have not exercised).
  If the pinned oracle cannot run DSpark, the whole row is oracle-blocked and
  that is the stop condition, recorded — not worked around.
- **R2 — Speculators format is a new subsystem for us.** Two of the three
  gate-model drafts are speculators-format. Scope it as its own slice (W3) so it
  can be reviewed independently; a `NEEDS_CONTEXT`-style stop is better than
  smuggling it into the loader.
- **R3 — `satgeze/Qwen3.6-27B-DSpark` is a community checkpoint** with
  `attn_output_gate: true`. That key is **not read** by upstream
  `qwen3_dflash.py` / `qwen3_dspark.py` / `qwen3.py` (grep, `555967922`), so
  either it is inert metadata or the tensors are gated and vLLM itself cannot
  load it. Verify against the oracle before adopting it as the 27B lane;
  `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` and `Koopah/…-NVFP4-DSPARK` are the
  alternates.
- **R4 — the `k >= dspark_block_size` trap.** Upstream documents that a smaller
  `k` yields *garbled* output, not merely lower acceptance
  (`speculative.py:1004-1027`). Our config gate must be a hard error with a red
  test, not a warning.
- **R5 — sequential sampling versus CUDA graphs.** Upstream captures the whole
  draft step (parallel backbone + the N-step Markov loop) in a FULL graph
  (`speculator.py:22-24`). An N-iteration host loop with a device round-trip per
  step would be a decode-path regression at exactly the point the feature is
  supposed to win. Memory: "CUDA-graph capture bakes stack addresses" — no
  function-local upload temporaries in the captured region.
- **R6 — probabilistic path.** The upstream tests gate at `temperature=1.0`
  with `draft_sample_method: probabilistic` (Gumbel + rejection). Our landed
  spec-decode gates are greedy. Greedy-first is the honest order; the
  probabilistic path is a named, separate slice, not an assumption.
- **R7 — GB10 memory.** `gpu_memory_utilization` reserves HOST RAM on GB10 and a
  big oracle beside ctest has rebooted the box. The 4B lane is chosen partly for
  this; keep the oracle and our engine serialized under the `flock`.

## 7. Evidence, authority, stop conditions

- Evidence root: `dgx:~/work/vllm.cpp-dspark-<slice>/`, one `flock`, named tmux.
- Authority still needed from the developer: (a) **downloading the draft +
  target checkpoints** (2.8–8.8 GB each — a large-asset download), (b) GPU-box
  time on dgx, (c) pushing `row/SPEC-DSPARK` and opening its draft PR. Until (a)
  lands, W1/W2/W4 remain CPU-gateable on synthetic fixtures and R1 is `PENDING`.
- Stop conditions: R1 unprovable at the pin (oracle cannot run DSpark) → record
  the exact external blocker and stop; any slice failing its red-then-green
  discipline → back to a fresh implementer, never repaired in the coordinating
  session (POL-REVIEW-NO-REPAIR).

Sources: pin files cited inline at `555967922`; HF API queried 2026-08-09;
our anchors cited against `bc6e3d72`.
