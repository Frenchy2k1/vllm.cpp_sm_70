# Live-state audit — 2026-08-06 (P0)

Read-only census of every LIVE matrix row against Git reality. It PROPOSES; it
changes no matrix. Corrections land in a separate reviewable change so this
reasoning can be read independently of the churn it justifies.

## Scope

- **Rows.** The 188 rows in a live state (`SPIKE`, `READY`, `ACTIVE`, `GATING`,
  `PARTIAL`, `BLOCKED`) across all seven `.agents/*-matrix.md` files. The five in
  `check-agent-record.py`'s `MATRIX_PATHS`, plus `feature-matrix.md` and
  `sglang-matrix.md`, which no CI gate parses as claim rows. All 11 of the live
  rows those two add are in `feature-matrix.md`; `sglang-matrix.md` holds **0**,
  because its lifecycle column is `Class` (FUSED / SGLANG-DISTINCT /
  OUT-OF-SCOPE), not `State`, so no table in it parses as a claim table. It stays
  in `AUDIT_MATRIX_PATHS` anyway — free today, automatic coverage the day it
  grows a `State` column.
- **Trees.** Rows are read from the working tree on `spec/issue-native-tracking`
  @ `4aaa9a8c`. Git evidence is read from `origin/main` @ `cf32c619` (fetched
  2026-08-06). The only matrix delta between the two is `engine-matrix.md`
  dropping nine `INVENTORIED` spec-decode rows and its count table; every one of
  those is `INVENTORIED`, so the LIVE census is identical to `origin/main`'s.
- **What this audit decides.** Which `ACTIVE` claims are unsupportable by Git;
  which `PARTIAL` rows fail to name their own missing modes; which stable IDs are
  live in two matrices at once; and, for each unsupportable `ACTIVE` row, which
  target states the state contracts actually permit.
- **What it does NOT decide.** Whether any row is *finished*. No verdict in this
  document promotes a row to `DONE`, and none may be cited as evidence that work
  completed. The strongest thing the tool can say is that a commit message
  mentions the row ID — see Findings ➁.

## Method

[`scripts/audit-live-rows.py`](../../scripts/audit-live-rows.py) — three modes:
markdown report (default), `--json`, and `--check` (exit 1 iff an `ACTIVE` row is
abandoned). Row parsing is *imported* from `check-agent-record.py` rather than
reimplemented, so the audit and the gate can never disagree about what a row is.

### Classification rules, verbatim

```python
def classify_active(branches, unmerged_by_branch, commits):
    live_branches = [b for b in branches if unmerged_by_branch[b]]
    if live_branches:
        return "IN-FLIGHT", f"unmerged commits on {joined}"
    if branches:
        return "LANDED", f"branch {joined} exists and is fully merged into main"
    if commits:
        return "LANDED", f"on main: {commits[0]}"
    return "ABANDONED", "no branch, no commit on main mentioning the row ID"
```

Supporting rules that matter when reading a verdict:

- **`IN-FLIGHT` beats `LANDED`.** A row can have landed groundwork and still have
  open follow-up; calling that finished would silently steal a live claim.
- **ID matching is anchored, never a substring** —
  `(^|[^A-Za-z0-9_-])<ID>([^A-Za-z0-9_-]|$)`. 55 pairs of live IDs are prefixes of
  longer ones, and a substring match would credit the short row with the long
  row's commits.
- **`origin/main` must resolve or the tool aborts.** Absence of information must
  never look like absence of work; without the guard every row would report
  abandoned at once.
- **The `PARTIAL` flag is a keyword heuristic, not a gate.** It reports the marker
  that fired so a reviewer can discount a bad hit; `--check` deliberately fails
  only on abandoned `ACTIVE` rows.

### Hand-verified sample

A classifier wrong on a sample is wrong on all 54, so six rows were checked by
hand before any of this was believed — four the tool called `ABANDONED`, two it
called `LANDED`. For each: `git log --oneline --all --fixed-strings --grep=<ID>`
(deliberately BROADER than the tool: all refs, substring match), `git branch -a
--list "*row/<ID>"`, and the tool's own anchored `origin/main` query.

| Row | Tool verdict | `--all --fixed-strings` hits | `row/<ID>` branches | Anchored `origin/main` hits | Holds? |
|---|---|---|---|---|---|
| `ENG-MM-AUDIO-ENCODER` | ABANDONED | none | none | none | yes |
| `KERNEL-GDN-SCRATCH` | ABANDONED | none | none | none | yes |
| `BACKEND-DISTRIBUTED-TP` | ABANDONED | none | none | none | yes |
| `QUANT-GGUF-IQ3_XXS` | ABANDONED | none | none | none | yes |
| `BACKEND-MLX` | LANDED | 1 (`d45c8cda`) | none | 1 (`d45c8cda`) | yes |
| `BACKEND-CUDA-OTHER` | LANDED | 1 (`d45c8cda`) | none | 1 (`d45c8cda`) | yes |

**Outcome: the sample holds, with a caveat that reshapes the whole report.** The
four `ABANDONED` rows have *zero* mentions anywhere in the repository — not on
`origin/main`, not on any other ref, not under a substring match. The verdict is
not a matching artefact; there is genuinely no ID-keyed Git evidence.

The caveat is on the `LANDED` side. Both sampled `LANDED` rows are credited to
`d45c8cda docs(reconcile): ... (records-only)`, whose entire diff is
`.agents/*`, `README.md` and `docs/BENCHMARKS.md` — **no code**. A record edit
that mentions a row ID satisfies the `LANDED` rule while proving nothing about the
work. `LANDED` therefore means *"has evidence worth reading"*, never *"finished"*.

## Findings

### ➀ Verdict distribution

| Measure | Value |
|---|---|
| Live rows | 188 |
| States | `PARTIAL` 68, `ACTIVE` 54, `SPIKE` 43, `GATING` 10, `BLOCKED` 7, `READY` 6 |
| `ACTIVE` verdicts | LANDED 44, ABANDONED 10 |
| `PARTIAL` rows that do not name their missing modes | 20 of 68 |
| Unique IDs among live rows | 186 (2 IDs live twice) |

**Zero rows classify `IN-FLIGHT`, and there is no merged-branch evidence anywhere.**
Not one live row has a `row/<ID>` branch, local or remote. Every `LANDED` verdict
in this repository rests on the weakest rule in the classifier — a commit message
mentioning the ID.

### ➁ What the 44 `LANDED` verdicts are actually worth

All 44 rest on `on main: <commit>`. Splitting them by whether that commit
touched code at all:

- **36 rows** are credited to a commit that touched `src/`, `include/`,
  `tests/`, `cmake/` or `scripts/`. That commit *mentions* the row ID; it is not
  established here that it implemented the row.
- **8 rows** are credited to a commit whose diff is records and docs ONLY.
  For these the "evidence" is somebody editing a record:

  - `BACKEND-CUDA-OTHER` — `d45c8cda` (.agents/feature-matrix.md:276)
  - `BACKEND-CUDA-SM087` — `aca8d7d7` (.agents/backend-matrix.md:170)
  - `BACKEND-CUDA-SM110` — `b34f9909` (.agents/backend-matrix.md:176)
  - `BACKEND-MLX` — `d45c8cda` (.agents/feature-matrix.md:279)
  - `KERNEL-GEMM-CPU-ELEM` — `428e20f0` (.agents/kernel-matrix.md:126)
  - `MODEL-SPEC` — `d45c8cda` (.agents/feature-matrix.md:154)
  - `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` — `2ff7252a` (.agents/model-matrix.md:174)
  - `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` — `2ff7252a` (.agents/model-matrix.md:219)

**No row is proposed for a `DONE` downgrade anywhere in this document.** A commit
mention is not a completion proof, and a records-only commit is not even a work
proof. Establishing completion requires a human reading the row's own code/test
anchors against the gate output — outside this audit's reach.

#### ➁a The gate's evidence rule self-blinds on the rows this audit vacated

**Recorded as a known hazard, not fixed here.** The rule in `main_commits()` is a
commit-MESSAGE mention with no code-touch discriminator: any commit on
`origin/main` whose subject or body names the row ID as a whole token makes the
row `LANDED`, whatever that commit changed. Record commits in this repository
routinely name row IDs, so the gate's discriminating power decays with every
records-only commit that lands.

That is not hypothetical for this branch. **9 of the 10 IDs vacated in ➂ are
named in this branch's OWN commit messages** — every one except
`MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` (`BACKEND-DISTRIBUTED-TP`
and `ENG-MM-AUDIO-ENCODER` twice each; `ENG-MM-INPUT-PIPELINE`,
`ENG-MM-VIDEO-FORWARD`, `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`,
`MODEL-TEXT-glm4-glm4-for-causal-lm`,
`MODEL-MM-voxtral-voxtral-for-conditional-generation` and `QUANT-GGUF-IQ3_XXS`
once each). Once those messages reach `origin/main` — and a squash merge
concatenates them, so squashing does not shed them — `main_commits()` is
non-empty for those nine FOREVER. **`--check` can therefore never flag those rows
abandoned again**, including if somebody re-flips one to `ACTIVE` with no work
behind it. The gate keeps its full force on every row it has not yet fired on;
it has spent itself precisely on the ten it just repaired.

The fix is already computed above: filter `main_commits()` by whether the commit
touched a code path (`src/`, `include/`, `tests/`, `cmake/`, `scripts/`) — the
same discriminator that splits the 44 `LANDED` verdicts 36/8 in this section.
**It is deliberately NOT applied in this change**, because applying it today
flips the 8 records-only-backed `LANDED` rows listed above to `ABANDONED` and
turns the standing gate RED for everyone. Vacating eight more rows is a
correction with its own evidence and its own decision; it does not get to ride in
on a tool tweak. Follow-up carried in `.agents/state.md`.

### ➂ The 10 abandoned `ACTIVE` rows

No `row/<ID>` branch and no commit on `origin/main` naming the ID:

| Row | Location | Active claims in `coordination.md` |
|---|---|---|
| `BACKEND-DISTRIBUTED-TP` | .agents/backend-matrix.md:273 | `CLAIM-PARALLELISM-MODES-SPIKE`, `CLAIM-SCALE-OUT-SPIKE`, `CLAIM-SCALE-OUT-W2` |
| `ENG-MM-INPUT-PIPELINE` | .agents/engine-matrix.md:72 | `CLAIM-MULTIMODAL-M1` |
| `ENG-MM-VIDEO-FORWARD` | .agents/engine-matrix.md:76 | `CLAIM-MULTIMODAL-M3C`, `CLAIM-MULTIMODAL-TOWER-FIDELITY` |
| `ENG-MM-AUDIO-ENCODER` | .agents/engine-matrix.md:78 | `CLAIM-AUDIO-ENCODER` |
| `KERNEL-GDN-AOT-BF16` | .agents/kernel-matrix.md:154 | `CLAIM-PR3`, `CLAIM-TRITON-AOT-PER-ARCH` |
| `KERNEL-GDN-SCRATCH` | .agents/kernel-matrix.md:155 | `CLAIM-PR3` |
| `MODEL-TEXT-glm4-glm4-for-causal-lm` | .agents/model-matrix.md:195 | `CLAIM-GLM-DSA-LATEST-DEEPSEEK` |
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | .agents/model-matrix.md:380 | `CLAIM-GEMMA4-G1`, `CLAIM-GEMMA4-G1B`, `CLAIM-GEMMA4-G2`, `CLAIM-GEMMA4-G2-IMPL`, `CLAIM-GEMMA4-G3`, `CLAIM-GEMMA4-MM-E2E`, `CLAIM-GEMMA4-MULTIMODAL`, `CLAIM-MULTIMODAL-TRACK` |
| `MODEL-MM-voxtral-voxtral-for-conditional-generation` | .agents/model-matrix.md:458 | `CLAIM-AUDIO-E2E` |
| `QUANT-GGUF-IQ3_XXS` | .agents/quantization-matrix.md:71 | `CLAIM-DEEPSEEK-V4-W8` |

By matrix: .agents/engine-matrix.md 3, .agents/model-matrix.md 3, .agents/kernel-matrix.md 2, .agents/quantization-matrix.md 1, .agents/backend-matrix.md 1.

**These rows are not necessarily unstarted work.** Every one of the ten carries
in-row anchors asserting passing gates (`ENG-MM-AUDIO-ENCODER`: "A2 encoder-tower
fidelity gate PASS 203/203"; `MODEL-MM-voxtral-...`: "audio→text e2e gate PASS
14/14"; `BACKEND-DISTRIBUTED-TP`: "CPU multi-rank TP gate 60/60"). What the
verdict establishes is narrower and still damning: **the `ACTIVE` claim is
unverifiable from Git**, because the convention of naming the stable row ID in the
commit message was not followed. The row says it is being worked on; the history
cannot corroborate that anyone ever did.

### ➃ `PARTIAL` rows that do not say what is missing

`PARTIAL` asserts partial support. 20 of 68 rows never name the missing
part, so a reader cannot tell what is absent:

| Row | Location |
|---|---|
| `BACKEND-CUDA-COMP-MARLIN` | .agents/backend-matrix.md:190 |
| `BACKEND-CUDA-COMP-SCALEDMM-C3X` | .agents/backend-matrix.md:194 |
| `BACKEND-CUDA-COMP-FP4` | .agents/backend-matrix.md:197 |
| `ENG-CUDAGRAPH` | .agents/engine-matrix.md:60 |
| `ENG-SCHED-KNOBS` | .agents/engine-matrix.md:67 |
| `TOOLS-STRUCTURED-CORE` | .agents/engine-matrix.md:149 |
| `TOOLS-STRUCTURAL-TAG` | .agents/engine-matrix.md:151 |
| `SERVE-DISCOVERY-HEALTH` | .agents/engine-matrix.md:188 |
| `SERVE-CLI-BENCH` | .agents/engine-matrix.md:198 |
| `LOAD-CONFIG-SURFACE` | .agents/engine-matrix.md:243 |
| `QUANT-VLLM-BREADTH` | .agents/feature-matrix.md:168 |
| `BACKEND-CUDA-SM121` | .agents/feature-matrix.md:275 |
| `BACKEND-CPU` | .agents/feature-matrix.md:277 |
| `KERNEL-ROPE-QKNORM` | .agents/kernel-matrix.md:131 |
| `KERNEL-MOE-UNQUANTIZED` | .agents/kernel-matrix.md:150 |
| `KERNEL-MOE-QUANTIZED` | .agents/kernel-matrix.md:151 |
| `MODEL-TEXT-stablelm-stablelm-for-causal-lm` | .agents/model-matrix.md:267 |
| `QUANT-GGUF-F32` | .agents/quantization-matrix.md:57 |
| `QUANT-MIXED-MODELOPT` | .agents/quantization-matrix.md:129 |
| `QUANT-KV-FP8` | .agents/quantization-matrix.md:158 |

Marker distribution over all `PARTIAL` rows:

| Flag | Rows |
|---|---|
| does not name its missing modes | 20 |
| explicit via 'no' | 15 |
| explicit via 'only' | 14 |
| explicit via 'gap' | 5 |
| explicit via 'PENDING' | 4 |
| explicit via 'NO' | 4 |
| explicit via 'pending' | 3 |
| explicit via 'blocked' | 2 |
| explicit via 'absent' | 1 |

**The true vague count is higher than 20.** Of the 48 rows the flag reads as
explicit, **11** qualify ONLY via a bare `no`/`gap` with no stronger marker
anywhere in the row — and in ten of the eleven the marker fires on prose asserting
GOODNESS, not absence:

| Row | Location | Prose the marker fired on | Genuine gap statement? |
|---|---|---|---|
| `LOAD-SAFETENSORS` | engine-matrix.md:236 | "the mirror build **no** longer double-resides with the full source mmap" | no — asserts a fix |
| `LOAD-GGUF` | engine-matrix.md:240 | "Pinned vLLM has **no** GGUF loader" | no — describes UPSTREAM's absence, not ours |
| `MODEL-TEXT-internlm2-intern-lm2-for-causal-lm` | model-matrix.md:215 | "max **gap** 0.0 nats, 0 divergent" | no — a passing near-tie metric |
| `QUANT-GGUF-F16` | quantization-matrix.md:58 | "Its speed **gap** ... was CLOSED" | no — asserts a closed gap |
| `QUANT-GGUF-Q4_0` | quantization-matrix.md:59 | "**no** bf16 expansion on the executed path" | no — asserts a good property |
| `QUANT-GGUF-Q8_0` | quantization-matrix.md:63 | "**no** bf16 expansion on the executed path" | no — asserts a good property |
| `QUANT-GGUF-Q3_K` | quantization-matrix.md:65 | "**no** bf16 expansion on the executed path" | no — asserts a good property |
| `QUANT-GGUF-Q4_K` | quantization-matrix.md:66 | "**no** bf16 expansion on the executed path" | no — asserts a good property |
| `QUANT-GGUF-Q5_K` | quantization-matrix.md:67 | "**no** bf16 expansion on the executed path" | no — asserts a good property |
| `QUANT-GGUF-Q6_K` | quantization-matrix.md:68 | "**no** bf16 expansion on the executed path" | no — asserts a good property |
| `MODEL-FACTORY` | feature-matrix.md:150 | "the two-model GPU **no**-regression campaign" | marker is a false hit inside a hyphenated word, but the surrounding prose DOES name an open campaign — the one of eleven that survives |

So at least **30 of 68** `PARTIAL` rows fail to state their missing modes
(20 flagged outright, 10 passing only on a marker that means the opposite of a
gap). The heuristic UNDER-flags; that is why the report names the marker rather
than printing a bare boolean. The remaining 13 bare-`no`/`gap` first-matches
(`KV-BLOCK-POOL`, `TOOLS-CALLING-CORE`, `MODEL-TEXT-minicpm...`,
`MODEL-TEXT-minicpm3...`, `MODEL-TEXT-mistral...`, `MODEL-TEXT-olmo2...`,
`MODEL-TEXT-phi...`, `MODEL-TEXT-phi3...`, `MODEL-MM-qwen3-vl...`,
`QUANT-GGUF-NVFP4`, `BACKEND-CUDA-COMP-FA`, `BACKEND-CPU` (backend-matrix.md:226),
`MODEL-MM`) also carry a stronger marker (`absent`, `pending`, `missing`, `only`,
`without`) elsewhere in the row, so their explicit reading does not depend on the
weak hit.

`BACKEND-CPU` reads as explicit here AND appears in the vague-20 table above,
which is not a contradiction: it is one of the two IDs live in two matrices at
once (see *Duplicate live IDs*), and its two instances read differently. The
`backend-matrix.md` instance (`:226`) is the explicit one; the
`feature-matrix.md` roll-up (`:277`) is the vague one the table flags. The other
duplicate, `BACKEND-CUDA-SM121`, splits the same way — explicit via `only` at
backend-matrix.md:178, vague at feature-matrix.md:275 — it just never reaches
this list, because `only` is a strong marker. Every entry in both lists is about
ONE instance; where an ID is duplicated the location disambiguates it.

### ➄ Full report

Verbatim output of `scripts/audit-live-rows.py`:

| Row | State | Location | Verdict | Evidence | Flag |
|---|---|---|---|---|---|
| `KV-PREFIX-MATCH-UNIT` | `PARTIAL` | .agents/engine-matrix.md:58 | - | - | explicit via 'absent' |
| `ENG-CUDAGRAPH` | `PARTIAL` | .agents/engine-matrix.md:60 | - | - | does not name its missing modes |
| `ENG-PRIORITY-SCHED` | `GATING` | .agents/engine-matrix.md:63 | - | - | - |
| `ENG-CORE-BUSY-LOOP` | `GATING` | .agents/engine-matrix.md:66 | - | - | - |
| `ENG-SCHED-KNOBS` | `PARTIAL` | .agents/engine-matrix.md:67 | - | - | does not name its missing modes |
| `ENG-CASCADE-ATTN` | `SPIKE` | .agents/engine-matrix.md:68 | - | - | - |
| `ENG-RUNNER-MODELSHAPE` | `ACTIVE` | .agents/engine-matrix.md:71 | LANDED | on main: 5ab3f111 docs(record): anchor backfill batch 1 — 12 engine rows made precise, 10 more leave ACTIVE | - |
| `ENG-MM-INPUT-PIPELINE` | `ACTIVE` | .agents/engine-matrix.md:72 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `ENG-MM-VISION-TOWER` | `ACTIVE` | .agents/engine-matrix.md:73 | LANDED | on main: d796187a docs(record): anchor backfill batch 4 — manual pass, 11 hand-verified anchors and the real blocker named | - |
| `ENG-MM-TEXT-BACKBONE` | `ACTIVE` | .agents/engine-matrix.md:74 | LANDED | on main: 2a8ff336 feat(multimodal): M2b/M2c — Qwen3-VL text-backbone numeric contracts unit-green vs vLLM 0.25.0 | - |
| `ENG-MM-QWEN36-VL-FORWARD` | `ACTIVE` | .agents/engine-matrix.md:75 | LANDED | on main: e89d51d8 feat(mm-speed): Qwen VISION-FORWARD — §14 flash kernel extended to the tower (byte-exact); ATTRIBUTION REFUTES the assumed lever, tower already BEATS vLLM (CLAIM-MM-SPEED-QWEN-IMAGE) | - |
| `ENG-MM-VIDEO-FORWARD` | `ACTIVE` | .agents/engine-matrix.md:76 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `ENG-MM-AUDIO-PIPELINE` | `ACTIVE` | .agents/engine-matrix.md:77 | LANDED | on main: adcac8e6 feat(multimodal): AUDIO track A0+A1 — audio INPUT pipeline on whisper-small, feature-parity gate PASS 77/77 | - |
| `ENG-MM-AUDIO-ENCODER` | `ACTIVE` | .agents/engine-matrix.md:78 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `ENG-MM-AUDIO-E2E` | `ACTIVE` | .agents/engine-matrix.md:79 | LANDED | on main: 9e34a19c perf(mm): ROAD-V1-MM lever #3 — ADOPT FA2 varlen as Voxtral audio decode; BEATS vLLM (0.97x), closes the LAST mm decode-speed gap | - |
| `KV-BLOCK-POOL` | `PARTIAL` | .agents/engine-matrix.md:92 | - | - | explicit via 'gap' |
| `KV-HYBRID-COORD` | `PARTIAL` | .agents/engine-matrix.md:95 | - | - | explicit via 'only' |
| `KV-MAMBA-ALIGN` | `SPIKE` | .agents/engine-matrix.md:96 | - | - | - |
| `KV-SLIDING-LOCAL-SPECS` | `READY` | .agents/engine-matrix.md:97 | - | - | - |
| `KV-SLIDING-WINDOW-SPEC` | `GATING` | .agents/engine-matrix.md:98 | - | - | - |
| `KV-CHUNKED-LOCAL-SPEC` | `GATING` | .agents/engine-matrix.md:99 | - | - | - |
| `KV-EVENTS` | `ACTIVE` | .agents/engine-matrix.md:105 | LANDED | on main: d796187a docs(record): anchor backfill batch 4 — manual pass, 11 hand-verified anchors and the real blocker named | - |
| `KV-SIZING` | `PARTIAL` | .agents/engine-matrix.md:108 | - | - | explicit via 'only' |
| `ENG-EXPERT-STREAM` | `READY` | .agents/engine-matrix.md:110 | - | - | - |
| `PAR-TP` | `READY` | .agents/engine-matrix.md:118 | - | - | - |
| `SAMPLE-PROMPT-LOGPROBS` | `PARTIAL` | .agents/engine-matrix.md:132 | - | - | explicit via 'pending' |
| `SERVE-COMPLETION-LONGTAIL` | `PARTIAL` | .agents/engine-matrix.md:135 | - | - | explicit via 'only' |
| `SAMPLE-BEAM` | `ACTIVE` | .agents/engine-matrix.md:136 | LANDED | on main: 0151314f feat(serve): async/production beam search (SAMPLE-BEAM) — BeamSearchAsync over AsyncLLM, token-identical to sync; C7 CLAIM-C7-BEAM-ASYNC | - |
| `SAMPLE-N` | `ACTIVE` | .agents/engine-matrix.md:142 | LANDED | on main: aed4718e feat(serve): best_of + use_beam_search OpenAI-endpoint surface — C7 SAMPLE-BEST-OF + SAMPLE-BEAM endpoint | - |
| `SAMPLE-BEST-OF` | `ACTIVE` | .agents/engine-matrix.md:143 | LANDED | on main: aed4718e feat(serve): best_of + use_beam_search OpenAI-endpoint surface — C7 SAMPLE-BEST-OF + SAMPLE-BEAM endpoint | - |
| `TOOLS-STRUCTURED-CORE` | `PARTIAL` | .agents/engine-matrix.md:149 | - | - | does not name its missing modes |
| `TOOLS-XGRAMMAR` | `ACTIVE` | .agents/engine-matrix.md:150 | LANDED | on main: d796187a docs(record): anchor backfill batch 4 — manual pass, 11 hand-verified anchors and the real blocker named | - |
| `TOOLS-STRUCTURAL-TAG` | `PARTIAL` | .agents/engine-matrix.md:151 | - | - | does not name its missing modes |
| `TOOLS-CALLING-CORE` | `PARTIAL` | .agents/engine-matrix.md:153 | - | - | explicit via 'no' |
| `TOOLS-STREAMING-PARSER` | `ACTIVE` | .agents/engine-matrix.md:154 | LANDED | on main: 05237562 feat(parser): JSON-schema tool-arg type coercion (_fix_arg_types) — ROAD-V1-C8 residual CLOSED | - |
| `SPEC-REJECTION` | `ACTIVE` | .agents/engine-matrix.md:164 | LANDED | on main: dfa610b2 feat(spec-decode): generic separate draft-model proposer (SPEC-DRAFT-MODEL) — W0 spike + W1 CPU greedy propose brick; Medusa spiked (SPEC-MEDUSA) | - |
| `SPEC-GDN-SEGMENTS` | `ACTIVE` | .agents/engine-matrix.md:165 | LANDED | on main: 3ae5cfe0 feat(spec-decode): SPEC-MTP I5a — GDN layer spec routing + runner spec-metadata upload | - |
| `SPEC-NGRAM` | `ACTIVE` | .agents/engine-matrix.md:169 | LANDED | on main: d796187a docs(record): anchor backfill batch 4 — manual pass, 11 hand-verified anchors and the real blocker named | - |
| `SPEC-EAGLE3` | `BLOCKED` | .agents/engine-matrix.md:170 | - | - | - |
| `SPEC-DRAFT-MODEL` | `ACTIVE` | .agents/engine-matrix.md:180 | LANDED | on main: dfa610b2 feat(spec-decode): generic separate draft-model proposer (SPEC-DRAFT-MODEL) — W0 spike + W1 CPU greedy propose brick; Medusa spiked (SPEC-MEDUSA) | - |
| `SPEC-MEDUSA` | `SPIKE` | .agents/engine-matrix.md:181 | - | - | - |
| `SERVE-DISCOVERY-HEALTH` | `PARTIAL` | .agents/engine-matrix.md:188 | - | - | does not name its missing modes |
| `SERVE-STREAM-USAGE` | `GATING` | .agents/engine-matrix.md:191 | - | - | - |
| `SERVE-ASYNC-LLM` | `GATING` | .agents/engine-matrix.md:194 | - | - | - |
| `SERVE-CLI-BENCH` | `PARTIAL` | .agents/engine-matrix.md:198 | - | - | does not name its missing modes |
| `SERVE-POOLING-ENDPOINTS` | `SPIKE` | .agents/engine-matrix.md:202 | - | - | - |
| `ENG-POOLER-SEQ` | `ACTIVE` | .agents/engine-matrix.md:203 | LANDED | on main: 2191f771 feat(pooling): W2 pooler HEADS composite + W3 pooling RUNNER path (CLAIM-POOLING) — CPU-gated, RED-first | - |
| `ENG-POOLING-RUNNER` | `ACTIVE` | .agents/engine-matrix.md:204 | LANDED | on main: 2191f771 feat(pooling): W2 pooler HEADS composite + W3 pooling RUNNER path (CLAIM-POOLING) — CPU-gated, RED-first | - |
| `ATTN-ROPE-FAMILY` | `READY` | .agents/engine-matrix.md:222 | - | - | - |
| `ATTN-CHUNKED-LOCAL` | `GATING` | .agents/engine-matrix.md:227 | - | - | - |
| `LOAD-SAFETENSORS` | `PARTIAL` | .agents/engine-matrix.md:236 | - | - | explicit via 'no' |
| `LOAD-SAFETENSORS-DIRECT-DENSE` | `GATING` | .agents/engine-matrix.md:237 | - | - | - |
| `LOAD-GGUF` | `PARTIAL` | .agents/engine-matrix.md:240 | - | - | explicit via 'no' |
| `LOAD-CONFIG-SURFACE` | `PARTIAL` | .agents/engine-matrix.md:243 | - | - | does not name its missing modes |
| `MODEL-FACTORY-registry` | `GATING` | .agents/model-matrix.md:152 | - | - | - |
| `MODEL-TEXT-chatglm-chat-glmfor-causal-lm` | `SPIKE` | .agents/model-matrix.md:167 | - | - | - |
| `MODEL-TEXT-commandr-cohere-for-causal-lm` | `BLOCKED` | .agents/model-matrix.md:168 | - | - | - |
| `MODEL-TEXT-llama-llama-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:170 | - | - | explicit via 'PENDING' |
| `MODEL-TEXT-deepseek-v2-deepseek-for-causal-lm` | `SPIKE` | .agents/model-matrix.md:173 | - | - | - |
| `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:174 | LANDED | on main: 2ff7252a docs(mla): campaign W10 — the blocked-row honesty pass; the W-plan is COMPLETE, the block is NOT closeable | - |
| `MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm` | `BLOCKED` | .agents/model-matrix.md:175 | - | - | - |
| `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:176 | LANDED | on main: ee3d5960 spike(model): DeepSeek-V4 (DeepseekV4ForCausalLM) W1/W2 — registry stub + config parse + loader VERIFIED vs real NVFP4 header; HW-fit REVERSAL (156.7 GiB, does not fit GB10) | - |
| `MODEL-TEXT-gemma-gemma-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:187 | - | - | explicit via 'only' |
| `MODEL-TEXT-gemma2-gemma2-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:188 | - | - | explicit via 'PENDING' |
| `MODEL-TEXT-gemma3-gemma3-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:189 | - | - | explicit via 'PENDING' |
| `MODEL-TEXT-gemma4-gemma4-for-causal-lm` | `BLOCKED` | .agents/model-matrix.md:192 | - | - | - |
| `MODEL-TEXT-glm-glm-for-causal-lm` | `SPIKE` | .agents/model-matrix.md:194 | - | - | - |
| `MODEL-TEXT-glm4-glm4-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:195 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `MODEL-TEXT-glm4-moe-glm4-moe-for-causal-lm` | `SPIKE` | .agents/model-matrix.md:196 | - | - | - |
| `MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:197 | LANDED | on main: d85fd04f feat(model): Glm4MoeLiteForCausalLM (GLM-4.7-Flash) G1 — SACRED gate 8/8 vs vLLM 0.25.0; closes the MLA campaign's q_lora + noaux_tc coverage gaps | - |
| `MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm` | `BLOCKED` | .agents/model-matrix.md:198 | - | - | - |
| `MODEL-TEXT-granite-granite-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:203 | - | - | explicit via 'pending' |
| `MODEL-TEXT-internlm2-intern-lm2-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:215 | - | - | explicit via 'gap' |
| `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:219 | LANDED | on main: 2ff7252a docs(mla): campaign W10 — the blocked-row honesty pass; the W-plan is COMPLETE, the block is NOT closeable | - |
| `MODEL-TEXT-laguna-laguna-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:222 | LANDED | on main: bfaea5cb feat(model): Laguna-S-2.1 W4 — fetch UD-Q4_K GGUF + 3 fidelity corrections from the real bytes | - |
| `MODEL-TEXT-minicpm-mini-cpmfor-causal-lm` | `PARTIAL` | .agents/model-matrix.md:227 | - | - | explicit via 'NO' |
| `MODEL-TEXT-minicpm3-mini-cpm3-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:228 | - | - | explicit via 'no' |
| `MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm` | `BLOCKED` | .agents/model-matrix.md:229 | - | - | - |
| `MODEL-TEXT-mistral-mistral-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:231 | - | - | explicit via 'NO' |
| `MODEL-TEXT-olmo2-olmo2-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:241 | - | - | explicit via 'NO' |
| `MODEL-TEXT-opt-optfor-causal-lm` | `PARTIAL` | .agents/model-matrix.md:244 | - | - | explicit via 'PENDING' |
| `MODEL-TEXT-phi-phi-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:252 | - | - | explicit via 'NO' |
| `MODEL-TEXT-phi3-phi3-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:253 | - | - | explicit via 'gap' |
| `MODEL-TEXT-qwen3-qwen3-for-causal-lm` | `ACTIVE` | .agents/model-matrix.md:259 | LANDED | on main: 6eee4437 fix(quant): Qwen3-32B-NVFP4A16 W4A16 — teacher-forcing PROVES the 4/6 is bf16 near-tie drift, not our defect; gate closes 6/6 | - |
| `MODEL-TEXT-qwen3-moe-qwen3-moe-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:260 | - | - | explicit via 'only' |
| `MODEL-TEXT-stablelm-stablelm-for-causal-lm` | `PARTIAL` | .agents/model-matrix.md:267 | - | - | does not name its missing modes |
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | `ACTIVE` | .agents/model-matrix.md:380 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `MODEL-MM-gemma4-unified-gemma4-unified-for-conditional-generation` | `SPIKE` | .agents/model-matrix.md:381 | - | - | - |
| `MODEL-MM-kimi-k3-kimi-k3-for-conditional-generation` | `SPIKE` | .agents/model-matrix.md:404 | - | - | - |
| `MODEL-MM-qwen3-vl-qwen3-vlfor-conditional-generation` | `PARTIAL` | .agents/model-matrix.md:447 | - | - | explicit via 'gap' |
| `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` | `PARTIAL` | .agents/model-matrix.md:449 | - | - | explicit via 'only' |
| `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` | `PARTIAL` | .agents/model-matrix.md:450 | - | - | explicit via 'only' |
| `MODEL-MM-voxtral-voxtral-for-conditional-generation` | `ACTIVE` | .agents/model-matrix.md:458 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp` | `ACTIVE` | .agents/model-matrix.md:487 | LANDED | on main: 28b5e866 feat(model): DeepSeek-V4 native MTP self-speculative draft head — W1 wiring (loader + draft forward + lossless gate) | - |
| `QUANT-GGUF-COMPUTE` | `READY` | .agents/quantization-matrix.md:33 | - | - | - |
| `QUANT-GGUF-F32` | `PARTIAL` | .agents/quantization-matrix.md:57 | - | - | does not name its missing modes |
| `QUANT-GGUF-F16` | `PARTIAL` | .agents/quantization-matrix.md:58 | - | - | explicit via 'gap' |
| `QUANT-GGUF-Q4_0` | `PARTIAL` | .agents/quantization-matrix.md:59 | - | - | explicit via 'no' |
| `QUANT-GGUF-Q8_0` | `PARTIAL` | .agents/quantization-matrix.md:63 | - | - | explicit via 'no' |
| `QUANT-GGUF-Q2_K` | `ACTIVE` | .agents/quantization-matrix.md:64 | LANDED | on main: d0bc0f41 feat(gguf): IQ2_XXS + Q2_K dequant — DeepSeek-V4-Flash single-Spark GGUF quant-path (W1) | - |
| `QUANT-GGUF-Q3_K` | `PARTIAL` | .agents/quantization-matrix.md:65 | - | - | explicit via 'no' |
| `QUANT-GGUF-Q4_K` | `PARTIAL` | .agents/quantization-matrix.md:66 | - | - | explicit via 'no' |
| `QUANT-GGUF-Q5_K` | `PARTIAL` | .agents/quantization-matrix.md:67 | - | - | explicit via 'no' |
| `QUANT-GGUF-Q6_K` | `PARTIAL` | .agents/quantization-matrix.md:68 | - | - | explicit via 'no' |
| `QUANT-GGUF-IQ2_XXS` | `ACTIVE` | .agents/quantization-matrix.md:69 | LANDED | on main: d0bc0f41 feat(gguf): IQ2_XXS + Q2_K dequant — DeepSeek-V4-Flash single-Spark GGUF quant-path (W1) | - |
| `QUANT-GGUF-IQ3_XXS` | `ACTIVE` | .agents/quantization-matrix.md:71 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `QUANT-GGUF-NVFP4` | `PARTIAL` | .agents/quantization-matrix.md:82 | - | - | explicit via 'no' |
| `QUANT-NVFP4-CT-W4A16` | `ACTIVE` | .agents/quantization-matrix.md:122 | LANDED | on main: 80d1da09 feat(quant): Qwen3-32B-NVFP4A16 (compressed-tensors W4A16) — quant-scheme additivity; strict gate GATING 4/6 | - |
| `QUANT-FP8-GENERIC` | `PARTIAL` | .agents/quantization-matrix.md:125 | - | - | explicit via 'only' |
| `QUANT-MIXED-MODELOPT` | `PARTIAL` | .agents/quantization-matrix.md:129 | - | - | does not name its missing modes |
| `QUANT-KV-FP8` | `PARTIAL` | .agents/quantization-matrix.md:158 | - | - | does not name its missing modes |
| `KERNEL-ACCEL-PROVIDER-SELECT` | `ACTIVE` | .agents/kernel-matrix.md:115 | LANDED | on main: 3a2d05d8 feat(backend): vt::OpProvider acceleration seam + MLX GEMM provider on Metal | - |
| `KERNEL-GEMM-CPU-ELEM` | `ACTIVE` | .agents/kernel-matrix.md:126 | LANDED | on main: 428e20f0 docs(roadmap): correct C4 punch-list — GGUF CPU decode is at parity, not a 10x-open blocker | - |
| `KERNEL-QUANT-CIQ-IQUANT` | `SPIKE` | .agents/kernel-matrix.md:127 | - | - | - |
| `KERNEL-QUANT-CIQ-GEMM-CUDA` | `ACTIVE` | .agents/kernel-matrix.md:128 | LANDED | on main: 3fb3149f feat(cuda): keep-quant GGUF k-quant GEMM (kCUDA kMatmulBTQuant) — DeepSeek-V4 experts on the GPU | - |
| `KERNEL-EW-NORM-QUANT` | `PARTIAL` | .agents/kernel-matrix.md:130 | - | - | explicit via 'only' |
| `KERNEL-ROPE-QKNORM` | `PARTIAL` | .agents/kernel-matrix.md:131 | - | - | does not name its missing modes |
| `KERNEL-ATTN-MLA-SPARSE` | `PARTIAL` | .agents/kernel-matrix.md:139 | - | - | explicit via 'only' |
| `KERNEL-ATTN-DSA-SPARSE-INDEX` | `SPIKE` | .agents/kernel-matrix.md:142 | - | - | - |
| `KERNEL-ATTN-DSA-COMPRESSOR` | `SPIKE` | .agents/kernel-matrix.md:143 | - | - | - |
| `KERNEL-MHC-SINKHORN` | `SPIKE` | .agents/kernel-matrix.md:144 | - | - | - |
| `KERNEL-MOE-SQRTSOFTPLUS-HASH` | `SPIKE` | .agents/kernel-matrix.md:145 | - | - | - |
| `KERNEL-DSV4-W7-DEVICE` | `SPIKE` | .agents/kernel-matrix.md:146 | - | - | - |
| `KERNEL-KDA-DELTA` | `SPIKE` | .agents/kernel-matrix.md:147 | - | - | - |
| `KERNEL-ATTN-DENSE-FLASH` | `ACTIVE` | .agents/kernel-matrix.md:148 | LANDED | on main: e89d51d8 feat(mm-speed): Qwen VISION-FORWARD — §14 flash kernel extended to the tower (byte-exact); ATTRIBUTION REFUTES the assumed lever, tower already BEATS vLLM (CLAIM-MM-SPEED-QWEN-IMAGE) | - |
| `KERNEL-MOE-UNQUANTIZED` | `PARTIAL` | .agents/kernel-matrix.md:150 | - | - | does not name its missing modes |
| `KERNEL-MOE-QUANTIZED` | `PARTIAL` | .agents/kernel-matrix.md:151 | - | - | does not name its missing modes |
| `KERNEL-GDN-AOT-BF16` | `ACTIVE` | .agents/kernel-matrix.md:154 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `KERNEL-GDN-SCRATCH` | `ACTIVE` | .agents/kernel-matrix.md:155 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `BACKEND-CUDA-SM060` | `SPIKE` | .agents/backend-matrix.md:164 | - | - | - |
| `BACKEND-CUDA-SM061` | `SPIKE` | .agents/backend-matrix.md:165 | - | - | - |
| `BACKEND-CUDA-SM070` | `SPIKE` | .agents/backend-matrix.md:166 | - | - | - |
| `BACKEND-CUDA-SM075` | `SPIKE` | .agents/backend-matrix.md:167 | - | - | - |
| `BACKEND-CUDA-SM080` | `SPIKE` | .agents/backend-matrix.md:168 | - | - | - |
| `BACKEND-CUDA-SM086` | `SPIKE` | .agents/backend-matrix.md:169 | - | - | - |
| `BACKEND-CUDA-SM087` | `ACTIVE` | .agents/backend-matrix.md:170 | LANDED | on main: aca8d7d7 feat(cuda-arch): Orin sm_87 RUNTIME gate — BACKEND-CUDA-SM087 portable bf16 SYNC path RUNTIME-VERIFIED (13/16 strict vs vLLM, 2nd non-GB10 proof) | - |
| `BACKEND-CUDA-SM089` | `SPIKE` | .agents/backend-matrix.md:171 | - | - | - |
| `BACKEND-CUDA-SM090` | `SPIKE` | .agents/backend-matrix.md:172 | - | - | - |
| `BACKEND-CUDA-SM100` | `SPIKE` | .agents/backend-matrix.md:173 | - | - | - |
| `BACKEND-CUDA-SM103` | `SPIKE` | .agents/backend-matrix.md:175 | - | - | - |
| `BACKEND-CUDA-SM110` | `ACTIVE` | .agents/backend-matrix.md:176 | LANDED | on main: b34f9909 feat(cuda): sm_110 (Jetson Thor) portable path RUNTIME-VERIFIED — first non-GB10 runtime proof | - |
| `BACKEND-CUDA-SM120` | `ACTIVE` | .agents/backend-matrix.md:177 | LANDED | on main: 88a0b869 feat(backend): sm_120a (consumer Blackwell) — BUILD-supported CUDA target, first arch through the additive seams | - |
| `BACKEND-CUDA-SM121` | `PARTIAL` | .agents/backend-matrix.md:178 | - | - | explicit via 'only' |
| `BACKEND-CUDA-ARCH-ADDITIVITY` | `ACTIVE` | .agents/backend-matrix.md:187 | LANDED | on main: 8a379182 feat(cuda-arch): cross-family BUILD-SUPPORTED fan-out (W10) — sm_80/86/87/89, sm_100a/103a, sm_110 portable-only; sm_70/75/101a scoped | - |
| `BACKEND-CUDA-COMP-CORE` | `PARTIAL` | .agents/backend-matrix.md:188 | - | - | explicit via 'only' |
| `BACKEND-CUDA-COMP-MARLIN` | `PARTIAL` | .agents/backend-matrix.md:190 | - | - | does not name its missing modes |
| `BACKEND-CUDA-COMP-MACHETE` | `SPIKE` | .agents/backend-matrix.md:191 | - | - | - |
| `BACKEND-CUDA-COMP-DSV3` | `SPIKE` | .agents/backend-matrix.md:192 | - | - | - |
| `BACKEND-CUDA-COMP-ALLSPARK` | `SPIKE` | .agents/backend-matrix.md:193 | - | - | - |
| `BACKEND-CUDA-COMP-SCALEDMM-C3X` | `PARTIAL` | .agents/backend-matrix.md:194 | - | - | does not name its missing modes |
| `BACKEND-CUDA-COMP-SCALEDMM-C2X` | `SPIKE` | .agents/backend-matrix.md:195 | - | - | - |
| `BACKEND-CUDA-COMP-MOE-CUTLASS` | `SPIKE` | .agents/backend-matrix.md:196 | - | - | - |
| `BACKEND-CUDA-COMP-FP4` | `PARTIAL` | .agents/backend-matrix.md:197 | - | - | does not name its missing modes |
| `BACKEND-CUDA-COMP-W4A8` | `SPIKE` | .agents/backend-matrix.md:198 | - | - | - |
| `BACKEND-CUDA-COMP-MLA` | `SPIKE` | .agents/backend-matrix.md:199 | - | - | - |
| `BACKEND-CUDA-COMP-FLASHMLA` | `SPIKE` | .agents/backend-matrix.md:212 | - | - | - |
| `BACKEND-CUDA-COMP-DEEPGEMM` | `SPIKE` | .agents/backend-matrix.md:213 | - | - | - |
| `BACKEND-CUDA-COMP-FA` | `PARTIAL` | .agents/backend-matrix.md:215 | - | - | explicit via 'no' |
| `BACKEND-CUDA-COMP-JIT` | `PARTIAL` | .agents/backend-matrix.md:216 | - | - | explicit via 'only' |
| `BACKEND-CPU` | `PARTIAL` | .agents/backend-matrix.md:226 | - | - | explicit via 'no' |
| `BACKEND-XPU` | `SPIKE` | .agents/backend-matrix.md:229 | - | - | - |
| `BACKEND-ACCEL-PROVIDER` | `ACTIVE` | .agents/backend-matrix.md:231 | LANDED | on main: 3a2d05d8 feat(backend): vt::OpProvider acceleration seam + MLX GEMM provider on Metal | - |
| `BACKEND-METAL-MLX` | `ACTIVE` | .agents/backend-matrix.md:232 | LANDED | on main: c351dff2 feat(model): Qwen3-dense (Qwen3ForCausalLM) on Metal — SECOND non-CUDA model, SACRED gate 16/16 + first ours-vs-MLX benchmark (INDICATIVE) | - |
| `BACKEND-VULKAN` | `ACTIVE` | .agents/backend-matrix.md:233 | LANDED | on main: 1cb5f643 feat(backend): Vulkan W0/V1 — vt::Backend/Platform skeleton with committed SPIR-V compute (BACKEND-VULKAN SPIKE->ACTIVE) | - |
| `BACKEND-GATE-CUDA-VLLM` | `PARTIAL` | .agents/backend-matrix.md:244 | - | - | explicit via 'blocked' |
| `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | `GATING` | .agents/backend-matrix.md:245 | - | - | - |
| `BACKEND-GATE-CUDA-SGLANG` | `BLOCKED` | .agents/backend-matrix.md:246 | - | - | - |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | `READY` | .agents/backend-matrix.md:247 | - | - | - |
| `BACKEND-GATE-METAL-MLXLM` | `ACTIVE` | .agents/backend-matrix.md:254 | LANDED | on main: 41d7f8d7 test(metal): pin the S5 reference-tier contract, and record the MLX provider A/B | - |
| `BACKEND-DISTRIBUTED-COMM` | `ACTIVE` | .agents/backend-matrix.md:272 | LANDED | on main: 9b516ab2 feat(scale-out): W1 vt::Communicator + CPU collective gate — BACKEND-DISTRIBUTED-COMM SPIKE→ACTIVE | - |
| `BACKEND-DISTRIBUTED-TP` | `ACTIVE` | .agents/backend-matrix.md:273 | ABANDONED | no branch, no commit on main mentioning the row ID | - |
| `BACKEND-DISTRIBUTED-PP` | `SPIKE` | .agents/backend-matrix.md:274 | - | - | - |
| `BACKEND-DISTRIBUTED-DP` | `SPIKE` | .agents/backend-matrix.md:275 | - | - | - |
| `BACKEND-DISTRIBUTED-EP` | `SPIKE` | .agents/backend-matrix.md:276 | - | - | - |
| `BACKEND-DISTRIBUTED-SP` | `SPIKE` | .agents/backend-matrix.md:277 | - | - | - |
| `BACKEND-DISTRIBUTED-MULTINODE-SPARK` | `SPIKE` | .agents/backend-matrix.md:278 | - | - | - |
| `BACKEND-DISTRIBUTED-MLX-RING` | `SPIKE` | .agents/backend-matrix.md:279 | - | - | - |
| `MODEL-GATE-QWEN35` | `PARTIAL` | .agents/feature-matrix.md:149 | - | - | explicit via 'pending' |
| `MODEL-FACTORY` | `PARTIAL` | .agents/feature-matrix.md:150 | - | - | explicit via 'no' |
| `MODEL-TEXT` | `PARTIAL` | .agents/feature-matrix.md:151 | - | - | explicit via 'blocked' |
| `MODEL-MM` | `PARTIAL` | .agents/feature-matrix.md:153 | - | - | explicit via 'no' |
| `MODEL-SPEC` | `ACTIVE` | .agents/feature-matrix.md:154 | LANDED | on main: d45c8cda docs(reconcile): fix behind/stale drift across records to match code state (records-only) | - |
| `QUANT-GGUF` | `PARTIAL` | .agents/feature-matrix.md:167 | - | - | explicit via 'only' |
| `QUANT-VLLM-BREADTH` | `PARTIAL` | .agents/feature-matrix.md:168 | - | - | does not name its missing modes |
| `BACKEND-CUDA-SM121` | `PARTIAL` | .agents/feature-matrix.md:275 | - | - | does not name its missing modes |
| `BACKEND-CUDA-OTHER` | `ACTIVE` | .agents/feature-matrix.md:276 | LANDED | on main: d45c8cda docs(reconcile): fix behind/stale drift across records to match code state (records-only) | - |
| `BACKEND-CPU` | `PARTIAL` | .agents/feature-matrix.md:277 | - | - | does not name its missing modes |
| `BACKEND-MLX` | `ACTIVE` | .agents/feature-matrix.md:279 | LANDED | on main: d45c8cda docs(reconcile): fix behind/stale drift across records to match code state (records-only) | - |

## Proposed corrections

### The legality rule

`scripts/check-agent-record.py` enforces state contracts, so an illegal target
breaks the build:

- `READY`, `ACTIVE`, `GATING`, `DONE`, `BLOCKED` (`READY_STATES`) require a real
  `.agents/specs/<slug>.md` link that resolves, names the exact token `` `<ID>` ``,
  and carries the structured spec sections.
- `PARTIAL`, `ANCHOR-BACKFILL`, `GATING`, `DONE`, `BUILD-ONLY`, `UNTRACED`
  (`EVIDENCED_STATES`) require resolving code AND test/evidence anchors.
- `SPIKE` and `ACTIVE` require a `CLAIM-*` owner that claims the row in
  `coordination.md`; conversely `coordination.md` may only reference rows in state
  `SPIKE` or `ACTIVE` (`check_row_contracts`, "references {id} in state X, not
  SPIKE/ACTIVE").

Therefore an abandoned `ACTIVE` row goes to `READY` if it has a real spec link and
to `INVENTORIED` otherwise. It may never simply be blanked.

### Per-row legality determination

Every one of the ten was checked with the checker's own `local_spec_paths` +
`check_spec`. **All ten resolve to a real spec that names their exact token and
passes the structured-section requirement, so all ten are legal at `READY` and
none needs `INVENTORIED`:**

| Row | Spec cell | Resolves | Names `` `<ID>` `` | Legal target |
|---|---|---|---|---|
| `BACKEND-DISTRIBUTED-TP` | `specs/scale-out-distributed.md` | yes | yes | `READY` |
| `ENG-MM-INPUT-PIPELINE` | `specs/multimodal-track.md` §3 (M0/M1) | yes | yes | `READY` |
| `ENG-MM-VIDEO-FORWARD` | `specs/multimodal-track.md` §M3 (M3c) | yes | yes | `READY` |
| `ENG-MM-AUDIO-ENCODER` | `specs/audio-track.md` §0b (A2) | yes | yes | `READY` |
| `KERNEL-GDN-AOT-BF16` | `specs/kernel-family-inventory.md` | yes | yes | `READY` |
| `KERNEL-GDN-SCRATCH` | `specs/kernel-family-inventory.md` | yes | yes | `READY` |
| `MODEL-TEXT-glm4-glm4-for-causal-lm` | `specs/glm-dsa-latest-deepseek.md` | yes | yes | `READY` |
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | `specs/gemma4-multimodal.md` §G1b (+ `specs/sweep-gemma.md`) | yes | yes | `READY` |
| `MODEL-MM-voxtral-voxtral-for-conditional-generation` | `specs/audio-track.md` §0c/§1 (A3) | yes | yes | `READY` |
| `QUANT-GGUF-IQ3_XXS` | `specs/cuda-keepquant-gemm.md`; `specs/gguf-iquant-dsv4.md` | yes | yes | `READY` |

For `MODEL-MM-gemma4-...` and `QUANT-GGUF-IQ3_XXS` only ONE of the two linked
specs names the token (`gemma4-multimodal.md` and `gguf-iquant-dsv4.md`
respectively); `check_spec` is satisfied by any one match, so both are legal.

### The coupled obligation nobody can skip

`READY` is not in `{SPIKE, ACTIVE}`, so leaving the row inside an active claim
turns the transition RED. **Every one of the ten is referenced by at least one
active claim**, and several by many, so each correction must retire or amend the
claim in `coordination.md` in the SAME change:

| Row | Claims that must be amended |
|---|---|
| `BACKEND-DISTRIBUTED-TP` | `CLAIM-PARALLELISM-MODES-SPIKE`, `CLAIM-SCALE-OUT-SPIKE`, `CLAIM-SCALE-OUT-W2` |
| `ENG-MM-AUDIO-ENCODER` | `CLAIM-AUDIO-ENCODER` |
| `ENG-MM-INPUT-PIPELINE` | `CLAIM-MULTIMODAL-M1` |
| `ENG-MM-VIDEO-FORWARD` | `CLAIM-MULTIMODAL-M3C`, `CLAIM-MULTIMODAL-TOWER-FIDELITY` |
| `KERNEL-GDN-AOT-BF16` | `CLAIM-PR3`, `CLAIM-TRITON-AOT-PER-ARCH` |
| `KERNEL-GDN-SCRATCH` | `CLAIM-PR3` |
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | `CLAIM-GEMMA4-G1`, `CLAIM-GEMMA4-G1B`, `CLAIM-GEMMA4-G2`, `CLAIM-GEMMA4-G2-IMPL`, `CLAIM-GEMMA4-G3`, `CLAIM-GEMMA4-MM-E2E`, `CLAIM-GEMMA4-MULTIMODAL`, `CLAIM-MULTIMODAL-TRACK` |
| `MODEL-MM-voxtral-voxtral-for-conditional-generation` | `CLAIM-AUDIO-E2E` |
| `MODEL-TEXT-glm4-glm4-for-causal-lm` | `CLAIM-GLM-DSA-LATEST-DEEPSEEK` |
| `QUANT-GGUF-IQ3_XXS` | `CLAIM-DEEPSEEK-V4-W8` |

`MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` alone is claimed by EIGHT
live claims (`CLAIM-GEMMA4-G1`, `-G1B`, `-G2`, `-G2-IMPL`, `-G3`, `-MM-E2E`,
`-MULTIMODAL`, and `CLAIM-MULTIMODAL-TRACK`) — the same rot in a second surface.
`BACKEND-DISTRIBUTED-TP` is claimed by three; `coordination.md` carries 110 active
claim entries in total.

Note the second half of the same contract: `check_row_contracts` also fails an
active claim with NO stable row IDs. **These claims reference nothing BUT the
abandoned rows, so removing the rows empties them — they must be RETIRED outright,
not merely emptied:**

- `CLAIM-AUDIO-ENCODER` (claims only `ENG-MM-AUDIO-ENCODER`)
- `CLAIM-GEMMA4-G1` (claims only `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`)
- `CLAIM-GEMMA4-G2` (claims only `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`)
- `CLAIM-GEMMA4-G2-IMPL` (claims only `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`)
- `CLAIM-GEMMA4-G3` (claims only `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`)
- `CLAIM-GEMMA4-MM-E2E` (claims only `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`)
- `CLAIM-MULTIMODAL-M1` (claims only `ENG-MM-INPUT-PIPELINE`)
- `CLAIM-MULTIMODAL-M3C` (claims only `ENG-MM-VIDEO-FORWARD`)
- `CLAIM-MULTIMODAL-TOWER-FIDELITY` (claims only `ENG-MM-VIDEO-FORWARD`)
- `CLAIM-PR3` (claims only `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`)
- `CLAIM-TRITON-AOT-PER-ARCH` (claims only `KERNEL-GDN-AOT-BF16`)

### Rows proposed for correction: 10. Rows needing human inspection: 10.

These are the same ten, and the distinction the brief demands is between two
different questions:

- **What the tool settles (no human needed):** the `ACTIVE` claim on all ten is
  unsupportable. Nothing in Git corroborates active work. Vacating `ACTIVE` is
  justified by the evidence in this document.
- **What the tool CANNOT settle (human required, all ten):** which state each row
  should land in. `READY` is the legality FLOOR from the rule above, not a
  semantic verdict — and for these rows it is probably the WRONG semantic answer,
  because each row's own anchors assert landed code and passing gates. Applying
  `READY` blindly would trade one records lie ("actively being worked") for
  another ("spec'd, not started").

**Nothing in this audit is sufficient to pick between `READY`, `PARTIAL`,
`GATING` and `DONE` for any of the ten.** The correcting change must have a human
read each row's code/test anchors and the linked spec, then pick the state whose
contract that evidence actually satisfies:

| Row | In-row evidence a human must adjudicate | Contract if promoted instead of `READY` |
|---|---|---|
| `BACKEND-DISTRIBUTED-TP` | "LANDED tensor_parallel.h"; CPU multi-rank TP gate 60/60 RED-verified | `PARTIAL` fits (CPU only, no GPU/multi-node); needs named missing modes |
| `ENG-MM-INPUT-PIPELINE` | processor-parity 23/23 BIT-identical vs the M0 oracle fixture | `PARTIAL`/`GATING` need resolving code AND test anchors |
| `ENG-MM-VIDEO-FORWARD` | "M3c BUILT + UNIT-GATED", video-processor 41/41, pixel_values_videos bit-exact | `PARTIAL`/`GATING` need resolving code AND test anchors |
| `ENG-MM-AUDIO-ENCODER` | "A2 encoder-tower fidelity gate PASS 203/203" | `PARTIAL`/`GATING` need resolving code AND test anchors |
| `KERNEL-GDN-AOT-BF16` | "AOT/safety/native gates green", 1.007989x, 16/20 timing, 2/4 memory — explicitly incomplete | `PARTIAL` fits the evidence; needs anchors + named missing modes |
| `KERNEL-GDN-SCRATCH` | stream-owned pool + poison/reuse/growth assertions | `PARTIAL`/`GATING` need resolving code AND test anchors |
| `MODEL-TEXT-glm4-glm4-for-causal-lm` | `test_glm4_paged_engine` 16/16 on dgx; rope 6692/6692 | `DONE` additionally needs an exact parity-ledger link and a commit-SHA owner |
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | "G1b LANDED — TEXT PATH STRICT 32/32 TOKEN-EXACT"; MM path open | `PARTIAL` needs anchors AND must name what is missing (see ➃) |
| `MODEL-MM-voxtral-voxtral-for-conditional-generation` | "audio→text e2e gate PASS 14/14" (near-tie-robust, GPU dgx-only) | `DONE` additionally needs an exact parity-ledger link and a commit-SHA owner |
| `QUANT-GGUF-IQ3_XXS` | reader trait + codebook anchors, "ADDED W8" | `PARTIAL` needs anchors AND must name what is missing |

### `PARTIAL` rows

No `PARTIAL` state change is proposed. The correction owed by the 30-odd rows in
➃ is EDITORIAL — name the missing mode in the row text — and it cannot be
mechanised, because only a reader of the row knows what is absent. Proposing a
state change for them would be inventing confidence.

## Duplicate live IDs

`BACKEND-CUDA-SM121` and `BACKEND-CPU` are each `PARTIAL` in TWO matrices, so the
188 live rows carry only 186 unique IDs:

| ID | Occurrence A | Occurrence B |
|---|---|---|
| `BACKEND-CUDA-SM121` | backend-matrix.md:178 | feature-matrix.md:275 |
| `BACKEND-CPU` | backend-matrix.md:226 | feature-matrix.md:277 |

`check-agent-record.py` never caught this because its duplicate check only walks
`MATRIX_PATHS`, which omits `feature-matrix.md`.

### Decision: `backend-matrix.md` OWNS both IDs

Reasons, in order of weight:

1. **Only the backend rows carry the contract.** The `backend-matrix.md` rows have
   the full semantic column set (upstream / code / tests / spec / owner) with
   resolving anchors. The `feature-matrix.md` entries are five-column roll-ups
   whose evidence column is one sentence of prose.
2. **Only the backend matrix is gated.** `backend-matrix.md` is in `MATRIX_PATHS`,
   so CI holds it to the row contract. `feature-matrix.md` is in `REQUIRED` (it
   must exist) but is never parsed for claim rows, so a state written there is
   enforced by nothing.
3. **The feature rows already declare themselves pointers.** Both link out — their
   spec cells are markdown links to `backend-matrix.md` (one to the
   `#cuda-target-rows` anchor). They were authored as references; they merely kept
   the stable ID in the first cell, which is what makes them parse as claims.

### What the feature-matrix entries become: non-claimable references

Change the FIRST CELL of `feature-matrix.md:275` and `:277` from the bare stable ID
to a markdown link naming the owner — the link TEXT keeps the ID for readers, the
cell stops being a bare ID for the parser.
`parse_claim_rows` skips a row whose first cell fails `ID_RE.fullmatch` (line 439)
**silently and without an error**, so the roll-up view survives verbatim, the
duplicate disappears, and the backfill mints exactly one issue per item.

Two alternatives were considered and rejected:

- **Blank the state cell — ILLEGAL.** `parse_claim_rows` reports "must have exactly
  one canonical state" for a row whose first cell is still an ID, and
  `audit-live-rows.py` aborts the whole census on any parse error.
- **Delete the two feature rows — legal but lossy.** `feature-matrix.md` is the
  feature-level roll-up; deleting the rows removes CPU and sm121 from the roll-up
  to fix a keying problem. The pointer keeps both.

Widening `check-agent-record.py`'s duplicate check to `feature-matrix.md` is NOT
proposed here: that file has never been held to the row contract, and turning the
gate on it would fail for reasons unrelated to this audit. Note this leaves the
duplicate check unable to see a future recurrence — `scripts/audit-live-rows.py`
is the only thing that would, so it should keep being run.

## Rows left alone

### `IN-FLIGHT` rows: none

**Zero of the 54 `ACTIVE` rows classify `IN-FLIGHT`, so this section is empty by
measurement, not by omission.** No live row has a `row/<ID>` branch at all, so no
row can show unmerged commits. The audit considered all 54 and kept none on that
basis.

### The 44 `LANDED` rows: kept `ACTIVE`, untouched

Kept because a commit mention is not grounds to move a row in either direction —
not up to `DONE` (it proves nothing finished) and not down (it is real evidence
worth reading). They stay `ACTIVE` until a human reads them. Ordered as reported:

| Row | Location | Evidence commit | Commit touched code? |
|---|---|---|---|
| `ENG-RUNNER-MODELSHAPE` | .agents/engine-matrix.md:71 | `5ab3f111` | yes |
| `ENG-MM-VISION-TOWER` | .agents/engine-matrix.md:73 | `d796187a` | yes |
| `ENG-MM-TEXT-BACKBONE` | .agents/engine-matrix.md:74 | `2a8ff336` | yes |
| `ENG-MM-QWEN36-VL-FORWARD` | .agents/engine-matrix.md:75 | `e89d51d8` | yes |
| `ENG-MM-AUDIO-PIPELINE` | .agents/engine-matrix.md:77 | `adcac8e6` | yes |
| `ENG-MM-AUDIO-E2E` | .agents/engine-matrix.md:79 | `9e34a19c` | yes |
| `KV-EVENTS` | .agents/engine-matrix.md:105 | `d796187a` | yes |
| `SAMPLE-BEAM` | .agents/engine-matrix.md:136 | `0151314f` | yes |
| `SAMPLE-N` | .agents/engine-matrix.md:142 | `aed4718e` | yes |
| `SAMPLE-BEST-OF` | .agents/engine-matrix.md:143 | `aed4718e` | yes |
| `TOOLS-XGRAMMAR` | .agents/engine-matrix.md:150 | `d796187a` | yes |
| `TOOLS-STREAMING-PARSER` | .agents/engine-matrix.md:154 | `05237562` | yes |
| `SPEC-REJECTION` | .agents/engine-matrix.md:164 | `dfa610b2` | yes |
| `SPEC-GDN-SEGMENTS` | .agents/engine-matrix.md:165 | `3ae5cfe0` | yes |
| `SPEC-NGRAM` | .agents/engine-matrix.md:169 | `d796187a` | yes |
| `SPEC-DRAFT-MODEL` | .agents/engine-matrix.md:180 | `dfa610b2` | yes |
| `ENG-POOLER-SEQ` | .agents/engine-matrix.md:203 | `2191f771` | yes |
| `ENG-POOLING-RUNNER` | .agents/engine-matrix.md:204 | `2191f771` | yes |
| `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` | .agents/model-matrix.md:174 | `2ff7252a` | **no — records/docs only** |
| `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` | .agents/model-matrix.md:176 | `ee3d5960` | yes |
| `MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm` | .agents/model-matrix.md:197 | `d85fd04f` | yes |
| `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` | .agents/model-matrix.md:219 | `2ff7252a` | **no — records/docs only** |
| `MODEL-TEXT-laguna-laguna-for-causal-lm` | .agents/model-matrix.md:222 | `bfaea5cb` | yes |
| `MODEL-TEXT-qwen3-qwen3-for-causal-lm` | .agents/model-matrix.md:259 | `6eee4437` | yes |
| `MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp` | .agents/model-matrix.md:487 | `28b5e866` | yes |
| `QUANT-GGUF-Q2_K` | .agents/quantization-matrix.md:64 | `d0bc0f41` | yes |
| `QUANT-GGUF-IQ2_XXS` | .agents/quantization-matrix.md:69 | `d0bc0f41` | yes |
| `QUANT-NVFP4-CT-W4A16` | .agents/quantization-matrix.md:122 | `80d1da09` | yes |
| `KERNEL-ACCEL-PROVIDER-SELECT` | .agents/kernel-matrix.md:115 | `3a2d05d8` | yes |
| `KERNEL-GEMM-CPU-ELEM` | .agents/kernel-matrix.md:126 | `428e20f0` | **no — records/docs only** |
| `KERNEL-QUANT-CIQ-GEMM-CUDA` | .agents/kernel-matrix.md:128 | `3fb3149f` | yes |
| `KERNEL-ATTN-DENSE-FLASH` | .agents/kernel-matrix.md:148 | `e89d51d8` | yes |
| `BACKEND-CUDA-SM087` | .agents/backend-matrix.md:170 | `aca8d7d7` | **no — records/docs only** |
| `BACKEND-CUDA-SM110` | .agents/backend-matrix.md:176 | `b34f9909` | **no — records/docs only** |
| `BACKEND-CUDA-SM120` | .agents/backend-matrix.md:177 | `88a0b869` | yes |
| `BACKEND-CUDA-ARCH-ADDITIVITY` | .agents/backend-matrix.md:187 | `8a379182` | yes |
| `BACKEND-ACCEL-PROVIDER` | .agents/backend-matrix.md:231 | `3a2d05d8` | yes |
| `BACKEND-METAL-MLX` | .agents/backend-matrix.md:232 | `c351dff2` | yes |
| `BACKEND-VULKAN` | .agents/backend-matrix.md:233 | `1cb5f643` | yes |
| `BACKEND-GATE-METAL-MLXLM` | .agents/backend-matrix.md:254 | `41d7f8d7` | yes |
| `BACKEND-DISTRIBUTED-COMM` | .agents/backend-matrix.md:272 | `9b516ab2` | yes |
| `MODEL-SPEC` | .agents/feature-matrix.md:154 | `d45c8cda` | **no — records/docs only** |
| `BACKEND-CUDA-OTHER` | .agents/feature-matrix.md:276 | `d45c8cda` | **no — records/docs only** |
| `BACKEND-MLX` | .agents/feature-matrix.md:279 | `d45c8cda` | **no — records/docs only** |

### The other live states: out of scope for a verdict

`SPIKE` 43, `GATING` 10, `BLOCKED` 7, `READY` 6 — 66 rows. The
classifier runs on `ACTIVE` only, and the flag on `PARTIAL` only, so these rows
appear in the census (they are live and they count toward the 188) but carry no
verdict. Nothing is proposed for them.

## Risks/decisions

| # | Question the tool could NOT decide | Human call made here |
|---|---|---|
| 1 | Does a commit mentioning a row ID mean the row is finished? | **No.** `LANDED` is renamed in this document to "has evidence worth reading". No `DONE` proposal anywhere. |
| 2 | Do the 8 records-only-backed `LANDED` rows have any work behind them? | **Unknown, and the audit cannot say.** A `.agents/`-only diff proves a record was edited. Flagged individually in ➁. |
| 3 | What state should each abandoned `ACTIVE` row actually land in? | **Deferred to a human, all ten.** `READY` is the legality floor and is recorded as such, NOT as a recommendation. Each row's own anchors assert passing gates, so `READY` is likely wrong semantically. |
| 4 | Is `ABANDONED` a true statement about the work? | **No — it is a statement about Git.** It means no branch and no ID-naming commit. In all ten cases the row asserts landed work, so the real defect is that commits never named the stable ID. |
| 5 | Which matrix owns a duplicated ID? | **`backend-matrix.md`**, for both. Reasons above. The feature-matrix entries become non-claimable pointers. |
| 6 | Is the `PARTIAL` flag trustworthy as a count? | **No — it UNDER-flags.** 20 flagged, but 11 more pass only on a bare `no`/`gap`, 10 of those on prose asserting goodness. The real count is ~30 of 68. This is why the report names the marker. |
| 7 | Should the correcting change edit `coordination.md` too? | **Yes, unavoidably.** All ten abandoned rows sit inside active claims, and `check_row_contracts` fails a claim that references a non-`SPIKE`/`ACTIVE` row. The matrix edit and the claim edit are one atomic change. |
| 8 | Should `check-agent-record.py`'s duplicate check be widened? | **Not here.** Widening a repo-wide gate onto a file that has never met the row contract is a separate decision with its own fallout; never weaken (or blindly widen) a checker to service an audit. |

### Method risks a reader should know

- **The classifier's `LANDED` rule is the weakest link and it fires 44 of 54
  times.** Its stronger rules (unmerged branch, merged branch) never fire in this
  repository because the `row/<ID>` branch convention is not in use. Any future
  reading of these verdicts must not treat them as equivalent.
- **`main_commits` caps at 20 and reports `commits[0]`,** the most recent mention.
  A row with many mentions is represented by one; the cited commit is not
  necessarily the one that did the work.
- **The census is the branch's working tree, not `origin/main`'s.** Verified
  equivalent for live rows (the only delta is nine `INVENTORIED` rows), but a
  re-run after that branch changes must re-verify.
- **`docs/STATUS.md` / `docs/BENCHMARKS.md` are not updated by this commit.**
  `check-doc-checkpoint.py` is already RED for every commit on this branch for the
  same reason (pre-existing, `--base origin/main --head HEAD`); this document
  records findings and moves no feature, and the debt is called out rather than
  silently carried.

