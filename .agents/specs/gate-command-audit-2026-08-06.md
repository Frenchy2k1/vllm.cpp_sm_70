# Gate-command audit — 2026-08-06

**72 of 97 gated rows cannot name a command a machine can run and see fail.**

**That is a statement about mechanical checkability, and nothing else.** It is
not a claim that those rows are unverified, unmeasured or ungated in the sense
this project uses the word. Many of them carry more evidence than the rows that
score `runnable` — assertion counts, exit statuses, mutation proofs, oracle
captures, ledger anchors. What they lack is a *command*: a literal string an
operator or a CI job could execute and read `$?` from. Read the finding as
"unverified work" and you would be slandering a large amount of landed,
carefully-gated engineering. See § What this does NOT mean, which is not a
footnote — it is the point.

**"Many", not "most" — the debt is not homogeneous, and it was counted.** The
72 rows split by state: **18 `DONE`, 37 `ACTIVE`, 11 `READY`, 5 `BLOCKED`,
1 `GATING`**. So the bucket holds at least three different things, and only the
first is landed work:

- **18 `DONE`** — finished work whose evidence is written as results rather than
  as commands. This is where the "better evidenced than `runnable`" claim is
  demonstrated (`SPEC-DFLASH-GGUF`, `KERNEL-EW-NORM-ACT`; § Method sample).
- **37 `ACTIVE`** — in flight, evidence partial by definition.
- **16 `READY` + `BLOCKED`** — **work not yet done, with no evidence to be more
  or less than anything.** Their `Gates` sections are *prospective*: they state
  the bar a future implementation must clear. `BACKEND-DISTRIBUTED-TP` ("a 2-GPU
  box **is required** to gate TP") and `KERNEL-GDN-AOT-BF16` ("ported upstream
  tests **pass** across boundary shapes") carry zero results, correctly, because
  nothing has run. Of three such rows opened by hand, only `SPEC-EAGLE3` carried
  MET evidence.

A prospective gate with no command is a different debt from a landed gate whose
evidence is prose, and step 4 should not treat them alike: the first needs the
work done, the second needs a transcription.

Recorded **before** anything enforces it, so the ratchet's baseline (step 4) is
a decision with reasoning attached rather than a number someone pasted.

---

## Scope

- **Subject:** every row at `READY`, `ACTIVE`, `GATING`, `DONE` or `BLOCKED` in
  the **six lifecycle matrices**, as of `0a23f966` on `spec/orchestration-harness`
  (`.agents/{engine,model,kernel,backend,quantization,feature}-matrix.md`).
  97 rows. `sglang-matrix.md` was listed as a seventh when this audit ran and
  contributed **0 rows**; step 5 dropped it from `AUDITED_MATRIX_PATHS` with the
  reasoning recorded — see risk 6, which is now **resolved**, not deferred.
  Nothing in the counts below changes: 0 rows in, 0 rows out.
- **Instrument:** `scripts/check-gate-commands.py` (landed step 2, `0a23f966`).
- **What the classifier decides:** whether the row's spec has a `Gates` heading,
  and whether the body under it contains at least one backticked or fenced span
  that names an executable and is not `true` / `:` / `echo …` / piped.
- **What the classifier cannot decide:** whether that command is *the row's
  gate*, whether it would pass, whether it is the right gate, or whether the row
  is verified by some other means. It reads shape, not meaning. Every one of
  those limits shows up concretely in § Risks and decisions.

`DONE` rows are in scope deliberately. A row that quietly lost its gate command
is exactly the regression worth catching, and `DONE` rows are the ones nobody
looks at again.

---

## Method

`python3 scripts/check-gate-commands.py --json`. The classification rules, as
the script implements them:

1. **Locate the section.** The body under the first `^#{1,6}\s*Gates\b` heading,
   up to the next heading of any level. Prose elsewhere in the spec does not
   count, however gate-like it reads.
2. **Extract candidates.** Every inline backticked span, plus every non-blank
   line of every fenced block, inside that section.
3. **A candidate names an executable** if it matches one of:
   - a known tool as a **whole word** —
     `ctest|pytest|python3?|cmake|bash|sh|make|nsys|ncu|git|gh`. Both word
     boundaries are load-bearing on the shipped record: without the trailing
     one, `sha256_cbor` matches `sh` and `python@3.14` matches `python`.
   - `flock <lockfile> <something>` — this repo's mandated shape for any
     GPU-touching gate. It quotes the real command, putting it out of reach of
     every other rule. It requires **both** a lockfile and something to run.
   - an **invoked** path — `./anything`, or a `scripts/`/`tests/` path carrying
     an executable suffix or arguments. A bare backticked filename is not a
     command: `` `docs/BENCHMARKS.md` `` is a thing the gate talks about, not a
     thing it runs.
4. **Reject what cannot fail.** `true`, `:`, `echo …` are recognised as
   commands *deliberately* and then rejected for the reason that matters — they
   cannot fail — rather than merely going unnoticed. Anything containing `|` is
   rejected too: `cmd | tail` reports `tail`'s exit status.
5. **Verdict:** `no-spec` (no resolving `.agents/specs/` link) → `no-gates-section`
   (no `Gates` heading) → `gates-no-command` (heading, no surviving candidate) →
   `runnable`.

### The hand-verified sample, and its outcome

A classifier wrong on a sample is wrong on all 97, so six rows were opened and
judged by hand before any number here was trusted — three it called `runnable`,
three it called `gates-no-command`.

| Row | Verdict | What the `Gates` section actually says | Holds? |
|---|---|---|---|
| `ENG-EXPERT-STREAM` | `runnable` | G1 names `` `flock /tmp/gpu -c './tests/parity/test_qwen36_expert_stream --resident-frac 0.5'` ``; G6 names `` `python3 scripts/check-agent-record.py` ``. Both real, both exit-status-bearing, both genuinely this row's gates. | **yes** |
| `QUANT-GGUF-COMPUTE` | `runnable` | `` `VLLM_CPP_CPU_THREADS=N ctest --test-dir build -L cpu` `` and `` `ctest -R gguf` ``. Real gates. (`N` is a metavariable needing substitution — a nit, not a misclassification.) | **yes** |
| `BACKEND-VULKAN` | `runnable` | The **only** credited command is `` `python3 -m venv ~/mlx-venv && ~/mlx-venv/bin/pip install -U pip mlx-lm` `` — an MLX install recipe, in a *Metal* subsection, for a *Vulkan* row. The section's real gating substance (token-exact vs our own CUDA backend on the same box; NMSE ≤ 5e-4, not `memcmp`) names **no** command. | **rule: yes. intent: no.** |
| `SPEC-DFLASH-GGUF` | `gates-no-command` | Seven gates, exhaustively evidenced: 302/302 assertions exit 0, 58/58 tensors byte-identical, a **mutation proof** (`kCrossQuantAcceptBand = 0` → 15/17, exit 1), a voided earlier reading honestly retracted. Commands: none. `tests/…/test_qwen3_dflash_gguf.cpp` is named as a *file*; `flock` appears bare. | **yes** |
| `MODEL-TEXT-laguna-…` | `gates-no-command` | Build flags (`-DVLLM_CPP_CUDA=OFF`, `-Werror`) and binary names with results (`test_laguna_scaffold` 8/8 · 166 assertions). No invocation. | **yes** |
| `KERNEL-EW-NORM-ACT` | `gates-no-command` | Four gates with 0-ulp bit-exactness, 140/140, 235/235 + 315/315, rollback arms, nsys per-shape timings. Names `test_ops_gdn`, `VT_RMSNORM_GATED_FAST=0`. No invocation. | **yes** |

**Outcome: 6/6 verdicts hold as the stated rule defines them.** No mismatch, so
the audit proceeds. `BACKEND-VULKAN` is right-for-the-wrong-reason — the rule is
satisfied by a line that is not a gate — which is a known limitation of the rule,
recorded in § Risks and decisions, not a defect in its implementation.

Note what rows 4–6 demonstrate, because it is the whole argument of this
document: the three `gates-no-command` rows sampled are **better evidenced than
the `runnable` one**. `SPEC-DFLASH-GGUF` carries a mutation proof. `BACKEND-VULKAN`
carries a `pip install`.

---

## Findings

```
  25  runnable
  51  gates-no-command
  20  no-gates-section
   1  no-spec

97 gated rows; 25 carry a command that can fail.
```

### Per matrix

| Matrix | Gated | `runnable` | `gates-no-command` | `no-gates-section` | `no-spec` |
|---|---:|---:|---:|---:|---:|
| `engine-matrix.md` | 43 | 14 | 24 | 5 | 0 |
| `model-matrix.md` | 19 | 5 | 11 | 3 | 0 |
| `backend-matrix.md` | 13 | 3 | 9 | 1 | 0 |
| `kernel-matrix.md` | 10 | 1 | 5 | 4 | 0 |
| `quantization-matrix.md` | 8 | 2 | 0 | 6 | 0 |
| `feature-matrix.md` | 4 | 0 | 2 | 1 | 1 |
| `sglang-matrix.md` — now **excluded**, risk 6 | 0 | 0 | 0 | 0 | 0 |
| **total** | **97** | **25** | **51** | **20** | **1** |

`quantization-matrix.md` is the outlier: 6 of its 8 gated rows have **no `Gates`
heading at all** and none are `gates-no-command`. Its specs record results in
prose sections under other names.

`sglang-matrix.md` contributed **zero rows, and not because its rows are all
below `READY`** — see risk 6. It was audited in name only, and is no longer
listed as audited at all.

### `gates-no-command` — 51 rows

The spec named is `specs[0]`, the only one the classifier reads (see risk 2).

| Row | State | Matrix:line | Spec |
|---|---|---|---|
| `KV-PREFIX-CACHE` | DONE | engine:57 | `prefix-prompt-caching-parity.md` |
| `ENG-RUNNER-MODELSHAPE` | ACTIVE | engine:71 | `first-additive-model-qwen3-dense.md` |
| `ENG-MM-INPUT-PIPELINE` | READY | engine:72 | `multimodal-track.md` |
| `ENG-MM-VISION-TOWER` | ACTIVE | engine:73 | `multimodal-track.md` |
| `ENG-MM-TEXT-BACKBONE` | ACTIVE | engine:74 | `multimodal-track.md` |
| `ENG-MM-QWEN36-VL-FORWARD` | ACTIVE | engine:75 | `multimodal-track.md` |
| `ENG-MM-VIDEO-FORWARD` | READY | engine:76 | `multimodal-track.md` |
| `ENG-MM-AUDIO-PIPELINE` | ACTIVE | engine:77 | `audio-track.md` |
| `ENG-MM-AUDIO-ENCODER` | READY | engine:78 | `audio-track.md` |
| `ENG-MM-AUDIO-E2E` | ACTIVE | engine:79 | `audio-track.md` |
| `KV-EVENTS` | ACTIVE | engine:105 | `kv-events.md` |
| `SAMPLE-LOGPROBS` | DONE | engine:131 | `sampling-controls-c7.md` |
| `SAMPLE-BEAM` | ACTIVE | engine:136 | `sampling-controls-c7.md` |
| `SAMPLE-N` | ACTIVE | engine:142 | `sampling-controls-c7.md` |
| `SAMPLE-BEST-OF` | ACTIVE | engine:143 | `sampling-controls-c7.md` |
| `TOOLS-XGRAMMAR` | ACTIVE | engine:150 | `xgrammar-backend.md` |
| `SPEC-MTP-GGUF` | DONE | engine:162 | `gguf-mtp-spec-decode.md` |
| `SPEC-DFLASH-GGUF` | DONE | engine:163 | `gguf-dflash-draft.md` |
| `SPEC-NGRAM` | ACTIVE | engine:169 | `spec-decode-breadth-d3.md` |
| `SPEC-EAGLE3` | BLOCKED | engine:170 | `spec-decode-breadth-d3.md` |
| `SPEC-DRAFT-MODEL` | ACTIVE | engine:180 | `draft-model-medusa-spec.md` |
| `ENG-POOLER-SEQ` | ACTIVE | engine:212 | `pooling-task-class.md` |
| `ENG-POOLING-RUNNER` | ACTIVE | engine:213 | `pooling-task-class.md` |
| `ENG-MOE-HOSTFREE` | DONE | engine:247 | `moe-marlin-host-free.md` |
| `MODEL-TEXT-commandr-cohere-for-causal-lm` | BLOCKED | model:172 | `sweep-recent-dense-batch.md` |
| `MODEL-TEXT-deepseek-v2-deepseek-v2-for-causal-lm` | ACTIVE | model:178 | `mla-deepseek-campaign.md` |
| `MODEL-TEXT-deepseek-v2-deepseek-v3-for-causal-lm` | BLOCKED | model:179 | `mla-deepseek-campaign.md` |
| `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` | ACTIVE | model:180 | `deepseek-v4-flash.md` |
| `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` | ACTIVE | model:223 | `kimi-linear.md` |
| `MODEL-TEXT-laguna-laguna-for-causal-lm` | ACTIVE | model:226 | `laguna-s21-w3-2026-07-31.md` |
| `MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm` | BLOCKED | model:233 | `mla-deepseek-campaign.md` |
| `MODEL-TEXT-qwen3-qwen3-for-causal-lm` | ACTIVE | model:263 | `first-additive-model-qwen3-dense.md` |
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | READY | model:384 | `gemma4-multimodal.md` |
| `MODEL-MM-voxtral-voxtral-for-conditional-generation` | READY | model:463 | `audio-track.md` |
| `MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp` | ACTIVE | model:492 | `deepseek-v4-mtp.md` |
| `KERNEL-ACCEL-PROVIDER-SELECT` | ACTIVE | kernel:115 | `metal-mlx-reuse-study.md` |
| `KERNEL-EW-NORM-ACT` | DONE | kernel:129 | `rmsnorm-gated-fast-2026-07-17.md` |
| `KERNEL-GDN-PACKED-DECODE` | DONE | kernel:153 | `gdn-packed-decode.md` |
| `KERNEL-GDN-AOT-BF16` | READY | kernel:154 | `kernel-family-inventory.md` |
| `KERNEL-GDN-SCRATCH` | READY | kernel:155 | `kernel-family-inventory.md` |
| `BACKEND-CUDA-SM087` | ACTIVE | backend:170 | `cuda-architecture-inventory.md` |
| `BACKEND-CUDA-SM110` | ACTIVE | backend:176 | `cuda-architecture-inventory.md` |
| `BACKEND-CUDA-SM120` | ACTIVE | backend:177 | `cuda-architecture-inventory.md` |
| `BACKEND-ACCEL-PROVIDER` | ACTIVE | backend:231 | `metal-mlx-reuse-study.md` |
| `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | GATING | backend:245 | `cuda-sglang-low-concurrency.md` |
| `BACKEND-GATE-CUDA-SGLANG` | BLOCKED | backend:246 | `cuda-sglang-low-concurrency.md` |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | READY | backend:247 | `cuda-sglang-low-concurrency.md` |
| `BACKEND-DISTRIBUTED-COMM` | ACTIVE | backend:272 | `scale-out-distributed.md` |
| `BACKEND-DISTRIBUTED-TP` | READY | backend:273 | `scale-out-distributed.md` |
| `QUANT-CUDA-GATES` | DONE | feature:166 | `quantization-coverage.md` |
| `BACKEND-CUDA-OTHER` | ACTIVE | feature:276 | `cuda-architecture-inventory.md` |

### `no-gates-section` — 20 rows

These specs have no `Gates` heading. Several record their gating under other
headings; the classifier does not look, by design (rule 1 — prose does not
count, or the section boundary means nothing).

| Row | State | Matrix:line | Spec |
|---|---|---|---|
| `PAR-TP` | READY | engine:118 | `tensor-parallelism.md` |
| `SPEC-MTP` | DONE | engine:161 | `mtp-spec-decode.md` |
| `SPEC-REJECTION` | ACTIVE | engine:164 | `mtp-spec-decode.md` |
| `SPEC-GDN-SEGMENTS` | ACTIVE | engine:165 | `mtp-spec-decode.md` |
| `SPEC-DFLASH` | DONE | engine:166 | `dflash-spec-decode.md` |
| `MODEL-SPEC-qwen3-dflash-dflash-qwen3-for-causal-lm` | DONE | model:481 | `dflash-spec-decode.md` |
| `MODEL-SPEC-qwen3-5-mtp-qwen3-5-mtp` | DONE | model:508 | `mtp-spec-decode.md` |
| `MODEL-SPEC-qwen3-5-mtp-qwen3-5-moe-mtp` | DONE | model:509 | `mtp-spec-decode.md` |
| `QUANT-GGUF-Q2_K` | ACTIVE | quant:64 | `cuda-keepquant-gemm.md` |
| `QUANT-GGUF-IQ2_XXS` | ACTIVE | quant:69 | `cuda-keepquant-gemm.md` |
| `QUANT-GGUF-IQ3_XXS` | READY | quant:71 | `cuda-keepquant-gemm.md` |
| `QUANT-NVFP4-MO-W4A16` | DONE | quant:121 | `marlin-dropin-feasibility.md` |
| `QUANT-NVFP4-CT-W4A4` | DONE | quant:123 | `qwen27b-w4a4-notes.md` |
| `QUANT-FP8-MO-STATIC` | DONE | quant:124 | `qwen36-forward-notes.md` |
| `KERNEL-QUANT-CIQ-GEMM-CUDA` | ACTIVE | kernel:128 | `cuda-keepquant-gemm.md` |
| `KERNEL-ATTN-DFLASH-BLOCK` | DONE | kernel:140 | `dflash-spec-decode.md` |
| `KERNEL-ATTN-DFLASH-PAGED-BLOCK` | DONE | kernel:141 | `dflash-spec-decode.md` |
| `KERNEL-ATTN-DENSE-FLASH` | ACTIVE | kernel:148 | `multimodal-speed.md` |
| `BACKEND-GATE-METAL-MLXLM` | ACTIVE | backend:254 | `competitive-benchmarks.md` |
| `MODEL-SPEC` | ACTIVE | feature:154 | `mtp-spec-decode.md` |

### `no-spec` — 1 row

| Row | State | Matrix:line | Why |
|---|---|---|---|
| `BACKEND-MLX` | ACTIVE | feature:279 | Its only link is to `backend-matrix.md`, an area matrix, not a `.agents/specs/` file. |

### `runnable` — 25 rows

| Row | State | Matrix:line | First credited command |
|---|---|---|---|
| `ENG-ASYNC-SCHED` | DONE | engine:62 | `flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'` |
| `ENG-PRIORITY-SCHED` | GATING | engine:63 | `flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'` |
| `ENG-CORE-BUSY-LOOP` | GATING | engine:66 | `flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'` |
| `KV-SLIDING-LOCAL-SPECS` | READY | engine:97 | `cmake -S . -B build-c5-cpu … && cmake --build …` |
| `KV-SLIDING-WINDOW-SPEC` | GATING | engine:98 | `cmake -S . -B build-c5-cpu … && cmake --build …` |
| `KV-CHUNKED-LOCAL-SPEC` | GATING | engine:99 | `cmake -S . -B build-c5-cpu … && cmake --build …` |
| `ENG-EXPERT-STREAM` | READY | engine:110 | `flock /tmp/gpu -c './tests/parity/test_qwen36_expert_stream --resident-frac 0.5'` |
| `TOOLS-STREAMING-PARSER` | ACTIVE | engine:154 | `git diff --stat` **(weak — see risk 3)** |
| `SERVE-STREAM-USAGE` | GATING | engine:200 | `git diff --check` **(weak — see risk 3)** |
| `SERVE-ASYNC-LLM` | GATING | engine:203 | `flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'` |
| `SERVE-HTTP-TRANSPORT` | DONE | engine:204 | `scripts/check-agent-record.py` |
| `ATTN-ROPE-FAMILY` | READY | engine:231 | `cmake -S . -B build-c5-cpu … && cmake --build …` |
| `ATTN-CHUNKED-LOCAL` | GATING | engine:236 | `cmake -S . -B build-c5-cpu … && cmake --build …` |
| `LOAD-SAFETENSORS-DIRECT-DENSE` | GATING | engine:246 | `scripts/check-agent-record.py` (9 commands total) |
| `MODEL-FACTORY-registry` | GATING | model:156 | `python3 scripts/check-agent-record.py` |
| `MODEL-TEXT-gemma4-gemma4-for-causal-lm` | BLOCKED | model:196 | `scripts/check-agent-record.py` (5 checkers) |
| `MODEL-TEXT-glm4-glm4-for-causal-lm` | READY | model:199 | `scripts/check-agent-record.py` |
| `MODEL-TEXT-glm4-moe-lite-…` | ACTIVE | model:201 | `scripts/check-agent-record.py` |
| `MODEL-TEXT-deepseek-v2-glm-moe-dsa-…` | BLOCKED | model:202 | `scripts/check-agent-record.py` |
| `QUANT-GGUF-COMPUTE` | READY | quant:33 | `VLLM_CPP_CPU_THREADS=N ctest --test-dir build -L cpu` |
| `QUANT-NVFP4-CT-W4A16` | ACTIVE | quant:122 | `scripts/qwen3-32b-nvfp4a16-oracle-capture.py --runs 5` |
| `KERNEL-GEMM-CPU-ELEM` | ACTIVE | kernel:126 | `ctest -j2` **(weak — see risk 3; lifted from prose about a FLAKE)** |
| `BACKEND-CUDA-ARCH-ADDITIVITY` | ACTIVE | backend:187 | `scripts/check-agent-record.py` (incl. `cuobjdump -lelf libvllm.a`) |
| `BACKEND-METAL-MLX` | ACTIVE | backend:232 | `python3 -m venv ~/mlx-venv && … pip install … mlx-lm` **(env setup — risk 3)** |
| `BACKEND-VULKAN` | ACTIVE | backend:233 | `python3 -m venv ~/mlx-venv && … pip install … mlx-lm` **(env setup — risk 3)** |

---

## What this does NOT mean

**A row without a runnable gate command is not ungated work.** The classifier
answers one narrow question — *can a machine re-check this without a human
reading prose?* — and the answer being "no" says nothing about whether the work
was done, measured, or verified.

The sample proves it directly. `SPEC-DFLASH-GGUF` is `gates-no-command` and its
`Gates` section contains: 302/302 assertions at exit 0; 58 of 58 tensors proven
byte-identical across two loaders; a demonstration that the pass is **not
vacuous** (pointed at `Q4_K_M`, the same case goes 21/302 red, exit 1); a
mutation proof that a tolerance band is load-bearing (`kCrossQuantAcceptBand = 0`
→ 15/17, exit 1, on exactly the two banded assertions); and an earlier "MET"
reading explicitly **voided** because the build lacked `-DVLLM_CPP_CUTLASS_DIR`.
That is a higher standard of evidence than most `runnable` rows meet.
`KERNEL-EW-NORM-ACT` is likewise `gates-no-command` and carries 0-ulp
bit-exactness over 140 assertions plus 235/235 and 315/315 engine token gates
with rollback arms.

Meanwhile `BACKEND-VULKAN` scores `runnable` on a `pip install`.

So the ordering the counts imply is not a quality ordering. What separates the
buckets is **notation**: whether the evidence was written as an invocation or as
a result. Much of this repo's record is written as results — "235/235,
exit 0" — which is *more* informative to a human reader and *useless* to a
machine that wants to re-run it.

Three further reasons a well-gated row lands outside `runnable`:

- **The evidence lives elsewhere.** `.agents/parity-ledger.md` holds the binding
  benchmark rows with their boxes, recipes and reps. The classifier never opens
  it.
- **The gate is a test anchor, not a command.** A spec naming
  `test_qwen36_paged_engine` and its assertion count is pointing at a real,
  runnable, CI-covered binary. It just isn't spelled as a command line.
- **The gate genuinely cannot be run yet** — HW-blocked (`BACKEND-*-XPU`),
  oracle-blocked, or needing two Sparks. Recording that plainly is the design's
  stated intent (§ Work breakdown item 5: "record honestly that they cannot be
  gated yet").

The actionable reading is therefore: **72 rows cannot be driven through the
operator loop as written.** Not: 72 rows are unverified.

What that costs differs by state, per the split in the lede. For the 18 `DONE`
and much of the 37 `ACTIVE`, it is a **transcription** — the evidence exists and
needs writing as an invocation. For the 16 `READY`/`BLOCKED` rows there is no
evidence to transcribe and none should be expected; their gates are prospective,
and they become runnable when the work is done, not before. Step 4 should not
report those 16 as a debt someone forgot to pay.

---

## The ratchet baseline

**The 25 named rows below — as a SET, pinned exactly.**

> **Superseding note (step 5).** An earlier draft of this section said the
> baseline was "**25**", "a **floor on a count**", and that the rule was
> **shrink-only**. Both descriptions are wrong about what shipped, and the next
> subsection (§ The baseline's principal false-red mode) already argued why a
> count cannot work. What `check-gate-commands.py` pins is the SET of row IDs,
> and it is pinned by **exact equality**, not as a floor. This section is the
> corrected text; the count-and-floor wording survives nowhere.

Twenty-five rows scored `runnable` at `0a23f966`. Step 4 pins **their IDs**, in
`RUNNABLE_BASELINE`, and the contract is an **exact pin**:

- `--check` refuses a row that **lost** its command, and distinguishes that from
  a row that legitimately **left** the gated population (next subsection);
- `tests/scripts/test_check_gate_commands.py` additionally asserts the shipped
  `runnable` set **equals** `RUNNABLE_BASELINE`, which is what makes "just lower
  the number" impossible to do quietly — and which means **growth is red too**.

So adding a real gate command to a row's spec — textbook improvement — leaves
`--check` at 0 while the suite, preflight and CI go red until the set is
re-pinned. **That is the intended cost, and it is stronger than shrink-only.**
Any movement, up or down, re-pins `RUNNABLE_BASELINE` in the SAME change, naming
the rows that moved and why. Growth is welcome, ordinary work; **silent** growth
is what the pin forbids.

The pin is not an assertion that these 25 rows are correctly gated. **Five of
them are not** (risk 3). They are pinned anyway, because a ratchet that waits for
a clean baseline never starts, and because the alternative — relaxing the rule
until everything passes — is the failure mode this whole subsystem exists to
prevent. A relaxed gate is worse than no gate.

Raising the set is ordinary work: transcribe a row's existing evidence into an
invocation, add the row ID here and to `RUNNABLE_BASELINE` in the same change.
That is § Work breakdown item 5, and it is now measured rather than estimated.

### The baseline's principal false-red mode: the population moves

**Twenty-five is a headcount over a population that is not fixed, which is the
first reason step 4 pins IDs rather than a number.** Step 2 already observed
the record move — three rows shifted during its own work — which is exactly why
the *total* (97) was deliberately not pinned. The `runnable` population inherits
the same exposure, and a rule written over a bare number reads a legitimate
record edit as a regression.

A `runnable` row can leave the population without anything being broken:

- it is **deleted** (a row retired, or folded into another);
- it is **merged** with another row;
- it **transitions out of `GATED_STATES`** — e.g. `READY` → `INVENTORIED` on a
  descope, or any move to a state below `READY`;
- its **matrix** leaves `AUDITED_MATRIX_PATHS` (risk 6, done in step 5 for
  `sglang-matrix.md` — which cost nothing here only because that matrix
  contributed no `runnable` rows to begin with).

Each drops the headcount below 25 and would turn a number-based gate red on a
correct edit. Risk 4 covers gaming the count *upward*; this is the opposite
failure and is the more likely one, because record edits are routine and gaming
is not.

**The distinguishing rule step 4 should implement:**

> A drop below the baseline is a **regression** only if the row that lost its
> `runnable` verdict **still exists and is still gated**. If the population
> itself shrank — the row was deleted, merged, or moved out of `GATED_STATES`,
> or its matrix left the audited set — the baseline is **re-pinned in the same
> change**, with the row ID and the reason recorded in the commit body.

Re-pinning is a disclosed decision, not an escape hatch: it names which row left
and why, so a reviewer can check the claim. The rule that must never be applied
is lowering the baseline because the number "went down" without saying which row
moved — that is the relaxation this subsystem exists to prevent.

**Consequence for the checker's output:** step 4 cannot enforce this on a bare
count — this is what supersedes the count-and-floor wording flagged above. It
needs the **set** of `runnable` row IDs, not just `len()`, so a drop
can be attributed to a named row and classified as "lost its command" versus
"left the population". Pinning an integer alone makes the two indistinguishable
— the repo's recorded defect class, one more time.

---

## Risks and decisions

### 1. The vocabulary misses real gate shapes — but the measured exposure inverts the expectation

Step 2's review flagged seven shapes the tool vocabulary does not know:
`compute-sanitizer`, `cuobjdump`, `/usr/bin/time -v`, `curl`, `brew install`,
`nvidia-smi`, and a built binary invoked without `./`. The concern was **false
reds** once ratcheting.

**Measured, this exposure is currently zero.** Every argument-bearing occurrence
of those seven shapes inside a gated row's `Gates` section lives in a row that
is *already* `runnable` via some other command:

All seven shapes, each measured against the rows currently NOT `runnable`:

| # | Shape | Argument-bearing occurrence | Rows it would flip |
|---|---|---|---|
| 1 | `compute-sanitizer` | `compute-sanitizer memcheck` — `sweep-qwen3-32b-nvfp4a16.md`, already `runnable` | **0** |
| 2 | `cuobjdump` | `cuobjdump -lelf libvllm.a` — `cuda-arch-additivity.md`, already `runnable` | **0** |
| 3 | `/usr/bin/time -v` | none anywhere in a gated row's `Gates` section | **0** |
| 4 | `curl` | `curl -N` — `async-serving.md` (4 rows), all already `runnable` | **0** |
| 5 | `brew install` | `brew install mlx` / `brew info mlx` — `backend-fanout-metal-vulkan-xpu.md`, already `runnable` | **0** |
| 6 | `nvidia-smi` | bare only, in `expert-streaming.md`, already `runnable` | **0** |
| 7 | built binary without `./` | no `test_*` name carries arguments anywhere | **0** |

(`vllm-bench --num-prompts 1 …` in `gguf-cpu-threadpool.md` is not among the
seven but was measured alongside them; that row is already `runnable` too.)

So widening the vocabulary to these seven would change **no row's verdict
today**. The risk is prospective — a *future* gate written with one of these
tools as its only command would be a false red — not a present miscount.

**The larger risk points the other way, and it is measured too.** Adding a bare
tool name credits *prose that merely mentions the tool*, and the exposure is
several times the false-red one:

| Naive entry | Falsely credits |
|---|---|
| bare **binary name** (shape 7, e.g. `test_*`) | **15 non-`runnable` rows** |
| bare `compute-sanitizer` (shape 1) | **6 non-`runnable` rows** |

Shape 7 is the worst of the seven and the easiest to get wrong, because "a built
binary invoked without `./`" reads like an instruction to credit the bare name.
Fifteen rows name a `test_*` binary with its assertion counts and no invocation
— `KERNEL-EW-NORM-ACT`, `MODEL-TEXT-laguna-…`, `ENG-MOE-HOSTFREE`,
`ENG-POOLER-SEQ`, `ENG-POOLING-RUNNER`, `ENG-RUNNER-MODELSHAPE`, `KV-EVENTS`,
`SPEC-NGRAM`, `SPEC-EAGLE3`, `SPEC-DRAFT-MODEL`, `TOOLS-XGRAMMAR`,
`MODEL-TEXT-qwen3-…`, `MODEL-TEXT-kimi-linear-…`, `MODEL-TEXT-deepseek-v4-…`,
`MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp`. Requiring an argument
(`test_\w+\s+\S`) credits **0** of them.

`compute-sanitizer` appears as a bare backticked word in the `Gates` sections of
**six** rows currently `gates-no-command` (`ENG-MM-INPUT-PIPELINE`,
`ENG-MM-VISION-TOWER`, `ENG-MM-TEXT-BACKBONE`, `ENG-MM-QWEN36-VL-FORWARD`,
`ENG-MM-VIDEO-FORWARD`, `MODEL-TEXT-commandr-cohere-for-causal-lm`).

The two sets are **disjoint** (verified), so naively adding both would credit
**21 distinct rows on nothing**, taking the ratchet baseline from **25 to 46** —
while every one of those rows stayed exactly as ungatable as it is today.

**This is the `flock` bug, exactly.** Step 2 added `flock` as a bare vocabulary
entry, credited five rows whose specs name the *lock idiom* rather than a gate,
and produced a count that matched an earlier prediction — which felt like
corroboration and was a bug. The shipped `_WRAPPER` rule requires
`flock <lockfile> <something>` for that reason.

**Decision for step 4:** widen only with argument-requiring patterns
(`compute-sanitizer\s+\S`, not `compute-sanitizer`), and re-run this audit
before and after so any baseline movement is attributed to a named row rather
than absorbed into a total. Do not widen as part of landing the ratchet; widen
in its own change, where the count delta is legible.

### 2. `classify_row` reads only `specs[0]` — this changes 12 verdicts, measured

`classify_row` takes the **first** resolving `.agents/specs/` link and ignores
the rest. 25 of the 97 gated rows link two or more existing specs.

Scanning **all** linked specs changes **12 verdicts**, in two groups. Six move
`gates-no-command` → `runnable`, which is the group that moves the baseline:

| Row | `specs[0]` verdict | Later spec | Command there |
|---|---|---|---|
| `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | `gates-no-command` | `sweep-gemma.md` | `scripts/check-agent-record.py` |
| `KERNEL-ACCEL-PROVIDER-SELECT` | `gates-no-command` | `accelerator-seam-audit.md` | `scripts/check-agent-record.py` |
| `BACKEND-CUDA-SM087` | `gates-no-command` | `cuda-arch-additivity.md` | `scripts/check-agent-record.py` |
| `BACKEND-CUDA-SM110` | `gates-no-command` | `cuda-arch-additivity.md` | `scripts/check-agent-record.py` |
| `BACKEND-CUDA-SM120` | `gates-no-command` | `cuda-arch-additivity.md` | `scripts/check-agent-record.py` |
| `BACKEND-ACCEL-PROVIDER` | `gates-no-command` | `dropin-kernel-abi.md` | `python3 scripts/check-agent-record.py` |

The other six move `no-gates-section` → `gates-no-command` — a later spec has a
`Gates` heading where `specs[0]` has none. These do **not** touch the baseline,
but they do move the published bucket totals: `QUANT-GGUF-Q2_K`,
`QUANT-GGUF-IQ2_XXS`, `QUANT-GGUF-IQ3_XXS`, `QUANT-NVFP4-CT-W4A4`,
`KERNEL-QUANT-CIQ-GEMM-CUDA`, `BACKEND-GATE-METAL-MLXLM`.

**So the honest all-specs classification is:**

| | shipped (`specs[0]`) | all-specs |
|---|---:|---:|
| `runnable` | 25 | **31** |
| `gates-no-command` | 51 | 51 |
| `no-gates-section` | 20 | **14** |
| `no-spec` | 1 | 1 |

(`gates-no-command` holds at 51 because it loses six upward and gains six from
`no-gates-section` — a coincidence of equal counts, not a fixed point.)

The baseline pinned above is 25 because that is what the shipped classifier
computes, and a baseline must be reproducible by running the tool. Recorded here
so that when `classify_row` is fixed to scan all specs, the jump from 25 to 31 —
and the § Findings `no-gates-section` total dropping 20 → 14 — is **understood as
the fix landing** and not mistaken for rows being gated or degated. Note this
fix is **not** free under the shipped pin, and an earlier draft here said it was:
the baseline is an EXACT pin, so taking `runnable` from 25 to 31 turns the suite
red until the six new IDs are added to `RUNNABLE_BASELINE` in the same change.
That is the pin working — the six rows are named right here, so the re-pin is a
transcription with its reasoning already written.

### 3. Five of the 25 `runnable` credits are not gates

They satisfy the stated rule. The ratchet should not lock them in silently.

- **`BACKEND-VULKAN`** and **`BACKEND-METAL-MLX`** — both credited to the same
  MLX `pip install` line, which is environment setup, in a *Metal* subsection.
  For `BACKEND-VULKAN` it is not even the right backend. Each row's single
  credited command is this line; remove it and both become `gates-no-command`.
- **`TOOLS-STREAMING-PARSER`** — credited solely to `git diff --stat`, which
  **exits 0 in every state this gate could encounter**: measured 0 on a clean
  tree, 0 on a dirty tree, and 0 on a bogus pathspec. (It returns 129 outside a
  git repo — immaterial, since a gate runs in the checkout.) This is a
  `_CANNOT_FAIL` shape the rule does not recognise: `true`, `:` and `echo` are
  blocked by name, but a real command with no reachable failure mode passes.
  Arguably the worst credit in the set, and it was not among the two the step-2
  review named.
- **`SERVE-STREAM-USAGE`** — credited solely to `git diff --check`. This *can*
  fail (whitespace errors), so it is not vacuous, but it is not this row's gate.
- **`KERNEL-GEMM-CPU-ELEM`** — credited solely to a bare `` `ctest -j2` ``
  extracted from prose reporting a **flake**: the spec says a test "flaked once
  under `ctest -j2` on the co-tenanted dev box". That is an incident report, not
  an invocation this row is gated by — no `-R`, no test-dir, no assertion of a
  result. The classifier reads shape, and the shape of a flake note is the shape
  of a command. Found in step 5's whole-branch review; it is the fifth weak
  credit and was missed by the count of four above.

A related weakness, not counted above: three GLM rows
(`MODEL-TEXT-glm4-*`, `MODEL-TEXT-deepseek-v2-glm-moe-dsa-*`) are credited
`scripts/check-doc-checkpoint.py --staged`. The harness spec
(`orchestration-harness.md` § Gate-command discipline, rule 3) names `--staged`
as the exact anti-pattern that let **eleven** commits on the P0 branch be red
while every preflight was green. Those rows also carry
`scripts/check-agent-record.py`, so they are not credited on the weak command
alone.

**Decision:** record, do not fix. Fixing means editing matrices and specs, which
is out of scope for this task by construction, and each is a judgement about
what a row's real gate should be — an operator call, not a classifier change.

### 4. What this gate stops being able to see once it has run

The harness spec asks this of any gate it introduces. For
`check-gate-commands.py` the answer is: **it cannot distinguish a row that
acquired a real gate from a row that acquired a plausible-looking string.**
Once ratcheting, the cheapest way to raise the count is to paste
`scripts/check-agent-record.py` — already among the credited commands of 9 of
the 25 — into a `Gates` section. That would pass, and it would gate nothing about the row.
The classifier reads shape, so shape is what it can be satisfied with.
No mitigation is proposed here beyond naming it; it is a review responsibility,
not a checkable one.

### 5. Rows the classifier could not decide, and the human call

- **`BACKEND-MLX` (`no-spec`)** — links only `backend-matrix.md`. **Call: not a
  defect.** It is a `feature-matrix.md` index row summarising a backend whose
  real gating lives in `backend-matrix.md`'s `BACKEND-METAL-*` rows, which are
  audited here in their own right. Counted as `no-spec`, excluded from the
  actionable debt.
- **`quantization-matrix.md`'s 6 `no-gates-section` rows** — these specs record
  results under other headings. **Call: real debt, low priority.** Adding a
  `Gates` heading to a spec that already states its results is a transcription
  job, and it is the same job the other 51 need.

### 6. `sglang-matrix.md` is audited in name only — an absence that looks like a pass

Step 2 widened `AUDITED_MATRIX_PATHS` beyond `check-agent-record.py`'s
`MATRIX_PATHS` to cover all seven matrices, `sglang-matrix.md` among them. It
contributes **0 rows** to this audit.

The first reading — "all its rows are below `READY`" — is **wrong**, and was
checked rather than assumed. `record.parse_claim_rows()` returns **zero rows and
zero parse errors** for that file, out of 87 table rows. The cause is in the
matrix's own header: it carries "a **classification** in place of a lifecycle
state" — `FUSED` / `ACTIVE` / `INVENTORIED` / `NOT-APPLICABLE` — so it has no
state column of the shape the row parser recognises, and the parser reports
nothing rather than failing.

**This is the repo's recorded defect class**: a failure and an absence looking
identical. A file listed as audited, returning no errors, contributing nothing.
Anyone reading `AUDITED_MATRIX_PATHS` would reasonably conclude SGLang rows were
examined and found clean.

**Resolved in step 5: dropped, with the reasoning recorded.** `sglang-matrix.md`
is no longer in `AUDITED_MATRIX_PATHS`; `scripts/check-gate-commands.py` carries
the reason at the constant, and `test_sglang_is_excluded_and_the_exclusion_is_justified`
pins the justification rather than the bare absence — it asserts that
`parse_claim_rows` really does find zero rows and zero errors there, so the
matrix goes back into the audited set the moment it gains lifecycle rows. The
`runnable` set is unchanged at 25, because 0 rows left. The original call
follows, unchanged.

**Original call (step 3): record, do not fix here.** Whether `SGLANG-*` rows *should* carry gate
commands is a real question — they are classification rows about a competitor's
surface, and most are `FUSED`, i.e. claims about our existing implementation
whose gates live on the rows they map to. But that argument must be made
explicitly, not arrived at by a silent zero. Step 4 should either drop
`sglang-matrix.md` from the audited set with that reasoning recorded, or teach
the parser its schema. **It must not leave it listed and empty.** — Step 4 did
neither and step 5 took the first branch, above: dropping is the cheaper option
and is defensible, because a classification column (`FUSED` / `INVENTORIED` /
`NOT-APPLICABLE`) is not a lifecycle state and `FUSED` rows' gates live on the
rows they map to, which are audited in their own right.

---

## Provenance

- Classifier: `scripts/check-gate-commands.py` @ `0a23f966`.
- Rows: the six lifecycle matrices @ `0a23f966` (`sglang-matrix.md` was listed
  as a seventh and contributed 0 rows; dropped in step 5, risk 6).
- Reproduce: `python3 scripts/check-gate-commands.py` (summary) or
  `--json` (per-row).
- Related: [orchestration-harness.md](orchestration-harness.md) § Gate-command
  discipline and § Work breakdown item 5.
