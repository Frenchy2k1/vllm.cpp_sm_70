# GATE-PIN-UNPINNED-SNAPSHOTS — a gate may not choose its own subject

Issue: [#471](https://github.com/mudler/vllm.cpp/issues/471) (the defect),
[#472](https://github.com/mudler/vllm.cpp/issues/472) (the goldens that record no
revision, which bounds how much of #471 can be closed from evidence)
Row: `GATE-PIN-UNPINNED-SNAPSHOTS`

Pinned in this row:

| repo | revision | evidence |
|---|---|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | `491c2f1ea524c639598bf8fa787a93fed5a6fbce` | `oracle.model` of `goldens/qwen36_{embed,norm,gdn_layer,fullattn_layer,logits}_35b/manifest.json` and `args.model` of `goldens/qwen3_5_mtp_head_35b/manifest.json` |
| `unsloth/Qwen3.6-27B-NVFP4` | `890bdef7a42feba6d83b6e17a03315c694112f2a` | already committed as `kQwen27NvfP4Revision`; `oracle.model` of the five `qwen36_*_27b` manifests |
| `z-lab/Qwen3.6-27B-DFlash` | `0919688658996800f86b895034249700e9481106` | **DETERMINISM pin only** — see "The one pin that is not ratified" |

## Scope

**In scope.** Pin every checkpoint gate whose revision is derivable from a
committed record, and make an unpinned resolution impossible to add later.

- `tests/parity/hf_snapshot.h` gains `kQwen36A3bNvfP4Revision` /
  `Qwen36A3bNvfP4Snapshot()` and `kQwen27DFlashDraftRevision` /
  `Qwen27DFlashDraftSnapshot()`, both built on the existing `HfSnapshot` so the
  skip-not-substitute discipline is inherited unchanged.
- The three DFlash gates (`test_qwen3_dflash_draft_parity`,
  `test_qwen3_dflash_kvprep_parity`, `test_qwen27_dflash_spec_decode`) lose their
  private `SnapDir` helpers and resolve through the pinned accessors.
- The five 35B gates (`test_op_parity` `Find35BSnapshot`/`FindMtpSnapshot`,
  `test_qwen36_paged_engine`, `test_qwen36_async_serving`,
  `test_qwen36_spec_decode`, `tests/vllm/test_qwen36_weights`) likewise.
- `tools/bench/online_gate.py::MODEL_GATE_CONTRACTS["test_qwen36_paged_engine"]`
  stops recording `golden_revision: None`; its comment currently asserts that the
  gate "pins no revision", which this row makes false.
- A new deterministic, GPU-free gate `test_hf_snapshot_pinning` that builds a
  synthetic two-revision cache and proves selection, refusal, and override.
- A new checker `scripts/check-snapshot-pins.py` that fails on any *new*
  unpinned checkpoint resolution under `tests/`, `tools/` and `scripts/`.

**Out of scope, recorded as owed on [#472](https://github.com/mudler/vllm.cpp/issues/472).**
The **55** remaining unpinned resolvers whose goldens record no revision at all
(19 `*_greedy*` corpora carry no manifest file whatsoever). They cannot be pinned
from evidence without re-capturing the goldens, and "pin to whatever is cached
here" is the defect wearing a constant's name. They are enumerated in the
checker's ledger, each of which is a line that must be *deleted*, never added to.
50 are the C++ gates; the last 5 are `scripts/` staging and reference-dump
helpers, which the widened scan (see §3) reaches — they FEED goldens rather than
assert them, which is the same hazard one step upstream.

`q3mxfp4` is out of scope by the same rule and needs no C++ change: no C++ gate
resolves `Yi30/Qwen3-8B-MXFP4`, and `tools/bench/online_gate.py` already records
`golden_revision: None` for `mxfp4_smoke_battery` honestly. The row leaves that
`None` in place rather than promoting `MODEL_REVISIONS["q3mxfp4"]` into it —
those are the benched model and the golden's provenance, and conflating them is
exactly the false-coverage claim this row exists to remove.

## The defect, MEASURED not inferred

`tests/parity/hf_snapshot.h` is the only revision-pinned resolver in the tree and
exactly **5 of 61** checkpoint-resolving test files use it. The other 56 take the
first entry `std::filesystem::directory_iterator` yields under
`<repo>/snapshots/`, through 19 differently-named private copies of the same
helper (`SnapDir`, `FindSnapshot`, `FindSnap`, `FindCkpt`, `Find35BSnapshot`,
`FindShard1`, …).

`unsloth/Qwen3.6-27B-NVFP4` caches two materially different models under one repo
name on the gate host — `@890bdef7` (NVFP4 W4A4, bf16 GDN tower, single-file
`model.safetensors`) and `@ccdaab7e` (the same repo silently re-quantized to FP8
W8A8 across 5 shards, every `*_global_scale` gone). Three gates resolve it
unpinned.

Read off `dgx.casa` today, read-only:

```
readdir order: ['890bdef7a42feba6d83b6e17a03315c694112f2a',
                'ccdaab7e68af2409599b8949a8f2685703c9bae5']
FIRST with config.json (what SnapDir returns): 890bdef7...
```

**So the two plain-`SnapDir` gates currently resolve the correct revision, and
this row is not repairing a live wrong-model measurement.** That is worth stating
plainly rather than overselling the find: readdir order is an ext4 hash artefact,
not insertion order, and it is stable only until the directory is rewritten. A
re-download, an eviction, a different filesystem, or a fresh CI cache reorders it
and the gate silently changes subject with nothing in its output naming which
model ran. `test_qwen27_dflash_spec_decode` is a half-step better and a whole
step more misleading: its `prefer_single_file` heuristic selects `@890bdef7`
because only that revision is single-file, but its fallback returns *the last*
entry seen, so on a cache where no snapshot is single-file it silently prefers
the newest re-quant.

A gate whose subject is a property of the filesystem is not a gate. The remedy is
not a better heuristic; it is naming the revision.

## The one pin that is not ratified

`z-lab/Qwen3.6-27B-DFlash` is pinned to `@0919688658996800f86b895034249700e9481106`
on weaker evidence than the other two, and the header says so in place.

What was checked, and what it proved:

- The DFlash goldens record **no** revision. `dflash_27b_spec_{on,off}.json`
  carry `draft="z-lab/Qwen3.6-27B-DFlash"` as a bare repo name;
  `dflash_27b_draft/` and `dflash_27b_kvprep/` name neither repo nor revision.
- `dflash_27b_draft/ckpt_keys.txt` looked like a checkpoint fingerprint and is
  not one. Compared against `@0919688` on the gate host: 47 golden keys vs 58
  snapshot keys, **MISMATCH** — the golden list is `model.`-prefixed with
  `mlp.gate_up_proj` fused and includes `lm_head`/`embed_tokens` that the draft
  checkpoint does not ship. It is vLLM's *loaded-module* namespace, not the
  safetensors key set, so it cannot identify a snapshot.
- The revision itself is committed, in
  `.agents/completed/state-events/0000-00/STATE-LEGACY-000001.md:30698,32279`,
  which records `0919688…` as the snapshot fetched and used for this work.

So the pin is a *determinism* pin resting on a committed provenance record, not a
*ratified* one resting on the golden it gates. It stops the gate silently
substituting a future re-quant of the same repo — the failure mode #471 is
about — and it does not prove the goldens belong to it. The distinction is
carried in the header comment and in the skip banner the gate prints, and the
re-capture is owed on #472. Pinning to "the only revision cached here" without
saying so would have been the forbidden move; saying so is what makes it
admissible.

## Design

### 1. Resolver

Two new constants and two new accessors in `tests/parity/hf_snapshot.h`, both
delegating to the existing `HfSnapshot(repo, revision, env_override)`. Neither is
named `kQwen27NvfP4Revision<suffix>`:
`tests/tools/test_online_gate_server_binary.py:617-621` parses that exact
identifier out of this header and asserts a **single** pin. That assertion is
correct, it is what ties the recorded golden revision to the gate that resolves
it, and this row leaves it untouched — the same reason
`kQwen27nFp8TowerRevision` was named apart from it on
`row/GATE-27B-FP8-TOWER-GOLDEN`.

### 2. Removing the unpinned path, not documenting it

Each converted file's private `SnapDir` / `Find*Snapshot` is **deleted**, not
left beside the pinned call. A helper that can resolve a checkpoint without a
revision is the defect; leaving one in the file with a comment telling the next
agent not to call it is how the defect comes back.

### 3. Making it impossible to add later

`scripts/check-snapshot-pins.py` scans `{tests,tools,scripts}/**/*.{cpp,cc,cxx,h,hpp,hh,py}`
for any of eleven directory ENUMERATORS — `directory_iterator`,
`recursive_directory_iterator`, `opendir`/`scandir`/`readdir`,
`listdir`/`iterdir`/`walk`, `glob`/`iglob`/`rglob` — whose call text (receiver,
name and *balanced* argument list, so line wrapping is not an evasion) names an
HF-cache marker or a value the file bound to one, followed to a fixpoint through
assignments, struct members and helpers that RETURN the path. It fails on any
occurrence not present in an explicit ledger inside the checker. `src/` is
deliberately unscanned: resolving a user's cache at run time is the product
working, not a gate choosing its own subject. The checker's guarantees and its
known evasions are stated in its own module docstring, which is the authority. The ledger is a
**shrinking** list: every entry names the file and the issue that owes its
golden's provenance (#472). Adding a line is a review event; deleting one is the
work. This is the same shape as the existing device-leakage allowlist
(issue #302) and is admissible under the record rules because it is written only
when an unpinned resolver is *added* — which is what it exists to prevent — and
not by every PR.

The checker ships with its own RED proof. **The first version of this claim was
FALSE and is corrected here rather than quietly dropped**, because a future agent
will cite it: that self-test synthesised ONE unpinned resolver, so it proved only
that the checker matched that one fixture. A fresh review mutated the checker five
ways and four of the five kept the self-test green — renaming the required
subject, requiring the `fs::` spelling, restricting the scan to `.cpp`, and
restricting it to `tests/parity` (caught only incidentally, by the STALE arm).

What replaces it is a `FIXTURES` corpus swept in BOTH directions: every positive
is an idiom some plausible narrowing would drop, every negative is code some
plausible widening would falsely flag, and each positive records in `kills` the
narrowing it exists to catch. The claim is now bounded rather than absolute:
**narrowing any single branch of the pattern that the corpus covers turns the
self-test red.** It is not a proof against an idiom nobody wrote a fixture for,
and the checker is lexical and defeatable on purpose — `find(1)`, a marker-free
environment variable, or a `#define`d enumerator all still evade it. The module
docstring states that scope, and is the authority on it.

**The bound was stated in two places and asserted in a third, and the third
still made the absolute claim.** A second fresh review found
`tests/scripts/test_check_snapshot_pins.py::test_the_fixture_corpus_sweeps_both_directions`
closing on "a mutation to any single branch of the pattern lands on one of
them" — the exact sentence the paragraph above retracts — and then disproved it
with four single-branch narrowings that left checker, self-test and suite all
green: dropping the three environment `CACHE_MARKERS`
(`HF_HUB_CACHE`/`HF_HOME`/`TRANSFORMERS_CACHE`), dropping
`scandir`/`walk`/`iglob`/`rglob`, dropping `listdir`, and dropping `readdir`.
Measured coverage at that point was **4 of 11 enumerators and 2 of 5 markers**
isolated by some fixture. The environment markers were the sharpest case: the
commit that added them asserted they "ARE caught", the branch is live, and no
fixture touched it.

Correcting the sentence again would have fixed the honesty and not the hole, so
both were closed. Ten fixtures were added, each exercising exactly ONE
uncovered enumerator or carrying exactly ONE uncovered marker, and the bound
itself became mechanical:
`test_narrowing_any_single_branch_drops_a_positive_fixture` builds the pattern
minus one branch — through the checker's own `enumerator_re`/`marker_re`, never
a restated copy — and requires some positive fixture to stop being reported, for
all 11 enumerators and all 5 markers. **An enumerator or marker added to the
checker without a fixture that isolates it is now RED**, so the two lists cannot
outgrow their corpus a third time. What stays unguaranteed is unchanged and
still stated: narrowings outside those two lists are covered only where a
fixture happens to exercise them.

Known gaps recorded rather than fixed here:
[#482](https://github.com/mudler/vllm.cpp/issues/482) (the checker
false-positives on the CORRECT pinned Python form, and the only relief is a
ledger line — the ratchet running backwards),
[#483](https://github.com/mudler/vllm.cpp/issues/483) (ordinary modern-C++
punctuation — brace init, compound assignment, a lambda body severed by the
`[;{}]` split — evades it, and no cheap subset closes the class),
[#484](https://github.com/mudler/vllm.cpp/issues/484) (the CI step has never
been observed on a real runner: UNVERIFIED, not absent and not successful), and
[#485](https://github.com/mudler/vllm.cpp/issues/485)
(`_MARK_FIXPOINT_ROUNDS = 6` truncates a long binding chain silently).

`SELF_EXCLUDED` was a second, unguarded ledger and is no longer one. It excuses
the two files that carry the corpus as text; as a bare `frozenset` it owed
neither a tracking issue nor a STALE ratchet, so adding any real gate path to it
left every gate green — strictly cheaper than `LEDGER`, which owes both.
`check_self_exclusion` now applies both of `LEDGER`'s rules plus OWNERSHIP: an
entry's file stem must be this checker's or its suite's, derived from
`__file__`, so a gate path is refused outright rather than obeyed.

### 4. The skip must be loud and must name the revision

Every converted gate's skip message names the repo **and** the pinned revision,
so a skipped run says which checkpoint was looked for. This does not fix the
zero-assertion-SUCCESS shape tracked on
[#463](https://github.com/mudler/vllm.cpp/issues/463) — that is a separate defect
across ~40 gates and is not in scope here.

## Tests

| test | what it falsifies |
|---|---|
| `test_hf_snapshot_pinning` (new, CPU-only, no checkpoint needed) | that the resolver picks the pinned revision when a *decoy* revision of the same repo is also cached; that it returns "" (→ skip) when only the decoy is cached; that the env override still works and still requires `config.json` |
| `scripts/check_snapshot_pins.py --self-test` | that the checker fails on a synthesised unpinned resolver |
| `test_narrowing_any_single_branch_drops_a_positive_fixture` (new) | that the corpus's coverage claim is EARNED: it removes each of the 11 `ENUMERATORS` and each of the 5 `CACHE_MARKERS` in turn and requires a positive fixture to notice. An enumerator or marker added without an isolating fixture is RED |
| `test_self_exclusion_refuses_a_path_that_is_not_this_checker` (new) + 3 siblings | that `SELF_EXCLUDED` cannot be used as a second, unguarded ledger: a gate path, a reason with no tracking issue, and an entry whose file no longer carries resolver text are each refused |
| `python3 -m unittest tests.tools.test_online_gate_server_binary` | that the single-`kQwen27NvfP4Revision` assertion and the 35B contract update stay consistent — 20/20 before and after |
| the 3 DFlash + 5 35B gates | build green; SKIP loudly and by name on a box without the pinned revision |

`test_hf_snapshot_pinning` is the direct proof for item 5 of the row and is
deliberately GPU-free: the DFlash gates check `HasCuda()` *before* resolving, so
on a CPU box they can never demonstrate selection. Building a synthetic cache
with **both** the real revision and a decoy is a stronger proof than running on
dgx anyway — it exercises the two-revision case on demand instead of waiting for
the cache to be in the hazardous state.

## Gates

- Clean CPU `Release` rebuild (`-DVLLM_CPP_CUDA=OFF`), `build_exit=0` captured
  alongside every run: a failed build re-runs the stale binary and prints
  SUCCESS.
- Assertion counts recorded before and after for every touched gate. A changed
  count is RED even when the line reads `Status: SUCCESS!`.
- `scripts/agent-preflight.sh --staged` and again on the committed HEAD.

## Stop conditions

- If a revision cannot be derived from a committed record, the gate is **not**
  pinned to whatever is cached — it is recorded as owed on #472 and left in the
  checker ledger.
- No GPU work is queued on `dgx.casa` (it OOM-rebooted at 13:45:45 and an
  operator gate build holds it). dgx is read only, for cache inspection.
  Every runtime claim about a gate actually *running* against a checkpoint is
  therefore UNVERIFIED in this row and is recorded as such.

## Evidence

Host: CPU dev box, **no nvcc, no nvidia-smi**. Build `Release`,
`-DVLLM_CPP_CUDA=OFF`, clean rebuild after the header change. `build_exit=0`
captured alongside every run; zero warnings or errors in the final build log.
Every claim below about a gate *running its body against real weights* is
therefore **UNVERIFIED** — dgx.casa OOM-rebooted and was inspected read-only
only, and no GPU work was queued.

### The resolver, RED then GREEN

`test_hf_snapshot_pinning` mutated by replacing the pinned join in
`HfSnapshot` with exactly the unpinned `directory_iterator` the 56 other files
use. Restoration verified by **checksum, not timestamp** — `cp -p`/`copy2`
preserve mtime and a "restored" tree can silently re-run the mutant binary:

| | source md5 | binary md5 | result |
|---|---|---|---|
| baseline | `cce879c9…` | `18184e6d…` | 4 cases / 19 assertions / SUCCESS / exit 0 |
| mutant | `c7ee1a19…` | `9069b3f0…` | 2 cases FAILED / **5 assertions FAILED** / FAILURE / exit 1 |
| restored | `cce879c9…` | `18184e6d…` | 4 cases / 19 assertions / SUCCESS / exit 0 |

The mutant binary md5 differs from baseline, which is what proves the recompile
actually happened rather than the old object being re-linked.

### Selection and refusal, MEASURED on a synthetic two-revision cache

Both revisions of `unsloth/Qwen3.6-27B-NVFP4` planted under a scratch `$HOME`.
Readdir yields the pinned one first in this cache, i.e. the *unpinned* resolver
would also have "passed" — which is exactly why the decoy-only case is the
load-bearing one.

| case | cache | all three DFlash gates |
|---|---|---|
| A | only `@ccdaab7e` (the FP8 re-quant) + the DFlash draft | SKIP, banner names `@890bdef7` and reports `got: ABSENT`. The repo is present and was **not** substituted. |
| B | `@890bdef7` **and** `@ccdaab7e` | all three resolve `.../snapshots/890bdef7…`; **0 mentions of the decoy** in any output |

In case B `test_qwen27_dflash_spec_decode` proceeds past the resolution and then
fails loading the synthetic (empty) `config.json`, exit 1 — which is the
"reaches its body when the pinned revision is present" half of the proof, on a
box that cannot hold the real 27 GB checkpoint.

### Assertion counts, before and after

Baseline taken by stashing the working tree back to `origin/main` and rebuilding
the same targets in place. Identical on both sides — a changed count would be RED
even under `Status: SUCCESS!`:

| gate | before | after |
|---|---|---|
| `test_qwen3_dflash_draft_parity` | 1 case / 0 assertions | 1 / 0 |
| `test_qwen3_dflash_kvprep_parity` | 1 / 0 | 1 / 0 |
| `test_qwen27_dflash_spec_decode` | 3 / 0 | 3 / 0 |
| `test_qwen36_paged_engine` | 2 / 0 | 2 / 0 |
| `test_qwen36_async_serving` | 1 / 0 | 1 / 0 |
| `test_qwen36_spec_decode` | 1 / 0 | 1 / 0 |
| `test_qwen36_weights` | 7 / 45 | 7 / 45 |
| `test_op_parity` | 10 / 123 | 10 / 123 |
| `test_hf_snapshot_pinning` | — (new) | 4 / 19 |
| `tests.tools.test_online_gate_server_binary` | 20/20 | 20/20 |
| `tests.scripts.test_check_snapshot_pins` | — (new) | 8/8 → 13/13 → **19/19** |

The zero-assertion `Status: SUCCESS!` shape on the checkpoint-gated arms is
**not** repaired here and is not this row's defect; it is
[#463](https://github.com/mudler/vllm.cpp/issues/463), open across ~40 gates.

### Second review: the corpus bound, and the merge-forward

The checker suite went 13 → **19** in the landing pass: six tests added, none
removed and none weakened. The four narrowings that used to pass now fail, and
each was confirmed GREEN on the pre-fix tree first, so the RED is the fix's and
not an artifact:

| single-branch narrowing | before | after |
|---|---|---|
| drop `HF_HUB_CACHE`/`HF_HOME`/`TRANSFORMERS_CACHE` | GREEN | **RED** |
| drop `scandir`/`walk`/`iglob`/`rglob` | GREEN | **RED** |
| drop `listdir` | GREEN | **RED** |
| drop `readdir` | GREEN | **RED** |
| add a gate path to `SELF_EXCLUDED` | GREEN | **RED** |
| delete the `--self-test` leg from `ci.yml` | GREEN | **RED** |

Detected files still equal the ledger exactly (55 = 55) with the 10 new
fixtures in place, so the added coverage introduced no false positive on the
real tree; the 5 shard-iteration negatives are unchanged. `tests/tools` 208/208,
`test_online_gate_server_binary` 20/20.

The merge-forward resolved `tests/parity/hf_snapshot.h` as a UNION — main's
#466 `kQwen27nFp8TowerRevision` beside this row's `kQwen36A3bNvfP4Revision` and
`kQwen27DFlashDraftRevision`, all four defined once and passed to `HfSnapshot`
once. `scripts/agent-preflight.sh` is a keyed record, so its automatic
three-way merge was discarded and main's version taken wholesale with this
row's two lines reapplied.

**UNVERIFIED on this host**: no `nvidia-smi`, no GPU. Every claim about a gate
running its body against real weights remains owed, as does one observed run of
the CI step (#484).

## Outcome

**What was measured.** The defect is real and latent rather than currently
firing: readdir on the gate host yields `@890bdef7` first today, so no committed
DFlash result is known to have been taken against the FP8 re-quant. The row
converts luck into a guarantee.

**What was rejected.** `prefer_single_file` — the heuristic
`test_qwen27_dflash_spec_decode` already carried — was deleted rather than copied
to the other two gates. It selects the right revision today only because
`@890bdef7` happens to be the single-file one, its fallback returns the *last*
entry seen, and it encodes a property the goldens correlate with instead of the
identity they were captured against.

Pinning the remaining 55 resolvers was also rejected, for the reason that bounds the
whole row: 80 of 92 committed goldens record no revision, so their pins would
have to be invented. They are enumerated in the checker ledger and owed on #472.

**Why the DFlash draft pin is set the way it is.** It is a determinism pin on a
committed provenance record, explicitly not a ratified one. `ckpt_keys.txt`
looked like it would ratify it and does not — it is vLLM's loaded-module
namespace, 47 keys against the snapshot's 58. The header says so where a reader
will hit it, and #472 owes the re-capture.

## Now

`ACTIVE` — implemented and gated on a CPU box; the GPU-side "runs against real
weights" arm is UNVERIFIED and owed on a dgx window.
