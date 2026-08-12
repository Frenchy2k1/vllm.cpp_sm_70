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
- A new checker `scripts/check_snapshot_pins.py` that fails on any *new*
  unpinned checkpoint resolution under `tests/`.

**Out of scope, recorded as owed on [#472](https://github.com/mudler/vllm.cpp/issues/472).**
The ~48 remaining unpinned gates whose goldens record no revision at all
(19 `*_greedy*` corpora carry no manifest file whatsoever). They cannot be pinned
from evidence without re-capturing the goldens, and "pin to whatever is cached
here" is the defect wearing a constant's name. They are enumerated in the
checker's ledger, each of which is a line that must be *deleted*, never added to.

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

`scripts/check_snapshot_pins.py` scans `tests/**/*.{cpp,h}` for a
`directory_iterator` whose subject path contains `snapshots`, and fails on any
occurrence not present in an explicit ledger inside the checker. The ledger is a
**shrinking** list: every entry names the file and the issue that owes its
golden's provenance (#472). Adding a line is a review event; deleting one is the
work. This is the same shape as the existing device-leakage allowlist
(issue #302) and is admissible under the record rules because it is written only
when an unpinned resolver is *added* — which is what it exists to prevent — and
not by every PR.

The checker ships with its own RED proof: a self-test that synthesises an
unpinned resolver in a scratch tree and asserts the checker fails on it, so the
checker cannot be turned green by weakening its own pattern.

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

## Now

`ACTIVE` — spec committed ahead of implementation.
