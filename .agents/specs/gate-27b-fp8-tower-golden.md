# GATE-27B-FP8-TOWER-GOLDEN — a token gate the FP8 tower can actually fail

Issue: [#466](https://github.com/mudler/vllm.cpp/issues/466)
Row: `GATE-27B-FP8-TOWER-GOLDEN`
Gate model (new arm): `nvidia/Qwen3.6-27B-NVFP4` @`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
Gate model (existing arm, unchanged): `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7a42feba6d83b6e17a03315c694112f2a`

## Scope

Add a SECOND checkpoint arm to the dense 27B greedy acceptance gate, pinned to
the `modelopt_mixed` checkpoint whose attention/GDN tower is FP8 W8A8, with its
own goldens captured from the pinned oracle.

**In scope:** a pinned resolver for `nvidia/Qwen3.6-27B-NVFP4`@`0893e160` in
`tests/parity/hf_snapshot.h`; goldens under
`tests/parity/goldens/qwen36_{embed,gdn_layer,fullattn_layer,norm,logits}_27n/`
(the paged-engine arm consumes `logits`; the other four are captured for the
op-parity follow-on); a new parity test that greedy-decodes
that checkpoint through the full paged engine and asserts the FP8 GDN
input-projection dispatch contract **unconditionally**; a skip that cannot be
read as coverage.

**Out of scope, recorded as owed:** wiring the new gate into
`tools/bench/online_gate.py::MODEL_GATE_CONTRACTS` so the `27n` performance key
stops recording a build-sanity precondition (needs a `27n` online-gate run to
validate, which this row does not take); pinning the three DFlash tests that
still resolve the unsloth 27B repo by unpinned `directory_iterator`; pinning the
four 35B tests likewise. Each is a separate row.

## The hole, VERIFIED not inferred

`tests/parity/hf_snapshot.h:31,59-62` resolves the dense 27B gate to
`unsloth/Qwen3.6-27B-NVFP4`@`890bdef7`. Every fp8-tower lever targets
`nvidia/Qwen3.6-27B-NVFP4`@`0893e160`. Read off the two cached snapshots on the
gate host:

| | `unsloth`@`890bdef7` | `nvidia`@`0893e160` |
|---|---|---|
| `quant_method` | `compressed-tensors`, `nvfp4-pack-quantized` | `modelopt_mixed` |
| MLP | NVFP4 **W4A4** (`input_activations.num_bits=4`, dynamic local) | `W4A16_NVFP4`, group 16 |
| `linear_attn.in_proj_{qkv,z,a,b}` | listed in `ignore` → **BF16** | `in_proj_qkv`, `in_proj_z` = **FP8** |
| `linear_attn.out_proj` | NVFP4 (`weight_packed`) | **FP8** |
| `self_attn.{q,k,v,o}_proj` | NVFP4 | **FP8** |
| `*.input_scale` tensors present | **0** | present |

The checkpoints are not two spellings of one model. On `@890bdef7` the FP8 tower
does not exist, so no amount of correctness pressure on the SACRED gate can reach
an fp8 code path.

The gate concedes this in place. `tests/parity/test_qwen27_paged_engine.cpp:227`
guards the entire FP8 dispatch-count contract on
`if (fp8_inproj.Total() != 0)`, and the comment above it states *"A BF16-owner
checkpoint issues neither and totals 0."* On the checkpoint this gate pins, that
predicate is always false: the contract never executes, prints nothing, and the
case still reports success. An unexercised contract that is indistinguishable
from a satisfied one is the defect this row closes.

`.agents/specs/cuda-online-serving-gate.md:89,101-106` already records the debt
in prose — `27n`'s correctness precondition is *"build sanity only, never a
golden for @0893e160"*, and a `27n` correctness claim *"owes a greedy
continuation against the pinned oracle on @0893e160 itself"*. This row is that
owed continuation.

## Design

### 1. Resolver

`tests/parity/hf_snapshot.h` gains `kQwen27nFp8TowerRevision` and
`Qwen27nFp8TowerSnapshot()`, built on the existing `HfSnapshot` helper so the
skip-not-substitute discipline is inherited unchanged: a cache holding some other
revision of the same repo returns `""` rather than being gated.

The constant is deliberately NOT named `kQwen27NvfP4Revision*`.
`tests/tools/test_online_gate_server_binary.py:613-628` asserts exactly one
`kQwen27NvfP4Revision` pin exists in the header and that it equals
`MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"]["golden_revision"]`. That
assertion is correct and must keep passing untouched; a second pin under a
distinct name does not disturb it.

### 2. Goldens

Captured by the EXISTING recipe, `tools/parity/dump_qwen36.py`, with `--tag
27n`. Its MEASURED behaviour is unmodified; only the manifest's provenance
strings changed (see `## Outcome`), because the old hardcoded `detail` asserted a
weight layout that is false for this checkpoint. Same prompt (`"The capital of France is Paris, and the"`), same
`N_GREEDY = 16`, same `TOPK = 1000`, same `SamplingParams(temperature=0.0)`, same
`LLM(enforce_eager=True, max_model_len=256, max_num_seqs=1, dtype="bfloat16")`,
same manifest emitter. The engine config is load-bearing, not incidental:
`.agents/specs/pin-advance.md:456-474` shows the same oracle emits a different
tok6 under `max_model_len=8192, max_num_seqs=4` — config decides the near-tie,
not oracle version. Nothing about the capture is invented for this row.

Oracle: `~/venvs/vllm-oracle-next` asserted to report a version containing
`555967922` before the capture runs, and aborting otherwise. The canonical
`~/venvs/vllm-oracle` symlink currently points at
`~/venvs/vllm-oracle-v0.25.0-stage` — the ROLLBACK (issue #375) — so this row
addresses the oracle by its explicit path and never by the symlink.

No emulation sidecar. `greedy_ids_emulation.npy` exists on the `@890bdef7` arm
because that checkpoint has a documented W4A4 tok6 whitespace near-tie between
the production and emulation NVFP4 kernels;
`tools/parity/dump_27b_emulation_greedy.py` hardcodes `EXPECT_TOK6 = 271` for it.
`@0893e160` is W4A**16** with an FP8 tower — a different kernel set and a
different near-tie question — so importing that fixture would be asserting a
property nobody measured. The new arm therefore gates on `greedy_ids` alone and
records the absent negative control as a limitation.

### 3. The new arm

`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp`, a sibling of the existing
gate, NOT a replacement. `@890bdef7` keeps its goldens, its assertions, and its
proof line, byte for byte.

The new arm differs from its sibling in exactly the places the checkpoint differs:

- The FP8 GDN input-projection dispatch contract is asserted
  **unconditionally**. `Total() == 0` is a FAILURE here — it is the precise
  signature of gating a checkpoint with no FP8 tower, which is how this hole
  stayed open. That single inverted guard is the whole point of the row.
- Packed GDN decode must issue ZERO launches, asserted against the REAL
  predicate rather than `detail::PackedGdnDecodeEnvSelected`, which mirrors the
  ENV couplings only and would demand 48 here (#470).
- No `greedy_ids_emulation` comparison (§2).

### 4. A skip that cannot be mistaken for coverage

The sibling prints a `MESSAGE` and returns — the case then reports **success with
zero assertions**, and `ctest` exits 0. That is what let an absent instrument read
as a pass. The new arm's absent-checkpoint path:

- prints a banner naming the row, the repo, the exact revision, and the words
  `NO FP8 TOWER COVERAGE`, so the intent cannot be reconstructed as a pass by a
  reader skimming a log;
- records at least one assertion in the skip path, so a skipped run is
  distinguishable from a covered run by the assertion count alone, which is the
  signal a gate reader actually diffs;
- prints its proof line ONLY after tokens have been compared, mirroring the
  existing `MODEL_GATE_CONTRACTS` proof discipline.

## Gates

Focused: `ctest -R qwen27n_fp8_tower --output-on-failure -V`, on the dgx
production build (`-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0
-DVLLM_CPP_TRITON=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
-DCMAKE_BUILD_TYPE=RelWithDebInfo`), which the gate itself refuses to run
without.

RED-before: the arm must FAIL under a mutation of the FP8 path it claims to
cover, proven by a changed assertion count and a non-zero exit — not by reading
the code. Restored byte-for-byte afterwards, verified by `git status` and a
`git diff --stat` that is empty.

The `@890bdef7` arm must remain at its established count in the same run, proving
the addition is additive.

## Risks

- **A near-tie on the new checkpoint.** If `@0893e160`'s greedy continuation sits
  on a tie the way `@890bdef7`'s tok6 does, the arm could be checkpoint-stable but
  build-sensitive. Mitigated by capturing with the same knobs the sibling uses and
  by recording the oracle's own continuation verbatim; a tie discovered later is a
  finding, not a reason to adjust a golden.
- **Unified-memory pressure.** GB10 memory is unified, so the capture's
  `gpu_memory_utilization` reserves HOST RAM. Capture and gate runs take the GPU
  mutex and never overlap another model job.

## Stop conditions

- The oracle cannot load `@0893e160` → STOP and report. Never substitute a
  different revision; that is the failure mode this row exists to close.
- The gate host's GPU lock is held by another agent's campaign → hand back rather
  than contend.

## Outcome

`DONE`. The hole was real and is closed. What was measured, what was rejected,
and what the arm still cannot see:

### Both arms, same production build, dgx (GB10 sm_121a)

`RelWithDebInfo`, `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0
-DVLLM_CPP_TRITON=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a`; configure log verified
to print `CUTLASS found … enabling sm120a NVFP4 cutlass GEMM`,
`FlashAttention-2 prefill/decode: ENABLED`, and the `sm_121a` Triton-AOT lines.
`ctest -j 1`, each arm alone, under `flock $HOME/gpu.lock`.

| run | `qwen3_5_weights.cpp` | 27n arm | 27b arm (SACRED) |
|---|---|---|---|
| base | pristine | 236 asserts, **SUCCESS** | 235 asserts, SUCCESS |
| mutant 1.02x | fp8 scale x1.02 | 236 asserts, SUCCESS | 235 asserts, SUCCESS |
| mutant 1.10x | fp8 scale x1.10 | 236 asserts, SUCCESS | 235 asserts, SUCCESS |
| **mutant 2.0x** | fp8 scale x2.0 | **236 asserts, 1 FAILED, ctest EXIT=8** | **235 asserts, SUCCESS** |
| restored | pristine, forced recompile | 236 asserts, SUCCESS | 235 asserts, SUCCESS |

The 2.0x row is the whole thesis in one binary. `LoadFp8RawShared` is the single
external-linkage seam through which every FP8 GDN in_proj / out_proj /
attention projection is loaded, and it is entered only for a projection whose
on-disk dtype is `F8_E4M3`. Under the mutation the 27n arm printed five
`[VT_RED_MUTATION]` reachability lines (layer 0 `in_proj_qkv`, `in_proj_z`,
`out_proj`) and failed on `CHECK(got == want_prod)`. The SACRED arm printed
**zero** — the mutated code was never called at all on its checkpoint — and
passed 235/235. An fp8 defect that the SACRED gate structurally cannot observe
is now observed.

### The arm's SENSITIVITY, honestly

A uniform **1.10x** perturbation of every FP8 weight scale and folded alpha is
REACHED (five reachability lines) and still emits the identical 16 tokens. So
this arm catches a gross fp8 defect and does NOT catch a ~10% per-tensor scale
error. It is a token gate on one short prompt, not a numerical-tolerance gate,
and it should never be quoted as if it bounded fp8 numerics.

The fix for that is already paid for: this row captured
`qwen36_{embed,gdn_layer,fullattn_layer,norm,logits}_27n` — per-tensor goldens
with `atol/rtol` 1e-3 to 5e-2 and top-1000 logits — and only the `logits`
directory is consumed so far. Giving `test_op_parity`'s Qwen27 arms a 27n
counterpart against the committed tensors would catch the 1.10x case. Owed,
not done here.

### The token stream does not discriminate the checkpoints

`@0893e160` and `@890bdef7` emit the SAME 16 greedy tokens on this prompt
(`[6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 9338, 369, 11751, 11,
321, 279]`), and the same 9 prefill argmaxes. The token comparison alone would
therefore NOT notice a wrong-checkpoint run. Two things carry that instead, and
both are load-bearing rather than decorative: the revision-pinned resolver, and
the unconditional `Total() != 0` fp8 dispatch assertion — which is the only
thing in either arm that can tell the two checkpoints apart at runtime.

### Skip behaviour, measured not asserted

| invocation | assertions | status | exit |
|---|---|---|---|
| checkpoint present | 236 | SUCCESS | 0 |
| absent | **1** | SUCCESS | 0 |
| absent + `VT_REQUIRE_27N_FP8_GATE=1` | **0**, `test case THREW` | **FAILURE** | **1** |

All three print the `NO-FP8-TOWER-COVERAGE` banner naming the exact revision.

### Capture reproducibility

Captured twice from two independent oracle loads of
`0.23.1rc1.dev1511+g555967922` (identity asserted before each load). Every
`.npy` is byte-identical between the two runs; only `manifest.json` differs, by
the provenance fix below.

### Rejected / repaired along the way

- **`dump_qwen36.py`'s hardcoded `detail`.** It asserted "GDN in_proj, embed,
  norms, lm_head are bf16" — true of `@890bdef7`, FALSE of `@0893e160`. A
  manifest whose job is to say which model a golden belongs to must not state
  the wrong layout for the next checkpoint captured, so `detail` now records
  the engine-resolved `quantization` and `kv_cache_dtype`, and `--pin` records
  the oracle revision actually run instead of inheriting `e24d1b24`. Measured
  content, knobs and schema unchanged.
- **`detail::PackedGdnDecodeEnvSelected` is NOT used by this arm.** It mirrors
  the ENV couplings only; the real predicate additionally requires
  `in_proj_qkv_fp8.Empty()` (`qwen3_5.cpp:4161-4165`). On an fp8 tower the mirror
  would demand 48 packed launches where the truth is 0, so the sibling gate
  would throw for the wrong reason if ever pointed here via
  `VT_QWEN27_SNAPSHOT`. This arm pins the real behaviour (0) directly. The
  mirror's drift is filed as [#470](https://github.com/mudler/vllm.cpp/issues/470).
- **Two harness traps hit and recorded, because both FAIL OPEN silently.**
  (1) `set -e` is disabled inside a function whose status is tested by `if`, so
  the first capture's checkpoint-identity guard printed its own failure message
  and carried on; the guard was re-run standalone afterwards and the checkpoint
  is confirmed (96 FP8 `linear_attn.in_proj`, 48 `out_proj`, 192 W4A16 MLP).
  (2) `shutil.copy2` restores the backup's ORIGINAL mtime, leaving the mutant
  object NEWER than the restored source: the first "restored" run silently
  re-used the mutant binary and reported GREEN. Caught by a reachability-line
  count, not by reading the code. Restoration was only accepted after a forced
  recompile and zero `[VT_RED_MUTATION]` lines.

### Owed, explicitly

- `tools/bench/online_gate.py::MODEL_GATE_CONTRACTS` still records
  `test_qwen27_paged_engine` as `27n`'s precondition, i.e. build sanity on a
  neighbour. Repointing it at this arm needs a `27n` online-gate run to
  validate and is a separate row.
- The three DFlash tests and the four 35B tests still resolve their snapshot by
  unpinned `directory_iterator`; the DFlash trio reads the very repo known to
  hold two revisions.
- 27n op-parity arms against the four other committed 27n golden directories.

## Now

`DONE` — arm landed with RED-mutation evidence; follow-ons above are separate
rows.
