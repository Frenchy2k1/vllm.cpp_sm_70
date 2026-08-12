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
`tests/parity/goldens/qwen36_logits_27n/`; a new parity test that greedy-decodes
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

Captured by the EXISTING recipe, `tools/parity/dump_qwen36.py`, unmodified, with
`--tag 27n`. Same prompt (`"The capital of France is Paris, and the"`), same
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
- The engine's own quantization ownership is asserted before decoding: the
  loaded model must report FP8 owners for the GDN input projections.
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

## Now

`ACTIVE` — spec committed, implementation and evidence follow on this branch.
