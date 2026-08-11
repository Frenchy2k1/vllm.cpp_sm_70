# FUSION-DENSE-MIGRATE — fold the five no-blocker dense SwiGLU MLPs onto the merged-GEMM seam

Issue: [#299](https://github.com/mudler/vllm.cpp/issues/299).
Owning row: `ROAD-V1-C1` (punch-list item 15, "route 5 drift models", in
[`roadmap-v1-completion.md`](roadmap-v1-completion.md)); shared-op history in
[`arch-fusion-fold-plan-2026-07-30.md`](arch-fusion-fold-plan-2026-07-30.md) §A1 and
[`fusion-consistency-audit.md`](fusion-consistency-audit.md).
Claim: `CLAIM-FUSION-DENSE-MIGRATE`.

## Scope

**IN — exactly five model TUs**, each a plain bf16 dense SwiGLU MLP that today
hand-rolls `{ResidentWeight(gate_up); MatmulBT[2I,H]; vt::SiluAndMul}` and therefore
sits on `scripts/merged-gemm-consistency-allowlist.txt` with the sole reason
`pending FOLD-MIGRATE`:

| Stem | Function | Weights struct |
|---|---|---|
| `commandr` | `CommandrMlpBlock` | `CommandrMlpWeights` |
| `glm4` | `Glm4MlpBlock` | `Glm4MlpWeights` |
| `minicpm` | `MiniCPMMlpBlock` | `Qwen3DenseMlpWeights` |
| `minicpm3` | `MlpBlock` (MLA arch) | `Qwen3DenseMlpWeights` |
| `phi3` | `Phi3MlpBlock` | `Qwen3DenseMlpWeights` |

Each routes its gate/up through `layers::UnquantizedMlpGateUpMethod` and loses its
allowlist entry.

**Mechanical correction to #299's stated end state.** The issue predicted the checker
would go from `6/15 routed` to `11/15 routed`. It cannot, and no earlier fold ever did:
`scan_models_gemm` only enters a TU that still HAND-CALLS `vt::SiluAndMul`/`GeluAndMul`,
and a fully-folded TU has no hand-call left, so it leaves the DENOMINATOR rather than
entering the numerator. (That is why `qwen3`, `olmo2`, `stablelm`, `granite` and
`deepseek_v2` — all folded by A1 — do not appear in the checker's count at all today.)
The real end state is `15 scanned / 6 routed / 11 allowlisted` →
`10 scanned / 6 routed / 6 allowlisted`. What matters, and what the gate asserts, is
identical either way: drift is 0 and the allowlist holds only entries that state a
blocker. The checker's semantics are NOT changed to make the issue's number come true.

**OUT — every other allowlist entry**, each of which names a real blocker that can
only be cleared by EXTENDING the shared layer (its own row, not this one):
`gemma4_vision` and `gemma4_moe` (GeGLU + clamp epilogue / PLE shared expert),
`laguna` (resident/graph decode ownership), `minimax_h3_device` (weight residency:
up-front device staging vs `OwnedTensor`/`ResidentWeight`),
`minimax_h3_video_vae_device` (f32-end-to-end activations + rank-1 biases the
bias-free method has no slot for), `minimax_h3_encoder_device` (ggml block-quant
weights against the explicitly UNQUANTIZED arm; gate/up not always merged).

Also OUT: the GLUE allowlist `scripts/fusion-consistency-allowlist.txt`. `glm4` and
`phi3` appear there too, for a DIFFERENT drift (add+RMSNorm → `kFusedAddRmsNormStd`);
that is the same historical follow-on name but a distinct check, and #299 scopes this
row to the merged-GEMM half only. Both of those entries read `pending
FUSION-DENSE-MIGRATE`, so closing this row would have left them pointing at closed work
— the exact stale-`pending` shape that let the merged-GEMM allowlist grow to hold most
of its population. They are therefore REPOINTED at a new issue,
[#314](https://github.com/mudler/vllm.cpp/issues/314), which owns the glue half. That
repoint is a comment correction this row creates the need for; the glue FOLD itself is
not done here.

**OUT:** any change to `include/vllm/model_executor/layers/linear.h`,
`.../schemes/nvfp4.h`, or any `vt::` op. The seam TU is untouched, so the 27B/35B/
qwen3-dense gates cannot move.

## Upstream chain

Pinned oracle `/home/mudler/_git/vllm` @ `555967922` (0.26.0.dev0), the pin recorded
in [`upstream-sync.md`](../upstream-sync.md).

Upstream expresses every one of these five as ONE `MergedColumnParallelLinear`
(`gate_up_proj`, `output_sizes=[intermediate_size] * 2`) followed by `SiluAndMul()` —
i.e. upstream has ALREADY merged the pair, which is exactly what our seam mirrors.

| Ours | Upstream `file:line` |
|---|---|
| `commandr` | `vllm/model_executor/models/commandr.py:91` `CohereMLP`; `:102-108` `MergedColumnParallelLinear`; `:116` `SiluAndMul()`; `:118-122` forward |
| `glm4` | `vllm/model_executor/models/glm4.py:46` `from .llama import LlamaMLP as Glm4MLP` → `llama.py:79` `LlamaMLP`, `:92-99` merged gate_up, `:113` `SiluAndMul()`, `:115-119` forward |
| `minicpm` | `vllm/model_executor/models/minicpm.py:193` `MiniCPMMLP`; `:204-211` merged gate_up; `:219` `SiluAndMul()` |
| `minicpm3` | `vllm/model_executor/models/minicpm3.py:186` `MiniCPM3DecoderLayer(MiniCPMDecoderLayer)` — inherits `MiniCPMMLP` verbatim (`minicpm.py:193`) |
| `phi3` | `vllm/model_executor/models/phi3.py:10` `Phi3ForCausalLM(LlamaForCausalLM)` → `llama.py:79` `LlamaMLP` |

`MergedColumnParallelLinear` itself: `vllm/model_executor/layers/linear.py`
(`MergedColumnParallelLinear`, the `output_sizes` concatenation), which
`layers::MlpGateUpMethodBase` mirrors — with the activation kept inside the method so
a quantized scheme may fuse GEMM+activation (`dense_nvfp4::GateUpFusedMarlinD`), a
thing a plain linear + separate `SiluAndMul` cannot express.

## Our baseline

- Seam: `include/vllm/model_executor/layers/linear.h:82` `MlpGateUpMethodBase`, `:90`
  `UnquantizedMlpGateUpMethod` (one `ResidentWeight` + one `MatmulBT[2I,H]` +
  `vt::SiluAndMul`), `:122` the GeGLU sibling; factory
  `include/vllm/model_executor/layers/quantization/compressed_tensors/schemes/nvfp4.h:104`
  `MakeMlpGateUpMethod`.
- Byte-exactness of the seam vs the standalone sequence is ALREADY gated:
  `tests/vllm/model_executor/layers/test_linear_method.cpp:304` (RED-first proven by
  the A1 fold, `18ed6f03`).
- Six TUs route today: `qwen3`, `granite`, `olmo2`, `stablelm`, `qwen3_dflash`,
  `deepseek_v2` (A1, `18ed6f03`) plus the Gemma family via the GeGLU arm — the
  checker reports 6 of the 15 scanned gated-MLP TUs.
- The five targets carry a LITERALLY IDENTICAL five-statement body; none of them
  references `*_fp4` weights anywhere, so all five are bf16-only paths.
- Honest gap: all five own checkpoint-gated, dgx-only paged-engine gates
  (`tests/parity/test_{commandr,glm4,minicpm,minicpm3,phi3}_paged_engine.cpp`), which
  SKIP on a CPU box. This row is executed CPU-only.

## Port map

No upstream file is newly ported; this is a local routing fold onto an
already-ported seam. Per TU the change is: `#include
"vllm/model_executor/layers/linear.h"` + replace the three-statement gate-up
sequence with one `layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, x)`.

| File | Site |
|---|---|
| `src/vllm/model_executor/models/commandr.cpp` | `CommandrMlpBlock` |
| `src/vllm/model_executor/models/glm4.cpp` | `Glm4MlpBlock` |
| `src/vllm/model_executor/models/minicpm.cpp` | `MiniCPMMlpBlock` |
| `src/vllm/model_executor/models/minicpm3.cpp` | `MlpBlock` |
| `src/vllm/model_executor/models/phi3.cpp` | `Phi3MlpBlock` |
| `scripts/merged-gemm-consistency-allowlist.txt` | five entries removed |
| `tests/scripts/test_check_fusion_consistency.py` | new regression assertion pinning the five |

Deviation from the A1 precedent: Granite used the `MakeMlpGateUpMethod` FACTORY
because its weights are `Qwen3DenseMlpWeights` (carrying `gate_proj_fp4`). `minicpm`,
`minicpm3` and `phi3` share that struct, but none of their loaders or forwards ever
populates or reads `*_fp4`, so the factory would resolve to the unquantized arm on
every checkpoint that exists while adding a heap allocation and an untestable
quantized branch. All five therefore take the direct
`UnquantizedMlpGateUpMethod` arm (the olmo2/stablelm/qwen3_dflash precedent), which
is byte-identical by construction. Promoting the three to the factory is a separate,
gateable step once an nvfp4 checkpoint for any of them exists.

## Tests to port

Nothing new from upstream: `MergedColumnParallelLinear`'s own upstream tests
(`tests/kernels/test_layernorm.py` neighbours, `tests/distributed/test_pynccl.py`
style shard tests) are TP-sharding tests with no single-GPU analogue, and the
activation identity is already covered by the ported byte-exact case at
`tests/vllm/model_executor/layers/test_linear_method.cpp:304` (which anchors
`tests/compile/passes/test_fusion.py` oracle discipline: raw byte compare, not
`assert_close`).

Added locally, both RED-first proven: a byte-exact case in
`tests/vllm/model_executor/layers/test_linear_method.cpp` for the DIRECTLY-constructed
`UnquantizedMlpGateUpMethod` (the spelling all five folds use, as distinct from the
factory the existing case covers) at BOTH the decode shape `M == 1` and a prefill shape
`M == 4`, byte-compared against the standalone `{ResidentWeight; MatmulBT[2I,H];
SiluAndMul}` sequence; and two regression cases in
`tests/scripts/test_check_fusion_consistency.py` asserting that each of the five
folded stems (a) is scanned by the merged-GEMM detector, (b) routes a shared seam,
and (c) is NOT on `scripts/merged-gemm-consistency-allowlist.txt` — so re-allowlisting
a folded model, or reverting a fold, goes RED instead of silently re-opening the
drift the allowlist was emptied of.

Added after review (finding F4 — there was NO executed coverage of the change
surface, proven by mutating `phi3`'s `I` to `I - 1` at the call site and watching it
survive 176 CPU tests): `tests/vllm/models/test_dense_gate_up_seam_forward.cpp`
drives the REAL forward of four of the five folded TUs (`commandr`, `glm4`,
`minicpm`, `phi3`) over synthetic in-memory weights on CPU — no checkpoint, no GPU —
and pins the gate/up split analytically rather than against a golden. The review
also rejected this row's original "no e2e evidence is possible on a CPU box" framing
as too strong: *oracle* evidence needs the GPU, *self-consistency* evidence does
not. `minicpm3` is the folded TU this harness does not drive: its MLA attention plus
the load-time `kv_b_proj` -> W_UK/W_UV absorption belong to the DeepSeek-V2 synthetic
harness, not the dense one, and it shares the identical `Qwen3DenseMlpWeights` call
shape with `minicpm` and `phi3`, both covered.

## Gates

CPU-only (no GPU available to this claim), foreground:

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-cpu -j 18
python3 scripts/check-fusion-consistency.py       # 0 drift; allowlist 11 -> 6 entries
python3 -m unittest tests.scripts.test_check_fusion_consistency -v
./build-cpu/tests/test_linear_method               # the seam byte-exactness gate
./build-cpu/tests/test_dense_gate_up_seam_forward # the folded TUs, really executed
ctest --test-dir build-cpu -j 6 --output-on-failure
```

Token-exactness argument (this is a ROUTING change, and the gate is the identity of
the op sequence, not a tolerance):

1. `UnquantizedMlpGateUpMethod::Apply` is `{ResidentWeight(*gate_up); DBuf[M,2I];
   MatmulBT; DBuf[M,I]; SiluAndMul}` with `M = x.shape[0]`. Each replaced body is
   that same sequence with `M` spelled `T`, and at every call site the input is a
   `DBuf(d, kBF16, {T,H})`, so `x.shape[0] == T` identically.
2. `test_linear_method`'s byte-exact case proves the seam equals the standalone
   sequence bit-for-bit on the CPU backend (RED-first proven at `18ed6f03`).
3. The seam TU is not edited, so no already-routed model can move either.

Dgx-only paged-engine gates (`test_{commandr,glm4,minicpm,minicpm3,phi3}_paged_engine`)
are the SACRED token-exact bar for these archs; they emit a loud SKIP on a CPU box.
Running them is OWED to whoever next holds the GPU and is recorded as PENDING, not as
passed — tracked by [#337](https://github.com/mudler/vllm.cpp/issues/337), so the
handle outlives #299 rather than living only in this prose.

## Dependencies

`KERNEL-FUSION-FRAMEWORK` (seam exists, A1 landed). No toolchain, model artifact,
hardware or license dependency. No other in-flight row owns these five TUs.

## Work breakdown

- W1 — spec (this file) + the #299 roadmap issue-table row. Committed alone, first.
- W2 — the five folds + allowlist shrink + the checker regression assertion + records.
  Single commit; the five folds are the same edit five times and splitting them would
  leave the allowlist and the checker count disagreeing with the tree.

## Risks/decisions

- **Decision: direct arm, not the factory.** See Port map. Keeps the change
  provably byte-identical on the only checkpoints that exist.
- **Decision: no adoption env flag.** A1 set the precedent — the fold is
  unconditional and bit-exact, so a `VT_*` rollback branch would add a dead
  same-binary arm nobody can gate.
- **Risk: a target turns out to have a real blocker.** Mitigation: leave it
  allowlisted and rewrite its comment to state the ACTUAL blocker rather than
  `pending FOLD-MIGRATE`. An honest allowlist entry beats a forced fold.
- **Risk: CPU-only evidence.** Named and carried; the dgx paged-engine gates are
  recorded as OWED rather than claimed.
- No product decision is opened; upstream defines all five as merged gate_up +
  SiluAndMul and this fold moves toward that shape.

## Evidence

Recorded in [`parity-ledger.md`](../parity-ledger.md) and in the
`KERNEL-FUSION-FRAMEWORK` row of [`kernel-matrix.md`](../kernel-matrix.md).
`git log --grep FUSION-DENSE-MIGRATE` is the history.

## Stop conditions

- A fold changes numerics (any test that was green goes red, or `test_linear_method`
  disagrees): STOP, report it as a finding, do NOT widen a tolerance.
- The checker cannot reach 11 routed / 6 allowlisted without touching an OUT-of-scope
  entry: STOP and report; do not extend the shared layer under this row.
- A serial re-run of a failing ctest binary still fails on a clean `main` tree:
  that failure is not this row's; report both numbers and stop chasing it.

## Outcome

**All five folded. No target turned out to have a real blocker.** Each of
`commandr`, `glm4`, `minicpm`, `minicpm3` and `phi3` now spells its gate-up as one
`layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, x)`, and all five
entries are gone from `scripts/merged-gemm-consistency-allowlist.txt` (11 -> 6
entries, every survivor naming a shared-layer blocker).

**Measured, on the CPU build (`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF
-DVLLM_CPP_METAL=OFF`, Release, `-j 18`, clean configure, foreground):**

| Gate | Before | After |
|---|---|---|
| `check-fusion-consistency.py` merged-GEMM | 15 scanned / 6 routed / 11 allowlisted, 0 drift | 10 scanned / 6 routed / 6 allowlisted, 0 drift |
| `check-fusion-consistency.py` glue | 13 scanned / 11 routed / 2 allowlisted | unchanged |
| `test_linear_method` | 5 cases / 65 assertions | 6 cases / 76 assertions |
| `test_check_fusion_consistency` | 18 tests | 20 tests |

Full CPU `ctest -j 6`, re-run on the rebased base `60e71a0e` (the final base `c70f42b9` adds only
record commits plus one unrelated parity test, re-gated focused: fusion checker 0
drift, `test_check_fusion_consistency` 20/20, `test_linear_method` 76/76):
**369 / 369 passed, 0 failed**, 1288.75 s.

An earlier `-j 6` run on the intermediate base `688eea12` read **367 / 369** with
`test_async_llm` and `test_openai_conformance` red, and BOTH numbers are reported
because that run is the honest one to explain rather than delete. It executed while
a SECOND worktree (`/home/mudler/_git/vllm.cpp-land-ext`) ran its own suite and the
box sat at load average 89-122. Re-run alone on a quiet box (load 12.8) they passed
in **0.04 s** and **20.76 s**, against 605 s of `statuses == -1` — the client's 30 s
read timeout — under load. Neither could have been this row's:
`test_openai_conformance` builds a synthetic `Qwen3_5MoeForConditionalGeneration`
in-process (`test_conformance.cpp:154`) and loads none of the five folded TUs. The
clean 369/369 on the final base settles it. The five ASan/UBSan failures of #274 did
not appear in either run — this gate is a Release build with no sanitizer.

**Why the issue's predicted number is unreachable, and was not chased.** #299 said
the checker would read `11 routed / 15`. It cannot: `scan_models_gemm` enters a TU
only while that TU still HAND-CALLS `vt::SiluAndMul`/`GeluAndMul`, so a fully-folded
TU leaves the denominator instead of entering the numerator — which is why the five
A1 folds (`qwen3`, `olmo2`, `stablelm`, `granite`, `deepseek_v2`) appear nowhere in
today's count either. Reaching `11/15` would have meant widening the detector to
count folded models, i.e. changing a checker's semantics to make a prediction come
true. Rejected. Drift is 0 and the allowlist holds only stated blockers, which is
what the gate actually asserts.

**Full CPU `ctest -j 6` on the MERGED head `4b99cefb`: 369 / 370 passed, 1
failed**, 1441.85 s (370 not 369 — this branch adds one binary). The single
failure is `test_engine_core_proc`, and it is a starvation flake, not a
regression: re-run SERIALLY on the same binary it is **10 / 10 cases, 93
assertions GREEN**. Its signature is the giveaway — under `-j 6` at load average
~170 (two other worktrees running their own suites concurrently) it failed in
0.06 s on `CHECK(abort_seen)` at `test_engine_core_proc.cpp:345` having spun to
**1089 assertions** waiting for the abort to be observed, against 93 assertions
when it runs alone. It is one of the four known starvation-prone binaries, it
loads none of the five folded TUs, and the other three (`test_async_llm`,
`test_openai_conformance`, `test_openai_api_server`) all passed in this run.
`test_dense_gate_up_seam_forward` passed inside the suite too (32.96 s), which
also proves the new binary is registered and wired.

**RED-first, executed, not asserted.** Three mutations were run and each was
restored byte-for-byte (`md5sum` verified against the pre-mutation digest):

1. Seam activation `SiluAndMul` -> `GeluAndMul` in `linear.h`, rebuilt: the new
   byte-exact case FAILS — `test_linear_method` 6 cases / 76 assertions -> 4 passed /
   2 failed, 11 assertions failed. Restored: 76/76 green.
2. `phi3` re-added to the merged-GEMM allowlist: `test_check_fusion_consistency`
   20/20 -> 2 failures (`test_folded_dense_models_stay_folded`,
   `test_refolded_model_reappearing_unfolded_would_fail`).
3. The `phi3` fold reverted **as a whole hunk** — the five pre-fold statements
   restored AND the explanatory comment (which names
   `layers::UnquantizedMlpGateUpMethod`) removed with them: the CHECKER ITSELF goes
   RC=1 naming `phi3.cpp (1 gated-MLP epilogue hand-call site(s))`, and the unit
   suite drops 2. **The scope of that claim, corrected after review:** a CODE-ONLY
   revert that leaves the comment in place leaves the shipped checker at **RC=0**
   with 1 unit failure, because `_MERGED_GEMM_SEAM` matches the seam token anywhere
   in the file, comments included, and the fold added that token to each TU's
   comment (e.g. `phi3.cpp:49`). Verified by executing both reverts. That is the
   detector working as documented — its docstring calls the seam token's presence
   in the file the adoption signal, precisely so a rollback hand-call in an `else`
   branch still reads as adopted — so the checker was NOT changed to make the
   stronger claim true; the claim was narrowed to what was actually run. The
   realistic regression (an entry coming back onto the allowlist) is caught by the
   unit suite's `assertNotIn(stem, scanned)` guard either way.

**Two honest limits on the token-exactness claim.**

- *No ORACLE token evidence exists on this box* (narrowed after review: the original
  wording said no end-to-end evidence was possible on a CPU box at all, which is too
  strong — oracle evidence needs the GPU, self-consistency evidence does not, and
  `test_dense_gate_up_seam_forward` now supplies the latter for four of the five
  folded TUs). Each folded arch's SACRED gate is
  `tests/parity/test_<arch>_paged_engine.cpp`, checkpoint-gated and dgx-only; all
  five run here as a loud SKIP (1 case, **0 assertions**). Running them is OWED to
  the next GPU holder ([#337](https://github.com/mudler/vllm.cpp/issues/337)) and is
  recorded as PENDING, never as passed. The proof that stands in the meantime is
  op-sequence identity: each replaced body WAS the seam's
  own `{ResidentWeight; MatmulBT[2I,H]; SiluAndMul}` with `M` spelled `T`, and all
  five call sites pass a `DBuf{T,H}` (`commandr.cpp:158`, `glm4.cpp:185`,
  `minicpm.cpp:180`, `minicpm3.cpp:208`, `phi3.cpp:147`) so `x.shape[0] == T`
  identically — plus the new byte-exact runtime case at both M=1 and M=4.
- *The op sequence is identical; the ALLOCATION sequence is not, and saying
  otherwise would overreach.* `DBuf` is pooled and returns its block on destruction
  (`dense_device_glue.h:99`). Pre-fold, the `[T,2I]` gate_up buffer stayed alive to
  the end of the MLP block; post-fold it is released when `Apply` returns, so the
  `[T,H]` output may reuse that block. This cannot move a value — `vt::MatmulBT`
  writes every output element from a fresh f32 accumulator rather than accumulating
  into `out` — and peak pool usage is unchanged or lower. It is the same shape every
  earlier fold produced, and it is stated here rather than hidden inside
  "byte-for-byte the same".

**Rejected as evidence: an object-code A/B.** Compiling each of the five TUs from
main's source and from the folded source with the identical production command and
diffing the disassembly gives 3689-6888 differing instruction lines per TU. That is
GCC re-allocating registers around a header-inlined method plus its emitted
out-of-line copy — it is not a numerical signal in either direction, and machine-code
identity is the wrong bar for a routing fold. Recorded so nobody re-runs it expecting
a proof.

**Decisions held, none reopened.** Direct `UnquantizedMlpGateUpMethod` rather than the
`MakeMlpGateUpMethod` factory (no loader for the three `Qwen3DenseMlpWeights` models
populates `*_fp4`, so the factory arm would be untestable); no adoption env flag (the
fold is unconditional and bit-exact, so a rollback branch would be dead code); no
Group-B allowlist entry touched and no shared-layer TU edited (`git status` on
`include/` is empty), so no already-routed model can move.

**One correction this row forced.** `glm4` and `phi3` also sit on the OTHER (glue)
allowlist with the reason `pending FUSION-DENSE-MIGRATE`. Closing this row would have
left both pointing at closed work — the stale-`pending` shape that let this allowlist
grow to hold most of its population. Both reasons are repointed at
[#314](https://github.com/mudler/vllm.cpp/issues/314), which owns the glue half. The
glue fold itself is NOT done here.

## Review findings repaired (2026-08-11)

An independent review returned **PASS with 5 MINOR findings**. The routing change
itself was not touched — it is byte-identical by construction and was verified per
model. What changed:

| # | Finding | Repair |
|---|---|---|
| F1 | The RED-first claim overstated its scope: only a **comment-inclusive** revert turns the shipped checker RC=1. | Claim narrowed to what was executed, with the code-only result stated. Checker semantics deliberately NOT changed — that would need its own spec, and the detector's documented adoption signal is the token's presence in the FILE. |
| F2 | The allowlist claimed every survivor "names a blocker that needs the SHARED LAYER extended"; `gemma4_moe` and `laguna` did not. | Both reasons REWRITTEN with the actual missing arm (a GeGLU arm on the SwiGLU-only grouped MoE op; an NVFP4-Marlin-resident arm for weights held as raw device pointers), and the allowlist header now says a bare `pending <ROW>` is not a reason. Echoes in `parity-ledger.md` and `kernel-matrix.md` follow. |
| F3 | `docs/FEATURES.md` said `check-fusion-consistency.py` "lists the rest with their blocker". The script lists nothing. | Cites `scripts/merged-gemm-consistency-allowlist.txt`, which is where the blockers live. |
| F4 | **No executed coverage of the change surface** — mutating `phi3`'s `I` to `I - 1` survived 176 CPU tests. | New `tests/vllm/models/test_dense_gate_up_seam_forward.cpp` (below). |
| F5 | `Closes #299` left the five owed dgx SACRED runs with no tracking handle. | [#337](https://github.com/mudler/vllm.cpp/issues/337) opened and cited from the ledger line, this spec and the PR body. |

Also opened, NOT fixed here (pre-existing on both sides of the fold, its own issue
per AGENTS.md): [#338](https://github.com/mudler/vllm.cpp/issues/338) —
`minicpm.cpp`/`minicpm3.cpp` hard-code SiLU while upstream `MiniCPMMLP`
(`minicpm.py:219-226`) selects `FatreluAndMul` on `hidden_act == "fatrelu"` and
RAISES otherwise; our `parse_config` hook never reads `hidden_act`, so such a
checkpoint would decode fluent-wrong rather than fail to load.

### F4: what the new test proves, and how that was verified

`test_dense_gate_up_seam_forward` drives the real `{MiniCPM,Phi3,Glm4,Commandr}Model::Forward`
over synthetic in-memory weights on CPU (no checkpoint, no GPU), four arms per model:

- **The split.** Zeroing the UP half of the merged `[2I, H]` gate_up makes
  `silu(gate)·up` exactly zero; zeroing `down_proj` makes the MLP contribute zero a
  different way. Both must give BYTE-IDENTICAL logits — which holds only if the seam
  splits the merged operand at exactly `I`.
- **The half order.** Swapping the two halves must CHANGE the logits, since
  `silu(gate)·up != silu(up)·gate`.
- **Vacuity guard.** baseline != MLP-disabled, so the split assertion is not two
  zeros compared to each other.
- **Determinism.** The same weights twice are byte-identical.

Green: **4 cases / 1940 assertions**. Both mutations executed, each restored
byte-for-byte (`md5sum` verified):

1. `I` -> `I - 1` at ALL FOUR call sites simultaneously: **all four cases fail**, each
   with `vt: matmul_bt: output shape mismatch` — i.e. every one of the four folded TUs
   is genuinely reached by this binary. Mutating `phi3` alone gives 3 passed / 1
   failed, so the cases are independent. Stated plainly: this mutation is caught by a
   SHAPE contract inside `vt::MatmulBT`, not by the numeric assertion — the point is
   that before this file nothing executed those TUs on CPU at all, so that contract
   never ran.
2. A SHAPE-PRESERVING mutation of the shared epilogue (`vt::SiluAndMul`'s CPU kernel,
   `silu * up` -> `silu + up`): **all four cases fail on `CHECK(Same(zero_up, zero_down))`**
   — the split assertion itself — while `test_linear_method` stays fully GREEN at
   **76/76**, because that test compares the seam against a standalone sequence which
   calls the SAME op. So the new coverage catches a defect class the PR's existing new
   test structurally cannot.
