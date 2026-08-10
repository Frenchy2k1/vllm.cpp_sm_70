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
allowlist entry. `scripts/check-fusion-consistency.py` goes from
`6 route … 11 allowlisted` to `11 route … 6 allowlisted`.

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
row to the merged-GEMM half only.

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

Added locally: one regression case in
`tests/scripts/test_check_fusion_consistency.py` asserting that each of the five
folded stems (a) is scanned by the merged-GEMM detector, (b) routes a shared seam,
and (c) is NOT on `scripts/merged-gemm-consistency-allowlist.txt` — so re-allowlisting
a folded model, or reverting a fold, goes RED instead of silently re-opening the
drift the allowlist was emptied of.

## Gates

CPU-only (no GPU available to this claim), foreground:

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-cpu -j 18
python3 scripts/check-fusion-consistency.py      # must read 11 routed / 6 allowlisted
python3 -m unittest tests.scripts.test_check_fusion_consistency -v
./build-cpu/tests/test_linear_method               # the seam byte-exactness gate
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
passed.

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

PENDING — W2 has not run. This section is filled in the implementation commit,
from gates that actually executed, and never written ahead of them.
