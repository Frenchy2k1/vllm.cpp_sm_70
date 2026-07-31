# qwen3_5 GGUF A3 — keep-quant grouped-MoE fold (spike + W1 scaffolding)

**Row:** `MODEL-TEXT-qwen3_5` GGUF keep-quant MoE. **Reuse exemplar:** the landed
**Laguna W9** grouped-expert fold (`96b4652f`, `src/vllm/model_executor/models/laguna.cpp`
`LqGemmGrouped` + `LagunaGroupedMoeEnabled`). This is the qwen3_5 half of fold-plan **A3**
(`.agents/specs/arch-fusion-fold-plan-2026-07-30.md`), the SAME shared descriptor
(`vt::MatmulBTQuantGrouped`) applied to a second model — "fold structurally, use consistently".

## Goal
qwen3_5's keep-quant GGUF MoE is a per-expert host-gather loop (`qwen3_5.cpp` ~`:5129`):
per activated expert it calls `ExpertMlp` = `MatmulF32(gate)` + `MatmulF32(up)` + host
SwiGLU + `MatmulBf16(down)`, each a device `vt::MatmulBT` over ONE expert's block-quant
weight + a host round-trip. Collapse the per-expert `{gate,up,down}` matvecs into 3
grouped `vt::MatmulBTQuantGrouped` launches over the stacked `[E*N,K]` tower (mirror W9).
Bit-exact by construction (the grouped op's CPU provider loops the EXACT `kMatmulBTQuant`
per group; ds4-gated byte-exact on CUDA — premise verified `c1349367`: `ResidentWeight`
keeps weights quantized, so the per-expert loop already runs the same vec_dot).

## The loader crux (WHY this is more than Laguna W9) — MEMORY-NEUTRAL
Unlike Laguna (experts pre-stacked), qwen3_5's `MoeBlockWeights.expert_gate/up/down`
(`qwen3_5_weights.h:296`) are per-expert `std::vector<OwnedTensor>` — SEPARATE
non-contiguous copies produced by `LoadExpertsT` (`qwen3_5_gguf_weights.cpp:1150`), which
for the keep-quant arm (`r != kExpandBf16`, `:1164-1174`) slices the GGUF's already-stacked
`[E,out,in]` tensor per-expert **without transpose** (block boundaries; the bf16-EXPAND
arm `:1176-1185` DOES transpose — out of A3 scope). The grouped op needs ONE contiguous
`[E*out,in]`.
**Memory-neutral fix:** load the tower WHOLE (`OwnGgufKeptSlice(g,pol,t,r, n=E*out_dim,
k=in_dim, row_offset=0)` — handles `kKeepQuant` blocks + `kKeepF16`, `:226-237`) into the
NEW stacked fields, and DO NOT populate the per-expert vector for keep-quant. This REPLACES
the E per-expert copies with ONE stacked copy (same total bytes) — NOT additive, so NO
duplicate ~30 GB expert copy on the 35B ⇒ no OOM-reboot (see
[[gb10-unified-memory-oom-reboots-box]]).

## W-plan
- **W1 (this change): spike + struct scaffolding.** Added `OwnedTensor expert_gate_kq /
  expert_up_kq / expert_down_kq` to `MoeBlockWeights` (keep-quant stacked `[E*N,K]`, EMPTY
  on fp4/bf16-expand — additive, the existing per-expert + fp4 vectors untouched). No
  loader/forward wiring yet ⇒ build stays green, behavior unchanged (dead fields until W2).
- **W2: loader.** New `LoadExpertsStackedKq` (whole-load via `OwnGgufKeptSlice`, returns
  empty on `kExpandBf16`). Rework `LoadExpertsOrNvfp4` (`:1192`) to a 3-arm route on
  `PeekRoute`: `kNvfp4Fp4` → `*fp4` (as now); `kExpandBf16` → `*bf16` per-expert (as now,
  the transposed fallback); else (`kKeepQuant`/`kKeepF16`) → `*kq_stacked` whole,
  leaving `*bf16` empty. Route/audit called exactly once (mirror the existing fp4 peek).
- **W3: forward.** In the `qwen3_5.cpp` GGUF MoE block, add a grouped path (guard
  `VT_QWEN35_GROUPED_MOE`, default-ON, `=0` = byte-exact per-expert fallback) taken when
  `!expert_gate_kq.Empty()`: gather activated `(token,expert)` pairs, build `act[P,K]`
  (bf16→f32 per row), 3× grouped `vt::MatmulBTQuantGrouped` over `expert_gate_kq`/`_up_kq`
  /`_down_kq` + f32 SwiGLU in SLOT ORDER (byte-identical combine), route-weight stays in
  `MoeCombine`. The fp4 arm + the bf16-expand per-expert `ExpertMlp` arm are unchanged.
  A `LqGemmRowSlice`-style view over `expert_*_kq` gives the `=0` fallback with no per-expert
  vector. Validation (`:591`) updated: keep-quant checks `expert_*_kq` non-empty, not the
  vector `.size()==E`.
- **W4: gate.** CPU RED-first byte A/B unit (grouped == per-expert-loop byte-identical on
  block-quant experts — the qwen3_5 WIRING; the OP is already gated by ds4's
  `MoeGateUpSwiGLUGrouped`/`MatmulBTQuantGrouped` tests). DGX SACRED: `test_qwen36_gguf_engine`
  on `~/llama-w4a16-phase3/Qwen3.5-35B-A3B-TBQ4_0.gguf` (GB10) — token-identical to the
  committed golden (grouped=1 vs =0 byte-equal). Build+gate harness proven by Laguna W8/W9
  (git-archive → CUDA build → same-binary `=1`/`=0` A/B).

## Tests to port
- `test_ops_moe_*` grouped keep-quant A/B (extend the ds4 pattern to the qwen3_5 stacked layout).
- `test_qwen36_gguf_engine` stays the e2e SACRED gate; add a `VT_QWEN35_GROUPED_MOE=0` A/B leg.

## ★★ W2/W3a attempt 2026-07-31 — CPU gate CAUGHT an incomplete change (multi-consumer)
Implemented W2 (3-arm loader → `expert_*_kq`) + W3a (byte-exact `ExpertMlpKq` slice forward),
committed LOCAL-only, and ran `test_gguf_keep_quant` (CPU) BEFORE pushing. It FAILED:
`:845 REQUIRE(expert_gate.size()==E)` → 0 vs 3. Root cause: emptying the per-expert
`expert_gate` vector for keep-quant breaks consumers that STILL read it. **A3 has THREE
keep-quant consumers of the per-expert vector, all must change together:**
1. the main MoE forward (`qwen3_5.cpp` reference loop) — handled by W3a `ExpertMlpKq`;
2. **`qwen3_5_mtp.cpp`** (`:122-129` populates `expert_gate`; `:230-237` `VT_CHECK`
   `expert_gate.size()==experts` + reads `expert_gate[expert]`) — the MTP draft MoE, NOT
   yet handled → its VT_CHECK fires;
3. **`tests/vllm/test_gguf_keep_quant.cpp:803` "loader keep-quant expert split is lossless
   per expert"** — asserts the loader fills the per-expert vector; must be re-expressed to
   assert the stacked `expert_*_kq` (the whole tower == the concat of the per-expert slices).
Keeping BOTH the per-expert vector AND the stacked tensor is NOT an option (2× ~30GB experts
→ OOM-reboot on the 35B). So the correct A3 W2/W3 is: load stacked, and route BOTH the main
forward AND the MTP MoE through slices of the stacked tower (add a shared keep-quant expert
accessor), and update the split-lossless test. Verify-before-push worked — nothing broken
was pushed. (Local commit `831e941b` reverted; the byte-exact `ExpertMlpKq`/slice-helper
code is correct and reusable, it just needs the MTP + test co-changes.)

## ★★★ W2/W3a full attempt #2 (2026-07-31) — DGX-gated, device bug ROOT-CAUSED, 1/2 passing
Completed the coordinated change (loader stacked + `ExpertMlpKq` slice forward + test
re-expressed). **CPU gate `test_gguf_keep_quant` 37/37 PASS** + MTP unaffected. On the DGX
strict `test_qwen36_gguf_engine` (keep-quant default-ON), the same-binary A/B was decisive:
- **`VT_GGUF_KEEP_QUANT=0` (expand, bypasses my `_kq` path): 2/2 PASS** (coherent "Paris,
  France…" + "9,10,11,12" == golden) ⇒ the failure is in MY keep-quant path.
- default keep-quant (my `_kq`): **ALL-ZEROS** ("!!!!") both cases.
**ROOT CAUSE #1 (fixed):** `KqSliceView` built a Tensor over `w.bytes.data()` (HOST) but
**CUDA `needs_weight_staging()` == TRUE** (`platforms/cuda.cpp:67`) — the device keep-quant
kernel does not read the host pointer (the CPU gate can't catch this: CPU host==device). Fix:
`KqResidentSlice` routes through `ResidentWeight(d, w)` (stages the WHOLE tower to `d_dev`
once, cached) and slices the RESIDENT ptr. **After the fix: case 1 (APEX-Balanced) PASSES
golden-correct; case 2 (APEX-Compact) STILL all-zeros → 1/2.**
**RESIDUAL (case 2 only, keep-quant only — expand passes both):** the two cases are DIFFERENT
checkpoints (APEX-Compact vs -Balanced, different quant mixes/sizes). Hypotheses to test next:
(a) whole-tower `ResidentWeight` `Alloc` is ONE large block per tower vs the pre-A3 per-expert
lazy staging — the larger file may hit an Alloc/copy-size issue; (b) the `repacked` marker
(my host slice carried `w.repacked`; the ResidentWeight staging branch leaves it false — a
mismatch on one file's quant type); (c) a quant-type my slice mishandles. **Attribution owed:
run pre-A3 main in keep-quant on case 2 — if it ALSO all-zeros, case 2 keep-quant is
PRE-EXISTING (my A3 is then correct: case-1 golden + CPU 37/37), not an A3 regression.**
The working code (loader + ExpertMlpKq + KqResidentSlice + test) is on the reverted local
commit; NOT pushed (1/2 strict is not landable). Verify-before-push caught both the staging
bug AND the residual — nothing broken shipped.
**★ CHECK LAGUNA:** `laguna.cpp` `LqGemm`/`LqGemmRowSlice` use the SAME raw-host-byte view
(no `ResidentWeight`) yet W9 gated byte-exact on GB10 — reconcile (GB10 is physically unified
per `cuda.cpp:64`, so a host ptr may work despite the staging POLICY flag; or laguna's
OwnedTensors are device-resident). If host-ptr works physically, root-cause #1 is really the
`repacked`/whole-tower-load angle, not the pointer — worth pinning before W3b.

## Constraints / hazards
- `MoeBlockWeights` is SHARED with the fp4/bf16 paths — keep the `_kq` fields additive.
- Route-weight in `MoeCombine` (invariant). No RopeNeox/impl swaps. `check-fusion-consistency`
  must pass by ROUTING through `vt::MatmulBTQuantGrouped` (the shared op), never an allowlist.
- Agent cap 200/200 this session ⇒ W2-W4 are hands-on.

## ★ W3 byte-exactness hazard (found during the 2026-07-31 W2/W3 attempt — MUST resolve)
qwen3.6-35B is a **STRICT-gated MVP gate model** (`test_qwen36_gguf_engine` token-exact),
so the grouped fold MUST be byte-exact, NOT a near-tie. The subtlety is the DOWN GEMM:
`ExpertMlp`'s `MatmulBf16` (`qwen3_5.cpp:858`) computes the keep-quant `vt::MatmulBT` with a
**bf16 OUTPUT tensor** (`dout` is `kBF16`) — the f32→bf16 cast happens INSIDE the kernel.
The shared `vt::MatmulBTQuantGrouped` outputs **f32 only** (`ops.h`), so the grouped path
would do f32 → then `F32ToBF16`. Byte-exact ONLY IF the kernel's internal store-cast ==
`F32ToBF16(same f32 accumulator)` — provable by construction (identical f32 vec_dot
accumulator — the grouped op's "same integer-dot core", ds4-gated byte-exact — + the same
round-to-nearest-even bf16 cast), but MUST be confirmed by the grouped=1-vs-=0 A/B BEFORE
claiming strict. If the A/B shows a 1-ULP diff, W3 needs a **bf16-output grouped down**
variant (mirror `MatmulBf16`'s in-kernel cast) rather than f32+`F32ToBF16`, since a near-tie
is NOT acceptable on this strict gate. The gate/up side is safe (`MatmulF32` f32-out ==
grouped f32-out; SwiGLU `F32ToBF16(Silu(hg)·hu)` identical). **Also:** the `=0`
byte-exact fallback for keep-quant must slice `expert_*_kq` per-expert (add `MatmulF32Slice`
/`MatmulBf16Slice`, mirror of laguna `LqGemmRowSlice`) — the per-expert `expert_gate` vector
is EMPTY for keep-quant after W2.

## ✅ RESOLVED — A3 W2+W3a LANDS (2026-07-31, attribution complete)
Ran PRE-A3 main (`origin/main` sans A3 code) `test_qwen36_gguf_engine` in keep-quant on the
DGX. Result is byte-for-byte the SAME as A3-fixed: **APEX-Compact (1st TEST_CASE) passes
golden-correct ("Paris, France…" + "9,10,11,12"); APEX-Balanced (2nd TEST_CASE) all-zeros.**
So the "case-2 all-zeros" is **PRE-EXISTING** — the SECOND engine load in the same test
binary fails identically WITHOUT any A3 code (a second-engine/async-scheduler-state harness
bug, tasks #50/#65), NOT the keep-quant forward. My earlier "case-2 is my A3 bug" read was
wrong (I mis-mapped the config↔order). With the ResidentWeight staging fix, **A3 is
behavior-IDENTICAL to pre-A3**: Compact keep-quant byte-identical golden + CPU
`test_gguf_keep_quant` 37/37. The staging fix (§W2/W3a #2) WAS required (host-ptr broke both).
**LANDED** `df3f5add`→(this commit). W3b grouping (`MatmulBTQuantGrouped`, additive perf) is
now unblocked. The pre-existing Balanced-2nd harness failure is out of A3 scope (tasks
#50/#65). laguna `LqGemmRowSlice` reconcile still worth a look but non-blocking (A3 now uses
the correct ResidentWeight-resident slice).
