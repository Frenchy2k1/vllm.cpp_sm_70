# A2-Q2b — the REAL-CHECKPOINT per-block gate, and `lm_head` on NVFP4

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Sibling / predecessor:** [`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md)
(A2-Q2a — the MoE arm, landed on the synthetic gate).
**State:** IMPLEMENTED on `row/A2-Q2b-lmhead-nvfp4`; the GPU legs of §3 are pending a
`dgx:gpu0` window. See `## Now`.

---

## 0. Why this exists as its own unit

A2-Q2 was split twice, and this file owns the second split.

The first split (A2-Q2a / A2-Q2b) was taken because keeping `lm_head` on the
host **preserves A2-R's attributability**: both arms end in the identical host
projection, so a token difference stays attributable to the MoE arm.

The second split — deferring the **real-checkpoint per-block numeric gate** out
of A2-Q2a — was a scheduling decision, and it is recorded here rather than left
implicit. A2-Q2a's device MoE arm is **unreached** (G-SAFE refuses before it,
see that spec's `## Owed`), and end-to-end NemotronH is blocked on **A2-P**, not
on the quantized arms. Holding A2-Q2a's branch for a 21 GB checkpoint load on a
contended, reboot-prone box would have held the critical path behind a gate that
unlocks nothing. So A2-Q2a landed on the synthetic NVFP4 fixture, and the
expensive gate came here.

**This is a deferral, not a cancellation.** The synthetic result is explicitly
bounded (A2-Q2a §13.6.1) and cannot stand in for what this unit measures.

---

## 1. Scope

| In A2-Q2b | Out |
|---|---|
| the per-block numeric gate on the REAL checkpoint: every one of the 23 MoE layers against `trace.mixer[l]` | anything A2-P owns (paging, carried state, batching, the G-SAFE narrowing) |
| `lm_head` through the NVFP4 dense route, with the `## 5. Owed` residency decision (#984) applied as A2-Q2a applied it | the FP8 mamba arm — A2-Q1 |
| hybrid-vs-host token identity, and the disclosure that A2-R's attributability property ENDS when `lm_head` moves | fixing [#984](https://github.com/mudler/vllm.cpp/issues/984) or [#962](https://github.com/mudler/vllm.cpp/issues/962) |
| the §5.3 mutations A2-Q2a left owed (Q2-M3 … Q2-M7) | any throughput number, on any axis |

---

## 2. What it must measure, and why the synthetic arm does not

A2-Q2a's fixture reported **bit-exact** device-vs-host agreement (`0` over 512
elements, against a routed-scale separation of `0.6`). That is a real result and
a NARROW one. Its output is bf16, its contraction is K=128, its E2M1 codes are
exactly representable and its group scales are powers of two — precisely the
conditions under which a bf16 store absorbs genuine reduction-order differences.

So this unit must measure what the synthetic arm structurally cannot:

- **the real geometry** — H=2688, I=1856, E=128, top_k=6. The contraction is 21x
  longer, so the reduction order genuinely differs.
- **the real scale VALUES** — `weight_scale_2` and the fp8 group scales the
  checkpoint ships are neither powers of two nor uniform, unlike the fixture's.
- **the other Marlin thread configs.** The fixture resolves on `{128,64,128}`
  (up) and `{64,128,128}` (down) only; the real shapes may select others.

**A red here after a bit-exact synthetic is an EXPECTED outcome, not a
contradiction.** It is what A2-Q2a §13.6.1 predicts. Report it as a result; never
widen a band to absorb it.

---

## 3. The gate

Per-block numeric equivalence against the host reference via `NemotronHTrace`,
on the real checkpoint, at **every** one of the 23 MoE layers, plus `lm_head`
against the host projection on the same gathered rows, plus hybrid-vs-host token
identity.

**Bands are MEASURED, and the guard is a PROPERTY** — the shape A2-Q2a arrived
at after its first attempt failed on its own instrument:

- Every comparison reports **how many elements it examined**, and the caller
  asserts that count **against the geometry**, never against the buffer's own
  size (which would agree with itself if the buffer were short). A maximum over
  zero elements is `0.0`, and so is a bit-exact comparison; without the count
  the two are indistinguishable.
- Agreement, separation and the property guard must report the **same** count,
  or the band between them is fiction.
- The band must **admit exact agreement**. A2-Q2a's first band was
  `sqrt(agreed * separation)`, which collapses to 0 when the arms agree exactly
  and failed on the best possible outcome. `separation / 2` with an explicit
  `REQUIRE(separation > 0)` has the intended property and keeps `<` strict.
  Never relax `<` to `<=`: that accepts a band of 0.

---

## 4. ★ Operational — the host working set is what takes the box down

`gpu_memory_utilization` does **not** bound host RAM on GB10, and `nvidia-smi`
attributes only the device side. **Both instruments are blind to the transient
host working set**, which is what actually reboots the machine
(`vm.overcommit_memory=1`, zero swap: the kernel grants memory it cannot back and
the box reboots rather than OOM-killing a process).

So:

1. **Claim `dgx:gpu0` through the fleet controller, and run the work as the
   lease's payload**: `rc run -d dgx:gpu0 --max-runtime <N>h -- <cmd>`, with
   `scripts/nemotron-h-a2q2b-gpu-gate.sh` as `<cmd>`. Never `ssh` to the box,
   and never `rc hold` an idle one. **Do not take `$GPU_LOCK` here.** `dgx:gpu0`
   is a fleet device, so `AGENTS.md` makes the lease the required path, and the
   file mutex is the wrong instrument for it: the fleet cannot see that mutex,
   so a `flock` taken over `ssh` does not exclude a concurrent `rc` holder and
   the controller keeps reporting the box free while somebody is on it. That is
   not hypothetical — on 2026-08-17 exactly this pair of non-excluding mutexes
   cost `.agents/specs/minimax-music3.md` §13.10 a whole speed axis, which is
   the [#777](https://github.com/mudler/vllm.cpp/issues/777) failure again. The
   gate script correspondingly takes **no** mutex of its own; the lease is what
   serialises this row's window.
2. **Check `free -g` headroom INSIDE the lease**, not before claiming it. A
   granted lease says the previous holder released; it never says the box
   recovered. Abort loudly below a stated floor (A2-Q2a used 60 GB), which is
   what `scripts/nemotron-h-a2q2b-gpu-gate.sh` PRECONDITION 1 does.
3. **Sample `free -g` on a loop for the whole load and record the PEAK**, not the
   final value. The figure that matters is transient.
4. Build `-j 4`. One log per run. Never hang holding the lease.
5. Verify the configure log reads `ENABLED for [121a]` — a `DISABLED` line or a
   `[121]` **voids** the result rather than failing it.

Budget to size against: ~17.6 GiB host weights + 16.5 GB device arena + page
cache for a 21 GB checkpoint. Checkpoint at
`${CHECKPOINT_ROOT}/nemotron-3.5-lightning-30b-nvfp4`, with `CHECKPOINT_ROOT` =
`/usr/local/nas_share/checkpoints` — `/usr/local` is COS_PERSISTENT and survives
a reboot; `/mnt` is the ephemeral root overlay of the immutable OS and does not.

**This row has lost four GB10 windows to environment rather than to code** —
`121` instead of `121a`, an unconstrained build, a CUTLASS fetch with no egress,
and an 8h19m outage ended by a human power cycle. Size the plan for that.

---

## 5. Owed

- [#962](https://github.com/mudler/vllm.cpp/issues/962) — NVFP4 Marlin disagrees
  with itself on sm_110 (`bitdiff=15/32768`). The **Thor leg stays PENDING** on
  it; do not quote a number from a kernel that contradicts itself.
- [#984](https://github.com/mudler/vllm.cpp/issues/984) — the address-keyed
  Marlin repack cache. A2-Q2a routed around it by never calling either
  `MarlinDenseResidentFor`; `lm_head` must do the same or say why not.
- The §5.3 mutations A2-Q2a left owed: **Q2-M3** (expert stride off by one),
  **Q2-M4** (`routed_scaling_factor` folded into the logits), **Q2-M5** (shared
  expert added before the routed scale), **Q2-M6** (`MoeRelu2` → plain relu),
  **Q2-M7** (device call site deleted). Q2-M1 and Q2-M2 were run in A2-Q2a.
  Q2-M3 matters most: `src/vt/ops.cpp:874-895` validates **no extent of
  `b_q_weight` and nothing at all about `b_scales`**, so a stride defect is
  silent at the op boundary and only the numeric gate can see it.


- [#1410](https://github.com/mudler/vllm.cpp/issues/1410) — `check-runner-routing-consistency.py`
  cannot resolve a cross-TU free-function device forward, so NemotronH still
  classifies HOST although `NemotronHPagedForward` now assigns both
  `fl.device_tensor` and `fl.device_storage`. The allowlist entry is NARROWED
  to that instrument limit rather than removed. Not fixed in this flow because
  it changes checker semantics, which `AGENTS.md` routes to its own row.
- **The 23-layer MoE per-block sweep of §3 and the `lm_head` real-checkpoint
  numeric leg** run only on `dgx:gpu0` and are PENDING a window, not waived.
  What this change lands is the implementation, the CPU-buildable arms, and the
  synthetic device gate that A2-Q2a's preamble argues for. `## Now` records the
  exact state of each leg, so a `PENDING` here is a scheduled measurement with
  a named blocker rather than a gate nobody ran.
- **Q2-M3 … Q2-M7** (A2-Q2a's owed mutations) are unchanged by this row and
  stay owed. They gate the MoE arm, not `lm_head`.

---

## 6. Now

**Implemented.** The device `lm_head` arm lands on `row/A2-Q2b-lmhead-nvfp4`.

### What was measured BEFORE anything was built

The row's premise — that host re-expansion dominates NemotronH decode and that
`lm_head` is a large share of it — was ARITHMETIC when this row was dispatched.
It is now a measurement, taken at the single dequant seam
(`NemotronHOwned::DenseBf16`) on the REAL 21 GB checkpoint through the
production ABI driver (`examples/nemotron_h_gen`), one decode step, T=1,
top_k=6, 23 MoE layers:

| group | shape | calls | elements | per call | % |
|---|---|---|---|---|---|
| routed expert `up_proj` | `[1856, 2688]` | 138 | 688 472 064 | 4 988 928 | 22.36% |
| routed expert `down_proj` | `[2688, 1856]` | 138 | 688 472 064 | 4 988 928 | 22.36% |
| shared expert `down_proj` | `[2688, 3712]` | 23 | 229 490 688 | 9 977 856 | 7.45% |
| shared expert `up_proj` | `[3712, 2688]` | 23 | 229 490 688 | 9 977 856 | 7.45% |
| **`lm_head`** | **`[131072, 2688]`** | **1** | **352 321 536** | **352 321 536** | **11.44%** |
| mamba `out_proj` (FP8) | `[2688, 4096]` | 23 | 253 231 104 | 11 010 048 | 8.23% |
| mamba `in_proj` (FP8) | `[10304, 2688]` | 23 | 637 034 496 | 27 697 152 | 20.69% |
| **TOTAL** | | **369** | **3 078 512 640** | | **100%** |

`138 == 6 * 23` exactly, which is what confirms this is the decode shape and not
a prefill aggregate.

Three things follow, and the third is why the row proceeded:

1. **The dispatching estimate was wrong in both of its numbers, in the same
   direction.** It put `lm_head` at 131072 x 4096 = 537e6 elements and at ~43%
   of the population. `hidden_size` is 2688, not 4096: the true count is
   352 321 536, and the true share of that population is 28.35%.
2. **The "~1.24e9 elements / ~2.49 GB per token" figure is real and now has a
   name.** It is not the host arm's total (3.079e9 / 6.157 GB). It is exactly
   `mamba + lm_head` = 1 242 587 136 elements = 2.485 GB — the residue AFTER
   A2-Q2a moved the MoE arm to the device. It matches to four significant
   figures, which is what identifies which regime the number describes.
3. **`lm_head` is the LAST one.** Against the three-leg discriminator on
   `dgx:gpu0` (`/workspace/a2d1-discriminate/20260819T200231Z`: device mamba ON
   1.554 s/token and 108.2x vs vLLM, OFF 10.319 s/token and 718.1x), the mamba
   arm is worth 6.64x and is in flight on A2-D1. `lm_head` is on the HOST in
   every one of those three legs. Once the mamba arm lands, `lm_head` is
   352 321 536 of 352 321 536 — 100% of the host re-expansion left in a decode
   step. It is also the largest SINGLE re-expansion in the model by 12.7x
   (352.3e6 in one call against 27.7e6 for mamba `in_proj`), so its 704.6 MB
   transient bf16 buffer is the allocation that matters most on a
   unified-memory box that reboots rather than OOM-kills.

The refutation is therefore narrow and the conclusion survives: the estimate's
share was wrong, the direction was right, and the case is STRONGER after the
discriminator than before it.

### Leg status

| Leg | State |
|---|---|
| seam extension (caller-owned `MarlinDenseResident`) | DONE, built on BOTH arms (see below) |
| device `lm_head` arm + `DeviceLmHeadEligible` | DONE, built on BOTH arms |
| production wiring in `NemotronHPagedForward` -> device `ForwardLogits` | DONE |
| host arm retained as the gate's operand + the non-NVFP4 fallback | DONE |
| routing-allowlist entry narrowed, [#1410](https://github.com/mudler/vllm.cpp/issues/1410) filed | DONE |
| the CPU-reachable half (`n_out`, the shared `final_normed` download) gated through `GPUModelRunner` | DONE and RUN — `test_nemotron_h_paged_forward.cpp` §12, red-first below |
| synthetic device `lm_head` numeric gate (measured band, asserted counts) | COMPILES; runs on CUDA only. **NEVER RUN** |
| `nvcc` build of the Marlin KERNEL + any execution of the device arm | PENDING a `dgx:gpu0` window |
| real-checkpoint `lm_head` numeric leg + token identity | PENDING a `dgx:gpu0` window |
| reachability deletion mutation | PENDING the same window |
| 23-layer MoE per-block sweep (§3) | PENDING; owed above |

### What "built" means here, corrected

The first submission of this row said "the CPU build compiles the `#else` arms
only, so it does not compile the Marlin path at all", and recorded the CUDA
build of the Marlin arm as blocked on a GPU window. **That was wrong, and it
overstated the blocker.** `include/vt/cuda/marlin_repack.h` includes only
`<cstdint>`, `<cstddef>` and `<vector>`, so the HOST side of the Marlin arm
needs no CUDA toolkit at all. Measured on this box, which has no `nvcc`:

```
c++ -std=c++20 -I include -I src -isystem third_party -DVT_MARLIN_NVFP4=1     -Wall -Wextra -Werror -c -o nhd_marlin.o     src/vllm/model_executor/models/nemotron_h_device.cpp
-> rc 0, 0 errors, 0 warnings, a 1 268 552-byte object
```

**That byte figure is anchored to `72d736867`**, gcc 13.3.0 (Ubuntu
13.3.0-6ubuntu2~24.04.1) on a box with no `nvcc`, and compilation is
deterministic here: two runs of the identical command produced the identical
`sha256 42d670b6...`. The anchor is the point. Measured with that one command
at seven commits of this branch, the size MOVES with the code:

| commit | object bytes |
|---|---|
| `806b263e7`, `8fa900a62`, `1c62d9974`, `29b1128e3` (the reviewed head) | 1 269 696 |
| `fedf78d86` (the `-Werror` declaration repair) | 1 272 808 |
| `bff2b7b2f` (the F4 predicate repair) onwards, through `18f3da188` and `72d736867` | 1 268 552 |

So the 1 269 696 this section carried until now was CORRECT when it was written
and correct at the head a fresh reviewer read; it went stale two commits later
and nothing said so, because the evidence block named no commit. An unanchored
byte count is a measurement nobody can reproduce or falsify, which is why the
number now travels with the SHA it was taken at. Nothing after `72d736867`
touches this translation unit or its headers.

The size is incidental either way; what the block asserts is `rc 0` with zero
errors and zero warnings on a toolchain with no CUDA, and that holds at every
one of those seven commits.

Both arms are therefore compiled, and both are compiled `-Werror`. What
genuinely needs `nvcc` is the Marlin **kernel** and every **execution** of the
device path. Landing without those is acceptable and is what the PENDING rows
above record; claiming the host arm could not be compiled was not.

### A gate this row was already failing, and cannot repair in place

`scripts/check-doc-checkpoint.py` is red on this branch, and it was red at the
reviewed head `29b1128e` with the identical two errors, measured with
`--base 96ed8346f --head 29b1128e3`:

```
ERROR: commit 1c62d9974: changed user_usage but did not update docs/USAGE.md
ERROR: commit 8fa900a62: changed .agents/benchmark-record.md: measurement
       recorded but did not update docs/STATUS.md
```

The PR description claimed this gate had one real failure and that it was
repaired. It was not, and `.github/workflows/ci.yml:519` runs the same
`--base/--head` invocation, so the lane is red for this reason independently of
[#1371](https://github.com/mudler/vllm.cpp/issues/1371).

The second error names a real gap and it is now closed: this row moved a
lifecycle state and recorded a measurement, and `docs/STATUS.md` said neither.
It does now.

The first error, and the historical form of the second, cannot be closed by a
later commit. `check_doc_checkpoint` iterates `commits_in_range` and judges each
commit on its own contents, so the obligation belongs to `1c62d9974` and
`8fa900a62` and nothing appended afterwards discharges it. Repairing those two
would mean rewriting commits that are already the reviewed base, which resets
the pull request's continuous-integration approval and moves the head a fresh
reviewer was asked to look at. That is a scheduling decision rather than a
technical one, so it is recorded here as owed and raised for the operator rather
than taken unilaterally by a repair pass.

### The gate, as it actually ran

Every number below is from a run on this box, `-DCMAKE_BUILD_TYPE=RelWithDebInfo`,
`-Wall -Wextra -Werror`, no CUDA toolkit.

**Read the `overlay` column before any other one.** This tree cannot run
`GPUModelRunner` at all — [#1371](https://github.com/mudler/vllm.cpp/issues/1371)
throws at its construction — so every green below was taken with
[#1392](https://github.com/mudler/vllm.cpp/pull/1392)'s production fix applied
to the working tree, never committed here and reverted byte-for-byte afterwards.
Without it the §12 case does not merely score differently, it never reaches its
first assertion. Saying "a run on this tree" without that clause made the §12
row false of the tree it named.

| binary | overlay | `run_rc` | cases | assertions | verdict |
|---|---|---|---|---|---|
| `test_nemotron_h_moe_device` | none | 0 | 4, **4 passed** | 4 | `SUCCESS!` — and **all four SKIP**, both A2-Q2b cases included |
| `test_nemotron_h_paged_forward`, whole binary | **none** | 1 | 13, 2 passed, **11 failed** | 18 | `FAILURE!` — [#1371](https://github.com/mudler/vllm.cpp/issues/1371), not this row |
| `test_nemotron_h_paged_forward`, whole binary | **#1392** | 0 | 13, **13 passed** | 3269 | `SUCCESS!` |
| the new §12 case alone | **none** | **1** | 1, **0 passed, 1 failed**, 12 skipped | **0** | **`FAILURE!` — throws #1371 before the first assertion** |
| the new §12 case alone | **#1392** | 0 | 1, 1 passed, 12 skipped | 13 | `SUCCESS!` |

The un-overlaid §12 row is the baseline the two mutations below are measured
against, and it is a red already. That is exactly why they are reported with the
overlay: a mutation cannot be shown to turn a case red when the case is red
without it. Re-derived at `72d736867`: `-tc=` the §12 case on the clean tree
gives `run_rc=1`, `test cases: 1 | 0 passed | 1 failed | 12 skipped`,
`assertions: 0`, and `ERROR: test case THREW exception: No valid attention
backend for device type 0 from {FLASH_ATTN: [head_size not supported]}`. Note
`assertions: 0` — the shape this project has read as a pass before.

**The moe_device row is the honest reading of the synthetic gate, and it is not
a pass.** The binary builds and exits 0, but all four cases take the
`TryCudaQueue` skip on a GPU-less box, so the 4 assertions are the skip notices
themselves. The numeric gate examined nothing. That is why the table above says
`NEVER RUN` and not `green`.

**The paged-forward red is [#1371](https://github.com/mudler/vllm.cpp/issues/1371)
and belongs to nobody here.** All 11 failing cases throw the identical
`No valid attention backend for device type 0 from {FLASH_ATTN: [head_size not
supported]}` at `GPUModelRunner` construction, 10 of them cases this row never
touched. Overlaying [#1392](https://github.com/mudler/vllm.cpp/pull/1392)'s
production fix in the working tree — never committed here, and reverted
afterwards — turns the same binary green at 13/13 and 3269 assertions. Note the
shape of the red: doctest reported `assertions: 18 | 18 passed | 0 failed` while
11 cases were throwing, so the assertion line alone would have read as a pass.

### The red-first, on the cases that can run

Each mutation was applied to a scratch copy of
`src/vllm/model_executor/models/nemotron_h_device.cpp`, verified to have applied
by `git diff --stat`, built (a mutation that fails to build proves nothing), run,
and restored to an identical sha256 (`abf6e21f...`).

**Every row here was run with [#1392](https://github.com/mudler/vllm.cpp/pull/1392)'s
production fix overlaid**, for the reason the table above gives: un-overlaid, the
§12 case is already `run_rc=1` on the unmutated tree, so an M1 or M2 red measured
there would prove nothing at all. The overlay is what makes the vehicle able to
report a green, and only then can a mutation take it away.

| mutation | overlay | applied | `compile_rc` | `run_rc` | verdict |
|---|---|---|---|---|---|
| — (unmutated control) | **#1392** | — | 0 | 0 | **GREEN**, `SUCCESS!`, 13 assertions |
| — (unmutated control) | **none** | — | 0 | **1** | already **RED** — #1371, and the reason the rest of this table is overlaid |
| **M1** — `n_out` -> `R` at the host projection | **#1392** | 1 ins / 1 del | 0 | **1** | **RED**, `FAILURE!` |
| **M2** — never fill `trace->final_normed` | **#1392** | 1 ins / 2 del | 0 | **1** | **RED**, `FAILURE!` |
| **M4** — restore the exact pre-repair two-download shape | **#1392** | 3 ins / 8 del | 0 | 0 | **GREEN — reported, not hidden** |

Re-derived independently at `72d736867`, on a clean tree with the same overlay
applied and reverted: the unmutated control is `run_rc=0`, `1 passed`,
`13 assertions`, `SUCCESS!`; M1 gives `compile_rc=0`, `run_rc=1`, and
`ERROR: test case THREW exception: vt: NemotronH lm_head: gathered row count
does not match hidden_size at nemotron_h.cpp:1029`; M2 gives `compile_rc=0`,
`run_rc=1`, `assertions: 8 | 7 passed | 1 failed` on `REQUIRE( 0 == 288 )`. The
M2 edit was written as one deletion rather than the recorded `1 ins / 2 del`
reshape, so the applied-lines cell is the original author's shape and the
verdict is reproduced. `M4`'s cell needs no separate attestation: un-overlaid,
every run of this case is red, so a GREEN is only reachable with the overlay in
place and its own verdict entails the column.

M1 is the red the `n_out` rename exists for: with the request count substituted
the returned row count is 1 where the gather asked for 3. M2 arms the
trace-operand assertion. Both showed the trap this project has been bitten by
before — doctest printed `assertions: 2 | 2 passed | 0 failed` on M1 while the
case was failing, because a `REQUIRE` throws rather than counting.

**M4 stays green and that is a finding, not a gap to paper over.** The duplicate
`DownloadF32` the first submission introduced copies the same unchanged buffer
twice and produces identical bytes, so nothing observable from outside the
function can distinguish it. It is repaired structurally — one call is reached on
any path — and no assertion here pretends to catch it.

The device arm's own red-first does not exist on a CPU box and is not claimed.
`MarlinW4A16Selects` is false on a CPU queue, so the device branch, the
`fallback_gemms` assertion and the reachability deletion mutation are all
unreachable here and stay PENDING a `dgx:gpu0` window.

### The fresh review, and what it found

A fresh reviewer returned FINDINGS on the first submission. The two blocking
ones are recorded here because both are about EVIDENCE, and evidence is what a
spec is for.

1. **`tests/vllm/models/test_nemotron_h_moe_device.cpp` had never compiled.**
   `NemotronHHostWeights` was used unqualified and was missing from the file's
   using-block, so the file failed `-Wall -Wextra -Werror` with 9 errors on a
   plain CPU build — the PR head at `rc 1`, the merge-base version of the same
   file at `rc 0` on the same command. **The red-first result claimed for the
   synthetic numeric gate therefore did not exist and could not have existed.**
   The declaration is repaired and the file now compiles at `rc 0` on both
   arms, but the gate itself is still CUDA-only and still has never executed;
   the table above says `NEVER RUN` rather than restating a red nobody saw.
   `vllm_cpp_add_test(test_nemotron_h_moe_device)` is deliberately registered
   with NO CUDA guard, and that is what surfaced this: a case that skips at run
   time still has to parse and type-check on every CPU build.
2. **The production source asserted a protection that does not exist.** A
   comment at the device branch claimed the allowlist entry was removed and
   that "the routing checker, not a comment, is what now holds this branch in
   place". All three parts were false: the entry is narrowed and still present,
   and deleting the entire `if (DeviceLmHeadEligible(...)) { ... }` block leaves
   `scripts/check-runner-routing-consistency.py` at `rc 0` with byte-identical
   output ("3 host-logits off-framework (3 allowlisted)"), reproduced on the
   repaired tree with the file restored byte-for-byte afterwards (identical
   sha256). **Nothing automated holds that branch.** The checker is not widened
   to make it — that changes checker semantics, and [#1410](https://github.com/mudler/vllm.cpp/issues/1410)
   owns it with its own red-before. The comment now says so, and the
   reachability deletion mutation stays PENDING rather than claimed.

The reviewer also found a real defect that a token gate structurally cannot
see. `DeviceLmHeadEligible` restated the shared dispatcher's selection clauses
and dropped `MarlinW4A16Enabled()`, so under an explicit `VT_NVFP4_MARLIN=0`
the predicate said eligible while `MatmulNvfp4W4A16D` took its naive
redundant-dequant arm — computing the SAME logits while re-uploading the whole
`[131072, 2688]` operand on every decode step, because `LmHeadNvfp4View` hands
out a stack temporary that `ResidentNvfp4`'s weight-keyed cache can never hit.
The repair is structural rather than a patched clause: the three clauses now
live once, in `dense_nvfp4::MarlinW4A16Selects`, and both the dispatcher and
the model call it, so they cannot drift. `DeviceLmHeadD` additionally refuses
BY NAME if that same predicate is false on the operand it is about to hand over,
which catches an eligibility answer taken against a different queue or dtype.
Deliberately the PREDICATE and not the seam's `fallback_gemms` counter:
`MutableW4A16Stats()` is a plain non-atomic process-wide static, so a counter
window in production would refuse a correct run whenever anything else took a
fallback GEMM concurrently, and a false refusal is worse than the silence it
replaces. The counter is the right instrument in a test, where
single-threadedness is a property of the harness, and the synthetic gate asserts
it there. It is demonstrably armed rather than assumed, because
`tests/vllm/models/test_qwen3_forward.cpp:559` already asserts on CPU that it
reaches exactly `5 * num_hidden_layers` when the dispatcher does fall back.

**And the counter is not the only process-wide static on this route.** A row
whose whole thesis is declining to inherit the sibling's process-static defect
([#984](https://github.com/mudler/vllm.cpp/issues/984), the address-keyed Marlin
repack cache) should say what it DOES inherit, so:
`dense_nvfp4::DenseMarlinWorkspace` (`dense_nvfp4_gemm.h:506`) is a
process-static **device** allocation — `static void* ws` behind a `static
std::mutex`, sized from `MarlinDeviceSms` and keyed on nothing, not even the
device index — and every caller of `MatmulNvfp4MarlinD` (`:539`) and
`GateUpFusedMarlinD` (`:700`) shares the one buffer. It already has three
consumers (the shared dense route in `dense_attn_block.h`, MiniMax-H3's
`minimax_h3_device.cpp`, and the compressed-tensors NVFP4 scheme in
`schemes/nvfp4.h`); NemotronH's `lm_head` becomes a fourth. It is PRE-EXISTING
and NOT addressed here: it arrived at `80d1da096`, and the function body is
byte-identical at this row's merge base and at its head
(`sha256 bf4685f4...` over the whole definition, both sides). Nothing in this
row's diff touches it. It is called out because the asymmetry — refusing one
shared static by name while silently taking another — is the kind of thing a
reader is entitled to see stated rather than to discover. The
behavioural red for this class needs CUDA and is PENDING with the rest.

### The DSR ratchet, which no review round ever saw run

`device-leakage` had not COMPLETED on this row through three review passes, so
its verdict was never an input to any of them. It completed after the third and
failed: `vt_ifdef` **35 against a baseline of 32**, `rc 1`. Three
`#ifdef VT_MARLIN_NVFP4` sites had been added to the device-agnostic shared
layer, and the ratchet exists precisely to stop that accreting.

The three, located by running `scripts/check-device-leakage.py --report` at the
failing head `7a3909187` and diffing the per-file table against `9ecaf1bb3`
rather than by reading the diff for guards:

| # | site at `7a3909187` | what the guard decided | resolution |
|---|---|---|---|
| 1 | `include/vllm/model_executor/models/dense_nvfp4_gemm.h:768` — `MarlinW4A16Selects` | **nothing.** `MarlinW4A16Enabled()` is declared above the guarded region and `vt::OpRegistered` is the op table's own availability answer | **removed** |
| 2 | `src/vllm/model_executor/models/nemotron_h_device.cpp:883` — around `LmHeadNvfp4View` | **nothing.** The function names no symbol the Marlin build adds | **removed** |
| 3 | `src/vllm/model_executor/models/nemotron_h_device.cpp:985` — `DeviceLmHeadD`'s body | `ResidentIn` and a complete `dense_nvfp4::MarlinDenseResident`, neither of which EXISTS without the guarded arena region | **`DSR-ALLOW(A2-Q2b)`** |

**(1) is the case the checker's own message describes.** The build flag and the
registration are the same condition, not two: `CMakeLists.txt`'s single
`if(VLLM_CPP_MARLIN)` block adds `src/vt/cuda/cuda_moe_marlin.cu` — whose
file-scope `Registrar` holds the tree's only
`RegisterOp(OpId::kMoeGroupedGemmNvfp4Marlin, …)` — and defines
`VT_MARLIN_NVFP4=1`, in that same block. A build without the macro therefore
registers nothing and the query already resolves false on exactly the builds the
`#ifdef` excluded. This is the call `nemotron_h_device.cpp`'s `moe_on_device`
selection had already made, in a comment that says so.

**(2) was measured, not reasoned.** The claim is that the function's external
linkage at `namespace vllm` scope is what makes an unused definition harmless in
a build where its only call site is compiled out. Adding `static` to that
definition — the mutation that removes exactly that property — turns the same
CPU compile RED, `error: 'vllm::Nvfp4Weight vllm::LmHeadNvfp4View(...)' defined
but not used [-Werror=unused-function]`, `rc 1`; the file was restored to an
identical sha256 (`9719ea70…`) afterwards.

**(3) is TYPES-not-behaviour, and takes the checker's documented escape hatch
rather than a baseline change.** `AGENTS.md` forbids making a red gate green by
widening an assertion, and a baseline bump is that. `DSR-ALLOW` is not: the site
is excluded from the count but COUNTED AND PRINTED on every run, so the
exemption is visible in CI output. It is the same class, and carries the same
stated reason, as the five sibling guards A2-Q2a and A2-P already hold in this
file. The SELECTION for this arm is a runtime op-table query
(`DeviceLmHeadEligible` → `MarlinW4A16Selects`, which now carries no guard);
only the call site needs the build guard, and its `#else` refuses by name.

**Measured on this tree**, a CPU build with `VT_MARLIN_NVFP4` absent from
`build/compile_commands.json` (positive control: 1020 `VLLM_CPP` hits in the
same file, so the grep is not silently wrong) — which is the configuration that
exercises both removals, since it is the arm the deleted `#else` branches used
to serve.

| what | before (`7a3909187`) | after | `rc` |
|---|---|---|---|
| `check-device-leakage.py` `vt_ifdef` | 35 | **32** | 1 → **0** |
| `DSR-ALLOW` exemptions in force | 20 | **21** | — |
| `scripts/device-leakage-baseline.json` | 32 | **32, untouched** | — |
| per-file table vs `origin/main` | +1 header, +2 model TU | **identical** | — |
| `nemotron_h_device.cpp`, `nemotron_h.cpp`, `qwen3_5.cpp` at `-Wall -Wextra -Werror` | — | compile | **0** |

Each of the three repairs is load-bearing, proven by reverting it alone in a
scratch worktree and re-running the gate. Every mutation was verified applied by
`git diff --stat` and restored to an identical sha256, with the unmutated
control green immediately before and after.

| mutation | applied | `vt_ifdef` | `rc` | verdict |
|---|---|---|---|---|
| — (control) | — | 32 | 0 | `ratchet holds` |
| **M-B** — restore the guard on `MarlinW4A16Selects` | 4 ins | **33** | **1** | **RED**, `DSR REGRESSION` |
| **M-C** — restore the guard around `LmHeadNvfp4View` | 2 ins | **33** | **1** | **RED**, `DSR REGRESSION` |
| **M-D** — delete the `DSR-ALLOW(A2-Q2b)` line | 1 del | **33** | **1** | **RED**, `DSR REGRESSION` |
| — (control, after restore) | — | 32 | 0 | `ratchet holds` |

The two pre-existing allowlist entries the report also prints —
`deepseek_v4_device.cpp [kcuda] x8` and `platform.cpp [dev_cast] x1` — are
byte-identical at `9ecaf1bb3` and here. This change moves one bucket and
nothing else.

**The #1392 overlay is retired.** The two evidence tables above in this section
carry an `overlay` column because this tree could not construct `GPUModelRunner`
at all ([#1371](https://github.com/mudler/vllm.cpp/issues/1371)) and every green
was taken with [#1392](https://github.com/mudler/vllm.cpp/pull/1392)'s fix
applied to the working tree and reverted. #1392 has since landed on `main` and
this branch has merged it, so the tree now carries the fix as a committed
object. The historical rows keep their `overlay` cells, because they describe
the tree they were measured on and rewriting them would make them false.

### The full CPU suite, and a red that arrived from `main` mid-repair

`origin/main` moved four times while this repair was gated, and the third sync
brought `4712dac40` (`VT-ACT-ROUND-POLARITY`, [#1322](https://github.com/mudler/vllm.cpp/issues/1322)
via [#1347](https://github.com/mudler/vllm.cpp/pull/1347)). Both `ctest` runs
below are on this box, `-Wall -Wextra -Werror`, no CUDA toolkit, and neither
carries an overlay of any kind — #1392 is a committed object here now.

| tree | base | `ctest` | result |
|---|---|---|---|
| this row + `9ecaf1bb3` | before `4712dac40` | 567 | **`100% tests passed, 0 tests failed out of 567`**, `rc 0` |
| this row + `01854663c` | after `4712dac40` | 569 | `rc 8`, **4 failed**: `test_minimax_music3_ar`, `test_ltx2_text_encoder`, `test_muse_glimmer_text`, `test_muse_glimmer_text_fallback` |

The four are [#1458](https://github.com/mudler/vllm.cpp/issues/1458), filed from
another flow before this control was run, and they are inherited rather than
caused. That is proven in both directions on the same tree, each mutation
verified applied by `git diff --stat` and restored to an identical sha256:

| control | change | `compile_rc` | `run_rc` (music3 / ltx2 / glimmer) | verdict |
|---|---|---|---|---|
| **A** | revert THIS row's two source files to their pre-repair `7a3909187` content | 0 | **1 / 1 / 1** | still RED — **not this row's** |
| **B** | revert `src/vt/cpu/cpu_ops.cpp` alone to `4712dac40^` | 0 | **0 / 0 / 0**, `37/37`, `27/27`, `24/24` | GREEN — **`4712dac40` is the cause** |

Control A is the one that answers the attribution question and it answers it
alone; control B is here because naming a cause is more useful to the next
reader than clearing oneself. Neither repairs anything: #1458 needs
`VT-ACT-ROUND-POLARITY` to decide whether the kernel or four never-re-derived
bf16 error floors are wrong, which is that row's oracle work and not a small,
clear, in-flow fix.

`test_nemotron_h_paged_forward` — the test all three of this row's red CI jobs
failed on — is green on both trees: `13 | 13 passed | 0 failed`,
`assertions: 3269 | 3269 passed | 0 failed`, `Status: SUCCESS!`, `run_rc 0`.
