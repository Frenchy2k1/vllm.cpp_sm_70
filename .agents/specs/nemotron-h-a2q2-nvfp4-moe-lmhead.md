# A2-Q2 — NemotronH's 23 MoE blocks and `lm_head` reach the device on NVFP4 W4A16

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Governing spec:** [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md) — this file
owns the NVFP4 half of what that spec's §1 called `A2` and what the A2-R landing
commit (`598226e96`) renamed `A2-Q`.
**Sibling:** [`nemotron-h-a2q1-fp8-mamba.md`](nemotron-h-a2q1-fp8-mamba.md).
**Base:** `origin/main` @ `0e1bee42f16b5f3fb3ae5a23869f6fd97bfc037d`.
**Not blocked on a build.** Its constraint is a REPORTING one
([#962](https://github.com/mudler/vllm.cpp/issues/962), §6) plus one DESIGN
decision ([#984](https://github.com/mudler/vllm.cpp/issues/984), §4.3).
**Pinned oracle:** `${VLLM_SOURCE}` @ `5559679229bc961848b121ccdeaa8fa5d79bec98`.
**Lifecycle at this commit:** unchanged; §9 records what the implementing change
owes.

**No product code is written by this spec.** Per the governing spec's §1.4 the
implementation is a separate PR by a different agent, and the reviewer is a
third.

---

## 0. Scope

The released checkpoint is `quant_algo: MIXED_PRECISION`. **5935 projections are
NVFP4 W4A16 group-16**: 5888 routed expert projections (128 experts x 2
projections x 23 layers), 23 shared-expert pairs, and `lm_head`. Together they
are 30.19e9 parameters, 15.8 GiB packed and 56.2 GiB dequantized to bf16 — which
is why a "dequant to bf16" device path was rejected at load
(`nemotron_h_loader.h:36-46`) and is rejected again here. Both gate hosts are
unified-memory, where that is a reboot rather than an OOM.

| In A2-Q2 | Out of A2-Q2 |
|---|---|
| the 23 MoE blocks on `vt::MoeGroupedGemmNvfp4Marlin` + the existing `vt::MoeRelu2` | the FP8 mamba arm — A2-Q1 |
| `lm_head` through `dense_nvfp4::MatmulNvfp4W4A16D`, with §4.3's residency decision made explicitly | paging, carried state, batching — A2-P / A2-B |
| supplementing the expert / `lm_head` weights with the shared `Nvfp4Weight` (`qwen3_5_weights.h:198`) | converting the FP8 weights — A2-Q1 |
| implementing `PrepareNemotronHForCausalLM` to host the load-time repack (§4.2) | fixing [#984](https://github.com/mudler/vllm.cpp/issues/984) or [#962](https://github.com/mudler/vllm.cpp/issues/962) |
| the per-block numeric gate for MoE and `lm_head` on the real checkpoint | any throughput number, on any axis |

**Explicitly out:** the MTP head (parent W5), the GGUF k-quant arm (parent W7),
`moe_latent_size` (null in the released checkpoint and already refused by name at
`nemotron_h.cpp:702-705`), and any change to the Qwen3.5, Laguna or Kimi-Linear
model files.

---

## 1. Geometry

From `tests/vllm/models/fixtures/nemotron_h_35_lightning/config.json`:

```
hidden_size 2688   n_routed_experts 128   num_experts_per_tok 6
moe_intermediate_size 1856   moe_shared_expert_intermediate_size 3712
n_shared_experts 1   vocab_size 131072   num_hidden_layers 52 (23 MoE)
```

**The experts are NON-GATED.** `NemotronHExpertWeights`
(`nemotron_h_forward.h:259-262`) holds only `up_proj` and `down_proj`;
`nemotron_h.py:220` is `ckpt_names=("up_proj","down_proj","")` and there is no
`gate_proj` tensor anywhere in the checkpoint. So the fused
`kMoeGroupedGemmBf16GateUpSilu` and Laguna's fused-`w13` lever **do not apply**.
The composition is: grouped GEMM (up) → `vt::MoeRelu2` → grouped GEMM (down) →
`vt::MoeCombine`.

`vt::MoeRelu2` (`include/vt/ops.h:406`, `:951`) already exists on CPU and CUDA
and was added for this architecture — its own comment cites
`nemotron_h.py:220`. Nothing new is owed for the activation.

---

## 2. Upstream anchors — `file:line` on both sides, at `555967922`

| Behaviour | Upstream | Ours |
|---|---|---|
| router, f32 out, `force_fp32_compute=True` | `nemotron_h.py:150-156` | `nemotron_h.cpp:712-731` |
| `e_score_correction_bias` | `nemotron_h.py:158-160` | `nemotron_h.cpp:715-716` |
| sigmoid scoring, grouped top-k | `nemotron_h.py:225` | `nemotron_h.cpp:733-750` |
| `routed_scaling_factor` on the OUTPUT, router's own factor forced to 1.0 | `nemotron_h.py:234`; `layer.py:291-300` | `nemotron_h.cpp:743-745`, `:819-823` |
| shared expert added UNSCALED after the routed sum is scaled | `moe_runner.py:402-406`, `:722-725` | `nemotron_h.cpp:808-815` |
| non-gated relu² expert | `nemotron_h.py:126-256` | `nemotron_h.cpp` `NonGatedExpert` |

**Three things this block is the whole reason to gate**, already stated at
`nemotron_h.cpp:678-685` and preserved verbatim by A2-Q2: the router runs in f32
(mirrored, not inherited — the annotated `f32` escape AGENTS.md allows); the
routed scale is applied to the output, not folded into the router weights or the
logits; and the shared expert is added unscaled.

---

## 3. ★ The arena correction — RECORDED so the next reader does not repeat it

The A2-Q brief stated that the MoE arm takes a **per-expert `[K,N]` device-pointer
array**, citing `deepseek_v2.cpp:369-413` and `qwen3_5.cpp:6066-6068`.

**That is `kMoeGroupedGemmBf16`, not the NVFP4 path.** Measured:

- `vt::MoeGroupedGemmBf16` (`include/vt/ops.h:1642`) does take
  `const Tensor& weight_ptrs` — an `[E]` i64 device array of `[K,N]` Matmul-B
  pointers.
- `vt::MoeGroupedGemmNvfp4Marlin` (`include/vt/ops.h:1685`) takes a **rank-3
  strided arena**: `b_q_weight [E, size_k/16, size_n*8/pack]` i32, validated at
  `src/vt/ops.cpp:884` (`"moe_marlin: b_q_weight must be rank-3 [E, K/16, N*8/pack]"`).
  At `pack = 4` that is `[E, K/16, N*2]`. Scales are `[E, K/16, N]` fp8-e4m3 and
  `global_scale` is `[E]` f32, both **processed**, not raw — `marlin_permute_scales`
  then `nvfp4_marlin_process_scales`, and `nvfp4_marlin_process_global_scale`.

The two are not interchangeable and the difference is the whole shape of the
work: the arena needs a **load-time repack**, the pointer array does not.

**Size A2-Q2 against the arena.** Per layer, non-gated:

| Slab | Shape | Bytes |
|---|---|---|
| up weight | `[128, 2688/16, 1856*2]` i32 | 319 MB |
| down weight | `[128, 1856/16, 2688*2]` i32 | 319 MB |
| up scales | `[128, 168, 1856]` i8 | 39.9 MB |
| down scales | `[128, 116, 2688]` i8 | 39.9 MB |

≈ **718 MB per layer x 23 layers ≈ 16.5 GB repacked, device-resident.** The raw
packed `ResidentNvfp4` copies must be freed as the repack proceeds, as
`qwen3_5.cpp` already does ("free the fp4 device originals... so peak"), or peak
device residency roughly doubles. **Measure peak on both hosts and record it**;
GB10 has OOM-rebooted on host RSS at `gpu_memory_utilization` 0.25, and Thor
reboots instead of OOM-killing.

**Follow the two existing instances, do not invent a third shape.** The
multi-expert arena machinery lives in the model TUs, not the shared header:
`qwen3_5.cpp:794` (`ResidentIn<MoeMarlinResident>(w->resident_marlin)`) and
`laguna.cpp:730-752`. The shared header's only use of
`MoeGroupedGemmNvfp4Marlin` is with `num_experts=1` as a dense GEMM
(`dense_nvfp4_gemm.h:41`, `:556`).

---

## 4. Design

### 4.1 The weights

Supplement, do not replace: add `Nvfp4Weight` beside the existing
`NemotronHOwned` on `NemotronHExpertWeights` and `NemotronHHostWeights::lm_head`,
filled on the CUDA path, with `NemotronHOwned` retained for the host reference
arm. Replacing outright deletes the operand the numeric gate compares against.

Loader anchors: `LoadNvfp4` (`nemotron_h_weights.cpp:578-609`), called at
`:703-704` (experts) and `:1088` (`lm_head`).

> **`HostBytesOf` (`nemotron_h_weights.cpp:745`) feeds `rep.host_bytes`, pinned
> as the exact literal `18888922112` at
> `tests/vllm/models/test_nemotron_h_loader.cpp:310`.** A copy-preserving
> supplementation keeps it green; a borrowing one changes it. Either is
> acceptable, but the literal is re-derived and updated **in the same change**
> with the arithmetic shown — never adjusted to whatever the run printed.

### 4.2 `PrepareNemotronHForCausalLM` must be implemented

`PrepareNemotronHForCausalLM` (`nemotron_h_registry.cpp:113-118`) is a pure
no-op — three `(void)` casts. **The repack belongs there**, not in the forward:
a 16.5 GB allocation-and-repack inside a forward would land inside a CUDA-graph
capture. `PrepareQwen3_5Dense` (`qwen3_5_dense.cpp:104-115`) is the pattern,
calling `PrepareLmHeadResident` and `PrepareGdnFp8Resident` from the prepare
hook.

**Implementing it is a shared-seam change**, not a model-local detail: it is a
`ModelFactory::prepare` slot that currently does nothing, and giving it work
changes when this model allocates. Call it out for the reviewer.

### 4.3 ★ `lm_head` residency — an EXPLICIT decision, not a default

There are **two** functions named `MarlinDenseResidentFor`, with different safety,
selected by translation unit:

```
dense_nvfp4_gemm.h:379   static std::unordered_map<const Nvfp4Weight*, MarlinDenseResident> cache;   ADDRESS-KEYED
qwen3_5.cpp:2429         return ResidentIn<MarlinDenseResident>(w->resident_marlin);                 SLOT-KEYED, safe
```

Each is used by the `MatmulNvfp4W4A16D` in its own TU — `dense_nvfp4_gemm.h:730`
and `qwen3_5.cpp:2521`. **The shared seam holds the unsafe one.** `Nvfp4Weight`
already carries an unused `resident_marlin` `ResidentSlot`
(`qwen3_5_weights.h:258-260`), which is exactly what
[#237](https://github.com/mudler/vllm.cpp/issues/237) added to fix this.

This inverts the incentive the shared-seam rule creates: following the seam gets
you the defect, hand-rolling gets you the fix. And the safe behaviour is **not
discoverable from the seam** — a reader of the header has no reason to suspect a
better implementation exists elsewhere under the same name. Filed as
[#984](https://github.com/mudler/vllm.cpp/issues/984), which asks for a
two-engine red-before, since an address key is invisible to any single-engine
gate.

> **A2-Q2 states its choice and its evidence rather than taking a default.** The
> options are: (a) route `lm_head` through the seam's `MatmulNvfp4W4A16D` and
> accept the address-keyed cache, recording the hazard; (b) route through the
> `resident_marlin` slot as `qwen3_5.cpp` does, diverging from the seam with the
> reason recorded; or (c) declare #984 a base and wait.
>
> A2-Q2 is precisely the second-consumer condition an address key cannot survive
> — `lm_head` plus 5935 expert projections through one header — so (a) is the one
> option that must not be taken silently. Whichever is chosen, it is argued in
> the commit message, where the reason attaches to the diff.

### 4.4 The forward

A `NemotronHMoeMixerDevice` in `nemotron_h_device.cpp` mirroring the host arm's
composition statement for statement: f32 router GEMM → `vt::MoeRouterTopK`
(sigmoid, grouped, `routed_scaling_factor = 1.0f`) → `MarlinMoeAlignBlockSize` →
grouped GEMM (up) → `vt::MoeRelu2` → grouped GEMM (down) → shared expert →
`vt::MoeCombine` with `params.routed_scaling_factor`.

`lm_head` replaces the host `NemotronHHostLmHead` call at
`nemotron_h_device.cpp:501`. **Note what that costs the gate:** A2-R's token
comparison was attributable *because* both arms ended in the identical host
projection (`nemotron_h_forward.h:452-457`). A2-Q2 removes that property, which
is exactly why its gate is per-block numeric and not token-only.

---

## 5. Gates

### 5.1 The gate

**Per-block numeric equivalence against the host reference on the REAL
checkpoint**, via `NemotronHTrace`: the device MoE arm compared against
`trace.mixer[l]` at **every one of the 23 MoE layers**, and `lm_head` compared
against the host projection on the same gathered rows — plus hybrid-vs-host token
identity.

Note that today's device tests are driven by a **synthetic tiny fixture**
(`TinyParams` / `BuildTiny`, `test_nemotron_h_forward.cpp`), all of it `kDense`.
A real-checkpoint per-layer comparison **does not exist yet** and is part of this
unit's work, not a harness it inherits.

**Hosts:** `dgx.casa` (GB10, sm_121a) and `192.168.68.23` (Thor, sm_110) — but
see §6 before quoting a Thor number.

### 5.2 Bands are MEASURED, and the guard is a PROPERTY

Same two lessons as the sibling spec, both already paid for on this row: a bf16
band of `3e-2` sat *above* a `2.11e-2` defect and accepted a wrong answer; and a
"safety factor" guard compared two compile-time constants and observed nothing.
`DevRelFor` (`test_nemotron_h_forward.cpp:1596`) is the model.

> **Derive every band from a measurement taken in the case. Make the guard a
> property: the perturbed answer, run through the same arithmetic that accepted
> the real one, must come out rejected. No stored twin, no invented safety
> factor.**

### 5.3 Mutations

Applied **alone**, in a scratch copy, rebuilt, run, tree restored to the
**baseline sha** — the restore is the control that catches `shutil.copy2`
preserving mtime so ninja skips the rebuild.

| # | Mutation | Must RED |
|---|---|---|
| Q2-M1 | NVFP4 nibble order flipped (high-first) | the MoE numeric gate AND the `lm_head` gate |
| Q2-M2 | `weight_scale_2` / `global_scale` ignored | both |
| Q2-M3 | expert stride off by one in the arena | the MoE numeric gate |
| Q2-M4 | `routed_scaling_factor` folded into the router logits instead of the output | the MoE numeric gate |
| Q2-M5 | the shared expert added BEFORE the routed sum is scaled | the MoE numeric gate |
| Q2-M6 | `vt::MoeRelu2` replaced by plain relu | the MoE numeric gate |
| Q2-M7 | the device call site deleted, host arm left in place | the gate must go RED — otherwise it measures a class, not a capability |

**Report per mutation:** the exact `[doctest] test cases:` / `assertions:` /
`Status:` lines, a **non-zero case count**, compile exit AND error count, and a
binary sha256 distinct from baseline. **Never put a comma in a `TEST_CASE` name**
— `-tc` splits on commas, selects zero cases, prints `SUCCESS!`, exits 0.

### 5.4 Baseline

Measured 2026-08-16 in a clean detached worktree at
`0e1bee42f16b5f3fb3ae5a23869f6fd97bfc037d`, local x86_64 dev box:
`scripts/agent-preflight.sh` **fully green** (all 26 record gates including the
seven #873 ones, plus both mutation suites). Release build, **0 warnings**,
1441/1441 linked. `ctest -R "nemotron_h|ops_mamba2|ops_fp8|linear_method"`
**13/13 passed**.

The #873 gates are FIXED — a red on any is the implementer's. Genuinely
inherited: `windows-msvc-*` (#584, PR-only), `test_cpu_x86_llamacpp_floor` under
load (`NO_QUIET_WINDOW` exit 4), `test_async_llm` under `ctest -j` (#294 — re-run
serially first), and on GB10 `test_qwen3_5_gdn_spec_routing` 119/123 and
`test_linear_method` 83/85 (#907).

---

## 6. ★ The Thor leg is OWED DEBT, stated — not a footnote

[#962](https://github.com/mudler/vllm.cpp/issues/962): on sm_110,
`tests/vt/test_ops_moe_grouped.cpp:1144` fails with

```
NVFP4 block8-vs-block16  M=8 K=4096 N=4096  bitdiff=15/32768
```

Two NVFP4 groupings that must agree disagree in 15 of 32768 output elements.
This is **not** feature absence: `marlin-nvfp4` is ENABLED for `[110]`
(`cmake/CudaArchFeaturesTest.cmake:145`), so the kernel is present, running, and
disagreeing with itself on a live path — the exact kernel family A2-Q2's MoE and
`lm_head` arms run. Which grouping is correct is **not established**, and a small
bitdiff is not evidence of a small problem.

> **A2-Q2 does not quote a Thor number as a result while #962 is open.** It
> records the Thor leg as **PENDING #962**, with the measurement above, in
> `docs/BENCHMARKS.md` and in the PR body. An arm whose gate cannot run on a
> required host is owed debt and must be visible as such — pending is a result;
> silence is not.

Running the gate on Thor anyway and reporting what it printed is the failure this
section exists to prevent: the numbers would look like a result and would rest on
a kernel already known to contradict itself.

#962 is listed under §11 `## Owed`.

---

## 7. ★ G-SAFE stays FULLY intact

All three clauses at `src/vllm/model_executor/models/nemotron_h_registry.cpp:162`
— `input.attn_kv.empty() && input.gdn_state.empty() && input.num_reqs <= 1`.

A2-Q2 creates no paging, no carried state and no batching, so **nothing may be
weakened or narrowed.** A2-P is where it narrows. A reviewer who cannot point at
all three intact returns FAIL.

## 8. A2-Q2 inherits A2-R's UNREACHED posture

`NemotronHDeviceForward` is called only from
`tests/vllm/models/test_nemotron_h_forward.cpp:1805`;
`ForwardNemotronHForCausalLM` still routes to the host `NemotronHForward`
(`nemotron_h_registry.cpp:185-187`). A2-R disclosed this and assigned the wiring
to **A2-P**.

A2-Q2 does not change it, and **its commit body and PR body must say so** —
naming what is unreached, the row that owns the wiring (A2-P) and the issue
(#810), per AGENTS.md §"Nothing lands dead". Listed under §11 `## Owed`.

**One nuance:** §4.2 makes `PrepareNemotronHForCausalLM` non-trivial, and that
hook **is** reached from production load. So A2-Q2's repack runs in production
while the forward that consumes it does not. Say that plainly rather than letting
"unreached" imply nothing new executes.

---

## 8.1 Risks

**R1 — peak device residency.** §3's 16.5 GB assumes the raw packed copies are
freed as the repack proceeds. If they are not, peak roughly doubles on two
unified-memory boxes, one of which reboots instead of OOM-killing. Measure peak,
do not estimate it, and do not run a large oracle alongside `ctest`.

**R2 — the address-keyed repack cache.** §4.3. Invisible to any single-engine
gate, so it will not show up in this unit's own testing. Decide explicitly.

**R3 — a too-wide dtype passes every gate this unit owns.** The router's f32 is
upstream's own polarity and stays; anything *else* that names f32 on this path
owes a one-line reason at the buffer. Compare the memory format against the
oracle explicitly rather than inferring it from matching tokens.

**R4 — `origin/main` moves under a shared checkout.** Merge an immutable SHA and
re-run the full gate after merging rather than reading the diff (#818). Never
force-push.

**R5 — the fixture must be the checkpoint the changed path loads.** Pin revision
`29f2d174`; leave `VT_NEMOTRON35_SNAPSHOT` UNSET so the revision check binds
(`tests/parity/test_hf_snapshot_pinning.cpp:62`) and record the resolved
directory.

### 8.2 Stop conditions

- The repack cannot be made to fit peak memory on either host →
  **`NEEDS_DECISION`**, with the measured peak; do not trim the gate to fit.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name, record as owed, never silently dequantize.
- #962 turns out to affect GB10 too → **stop and report.** The unit then has no
  trustworthy host and that is a finding about the Marlin kernel, not about
  A2-Q2.
- Any temptation to widen a band until it passes → **stop.** §5.2.

---

## 9. Records owed on landing

A2-Q2 changes lifecycle state, so it owes `docs/STATUS.md`, `docs/BENCHMARKS.md`
(with the Thor leg recorded **PENDING #962** — a pending result is a result), the
row spec's `## Now`, and the row + checklist entry + rollup in
`.agents/model-matrix.md` in the **same** change
(`scripts/check-model-checklist.py` enforces the rollup). Plus `docs/FEATURES.md`
if the NemotronH row's wording changes, and `docs/USAGE.md` if
`scripts/check-doc-checkpoint.py` classifies anything new as user-facing — run
the checker rather than assuming.

It does **not** remove `scripts/runner-routing-allowlist.txt:26`: that entry's
removal condition is the device/paged runner path, which is A2-P's.

## 10. Now

**State at this commit:** spec only. No product code, no lifecycle change.

A2-Q2 is **claimable now** — its constraints are a reporting one (§6) and a
design decision (§4.3), neither of which blocks a build. A fresh implementer
claims this file, captures a RED per §5.3 first, and lands the MoE and `lm_head`
device arms with G-SAFE untouched. A fresh reviewer — never the implementer —
runs the §5.3 mutations.

**Three things to read before the first edit:** §3, so the arena is not sized as
a pointer array; §4.3, so the `lm_head` residency is a decision rather than a
default; and §6, so the Thor leg is reported as pending rather than quoted.

## 11. Owed

- [#962](https://github.com/mudler/vllm.cpp/issues/962) — NVFP4 Marlin disagrees
  with itself on sm_110. A2-Q2's Thor leg is PENDING on it (§6); the fix is not
  this unit's.
- [#984](https://github.com/mudler/vllm.cpp/issues/984) — `dense_nvfp4_gemm.h`
  keys the Marlin repack cache on the weight's address. A2-Q2 decides how to
  route around it (§4.3); the fix is not this unit's.
- The device arm has **no production caller** until A2-P wires it through
  `ModelRegistry::Forward` (§8). Tracked on
  [#810](https://github.com/mudler/vllm.cpp/issues/810).

## 12. Outcome

Not yet written. Filled when the unit reaches `DONE`: what was measured, what
was rejected and why, and why each default is set the way it is.
