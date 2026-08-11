# PERF-27B-DENSE-MARLIN-GATEUP — the dense W4A16 MLP launches two Marlin GEMMs where vLLM launches one

Issue: [#365](https://github.com/mudler/vllm.cpp/issues/365) (ranked work item 1)
Row: `PERF-27B-DENSE-MARLIN-GATEUP`
Base SHA: `d928e2c39e674a653ab4ca3366ef62eef6a8183e`
Measurement: [#362](https://github.com/mudler/vllm.cpp/issues/362) (our decode map)

## 0. Scope

Wire the **existing, default-ON** fused Marlin gate_up pair into the dense MLP
for the **W4A16** case, so the 27B issues one `[T,H]x[2I,H]` GEMM per layer
instead of two.

Out of scope: the cuDNN fp8 tower (#365 item 2, 48% of the gap, needs its own
spec and a new dependency); the CUTLASS W4A4 merged path, which already exists
and is untouched; anything in the MoE runners.

## 1. The defect, measured

Both arms traced on the identical batch-1 decode workload (#365):

| | ours | vLLM |
|---|---:|---:|
| Marlin ms/step | 47.0590 | 45.8048 |
| **Marlin calls/step** | **193** | **129** |

48 layers: ~4 Marlin GEMMs/layer for us, ~2.7 for vLLM. vLLM's production
topology is one `MergedColumnParallelLinear` `gate_up_proj`.

The fusion exists here and is not reachable on this path:

- `SharedGateUpFusedMarlinD` (`qwen3_5.cpp:2606`) issues ONE fused Marlin
  gate_up GEMM into `[M, 2N]`, backed by `MarlinDensePairResident`
  (`:2535`, built once at `:2549`).
- Its eligibility `SharedGateUpFusedEligible` (`:2600`) explicitly admits the
  non-W4A4 case: `!gw.IsTrueW4A4() && !uw.IsTrueW4A4()`.
- Its ONLY callers are `:4984` and `:6840`, both the **MoE shared expert**.

`DenseMlpBlock` (`:5901`) does have a merged path, but `MergedGateUpEligible`
(`:1980`) requires `gate.IsTrueW4A4() && up.IsTrueW4A4() && TrueW4A4Enabled()`.
`nvidia/Qwen3.6-27B-NVFP4`@`0893e160` is `modelopt_mixed` **W4A16**, so it is
ineligible and falls back to split gate + up Marlin GEMMs. That is the 193/129.

AGENTS.md §"Shared seams": "Route mergeable MLP projections through
`layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`." This path bypasses it.

## 2. Upstream anchor

- `vllm/model_executor/layers/linear.py` `MergedColumnParallelLinear` — the
  fused `gate_up_proj` topology.
- Qwen3.6 dense MLP builds `gate_up_proj` as one merged linear, so ONE GEMM per
  layer is the mirrored behavior, not an optimization we invented.

## 3. Design

1. Add a `DenseGateUpFusedMarlinEligible(w, d)` that mirrors
   `SharedGateUpFusedEligible`'s shape and scale conditions against
   `DenseMlpWeights` (`gate_proj_fp4`, `up_proj_fp4`): both non-empty, both NOT
   true-W4A4, `n` and `k` equal, `scale2` equal.
2. In `DenseMlpBlock`, when the CUTLASS W4A4 merged branch is ineligible and the
   new predicate holds, call the fused Marlin pair instead of two
   `MatmulNvfp4MarlinD` launches, then feed the existing `[T,2I]` silu/mul sink.
3. Behind an env toggle for a same-binary A/B, default **OFF** while unmeasured.
   Flip the default only after the A/B measures a win, per the standing lesson
   that a lever's default moves on evidence. **The A/B measured a win and the
   default is now ON** -- see `## Outcome`.
4. Do NOT duplicate `MarlinDensePairResident`. Reuse it. If its keying (`const
   Nvfp4Weight* gate`) does not admit the dense weights, extend the existing
   cache rather than adding a parallel one — a hand-rolled second resident is
   exactly the parallel path AGENTS.md forbids.

## 4. Risks

- **Numerics.** The pair fusion concatenates the two operands; it must be
  bit-identical to the split path. `SharedGateUpFusedEligible` requires
  `gw.scale2 == uw.scale2` precisely because a shared scale is what makes the
  concatenation legal. Assert that, do not assume it, and prove byte-identity.
- **The gain may be launch-overhead only.** 193 -> ~145 calls/step saves ~48
  launches. If the fused GEMM is not itself faster, the saving is bounded by
  launch overhead, which at 0.24 ms/call average is not the whole +1.26 ms.
  Measure; do not assume the full delta is recoverable.
- **`scale2` may differ per shard** on this checkpoint, exactly as the GDN
  alphas turned out to be 0-of-48 folded (#339). CHECK IT on the real weights
  before assuming the fusion is reachable — that check is cheap and it is what
  made #339's Phase A measure flat.

## 5. Tests and evidence

- RED-first unit: fused vs split gate_up on the same weights, asserting
  **byte-identical** output, and an f32 comparison arm (a bf16 store is known on
  this project to absorb a real defect that only f32 catches).
- Token-exact gate on the 27B with the toggle ON.
- Same-binary A/B at c1 and c8, toggle the only variable, arms interleaved.
- A decode-window trace showing Marlin calls/step actually fell from 193, per
  #362's method (whole-run trace, windowed by the profiler's own
  cudaProfilerStart/Stop; `--cuda-graph-trace=node` mandatory).

## 6. Stop conditions

- If `scale2` differs across gate/up on the gate checkpoint, the fusion is
  unreachable as designed. STOP and report; do not silently relax the equality,
  which would change numerics.
- If the A/B measures flat, record the negative with its regime and do NOT flip
  the default. (NOT triggered: the A/B separated completely at both c1 and c8.)

## 7. Honest sizing

This targets **29%** of a measured +4.40 ms/step gap, i.e. ~1.26 ms of 84.85
(~1.5%). On its own it would move the 27B from 0.9561x to roughly 0.970x. It
does NOT reach parity. The larger term is the cuDNN fp8 tower at 48%, which is
a separate row. Nothing here should be described as closing the gap.

## Outcome

**Landed and measured. The default is ON.**

### What was measured

Same-binary interleaved A/B, `VT_DENSE_MARLIN_GATEUP` the only variable, 4 reps
per arm, caches dropped between arms, FA2 marker verified in the configure log,
and the toggle verified present in the binary -- an A/B against a binary lacking
the toggle silently compares OFF against OFF and reports a confident zero.

| c | split mean | fused mean | effect |
|---|---:|---:|---:|
| 1 | 11.8313 (spread 1.0051) | 12.0823 (spread 1.0120) | **+2.12%** |
| 8 | 82.2217 (spread 1.0054) | 83.6186 (spread 1.0102) | **+1.70%** |

Complete separation at both concurrencies -- every fused rep beats every split
rep (c1: min fused 12.0203 > max split 11.8729; c8: min fused 83.2511 > max
split 82.4959). 4/4 paired, effect well outside each arm's measured spread.

### The mechanism, proven rather than inferred

A throughput A/B cannot distinguish a working lever from one silently not taken
-- both can read flat. A decode-window trace counted the launches on the SAME
binary, both arms passing the integrality check (0/20 non-integral):

| arm | Marlin calls/step | Marlin ms/step | GPU busy ms/step |
|---|---:|---:|---:|
| split | **193.000** | 46.7747 | 84.5544 |
| fused | **129.000** | 45.4176 | 83.1344 |
| vLLM (reference) | **129** | 45.8048 | 80.4492 |

We now issue **exactly vLLM's 129 Marlin calls per step** -- 129.000 against 129
-- and our Marlin time is marginally better than the reference's. GPU busy fell
1.42 ms/step (-1.68%), independently corroborating the A/B from a different
instrument.

### Numerics

**Tokens identical between arms**, verified by a greedy 64-token continuation
captured on each arm and diffed.

Byte-identity was deliberately NOT asserted, and this spec's original §4/§5
wording demanding it was **wrong**. Marlin's fp32 split-K groups K-slices
differently for a `[2N,K]` operand than for two `[N,K]` operands -- ~1 bf16 ULP
on ~0.1% of elements, recorded from a measured run at
`tests/vllm/model_executor/layers/test_linear_method.cpp:190-202`. The
implementer refused to write a gate it knew to be false and used token-exactness
instead, the bar the sibling shared-expert pair flipped ON under. That was the
correct call, and it held: the ULP moved no tokens.

### What this did NOT establish

A new **parity ratio**. The A/B harness reads 11.83 tok/s at c1 where the
canonical binding grid reads 10.756 -- a different denominator. Translating a
percentage across harnesses is the error that produced the 0.964-vs-0.867
confusion earlier in this campaign, where two ratios 8x apart in absolute scale
were compared as equivalent. The binding number requires re-running the
canonical grid against vLLM with the default ON, and is **owed**.

### Where it leaves the gap

The measured 27B gap was +4.40 ms/step (#365). This row closes ~1.42 ms of it,
**32%**. Remaining **+2.68 ms**: the fp8 tower at +2.12 and an unattributed
residue at +1.02, with Marlin now a -0.39 credit. GPU-busy ratio 0.948 -> 0.968.

The tower term was **reframed** during this row: vLLM does not reach cuDNN
through any linear layer -- those kernels come from torch/Inductor, and
`matMul_pointwise_pointwise` is cuDNN's fusion engine folding the epilogue into
the GEMM. So that gap is **epilogue fusion**, not a missing dependency, and
`vt::FusedChain` is the named seam. That row must map the kernels via NVTX
scopes before implementing -- inferring instead of tracing already caused one
retraction in this campaign.
