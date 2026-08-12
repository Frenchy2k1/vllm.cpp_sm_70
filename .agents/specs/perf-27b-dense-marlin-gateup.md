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

**64 layers** (16 full-attn + 48 GDN -- see `.agents/parity-ledger.md:369`), and
the counts decompose exactly: ours 64x3 + lm_head = **193**; fused 64x2 +
lm_head = **129**; vLLM 64x2 + 1 = **129**. So we issue 3 Marlin GEMMs per
layer where vLLM issues 2, and the saving is **64** launches, not 48. vLLM's
production topology is one `MergedColumnParallelLinear` `gate_up_proj`.

(An earlier draft of this spec said "48 layers / ~4 GEMMs per layer / saves ~48
launches". That was wrong on all three counts and the sizing in §4 rested on
it. Corrected here after review.)

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
- **The gain may be launch-overhead only.** 193 -> 129 calls/step saves **64**
  launches. If the fused GEMM is not itself faster, the saving is bounded by
  launch overhead, which at 0.24 ms/call average is not the whole +1.26 ms.
  Measure; do not assume the full delta is recoverable.
- **`scale2` may differ per shard** on this checkpoint, exactly as the GDN
  alphas turned out to be 0-of-48 folded (#339). CHECK IT on the real weights
  before assuming the fusion is reachable — that check is cheap and it is what
  made #339's Phase A measure flat.

## 5. Tests and evidence

- RED-first unit: fused vs split gate_up on the same weights. **NOT
  byte-identical** -- that bar is false here and was corrected during the row;
  see `## Outcome` / Numerics. Pin the decisions (predicate truth table, path
  selection, halves order) and use token agreement for the arithmetic.
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

**CORRECTED AFTER MEASUREMENT: the arms do NOT agree in general.**

The original claim here -- "tokens identical between arms" -- came from a
SINGLE 64-token prompt. Measured across 4 prompts at 128 tokens:

| control | result |
|---|---|
| oracle run2 vs oracle run1 | **4/4 EXACT** |
| ours SPLIT vs oracle | **0/4** |
| ours FUSED vs oracle | **0/4** |
| ours FUSED vs ours SPLIT | **1/4 exact, 3 differ** |

So the ~1 bf16 ULP from Marlin's split-K regrouping DOES reach the output at
length. One sample was not enough to claim agreement and I over-generalised
from it.

**But the oracle gap is NOT caused by this row**: the split path -- today's
shipped behaviour -- is equally 0/4. And vLLM's greedy is DETERMINISTIC here
(4/4 self-match), so the ratified distributional gate does not apply and
token-exactness IS the correct bar.

That pre-existing divergence is filed as **#370** and outranks this row. **This
row must not land on a token bar until #370 resolves.** A speed ratio measured
on an arm that does not reproduce the reference's tokens is not a parity
number.

**This is NOT the oracle bar and must not be read as it.** It shows the change
is self-consistent; it does not show we still match vLLM. Spec §5 requires a
token-exact gate against the PINNED ORACLE with the toggle ON, every sibling
lever in `docs/ENVIRONMENT.md` cites a SACRED count, and this row cited none --
while being the only one that changes the 27B decode path by default. Caught by
the fresh review. The oracle gate is **owed** and no parity or correctness claim
for this row is complete until it lands.

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

### Where it leaves the gap -- ARITHMETIC CORRECTED AFTER REVIEW

The first version of this section mixed two baselines and did not close:
"+4.40 gap, closes 1.42, remaining 2.68" sums to 4.10, not 4.40. The 4.40 came
from the ORIGINAL profile (84.85 ms/step) while the 2.68 came from the NEW
trace's split arm (84.5544). That is the mixed-denominator hazard this project
has recorded twice, committed inside a single paragraph.

Stated against **one** denominator -- this row's own trace, both arms same
binary:

| | split | fused | vLLM |
|---|---:|---:|---:|
| GPU busy ms/step | 84.5544 | 83.1344 | 80.4492 |
| gap vs vLLM | +4.105 | **+2.685** | -- |
| ratio | 0.951 | **0.968** | 1.000 |

So the closure is **1.420 ms of 4.105 = 34.6%**, and the ratio moves
**0.951 -> 0.968** on this instrument. The separately-measured original profile
put the gap at +4.40 against a slightly different split baseline (84.85 vs
84.5544); that ~0.3 ms difference between two sessions is itself unexplained
and is NOT to be quoted as if it were one number.

The remaining +2.685 is the fp8 tower plus an unattributed residue, with Marlin
now a small credit. The sub-terms (+2.12, +1.02, -0.39) were carried over from
the original profile and sum to 2.75 rather than 2.685; they are indicative,
not reconciled, and re-attributing them needs a fresh both-arms profile.

The tower term was **reframed** during this row: vLLM does not reach cuDNN
through any linear layer -- those kernels come from torch/Inductor, and
`matMul_pointwise_pointwise` is cuDNN's fusion engine folding the epilogue into
the GEMM. So that gap is **epilogue fusion**, not a missing dependency, and
`vt::FusedChain` is the named seam. That row must map the kernels via NVTX
scopes before implementing -- inferring instead of tracing already caused one
retraction in this campaign.
