# Spike: SGLang as an ORACLE — the SGLang-parity gate methodology

**Owning program:** `CLAIM-SGLANG-PARITY-PROGRAM` (the SGLang parity PROGRAM —
the SGLang analogue of the vLLM-parity approach). This spec is the
**gate methodology**; the enumerated surface is
[sglang-matrix.md](../sglang-matrix.md). Read-only scoping spike: no engine code,
no measurement taken here.

**User directive (2026-07-27):** "we need the same vLLM approach there [SGLang]
for reaching parity." This replicates, for SGLang, the four vLLM-parity standing
directives of [AGENTS.md](../../AGENTS.md): tabular inventory spike-first;
always compare vs the oracle on the same workload; port the tests with the code;
mirror, never ask; never accept a ceiling.

**The one policy caveat that makes SGLang DIFFERENT from the vLLM oracle.**
MIRROR-vLLM is the PRIME directive ([AGENTS.md](../../AGENTS.md) STANDING
DIRECTIVE): vLLM is the source of truth for BEHAVIOR (when vLLM has an answer we
mirror it). SGLang is a **competitor engine**, not the mirror source. So SGLang
serves two honest roles and one forbidden one:

- **Correctness CROSS-CHECK (not a behavior spec).** Where SGLang and vLLM agree
  on greedy tokens for the same model, ours must match both; where they diverge
  (different sampling tie-breaks, different default rope/quant resolution), **vLLM
  wins** — we do not re-mirror to SGLang. SGLang correctness capture is a
  regression net and a divergence detector, never an instruction to change a
  vLLM-derived behavior.
- **Performance FLOOR (binding).** Wherever SGLang is faster than vLLM on an
  equivalent workload, SGLang is a binding floor (AGENTS.md "Additional
  competitor floor"). This is the primary reason SGLang is a full parity target,
  not just a curiosity.
- **Forbidden:** copying SGLang's Python data structures as a second,
  incompatible abstraction (recorded rule,
  [cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md#L594);
  restated in [sglang-radixattention.md](sglang-radixattention.md) §10). A
  SGLANG-DISTINCT behavior is added as an opt-in over our vLLM-derived design,
  never as a fork of the engine.

---

## 1. Reference pins and artifacts

| Reference | Pin | Ground |
|---|---|---|
| SGLang source | tag `v0.5.15`, commit `f63458b5beaceabbd9d749b9fc956370e1b649e6` | cloned to `/home/mudler/_git/sglang`; every SGLang `file:line` in the matrix + specs is from this tree |
| SGLang runtime image (perf + shared-prefix) | `lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667eca4e6fff64f7758c5fb1720e16faa806f90ea767e018bb8fa1b09dd44` (manifest list carries **arm64 AND amd64**) | the digest-pinned production artifact recorded by `BACKEND-GATE-CUDA-SGLANG-PREFIX`; CUDA 13 for CUDA-13 hosts |
| SGLang runtime image (existing v0.5.13 harness) | `lmsysorg/sglang:v0.5.13-cu130-runtime@sha256:9631280f57d95503ed64cf3892de72190aafbfe6e58e90718a019fa775113bfb` | the pin the existing `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` harness already targets; re-pin to v0.5.15 deliberately, never silently |
| vLLM oracle (the mirror source) | `555967922` / 0.26.0.dev0 (+ transformers 5.14.1) | the correctness + CUDA-perf floor; SGLang parity NEVER weakens the vLLM gate |
| Load client | SGLang `bench_serving` from the same pinned image | it maps `sglang-oai` and `vllm` to the SAME OpenAI completions request path (`bench_serving.py:874-887`), so the two engines are driven identically |

The official image is a verified production artifact, not a floating tag; the
release workflow builds arm64 + amd64 and joins them into the manifest list
([low-concurrency spike](cuda-sglang-low-concurrency.md) §"SGLang runtime image").

---

## 2. Which board can run SGLang — and the honest setup cost

**The headline: SGLang runs on our gate box (dgx GB10, sm_121a, aarch64,
CUDA 13) via the OFFICIAL arm64 CUDA-13 image — NO from-source sm_121a build is
required.** This is the crucial asymmetry vs our vLLM oracle: the vLLM oracle on
GB10 is a from-source sm_121a build with no wheel
([pin-advance.md](pin-advance.md)), whereas SGLang ships a digest-pinned arm64
CUDA-13 runtime image that already carries (or JITs) the Blackwell kernels. A
public reproduction of SGLang serving Qwen on a GB10 Spark exists
([DGX recipe](https://github.com/Weschera/qwen-sglang-dgx-spark/tree/03253ef98c01de59a21c85b9a5cc6a27a871c383),
referenced by `BACKEND-GATE-CUDA-SGLANG-PREFIX`).

| Board | Arch | Can it run pinned SGLang? | Cost / caveat |
|---|---|---|---|
| **dgx GB10 (`dgx.casa`)** | sm_121a, aarch64, CUDA 13 | **YES — via the arm64 `v0.5.15-cu130` image.** The primary board. | Pull the digest-pinned image (`--pull=never` at run); needs ≥200 GB free disk headroom for pull+decompress ([low-conc spike](cuda-sglang-low-concurrency.md) §disk). **GB10 unified-memory hazard:** `gpu_memory_utilization` reserves HOST RAM on the 119 GiB unified pool; 0.85 hard-rebooted the box (MEMORY: [[gb10-unified-memory-oom-reboots-box]]). Keep it LOW; never run a big vLLM oracle alongside it. Run the whole series under one `flock $GPU_LOCK`. `nvidia-smi` returns `N/A` for total/used on unified memory → use the process-tree PSS + system-available sampler (`tools/bench/sample_process_memory.py`). |
| Jetson Thor | sm_110, aarch64, CUDA 13 | LIKELY, unverified — the cu130 arm64 image would need sm_110 cubins or runtime JIT. | The sm_110 runtime path just landed for us (`CLAIM-CUDA-SM110-RUNTIME`); a SGLang-on-Thor arm is a stretch goal, not the primary gate. Verify the image carries/ JITs sm_110 before claiming. |
| Cloud H100 / B200 | sm_90a / sm_100a, x86_64 | YES — SGLang's best-supported path (amd64 image, the arch SGLang is tuned for). | Easiest to stand up, but NOT our gate hardware. Use only for arch-coverage cross-checks or when GB10 is contended; a cloud number never substitutes for the GB10 gate. |
| ROCm / Intel XPU | — | image variants exist; out of current scope | `BACKEND-GATE-ROCM-SGLANG` stays `INVENTORIED`. |

**Honest setup cost for the primary (GB10) path:** (1) pull the digest-pinned
arm64 image once (large; needs the disk headroom above); (2) stand SGLang up
under one GPU lock with a LOW memory-utilization to respect the unified pool;
(3) the correctness oracle additionally needs a shared model both engines load
byte-identically (a dense Qwen3 or a gate model); (4) the perf oracle reuses the
EXISTING `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` harness (`tools/bench/*serve_low*`
+ `scripts/dgx-sglang-low-concurrency.sh`) — **do NOT reinvent it**. The residual
blocker to the BINDING perf gate is OUR side, not SGLang's: `BACKEND-GATE-CUDA-SGLANG`
is `BLOCKED on SERVE-ASYNC-LLM` (our comparable async HTTP server) + a successful
exact-equivalence preflight, NOT on SGLang runnability.

---

## 3. The correctness gate (SGLang analogue of verification.md)

**Same ratified near-tie methodology as the vLLM oracle** (MEMORY:
[[near-tie-distributional-gate]]). SGLang is stood up as a greedy token capturer
on a shared model:

1. **Shared model, byte-identical weights.** Both engines load the SAME
   checkpoint (same safetensors snapshot hash). Where SGLang converts/repacks,
   record it; a converted arm cannot bind until weight/quant equivalence is
   proven (mirror the benchmark-protocol "equivalent" rule).
2. **Greedy token-exact where SGLang is deterministic.** Capture SGLang greedy
   (temperature 0) output token ids for a fixed prompt set; ours must be
   token-for-token identical AND identical to the vLLM oracle. This is the strict
   gate on models where BOTH competitor engines are bf16-deterministic.
3. **Near-tie band where SGLang's own greedy is bf16-nondeterministic.** On small
   dense models where SGLang's greedy is itself unstable at bf16 near-ties (same
   phenomenon we verified for vLLM), the gate is DISTRIBUTIONAL: ours ∈ the
   K-run set of SGLang greedy AND ∈ vLLM's K-run set, with a strict pass on a
   BIGGER deterministic model as the anchor. Teacher-forced logprob gap ≤ the
   ratified near-tie nat threshold, 0 forward-divergent.
4. **Divergence rule (the mirror caveat).** If SGLang and vLLM greedy DISAGREE on
   a token, that is NOT a bug in ours — ours mirrors vLLM (§ policy caveat). Log
   the divergence as an engine-difference datum; do not re-mirror to SGLang.
5. **Port the tests with the behavior.** Any SGLANG-DISTINCT behavior we adopt
   (e.g. LPM scheduling) carries its upstream SGLang test module re-expressed in
   our suite, named traceably, per the port-the-tests directive. A behavior that
   is output-neutral (scheduling/eviction/jump-forward) additionally gets a
   token-neutral A/B: greedy tokens must be byte-identical ON vs OFF.

**Correctness gate PASS =** greedy token-exact (or ratified near-tie) vs SGLang
AND vs vLLM on the shared model, with the DISTINCT-behavior A/B token-neutral.

---

## 4. The performance gate (SGLang analogue of verification.md)

**Every-axis, same workload, idle box, reproduced — identical rigor to the vLLM
perf gate.** SGLang is the binding floor wherever it is faster
([verification.md](../verification.md) "Additional competitor floor").

- **Axes (all of them, both gate models where they fit):** total + output
  throughput, req/s (higher-is-better, ours ≥ SGLang); TTFT, TPOT/ITL
  mean/median/P99, peak memory (lower-is-better, ours ≤ SGLang). Below SGLang on
  any axis where SGLang beats vLLM = an open gap, never "near parity".
- **Two independent cache gates** (cache is part of the workload, never an
  incidental default): a **cache-neutral / cache-off** workload
  (`BACKEND-GATE-CUDA-SGLANG`) and a **deterministic shared-prefix cache-ON**
  workload (`BACKEND-GATE-CUDA-SGLANG-PREFIX`). Each arm gets equal cache
  capacity, warmup, request order, and a MEASURED hit/reuse proof. For hybrid
  Qwen cache-ON, vLLM must be run with `mamba_cache_mode=align` — comparing
  SGLang's default radix cache against vLLM's default-off hybrid policy is a
  configuration comparison and cannot bind (verification.md §cache).
- **Equivalence preflight is a precondition (the existing P1/P2 work).** Before
  any number binds, prove same model/quant, same prompt tokens + token-ID
  counts, same sampling, same concurrency, same cache capacity/policy, same
  serving features across the SGLang and ours arms
  (`BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT`, P1 CPU contract cases green; P2
  image/model/GPU classification pending).
- **Reproduction is a gate.** Record commit, exact `bench_serving` command,
  model snapshot, in/out len, concurrency, num-blocks, seed, build, image
  digest + OCI revision; re-run ≥2-3× within run-noise; same-binary A/B for our
  own toggles; one `flock $GPU_LOCK` over the WHOLE series; thermal/power +
  memory-return disclosure per the multi-arm rule. A contended or non-reproducing
  run is void.
- **Denominator honesty.** SGLang runs in its production config (its overlap
  scheduler ON, radix cache ON) — the SGLang analogue of "graphed vLLM, never
  `--enforce-eager`". Never handicap the competitor to flatter ours.

**Perf gate PASS =** ours ≥ SGLang (throughput/req-s) AND ≤ SGLang (latency,
memory) on EVERY axis, BOTH cache gates, correctness holding, reproduced on an
idle GB10.

---

## 5. Relationship to the existing SGLang rows (no duplication)

| Existing row | Owner | Role | This program |
|---|---|---|---|
| `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | benchmark track | the perf-oracle harness (corpus, client, preflights) | REUSED as the perf-oracle stand-up; not modified here |
| `BACKEND-GATE-CUDA-SGLANG` | benchmark track | binding cache-neutral perf gate | the perf gate §4; `BLOCKED on SERVE-ASYNC-LLM` |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | benchmark track | binding shared-prefix cache-ON perf gate | the second cache gate §4 |
| `KV-SGLANG-RADIX-CACHE` | `CLAIM-SGLANG-RADIX-SCOPE` | RadixAttention == APC (FUSED; alias) | matrix row `SGLANG-KV-RADIX`; verdict carried, not re-derived |
| `ENG-SGLANG-BEHAVIOR-FLAG` | `CLAIM-SGLANG-RADIX-SCOPE` | LPM / overlap / jump-forward survey + flag | matrix rows `SGLANG-SCHED-LPM` etc.; verdict carried |
| `KV-MAMBA-ALIGN` | `CLAIM-PREFIX-PROMPT-CACHING` | hybrid-prefix retention (the cache-ON precondition) | precondition for `BACKEND-GATE-CUDA-SGLANG-PREFIX` |
| `BACKEND-GATE-ROCM-SGLANG` | benchmark track | ROCm floor | `INVENTORIED` |

This program's NEW artifacts are ONLY: this spec, [sglang-matrix.md](../sglang-matrix.md)
(the whole-surface inventory + classification), and the records notes. It creates
NO new claimable engine/kernel rows; the implementation rows already exist above
and are owned by their claims.

---

## 6. Ranked execution plan (value × unblocks ÷ size)

The value of this program is almost entirely the **precise map of what is
SGLANG-DISTINCT** — most of SGLang's surface is FUSED (both engines descend from
the same ideas; full table in [sglang-matrix.md](../sglang-matrix.md)). Ranked
by (value × what-it-unblocks ÷ size); "oracle-first" = needs the SGLang oracle
stood up on GB10 before it can be gated.

| Rank | Item | Matrix row | Size | Needs oracle up first? | Rationale |
|---:|---|---|---|---|---|
| 1 | **Stand up the SGLang perf oracle on GB10** (pull the arm64 image, wire it into the existing preflight harness, re-pin P1→v0.5.15, close P2 image/model/GPU classification) | `SGLANG-ORACLE-PERF` | M | — (this IS the stand-up) | Unblocks EVERY binding SGLang perf number; the single highest-leverage step. Gated by disk + the unified-memory caution, not by new code. |
| 2 | **`SERVE-ASYNC-LLM`** (our comparable async HTTP server) | (engine track) | L | no | The residual blocker on `BACKEND-GATE-CUDA-SGLANG`; without it the two arms are not driven equivalently. Already tracked; called out as THE unblock for the cache-neutral gate. |
| 3 | **Cache-aware LPM scheduling** `--schedule-policy=lpm` | `SGLANG-SCHED-LPM` | M | for the throughput A/B, yes | The ONE genuinely-distinct default-adjacent throughput lever (SGLang's own default is `fcfs`, so it is opt-in). CPU-testable behaviorally; the cache-ON throughput win needs the oracle. Owned by `ENG-SGLANG-BEHAVIOR-FLAG`. |
| 4 | **`--enable-radix-attention` alias + C-ABI `enable_prefix_caching` tri-state** | `SGLANG-ALIAS` | S | no | Trivial ergonomics over the already-shipped APC; reuses the existing APC gate. Highest value÷size but tiny absolute value (naming). |
| 5 | **Correctness cross-check oracle** (SGLang greedy capture on a shared dense model) | `SGLANG-ORACLE-CORRECT` | S–M | yes | A regression net + divergence detector; low urgency because vLLM already binds correctness. |
| 6 | **In-batch prefix-collision de-prioritization** (`IN_BATCH_PREFIX_CACHING_*`) | `SGLANG-SCHED-INBATCH` | S | yes | Follows LPM; small throughput lever on colliding prefixes. |
| 7 | **Radix eviction strategies** (lfu / slru / priority) `--radix-eviction-policy` | `SGLANG-KV-EVICT` | S | no | Opt-in knob over the block pool; minor. |
| 8 | **Jump-forward decoding** `--enable-jump-forward` | `SGLANG-CONSTRAIN-JUMP` | L | no (output-neutral) | Distinct constrained-decode speed lever; deferred (FSM-run precompute + cross-boundary retokenization is a substantial leaf, and neither mirror engine defaults it on). |
| 9+ | Everything else | see matrix | — | — | FUSED or OUT-OF-SCOPE (PD-disaggregation, HiCache == KV-OFFLOAD, EAGLE draft = a model port, custom kernels = the benchmark track). |

**Sequence:** land #1+#2 to make the perf gate bindable → drive #3 (the real
throughput lever) under the stood-up oracle → #4 (trivial) any time → #5-#8 as
prioritized. Items #3/#6 also need the `BACKEND-GATE-CUDA-SGLANG-PREFIX` cache-ON
A/B (owned by the benchmark track), which itself needs `KV-MAMBA-ALIGN`.

---

## 7. Dependencies

- Rows: `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` (harness), `BACKEND-GATE-CUDA-SGLANG`
  + `BACKEND-GATE-CUDA-SGLANG-PREFIX` (perf gates), `KV-SGLANG-RADIX-CACHE` +
  `ENG-SGLANG-BEHAVIOR-FLAG` (behavior verdicts), `KV-MAMBA-ALIGN` (cache-ON
  precondition), `SERVE-ASYNC-LLM` (our async server, the cache-neutral unblock).
- Hardware: dgx GB10 for every binding number; ≥200 GB disk; one `flock $GPU_LOCK`;
  low unified-memory utilization.
- No new library dependency. The image is the only new artifact and it is
  digest-pinned.

## 8. Risks and decisions

- **Do NOT re-mirror to SGLang.** vLLM is the behavior source; SGLang is a
  cross-check + a perf floor. A greedy divergence from SGLang is logged, not
  "fixed" against vLLM-derived behavior.
- **Unified-memory OOM reboots the box.** Keep SGLang's memory-utilization low;
  never co-locate a big vLLM oracle. A voided (contended/rebooted) run is not a
  gate.
- **`--pull=never`, digest-pinned.** A moved SGLang tag never silently repins
  evidence; record the manifest digest + OCI revision + platform digest.
- **The perf gate is our-side blocked, not SGLang-blocked.** The honest framing:
  SGLang runs on GB10 today via the image; what is missing is our comparable
  async server + the exact-equivalence preflight. This program does not conflate
  the two.

---

## 9. Results — first measured floor (`CLAIM-SGLANG-PERF-BENCH`, 2026-07-28)

**The oracle is STOOD UP and a first floor is MEASURED, reproduced on an idle
GB10.** Rank-1 `SGLANG-ORACLE-PERF` is no longer INVENTORIED: the arm64 cu130
image (`@sha256:d0a667e`) pulled and RAN the 27B-NVFP4 gate model on GB10
sm_121a with no from-source build — confirming §2's headline asymmetry. The
`run_serve_low.py` harness's `bench` subcommand drove `sglang.bench_serving`
against both engines identically; note the harness's `SGLANG_IMAGE` pin
(`serve_low_common.py:21`) is still v0.5.13 while this run used the oracle-spec
v0.5.15 pin — a deliberate re-pin to record, not silently adopt.

**Config + equivalence.** vllm.cpp `7e9ffbff` vs SGLang `v0.5.15-cu130` in
production config (overlap + radix cache on, CUDA graphs captured); model
`unsloth/Qwen3.6-27B-NVFP4` `890bdef7` byte-identical both arms; deterministic
corpus (seed 0, 80 × 1024-in/128-out exact, common-prefix ≤32 = cache-neutral),
greedy `ignore_eos`, KV 20480 tokens matched; strictly sequential under one
`flock`; 3 reps + warmup, all within run-noise; both emitted exactly 80×128
tokens, 0 errors, identical peak concurrency. Full per-axis table +
repro recipe: [sglang-matrix.md](../sglang-matrix.md) § "Perf oracle results",
[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md).

**Verdict against the §4 perf gate.** The gate is "ours ≥ SGLang on
throughput/req-s AND ≤ SGLang on latency/memory, EVERY axis, BOTH cache gates."
On the cache-neutral arm at c8/c16: ours PASSES throughput (2.21×/1.44×), req/s,
TTFT (6–12× lower), and ~ties peak memory — but FAILS the per-token
latency axis (TPOT/ITL 1.18–1.49× above SGLang). Per the gate's own rule
("below SGLang on any axis where SGLang beats vLLM = an open gap, never near
parity"), the perf gate is **NOT fully met**: it is a decisive throughput+TTFT
win with a reproduced, honestly-recorded TPOT/ITL gap (candidate lever: our
higher throughput comes from larger decode batches that raise per-request
per-token latency). The second (shared-prefix cache-ON) gate,
`BACKEND-GATE-CUDA-SGLANG-PREFIX`, remains unrun — as does 35B, the c1/c2/c4
low-concurrency sweep (SGLang c1 ~13.3 s/it, impractical for 3-rep reproduction
this pass), and the `SGLANG-ORACLE-CORRECT` token-exact cross-check.
