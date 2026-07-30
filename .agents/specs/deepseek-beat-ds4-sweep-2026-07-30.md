# DeepSeek-V4-Flash — the BEAT-ds4 sweep (get PAST 16.5-17.2 tok/s, not just match)

Status: **SWEEP / PLAN — analysis-only, no code changed.** Base main `5c3da2f1`.
`CLAIM-DSV4-BEAT-SWEEP`. Branch `deepseek-beat-ds4-sweep` off `5c3da2f1`, NOT pushed.

## 0. Headline verdict (brutally honest)

- **Our raw autoregressive decode is at its GB10 ceiling: 13.19 tok/s, bit-exact, and the
  Q8_0-GEMV front is CLOSED** (Brick 13, `ds4-q8-ncu-2026-07-30`). Six structural axes were
  built and refuted (Bricks 3 dp4a, 4 aligned-repack, 8 quant-fusion, 11 sub-warp, 12 launch-
  consolidation, 13 ILP). `ncu` proved our Q8_0 kernel **memory-LATENCY-bound at 71.9% occupancy**
  (long-scoreboard 54.4, L1-hit 96.7%). ds4 moves the **byte-identical** 6.15 GiB Q8_0 tower at
  **90% of the roofline vs our 67%**, and that 90% is attributable ONLY to a mechanism our
  counters can't observe (L2 residency / access pattern; `ERR_NVGPUCTRPERM` HW-counter-blocked
  for the ds4 side; Brick 12 reproduced ds4's EXACT 259-launch structure bit-for-bit and stayed
  at 69%). **So at raw-decode-vs-raw-decode we are ~13.2 and cannot close to ds4's ~16.5 on the
  Q8_0 kernel — that axis is exhausted.**

- **ds4's 16.5-17.2 IS RAW AUTOREGRESSIVE DECODE, spec OFF** (verified §1). Its own MTP/DSpark
  speculative path is opt-in, needs a separate support GGUF, and is NOT in the 16.5 measurement.

- **Therefore beating ds4's 16.5-17.2 is reachable via EXACTLY ONE path: our MTP self-spec
  decode.** It is a throughput MULTIPLIER on our raw 13.19 that ds4's DEFAULT/published number
  does not use. **Break-even acceptance to pass 16.5 is only p≈0.31** (§2); DeepSeek MTP typically
  delivers p≈0.6-0.85, landing **~18-23 tok/s** — comfortably past 16.5. **Everything else on the
  board is sub-16.5 and cannot get us past ds4 alone.**

- **The honesty razor:** "beat ds4" is unambiguous for ds4's DEFAULT 16.5-17.2. If the user later
  enables ds4's **DSpark** (its purpose-built 2-stage draft — ds4's README calls its own legacy
  ONE-stage MTP, which is exactly OUR architecture, "at most a slight speedup"), ds4 leaps ahead
  again and matching it becomes a draft-quality race (R5, future — a DSpark-class draft, not the
  single nextn layer). See §2 caveat.

---

## 1. Is ds4's 16.5-17.2 WITH or WITHOUT spec? — DECISIVE: WITHOUT (raw decode)

Grounded on the ds4 oracle (`~/w8run/ds4/` on `dgx.casa`, source `ds4_cuda.cu`, `README.md`):

1. **The profiled 16.51 tok/s run used NO spec flags.** Reproduce command
   (`ds4-q8-kernel-profile-2026-07-30.md` §5): `./ds4 -m ds4flash.gguf --cuda --gpu-vram 90
   -p "..." -n 60 --temp 0` — **no `--mtp`, no `--dspark`, no `--glm-mtp`.**
2. **ds4's README is explicit:** "Ordinary decode remains the default." Speculation "must be
   enabled explicitly with `--mtp`" / `--dspark` and needs the separate `DeepSeek-V4-Flash-
   DSpark-support.gguf`. The headline "~16.8 t/s (15.4 at 4k)" table is the **two-M5-Mac tensor-
   parallel** figure; the **GB10** apples-to-apples number is the profile's **16.51 wall / 17.6
   GPU-active**, single-token autoregressive.
3. **The per-step kernel structure proves single-token decode:** ds4 reads the ENTIRE 6.15 GiB
   Q8_0 tower exactly once per step in 259 launches (byte-impossibility proof: >6.6 GB in 30 ms
   would exceed 240 GB/s). A 15-token DFlash/DSpark verify block would show a multi-token verify
   forward and draft-model kernels — it does not.
4. **ds4's own verdict on its ONE-stage MTP** (== our `nextn_predict_layers=1` head): "provides
   **at most a slight speedup, not a meaningful generation-speed win**" / "experimental slight-
   speedup path." Its REAL speculator is **DSpark**, a 2-stage auxiliary draft model that
   "advances the stream by several tokens" (`--mtp DSpark-support.gguf --dspark`).

**Conclusion:** the 16.5-17.2 target is ds4's RAW decode. Our raw 13.19 is behind on the Q8_0
kernel (closed front). The multiplier that gets us past it is spec-decode, which the target does
not include.

---

## 2. MTP self-spec assessment — the #1 (and only) path to EXCEED ds4

### State of the MTP wiring (task #196, `deepseek-v4-mtp.md`)
- **W1 LANDED (host):** loader `DeepseekV4MtpHostWeights` + `DeepseekV4GgufHasMtp` absence guard;
  host draft forward `DeepseekV4MtpDraftLogitsHost` (1:1 nvidia/mtp.py:128-258, reuses DS4
  attn/MoE/MHC); lossless self-spec gate via the SHARED `RejectionSampler` — **MTP-on greedy is
  token-IDENTICAL to MTP-off by construction** (target verifies every draft token). Gate
  `test_deepseek_v4_mtp` 5/5, RED-first miswires.
- **BLOCKER R1 (weight):** BOTH shipped GGUFs advertise `nextn_predict_layers=1` but the llama.cpp
  `deepseek4` converter **dropped every `blk.43`/nextn tensor** (1328 tensors, blocks 0-42 only).
  `DeepseekV4GgufHasMtp` returns false → clean fall-back to MTP-off. The NVFP4 safetensors DOES
  carry `mtp.*` but is 156.7 GiB (needs multi-Spark). **No shipped file can run MTP today.**
- **RESIDUALS:** R2 = DS4-native propose/verify decode loop over `ForwardResidentDecodeGguf` /
  `V4Graph::Step` (stash the target's pre-hc_head MHC residual per step, draft k=1, verify next
  step). R3 = engine speculator registration. R4 = DEVICE MTP draft forward (reuse DS4 device
  kernels) for decode-graph speed (W1 is the host oracle).

### The acceptance-rate → tok/s math (honest)
Draft depth k=1 (single nextn layer). Effective tokens per verified step = **1 + p** (p = accept
prob of the drafted token). **Per-step overhead on GB10 is small and favorable:**
- The target verify at M=2 (draft position + real position) is **nearly free** on our keep-quant
  **device-resident** decode: the Q8_0/MoE GEMMs are **weight-read-bound**, and M=1 vs M=2 reads
  the same 8.4 GB of weights ONCE. (This is the structural reason the Mac "slight speedup" does
  NOT carry over — on Metal the verify is not free; on GB10 it is.)
- The draft head is ONE extra decoder layer ≈ **+2-3%** of the step; + rejection sampling (tiny).
- Model per-step cost ≈ **1.05× raw**.

**Effective tok/s ≈ 13.19 × (1 + p) / 1.05:**

| accept p | regime | effective tok/s | vs ds4 16.5 |
|---|---|---|---|
| 0.31 | **break-even** | **16.5** | ties |
| 0.40 | ds4's pessimistic "slight" (Mac) | 17.6 | **+7%** |
| 0.55 | quant-degraded IQ2 mid | 19.5 | **+18%** |
| 0.70 | typical MTP | 21.4 | **+30%** |
| 0.85 | DeepSeek-paper regime | 23.2 | **+41%** |

**Break-even is only p > ~0.31.** DeepSeek's own MTP module reports ~85-90% 2nd-token acceptance
(~1.8× TPS) on the full model; even with heavy IQ2_XXS/Q2_K quant degrading the draft, clearing
31% is very likely. **So MTP self-spec beats ds4's raw 16.5-17.2 across the entire plausible
acceptance band.**

### Honest caveats (do not oversell)
1. **Acceptance is UNMEASURED** — weight-blocked (R1). The band above is inference from DeepSeek's
   published MTP + the favorable GB10 verify economics, NOT a measurement. The real number needs
   R1+R2 to land and profile on the 80.7 GB model.
2. **IQ2_XXS is aggressive** — the 2-bit quant may pull acceptance toward the low end. Even so,
   p=0.40 already beats 16.5.
3. **ds4-DSpark-on is a different bar.** ds4's DEFAULT is raw (what we beat). ds4's DSpark (2-stage
   draft, "several tokens/verify", opt-in) would out-accept our single nextn layer. Matching a
   *spec-on ds4* would need a DSpark-class draft (R5), not the shipped MTP head.

---

## 3. The Q8_0 90%-vs-67% question, the fp16 cache, and the ncu (items 2+3)

### 3a. fp16-dequant weight cache (`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB`) — NET-SLOWER on GB10, NOT a lever
- In the profiled 16.5 run the cache was **budget-EXHAUSTED** (`"q8 fp16 cache budget exhausted"`)
  → ds4 read the **same raw 34-B q8_0 blocks we do**. So ds4's 90% is NOT the fp16 cache.
- **Arithmetic:** fp16 = 2.0 B/elem vs q8_0 = 1.0625 B/elem = **1.88× the bytes**. Decode is
  memory-bound. A fully-fp16 tower reads 6.15 GiB × 1.88 ≈ 11.6 GB/step; even at ds4's 217 GB/s
  that is **~53 ms vs 30 ms** for raw q8. **fp16-cache → cuBLAS fp16 GEMV is strictly SLOWER for
  M=1 decode on GB10** (more bytes on a BW-bound step). ds4 exhausts/disables it for exactly this
  reason; it targets small 8-12 GB cards, not the GB10 regime. **Verdict: not a beat-lever.**

### 3b. Why ds4 is 90% and we are 67% — unresolved, counter-blocked, front CLOSED
- OUR side is fully measured (Brick 13 sudo-ncu): latency-bound, occ 71.9%, L1-hit 96.7%,
  long-scoreboard 54.4. **ds4's ncu side was NEVER captured** (the agent stalled in prefill).
- Even IF ds4 shows low long-scoreboard / higher ILP, it yields **no new portable lever**: we
  tried ILP (Brick 13, occupancy collapses), launch-consolidation (Brick 12, reproduced ds4's
  259 structure → still 69%), aligned repack (Brick 4), sub-warp (Brick 11), fusion (Brick 8).
  All flat/negative. The residual is an unobservable L2/access-pattern effect on GB10's LPDDR5X.

### 3c. Should we run the ds4 ncu now? — NO (scoped, not run). Reasoning + exact command.
**Decision: DO NOT run it this session.** (a) The Q8_0 front is CLOSED — a confirmed ds4
long-scoreboard number is confirmatory, not lever-opening. (b) `ncu` replay of a decode kernel
needs the 80 GB model resident PLUS replay buffers in the **119 GB UNIFIED** pool — reboot-risky
(`gb10-unified-memory-oom-reboots-box`), and ncu graph-replay is slow, not "quick." (c) The
beat-path is MTP, not Q8_0. Running it cannot change the ranking. **Exact command for the record
(needs `--launch-skip` tuned to land in a steady decode step, the trap that stalled the prior
agent):**
```
docker stop local-ai-worker 2>/dev/null           # free the GPU (restore after)
cd ~/w8run/ds4
flock $HOME/gpu.lock sudo env DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=28000 \
  /usr/local/cuda-13.0/bin/ncu \
    --graph-profiling node \
    --kernel-name-base function \
    --kernel-name 'regex:matmul_q8_0_(preq|pair_preq|hc_expand_preq)_warp8_kernel' \
    --launch-skip 20000 --launch-count 8 \
    --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct \
    --csv -o ~/w8run/ncu_out/ds4_decode \
    ./ds4 -m ds4flash.gguf --cuda --gpu-vram 90 -p "The capital of France is" -n 30 --temp 0
docker start local-ai-worker 2>/dev/null           # restore
```
(`--launch-skip` must clear the 43-layer prefill graph; sweep 15k-25k until the CSV shows a decode
Q8_0 kernel at grid 256-512. Interpretation: LS ≪ 54.4 or L2-hit ≫ 10.7% on ds4 would confirm the
access-pattern hypothesis — but does not re-open a portable lever given Bricks 4/8/11/12/13.)

---

## 4. Non-Q8_0 residual (item 4) — all measured, none exceeds 16.5

From the Lane-A group diff (`deepseek-ds4-gap-lever-plan-2026-07-30.md`, measured, GB10 30-step):
- **Routed-expert MoE (grouped IQ2_XXS + Q2_K GEMMs): WE WIN by 2.6 ms** (Brick 6 gate+up+SwiGLU
  fusion). Grouped kernels are memory-bound (IQ2 46%, Q2_K 56% of roofline after Bricks 1/1b).
  **KEEP as-is, no headroom to chase.**
- **MHC / `MhcPreFinishKernel` (~7.6% glue): MEASURED TIE** (ours 9.72 vs ds4 9.82 ms — we're
  fractionally ahead). Lane C's "MHC = 7.7% headroom" is REFUTED by Lane A. **DROP as a lever.**
- **Residual glue** (`QuantizeQ8K` 7.6%, `RouteKernel` already warp-parallel −5.59 ms Brick 10,
  `NormRopeRows` fused Brick 7): the only un-fused reduction left is the MhcPre near-tie; a fold
  is **~0.2-0.5 tok/s** at most and near-tie-risky. Not a path past 16.5.

## 5. fp8 KV + prefill (item 5)
- **fp8 KV**: ~1.3% of the short-context step (attention `DecodeAttnGKernel`); it is a
  footprint / long-context (256+) / ds4-apples-to-apples parity lever, **0 tok/s at short ctx**.
  Quantify only when a long-context benchmark is stood up. NOT a beat-lever.
- **Prefill**: the ds4 comparison is DECODE. **No GB10 prefill A/B is measured** — a genuine
  measurement gap. ds4's README prefill (94 t/s) is the 2-Mac number, not GB10. If a beat claim
  ever needs prefill, measure it first; today it is out of scope for the tok/s (decode) target.

## 6. Anything ds4 does that we DON'T (item 6, from `ds4_cuda.cu`)
- **Q8_0 projection pairing / hc-expand / grouped-a launch structure** — PORTED (Brick 12,
  bit-exact, default-ON) → measured flat (the graph already amortized launches). Done.
- **DSpark 2-stage speculative draft** — the real structural thing ds4 has that we don't at
  equivalent quality. Our equivalent is the single nextn MTP head (weaker). A DSpark-class draft
  is R5 (future) if a spec-on-vs-spec-on race is ever required.
- **fp16 weight cache** — refuted net-slower on GB10 (§3a).
- **TP mirror / stream overlap** — 98.5% GPU-bound at T=1 leaves no slack (Lane A); not a lever.

---

## 7. RANKED BEAT-ds4 lever table (by path-to-EXCEED 16.5-17.2)

| # | Lever | Path to EXCEED 16.5 | Grounded gain (tok/s) | Risk | Effort | Dependency |
|---|---|---|---|---|---|---|
| **1** | **MTP self-spec decode** (single nextn head, k=1, lossless) | **THE path** — multiplier `13.19×(1+p)/1.05`; break-even p>0.31; typical p 0.55-0.85 | **19.5-23** (p 0.55-0.85); 17.6 at p=0.40; **>16.5 for any p>0.31** | med (accept unmeasured; lossless by construction so ZERO correctness risk) | **high** — R1 GGUF + R2 loop + R4 device draft | **R1: a nextn-carrying GGUF (BLOCKED)** |
| 2 | Q8_0 90% capture (ds4's unobservable L2/access edge) | would lift raw 13.19→~14 (still <16.5) | +0.9 max, **cannot exceed 16.5 alone** | high — 6 axes refuted, counter-blocked | high, low-probability | sudo ncu + a NEW mechanism (none found) |
| 3 | Residual MhcPre glue fold (near-tie) | raw +0.2-0.5 (still <16.5) | +0.3 | near-tie | low-med | none |
| — | Routed-MoE grouped GEMM | **WE WIN +2.6 ms — KEEP** | — | — | — | — |
| — | MHC mix→GEMM | **MEASURED TIE — DROP** | — | — | — | — |
| — | fp16 dequant cache | **net-SLOWER on GB10 (§3a) — REJECT** | — | — | — | — |
| — | fp8 KV / prefill | long-ctx / footprint / unmeasured; **0 at short-ctx decode** | — | — | — | DEFER |

**TOP RECOMMENDATION:** unblock and land **MTP self-spec decode (lever 1)** — it is the ONLY
lever that gets us past 16.5-17.2, it is **lossless** (token-identical to greedy, zero correctness
risk), and the GB10 verify economics make the effective multiplier favorable. Execution order:
1. **R1 — the gating dependency:** produce a `deepseek4` GGUF that RETAINS `blk.43.*`/nextn (fix
   the llama.cpp converter's nextn drop), OR run the NVFP4 safetensors on multi-Spark. Nothing else
   in the campaign matters until a weight-carrying file exists.
2. **R2 — DS4-native propose/verify loop** over `ForwardResidentDecodeGguf` / `V4Graph::Step`
   (stash pre-hc_head MHC residual → draft k=1 via `DeepseekV4MtpDraftLogitsHost` → verify via the
   shared `RejectionSampler`). **GATE:** MTP-on == MTP-off token-identical on the 80.7 GB model +
   MEASURE acceptance p and effective tok/s. This is the go/no-go for the beat claim.
3. **R4 — device MTP draft forward** (reuse DS4 device kernels) so the draft rides the decode graph
   — needed to realize the full multiplier at speed rather than paying a host draft each step.

**Honest bottom line:** 13.19 IS the raw-decode ceiling (Q8_0 front closed; ds4's raw-kernel edge
is unreachable/unobservable). **Beating ds4's 16.5-17.2 REQUIRES spec-decode — and MTP self-spec
gets us there** (~18-23 tok/s at plausible acceptance, break-even p≈0.31), **conditioned on
unblocking the nextn-carrying GGUF and wiring the propose/verify loop.** The residual glue/Q8_0
levers cannot, alone or together, reach 16.5.

## 8. Grounding
- Our decode: `src/vt/cuda/cuda_deepseek_v4.cu`, `src/vt/cuda/cuda_quant_dot.cu`,
  `src/vllm/model_executor/models/deepseek_v4*.cpp`. MTP: `DeepseekV4MtpDraftLogitsHost`
  (`deepseek_v4.cpp:1985`), `DeepseekV4MtpHostWeights`/`DeepseekV4GgufHasMtp`.
- ds4 oracle (`dgx.casa:~/w8run/ds4/`): `ds4_cuda.cu` (spec `g_glm_mtp_verify_mode`,
  `matmul_q8_0_{pair,hc_expand,grouped_a}_preq_warp8`, DSpark draft block :13462), `README.md`
  (decode 16.8/GB10 16.51 raw; `--mtp`/`--dspark` opt-in; one-stage MTP "slight speedup").
- Prior campaign records: `deepseek-v4-last-mile.md` (Bricks 0-13), `ds4-q8-kernel-profile-2026-07-30.md`
  (ds4 90% vs our 67%), `ds4-q8-ncu-2026-07-30.md` (our kernel latency-bound), `deepseek-v4-mtp.md`
  (MTP W1 + weight blocker), `deepseek-ds4-gap-lever-plan-2026-07-30.md` (Lane-A group diff).
