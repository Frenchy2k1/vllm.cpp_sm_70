# Laguna-S-2.1 NVFP4 forward arm — bring-up spec (task #230)

**Goal.** Give our engine an NVFP4 Laguna forward so we can compare ours-NVFP4
vs vLLM-NVFP4 on the SAME quant (the true apples-to-apple). Today Laguna runs
only the GGUF Q4_K keep-quant path (7.7 tok/s); vLLM-NVFP4 = ~18.8 tok/s
(MARLIN, lower bound — `CLAIM-LAGUNA-VLLM-NVFP4`). The gap should shrink on the
shared tensor-core regime we already reach vLLM parity on for 27B/35B.

**This is NOT a from-scratch forward.** It reuses the landed qwen3_5 (35B)
NVFP4 W4A4 MoE infrastructure + the existing Laguna host ops. Estimated ~85%
reuse; the genuinely-new work is a name-map + a second weight-arm + the forward
quant-branch selection.

## Checkpoint (verified 2026-07-31, `poolside/Laguna-S-2.1-NVFP4`, ~67 GiB)
Recipe = **bf16 attention + dense MLP, W4A4-NVFP4 routed experts**:
- Standard HF names: `model.embed_tokens.weight`, `lm_head.weight`,
  `model.layers.N.*`.
- Attention (all layers) BF16 `.weight` only: `self_attn.{q,k,v,o,g}_proj.weight`
  (`g_proj` = Laguna softplus-gate proj), `self_attn.{q,k}_norm.weight`
  (QK-RMSNorm), `self_attn.{k,v}_scale` (fp8 KV-cache scales — our KV is f32,
  so ignore or wire to fp8-KV later).
- Layer 0 dense MLP BF16 `.weight`: `mlp.{gate,up,down}_proj.weight`.
- Layers 1..47 MoE experts NVFP4 **W4A4**: `mlp.experts.E.{gate,up,down}_proj.{`
  `weight_packed` (U8 [N,K/2]), `weight_scale` (F8 [N,K/16]),
  `weight_global_scale` (f32 scalar), `input_global_scale` (f32 scalar, the
  activation-quant scale ⇒ activations are ALSO fp4)`}`. Plus router
  `mlp.gate.weight` + a shared expert (`mlp.shared_expert*`, verify name).
- **Name-map delta vs qwen3_5:** qwen3_5 `LoadNvfp4Raw` reads `.weight_scale`
  + `.weight_scale_2`; here the global scale is `.weight_global_scale` and
  there is an extra `.input_global_scale` (W4A4). Adapt the resolver or add a
  `LoadNvfp4RawW4A4` sibling.

## Reuse map (existing, landed)
- `Nvfp4Weight` POD + `LoadNvfp4Raw` (`qwen3_5_weights.cpp:259`) — the packed/
  scale/scale2 loader; free function over a `TensorResolver`, directly reusable.
- Ops: `vt::MatmulNvfp4` (W4A16), `vt::MatmulNvfp4Fp4` / `MatmulNvfp4Cutlass`
  (W4A4 fp4×fp4), `kMoeGroupedGemmNvfp4` / `...Marlin` (grouped routed experts)
  — all in `include/vt/ops.h`, used by the 35B path.
- Laguna host ops UNCHANGED (`laguna_ops.cpp`): `LagunaSoftplusHeadGate`,
  `LagunaUngroupedRouterTopK`, dual RoPE cos/sin builders, QK-RMSNorm. Only the
  ~9 GEMM sites in `laguna.cpp` change quant arm; the glue is identical.
- Laguna KV cache + incremental decode (`LagunaKvCache`, W6) UNCHANGED.

## W-plan (bricks; each CPU-buildable, DGX-gated where noted)
- **N1 — weight struct + loader.** Add an NVFP4 arm to `MoeBlockWeights` /
  Laguna weights (bf16 attn/dense + `Nvfp4Weight` experts, mirroring
  qwen3_5's `expert_{gate,up,down}_fp4` + `shared_*_fp4`). `LoadLagunaNvfp4`
  over the HF name-map with the W4A4 global-scale delta. CPU build-verify +
  a loader unit test (shapes + a byte-identity check vs the raw file, RED-first).
  - **N1a DONE (98a67a01):** additive `Nvfp4Weight` expert fields on
    `LagunaMoeWeights` (CPU-built, `test_laguna_scaffold` 8/8·167).
  - **N1b DONE (0d014016 loader, 218655f2 test).** `LoadLagunaForCausalLMWeights`
    body landed: resolver + per-layer bf16 attn/dense/norms/embed/lm_head +
    BF16 router + F32 `e_score_correction_bias` + `LnLoadCtNvfp4Raw` W4A4 experts
    (`alpha = scale2 / input_global_scale_inv`) + BF16 shared expert; sets
    `has_nvfp4_weights`. Gated by `test_laguna_nvfp4_loader` (synthetic 2-layer
    NVFP4 checkpoint round-trips byte-identically + missing-tensor RED). The
    concrete recipe below was the derivation:
  - **N1b recipe (derived 2026-07-31).** Replace the
    `VT_CHECK(false)` in `LoadLagunaForCausalLMWeights(const
    std::vector<SafetensorsFile>& shards, const HfConfig&)`
    (`laguna_weights.cpp:240`). Steps, all reusing qwen3_5_weights.cpp helpers
    (same TU family — extract them to a shared header or duplicate the ~5-line
    resolver):
    1. **Resolver** (copy `qwen3_5_weights.cpp:423-434`): build
       `where = unordered_map<string,const SafetensorsFile*>` from each
       `shard.Names()`, then a `TensorResolver get` lambda capturing `where`
       that maps a tensor name to `it->second->Get(name)`.
    2. **Top-level** (HF names, all BF16): `LoadBf16Direct(get,
       "model.embed_tokens.weight")`, `..."model.norm.weight"`, `lm_head` =
       `LoadBf16Direct(get,"lm_head.weight")` (NOT fp4 — the checkpoint keeps it
       bf16; verified in the index).
    3. **Per layer l in 0..num_hidden_layers-1** (`model.layers.l.`):
       - norms BF16: `input_layernorm.weight`, `post_attention_layernorm.weight`.
       - attn BF16: `self_attn.{q,k,v,o,g}_proj.weight` +
         `self_attn.{q,k}_norm.weight` (per-head [head_dim]). Q width is
         per-layer variable (`LagunaParams::QHeadsForLayer`) — size from the
         tensor shape, do NOT assume uniform.
       - **layer 0 (dense, `mlp_only_layers=={0}`):** `mlp.{gate,up,down}_proj.weight` BF16.
       - **layers 1..47 (MoE):** router `mlp.gate.weight` BF16->F32; the
         `e_score_correction_bias` (verify HF name, likely
         `mlp.gate.e_score_correction_bias` or absent — if absent, leave Empty,
         the sigmoid-noaux router tolerates a zero bias); per-expert e in
         0..num_experts-1: `experts_{gate,up,down}_fp4.push_back(LoadNvfp4RawW4A4(
         get, "mlp.experts.e.{gate,up,down}_proj"))`; shared expert
         `mlp.shared_expert{,s}.{gate,up,down}_proj` -> `shared_*_fp4` (verify the
         exact shared-expert HF stem from the index).
    4. **`LoadNvfp4RawW4A4`** = `LoadNvfp4Raw` (`:259`) with the name delta:
       reads `.weight_packed`, `.weight_scale` (F8), and — instead of
       `.weight_scale_2` — `.weight_global_scale` (f32) into `scale2` +
       `weight_global_scale_inv`, plus `.input_global_scale` (f32) into
       `input_global_scale_inv`, and `alpha = scale2 / input_global_scale_inv`
       (the W4A4 recipe already documented on `Nvfp4Weight`, qwen3_5_weights.h:120-132).
    5. **Dual RoPE caches** (Laguna-specific, already implemented): build the
       full-attn YaRN-64 + sliding-128 cos/sin via the W3 `BuildLaguna{FullYarn,
       Sliding}CosSin` (`laguna_ops.cpp`) into `LagunaWeights`, same as the GGUF
       path does.
    - **Gate:** truly validating N1b needs the real 67 GiB checkpoint (DGX) OR a
      synthetic 2-layer safetensors fixture (a `test_laguna_nvfp4_loader` that
      writes a tiny fp4 tensor + checks `experts_gate_fp4[0].{packed,scale,scale2,
      n,k}` round-trip byte-identical, RED-first on a wrong-scale-name mutant).
      The fixture is the CPU-gateable path; recommend it over deferring to DGX.
- **N2 DONE (this commit).** Forward quant-branch landed in `laguna.cpp` +
  CPU-gated. **CORRECTION vs the original recipe:** the routed experts are TRUE
  **W4A4** (each expert carries `input_global_scale` ⇒ `alpha>0`), so they use
  the **per-expert `MatmulNvfp4Fp4`** (fp4 activation quant via
  `ScaledFp4Quant(w.input_global_scale_inv)` → `MatmulNvfp4Fp4(..., w.alpha)`),
  NOT the grouped `MoeGroupedGemmNvfp4` op — that op is **W4A16** (bf16
  activation), wrong numerics for this checkpoint. The grouped W4A4 fast-path is
  deferred to N5 (a speed fold, not correctness). Concretely:
  - New helper `LqGemmNvfp4Fp4` (after `LqGemm`): mirrors `LqGemm`'s
    unified-memory pattern (host ptrs as device tensors on GB10, raw weight view
    retagged), does fp4-activation-quant then the W4A4 GEMM.
  - `LagunaFfnBlock` branches on `fp4 = !experts_gate_fp4.empty()`: routed-expert
    gate/up/down call `LqGemmNvfp4Fp4` per expert; the keep-quant grouped
    fast-path is gated off (`!fp4`); an `IsTrueW4A4()` assert guards the arm.
  - **Everything else flows through the SAME bf16 path unchanged:** attention,
    dense L0 MLP, router (BF16→F32), shared expert (BF16 in this checkpoint), and
    lm_head are all bf16 and route through `LqGemm`/`ReadF32` — softplus gate,
    dual RoPE, sigmoid router, sliding-window mask all identical to the GGUF path.
  - The two `LagunaForwardGguf{,Cached}` guards relaxed to
    `has_gguf_weights || has_nvfp4_weights`.
  - **CPU run-gate** (`test_laguna_nvfp4_loader`, +1 case): the fp4 MoE branch
    executes through the real `LagunaForwardGguf` on a controlled-finite synthetic
    checkpoint → finite + deterministic logits + routed-experts-consumed proof.
- **N3 — engine wiring + run entrypoint.** Register the NVFP4 Laguna in the
  loader dispatch (safetensors dir → NVFP4 arm; GGUF → keep-quant arm);
  `examples/laguna_gen` accepts the NVFP4 dir.
- **N4 — CORRECTNESS gate (DGX).** Greedy on the NVFP4 dir must be coherent and
  match a vLLM-NVFP4 golden within a ratified near-tie band (vLLM MARLIN is not
  bit-deterministic vs our kernel; use the distributional gate methodology).
  Capture the golden from the same `run_vllm4.py` MARLIN run (extend to dump
  logits/ids).
- **N5 — SPEED gate (DGX).** ours-NVFP4 tok/s vs vLLM-NVFP4 18.8 (MARLIN) on the
  identical checkpoint, same prompt, GB10 safe-memory recipe. Record vs the bar.

## Gates / risks
- **W4A4 activation quant.** Experts carry `input_global_scale` ⇒ activations
  quantize to fp4 (`MatmulNvfp4Fp4`), not the W4A16 path. Confirm which the 35B
  routed-expert grouped op uses and match it (35B APEX-Compact is W4A4 — likely
  the same op, so this is reuse not new).
- **DGX memory.** 67 GiB weights on the 119 GiB pool — use the saved GB10 recipe
  (drop_caches, low util, single engine). Our engine's footprint differs from
  vLLM's; profile before assuming fit.
- **Near-tie gate, not strict.** vLLM MARLIN NVFP4 is not bit-identical to our
  CUTLASS/fp4 kernels; correctness is a distributional/near-tie gate (ratified
  methodology), plus coherence + the GGUF path stays token-exact to ITS golden.
- **Alternative considered (1b):** optimize the EXISTING GGUF path (W11 levers:
  QuantizeQ8K-dedup via `MoeGateUpSwiGLUGrouped` + GEMV BW-tune) → 7.7 → ~13-20
  tok/s, gateable cheaply in the GGUF regime. Cheaper + lower-risk than this arm
  but a different quant than vLLM's NVFP4; not mutually exclusive.

## Status
**N1a + N1b + N2 + N3 LANDED + CPU-gated (2026-07-31).** The NVFP4 W4A4 arm is
runnable end-to-end on CPU: weight struct → loader → forward branch → greedy
driver. `test_laguna_nvfp4_loader` (3 cases, 61 assertions) covers byte-identity
load, missing-tensor RED, and a finite+deterministic forward through the fp4 MoE
branch; the GGUF keep-quant path stays byte-identical (`test_laguna_scaffold`
8/8). **N3 (driver):** `examples/laguna_gen` auto-detects a safetensors DIRECTORY
(→ NVFP4: `LoadHfConfig(config.json)` + `LoadLagunaForCausalLMWeights` +
`LagunaForwardGguf{,Cached}`) vs a `.gguf` FILE (→ keep-quant), sharing the greedy
loop; injected `--token-ids` bypass the tokenizer for the id-vs-golden gate.
CPU-smoke-verified on a synthetic NVFP4 dir with a REAL config.json (exercises the
`LoadHfConfig`→`ParseLagunaParams` seam the loader test bypassed) → `has_nvfp4=1`,
KV-cache decode runs finite. **N4 RAN on GB10 (2026-08-01) — coherent, near-tie; the arm works end-to-end.**
git-archived `84fab587` → clean CUDA build (`121a`, `-DVLLM_CPP_CUTLASS_DIR`) →
`laguna-gen --model ~/laguna-nvfp4/ckpt --gpu --token-ids 2,785,9626,377,15360,395`
(vLLM's exact prompt ids, captured cheaply via the HF tokenizer — no model load).
Two GB10 memory fixes were needed to run (both landed): release the mmap'd shards
after the loader's memcpy-copy (114→67 GiB RSS), and create the CUDA context BEFORE
the load (else the 67 GiB reclaimable page cache starves `cudaStreamCreate`).
- **Result:** ours `22345 83 350 71070 395 340 9626 372 1703 29339 5705 377 15360
  81 466 330 7097 377 290 87 81 86 91 86` vs golden `22345 83 290 350 674 330 5541
  966 340 9626 377 15360 81 2498 …`. **First 2 tokens match vLLM exactly**, then
  near-tie divergence — coherent (9626/377/15360 = "France is"; shares the golden's
  340/330/290/81 vocabulary). EXPECTED: our path is TRUE **W4A4** (fp4 activations)
  vs the MARLIN golden's **W4A16** (bf16 activations) — genuinely different
  precision, so token divergence is not a bug. Forward is CORRECT (coherence + near-tie).
- **N5 SPEED = 6.34 s/tok (0.16 tok/s), prefill 17.3s — ~120× slower than vLLM's
  18.8 tok/s.** ROOT CAUSE (confirmed in source): `LqGemmNvfp4Fp4` calls the generic
  `vt::MatmulNvfp4Fp4` op, which on CUDA is the hand-written EMULATION kernel
  (`MatmulNvfp4Fp4KernelCuda` → Naive/Wmma; reproduces vLLM's emulation token stream),
  NOT the cutlass sm120a fp4 tensor-core GEMM that the 27B/35B W4A4 path uses
  (`MatmulNvfp4Fp4DirectD`, qwen3_5.cpp). Compounded by the per-expert loop (top-k×3
  GEMMs/token), a host `ScaledFp4Quant`+`DrainQueue` per GEMM, and the unified-memory
  raw-view retag (no device residency).

**N5 nsys TRACE (2026-08-01, `cuda_gpu_kern_sum`, our `laguna-gen --gpu` decode, 5 tok):**
only TWO kernels ran on the GPU — **`MatmulNvfp4Fp4Naive<float>` = 99.3%** of GPU time
(14,100 instances, avg 526µs, med 254µs — the emulation fp4 GEMV, ~100× a tensor-core
fp4 GEMM), `ScaledFp4QuantFastKernel` = 0.7%. GPU busy only ~18% of wall time. The
absence of ANY bf16 GEMM kernel proves the second lever: `LqGemm`'s bf16 branch runs the
HOST `MatmulNK` reference (`laguna.cpp:184`) even on the CUDA queue, so ALL bf16 work
(attention q/k/v/o, dense L0, router, shared expert, embed, lm_head) executes on the CPU
— ~4.8 s/tok of the 6.34. Source-only scan would have missed this; the trace found it.

N5 SPEED PLAN (the user's goal — reach ≥ vLLM 18.8 tok/s, both at NVFP4; trace-ranked):
1. **Native fp4 tensor-core MMA — DONE (2026-08-01), another ~2×.** The post-lever-2 nsys
   showed `MatmulNvfp4Fp4Naive` at 92% of GPU time. TURNS OUT the engine ALREADY has a native
   sm120a fp4 tensor-core kernel (`MatmulNvfp4Fp4Native`, `mma.sync kind::mxf4nvf4`,
   `cuda_matmul_nvfp4.cu:2726`) that reads the SAME LINEAR scale layout `LqGemmNvfp4Fp4`
   produces (`a_scale[row*groups+col]`, :2763) — so NO swizzle/DirectD machinery was needed;
   it was just gated OFF behind `VT_NVFP4_FP4_NATIVE` (`NativeFp4MmaEnabled`, :1669). The
   Laguna driver now defaults it ON (`setenv(...,0)`, scoped — 27B/35B use the separate
   `MatmulNvfp4CutlassModel`/DirectD path via `NvfpCutlassEnabled`, untouched). GB10: decode
   0.39 → ~0.20-0.24 s/tok (~2×; ~4.2-5.0 tok/s); byte-identical ids to the emulation path
   (numerically equivalent), first token matches the golden. **Cumulative: 0.16 → ~4.5 tok/s
   (~28×), ~4× from vLLM 18.8.** NOTE: the DirectD swizzled-scale cutlass path remains a
   possible further lever if the native MMA trails cutlass on this shape — but the native
   kernel is already tensor-core, so the bigger remaining wins are the host-orchestration tail:
2. **Route the bf16 GEMMs to the GPU — DONE (2026-08-01), 16× decode.** `LqGemm`'s bf16
   branch now casts the small `[T,K]` f32 activation to bf16 on-device (`vt::CastBf16`) and
   runs `vt::MatmulBT` (bf16×bf16→f32, cuBLASLt nvjet), keeping the weight bf16 (no per-token
   `ReadF32` of `[N,K]`; `lm_head [Vsz,H]` was the dominant host cost). GB10: decode
   6.34 → 0.39 s/tok (16.3×; 0.16 → 2.56 tok/s), prefill 17.3 → 2.24s; coherence preserved
   (near-tie shifted slightly from bf16-activation rounding). CPU path keeps `MatmulNK` (run-gate
   byte-identical); GGUF tower is block-quant so nvfp4-arm-only. STILL 7.3× short of vLLM 18.8.
3. **Grouped W4A4 MoE — NEXT lever (post-lever-1 nsys: 22,115 `cudaStreamSynchronize`/run =
   78.6% of API time, ~2,760/token — the per-GEMM `DrainQueue` in the per-expert loop; GPU
   kernels are fast, the arm is HOST-SYNC-bound).** Collapse the top_k×3 per-expert
   GEMMs+drains/layer into 3 grouped GEMMs. DESIGN INPUT (checked 2026-08-01): `vt::MoeGroupedGemmNvfp4`
   exists but is **W4A16** (bf16 activation, only `scale2` — the 35B uses it at `qwen3_5.cpp:4645`),
   so it is NOT drop-in for our TRUE-W4A4 experts (wrong numerics — no fp4 activation quant / no
   `input_global_scale`/`alpha`). Options: (a) a NEW grouped fp4×fp4 (W4A4) op — quantize the
   shared token activation to fp4 ONCE, then a grouped fp4×fp4 GEMM over the top_k expert weights;
   (b) switch the experts to the `use_a16` W4A16 grouped mode (valid per the MIRROR directive, but
   a numerics change to verify) — both ALSO need the per-expert `Nvfp4Weight` vector STACKED at load
   (like qwen3_5 A3 W2) so the grouped op sees one `[E*N,K]` block. Needs a spike.
4. **Device-resident decode (RECOMMENDED next brick — kills the 22k syncs at the root).**
   The current Laguna forward is HOST-style (every `LqGemm`/`LqGemmNvfp4Fp4` returns a
   `std::vector<float>` + `DrainQueue`; host `GateUpSilu`/combine), which is why there are
   ~2,760 `cudaStreamSynchronize`/token. Restructure the decode to keep activations ON-DEVICE
   across a layer's GEMMs and drain ONCE per step. REUSE TARGET (checked 2026-08-01): qwen3_5
   already has the full device-resident MoE machinery — `struct Dev`/`Nvfp4Dev`
   (`qwen3_5.cpp:449/878`), `ResidentNvfp4` (:887, lazy `d_packed`/`d_scale` staging),
   device `ScaledFp4Quant`/`MoeGroupedGemmNvfp4`/`MoeGateUpSwiGLUGrouped`, and `RunMoeBlock`
   (`qwen3_5_moe_block.h`) — but these are qwen3_5-internal (static); the Laguna device forward
   mirrors the pattern (expose/replicate the seam). This is the SAME lever the GGUF path owes
   (task #228) and lifts BOTH Laguna quant paths. Substantial; spike-first per protocol.
   **DECISIVE PRECEDENT (DeepSeek-V4 device-decode campaign, Bricks A–D, `state.md`/`CLAIM-
   DEEPSEEK-V4-DEVICE-DECODE`):** ds4 built exactly this (`ForwardResidentDecodeGguf`, whole
   decode as one async device chain, drop the per-op syncs) and it was TOKEN-IDENTICAL but
   **EAGER-SLOWER (−20%)** — the ~1700 small device-kernel launches/step leave the GPU ~45%
   idle in host-launch GAPS; the sync-drop is a PREREQUISITE, not the payoff. **The payoff is
   the decode CUDA-graph (lever 5): collapse the launches into ONE `cudaGraphLaunch`.** So
   levers 4+5 are ONE campaign (device-resident THEN graph); per the ds4 projection a graph
   reaches maybe ~10-13 tok/s (2-2.5×) — clears our current ~4.5 but may still trail vLLM 18.8
   (vLLM also graphs + has tuned cutlass kernels). Honest ceiling uncertain vs 18.8; the
   campaign is how we find out, and it also lifts the GGUF path (#228).
5. Decode CUDA-graph + on-GPU sampling (the ds4/35B host-orchestration levers) — the PAYOFF
   step of lever 4 (see the ds4 precedent above), not a separate lever.

**CORRECTED CEILING (2026-08-01, from the measured lever-1 state — CRITICAL for planning):**
at 0.20 s/tok the GPU is ~87% busy ⇒ GPU compute ≈ 0.174 s/tok, host gaps only ≈ 0.026. So a
PERFECT decode graph (levers 4+5, removing ALL host gaps) caps at **~5.9 tok/s** — it clears
our ~4.5 but is still **3.3× short of vLLM's 18.8**. The graph is NECESSARY BUT NOT SUFFICIENT.
The remaining 3.3× is **KERNEL EFFICIENCY**, a SEPARATE lever set: our enabled native fp4 MMA
runs ~302µs per M=1 expert GEMM (tensor-core tiles waste the single row), vs vLLM's tuned
cutlass sm120a fp4 GEMM (`MatmulNvfp4CutlassModel`/DirectD, swizzled scales, resident weights)
+ FUSED norm+quant / silu+quant kernels (fewer launches, no bf16 intermediate). So the true
parity plan is TWO campaigns: (A) device-resident forward + graph → ~5.9 tok/s; (B) route the
experts to cutlass DirectD + adopt the fused norm/quant/silu ops (mirror 27B/35B) + an
M=1-tuned fp4 GEMV → the remaining 3.3×. Both are substantial; (B) is the bigger lift and the
one that actually reaches 18.8. Honest: single-GB10 Laguna-NVFP4 parity is a multi-campaign
effort, not one graph.

**N5 update (2026-08-01) — hw-fp8 GEMV dequant: MEASURED NEGATIVE, reverted.**
Hypothesis: the M=1 fp4 GEMV (MatmulNvfp4Fp4Gemv) is compute-bound on the per-byte
software fp8-e4m3 scale decode (F8E4M3ToF32Dev's `ldexpf`), so a hardware
`cvt.rn.f16.e4m3` (via `__nv_fp8_e4m3`→float) would cut it. Implemented bit-exact
(generated ids byte-identical on the real 67 GiB ckpt) + templated behind
VT_NVFP4_FP4_GEMV_HWFP8 for a same-binary A/B. **Paging-immune ncu (sm_121a) verdict:
hardware cvt is NEUTRAL-to-slightly-WORSE** — grid768 41.2us(hw) vs 41.9us(sw) = tie;
grid256 59.8 vs 53.2 (within the 45-87us per-sample spread); mean 53.6(hw) vs 49.4(sw).
On the GPU `ldexpf` compiles to a cheap exponent-bit add, so the software decode was
already efficient. Reverted (commit ab7a1c1e). Recorded so the LUT-vs-cvt question
is closed, not re-litigated. NOTE the end-to-end wall-clock A/B was USELESS: the 67 GiB
unified reload + paging swings TPOT 0.16↔1.08 s/tok run-to-run — the kernel-duration
ncu is the only honest anchor here (matches [[dgx-passwordless-sudo-clean-measurement]]).

**N5 ATTRIBUTION CORRECTION (2026-08-01) — the "host-orchestration" framing is GGUF-only.**
[[laguna-speed-host-orchestration-bound]] (⅔ host-sync, 22k syncs/step, GPU 32.7% active)
was measured on the GGUF keep-quant path, NOT NVFP4. The NVFP4 lever-1 state is ~87%
GPU-busy (line 254), and current best decode ≈ 0.16 s/tok = **6.25 tok/s already MEETS the
~5.9 tok/s perfect-graph ceiling**. So for NVFP4 the device-resident-decode + graph campaign
(#228) is LARGELY SPENT — it cannot clear the current number by more than noise. The ONLY
lever left to 18.8 is **kernel efficiency of the M=1 fp4 GEMMs** (campaign B): the compute-
bound GEMV (68% SM at grid768) + routing experts to vLLM's tuned cutlass DirectD (swizzled
scales, resident weights) + fused norm/quant/silu. Next bounded bricks, ranked:
  B1. GEMV dequant arithmetic: replace the kE2M1 `__constant__` LUT (data-dependent index →
      constant-memory serialization across a warp) with branchless bit-arithmetic e2m1→f32;
      + vectorized (uint4) packed-weight loads. Cheapest, self-contained, ncu-gated.
  B2. **Route Laguna experts to the MARLIN W4A16 grouped MoE GEMM — the HIGHEST-VALUE lever
      and the true 18.8 path (scoped 2026-08-01, zero-DGX source study).** KEY: vLLM's 18.8 bar
      is **MARLIN W4A16** (`VLLM_TEST_FORCE_FP8_MARLIN=1`; bf16 activations + fp4 weights), NOT
      cutlass W4A4 — and Marlin is LOW-M-optimized (decode/batch-1), exactly where a tensor-core
      cutlass/native-MMA GEMM wastes tile rows on M=1. The engine already ships the EXACT kernel:
      `vt::MoeGroupedGemmNvfp4Marlin` (`include/vt/ops.h:1340`, a 1:1 lift of vLLM
      `moe_wna16_marlin_gemm`, `ops.cu:543`) + the shared repack primitive
      `MarlinRepackExpertWeight` (`src/vt/cuda/cuda_marlin_repack.cu:131`, vendored
      `gptq_marlin_repack_kernel`). **qwen3_5 (27B/35B) ALREADY uses this path** — DEFAULT ON
      (`MarlinMoeEnabled()`/`VT_NVFP4_MARLIN`), validated **16/16 token-for-token vs oracle,
      +22% gate / +80% decode-heavy** — via `BuildMoeMarlinResident` (`qwen3_5.cpp:4674`, repacks
      each expert's fp4 weight → i32 marlin `[E,K/16,N*2]` + fp8 scales `[E,K/16,N]` + f32 global
      scale at LOAD) + `MoeMarlinResident` (:617) + the fused-w13 forward (:2210). Laguna reuses
      qwen3_5's whole infra, so B2 = **(i)** a `BuildLagunaMoeMarlinResident` mirroring
      `BuildMoeMarlinResident` but over `LagunaMoeWeights.experts_{gate,up,down}_fp4`
      (`Nvfp4Weight`, already stacked `[E,N,K/2]`) — call the SAME shared `MarlinRepackExpertWeight`
      + scale/global-scale packing; **(ii)** route `LagunaFfnBlock`'s fp4 branch (currently
      `LagunaMoeResidentFp4` → per-expert `MatmulNvfp4Fp4` GEMV) to `MoeGroupedGemmNvfp4Marlin`
      when `MarlinMoeEnabled()` + resident ready, mirroring qwen3_5:2210; keep the GEMV path as
      the `VT_NVFP4_MARLIN=0` escape hatch for A/B. Gate: near-tie vs the vLLM-**Marlin** golden
      (W4A16 numerics → should match BETTER than our current W4A4) + kernel-duration ncu +
      same-binary A/B. This is pure reuse of proven shared ops ([[ground-every-impl-in-upstream]],
      MIRROR-vLLM + FOLD-onto-shared-MoE-Marlin directives) — no new kernel. NOTE the shared
      expert also has a Marlin arm (`SharedGateUpFusedMarlinD`) for the same fold.
  (B0 — hw-fp8 cvt — DONE, negative. B2 supersedes the earlier "cutlass DirectD" framing: Marlin
   is vLLM's ACTUAL 18.8 kernel and is the low-M-correct choice; DirectD/cutlass W4A4 is a
   prefill/large-M lever, not the decode bar.)
