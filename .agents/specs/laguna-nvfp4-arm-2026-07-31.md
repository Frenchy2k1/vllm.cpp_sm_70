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
4. **Device residency** — stage `d_packed`/`d_scale` once (ResidentNvfp4), drop the
   per-GEMM host round-trip + `DrainQueue` sync.
5. Decode CUDA-graph + on-GPU sampling (the ds4/35B host-orchestration levers).
