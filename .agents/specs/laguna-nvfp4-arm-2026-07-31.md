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
**N1a + N1b + N2 LANDED + CPU-gated (2026-07-31).** The NVFP4 W4A4 arm is
buildable end-to-end on CPU: weight struct → loader → forward branch, with
`test_laguna_nvfp4_loader` (3 cases, 61 assertions) covering byte-identity load,
missing-tensor RED, and a finite+deterministic forward run through the fp4 MoE
branch. The GGUF keep-quant path stays byte-identical (`test_laguna_scaffold`
8/8). REMAINING:
- **N3 — run entrypoint.** Point `examples/laguna_gen` (the greedy driver the DGX
  benchmark loops) at the NVFP4 safetensors dir → `LoadLagunaForCausalLMWeights`
  → `LagunaForwardGguf`. (Loader dispatch already keys on dir vs GGUF.)
- **N4 — CORRECTNESS gate (DGX).** Greedy on the real 67 GiB checkpoint, near-tie
  vs the recorded vLLM-NVFP4 golden (`~/laguna-nvfp4/vllm_golden.txt`).
- **N5 — SPEED gate (DGX).** ours-NVFP4 tok/s vs vLLM-NVFP4 18.8 (MARLIN);
  grouped-W4A4 fold + device residency are the speed levers if the per-expert
  loop trails. This is the apples-to-apple bar the user set (both at NVFP4).
