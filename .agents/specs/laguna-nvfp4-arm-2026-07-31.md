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
  - **N1b — the loader body (CONCRETE RECIPE, derived 2026-07-31).** Replace the
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
- **N2 — forward quant-branch.** In `laguna.cpp`, select the NVFP4 arm when the
  weights are fp4-resident: attention/dense via bf16 `MatmulBT`; routed experts
  via `kMoeGroupedGemmNvfp4` (P = T·top_k, mirror the A3 grouped shape); shared
  expert + lm_head via the dense NVFP4/bf16 sink. Keep softplus gate / dual RoPE
  / sigmoid router / sliding-window mask exactly as the GGUF path.
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
SPEC ONLY (2026-07-31). No code. Blocked on user direction between this arm (1a)
and the GGUF-path optimization (1b); both need DGX gates. Reference studied:
qwen3_5 NVFP4 loader + ops confirmed reusable; checkpoint structure verified.
