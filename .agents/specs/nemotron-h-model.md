# Nemotron-H — the first hybrid Mamba2 model, and the first MIXED_PRECISION checkpoint

**Claim:** `CLAIM-MODEL-NEMOTRON-H`. **Model row:**
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` (existing, stays `INVENTORIED`
at this spec commit). **Issue:** [#517](https://github.com/mudler/vllm.cpp/issues/517).

**Hard blocker:** `KERNEL-SSM-MAMBA` [#496](https://github.com/mudler/vllm.cpp/issues/496)
([spec](mamba2-ssd.md)). W1 (CPU host references) is in fresh review on
`row/KERNEL-SSM-MAMBA-SSD-W1`; the CUDA arm is W2 of that row. **No forward
path in this spec can be gated before that lands**, and W3 below says so
explicitly rather than pretending otherwise.

**Base:** `main` HEAD `66deca15`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).
**Driver checkpoint:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
at pinned revision `29f2d1746d8f41e316523194b19018707749b1b1`, staged on the
NAS at `$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` (20.1 GiB, fits one
GB10).

---

## 0. Scope (headline verdict)

Two things make this arch more than "another model file", and both are the
first of their kind in this tree:

1. **It is the first true hybrid Mamba2 model.** 52 layers: 23 Mamba2, 6 GQA
   attention, 23 MoE. Our GDN hybrids (Qwen3.5, Kimi-Linear) supply the het-KV
   machinery, but not the recurrence — see [mamba2-ssd.md](mamba2-ssd.md).
2. **It is the first `MIXED_PRECISION` checkpoint.** One file carries NVFP4
   W4A16 group-16 experts, FP8 W8A8 static mamba projections, bf16 attention,
   and an fp8 KV scheme. We have never resolved a quant algorithm *per module*;
   `quantization_config` is read ad-hoc in exactly two weight files today
   (`kimi_k3_weights.cpp:171`, `deepseek_v2_weights.cpp:365`).

A third thing is smaller but has no local precedent either: the MoE is
**non-gated**. There is no `gate_proj` anywhere in the checkpoint — the expert
is `up_proj -> relu² -> down_proj`. Every grouped-MoE op we have is
SwiGLU-shaped.

**Out of scope, explicitly:** `NemotronHPuzzleForCausalLM` heterogeneous
per-layer configs (`get_nemotron_h_config_for_layer`), `moe_latent_size`
(null in this checkpoint, so `fc1_latent_proj`/`fc2_latent_proj` are absent),
TP sharding of `n_groups`, ReplaySSM, and any speed claim. Correctness first,
always; no ratio is recorded by this row until the token gate passes.

## 1. The checkpoint, exactly

`architectures: ["NemotronHForCausalLM"]`, `model_type: nemotron_h`,
`hidden_size=2688`, `vocab_size=131072`, `max_position_embeddings=1048576`,
`tie_word_embeddings=false`, weight prefix `backbone.`.

`layers_block_type` (52 entries) gives 23 `mamba`, 23 `moe`, 6 `attention`
at indices **5, 12, 19, 26, 33, 42**.

| Block | Parameters |
|---|---|
| Mamba2 | `mamba_num_heads=64`, `mamba_head_dim=64`, `n_groups=8`, `ssm_state_size=128`, `conv_kernel=4`, `chunk_size=128`, `mamba_hidden_act=silu`, `use_conv_bias=true`, `use_bias=false`, `mamba_proj_bias=false`, `mamba_ssm_cache_dtype=float32`, `time_step_min/max/floor = 1e-3 / 1e-1 / 1e-4` |
| Attention | 32 q / 2 kv heads, `head_dim=128`, `rope_theta=10000`, `partial_rotary_factor=1.0`, `attention_bias=false`, `sliding_window=null` |
| MoE | `n_routed_experts=128`, `num_experts_per_tok=6`, `moe_intermediate_size=1856`, `n_shared_experts=1`, `moe_shared_expert_intermediate_size=3712`, `mlp_hidden_act=relu2`, `norm_topk_prob=true`, `n_group=1`, `topk_group=1`, `routed_scaling_factor=2.5`, `moe_shared_expert_overlap=true` |
| MTP | `num_nextn_predict_layers=1`, `mtp_layers_block_type=["attention","moe"]`, weights `mtp.layers.0.{eh_proj,enorm,hnorm,final_layernorm,norm,mixer.*}`, unquantized |

**Quantization** — `quant_method: modelopt`, `quant_algo: MIXED_PRECISION`,
`producer: modelopt 0.44.0rc5`, two `config_groups` plus a 5981-entry
`quantized_layers` map and an `ignore` list:

| Target | Scheme | Tensors |
|---|---|---|
| routed experts, shared experts, `lm_head` | `W4A16_NVFP4`, **`group_size=16`** | `weight`, `weight_scale` (e4m3 per-16-block), `weight_scale_2` (fp32 global) |
| mamba `in_proj` / `out_proj` (46 targets) | FP8 W8A8 static | `weight`, `weight_scale`, `input_scale` |
| attention `q/k/v/o_proj`, `conv1d`, gates, norms, embeddings | unquantized bf16 | — |
| KV cache | `kv_cache_scheme` fp8 | `k_proj.k_scale`, `v_proj.v_scale` |

Note the polarity trap: the repo name says NVFP4, and most of the *parameters*
are, but the mamba projections are FP8 and the attention tower is bf16. Reading
it as uniform NVFP4 gets the loader wrong in a way a token gate can still pass
while moving the wrong bytes — see [porting.md](../porting.md) on checking the
memory format against the oracle explicitly.

## 2. Upstream chain (`file:line` @ `555967922`)

| What | Anchor |
|---|---|
| registry | `registry.py:179` -> `models/nemotron_h.py::NemotronHForCausalLM` |
| layer dispatch (`ALL_DECODER_LAYER_TYPES`, `M`/`*`/`E`/`-`) | `nemotron_h.py:531-536`, model `:546-600` |
| Mamba2 layer | `nemotron_h.py:373-389` (`MambaMixer2`, fed `mamba_num_heads * mamba_head_dim`) |
| MoE | `nemotron_h.py:126-256` (`NemotronHMoE`), decoder layer `:317` |
| non-gated activation | `activation_without_mul(config.mlp_hidden_act)` -> `ReLUSquaredActivation` (`layers/activation.py`) |
| expert ckpt naming | `ckpt_names=("up_proj", "down_proj", "")` (`nemotron_h.py:220`) |
| routed scale applied to OUTPUT | `apply_routed_scale_to_output=True` (`nemotron_h.py:246`) |
| router dtype | `GateLinear(..., out_dtype=torch.float32, force_fp32_compute=True)` (`nemotron_h.py:150-156`) |
| state shape / dtype | `mamba_utils.py:174-199`, `:73-81` |
| MTP | `models/nemotron_h_mtp.py::NemotronHMTP` (`registry.py:638`) |
| MIXED_PRECISION resolution | `layers/quantization/modelopt.py:2280-2450`, per-layer lookup `:2416-2445` |

The `quantized_layers` lookup is **direct name first, then shard prefix**
(`modelopt.py:2426`, `:2437`). Mirror both, in that order; a merged/sharded
local name that resolves only by prefix is the case that will bite.

Config note: `nemotron_h.py` reads `config.hybrid_override_pattern`, which
newer transformers exposes as a property derived from `layers_block_type`
(`_list_to_pattern`, mapping `mamba->M`, `moe->E`, `attention->*`). Our loader
reads `layers_block_type` directly and does not reconstruct the char pattern.

## 3. Our baseline — reuse vs new

### REUSE (landed)

- Het-KV: `MambaSpec` + `HybridKVCacheCoordinator` + per-group managers
  (porting-inventory.md:78-79,109). Only 6 of 52 layers hold a paged KV group.
- Causal conv1d, all three arms (`kCausalConv1dFwd/Update/SpecUpdate`).
- Grouped-topk sigmoid routing with `e_score_correction_bias` and
  `routed_scaling_factor` — the DeepSeek-V2/V4 path (`kMoeRouterTopK`).
- Shared experts (`kSharedExpertGate`, `kMoeCombineGate`).
- NVFP4 W4A16 grouped MoE Marlin (`kMoeGroupedGemmNvfp4Marlin`), FP8 W8A8
  linear, fp8 KV (`kReshapeAndCacheFp8`).
- MTP spec-decode machinery (`qwen3_5_mtp.cpp`, `SPEC-MTP`).
- Dense attention + rope (`dense_attn::AttnBlock`, `kAttnQkNormRope`).

### NEW

- `vt::Mamba2ChunkScan` / `Mamba2StateUpdate` / `RmsNormGatedGroup` — **owned
  by #496, not by this row.**
- A non-gated `relu²` grouped-MoE arm on the merged-GEMM seam.
- A ModelOpt `MIXED_PRECISION` per-module quant resolver.
- `nemotron_h.cpp` / `nemotron_h_weights.cpp` / `nemotron_h_registry.cpp`,
  and the MTP head.

## 4. W-breakdown

Each W is one delegated task with its own fresh implementer and fresh reviewer.
W1 and W2 are independent of #496 and can run in parallel with it; W3 onward
cannot.

| W | Content | Gate | Depends on |
|---|---|---|---|
| **W1** | ModelOpt `MIXED_PRECISION` resolver: parse `quantization_config`, resolve per-module `quant_algo` (direct then shard-prefix), expose it to weight loading. Refuse an unknown algo by name | unit tests on the REAL `config.json` (committed as a fixture, weights not needed): every one of the 5981 entries resolves, the `ignore` list resolves to unquantized, an unknown algo refuses | — |
| **W2** | Non-gated `relu²` grouped MoE through `MlpGateUpMethodBase` / `vt::MergedGemmGroup`; bf16 arm then NVFP4 W4A16 g16 | byte/tolerance tests vs a host reference; `relu²` mutation caught; routed scale applied to the OUTPUT, not the logits | — |
| **W3** | `nemotron_h_weights.cpp` + `_registry.cpp`: `layers_block_type` dispatch, `backbone.` prefix, het-KV group construction (1 Mamba group + 1 full-attn group over 6 layers), enumeration gate vs the released index | enumeration: every tensor in `model.safetensors.index.json` is claimed or explicitly refused; KV spec shapes match `mamba2_state_shape` | #496 W1 |
| **W4** | `nemotron_h.cpp` forward: hybrid layer loop, Mamba2 mixer wiring, 6 attention layers, MoE layers | CPU forward runs; per-layer activations vs a dumped oracle reference | #496 W1, W2, W3 |
| **W5** | MTP head (`mtp.layers.0`, `eh_proj`/`enorm`/`hnorm`) on the existing spec-decode seam | draft acceptance non-zero; spec-off and spec-on token-identical | W4 |
| **W6** | **GB10 e2e token gate vs the pinned oracle** | token-exact greedy, identical prompts/counts/batching/sampling; oracle identity asserted | #496 W2 (CUDA), W4, W5 |
| **W7** | GGUF k-quant / i-quant arm through the shared GGUF loader (see §5b); refused by name until it lands | quant-matched load + token gate | W4 |

## 5. Gates

**Correctness first, always.** No throughput number is recorded by this row
until W6 passes. When speed is measured later, the denominator is vLLM's
production configuration, never `--enforce-eager`.

**Oracle identity is asserted, not assumed.** `$HOME/venvs/vllm-oracle` on the
dgx host symlinks to `vllm-oracle-v0.25.0-stage` — vLLM **0.25.0**, transformers
5.13.1 — which predates `NemotronHMoEDecoderLayer` entirely. A run through that
venv fails on this checkpoint and reads as "the model is unsupported". The pin
is `vllm-oracle-next`: `0.23.1rc1.dev1511+g555967922`, transformers 5.14.1,
flashinfer 0.6.15.post1. Every oracle run asserts all three and ABORTS on
mismatch before producing a number.

**The fixture must be the checkpoint the changed path loads.** Pin the
revision (`29f2d174`) explicitly; a repo silently re-quantized under the same
name has cost this project a full campaign before.

**A token gate cannot see a dtype that is too wide.** Every f32 buffer on this
path owes a one-line reason, and the loaded memory format is checked against
the oracle explicitly, not inferred from matching tokens.

**GPU discipline on dgx:** `flock $HOME/gpu.lock`, `local-ai-worker` parked,
never a large oracle alongside `ctest` — `gpu_memory_utilization` reserves HOST
RAM on GB10 and has OOM-rebooted the box.

## 5a. Gateability: CLOSED — the pinned oracle loads and runs it (2026-08-12)

`.agents/porting-a-model.md` §4 requires saying plainly whether the pinned
oracle can load the model. **It can.** On GB10, from
`$HOME/venvs/vllm-oracle-next` (vLLM `0.23.1rc1.dev1511+g555967922`,
transformers 5.14.1, flashinfer 0.6.15.post1 — all three asserted, the run
aborting on mismatch), against
`$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` at pinned revision
`29f2d1746d8f41e316523194b19018707749b1b1`:

- `ORACLE_IDENTITY_OK`
- `CONFIG arch = ['NemotronHForCausalLM']`, `nlayers = 52`
- `CONFIG pattern = MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` —
  confirming on the real checkpoint that newer transformers derives
  `hybrid_override_pattern` from `layers_block_type`, which `nemotron_h.py`
  reads. That was previously a source-level assumption.
- `MODEL_LOADED_OK`, greedy decode, `EXIT=0`.

Constructing a config proves nothing; this ran the model. Three greedy goldens
(`temperature=0.0`, `max_tokens=32`) are committed at
[`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json`](../../tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json),
with the revision pinned as `parity::kNemotron35LightningNvfP4Revision`
(`tests/parity/hf_snapshot.h`) so the gate names the revision its golden belongs
to. Output is coherent — Fibonacci is `0, 1, 1, 2, 3, 5, 8, 13, 21, 34` — so
this is a healthy denominator, not a degraded one.

**Two traps recorded here because they cost time.** (1) `$HOME/venvs/vllm-oracle`
on dgx symlinks to `vllm-oracle-v0.25.0-stage`, which predates
`NemotronHMoEDecoderLayer`; a run through it fails and reads as "the model is
unsupported". Use `vllm-oracle-next`. (2) A vLLM v1 driver script MUST guard its
body with `if __name__ == "__main__":` — EngineCore is spawned, the module
re-imports, and the failure surfaces as a `multiprocessing` traceback naming
neither vLLM nor the caller. The tell is the banner printing twice.

W6 consumes this golden; nothing else in this spec is unblocked by it.

## 5b. Quantized arms owed (`porting-a-model.md` §2)

`AGENTS.md` makes the quantized arms part of a model port, not a follow-up, and
names **GGUF k-quants** a standing requirement rather than a per-model choice.
For this row that means:

| Arm | State |
|---|---|
| ModelOpt NVFP4 W4A16 g16 + FP8 W8A8 (the shipped checkpoint) | W1-W6, the critical path |
| Safetensors bf16 | reachable via the same loader; owed a fixture |
| **GGUF k-quants / i-quants** through the shared GGUF loader | **OWED.** No NemotronH GGUF arm exists |

Until the GGUF arm lands it is **refused by name** at load, naming the missing
piece, never silently dequantized to a supported path — a silent fallback is
exactly what a token gate cannot see. Tracked as W7.

## 6. Risks / decisions

- **Non-gated MoE must not become a parallel path.** If
  `MlpGateUpMethodBase` / `vt::MergedGemmGroup` cannot represent a
  gate-half-absent expert, extend the seam or record one exact tracked
  exception. Never hand-roll a sibling.
- **`group_size=16` NVFP4.** Confirm our Marlin grouped path actually supports
  16 and does not silently assume another group size. Prove it on the real
  tensors, not on a synthetic fixture.
- **Router in f32.** Upstream forces fp32 router compute
  (`force_fp32_compute=True`). Mirror the polarity; do not inherit the model
  dtype here.
- **`routed_scaling_factor` position.** Applied to the OUTPUT
  (`apply_routed_scale_to_output=True`), not folded into the router weights. A
  mis-placed scale is exactly the class of error a token gate catches late and
  a unit test catches immediately.
- **6 attention layers out of 52** means KV is small and the 1M context is
  cheap — but it also means an attention-side defect is diluted across 46
  non-attention layers and may not move tokens on short prompts. Gate with a
  long-prompt arm, not only a 6-token one.

## 7. Now

**State at this commit:** spec committed, implementation **not started**. The
row stays `INVENTORIED`; this commit changes no lifecycle state. The checkpoint
is staged on the NAS and the oracle smoke run is queued behind the GPU lock.

**Next action:** dispatch fresh implementers for **W1** and **W2** (both
independent of #496) as soon as `row/KERNEL-SSM-MAMBA-SSD-W1` clears review,
so the ops-header churn does not collide.

## 8. Stop conditions

- The pinned oracle cannot be made to load and run this checkpoint on GB10 →
  stop and report; without a running oracle there is no gateable denominator
  and the row does not proceed on source inspection alone.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name and record it as owed. Never silently dequantize to a supported path;
  that is invisible to a token gate.
- The non-gated expert cannot be expressed on the shared merged-GEMM seam →
  `NEEDS_DECISION`, do not fork a parallel MoE path.
