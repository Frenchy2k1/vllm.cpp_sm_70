# Per-channel FP8 GDN projection support (`Qwen3.8` loader)

Option A — port the per-channel-FP8 GDN loader so `unsloth/Qwen3.8-27B-NVFP4`
loads on the sm70 branch (mirror of the dense/FP8 arms already shipped for 3.6).

## Status

`PENDING`. Filed 2026-08-20 against the working tree of
`row/BACKEND-DISTRIBUTED-TP-CLEAN`. This is the "Option A" the operator and
developer agreed to record as a plan, after the developer chose "do B, add A to
the plan" for the qwen3.8-fp4 benchmark ask. It has NO open GitHub issue yet
(the repo requires one before the row can enter `.agents/issue-index.md`);
opening the issue is the first unblocking step (needs `gh`/a token or the
developer opening it).

## Scope

- **Defect**: serving dies at load with
  `dense loader: 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale'
  ships shape [10240, 1] (10240 elements), not the ONE element a per-tensor scale
  is` — `include/vllm/model_executor/models/dense_weight_loaders.h:168`,
  `ReadF32Scalar` (this is the per-channel scale class the loader refuses by
  design — `dense_weight_loaders.h:151-163` documents why a silent misread yields
  fluent-wrong tokens).
- **Checkpoint shape census** (`unsloth/Qwen3.8-27B-NVFP4 @ 7d6f8d4d...`, model.safetensors):
  - `in_proj_qkv.weight` `F8_E4M3 [10240,5120]`, `.weight_scale` `BF16 [10240,1]` per-channel
  - `in_proj_z.weight` `F8_E4M3 [6144,5120]`, `.weight_scale` `BF16 [6144,1]` per-channel
  - attn q/k/v/o + MoE projections use fp8-e4m3 + per-column scales or the existing NVFP4 arms
  - 336 x F32 (1,) per-tensor, 112 x fp8 grid `[out, in/16]`, per-channel BF16 `[out,1]` rows
- **Deliverable**: a `LoadGdnFp8PerChannel` arm that stores the fp8 bytes + the
  per-channel BF16 scale and applies the scale in the GDN in_proj GEMM epilogue
  (the same per-column-alpha machinery the fp8 tower already uses), token-gated
  against the CPU oracle like the other arms.

## Gates

- `test_sm70_nvfp4_gemm` (existing) stays green.
- New: a model-free GDN in_proj per-channel fixture (device vs CPU oracle) at the
  10240/6144 x H scales; then `test_qwen38` load-and-forward (loads the real
  checkpoint, GPU 0,1).
- `test_sm70_fa2_*` and `cuda_backend` unchanged.

## Stop when

- A `Qwen3.8-27B-NVFP4` load reaches the first forward on the sm70 branch and
  the GDN projections produce fp8-with-per-channel-scale equals the oracle.

## Owner

Deferred to Option A of the "explore + port v100-skinny / 1Cat" ask (the
developer: "do B, add A to the plan").