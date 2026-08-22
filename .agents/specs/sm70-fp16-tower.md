# V100 fp16 decode tower for the dense NVFP4 MLP (route through sm70 SIMT/QPN)

Issue: PENDING — no `gh` CLI or GitHub token in this session; open in
`mudler/vllm.cpp` and link from here, from the index, and from the PR.
Row: `SM70-FP16-TOWER`

## Why

The V100 (sm70) NVFP4 dense MLP decode is `MatmulNvfp4KernelNaive`
(`cuda_matmul_nvfp4.cu:266`), a one-thread-per-output f32-accumulating bf16
dequant-GEMM. On the served 2.62 tok/s session it is 66% of GPU time (nsys,
this work session). The subtree's fast decode path is the sm70 SIMT/QPN lane
(`cuda_sm70_nvfp4_gemm.cu` `LaunchSm70Nvfp4W4a16` family), which reaches the
Volta tensor-core rate on decode, but it is **fp16-domain**: the
`mma.sync.aligned.m8n8k4.f32.f16.f16.f32` (Volta tensor core) consumes fp16
fragments (`__half`, `dequant8_k2`), unlike the model's bf16 dtype.

They cannot be bridged: E2M1 dequant rounds at bf16 (7-bit,
`DecodeFp4Byte`'s `__float2bfloat16`) in the dense kernel and at fp16 (10-bit)
in the SIMT deck; the f32 product of `gscale` (per-tensor f32) differs across
the two mantissas. Routing bf16 decode through the fp16 SIMT silently drifts
tokens from the bf16 oracle.

This row adopts the reference's own answer: run the fp4 tower in **fp16
end-to-end**, exactly as v100-skinny and 1Cat do (their engines are fp16), and
re-pin the V100 oracle to fp16. That is a recorded developer decision:
**full fp16 tower** — activations into and out of the fp4 tower, the KV cache,
and attention follow fp16 so the fp16 SIMT/QPN deck becomes the true
denominator instead of a bridging workaround.

References in the tree: the bf16 anchor is `cuda_matmul_nvfp4.cu:152-184`
(`Store`/`DecodeFp4Byte`/`Fp4ColDot`); the fp16 SIMT/QPN deck is
`cuda_sm70_nvfp4_gemm.cu:78-108` (`dequant8_k2`), `:330-340` (`MMA_8N8K4 =
mma.f32.f16.f16.f32`), `:418-582` (`Sm70Nvfp4QpnNacc`/`Mt2`). No external
reference implements a bf16 SIMT/QPN on sm70 — Volta MMA is f16-only; a bf16
SIMT is CUDA-core FMA and cannot reach the tensor-core rate (measured this
session).

## Design

Three waves, shipped inside ONE change (they share the dtype contract).

### Wave 1 — fp16 SIMT/QPN route for the dense MLP (M ≤ 16)

- New registered op in the sm70 TU mirroring the fp8 bridge
  (`MatmulFp8W8a16QueueCuda`'s shape, `cuda_sm70_nvfp4_gemm.cu:1292-1377`): a
  `kMatmulNvfp4Sm70Fp16` op implements the dense fp16 decode via
  `LaunchSm70Nvfp4W4a16` with the same
  bf16→fp16 activation staging, grow-only fp16 scratch (graph-capture legal),
  and the same-math fallback for any non-QPN shape (M ≤16, `n%32==0`,
  `k%64==0`; M-band via `Sm70Nvfp4QpnNacc`/`Mt2`).
- `MatmulNvfp4KernelCuda` bf16 branch dispatches to it when the device is sm70
  and the shape passes the QPN gate; the naive bf16 arm is the byte-exact
  fallback for every other shape and every other device.

### The fp16 dtype wall

The fp4 MLP output feeds `MlpGateUpSilu` and the next layer; to make the fp16
route the real denominator, the tower's activations, the KV cache, and
attention run **fp16 end-to-end** (the coherent v100-skinny shape), and the
loader keeps the fp8/fp4 weights raw (the fp16 dequant happens in-kernel).
Only the bf16 activation/io tensors become fp16 at the tower boundary.

Rejected: an fp16-only-inside-the-GEMM hybrid with bf16 KV/attn — a mixed
precision state no single oracle can validate.

### Wave 2 — re-pin the oracle

Run the pinned vLLM on the identical workload with the model dtype = fp16
(the V100 fp16 path), read the matching upstream path, and advance the pin
in `.agents/upstream-sync.md` per AGENTS.md after reconciling every affected
row and gate.

## Scope

- `src/vt/cuda/cuda_sm70_nvfp4_gemm.cu` — new `kMatmulNvfp4Sm70Fp16` op +
  QPN-gate dispatch (mirrors the fp8 bridge at `:1292-1377`).
- `src/vt/cuda/cuda_matmul_nvfp4.cu` — dense bf16 branch routes sm70 to the new
  op when QPN-shaped; the naive stays the fallback.
- The dense-weight loader / tower tensors: fp16 dtype for the fp4 tower
  activations, KV, attention; norms fp32 per v100-skinny.
- `include/vt/ops.h`, `.agents/issue-index.md`, `.agents/upstream-sync.md`,
  `docs/FEATURES.md`, `docs/USAGE.md`, `docs/STATUS.md` as the dtype surface
  changes.

## Gates

- **Wave 0 (route)**: mutation test `test_sm70_nvfp4_fp16_decode` — with the
  QPN gate disabled in a scratch copy, the route misses and falls back to the
  bf16 naive; green proves the fp16 op is what a decode reaches. Plus the full
  gate.
- **Wave 1 (tower)**: token-exact under the fp16 re-racking oracle; the A/B on
  the idle host with the same binary is the comparator; plus
  `test_qwen3_5_decode_graph_seam` under the fp16 dtype.
- **Wave 2 (pin)**: `upstream-sync.md` gate — the pin advances only after the
  fp16 oracle demonstrably builds and runs the model; reconcile affected rows.

```
cd build-all && ninja examples/vllm-server && ctest
```

## Stop conditions

- If the fp16 oracle does not reproduce the bf16 token stream on the same
  weights (it should legitimately differ — fp16 is a different dtype and the
  pin advances to fp16 as the V100 oracle), stop and record the fp16-vs-bf16
  divergence as the new oracle baseline; never smooth it.
- If the fp16 tower shows an unexplained token difference at a shape within
  oracle coverage, stop and reduce the fp16-GEMM-only lane rather than widen.

## Owed

- The GitHub issue (PENDING — external: no credentials this session; owning
  operator opens it).
- The oracle re-pin (needs an idle host + the vLLM fp16 run): operator-held.