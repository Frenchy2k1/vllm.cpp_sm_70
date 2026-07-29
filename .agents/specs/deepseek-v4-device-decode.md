# DeepSeek-V4-Flash — device-resident decode forward (toward a capturable decode CUDA graph)

Owner: `CLAIM-DEEPSEEK-V4-DEVICE-DECODE` (under `CLAIM-DEEPSEEK-V4-IMPL`). Base
`main` `59260579`. Row stays `ACTIVE` (speed campaign on the coherent model). The
campaign's largest brick, run STAGED with per-brick correctness gates. NOT pushed.

## 0. Goal + why

Stage 3 (Stage-3 finding, merged `59260579`) proved a decode CUDA graph is
IMPOSSIBLE on the host-orchestrated forward: the vt capture contract
(`cuda_backend.cu:173-182`) needs a pure ASYNC device sequence (no Synchronize, no
host<->device blocking copies, no malloc; fixed pointers), but
`ForwardComposeImpl` (`device=false`) runs MHC/Sinkhorn, the attention
QK/softmax-sink/AV Dot loop, rope, RMSNorms, the MoE router, SwiGLU and combine ON
THE HOST between GPU GEMMs, reading each GEMM output on the host. To make the
decode CUDA graph capturable, the WHOLE decode step (T=1) must become device-
resident on persistent `DBuf`s with NO per-op host sync — mirroring
`Qwen3_5DenseDecodeGraph`'s `ForwardLayers` (`qwen3_5.cpp:5820`; ":5747
captures/replays ForwardLayers over that fixed hidden address").

Current FINAL (host-orchestrated, Stages 1-2): decode **5.83 tok/s** (23-tok),
token-identical "…Paris."; ds4 **16.5**. Target: ds4-class.

**CORRECTNESS ANCHOR (never regress):** every brick stays token-identical to the
current `--gpu --kv-cache` path (ids `11111 16 455 6102 294 8760 344 11111` =
"…Paris."), OR — where a device fp reduction reorders vs host f32 (softmax,
Sinkhorn, RMSNorm, AV-accumulate) — a CHARACTERIZED distributional near-tie
(coherence preserved, ours ∈ the host path's greedy set), never silent.
`test_deepseek_v4_gguf_load` stays green at every brick.

## 1. What runs where today (the decode step, T=1)

Per layer (`deepseek_v4.cpp` `AttentionBlock`/`MoeBlock`/`ForwardComposeImpl`):
- **GPU (kMatmulBTQuant / kMatmulBTQuantGrouped):** MLA wq_a/wq_b/wkv, grouped
  o-LoRA (8 group GEMMs), router gate, 3 grouped routed-expert GEMMs, 3 shared,
  lm_head. ~18 launches/layer, each followed by a host read.
- **HOST (`std::vector`, between the GPU GEMMs — the blockers):** per-head q
  RMS-norm + dual-RoPE (`:465-499`), the attention QK/softmax-sink/AV Dot loop
  (`:736-756`) + inverse-RoPE (`:758-`), MHC pre/post + 20-iter Sinkhorn +
  hc_head (`ForwardComposeImpl`), kv RMS-norm, the MoE router
  (sqrtsoftplus + top-k/hash, decides `expert_ids`), clamped-SwiGLU, expert
  combine, final RMSNorm.

The real dense run has NO compressor/indexer (`dsa_dense` → `is_comp`/`is_indexer`
false, `:682-684`), so those are OUT of the decode-graph scope (long-context
residual). KV cache = per-layer `deck` latent `[hd]` (`DeepseekV4KvCache`), already
unified-memory.

## 2. Persistent-DBuf layout (Brick C)

Mirror `Qwen3_5DenseModel`'s persistent buffers (`qwen3_5.cpp:3130` "read by EVERY
layer", `:5845` "into persistent device buffers all layers read"). Decode T=1:
- `hidden` `DBuf[hc*H]` (the MHC residual manifold, persistent, in-place across
  layers) + `x` `DBuf[H]` working.
- Per-layer scratch (persistent, reused): `qa[qlr]`, `q[nh*hd]`, `kraw[hd]`,
  `deck_new[hd]`, `o[nh*hd]`, MoE `gate_up`/`act`/`eo` buffers, `gating[E]`,
  `eids[topk]`.
- The KV cache `deck[layer]` promoted to a **fixed-capacity** device buffer
  `[max_ctx * hd]` per layer (capture against fixed base pointer; the append writes
  row `len`; replay updates `len` + the new token's row — NOT re-capture). This is
  how the dense/DFlash graphs handle the growing KV (paged/fixed-capacity, pointers
  fixed, contents change). Watch [[cudagraph-capture-bakes-stack-addresses]]:
  positions/token-id/len live in persistent device buffers written by an async
  copy on the stream BEFORE replay, never function-local temporaries.

## 3. Bricks

### Brick 0 — SCOPE (this doc). Gate: committed + reviewed.

### Brick A — DEVICE MLA ATTENTION KERNEL ← implement next
Replace the host QK/softmax-sink/AV Dot loop (`:736-756`) with a real parallel
device kernel reading the unified KV cache. Math (per query t, head h; num KV
heads = 1, shared latent): `score_s = (q_h · kv[s]) * scale` for `s ∈ [0,
kv_base+t]`; sink softmax `m=max(sink_h, max_s score)`, `denom=exp(sink_h-m)+
Σexp(score_s-m)`, `o_h[d]=Σ (exp(score_s-m)/denom)·kv[s][d]`
(`SoftmaxWithSink`, `deepseek_v4_dsa.cpp:116`). Bit-identity is ACHIEVABLE by
preserving host accumulation order (per-key dot sequential over d; denom sequential
over s incl. sink; AV sequential over s) — no reduction reorder. Exposed as a
direct unified-pointer launch on the queue stream (no upload/download — unlike the
#183 `<<<1,1>>>` glue kernels; this is the first REAL device V4 kernel). Flag
`VT_V4_DEVICE_ATTN` (default OFF). GATE: correctness + build (see §4); Brick A alone
likely does NOT speed decode (surrounding glue still syncs) — measure + report the
real number anyway.

### Brick B — DEVICE GLUE KERNELS (after review)
Real parallel device kernels on resident DBufs for: per-head q-RMSNorm, dual-RoPE
(+ inverse-RoPE), kv-RMSNorm, final RMSNorm; MHC pre/post + hc_head + the 20-iter
Sinkhorn; the MoE router (sqrtsoftplus + top-k/hash); clamped-SwiGLU; expert
combine. Reuse the #183 kernels' MATH but rewrite them as real parallel kernels
that read/write persistent DBufs with NO upload/download/sync (the #183 versions
Upload/Download+`cudaStreamSynchronize` per call — unusable for capture). **HONEST
RISK — the 20-iter Sinkhorn** (`MhcSinkhorn`, hc=4): a 4×4 doubly-stochastic
iteration, tiny (hc²=16); a parallel kernel is trivial in size but 20 sequential
iters. It's small per token, fine as one kernel (16 threads, 20 iters in-kernel) —
no host round-trip needed. If any glue op proves impractical as a good parallel
kernel, the FALLBACK is a single-thread device kernel (correct, graph-capturable —
correctness+capturability matter more than that op's own speed, since it's tiny).

### Brick C — ASSEMBLE DEVICE-RESIDENT DECODE (after review)
A `DeepseekV4DeviceDecode` path (T=1) composing Bricks A+B on persistent DBufs with
NO per-op sync — the analog of `ForwardLayers`. `EmbedInto`-style embed outside;
one drain at the end before the host argmax. Gate: token-identical / near-tie.

### Brick D — CAPTURE THE DECODE CUDA GRAPH (the SPEED gate, after review)
`BeginCapture`/`EndCapture`/`Replay` (`cuda_backend.cu:186-201`) over the Brick-C
step; per-step token-id/position/len written to persistent DBufs by an async copy
before `Replay`. Gate: replay == eager TOKEN-IDENTICAL; then the decode benchmark.

## 4. Per-brick gates

| Brick | Correctness | Build | Speed |
|---|---|---|---|
| A | device-attn == host attn (unit: `test_cuda_deepseek_v4` device-vs-host-ref, RED-first; real-model token-identical ON vs OFF) | CUDA + CPU `-Werror`; `test_deepseek_v4_gguf_load` 12/12 | measure (expected flat/regressed — reported honestly) |
| B | each glue kernel == host ref (unit) | same | flat (still host-synced) |
| C | device-resident decode == host decode token-identical/near-tie | same | may improve (fewer syncs) |
| D | graph replay == eager token-identical | same | **the payoff — decode tok/s vs ds4** |

## 5. Honest projected ceiling

The profiler put the host-orchestrated decode step at ~0.17 s = GEMM-dispatch
(~0.071s, ~18 launches/layer × 43) + host-glue + per-op sync, GB10 util ~36%. A
captured graph collapses ALL launches into ONE `cudaGraphLaunch` (the dense-decode
graph measured host tax ~5→2.7 ms/step, ~86%→~92% GPU-busy, `docs/BENCHMARKS.md`
Qwen3-Coder W7). If the launch/sync tax (~55-60% of our step) largely vanishes,
decode could reach **~10-13 tok/s** — a real 2-2.5× over 5.83, but **honestly
likely SHORT of ds4's 16.5**: ds4 also has (a) fp8 KV (smaller cache reads), (b)
hand-fused kernels, (c) a native fp4/int-quant expert GEMM tuned for GB10, whereas
our expert GEMM is the keep-quant MMVQ path and our KV is f32. So the realistic
ceiling of THIS campaign (graph over the current kernels) is ~60-80% of ds4;
closing the last gap needs kernel-efficiency work (fp8 KV cache, a tuned grouped
MMQ) that is BEYOND the decode-graph and is a NAMED follow-on, not this campaign.
Stated up front so the final number isn't a surprise.

## 6. Risks / decisions

- **R1 (bit-identity):** Brick A preserves host accumulation order → aim
  bit-identical; if a later brick's reduction reorders (Sinkhorn/RMSNorm), gate
  distributionally + CHARACTERIZE. DECIDED: never silent.
- **R2 (capture hazard):** [[cudagraph-capture-bakes-stack-addresses]] — all
  per-step inputs in persistent DBufs, written by async stream copies pre-replay;
  verify by TOKENS (compute-sanitizer clean ≠ capture-safe).
- **R3 (fixed-capacity KV):** capture against a `max_ctx`-sized KV buffer; assert
  `len < max_ctx`; long-context is a named residual.
- **R4 (rollback):** every brick behind a flag; the host path stays DEFAULT until
  Brick D proves the graph token-identical.
- **R5 (ceiling):** §5 — this campaign targets the launch tax; kernel-efficiency
  (fp8 KV, tuned MMQ) is a separate named follow-on.
