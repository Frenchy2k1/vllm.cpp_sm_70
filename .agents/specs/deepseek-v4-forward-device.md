# DeepSeek-V4-Flash — ForwardDevice parity campaign (host-orchestrated stateless → ds4-class)

Owner claim: `CLAIM-DEEPSEEK-V4-FORWARD-DEVICE` (under `CLAIM-DEEPSEEK-V4-IMPL`).
Base: `main` `fd9e191c` (the `--gpu` keep-quant GEMM route merged). Row stays
`ACTIVE` (this is a speed campaign on an already-coherent model). NOT pushed.

## 0. Goal + anchor

Take `DeepseekV4ForwardGguf` from a **host-orchestrated stateless full-recompute**
(every decode token re-runs the whole 43-layer forward over the growing context)
to **ds4-class** throughput on ONE GB10.

Measured starting point (fd9e191c, real 80.7 GB ds4 keep-quant file, DGX GB10,
prompt "The capital of France is"):

| Engine | prefill tok/s | decode tok/s | peak GiB |
|---|---|---|---|
| ds4 (GPU, KV-cache incremental decode) | 325.9 (512 ctx) | 16.3–16.6 | 80.8 |
| ours CPU (stateless full-recompute) | 4.48 (5 ctx) | 0.51 | 85.96 |
| ours `--gpu` (experts on GB10, stateless) | 6.26 (5 ctx) | 0.68 | 86.33 |

**CORRECTNESS ANCHOR (never regress).** Today CPU and `--gpu` are byte-identical
and coherent: "The capital of France is" → generated ids `11111 16 455 6102 294
8760 344 11111` = " Paris. The capital of France is Paris". Every stage stays
token-identical to that, or (only if a device-resident fp path introduces bf16
associativity differences) states it and gates distributionally like the other
near-tie models. `test_deepseek_v4_gguf_load` **10/10·403** must stay green.

## 1. Why the gap (source-grounded)

The gap to ds4 is NOT "experts on CPU" (fixed at fd9e191c — the block-quant
expert/MLA/lm_head GEMMs run on the GB10 kMatmulBTQuant provider, nvidia-smi
38–50% util). Two independent causes remain:

1. **No KV cache → decode is repeated prefill.** `ForwardComposeImpl`
   (`deepseek_v4.cpp:788`) is stateless: the driver
   (`examples/deepseek_v4_gen/main.cpp:149`) re-passes ALL tokens each step, so
   `AttentionBlock` (`deepseek_v4.cpp:428`) recomputes every token's KV and runs
   the full stack over the growing context — O(ctx) work per decode token. ds4
   keeps a per-layer KV cache (`ds4.c:12089 ds4_layer_cache.raw_kv`,
   `:12351 kv_cache_push_raw`) and does O(1) incremental decode.
2. **Host-orchestration + per-GEMM sync.** With `device=false` the MHC/DSA/
   compressor/MoE-glue run on the host between GPU GEMMs, each GEMM drains the
   stream (`SyncDeviceGemm`, `deepseek_v4.cpp`), and at tiny batch (T=1) the GEMMs
   are latency-bound — GB10 sits at ~45% util, weights read from system memory in
   place (no HBM-resident, no kernel overlap/graphs).

## 2. KV-cache shape (the Stage-1 object), grounded in our + ds4 source

For the REAL keep-quant single-Spark run, `AttentionBlock` runs **dense MLA**:
`dsa_dense = (be.gguf != nullptr)` is true, so `is_comp` and `is_indexer` are both
FALSE (`deepseek_v4.cpp:457-459`). Therefore, per layer:

- `deck[t] = latent[t] = kraw[t] = rope(kv_norm(wkv · x[t]))`
  (`deepseek_v4.cpp:494-499,504,542-543`) — a single **`[hd]`** latent per token
  (num_key_value_heads=1 MLA; `hd = p.head_dim` = 576 = nope 512 + rope 64;
  ds4 `DS4_N_HEAD_DIM=576`). It depends ONLY on `x[t]` (the post-MHC-pre attn
  input) and `positions[t]` (the rope), NEITHER of which changes across decode
  steps.
- Attention (`deepseek_v4.cpp:594-614`): query `t` attends over selected keys
  `S = {0..t}` (dense causal), with **key = value = `deck[s]`** and per-head sink
  softmax; then inverse-RoPE on the output (`:621-625`) and grouped o-LoRA.

So the cache is exactly **per-layer `deck` `[n_ctx × hd]` f32** — the mirror of
ds4's `raw_kv[cap_raw × DS4_N_HEAD_DIM]` (`ds4.c:12305`), appended one row/token
(`kv_cache_push_raw`), sliding-window (SWA cap `DS4_N_SWA`) when full. ds4 stores
raw_kv at fp16 (`f16_to_f32(f32_to_f16(...))`); **we cache f32 to stay bit-exact
to our own full-recompute anchor** (matching ds4's fp16 KV is a separate fidelity
question, not the equivalence gate). For seq ≤ 512 (our regime; ds4 `index_topk`)
no sliding/compression is hit — a plain growing buffer is exact. Long-context SWA
+ the compressed `attn_comp_kv`/indexer cache (`ds4.c:12096-12108`) is a NAMED
residual.

**No other cross-token/cross-step state.** The MHC residual manifold
(`residual[T,hc,H]`, `post_mix`, `res_mix`, `deepseek_v4.cpp:830-834,851-901`) is
per-token and intra-forward (first layer broadcast-expands, later layers fuse
MhcPost+MhcPre) with NO cross-token mixing — each token's stream is built fresh as
it flows down the layers. MoE hash routing keys on `token_ids[t]`
(`MoeBlock`, `:909`). So a T=1 decode token carries no state but its own; only the
attention needs prior tokens' cached `deck`.

## 3. What already exists for stages 2–3 (#183 device kernels)

`DeepseekV4Model::ForwardDevice` (`deepseek_v4.cpp:1152`) runs the SAME
`ForwardComposeImpl` with `device=true`, so the `Disp*` shims
(`deepseek_v4.cpp:140-224`) dispatch to the landed W7-device CUDA kernels
(`include/vllm/model_executor/models/deepseek_v4_device.h`,
`src/vt/cuda/cuda_deepseek_v4.cu`, DGX-gated `test_cuda_deepseek_v4` 11/11·153):

- `MhcDeviceKernels` — pre / post / head (`deepseek_v4_device.h:48`).
- `DsaDeviceKernels` — weight_fold / logits / topk / softmax_sink / grouped_olora
  (`:69`).
- `CompressorDeviceKernels` — save_score_ape / pool_norm / encode / decode (`:93`).
- `MoeDeviceKernels` — route / clamped_swiglu (`:111`).

LIMITATION (the stage-2 target): each kernel's signature is
`std::vector<float>(vt::Queue&, const std::vector<float>&, …)` — it takes and
returns HOST vectors, so today each glue op is a separate host→device→host
round-trip. Stage 2 keeps activations resident on-device across the stack; stage 3
adds overlap/graphs.

## 4. Stage plan (each: correctness gate BEFORE speed; report; await review)

### Stage 0 — SCOPE (this document). Gate: committed + reviewed.

### Stage 1 — KV CACHE (the single biggest lever). ← implement next
Add the MLA compressed-latent KV cache; switch DECODE from full-recompute to
incremental. Prefill fills the cache (all prompt tokens → their `deck`); each
decode step processes ONE new token against the cached KV.

- NEW `DeepseekV4KvCache` (per-layer `std::vector<float> deck` + `int64_t len`),
  owned by the driver, threaded into a NEW cached forward entry.
- `AttentionBlock` becomes cache-aware: compute `q`/`deck` for the T NEW tokens,
  APPEND their `deck` to `cache.layer[layer]`, and attend each new query (global
  pos `g = base + t`) over `cache.layer[layer][0 .. g]` (all cached decks incl.
  the new ones). `base = 0` for prefill; `base = len` for decode.
- NEW `DeepseekV4ForwardGgufCached(weights, queue, new_token_ids, positions,
  cache, logits_indices)`; driver `--kv-cache` mode: step 0 = prefill (all prompt,
  positions 0..n-1), steps 1.. = decode (1 token, position = len). Rollback-able:
  default off; `--gpu` unaffected.
- **Files:** `deepseek_v4.cpp` (cache struct + cache-aware `AttentionBlock` +
  `ForwardComposeImplCached`/entry), `deepseek_v4.h` (decls),
  `examples/deepseek_v4_gen/main.cpp` (`--kv-cache` loop),
  `tests/vllm/models/test_deepseek_v4_gguf_load.cpp` (equivalence gate),
  records.
- **CORRECTNESS GATE (pure equivalence — same tokens, ~ctx× fewer FLOPs):**
  incremental decode token-IDENTICAL to the full-recompute path, on the tiny
  synthetic model (doctest) AND on the real 80.7 GB model (DGX greedy, same
  prompt → same `11111 16 455 6102 294 8760 344 11111`). RED-first: a deliberate
  off-by-one in the cache base flips tokens.
- **BENCHMARK GATE:** re-run the DGX ours-vs-ds4 table; decode should move from
  0.68 toward ds4's ~16 (decode stops being repeated prefill). Capture nvidia-smi
  DURING. Report, STOP.

### Stage 2 — DEVICE-RESIDENT ACTIVATIONS (after review).
Keep activations on-device across the layer stack: run MHC/DSA/compressor/MoE-glue
via the #183 kernels with resident buffers (drop the per-op host round-trip and
the per-GEMM `SyncDeviceGemm`). Gate: token-identical or ratified near-tie
(state + gate distributionally if bf16 associativity diverges). Benchmark.

### Stage 3 — OVERLAP / CUDA GRAPHS (after review).
Close the residual: capture the per-token decode step as a CUDA graph / overlap
host glue with device GEMMs. Gate: unchanged tokens. Benchmark to ds4 parity.

## 5. Risks / decisions

- **R1 (Stage 1 exactness):** cache f32 `deck` (not ds4's fp16) so incremental ==
  our full-recompute bit-for-bit — the anchor is OUR path, not ds4. DECIDED.
- **R2 (compressor/indexer layers):** inert in the real dense run
  (`is_comp/is_indexer` false); the sliding-window + compressed/indexer caches are
  a long-context (>512) NAMED residual, not on the Stage-1 path.
- **R3 (SWA sliding):** not hit for seq ≤ 512; a growing buffer is exact. Long
  context needs ds4's `kv_cache_push_raw` slide + the compressed caches — residual.
- **R4 (Stage 2 numerics):** device-resident fp reductions may diverge from host
  at bf16 near-ties → gate distributionally (near-tie methodology) if so, and say
  it; do not silently accept drift.
- **R5 (OOM):** ONE engine resident at a time under `flock $HOME/gpu.lock`;
  `docker stop local-ai-worker` for the run; the cache adds only n_ctx×hd×43×4 B
  (≈ 0.1 GiB at 1k ctx) — negligible vs the 119 GiB pool.

## 6. Gates summary

| Stage | Correctness gate | Speed gate |
|---|---|---|
| 1 | incremental == full-recompute (doctest + real greedy token-identical) | decode 0.68 → toward ~16 |
| 2 | token-identical or ratified near-tie (stated) | GB10 util ↑, prefill/decode ↑ |
| 3 | tokens unchanged | approach ds4 326 / 16.5 |
