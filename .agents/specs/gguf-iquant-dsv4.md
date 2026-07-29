# GGUF i-quant (IQ2_XXS) + Q2_K dequant + DeepSeek-V4 GGUF loadability

**Claim:** `CLAIM-DSV4-GGUF-LOADER` (dequant); keep-quant compute + `QUANT-GGUF-IQ3_XXS`
advanced by `CLAIM-DEEPSEEK-V4-W8`. **Rows:** `QUANT-GGUF-IQ2_XXS` (id 16),
`QUANT-GGUF-IQ3_XXS` (id 18), `QUANT-GGUF-Q2_K` (id 10) in
[quantization-matrix.md](../quantization-matrix.md);
cross-references `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` (stays `SPIKE`,
owned by the DeepSeek-V4 impl/forward lane — this claim owns only the GGUF/quant
path, NOT the forward). **Base:** `main` HEAD `4d1be010`.

**Driver:** make the single-Spark-fitting `unsloth/DeepSeek-V4-Flash-GGUF
UD-IQ2_XXS` (~91 GB, the only DeepSeek-V4 build that fits ONE GB10's 119 GiB
unified pool — the NVFP4/fp8 builds are 156.7/167 GiB → 2 Sparks) LOADABLE by our
engine. This is the vehicle for the user's DeepSeek-V4 real-GPU benchmark. Two
independent blockers existed (spike `CLAIM-DSV4-GGUF-SPIKE`, see
[deepseek-v4-flash.md](deepseek-v4-flash.md) § "GGUF benchmark loadability"): (1)
our GGUF dequant lacked IQ2_XXS and Q2_K; (2) the V4 registry hard-rejects GGUF
and there is no V4-GGUF name map. This brick clears **(1)**; **(2)** is scoped +
partly ground-verified here but remains blocked on the tensor manifest (below).

---

## Scope

**IN (W1, landed):** portable CPU dequant of the two ~2-bit GGUF encodings the
DeepSeek-V4 vehicles use — **IQ2_XXS (ggml id 16)** and **Q2_K (ggml id 10)** —
to f32/bf16, ported 1:1 from llama.cpp `ggml-quants.c` @ `237ad9b96` (the same
pin the rest of our GGUF path cites). Registered as vt block dtypes
(`kIQ2_XXS`/`kQ2_K`) + reader trait (id 16; id 10 already had one) + routed
through the loader's `DequantGgufRowToF32/Bf16`.

**OUT (this claim):** the DeepSeek-V4 FORWARD (MHC, DSA indexer/compressor,
512-wide MLA, sqrtsoftplus/hash MoE) — that is the W3-W8 forward lane
(`CLAIM-DEEPSEEK-V4-*`). This claim does NOT edit the forward TUs. Compute-in-quant
(`vec_dot`) GEMM for IQ2_XXS/Q2_K is also OUT: both are dequant-only here (no
`vec_dot`), so `HasQuantDotKernel` is false and the loader routes them to
expand-to-bf16, never keep-quant GEMM. A `vec_dot` for these is a future perf leaf.

**Dispatch contract:** a block dtype with a `to_float` decode but NO `vec_dot`
row is automatically NOT keep-quant capable —
[`KeepQuantDType`](../../src/vllm/model_executor/model_loader/gguf_keep_quant.cpp#L106)
gates on `vt::cpu::HasQuantDotKernel`, and
[`FindQuantTraits`](../../src/vt/cpu/cpu_quant_traits.cpp#L29) returns `nullptr`
for these two (its `default:`), so they cleanly route to `kExpandBf16`. This is
identical to how `Q8_K` (activation-only, no file weight) is handled, so no new
routing code is needed.

## Upstream chain (llama.cpp @ `237ad9b96`, cite file:line)

- **Q2_K block** `ggml/src/ggml-common.h:288-299` `block_q2_K = { u8 scales[16];
  u8 qs[64]; f16 d; f16 dmin; }` = 84 B, 256 elems. Decode
  `ggml/src/ggml-quants.c:903` `dequantize_row_q2_K`: 2-bit code (`qs`, shift
  0/2/4/6) × per-16 4-bit sub-scale (low nibble of `scales[]`) − 4-bit sub-min
  (high nibble), scaled by f16 `d`/`dmin`.
- **IQ2_XXS block** `ggml-common.h:371-374` `block_iq2_xxs = { f16 d; u16 qs[32]; }`
  = 66 B, 256 elems. Decode `ggml-quants.c:2416` `dequantize_row_iq2_xxs`:
  codebook — each 32-elem sub-block reads two u32 from `qs` (`aux32[0]` = four
  8-bit grid indices, `aux32[1]` = four 7-bit sign selectors + a 4-bit block
  scale in bits 28-31, `db = d·(0.5 + (aux32[1]>>28))·0.25`); the 8 grid bytes
  come from `iq2xxs_grid` (`ggml-common.h:550`, 256 u64 codebook) and are
  sign-flipped via `ksigns_iq2xs[selector]` (`:503`) & `kmask_iq2xs[j]` (`:499`).

## Our baseline + reuse

- **Block dequant layer** [`cpu_quant_dequant.cpp`](../../src/vt/cpu/cpu_quant_dequant.cpp)
  `BlockToFloat(dtype)` — already hosts the Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K
  `dequantize_row_*` decoders. IQ2_XXS/Q2_K land as two more decoders + their
  static grid/sign tables; wired into the same `switch`.
- **vt dtype system** [`dtype.{h,cpp}`](../../src/vt/dtype.cpp) — `kQ2_K`/`kIQ2_XXS`
  join the block-dtype enum, geometry table, `BlockDTypeFromGgmlTypeId`, `SizeOf`
  (block-quant throw group), `Name`; [`ops.cpp`](../../src/vt/ops.cpp)
  `ToScalarType` block group. All exhaustive `-Wswitch` sites updated; every
  vec_dot/traits site has a `default:` (verified).
- **GGUF reader** [`gguf_reader.cpp`](../../src/vllm/model_executor/model_loader/gguf_reader.cpp)
  `FindGgmlTraits` — id 10 (Q2_K, 84 B) already present; id 16 (IQ2_XXS, 66 B)
  added.
- **Loader façade** [`gguf_dequant.cpp`](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp)
  `DequantGgufRowToF32` — ids 10 + 16 join the block-path `case` group (routes to
  `vt::cpu::BlockToFloat`).

## Port map (upstream → ours)

| Upstream (`237ad9b96`) | Ours |
|---|---|
| `ggml-quants.c:903` `dequantize_row_q2_K` | `cpu_quant_dequant.cpp` `DequantQ2_K` |
| `ggml-quants.c:2416` `dequantize_row_iq2_xxs` | `cpu_quant_dequant.cpp` `DequantIQ2_XXS` |
| `ggml-common.h:550` `iq2xxs_grid` (256 u64) | `cpu_quant_dequant.cpp` `kIq2xxsGrid[256]` |
| `ggml-common.h:503` `ksigns_iq2xs` (128 u8) | `cpu_quant_dequant.cpp` `kKsignsIq2xs[128]` |
| `ggml-common.h:499` `kmask_iq2xs` (8 u8) | `cpu_quant_dequant.cpp` `kKmaskIq2xs[8]` |

Recorded in [porting-inventory.md §9](../porting-inventory.md) as a
from-necessity port (llama.cpp is the reference impl for GGUF i-quants; pinned
vLLM has no in-tree GGUF at all — it migrated to the OOT `vllm-gguf-plugin`).

## Tests to port (RED-first, hand-derived over known packed bytes)

- [`test_gguf_dequant.cpp`](../../tests/vllm/test_gguf_dequant.cpp): Q2_K case
  (`:228`) — literal nibble sub-scale/min values (`5.75`, `-0.25`, `2.5`, `0.25`)
  computed by hand from the algorithm; IQ2_XXS case (`:265`) — hand-verified
  against the codebook: `grid[0]`=all-8, `grid[1]` byte0=`0x2b`(43), `ksigns[1]`=129
  flips lanes j=0,7, 4-bit scale (`db`=0.125 vs 0.375), multi-sub-block striding,
  plus the bf16 round-trip. A wrong grid index, sign mask, block stride, or scale
  formula flips these — they are RED-first, not consistency checks.
- [`test_ops_quant_traits.cpp`](../../tests/vt/test_ops_quant_traits.cpp) `:143`:
  the dequant-only contract — geometry agrees vt↔reader, `to_float` present,
  `HasQuantDotKernel` FALSE, `QuantTraits` throws (no vec_dot row). This pins that
  the two types route to expand-bf16, never keep-quant GEMM.

Upstream `ggml-quants.c` has no standalone unit test for these decoders (llama.cpp
gates them only through end-to-end model runs); the hand-derived literals are the
executable spec, mirroring the existing k-quant cases in the same file.

## Gates

- **W1 correctness (LANDED, PASS):** `test_gguf_dequant` 15/15 · 480 assertions +
  `test_ops_quant_traits` 9/9 · 5643 assertions, GREEN on a CPU build. RED-first by
  construction — the literals are computed from the algorithm spec independent of the
  code (a wrong grid index, sign mask, block stride, or scale formula flips them).
- **W1 build-verify (PASS):** all 7 changed TUs compile clean under full `-Werror`
  (0 warnings, verbatim `compile_commands.json`). The pre-existing GCC-13
  `-Werror=array-bounds` false positive in `voxtral.cpp` (unrelated to this diff —
  proven pre-existing: fails identically at base HEAD with this diff's `dtype.h`
  reverted) is neutralized ONLY to link the runnable test binaries.
- **Speed:** NOT APPLICABLE — a CPU dequant primitive owes no throughput number;
  a `vec_dot` keep-quant GEMM (perf) is a future leaf.
- **W2 loader-accounts-for-tensors (BLOCKED):** the "100% of the 1328 tensors mapped,
  no unmapped/leftover" gate needs the tensor manifest (§ DeepSeek-V4 GGUF loadability)
  — DERIVED (config KV verified), not run.
- **Eventual V4-GGUF e2e:** vs llama.cpp-on-card (the pinned vLLM cannot load V4 from
  GGUF), once the W2 loader + the W3-W8 forward land.

## DeepSeek-V4 GGUF loadability (W2) — DERIVED, NOT a run

**Verified read-only (HTTP-range, NO 90 GB download)** of
`unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS-00001-of-00003.gguf` header (the CDN
capped the range at ~5 MB, which covers the 72-KV metadata block but NOT the
tensor-info table):

- `general.architecture = **deepseek4**` (the GGUF arch string; our registry key
  is `DeepseekV4ForCausalLM`, vLLM `deepseek_v4` — a V4-GGUF loader must map
  `deepseek4` → our V4 arch).
- `general.file_type = **19**` = `LLAMA_FTYPE_MOSTLY_IQ2_XXS` → **confirms this
  brick's IQ2_XXS dequant targets exactly the vehicle's dominant weight type.**
- `split.count = 3`, `split.tensors.count = **1328**` (total tensors across shards).
- Full V4 config maps to `deepseek4.*` KV keys (all consistent with the verified
  safetensors arch scalars): `block_count=43`, `attention.{head_count=64,
  head_count_kv=1, key_length=512, value_length=512, q_lora_rank=1024,
  output_lora_rank=1024, output_group_count=8, sliding_window=128,
  compress_rope_freq_base=160000, indexer.{head_count=64, key_length=128,
  top_k=512}}`, `attention.compress_ratios=[0,0,4,128,…]`, `expert_count=256`,
  `expert_used_count=6`, `expert_shared_count=1`, `expert_feed_forward_length=2048`,
  `expert_gating_func=4` (sqrtsoftplus), `hash_layer_count=3`,
  `hyper_connection.{count=4, sinkhorn_iterations=20, epsilon=1e-6}`,
  `swiglu_clamp_exp/shexp=[10.0,…]`, `rope.{dimension_count=64, freq_base=10000,
  scaling.{type=yarn, factor=16, original_context_length=65536}}`,
  `tokenizer.ggml.model=gpt2`.

**Verdict (honest 3-state):** the config half of a V4-GGUF loader is now
**ground-verified** (arch string + every param KV). The tensor **name map** — the
1328-tensor `blk.N.*` manifest needed for the "100% accounted, no unmapped/leftover"
gate — is **NOT obtainable in this lane**: the tensor-info table is beyond the CDN
range cap, the file is NOT cached on `dgx.casa` (only `config.json`), and the 90 GB
download is prohibited. Lifting the registry GGUF reject onto a GUESSED name map
would be a speculative half-loader routing into a W4-owned stub forward, so the
reject **stays in place**. W2 is therefore **DERIVED (config) + BLOCKED (manifest)**,
not landed.

## Work breakdown (W-plan)

- **W1 — landed (this):** IQ2_XXS + Q2_K dequant to f32/bf16, vt/reader/loader
  registration, RED-first unit gates, clean `-Werror` build-verify of the changed
  TUs. DONE.
- **W2 — V4-GGUF loader:** blocked on the tensor manifest (§6). Next step when a
  manifest is available (range-read the tensor table past the CDN cap via a
  multi-range fetch, or read the header of a cached shard on a box with disk):
  map `deepseek4` arch + the `blk.N.*` tensor names → our V4 layout, lift the
  reject, run the accounting pass. The config-KV → `DeepseekV4Params` mapping in
  §6 is the verified input.
- **W3-W8 — V4 forward:** the parallel forward lane (MHC, DSA, 512-wide MLA,
  sqrtsoftplus/hash MoE). A V4-GGUF run also needs a correctness reference: the
  pinned vLLM oracle cannot load V4 from GGUF (V4 has no `packed_modules_mapping`
  in the OOT plugin, and GGUF is OOT in 0.26), so the GGUF vehicle's reference is
  **llama.cpp-on-card** (Unsloth publishes these FOR llama.cpp), a
  derive-and-ship correctness reference, not a vLLM oracle gate.

## Dependencies

- **W1 (this brick):** none beyond the landed GGUF dequant layer + vt block-dtype
  system (all present). No GPU, no download — the unit gate uses known packed bytes.
- **W2 (V4-GGUF loader):** the 1328-tensor `blk.N.*` manifest (a multi-range header
  fetch past the CDN cap, or a cached shard on a disk-freed box) + the shared
  `sqrtsoftplus`/MLA-geometry primitives (coordinate with `CLAIM-DEEPSEEK-V4-*`
  before wiring). Does NOT depend on this claim editing any forward TU.
- **A runnable V4-GGUF vehicle:** the V4 forward (W3-W8, multi-Spark on one GB10 is
  not needed — IQ2_XXS is ~91 GB and FITS, so a single-Spark run is memory-feasible
  once the forward + loader land) + a correctness reference (llama.cpp-on-card, since
  the pinned vLLM cannot load V4 from GGUF).

## Risks/decisions and residuals

- **What still blocks an actual V4-GGUF RUN:** the entire V4 forward (W3-W8) +
  the V4-GGUF name map (W2, manifest-blocked). This brick makes the QUANT-TYPE
  half loadable; the MODEL-ARCH half is untouched.
- **vec_dot for IQ2_XXS/Q2_K** (compute-in-quant GEMM) — a future perf leaf; today
  both expand to bf16.
- The IQ2_XXS `db` scale, sign selector, and grid layout are pinned by the
  hand-derived unit case, but a REAL-checkpoint byte-level gate (like the NVFP4
  cross-format gate) awaits a manifest + slice of real IQ2_XXS bytes.
