# Spec: DFlash speculative decoding (task #51, from B5 scoping)

Block-diffusion drafting — the headline speculative method after MTP. Derived from
[.agents/specs/spec-decode-scoping-2026-07-10.md](spec-decode-scoping-2026-07-10.md);
re-grounded against the pin (`/home/mudler/_git/vllm` @ `e24d1b24`) and, for the
reuse map, **against the NOW-LANDED MTP machinery on `origin/main` `72f9fb1`**
(the MTP campaign closed I1..I7 — `SPEC-MTP` `ACTIVE`, single-request greedy
token-identical to vLLM + ~1.04x TPOT / +4% output-tput at c1, on-par-or-above at
c2/c4/c8). **Readiness re-assessment: 2026-07-25.** This is a READINESS ASSESSMENT
+ dispatch-sized W-plan; no DFlash code exists yet.

## 0. READINESS VERDICT

### D0 RESULT (2026-07-26, `CLAIM-DFLASH-D0D1`) — **UNBLOCKED on vLLM 0.26.0.dev0, GOLDEN CAPTURED. D1 LANDED.**

The pin advance `e24d1b24`/0.25.0 -> `555967922`/**vLLM 0.26.0.dev0+g5559679**
(`chore(pin)` `bc415a3e`) resolves upstream **vllm#40898** (mixed sliding/full
attention `layer_types` for DFlash drafts, under `VLLM_USE_V2_MODEL_RUNNER=1`).
The prior 0.25.0 ORACLE-BLOCKED verdict (draft construction aborted at
`qwen3_dflash.py _resolve_layer_attention`) is SUPERSEDED. Full verdict +
determinism + gate-form analysis: `tests/parity/goldens/dflash_27b/D0_VERDICT.md`.

- **Decisive check RAN** (dgx, `flock`, GPU sole-owner, `VLLM_USE_V2_MODEL_RUNNER=1`,
  no FLASH_ATTN pin): the mixed-attention z-lab 27B draft **CONSTRUCTS** and the
  propose/verify/rejection loop **RUNS**. The **DRAFTER IS ALIVE** — acceptance_len
  France 2.214 / fibonacci 8.800 / "17*23" 4.750 / three-laws 4.571 (drafts/accepted
  14·17, 5·39, 8·30, 7·25); all > 1 (dead-drafter trap cleared, content-dependence as
  §5). Backend auto-selects `flashinfer-native` fp8-KV sm121 (NOT FLASH_ATTN — the §0
  soft risk did not materialize). `num_speculative_tokens=16`; oracle log confirms
  `eagle3 aux layers (2,17,32,47,62)` = `target_layer_ids [1,16,31,46,61] + 1`.
- **MEMORY:** the first attempt (gpu_util 0.55, mm ON) OOM-**rebooted** dgx — the 27B
  is multimodal and profiles the vision encoder at max image size. Fix (now the
  capture-tool default): `limit_mm_per_prompt={image:0,video:0}` + gpu_util 0.30.
- **Gate FORM (measured):** vLLM-DFlash-ON greedy is **run-deterministic** (K>=3
  token+acceptance identical) BUT is **NOT** token-identical to vLLM-spec-OFF (3/4
  prompts differ — the k=16 block verify diverges from sequential decode at bf16
  near-ties). So the D5 gate is **STRICT + MODE-MATCHED** (`our-DFlash-ON ==
  vLLM-DFlash-ON`), NOT the MTP three-way identity; spec-OFF byte-identical SACRED is a
  separate inertness gate. (Supersedes the §6 "three-way identity" D5 row below.)
- **Goldens committed:** `tests/parity/goldens/dflash_27b/dflash_27b_spec_{on,off}.json`
  (tokens + acceptance); capture tool `scripts/spec/d0_dflash_oracle_capture.py`
  (extended: `disable_log_stats=False`, mm-off + gpu_util 0.30).

### D1 RESULT (2026-07-26, `CLAIM-DFLASH-D0D1`) — **DF-AUX-TAPS DONE.**

The landed single hidden tap (`hidden_tap`/`ForwardDeviceTap`) is generalized to the
DFlash multi-tap. New `Qwen3_5AuxTaps` carrier + `ModelForwardInput::aux_tap` out-field
route the Qwen3.5 dense/MoE forward to `Qwen3_5{,Dense}Model::ForwardDeviceMultiTap`,
which captures `(hidden + residual)` — the exact eagle3 `_maybe_add_hidden_state`
(`interfaces.py:1382`) value at aux key `L+1` — at each configured `target_layer_id L`
into a `[T, H×taps]` buffer (column order = ascending `layer_ids` = `cat(aux, dim=-1)`
fed to the DFlash `fc`, `qwen3_dflash.py:411-419`/`:750-770`). Config-gated: null
`aux_tap` -> byte-identical (the single-tap inertness discipline). Anchors:
`include/.../qwen3_5.h` (struct + MoE decl), `qwen3_5_dense.h` (dense decl),
`model_registry.h` (field), `src/.../qwen3_5.cpp` (`MaybeCaptureAuxTap` +
`ValidateAuxTapLayerIds` + both `ForwardDeviceMultiTap`), `qwen3_5_moe.cpp`/
`qwen3_5_dense.cpp` (routing). Unit gate: `test_qwen27_paged_forward` new case
(598 assertions) — each tap column == an INDEPENDENT truncated-model reference at that
layer; RED-first proven (reversed concat order -> 384 assertions fail). Inertness: the
tap struct is additive/config-gated (`git diff --stat` = new methods only), the 27B
MTP e2e + 27B text SACRED gates re-run byte-identical on the new oracle. CUDA `-Werror`
0 warnings; compute-sanitizer 0 on the multi-tap.

**Honest scope note (D1 reference):** the D1 unit gate validates the multi-tap capture
against an INDEPENDENT in-engine reference (a truncated-depth rebuild — layers > L cannot
affect the depth-L residual), not a direct vLLM aux-hidden dump. Our full 27B forward is
already bit-exact to vLLM (SACRED), and `(hidden+res)` is a deterministic function of the
bit-exact per-layer state, so the captured taps equal vLLM's aux within the bf16 envelope
transitively; the direct per-layer vLLM aux-dump parity folds into D2's drafter golden
(where the `fc(cat(aux))` output is checked end-to-end).

### D2 RESULT (2026-07-26, `CLAIM-DFLASH-D2`) — **DF-DRAFT-MODEL DONE: GPU promotion GREEN on dgx (all 4 gates pass), draft forward matches vLLM.**

**GPU PROMOTION COMPLETE (dgx GB10, CUDA 13.0, sm_121a, oracle vLLM 0.26.0.dev0+g5559679).**
The dev-box CPU session's 4 pending GPU gates all pass:
1. **CUDA `-Werror` build clean** — 0 warnings, 0 errors. The new
   `DFlashBlockAttentionKernelCuda` (`cuda_ops.cu:1300`, never before compiled)
   builds AS-WRITTEN under `-Werror=all-warnings`; NO algorithm change needed.
2. **New-primitive CUDA==CPU + compute-sanitizer 0** — added CUDA-vs-CPU parity
   cases to `test_ops_dflash_block_attn` covering all 5 semantic corners
   (non-causal, causal, block isolation, SWA window, GQA): 198412/198412
   assertions pass within the 1e-4 f32-online-softmax envelope; `compute-sanitizer
   --tool memcheck` = **0 errors** on the new kernel.
3. **Draft-forward parity vs the REAL vLLM DFlash draft** (the decisive gate) —
   `scripts/spec/d2_dflash_draft_ref.py` dumps, from the loaded
   `DFlashQwen3ForCausalLM` (reached via `collective_rpc` into the V2 worker's
   `model_runner.speculator.model`), vLLM's real `combine_hidden_states` + the
   context-free (1+k) block forward (per-layer + final hidden + top-k proposed
   ids) built from vLLM's OWN loaded submodules. `test_qwen3_dflash_draft_parity`
   loads the same z-lab draft + the target-SHARED embed/lm_head and gates: **fc
   rel-L2 0.46%**, per-layer hidden rel-L2 **0.44/1.00/1.27/1.30/0.65%**, final
   hidden **0.88%** (all ≪ the 5% bf16/f32-softmax envelope); proposed ids **11
   deterministic rows STRICT-matched vLLM's top-1 + 5 bf16-near-tie rows
   cluster-matched** (the 2 non-strict flips are within ONE bf16 ULP = 0.0625 at
   magnitude ~8 — a genuine vocab-head near-tie, per [[near-tie-distributional-gate]],
   NOT a root divergence). vLLM's `get_draft_attn_causal()` = `[T,T,T,T,F]`
   matches our `ResolveQwen3DFlashAttnModes` exactly. The RED proofs (causal-flip,
   reversed-tap) stay in the deterministic CPU unit test.
4. **Inertness re-runs byte-identical** — 27B text SACRED `test_qwen27_paged_engine`
   **235/235** (16/16 token-exact vs vLLM) + 27B MTP `test_qwen27_spec_decode`
   **9/9**, both unchanged (the new op is separate).

**Loader fix (found on-box):** the z-lab draft ckpt ships only 58 tensors
(fc/hidden_norm/norm + 5 layers) — it OMITS embed_tokens + lm_head, which the draft
SHARES from the target (mirrors vLLM's `skip_substrs.append("embed_tokens")`). The
loader now tolerates their absence (`TryLoadBf16`); the parity harness supplies the
target's bf16 embed/lm_head, exactly as vLLM does.

Below = the pre-promotion record (retained for context).

### D2 RESULT (2026-07-26, `CLAIM-DFLASH-D2`) — **DF-DRAFT-MODEL code LANDED + CPU-GATED; GPU promotion gate PENDING on dgx.**

The biggest new DFlash brick is implemented and CPU-verified. Three genuinely-new
pieces (§1.3) landed:

- **The NEW non-causal in-block attention primitive** (kernel-matrix row
  `KERNEL-ATTN-DFLASH-BLOCK`) — a SEPARATE `vt::` op
  `vt::DFlashBlockAttention` (`OpId::kDFlashBlockAttention`, `DFlashBlockAttentionArgs`),
  the project's FIRST bidirectional attention. Per-request uniform (1+k) query block
  attends within its own block: FULL layers BIDIRECTIONAL (`causal=false`, no mask),
  SWA layers causal-within-window; f32 online softmax, GQA broadcast. CPU
  `DFlashBlockAttentionKernel` (`cpu_ops.cpp`) is the authoritative three-pass
  reference; CUDA `DFlashBlockAttentionKernel` (`cuda_ops.cu`) mirrors the causal
  `AttentionKernel` block-reduction recurrence with per-block bounds + the
  bidirectional/window mask. Grounded in `_resolve_layer_attention`
  (`qwen3_dflash.py:86-146`) + the flashinfer non-causal path (vllm#48167, in-pin).
  **The causal `kAttention`/`kPagedAttention` are UNTOUCHED** (a new op, not a
  modification) so every other model is byte-identical.
- **The `qwen3_dflash` draft model** (`include/.../qwen3_dflash.h` +
  `src/.../qwen3_dflash.cpp`): a plain 5-layer Qwen3-dense decoder (merged qkv,
  per-head q/k RMSNorm, NeoX RoPE theta=1e7, SwiGLU, standard add+RMSNorm) reusing the
  landed `dense_attn_block.h` block ops, with attention routed through the new
  primitive per the config-resolved per-layer mode (`ResolveQwen3DFlashAttnModes`
  mirrors `_resolve_layer_attention`). NO GDN/MoE/gate.
- **The `fc` aux-combine** (`CombineAuxFeatures`, Linear H*taps->H over the D1
  `[T,H*taps]` multi-tap, `qwen3_dflash.py:411-419`/`:750-770`), **mask embedding**
  (`:352-361`/`:432-438`), and the **z-lab weight loader** (`qwen3_dflash_weights.cpp`,
  q/k/v->qkv + gate/up concat per the vLLM stacked mapper).

**CPU GATE GREEN (deterministic, RED-first):**
`tests/vt/test_ops_dflash_block_attn.cpp` 5 cases / 12 assertions — hand-checked
non-causal (query 0 sees the FUTURE key, value 1.538 vs causal 1.0), the **RED
causal-vs-non-causal separation** (the non-causal mask is load-bearing), per-request
`cu_seqlens` block isolation, SWA window bound, GQA. `tests/vllm/models/
test_qwen3_dflash_forward.cpp` 5 cases / 95 assertions — the context-free block
forward runs + finite; the **RED full-layer-causal-flip** shifts the logits
(~1e-2, byte-identical rerun = 0); block isolation end-to-end; the fc combine
matches an independent reference + **RED reversed tap order**; the 4xSWA+1xfull
attn-mode resolution. **INERTNESS at the behavior level:** existing causal
`test_ops_attention` 9/9·23 and `test_qwen3_forward` 5/1028 UNCHANGED (the new op
is separate).

**GPU PROMOTION GATE STILL PENDING (dgx — no GPU/nvcc on the dev box this session):**
(1) draft-forward parity vs a dumped vLLM DFlash-draft reference (empty-context block
forward: fc output + per-layer hidden + logits) via `scripts/spec/d2_dflash_draft_ref.py`
(mm-limited + gpu_util 0.30); (2) CUDA `-Werror` build + `compute-sanitizer` 0 on the
new kernel; (3) 27B text SACRED 235/235 + 27B MTP 9/9 re-runs (inertness) on the new
oracle; (4) the loader's exact on-disk key spelling confirmed against the checkpoint
key dump. D2 is DONE only when (1)-(3) pass.

### D3 RESULT (2026-07-26, `CLAIM-DFLASH-D3`) — **DF-DRAFT-KV-PREP code LANDED + CPU-GATED; GPU numeric-parity PENDING on dgx.**

The two genuinely-new D3 pieces (§1.3) are implemented ADDITIVE and CPU-gated. The diff is
`git diff --stat` = `qwen3_dflash.h` +92, `qwen3_dflash.cpp` +393, `tests/CMakeLists.txt` +2,
new `tests/vllm/v1/spec_decode/test_dflash_kvprep.cpp` + `scripts/spec/d3_dflash_kvprep_ref.py`
— **NO CUDA, NO edit to any shared causal-path / scheduler / `vt` op** (so the D2/MTP/SACRED
gates are byte-identical by construction, not just re-measured).

- **`PrepareDflashInputs`** (`qwen3_dflash.{h,cpp}`) — a pure-integer HOST port of
  `_prepare_dflash_inputs_kernel` (`dflash/speculator.py:472-618`). Every store the Triton
  kernel makes is reproduced: the (1+k) query block (anchor = bonus/verified token at offset 0,
  then k `mask_token_id` rows), query positions (`last_valid_pos+1+off`, clamped
  `max_model_len-1`) / slots (via the target block table), the context positions/slots, the
  per-mask sample maps (`sample_indices`/`sample_pos`/`sample_idx_mapping`), `seq_lens =
  last_valid_pos+1+num_query_per_req`, and the CG-replay padding (`PAD_SLOT_ID=-1`, sample
  mapping -1, `query_start_loc` fill). Rejected positions excluded via `valid_ctx_end = ctx_end -
  num_rejected`. Integer, so bit-exact by construction — no float math, no CUDA, no sanitizer.
- **`PrecomputeContextKV` + `ForwardBlockLogitsWithContext`** (`qwen3_dflash.cpp`) — the
  context-KV precompute (`precompute_and_store_context_kv`, `qwen3_dflash.py:548-619`): one shared
  `hidden_norm` over the target's combined features, then per draft layer the K/V slices of
  `qkv_proj` (= the fused `_fused_kv_weight` split per layer), a per-head k-norm over `head_dim`,
  and a NeoX RoPE on K at the context positions (V raw). It reuses the LANDED
  `vt::MatmulBT`/`vt::RmsNorm`/`vt::RopeNeox` (the fused multi-layer GEMM + grouped norm are perf
  fusions, numerically the same bf16 projection). The context-aware forward attends the (1+k)
  block over its pre-inserted context K/V by laying out `[context; block]` per request and
  calling the **UNCHANGED D2 `vt::DFlashBlockAttention`** (offset-based mask exact because context
  precedes the block and the z-lab 27B SWA window 2048 >> any block, so SWA degenerates to plain
  causal-over-[context;block]); **no new kernel**, so the D2 CUDA build/sanitizer already covers
  the only kernel on this path. Block outputs are sliced back out; the rest of the layer (o_proj,
  post-norm, SwiGLU, final norm, lm_head) is shared with `ForwardBlockLogits`.

**CPU GATE GREEN (deterministic, RED-first):** `tests/vllm/v1/spec_decode/test_dflash_kvprep.cpp`
6 cases / **114 assertions**:
1. `prepare_dflash_inputs` INTEGER bit-exact vs a hand-computed 2-request reference (distinct ctx
   lengths, rejection on req1, chunked-prefill bonus on req1): ids `[42,7,7,88,7,7]`, query pos
   `[3,4,5,1,2,3]`, context pos `[0,1,2,0,1]`, context slots `[40,41,42,80,81]`, query slots
   `[43,44,45,81,82,83,-1,...]`, `query_start_loc [0,3,6,6]`, `seq_lens [6,4,0]`, sample maps all
   exact. **RED proven:** replacing `valid_ctx_end = ctx_end - num_rejected` with `ctx_end` fails
   1 case / 4 assertions (the query positions move — rejected-token exclusion is load-bearing).
2. context-KV V matches an independent f32 envelope reference (hidden_norm + KV projection); RED —
   hidden_norm perturbed changes BOTH K and V, k_norm perturbed changes K only, a context position
   perturbed changes K only (via RoPE) — each load-bearing.
3. the context-aware forward DEGENERATES exactly to the D2 context-free forward at empty context
   (bf16-envelope equal), DIVERGES when context is present (context participates in attention),
   and keeps per-request block isolation.
   Inertness: `test_qwen3_dflash_forward` **95/95** + `test_ops_dflash_block_attn` **12/12**
   byte-identical (additive-only diff).

**GPU NUMERIC-PARITY GATE PENDING on dgx** (no GPU/nvcc/cutlass on the dev box, exactly the D2
CPU-gated situation): (1) the context-KV envelope + `prepare_dflash_inputs` bit-exact +
block-proposal STRICT-or-ratified-near-tie vs the dumped vLLM reference
(`scripts/spec/d3_dflash_kvprep_ref.py`, mm-limited + gpu_util 0.30, `VLLM_USE_V2_MODEL_RUNNER=1`);
(2) 27B SACRED 235/235 + MTP 9/9 + the D2 draft-parity re-run (inertness on GPU — expected trivially
byte-identical given the additive-only diff). No new CUDA kernel ⇒ the `-Werror`/compute-sanitizer
gate is **N/A for D3** (the only kernel on the path, `DFlashBlockAttention`, was compiled +
sanitized at D2). D3 is DONE only when (1) passes.

### (Superseded) 2026-07-25 pre-run assessment — GREEN, no HW/oracle/download blocker

**Superseded by the D0 RESULT above:** the "oracle CONSTRUCTS DFlash" bullet was true only
of the config/registry, not a model run — construction of the mixed-attention draft aborts.
Retained for the reuse map (§1, still valid) and sizing (§5). Original text:

DFlash is **dispatch-ready**. The three gating facts all clear:

- **Draft checkpoints exist and are tiny** (HF API, fetched 2026-07-25):
  `z-lab/Qwen3.6-27B-DFlash` (arch `DFlashDraftModel`, model_type qwen3, **1.73 GB**
  bf16 safetensors, 91.9k downloads, config.json present) and
  `z-lab/Qwen3.6-35B-A3B-DFlash` (**368 MB** bf16 safetensors, 228k downloads).
  Both are plain Qwen3-dense decoders (§2). **NOT on dgx yet** (dgx has only the
  NVFP4 targets `nvidia--Qwen3.6-35B-A3B-NVFP4`, `unsloth--Qwen3.6-27B-NVFP4`);
  the draft download is ≤1.73 GB, public, trivial — **not a blocker** (D0 task).
- **The oracle CONSTRUCTS DFlash.** The active dgx oracle
  (`~/venvs/vllm-oracle` → `vllm-oracle-v0.25.0-stage`, vLLM **0.25.0**) ships the
  full DFlash stack: `vllm/v1/worker/gpu/spec_decode/dflash/{speculator,utils,cudagraph}.py`,
  `vllm/model_executor/models/qwen3_dflash.py`, the registry arch
  `"DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM")`, and
  `config/speculative.py` method `"dflash"` resolution. So the parity oracle
  (our-ON == our-OFF == vLLM `--speculative-config dflash`) is runnable.
  **To verify at D0** (the one soft risk): vLLM's own note "DFlash needs a
  non-causal-capable backend like FLASH_ATTN" (`speculative.py:117`) — confirm the
  pinned oracle serves DFlash on sm_121/GB10 with the NVFP4 target end-to-end (the
  community `AEON-7/vllm-dflash` DGX-Spark container proves this combination runs
  on exactly this hardware, so this is a config/backend confirmation, not a
  research question).
- **It fits the 119 GiB unified pool.** The bf16 draft (≤1.73 GB) + the ~15 GiB
  NVFP4 target is negligible. The real sizing input is the **GDN spec-state slots
  at DFlash's k** (§5): at block-16 (k=15, 16 slots/req) the 27B GDN SSM state is
  ~16 × 144 MiB ≈ **2.3 GiB per concurrent request**; 35B ≈ 16 × 60 MiB ≈ 0.96 GiB.
  This caps effective concurrency (vLLM pays the identical cost — mirror first),
  and motivates measuring the block-8 draft variant (D6). Not a correctness or
  fit blocker at low concurrency; it IS the single biggest DFlash-on-hybrid risk.

**Prerequisite status:** the MTP machinery DFlash reuses is LANDED (I1..I7 on
`72f9fb1`). The frozen spec-metadata ABI, the greedy rejection sampler, the GDN
spec slot path + rollback, the mixed spec+non-spec GDN batch, the draft-KV layer
pattern, and `--speculative-config` (server + CLI + C-ABI) all exist and are
default-off byte-identical. §1 refreshes exactly what DFlash reuses AS-IS vs must
EXTEND vs build NEW.

## Protocol compliance map

| Required field | Grounded content |
|---|---|
| Row IDs | `SPEC-DFLASH`, `MODEL-SPEC-qwen3-dflash-dflash-qwen3-for-causal-lm` |
| Scope | Qwen3.6 27B then 35B DFlash, greedy first, production graphs and sampling later; §1-§6 |
| Upstream chain | drafter model, speculator, prepare-inputs, target taps, draft KV, scheduler and rejection paths; §1-§4 |
| Our baseline | LANDED MTP infra reused (§1); genuinely-new pieces = multi-tap, non-causal in-block attn, context-KV precompute, `prepare_dflash_inputs`, full CG; exact anchors below |
| Port map | upstream-to-local files and ownership table below (§4 leaves) |
| Tests to port | lookahead, acceptance, rejection, GDN metadata (k=15), max-length and dynamic-SD cases; §7 |
| Gates | standalone draft parity, 27B/35B token-exact e2e, acceptance, memory, graphs and same-config vLLM performance; §5-§6 |
| Dependencies | LANDED `SPEC-MTP`/`SPEC-REJECTION`/`SPEC-GDN-SEGMENTS` rows, both gate model/quant rows, sliding-window attention; §1 |
| Work breakdown | ordered `D0`..`D6` increments (§6), non-overlapping `DF-*` claim leaves (§4) |
| Risks/decisions | k resolution, GDN state growth at k=15, target feature width, non-causal backend, oracle-on-sm121, remote-code policy; §0, §5, §8 |

## 1. Refreshed reuse-vs-NEW map (against LANDED MTP `72f9fb1`)

The DFlash spec's original reuse map predated the MTP landing and was stale. This
is the re-derived map — verified `file:line` at HEAD.

### 1.1 REUSED AS-IS (landed, DFlash needs zero change)

| Landed machinery | Anchor at `72f9fb1` | DFlash use |
|---|---|---|
| Frozen spec-metadata ABI | `include/vllm/config/speculative.h`; `Request::spec_token_ids`/`NumTokensWithSpec` (`include/vllm/v1/request.h`); `SchedulerOutput::scheduled_spec_decode_tokens`; `Scheduler::update_draft_token_ids`; `ModelRunnerBase::take_draft_token_ids`; `EngineCore::post_step` | AS-IS. Same scheduler/engine plumbing; DFlash only supplies drafts of length k. |
| **Scheduler `num_lookahead_tokens = k+1`** | **`speculative.h:91-108` — `use_dflash()` + `NumLookaheadTokens()` ALREADY returns `k+1` for method "dflash"** (mirrors `scheduler.py:289-292`) | **AS-IS — already implemented.** The one MTP-vs-DFlash scheduler delta (the extra slot) is already coded and inert. |
| Greedy rejection sampler | `src/vllm/v1/spec_decode/rejection_sampler.cpp` + `vt::GreedyRejectionSample` (`src/vt/cpu/cpu_sample.cpp`, `src/vt/cuda/cuda_sample.cu`) | AS-IS. The accept loop is general over `k_r` (I3 tested k∈{1,3}); no k==1 hardwiring. See §4.5. |
| Per-request logits expansion | `prepare_inputs.cpp` `cu_num_logits`/`num_draft_tokens_per_req`/widened `logits_indices` | AS-IS — general k. |
| GDN spec slot path + rollback | `vt::GdnSpecDecode`/`vt::CausalConv1dSpecUpdate` (`src/vt/cuda/cuda_gdn.cu`, `src/vt/cpu/cpu_ops.cpp`); the spec builder overload (`src/vllm/v1/attention/backends/gdn_attn.cpp`); k+1 slot alloc via `MakeQwen3_5KVCacheSpec(num_spec)` | AS-IS mechanism; `num_spec` jumps 1→15/16. See §4.5 + §5. |
| Mixed spec+non-spec GDN batch | `GdnBlockPagedMixedSpec` + `vt::IndexSelect`/`vt::IndexCopy` (`src/vllm/model_executor/models/qwen3_5.cpp`) | AS-IS — the concurrency split/merge, general k. |
| Widened-cache-aware non-spec conv ops | I5e (`GdnStateGather`/`GdnStateScatter`/`CausalConv1dUpdate` widened-cache-aware, `src/vt/ops.cpp`, `cuda_gdn.cu`) | AS-IS — already handles a `(K-1)+num_spec` conv row for any num_spec. |
| `--speculative-config` (server + CLI + C-ABI) | `examples/server/main.cpp`; `examples/cli/main.cpp`; `include/vllm.h` ABI v6; `ParseSpeculativeConfigJson` (`src/vllm/config/speculative.cpp`, `src/capi/vllm_c.cpp`) | EXTEND — parser currently throws on any method ≠ "mtp" (`speculative.h:111-118` note). Add "dflash" acceptance + resolution. Small. |

### 1.2 EXTENDED from a landed seam (reuse the pattern, widen it)

| Landed seam | What DFlash extends | Anchor |
|---|---|---|
| **Single hidden tap** → **multi-tap** | The I5d-pre `hidden_tap` out-field on `ModelForwardInput` (`include/vllm/model_executor/models/model_registry.h:155-162`) + `Qwen3_5{,Dense}Model::ForwardDeviceTap` (`qwen3_5.h:167-181`) capture ONE post-final-norm `[T,H]` tensor. DFlash needs `[T, H×taps]` from 5 (27B) / 8 (35B) residual boundaries. | Extend `ForwardDeviceTap` / the tap struct to accept a list of `target_layer_ids` and write a `[T, H×taps]` buffer; keep config-gated inert-when-off (identical discipline to the single tap). See §3.3. This is `DF-AUX-TAPS`. |
| Draft-KV layer pattern (1 layer) → 5-6 layers + SWA groups | `fa_draft` group added by `MakeQwen3_5KVCacheSpec(num_spec>0)` (one `FullAttentionSpec` at index `num_hidden_layers`, `speculator.py:163-169`). DFlash's draft has 5-6 attention layers (mixed SWA + full) → several draft KV groups + a per-layer group index. | Extend the KV-spec builder to register the DFlash draft's layer set (SWA specs + full specs). The I5d-pre `full_attn_group_id_` latent fix already keeps the target group deterministic when draft groups are appended. Part of `DF-DRAFT-MODEL`. |
| MTP paged propose (`MtpProposePrefill`, k=1 early-exit) | DFlash does NOT reuse `MtpProposePrefill` — it is a k=1 autoregressive early-exit. DFlash proposes the whole k-token block in ONE non-autoregressive forward via its own speculator. | NEW speculator (`DF-DRAFT-KV-PREP` / `DF-ENGINE-INTEGRATION`), but it plugs into the SAME `propose()`/`take_draft_token_ids` runner seam MTP wired at I5d. |
| Runner verify/propose loop | I5d/I5e/I7 runner loop (draft splice, hidden-tap capture, GDN spec feed, rejection, `take_draft_token_ids`, acceptance telemetry, mixed batch) | Reuse the loop SHAPE; swap the MTP propose call for the DFlash speculator call (multi-tap combine → context-KV precompute → one draft forward → sample k). |

### 1.3 GENUINELY NEW (no landed analogue)

| New piece | Upstream | Why new |
|---|---|---|
| **Multi-layer aux hidden taps → `[T, H×taps]`** | `eagle3_utils.py:36-55`; `qwen3_dflash.py:377-395` (`fc`, input H×num_taps→H); `dflash/speculator.py:285-291` `combine_hidden_states` | Our tap is single. DFlash conditions the drafter on 5/8 residual-boundary features combined by an `fc`. (`DF-AUX-TAPS`) |
| **Non-causal in-block attention** (project's FIRST) | `qwen3_dflash.py:55-122` `_resolve_layer_attention` — full-attn layers default **non-causal**, SWA layers causal | Every `vt` attention primitive we ship (`PagedFlash*`) is causal. The DFlash draft's full-attention layers attend BIDIRECTIONALLY within the 1+k query block. NEW primitive: a dense/bidirectional block attention over the uniform `(1+k)` query (no causal mask), causal only for the SWA layers. (`DF-DRAFT-MODEL`) |
| **The `qwen3_dflash` draft model** | `vllm/model_executor/models/qwen3_dflash.py` (`DFlashQwen3ForCausalLM`, `DFlashQwen3Model`) | Plain Qwen3-dense decoder (gate/up/down MLP, q/k/v/o + q/k norm), 5-6 layers, SWA + full, NO GDN/MoE. New model + loader + goldens. (`DF-DRAFT-MODEL`) |
| **Context-KV precompute** | `qwen3_dflash.py:471-568` `precompute_and_store_context_kv` | Fused multi-layer KV proj (`_fused_kv_weight`) + bulk RoPE + direct cache insert of the target features → the draft never re-runs context tokens. Runs eagerly, outside CG. NEW. (`DF-DRAFT-KV-PREP`) |
| **`prepare_dflash_inputs`** | `dflash/speculator.py:416-563` (one Triton kernel) | Mask-token blocks (anchor = bonus token, then k `mask_token_id` embeddings), context positions/slots from the target block table, per-mask sample maps, `valid_ctx_end = ctx_end − num_rejected`. DISTINCT from MTP's shift-splice `prepare_prefill_inputs`. NEW kernel. (`DF-DRAFT-KV-PREP`) |
| **FULL CUDA graph for the uniform `1+k` draft step** | `dflash/cudagraph.py`; `speculator.py:369-378` (`uniform_token_count=1+k`) | The uniform block shape makes this the easy CG case, but it is a new graph registration for the draft forward. (`DF-GRAPH-SAMPLING`) |
| Separate mask-embedding fallback | `qwen3_dflash.py:352-361` | NEW-trivial; our drafts use in-vocab mask tokens (248077/248070), so the fallback is unused but mirrored. |

## 2. Draft checkpoints for OUR exact gate models (HF, re-verified 2026-07-25)

| | `z-lab/Qwen3.6-35B-A3B-DFlash` | `z-lab/Qwen3.6-27B-DFlash` |
|---|---|---|
| arch / model_type | `DFlashDraftModel` / qwen3 | `DFlashDraftModel` / qwen3 |
| bf16 safetensors size | **368 MB** | **1.73 GB** |
| downloads (HF) | 228k | 91.9k |
| layers | 6 (5× SWA-4096 + 1 full) | 5 (4× SWA-2048 + 1 full) |
| hidden / heads / kv / head_dim | 2048 / 32 / 8 / 128 | 5120 / 32 / 8 / 128 |
| block_size (drafted block) | 16 (`dflash_config`) | 16 (top-level key) |
| mask_token_id | 248077 | 248070 |
| target_layer_ids (aux taps) | [1, 6, 11, 16, 22, 27, 32, 37] (8 taps, num_target_layers 40) | [1, 16, 31, 46, 61] (5 taps, num_target_layers 64) |
| dtype / vocab | bf16 / 248320 | bf16 / 248320 |
| rope_theta | 1e7 (full rotary, head_dim 128) | 1e7 |

Both are plain Qwen3-dense-style decoders (gate/up/down MLP, q/k/v/o + q/k norm) —
NO GDN, NO MoE in the draft. The draft is bf16 while targets are NVFP4 — exactly
the combination the DGX-Spark community container (`AEON-7/vllm-dflash`) runs on
GB10. **Neither draft is on dgx yet** — D0 downloads them (≤1.73 GB, trivial).

## 3. Anatomy at pin (what a port implements)

### 3.1 Per-step flow (`dflash/speculator.py:246-413`)
1. Target step finishes; `propose()` receives the target's per-token hidden states
   (aux-combined, §3.3) + `num_sampled/num_rejected/last_sampled`.
2. `prepare_dflash_inputs` (Triton, `:416-563`): per request build a query block of
   `num_query_per_req = 1 + k` tokens (`:45`) — the **anchor** = bonus token (last
   sampled, or next prefill token for chunked prefill `:464-469`), then **k mask
   tokens** (`mask_token_id` embedding, `:497`); positions `last_valid_pos+1 …`;
   also context positions/slots from the target block table and per-mask sample
   indices. Rejected positions excluded via `valid_ctx_end = ctx_end − num_rejected`
   (`:461-471`).
3. **Context-KV pre-insert** (`qwen3_dflash.py:471-568`): the target's hidden states
   for THIS step's tokens are projected to K/V for ALL draft layers in one fused
   GEMM (`_fused_kv_weight`), k-normed + RoPE'd in bulk, and written directly into
   the draft's KV caches at the context slots. The draft never re-runs context.
   Runs eagerly (context length varies), outside CUDA graphs (`:350-353`).
4. One draft forward over the `num_reqs × (1+k)` query tokens (uniform → FULL CG,
   `:369-378`); **non-causal in-block attention** for full-attention layers, causal
   for SWA layers (`qwen3_dflash.py:55-122`).
5. Sampling: each mask position t yields draft token t via the shared `sample_draft`
   (`:209-224`; Gumbel positions offset −2 so verify noise matches, `:212-215`).
   Anchor is not sampled (`sample_from_anchor = False`, `:53-56`; the True variant
   is DSpark — out of scope).
6. Verify next step: the target runs the k+1-token query and the **SAME
   `RejectionSampler`** as MTP (`model_runner.py:1065-1079` — no DFlash-specific
   verify path). Scheduler rollback identical.

### 3.2 Draft KV cache
The draft's 5-6 attention layers register as extra attention layers → own KV-cache
group(s) (`speculator.py:125-169`: `draft_kv_cache_group_ids`, per-group context
slot buffers, and, for mixed SWA/full drafts, a per-layer group index). Slot
mappings for query tokens are written into the shared `BlockTables.slot_mappings`
so captured graphs replay correctly (`:321-348`). Extends the landed single
`fa_draft` group pattern (§1.2).

### 3.3 Aux hidden capture (target-side change — extends the single tap)
The drafter conditions on MULTI-LAYER target features: `target_layer_ids` +1-shifted
via eagle3 (`eagle3_utils.py:36-55`) → `set_aux_hidden_state_layers(...)`; the target
forward returns `(hidden, aux_hidden_states)` (`model_runner.py:1324-1332`); the
drafter combines `hidden = model.combine_hidden_states(cat(aux, dim=-1))`
(`speculator.py:285-291`; the `fc` lives in the draft, `qwen3_dflash.py:377-395`,
input = hidden_size × num_taps → hidden_size).

**For us:** extend the landed I5d-pre `hidden_tap` seam (`model_registry.h:155-162`,
`ForwardDeviceTap`) — instead of one post-final-norm `[T,H]` copy, tap the residual
stream at 8 (35B) / 5 (27B) configured layer boundaries into a `[T, H×taps]` buffer.
Cheap (copies only) but touches the fused forward — keep the taps config-gated so
the non-spec (and non-DFlash-spec, e.g. MTP-spec) hot path is byte-identical.

### 3.4 What DFlash does NOT need
- No multi-step draft loop (one forward per block).
- No draft-side GDN (drafts are plain attention).
- No new rejection sampler, scheduler mechanics, or GDN target machinery beyond
  what MTP landed — but the GDN state slots scale as k+1 per request (§4.5, §5).

## 4. Port map and non-overlapping claim leaves

| Claim leaf | Upstream source | Owned local scope | Exit gate |
|---|---|---|---|
| `DF-DRAFT-MODEL` | `qwen3_dflash.py`, DFlash config/loader | new `include/vllm/model_executor/models/qwen3_dflash.h`, `src/.../qwen3_dflash.cpp`, the NEW non-causal block-attn `vt` primitive, target-specific weight loader + model unit tests | standalone draft forward parity on captured target features |
| `DF-AUX-TAPS` | eagle3 aux capture + `combine_hidden_states` fc | extend `qwen3_5*.{h,cpp}` `ForwardDeviceTap` + the tap struct to multi-tap `[T,H×taps]`; runner metadata; no scheduler/speculator files | exact configured layer/features, non-spec + MTP-spec byte-identical |
| `DF-DRAFT-KV-PREP` | `dflash/speculator.py`, `dflash/utils.py`, prepare-input + context-KV Triton | new `src/vllm/v1/worker/gpu/spec_decode/dflash/`, headers, the `prepare_dflash_inputs` + context-KV precompute CUDA kernels + unit tests | context-KV reuse + block-proposal parity |
| `DF-ENGINE-INTEGRATION` | shared speculator/scheduler/rejection path (LANDED) | integration-only wiring of the DFlash speculator into the I5d/I7 runner loop; `--speculative-config dflash` parse/resolve; DFlash lookahead/e2e tests | 27B then 35B token-exact + acceptance gates |
| `DF-GRAPH-SAMPLING` | `dflash/cudagraph.py`, stochastic verify | DFlash FULL-CG registration for the uniform 1+k step + sampler parametrization | production-graph + seeded-sampling parity/perf |

No leaf edits another leaf's owned files without an explicit lead claim in
`coordination.md`.

### 4.5 The k>1 question — VERDICT: landed rejection + GDN handle it MECHANICALLY; validation + memory are the work

DFlash's block-diffusion drafts k tokens per step (k = block−1, typically 15 at
block 16), unlike MTP's k=1. Assessment against the landed code:

- **Greedy rejection sampler: k-general AS-IS.** The accept walk is `for i in
  [0, k_r)` (MTP spec §2.8), no `k==1` hardwiring anywhere in `rejection_sampler.cpp`
  / `cpu_sample.cpp` / `cuda_sample.cu`; I3 already ships and tests k∈{1,3}. No
  change needed for greedy DFlash. (Stochastic rejection stays deferred — the DFlash
  gate is greedy-first, so it does not block; it lands with `DF-GRAPH-SAMPLING`,
  mirroring MTP M-mtp-3.)
- **GDN spec slots + recurrence + mixed batch: k-general AS-IS.** `vt::GdnSpecDecode`
  is a general `T>1` loop with per-timestep snapshots; `MakeQwen3_5KVCacheSpec(num_spec)`
  sizes `num_spec+1` slots for any `num_spec`; `GdnBlockPagedMixedSpec` + `IndexSelect`/
  `IndexCopy` split/merge is general; I5e made the non-spec conv ops
  `(K-1)+num_spec`-cache-aware for any num_spec. No `num_spec<=1` assert exists.
  Setting `num_spec = k = 15` is a configuration, not a code change.
- **What IS the work at k=15, not a mechanism gap:** (a) EXERCISE/validate the
  landed k-general paths at k=15/16 — MTP only ever ran them at k=1 end-to-end, so
  the k=15 GDN spec + rejection + mixed batch are code-general but UNTESTED at scale
  (D4 gate); (b) the **memory sizing** at 16-17 slots/req (§5 — the 2.3 GiB/req 27B
  GDN state is the real gating cost, not a code change); (c) DFlash's own drafting is
  non-autoregressive block, so it never touches MTP's k=1-only `MtpProposePrefill`
  early-exit — that limitation is irrelevant to DFlash.

**Bottom line:** DFlash needs NO extension to the rejection or GDN-slot MECHANISM.
It needs the NEW drafter (model + taps + context-KV + prepare_dflash), then to
exercise the k-general shared paths at k=15 and measure the k+1-slot memory.

## 5. Sizing flags (must-measure before committing to defaults)

- **GDN state slots (the #1 DFlash-on-hybrid risk):** k+1 slots/request on the
  TARGET's GDN layers. At block 16 (k=15, 16 slots): 27B ≈ 16 × 144 MiB ≈ **2.3 GiB
  per concurrent request**; 35B ≈ 16 × 60 MiB ≈ 0.96 GiB (measured slot cost from
  SPEC-GDN-SEGMENTS I4: 27B 144 MiB/slot/req over 48 GDN layers, 35B 60 MiB over 30).
  On GB10 (119 GiB unified) this caps effective concurrency; vLLM pays the same —
  mirror first, then measure the `block_size: 8` draft variant acceptance-vs-memory
  trade (D6). NOT visible in the B200 dense-model numbers.
- **Verify width:** 17-token queries/request/step through a 256-expert MoE (35B)
  unions many experts; GB10's compute:bandwidth ratio thins the B200 margins. Gate
  at both operating points.
- **Acceptance content-dependence:** natural-language + code workloads, never
  synthetic (B5: prose ~2/15 vs code ~5.5/15 accepted for a k=15 block).
- **Non-causal backend on sm_121:** confirm the draft's bidirectional block attention
  path (our new primitive) is correct AND that the vLLM oracle serves DFlash on
  sm_121 (§0). Soft risk, D0.

## 6. Ordered W-plan (D0..D6, MTP-style, independently gateable)

Critical path = **D0 → D1 → D2 → D3 → D4 → D5 → D6**. D1 (`DF-AUX-TAPS`) and the
D2 model can be developed in parallel worktrees once D0 confirms configs, but the
D4 loop needs both. The single throughput-relevant blocker to surface EARLY is the
D0 oracle-on-sm121 check (if the pinned oracle can't serve DFlash+NVFP4 on GB10,
D5's three-way gate loses its vLLM arm — mitigation: the community container proves
the combination runs, so worst case is a config/version bump, not a dead row).

| # | Builds | Gate | GPU/CPU | Hardest risk |
|---|---|---|---|---|
| **D0** ground + download | Fetch both drafts to dgx; dump their resolved `SpeculativeConfig`; confirm the pinned oracle SERVES DFlash + NVFP4 on sm_121 end-to-end (short greedy run) | **✓ DONE 2026-07-26 (§0 D0 RESULT) on the NEW pin `555967922`/vLLM 0.26.0.dev0.** vllm#40898 resolved: mixed-attn draft CONSTRUCTS, drafter ALIVE (acceptance 2.21/8.80/4.75/4.57 > 1), `num_spec=16`, backend flashinfer-native. Goldens `dflash_27b_spec_{on,off}.json` committed. Gate FORM measured = STRICT mode-matched (vLLM-ON run-deterministic K>=3 but != vLLM-OFF). | GPU (one short oracle run under `flock`) | ~~Oracle can't serve DFlash on sm_121~~ RESOLVED by pin advance + `VLLM_USE_V2_MODEL_RUNNER=1`. New hazard hit+fixed: multimodal 27B vision-encoder profiling OOM-rebooted dgx -> `limit_mm_per_prompt=0` + gpu_util 0.30 |
| **D1** aux multi-tap seam (`DF-AUX-TAPS`) | Extend `ForwardDeviceTap`/tap struct: capture `[T,H×taps]` at 5(27B)/8(35B) residual boundaries, config-gated | **✓ DONE 2026-07-26 (§0 D1 RESULT).** `ForwardDeviceMultiTap` (MoE+Dense) captures `(hidden+res)` at `target_layer_ids` into `[T,H×taps]`; unit gate 598 assertions vs an independent truncated-model reference (RED-first: reversed concat -> 384 fail); CUDA 697/697 + compute-sanitizer 0; MTP e2e 9/9 + 27B SACRED **235/235** byte-identical on the new oracle (inertness) | GPU (parity dump) + CPU (unit) | ~~byte-identical when off~~ PROVEN (config-gated, additive); the direct vLLM aux-dump parity folds into D2 (see §0 D1 honest scope note) |
| **D2** the `qwen3_dflash` drafter + non-causal in-block attention (`DF-DRAFT-MODEL`) | New draft model (5-6 dense layers, SWA+full), loader, the NEW non-causal block-attention `vt` primitive, draft KV groups | **CODE LANDED + CPU-GATED 2026-07-26 (`CLAIM-DFLASH-D2`, §0 D2 RESULT):** model + `vt::DFlashBlockAttention` (CPU authoritative + CUDA port) + fc + mask-embed + loader; CPU gate op 12/12 (RED non-causal) + model 95/95 (RED full-layer-causal-flip, block isolation, fc RED); causal path byte-identical (test_ops_attention/test_qwen3_forward unchanged). **✓ DONE 2026-07-26 — GPU promotion GREEN on dgx (§0 D2 RESULT):** CUDA `-Werror` clean (kernel compiles as-written); CUDA==CPU 198412/198412 + compute-sanitizer 0; draft-forward parity vs the REAL vLLM draft (fc rel-L2 0.46%, hidden ≤1.3%, 11 strict + 5 near-tie proposed ids); 27B SACRED 235/235 + MTP 9/9 byte-identical | GPU (forward + parity) + CPU (op ref) | **The non-causal primitive is the project's first** — bidirectional block attention; get the mask + SWA-vs-full per-layer routing right |
| **D3** context-KV precompute + `prepare_dflash_inputs` (`DF-DRAFT-KV-PREP`) | Multi-layer KV proj + k-norm + RoPE (context-KV precompute); the mask-block/context-slot/sample-map input builder; the context-aware draft forward | **CODE LANDED + CPU-GATED 2026-07-26 (`CLAIM-DFLASH-D3`, §0 D3 RESULT):** `PrepareDflashInputs` (pure-integer HOST port of `_prepare_dflash_inputs_kernel`) + `PrecomputeContextKV` + `ForwardBlockLogitsWithContext` (reuse the LANDED MatmulBT/RmsNorm/RopeNeox + the UNCHANGED D2 `vt::DFlashBlockAttention` via a combined `[context;block]` sequence — NO new kernel). CPU gate `test_dflash_kvprep` 6/114: prepare INTEGER bit-exact vs a hand reference (RED-first: breaking `valid_ctx_end` fails 4 assertions); context-KV V envelope-matched + hidden_norm/k_norm/RoPE-pos RED load-bearing; context forward degenerates EXACTLY to the D2 context-free forward at empty context + diverges with context + block isolation. Additive-only diff ⇒ D2/MTP/SACRED byte-identical. **GPU numeric-parity PENDING on dgx:** context-KV envelope + prepare bit-exact + block-proposal STRICT/near-tie vs `scripts/spec/d3_dflash_kvprep_ref.py`; 27B SACRED 235/235 + MTP 9/9 + D2 parity re-run. No new CUDA ⇒ `-Werror`/sanitizer N/A. | GPU (parity dump) + CPU (input-prep ref) | Rejected-position exclusion (`ctx_end − num_rejected`) + slot arithmetic off-by-one — both RED-gated; eager (non-CG) context path |
| **D4** engine integration + k>1 exercise (`DF-ENGINE-INTEGRATION`) | Wire the DFlash speculator into the landed I5d/I7 runner loop; `--speculative-config dflash` parse/resolve; run the k-general shared paths at k=15 | the loop runs end-to-end at k=15 on the 27B, drafts proposed & accepted (nonzero); the k=15 GDN spec slots + rejection + mixed batch exercised (no crash, bit-exact vs I4 ops as a token chain at k=15) | GPU | First real k=15 exercise of the shared GDN/rejection paths (code-general but untested at scale); k+1=16 slot alloc + widened conv at k=15 |
| **D5** correctness gate | The DFlash correctness bar | **REVISED by the D0 measurement (§0):** NOT the three-way identity — vLLM-DFlash-ON != vLLM-spec-OFF at k=16 near-ties. Gate = STRICT MODE-MATCHED **`our-DFlash-ON == vLLM-DFlash-ON` greedy token-for-token** (the committed golden is the reference) + measured nonzero acceptance; 27B then 35B; spec-OFF byte-identical SACRED as a SEPARATE inertness gate | GPU (`flock`, standalone big-model) | Batch-nondeterminism at c>1 (near-tie-distributional-gate applies as in I7 — exact identity only at c1); acceptance content-dependence |
| **D6** throughput A/B + memory + block-8 | c1 then c>1 A/B vs vLLM DFlash same-config; GDN k+1-slot memory measurement; block-8 vs block-16 acceptance-vs-memory | ours ≥ vLLM (throughput) / ≤ vLLM (latency, memory) on every axis at c1 AND the throughput operating point, both models; §5 memory within the 119 GiB pool at the target concurrency; `benchmark_binding=true` | GPU (A/B series, one `flock`, cold-leg discarded, ≥3 reps) | The 2.3 GiB/req 27B GDN state at k=15 caps concurrency; DFlash may go <1× at high concurrency (mirror vLLM's behavior at each point) |

Milestone rollup: **M-df-0** = D0-D3 (draft runs standalone + inputs), **M-df-1** =
D4-D5 27B e2e greedy, **M-df-2** = D5 35B (8 taps + §5 memory + concurrency sweep),
**M-df-3** = D6 sampling + FULL CG.

## 7. Tests to port (protocol: .agents/test-porting.md)

| upstream | what it asserts | tier → ours |
|---|---|---|
| `tests/v1/spec_decode/test_dflash_lookahead.py` (`test_dflash_prefill_reserves_lookahead_blocks` :98, `..._first_prefill_query_window_fits_allocated_blocks` :117, `..._drafter_window_reserves_bonus_token` :134) | k+1 lookahead slot accounting: prefill reserves blocks for the query window incl. bonus | T-unit → `tests/vllm/v1/spec_decode/test_dflash_lookahead.cpp` |
| `tests/v1/e2e/spec_decode/test_spec_decode.py::test_dflash_acceptance_rates` (:1323-1345) | e2e acceptance-rate floor with a real DFlash draft | T-e2e (nightly dgx) → paged-engine spec config with the z-lab drafts; we additionally gate 16/16 greedy token-exactness vs the oracle (stricter, per gates.md) |
| shared rejection-sampler suite (MTP §6) | verify correctness independent of drafter | ALREADY ported (I3); DFlash adds parametrizations with k=block−1 (=15) |
| `tests/v1/attention/test_gdn_metadata_builder.py` spec cases | GDN split at k≫1 | ALREADY ported (I4); extend params to k=15 |
| draft-model unit coverage (no dedicated upstream `qwen3_dflash` unit test at pin — correctness via e2e acceptance) | non-causal in-block attention, SWA layers, context-KV precompute numerics | T-parity → golden-dump tests vs the HF draft (`tools/parity/`); cite `qwen3_dflash.py` in the test header |
| `tests/v1/spec_decode/test_max_len.py`, `test_dynamic_sd.py` | max-model-len clamping with lookahead; dynamic SD on/off | T-unit, port applicable cases; SKIP dynamic-SD until we mirror that feature |

## 8. Open questions

- vLLM's default `num_speculative_tokens` for these drafts (block 16 → k=15? or the
  block-8 variant): resolve by running the oracle with the z-lab cards' recommended
  config and mirroring what `SpeculativeConfig` resolves (`speculative.py:865-880`)
  — never hand-pick. **D0 task.**
- Whether the 27B draft's top-level `block_size: 16` (vs in-`dflash_config` for the
  35B) resolves identically through `EAGLEConfig` wrapping — check at D0 with the
  oracle.
- Oracle serves DFlash + NVFP4 on sm_121/GB10 (non-causal backend gate,
  `speculative.py:117`) — **D0**, community container is the fallback config.
- trust_remote_code: the drafts ship `custom_code` (`dflash.DFlashDraftModel`) —
  vLLM uses its in-tree `qwen3_dflash.py`; we mirror the in-tree model, so no remote
  code in our engine.
- DSpark (`dspark/`, deepseek drafts) shares this scaffolding — explicitly out of
  scope (`SPEC-DSPARK`); revisit only if a Qwen3.6 dspark draft appears.

Sources: pin files cited inline · HF API `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`
(fetched 2026-07-25) · dgx oracle `vllm-oracle-v0.25.0-stage` (DFlash registry +
speculator confirmed 2026-07-25) · landed MTP anchors at `72f9fb1` (verified
file:line) · arxiv.org/abs/2602.06036 · github.com/AEON-7/vllm-dflash · B5 scoping.
