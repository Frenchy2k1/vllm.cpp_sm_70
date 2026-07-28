# Spike: MTP speculative decoding from a GGUF target (`SPEC-MTP-GGUF`)

Stable row: `SPEC-MTP-GGUF` (`.agents/engine-matrix.md`).
Depends on: `SPEC-MTP` (`DONE`, safetensors), [mtp-spec-decode.md](mtp-spec-decode.md).

## Scope

Make `--speculative-config '{"method":"mtp"}'` work when the TARGET is a `.gguf`
file, not only a safetensors directory.

Today `LoadedEngine::FromModelDir` refuses the combination outright
(`src/vllm/entrypoints/model_loader.cpp:717-723`):

```
speculative decoding requires a safetensors target checkpoint: GGUF exports
lack the mtp.* draft tensors / bf16 target embed+lm_head the DFlash draft shares
```

That message encodes an assumption from the original MTP spike
([mtp-spec-decode.md](mtp-spec-decode.md):979-980, "GGUF checkpoints lack
`mtp.*` -> document as safetensors-only feature **until we re-export GGUFs with
the head**"). The assumption is now stale in one direction: llama.cpp's Qwen3.5
converter DOES emit the MTP head into GGUF, under its own layer-indexed `nextn`
naming, and our own `HfConfigFromGguf` ALREADY reads the metadata key that
announces it. So the head is present and detectable in third-party GGUFs we do
not produce; what is missing is our loader mapping.

IN SCOPE: the GGUF weight resolver for the `mtp.*` block, the config plumbing so
`ResolveSpecConfig` learns the head depth from GGUF metadata, removing the
rejection for `method == "mtp"`, and the token-identity gate.

OUT OF SCOPE: `dflash` over GGUF (separate row `SPEC-DFLASH-GGUF`,
[gguf-dflash-draft.md](gguf-dflash-draft.md)); a standalone MTP-only drafter
GGUF as a SECOND model handle (noted under Risks); non-Qwen3.5/3.6
architectures, which the widened spec KV path does not serve at all
(`MakeKVCacheMaybeSpec`, `src/vllm/entrypoints/model_loader.cpp`).

## Upstream chain

The producer side is llama.cpp, not vLLM (vLLM has no GGUF MTP path), so the
"upstream" to mirror here is the GGUF CONTRACT llama.cpp writes.

- `gguf-py/gguf/constants.py:129` — `{arch}.nextn_predict_layers`, the head-depth
  metadata key (the GGUF spelling of HF `mtp_num_hidden_layers`).
- `gguf-py/gguf/constants.py:910-917` — the `NEXTN_*` tensor enums;
  `:1494-1501` — their GGUF names: `blk.{bid}.nextn.eh_proj`,
  `.nextn.embed_tokens`, `.nextn.enorm`, `.nextn.hnorm`,
  `.nextn.shared_head_head`, `.nextn.shared_head_norm`, plus the unindexed
  `nextn.pre_projection` / `nextn.post_projection`.
- `gguf-py/gguf/tensor_mapping.py` — `NEXTN_*` map to HF
  `model.layers.{bid}.eh_proj` / `.enorm` / `.hnorm` / `.shared_head.head` /
  `.shared_head.norm` / `.embed_tokens`, i.e. the DeepSeek-V3 MTP spelling.
- `conversion/qwen.py:535-604` — `_Qwen35MtpMixin`, the authoritative bridge.
  It extends `block_count` by `mtp_num_hidden_layers`, emits
  `add_nextn_predict_layers(n)`, and in `filter_tensors` remaps Qwen3.5's
  `mtp.*` onto the layer-indexed DeepSeek spelling with exactly:

  | Qwen3.5 HF | remapped to |
  |---|---|
  | `mtp.fc` | `model.layers.{L}.eh_proj` |
  | `mtp.pre_fc_norm_embedding` | `model.layers.{L}.enorm` |
  | `mtp.pre_fc_norm_hidden` | `model.layers.{L}.hnorm` |
  | `mtp.norm` | `model.layers.{L}.shared_head.norm` |
  | `mtp.layers.{i}.<rest>` | `model.layers.{L+i}.<rest>` |

  where `L` = the ORIGINAL `num_hidden_layers` (`_original_block_count`). Note
  the asymmetry: the four scalar tensors land on block `L` while
  `mtp.layers.{i}.*` land on `L+i`. At `mtp_num_hidden_layers == 1` (both gate
  checkpoints) these coincide.
- Same mixin: `no_mtp` and `mtp_only` conversion flags, so a Qwen3.5 GGUF in the
  wild may carry the head, omit it, or BE only the head.

Also relevant, and the reason `llama.cpp` archs matter beyond Qwen: the same
`nextn` tensor set is declared for `QWEN35`, `QWEN35MOE`, `GEMMA4_ASSISTANT`,
`COHERE2MOE`, `DEEPSEEK32`, `GLM4_MOE`, `GLM_DSA`, `EXAONE4`, `EXAONE_MOE`,
`BAILINGMOE2`. Only the two Qwen35 archs are reachable for us at this pin.

## Our baseline

What already exists, and is the reason this row is small:

- **The head-depth key is already read.** `HfConfigFromGguf`
  (`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:568`) does
  `const int64_t nextn = OptInt(gguf, p + "nextn_predict_layers", 0);` and uses
  it as `c.num_hidden_layers = block_count - nextn`. The trunk layer count is
  therefore ALREADY correct on a head-carrying GGUF; the head blocks are simply
  never loaded. The value is then discarded.
- **The MTP weight loader is already resolver-shaped.**
  `LoadQwen3_5MTP(const TensorResolver& get, const HfConfig&, Qwen3_5MTPKind)`
  (`src/vllm/model_executor/models/qwen3_5_mtp.cpp:272`) is source-agnostic; the
  safetensors entry point (`:337`) only builds a name -> shard map and delegates.
  `TensorResolver` is `std::function<const StTensor&(const std::string&)>`
  (`include/vllm/model_executor/models/qwen3_5_weights.h:343`).
- **`StTensor` is a plain view**, not a safetensors-owned type:
  `{dtype, shape, data, nbytes}`
  (`include/vllm/model_executor/model_loader/safetensors_reader.h:17-22`). A
  GGUF-backed resolver can hand out views over dequantized buffers it owns.
- **GGUF -> bf16 dequant already exists** and is already used by the trunk GGUF
  loader: `DequantGgufRowToBf16` via
  `vllm/model_executor/model_loader/gguf_dequant.h`, wrapped locally in
  `qwen3_5_gguf_weights.cpp`.
- **The spec engine itself is architecture-complete and gated** for these
  checkpoints (`SPEC-MTP` `DONE`: three-way token-exact at c1, c2-c8 on-par with
  vLLM). Nothing about the propose/verify loop changes here.

So the gap is exactly: a `TensorResolver` over `GgufFile`, a config field, and
one deleted `throw`.

## Port map

| # | File | Change |
|---|---|---|
| 1 | `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:568` (`HfConfigFromGguf`) | Stop discarding `nextn`: set `c.raw["mtp_num_hidden_layers"] = nextn` so `ResolveSpecConfig` (`src/vllm/entrypoints/model_loader.cpp:458-470`) resolves the real depth instead of its `int64_t{1}` fallback. Keep `num_hidden_layers = block_count - nextn` unchanged. |
| 2 | **CORRECTED at implementation time.** `LoadQwen3_5MTPFromGguf` in the EXISTING `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`, NOT a `TensorResolver` in a new TU. **The resolver seam is the wrong seam**: it hands out raw tensor views, but the GGUF conventions live one level up in that file's helpers, and there are three of them, all silent if missed. (a) Norm weights are stored **(w + 1)** and must be un-shifted (`OwnNormMinus1`) - a plain read leaves every norm off by one and poisons every proposal without failing anything. (b) Matmul weights carry the file's quantization + residency routing (`OwnMatmulWeight`, `RequireExpand`). (c) `GgufTensorInfo::shape` is already torch `[N, K]`, so `fc` is the VERBATIM path, not the transposing one - the transposing helper would yield `[2H, H]` and only fail deep inside the draft forward. Going through the resolver means re-deriving all three in a second place; reusing the trunk helpers is what makes the head obey the same conventions as the trunk it speculates for. |
| 3 | same | Scalars resolve at block `L` (`nextn.eh_proj`/`enorm`/`hnorm`/`shared_head_norm`); the head's transformer block(s) at `L+i` reuse `LoadAttnGguf` / `OwnMatmulWeight` / `LoadMoeGguf`. The head block is ALWAYS full-attention and is deliberately NOT looked up in `config.layer_types`, which covers the trunk only (a lookup there would run off the end). |
| 4 | `src/vllm/entrypoints/model_loader.cpp:717-723` | Narrow the rejection: keep it for `method == "dflash"` (until `SPEC-DFLASH-GGUF`), drop it for `"mtp"`. `"ngram"` is already unaffected. |
| 5 | `src/vllm/entrypoints/model_loader.cpp` (GGUF branch, ~`:710-730`) | After `ModelRegistry::Load`, when a spec config is present call `LoadQwen3_5MTP` with the GGUF resolver and `AttachMtpDraftWeights`, mirroring the safetensors branch at `:770`. |
| 6 | `include/vllm.h` / `docs` | No ABI change. The capability is reachable through the existing `speculative_config` field. |

Decision to make in step 2 (record the outcome in this file): whether the MTP
head uses the GGUF's own `blk.{L}.nextn.shared_head_head` or the target's
`output.weight`. Qwen3.5 SHARES the target lm_head
(`LoadQwen3_5MTP` asserts `!UsesDedicatedEmbeddings(config)`), and the converter
emits `shared_head.head` only because the DeepSeek layout has that slot, so the
two should be the same tensor. Verify byte-equality on a real GGUF; if they
differ, the target's is authoritative.

## Tests to port

There is no upstream test to port (vLLM has no GGUF MTP path), so the gates are
ours, built to the same shape as the safetensors MTP gates.

| Test | Location | What it pins |
|---|---|---|
| Resolver name mapping | new `tests/vllm/models/test_qwen3_5_gguf_mtp.cpp` | Every name `LoadQwen3_5MTP` requests resolves; the four scalars land on block `L`, `mtp.layers.{i}` on `L+i`. Synthetic in-memory `GgufFile`, no weights. RED-first by asserting against a deliberately off-by-one block index. |
| Dequant equivalence | same | A GGUF-backed resolver over an F32/F16 head produces bit-identical bf16 to the safetensors resolver over the same values. |
| Config depth | extend `tests/vllm/models/test_qwen3_5_gguf_weights.cpp` (or the owning GGUF config test) | `nextn_predict_layers=N` yields `raw["mtp_num_hidden_layers"] == N` AND `num_hidden_layers == block_count - N`. Currently only the second holds. |
| Rejection narrowing | `tests/capi/test_capi.cpp` or the model-loader test | `mtp` + GGUF no longer throws; `dflash` + GGUF still does, with the message naming dflash. |
| Three-way token gate | new `tests/parity/test_qwen35_gguf_spec_decode.cpp`, modelled on `tests/parity/test_qwen27_spec_decode.cpp` | GGUF spec-ON == GGUF spec-OFF token-for-token at c1, plus acceptance > 0. |

## Gates

1. **Spec-OFF byte-identical (SACRED).** The GGUF trunk path must not move:
   27B 235/235, 35B 315/315, Coder 138/138. Step 1 touches `HfConfigFromGguf`,
   which every GGUF load runs, so this is the gate that matters most.
2. **Token identity, c1.** GGUF spec-ON == GGUF spec-OFF, token-for-token,
   greedy, single request. This is the `SPEC-MTP` I5e bar restated for GGUF.
3. **Cross-format agreement (the strong gate, if the checkpoint allows it).**
   For a checkpoint available BOTH as safetensors and as an F16 GGUF, the GGUF
   spec-ON tokens should equal the safetensors spec-ON tokens. Only meaningful
   at F16/F32 (a quantized GGUF legitimately diverges), so record it as
   NOT APPLICABLE when only a quantized export exists.
4. **Acceptance parity.** GGUF acceptance rate within noise of the safetensors
   rate on the same prompts; a near-zero rate means the head loaded but is
   wired wrong (the exact failure mode I5e hit and RCA'd).
5. **Speed: PENDING, not owed by this row.** Spec-decode throughput is already
   recorded under `SPEC-MTP` (I6/I7). This row owes a `docs/BENCHMARKS.md`
   entry, but it may legitimately be `PENDING` with the reproduction command
   until a GGUF A/B is run.

## Dependencies

- `SPEC-MTP` `DONE` (the propose/verify loop, widened KV, GDN spec routing).
  Nothing here modifies it.
- The GGUF trunk loader and `GgufLoadPolicy` residency routing
  (`qwen3_5_gguf_weights.cpp`), unchanged but depended on for stem naming.
- `gguf_dequant.h` (`DequantGgufRowToBf16`), unchanged.
- **A test asset.** A Qwen3.5/3.6 GGUF that actually carries the head, i.e.
  converted WITHOUT `--no-mtp`. Sourcing or producing one is the first task and
  can block everything after step 1 — see Risks.
- No new third-party dependency. No ABI change. No CUDA/GPU dependency for
  steps 1-3 (CPU build suffices for the resolver tests).

## Work breakdown

| Row | Work | Gate | Blocked by |
|---|---|---|---|
| `G0` | **DONE 2026-07-28.** Confirmed against a real llama.cpp-converted Qwen3.5-2B (Q8_0 body) already on disk: `qwen35.block_count=25`, `qwen35.nextn_predict_layers=1` => trunk `L=24`, head at `blk.24`, carrying `nextn.eh_proj [2048,4096]` (torch `[N,K]`), `nextn.enorm/hnorm/shared_head_norm [2048]`, plus the block's own `attn_*`/`ffn_*`/`attn_norm`/`post_attention_norm`. **NO `nextn.embed_tokens` and NO `nextn.shared_head_head`** - Qwen3.5's head shares the target's embedding and lm_head, so the converter emits neither and the loader must not ask. Mapping table above is evidence-backed. | Evidence-backed, not inferred | - |
| `G1` | `HfConfigFromGguf` publishes `mtp_num_hidden_layers` into `c.raw` | Config unit test; SACRED spec-OFF byte-identical | - |
| `G2` | `MakeGgufMtpResolver` + name-mapping and dequant-equivalence tests | Resolver unit tests, RED-first | `G0`, `G1` |
| `G3` | Narrow the rejection; wire `LoadQwen3_5MTP` + `AttachMtpDraftWeights` into the GGUF branch | Rejection test; engine loads spec-ON on GGUF | `G2` |
| `G4` | **RUNS and FAILS 2026-07-28 - gate landed, defect OPEN.** `tests/parity/test_qwen35_gguf_spec_decode.cpp` drives the real 2B GGUF twice (spec-OFF then spec-ON) on CPU. It found and fixed one real loader defect (`fc.nk` unset: shape checks passed, the draft forward refused with "fc must be raw bf16 [H,2H]"), then surfaced a SECOND, still-open failure: **spec-ON diverges from spec-OFF while acceptance is HIGH (13 proposed / 10 accepted)**. Greedy MTP is exactness-preserving, so this is a genuine correctness defect. NOT yet attributed - see `G4a`. Needs `VT_GDN_STATE_BF16=0` on CPU (`causal_conv1d_spec_update` requires f32 conv state off CUDA) | Gates 2 and 4 | `G3` |
| `G4a` | **ANSWERED 2026-07-28 by bisect - the defect is NOT in this row.** Killing the drafter outright (zeroing `fc`, so every proposal is garbage) gives **23 proposed / 0 accepted** and *byte-identical output to the live-drafter run*, still diverging from spec-OFF. With ZERO drafts accepted the emitted sequence must be the target's own greedy sequence; it is not. So enabling speculation changes the TARGET's forward on CPU, independently of the head. This row's loader is EXONERATED (and, with a live head, earns 10/13 acceptance). Tracked below as `CPU-SPEC-DIVERGENCE`, which is a pre-existing engine defect this row merely surfaced | Attribution complete, no safetensors download needed | `G4` |

`G0` is deliberately a work row, not an assumption. `G1` is independently
landable and inert.

## Open defect surfaced by this row: `CPU-SPEC-DIVERGENCE`

**Not caused by this row, and not fixed by it.** Turning speculative decoding ON
changes the target's own greedy output on CPU even when NO draft is ever
accepted. Established by bisect, not inference: with the head zeroed
(23 proposed / 0 accepted) the spec-ON continuation is byte-identical to the
live-head spec-ON continuation and still differs from spec-OFF.

    spec-OFF          " Paris.\nA. True\nB. False..."      {11751, 13, 198, 32, ...}
    spec-ON (live)    " Paris is the capital of France..."  {11751, 369, 279, 6511, ...}
    spec-ON (dead)    identical to spec-ON (live)           {11751, 369, 279, 6511, ...}

Why this is engine-level and not loader-level: with zero accepted drafts the
verify path must emit exactly the target's argmax at every position. It does not.
So the target's forward or its state bookkeeping differs under the widened
speculative KV / conv cache.

Scope note: there is NO CPU spec-decode gate anywhere in the tree. `SPEC-MTP`
I5d/I5e/I6/I7 were all gated on GB10, so this is plausibly the first end-to-end
CPU spec run, and the divergence is plausibly as old as the widened-cache work.

**First lead RAISED AND REFUTED (do not re-chase).**
`CausalConv1dSpecUpdateKernel` (`src/vt/cpu/cpu_ops.cpp:1010+`) computes
`off = num_accepted_tokens[i] - 1` and guards the state read with
`src < state_len` only, with no `src >= 0`, so a zero `nat` would read
`srow[-1]`. It cannot: `src/vllm/v1/worker/gpu/runner.cpp:1187` clamps
`num_accepted_tokens[i] = ns > 1 ? ns : 1`, so `nat >= 1` and `off >= 0` always.
The missing `>= 0` guard is a latent robustness gap, not this bug.

**Where to look next.** The op-level CPU suites PASS (`test_ops_gdn` 1825,
`test_qwen3_5_gdn_spec_routing` 12), so the kernels are bit-exact and the defect
is in what the runner FEEDS them. The spec-only branch that has no non-spec
counterpart is the GDN metadata spec overload,
`src/vllm/v1/worker/gpu/runner.cpp:896-925`: `num_decode_draft_tokens` (-1 for a
non-spec row, else the scheduled draft count), the PREVIOUS step's
`num_accepted_tokens`, and the k+1 GDN state-slot remap behind `gdn_bt`. That
triple is what changes the target's own recurrent state under speculation, which
is exactly the observed symptom (target output moves with speculation ON even at
zero acceptance). Suggested next probe: assert spec-ON vs spec-OFF equality of
the GDN state itself after step 1 on a 1-token prompt, which localises to the
state feed without needing a full generation.

**Reproduction:**
`VT_GDN_STATE_BF16=0 VLLM_MTP_GGUF_MODEL=<head-carrying .gguf> ./build-cpu/tests/test_qwen35_gguf_spec_decode`
(CPU `causal_conv1d_spec_update` refuses bf16 conv state, hence the env var.)

## Risks/decisions

- **RISK (highest): the test asset may not exist publicly.** Most published
  Qwen3.5/3.6 GGUFs may have been converted before the MTP mixin landed, or with
  `--no-mtp`. If no head-carrying GGUF can be obtained, `G0` must produce one
  locally from the safetensors gate checkpoint, which needs a llama.cpp
  converter run and disk for a second full export. If that is not affordable,
  the row stays `BLOCKED` at `G0` rather than shipping an ungated mapping. Do
  NOT infer the layout from names alone and call it done.
- **RISK: two MTP layouts, one loader.** Our `LoadQwen3_5MTP` is written to the
  Qwen3.5 `mtp.*` layout; the GGUF spelling is DeepSeek's. The remapper is
  purely nominal per `conversion/qwen.py`, but that is a claim about the
  CONVERTER, not about every producer. A GGUF written by another tool under the
  same `nextn` names could carry genuinely DeepSeek-shaped tensors. Mitigation:
  `RequireShape` on every resolved tensor (the safetensors path already does
  this at `qwen3_5_mtp.cpp:317`), so a layout mismatch fails loudly at load.
- **RISK: quantized head accuracy.** The head's tensors are quantized at
  whatever the export used. Dequant-to-bf16 keeps the compute path identical to
  safetensors but costs memory; computing on blocks would be faster and is what
  the trunk does for matmul weights via `GgufLoadPolicy`. DECISION for the first
  cut: dequant to bf16, because it makes gate 3 (cross-format agreement)
  meaningful at F16 and keeps the diff to the resolver. Revisit under a
  follow-on row if the head shows up in a profile.
- **RISK: a low-bit head may not be worth speculating with.** Acceptance rate is
  the whole value of MTP; a Q4 head could accept so rarely that spec-ON is
  SLOWER than spec-OFF. Gate 4 measures this. If acceptance collapses at low
  bit-width, the honest outcome is a documented minimum quantization, not a
  silent capability claim.
- **DECISION: `mtp_only` drafter GGUFs are out of scope.** llama.cpp can emit a
  GGUF containing only the head (`conversion/qwen.py`, `mtp_only`). Consuming
  one means a SECOND model handle plus a target/draft pairing contract, which is
  the `dflash` problem shape, not this one. Deliberately deferred; if wanted, it
  is a follow-on row that reuses `G2`'s resolver.
- **DECISION: no ABI change.** `speculative_config` already carries this. A
  caller that works on safetensors works unchanged on GGUF once this lands,
  which is the point.
- **NON-RISK, stated to close it:** `ngram` over GGUF already works and is
  untouched; the current rejection never covered it.
