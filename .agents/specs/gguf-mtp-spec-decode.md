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
| 2 | new `src/vllm/model_executor/models/qwen3_5_gguf_mtp.cpp` (+ header) | `MakeGgufMtpResolver(const GgufFile&, const HfConfig&)`: returns a `TensorResolver` plus an owning cache. Maps the `mtp.*` names `LoadQwen3_5MTP` asks for onto `blk.{L+i}.nextn.*` / `blk.{L+i}.*`, dequantizing to bf16 on first request and caching the `OwnedTensor` + its `StTensor` view. Inverse of the `conversion/qwen.py:579-586` remapper table above. |
| 3 | same | The four scalar names resolve at block `L`; `mtp.layers.{i}.<rest>` resolves at block `L+i` through the EXISTING trunk stem mapping (`blk.N.<stem>`) so attention/MLP/MoE naming is not duplicated. |
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
| `G0` | Obtain/convert a head-carrying Qwen3.5 GGUF; dump its tensor list and metadata; CONFIRM the `blk.{L}.nextn.*` names and the `L` vs `L+i` placement against the table above. Record the actual dump in this file. | The mapping table is evidence-backed, not inferred | - |
| `G1` | `HfConfigFromGguf` publishes `mtp_num_hidden_layers` into `c.raw` | Config unit test; SACRED spec-OFF byte-identical | - |
| `G2` | `MakeGgufMtpResolver` + name-mapping and dequant-equivalence tests | Resolver unit tests, RED-first | `G0`, `G1` |
| `G3` | Narrow the rejection; wire `LoadQwen3_5MTP` + `AttachMtpDraftWeights` into the GGUF branch | Rejection test; engine loads spec-ON on GGUF | `G2` |
| `G4` | Three-way token gate + acceptance | Gates 2 and 4 | `G3` |
| `G5` | Cross-format agreement at F16, or an explicit NOT APPLICABLE | Gate 3 | `G4` |
| `G6` | Record: `docs/STATUS.md`, `docs/BENCHMARKS.md` (PENDING or measured), matrix row to `GATING`/`DONE`, ledger | Checkers green | `G5` |

`G0` is deliberately a work row, not an assumption. `G1` is independently
landable and inert.

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
