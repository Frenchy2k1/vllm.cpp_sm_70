# Spike: DFlash speculative decoding from GGUF (`SPEC-DFLASH-GGUF`)

Stable row: `SPEC-DFLASH-GGUF` (`.agents/engine-matrix.md`).
Depends on: `SPEC-DFLASH`, [dflash-spec-decode.md](dflash-spec-decode.md);
sibling: [gguf-mtp-spec-decode.md](gguf-mtp-spec-decode.md) (`SPEC-MTP-GGUF`).

**Row-ID namespace.** This spike's work rows are `GD0`-`GD8`. They are NOT the
`D0`-`D13` of [dflash-spec-decode.md](dflash-spec-decode.md), which are that
track's own rows for the safetensors DFlash engine (landed through D13: paged
draft-KV + capture-safe CUDA graph, 0.978x vLLM, `SPEC-DFLASH` `DONE`). Renamed
after the collision was spotted; do not renumber back.

**Relationship to the landed DFlash track.** That work is merged and is the
BASELINE this row builds on, not a parallel effort: it contains no GGUF handling
whatsoever (`grep -i gguf` over its spec returns nothing), so the draft SOURCE is
the one axis it never varied. The three loader seams this row must change
(`ResolveDflashDraftDir` `model_loader.cpp:116`, `MakeDflashDraftConfig` `:173`,
`LoadDflashDraft` `:197`, still typed on `std::vector<SafetensorsFile>`) are
unchanged by D13 and verified against post-merge `main`.

Upstream reference pin for this spike: llama.cpp `origin/master` as fetched
2026-07-28 (tag era `b10158`). This matters: a local checkout at `237ad9b9`
(2026-07-01, 334 commits behind) has NO dflash GGUF contract at all, and reading
it would lead to the wrong conclusion that one must be invented. It must not.

## Scope

Make `--speculative-config '{"method":"dflash","model":...}'` work when the
DRAFT is a `.gguf`, and (second, harder) when the TARGET is a `.gguf` too.

Two independent axes, deliberately separated because they have very different
costs:

- **A. GGUF draft, safetensors target.** The draft checkpoint is a `dflash`-arch
  GGUF; the target stays safetensors. Contained: one new weight resolver plus a
  config source swap.
- **B. GGUF draft, GGUF target.** Additionally removes the safetensors-typed
  dependency on the target's bf16 `embed_tokens` + `lm_head`, which the draft
  SHARES. This is where the real work is.

IN SCOPE: both axes, sequenced A then B, with A independently shippable.

OUT OF SCOPE: MTP over GGUF (`SPEC-MTP-GGUF`); EAGLE3, which llama.cpp also
carries as a GGUF arch and we do not implement at all; any change to the DFlash
propose/verify algorithm.

## Upstream chain

llama.cpp `origin/master` (the producer contract; vLLM has no GGUF path):

- `gguf-py/gguf/constants.py:547` — `MODEL_ARCH.DFLASH`; `:1151` — its GGUF
  architecture string `"dflash"`.
- `gguf-py/gguf/constants.py:4350` — the DFLASH tensor set:
  `OUTPUT_NORM`, `ATTN_NORM`, `ATTN_Q/K/V/OUT`, `ATTN_Q_NORM`, `ATTN_K_NORM`,
  `FFN_NORM`, `FFN_GATE/DOWN/UP`, `FC`, `ENC_OUTPUT_NORM`.
  **Note what is ABSENT: no `TOKEN_EMBD`, no `OUTPUT`.** The draft shares the
  target's embedding and lm_head, which is exactly the sharing our
  `LoadDflashDraft` already implements. The GGUF contract and our loader agree
  on the model's shape.
- `gguf-py/gguf/tensor_mapping.py:1303` — `MODEL_TENSOR.FC` <- HF `model.fc`;
  `:1297-1301` — `ENC_OUTPUT_NORM` <- HF `model.hidden_norm` (`# dflash`).
  Tensor names on disk: `fc`, `enc.output_norm`, `output_norm`, and the standard
  `blk.N.attn_*` / `blk.N.ffn_*`.
- Metadata keys: `dflash.target_layers` (the `target_layer_ids` list) and
  `dflash.target_hidden_size`; plus the standard `dflash.block_count`,
  `.embedding_length`, `.attention.head_count[_kv]`, `.attention.key_length`,
  `.feed_forward_length`, `.rope.freq_base`, `.attention.layer_norm_rms_epsilon`.
- `conversion/qwen.py:351` — the mask token rides the STANDARD tokenizer KV via
  `add_mask_token_id(...)`, not a dflash-specific key.
- `convert_hf_to_gguf.py` `--target-model-dir` — required when converting a
  standalone draft (EAGLE3 / DFlash), because the draft's GGUF must be populated
  with target metadata (tokenizer, hidden size, layer count). This is the
  producer-side reason `dflash.target_hidden_size` exists.

## `GD0` verified dump (the implementation contract)

From the real Q4_K_M draft. `H = 5120`, `num_taps = 5`.

    general.architecture                     dflash
    dflash.block_count                       5        -> num_hidden_layers
    dflash.embedding_length                  5120     -> hidden_size
    dflash.feed_forward_length               17408    -> intermediate_size
    dflash.attention.head_count              32       -> num_attention_heads
    dflash.attention.head_count_kv           8        -> num_key_value_heads
    dflash.attention.key_length              128      -> head_dim (= rotary_dim)
    dflash.rope.freq_base                    1e7      -> rope_theta
    dflash.attention.layer_norm_rms_epsilon  1e-6     -> rms_norm_eps
    dflash.attention.sliding_window          2048     -> sliding_window
    dflash.attention.sliding_window_pattern  [T,T,T,T,F]  -> layer_types
    dflash.block_size                        16       -> raw["block_size"]
    dflash.target_layers                     [2,17,32,47,62] -> raw["dflash_config"]["target_layer_ids"], num_taps=5
    tokenizer.ggml.mask_token_id             248070   -> raw["dflash_config"]["mask_token_id"]

    fc.weight              ggml [25600, 5120] -> torch [5120, 25600] = [H, H*num_taps]  Q4_K
    enc.output_norm.weight [5120] F32   -> hidden_norm
    output_norm.weight     [5120] F32   -> final_norm
    blk.{0..4}.{attn_q,attn_k,attn_v,attn_output,attn_q_norm,attn_k_norm,attn_norm,
                ffn_norm,ffn_gate,ffn_up,ffn_down}.weight
    58 tensors total, 5 blocks. NO token_embd, NO output.

Three implementation facts this pins down:

1. **`fc` is the VERBATIM path, not the transposing one.** `GgufTensorInfo::shape`
   is torch `[N, K]`, so the file's `[25600, 5120]` arrives as `[5120, 25600]` =
   `[H, H*num_taps]`, which is exactly what `Qwen3DFlashWeights::fc` wants as
   raw-NK. Same trap as `SPEC-MTP-GGUF`, same answer - and `fc.nk` must be SET,
   which is the defect that cost that row a debug cycle.
2. **`mask_token_id` 248070 matches** the value `qwen3_dflash.h:90` documents for
   the z-lab 27B, independently confirming the whole mapping.
3. **No `vocab_size` KV and no embedding/lm_head tensors.** `draft_vocab_size`
   must come from the TARGET's `lm_head` (which `LoadDflashDraft` already does),
   NOT from the draft. `MakeDflashGgufConfig` must therefore leave `vocab_size`
   to the caller rather than VT_CHECK a missing key.

## Two off-by-one traps, RESOLVED from the producer's source (`GD1` prerequisite)

Both are invisible to name/shape validation and point in OPPOSITE directions.
Settled by reading llama.cpp `origin/master` `conversion/qwen.py`, not by
inspecting values (a value-distribution check was run first and was AMBIGUOUS -
draft and trunk norms both cluster near ~1).

**1. Norms are stored RAW. Do NOT apply the `-1` un-shift.**
The `+1` shift lives in `Qwen3NextModel.modify_tensors`
(`elif name.endswith("norm.weight") ...: data_torch = data_torch + 1`).
`class DFlashModel(Qwen3Model)` inherits from `Qwen3Model`, NOT
`Qwen3NextModel`, and overrides only `set_vocab`, `set_gguf_parameters` and
`filter_tensors` - there is NO `modify_tensors` override, and `Qwen3Model`'s own
`modify_tensors` does not touch norms.

So the DFlash draft is the OPPOSITE of the Qwen3.5 trunk and the MTP head: those
need `OwnNormMinus1`, this one needs a plain load. Copying the `SPEC-MTP-GGUF`
approach - the natural move, since it is the sibling row - would compile, load,
produce valid-looking logits, and be quietly wrong on every norm.

**2. `dflash.target_layers` is written OFFSET BY +1.**
`set_gguf_parameters` does `extract_layer_ids = [i + 1 for i in target_layer_ids]`
before `add_target_layers(...)`. So the file's `[2, 17, 32, 47, 62]` means HF
`target_layer_ids = [1, 16, 31, 46, 61]`. `MakeDflashGgufConfig` MUST subtract 1
when synthesizing `raw["dflash_config"]["target_layer_ids"]`, or every
hidden-state tap reads the wrong target layer - silently, because `num_taps`
(5) stays correct and every shape check still passes.

Consequence for `GD2`: the DFlash GGUF loader must NOT be written by analogy to
the MTP one. It shares the dequant/quantization-routing helpers but differs on
the norm convention, and its config differs on the tap indices.

## Our baseline

- **The DFlash weight loader is already resolver-shaped**, exactly like the MTP
  one: `LoadQwen3DFlash(const TensorResolver& get, const HfConfig&, ...)`
  (`include/vllm/model_executor/models/qwen3_dflash.h:104`), with a
  safetensors-shards overload at `:106`. A GGUF source needs only a second
  resolver.
- **The weight struct maps cleanly onto the GGUF tensor set**
  (`qwen3_dflash.h:79-92`): `fc` <- `fc`, `hidden_norm` <- `enc.output_norm`,
  `final_norm` <- `output_norm`, `layers` <- `blk.N.*`, and
  `embed_tokens` / `lm_head` come from the TARGET (which is why the GGUF DFLASH
  arch omits them). `num_taps` = `len(target_layer_ids)` <- `dflash.target_layers`.
  `mask_token_id` <- the tokenizer mask KV.
- **Blockers, all in `src/vllm/entrypoints/model_loader.cpp`:**
  - `MakeDflashDraftConfig` (`:172`) builds the draft `HfConfig` by reading
    `draft_dir/config.json` with `nlohmann::json` and `c.at(...)` on
    `dflash_config.{mask_token_id,target_layer_ids}`, `block_size`,
    `layer_types`, `sliding_window`. A GGUF draft has no `config.json`.
  - `ResolveDflashDraftDir` (`:115`) probes for `config.json` to decide a path
    is a checkpoint, so it does not recognise a `.gguf` FILE as a draft at all.
  - `LoadDflashDraft` (`:196`) is typed
    `(const SpeculativeConfig&, const std::vector<SafetensorsFile>& target_shards)`
    and pulls the shared tensors with
    `LoadNamedBf16(target_shards, "model.language_model.embed_tokens.weight", ...)`
    and `"lm_head.weight"`. This is the axis-B blocker: the parameter type
    itself, not just the lookup.
  - The GGUF branch rejection (`:717-723`) currently covers `mtp` AND `dflash`.
- `LoadShards(draft_dir)` assumes a directory of `*.safetensors`.

## Port map

**Axis A (GGUF draft, safetensors target):**

| # | File | Change |
|---|---|---|
| A1 | `src/vllm/entrypoints/model_loader.cpp:115` (`ResolveDflashDraftDir`) | Accept a path that IS a `.gguf` file (and a directory containing exactly one), alongside today's `config.json` probe and HF-cache snapshot search. Return a discriminated result (dir vs gguf file) rather than a bare string. |
| A2 | new `src/vllm/model_executor/models/qwen3_dflash_gguf.cpp` (+ header) | `MakeDflashGgufConfig(const GgufFile&)` -> `HfConfig`, the GGUF counterpart of `MakeDflashDraftConfig`: read `dflash.*` KVs, synthesize `raw["dflash_config"]["target_layer_ids"]` from `dflash.target_layers` and `raw["block_size"]`, so downstream code that reads `config.raw` is unchanged. VT_CHECK every required key with the GGUF key name in the message. |
| A3 | same | `MakeDflashGgufResolver(const GgufFile&)` -> `TensorResolver` + owning dequant cache, mapping our HF-style names onto `fc` / `enc.output_norm` / `output_norm` / `blk.N.*` and dequantizing to bf16. Same shape as `SPEC-MTP-GGUF` `G2`; factor the cache helper so both rows share it. |
| A4 | `src/vllm/entrypoints/model_loader.cpp:196` (`LoadDflashDraft`) | Branch on A1's result: GGUF draft -> A2 + A3 + `LoadQwen3DFlash(get, ...)`; directory -> today's path unchanged. Target-shared tensors still come from `target_shards`. |
| A5 | `src/vllm/entrypoints/model_loader.cpp:717-723` | Unchanged for axis A: a safetensors TARGET does not hit the GGUF branch. |

**Axis B (GGUF target as well):**

| # | File | Change |
|---|---|---|
| B1 | `model_loader.cpp` | Introduce a small `SharedHeadSource` abstraction returning bf16 `embed_tokens` + `lm_head` from EITHER safetensors shards or a `GgufFile` (dequantizing `token_embd.weight` / `output.weight`, with the `output.weight`-absent tied-embedding fallback the trunk GGUF loader already handles). Re-type `LoadDflashDraft`'s second parameter to it. |
| B2 | `model_loader.cpp:717-723` | Drop `dflash` from the GGUF rejection once B1 lands. |
| B3 | GGUF branch | Wire `maybe_load_dflash`'s equivalent into the GGUF path (today it is only reachable from the safetensors branch). |

No ABI change on either axis.

## Tests to port

No upstream tests exist to port; gates are ours, shaped like the existing DFlash
gates.

| Test | Location | What it pins |
|---|---|---|
| Config from GGUF KVs | new `tests/vllm/models/test_qwen3_dflash_gguf.cpp` | `dflash.target_layers` -> `num_taps` and `raw["dflash_config"]["target_layer_ids"]`; a missing required KV throws naming that KV. Synthetic in-memory `GgufFile`. RED-first. |
| Resolver name mapping | same | Every name `LoadQwen3DFlash` requests resolves; `fc`/`enc.output_norm`/`output_norm` land on the right fields. RED-first via a deliberately swapped norm. |
| Draft-path discrimination | same or the model-loader test | `ResolveDflashDraftDir` accepts a `.gguf` file, a dir with `config.json`, and an HF-cache snapshot; rejects a dir with neither, with both the reference and the searched locations in the message. |
| Cross-format draft equivalence | `tests/parity/` | The SAME draft, loaded from safetensors and from an F16 GGUF, produces bit-identical `Qwen3DFlashWeights` (compare tensor bytes). This is the strongest cheap gate and needs no GPU. |
| Axis-A token gate | new `tests/parity/test_qwen36_dflash_gguf_draft.cpp`, modelled on `tests/parity/test_qwen36_spec_decode.cpp` | GGUF-draft spec-ON == safetensors-draft spec-ON, token-for-token, c1 greedy; acceptance within noise. |
| Axis-B token gate | same file, second case | GGUF-target spec-ON == GGUF-target spec-OFF at c1; plus acceptance > 0. |
| Shared-head equivalence (B) | unit | `SharedHeadSource` over an F16 GGUF yields bf16 bit-identical to the safetensors path. |

## Gates

1. **Spec-OFF byte-identical (SACRED).** 27B 235/235, 35B 315/315, Coder 138/138.
   Axis B touches the GGUF branch, so this is mandatory there.
2. **Cross-format draft equivalence.** Bit-identical weights from F16 GGUF vs
   safetensors for the same draft. Cheap, CPU-only, and it is what makes the
   token gates interpretable.
3. **Axis-A token identity, c1.** GGUF draft == safetensors draft, token-exact,
   greedy, single request.
4. **Axis-B token identity, c1.** GGUF target spec-ON == spec-OFF.
5. **Acceptance parity.** Within noise of the safetensors-draft rate on the same
   prompts. A collapsed rate means the draft loaded but is wired wrong.
6. **Speed: PENDING, not owed by this row.** Owes a `docs/BENCHMARKS.md`
   disposition, which may be `PENDING` with the reproduction command.

## Dependencies

- `SPEC-DFLASH` and its runner loop, unchanged by this row. **If `SPEC-DFLASH`
  is still oracle-blocked, this row inherits that block for its ORACLE
  comparisons** but not for gates 2-4, which are self-referential (ours vs ours).
- `SPEC-MTP-GGUF` `G2`'s dequant-cache helper. Not a hard dependency; if this
  row lands first it owns the helper and the MTP row reuses it. Sequence
  whichever starts first, do not duplicate the cache.
- A `dflash`-arch GGUF draft asset. **Producing one is NOT required** (see `GD0`):
  pre-converted drafts are published, smallest useful being
  `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` `Q4_K_M` at 1.03 GB. The
  fetch is the only blocked step and needs developer approval per the AGENTS.md
  safe defaults. Required for gates 2-5; NOT required for `GD1`-`GD3`.
- For `GD4`+ only: the matching TARGET (Qwen3.6-27B safetensors for axis A). This
  is the expensive dependency, not the draft.
- llama.cpp at a commit that HAS the DFLASH arch. Pin and record it; a stale
  checkout silently lacks the whole contract.
- The GGUF trunk loader + `gguf_dequant.h`, unchanged.

## Work breakdown

| Row | Work | Gate | Blocked by |
|---|---|---|---|
| `GD0` | **DONE 2026-07-28 - contract CONFIRMED against a real file**, `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` Q4_K_M (1.03 GB, fetched; no local conversion run was needed, contrary to the original plan). Verified dump below. Every name and KV in Upstream chain holds, and two gaps in it are now closed: `dflash.block_size` IS emitted as its own KV, and `dflash.target_hidden_size` is NOT emitted by this converter (llama.cpp declares the key; do not require it) | Evidence-backed | - |
| `GD1` | **DONE 2026-07-28.** `MakeDflashGgufConfig` (`src/vllm/model_executor/models/qwen3_dflash_gguf.cpp`) builds the draft HfConfig from the `dflash.*` KVs, undoing the +1 target-layer offset and rebuilding `raw["dflash_config"]` + `raw["block_size"]` so downstream `config.raw` readers are unchanged. `vocab_size` deliberately left 0 (the draft shares the target's lm_head). | Config unit tests, RED-first | `GD0` |
| `GD2` | **DONE 2026-07-28.** `LoadQwen3DFlashFromGguf` via a `TensorResolver` over dequantized bf16 views, delegating to the EXISTING `LoadQwen3DFlash` so its qkv / gate_up row concatenation is reused unchanged. The resolver seam is correct HERE (unlike the MTP head) precisely because dflash norms are RAW and the draft is small enough to dequant wholesale. | Resolver unit tests; gate 2 | `GD0` |
| `GD3` | **DONE 2026-07-28.** `IsDflashGgufDraft` + the `.gguf` branch in `ResolveDflashDraftDir` / `LoadDflashDraft` (`src/vllm/entrypoints/model_loader.cpp`). Target-shared embed/lm_head still come from `target_shards`, so axis A is complete. | Path-discrimination test | `GD1`, `GD2` |
| `GD4` | Axis-A token + acceptance gate | Gates 3, 5 | `GD3` |
| `GD5` | `SharedHeadSource` + its equivalence test | Gate 7 | `GD4` |
| `GD6` | Drop `dflash` from the GGUF rejection; wire the GGUF branch | Gate 1 | `GD5` |
| `GD7` | Axis-B token gate | Gate 4 | `GD6` |
| `GD8` | Record: STATUS, BENCHMARKS, matrix, ledger | Checkers green | `GD7` |

`GD0`-`GD4` are axis A and independently shippable; the row can legitimately rest
at `PARTIAL` after `GD4`.

## Risks/decisions

- **RISK (highest): the shared-head sharing is a real semantic coupling, not a
  loading detail.** The draft runs the TARGET's `lm_head` over ITS own hidden
  states. On axis B that lm_head is quantized in the target GGUF. Dequantizing
  it to bf16 for the draft, while the target itself computes on quantized
  blocks, means draft and target score with numerically different heads. That is
  not obviously wrong (the draft only PROPOSES; the target verifies) but it will
  move the acceptance rate, and could move it a lot at low bit-width. Gate 5
  measures it. DECISION: treat an acceptance drop here as a finding to document,
  not a bug to hide; if it collapses, the honest output is a documented minimum
  target quantization for dflash.
- **RISK: `dflash.target_layers` must MATCH the target actually being served.**
  The draft taps specific target layers. A draft converted against a different
  target build, or paired with a different target than the one it was made for,
  will load fine and produce garbage proposals. There is no checksum in the GGUF
  contract tying them. DECISION: validate what we can at load
  (`dflash.target_hidden_size` == target `hidden_size`; every id in
  `dflash.target_layers` < target `num_hidden_layers`) and REFUSE on mismatch.
  This is strictly better than the current safetensors path, which does not
  check either.
- **RISK: `mask_token_id` arrives by a different route than the safetensors
  path.** Ours reads `dflash_config.mask_token_id` from `config.json`; the GGUF
  carries it as the standard tokenizer mask KV (`conversion/qwen.py:351`). If a
  given export omits it, `mask_token_id` stays -1 and the block drafter masks
  nothing. `GD1` must VT_CHECK it present rather than defaulting.
- **RISK: my earlier reading was wrong on a stale checkout, and the same trap is
  live for the implementer.** At llama.cpp `237ad9b9` there is no DFLASH arch,
  no dflash tensors and no dflash KVs; only a `--target-model-dir` help string
  mentioning DFlash next to EAGLE3. Anyone who greps a stale tree will conclude
  no contract exists and start inventing one. `GD0` exists to prevent that.
  Record the exact llama.cpp commit used.
- **DECISION: mirror llama.cpp's contract, do not invent one.** The house rule
  is MIRROR upstream. Where our loader wants something the GGUF contract does
  not carry, prefer deriving it from what IS carried (as with
  `target_layer_ids` <- `dflash.target_layers`) over adding a vendor KV. If a
  vendor key becomes unavoidable, that is a decision to record here first.
- **DECISION: axis A before axis B.** Axis A is a contained loader addition with
  a safetensors target as the control, which makes every failure attributable.
  Axis B changes a shared code path under the SACRED gate and should not be
  entangled with first-time resolver bugs.
- **NON-RISK, stated to close it:** the DFLASH GGUF arch omitting `token_embd` /
  `output` is not a gap; it is the contract agreeing with our loader that those
  come from the target.
