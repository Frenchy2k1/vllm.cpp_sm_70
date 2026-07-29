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

## A THIRD trap, found by `GD4` and only findable by GENERATING

The two above are load-time conventions and both were caught by unit tests. The
third is not: it is a field the GGUF contract legitimately cannot supply, whose
absence is invisible until the draft's first forward.

**`vocab_size` must be back-filled from the TARGET; leaving it 0 is fatal.**
`MakeDflashGgufConfig` deliberately leaves `HfConfig::vocab_size` at 0 (`GD1`),
which is right: the DFLASH arch carries no vocab KV and no `token_embd`, because
the draft SHARES the target's table. But the draft's forward sizes its embedding
lookup as `ResidentWeight(d, weights.embed_tokens, {config.vocab_size, H})`
(`src/vllm/model_executor/models/qwen3_dflash.cpp:245,477,1008,1043`), so a 0
there is an EMPTY table over a perfectly good buffer, and the FIRST propose
throws `vt: cuda embedding: empty table (vocab 0) with nonempty ids`
(`src/vt/cuda/cuda_ops.cu:648`). The safetensors draft never met this because its
`config.json` declares `vocab_size`.

Every load-time assertion passed: the weights were loaded, the shared bf16
embed/lm_head were found, `draft_vocab_size` was set from `lm_head`, and the 47
`GD1`-`GD3` assertions were green - including one asserting `vocab_size == 0`,
which is correct for the config-building step and says nothing about the loader
that consumes it. **The fix belongs in `LoadDflashDraft`, not in
`MakeDflashGgufConfig`**: take the row count from the TARGET tensor actually
being indexed (`weights.embed_tokens.shape[0]`) rather than from any declared
number, so the view and the buffer cannot disagree.

The general lesson this row now carries alongside the MTP row's: a loader gated
only at load level is not gated. Here it was not even a wrong VALUE - it was a
value nothing had a reason to write.

## The axis-A acceptance delta, root-caused in WEIGHT SPACE (`GD9`, 2026-07-29)

The production-build re-measurement left one open item: on the 48-token prompt
the `Q4_K_M` GGUF draft measures **46/112** where the bf16 safetensors draft
measures **47/96**, with the emitted tokens IDENTICAL. Two candidate causes had
to be separated - ordinary quantization cost, or a structural defect in our GGUF
draft path - and "plausible statistical explanation" is exactly the shape the
`fc.nk`, `(w+1)`-norm and `vocab_size` defects this row already produced also
had. So it was settled by measurement, not by argument.

**First, read the accept rule.** `include/vllm/v1/spec_decode/rejection_sampler.h`
pins ACCEPT-IFF-EQUAL, STOP-AT-FIRST-MISMATCH, and the mismatch position emitting
the TARGET's argmax. A consequence worth stating plainly, because it changes what
each half of the bar is worth: under greedy verification the emitted token stream
is the target's own greedy continuation NO MATTER WHAT THE DRAFT PROPOSES. Token
identity across two drafts is therefore a property of the VERIFIER, not evidence
about either draft. The accept counts are the only channel that carries
information about draft quality - which is why they diverged while the tokens did
not, and why "one extra 16-wide propose block" is not a block-accounting
off-by-one but the arithmetic consequence of accepting less per block (`proposed`
is `16 x nblocks` in every arm; 144, 96 and 112 are all exact multiples).

**The bar's own premise did not hold.** The assertion block reads "Same weights,
two containers, one target ... both arms run the identical code path"
(`tests/parity/test_qwen27_dflash_spec_decode.cpp`). Against a `Q4_K_M` file the
first clause is false: it is not the same weights, it is a 4-bit re-encoding of
them. The exact accept-count bar was pointed at an asset that never satisfied its
stated premise.

**What was measured (CPU, no GPU needed).**

1. **An UNQUANTIZED GGUF of this draft exists**, and through our loader it is
   BYTE-IDENTICAL to the safetensors draft, 58/58 tensors (gate 2 above). Every
   structural way the GGUF arm could differ - a mis-mapped name, a transposed
   `fc`, a stray `+1` norm shift, a mis-scaled tensor - is ruled out over the
   SAME `MapName` / dequant-to-bf16 / `LoadQwen3DFlash` path a `Q4_K_M` draft
   takes. The only thing left that a quantized draft can change is the
   quantization.
2. **Our Q4_K and Q6_K dequantizers agree with the producer's reference to the
   bit.** Dequantizing the real draft's `fc.weight` (Q4_K, 131,072,000 elems),
   `blk.0.attn_q.weight` (Q4_K, 20,971,520) and `blk.2.ffn_down.weight` (Q6_K,
   89,128,960) through `DequantGgufRowToBf16` and comparing against `gguf-py`'s
   `gguf.quants.dequantize` gives **zero differing bf16 values, max|d| = 0** on
   all three. So the dequant is not degrading the draft below what the file
   encodes.
3. **The error is ordinary quantization error and it is UNIFORM.** Mean
   `|w_gguf - w_st| / mean|w_st|` over the whole published ladder, all 58
   tensors, against the safetensors draft:

       BF16     0            (58/58 bit-identical)
       Q8_0     5.6e-3       (22/58 bit-identical: the F32 norms)
       Q6_K     1.85e-2      (22/58)
       Q5_K     3.85e-2      (22/58)
       Q4_K_M   7.6e-2       (22/58)

   Monotone in bit-width, and the spread WITHIN each level is negligible (Q8_0
   ranges 5.53e-3 to 5.63e-3 across every matmul tensor). There is no outlier
   tensor, no tensor out of family, and the norms - which the two off-by-one
   traps above are about - are bit-identical at every level because llama.cpp
   keeps them F32.

**Config delta between the two arms, enumerated and excluded.** After load the
two sources converge on the same struct and the same code, so the only remaining
arm-dependent input is the draft `HfConfig`. Field by field against the z-lab
`config.json` (`hidden_size` 5120, layers 5, heads 32/8, `head_dim` 128,
`rope_theta` 1e7, `intermediate_size` 17408, `sliding_window` 2048,
`layer_types` `[SWA x4, full]`, `block_size` 16, `target_layer_ids`
`[1,16,31,46,61]`, `mask_token_id` 248070) the GGUF-derived config matches
exactly. Two fields differ and neither can move a proposal:

- `rms_norm_eps`. `config.json` says `1e-06`; the GGUF stores it as f32, so it
  reads back `9.999999974752427e-07`. Relative difference 2.5e-9, which is ~50x
  below f32 epsilon and ~5 orders below bf16 resolution.
- `vocab_size`. The safetensors arm takes the draft's declared 248320; the GGUF
  arm has no vocab KV and is back-filled from the TARGET's `embed_tokens` rows
  (`GD4`). It only sizes the shared embedding VIEW; the value the draft's argmax
  actually ranges over is `draft_vocab_size`, which BOTH arms overwrite from the
  target's `lm_head` rows (`model_loader.cpp:312`), so they agree by
  construction.

**Category: (a), expected quantization cost.** A structural defect is eliminated
in the loader, in the dequantizer, and in the config; the residual difference is
a 7.6e-2 mean relative perturbation of the draft's weights, which is what
`Q4_K_M` is. The effect size is also the right shape for that cause and the wrong
shape for a defect: on one prompt it costs exactly nothing (24 tokens, both
drafts 15/144) and on the other it costs ONE acceptance out of 47. A mis-loaded
draft collapses acceptance; it does not shave one.

**Consequence for the bar.** Split it by what the arms actually vary:

- **cross-FORMAT** (safetensors vs an UNQUANTIZED GGUF): same weights to the
  byte, so exact tokens AND exact `proposed`/`accepted` is a real invariant and
  is the right bar. Gate 2 now proves the premise at load; the end-to-end form
  is the `bf16_vs_st` arm below.
- **cross-QUANTIZATION** (safetensors vs `Q4_K_M`): different weights by
  construction. Tokens stay EXACT - that is guaranteed by the verifier, not
  hoped for - and the accept counts get a band, the way the landed `SPEC-DFLASH`
  golden arm's already are (`abs(acc - want) <= 4`).

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
| Cross-format draft equivalence | LANDED 2026-07-29 in `tests/vllm/models/test_qwen3_dflash_gguf.cpp` third case (a unit, not a parity, test: it needs no engine) | The SAME draft, loaded from safetensors and from an UNQUANTIZED GGUF, produces bit-identical `Qwen3DFlashWeights` (compare tensor bytes). The strongest cheap gate and needs no GPU, as the spike said - it was simply believed unavailable. Asset-gated on `VLLM_DFLASH_GGUF_BF16_MODEL` + `VLLM_DFLASH_ST_DIR`. |
| Axis-A token gate | `tests/parity/test_qwen27_dflash_spec_decode.cpp` second case (NOT a new file: the existing one is the same engine path and only lacked a draft-source override) | MODE-MATCHED and CROSS-FORMAT, both drafts in ONE process: `VLLM_DFLASH_DRAFT` vs `VLLM_DFLASH_DRAFT_B` must produce identical DFlash-ON tokens AND identical accepted/proposed, each with nonzero acceptance. Companions: near-tie-robust vs spec-OFF (`D5` form), a spec-OFF self-reproducibility control, and a target margin sweep. **Not `spec-ON == spec-OFF`**: that is the MTP bar, and `SPEC-DFLASH` `D5` measured vLLM's own DFlash-ON as token-different from vLLM's own spec-OFF on 3 of 4 prompts. |
| Axis-B token gate | same file, second case | GGUF-target spec-ON == GGUF-target spec-OFF at c1; plus acceptance > 0. |
| Shared-head equivalence (B) | unit | `SharedHeadSource` over an F16 GGUF yields bf16 bit-identical to the safetensors path. |

## Gates

1. **Spec-OFF byte-identical (SACRED).** 27B 235/235, 35B 315/315, Coder 138/138.
   Axis B touches the GGUF branch, so this is mandatory there.
2. **Cross-format draft equivalence (weights). MET 2026-07-29.** The earlier
   `NOT APPLICABLE` rested on a factual error about the asset: it read "the only
   published one is `Q4_K_M`", and `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp`
   in fact publishes a whole ladder - `BF16` (3.47 GB, ggml type 30), `Q8_0`,
   `Q6_K`, `Q5_K` and `Q4_K_M`. With the UNQUANTIZED member the gate is exactly
   as the spike wrote it and it passes: `tests/vllm/models/test_qwen3_dflash_gguf.cpp`
   third case, **302/302 assertions, exit 0**, every one of the 58 tensors
   BYTE-IDENTICAL between `LoadQwen3DFlashFromGguf(BF16.gguf)` and
   `LoadQwen3DFlash(z-lab safetensors shards)` - `fc`, both norms, all five
   layers' concatenated `qkv_proj`/`gate_up_proj`, `o_proj`, `down_proj`, the
   per-head `q_norm`/`k_norm`, `nk` flags and attention modes included. It is
   NOT a vacuous pass: pointed at the `Q4_K_M` file the same case FAILS with 21
   of 302 assertions red (exactly the 21 quantized matmul tensors; the 22 F32
   norms stay bit-equal), exit 1.
3. **Axis-A cross-format equivalence, c1. TOKEN half MET; ACCEPT-COUNT half RED
   as of the 2026-07-29 production-build re-measurement.** The DFlash-ON greedy
   continuation from the Q4_K_M GGUF draft is still token-for-token identical to
   the one from the bf16 safetensors draft on BOTH prompts, so the TOKEN half
   stands. The proposed/accepted counts no longer agree: at 24 tokens both drafts
   are 15/144 and the gate passes 17/17, at 48 tokens the GGUF draft is 46/112
   against the safetensors draft's 47/96 and the gate FAILS 15/17, exit 1,
   reproduced 3 of 3 runs. The 2026-07-28 "MET" reading came from a build
   configured without `-DVLLM_CPP_CUTLASS_DIR` and without `-DVLLM_CPP_TRITON=ON`,
   i.e. running the emulation fp4 GEMM, which masked the difference. **Note the
   restatement:** the original wording of this gate was "GGUF draft ==
   safetensors draft", which is what is measured; an earlier attempt to gate on
   `spec-ON == spec-OFF` instead was wrong for DFlash and is recorded under `GD4`
   below. **Open question this now raises:** whether the bar should stay
   accept-count-EXACT or become accept-count-BANDED the way the landed
   `SPEC-DFLASH` golden arm's is (`abs(acc - want) <= 4`, which 46 vs 47 would
   satisfy). **ANSWERED 2026-07-29 by the `GD9` root cause above, in weight
   space:** the exact bar is right for a cross-FORMAT arm and wrong for the
   cross-QUANTIZATION arm it was pointed at, because its own premise ("same
   weights, two containers") is false against a `Q4_K_M` file and true against
   the `BF16` one. The end-to-end confirmation (`bf16_vs_st` at 48 tokens must
   read exactly `47/96`, `Q4_K_M` a band around it) is NOT YET RUN - dgx.casa
   dropped off the network mid-session on 2026-07-29 and did not return, so no
   GPU number was produced this round and none is claimed. The RED stands.
4. **Axis-B token identity, c1. MET 2026-07-28, RE-CONFIRMED 2026-07-29 on a
   production build**, in its strict form: on the GGUF target the DFlash-ON
   continuation is token-for-token identical to that same target's spec-OFF,
   24/24, with acceptance 14/160 and the cross-target spec-OFF divergence still
   at index 4, every number unchanged. Note this is a WITHIN-target bar. A
   CROSS-target one was considered and rejected on evidence: the two containers
   diverge at index 4 without any speculation, so it would gate their arithmetic
   rather than this row (see the axis-B section above).
5. **Acceptance parity. NOT MET for axis A as of 2026-07-29.** The 2026-07-28
   reading ("EXACTLY equal on both prompts, 20/80 and 42/96, so Q4_K_M costs this
   draft nothing") was measured on the defective build and is void. On a
   production build the 24-token prompt is 15/144 for both drafts but the
   48-token prompt is 46/112 for the GGUF draft against 47/96 for the bf16 one,
   so the spike's flagged risk is REAL: the 4-bit draft needs one extra 16-wide
   propose block and lands one fewer acceptance, while emitting identical tokens.
   **ROOT-CAUSED IN WEIGHT SPACE 2026-07-29** (the `GD9` section above):
   category (a), ordinary `Q4_K_M` cost. A structural defect is eliminated in the
   loader (gate 2, 58/58 byte-identical over the same code path), in the
   dequantizer (bit-equal to `gguf-py` on the real tensors) and in the config
   (only `rms_norm_eps` at 2.5e-9 relative and a benign `vocab_size` provenance
   differ). What remains is the end-to-end confirmation on GB10; it is the
   reason axis A is still not closed.
   **For axis B: MEASURED, nonzero, and lower** - 14/160 vs the safetensors
   target's 15/144 same-draft same-prompt on the production build (the older
   20/80 comparison figure is void) - with the cause established as the GGUF target's bf16 compute
   path and the shared head EXCLUDED by a byte comparison. Recorded as a
   finding, per the risk decision below.
6. **Speed: PENDING, not owed by this row.** Owes a `docs/BENCHMARKS.md`
   disposition, which may be `PENDING` with the reproduction command.
7. **Shared-head equivalence (B). MET 2026-07-28**, twice over: three synthetic
   `LoadGgufSharedEmbedAndHeadBf16` unit cases (untied head really comes from
   `output.weight`, the tied fallback, the `nk` orientations, a missing
   `token_embd` refused) plus the byte comparison on the REAL 27B pair. The
   spike wrote this gate as "over an F16 GGUF"; the real asset stores the pair
   as ggml BF16, which is a strictly stronger substrate for the same claim.

## Dependencies

- `SPEC-DFLASH` and its runner loop, unchanged by this row. **If `SPEC-DFLASH`
  is still oracle-blocked, this row inherits that block for its ORACLE
  comparisons** but not for gates 2-4, which are self-referential (ours vs ours).
- `SPEC-MTP-GGUF` `G2`'s dequant-cache helper. Not a hard dependency; if this
  row lands first it owns the helper and the MTP row reuses it. Sequence
  whichever starts first, do not duplicate the cache.
- A `dflash`-arch GGUF draft asset. **Producing one is NOT required** (see `GD0`):
  pre-converted drafts are published. `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp`
  publishes a FULL LADDER, not just the `Q4_K_M` this spec originally named:
  `BF16` 3,471,497,440 B, `Q8_0` 1,849,481,440 B, `Q6_K` 1,430,460,640 B,
  `Q5_K` 1,225,742,560 B, `Q4_K_M` 1,033,066,720 B
  (md5 `255fb6188e8eaaa0b0938dcd806dd3a1`). The `BF16` member is what makes
  gate 2 available and what makes the cross-FORMAT bar separable from the
  cross-QUANTIZATION one; the intermediate levels are a ready-made dose-response
  ladder for any future acceptance-vs-bit-width question. The fetch is the only
  blocked step and needs developer approval per the AGENTS.md safe defaults.
  Required for gates 2-5; NOT required for `GD1`-`GD3`.
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
| `GD4` | **DONE 2026-07-28 - axis A PASSES end to end on GB10, and it found a defect that only generating could find.** `tests/parity/test_qwen27_dflash_spec_decode.cpp` second case, env-driven draft source. Fixed `LoadDflashDraft`'s GGUF branch to back-fill `config.vocab_size` from the target's `embed_tokens` rows: it was 0, the draft's embedding view was therefore EMPTY, and the first propose threw. Result: GGUF-draft DFlash-ON == safetensors-draft DFlash-ON token-for-token with identical accepted/proposed, on two prompts. See the gate-form correction below | Gates 3, 5 | `GD3` |
| `GD5` | **DONE 2026-07-28.** `SharedHeadSource` (`src/vllm/entrypoints/model_loader.cpp`) re-expresses the shared-head seam as a SOURCE and re-types `LoadDflashDraft`'s second parameter; its GGUF arm is `LoadGgufSharedEmbedAndHeadBf16` (`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`), which reuses the trunk loader's tied-embedding rule and its sidecar-aware dequant rather than restating either. The shared-head load also MOVED out of the two draft-source branches into one common tail, so the four (draft format x target container) combinations run identical code | Gate 7 | `GD4` |
| `GD6` | **DONE 2026-07-28.** The `dflash` half of the GGUF-branch rejection is deleted (`model_loader.cpp`); the `mtp` half is untouched | Gate 1 | `GD5` |
| `GD7` | **DONE 2026-07-28 - axis B GENERATES on GB10.** `maybe_load_dflash`'s equivalent wired into the GGUF branch; e2e gate `tests/parity/test_qwen27_dflash_spec_decode.cpp` third case. See the axis-B result below | Gate 4 | `GD6` |
| `GD8` | Record: STATUS, BENCHMARKS, matrix, ledger | Checkers green | `GD7` |
| `GD9` | **Root-cause the axis-A accept-count RED. WEIGHT-SPACE HALF DONE 2026-07-29, category (a) established; the end-to-end half is what remains.** Enabled gate 2 with the previously-overlooked `BF16` GGUF (58/58 byte-identical, RED-first against `Q4_K_M`), bit-checked our Q4_K/Q6_K dequant against `gguf-py`, measured the whole quantization ladder, enumerated and excluded the config delta, and added the `VT_SPEC_TRACE=1` per-block propose/accept trace the totals cannot substitute for. See the root-cause section above | Gates 2, 3, 5 | `GD4` |

`GD0`-`GD4` are axis A and independently shippable; the row can legitimately rest
at `PARTIAL` after `GD4`.

## The axis-B result, and the risk that did NOT materialize (`GD5`-`GD7`)

Axis B WORKS. On dgx GB10 sm_121a, the Qwen3.6-27B NVFP4 **GGUF** target plus the
`Q4_K_M` **GGUF** draft loads, wires the shared head out of the target file
(`shared head from the GGUF target file`) and generates a coherent continuation,
with the DFlash-ON sequence **token-for-token IDENTICAL to that same target's
spec-OFF** on the c1 24-token prompt - the strict form of gate 4, not merely the
near-tie-robust one. Acceptance is alive at **14/160**. One prompt at c1; the
case asserts the near-tie-robust form so it does not become brittle.

**The spike ranked the shared-head dequant as its highest risk** (the draft would
score with a bf16 head dequantized from quantized target blocks while the target
computed on its own path). On this asset that risk is EMPTY, and the reason is
worth recording because it is a property of the container, not of luck: the 27B
NVFP4 GGUF stores `token_embd.weight` and `output.weight` as ggml **BF16 (type
30)** beside its NVFP4 body. Byte-compared against the safetensors sibling of the
same quantization run, both are **bit-identical - 2,542,796,800 bytes each, zero
differing bytes**. So B1's "dequant" of the shared head is a verbatim bf16 read,
and the draft scores with exactly the head it scores with on axis A. A GGUF that
DID quantize its head would re-open the risk, which is why the gate stays a
measurement rather than an assumption.

**Acceptance IS lower on the GGUF target - 14/160 (0.0875) vs 20/80 (0.25) on the
safetensors target, same draft, same prompt - and it is NOT chargeable to the
shared head.** The two containers are not the same target: their spec-OFF
continuations already diverge at index 4 with NO speculation anywhere, and the
DFlash-ON pair diverges at exactly that same index 4. The cause is `QUANT-GGUF-NVFP4`
being dequant-only - there is no NVFP4 GGUF GEMM, so on CUDA the GGUF target
EXPANDS to bf16 and computes in bf16 while the safetensors target runs the true
W4A4 fp4 kernels. A drafter proposing for a numerically different target agrees
with it less often. The honest disposition is therefore: axis B's wiring is
correct and proven, and the acceptance number belongs to the GGUF target's
compute path, which a native NVFP4 GGUF GEMM would change. This also settles
what a cross-target token gate would have measured: the containers' arithmetic,
never this row.

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
  **OUTCOME 2026-07-28: the premise is FALSE on this asset, and the decision
  still stands for others.** "On axis B that lm_head is quantized in the target
  GGUF" is what the spike assumed; the 27B NVFP4 GGUF actually stores
  `token_embd`/`output` as ggml BF16 beside its NVFP4 body, byte-identical to the
  safetensors sibling, so nothing is dequantized and both arms score with the
  SAME head. Acceptance did drop (14/160 vs 20/80), and it is precisely the byte
  comparison that let that be attributed to the GGUF target's bf16 compute path
  instead. The "documented minimum target quantization" contingency is therefore
  NOT owed; what a future quantized-head GGUF owes is a re-measurement, not a
  re-design.
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
