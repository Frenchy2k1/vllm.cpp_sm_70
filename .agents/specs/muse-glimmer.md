# Muse Glimmer — Meta's 30B open agentic multimodal model

**Row:** `MODEL-MUSE-GLIMMER`
**Issue:** [#268](https://github.com/mudler/vllm.cpp/issues/268)
**Base SHA:** `a0fa12c7219a86832412a6ece1490f452c1d1c40`
**Upstream anchor:** vLLM PR [#51655](https://github.com/vllm-project/vllm/pull/51655) head `075d645af`
**Secondary C++ reference:** llama.cpp PR [#26841](https://github.com/ggml-org/llama.cpp/pull/26841) (MERGED 2026-08-10)
**Checkpoint:** `meta-models/Muse-Glimmer-30B`, Apache-2.0, bf16

## 0. Honesty statement — what is and is not claimed

Three things are unusual about this row and none of them may be papered over.

**The anchor is not the pin.** Our parity pin `555967922` (2026-07-26) contains
no Muse Glimmer code — `grep -ril 'muse\|glimmer' vllm/model_executor/models/`
at the pin returns nothing — and neither does vLLM `main`. The only upstream
implementation is PR #51655, opened 2026-08-10T10:20Z, approved but **unmerged**,
with 3 of 20 CI checks red. This row therefore ports from a **branch head**, not
from the pin. That is a deliberate exception, taken on explicit developer
direction (2026-08-10), recorded as deviation 16 in
[`.agents/porting-inventory.md`](../porting-inventory.md) §9. It is NOT a
`waivers.csv` entry: no checker enforces the anchor rule, and a waiver naming a
checker that does not exist would be a false record. It owes a re-anchor when
#51655 merges.

**There is no gateable oracle, so no speed number is claimable.** AGENTS.md
requires both sides of any throughput comparison to run the pinned oracle on
identical workloads, and requires an oracle to demonstrably build and run the
model before it is gateable. The pinned vLLM cannot load `muse_glimmer` at all,
and the checkpoint needs `transformers 5.15.0.dev0` against the pin's 5.14.1.
**Every performance axis for this model is an open gap** until the pin advances.
Correctness is gated against the HF reference implementation instead. No
"parity" or "faster than vLLM" claim may be made from this row's evidence.

**llama.cpp is a secondary reference only.** Per
[[vllm-is-the-bar-not-llamacpp]], llama.cpp informs C++ structure and is useful
for cross-checking dequant and tokenizer behaviour, but it is never the
correctness oracle and never the speed denominator. Anything taken from it is
cited as such and marked in the porting inventory.

## 1. Architecture

`MuseGlimmerForConditionalGeneration` → `MuseGlimmerForCausalLM`,
`model_type: muse_glimmer`. The registry maps both the conditional-generation
and causal-LM architecture strings onto one class
(`registry.py`, PR #51655).

### 1.1 Text tower

52 layers, hidden 6656, 32 query heads / 2 KV heads (GQA 16:1), head_dim 128,
vocab 202048, max position 131072, rope theta 500000, `hidden_activation` gated
MLP, no attention logit softcapping.

| Mechanism | Detail | Anchor |
|---|---|---|
| Sandwich norms | `input`, `post_attention`, `pre_feedforward`, `post_feedforward` RMSNorm, fp32 compute, **baked `+1` weight offset** | `muse_glimmer.py:1236-1247` |
| Split eps | pre-norms use `rms_norm_eps`; post-norms use a separate, smaller `post_norm_eps` | `muse_glimmer.py:1239-1246` |
| iRoPE | `no_rope_layers[i] == 1` → RoPE **and** sliding window; `== 0` → NoPE **and** full attention | `muse_glimmer.py:1114-1116`, `:1167-1168` |
| QK-norm | weightless RMSNorm over `head_dim`, fp32, applied **before** RoPE | `muse_glimmer.py:1189-1196` |
| Query pre-scale | `scale_query_by` (~3.87) multiplies q *after* QK-norm; softmax scaling stays `head_dim**-0.5` | `muse_glimmer.py:1112`, `:1192` |
| Attention output gate | `sigmoid(output_gate_proj(hidden_states)) * attn_out`, gate reads the **layer input** | `muse_glimmer.py:1203-1206` |

Two of these are correctness traps and get their own RED-first tests:

- **The query pre-scale has two config schemas.** The native `params.json` ships
  the raw `qk_scale_factor` (~43.784); the modular HF `text_config` ships it
  pre-folded by `1/sqrt(head_dim)` (~3.87). Upstream disambiguates *by
  magnitude* — `qk_scale >= sqrt(head_dim)` means native, divide; otherwise use
  as-is (`muse_glimmer.py:472-517`). Getting this wrong scales every query by
  11.3x and is not a subtle drift.
- **`use_qk_norm` and `use_attn_output_gate` read as `None`, not `True`,** in the
  modular schema. Muse Glimmer always applies both; only an explicit `False`
  disables them (`muse_glimmer.py:456-469`). A naive `getattr(..., False)`
  silently drops both mechanisms and still produces plausible text.

### 1.2 Perception encoder

50 layers, hidden 1536, 16 heads (head_dim 96), patch 14×14, `patch_temporal` 2,
`pos_emb` grid 32×32, interleaved `window_attention` / `full_attention` per
`layer_types`, projector 4096 → output 6144. Image token 200092, video token
200091; placeholder strings `<|patch|>` / `<|image|>` / `<|video|>`
(`muse_glimmer.py:127-129`).

| Stage | Detail | Anchor |
|---|---|---|
| Patchify + embed | `conv1_linear: Linear(patch_temporal*3*patch_size^2 → hidden, bias=False)` — a **linear on patchified input**, not a conv | `muse_glimmer.py:696`, `:710` |
| Positional embedding | learned `[32*32, hidden]`, **bilinear-interpolated** to the actual grid with per-corner validity masking | `muse_glimmer.py:761-820` |
| 2D RoPE | `spatial_dim = head_dim//2`, base 10000, `freqs = cat([freq_w, freq_h])` — **width first** | `muse_glimmer.py:741-759` |
| Window attention | blocks of `pos_emb_height × pos_emb_width`, `-1`-padded permutation, per-block valid counts become `seq_lens` | `muse_glimmer.py:844-867` |
| Pixel-shuffle downsample | by `merge_kernel_size`; asserts `output_dim == hidden * merge^2` | `muse_glimmer.py:822-842`, `:734-739` |
| ln_pre / ln_post | plain `LayerNorm` (not RMSNorm) | `muse_glimmer.py:714`, `:732` |

The width-before-height RoPE concatenation and the `+0.5 / -0.5` half-pixel
convention in the positional interpolation are both easy to transpose and both
produce a plausible-but-wrong image understanding. Each gets a fixture test.

### 1.3 DFlash drafter

The PR adds no new drafter. It reuses `qwen3_dflash`, recognising
`MuseGlimmerAssistantModel` as a `dflash` method (`config/speculative.py`), and
threads the **target's** `is_neox_style` into the draft config
(`qwen3_dflash.py:75-94`, `v1/spec_decode/dflash.py:83-95`). Upstream's own
comment is the important part: a RoPE-layout mismatch between draft and target
is **silent** — acceptance collapses, nothing errors, output stays correct. Our
DFlash row already exists ([[dflash-correctness-done-speed-bf16-blocked]]), so
this is a recognition-and-threading change, not a new speculator.

## 2. Reuse — the shared seams this must route through

A capability not reachable through the shared surface is not done. Every Muse
mechanism has an existing home:

| Muse piece | Existing seam | Files |
|---|---|---|
| Sandwich norms, `+1` offset, fp32 | Gemma 2/3/4 norm path | `gemma2.cpp`, `gemma4.cpp` |
| Attention output gate | Qwen3.5 gated attention | `qwen3_5.cpp` |
| Weightless QK-norm | Qwen3 family | `qwen3_vl_text.cpp` |
| Windowed-attention vision tower | Qwen3-VL vision seam | `qwen3_vl_vision.cpp` |
| Gated MLP | `layers::MlpGateUpMethodBase`, `vt::MergedGemmGroup` | — |
| Decode | `ModelRegistry::Forward`, `dense_attn::AttnBlock`, on-device sampling | — |
| Fusion | `vt::FusedChain` | — |
| DFlash | existing speculator path | — |

**iRoPE / per-layer NoPE has no analogue** — we have no Llama-4 — and is the one
genuinely new text-side mechanism. It extends the existing per-layer attention
config rather than introducing a parallel path; if the current layer-type seam
cannot express "NoPE + full attention" alongside "RoPE + sliding", the seam is
extended, not bypassed.

Everything ships as **additive files** mirroring vLLM's structure:
`muse_glimmer.cpp`, `muse_glimmer_text.cpp`, `muse_glimmer_vision.cpp`,
`muse_glimmer_weights.cpp`, `muse_glimmer_registry.cpp`, plus
`include/vllm/model_executor/models/muse_glimmer.h`. The capability is exposed
through `include/vllm.h`; examples and the server stay thin ABI clients
([[examples-are-abi-clients-only]]).

## 3. Work breakdown

| W | Scope | Gate |
|---|---|---|
| **W0** ✅ | Config parse + registry + weight map. No forward. | **LANDED 2026-08-10**, see §8 |
| **W1** | Text tower forward: sandwich norms, iRoPE/NoPE, QK-norm, query pre-scale, output gate | Per-layer reference-dump match, RED-first per mechanism |
| **W2** | Text e2e greedy vs HF reference | Token-exact on a fixed prompt set |
| **W3** | Perception encoder: patchify, pos-emb interp, 2D RoPE, window attention, pixel shuffle | Tower dump match, per-stage fixtures |
| **W4** | Image e2e (placeholder expansion, projector, scatter into text) | STRICT vs HF reference |
| **W5** | Video (`patch_temporal`, frame sampling) | STRICT vs HF reference |
| **W6** | DFlash drafter recognition + `is_neox_style` threading | Acceptance-rate check vs spec-off; drafts must actually be consumed |
| **W7** | Reasoning + tool parsers on the server surface | Ported upstream parser tests |

W0–W2 are the critical path. W3–W5 depend on the vision seam. W6 must force
async off on both arms ([[engine-proc-dropped-every-speculator-draft]]) or the
A/B is meaningless.

## 4. Gates

**Correctness, against the HF reference** (not the pinned oracle, which cannot
run this model — see §0):

1. Config parse: both schemas resolve to the same `scale_query_by` (~3.87), and
   `use_qk_norm` / `use_attn_output_gate` default **on**.
2. Per-mechanism RED-first: disabling QK-norm, the output gate, the query
   pre-scale, the `+1` norm offset, or the NoPE/RoPE layer split must each turn
   its test red. Mutation is the proof, not inspection.
3. Text e2e: token-exact greedy vs the HF reference on a fixed prompt set.
4. Vision: per-stage tower dumps within the bf16-depth envelope; image and video
   e2e STRICT.
5. DFlash: drafts demonstrably consumed (not silently dropped), acceptance rate
   recorded, output identical to spec-off.

**Ported tests.** The upstream tests in PR #51655 —
`tests/transformers_utils/test_muse_glimmer_config.py`,
`test_muse_glimmer_config_schema_norm.py`, and the five
`tests/tool_use/test_muse_glimmer_*.py` — are ported in the same change with
parameters, fixtures, tolerances and failure cases preserved, each carrying the
`075d645af` revision anchor. Harness adaptation is documented where unavoidable.

**Speed: OPEN GAP, not measured, not waived.** Recorded in `docs/BENCHMARKS.md`
as pending on the named external unblocker: #51655 merging and the pin advancing
to include it plus transformers 5.15. No ceiling is declared
([[when-stuck-workflow-scan-vllm]]).

## 5. Risks

- **The anchor can change under us.** #51655 is an open branch with red CI; a
  force-push or review round rewrites what we cite. Mitigation: cite the exact
  head `075d645af`, keep the fetched ref, and diff before every re-anchor.
- **Unmerged upstream may itself be wrong.** The approved-but-red state means
  upstream's own gates have not fully passed. Where our HF-reference gate
  disagrees with PR #51655, the HF reference wins and the divergence is reported
  upstream rather than silently mirrored.
- **No oracle means correctness rests on one leg.** The HF reference is the only
  cross-check until the pin moves. Per-mechanism mutation testing carries more
  weight than usual here.
- **Weights are not local.** ~60 GB bf16. Download needs explicit authority.
- **GB10 memory.** 30B bf16 will not fit comfortably alongside anything else;
  `gpu_memory_utilization` reserves host RAM on unified memory
  ([[gb10-unified-memory-oom-reboots-box]]). Quantized arms first where possible.

## 6. Stop conditions

- #51655 is force-pushed or substantially rewritten → stop, re-diff, re-anchor
  before continuing.
- The HF reference and PR #51655 disagree on a mechanism → `NEEDS_DECISION`,
  do not guess which is authoritative.
- Any request to state a speed number for this model before the pin advances →
  refuse; the axis is an open gap by construction.
- Weight download or GPU time beyond what is already authorised → stop and ask.

## 7. Outcome

Pending overall.

## 8. W0 — the CPU scaffold (2026-08-10, `CLAIM-MUSE-GLIMMER-W0`)

Additive only: `include/vllm/model_executor/models/muse_glimmer.h`,
`src/vllm/model_executor/models/muse_glimmer{,_weights,_registry}.cpp`,
`tests/vllm/models/test_muse_glimmer_scaffold.cpp`. No forward, no checkpoint,
no GPU, no download.

Both architecture strings register onto one factory, mirroring upstream. The
config parse handles the canonical nested layout *and* normalizes the older flat
layout. The weight-name mapper ports `hf_to_vllm_mapper` for both checkpoint
conventions. The structural enumeration deliberately omits the three weightless
modules (`embed_norm`, the per-head `qk_norm`, `perception_emb_norm`) that ship
no tensor — enumerating them would make the loader demand tensors that do not
exist in any checkpoint. The forward refuses by name.

**Gate:** `test_muse_glimmer_scaffold` 11/11 cases, 73/73 assertions, clean CPU
`-Werror` build. Full CPU `ctest` green (regression: the change is additive TUs
plus two registry entries).

**RED-first mutation evidence.** Each of the four named traps was mutated in
tree, rebuilt, and confirmed to turn the gate red; the tree was then restored
byte-for-byte (verified by an empty `git diff`) and the gate re-run green:

| Mutation | Result |
|---|---|
| Native raw `qk_scale_factor` treated as already-folded (the 11.3x query blow-up) | 3 assertions RED |
| Absent `use_qk_norm` / `use_attn_output_gate` defaulted to `false` | 4 assertions RED |
| iRoPE mask counted forward instead of backward from the last layer | 5 assertions RED |
| Legacy sandwich-norm renames applied in the wrong order (swapping post-attention with pre-feedforward) | 1 assertion RED |

**What W0 does NOT establish.** No forward runs, so nothing here says the model
produces correct tokens. The KV-cache spec is a documented placeholder: the real
sliding/full split rides the Gemma-4 per-layer spec seam and lands with W1. And
per §0 no speed axis is measurable at all while the pin lacks `muse_glimmer`. Filled in when the row reaches `DONE`: what was measured, what was
rejected and why, and why each default is set the way it is.
