# SPIKE: Gemma-4 multimodal (image + video + AUDIO) + the AUDIO track

**SPIKE ONLY — READ-ONLY design + a checkpoint/oracle/HW-fit check. No
implementation, no tower built, no build, no download, no gate.** Grounds the
remaining piece of the user's #1 roadmap priority (2026-07-25: *"Multimodal
Audio/Video/Image with Gemma-4 and Qwen3.6"*). Qwen3.6 image+video is LANDED
(M2/M3, [multimodal-track.md](multimodal-track.md)); **audio exists ONLY in
Gemma-4 / gemma3n**, so this spike scopes (a) the genuinely-new AUDIO modality
and (b) the Gemma-4 backbone+towers, honestly separating what is reachable on
GB10 against the pinned oracle from what is staged/blocked.

**Base:** `origin/main` `64a01af` (M3c video near-tie gate landed). **Oracle pin:**
`/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:** `~/venvs/vllm-oracle` = vLLM
**0.25.0** + **transformers 5.13.1** (measured 2026-07-25). **Claim:**
`CLAIM-GEMMA4-MULTIMODAL`.
**Precedent spikes mirrored:** [`multimodal-track.md`](multimodal-track.md) (the
landed M0–M3 mm infra this reuses), [`sweep-gemma.md`](sweep-gemma.md) (the
Gemma-4 text-backbone characterization: PLE/YOCO/Gemma-4-MoE), and the
blocked-row honesty precedent [`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md).

Rows this spike advances (owned; the concurrent Qwen3.6-video agent owns the Qwen
`MODEL-MM-*` rows + `multimodal-track.md` — NOT touched here):
- `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` — stays `SPIKE`, verdict
  sharpened to **oracle-BLOCKED (decisive)** + re-pointed to this spec.
- `MODEL-MM-gemma4-unified-gemma4-unified-for-conditional-generation` — same.

---

## 0. Headline findings

> **W0 RUN-VERIFIED (2026-07-28, `CLAIM-GEMMA4-W0`): the oracle LOADS + RUNS +
> GENERATES — Gemma-4 is GATEABLE, greedy golden captured.** The decisive
> oracle-gateability gate ([[oracle-gateability-model-runs-not-config-constructs]])
> is now PASSED, not inferred: on dgx, `~/venvs/vllm-oracle` = vLLM **0.25.0** +
> transformers **5.13.1**, `vllm.LLM(model=unsloth/gemma-4-E4B-it)` (ungated,
> `Gemma4ForConditionalGeneration`, 15.99 GB bf16 single shard) **resolved the arch,
> loaded the weights onto the GB10, built the KV cache, and greedily generated 32
> coherent tokens** (`enforce_eager`, `temperature=0`, GMU 0.30 under `flock`). vLLM
> configured Gemma-4's heterogeneous head dims (head_dim=256 / global_head_dim=512 →
> forced `TRITON_ATTN`) and ran the PLE/YOCO/Gemma-4-MoE backbone e2e. **K=5
> self-determinism = ALL-DETERMINISTIC ⇒ the future bring-up bar is STRICT
> token-exact.** This is the opposite of OLMo-3 (which CONSTRUCTS but ABORTS on run):
> Gemma-4 CONSTRUCTS **and RUNS**. Golden fixture:
> `tests/parity/goldens/gemma4_e4b_text/gen_manifest.json` (prompt + exact 32 output
> token ids + K=5 runs + sha256; capture script `scripts/mm/g0_gemma4_oracle_capture.py`).
> **Verdict: the oracle-block is fully retired; the ONLY remaining work is
> implementation** (the PLE/YOCO/Gemma-4-MoE backbone + the SigLIP/USM-Conformer
> towers — see the G-plan §2.2). Text-only ran (sufficient for W0 — it exercises the
> full backbone); an image/audio prompt was NOT run this pass (staged to G2/G3).
> Golden greedy tokens (ref, STRICT): `[236776, 2455, 5192, 2028, 563, 496, 3996,
> 16477, 14020, 1948, 15453, 580, 12566, 12136, 529, 1816, 1262, 531, 3050, 236764,
> 8729, 236764, 532, 8932, 531, 3246, 5192, 528, 496, 40137, 532, 4403]` = *"A large
> language model is a complex artificial intelligence program trained on massive
> amounts of text data to understand, generate, and respond to human language in a
> coherent and context"*. Oracle input prompt ids (chat-templated, len 20): BOS 2 →
> `<|turn>user…`. The two subsections below are preserved as the (now-superseded)
> gate-time record.

> **POST-PIN-ADVANCE UPDATE (2026-07-26): the decisive block is DISSOLVED.** The
> parity pin advanced to `555967922` / vLLM 0.26.0.dev0, which carries **transformers
> 5.14.1 - and 5.14.1 SHIPS `transformers.models.gemma4`** (environment.md:39-40;
> the advance explicitly lists Gemma-4 among its unblocks). So the "oracle cannot
> construct the mm path" verdict below was correct against the *gate-time* 0.25.0 /
> transformers-5.13.1 oracle but no longer holds on the current pin. Re-assessed
> verdict: Gemma-4 multimodal is **reachable on the advanced pin, implementation
> pending** (the ≥12B HF-gated mm-wrapped checkpoint + the PLE/YOCO/Gemma-4-MoE
> backbone + the SigLIP/USM-Conformer towers remain the real work; the oracle is no
> longer the blocker). The analysis below is preserved as the gate-time record.

### 0.0 The DECISIVE gating fact (AT GATE TIME): the 0.25.0 oracle CANNOT construct Gemma-4 mm (transformers 5.13.1 has no `gemma4`)

The Gemma-4 mm wrapper does **not** implement its towers natively. It loads them
from **Transformers** via `AutoModel.from_config`:

- vision tower — `gemma4_mm.py:1040` `self.vision_tower = AutoModel.from_config(config=config.vision_config)`
- audio tower — `gemma4_mm.py:1056` `self.audio_tower = AutoModel.from_config(config=config.audio_config)` (+ `post_init()` at `:1061` to build the Conformer `inv_timescales`/softcap/grad-clip buffers absent from the checkpoint)

and the config classes are imported straight from Transformers:
`gemma4_mm.py:25-33` `from transformers.models.gemma4 import (Gemma4Config,
Gemma4Processor, Gemma4VisionConfig)` and `...configuration_gemma4 import
(Gemma4AudioConfig, Gemma4TextConfig)`.

**Measured on dgx (2026-07-25):** `~/venvs/vllm-oracle` = vLLM 0.25.0 +
**transformers 5.13.1**, and `import transformers.models.gemma4` **FAILS**
(ModuleNotFoundError) while `import transformers.models.gemma3n` **succeeds**. So
constructing `Gemma4ForConditionalGeneration` on the pinned oracle dies at
`AutoModel.from_config(config.vision_config)` — **there is no SACRED oracle for
the Gemma-4 mm path, hence no gateable target.** This is stronger and more
decisive than [`sweep-gemma.md`](sweep-gemma.md) §W6 found for the bare text row
(which only worried oracle *registry* listing): even though vLLM 0.25.0 ships
`gemma4_mm.py`/`gemma4_unified.py`, the towers are Transformers modules the pinned
Transformers does not carry. **A model the oracle cannot run has no gate**
([gates.md](../gates.md)). This is the load-bearing reason Gemma-4 mm is
HONESTY-PASS-BLOCKED, independent of checkpoints and HW.

### 0.1 The Gemma-4 mm architecture — two variants, a SigLIP vision tower, and a USM Conformer audio tower

`gemma4_mm.py` (1708 lines) registers `Gemma4ForConditionalGeneration`
(`registry.py:392`); `gemma4_unified.py` (469 lines) registers
`Gemma4UnifiedForConditionalGeneration` (`registry.py:393-396`), which
**subclasses** `Gemma4ForConditionalGeneration` (`gemma4_unified.py:40,220`).

**Variant A — `Gemma4ForConditionalGeneration` (the encoder variant):**
- **Vision tower** — a **SigLIP-class ViT** loaded via `AutoModel.from_config(config.vision_config)`
  (`gemma4_mm.py:1040`), shared by image AND video (`_mark_tower_model(..., {"image","video"})`
  `:1039`). Structurally a learned-abs-pos ViT (patch-embed → pre-LN transformer
  blocks with bidirectional attention + gelu-tanh MLP), the Gemma-3 vision lineage
  — NO patch-merger / NO DeepStack / NO vision-RoPE (simpler than the Qwen3-VL
  tower we landed in M2a). Video path: `_VIDEO_MAX_SOFT_TOKENS=70`, `_VIDEO_MAX_FRAMES=32`
  (`gemma4_mm.py:91-92`).
- **Audio tower** — a **USM-class Conformer** loaded via `AutoModel.from_config(config.audio_config)`
  (`gemma4_mm.py:1056`), present **only** on variants whose config carries an
  `audio_config` (`:1055`, else `audio_tower=None` `:1074`). Frontend per the
  processor arithmetic (`gemma4_mm.py:348-368` `_compute_audio_num_tokens`): **mel
  framing** (`Gemma4AudioFeatureExtractor._unfold`) → **two Conv2d subsampling
  layers** (kernel 3, stride 2, semicausal pad top=1/bottom=1) → Conformer blocks
  with relative-position attention + softcap (the `post_init()` buffers). Forward
  `gemma4_mm.py:1468-1490`: `audio_tower(input_features, input_features_mask)` →
  `(audio_encodings, audio_mask)`, then the projector, then per-audio padding strip.
  `hidden=1024`, `output_proj_dims=1536` (`gemma4_mm.py:936-940`).
- **Projector(s)** — `Gemma4MultimodalEmbedder` (`gemma4_mm.py:908-965`): a
  **2-layer** design = `RMSNorm(has_weight=False)` (`:942`) → `ReplicatedLinear`
  bias-free to `text_hidden_size` (`:949`). One instance each for vision
  (`embed_vision`, in-dim `hidden_size`=768) and audio (`embed_audio`, in-dim
  `output_proj_dims`=1536). No embedding table, no pre-projection weights (the
  checkpoint only has `embedding_projection.weight`).
- **Backbone** — `Gemma4ForCausalLM` (`gemma4_mm.py:44`), the PLE/YOCO/KV-sharing/
  Gemma-4-MoE/k_eq_v text stack characterized in [`sweep-gemma.md`](sweep-gemma.md)
  §0.1 (none of it landed).
- **Merge** — soft tokens scatter into audio/image/video placeholder rows of
  `input_embeds` (the `SupportsMultiModal` mixin, `interfaces.py`), identical in
  shape to the vision merge we landed (M2c/M3b `Qwen3VLMergeMultimodal`).
- **Modality support gating** (`gemma4_mm.py:216-235,270-275`): image+video always;
  **audio only if `audio_config is not None`** — a non-audio Gemma-4 checkpoint
  raises "does not have an audio tower" (`:225-229`). `audio` mm-limit and
  `audio_seq_length` come from the processor (`:235,255`).

**Variant B — `Gemma4UnifiedForConditionalGeneration` (encoder-FREE):**
`gemma4_unified.py:3-16` — **no SigLIP vision tower and no audio tower**; images
flow through `Gemma4UnifiedVisionEmbedder` (`:73`), a lightweight patch pipeline
with **factorized 2-D positional embeddings** straight into text-embed space
(`:266-291` overrides `embed_vision`/`embed_audio`). Audio still gated on
`audio_config` (`:172,288-291`). This is a **different, simpler** design than
Variant A (no `AutoModel` vision tower) — but it STILL imports
`transformers.models.gemma4_unified` configs (`:23`), so it is oracle-blocked by
0.0 for the same reason.

**Checkpoint→variant map** (HF metadata, [`sweep-gemma.md`](sweep-gemma.md) §0.0,
nothing downloaded): `google/gemma-4-12B-it` (11.96B) = **Unified** (encoder-free);
`google/gemma-4-31B-it` (32.68B) + `google/gemma-4-26B-A4B-it` (26.5B MoE) =
**Variant A** (encoder). Which of them carry an `audio_config` (hence do audio) is
per-checkpoint and unverified — the audio tower is optional; a fetch-time
`config.json` read decides it.

### 0.2 Reuse-vs-new against the LANDED M0–M3 mm infra

The image+video track (M0–M3) built a reusable spine. Mapping each Gemma-4/audio
piece onto it:

| Gemma-4 / audio piece | Landed M0–M3 anchor | Verdict |
|---|---|---|
| mm INPUT container (`MultiModalKwargs`, feature specs) | `src/vllm/multimodal/` (`inputs.h`) | **REUSE** — modality-agnostic; audio adds a new feature-tensor kind (`input_features`+mask) |
| mm-hash | `MultiModalHasher` blake3 (`src/vllm/multimodal/hasher.cpp`) | **REUSE** — hashes any serialized media incl. audio bytes |
| encoder-cache engine seam | `EncoderCacheManager` + budget (`src/vllm/v1/core/encoder_cache_manager.cpp`) | **REUSE** — keyed by mm-hash, modality-agnostic |
| placeholder-token expansion + masked-scatter merge | `Qwen3VLMergeMultimodal` (M2c/M3b, `qwen3_vl.cpp`/`qwen3_5.cpp`) | **REUSE (pattern)** — audio soft tokens scatter into audio-placeholder rows exactly like image tokens; a Gemma-4 audio-token id + count-arithmetic is the only new wiring |
| LMCache `extra_keys` mm slot | `chunked_token_database.{h,cpp}` | **REUSE** — already carries the slot |
| ViT tower scaffold (patch-embed matmul, attention, LayerNorm, GELU MLP) | M2a `qwen3_vl_vision.{h,cpp}` + `vt::` GEMM/attn/norm + `GeluTanh`/`GeluErf` ops | **REUSE (scaffold) for the SigLIP vision tower** — SigLIP delta: learned abs pos-embed (no vision-RoPE), no patch-merger, no DeepStack → SIMPLER than M2a; drop those three, keep blocks+MLP+LN |
| the merge/decode fork driver | M2c/M3b `VLGenerateCore` | **REUSE (pattern)** for any tower→merge→decode |
| **AUDIO input pipeline** (decode→resample→log-mel→_unfold framing→count) | — none (grep: only image/video processors) | **GENUINELY NEW** — the largest audio piece; no audio preprocessing exists today |
| **AUDIO encoder tower** (USM Conformer: 2×Conv2d subsample + relative-pos attn + conv-module + softcap) | — none | **GENUINELY NEW tower TYPE** — attention/FFN reuse `vt::` ops; the Conformer conv-module + relative-position bias + Conv2d subsampling are new kernels |
| audio projector | `Gemma4MultimodalEmbedder` = RMSNorm(no-weight)+Linear | **REUSE** ops (`vt::RmsNorm`+GEMM), trivial |
| **Gemma-4 backbone** (PLE/YOCO/KV-share/Gemma-4-MoE/k_eq_v) | — none | **GENUINELY NEW** — the [`sweep-gemma.md`](sweep-gemma.md) §0.1 stack; a separate campaign, no other row needs it |
| Gemma sandwich norms / soft-cap / GeGLU / gemma-RMSNorm | LANDED (Gemma-1/2/3 text, `gemma{,2,3}.cpp`; `kGeluAndMul`, `kSoftCap`) | **REUSE** — the Gemma text primitives are done |

**Net:** the mm SPINE (input container, hash, encoder cache, merge, decode fork)
and the ViT scaffold + Gemma text primitives are **ours**. Genuinely NEW =
**(a) the audio input pipeline, (b) the audio USM-Conformer tower, (c) the Gemma-4
backbone**; the SigLIP vision tower is a REUSE-with-simplification of M2a.

### 0.3 Checkpoint + oracle + GB10-fit — the gating facts (Gemma-4)

| Fact | Verdict |
|---|---|
| **Oracle constructs the mm path?** | **NO at gate time (decisive then; DISSOLVED on the advanced pin)** — transformers 5.13.1 had no `gemma4`; `AutoModel.from_config(vision_config)` failed (0.0). The current 0.26.0.dev0 pin carries transformers 5.14.1, which ships `gemma4`, so the oracle can now construct it. |
| Oracle ships the vLLM files? | Yes (`gemma4_mm.py`,`gemma4_unified.py` present) — necessary but NOT sufficient; the towers are Transformers modules. |
| Smallest checkpoint | `google/gemma-4-12B-it` (11.96B, **Unified/encoder-free**); next `26B-A4B`/`31B` (Variant A). All **≥12B, mm-wrapped, `google/*` HF-gated**. No ungated bare-text or small mirror. Audio-capable variant = whichever carries `audio_config` (per-checkpoint, unverified). |
| dgx cache | **none** — `ssh dgx 'ls ~/.cache/huggingface/hub ~/bench \| grep -i gemma'` = gemma-3-1b-it, unsloth gemma-2/2b only. No Gemma-4, no audio-LLM. |
| GB10 fit (119 GiB unified) | 12B (~24 GiB bf16) + SigLIP tower + audio tower **FITS**; 26B-A4B/31B **HW-marginal-to-blocked** ([`sweep-gemma.md`](sweep-gemma.md) §0.6). |

**Verdict: Gemma-4 mm = HONESTY-PASS-BLOCKED / STAGED.** Not e2e-reachable at the
pin: (1) oracle cannot construct it (no gate); (2) checkpoints ≥12B, gated,
mm-wrapped, none cached; (3) three new subsystems unbuilt (SigLIP tower is a
reuse, but the **audio pipeline + USM Conformer tower + the PLE/YOCO/MoE
backbone** are all new). HW is the ONLY non-blocker (12B fits). Reopen when the
oracle's Transformers advances to carry `gemma4` (or the pin advances) AND a
fitting checkpoint downloads.

### 0.4 Audio reachability BEYOND Gemma-4 — the smallest oracle-runnable vehicle to stand the modality up

Audio is the genuinely-new modality (nothing built). Mirroring how vision was
stood up on Qwen3-VL-4B before Qwen3.6-27B, land the audio subsystems on the
smallest model the pinned oracle CAN run, then carry them to Gemma-4. All the
following have **native vLLM towers** (grep: 0 `AutoModel.from_config` in
`whisper.py`/`qwen2_audio.py`*/`voxtral.py`/`ultravox.py`/`granite_speech.py`) and
their Transformers deps ARE in 5.13.1 — so unlike Gemma-4 they are
oracle-constructible.

| Vehicle | Params (approx, verify at fetch) | Audio encoder | Backbone | Oracle 0.25.0 | GB10 fit | Role |
|---|---|---|---|:--:|:--:|---|
| **Whisper** (`whisper-small` 244M / `base` 74M / `tiny` 39M) | 39M–244M | native WhisperEncoder (log-mel 128 + 2×Conv1d + transformer), `whisper.py:458` | encoder-DECODER (ASR) | ✅ native | ✅ trivial | **A1/A2 vehicle** — smallest, stands up the audio INPUT pipeline (log-mel) + first encoder tower in isolation |
| **Voxtral-Mini-3B-2507** | ~4.7B | native `WhisperCausalEncoder` (`voxtral.py:671,737`) | **Mistral** (LANDED: Mistral-7B-v0.3 text SACRED 16/16) | ✅ native | ✅ ~9.4 GiB | **A3 vehicle** — smallest decoder-MERGE audio-LLM on a backbone we own → e2e audio→text |
| Qwen2-Audio-7B | ~8.4B | transformers `Qwen2AudioEncoder` (Whisper-class; qwen2_audio in 5.13.1), `qwen2_audio.py:349` | Qwen2 decoder | ✅ | ✅ ~17 GiB | A3 fallback (larger; Qwen2 backbone) |
| Granite-Speech-3.3-2b | ~2–3B | native **Conformer** (`granite_speech.py:294-297`) | Granite decoder | ✅ native | ✅ | **Conformer reference** — closest encoder to Gemma-4's USM (proves the conv-module) |
| gemma3n-E2B-it | 5.44B | native USM **Conformer** (`gemma3n.py`, `gemma3n_audio_utils.py`) | MatFormer/AltUp (complex, out-of-scope per sweep-gemma D6) | ✅ (transformers has gemma3n) | ✅ ~10.9 GiB | Gemma-family Conformer ref; backbone too heavy to be the first vehicle |

**Named smallest audio-runnable vehicle: `whisper-small` (244M, native,
oracle-certain, fits trivially)** to stand up the audio INPUT pipeline + first
encoder tower, then **`Voxtral-Mini-3B`** (native Whisper-class encoder + our
LANDED Mistral backbone + projector-merge) for the e2e audio→text gate — the
low-risk path that reuses a text backbone we already own. **Encoder-family caveat
(honest):** Whisper/Voxtral/Qwen2-Audio use a **Whisper-class** encoder (2×Conv +
vanilla transformer); Gemma-4/gemma3n/Granite use a **USM Conformer** (Conv2d
subsampling + conv-module + relative-pos attn + softcap). So these vehicles
de-risk the audio *pipeline + merge pattern* fully, but the **Gemma-4-specific
Conformer tower** is proven separately on **Granite-Speech-2b** (or gemma3n-E2B) —
not by Whisper/Voxtral.

---

## G1 — TEXT backbone LANDED (2026-07-28, `CLAIM-GEMMA4-G1`)

The Gemma-4 **text backbone** (`Gemma4ForConditionalGeneration` language_model
stack of `unsloth/gemma-4-E4B-it`) is implemented as NEW additive files —
`include/vllm/model_executor/models/gemma4.h`,
`src/vllm/model_executor/models/{gemma4,gemma4_weights,gemma4_registry}.cpp` —
mirroring the OLMo-2/gemma3 registration seam (one `REGISTER_VLLM_MODEL` line, no
shared-array edit). CPU `-Werror` 0-warn on all three TUs + full `libvllm.a` link
(SACRED inertness: the whole existing model set still builds).

### G1.0 E4B config — which primitives are actually ON

From `unsloth/gemma-4-E4B-it` `text_config` (fetched HF, 2026-07-28): hidden 2560,
42 layers, GQA 8/2, head_dim 256 / **global_head_dim 512**, intermediate 10240,
`hidden_size_per_layer_input` 256, `num_kv_shared_layers` 18, sliding_window 512,
`final_logit_softcapping` 30.0, vocab 262144, tie_word_embeddings true. Crucially
**`enable_moe_block=false`, `attention_k_eq_v=false`, `use_double_wide_mlp=false`**
→ the Gemma-4 MoE router / per_expert_scale, k_eq_v, and double-wide MLP are OFF
for E4B and are the ≥12B-checkpoint follow-on, NOT G1.

### G1.1 Primitive-by-primitive port map (grounded, file:line)

| Primitive | vLLM ground | Our realization | REUSE / NEW |
|---|---|---|---|
| **PLAIN RMSNorm** (`x·w`, NOT gemma `(1+w)`) | `gemma4.py:45` imports `layernorm.RMSNorm` (not `GemmaRMSNorm`); used at every norm | `vt::RmsNorm(...,{eps,false})` | REUSE (the `gemma=false` mode) — **the load-bearing divergence from gemma2/3** |
| **PLE** (Per-Layer Embeddings) | `gemma4.py:986-1063` (tables) + `:845-898` (combine) + `:680-761` (per-layer gate/proj/norm) | `embed_tokens_per_layer` lookup·√ple + `per_layer_model_projection`·h^-0.5 → RMSNorm → `(proj+emb)·rsqrt2`; per layer `gelu(gate_lin(h))*ple` → proj → norm → add | NEW wiring; the gate reuses `vt::GeluAndMul` on `[gate_lin ‖ ple]` (no elementwise-Mul op exists) |
| **YOCO KV-sharing** | `gemma4.py:463-489`, forward `:535-548` | shared layers (24-41) read the target layer's cache (sliding→22, full→23) IN-forward, compute no K/V | NEW (in-forward target-cache index; no runner aliasing) |
| **heterogeneous head_dim** 256/512 | `gemma4.py:572-578` | per-layer `Dh` threaded through the attn block | NEW |
| **proportional partial-RoPE** (full) | `gemma4_rope.py` (head_dim denom + zero-pad), `rope_parameters.full_attention` (θ 1e6, pf 0.25 → rotary 128/512) | custom host cos/sin cache → `vt::RopeFromCache` | REUSE `RopeFromCache` + NEW cache builder |
| **standard sliding rope** | `rope_parameters.sliding_attention` (θ 1e4, full 256) | `vt::RopeNeox` | REUSE |
| **weight-less V-norm** | `gemma4.py:437` `has_weight=False` | `vt::RmsNorm` with a ones[Dh] weight (identity) | REUSE |
| **GeGLU MLP** | `gemma4.py:224-254` `gelu_pytorch_tanh` | `vt::GeluAndMul` | REUSE (gemma2/3 W1 primitive) |
| **√hidden embed-scale** | `gemma4.py:1067-1074` | `vt::MulScalar` bf16 normalizer | REUSE |
| **per-layer scalar** | `gemma4.py:707,765` `layer_scalar` [1] | host-read bf16 → `vt::MulScalar` | REUSE |
| **final logit soft-cap 30** | `gemma4.py:1569-1572` | `vt::SoftCap` | REUSE (gemma2 W3 primitive) |
| **tied lm_head** | `gemma4.py:1566-1567` | `MatmulBT` over embed table | REUSE |

Residual pattern is standalone-norm + explicit `vt::Add` (NOT the gemma2/3 fused
add-norm sandwich), because PLE + `layer_scalar` intervene after the second add.

### G1.2 Weight loader — VERIFIED (no download)

`LoadGemma4ForConditionalGenerationWeights` strips the `model.language_model.`
prefix and skips the mm towers (`audio_tower`/`vision_tower`/`embed_audio`/
`embed_vision`, per `gemma4.py:1716-1723`). VERIFIED against the real E4B
safetensors HEADER (HTTP range, no 16 GB download): 2130 total tensors; every one
of the 336 `language_model.*` names + shapes matches the loader's expected map
(incl. per-layer q/k/v/o at the correct 256/512 head widths, PLE tables, per-layer
gate/proj/norm, `layer_scalar` [1], tied embeddings). Shared layers (24-41) DO
carry their own q/k/v_proj in the checkpoint (loaded but K/V discarded at forward).

### G1.3 HONEST e2e GATE STATUS — BLOCKED on runner KV topology (named)

The strict 32/32 gate vs `tests/parity/goldens/gemma4_e4b_text/gen_manifest.json`
(golden `[236776, 2455, 5192, ...]`) is **NOT reached this pass**, and the reason is
precise, not a numeric divergence: the runner allocates **one uniform KV head_dim**
per non-GDN layer (`src/vllm/v1/worker/gpu/runner.cpp:600-646`, `attn_kv_` built
with a single `Hkv`/`Dh`). Gemma-4's per-layer **256 (sliding) / 512 (full)** head
dims cannot be represented without a shared-path change to `attn_kv_` construction
(per-layer/per-group head_dim). The forward's per-layer
`VT_CHECK(kv.head_size == Dh)` turns this into an explicit failure rather than a
silent wrong answer. This is the additive-vs-shared-path boundary: G1 is clean
additive files; the KV-topology change is a runner edit deferred to **G-next**.

**G-next (to reach strict 32/32):** (1) runner heterogeneous per-layer KV head_dim
(+ optional YOCO cache aliasing to reclaim the 18 shared caches' memory); (2) a full
CUDA build + STRICT gate on dgx under `flock`; (3) verify the two named bf16-rounding
nuances (the f32-accumulated PLE combine in vLLM vs our per-op bf16; the proportional
cos/sin cache dtype) do not perturb the token match. G2 (SigLIP vision, reuses M2a)
and G3 (USM-Conformer audio) remain separate and unbuilt.

---

## 1. Per-model / per-modality DISPOSITION

| Target | Image | Video | Audio | Disposition |
|---|:--:|:--:|:--:|---|
| **Gemma-4 `Gemma4ForConditionalGeneration`** | ✅ (SigLIP) | ✅ (SigLIP) | ✅ (if `audio_config`) | **GATEABLE — W0 RUN-VERIFIED 2026-07-28, greedy golden captured, IMPLEMENTATION PENDING.** vLLM 0.25.0 (transformers 5.13.1) LOADS+RUNS+GENERATES `unsloth/gemma-4-E4B-it` (ungated, 15.99 GB) — STRICT K=5 golden in `tests/parity/goldens/gemma4_e4b_text/`. Oracle block RETIRED; remaining work is pure impl: the PLE/YOCO/Gemma-4-MoE backbone + the SigLIP/audio towers, all unbuilt. No engine e2e yet. |
| **Gemma-4 `Gemma4UnifiedForConditionalGeneration`** | ✅ (encoder-free embedder) | ✅ | ✅ (if `audio_config`) | **GATEABLE (by the E4B W0 proof — same oracle path), IMPLEMENTATION PENDING** — simpler (no SigLIP/audio `AutoModel` tower); the ungated `unsloth/gemma-4-12b-it` (23.92 GB) is this variant and fits GB10. No standalone run this pass (E4B is the smaller vehicle); the oracle-runnability is proven by the shared registered path. No engine e2e yet. |
| **AUDIO modality** (as a subsystem) | — | — | ✅ | **STAGED — reachable, land it first on the smallest oracle-runnable vehicle** (Whisper→Voxtral-Mini-3B), NOT on Gemma-4. This is the genuinely-new work. |
| Whisper (vehicle) | — | — | ✅ ASR | IMPLEMENTABLE-ADDITIVE — oracle-runnable, fits; the audio-pipeline standup vehicle |
| Voxtral-Mini-3B (vehicle) | — | — | ✅ | IMPLEMENTABLE-ADDITIVE — oracle-runnable, fits, Mistral backbone LANDED; the e2e audio-merge vehicle |

---

## 2. The W-plan — the AUDIO track (modality-first) then the Gemma-4 campaign

Two SEPARATE campaigns. The AUDIO track (A0–A3) is the genuinely-new, reachable
work and does NOT depend on Gemma-4. The Gemma-4 campaign (G0–G3) is
staged/blocked behind the oracle. Every increment carries the **inertness gate**
(all current SACRED gates byte-identical — the audio/tower subsystems are additive
and gated on mm input, exactly as the landed vision track proved).

### 2.1 AUDIO track (genuinely-new modality; the low-risk path)

```
 A0  Ground + vehicle + oracle/fit                     [CRITICAL PATH]
      |
 A1  AUDIO INPUT pipeline (decode→resample→log-mel→framing→placeholder count)  [foundation]
      |
 A2  AUDIO encoder TOWER (Whisper-class first; Conformer as the Gemma-family delta)
      |
 A3  e2e AUDIO→text gate on the smallest decoder-merge audio-LLM (Voxtral-Mini-3B)
```

- **A0 — Ground + vehicle + oracle/fit.** Confirm on dgx the oracle constructs +
  runs `whisper-small` and `Voxtral-Mini-3B` (native towers, transformers 5.13.1
  OK); fetch `whisper-small` (~0.5 GiB) then `Voxtral-Mini-3B` (~9.4 GiB, staged);
  capture the oracle log-mel feature reference + a fixed audio→text greedy golden +
  K=5 self-determinism (selects the gate form). **Gate:** oracle references
  produced; vehicle+fit confirmed. **GPU:** minimal (oracle under `flock`).
  **Risk:** audio decode determinism (resampling filter) — pin the resampler.
  **Critical path:** YES.
- **A1 — AUDIO INPUT pipeline (the largest new piece).** `src/vllm/multimodal/`
  audio processor: waveform decode → resample to the feature sampling_rate →
  log-mel feature extraction → `_unfold` framing → the placeholder-count
  arithmetic (mel frames → 2×stride-2 subsample → num soft tokens, cf.
  `gemma4_mm.py:348-368` for the Gemma shape). REUSE `MultiModalKwargs`,
  `MultiModalHasher`, `EncoderCacheManager`, the placeholder-expansion seam.
  **Gate:** feature-parity — our log-mel `input_features` + mask BIT/near-exact vs
  the oracle's feature extractor on a fixed clip, and the mm-hash byte-identical
  (mirrors M1 processor-parity 23/23, RED-first). **GPU:** CPU-only. **Risk:**
  log-mel numerics (window/FFT/mel-filterbank) are a silent-corruption hazard —
  gate against the oracle, never eyeball. **Critical path:** YES.
- **A2 — AUDIO encoder TOWER.** Start with the **Whisper-class encoder** (2×Conv +
  transformer; reuse `vt::` GEMM/attn/LayerNorm/GELU) proven faithful in ISOLATION
  vs a dumped oracle reference (mirrors M2a tower fidelity, bf16-envelope tol).
  Then the **USM Conformer** delta (Conv2d subsampling stride-2 semicausal +
  conv-module + relative-position attention + softcap) proven on
  **Granite-Speech-2b** — the Gemma-4-family tower. **Gate:** tower output rel-L2
  within the bf16 envelope vs the oracle dump; RED-first (disable a block → fail).
  **GPU:** tower forward under `flock`; **new kernels:** Conv2d subsample,
  Conformer conv-module, relative-position bias (compute-sanitizer 0). **Risk:**
  the Conformer relative-position + softcap buffers (the `post_init()` set) must
  match. **Critical path:** YES (proves the tower Gemma-4 audio reuses).
- **A3 — e2e AUDIO→text gate.** `Voxtral-Mini-3B`: native Whisper-class encoder
  (A2) + projector (RMSNorm+Linear, `vt::` ops) + masked-scatter merge (REUSE
  `VLGenerateCore`/`Qwen3VLMergeMultimodal` pattern) into the **LANDED Mistral**
  decoder → forked greedy decode. **Gate:** audio→text token-exact vs vLLM 0.25.0,
  greedy, gate form BY MEASUREMENT (K=5 self-determinism → STRICT else near-tie,
  per the ratified rule). **GPU:** full decode under `flock`, memory-careful (~9.4
  GiB, never OOM-reboot GB10). **Risk:** the projector-merge silent-corruption
  (wrong count/offset → fluent-wrong); localize with A1/A2 unit gates first.
  **Critical path:** YES — this is the first end-to-end AUDIO capability.

### 2.2 Gemma-4 campaign (STAGED / oracle-blocked)

```
 G0  Honesty pass (THIS spike): oracle verdict + config/registry + primitive inventory   [reachable now]
      |  ── BLOCKED until: oracle Transformers carries `gemma4` AND a fitting checkpoint downloads ──
 G1  Gemma-4 backbone (PLE/YOCO/KV-share/Gemma-4-MoE/k_eq_v) via nested text_config  [sweep-gemma §0.1 campaign]
      |
 G2  SigLIP vision tower (REUSE M2a scaffold minus merger/DeepStack/RoPE) + video → IMAGE/VIDEO gate
      |
 G3  Gemma-4 AUDIO (REUSE the A-track pipeline + the USM Conformer tower from A2) → AUDIO gate
```

- **G0 — Oracle-gateability + greedy golden [DONE 2026-07-28, `CLAIM-GEMMA4-W0`].**
  Supersedes the original honesty-pass scope: the oracle does NOT merely construct —
  it **LOADS+RUNS+GENERATES** `unsloth/gemma-4-E4B-it` on vLLM 0.25.0 (§0 banner).
  Captured: the STRICT K=5 greedy golden (`tests/parity/goldens/gemma4_e4b_text/gen_manifest.json`),
  the oracle command/versions, the arch/head-dim config resolution, the two-variant
  architecture (0.1), the reuse-vs-new map (0.2). **Gate PASSED:** a real greedy
  generation (32 coherent tokens, deterministic ⇒ STRICT bar for G-bringup). **GPU:**
  used (E4B under `flock`, GMU 0.30). This is the anchor for the future G1–G3 gates.
- **G1 — Gemma-4 backbone** (only after the block clears): the PLE/YOCO/Gemma-4-MoE/
  k_eq_v/double-wide-MLP/layer-scalar stack ([`sweep-gemma.md`](sweep-gemma.md)
  §0.1), reusing the landed Gemma text primitives (gemma-RMSNorm, sandwich norms,
  GeGLU `kGeluAndMul`, soft-cap `kSoftCap`) + the BF16 grouped-MoE GEMM. A large
  NEW-primitive campaign; **the critical Gemma-4 blocker beyond the oracle.**
- **G2 — SigLIP vision tower + video.** REUSE the M2a ViT scaffold, DROP
  merger/DeepStack/vision-RoPE, ADD learned abs pos-embed; projector = the audio
  projector op. **Gate:** image then video token-exact vs oracle.
- **G3 — Gemma-4 AUDIO.** REUSE the A-track audio pipeline + the USM Conformer
  tower (A2) + the projector-merge; wire the Gemma-4 audio-token id + count
  arithmetic (`gemma4_mm.py:348-395`). **Gate:** audio→text token-exact.

**Critical path:** the AUDIO track (A0–A3) is independent and reachable now; the
Gemma-4 campaign (G1–G3) is gated on the oracle advancing AND a checkpoint. G3
consumes the A-track output, so landing audio on Whisper/Voxtral first is strictly
on the Gemma-4 critical path too.

---

## 3. Honest blockers (mirror the GLM/DeepSeek blocked-row precedent)

- **Gemma-4 mm — oracle block RETIRED (W0 RUN-VERIFIED 2026-07-28); now
  IMPLEMENTATION-BLOCKED only (backbone + towers unbuilt).** The two gate-time
  blockers are both dissolved by measurement: (1) the pinned oracle vLLM 0.25.0 +
  transformers 5.13.1 DOES carry `transformers.models.gemma4` and **loads + runs +
  generates** the mm wrapper (STRICT greedy golden captured on `unsloth/gemma-4-E4B-it`);
  (2) an **ungated** vehicle exists and is cached (`unsloth/gemma-4-E4B-it` 15.99 GB,
  `unsloth/gemma-4-12b-it` 23.92 GB — no HF token needed). The ONLY remaining
  blockers are implementation: the PLE/YOCO/Gemma-4-MoE backbone + the USM-Conformer
  audio tower are unbuilt (the SigLIP vision tower is a reuse of M2a). HW fits (E4B
  ~15 GB in the 119 GiB pool). **The G-campaign is now unblocked to start.**
- **AUDIO modality — NOT blocked; STAGED on a smaller vehicle.** Reachable via
  native oracle-runnable audio models (Whisper, Voxtral-Mini-3B, Qwen2-Audio,
  Granite-Speech). Genuinely-new subsystems (audio pipeline + encoder tower), no
  reuse from image/video except the mm spine + merge pattern. Land it on
  `whisper-small`→`Voxtral-Mini-3B` (Mistral backbone LANDED), then carry to
  Gemma-4 (G3). Downloads not performed (staged, dgx disk tight —
  [[grid-per-sha-trees-fill-disk]]).
- **No HW-blocked modality for the audio vehicles** — all fit the 119 GiB unified
  pool with vast headroom.

---

## 4. Structured contract

### Scope
Design — not build — the Gemma-4 mm readiness (vision + audio towers + backbone)
and the genuinely-new AUDIO track, with the honest oracle/checkpoint/HW verdict
and a per-modality/per-model disposition + W-plan. Covers the two Gemma-4
`MODEL-MM-*` rows. In scope: the mm architecture characterization (§0.1); the
oracle-block finding (§0.0); the reuse-vs-new map against landed M0–M3 (§0.2); the
checkpoint/oracle/GB10 verdict (§0.3); the smallest audio-runnable vehicle (§0.4);
the AUDIO A0–A3 + Gemma-4 G0–G3 W-plans (§2); the blockers (§3).

OUT of scope, each with a reason: **implementation of anything** (spike — no code,
no tower, no build, no download, no gate). **Gemma-4 e2e** — oracle-blocked (§0.0)
+ ≥12B gated checkpoints + unbuilt backbone; honesty-pass only. **The Qwen3.6 mm
rows + `multimodal-track.md`** — owned by the concurrent Qwen3.6-video agent; not
touched. **gemma3n / GlmAsr / other audio rows** as targets — named as references
only; they stay `INVENTORIED`.

### Upstream chain
Registry `registry.py:392-396` (`gemma4_mm`,`gemma4_unified`). Wrapper+towers
`gemma4_mm.py` (vision `:1039-1051`, audio `:1055-1073`, projector `:908-965`,
audio forward `:1468-1490`, audio count `:348-395`, modality gating `:216-235`);
`gemma4_unified.py` (encoder-free `:3-16,73,220-291`). Backbone `gemma4.py`
([`sweep-gemma.md`](sweep-gemma.md) §0.1). Audio vehicles: `whisper.py:458`,
`voxtral.py:671,737`, `qwen2_audio.py:349`, `granite_speech.py:294`,
`gemma3n.py`/`gemma3n_audio_utils.py`. Landed mm spine (§0.2 anchors).
**Anchor-drift warning:** re-anchor every `file:line` at implementation time.

### Our baseline
REUSE §0.2 (mm spine, ViT scaffold, Gemma text primitives, Mistral backbone for
Voxtral). NEW §0.2 (audio pipeline, audio Conformer tower, Gemma-4 backbone).
No `MODEL-MM-gemma4-*` row is or becomes `DONE`; both stay `SPIKE` with the
oracle-blocked verdict.

### Tests to port (inventory only — nothing ported here)
| Upstream test | Tier | Ours (increment) |
|---|---|---|
| audio feature-extractor / processor parity (whisper/gemma4 feature extractor) | T-unit | log-mel `input_features` + mm-hash bit-parity (A1) |
| `tests/models/multimodal/generation/test_*audio*` / whisper / voxtral | T-e2e | audio→text token-exact (A2 tower, A3 e2e) |
| `tests/models/multimodal/generation/test_gemma4*` | T-e2e | SKIPPED — oracle-blocked (G-campaign), tracked reason |
| `tests/models/registry.py` Gemma-4 `_HfExamplesInfo` | T-unit | config/registry resolution (G0, no checkpoint) |

### Gates
1. **Inertness (SACRED, every increment).** All current SACRED gates
   byte-identical (audio/tower subsystems additive + gated on mm input).
2. **Audio feature parity (A1).** log-mel `input_features`+mask + mm-hash
   bit/near-exact vs the oracle on a fixed clip (RED-first).
3. **Audio tower fidelity (A2).** tower output rel-L2 within the bf16 envelope vs
   the oracle dump; RED-first.
4. **Audio e2e (A3, SACRED).** audio→text token-exact vs vLLM 0.25.0, gate form by
   measurement.
5. **Build/memcheck/records** — clean `-Werror`; compute-sanitizer 0 on new audio
   kernels (Conv2d subsample, Conformer conv-module, relative-pos bias); the record
   checkers green.
6. **SPEED** — PENDING; a target is `DONE` only at token-exact AND vLLM throughput.
7. **Blocked-row honesty (Gemma-4).** Record the oracle/checkpoint verdict + the
   primitive inventory; never claim more than a runnable gate backs.

### Dependencies
No hard upward dependency for the AUDIO track (the mm spine + Mistral backbone are
landed). Checkpoint deps (NOT performed): `whisper-small` ~0.5 GiB,
`Voxtral-Mini-3B` ~9.4 GiB, Granite-Speech-2b ~5 GiB; Gemma-4 ≥12B `google/*`
HF-gated. Stage sequentially (dgx disk tight). Downward deps introduced: the audio
input pipeline + audio tower are reusable by the entire audio family incl.
Gemma-4 (G3). **Blocking preconditions for the Gemma-4 rows only:** (a) the oracle
Transformers must carry `gemma4` (currently absent at 5.13.1); (b) a fitting
checkpoint downloads; (c) the Gemma-4 backbone campaign lands. Until all hold, the
Gemma-4 rows are an honesty pass.

### Risks/decisions
- **D1 — Gemma-4 mm is oracle-blocked, not just checkpoint-gated.** The towers are
  Transformers `AutoModel` modules absent from the pinned Transformers; this is the
  decisive blocker, verified by measurement (§0.0). Do not schedule Gemma-4 mm e2e
  until it clears.
- **D2 — Stand audio up on the smallest oracle-runnable vehicle, not Gemma-4.**
  Mirror Qwen3-VL-4B→Qwen3.6: Whisper (pipeline+tower) then Voxtral-Mini-3B (e2e
  merge on our landed Mistral). The Gemma-4 Conformer tower is the family delta,
  proven on Granite-Speech-2b.
- **D3 — log-mel + placeholder-count are silent-corruption hazards.** Wrong
  feature numerics or soft-token count emits fluent-wrong text (the OPT failure
  mode). Gate A1 against the oracle bit/near-exact, never eyeball.
- **D4 — Whisper-class ≠ Conformer.** The small vehicles de-risk the pipeline +
  merge but NOT the USM Conformer tower; do not claim the Gemma-4 audio tower is
  proven by a Whisper/Voxtral gate.
- **D5 — the Gemma-4 backbone is a separate campaign.** PLE/YOCO/Gemma-4-MoE/k_eq_v
  ([`sweep-gemma.md`](sweep-gemma.md) §0.1) is unbuilt and no other row needs it;
  it is not smuggled into an mm bring-up.
