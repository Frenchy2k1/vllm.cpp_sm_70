# SPIKE: Multimodal track (Audio / Video / Image) — Gemma-4 + Qwen3.6

**SPIKE ONLY — READ-ONLY design + a checkpoint/oracle-availability check. No
implementation, no vision tower built, no build, no download, no gate.** Grounds
the user's NEW TOP `roadmap_v1` priority (2026-07-25): *"Multimodal support
Audio/Video/Image with Gemma-4 and Qwen3.6"*, ahead of GPU-arch expansion, model
expansion, KV-to-disk, LMCache, and BEFORE DFlash (which moves to last). Framing
memory: [[multimodal-is-top-priority-gate-models-already-mm]].

**Base:** `72f9fb1` (`origin/main`, SPEC-MTP I7 landed). **Oracle pin:**
`/home/mudler/_git/vllm` @ `e24d1b24`. **dgx oracle:** `~/venvs/vllm-oracle` =
vLLM **0.25.0**. **Claim:** `CLAIM-MULTIMODAL-TRACK`.
**Gold-standard spike shape mirrored:** [`sweep-gemma.md`](sweep-gemma.md),
[`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md) (the BLOCKED-row
honesty precedent), [`mtp-spec-decode.md`](mtp-spec-decode.md) (the
campaign-decomposition + falsifiable-per-brick pattern).

Rows this spike advances:
- `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` (Qwen3.6-27B) — stays
  `PARTIAL` (text-only), narrative advanced with the mm-completion plan.
- `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` (Qwen3.6-35B) — same.
- `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` — `INVENTORIED` →
  `SPIKE` (BLOCKED-for-now honesty verdict).
- `MODEL-MM-gemma4-unified-gemma4-unified-for-conditional-generation` — same.

Named but LEFT at their current state (the vision family this UNLOCKS, not this
spike's targets): every other `MODEL-MM-*` row (Qwen2.5-VL, Qwen3-VL, GLM-4V,
InternVL, gemma3_mm, gemma3n_mm, …). Qwen3-VL-4B is named below as the first
*vehicle* but its row is not advanced until M2 owns it.

---

## 0. Headline findings (lead with the high-leverage framing)

### 0.0 Our two production GATE models are ALREADY multimodal — brought up text-only

The MVP gate models are not text architectures we would *add* vision to; they are
multimodal architectures we shipped with the vision half switched off:

- **Qwen3.6-27B** = `Qwen3_5ForConditionalGeneration`
  (`qwen3_5.py:389`), which **subclasses `Qwen3VLForConditionalGeneration`**
  (`qwen3_5.py:81-82,389`) and instantiates the **`Qwen3_VisionTransformer`**
  (`qwen3_vl.py:519`) over modalities `{"image", "video"}`
  (`qwen3_5.py:412-414`).
- **Qwen3.6-35B-A3B** = `Qwen3_5MoeForConditionalGeneration`
  (`qwen3_5.py:604`), same vision tower and modalities (`qwen3_5.py:624-626`).

So **completing Qwen3.6 multimodal completes models we already ship + benchmark**
(text-gen token-exact 235/235 and 315/315 vs vLLM 0.25.0). The vision half is the
*same* `Qwen3_VisionTransformer` used by the whole Qwen3-VL family, so the same
tower port unlocks Qwen3-VL-{2B,4B,8B,30B-A3B} as a side effect. The PRIMARY
vehicle question is therefore: **what does completing Qwen3.6-27B image+video
take, given we already own its text path + GDN-hybrid backbone?**

### 0.1 The oracle CAN construct the multimodal path (0.25.0), but our cached gate checkpoints are TEXT-ONLY

Two independently-measured gating facts (dgx, 2026-07-25):

1. **vLLM 0.25.0 (`~/venvs/vllm-oracle`) ships the multimodal model files** —
   `qwen3_5.py`, `qwen3_vl.py`, `qwen2_5_vl.py`, `qwen2_vl.py`, `gemma4_mm.py`,
   `gemma4_unified.py`, `gemma3_mm.py`, `gemma3n_mm.py` are all present in
   `site-packages/vllm/model_executor/models/`. So the SACRED oracle for the mm
   path EXISTS for every target here — this is a stronger starting position than
   Gemma-4-text had (there the worry was oracle construction; here the files are
   present). **Actual mm-forward run** on each is a W0/M0 verification, not
   assumed.
2. **Our cached NVFP4 gate checkpoints do NOT contain the vision tower.** Read
   the safetensors headers directly (metadata only, files already cached):
   - `unsloth/Qwen3.6-27B-NVFP4`: `architectures=["Qwen3_5ForConditionalGeneration"]`,
     `config.json` DOES declare a `vision_config` (depth 27, hidden 1152,
     out_hidden_size 5120, `deepstack_visual_indexes`, `spatial_merge_size`,
     `temporal_patch_size`), **but the weights contain 2111 tensors, ZERO named
     `visual.*`** — only `model.language_model.*`, `lm_head`, `mtp.*`. The unsloth
     NVFP4 quant is TEXT-ONLY (which is exactly why we could bring it up text-only).
   - `nvidia/Qwen3.6-35B-A3B-NVFP4`: same — `Qwen3_5MoeForConditionalGeneration`,
     `vision_config` declared, vision weights absent.
   - **No vision/VL checkpoint of any family is cached on dgx** (`ls ~/.cache/
     huggingface/hub` — only `wavlm-base-plus`, an unrelated audio model).

   **Consequence:** Qwen3.6 mm is NOT hardware-blocked and NOT oracle-blocked —
   it is **CHECKPOINT-blocked** until a vision-inclusive checkpoint is fetched
   (the full-precision Qwen3.6 release, or a mm-inclusive quant). M0 owns that
   fetch + the matching-checkpoint oracle run. The vision tower itself is tiny
   (depth 27 × hidden 1152 ≈ 0.5–0.7 B params, ~1–1.4 GiB bf16) — it fits the
   119 GiB GB10 unified pool trivially alongside the 27B/35B LLM.

### 0.2 Per-target modality — which of Audio/Video/Image is actually reachable

| Target | Image | Video | Audio | Vision tower | Oracle file (0.25.0) | Checkpoint on dgx | GB10 fit |
|---|:--:|:--:|:--:|---|:--:|:--:|---|
| **Qwen3.6-27B** (`Qwen3_5ForConditionalGeneration`) | ✅ | ✅ | ❌ | `Qwen3_VisionTransformer` (DeepStack) | ✅ `qwen3_5.py`+`qwen3_vl.py` | text-only quant cached; **vision-inclusive needed** | tower ~1 GiB + 27B — **FITS** |
| **Qwen3.6-35B-A3B** (`Qwen3_5MoeForConditionalGeneration`) | ✅ | ✅ | ❌ | same | ✅ | text-only quant cached; vision-inclusive needed | tower ~1 GiB + 35B MoE — **FITS** (per landed text run) |
| **Gemma-4** (`Gemma4ForConditionalGeneration` / `Gemma4Unified…`) | ✅ | ✅ | ✅ | SigLIP-class + **audio tower** | ✅ `gemma4_mm.py`,`gemma4_unified.py` | **none cached**; all public ckpts ≥12B, mm-wrapped, `google/*` HF-gated | 12B fits; 26B-A4B/31B HW-marginal-to-blocked (sweep-gemma §0.6) |
| **Qwen3-VL-4B** (first vehicle, `Qwen3VLForConditionalGeneration`) | ✅ | ✅ | ❌ | **`Qwen3_VisionTransformer` (identical to Qwen3.6)** | ✅ `qwen3_vl.py` | none cached (~9 GiB bf16 download) | **FITS trivially** |
| Qwen2.5-VL-3B (fallback vehicle, `Qwen2_5_VLForConditionalGeneration`) | ✅ | ✅ | ❌ | Qwen2.5-VL ViT (older, no DeepStack) | ✅ `qwen2_5_vl.py` | none cached (~7 GiB) | FITS |

**Audio honesty:** among the two named model targets, **only Gemma-4 does audio**
(image+video+audio; `gemma4_mm.py` wires an audio/ASR frontend). **Qwen3.6 has NO
audio path at all.** Audio is therefore reachable ONLY through the Gemma-4 lift
(or the separate `gemma3n_mm`/`*Speech*`/`*ASR*` families), and it needs a
genuinely NEW subsystem (audio encoder + feature frontend + audio preprocessing)
that no image/video work produces. It is deferred to M5, behind M4's Gemma-4
backbone, and called out as a large separate lift, not a near-term gate.

### 0.3 The first vehicle: **Qwen3-VL-4B-Instruct** (stands up the EXACT tower Qwen3.6 needs)

`Qwen/Qwen3-VL-4B-Instruct` (`Qwen3VLForConditionalGeneration`,
`tests/models/registry.py:1293-1294`) is the smallest genuinely-gateable vehicle
that de-risks the target directly: it instantiates the **same
`Qwen3_VisionTransformer`** (DeepStack, `qwen3_vl.py:519`) that
`Qwen3_5ForConditionalGeneration` reuses, is image+video, is oracle-runnable
(0.25.0 has `qwen3_vl.py`), is ~9 GiB bf16 (fits GB10 with vast headroom), and its
LLM backbone is a plain Qwen3-VL **dense text** decoder — a near-sibling of our
landed Qwen3-dense. So the ONLY new subsystem on the first gate is the vision +
input pipeline; the LLM half is already ours. Once M2 gates Qwen3-VL-4B image,
M3 reuses the tower **verbatim** and re-wires it onto the landed Qwen3_5
GDN-hybrid backbone for the actual Qwen3.6 target. Fallback vehicle
(`Qwen/Qwen2.5-VL-3B-Instruct`) is smaller and older but its ViT is NOT the
DeepStack tower — use it only if the Qwen3-VL-4B download is blocked.

### 0.4 Reuse-vs-new — most of the ENGINE is ours; the mm SUBSYSTEMS are all new

**REUSE (landed, unchanged):**
- **Paged attention + KV cache + the model-shape-agnostic runner**
  (`ENG-RUNNER-MODELSHAPE`) — the LLM decode over vision-produced embeds is the
  same paged path we run today; mm only changes how `input_embeds` are populated
  before the first decode.
- **Both LLM backbones for the target** — Qwen3_5 dense + MoE GDN-hybrid text is
  landed and token-exact; Qwen3-VL-4B's dense text is a Qwen3-dense sibling.
- **Tokenizer, sampling, logits path** — unchanged; mm tokens are placeholder ids
  the tokenizer already emits.
- **The LMCache key structure is already mm-AWARE** — our
  `chunked_token_database.{h,cpp}` mirrors vLLM's `ChunkedTokenDatabase` whose
  per-chunk key hashes the 3-tuple `(prefix, tokens, extra_keys)`; we currently
  pass `extra_keys=()` (text-only scope, `KV-EXTERNAL-CACHE` W4). Populating
  `extra_keys` with the mm-hash is a small, localized addition, not a new key
  scheme.

**GENUINELY NEW (nothing in `src/`/`include/` today — grep confirms zero mm
subsystems: no `MultiModalKwargs`, no encoder cache, no vision tower, no pixel
preprocessing):**
1. The **mm INPUT pipeline** — `MultiModalKwargs`, the `BaseMultiModalProcessor`
   (image/video preprocessing → pixel/grid features), placeholder-token
   expansion, and `MultiModalHasher` (mm-hash).
2. The **vision TOWER** — `Qwen3_VisionTransformer` (patch embed, ViT blocks,
   patch merger, DeepStack multi-level injection) + the projector/merge into
   `input_embeds`.
3. The **encoder-cache ENGINE seam** — `EncoderCacheManager`, `EncoderRunner`,
   the scheduler encoder-budget + chunked-prefill-mm hooks.
4. The **serving mm ingestion** — `image_url`/video parsing on the OpenAI server,
   `--limit-mm-per-prompt`.
5. (M5) the **audio encoder + ASR frontend** — Gemma-4/gemma3n only.

---

## 1. The multimodal SEAM MAP — vLLM `file:line` → what we build

### 1.1 Input pipeline (`vllm/multimodal/`)

| vLLM seam | file:line @ `e24d1b24` | What we build |
|---|---|---|
| `MultiModalKwargs` / mm inputs container | `vllm/multimodal/inputs.py` (32 KB) | NEW C++ `MultiModalKwargs` (typed per-modality feature tensors + grid metadata) |
| `BaseMultiModalProcessor` (the input processor) | `vllm/multimodal/processing/processor.py:972`; `apply` `:1663`; `_get_prompt_updates` `:1020`; `_get_mm_fields_config` `:1011`; `_call_hf_processor` `:1097` | NEW processor: HF image/video preprocess → pixel/grid features + placeholder-token expansion |
| `BaseProcessingInfo` | `vllm/multimodal/processing/context.py:296` | NEW per-model processing-info (num-mm-tokens per item, grid sizing) |
| `MultiModalHasher` (mm-hash) | `vllm/multimodal/hasher.py:50` (`serialize_item`) | NEW mm-hash (blake3/sha256 over serialized media) → feeds encoder cache + LMCache `extra_keys` |
| image / video / audio preprocessing | `vllm/multimodal/image.py`, `video.py` (64 KB), `audio.py` | NEW image preprocess (M1); video frame sampling (M3); audio (M5) |
| placeholder-token expansion + merge target | `vllm/model_executor/models/utils.py::_merge_multimodal_embeddings:524` (masked scatter `inputs_embeds[is_multimodal]=…:545`) | NEW `_merge_multimodal_embeddings` (scatter mm embeds into the placeholder positions of `input_embeds`) |

### 1.2 The `*ForConditionalGeneration` wrapper + vision tower

| vLLM seam | file:line | What we build |
|---|---|---|
| `SupportsMultiModal` protocol (`embed_multimodal`, `get_input_embeddings`, `get_language_model`, `get_placeholder_str`) | `vllm/model_executor/models/interfaces.py:94,147,176,390-404` | NEW `SupportsMultiModal` mixin contract on the wrapper |
| Qwen3.6 wrapper | `qwen3_5.py:389-453` (`_mark_tower_model {"image","video"}` `:412`; `self.visual = Qwen3_VisionTransformer` `:413`; merge `:447`) | NEW `Qwen3_5ForConditionalGeneration` wrapper over the LANDED text model |
| Qwen3-VL vision tower | `qwen3_vl.py`: `Qwen3_VisionPatchEmbed:347`, `Qwen3_VisionBlock:413`, `Qwen3_VisionPatchMerger:467`, `Qwen3_VisionTransformer:519`, DeepStack `:543-608,821-840` | NEW vision tower (patch embed, ViT blocks, merger, DeepStack) — reuse `vt::` GEMM/attn/norm |
| image/video feature → embeds | `qwen3_vl.py::_process_image_input:2143`, `_process_video_input:2165`, `embed_multimodal:2731`, `get_input_embeddings`/forward `:2843` | NEW per-modality feature-extraction glue |
| tower weight loading | `qwen3_vl.py::load_weights:2905`, vision-tower `load_weights:843` | NEW `visual.*` weight map (added to the landed text loader) |

### 1.3 Engine seams (scheduler / encoder cache / runner / serving)

| vLLM seam | file:line | What we build |
|---|---|---|
| `EncoderCacheManager` (allocate/free/budget/mm-hash) | `vllm/v1/core/encoder_cache_manager.py:17` (`check_and_update_cache:94`, `can_allocate:123`, `allocate:184`, `get_freed_mm_hashes:255`, `compute_mm_encoder_budget:269`) | NEW encoder-cache manager (keyed by mm-hash) |
| scheduler mm hooks | `vllm/v1/core/sched/scheduler.py:205-228` (budget wiring), `:616,1004-1011` (allocate), `:1356-1467` (mm-hash schedule + chunked-mm), `:1905-1935` (free) | NEW scheduler encoder-budget + per-item scheduling + free path |
| chunked-prefill × mm placeholders | `scheduler.py:1405-1420` (`disable_chunked_mm_input`, don't split a mm item) | NEW chunked-prefill mm-placeholder guard |
| `MultiModalBudget` | `vllm/multimodal/encoder_budget.py` | NEW encoder-budget sizing |
| `EncoderRunner` (execute encoder, gather embeds, splice) | `vllm/v1/worker/gpu/mm/encoder_runner.py:13` (`prepare_mm_inputs:35`, `execute_mm_encoder:52`, `gather_mm_embeddings:64`, `get_inputs_embeds:148`) | NEW worker-side encoder runner + `EncoderCache` |
| model-runner mm wiring | `vllm/v1/worker/gpu/model_runner.py:181-186,315,668-673,737-754` | NEW: build encoder cache + reset/remove-request hooks (inert when `supports_mm_inputs=False`) |
| serving `image_url`/video ingestion | `vllm/entrypoints/chat_utils.py:726` (`MultiModalDataDict`), `:873,955` (`parse_image`, `fetch_image`) | NEW OpenAI-server mm content parsing |
| `--limit-mm-per-prompt` + mm config | `vllm/config/multimodal.py` | NEW mm config + CLI/server flag |

### 1.4 mm-hash ↔ LMCache

`MultiModalHasher` (`hasher.py:50`) produces the per-item mm-hash that keys the
encoder cache AND flows into the KV-cache key as `extra_keys`. vLLM's
`ChunkedTokenDatabase` (LMCache) hashes `(prefix, tokens, extra_keys)`; our
landed `src/vllm/v1/kv_offload/lmcache/chunked_token_database.{h,cpp}` already
carries the `extra_keys` slot (currently `()`, text-only). Populating it with the
mm-hash is the ONLY LMCache change the mm track needs — the key scheme is
unchanged.

---

## 2. Structured contract

### Scope
Design — not build — the multimodal track (Image → Video → Audio) with the
primary vehicle question framed on completing Qwen3.6-27B/35B (already-shipped
text-only mm architectures), plus a smaller first vehicle (Qwen3-VL-4B) to stand
the track up on, and the honest Gemma-4 verdict. Covers the four rows named in
the header. In scope: the seam map (§1); per-target modality + oracle + GB10-fit
+ checkpoint gateability (§0.2); the reuse-vs-new factoring (§0.4); the ordered
M0–M5 W-plan (§3); per-increment gates, GPU/CPU, risk, critical path; and the
HW/oracle/checkpoint-blocked honesty (§0.1, §4).

OUT of scope, each with a reason: **implementation of anything** (spike — no code,
no tower, no build, no download, no gate). **Audio e2e** beyond a design + honesty
verdict — it needs a NEW audio encoder + ASR frontend reachable only via Gemma-4
(M5), a large separate lift. **Gemma-4 e2e** beyond a characterization/honesty
pass — its only checkpoints are ≥12B mm-wrapped + `google/*` HF-gated and it needs
the PLE/YOCO/Gemma-4-MoE backbone (sweep-gemma §0.1) PLUS vision PLUS audio
towers; staged in M4. **The rest of the vision family** (GLM-4V, InternVL,
gemma3_mm, …) — unlocked by the track but not this spike's targets; they stay at
their current state.

### Upstream chain
Registry: `qwen3_5.py:389,604` (`registry.py:556,557-560`), `qwen3_vl.py`
(`registry.py:551`), `qwen2_5_vl.py` (`:526-529`), `gemma4_mm.py` (`:392`),
`gemma4_unified.py` (`:393-396`), `gemma3_mm.py` (`:383`). Input pipeline
`vllm/multimodal/` (§1.1). Wrapper + tower `qwen3_5.py`+`qwen3_vl.py` (§1.2).
Engine `vllm/v1/core/encoder_cache_manager.py`, `sched/scheduler.py`,
`v1/worker/gpu/mm/encoder_runner.py`, `model_runner.py`, `entrypoints/chat_utils.py`,
`config/multimodal.py` (§1.3). **Anchor-drift warning:** re-anchor every cited
`file:line` at implementation time.

### Our baseline
REUSE §0.4 (paged path, both landed LLM backbones, tokenizer, sampling, the
LMCache `extra_keys` seam). NEW §0.4 (the five mm subsystems). No `MODEL-MM` row
is or becomes `DONE`; the two Qwen rows stay `PARTIAL` and the Gemma-4 rows move
to `SPIKE` with a BLOCKED-for-now verdict.

### Tests to port (inventory only — nothing ported here)
Per [`.agents/test-porting.md`](../test-porting.md):
| Upstream test | Tier | Ours (increment) |
|---|---|---|
| `tests/multimodal/test_processing.py` (processor + placeholder expansion) | T-unit | processor-parity vs vLLM's processor output (M1) |
| `tests/multimodal/test_hasher.py` | T-unit | mm-hash byte-agreement (M1) |
| `tests/v1/core/test_encoder_cache_manager.py` | T-unit | encoder-cache manager (M1) |
| `tests/models/multimodal/generation/test_qwen*_vl.py` (or the pinned equivalent) | T-e2e | image token-exact on Qwen3-VL-4B (M2), Qwen3.6 (M3) |
| `tests/models/multimodal/generation/test_common.py` (video) | T-e2e | video token-exact (M3) |
| `tests/models/multimodal/generation/test_gemma*` | T-e2e | SKIPPED — Gemma-4 (M4) / audio (M5), tracked reason |

### Gates
1. **Inertness (SACRED, non-negotiable, EVERY increment).** With no mm input, all
   current SACRED gates byte-identical (27B 235/235, 35B 315/315, Qwen3-Coder,
   Qwen3-dense, OPT, DeepSeek-V2-Lite, Llama-3.2-1B, Mistral-7B, GLM, Gemma-1/2/3,
   OLMo-2). The mm subsystems are additive + default-inert (`supports_mm_inputs`
   gates every hook).
2. **Processor parity (M1).** Our processor's placeholder-token ids + `mm_kwargs`
   tensor shapes/values match vLLM's `BaseMultiModalProcessor.apply` on a fixed
   image, and the mm-hash is byte-identical.
3. **Image token-exact (M2/M3, SACRED).** Greedy token-exact vs vLLM 0.25.0 on a
   FIXED image+prompt — Qwen3-VL-4B (M2), then Qwen3.6-27B (M3). Gate form by
   measurement per [[near-tie-distributional-gate]] (a vision-conditioned decode
   may sit in a bf16 near-tie band → distributional fallback only if measured).
4. **Video token-exact (M3).** Same, on a fixed short video (frame-sampling
   parity is the risk).
5. **Build / memcheck / records** — clean `-Werror`; `compute-sanitizer` 0 on new
   vision/encoder kernels; the five record checkers green.
6. **SPEED.** PENDING per increment; a target is `DONE` only at token-exact AND
   vLLM throughput on every axis (mm encoder-cache reuse + chunked-prefill mm are
   the throughput levers).
7. **Blocked-row honesty (Gemma-4, audio).** Record the oracle/checkpoint verdict
   and the primitive inventory; never claim more than a runnable gate backs.

### Dependencies
No hard upward dependency on unlanded engine work — the paged path + both LLM
backbones are landed. Checkpoint dependencies (downloads, NOT performed): a
vision-inclusive Qwen3.6-27B/35B checkpoint (our NVFP4 caches are text-only, §0.1);
`Qwen/Qwen3-VL-4B-Instruct` (~9 GiB); `Qwen/Qwen2.5-VL-3B-Instruct` (~7 GiB,
fallback); Gemma-4 (≥12B, `google/*` HF-gated). Stage sequentially
([[grid-per-sha-trees-fill-disk]] — dgx disk tight). Downward dependency this
introduces: the mm input pipeline + encoder cache are reusable by the ENTIRE
vision family; the mm-hash `extra_keys` closes the LMCache mm gap.

---

## 3. The dispatch-sized W-PLAN (M0–M5)

Each increment is independently gateable; the inertness gate (Gate 1) rides on
every one.

```
 M0  Ground + vehicle + oracle/fit/checkpoint         [CRITICAL PATH]
      |
 M1  mm INPUT pipeline + encoder-cache engine seam     [CRITICAL PATH, foundation]
      |
 M2  first vision TOWER + merge on Qwen3-VL-4B -> IMAGE gate   [CRITICAL PATH]
      |
 M3  complete Qwen3.6 IMAGE (reuse M2 tower) -> then VIDEO     [CRITICAL PATH -> user's target]
      |
 M4  Gemma-4 (staged: vision + PLE/YOCO/MoE backbone; honesty-pass blocked pieces)
      |
 M5  AUDIO if reachable (Gemma-4 / gemma3n audio encoder + ASR frontend)
```

**M0 — Ground the facts; pick the first vehicle; confirm oracle + fit + checkpoint.**
- Builds: nothing (design + measurement). Confirm (DONE in this spike): our cached
  NVFP4 gate checkpoints are text-only (no `visual.*`); 0.25.0 has the mm model
  files. Remaining M0 work: fetch `Qwen/Qwen3-VL-4B-Instruct`; run the 0.25.0
  oracle on it on a FIXED image+prompt to produce reference tokens; measure the
  vision-tower param count + peak-mem fit; decide the vision-inclusive Qwen3.6
  checkpoint source for M3.
- Gate: oracle mm reference outputs produced; vehicle selected; fit measured;
  checkpoint plan recorded. GPU: minimal (one oracle run under `flock`).
- Hardest risk: the Qwen3.6 vision-inclusive checkpoint may only exist as a large
  bf16 release (no mm-inclusive NVFP4) → M3 loads a bf16 tower alongside the NVFP4
  LLM (mixed-precision load) or downloads more.
- Critical path: YES (unblocks everything).

**M1 — The mm INPUT pipeline + encoder-cache engine seam (the foundation, inert
without mm input).**
- Builds: `MultiModalKwargs`; a `BaseMultiModalProcessor` port (image preprocess +
  placeholder expansion + `_get_mm_fields_config`); `MultiModalHasher`;
  `EncoderCacheManager` + `EncoderRunner` scaffold + scheduler encoder-budget /
  chunked-mm hooks; `_merge_multimodal_embeddings` (masked scatter). All gated on
  `supports_mm_inputs` so text engines are byte-identical.
- Gate: Gate 1 (inertness) + Gate 2 (processor parity: placeholder ids +
  `mm_kwargs` shapes + mm-hash byte-identical vs vLLM on a fixed image);
  encoder-cache-manager unit vs `test_encoder_cache_manager.py`. GPU: mostly CPU
  (preprocessing/hashing/scheduling); no vision compute yet.
- Hardest risk: placeholder-token expansion is a SILENT-corruption hazard (wrong
  count/position emits fluent-but-wrong text, the OPT failure mode) — gate the
  processor output against vLLM, not by eyeball. Chunked-prefill must not split a
  mm item (`disable_chunked_mm_input`).
- Critical path: YES.

**M2 — First vision TOWER + projector/merge on Qwen3-VL-4B → first IMAGE gate.**
- Builds: `Qwen3_VisionTransformer` (patch embed, ViT blocks over `vt::` GEMM/attn/
  norm, patch merger, DeepStack multi-level injection); the `visual.*` weight map;
  wire `embed_multimodal` → `_merge_multimodal_embeddings` → the landed
  Qwen3-VL-dense decode.
- Gate: Gate 3 (image token-exact vs vLLM 0.25.0 on Qwen3-VL-4B, fixed
  image+prompt, greedy; form by measurement). GPU: YES (vision GEMM/attn +
  paged LLM decode). Fits trivially.
- Hardest risk: DeepStack multi-level feature injection (`qwen3_vl.py:821-840`);
  vision-tower attention numerics + vision position/RoPE; the pixel→patch grid
  (`grid_thw`) must match the processor exactly.
- Critical path: YES (proves the tower Qwen3.6 reuses).

**M3 — Complete Qwen3.6-27B IMAGE (reuse M2 tower verbatim), then VIDEO.**
- Builds: attach the M2 `Qwen3_VisionTransformer` to the LANDED
  `Qwen3_5ForConditionalGeneration`/`…Moe…` wrapper over the GDN-hybrid backbone
  (the vision half is identical to Qwen3-VL; only the LLM backbone differs, and it
  is already ours + token-exact); then video frame sampling + `video_grid_thw`.
- Gate: Gate 3 on Qwen3.6-27B image (needs the M0 vision-inclusive checkpoint),
  then Gate 4 video. Flips both Qwen rows `PARTIAL` → mm-complete (correctness;
  speed still pending). GPU: YES; GB10 fit confirmed (tower ~1 GiB + 27B/35B).
- Hardest risk: the vision-inclusive-checkpoint precision (bf16 tower + NVFP4 LLM
  mixed load, from M0); video preprocessing parity (temporal patch + frame
  sampling).
- Critical path: YES — this is the user's stated target.

**M4 — Gemma-4 (staged: vision tower + backbone stack; honesty-pass the blocked
pieces).**
- Builds (only if the preconditions clear): the Gemma-4 SigLIP-class vision tower
  + the PLE/YOCO/Gemma-4-MoE backbone (sweep-gemma §0.1 — a large NEW-primitive
  stack no other row uses) via the nested `text_config`. Until a fitting checkpoint
  downloads AND the 0.25.0 oracle runs the mm forward, this is a
  characterization/honesty pass (the row records `SPIKE`/BLOCKED-for-now).
- Gate: Gate 7 (honesty) first; Gate 3 image only after the preconditions clear.
  GPU: heavy (12B mm). 
- Hardest risk: ALL public Gemma-4 checkpoints are ≥12B mm-wrapped +`google/*`
  HF-gated + the PLE/YOCO/MoE backbone is unbuilt; audio tower on top.
- Critical path: NO — parallel/deferred; does not block the Qwen3.6 target.

**M5 — AUDIO if reachable (Gemma-4 / gemma3n).**
- Builds: a NEW audio encoder + ASR/feature frontend + audio preprocessing —
  reachable ONLY via Gemma-4 (or `gemma3n_mm`/`*Speech*`/`*ASR*`); Qwen3.6 has no
  audio path. A large separate lift with no reuse from image/video.
- Gate: Gate 7 honesty; then audio token-exact on the smallest oracle-runnable
  audio+text model that fits. GPU: model-dependent.
- Hardest risk: it is a genuinely new modality subsystem gated behind M4's
  Gemma-4 backbone; honestly deferred until image+video land.
- Critical path: NO — the last, largest, most-deferred lift.

---

## 4. Blocked / deferred honesty (mirror the GLM/DeepSeek blocked-row precedent)

- **Qwen3.6-27B/35B multimodal — NOT blocked, CHECKPOINT-gated.** Oracle present
  (0.25.0 `qwen3_5.py`+`qwen3_vl.py`), tower fits GB10 trivially; the only gate to
  a run is fetching a vision-inclusive checkpoint (our NVFP4 caches are text-only,
  §0.1). Reachable modalities: **image + video** (no audio). This is the primary,
  achievable target.
- **Gemma-4 (`Gemma4ForConditionalGeneration`, `Gemma4Unified…`) — SPIKE /
  BLOCKED-for-now.** Oracle FILE present (0.25.0 `gemma4_mm.py`), but no checkpoint
  cached, all public checkpoints ≥12B mm-wrapped + `google/*` HF-gated, and it
  needs the PLE/YOCO/Gemma-4-MoE backbone (sweep-gemma §0.1) + vision + audio
  towers. Advanced `INVENTORIED` → `SPIKE`; NOT implemented; M4 is a
  characterization pass until a fitting checkpoint downloads and the oracle runs
  the mm forward.
- **Audio — deferred (M5), reachable only via Gemma-4/gemma3n.** A large separate
  lift (new audio encoder + ASR frontend); Qwen3.6 has no audio. Honestly not a
  near-term gate.
- No HW-blocked modality for the Qwen3.6 target: the vision tower is ~1 GiB and
  fits the 119 GiB unified pool alongside the 27B/35B LLM.
