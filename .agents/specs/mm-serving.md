# Multimodal serving — wiring image/audio/video into the OpenAI server (`ROAD-V1-MM` serving)

Row IDs: `MM-SERVE-PARSE` (this brick), `MM-SERVE-ENGINE`, `MM-SERVE-E2E`.
Owner claim: `CLAIM-MM-SERVING-W1`. Pinned vLLM oracle: `${VLLM_SOURCE}` @ `555967922`
(0.26.0.dev0).

## Problem

The multimodal INPUT pipeline (image/video/audio → processed features +
placeholder-expanded prompt) is correctness-complete on the single-sequence path
(`MODEL-MM`: Qwen3-VL-4B + Qwen3.6-27B image/video STRICT 32/32, Voxtral audio
14/14). It is **NOT wired into the OpenAI server**: the chat request `content` was
a bare string and multimodal content-part arrays were dropped
(`protocol.cpp` `from_json(ChatMessage)` handled only `content.is_string()`;
`serving_chat.{h,cpp}` deferred "multimodal").

## Ground truth (both sides, file:line)

### Ours — the mm INPUT pipeline that already exists (unwired)
- `include/vllm/multimodal/inputs.h` — `MultiModalInputs` (placeholder-expanded
  `prompt_token_ids` + `mm_features`), `MultiModalFeatureSpec`
  (mm_hash/modality/offset/length + `ImageKwargs`/`AudioKwargs`).
- `include/vllm/multimodal/qwen3vl_processor.h:87` `ProcessImage(rgb,H,W)` →
  `ImageKwargs`; `:113` `ExpandImagePlaceholders(...)`; `:90` `HashImage`. Needs
  only `Qwen3VLProcessorConfig` (no model weights).
- `include/vllm/multimodal/audio_processor.h:50` `DecodeWavPcm16Mono`; `:81`
  `ProcessWaveform` → `AudioKwargs`; `:98` `ExpandAudioPlaceholders`; `:86`
  `HashAudio`. Needs `AudioProcessorConfig` + the golden mel filterbank (no model
  weights).
- How the single-sequence path drives these (the reference): the e2e test
  `tests/vllm/multimodal/test_qwen3vl_e2e.cpp:112-158` — fixture RGB →
  `ProcessImage` → grid `[1,28,28]`/196 → tower → greedy. **Note:** even the
  single-seq e2e consumes PRE-DECODED raw RGB (`image_rgb_uint8_448x448x3.bin`)
  and a PRE-TOKENIZED prompt (`input_ids_i32.bin`); it never decodes PNG/JPEG and
  never tokenizes. The audio parity gate `test_audio_processor.cpp:91-107` decodes
  a committed PCM16 WAV → processor entirely on CPU.
- `include/vllm/v1/core/encoder_cache_manager.h` — the scheduler/encoder-cache
  seam that budgets/allocates per mm item (consumer of `mm_features`).

### The serving gap
- `include/vllm/entrypoints/openai/protocol.h:352` (`ChatMessage`) — content was a
  bare-string T0; `src/.../protocol.cpp` `from_json(ChatMessage)` only read
  `content.is_string()`.
- `include/vllm/entrypoints/openai/serving_chat.h` — `ChatPromptFn` seam renders
  `messages` → a prompt STRING; `create_chat_completion`
  (`serving_chat.cpp:577`) builds `prompt` then `add_request(prompt, params)` /
  `generate(prompt, ...)`. The engine `add_request` (`llm_engine.h:87,96`) takes a
  string OR pre-tokenized ids — **no mm-features overload**.

### vLLM's mm chat handling (MIRROR)
- `vllm/entrypoints/chat_utils.py:1478` `MM_PARSER_MAP` + `:1524`
  `_parse_chat_message_content_mm_part` — the content-part schema and dispatch:
  - `"text"` → `part["text"]`;
  - `"image_url"` → `part["image_url"]["url"]` (a `data:image/…;base64,…` URI or
    an http URL);
  - `"input_audio"` → `part["input_audio"]` = `{data:<base64>, format:str}`;
  - `"audio_url"` → `part["audio_url"]["url"]`;
  - others (`image_embeds`/`video_url`/`prompt_embeds`/`image_pil`/…) — residual.
- `vllm/multimodal/utils.py:35-113` — the `data:{mimetype};base64,{b64}` media
  encoding (we mirror the decode). Decode/route to a `MultiModalDataDict`, then
  the placeholder string is inserted into the templated prompt by the per-model
  parser (`_add_placeholder`, `chat_utils.py:886`).
- The OpenAI serving layer passes the `MultiModalDataDict` to the engine
  alongside the rendered prompt; the engine's input processor runs the mm
  processor and the encoder cache holds the features.

## Full wiring path (decomposed into bricks)

```
chat request content-part array          [MM-SERVE-PARSE — CPU, THIS BRICK]
  → parse typed parts (text/image_url/input_audio/audio_url)   protocol.cpp
  → decode data:/base64 payload → raw bytes                    chat_mm.cpp
  → route bytes through the EXISTING processor                 chat_mm.cpp
      · audio: WAV → ProcessWaveform → ExpandAudioPlaceholders (FULL on CPU)
      · image: RGB → ProcessImage    → ExpandImagePlaceholders (RGB in; codec = residual)
  → MultiModalInputs (expanded prompt ids + mm_features)       [asserted, CPU]

  → attach MultiModalInputs to the engine request              [MM-SERVE-ENGINE — CPU-ish]
      · new LLMEngine::add_request(prompt_ids, mm_features, params) overload
      · thread mm_features onto Request / EngineCoreRequest (fields already exist
        on Request for APC extra-keys) + encoder-cache budgeting
      · placeholder-string insertion into the chat-template prompt (chat_mm ↔
        ChatPromptFn), mirroring chat_utils `_add_placeholder`

  → mm model forward on the GPU worker                         [MM-SERVE-E2E — GPU, DGX]
      · encoder tower consumes mm_features; DeepStack/MRoPE inject
      · a real image+prompt OpenAI request → token-correct output
```

### Bricks

| Row | Scope | vLLM mirror | Tests to port | HW |
|-----|-------|-------------|---------------|----|
| `MM-SERVE-PARSE` (**LANDED, this commit**) | content-part parse + base64/data-URI decode + route to the existing processor → `MultiModalInputs` (shapes asserted) | `chat_utils.py:1478,1524`; `multimodal/utils.py:35-113` | `tests/entrypoints/test_chat_utils.py` (`test_parse_chat_messages_*`, the `image_url`/`input_audio` cases) → `test_chat_mm.cpp` | CPU |
| `MM-SERVE-ENGINE` (residual) | attach `MultiModalInputs` to the engine request (new `add_request` mm overload + `Request`/`EngineCoreRequest` plumbing + encoder-cache budget + placeholder-string insertion into the chat template) | `v1/engine/llm_engine.py add_request`; `v1/engine/core.py`; `chat_utils.py _add_placeholder:886` | `tests/v1/engine/test_processor_multi_modal_uuids.py`; `tests/entrypoints/test_chat_utils.py` placeholder-insertion cases | CPU-verifiable (shapes/counts); no forward |
| `MM-SERVE-E2E` (**MANDATORY closing gate, residual**) | a real image+prompt OpenAI `/v1/chat/completions` request → token-correct output on **Qwen3-VL-4B**, gated vs the mm oracle (reuse the M2c e2e golden `gen_tokens_i32.bin`) | `entrypoints/openai/chat_completion/serving.py` full mm path | new `test_openai_serving_chat_mm_e2e.cpp` (dgx-only, skip without ckpt+CUDA) | **DGX GB10 + Qwen3-VL-4B checkpoint** |

## This brick (`MM-SERVE-PARSE`) — landed

- `include/vllm/entrypoints/openai/protocol.h` — `struct ChatContentPart` (mirrors
  MM_PARSER_MAP fields) + `ChatMessage.content_parts` (nullopt for bare-string ⇒
  T0 byte-identical).
- `src/.../protocol.cpp` `from_json(ChatMessage)` + `ParseChatContentPart` — the
  array form fills `content_parts` and sets `content` to the joined text spans so
  the existing text prompt path is unchanged.
- `include/vllm/entrypoints/openai/chat_mm.h` + `src/.../chat_mm.cpp` —
  `DecodeBase64` (RFC 4648), `DecodeDataUri` (RFC 2397 `;base64`), the per-part
  decoders, and `RouteAudioWav` / `RouteImageRgb` reusing the existing processor
  seams to produce `MultiModalInputs`.
- Gate `tests/vllm/entrypoints/openai/test_chat_mm.cpp` (reuses the M1 image + A1
  audio processor-parity fixtures): inertness (bare-string unchanged); base64 /
  data-URI decode vectors; audio `input_audio` part → features `[80,3000]` + 1500
  placeholder tokens + byte-exact mm-hash; image `image_url` part → grid
  `[1,28,28]` + 196 merged tokens.

### Named residuals (out of this brick)
- **Container-format image decode (PNG/JPEG → RGB + dims):** no codec is vendored;
  `RouteImageRgb` consumes raw RGB (as the single-seq e2e path itself does). A
  real `data:image/png;…` needs a decoder — belongs to `MM-SERVE-ENGINE`/E2E or a
  small codec brick.
- **http(s) media-URL fetch** (vs inline `data:`/base64) — `DecodeDataUri`/
  `DecodeInputAudioPart` handle only inline payloads.
- **Engine plumbing** (`MM-SERVE-ENGINE`): no `add_request` mm overload yet; the
  produced `MultiModalInputs` is asserted but not yet handed to the engine.
- **Streaming mm, multiple images, video parts, image_embeds/prompt_embeds** — the
  parse tolerates them (empty payload under their `type`) but they are not routed.

## Correctness / gates
- CPU (this brick): `test_chat_mm` 5/5, 65 asserts; inertness suites
  `test_openai_protocol`/`test_openai_serving`/`test_openai_serving_chat_stream`
  byte-identical for bare-string; clean `-Werror` full-library + `server` build.
- **GPU closing gate (`MM-SERVE-E2E`):** a real image+prompt OpenAI request →
  token-correct output on Qwen3-VL-4B vs the mm oracle. Needs the DGX + the
  Qwen3-VL-4B checkpoint; out of scope for the CPU commit.
