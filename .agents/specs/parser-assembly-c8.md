# ROAD-V1-C8 — parser ASSEMBLY layer (`TOOLS-STREAMING-PARSER` assembly)

Status: **LANDED + CPU-gated 2026-07-27** (`CLAIM-ROADMAP-C8-ASSEMBLY`; config
families `CLAIM-ROADMAP-C8-CONFIGS`; not pushed). Builds directly on the merged
engine CORE ([streaming-parser-engine.md](streaming-parser-engine.md)); do NOT
re-port the engine core. Engine-backed families now ported: qwen3, seed_oss,
kimi_k2, **minimax_m2, glm47_moe, deepseek_v4, deepseek_v32, nemotron_v3**;
**DEFERRED: gemma4, inkling** (each needs an unported assembly hook — see
RESIDUALS).

## What this is

The engine core (`StreamingParserEngine`) emits a `SemanticEvent` stream. This
layer is the vLLM 0.26 `ParserEngine` that CONSUMES that stream and produces the
serving-visible output:

- streaming `DeltaMessage`s (reasoning / content, and tool-call deltas:
  name-first then argument deltas), and
- the one-shot `ExtractedToolCallInformation` (`extract_tool_calls`), plus the
  non-streaming `extract_reasoning` / `parse`.

## Ported FROM (pinned oracle, vLLM `555967922`, 0.26.0.dev0)

| Upstream | Ported to |
|---|---|
| `vllm/parser/engine/parser_engine.py:79` (`ParserEngine`, `ToolCallSlot`, `_events_to_delta`, `_safe_arg_prefix`, `_compute_arg_delta`/`_flush_arg_converter`, `_build_extracted_result`, `_coalesce_tool_call_deltas`, `_handle_*`, `_strip_trailing_reasoning`) | `include/vllm/parser/engine/parser_engine.h`, `src/vllm/parser/engine/parser_engine.cpp` |
| `vllm/parser/qwen3.py:68` (`_qwen3_arg_converter`, `_PARAM_RE`, `_PARTIAL_PARAM_RE`) + `:194-197` assembly fields | `src/vllm/parser/engine/configs.cpp` (converter in `qwen3_config`) |
| `vllm/parser/kimi_k2.py:150` (`KimiK2Parser` header-id/name overrides) | `include/vllm/parser/kimi_k2.h`, `src/vllm/parser/kimi_k2.cpp` |
| `vllm/parser/seed_oss.py:23` (wrapper-token variant of qwen3) | `get_parser_engine("seed_oss", …)` |
| `vllm/parser/parser_manager.py` + `vllm/parser/engine/registered_adapters.py` (name→parser dispatch) | `include/vllm/parser/parser_manager.h`, `src/vllm/parser/parser_manager.cpp` |
| `parser_engine_config.py:76-97` assembly fields (arg_converter, stream_arg_deltas, strip_*/validate_tool_names/drop_ws/arg_structural_chars) | `include/vllm/parser/engine/parser_engine_config.h` (added to the existing struct) |
| `json.dumps(…, ensure_ascii=False)` default `(", ", ": ")` separators | `include/vllm/parser/engine/py_json.h` |
| **ROAD-V1-C8 config families (`CLAIM-ROADMAP-C8-CONFIGS`, 2026-07-27):** `vllm/parser/minimax_m2.py:94` (`minimax_m2_config` + `_minimax_m2_arg_converter`), `vllm/parser/glm47_moe.py:74` (`glm47_moe_config` + `_glm47_arg_converter`), `vllm/parser/deepseek_v4.py:125` (`deepseek_v4_config` + `_dsml_arg_converter`), `vllm/parser/deepseek_v32.py:51` (`deepseek_v32_config`), `vllm/parser/nemotron_v3.py:34` (`nemotron_v3_config`) | `src/vllm/parser/engine/configs.cpp` (5 builders + 3 std::regex arg-converters), `src/vllm/parser/engine/registry.cpp` + `src/vllm/parser/parser_manager.cpp` (dispatch), and the `glm47_moe.py:198,203` name-`.strip()` overrides as `include/vllm/parser/glm47_moe.{h,cpp}` (existing `emit_name_delta`/`handle_tool_end` hooks) |

Types reused (not re-defined): `DeltaMessage`, `DeltaToolCall`,
`DeltaFunctionCall`, `FunctionCall`, `ToolCall`
(`include/vllm/entrypoints/openai/protocol.h`) and
`ExtractedToolCallInformation`
(`include/vllm/entrypoints/openai/tool_parsers/abstract.h`).

## Gate (CPU, EXACT, RED-first)

`tests/vllm/parser/engine/test_parser_engine_assembly.cpp` (+ generated
`..._goldens.inc` by `tools/parity/dump_parser_engine_assembly.py`). A parser is
a pure function of the `(delta_text, delta_token_ids)` stream, so it gates
EXACTLY: our per-delta `DeltaMessage` sequence AND the one-shot
`extract_tool_calls` result equal, field-for-field, the vLLM 0.26 assembly's for
the identical stream.

- **2 cases / 3510 assertions, 19 scenarios:** scenarios 1-9 as before (qwen3
  reasoning+XML tool call whole-delta AND char-by-char, reasoning-suppressed
  `include_reasoning=false`, thinking-off plain content, two consecutive tool
  calls `tool_index` 0→1, unfinished call flushed by `finish()`, seed_oss
  wrapper-token variant, kimi_k2 JSON args held-back top-level brace char-by-char
  AND whole-delta); **scenarios 10-19 (ROAD-V1-C8 config families, 1858 added
  assertions), each whole-delta AND char-by-char:** minimax_m2 (`<invoke>` /
  `<parameter>` XML, held-back JSON arg diffs), glm47_moe (`<arg_key>` /
  `<arg_value>`, function-name `.strip()`), deepseek_v4 (`<think>` + DSML
  tool_calls, `string="true/false"` typed-value coercion — `days` streamed as int
  `5`), deepseek_v32 (DSML `function_calls`, no reasoning, `count` int `3`),
  nemotron_v3 (qwen3 grammar + `strip_trailing_reasoning`).
- **RED-first (engine core, assembly):** dropping the held-back tool-args tail in
  `_safe_arg_prefix` fails 32 assertions; first divergent boundary
  `qwen3_reasoning_xml_wholedelta delta[3] tc[0] tc args` (expected held-back
  `{"city": "Tokyo`, break emitted `{"city": `). Restoring → GREEN.
- **RED-first (ROAD-V1-C8 configs):** disabling the glm47_moe name-`.strip()`
  fails exactly 2 assertions at boundary
  `glm47_reasoning_xml_wholedelta delta[2] tc[0] tc name` (and the char-by-char
  twin `delta[54]`): expected `get_weather`, break emitted `get_weather\n`.
  Restoring → 3510/3510.
- Goldens reproduce byte-identically from the pinned oracle.
- **Deterministic ids:** the qwen3 "random" id_type normally yields a random
  uuid; both the dump (monkeypatched `make_tool_call_id`) and the gate (injected
  `set_id_factory`) use `chatcmpl-tool-<idx>` so ids compare exactly. Production
  keeps the random factory (default).

## Inertness

Additive opt-in subsystem. `git diff --stat` for code = new files
(`parser_engine.{h,cpp}`, `kimi_k2.{h,cpp}`, `parser_manager.{h,cpp}`,
`py_json.h`; ROAD-V1-C8: `glm47_moe.{h,cpp}`) + config builders/converters
appended to `configs.{h,cpp}` + registry/manager dispatch lines (the streaming
engine ignores the assembly fields: engine-core gate still 586/586; serving-SSE
still 210/210) + 1 lib src line in CMake. The engine CORE, assembly, and serving
logic are UNCHANGED — only new configs/adapters + registration were added. No
existing tool_parsers / serving TU modified. Plain generation (no tool parser)
byte-identical.

## RESIDUALS (honest, stay open under C8)

- **Serving-SSE dispatch swap** (wire the assembled parser behind
  `--tool-call-parser` in the OpenAI serving path) — **DONE + CPU-GATED 2026-07-27
  (`CLAIM-ROADMAP-C8-SERVING`):** the chat streaming path routes engine-backed
  names through `parser_manager get_parser_engine` (`ShapeChatDeltaEngine` ->
  `parse_delta`; `ShapeChatMessageEngine` -> `parse`), EXACT chunk-for-chunk vs
  vLLM 0.26 `chat_completion_stream_generator` (`test_openai_serving_chat_stream`
  210/210, RED-first 6 CHECKs; goldens `tools/parity/dump_serving_chat_stream.py`).
  OFF by default (name-selected), legacy `tool_parsers` path byte-identical.
- **JSON-schema type coercion** (`_fix_arg_types` with a non-empty tool schema,
  `_streamable_string_keys` from properties, `find_tool_properties`) — modeled as
  identity because no tool schema is carried; only triggers when tools with
  schemas are supplied. `validate_tool_names` path is ported (gate configs have
  it false).
- **The other engine configs — 5 PORTED, 2 DEFERRED (`CLAIM-ROADMAP-C8-CONFIGS`,
  2026-07-27):**
  - **PORTED (exact-gated, scenarios 10-19):** `minimax_m2`, `glm47_moe`,
    `deepseek_v4`, `deepseek_v32`, `nemotron_v3`. Each maps onto the literal
    `LexerShape` engine; the regex arg-converters mirror qwen3's `std::regex`
    approach (`minimax`/`glm47`/`dsml`), overlapping literals (`">"` vs `<invoke
    name="`, DSML fullwidth `｜`) resolve via the lexer's longest-match +
    prefix-wait. glm47 uses a `Glm47MoeParser` subclass over the EXISTING
    `emit_name_delta`/`handle_tool_end` hooks (name `.strip()`). deepseek's
    subclass `_convert_args`→`_unwrap_wrapper_args` (schema wrapper-key unwrap)
    and `_dsml` typed values fall under the no-tool-schema model: with `tools`
    empty it degenerates to the config converter EXACTLY (same identity bucket as
    `_fix_arg_types` below), which the gate exercises. nemotron's empty-content
    reasoning/content SWAP (`_should_force_content`) needs
    `request.chat_template_kwargs` (unmodeled in `ParserRequest`) and fires only
    on `enable_thinking=False`/`force_nonempty_content` — a serving-edge residual,
    not exercised by the normal streaming/extract gate; the config maps cleanly.
  - **DEFERRED — 2, each needs a specific UNPORTED assembly hook (do NOT stub):**
    - **`gemma4`** — needs a per-parser `_events_to_delta` reasoning-REWRITE hook
      (`gemma4.py:530` strips the `thought\n` channel prefix from streamed
      reasoning; the `<|channel>thought\n…` header is intrinsic to the format, so
      every realistic stream hits it) AND `_preprocess_feed` first-feed
      `<|channel>` injection (`gemma4.py:424`). The current `ParserEngine` has no
      per-parser `_events_to_delta`/`_preprocess_feed` override seam; adding it
      modifies the assembly core (out of scope). The custom `_parse_gemma4_args`
      scanner itself is portable — only the delta-rewrite/feed hooks block it.
    - **`inkling`** — needs a virtual `_extract_args_value` hook to unwrap the
      Inkling `args` wrapper key in the non-streaming name-from-args path
      (`inkling.py:402`; the base `extract_name_and_args` inlines a fixed
      `{"arguments","parameters"}` key list, so `extract_tool_calls` would keep
      the `{"name":…,"args":{…}}` wrapper) AND the `_single_pass_parse`
      trailing-text flush (`inkling.py:376`, text blocks after a tool block). Both
      require editing the private non-virtual assembly methods. The
      `_inkling_arg_converter` JSON-span scanner itself is portable.
- **Prompt-state hooks** (`adjust_initial_state_from_prompt`) — base no-op ported;
  family token-ID `is_reasoning_end` overrides not needed by the assembly gate.
- **Live-engine metric wiring, `SERVE-RESPONSE-METRICS`, chat-form `/tokenize`**
  (tracked elsewhere under C8).
