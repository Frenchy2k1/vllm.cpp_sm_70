# ROAD-V1-C8 — parser ASSEMBLY layer (`TOOLS-STREAMING-PARSER` assembly)

Status: **LANDED + CPU-gated 2026-07-27** (`CLAIM-ROADMAP-C8-ASSEMBLY`; config
families `CLAIM-ROADMAP-C8-CONFIGS`; last 2 families `CLAIM-ROADMAP-C8-CONFIGS-2`;
not pushed). Builds directly on the merged engine CORE
([streaming-parser-engine.md](streaming-parser-engine.md)); do NOT re-port the
engine core. **ALL 10 engine-backed families now ported — vLLM tool-parser family
parity CLOSED:** qwen3, seed_oss, kimi_k2, minimax_m2, glm47_moe, deepseek_v4,
deepseek_v32, nemotron_v3, **gemma4, inkling** (the last two ported 2026-07-27 by
adding the specific default-inert assembly-core virtual seams each needs — see
"gemma4 + inkling seams" below).

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
| **ROAD-V1-C8 last-2 families (`CLAIM-ROADMAP-C8-CONFIGS-2`, 2026-07-27):** `vllm/parser/gemma4.py:296` (`gemma4_config` + `_gemma4_arg_converter`/`_parse_gemma4_args`/`_parse_gemma4_array`) + `gemma4.py:387` `Gemma4Parser` (`_preprocess_feed`:424, `_events_to_delta`:530, `_reset`:418, `extract_reasoning`:572); `vllm/parser/inkling.py:175` (`inkling_config` + `_inkling_arg_converter`/`_args_value_span`/`_scan_json_value`) + `inkling.py:298` `InklingParser` (`_extract_args_value`:402, `_single_pass_parse`:376); base seams `parser_engine.py:210` `_preprocess_feed`, `:706` `_events_to_delta`, `:645` `_single_pass_parse`, `:1064` `_extract_args_value`, `:195` `_reset`, `:141-149` reasoning token-id resolve | `src/vllm/parser/engine/configs.cpp` (`gemma4_config`/`inkling_config` + 2 hand-rolled arg-converters), `include/vllm/parser/{gemma4,inkling}.{h,cpp}` + `src/vllm/parser/{gemma4,inkling}.cpp` (the two subclasses), the 4 additive virtual seams + `args_wrapper_keys`/`engine_state`/reasoning-token-id members on `parser_engine.{h,cpp}`, `registry.cpp` + `parser_manager.cpp` (dispatch) |

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

- **3 cases / 5038 assertions, 30 scenarios** (scenarios 27-30 =
  `CLAIM-C8-ARG-COERCION` JSON-schema arg-type coercion; see RESIDUALS below)**:**
  scenarios 1-9 as before (qwen3
  reasoning+XML tool call whole-delta AND char-by-char, reasoning-suppressed
  `include_reasoning=false`, thinking-off plain content, two consecutive tool
  calls `tool_index` 0→1, unfinished call flushed by `finish()`, seed_oss
  wrapper-token variant, kimi_k2 JSON args held-back top-level brace char-by-char
  AND whole-delta); **scenarios 10-19 (ROAD-V1-C8 config families), each
  whole-delta AND char-by-char:** minimax_m2 (`<invoke>` / `<parameter>` XML,
  held-back JSON arg diffs), glm47_moe (`<arg_key>` / `<arg_value>`, function-name
  `.strip()`), deepseek_v4 (`<think>` + DSML tool_calls, `string="true/false"`
  typed-value coercion — `days` streamed as int `5`), deepseek_v32 (DSML
  `function_calls`, no reasoning, `count` int `3`), nemotron_v3 (qwen3 grammar +
  `strip_trailing_reasoning`); **scenarios 20-26 (`CLAIM-ROADMAP-C8-CONFIGS-2`),
  each whole-delta AND char-by-char:** gemma4 explicit-`<|channel>` reasoning +
  `<|tool_call>` custom key:value args (intrinsic `thought\n` stripped), gemma4
  ELIDED-channel (first delta starts `thought\n`, `_preprocess_feed` injects the
  opener; whole-delta only), inkling thinking + `invoke_tool_json` + trailing text
  block (args unwrap + `_single_pass_parse` flush), inkling non-object `args`
  (converter-failure → `_extract_args_value` fallback). A THIRD test case gates
  the non-streaming `parse()` (reasoning/content/tool_calls) for the gemma4/inkling
  scenarios — the only path that reaches inkling's `_single_pass_parse` flush.
- **RED-first (engine core, assembly):** dropping the held-back tool-args tail in
  `_safe_arg_prefix` fails 32 assertions; first divergent boundary
  `qwen3_reasoning_xml_wholedelta delta[3] tc[0] tc args` (expected held-back
  `{"city": "Tokyo`, break emitted `{"city": `). Restoring → GREEN.
- **RED-first (ROAD-V1-C8 configs):** disabling the glm47_moe name-`.strip()`
  fails exactly 2 assertions at boundary
  `glm47_reasoning_xml_wholedelta delta[2] tc[0] tc name` (and the char-by-char
  twin `delta[54]`): expected `get_weather`, break emitted `get_weather\n`.
  Restoring → 3510/3510.
- **RED-first (`CLAIM-ROADMAP-C8-CONFIGS-2`, all 4 new seams):** gemma4
  `events_to_delta` skip → 13 asserts, first boundary
  `gemma4_channel_tool_wholedelta delta[0] reasoning` (expected `Let me check the
  weather.`, break kept `thought\nLet me check the weather.`); gemma4
  `preprocess_feed` identity → 5 asserts, ONLY `gemma4_elided_channel_wholedelta
  delta[0] content` (reasoning leaked as content — explicit-channel scenarios
  still pass, isolating the inject); inkling `args_wrapper_keys` drop-`args` → 4
  asserts, first boundary `inkling_nonobject_args_wholedelta extract tc[0]
  arguments` (expected `[1, 2, 3]`, break emitted the `{"args": [1, 2, 3]}`
  wrapper); inkling `single_pass_parse` skip-flush → 2 asserts, ONLY
  `inkling_think_tool_text_wholedelta parse content` (expected `Here you go.`,
  break emitted none). Each restored → 4526/4526.
- Goldens reproduce byte-identically from the pinned oracle.
- **Deterministic ids:** the qwen3 "random" id_type normally yields a random
  uuid; both the dump (monkeypatched `make_tool_call_id`) and the gate (injected
  `set_id_factory`) use `chatcmpl-tool-<idx>` so ids compare exactly. Production
  keeps the random factory (default).

## Inertness

Additive opt-in subsystem. `git diff --stat` for code = new files
(`parser_engine.{h,cpp}`, `kimi_k2.{h,cpp}`, `parser_manager.{h,cpp}`,
`py_json.h`; ROAD-V1-C8: `glm47_moe.{h,cpp}`; C8-2: `gemma4.{h,cpp}`,
`inkling.{h,cpp}`) + config builders/converters appended to `configs.{h,cpp}` +
registry/manager dispatch lines (the streaming engine ignores the assembly
fields: engine-core gate still 586/586; serving-SSE still 210/210) + lib src
lines in CMake. For C8-2 the ONLY assembly-core change is 4 ADDITIVE virtual
methods (`preprocess_feed`, `events_to_delta`, `single_pass_parse`,
`args_wrapper_keys`) + virtual `reset`/`extract_reasoning` + `engine_state()`
accessor + `_reasoning_{start,end}_token_id` members whose bases reproduce the
prior behavior EXACTLY — the 8 non-overriding families are byte-identical (the
existing 3510 stream/extract asserts, 586/586, 210/210 all UNCHANGED; the goldens
`.inc` diff is pure insertions). No existing tool_parsers / serving TU modified.
Plain generation (no tool parser) byte-identical.

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
  `_streamable_string_keys` from properties, `find_tool_properties`) — **DONE +
  CPU-GATED 2026-07-28 (`CLAIM-C8-ARG-COERCION`; not pushed):** `ParserTool` now
  carries the function `parameters` JSON-schema (threaded from `serving_chat.cpp`
  `ToParserRequest`); `_fix_arg_types` (parser_engine.py:365) + the recursive
  `_coerce_dict`/`_coerce_value` (parser_engine.py:227,269) + `_streamable_string_keys`
  (:348) + `find_tool_properties` (tool_parsers/utils.py:271) are ported in
  `parser_engine.{h,cpp}`, reusing the ALREADY-ported `extract_types_from_schema` /
  `coerce_to_schema_type` (`tool_parsers/utils.cpp`). A request whose tools declare
  typed params has its assembled `tool_calls[].function.arguments` coerced to the
  declared JSON types (int/number/bool/string/array/null; priority
  null>int>number>bool>object>array>string, uncoercible values left as-is, non-object
  args left as-is) in BOTH streaming (`parse_delta`, per-tick + held-back
  `_streamable_string_keys` string keys) and one-shot (`extract_tool_calls`/`parse`).
  With no schema / no tools the path is IDENTITY (byte-identical). Gate:
  `test_parser_engine_assembly` extended to 30 scenarios / 5038 asserts (27-30: qwen3
  typed-schema whole+char, qwen3 schema-mismatch + nullable, kimi_k2 JSON-native
  `"5"`->int in the converter-less extract path — streaming stays raw, extract
  coerces, divergence gated); RED-first 38 asserts, first boundary
  `qwen3_typed_schema_wholedelta extract tc[0] arguments` (identity
  `{"days": "5", …}` vs coerced `{"days": 5, "unit": "celsius", "active": true,
  "temp": 3.14, "tags": [1, 2, 3]}`); no-schema path byte-identical (the 26 prior
  scenarios + 586/586 engine-core + 210/210 serving unchanged; `.inc` diff = pure
  insertions). `validate_tool_names` path is ported (gate configs have it false).
- **The other engine configs — ALL 7 PORTED (`CLAIM-ROADMAP-C8-CONFIGS` +
  `-CONFIGS-2`, 2026-07-27); vLLM tool-parser family parity CLOSED:**
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
  - **PORTED (`CLAIM-ROADMAP-C8-CONFIGS-2`, exact-gated, scenarios 20-26) — the
    last 2, each by ADDING the specific assembly-core seam it needs as a
    DEFAULT-INERT virtual (the other 8 families byte-identical: 586/586 + the 3510
    existing stream/extract asserts + 210/210 UNCHANGED; `.inc` diff = pure
    insertions):**
    - **`gemma4`** (`Gemma4Parser`, `parser/gemma4.{h,cpp}` + `gemma4_config`) —
      virtual `preprocess_feed` (base identity, parser_engine.py:210) for the
      first-feed `<|channel>` injection (gemma4.py:424), virtual `events_to_delta`
      (parser_engine.py:706) for the `thought\n` channel-prefix REWRITE
      (gemma4.py:530), + virtual `reset` (:195) and `extract_reasoning` (:490,
      gemma4.py:572). The `_gemma4_arg_converter` key:value scanner lives in
      `gemma4_config`. `_reasoning_{start,end}_token_id` are resolved from the
      tokenizer vocab in the ctor (parser_engine.py:141) — the gate passes a
      gemma4 mock-tokenizer so the injection guard fires. Two RED-first scenarios:
      explicit-channel (`_events_to_delta` strip → 13 asserts) + ELIDED-channel
      (`_preprocess_feed` inject → 5 asserts, isolated).
    - **`inkling`** (`InklingParser`, `parser/inkling.{h,cpp}` + `inkling_config`)
      — `_extract_args_value` (inkling.py:402) refactored to a virtual
      `args_wrapper_keys()` (base returns the fixed `{"arguments","parameters"}`;
      inkling prepends `"args"`) + virtual `single_pass_parse`
      (parser_engine.py:645) for the trailing-text flush (inkling.py:376). The
      `_inkling_arg_converter` JSON-span carver lives in `inkling_config`. NOTE:
      for well-formed inkling the `args` unwrap happens in the CONVERTER (streaming
      + extract both), so `args_wrapper_keys` is only reached on a converter
      FAILURE (non-object `args`) — gated by the dedicated `inkling_nonobject_args_*`
      scenario (RED-first 4 asserts); the `_single_pass_parse` flush is reached
      ONLY by `parse()` (extract_tool_calls flushes via finish_streaming), gated by
      the NEW non-streaming parse() test case (RED-first 2 asserts).
- **Prompt-state hooks** (`adjust_initial_state_from_prompt`) — base no-op ported;
  family token-ID `is_reasoning_end` overrides not needed by the assembly gate.
- **Live-engine metric wiring, `SERVE-RESPONSE-METRICS`, chat-form `/tokenize`**
  (tracked elsewhere under C8).
