# ROAD-V1-C8 — parser ASSEMBLY layer (`TOOLS-STREAMING-PARSER` assembly)

Status: **LANDED + CPU-gated 2026-07-27** (`CLAIM-ROADMAP-C8-ASSEMBLY`, not
pushed). Builds directly on the merged engine CORE
([streaming-parser-engine.md](streaming-parser-engine.md)); do NOT re-port the
engine core.

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

- **2 cases / 1652 assertions, 9 scenarios:** qwen3 reasoning+XML tool call
  (whole-delta AND char-by-char), reasoning-suppressed
  (`include_reasoning=false`), thinking-off plain content, two consecutive tool
  calls (`tool_index` increments 0→1), an unfinished call flushed by `finish()`,
  seed_oss wrapper-token variant, kimi_k2 JSON args with held-back top-level
  brace (char-by-char AND whole-delta).
- **RED-first:** dropping the held-back tool-args tail in `_safe_arg_prefix`
  fails 32 assertions; first divergent boundary
  `qwen3_reasoning_xml_wholedelta delta[3] tc[0] tc args` (expected held-back
  `{"city": "Tokyo`, break emitted `{"city": `). Restoring → 1652/1652.
- Goldens reproduce byte-identically from the pinned oracle.
- **Deterministic ids:** the qwen3 "random" id_type normally yields a random
  uuid; both the dump (monkeypatched `make_tool_call_id`) and the gate (injected
  `set_id_factory`) use `chatcmpl-tool-<idx>` so ids compare exactly. Production
  keeps the random factory (default).

## Inertness

Additive opt-in subsystem. `git diff --stat` for code = new files
(`parser_engine.{h,cpp}`, `kimi_k2.{h,cpp}`, `parser_manager.{h,cpp}`,
`py_json.h`) + assembly-field additions to `parser_engine_config.h` /
`configs.cpp` (the streaming engine ignores them: engine-core gate still
586/586) + 3 lib src lines + 1 test line in CMake. No existing tool_parsers /
serving TU modified. Plain generation (no tool parser) byte-identical.

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
- **The other 6 engine configs** (gemma4, deepseek_v4/v32, minimax_m2,
  nemotron_v3, glm47_moe, inkling) — configs + any family overrides.
- **Prompt-state hooks** (`adjust_initial_state_from_prompt`) — base no-op ported;
  family token-ID `is_reasoning_end` overrides not needed by the assembly gate.
- **Live-engine metric wiring, `SERVE-RESPONSE-METRICS`, chat-form `/tokenize`**
  (tracked elsewhere under C8).
