# Unified streaming parser engine (`TOOLS-STREAMING-PARSER`, ROAD-V1-C8)

Live spec for the `TOOLS-STREAMING-PARSER` engine-matrix row: vLLM 0.26's
declarative, config-driven streaming parser engine that turns a streamed
completion's `(delta_text, delta_token_ids)` pairs into an incremental stream of
reasoning/tool-call semantic events. Grounded 1:1 in vLLM 0.26.0.dev0
(`555967922`).

## Why this row exists (0.25 -> 0.26 shift)

Through 0.25 every model's tool-call / reasoning format was a hand-rolled
parser under `vllm/entrypoints/openai/tool_parsers/` with its own
`extract_tool_calls_streaming`. This project already ports that legacy surface
(40 tool-parser families + 9 reasoning parsers, each individually gated). vLLM
0.26 introduced a NEW unified abstraction, `vllm/parser/engine/`, in which each
format is a declarative `ParserEngineConfig` (terminals + a transition state
machine) and ONE shared `StreamingParserEngine` handles streaming, ambiguity
buffering, token-ID mapping, and delta computation. The legacy parsers still
exist at `vllm/tool_parsers/`; the engine-backed families
(qwen3, seed_oss, kimi_k2, gemma4, deepseek_v4/v32, minimax_m2, nemotron_v3,
glm47_moe, inkling) route through the new engine. This row ports the engine.

## Upstream chain (cite file:line @ 555967922)

- `vllm/parser/engine/events.py:11,22` — `EventType` (8 variants) + `SemanticEvent`.
- `vllm/parser/engine/parser_engine_config.py:28,38,45` — `ParserState`,
  `Transition`, `ParserEngineConfig` (terminals, token_id_terminals,
  transitions, content_events, initial_state, tool_args_json, preserve_tokens,
  arg_converter, and the assembly-layer flags).
- `vllm/parser/engine/token_id_scanner.py:29` — `TokenIDScanner`: maps special
  token IDs in the delta to `PreLexedTerminal`s, defers terminals whose text has
  not yet arrived, and recovers detokenizer hold-back text
  (`scan`/`flush_pending`/`_resolve_deferred`/`_recover_holdback_text`/
  `_rebuild_from_anchors`).
- `vllm/parser/engine/incremental_lexer.py:29,80,214` — `LexerShape`,
  `IncrementalLexer` (prefix-match buffering), `terminals_from_literals`.
- `vllm/parser/engine/streaming_parser_engine.py:89` — `StreamingParserEngine`:
  the pipeline `scan -> lex -> state machine -> events`, plus `_build_drop_info`
  (`:39`), the JSON argument streamer `_feed_args_char`/`_flush_safe_args`
  (`:422,459`, holds back top-level closing braces), and `finish` (`:237`).
- `vllm/parser/engine/parser_engine.py:79` — the higher `ParserEngine` assembly
  layer that turns `SemanticEvent`s into `DeltaMessage` /
  `ExtractedToolCallInformation` (non-streaming + streaming DeltaToolCall).
- `vllm/parser/engine/adapters.py:35,128` + `registered_adapters.py` — the
  reasoning/tool adapter split and the manager registration binding a format
  name to its config.
- Configs ported: `vllm/parser/qwen3.py:88` (`qwen3_config`, XML args,
  `tool_args_json=False`), `vllm/parser/seed_oss.py:23` (Qwen3 grammar, different
  wrapper tokens), `vllm/parser/kimi_k2.py:52` (`kimi_k2_config`, JSON args,
  `tool_args_json=True`).
- Tests (executable spec): `tests/parser/engine/test_engine.py`,
  `test_parser_engine.py`, `test_token_id_scanner.py`, `test_qwen3.py`,
  `test_qwen3_reasoning.py`, `test_replay.py`, `test_delegating_replay.py`.

## Our baseline

Before this change the engine did not exist in C++: `include/vllm/parser/` and
`src/vllm/parser/` were absent, and all tool/reasoning streaming ran through the
40 hand-rolled per-family parsers under
`include|src/vllm/entrypoints/openai/tool_parsers/` +
`reasoning_parsers/` (0.25-style, each with its own
`extract_tool_calls_streaming`, individually gated). The `TOOLS-STREAMING-PARSER`
row was `INVENTORIED` with no code and no spec.

## Port map (this change — ACTIVE)

The CORE streaming engine as an additive, self-contained C++ subsystem under
`include/vllm/parser/engine/` + `src/vllm/parser/engine/`:

- `events.h` — `EventType` + `SemanticEvent` (value semantics + `==`).
- `parser_engine_config.h` — `ParserState`, `Transition`, `ParserEngineConfig`
  (only the streaming-relevant fields; see scope note below).
- `incremental_lexer.{h,cpp}` — literal-terminal `LexerShape` + `IncrementalLexer`
  with prefix-match buffering, byte-wise iteration equivalent to upstream's
  char-wise iteration for ASCII terminals.
- `token_id_scanner.{h,cpp}` — full `TokenIDScanner` incl. deferral +
  hold-back / anchor-rebuild recovery, over an abstract `EngineTokenizer` seam.
- `streaming_parser_engine.{h,cpp}` — `StreamingParserEngine`
  (`feed`/`finish`/`parse_complete`), the state machine, JSON argument streaming
  with brace hold-back, and `_build_drop_info`.
- `configs.{h,cpp}` — `qwen3_config` (reused by seed_oss) + `kimi_k2_config`.
- `registry.{h,cpp}` — the unified name -> config dispatch
  (`get_engine_config`/`is_engine_backed`/`engine_backed_names`), the C++ analogue
  of `registered_adapters.py` + the manager name registration.

## Gates (exact, CPU, RED-first)

`tests/vllm/parser/engine/test_streaming_parser_engine.cpp` +
`..._goldens.inc`. The engine is a pure function of the delta stream, so it is
gated EXACTLY: for 8 fixed deterministic delta streams the C++ engine's emitted
`SemanticEvent` sequence must equal, event-for-event (type, value, tool_index),
the sequence the vLLM 0.26 Python engine emits for the identical stream. The
goldens are captured directly from upstream by
`tools/parity/dump_streaming_parser_engine.py` (copies the six pinned engine
modules, rebuilds the two configs inline, drives the identical streams) and
reproduce the committed `.inc` byte-for-byte. Coverage: reasoning->content->XML
tool call (qwen3); char-by-char vs whole-string cadence; thinking-disabled plain
content; two consecutive tool calls (tool_index increment); an unfinished tool
call flushed by `finish()`; JSON argument streaming with the held-back top-level
brace (kimi_k2); a token-ID-driven stream through the `TokenIDScanner` + a mock
vocab. RED-first proof: dropping the held-back argument delta on tool-args exit
fails 12 assertions at the exact divergent `}` boundary; restoring it returns
586/586 green. Inertness: additive subsystem, opt-in — nothing dispatches to it
unless a request selects an engine-backed parser; plain generation is byte-
identical (`git diff --stat` touches only new files + explicit wiring lines).

## Scope note / residual (honest)

The `ParserEngineConfig` struct carries only the fields the `StreamingParserEngine`
reads. The remaining honest residual (stays open under this row):

1. The higher `parser_engine.py` ASSEMBLY layer (SemanticEvent ->
   `DeltaMessage` / `ExtractedToolCallInformation`, tool-call id/name/arguments
   delta objects, `arg_converter` application, `validate_tool_names`, the
   whitespace/strip flags) and its replay adapters.
2. The serving-SSE DISPATCH swap: routing engine-backed families in
   `serving_chat` through the engine instead of the legacy per-family parser.
   The legacy parsers remain the live path; the engine ships gated but not yet
   wired into the response pipeline.
3. The remaining engine configs: gemma4, deepseek_v4/v32, minimax_m2,
   nemotron_v3, glm47_moe, inkling (each is a declarative config + optional
   subclass hooks; no fixture faked).
4. Non-literal (regex) terminals in the lexer (no 0.26 engine config uses them).

## Dependencies

- No new third-party dependency: the core is standard C++ (`<map>`/`<set>`/
  `<string>`/`<vector>`/`<optional>`) plus the existing doctest harness. The
  legacy `nlohmann/json` and per-family parsers are untouched.
- Regenerating the goldens needs Python 3 with the `regex` package and a checkout
  of the pinned vLLM source (`VLLM_SOURCE`); no GPU, no model, no vLLM wheel.
- Row dependency: the serving-SSE dispatch swap (residual #2) depends on the
  assembly layer (residual #1) and on `TOOLS-CALLING-CORE` / the chat-completion
  serving seam already in the tree.

## Risks/decisions

- DECISION: gate the pure streaming engine (SemanticEvent stream) EXACTLY rather
  than approximate the higher assembly-layer DeltaMessage output. A parser is a
  pure function of the token stream, so the event stream is the tightest,
  deterministic, CPU-only oracle; the assembly layer is a thin, separately
  gateable transform (residual #1).
- DECISION: literal-terminal lexer only. Every 0.26 engine config uses
  `terminals_from_literals`; regex terminals are unused, so porting the regex
  path now would be dead code (residual #4).
- RISK: byte-wise vs upstream char-wise iteration in the lexer/scanner. Mitigated
  because every terminal first-char is ASCII (< 0x80); a UTF-8 lead/continuation
  byte can never match a terminal first-char and content boundaries always land
  on char boundaries. Documented in `incremental_lexer.h`.
- RISK: the drop-info / special-token machinery is ported but only lightly
  exercised (the gate's mock tokenizer exposes no special-token table, mirroring
  the Python `MockTok`). A dedicated drop gate with a full tokenizer ships with
  the assembly layer.
- DECISION: ship the engine additive and INERT (registry present, no serving
  dispatch swap) so plain generation stays byte-identical and the landed
  per-family parsers remain the live path until the assembly layer + swap land.

## Tests to port (tracked)

`tests/parser/engine/test_engine.py`, `test_parser_engine.py`,
`test_token_id_scanner.py`, `test_qwen3*.py`, `test_replay.py` — the core
streaming cases are represented by the 8 exact-gated scenarios here; the
assembly-layer and replay cases port with residual #1/#2.

## Work breakdown (row-sized)

- W0 core engine (scanner + lexer + state machine + JSON args) + qwen3/kimi
  configs + registry + exact gate. **DONE (this change, ACTIVE).**
- W1 remaining engine configs (residual #3).
- W2 assembly layer + serving-SSE dispatch swap (residual #1/#2) -> DONE.
