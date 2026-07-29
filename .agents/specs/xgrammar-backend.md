# xgrammar structured-output backend (`TOOLS-XGRAMMAR`)

The xgrammar backend is vLLM's **default** (`auto`) structured-output backend: it
compiles JSON-schema / regex / EBNF-grammar / choice / structural-tag to a
token-masking pushdown FSM and, each decode step, fills a per-sequence token
bitmask that the sampler uses to force grammar-valid tokens. This spike inventories
the whole upstream surface and lays out the portable 1:1 port. Pin `555967922`
(vLLM 0.26.0.dev0); xgrammar library pinned by content at
`mlc-ai/xgrammar a32ac892676d2eedc0327416105b9b06edfb94b2` (the SHA cited across
`backend_xgrammar.py`).

## Scope

- **Row:** `TOOLS-XGRAMMAR` (was `INVENTORIED`; W0 spike -> `SPIKE`, W1 brick ->
  `ACTIVE`). Adjacent rows unchanged: `TOOLS-STRUCTURED-CORE` (`PARTIAL`, the seam
  + native engine), `TOOLS-STRUCTURAL-TAG` (`PARTIAL`), `TOOLS-GUIDANCE-OUTLINES`
  (`INVENTORIED`, the other backends).
- **In:** a SECOND registerable `StructuredOutputBackend` (behind the existing
  `backend_types.h` seam) that mirrors xgrammar's grammar input modes
  (json_schema / json_object / regex / grammar-EBNF / choice / structural_tag),
  the engine backend-selection logic (`auto` -> `xgrammar` with fallbacks), and
  the token-bitmask contract already shared with the native backend.
- **Out (this row):** the guidance / outlines / lm-format-enforcer backends
  (`TOOLS-GUIDANCE-OUTLINES`); the full structural-tag surface
  (`TOOLS-STRUCTURAL-TAG`); reasoning-gated structured output (`SAMPLE-REASONING`
  W4); GPU bitmask-apply kernel breadth (already covered by
  `test_apply_grammar_bitmask` on CPU).
- **Modes & dispatch:** `StructuredOutputOptions` {JSON, JSON_OBJECT, REGEX,
  GRAMMAR, CHOICE, STRUCTURAL_TAG} (`backend_types.py:19-25`). The manager holds
  ONE engine-wide backend, built lazily on first grammar; per-request backend
  selection is NOT supported (`__init__.py:127-129`), matching our manager's
  single-factory design.

## Upstream chain

- **Backend + grammar:** `vllm/v1/structured_output/backend_xgrammar.py`
  - `XgrammarBackend.__post_init__:37-76` — builds `xgr.TokenizerInfo`
    (`from_huggingface`, or the Mistral/tekken special-case :42-59) and
    `xgr.GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=True,
    cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB*MB)`; reads
    `structured_outputs_config.disable_any_whitespace` (:38-40); pulls
    `num_speculative_tokens` for `max_rollback_tokens` (:72-76).
  - `compile_grammar:78-126` — the mode switch: JSON ->
    `compiler.compile_json_schema(spec, any_whitespace=not disable_any_whitespace)`;
    JSON_OBJECT -> `compile_json_schema('{"type": "object"}', ...)`; GRAMMAR ->
    `compile_grammar(spec)` (EBNF); REGEX ->
    `compile_regex_with_timeout(compiler.compile_regex, spec)`; STRUCTURAL_TAG ->
    `compile_structural_tag(...)` (deprecated `structures` list :98-108, or the
    new single-arg form :110). Returns an `XgrammarGrammar` wrapping
    `xgr.GrammarMatcher(ctx, max_rollback_tokens=num_speculative_tokens)`.
  - `allocate_token_bitmask:128-129` -> `xgr.allocate_token_bitmask(max_num_seqs,
    vocab_size)` (a `[max_num_seqs, ceil(vocab/32)]` int32 tensor).
  - `XgrammarGrammar:135-204` — the request-level FSM: `accept_tokens:152-171`
    (advance `matcher.accept_token` per token, then `matcher.is_terminated()`),
    `validate_tokens:173-188` (accept + rollback, no advance),
    `rollback:190-193`, `fill_bitmask:195-196`
    (`matcher.fill_next_token_bitmask(bitmask, idx)`), `is_terminated:198-199`,
    `reset:201-203`.
  - `has_xgrammar_unsupported_json_features:225-269` — the JSON-schema feature
    guard (multipleOf, uniqueItems/contains, non-standard string format,
    patternProperties/propertyNames) that drives the `auto` fallback.
  - `validate_xgrammar_grammar:272-364` — per-mode pre-validation (regex/choice/
    json/grammar/structural_tag); `choice` is rewritten to an EBNF grammar
    (`choice_as_grammar`), Lark grammars converted via `convert_lark_to_ebnf`.
- **Backend abstraction:** `backend_types.py` — `StructuredOutputOptions` (enum),
  `StructuredOutputGrammar` (accept/validate/rollback/fill/is_terminated/reset),
  `StructuredOutputBackend` (compile_grammar / allocate_token_bitmask / destroy).
- **Engine dispatch:** `vllm/v1/structured_output/__init__.py:114-175` —
  `grammar_init` builds the backend once off `sampling_params.structured_outputs.
  _backend` (`xgrammar` :133-138, then guidance/outlines/lm-format-enforcer),
  `grammar_bitmask:204-303` fills the batched mask.
- **Backend selection (`auto`):** `vllm/config/structured_outputs.py:13,21`
  (supported list + default `"auto"`); `vllm/sampling_params.py:932-949`
  (engine-level pin + the "request-level selection unsupported" guard),
  `:1002-1061` (the `auto` resolution: try `validate_xgrammar_grammar` ->
  `_backend="xgrammar"`; on `ValueError`, fall back to outlines/guidance;
  `_backend_was_auto=True`).
- **xgrammar library (grammar->bitmask algorithm):**
  `mlc-ai/xgrammar cpp/json_schema_converter.cc` (JSON schema -> EBNF; the
  `basic_*` rule set, `any_whitespace`, declaration key order),
  `cpp/grammar_matcher.cc` / `cpp/grammar.cc` (the pushdown automaton + token mask
  fill), `cpp/grammar_functor.cc` (grammar normalization). The bitmask layout is
  `[batch, ceil(vocab/32)]` int32, bit set == token allowed — IDENTICAL to our
  `TokenBitmask`.
- **Runtime-trace note:** structured output is a HOST-side masking path (no GPU
  kernel divergence beyond the already-ported `apply_grammar_bitmask` scatter);
  no nsys is owed. Correctness is the whole gate.

## Our baseline

The seam and a from-scratch, correctness-grade native engine already exist and
ARE xgrammar's algorithm (grammar -> byte-level pushdown FSM -> per-step token
bitmask via a token-byte trie):

- Seam: `include/vllm/v1/structured_output/backend_types.h` +
  `.cpp` (`StructuredOutputOptions`, both ABCs, `TokenBitmask`,
  `StructuredOutputKey`) — 1:1 with `backend_types.py`.
- Manager: `src/vllm/v1/structured_output/manager.cpp` — `grammar_init` /
  `grammar_bitmask`, ONE backend built via an injected `BackendFactory` (the
  factory IS the backend-selection decision).
- Native engine: `src/vllm/v1/structured_output/backend_native.cpp` — GBNF/EBNF
  parser, regex/choice lowering, JSON-schema->GBNF
  (`json_schema_to_gbnf.cpp`), the push-down FSM + token-byte trie (sub-O(vocab)
  `fill_bitmask`), lazy/trigger structural tags, the `forced_token` jump-forward
  hook.
- Production wiring: `src/vllm/entrypoints/model_loader.cpp:690-693` builds the
  manager with `MakeNativeBackendFactory`.
- Tests: `tests/vllm/v1/structured_output/` (native backend, json-schema->gbnf,
  apply-bitmask, response-format e2e, jump-forward).

**Honest gap (why this row exists):** the native JSON path parses into
`nlohmann::json`, which **sorts** object keys lexicographically and emits its own
whitespace/`basic_*` grammar. xgrammar preserves **declaration** key order and
emits a specific `any_whitespace` + `basic_*` grammar. So for schemas whose key
order or whitespace/exotic features differ, our native output diverges from what
vLLM's default backend produces. The xgrammar backend closes that.

## Port map (decision: mirror xgrammar's algorithm PORTABLY; do NOT vendor the C++ lib)

Recorded in porting-inventory §9. xgrammar IS "grammar -> pushdown automaton ->
per-step token bitmask", which the native engine already implements portably.
Vendoring the xgrammar C++ library (its own tokenizer-info, thread pool, cache,
CMake/deps) would duplicate that machinery and add a heavy dependency against the
no-extra-deps posture. So the xgrammar backend REUSES the native matcher and adds
only the xgrammar-FAITHFUL front-end where the two differ:

| Upstream | Local | Notes |
|---|---|---|
| `backend_xgrammar.py:36` `XgrammarBackend` | `include/.../backend_xgrammar.h` + `src/.../backend_xgrammar.cpp` (`XgrammarStructuredOutputBackend`) | composes a `NativeStructuredOutputBackend` for the matcher; owns `disable_any_whitespace` |
| `compile_json_schema` (xgrammar `json_schema_converter.cc`) | `include/.../xgrammar_json_schema.h` + `src/.../xgrammar_json_schema.cpp` | schema -> EBNF: **declaration key order** (`nlohmann::ordered_json`), `any_whitespace` `ws` rule, `basic_*` set VERBATIM |
| `compile_grammar/compile_regex/compile_structural_tag` | delegate to `NativeStructuredOutputBackend::compile_grammar` | xgrammar accepts the same EBNF/GBNF + regex + structural-tag surface the native paths implement |
| `XgrammarGrammar` (matcher) | reused `NativeGrammar` (FSM + trie) | accept/validate/rollback/fill/is_terminated/reset already 1:1 with `backend_types.py` |
| `allocate_token_bitmask` | reused `NativeStructuredOutputBackend::allocate_token_bitmask` | same `[max_num_seqs, ceil(vocab/32)]` int32 layout |
| `__init__.py:133-165` dispatch + `sampling_params.py:1031` `auto`->xgrammar | `ResolveStructuredOutputBackend` + `MakeStructuredOutputBackendFactory` (`backend_xgrammar.cpp`) | `auto`/`xgrammar` -> xgrammar; guidance/outlines/lm-format-enforcer -> native (named, until those rows land); unknown -> throw |

**Deviations (recorded):** the reused matcher is byte-level (mirrors xgrammar's
byte-based grammar); xgrammar's `basic_string_sub` lookahead `(=...)` optimization
is dropped (a performance-only anchor; the accepted language is identical). W1
supports required-only object properties, primitives, arrays, enum/const,
anyOf/oneOf; optional-property comma/permutation emission and the strict compact
(`disable_any_whitespace`) separators are W2 (see Work breakdown).

## Tests to port

| Upstream | Local | Tier | Status |
|---|---|---|---|
| `tests/entrypoints/llm/test_struct_output_generate.py:214` (json-schema constrain), key-order + whitespace behavior | `tests/vllm/v1/structured_output/test_backend_xgrammar.cpp` | doctest (model-free) | **W1 landed** — bitmask allows exactly the valid next tokens; declaration key order vs native sort; any_whitespace flag; json_object; converter EBNF; `auto`->xgrammar selection |
| `tests/v1/structured_output/test_utils.py` (has_xgrammar_unsupported_json_features) | `test_backend_xgrammar.cpp` (converter throws) | doctest | W2 — the full feature-guard table |
| `tests/v1/structured_output/test_validation.py` | (validation port) | doctest | W2 — per-mode `validate_xgrammar_grammar` |
| `tests/entrypoints/.../test_guided_*` (regex/choice/grammar/structural_tag) | delegate coverage via `test_backend_native.cpp` | doctest | present via native; xgrammar-specific regex compile is W3 |

## Gates

- **Correctness (W1, MET):** model-free doctest `test_backend_xgrammar` — given a
  schema + partial token sequence, the per-step bitmask allows EXACTLY the
  grammar-valid next tokens; RED-first witness = the native (key-sorting) backend
  admits a token the xgrammar (declaration-order) grammar forbids. CPU-only, runs
  everywhere.
- **Build:** clean CPU `-Werror` full-library build, 0 warnings.
- **Correctness (W2+):** the feature-guard + validation tables; optional-property
  schemas; e2e response_format through the manager on the CPU engine.
- **Parity (future, GPU):** against the pinned vLLM oracle with
  `structured_outputs.backend="xgrammar"` on a real model — token-identical
  constrained decode. DGX-gated (currently offline); not owed by the CPU brick.
- **No performance gate** is owed (host-side masking; the GPU scatter is the
  already-gated `apply_grammar_bitmask`).

## Dependencies

- `TOOLS-STRUCTURED-CORE` (the seam + manager + native matcher) — DONE/PARTIAL,
  reused directly.
- No new third-party dependency (the vendor-vs-port decision above).
- No hardware for the CPU brick; the GPU parity gate needs the DGX oracle (pin
  `555967922`), currently offline.

## Work breakdown

- **W0 (this spike):** the spec + records. `TOOLS-XGRAMMAR` -> `SPIKE`.
- **W1 (CPU brick, landed):** the `XgrammarStructuredOutputBackend` behind the
  seam; the xgrammar-faithful JSON-schema->EBNF converter (declaration key order,
  `any_whitespace`, `basic_*`); JSON + JSON_OBJECT compile->bitmask; the
  `auto`->xgrammar selection + factory; the RED-first unit gate. Row -> `ACTIVE`.
- **W2:** optional object properties (xgrammar comma/permutation emission); the
  strict compact `disable_any_whitespace` separators; the
  `has_xgrammar_unsupported_json_features` guard + `validate_xgrammar_grammar`
  ports feeding the `auto` fallback; production wiring in `model_loader.cpp`
  (thread the configured backend name through `MakeStructuredOutputBackendFactory`).
- **W3:** xgrammar-specific regex compile semantics (vs the native regex->GBNF
  lowering) where they diverge; structural-tag parity against
  `TOOLS-STRUCTURAL-TAG`.
- **W4 (GPU):** the oracle parity gate on a real model when the DGX is back.

## Risks / decisions

- **Vendor vs port (product-adjacent, decided):** mirror xgrammar's algorithm
  portably, reuse the native matcher, port only the divergent front-end. No new
  dependency; recorded in porting-inventory §9. This is the same multi-backend
  design vLLM uses (xgrammar is one of several behind the seam).
- **Throw rather than mis-constrain:** the converter throws on any construct
  outside its supported subset, so an emitted grammar never accepts
  schema-invalid JSON (matches the native path's philosophy). The `auto` fallback
  (W2) is what turns an unsupported schema into a different backend upstream.
- **No behavior reopened:** key order, whitespace, and the `basic_*` grammar are
  taken 1:1 from xgrammar; we do not invent an alternative.
