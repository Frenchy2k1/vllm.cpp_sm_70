# Utility endpoints (`SERVE-UTILITY-ENDPOINTS`, ROAD-V1-C8)

Live spec for the `SERVE-UTILITY-ENDPOINTS` engine-matrix row: `/tokenize`,
`/detokenize`, `/ping`, `/server_info` and `/reset_prefix_cache`, mirroring vLLM
0.26.0.dev0 (`555967922`).

## Scope

Add the OpenAI-server utility endpoints as ADDITIVE, opt-in routes on the
`ApiServer`: `/tokenize` + `/detokenize` (prompt/tokens forms), `/ping`
(liveness), `/server_info`, and `/reset_prefix_cache`. Each matches the vLLM 0.26
request/response schema on a fixed CPU-deterministic case. Chat-form `/tokenize`
(messages + chat template) is a bounded deferral (needs the renderer path) and is
rejected loudly rather than mis-answered.

## Upstream chain

- `vllm/entrypoints/serve/tokenize/api_router.py:37,63` — `/tokenize`,
  `/detokenize`; `serve/tokenize/protocol.py:23,158,166,181` —
  `TokenizeCompletionRequest{prompt, add_special_tokens=true,
  return_token_strs=false}` → `TokenizeResponse{count, max_model_len, tokens,
  token_strs?}`; `DetokenizeRequest{tokens[]}` → `DetokenizeResponse{prompt}`.
- `vllm/entrypoints/serve/tokenize/serving.py:57,101-123,126-152` — encode with
  `add_special_tokens`, `convert_ids_to_tokens`, decode.
- `vllm/entrypoints/serve/sagemaker/api_router.py:47-50` — GET/POST `/ping` (empty
  200 liveness, mirrors `/health`).
- `vllm/entrypoints/serve/dev/server_info/api_router.py:43` — `/server_info`
  `{vllm_config, vllm_env, system_env}`.
- `vllm/entrypoints/serve/dev/cache/api_router.py:20` — POST `/reset_prefix_cache`
  with `reset_running_requests`/`reset_external` query params → `{"success":
  bool}`.
- `vllm/entrypoints/serve/tokenize/api_router.py:95-108` — `attach_router` only
  registers `GET /tokenizer_info` when
  `app.state.args.enable_tokenizer_info_endpoint`; `serve/tokenize/serving.py:154-195`
  — `get_tokenizer_info` → `TokenizerInfo(tokenizer, chat_template).to_dict()`
  (`_get_tokenizer_config` = the HF `tokenizer.init_kwargs` minus
  `vocab_file`/`merges_file`, plus `tokenizer_class` + optional `chat_template`);
  `serve/tokenize/protocol.py:185` — `TokenizerInfoResponse(ConfigDict
  extra="allow", tokenizer_class: str)`.

## `/tokenizer_info` (`CLAIM-C8-SERVE-ENDPOINTS`, 2026-07-28)

`GET /tokenizer_info` is gated behind `set_tokenizer_info_enabled(true)` mirroring
vLLM's `enable_tokenizer_info_endpoint` CLI flag: OFF by default, so the route is
not registered (→ 404) unless the flag is on AND a tokenizer is attached. vLLM
emits the whole `tokenizer_config.json`-equivalent `init_kwargs` dict verbatim
(`extra="allow"`) plus the required `tokenizer_class`. We surface EXACTLY the
fields our byte-level / SentencePiece BPE tokenizer can GENUINELY back and OMIT
(never fabricate) the ones it does not carry:

- `tokenizer_class` (REQUIRED): the genuine BPE family — `ByteLevelBPETokenizer`
  or `SentencePieceBPETokenizer` (the HF `tokenizers` library's own class names),
  from `Tokenizer::GetFamily()`.
- `model_max_length`: the attached `max_model_len` (omitted when ≤0).
- `vocab_size`: `Tokenizer::VocabSize()`.
- `bos_token_id` / `eos_token_id`: `Tokenizer::BosId()`/`EosId()`, omitted when -1.
- `added_tokens_decoder`: id → `{content, special, lstrip, rstrip}` for each
  `Tokenizer::AddedTokens()` entry (the HF `tokenizer_config.json` map shape).

NAMED GAPS (vLLM emits, our tokenizer cannot back, so OMITTED — recorded honestly):

- The raw `chat_template` string — our chat template lives in the `ChatPromptFn`
  render seam (`chat_.prompt_fn()`), not on the `Tokenizer` object, so we cannot
  echo the source template here.
- The HF `init_kwargs` general contents (`clean_up_tokenization_spaces`,
  `add_bos_token`, `model_input_names`, padding/truncation defaults, …) — not
  parsed by our loader.
- The added-token `normalized` / `single_word` flags — not stored on
  `vllm::tok::SpecialToken` (which carries only content/id/special/lstrip/rstrip).

## Our baseline

Before this row: `ApiServer` exposed only `/v1/completions`, `/v1/chat/
completions`, `/v1/models`, `/health`, `/version`. The tokenizer already provides
`Encode`/`EncodeWithSpecialTokens`/`Decode`/`TokenText`
(`include/vllm/tokenizer/tokenizer.h`); `reset_prefix_cache()` exists on
`KVCacheManager`/`BlockPool`.

## Port map

- `src/vllm/entrypoints/openai/api_server.cpp` + `.h` — `handle_tokenize`,
  `handle_detokenize`, `handle_ping`, `handle_server_info`,
  `handle_reset_prefix_cache`; opt-in `set_tokenizer(tokenizer, max_model_len)`
  and `set_reset_prefix_cache(callback)`; conditional route registration
  (`/tokenize`+`/detokenize` when a tokenizer is attached, `/reset_prefix_cache`
  when a callback is attached, `/ping`+`/server_info` always).
- Encoding: `add_special_tokens` → `EncodeWithSpecialTokens`, else `Encode`;
  `return_token_strs` → per-id `TokenText`. Detokenize → `Decode`.

## Tests to port

Upstream `tests/entrypoints/openai/test_tokenization.py` (tokenize/detokenize
round-trip + schema) re-expressed in
`tests/vllm/entrypoints/openai/test_api_server.cpp`: `/tokenize` schema
(`count`/`max_model_len`/`tokens`/`token_strs`), `/detokenize` round-trip,
`token_strs` null default, 400 errors (missing `prompt`, chat-form `messages`,
missing `tokens`), `/ping` 200, `/server_info` three-key shape,
`/reset_prefix_cache` `{"success": bool}` with the callback invoked.

## Gates

- Parity: each endpoint returns the vLLM-0.26 schema on the fixed BPE-fixture
  case (CPU). RED-first where feasible (a missing `prompt` / wrong token boundary
  fails).
- Inertness: opt-in — a server without the backings does not register
  `/tokenize`/`/detokenize`/`/reset_prefix_cache`; existing serving byte-identical
  (22 pre-existing api_server cases green). Clean CPU `-Werror`.

## Dependencies

The tokenizer (`SERVE`-side, already present) and the prefix-cache reset
(`KV-PREFIX-CACHE`, `D4-APC` sibling — consumed only via an injected callback, no
KV-block-hash code touched).

## Work breakdown

- W1: `/tokenize` + `/detokenize` (prompt/tokens forms) + tests — DONE.
- W2: `/ping` + `/server_info` + `/reset_prefix_cache` + tests — DONE.
- W3: chat-form `/tokenize` (messages + chat template + tools) — DONE
  (`CLAIM-C8-CHAT-TOKENIZE`); `/tokenizer_info` — DONE (`CLAIM-C8-SERVE-ENDPOINTS`,
  2026-07-28, with the named-gap omissions above).
- W4 (residual): `/ready`, the full `server_info` config dump, and production
  `main.cpp` wiring of the whole opt-in C8 endpoint family (tokenize/detokenize/
  metrics/reset/tokenizer_info are handler+setter+route+test complete but NOT yet
  attached in the production server binary — the same deferred state for all of
  them) — OPEN.

## Risks/decisions

- Decision: chat-form `/tokenize` is rejected with 400 rather than partially
  implemented — the schema for the prompt form is exact, and the chat form needs
  the renderer + tool surface (bounded deferral, recorded honestly).
- Decision: `/reset_prefix_cache` takes an injected callback so the serving layer
  does not reach into KV internals owned by the `D4-APC` sibling.
- Risk: none to generation — all endpoints are read-only or cache-control and are
  opt-in additive routes.
