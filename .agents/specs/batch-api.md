# SERVE-BATCH-API — the offline OpenAI Batch API runner (spike)

Row: `SERVE-BATCH-API` (engine-matrix, *Serving surface, CLI, and library*).
Claim: `CLAIM-BATCH-API`. Upstream: vLLM `555967922` (0.26.0.dev0)
`vllm/entrypoints/openai/run_batch.py`. Upstream test:
`tests/entrypoints/openai/test_run_batch.py`.

## Scope

vLLM's offline Batch API: a standalone runner (`vllm run-batch -i in.jsonl -o
out.jsonl --model M`) that reads a JSONL file of OpenAI-format requests, runs
each through the engine's serving handlers, and writes a JSONL of responses —
the OpenAI Batch API *offline* shape (the file-based bulk job, not the hosted
`/v1/batches` REST lifecycle). It is an **orchestrator** over the existing
serving handlers (`OpenAIServingChat`, `serving_embedding`, `serving_scores`,
transcription/translation), NOT a new generation path: it drives the SAME
`create_chat_completion` the online `/v1/chat/completions` route drives
(run_batch.py:722-731 maps the url to `openai_serving_chat.create_chat_completion`).

Named a RECORDS-GAP by the feature-gap analysis
([vllm-feature-gap-analysis.md](vllm-feature-gap-analysis.md) line 85,
recommending `SERVE-BATCH-API`); this spike creates the row.

## Upstream chain

- `vllm/entrypoints/openai/run_batch.py` @ `555967922` — the whole runner:
  schema (`:148-228`), `make_error_request_output` (`:508-520`), `run_request`
  (`:529-570`), `handle_endpoint_request` (`:576-610`), `build_endpoint_registry`
  (`:687-777`), `run_batch` (`:793-847`), file I/O (`:334-505`), CLI
  (`:232-297,850-877`).
- The serving handlers it drives are UNCHANGED deps: `OpenAIServingChat`
  (`chat_completion/serving.py`), `serving_embedding`, `serving_scores`,
  transcription/translation — resolved through `init_app_state`. No kernel /
  dependency-chain code is involved (this is a pure orchestration layer).
- Upstream test `tests/entrypoints/openai/test_run_batch.py`.

## Our baseline

- `OpenAIServingChat::create_chat_completion`
  (`include/vllm/entrypoints/openai/serving_chat.h:208`,
  `src/…/serving_chat.cpp:231`) — the SAME handler
  `ApiServer::handle_chat_completions` (`src/…/api_server.cpp:169`) drives; the
  batch runner reuses it verbatim (no reimplemented generation).
- The protocol structs + (de)serializers already exist
  (`include/vllm/entrypoints/openai/protocol.h`,
  `src/…/protocol.cpp`: `from_json(ChatCompletionRequest)`,
  `to_json(ChatCompletionResponse)`, `to_json(ErrorResponse)`).
- `OpenAIServingModels::check_model`
  (`include/vllm/entrypoints/openai/serving_models.h:58`).
- The uuid idiom mirrors `make_tool_call_id`
  (`src/…/tool_parsers/abstract.cpp:58-68`).
- The synthetic-engine serving-test harness
  (`tests/vllm/entrypoints/openai/test_serving.cpp::Harness`) is the W1 gate
  vehicle.

## JSONL schema (run_batch.py:148-228)

`BatchRequestInput` — one per input line:
```
{ "custom_id": str,        # developer id echoed to the matching output row
  "method": "POST",        # only POST
  "url": str,              # "/v1/chat/completions" | "/v1/embeddings" | .../score | .../rerank | /v1/audio/{transcriptions,translations}
  "body": {…} }            # the endpoint's request object (disambiguated BY url, :170-187)
```
`BatchResponseData` (run_batch.py:202-210):
```
{ "status_code": int=200, "request_id": str, "body": AllResponse | null }
```
`BatchRequestOutput` — one per output line (run_batch.py:213-228):
```
{ "id": "vllm-<uuid>", "custom_id": str, "response": BatchResponseData | null, "error": (str | ErrorResponse) | null }
```
Ids: `id = f"vllm-{random_uuid()}"`, `request_id = f"vllm-batch-{random_uuid()}"`
(:511-519,:546-563). `random_uuid()` == uuid4().hex (32 hex chars); we mirror it
with the same 128-bit-hex generator as `make_tool_call_id`
(tool_parsers/abstract.cpp:58-68).

## Endpoint dispatch table (run_batch.py:722-777,815-842)

The last path segment of `url` is the registry key; a per-entry `url_matcher`
disambiguates. Each key resolves a `handler_getter`; `None` (task unsupported by
the model) → the "Model does not support endpoint" error row (:600-603). A url
that matches no key → the "Supported endpoints" error row (:832-842).

| url | handler | our status |
|---|---|---|
| `/v1/chat/completions` | `openai_serving_chat.create_chat_completion` | **W1 wired** (`OpenAIServingChat::create_chat_completion`) |
| `/v1/embeddings` | `serving_embedding` | residual → unsupported-endpoint row (no embeddings serving handler yet; `SERVE-POOLING-ENDPOINTS` W4) |
| `…/score`, `…/rerank` | `serving_scores` | residual → unsupported-endpoint row |
| `/v1/audio/{transcriptions,translations}` | transcription/translation + `make_transcription_wrapper` (URL/data-URL audio fetch) | residual → unsupported-endpoint row (`SERVE-RESPONSES-MESSAGES`) |

## Run loop (run_batch.py:793-847)

`build_endpoint_registry` (from `init_app_state`'s serving objects) → read the
file → for each non-blank line: `BatchRequestInput.model_validate_json` →
`endpoint_key = url.split("/")[-1]` → `handle_endpoint_request` (match → submit →
`run_request(handler, body)`), else the unsupported-url error → collect →
`asyncio.gather` → `write_file`. `run_request` (:529-570) try/excepts the handler
(→ `create_error_response`), maps `AllResponse` → success row, `ErrorResponse` →
error row (status = `error.code`), and a streamed/other response → the
"must not be sent in stream mode" error row.

## Concurrency

Upstream submits every line "concurrently" via `asyncio.gather` over ONE async
engine (cooperative single thread). W1 runs lines **sequentially** through the
sync `LLMEngine`-backed `OpenAIServingChat` (each `create_chat_completion` runs
to completion) — the correct offline shape, just not overlapped. Overlapped
submission over `AsyncLLM` (the online frontend `main.cpp` already builds) is a
later brick.

## File I/O (run_batch.py:334-505)

Upstream `read_file` / `write_file` support local paths AND http(s) (GET / PUT
with retry, optional temp-dir staging) plus the transcription data:/http(s)
audio fetch (`download_bytes_from_url`, with `allowed_media_domains`). W1 ships
**local paths only** (`RunBatchFile`); the http(s)/data-URL fetch + the s3-style
upload are a NAMED later brick. `s3://` is not an upstream scheme (upstream is
http/https/data) — recorded so we do not invent it.

## Deviations (recorded)

- **Per-line isolation.** Upstream does NOT wrap `model_validate_json` in
  try/except (:813), so a MALFORMED line aborts the whole job
  (`test_completions_invalid_input` asserts returncode != 0). Our `RunBatch`
  library instead isolates a bad line into an error row and continues — the
  robust offline-batch behavior the pickup requires. The per-HANDLER isolation
  (an exception from the serving handler → an error row) IS faithful to
  `run_request` (:534-537). If the vLLM abort-on-bad-line contract is wanted, it
  belongs in the (unbuilt) `vllm run-batch` CLI wrapper's exit code, not the
  library.

## Dependencies

- `SERVE-OAI-BASIC` (the chat serving handler) — present; the runner is an
  orchestrator over it.
- Embeddings/score/rerank dispatch DEPENDS on `SERVE-POOLING-ENDPOINTS` (the
  pooling serving handlers, W4 of `specs/pooling-task-class.md`) — until those
  land, those urls resolve to the unsupported-endpoint error row.
- Audio transcription/translation dispatch DEPENDS on
  `SERVE-RESPONSES-MESSAGES` (the speech-to-text handlers) + a media-fetch
  brick.
- No GPU / kernel / hardware dependency (host-only orchestration; DGX not
  required).

## Port map

- `include/vllm/entrypoints/openai/run_batch.h`, `src/…/run_batch.cpp` —
  `BatchRequestInput`(implicit)/`BatchResponseData`/`BatchRequestOutput`,
  `RunBatch` (RunLine/RunLines/Run), `RunBatchFile`. **[W1 landed]**
- CLI `vllm run-batch` + `BatchFrontendArgs` (`-i`/`-o`/`--model`, metrics)
  (run_batch.py:232-297,850-877). **[residual — W2]**
- embeddings/score/rerank/audio dispatch entries. **[residual — depends on the
  respective serving handlers]**
- http(s)/data-URL file I/O + transcription media fetch. **[residual]**

## Tests to port (from `tests/entrypoints/openai/test_run_batch.py`)

The upstream cases drive the `vllm run-batch` SUBPROCESS CLI + real models, so
they are not directly runnable without the CLI brick. Re-expressed at the
LIBRARY level over the synthetic `OpenAIServingChat` engine harness (the
serving-test pattern) in `tests/vllm/entrypoints/openai/test_run_batch.cpp`:

| upstream case | our expression |
|---|---|
| `test_empty_file:375` | `run_batch: empty input yields empty output` |
| `test_completions:402` | `run_batch: chat batch yields ordered rows, custom_id echoed` (+ per-line BatchRequestOutput schema round-trip) |
| `test_completions_invalid_input:432` | `run_batch: a line missing custom_id yields an error row` (records our isolation deviation) + `…a malformed line…batch continues` |
| dispatch (`:832-842`, `:600-603`) | `…unknown url…supported-endpoints row`, `…embeddings url…unsupported-endpoint row` |
| `run_request` ErrorResponse branch (`:554-563`) | `…unknown model yields a 404 ErrorResponse row` |
| `test_embeddings`/`test_score`/`test_transcription`/`test_tool_calling` | SKIPPED-residual (need embeddings/score/audio handlers + the CLI subprocess) |

## Gates

- **W1 (done):** `test_openai_run_batch` 7/7 (80 assertions) over the synthetic
  engine, RED-first proven (dropping the custom_id echo fails 9 assertions).
  Clean CPU `-Werror` build. Correctness = the JSONL round-trip shape (custom_id
  echo, ordering, per-line error isolation, dispatch); no model-token gate is
  owed here (the handler it drives, `SERVE-OAI-BASIC`, owns token parity).
- **Later:** the `vllm run-batch` CLI e2e against the pinned oracle on a real
  model (a `SERVE-E2E-NIGHTLY`-class gate); embeddings/audio dispatch once those
  handlers land.

## Risks and decisions

- **Isolation vs upstream abort (decided):** the library isolates a malformed
  line into an error row rather than aborting the job (see § Deviations). The
  risk is a silent divergence from `test_completions_invalid_input`; mitigated
  by recording it here + in the header, and by keeping the abort contract
  available for the (unbuilt) CLI exit code.
- **Reuse-not-reimplement (decided):** the runner MUST call the existing
  `create_chat_completion`; it never re-derives sampling/generation. Risk if a
  future refactor tempts a copy — the row's code anchor + this spec pin the
  reuse.
- **`s3://` (decided):** not an upstream scheme (upstream is http/https/data);
  we will NOT invent an s3 path — the residual is the upstream http(s)/data-URL
  I/O only.
- **Row lifecycle:** `ACTIVE` (not `DONE`) — the CLI + non-chat dispatch + remote
  I/O residuals are real and named; `DONE` is owed only when the CLI e2e gate
  passes.

## Work breakdown

- **W0** — this spike + the `SERVE-BATCH-API` row. **[done]**
- **W1** — `RunBatch` + `RunBatchFile` (chat dispatch, custom_id echo, per-line
  isolation, local file I/O) + unit gate. **[done]**
- **W2** — the `vllm run-batch` CLI + `BatchFrontendArgs`; the abort-on-bad-line
  CLI exit code if mirrored.
- **W3** — embeddings/score/rerank dispatch (rides `SERVE-POOLING-ENDPOINTS`).
- **W4** — audio transcription/translation dispatch + media fetch.
- **W5** — http(s)/data-URL file I/O + upload retry; Prometheus metrics server;
  overlapped submission over `AsyncLLM`.
