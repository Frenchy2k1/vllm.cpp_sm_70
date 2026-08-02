# `SERVE-CLI-CHAT`: interactive chat and completion commands

Date: 2026-08-01
Row: `SERVE-CLI-CHAT`
Claim: `CLAIM-SERVE-CLI-CHAT-SPIKE`
Pinned vLLM: `5559679229bc961848b121ccdeaa8fa5d79bec98`

## Scope

Port the pinned vLLM `chat` and `complete` command contracts into
`vllm-cli`. Both commands target a running OpenAI-compatible server, stream
output, support one-shot and interactive modes, select an explicit model or
the first model returned by `/v1/models`, and optionally print TTFT/TPS. Keep
the current in-process `vllm-cli --model DIR --prompt TEXT` form as a deprecated
compatibility alias for `vllm-cli complete --local-model DIR --quick TEXT`.

In scope:

- `vllm-cli chat [--url URL] [--model-name NAME] [--api-key KEY]
  [--system-prompt TEXT] [-q MESSAGE] [--stats]`;
- `vllm-cli complete [--url URL] [--model-name NAME] [--api-key KEY]
  [--max-tokens N] [-q PROMPT] [--stats]`;
- interactive stdin loops, EOF and interrupt handling, streamed content,
  usage-derived statistics, model discovery, HTTP/SSE failure reporting, and
  stable exit codes;
- preservation of the existing local-model completion flags and C-ABI-only
  implementation path.

Out of scope:

- starting or managing a server from the client;
- adding a new public C ABI or changing generation semantics;
- terminal editing/history, markdown rendering, tool execution, multimodal
  file attachment, TLS policy, or non-OpenAI endpoints;
- model correctness or throughput claims. This is a client/packaging row.

The current matrix statement that the pin has no direct commands is false.
The pin imports `vllm.entrypoints.cli.openai` and registers its commands in
`vllm/entrypoints/cli/main.py:17-37,83-95`; `openai.py:155-234` implements
`chat`, and `openai.py:237-312` implements `complete`.

## Upstream chain

All upstream anchors below are at the pinned SHA.

| Surface | Pinned behavior |
|---|---|
| CLI registration | `vllm/entrypoints/cli/main.py:17-37,73-98` lazily imports command modules, registers each `CLISubcommand`, validates, then dispatches. |
| Command interface | `vllm/entrypoints/cli/types.py:13-29` defines the name, argument-registration, validation, and dispatch contract. |
| Client/model resolution | `vllm/entrypoints/cli/openai.py:30-45` uses `--url`, resolves `--api-key` over `OPENAI_API_KEY` over `EMPTY`, and selects `--model-name` or the first `/models` result. |
| Stream rendering | `vllm/entrypoints/cli/openai.py:48-100` prints chat content or completion text as it arrives, collects usage, and prints TTFT/TPS only when measurable. |
| Chat loop | `vllm/entrypoints/cli/openai.py:155-234` preserves system/user/assistant turns, supports `--quick`, streams every request, and exits interactive mode on EOF. |
| Complete loop | `vllm/entrypoints/cli/openai.py:237-308` supports `--max-tokens`, `--quick`, streamed independent prompts, and EOF exit. |
| Server endpoints | `src/vllm/entrypoints/openai/api_server.cpp` locally exposes `/v1/models`, `/v1/chat/completions`, and `/v1/completions`; `serving_chat.cpp` and `serving_completion.cpp` own response semantics. |

Dispatch is dynamic at the HTTP boundary, not at a device kernel. Runtime trace
is therefore a deterministic fake-server transcript, not `nsys`: record the
request method/path/headers/body and the exact SSE frames consumed. No GPU or
dependency kernel chain participates.

## Our baseline

`examples/cli/main.cpp:1-207` is a single in-process completion program. It
loads a model through the stable C ABI, maps sampling flags, and calls
`vllm_complete` or `vllm_complete_stream`. It has no subcommands, remote URL,
model discovery, chat conversation state, OpenAI JSON/SSE client, system
prompt, quick mode, or TTFT/TPS display.

Useful existing seams:

- `include/vllm.h:370-453` provides local streaming completion and chat;
- `src/capi/vllm_c.cpp:604-951` owns their no-throw implementations;
- `third_party/httplib/httplib.h` is the already-vendored HTTP transport used
  by the server and can also drive a client without adding a dependency;
- `nlohmann::json` is already vendored and used throughout the entrypoints;
- `tests/capi/test_capi.cpp:567-711` proves completion stream concatenation,
  early stop, recovery, determinism, and argument errors;
- `tests/capi/test_chat_prompt.cpp:37-89` proves template-backed chat prompt
  construction.

Honest gaps: there is no CLI unit target, parsing and execution are fused in
`main`, numeric parsing uses permissive `atoi`/`atof`, `NextArg` exits from a
helper, and the current help smoke test covers only the server. These are
testability defects to fix as part of the port, not reasons to fork behavior.

## Design decision and alternatives

Three approaches were considered:

1. **Remote-only rewrite.** This is the closest vLLM mirror but would remove
   the documented local, library-first example and break existing scripts.
2. **In-process-only subcommands.** This reuses the C ABI but does not mirror
   vLLM's running-server commands, model discovery, or OpenAI/SSE boundary.
3. **Dual-mode command family (selected).** Add exact remote `chat` and
   `complete` subcommands, keep the existing invocation as a compatibility
   alias, and factor parsing/stream shaping behind testable helpers. This
   mirrors upstream without regressing the project's existing consumption
   example.

Remote mode is the default for named subcommands. Local mode is explicit via
`--local-model`; `--url` and `--local-model` are mutually exclusive. The legacy
top-level form remains local. No implicit server start, silent fallback, or
network-to-local mode switch is allowed.

The remote client uses the vendored `httplib` and `nlohmann::json`; it does not
vendor the OpenAI SDK. It accepts `http://` URLs in W2. An `https://` URL fails
clearly until the existing optional OpenSSL build can provide a verified TLS
client; it must never downgrade to HTTP.

## Port map

| Upstream | Local target | Port/deviation |
|---|---|---|
| `entrypoints/cli/types.py:13-29` | NEW `examples/cli/cli_app.{h,cpp}` | `Command` enum plus parse/validate/run result, no process exit in helpers. |
| `entrypoints/cli/main.py:73-98` | `examples/cli/main.cpp` | Dispatch `chat`/`complete`; retain legacy compatibility translation. |
| `entrypoints/cli/openai.py:30-45` | NEW `examples/cli/openai_client.{h,cpp}` | URL/key/model resolution; dependency-free transport. |
| `entrypoints/cli/openai.py:48-100` | NEW `examples/cli/stream_printer.{h,cpp}` | Incremental SSE JSON shaping, content/text extraction, usage, TTFT/TPS. |
| `entrypoints/cli/openai.py:155-234` | `cli_app.cpp` chat runner | Same conversation accumulation, quick/interactive behavior, EOF exit. |
| `entrypoints/cli/openai.py:237-308` | `cli_app.cpp` complete runner | Same quick/interactive behavior and `max_tokens` omission semantics. |
| Existing project-only local CLI | `cli_app.cpp` local runner | Preserve flags and C ABI calls; documented compatibility deviation. |

## Tests to port

The pin has no dedicated `tests/entrypoints/cli` module for these commands, so
the executable spec is `openai.py` plus endpoint tests. Port the following
behaviors into NEW `tests/examples/test_cli.cpp` with a fake OpenAI server and
in-memory stdin/stdout/stderr:

| Case | Source | Local assertion |
|---|---|---|
| model selection | `openai.py:30-45` | explicit name avoids `/models`; absent name uses the first returned ID; empty list is a clear error |
| auth precedence | `openai.py:33-35` | flag beats environment; absent values send `Bearer EMPTY` |
| chat quick | `openai.py:163-182` | system+user request, streamed text, one request, exit 0 |
| chat interactive | `openai.py:184-196` | second request contains prior assistant output; EOF exits cleanly |
| complete quick/interactive | `openai.py:243-268` | correct endpoint/body, streamed text, independent prompts |
| optional max tokens | `openai.py:247-258` | omitted when unset, present with exact integer when set |
| stats | `openai.py:48-100,170-173,251-254` | requests usage frames and prints TTFT/TPS only with usable timing+usage |
| malformed/failed stream | endpoint contract | non-2xx, invalid JSON, truncated SSE, missing choices, and `[DONE]` handling are deterministic and never crash |
| CLI validation | local contract | help 0; unknown/missing/invalid/conflicting args 2; network/runtime failure 1 |
| compatibility | `examples/cli/main.cpp:68-207` | legacy argv translates to the local complete configuration byte-for-byte |

Tests use real parsing, JSON, HTTP, and SSE code. Transport injection is only
for deterministic clock/input/output control; assertions target emitted
requests and user-visible output, not mock call counts. No upstream test is
dropped. The absence of upstream command tests is recorded rather than hidden.

## Error and lifecycle contract

- Parse/help errors happen before network or model work. Help exits 0; invalid
  invocation exits 2.
- Connection, HTTP, JSON/SSE, model-discovery, and generation failures include
  endpoint context on stderr and exit 1. API keys are never printed.
- EOF ends an interactive session with exit 0. SIGINT cancels the current
  request or exits the prompt loop without a stack trace; it never leaves a
  local C-ABI request alive.
- Each remote request has a finite connect/read timeout. A stalled peer cannot
  hang CI or the terminal forever.
- The chat history appends the assistant turn only after a successful complete
  stream. A failed/aborted response is not committed to history.
- Local engine ownership remains load once, free once. All helper APIs return
  status objects and are no-throw across `main`.

## Gates

Spike checkpoint (this change, CPU-only):

```bash
python3 scripts/check-agent-record.py
python3 tests/scripts/test_agent_record.py
python3 scripts/check-doc-checkpoint.py --staged
python3 tests/scripts/test_doc_checkpoint.py
python3 scripts/check-readme-structure.py
python3 tests/scripts/test_check_readme_structure.py
git diff --check
```

Implementation gates:

```bash
cmake -S . -B build-cli-cpu -DVLLM_CPP_CUDA=OFF \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-Wall -Wextra -Werror'
cmake --build build-cli-cpu --target vllm-cli test_cli -j2
ctest --test-dir build-cli-cpu --output-on-failure \
  -R 'test_cli|test_capi|test_chat_prompt|test_openai_(api_server|serving|conformance)'
```

Correctness is exact request/transcript and stdout/stderr parity for the
inventoried cases. Sanitizer gates run `test_cli` under ASan+UBSan and TSan.
End-to-end uses the CPU synthetic server/engine so no checkpoint or GPU is
needed. A later real-server smoke on a gate model is useful packaging evidence
but does not block the CPU-complete client row.

Performance and memory are `NOT APPLICABLE` for model inference: no compute
path changes. The client gate instead enforces bounded buffering (stream
content is appended only where chat history requires it), finite timeouts, and
no per-turn history duplication. Backend/architecture matrix: CPU execution
for the client; CUDA/ROCm/Metal are inherited from the independently running
server and are not exercised or claimed here.

## Dependencies

- Row dependency: the existing OpenAI completion/chat serving and C ABI seams;
  no open lifecycle row must advance for W1/W2.
- Toolchain: C++17, vendored `httplib`, vendored `nlohmann::json`, doctest,
  CMake. No new package or license.
- Hardware/data: CPU and loopback TCP only; no GPU, model, oracle venv, or
  downloaded checkpoint.
- The HTTP client must coordinate with `SERVE-HTTP-TRANSPORT` only if shared
  transport code is edited. The plan avoids that by keeping client code under
  `examples/cli/`.

## Work breakdown

| Leaf | Files/ownership | Deliverable | Gate |
|---|---|---|---|
| W1 parse/dispatch | `examples/cli/{main,cli_app}.{cpp,h}`, `tests/examples/test_cli.cpp` | subcommands, validation, legacy translation, injectable IO/clock | parse/exit-code tests, RED first |
| W2 OpenAI transport | `examples/cli/openai_client.{h,cpp}`, same test | URL/auth/model discovery, POST, incremental SSE, timeouts/errors | fake-server transcript and failure matrix |
| W3 complete | `cli_app.cpp`, same test | quick+interactive completion, max-tokens, stats | exact bodies/output, EOF/SIGINT |
| W4 chat | `cli_app.cpp`, same test | system prompt, conversation history, quick+interactive, stats | multi-turn transcript and failed-turn rollback |
| W5 packaging | `examples/CMakeLists.txt`, README/help docs, record surfaces | build wiring, CPU sanitizer/conformance closure | clean Release `-Werror`, focused CTest, record checks |

W1 and W2 may be claimed separately because their owned production files do
not overlap; both share the test file only under an explicit lead claim. W3
depends on W1+W2. W4 depends on W1+W2 and can run alongside W3 with split test
sections. W5 closes after W3+W4.

## Risks and decisions

- **Corrected inventory:** chat/complete are present at the pin. Future work
  mirrors them; it is not a greenfield UX design.
- **Compatibility:** removing the current local CLI would be an unnecessary
  breaking change. The compatibility alias is explicit and test-pinned.
- **SSE boundaries:** arbitrary TCP chunks do not equal SSE frames. W2 buffers
  through the blank-line delimiter and parses multi-line `data:` fields.
- **Secrets:** authorization values never appear in diagnostics or captured
  golden output.
- **Unbounded sessions:** conversation history necessarily grows by turns;
  no hidden truncation policy is invented. Model context-limit errors surface
  from the server exactly as errors.
- **Product calls:** none remain. Remote behavior mirrors vLLM; preserving the
  already-shipped local invocation is the only project-specific compatibility
  decision.
