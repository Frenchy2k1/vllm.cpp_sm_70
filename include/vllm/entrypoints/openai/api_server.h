// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 (the route
// shapes) + the per-endpoint api_router modules
//   - completion/api_router.py:34    POST /v1/completions
//   - chat_completion/api_router.py:40 POST /v1/chat/completions
//   - models/api_router.py:20        GET  /v1/models
//   - serve/instrumentator/health.py:22 GET /health
//   - serve/instrumentator/basic.py:53  GET /version
//
// SCOPE (M3.1 Task 4 / T0): the thin HTTP transport over the Task-2 serving
// handlers. The OpenAI-protocol logic (parse → SamplingParams → engine →
// response/SSE) already lives in serving_{completion,chat}; this file only
// binds it to routes over the vendored cpp-httplib (third_party/httplib.h).
//
// DEPENDENCY DEVIATION (recorded): cpp-httplib is a header-only MIT HTTP
// TRANSPORT library (the same choice as llama.cpp's server) — NOT a compute/ML
// dependency, so it is consistent with the no-pytorch / no-ggml rule. It is
// gated behind the CMake option VLLM_CPP_SERVER.
//
// The route DISPATCH (parse body → handler → status + body / SSE chunks) is
// decoupled from the socket via the handle_* methods below, so the request
// handling is unit-testable WITHOUT binding a port; listen() wires the same
// methods onto an httplib::Server.
#ifndef VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_
#define VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"

namespace vllm::tok {
class Tokenizer;
}  // namespace vllm::tok

namespace vllm::v1::metrics {
class PrometheusStatLogger;
}  // namespace vllm::v1::metrics

namespace vllm::entrypoints::openai {

// The HTTP api_server. Holds non-owning references to the serving handlers +
// model registry (constructed + owned by the caller; they outlive it).
class ApiServer {
 public:
  enum class HttpWorkerPoolMode {
    kCapacityFixed,
    // Diagnostic same-binary fallback for A/B attribution only.
    kLegacyDynamic,
  };

  // A live SSE response occupies one cpp-httplib worker while it waits on its
  // collector. Keep enough fixed workers for every scheduler-visible stream,
  // plus a small bounded reserve for health/discovery/control requests.
  static constexpr size_t kDefaultMaxConcurrentStreams = 8;
  static constexpr size_t kControlWorkerHeadroom = 4;

  ApiServer(OpenAIServingCompletion& completion, OpenAIServingChat& chat,
            OpenAIServingModels& models, std::string version,
            size_t max_concurrent_streams = kDefaultMaxConcurrentStreams,
            HttpWorkerPoolMode worker_pool_mode =
                HttpWorkerPoolMode::kCapacityFixed);
  ~ApiServer();

  ApiServer(const ApiServer&) = delete;
  ApiServer& operator=(const ApiServer&) = delete;

  // The socket-free result of dispatching one request (shared by the HTTP layer
  // and unit tests). Streaming uses either the legacy precomputed vector or
  // W2's live pull source.
  struct DispatchResult {
    int status = 200;
    bool streaming = false;
    std::string content_type = "application/json";
    std::string body;                     // populated when !streaming
    std::vector<std::string> sse_chunks;  // populated when streaming
    std::shared_ptr<SseStream> sse_stream;  // live AsyncLLM streaming path
  };

  // Route dispatch (parse the JSON body → the Task-1 request → the Task-2
  // handler → a DispatchResult). On a malformed body → 400; an unknown model →
  // 404; an internal failure → 500 — each carrying the OpenAI ErrorResponse
  // JSON. These are what listen()'s handlers call, and what the tests exercise
  // without a socket.
  DispatchResult handle_completions(const std::string& request_body);
  DispatchResult handle_chat_completions(const std::string& request_body);
  DispatchResult handle_models() const;
  DispatchResult handle_health() const;
  DispatchResult handle_version() const;

  // ── C8 utility + observability surface (SERVE-METRICS /
  // SERVE-UTILITY-ENDPOINTS). Each is ADDITIVE and OPT-IN: a route is only
  // registered when its backing dependency is configured below, so a server
  // constructed without them behaves byte-identically to before. ──────────────

  // GET /metrics — Prometheus text-0.0.4 exposition
  // (serve/instrumentator/metrics.py). Registered only when a metrics logger is
  // attached. Returns 404 (route absent) otherwise; the handler itself returns
  // the exposition with the prometheus content type.
  DispatchResult handle_metrics() const;
  // POST /tokenize, POST /detokenize (serve/tokenize/api_router.py). Registered
  // only when a tokenizer is attached. /tokenize accepts BOTH forms of the
  // TokenizeRequest union (serve/tokenize/protocol.py:156): the raw
  // TokenizeCompletionRequest{prompt} and the TokenizeChatRequest{messages, ...}
  // — the chat form renders through chat_.prompt_fn() (the same model chat
  // template create_chat_completion uses) then tokenizes. Schemas match vLLM.
  DispatchResult handle_tokenize(const std::string& request_body) const;
  DispatchResult handle_detokenize(const std::string& request_body) const;
  // POST /reset_prefix_cache (serve/dev/cache/api_router.py) → {"success": b}.
  DispatchResult handle_reset_prefix_cache(bool reset_running_requests,
                                           bool reset_external) const;
  // GET /ping (serve/sagemaker/api_router.py) — liveness, mirrors /health.
  DispatchResult handle_ping() const;
  // GET /server_info (serve/dev/server_info/api_router.py).
  DispatchResult handle_server_info() const;
  // GET /tokenizer_info (serve/tokenize/api_router.py:95-108 attach_router +
  // serving.py:154-160 get_tokenizer_info → TokenizerInfoResponse). Gated in
  // vLLM behind `enable_tokenizer_info_endpoint`; we mirror that with the
  // tokenizer-info flag below (default off → the route is not registered → 404).
  // Surfaces the tokenizer_config.json-equivalent fields our BPE tokenizer can
  // genuinely back (tokenizer_class, model_max_length, vocab_size,
  // bos/eos_token_id, added_tokens_decoder); fields vLLM emits that our
  // tokenizer does not carry are OMITTED + named in the spec.
  DispatchResult handle_tokenizer_info() const;
  // POST /abort_requests (serve/dev/rlhf/api_router.py:94-138) → {"status":
  // "aborted","aborted":<count>}. Parses {request_ids:[...]}; empty/missing ids
  // means "abort all in-flight". Wired to the injected abort callback below.
  DispatchResult handle_abort_requests(const std::string& request_body) const;

  // Attach the Prometheus stat logger backing GET /metrics (non-owning; must
  // outlive the server). Enables the /metrics route.
  void set_metrics_logger(const v1::metrics::PrometheusStatLogger* logger) {
    metrics_ = logger;
  }
  // Attach the tokenizer + max_model_len backing /tokenize and /detokenize
  // (non-owning; must outlive the server).
  void set_tokenizer(const vllm::tok::Tokenizer* tokenizer,
                     int64_t max_model_len) {
    tokenizer_ = tokenizer;
    max_model_len_ = max_model_len;
  }
  // Enable GET /tokenizer_info (mirrors vLLM's `enable_tokenizer_info_endpoint`
  // CLI flag: off by default, so the route is absent → 404 unless enabled AND a
  // tokenizer is attached).
  void set_tokenizer_info_enabled(bool enabled) {
    tokenizer_info_enabled_ = enabled;
  }
  // Attach the abort callback backing POST /abort_requests. Mirrors
  // EngineClient.abort(request_ids); an empty id list means "abort all
  // in-flight". Returns the number of requests aborted (the response's
  // "aborted" count).
  void set_abort_requests(
      std::function<int(const std::vector<std::string>&)> abort_requests) {
    abort_requests_ = std::move(abort_requests);
  }
  // Attach the prefix-cache reset callback backing POST /reset_prefix_cache.
  // Signature mirrors EngineClient.reset_prefix_cache(reset_running_requests,
  // reset_external) → success.
  void set_reset_prefix_cache(
      std::function<bool(bool, bool)> reset_prefix_cache) {
    reset_prefix_cache_ = std::move(reset_prefix_cache);
  }

  // Bind host:port, register the routes and serve until stop() (blocking).
  // Returns false if the bind fails.
  bool listen(const std::string& host, int port);

  // Bind to an OS-assigned ephemeral port (port 0) on `host`; returns the bound
  // port (or -1 on failure). Follow with serve() to run the loop. Used by the
  // smoke test to avoid a fixed-port race.
  int bind_to_any_port(const std::string& host);
  // Run the accept loop after a successful bind_to_any_port (blocking).
  bool serve();
  // Signal the accept loop to stop (thread-safe); listen()/serve() then return.
  void stop();
  // True once the server is accepting connections (poll before issuing
  // requests against a bind_to_any_port + serve() server on another thread).
  bool is_running() const;
  // Configured fixed worker count. Exposed for startup diagnostics and the
  // transport-capacity regression; it is max_concurrent_streams + headroom,
  // or zero when the diagnostic legacy-dynamic mode is selected.
  size_t http_worker_count() const;

 private:
  OpenAIServingCompletion& completion_;
  OpenAIServingChat& chat_;
  OpenAIServingModels& models_;
  std::string version_;

  // Opt-in C8 backings (all nullptr/empty by default → routes not registered).
  const v1::metrics::PrometheusStatLogger* metrics_ = nullptr;
  const vllm::tok::Tokenizer* tokenizer_ = nullptr;
  int64_t max_model_len_ = 0;
  bool tokenizer_info_enabled_ = false;
  std::function<bool(bool, bool)> reset_prefix_cache_;
  std::function<int(const std::vector<std::string>&)> abort_requests_;

  // Opaque httplib::Server (pimpl keeps third_party/httplib.h out of this
  // header — only api_server.cpp and the smoke test pull it in).
  struct Impl;
  std::unique_ptr<Impl> impl_;

  void register_routes();
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_
