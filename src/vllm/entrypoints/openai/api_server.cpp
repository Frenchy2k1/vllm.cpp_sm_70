// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 + the
// per-endpoint api_router modules. See api_server.h for scope + the cpp-httplib
// dependency deviation.
#include "vllm/entrypoints/openai/api_server.h"

#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/metrics/loggers.h"

namespace vllm::entrypoints::openai {

namespace {

size_t HttpWorkerCount(size_t max_concurrent_streams) {
  if (max_concurrent_streams == 0) {
    throw std::invalid_argument("max_concurrent_streams must be positive");
  }
  if (max_concurrent_streams >
      std::numeric_limits<size_t>::max() -
          ApiServer::kControlWorkerHeadroom) {
    throw std::invalid_argument("max_concurrent_streams is too large");
  }
  return max_concurrent_streams + ApiServer::kControlWorkerHeadroom;
}

size_t HttpWorkerCount(size_t max_concurrent_streams,
                       ApiServer::HttpWorkerPoolMode mode) {
  const size_t fixed_count = HttpWorkerCount(max_concurrent_streams);
  return mode == ApiServer::HttpWorkerPoolMode::kCapacityFixed ? fixed_count
                                                               : 0;
}

}  // namespace

// Opaque httplib::Server (pimpl — keeps httplib.h out of api_server.h).
struct ApiServer::Impl {
  Impl(size_t max_concurrent_streams, HttpWorkerPoolMode mode)
      : http_worker_count(HttpWorkerCount(max_concurrent_streams, mode)) {
    // cpp-httplib's default pool starts at hardware_concurrency()-1 and only
    // grows if idle_thread_count_ is exactly zero at enqueue. A burst can queue
    // accepted sockets while that counter is stale-positive; long-lived SSE
    // jobs then prevent the queued sockets from ever being read. A fixed floor
    // derived from the configured stream capacity removes that race and makes
    // resource use reproducible.
    if (http_worker_count != 0) {
      server.new_task_queue = [workers = http_worker_count]() {
        return new httplib::ThreadPool(workers);
      };
    }
    // Mirror vLLM's serving transport: vLLM serves through uvicorn over asyncio
    // (entrypoints/launcher.py:71,76), and asyncio disables Nagle on every
    // accepted TCP stream socket by default (CPython asyncio/base_events.py
    // _set_nodelay → setsockopt(IPPROTO_TCP, TCP_NODELAY, 1), invoked from
    // selector_events.py _SelectorSocketTransport). cpp-httplib defaults it off
    // (third_party/httplib/httplib.h:142) and applies it to the accepted socket
    // only when tcp_nodelay_ is set (httplib.h:12083). Per-token SSE frames are
    // tiny writes, so enabling TCP_NODELAY here puts each streamed frame on the
    // wire immediately instead of coalescing it under Nagle/delayed-ACK.
    server.set_tcp_nodelay(true);
  }

  httplib::Server server;
  size_t http_worker_count;
  // The legacy LLMEngine serving constructors remain for small synthetic
  // tests and embedding compatibility. Unlike AsyncLLM, that engine is driven
  // synchronously by its caller, so retain one shared lock for that seam only.
  // Production handlers use AsyncLLM and never take this request-level lock.
  std::mutex legacy_engine_mutex;
};

namespace {

// Build the OpenAI ErrorResponse JSON body for a failed request
// (serve/utils/error_response.py::create_error_response). `code` == the HTTP
// status code (upstream ErrorInfo.code carries it).
ApiServer::DispatchResult MakeError(int status, const std::string& type,
                                    const std::string& message) {
  ErrorResponse err;
  err.error.message = message;
  err.error.type = type;
  err.error.code = status;
  ApiServer::DispatchResult r;
  r.status = status;
  r.content_type = "application/json";
  r.body = nlohmann::json(err).dump();
  return r;
}

}  // namespace

ApiServer::ApiServer(OpenAIServingCompletion& completion,
                     OpenAIServingChat& chat, OpenAIServingModels& models,
                     std::string version, size_t max_concurrent_streams,
                     HttpWorkerPoolMode worker_pool_mode)
    : completion_(completion),
      chat_(chat),
      models_(models),
      version_(std::move(version)),
      impl_(std::make_unique<Impl>(max_concurrent_streams, worker_pool_mode)) {}

ApiServer::~ApiServer() = default;

ApiServer::DispatchResult ApiServer::handle_completions(
    const std::string& request_body) {
  // completion/api_router.py:46 (create_completion): parse → check_model →
  // handler → JSON (non-stream) or text/event-stream (stream).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  CompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid request: ") + e.what());
  }
  if (!models_.check_model(request.model)) {
    return MakeError(404, "NotFoundError",
                     "The model `" + request.model.value_or("") +
                         "` does not exist.");
  }

  CompletionResult result;
  try {
    std::unique_lock<std::mutex> legacy_lock(impl_->legacy_engine_mutex,
                                             std::defer_lock);
    if (!completion_.uses_async_engine()) legacy_lock.lock();
    result = completion_.create_completion(request);
  } catch (const std::exception& e) {
    // DISCRIMINATOR: attribute a 500 to its endpoint + model + raw cause so a
    // benchmark driver that only sees the generic HTTP body can still recover
    // the true failure. std::cerr only (survives SIGKILL escalation).
    std::cerr << "api-server: 500 endpoint=/v1/completions model="
              << request.model.value_or("") << " what=" << e.what() << "\n";
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
    out.sse_stream = std::move(result.sse_stream);
  } else {
    out.status = 200;
    out.content_type = "application/json";
    out.body = nlohmann::json(*result.response).dump();
  }
  return out;
}

ApiServer::DispatchResult ApiServer::handle_chat_completions(
    const std::string& request_body) {
  // chat_completion/api_router.py:53 (create_chat_completion).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  ChatCompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid request: ") + e.what());
  }
  if (!models_.check_model(request.model)) {
    return MakeError(404, "NotFoundError",
                     "The model `" + request.model.value_or("") +
                         "` does not exist.");
  }

  ChatCompletionResult result;
  try {
    std::unique_lock<std::mutex> legacy_lock(impl_->legacy_engine_mutex,
                                             std::defer_lock);
    if (!chat_.uses_async_engine()) legacy_lock.lock();
    result = chat_.create_chat_completion(request);
  } catch (const std::exception& e) {
    std::cerr << "api-server: 500 endpoint=/v1/chat/completions model="
              << request.model.value_or("") << " what=" << e.what() << "\n";
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
    out.sse_stream = std::move(result.sse_stream);
  } else {
    out.status = 200;
    out.content_type = "application/json";
    out.body = nlohmann::json(*result.response).dump();
  }
  return out;
}

ApiServer::DispatchResult ApiServer::handle_models() const {
  // models/api_router.py:21 (show_available_models).
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json(models_.show_available_models()).dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_health() const {
  // Upstream calls engine_client.check_health() before returning an empty 200.
  // This bounded server currently exposes process liveness only.
  DispatchResult out;
  out.status = 200;
  out.content_type = "text/plain";
  out.body.clear();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_version() const {
  // serve/instrumentator/basic.py:53 — {"version": <ver>}.
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json{{"version", version_}}.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_ping() const {
  // sagemaker/api_router.py:47-50 — GET/POST /ping is a liveness probe that
  // returns the same empty 200 as /health.
  return handle_health();
}

ApiServer::DispatchResult ApiServer::handle_metrics() const {
  // serve/instrumentator/metrics.py:82 — the prometheus text exposition served
  // by make_asgi_app(registry). The PrometheusResponse content type is
  // "text/plain; version=0.0.4; charset=utf-8".
  DispatchResult out;
  out.status = 200;
  out.content_type = v1::metrics::kContentTypeLatest;
  out.body = (metrics_ != nullptr) ? metrics_->Expose() : std::string();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_tokenize(
    const std::string& request_body) const {
  // serve/tokenize/api_router.py:46 (tokenize) over the TokenizeRequest union
  // (serve/tokenize/protocol.py:156): TokenizeCompletionRequest{prompt} OR
  // TokenizeChatRequest{messages, ...}. vLLM discriminates on the body shape
  // (pydantic Union) and, for the chat form, renders the model chat template
  // then tokenizes (serve/tokenize/serving.py:70-124). The response is
  // {count, max_model_len, tokens, token_strs} for both forms (:119).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  if (tokenizer_ == nullptr) {
    return MakeError(500, "InternalServerError", "No tokenizer configured.");
  }

  const bool return_token_strs = body.value("return_token_strs", false);
  std::string prompt;
  // add_special_tokens default differs by form: True for the completion form
  // (protocol.py:28), False for the chat form (protocol.py:78 — the chat
  // template already emits the model's special tokens).
  bool add_special_tokens;

  if (body.contains("messages")) {
    // ── TokenizeChatRequest (serve/tokenize/protocol.py:50). Render the
    // messages through the SAME chat-template seam create_chat_completion
    // tokenizes through (chat_.prompt_fn()), then tokenize — never reinvent
    // template rendering here (serve/tokenize/serving.py:84 preprocess_chat).
    if (!body.at("messages").is_array()) {
      return MakeError(400, "BadRequestError",
                       "`messages` must be an array.");
    }
    // check_generation_prompt (protocol.py:120-128): the two are exclusive.
    const bool add_generation_prompt =
        body.value("add_generation_prompt", true);
    const bool continue_final_message =
        body.value("continue_final_message", false);
    if (add_generation_prompt && continue_final_message) {
      return MakeError(400, "BadRequestError",
                       "Cannot set both `continue_final_message` and "
                       "`add_generation_prompt` to True.");
    }
    add_special_tokens = body.value("add_special_tokens", false);

    std::vector<ChatMessage> messages;
    std::vector<ChatCompletionToolsParam> tools;
    try {
      messages = body.at("messages").get<std::vector<ChatMessage>>();
      if (auto it = body.find("tools");
          it != body.end() && it->is_array()) {
        tools = it->get<std::vector<ChatCompletionToolsParam>>();
      }
    } catch (const std::exception& e) {
      return MakeError(400, "BadRequestError",
                       std::string("Invalid request: ") + e.what());
    }

    // build_chat_params folds add_generation_prompt / continue_final_message
    // into the template kwargs (protocol.py:130-146). Our ChatPromptFn seam
    // renders through the `add_generation_prompt` gate; continue_final_message
    // (open-ended final turn) suppresses the generation header, so it maps to
    // add_generation_prompt=false here.
    const bool render_generation_prompt =
        add_generation_prompt && !continue_final_message;
    try {
      prompt = chat_.prompt_fn()(messages, render_generation_prompt, tools);
    } catch (const std::exception& e) {
      return MakeError(400, "BadRequestError",
                       std::string("Chat template render failed: ") + e.what());
    }
  } else {
    // ── TokenizeCompletionRequest (serve/tokenize/protocol.py:24).
    if (!body.contains("prompt") || !body.at("prompt").is_string()) {
      return MakeError(400, "BadRequestError",
                       "`prompt` (string) is required.");
    }
    prompt = body.at("prompt").get<std::string>();
    add_special_tokens = body.value("add_special_tokens", true);
  }

  std::vector<int32_t> ids = add_special_tokens
                                 ? tokenizer_->EncodeWithSpecialTokens(prompt)
                                 : tokenizer_->Encode(prompt);

  nlohmann::json resp;
  resp["count"] = ids.size();
  resp["max_model_len"] = max_model_len_;
  resp["tokens"] = ids;
  if (return_token_strs) {
    std::vector<std::string> token_strs;
    token_strs.reserve(ids.size());
    for (int32_t id : ids) token_strs.push_back(tokenizer_->TokenText(id));
    resp["token_strs"] = token_strs;
  } else {
    resp["token_strs"] = nullptr;
  }
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = resp.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_detokenize(
    const std::string& request_body) const {
  // serve/tokenize/api_router.py:73 (detokenize) over DetokenizeRequest
  // (serve/tokenize/protocol.py:166) → DetokenizeResponse{prompt}.
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  if (!body.contains("tokens") || !body.at("tokens").is_array()) {
    return MakeError(400, "BadRequestError",
                     "`tokens` (array of token ids) is required.");
  }
  if (tokenizer_ == nullptr) {
    return MakeError(500, "InternalServerError", "No tokenizer configured.");
  }
  std::vector<int32_t> ids;
  try {
    for (const auto& t : body.at("tokens")) {
      ids.push_back(t.get<int32_t>());
    }
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid token id: ") + e.what());
  }
  nlohmann::json resp;
  resp["prompt"] = tokenizer_->Decode(ids);
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = resp.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_reset_prefix_cache(
    bool reset_running_requests, bool reset_external) const {
  // serve/dev/cache/api_router.py:20 → {"success": bool}.
  bool success = false;
  if (reset_prefix_cache_) {
    try {
      success = reset_prefix_cache_(reset_running_requests, reset_external);
    } catch (const std::exception& e) {
      return MakeError(500, "InternalServerError", e.what());
    }
  }
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json{{"success", success}}.dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_server_info() const {
  // serve/dev/server_info/api_router.py:43 — the three-key server_info shape.
  // vllm_config is rendered as a string; vllm_env/system_env are objects. This
  // bounded server exposes version + served model rather than the full config.
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  nlohmann::json info;
  info["vllm_config"] = std::string("served_model_name=") + models_.model_name();
  info["vllm_env"] = nlohmann::json::object();
  info["system_env"] =
      nlohmann::json{{"vllm_cpp_version", version_}};
  out.body = info.dump();
  return out;
}

void ApiServer::register_routes() {
  httplib::Server& server = impl_->server;

  // Write a DispatchResult onto an httplib::Response — either the full JSON body
  // or a chunked text/event-stream (SSE), matching upstream's JSONResponse vs
  // StreamingResponse(media_type="text/event-stream").
  auto write = [](const DispatchResult& result, httplib::Response& res) {
    res.status = result.status;
    if (result.streaming) {
      if (result.sse_stream != nullptr) {
        // W2 live StreamingResponse: one provider invocation pulls one
        // per-request collector output. A slow/disconnected client occupies
        // only its httplib worker; AsyncLLM keeps batching other requests.
        std::shared_ptr<SseStream> stream = result.sse_stream;
        res.set_chunked_content_provider(
            result.content_type,
            [stream](size_t /*offset*/, httplib::DataSink& sink) -> bool {
              try {
                std::string chunk;
                if (!stream->next(chunk)) {
                  sink.done();
                  return true;
                }
                if (!sink.write(chunk.data(), chunk.size())) {
                  stream->abort();
                  return false;
                }
                return true;
              } catch (...) {
                // MID-FLIGHT WITNESS: a provider exception here means the live
                // stream died after headers were already sent (the client sees
                // a truncated body, not a 500). Rethrow-to-inspect so the raw
                // cause reaches stderr before the abort.
                try {
                  std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                  std::cerr << "sse: stream aborted mid-flight: " << e.what()
                            << "\n";
                } catch (...) {
                  std::cerr << "sse: stream aborted mid-flight: unknown error\n";
                }
                stream->abort();
                return false;
              }
            },
            [stream](bool success) {
              if (!success) stream->abort();
            });
        return;
      }

      // Legacy synchronous compatibility/test seam: write the precomputed
      // chunks exactly as before.
      auto chunks = std::make_shared<std::vector<std::string>>(result.sse_chunks);
      res.set_chunked_content_provider(
          result.content_type,
          [chunks](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            for (const std::string& chunk : *chunks) {
              if (!sink.write(chunk.data(), chunk.size())) return false;
            }
            sink.done();
            return true;
          });
    } else {
      res.set_content(result.body, result.content_type);
    }
  };

  server.Post("/v1/completions",
              [this, write](const httplib::Request& req, httplib::Response& res) {
                write(handle_completions(req.body), res);
              });
  server.Post("/v1/chat/completions",
              [this, write](const httplib::Request& req, httplib::Response& res) {
                write(handle_chat_completions(req.body), res);
              });
  server.Get("/v1/models",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_models(), res);
             });
  server.Get("/health",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_health(), res);
             });
  server.Get("/version",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_version(), res);
             });

  // ── C8 additive routes: each registered only when its backing is attached,
  // so a default-constructed server is byte-identical to before. /ping and
  // /server_info are read-only liveness/introspection and always present. ─────
  server.Get("/ping",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_ping(), res);
             });
  server.Post("/ping",
              [this, write](const httplib::Request&, httplib::Response& res) {
                write(handle_ping(), res);
              });
  server.Get("/server_info",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_server_info(), res);
             });

  if (metrics_ != nullptr) {
    server.Get("/metrics",
               [this, write](const httplib::Request&, httplib::Response& res) {
                 write(handle_metrics(), res);
               });
  }
  if (tokenizer_ != nullptr) {
    server.Post(
        "/tokenize",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          write(handle_tokenize(req.body), res);
        });
    server.Post(
        "/detokenize",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          write(handle_detokenize(req.body), res);
        });
  }
  if (reset_prefix_cache_) {
    server.Post(
        "/reset_prefix_cache",
        [this, write](const httplib::Request& req, httplib::Response& res) {
          // Query params default false (cache/api_router.py:22-26).
          const bool reset_running =
              req.has_param("reset_running_requests") &&
              req.get_param_value("reset_running_requests") == "true";
          const bool reset_external =
              req.has_param("reset_external") &&
              req.get_param_value("reset_external") == "true";
          write(handle_reset_prefix_cache(reset_running, reset_external), res);
        });
  }
}

bool ApiServer::listen(const std::string& host, int port) {
  register_routes();
  return impl_->server.listen(host, port);
}

int ApiServer::bind_to_any_port(const std::string& host) {
  register_routes();
  return impl_->server.bind_to_any_port(host);
}

bool ApiServer::serve() { return impl_->server.listen_after_bind(); }

void ApiServer::stop() { impl_->server.stop(); }

bool ApiServer::is_running() const { return impl_->server.is_running(); }

size_t ApiServer::http_worker_count() const {
  return impl_->http_worker_count;
}

}  // namespace vllm::entrypoints::openai
