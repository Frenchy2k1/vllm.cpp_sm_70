// OpenAI serve request logging — shaped like Python vLLM --enable-log-requests.
// Ported concepts from: vllm/entrypoints/logger.py + api_server access logs.
#pragma once

#include <cstdint>
#include <string>

namespace vllm::entrypoints::openai {

struct RequestLogConfig {
  // Mirrors --enable-log-requests / --disable-log-requests (default ON for serve).
  bool enable_log_requests = true;
  // Mirrors --enable-log-outputs (requires enable_log_requests).
  bool enable_log_outputs = false;
  // Mirrors --max-log-len (chars of prompt/output preview).
  int max_log_len = 256;
  // Lab deep stages (former VT_SERVER_VERBOSE chat-dbg).
  bool debug_stages = false;
};

// Process-wide config (set once at server startup).
void ConfigureRequestLogger(const RequestLogConfig& cfg);
const RequestLogConfig& GetRequestLogConfig();

// Truncate for log lines (max_log_len); escapes newlines.
std::string LogPreview(const std::string& s, int max_len);

// HTTP ingress (api_server).
void LogHttpIngress(const char* method, const char* path, size_t body_bytes);

// After chat/completions parse + template.
void LogRequestReceived(const std::string& request_id, const std::string& endpoint,
                        const std::string& model, bool stream, int max_tokens,
                        int n_messages, int n_tools, size_t prompt_chars,
                        const std::string& prompt, const std::string& roles_summary);

// Mid-flight stages (debug_stages only) — heartbeats during prefill/decode.
void LogRequestStage(const std::string& request_id, const std::string& stage);

// Completion of a request.
void LogRequestFinished(const std::string& request_id, int prompt_tokens,
                        int completion_tokens, const std::string& finish_reason,
                        double elapsed_sec, const std::string& output_text);

// Errors.
void LogRequestError(const std::string& request_id, const std::string& endpoint,
                     const std::string& what);

}  // namespace vllm::entrypoints::openai
