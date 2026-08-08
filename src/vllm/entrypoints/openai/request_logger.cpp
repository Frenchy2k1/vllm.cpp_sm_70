#include "vllm/entrypoints/openai/request_logger.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>

namespace vllm::entrypoints::openai {
namespace {

RequestLogConfig g_cfg{};
std::mutex g_mu;
std::chrono::steady_clock::time_point g_t0 = std::chrono::steady_clock::now();

int64_t MsSinceStart() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - g_t0)
      .count();
}

void Emit(const std::string& line) {
  std::lock_guard<std::mutex> lock(g_mu);
  // Upstream-ish prefix: INFO level one-liners on stderr (no python logging stack).
  std::cerr << "INFO " << line << "\n";
  std::cerr.flush();
}

}  // namespace

void ConfigureRequestLogger(const RequestLogConfig& cfg) {
  g_cfg = cfg;
  if (g_cfg.enable_log_outputs && !g_cfg.enable_log_requests) {
    g_cfg.enable_log_outputs = false;
  }
  if (g_cfg.max_log_len < 16) g_cfg.max_log_len = 16;
}

const RequestLogConfig& GetRequestLogConfig() { return g_cfg; }

std::string LogPreview(const std::string& s, int max_len) {
  std::string out;
  out.reserve(static_cast<size_t>(max_len) + 8);
  const size_t n = std::min(s.size(), static_cast<size_t>(max_len));
  for (size_t i = 0; i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '\n')
      out += "\\n";
    else if (c == '\r')
      out += "\\r";
    else if (c == '\t')
      out += "\\t";
    else if (c < 32 || c == 127)
      out += '?';
    else
      out += static_cast<char>(c);
  }
  if (s.size() > n) out += "...";
  return out;
}

void LogHttpIngress(const char* method, const char* path, size_t body_bytes) {
  if (!g_cfg.enable_log_requests && !g_cfg.debug_stages) return;
  std::ostringstream os;
  os << "api: " << method << " " << path << " body_bytes=" << body_bytes
     << " t+" << MsSinceStart() << "ms";
  Emit(os.str());
}

void LogRequestReceived(const std::string& request_id, const std::string& endpoint,
                        const std::string& model, bool stream, int max_tokens,
                        int n_messages, int n_tools, size_t prompt_chars,
                        const std::string& prompt, const std::string& roles_summary) {
  if (!g_cfg.enable_log_requests) return;
  std::ostringstream os;
  os << "Received request " << request_id << " endpoint=" << endpoint
     << " model=" << model << " stream=" << (stream ? "1" : "0")
     << " max_tokens=" << max_tokens << " msgs=" << n_messages
     << " tools=" << n_tools << " prompt_chars=" << prompt_chars;
  if (!roles_summary.empty()) os << " roles=" << roles_summary;
  os << " prompt: '" << LogPreview(prompt, g_cfg.max_log_len) << "'";
  Emit(os.str());
}

void LogRequestStage(const std::string& request_id, const std::string& stage) {
  if (!g_cfg.debug_stages) return;
  std::ostringstream os;
  os << "chat-dbg t+" << MsSinceStart() << "ms id=" << request_id << " " << stage;
  Emit(os.str());
}

void LogRequestFinished(const std::string& request_id, int prompt_tokens,
                        int completion_tokens, const std::string& finish_reason,
                        double elapsed_sec, const std::string& output_text) {
  if (!g_cfg.enable_log_requests) return;
  std::ostringstream os;
  os << "Finished request " << request_id << " prompt_tokens=" << prompt_tokens
     << " completion_tokens=" << completion_tokens
     << " total_tokens=" << (prompt_tokens + completion_tokens)
     << " finish_reason=" << finish_reason << " elapsed_s=" << elapsed_sec;
  if (completion_tokens > 0 && elapsed_sec > 0.001) {
    os << " gen_tok_s=" << (static_cast<double>(completion_tokens) / elapsed_sec);
  }
  if (g_cfg.enable_log_outputs && !output_text.empty()) {
    os << " output: '" << LogPreview(output_text, g_cfg.max_log_len) << "'";
  }
  Emit(os.str());
}

void LogRequestError(const std::string& request_id, const std::string& endpoint,
                     const std::string& what) {
  std::ostringstream os;
  os << "ERROR request " << (request_id.empty() ? "-" : request_id)
     << " endpoint=" << endpoint << " what=" << what;
  Emit(os.str());
}

}  // namespace vllm::entrypoints::openai
