#include "vllm/platform/console_shutdown.h"

#include <atomic>
#include <iostream>
#include <thread>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace vllm::platform {

class ConsoleShutdown::Impl {
 public:
  explicit Impl(std::function<void()> stop) : stop_(std::move(stop)) {}

  void RequestStop() {
    bool expected = false;
    if (requested_.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
      stop_();
    }
  }

  std::function<void()> stop_;
  std::atomic<bool> requested_{false};
#if !defined(_WIN32)
  std::thread watcher_;
  int pipe_fds_[2] = {-1, -1};
#endif
  bool armed_ = false;
};

namespace {
#if defined(_WIN32)
std::atomic<ConsoleShutdown*> g_console_shutdown{nullptr};

BOOL WINAPI ConsoleControlHandler(DWORD event) {
  if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
  ConsoleShutdown* shutdown =
      g_console_shutdown.load(std::memory_order_acquire);
  if (shutdown == nullptr) return FALSE;
  shutdown->RequestStop();
  return TRUE;
}
#else
std::atomic<int> g_signal_write_fd{-1};
constexpr char kCancel = 0;

void SignalHandler(int signum) {
  const int fd = g_signal_write_fd.load(std::memory_order_acquire);
  if (fd < 0) return;
  const char byte = static_cast<char>(signum);
  const ssize_t ignored = write(fd, &byte, 1);
  (void)ignored;
}
#endif
}  // namespace

ConsoleShutdown::ConsoleShutdown(std::function<void()> stop,
                                 bool install_handlers)
    : impl_(std::make_unique<Impl>(std::move(stop))) {
  if (!install_handlers) return;
#if defined(_WIN32)
  ConsoleShutdown* expected = nullptr;
  if (!g_console_shutdown.compare_exchange_strong(expected, this,
                                                   std::memory_order_acq_rel) ||
      !SetConsoleCtrlHandler(ConsoleControlHandler, TRUE)) {
    if (expected == nullptr) {
      g_console_shutdown.store(nullptr, std::memory_order_release);
    }
    std::cerr << "server: could not install console control handler; shutdown "
                 "will not be graceful\n";
    return;
  }
  impl_->armed_ = true;
#else
  if (pipe(impl_->pipe_fds_) != 0) {
    std::cerr << "server: could not install signal handlers (pipe failed); "
                 "shutdown will not be graceful\n";
    return;
  }
  g_signal_write_fd.store(impl_->pipe_fds_[1], std::memory_order_release);
  const bool term_installed = std::signal(SIGTERM, SignalHandler) != SIG_ERR;
  const bool int_installed =
      term_installed && std::signal(SIGINT, SignalHandler) != SIG_ERR;
  impl_->armed_ = term_installed && int_installed;
  if (!impl_->armed_) {
    if (term_installed) std::signal(SIGTERM, SIG_DFL);
    if (int_installed) std::signal(SIGINT, SIG_DFL);
    g_signal_write_fd.store(-1, std::memory_order_release);
    return;
  }
  impl_->watcher_ = std::thread([this]() {
    char byte = 0;
    while (true) {
      const ssize_t bytes = read(impl_->pipe_fds_[0], &byte, 1);
      if (bytes == 1) {
        if (byte == kCancel) return;
        std::cerr << "server: shutting down on signal "
                  << static_cast<int>(byte) << "\n";
        RequestStop();
        return;
      }
      if (bytes < 0 && errno == EINTR) continue;
      return;
    }
  });
#endif
}

ConsoleShutdown::~ConsoleShutdown() {
#if defined(_WIN32)
  if (impl_->armed_) {
    SetConsoleCtrlHandler(ConsoleControlHandler, FALSE);
    ConsoleShutdown* expected = this;
    g_console_shutdown.compare_exchange_strong(expected, nullptr,
                                                std::memory_order_acq_rel);
  }
#else
  if (impl_->armed_) {
    std::signal(SIGTERM, SIG_DFL);
    std::signal(SIGINT, SIG_DFL);
    const char cancel = kCancel;
    const ssize_t ignored = write(impl_->pipe_fds_[1], &cancel, 1);
    (void)ignored;
  }
  if (impl_->watcher_.joinable()) impl_->watcher_.join();
  for (int fd : impl_->pipe_fds_) {
    if (fd >= 0) close(fd);
  }
  g_signal_write_fd.store(-1, std::memory_order_release);
#endif
}

void ConsoleShutdown::RequestStop() { impl_->RequestStop(); }

}  // namespace vllm::platform
