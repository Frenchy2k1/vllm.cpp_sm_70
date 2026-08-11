#include "vllm/platform/console_shutdown.h"

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace vllm::platform {

#if defined(_WIN32)
namespace {
struct WindowsHandlerState {
  std::atomic<unsigned> in_flight{0};
  std::atomic<bool> pending{false};
  HANDLE stop_event = nullptr;
  HANDLE quit_event = nullptr;
  HANDLE drained_event = nullptr;
  HANDLE before_drain_test_event = nullptr;
};

std::atomic<WindowsHandlerState*> g_handler_state{nullptr};
std::atomic<unsigned> g_handler_acquisitions{0};

bool DispatchControlEvent(DWORD event, HANDLE acquired_event = nullptr,
                          HANDLE resume_event = nullptr) {
  if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return false;
  g_handler_acquisitions.fetch_add(1, std::memory_order_acq_rel);
  WindowsHandlerState* state =
      g_handler_state.load(std::memory_order_acquire);
  if (state != nullptr) {
    state->in_flight.fetch_add(1, std::memory_order_acq_rel);
  }
  g_handler_acquisitions.fetch_sub(1, std::memory_order_acq_rel);
  if (state == nullptr) return false;

  if (acquired_event != nullptr) SetEvent(acquired_event);
  if (resume_event != nullptr) WaitForSingleObject(resume_event, INFINITE);
  state->pending.store(true, std::memory_order_release);
  SetEvent(state->stop_event);
  if (state->in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    SetEvent(state->drained_event);
  }
  return true;
}
}  // namespace
#endif

class ConsoleShutdown::Impl {
 public:
  explicit Impl(std::function<void()> stop) : stop_(std::move(stop)) {
#if defined(_WIN32)
    win_state_ = std::make_unique<WindowsHandlerState>();
    win_state_->stop_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    win_state_->quit_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    win_state_->drained_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (win_state_->stop_event == nullptr || win_state_->quit_event == nullptr ||
        win_state_->drained_event == nullptr) {
      throw std::runtime_error("server: could not create console shutdown events");
    }
    worker_ = std::thread([this] {
      const HANDLE events[] = {win_state_->quit_event, win_state_->stop_event};
      while (true) {
        const DWORD result = WaitForMultipleObjects(2, events, FALSE, INFINITE);
        if (win_state_->pending.exchange(false, std::memory_order_acq_rel)) {
          RequestStop();
        }
        if (result == WAIT_OBJECT_0) return;
      }
    });
#endif
  }

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
#if defined(_WIN32)
  std::unique_ptr<WindowsHandlerState> win_state_;
  std::thread worker_;
#endif
  bool armed_ = false;
};

namespace {
#if defined(_WIN32)
BOOL WINAPI ConsoleControlHandler(DWORD event) {
  return DispatchControlEvent(event) ? TRUE : FALSE;
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
  WindowsHandlerState* expected = nullptr;
  if (!g_handler_state.compare_exchange_strong(expected, impl_->win_state_.get(),
                                                std::memory_order_acq_rel) ||
      !SetConsoleCtrlHandler(ConsoleControlHandler, TRUE)) {
    if (expected == nullptr) {
      g_handler_state.store(nullptr, std::memory_order_release);
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
    WindowsHandlerState* expected = impl_->win_state_.get();
    g_handler_state.compare_exchange_strong(expected, nullptr,
                                             std::memory_order_acq_rel);
    while (g_handler_acquisitions.load(std::memory_order_acquire) != 0) {
      SwitchToThread();
    }
    if (impl_->win_state_->before_drain_test_event != nullptr) {
      SetEvent(impl_->win_state_->before_drain_test_event);
    }
    while (impl_->win_state_->in_flight.load(std::memory_order_acquire) != 0) {
      WaitForSingleObject(impl_->win_state_->drained_event, INFINITE);
    }
  }
  SetEvent(impl_->win_state_->quit_event);
  if (impl_->worker_.joinable()) impl_->worker_.join();
  CloseHandle(impl_->win_state_->drained_event);
  CloseHandle(impl_->win_state_->quit_event);
  CloseHandle(impl_->win_state_->stop_event);
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

#if defined(_WIN32)
void ConsoleShutdown::SetBeforeDrainEventForTest(void* event) {
  impl_->win_state_->before_drain_test_event = static_cast<HANDLE>(event);
}

bool ConsoleShutdown::DispatchControlEventForTest(unsigned long event,
                                                  void* acquired_event,
                                                  void* resume_event) {
  return DispatchControlEvent(static_cast<DWORD>(event),
                              static_cast<HANDLE>(acquired_event),
                              static_cast<HANDLE>(resume_event));
}
#endif

}  // namespace vllm::platform
