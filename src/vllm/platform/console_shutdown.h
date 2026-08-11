#ifndef VLLM_PLATFORM_CONSOLE_SHUTDOWN_H_
#define VLLM_PLATFORM_CONSOLE_SHUTDOWN_H_

#include <functional>
#include <memory>

namespace vllm::platform {

class ConsoleShutdown {
 public:
  explicit ConsoleShutdown(std::function<void()> stop,
                           bool install_handlers = true);
  ~ConsoleShutdown();

  ConsoleShutdown(const ConsoleShutdown&) = delete;
  ConsoleShutdown& operator=(const ConsoleShutdown&) = delete;

  // Request the same clean stop used by OS console/signal handlers. Multiple
  // concurrent requests invoke the callback exactly once.
  void RequestStop();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::platform

#endif  // VLLM_PLATFORM_CONSOLE_SHUTDOWN_H_
