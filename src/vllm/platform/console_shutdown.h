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

#if defined(_WIN32)
  // Deterministic native test seam: pause a simulated handler after it has
  // acquired stable state, and observe teardown immediately before it drains.
  void SetBeforeDrainEventForTest(void* event);
  static bool DispatchControlEventForTest(unsigned long event,
                                          void* acquired_event,
                                          void* resume_event);
#endif

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::platform

#endif  // VLLM_PLATFORM_CONSOLE_SHUTDOWN_H_
