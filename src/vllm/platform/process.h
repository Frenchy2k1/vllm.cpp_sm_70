#ifndef VLLM_PLATFORM_PROCESS_H_
#define VLLM_PLATFORM_PROCESS_H_

#include <string>
#include <vector>

namespace vllm::platform {

// Encode argv using the quoting rules consumed by CommandLineToArgvW. Every
// argument is quoted so whitespace, quotes, empty strings, and trailing
// backslashes survive the CreateProcessW command-line boundary exactly.
std::wstring BuildWindowsCommandLine(const std::vector<std::wstring>& args);

// Execute one program directly from argv, wait, and return its exit code.
// This never invokes a shell.
int RunProcessArgv(const std::vector<std::string>& args);

}  // namespace vllm::platform

#endif  // VLLM_PLATFORM_PROCESS_H_
