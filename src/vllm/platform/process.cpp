#include "vllm/platform/process.h"

#include <stdexcept>
#include <string_view>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace vllm::platform {
namespace {

std::wstring QuoteWindowsArg(std::wstring_view arg) {
  std::wstring quoted;
  quoted.push_back(L'"');
  std::size_t backslashes = 0;
  for (wchar_t ch : arg) {
    if (ch == L'\\') {
      ++backslashes;
      continue;
    }
    if (ch == L'"') {
      quoted.append(backslashes * 2 + 1, L'\\');
      quoted.push_back(L'"');
    } else {
      quoted.append(backslashes, L'\\');
      quoted.push_back(ch);
    }
    backslashes = 0;
  }
  quoted.append(backslashes * 2, L'\\');
  quoted.push_back(L'"');
  return quoted;
}

#if defined(_WIN32)
std::wstring Utf8ToWide(std::string_view value) {
  if (value.empty()) return {};
  const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                       value.data(),
                                       static_cast<int>(value.size()), nullptr,
                                       0);
  if (size <= 0) {
    throw std::runtime_error("process argv is not valid UTF-8");
  }
  std::wstring wide(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), wide.data(), size) !=
      size) {
    throw std::runtime_error("could not convert process argv to UTF-16");
  }
  return wide;
}
#endif

}  // namespace

std::wstring BuildWindowsCommandLine(const std::vector<std::wstring>& args) {
  std::wstring command_line;
  for (const std::wstring& arg : args) {
    if (!command_line.empty()) command_line.push_back(L' ');
    command_line.append(QuoteWindowsArg(arg));
  }
  return command_line;
}

int RunProcessArgv(const std::vector<std::string>& args) {
  if (args.empty() || args.front().empty()) {
    throw std::invalid_argument("process argv must name an executable");
  }
#if defined(_WIN32)
  std::vector<std::wstring> wide_args;
  wide_args.reserve(args.size());
  for (const std::string& arg : args) wide_args.push_back(Utf8ToWide(arg));
  std::wstring command_line = BuildWindowsCommandLine(wide_args);
  command_line.push_back(L'\0');

  STARTUPINFOW startup{};
  startup.cb = static_cast<DWORD>(sizeof(startup));
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(wide_args.front().c_str(), command_line.data(), nullptr,
                      nullptr, FALSE, CREATE_UNICODE_ENVIRONMENT, nullptr,
                      nullptr, &startup, &process)) {
    throw std::runtime_error("CreateProcessW failed with error " +
                             std::to_string(GetLastError()));
  }
  const auto close_handles = [&]() {
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
  };
  if (WaitForSingleObject(process.hProcess, INFINITE) != WAIT_OBJECT_0) {
    const DWORD error = GetLastError();
    close_handles();
    throw std::runtime_error("waiting for child failed with error " +
                             std::to_string(error));
  }
  DWORD exit_code = 0;
  if (!GetExitCodeProcess(process.hProcess, &exit_code)) {
    const DWORD error = GetLastError();
    close_handles();
    throw std::runtime_error("reading child exit code failed with error " +
                             std::to_string(error));
  }
  close_handles();
  return static_cast<int>(exit_code);
#else
  std::vector<char*> native_args;
  native_args.reserve(args.size() + 1);
  for (const std::string& arg : args) {
    native_args.push_back(const_cast<char*>(arg.c_str()));
  }
  native_args.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error(std::string("fork failed: ") +
                             std::strerror(errno));
  }
  if (pid == 0) {
    execvp(native_args[0], native_args.data());
    _exit(127);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) continue;
    throw std::runtime_error(std::string("waitpid failed: ") +
                             std::strerror(errno));
  }
  if (WIFSIGNALED(status)) {
    throw std::runtime_error("child died on signal " +
                             std::to_string(WTERMSIG(status)));
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

}  // namespace vllm::platform
