#!/usr/bin/env python3
"""Behavior tests for the fail-closed Windows portability checker."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "scripts" / "check-windows-portability.py"


SAFE_FILES = {
    "CMakeLists.txt": """
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        if(MSVC)
          add_compile_options(/fp:strict /W4 /WX)
        else()
          add_compile_options(-ffp-contract=off)
        endif()
        set_source_files_properties(src/vt/cpu/cpu_matmul_elem_avx2.cpp
          PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>")
        set_source_files_properties(src/vt/cpu/cpu_matmul_elem_f16c.cpp
          PROPERTIES COMPILE_OPTIONS
            "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mf16c>")
    """,
    "src/vllm/entrypoints/openai/server_main.cpp": """
        int server_main_contract;
    """,
    "src/vllm/platform/process.cpp": """
        #ifdef _WIN32
        CreateProcessW(nullptr, command_line.data(), nullptr, nullptr, FALSE,
                       0, nullptr, nullptr, &startup, &process);
        #else
        #include <sys/wait.h>
        fork(); execvp(argv[0], argv); waitpid(pid, &status, 0);
        #endif
    """,
    "src/vllm/platform/console_shutdown.cpp": """
        #ifdef _WIN32
        SetConsoleCtrlHandler(ConsoleControlHandler, TRUE);
        #else
        #include <unistd.h>
        pipe(fds); read(fds[0], data, 1); write(fds[1], data, 1); close(fds[0]);
        #endif
    """,
    "src/vllm/v1/kv_offload/lmcache/remote_client.cpp": """
        #ifdef _WIN32
        WSAStartup(MAKEWORD(2, 2), &data);
        WSAGetLastError(); closesocket(socket);
        #else
        #include <sys/socket.h>
        close(socket);
        #endif
    """,
    "src/vllm/v1/kv_offload/fs_io.cpp": """
        #ifdef _WIN32
        CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
        FlushFileBuffers(file);
        MoveFileExW(source.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        #else
        #include <unistd.h>
        open(path, O_RDONLY); read(fd, data, size); close(fd);
        #endif
    """,
    "scripts/build-windows-release.ps1": """
        cmake -S . -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
          -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        cmake --build $BuildDir --config Release --target install
        & "$StageDir/bin/vllm-server.exe" --help
        Invoke-WebRequest "$BaseUrl/health"
        Invoke-WebRequest "$BaseUrl/version"
        CTRL_BREAK_EVENT
        $env:VT_CPU_MATMUL_TIER = "portable"
        & "$StageDir/bin/vllm-server.exe" --help
        $env:VT_CPU_MATMUL_TIER = "avx2"
        & "$StageDir/bin/vllm-server.exe" --help
        $env:VT_CPU_MATMUL_TIER = "amx"
        & "$StageDir/bin/vllm-server.exe" --help
    """,
    "src/vt/cpu/cpu_matmul_elem.cpp": """
        void PortableDispatcher();
    """,
}


class WindowsPortabilityCheckerTest(unittest.TestCase):
    def make_tree(self, mutations: dict[str, str] | None = None) -> Path:
        root = Path(self.tempdir.name)
        files = dict(SAFE_FILES)
        files.update(mutations or {})
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content), encoding="utf-8")
        return root

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_checker(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root)],
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_rejected(self, relative: str, content: str, reason: str) -> None:
        result = self.run_checker(self.make_tree({relative: content}))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(reason, result.stdout + result.stderr)

    def test_accepts_complete_guarded_contract(self) -> None:
        result = self.run_checker(self.make_tree())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_unguarded_posix_surface(self) -> None:
        self.assert_rejected(
            "src/vllm/entrypoints/openai/server_main.cpp",
            "#include <sys/wait.h>\nfork();\n",
            "unguarded POSIX",
        )

    def test_rejects_global_avx2_and_missing_static_runtime(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            "add_compile_options(/arch:AVX2)\n",
            "static MSVC runtime",
        )
        self.assert_rejected(
            "CMakeLists.txt",
            'set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")\n'
            "add_compile_options(/arch:AVX2)\n",
            "global /arch:AVX2",
        )

    def test_rejects_lossy_windows_paths(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            "#ifdef _WIN32\nCreateFileA(path.string().c_str(), 0, 0, nullptr, 0, 0, nullptr);\n#endif\n",
            "lossy Windows path",
        )

    def test_rejects_shell_process_launch(self) -> None:
        self.assert_rejected(
            "src/vllm/entrypoints/openai/server_main.cpp",
            "#ifdef _WIN32\nsystem(command.c_str());\n#endif\n",
            "shell invocation",
        )

    def test_rejects_missing_winsock_support(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
            "#ifdef _WIN32\nthrow unsupported;\n#else\n#include <sys/socket.h>\n#endif\n",
            "LMCache Winsock support",
        )

    def test_rejects_missing_windows_kv_filesystem_support(self) -> None:
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            "#ifdef _WIN32\nthrow unsupported;\n#else\n#include <unistd.h>\n#endif\n",
            "Windows KV filesystem support",
        )

    def test_rejects_function_target_attribute_in_portable_cpu_tu(self) -> None:
        self.assert_rejected(
            "src/vt/cpu/cpu_matmul_elem.cpp",
            '__attribute__((target("f16c"))) void leaked_baseline();\n',
            "F16C must be isolated",
        )

    def test_rejects_missing_compiler_specific_f16c_flags(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            textwrap.dedent(SAFE_FILES["CMakeLists.txt"]).replace(
                'set_source_files_properties(src/vt/cpu/cpu_matmul_elem_f16c.cpp\n'
                '  PROPERTIES COMPILE_OPTIONS\n'
                '    "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mf16c>")',
                "",
            ),
            "dedicated F16C translation unit",
        )


if __name__ == "__main__":
    unittest.main()
