#!/usr/bin/env python3
"""Behavior tests for the fail-closed Windows portability checker."""

from __future__ import annotations

import json
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
            "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mf16c>")
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
        std::atomic<unsigned> handler_acquisitions;
        std::atomic<unsigned> in_flight;
        bool DispatchControlEvent(DWORD event) {
          handler_acquisitions.fetch_add(1); in_flight.fetch_add(1);
          SetEvent(stop_event); return true;
        }
        BOOL WINAPI ConsoleControlHandler(DWORD event) {
          return DispatchControlEvent(event);
        }
        SetConsoleCtrlHandler(ConsoleControlHandler, TRUE);
        SetEvent(stop_event);
        WaitForSingleObject(impl_->win_state_->drained_event, INFINITE);
        #else
        #include <unistd.h>
        pipe(fds); read(fds[0], data, 1); write(fds[1], data, 1); close(fds[0]);
        #endif
    """,
    "src/vllm/v1/kv_offload/lmcache/remote_client.cpp": """
        #ifdef _WIN32
        WSAStartup(MAKEWORD(2, 2), &data);
        WSAGetLastError(); closesocket(socket);
        if (recv_result == 0) { Close(); throw peer_closed; }
        #else
        #include <sys/socket.h>
        close(socket);
        #endif
    """,
    "src/vllm/v1/kv_offload/fs_io.cpp": """
        #ifdef _WIN32
        #define NOMINMAX
        CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr); CREATE_NEW;
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
        dumpbin /directives library.lib
        dumpbin /imports bin/vllm-server.exe
        VCRUNTIME MSVCP CONCRT UCRTBASE api-ms-win-crt- MSVCR LIBCMT
    """,
    "src/vt/cpu/cpu_matmul_elem.cpp": """
        void PortableDispatcher();
    """,
    "src/vllm/model_executor/models/minimax_h3_sharded.cpp": """
        #include <filesystem>
        bool ok = std::filesystem::is_directory(path) ||
                  std::filesystem::is_regular_file(path);
    """,
    "include/vllm/model_executor/models/device_pool.h": """
        #include <bit>
        auto width = std::bit_width(bytes);
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
        (root / ".windows-portability-sources.json").write_text(
            json.dumps({"sources": sorted(files)}), encoding="utf-8"
        )
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
                '    "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mf16c>")',
                "",
            ),
            "dedicated F16C translation unit",
        )

    def test_f16c_tu_accepts_avx_but_rejects_avx2(self) -> None:
        avx = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        result = self.run_checker(self.make_tree({"CMakeLists.txt": avx}))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "CMakeLists.txt",
            avx.replace("/arch:AVX>", "/arch:AVX2>"),
            "F16C translation unit must not require AVX2",
        )

    def test_rejects_dynamic_msvc_runtime(self) -> None:
        self.assert_rejected(
            "CMakeLists.txt",
            textwrap.dedent(SAFE_FILES["CMakeLists.txt"]).replace(
                '"MultiThreaded$<$<CONFIG:Debug>:Debug>"',
                '"MultiThreadedDLL"',
            ),
            "exact static MSVC runtime",
        )

    def test_scans_all_shipped_server_sources(self) -> None:
        self.assert_rejected(
            "src/vllm/model_executor/models/minimax_h3_sharded.cpp",
            "#include <sys/stat.h>\nstruct stat st; stat(path, &st);\n",
            "unguarded POSIX",
        )
        self.assert_rejected(
            "include/vllm/model_executor/models/device_pool.h",
            "auto n = __builtin_clzll(bytes);\n",
            "non-portable compiler builtin",
        )

    def test_pins_windows_file_publish_and_header_contracts(self) -> None:
        fs = textwrap.dedent(SAFE_FILES["src/vllm/v1/kv_offload/fs_io.cpp"])
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            fs.replace("#define NOMINMAX\n", ""),
            "NOMINMAX",
        )
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            fs.replace("CREATE_NEW", "CREATE_ALWAYS"),
            "CREATE_NEW",
        )
        self.assert_rejected(
            "src/vllm/v1/kv_offload/fs_io.cpp",
            fs.replace("MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH",
                       "MOVEFILE_REPLACE_EXISTING"),
            "MOVEFILE_WRITE_THROUGH",
        )

    def test_pins_peer_close_invalidation(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/v1/kv_offload/lmcache/remote_client.cpp"]
        )
        result = self.run_checker(self.make_tree({
            "src/vllm/v1/kv_offload/lmcache/remote_client.cpp": source,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
            source.replace("Close(); throw peer_closed", "throw peer_closed"),
            "peer-close must invalidate",
        )

    def test_console_handler_uses_stable_event_and_drains_inflight(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        for mutation in (
            "std::atomic<unsigned> handler_acquisitions;",
            "std::atomic<unsigned> in_flight;",
            "SetEvent(stop_event);",
            "WaitForSingleObject(impl_->win_state_->drained_event, INFINITE);",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace(mutation, ""),
                "stable event/in-flight",
            )

    def test_rejects_disabled_smoke_and_missing_crt_audit(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace('& "$StageDir/bin/vllm-server.exe" --help',
                           '# & "$StageDir/bin/vllm-server.exe" --help'),
            "live --help smoke",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace("dumpbin /imports bin/vllm-server.exe", ""),
            "PE CRT import audit",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace("UCRTBASE", "UNRELATED_SYSTEM_DLL"),
            "dynamic CRT rejection",
        )


if __name__ == "__main__":
    unittest.main()
