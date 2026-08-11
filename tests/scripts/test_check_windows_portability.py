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
        #include "vllm/platform/process.h"
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
        struct WindowsHandlerRegistry {
          std::atomic<WindowsHandlerState*> published;
          std::atomic<unsigned> entrants;
        };
        WindowsHandlerState::~WindowsHandlerState() {
          if (quit_event) CloseHandle(quit_event);
          if (stop_event) CloseHandle(stop_event);
        }
        WindowsHandlerRegistry& HandlerRegistry() {
          static auto* registry = new WindowsHandlerRegistry();
          return *registry;
        }
        std::atomic<unsigned> in_flight;
        bool DispatchControlEvent(DWORD event) {
          auto& registry = HandlerRegistry();
          registry.entrants.fetch_add(1, std::memory_order_seq_cst);
          WindowsHandlerState* state =
              registry.published.load(std::memory_order_seq_cst);
          if (state != nullptr) {
            state->in_flight.fetch_add(1, std::memory_order_seq_cst);
          }
          registry.entrants.fetch_sub(1, std::memory_order_seq_cst);
          if (state == nullptr) return false;
          SetEvent(state->stop_event);
          state->in_flight.fetch_sub(1, std::memory_order_seq_cst);
          return true;
        }
        BOOL WINAPI ConsoleControlHandler(DWORD event) {
          return DispatchControlEvent(event);
        }
        SetConsoleCtrlHandler(ConsoleControlHandler, TRUE);
        registry.published.store(nullptr, std::memory_order_seq_cst);
        DrainEntrantsWithTimeout(registry.entrants);
        DrainInFlightWithTimeout(state->in_flight);
        if (!safe_to_close) RetireHandlerState(std::move(state));
        else state.reset();
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
        New-Item .cmake/api/v1/query/codemodel-v2
        cmake -S . -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
          -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
        python scripts/check-windows-portability.py --build-dir $BuildDir
        cmake --build $BuildDir --config Release --target server tests
        & "$BuildDir/tests/test_openai_api_server.exe"
        cmake --install $BuildDir --config Release
        dumpbin /directives library.lib
        dumpbin /imports bin/vllm-server.exe
        Invoke-CrtAudit
        Invoke-Checked $server @("--help")
        $env:VT_CPU_MATMUL_TIER = "portable"
        Invoke-Checked $tierTest @()
        $env:VT_CPU_MATMUL_TIER = "avx2"
        Invoke-Checked $tierTest @()
        Invoke-WebRequest "$BaseUrl/health"
        Invoke-WebRequest "$BaseUrl/version"
        CTRL_BREAK_EVENT
        $env:VT_CPU_MATMUL_TIER = "amx"
        Invoke-Checked python @($smokeHarness, $server)
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
        self.source_manifest = root / ".windows-portability-sources.json"
        self.source_manifest.write_text(
            json.dumps({"sources": sorted(files)}), encoding="utf-8"
        )
        return root

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_checker(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--test-source-manifest", str(self.source_manifest)],
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
            "registry.entrants.fetch_add(1, std::memory_order_seq_cst);",
            "registry.published.load(std::memory_order_seq_cst);",
            "state->in_flight.fetch_add(1, std::memory_order_seq_cst);",
            "SetEvent(state->stop_event);",
            "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);",
            "registry.published.store(nullptr, std::memory_order_seq_cst);",
            "DrainEntrantsWithTimeout(registry.entrants);",
            "DrainInFlightWithTimeout(state->in_flight);",
            "RetireHandlerState(std::move(state));",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace(mutation, ""),
                "stable event/in-flight",
            )

    def test_console_partial_event_creation_cleanup_is_structural(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        for mutation in (
            "CloseHandle(quit_event);",
            "CloseHandle(stop_event);",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace(mutation, ""),
                "partial event creation cleanup",
            )

    def test_console_handler_rejects_object_access_callback_and_blocking(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        for statement in (
            "impl_->RequestStop();",
            "stop_();",
            "std::mutex lock;",
            "std::condition_variable ready;",
        ):
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                source.replace("SetEvent(state->stop_event);",
                               f"SetEvent(state->stop_event); {statement}"),
                "OS handler may use only stable atomics and Win32 events",
            )

    def test_console_teardown_order_is_structural(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                "registry.published.store(nullptr, std::memory_order_seq_cst);\n"
                "DrainEntrantsWithTimeout(registry.entrants);",
                "DrainEntrantsWithTimeout(registry.entrants);\n"
                "registry.published.store(nullptr, std::memory_order_seq_cst);",
            ),
            "stable event/in-flight",
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                "DrainInFlightWithTimeout(state->in_flight);\n"
                "if (!safe_to_close) RetireHandlerState(std::move(state));\n"
                "else state.reset();",
                "else state.reset();\n"
                "if (!safe_to_close) RetireHandlerState(std::move(state));\n"
                "DrainInFlightWithTimeout(state->in_flight);",
            ),
            "stable event/in-flight",
        )

    def test_build_script_requires_live_gate_order(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                'Invoke-CrtAudit\nInvoke-Checked $server @("--help")',
                'Invoke-Checked $server @("--help")\nInvoke-CrtAudit',
            ),
            "native gate order",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                'Invoke-Checked $server @("--help")',
                'if ($false) { Invoke-Checked $server @("--help") }',
            ),
            "live --help smoke",
        )

    def test_real_codemodel_requires_reachable_sources_and_internal_headers(self) -> None:
        root = self.make_tree()
        process_header = root / "src/vllm/platform/process.h"
        process_header.parent.mkdir(parents=True, exist_ok=True)
        process_header.write_text("#include <unistd.h>\n", encoding="utf-8")
        cmake_contract = textwrap.dedent(SAFE_FILES["CMakeLists.txt"])
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(portability LANGUAGES CXX)\n" + cmake_contract +
            "\nadd_library(runtime STATIC\n"
            "  src/vllm/platform/process.cpp\n"
            "  src/vllm/platform/console_shutdown.cpp\n"
            "  src/vllm/v1/kv_offload/lmcache/remote_client.cpp\n"
            "  src/vllm/v1/kv_offload/fs_io.cpp)\n"
            "target_include_directories(runtime PRIVATE src include)\n"
            "add_executable(server src/vllm/entrypoints/openai/server_main.cpp)\n"
            "target_link_libraries(server PRIVATE runtime)\n",
            encoding="utf-8",
        )
        build = root / "build"
        query = build / ".cmake/api/v1/query"
        query.mkdir(parents=True)
        (query / "codemodel-v2").touch()
        configured = subprocess.run(
            ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(configured.returncode, 0,
                         configured.stdout + configured.stderr)
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--build-dir", str(build)],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("src/vllm/platform/process.h", result.stdout + result.stderr)

        cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        (root / "CMakeLists.txt").write_text(
            cmake.replace("  src/vllm/platform/process.cpp\n", ""),
            encoding="utf-8",
        )
        subprocess.run(
            ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja"],
            text=True, capture_output=True, check=True,
        )
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--build-dir", str(build)],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("required implementation is not reachable",
                      result.stdout + result.stderr)

    def test_representative_real_source_mutations_fail(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            console.replace(
                "registry.entrants.fetch_add(1, std::memory_order_seq_cst);",
                "",
                1,
            ),
            "stable event/in-flight",
        )

        process = (REPO / "src/vllm/platform/process.cpp").read_text(
            encoding="utf-8"
        )
        process_header = (REPO / "src/vllm/platform/process.h").read_text(
            encoding="utf-8"
        )
        self.assert_rejected(
            "src/vllm/platform/process.cpp",
            process.replace(
                '#include "vllm/platform/process.h"',
                '#include "vllm/platform/process.h"\n'
                '#include <unistd.h>\n'
                'void leaked_posix() { fork(); }',
                1,
            ),
            "unguarded POSIX",
        )
        root = self.make_tree({
            "src/vllm/platform/process.cpp": process,
            "src/vllm/platform/process.h": process_header +
            "\n#include <unistd.h>\nvoid leaked_posix() { fork(); }\n",
        })
        result = self.run_checker(root)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("src/vllm/platform/process.h", result.stdout + result.stderr)

    def test_rejects_disabled_smoke_and_missing_crt_audit(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace('Invoke-Checked $server @("--help")',
                           '# Invoke-Checked $server @("--help")'),
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
