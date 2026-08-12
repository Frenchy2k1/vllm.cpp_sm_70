#!/usr/bin/env python3
"""Behavior tests for the fail-closed Windows portability checker."""

from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "scripts" / "check-windows-portability.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("windows_portability", CHECKER)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(checker)
UNSUPPORTED_TIER_FILTER = (
    "--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran"
)


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
        bool DrainEntrantsWithTimeout(std::atomic<unsigned>& entrants) {
          const ULONGLONG start = GetTickCount64();
          while (entrants.load(std::memory_order_seq_cst) != 0) {
            if (GetTickCount64() - start >= 5000) return false;
            SwitchToThread();
          }
          return true;
        }
        bool DrainInFlightWithTimeout(std::atomic<unsigned>& in_flight) {
          const ULONGLONG start = GetTickCount64();
          while (in_flight.load(std::memory_order_seq_cst) != 0) {
            if (GetTickCount64() - start >= 5000) return false;
            SwitchToThread();
          }
          return true;
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
          const bool resumed = true;
          SetEvent(state->stop_event);
          state->in_flight.fetch_sub(1, std::memory_order_seq_cst);
          return resumed;
        }
        BOOL WINAPI ConsoleControlHandler(DWORD event) {
          return DispatchControlEvent(event);
        }
        SetConsoleCtrlHandler(ConsoleControlHandler, TRUE);
        ConsoleShutdown::~ConsoleShutdown() {
        registry.published.store(nullptr, std::memory_order_seq_cst);
        DrainEntrantsWithTimeout(registry.entrants);
        DrainInFlightWithTimeout(state->in_flight);
        if (!safe_to_close) RetireHandlerState(std::move(state));
        else state.reset();
        }
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
        function Invoke-UnsupportedTierProbe {
          param([Parameter(Mandatory)][string]$TierTest,
                [scriptblock]$Runner)
          $arguments = @(
            '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran'
          )
          if ($null -eq $Runner) {
            $probeOutput = @(& $TierTest @arguments 2>&1)
            $probeExitCode = $LASTEXITCODE
          } else {
            $probeResult = & $Runner $TierTest $arguments
            $probeOutput = @($probeResult.Output)
            $probeExitCode = [int]$probeResult.ExitCode
          }
          if ($probeExitCode -ne 1) { throw "unexpected exit" }
          $diagnostic = $probeOutput -join "`n"
          if ($diagnostic -notmatch
              [regex]::Escape("unknown x86 ISA tier 'amx'")) {
            throw "missing diagnostic"
          }
        }
        function Invoke-UnsupportedTierContractTests {
          $calls = [System.Collections.Generic.List[object]]::new()
          $good = {
            param([string]$Program, [string[]]$Arguments)
            $calls.Add([pscustomobject]@{ Arguments = @($Arguments) })
            [pscustomobject]@{
              ExitCode = 1
              Output = @("unknown x86 ISA tier 'amx'")
            }
          }.GetNewClosure()
          Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" -Runner $good
          if ($calls.Count -ne 1) { throw "wrong fake call count" }
          if ($calls[0].Arguments.Count -ne 1) { throw "wrong fake argument count" }
          if ($calls[0].Arguments[0] -ne '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran') {
            throw "wrong fake argument"
          }
          $badResults = @(
            [pscustomobject]@{ ExitCode = 0 },
            [pscustomobject]@{ ExitCode = 134 },
            [pscustomobject]@{ ExitCode = -1073741819 },
            [pscustomobject]@{ ExitCode = 3 },
            [pscustomobject]@{ ExitCode = 2 }
          )
        }
        if ($ContractTest) {
          Invoke-CrtContractTests
          Invoke-UnsupportedTierContractTests
        }
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
        $savedTier = $env:VT_CPU_MATMUL_TIER
        try {
          $env:VT_CPU_MATMUL_TIER = "portable"
          Invoke-Checked $tierTest @()
          $env:VT_CPU_MATMUL_TIER = "avx2"
          Invoke-Checked $tierTest @()
          $env:VT_CPU_MATMUL_TIER = "amx"
          Invoke-UnsupportedTierProbe -TierTest $tierTest
        } finally {
          $env:VT_CPU_MATMUL_TIER = $savedTier
        }
        Invoke-WebRequest "$BaseUrl/health"
        Invoke-WebRequest "$BaseUrl/version"
        CTRL_BREAK_EVENT
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

    def test_real_tree_declares_strict_msvc_platform_contract(self) -> None:
        cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
        warnings = (REPO / "cmake/CompilerWarnings.cmake").read_text(
            encoding="utf-8"
        )
        contract = cmake + "\n" + warnings
        for token in (
            "NOMINMAX",
            "_CRT_SECURE_NO_WARNINGS",
            "/utf-8",
            "/W4",
            "/WX",
        ):
            with self.subTest(token=token):
                self.assertIn(token, contract)

    def test_real_tree_uses_portable_windows_allocation_and_math(self) -> None:
        backend = (REPO / "src/vt/cpu/cpu_backend.cpp").read_text(
            encoding="utf-8"
        )
        for token in ("_aligned_malloc", "_aligned_free", "std::aligned_alloc"):
            with self.subTest(token=token):
                self.assertIn(token, backend)

        offenders = []
        for root in (REPO / "src", REPO / "tests"):
            for path in root.rglob("*"):
                if path.suffix in {".cc", ".cpp", ".h", ".hpp"}:
                    if re.search(r"\bM_PI\b", path.read_text(encoding="utf-8")):
                        offenders.append(path.relative_to(REPO).as_posix())
        self.assertEqual(offenders, [])

    def test_real_tree_closes_observed_msvc_source_warnings(self) -> None:
        forbidden = {
            "src/vt/cpu/cpu_paged_attn.cpp": "constexpr KvKind kKV",
            "src/vllm/v1/worker/gpu/runner.cpp": (
                "std::vector<int32_t> positions(step.positions.begin()"
            ),
            "src/vllm/model_executor/models/gemma4_audio.cpp": (
                "std::max<double>(nts - 1, 1)"
            ),
            "src/vllm/v1/kv_offload/lmcache/lmcache_connector.cpp": (
                "conn_cfg.num_layers = ctx.identity->num_hidden_layers"
            ),
            "src/vllm/v1/core/kv_cache_utils.cpp": "MemAvailable: %ld kB",
            "src/vllm/model_executor/models/minimax_h3_video_vae.cpp": (
                "float* v = target->data()"
            ),
        }
        for relative, token in forbidden.items():
            with self.subTest(relative=relative):
                source = (REPO / relative).read_text(encoding="utf-8")
                self.assertNotIn(token, source)

        threadpool = (REPO / "src/vt/cpu/cpu_threadpool.h").read_text(
            encoding="utf-8"
        )
        for token in (
            "#pragma warning(push)",
            "#pragma warning(disable : 4324)",
            "#pragma warning(pop)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, threadpool)

    def test_posix_cache_source_requires_exact_not_win32_cmake_guard(self) -> None:
        source = "src/vt/cuda/nvfp4_persistent_cache.cpp"
        for condition, expected in (("NOT WIN32", {source}), ("WIN32", set()), ("NOT APPLE", set())):
            root = self.make_tree({
                "CMakeLists.txt": textwrap.dedent(SAFE_FILES["CMakeLists.txt"]) +
                f"\nif({condition})\n  target_sources(vllm PRIVATE {source})\nendif()\n",
                source: "#include <unistd.h>\nvoid f() { close(1); }\n",
            })
            with self.subTest(condition=condition):
                self.assertEqual(checker.windows_excluded_sources(root), expected)

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
        for source in (
            "#ifdef _WIN32\nsystem(command.c_str());\n#endif\n",
            '#ifdef _WIN32\n'
            'CreateProcessW(L"cmd.exe", nullptr, nullptr);\n'
            '#endif\n',
            '#ifdef _WIN32\n'
            'CreateProcessW(L"C:\\\\Windows\\\\System32\\\\cmd.exe", '
            'nullptr, nullptr);\n'
            '#endif\n',
        ):
            with self.subTest(source=source):
                self.assert_rejected(
                    "src/vllm/entrypoints/openai/server_main.cpp",
                    source,
                    "shell invocation",
                )

    def test_accepts_cpu_and_vulkan_command_buffer_identifiers(self) -> None:
        closures = {
            "cpu": "void RecordCpuCommandBuffer(int command_buffer) {}\n",
            "vulkan": """
                void RecordVulkanCommands(VkCommandBuffer cmd) {
                  // A cmd.exe shell is not launched by a command-buffer variable.
                  const char* note = "the cmd command-buffer variable is inert prose";
                  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                }
            """,
        }
        for backend, source in closures.items():
            with self.subTest(backend=backend):
                result = self.run_checker(self.make_tree({
                    f"src/vt/{backend}/portability_audit_fixture.cpp": source,
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_powershell_ast_path_is_opaque_for_cpu_and_vulkan(self) -> None:
        ast_commands = [{
            "text": "\n".join((
                "Invoke-CrtAudit",
                'Invoke-Checked $server @("--help")',
                "Invoke-UnsupportedTierProbe -TierTest $tierTest",
                "Invoke-Checked python @($smokeHarness)",
                "Invoke-UnsupportedTierContractTests",
            )),
            "offset": 0,
        }]
        path_env = "VLLM_CPP_POWERSHELL_AUDIT_PATH"

        for backend, windows_root in (
            ("cpu", r"C:\actions\vllm.cpp CPU checkout"),
            ("vulkan", r"D:\a\vllm.cpp Vulkan checkout"),
        ):
            with self.subTest(backend=backend):
                root = Path(self.tempdir.name) / windows_root
                script = root / "scripts" / "build-windows-release.ps1"
                calls: list[tuple[list[str], dict[str, object]]] = []

                def fake_run(
                    command: list[str], **kwargs: object
                ) -> subprocess.CompletedProcess[str]:
                    calls.append((command, kwargs))
                    command_index = (
                        command.index("-Command") if "-Command" in command
                        else -1
                    )
                    if command_index >= 0 and command_index != len(command) - 2:
                        return subprocess.CompletedProcess(
                            command,
                            1,
                            "",
                            "The file could not be read: Cannot process argument "
                            "because the value of argument path is not valid.",
                        )
                    if command_index >= 0:
                        return subprocess.CompletedProcess(
                            command, 0, json.dumps(ast_commands), ""
                        )
                    return subprocess.CompletedProcess(command, 0, "", "")

                errors: list[str] = []
                with mock.patch.object(
                    checker.shutil,
                    "which",
                    return_value=r"C:\Program Files\PowerShell\7\pwsh.exe",
                ), mock.patch.object(
                    checker.subprocess, "run", side_effect=fake_run
                ), mock.patch.object(
                    checker, "_validate_powershell_ast_order"
                ):
                    checker._validate_powershell_ast(script, errors)

                self.assertEqual(errors, [])
                self.assertEqual(len(calls), 2)
                ast_argv, ast_kwargs = calls[0]
                self.assertEqual(ast_argv.index("-Command"), len(ast_argv) - 2)
                self.assertNotIn("-Path", ast_argv)
                self.assertEqual(ast_kwargs["env"][path_env], str(script))
                self.assertFalse(ast_kwargs.get("shell", False))
                contract_argv, contract_kwargs = calls[1]
                self.assertEqual(
                    contract_argv[contract_argv.index("-File") + 1],
                    str(script),
                )
                self.assertEqual(
                    contract_argv[contract_argv.index("-SourceDir") + 1],
                    str(root),
                )
                self.assertFalse(contract_kwargs.get("shell", False))

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

    def test_console_function_body_skips_prototypes_and_decoys(self) -> None:
        source = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        declarations = {
            "bool DrainEntrantsWithTimeout(std::atomic<unsigned>& entrants) {":
                "bool DrainEntrantsWithTimeout(std::atomic<unsigned>& entrants);\n"
                "bool EntrantsDeclarationDecoy() { return false; }\n",
            "bool DrainInFlightWithTimeout(std::atomic<unsigned>& in_flight) {":
                "bool DrainInFlightWithTimeout(std::atomic<unsigned>& in_flight);\n"
                "bool InFlightDeclarationDecoy() { return false; }\n",
            "bool DispatchControlEvent(DWORD event, HANDLE acquired_event = nullptr,":
                "bool DispatchControlEvent(DWORD event, HANDLE acquired_event,\n"
                "                          HANDLE resume_event);\n"
                "bool DispatchDeclarationDecoy() { return false; }\n",
        }
        for definition, prefix in declarations.items():
            with self.subTest(definition=definition):
                self.assertEqual(source.count(definition), 1)
                mutated = source.replace(definition, prefix + definition, 1)
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp": mutated,
                }))
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)

    def test_console_final_tail_requires_bound_local_resumed(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace("return resumed;", "return true;", 1),
            "final in-flight decrement",
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                "bool DispatchControlEvent(DWORD event) {",
                "bool resumed = true;\n"
                "bool DispatchControlEvent(DWORD event) {",
                1,
            ).replace("const bool resumed = true;\n", "", 1),
            "final in-flight decrement",
        )

    def test_console_final_tail_rejects_object_and_function_macros(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        mutations = (
            (
                "#define POST_DECREMENT_RESULT "
                "(SetEvent(state->stop_event), resumed)\n  " + decrement,
                "return POST_DECREMENT_RESULT;",
                "final in-flight decrement",
            ),
            (
                "#define resumed (SetEvent(state->stop_event), true)\n  " +
                decrement,
                "return resumed;",
                "trusted final-tail token",
            ),
            (
                "#define POST_DECREMENT_RESULT() "
                "(SetEvent(state->stop_event), resumed)\n  " + decrement,
                "return POST_DECREMENT_RESULT();",
                "final in-flight decrement",
            ),
        )
        for macro_and_decrement, returned, reason in mutations:
            with self.subTest(returned=returned):
                mutated = source.replace(decrement, macro_and_decrement, 1)
                mutated = mutated.replace("return resumed;", returned, 1)
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    reason,
                )

    def test_console_final_tail_rejects_macros_for_trusted_tokens(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        mutations = (
            "#define return SetEvent(state->stop_event); return\n  ",
            "#define return(value) SetEvent(state->stop_event); return value\n  ",
            "#define resumed \\\n  (SetEvent(state->stop_event), true)\n  ",
            "#define fetch_sub(value, order) \\\n  (SetEvent(state->stop_event), 0)\n  ",
        )
        for directive in mutations:
            with self.subTest(directive=directive):
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    source.replace(decrement, directive + decrement, 1) +
                    "\n#undef return\n#undef resumed\n#undef fetch_sub\n",
                    "trusted final-tail token",
                )

    def test_console_final_tail_rejects_split_keyword_macro(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        split_keyword = (
            "#define \\\n"
            "return SetEvent(state->stop_event); return\n  "
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(decrement, split_keyword + decrement, 1),
            "trusted final-tail token",
        )

    def test_console_final_tail_macro_state_is_checked_at_each_token(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        mutations = (
            "#define return SetEvent(state->stop_event); return",
            "#define return(value) SetEvent(state->stop_event); return value",
        )
        for directive in mutations:
            with self.subTest(directive=directive):
                mutated = source.replace(
                    decrement,
                    decrement + "\n  " + directive,
                    1,
                )
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    "trusted final-tail token",
                )

    def test_console_trusted_tails_restore_pushed_macros(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        cases = (
            (
                timeout,
                '#define false (SwitchToThread(), false)\n'
                '#pragma push_macro("false")\n'
                '#undef false\n'
                '#pragma pop_macro("false")\n' + timeout,
                "DrainEntrantsWithTimeout trusted return-tail token",
                0,
            ),
            (
                timeout,
                '#define GetTickCount64() (SwitchToThread(), GetTickCount64())\n'
                '#if UNKNOWN_WINDOWS_BUILD_FLAG\n'
                '#pragma push_macro( \\\n'
                '                    "GetTickCount64" )\n'
                '#undef GetTickCount64\n'
                '#pragma pop_macro("GetTickCount64")\n'
                '#endif\n' + timeout,
                "DrainInFlightWithTimeout trusted return-tail token",
                1,
            ),
            (
                decrement,
                '#define fetch_sub(value, order) '
                '(SetEvent(state->stop_event), 0)\n'
                '#pragma push_macro("fetch_sub")\n'
                '#undef fetch_sub\n'
                '#pragma pop_macro("fetch_sub")\n  ' + decrement,
                "trusted final-tail token",
                0,
            ),
            (
                decrement,
                decrement + '\n  '
                '#define return SetEvent(state->stop_event); return\n'
                '#pragma push_macro("return")\n'
                '#undef return\n'
                '#pragma pop_macro("return")',
                "trusted final-tail token",
                0,
            ),
        )
        for old, replacement, reason, occurrence in cases:
            with self.subTest(reason=reason, occurrence=occurrence):
                position = -1
                for _ in range(occurrence + 1):
                    position = source.find(old, position + 1)
                self.assertGreaterEqual(position, 0)
                mutated = source[:position] + replacement + source[position + len(old):]
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp", mutated, reason
                )

    def test_console_trusted_tails_accept_push_pop_restoring_undefined(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        safe = (
            '#pragma push_macro("false")\n'
            '#define false unsafe_false\n'
            '#pragma pop_macro("false")\n' + timeout
        )
        for occurrence in (0, 1):
            with self.subTest(occurrence=occurrence):
                position = -1
                for _ in range(occurrence + 1):
                    position = source.find(timeout, position + 1)
                self.assertGreaterEqual(position, 0)
                mutated = source[:position] + safe + source[position + len(timeout):]
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp": mutated,
                }))
                self.assertEqual(result.returncode, 0,
                                 result.stdout + result.stderr)

    def test_console_trusted_tails_honor_phase_two_comment_splicing(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        cases = (
            (
                timeout,
                "#define false (SwitchToThread(), false)\n"
                "// the physical splice swallows the undef \\\n"
                "#undef false\n" + timeout,
                "DrainEntrantsWithTimeout trusted return-tail token",
                0,
            ),
            (
                timeout,
                "#define return SwitchToThread(); return\n"
                "// the physical splice swallows the undef \\\n"
                "#undef return\n" + timeout,
                "DrainInFlightWithTimeout trusted return-tail token",
                1,
            ),
            (
                decrement,
                "#define return SetEvent(state->stop_event); return\n"
                "// the physical splice swallows the undef \\\n"
                "#undef return\n  " + decrement,
                "trusted final-tail token",
                0,
            ),
        )
        for old, replacement, reason, occurrence in cases:
            with self.subTest(reason=reason, occurrence=occurrence):
                position = -1
                for _ in range(occurrence + 1):
                    position = source.find(old, position + 1)
                self.assertGreaterEqual(position, 0)
                mutated = source[:position] + replacement + source[position + len(old):]
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp", mutated, reason
                )

    def test_console_final_tail_accepts_only_macros_active_at_the_tail(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        safe_directives = (
            "#define return unsafe_return\n#undef return\n  ",
            "#if 0\n#define return unsafe_return\n#endif\n  ",
            "#if 0\n#define return unsafe_return\n"
            "#else\n#define harmless_tail_macro 1\n#endif\n  ",
        )
        for directives in safe_directives:
            with self.subTest(directives=directives):
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp":
                        source.replace(decrement, directives + decrement, 1),
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_console_final_tail_accepts_resumed_macros_inactive_at_return(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        safe_directives = (
            "#define resumed unsafe_resumed\n#undef resumed\n  ",
            "#if 0\n#define resumed unsafe_resumed\n#endif\n  ",
            '#pragma push_macro("resumed")\n'
            "#define resumed unsafe_resumed\n"
            '#pragma pop_macro("resumed")\n  ',
        )
        for directives in safe_directives:
            with self.subTest(directives=directives):
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp":
                        source.replace(decrement, directives + decrement, 1),
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(
                decrement,
                "#define resumed unsafe_resumed\n  " + decrement,
                1,
            ),
            "trusted final-tail token",
        )

        result = self.run_checker(self.make_tree({
            "src/vllm/platform/console_shutdown.cpp":
                source + "\n#define return unsafe_after_dispatch\n",
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_console_final_tail_ignores_comment_and_string_decoys(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        decoys = (
            'const char* macro_decoy = "#define resumed false";\n'
            "  (void)macro_decoy;\n"
            "  // #define POST_DECREMENT_RESULT false\n  "
        )
        result = self.run_checker(self.make_tree({
            "src/vllm/platform/console_shutdown.cpp":
                source.replace(decrement, decoys + decrement, 1),
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_console_drain_tails_reject_active_keyword_macros_independently(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        boundaries = {
            "DrainEntrantsWithTimeout": "bool DrainInFlightWithTimeout",
            "DrainInFlightWithTimeout": "std::atomic<unsigned> in_flight",
        }
        for function, boundary in boundaries.items():
            with self.subTest(function=function):
                definition = f"bool {function}"
                mutated = source.replace(
                    definition,
                    "#define return SwitchToThread(); return\n" + definition,
                    1,
                ).replace(boundary, "#undef return\n" + boundary, 1)
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    f"{function} trusted return-tail token",
                )

    def test_console_drain_tails_reject_one_spliced_macro_across_both(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        mutated = source.replace(
            "bool DrainEntrantsWithTimeout",
            "#define \\\nreturn SwitchToThread(); return\n"
            "bool DrainEntrantsWithTimeout",
            1,
        ).replace(
            "std::atomic<unsigned> in_flight",
            "#undef return\nstd::atomic<unsigned> in_flight",
            1,
        )
        for function in ("DrainEntrantsWithTimeout", "DrainInFlightWithTimeout"):
            with self.subTest(function=function):
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    f"{function} trusted return-tail token",
                )

    def test_console_drain_tails_reject_aliases_of_every_trusted_token(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        boundaries = {
            "DrainEntrantsWithTimeout": "bool DrainInFlightWithTimeout",
            "DrainInFlightWithTimeout": "std::atomic<unsigned> in_flight",
        }
        directives = {
            "if": "#define if(condition) if ((SwitchToThread(), condition))",
            "GetTickCount64": "#define GetTickCount64() (SwitchToThread(), GetTickCount64())",
            "start": "#define start (SwitchToThread(), start)",
            "return": "#define return SwitchToThread(); return",
            "false": "#define false (SwitchToThread(), false)",
            "true": "#define true (SwitchToThread(), true)",
        }
        for function, boundary in boundaries.items():
            for token, directive in directives.items():
                with self.subTest(function=function, token=token):
                    definition = f"bool {function}"
                    mutated = source.replace(
                        definition, f"{directive}\n{definition}", 1
                    ).replace(boundary, f"#undef {token}\n{boundary}", 1)
                    self.assert_rejected(
                        "src/vllm/platform/console_shutdown.cpp",
                        mutated,
                        f"{function} trusted return-tail token",
                    )

    def test_console_drain_tail_macro_state_is_bound_to_each_return(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        safe_mutations = (
            source.replace(
                "bool DrainEntrantsWithTimeout",
                "#define return unsafe_return\n#undef return\n"
                "bool DrainEntrantsWithTimeout",
                1,
            ),
            source.replace(
                "bool DrainInFlightWithTimeout",
                "#if 0\n#define true unsafe_true\n#endif\n"
                "bool DrainInFlightWithTimeout",
                1,
            ),
            source.replace(
                "std::atomic<unsigned> in_flight",
                "#define false unsafe_after_drain\n#undef false\n"
                "std::atomic<unsigned> in_flight",
                1,
            ),
        )
        for index, mutated in enumerate(safe_mutations):
            with self.subTest(index=index):
                result = self.run_checker(self.make_tree({
                    "src/vllm/platform/console_shutdown.cpp": mutated,
                }))
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )

    def test_console_drain_structural_view_blanks_safe_directives(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        safe_replacements = (
            (
                "if (GetTickCount64() - start >= 5000) return false;",
                "#define false unsafe_false\n"
                "#undef false\n"
                "if (GetTickCount64() - start >= 5000) return false;",
            ),
            (
                "if (GetTickCount64() - start >= 5000) return false;",
                "#define \\\n"
                "false unsafe_false\n"
                "#undef false\n"
                "if (GetTickCount64() - start >= 5000) return false;",
            ),
            (
                "return true;",
                "#define true unsafe_true\n"
                "#undef true\n"
                "return true;",
            ),
        )
        for old, new in safe_replacements:
            for occurrence in (0, 1):
                with self.subTest(old=old, occurrence=occurrence):
                    position = -1
                    for _ in range(occurrence + 1):
                        position = source.find(old, position + 1)
                    self.assertGreaterEqual(position, 0)
                    mutated = source[:position] + new + source[position + len(old):]
                    result = self.run_checker(self.make_tree({
                        "src/vllm/platform/console_shutdown.cpp": mutated,
                    }))
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_console_drain_structural_view_keeps_macro_state_fail_closed(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        uncertain = (
            "#if UNKNOWN_WINDOWS_BUILD_FLAG\n"
            "#define false (SwitchToThread(), false)\n"
            "#endif\n" + timeout
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(timeout, uncertain, 1),
            "trusted return-tail token",
        )
        spliced = (
            "#define \\\n"
            "false (SwitchToThread(), false)\n" + timeout
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            source.replace(timeout, spliced, 1),
            "trusted return-tail token",
        )

    def test_console_function_body_skips_constrained_template_prototypes(self) -> None:
        source = textwrap.dedent(
            SAFE_FILES["src/vllm/platform/console_shutdown.cpp"]
        )
        prototypes = {
            "DispatchControlEvent": textwrap.dedent("""
                template <typename T>
                bool DispatchControlEvent(T item)
                    requires requires(T value) { value(); };
                template <typename T>
                bool DispatchControlEvent(T item) noexcept
                    requires (sizeof(T) > 0);
                template <typename T>
                bool DispatchControlEvent(T item) [[deprecated]];
            """),
            "DrainEntrantsWithTimeout": textwrap.dedent("""
                template <typename T>
                bool DrainEntrantsWithTimeout(T item)
                    requires requires(T value) { value(); };
                template <typename T>
                bool DrainEntrantsWithTimeout(T item) noexcept
                    requires (sizeof(T) > 0);
                template <typename T>
                bool DrainEntrantsWithTimeout(T item) [[deprecated]];
            """),
            "DrainInFlightWithTimeout": textwrap.dedent("""
                template <typename T>
                bool DrainInFlightWithTimeout(T item)
                    requires requires(T value) { value(); };
                template <typename T>
                bool DrainInFlightWithTimeout(T item) noexcept
                    requires (sizeof(T) > 0);
                template <typename T>
                bool DrainInFlightWithTimeout(T item) [[deprecated]];
            """),
        }
        all_prototypes = "\n".join(prototypes.values())
        valid = source.replace(
            "bool DrainEntrantsWithTimeout", all_prototypes +
            "\nbool DrainEntrantsWithTimeout", 1
        )
        result = self.run_checker(self.make_tree({
            "src/vllm/platform/console_shutdown.cpp": valid,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            valid.replace(decrement, "", 1),
            "stable event/in-flight",
        )
        timeout = "if (GetTickCount64() - start >= 5000) return false;"
        self.assertEqual(valid.count(timeout), 2)
        for occurrence in (0, 1):
            position = -1
            for _ in range(occurrence + 1):
                position = valid.find(timeout, position + 1)
            mutated = valid[:position] + valid[position + len(timeout):]
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                mutated,
                "finite timeout",
            )

    def test_unsigned_elapsed_subtraction_survives_tick_wrap(self) -> None:
        mask = (1 << 64) - 1
        start = mask - 2
        self.assertEqual((1 - start) & mask, 4)
        self.assertLess((1 - start) & mask, 5)
        self.assertEqual((2 - start) & mask, 5)

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

    def test_build_script_requires_isolated_unsupported_tier_probe(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script,
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for old, new, reason in (
            (
                UNSUPPORTED_TIER_FILTER,
                "",
                "isolated unsupported-tier filter",
            ),
            ("$probeExitCode -ne 1", "$probeExitCode -eq 0",
             "exact unsupported-tier exit status"),
            ("unknown x86 ISA tier 'amx'", "unrelated diagnostic",
             "unsupported-tier diagnostic"),
            ("ExitCode = 134", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("ExitCode = -1073741819", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("ExitCode = 3", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("ExitCode = 2", "ExitCode = 1",
             "unsupported-tier crash contract"),
            ("@(& $TierTest @arguments 2>&1)",
             "@(& $TierTest @arguments)",
             "merged unsupported-tier stdout/stderr capture"),
            ("$env:VT_CPU_MATMUL_TIER = $savedTier", "",
             "unsupported-tier environment restoration"),
            ("  Invoke-UnsupportedTierContractTests\n", "",
             "unsupported-tier fake-tool contract"),
        ):
            with self.subTest(old=old):
                self.assertIn(old, script)
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    script.replace(old, new, 1),
                    reason,
                )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                "Invoke-UnsupportedTierProbe -TierTest $tierTest",
                "& $tierTest",
                1,
            ),
            "AMX refusal must use only the isolated",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                "Invoke-UnsupportedTierProbe -TierTest $tierTest",
                "Invoke-Checked $tierTest @()\n"
                "  Invoke-UnsupportedTierProbe -TierTest $tierTest",
                1,
            ),
            "AMX refusal must use only the isolated",
        )
        quoted_filter = f"'{UNSUPPORTED_TIER_FILTER}'"
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            script.replace(
                quoted_filter,
                f"{quoted_filter},\n    {quoted_filter}",
                1,
            ),
            "one exact unsupported-tier filter argument",
        )
        self.assert_rejected(
            "scripts/build-windows-release.ps1",
            f"$filterDecoy = {quoted_filter}\n" + script.replace(
                quoted_filter, "'--test-case=unrelated test'", 1
            ),
            "isolated unsupported-tier filter",
        )

    def test_real_unsupported_tier_helper_is_structurally_scoped(self) -> None:
        script = (REPO / "scripts/build-windows-release.ps1").read_text(
            encoding="utf-8"
        )
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        self.assertEqual(script.count(capture), 1)
        mutations = [(
            script.replace(
                capture, "& $TierTest\n        " + capture, 1
            ),
            "exactly one filtered process invocation",
        )]

        runner_call = "$probeResult = & $Runner $TierTest $arguments"
        self.assertEqual(script.count(runner_call), 1)
        mutations.extend((
            (
                script.replace(
                    runner_call,
                    "$null = & $Runner $TierTest @()\n        " + runner_call,
                    1,
                ),
                "exactly one fake-runner invocation",
            ),
            (
                script.replace(
                    runner_call,
                    "$probeResult = & $Runner $TierTest @()",
                    1,
                ),
                "exact unsupported-tier filter arguments",
            ),
        ))

        helper_start = script.index("function Invoke-UnsupportedTierProbe")
        helper_end = script.index(
            "function Invoke-UnsupportedTierContractTests", helper_start
        )
        helper = script[helper_start:helper_end]
        diagnostic = "unknown x86 ISA tier 'amx'"
        self.assertEqual(helper.count(diagnostic), 1)
        mutated_helper = helper.replace(diagnostic, "wrong helper diagnostic", 1)
        mutated = script[:helper_start] + mutated_helper + script[helper_end:]
        self.assertIn(diagnostic, mutated)
        mutations.append((mutated, "unsupported-tier diagnostic"))
        canonical_diagnostic = (
            '    $diagnostic = $probeOutput -join "`n"\n'
            '    if ($diagnostic -notmatch '
            '[regex]::Escape("unknown x86 ISA tier \'amx\'")) {'
        )
        fixture_only_diagnostic = (
            '    if (($probeOutput -join "`n") -notmatch '
            '[regex]::Escape("unknown x86 ISA tier \'amx\'")) {'
        )
        self.assertIn(canonical_diagnostic, helper)
        mutations.append((
            script.replace(
                canonical_diagnostic, fixture_only_diagnostic, 1
            ),
            "exact unsupported-tier probe body",
        ))
        mutations.extend((
            (
                script.replace("$calls.Add(", "$calls.Append(", 1),
                "fake runner call recording",
            ),
            (
                script.replace("$calls.Count -ne 1", "$calls.Count -lt 0", 1),
                "exactly one recorded fake call",
            ),
            (
                script.replace(
                    "$calls[0].Arguments.Count -ne 1",
                    "$calls[0].Arguments.Count -lt 0",
                    1,
                ),
                "exactly one recorded fake argument",
            ),
            (
                script.replace(
                    'Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" '
                    "-Runner $good",
                    'Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" '
                    "-Runner $good\n    "
                    'Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" '
                    "-Runner $good",
                    1,
                ),
                "exactly one unsupported-tier probe invocation",
            ),
        ))
        contract_start = script.index("function Invoke-UnsupportedTierContractTests")
        contract = script[contract_start:]
        self.assertIn(UNSUPPORTED_TIER_FILTER, contract)
        mutated_contract = contract.replace(
            UNSUPPORTED_TIER_FILTER, "--test-case=wrong fake filter", 1
        )
        mutations.append((
            script[:contract_start] + mutated_contract,
            "exact recorded fake filter argument",
        ))
        for mutation, reason in mutations:
            with self.subTest(reason=reason):
                self.assert_rejected(
                    "scripts/build-windows-release.ps1", mutation, reason
                )

    def test_unsupported_tier_helper_requires_mandatory_tier_parameter(self) -> None:
        canonical = "[Parameter(Mandatory)][string]$TierTest"
        optional = "[string]$TierTest"
        fixture = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        real = (REPO / "scripts/build-windows-release.ps1").read_text(
            encoding="utf-8"
        )
        for label, script in (("fixture", fixture), ("real", real)):
            with self.subTest(label=label):
                self.assertEqual(script.count(canonical), 1)
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    script.replace(canonical, optional, 1),
                    "exact unsupported-tier probe body",
                )

    def test_unsupported_tier_rejects_equivalent_unfiltered_target_forms(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        equivalent_targets = (
            "${TierTest}",
            "${TiErTeSt}",
            "($TierTest)",
            "$($TierTest)",
            "$local:TierTest",
            "${local:TierTest}",
            "$script:TiErTeSt",
            "${private:TierTest}",
        )
        for target in equivalent_targets:
            with self.subTest(target=target):
                mutated = script.replace(
                    capture, f"$null = & {target}\n            {capture}", 1
                )
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    mutated,
                    "exactly one filtered process invocation",
                )

    def test_unsupported_tier_helper_rejects_every_extra_executable_form(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        before_forms = (
            "$alias = $TierTest\n            & $alias",
            "$env:TierAlias = $TierTest\n"
            "            Invoke-Expression $env:TierAlias",
            "$alias = Get-Variable TierTest -ValueOnly\n            . $alias",
            "Start-Process -FilePath $TierTest -Wait",
            "[System.Diagnostics.Process]::Start($TierTest)",
        )
        for executable in before_forms:
            for placement in ("before", "after"):
                with self.subTest(executable=executable, placement=placement):
                    replacement = (
                        executable + "\n            " + capture
                        if placement == "before" else
                        capture + "\n            " + executable
                    )
                    self.assert_rejected(
                        "scripts/build-windows-release.ps1",
                        script.replace(capture, replacement, 1),
                        "exact unsupported-tier probe body",
                    )

    def test_unsupported_tier_amx_scope_rejects_indirect_execution(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        assignment = '$env:VT_CPU_MATMUL_TIER = "amx"'
        probe = "Invoke-UnsupportedTierProbe -TierTest $tierTest"
        indirect_forms = (
            "$alias = $tierTest; & $alias",
            "$env:TierAlias = $tierTest; Invoke-Expression $env:TierAlias",
            "$alias = Get-Variable tierTest -ValueOnly; . $alias",
            "Start-Process -FilePath $tierTest -Wait",
            "[System.Diagnostics.Process]::Start($tierTest)",
        )
        for executable in indirect_forms:
            for placement in ("before", "after"):
                with self.subTest(executable=executable, placement=placement):
                    mutation = (
                        script.replace(
                            assignment,
                            assignment + "\n          " + executable,
                            1,
                        ) if placement == "before" else
                        script.replace(
                            probe,
                            probe + "\n          " + executable,
                            1,
                        )
                    )
                    self.assert_rejected(
                        "scripts/build-windows-release.ps1",
                        mutation,
                        "exact AMX refusal block",
                    )

    def test_unsupported_tier_structural_view_ignores_literal_and_comment_decoys(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        capture = "$probeOutput = @(& $TierTest @arguments 2>&1)"
        decoy = "@(& $TierTest @arguments 2>&1)"
        accepted = (
            script.replace(
                'throw "unexpected exit"',
                f"throw 'diagnostic text {decoy} {UNSUPPORTED_TIER_FILTER} "
                "unknown x86 ISA tier ''amx'''",
                1,
            ),
            script.replace(
                capture,
                capture + f" # inline decoy {decoy}",
                1,
            ),
            script.replace(
                capture,
                f"# full-line decoy {decoy}\n            " + capture,
                1,
            ),
        )
        for mutation in accepted:
            result = self.run_checker(self.make_tree({
                "scripts/build-windows-release.ps1": mutation,
            }))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unsupported_tier_filter_is_one_exact_process_argument(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        arguments = re.findall(
            r"(?m)^\s*['\"](--test-case=.*)['\"]\s*$", script
        )
        self.assertEqual(arguments, [UNSUPPORTED_TIER_FILTER])

    def test_unsupported_tier_exact_literals_reject_runtime_backtick_escapes(self) -> None:
        fixture = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        real = (REPO / "scripts/build-windows-release.ps1").read_text(
            encoding="utf-8"
        )
        for label, script in (("fixture", fixture), ("real", real)):
            with self.subTest(label=label, literal="filter"):
                quoted_filter = f"'{UNSUPPORTED_TIER_FILTER}'"
                self.assertIn(quoted_filter, script)
                escaped_filter = '"--`test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran"'
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    script.replace(quoted_filter, escaped_filter, 1),
                    "exact unsupported-tier probe body",
                )
            with self.subTest(label=label, literal="diagnostic"):
                helper_start = script.index("function Invoke-UnsupportedTierProbe")
                helper_end = script.index(
                    "function Invoke-UnsupportedTierContractTests", helper_start
                )
                helper = script[helper_start:helper_end]
                diagnostic = '"unknown x86 ISA tier \'amx\'"'
                self.assertIn(diagnostic, helper)
                mutated_helper = helper.replace(
                    diagnostic, '"u`nknown x86 ISA tier \'amx\'"', 1
                )
                mutated = script[:helper_start] + mutated_helper + script[helper_end:]
                self.assert_rejected(
                    "scripts/build-windows-release.ps1",
                    mutated,
                    "exact unsupported-tier probe body",
                )

    def test_unsupported_tier_exact_filter_accepts_plain_double_quotes(self) -> None:
        script = textwrap.dedent(SAFE_FILES["scripts/build-windows-release.ps1"])
        quoted_filter = f"'{UNSUPPORTED_TIER_FILTER}'"
        result = self.run_checker(self.make_tree({
            "scripts/build-windows-release.ps1": script.replace(
                quoted_filter, f'"{UNSUPPORTED_TIER_FILTER}"', 1
            ),
        }))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_real_console_rejects_state_access_after_final_decrement(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        decrement = (
            "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        )
        self.assertEqual(console.count(decrement), 1)
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            console.replace(
                decrement,
                decrement + "\n  SetEvent(state->stop_event);",
                1,
            ),
            "final in-flight decrement",
        )

    def test_real_console_rejects_macro_post_decrement_result(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        decrement = "state->in_flight.fetch_sub(1, std::memory_order_seq_cst);"
        self.assertEqual(console.count(decrement), 1)
        self.assertEqual(console.count("return resumed;"), 1)
        mutated = console.replace(
            decrement,
            "#define POST_DECREMENT_RESULT "
            "(SetEvent(state->stop_event), resumed)\n  " + decrement,
            1,
        ).replace(
            "return resumed;", "return POST_DECREMENT_RESULT;", 1
        )
        self.assert_rejected(
            "src/vllm/platform/console_shutdown.cpp",
            mutated,
            "final in-flight decrement",
        )

    def test_real_console_drains_require_reachable_timeout_branches(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        timeout = (
            "if (GetTickCount64() - start >= kHandlerDrainTimeoutMs) return false;"
        )
        self.assertEqual(console.count(timeout), 2)
        decoy = (
            'const char* timeout_decoy = "if (GetTickCount64() - start >= '
            'kHandlerDrainTimeoutMs) return false;";\n'
            "    // if (GetTickCount64() - start >= kHandlerDrainTimeoutMs) "
            "return false;"
        )
        for occurrence in (0, 1):
            position = -1
            for _ in range(occurrence + 1):
                position = console.find(timeout, position + 1)
            self.assertGreaterEqual(position, 0)
            mutated = console[:position] + decoy + console[position + len(timeout):]
            self.assert_rejected(
                "src/vllm/platform/console_shutdown.cpp",
                mutated,
                "finite timeout",
            )

    def test_real_console_rejects_absolute_deadline_in_each_drain(self) -> None:
        console = (REPO / "src/vllm/platform/console_shutdown.cpp").read_text(
            encoding="utf-8"
        )
        start = "const ULONGLONG start = GetTickCount64();"
        elapsed = (
            "if (GetTickCount64() - start >= kHandlerDrainTimeoutMs) return false;"
        )
        self.assertEqual(console.count(start), 2)
        self.assertEqual(console.count(elapsed), 2)
        for function in ("DrainEntrantsWithTimeout", "DrainInFlightWithTimeout"):
            with self.subTest(function=function):
                function_at = console.index(f"bool {function}")
                next_function = console.find("\nbool ", function_at + 1)
                function_end = len(console) if next_function < 0 else next_function
                body = console[function_at:function_end]
                self.assertIn(start, body)
                self.assertIn(elapsed, body)
                body = body.replace(
                    start,
                    "const ULONGLONG deadline = "
                    "GetTickCount64() + kHandlerDrainTimeoutMs;",
                    1,
                ).replace(
                    elapsed,
                    "if (GetTickCount64() >= deadline) return false;",
                    1,
                )
                mutated = console[:function_at] + body + console[function_end:]
                self.assert_rejected(
                    "src/vllm/platform/console_shutdown.cpp",
                    mutated,
                    "finite timeout",
                )

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
