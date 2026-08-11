[CmdletBinding()]
param(
    [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$BuildDir = (Join-Path $SourceDir "build-release-windows-cpu"),
    [string]$StageDir = (Join-Path $BuildDir "stage"),
    [string]$SmokeModel = (Join-Path $SourceDir "tests/vllm/models/fixtures/llama_embed_e2e"),
    [int]$SmokePort = 18080
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([Parameter(Mandatory)][string]$Program,
          [Parameter(Mandatory)][string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with status $LASTEXITCODE"
    }
}

if (-not (Test-Path (Join-Path $SmokeModel "config.json"))) {
    throw "Windows runtime smoke model is incomplete: $SmokeModel"
}

Invoke-Checked cmake @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DVLLM_CPP_BUILD_TESTS=ON",
    "-DVLLM_CPP_BUILD_EXAMPLES=ON",
    "-DVLLM_CPP_SERVER=ON",
    "-DVLLM_CPP_CUDA=OFF",
    "-DVLLM_CPP_CUDA_ARCHITECTURES=",
    "-DVLLM_CPP_HIP=OFF",
    "-DVLLM_CPP_HIP_ARCHITECTURES=",
    "-DVLLM_CPP_METAL=OFF",
    "-DVLLM_CPP_MLX=OFF",
    "-DMLX_ROOT=",
    "-DVLLM_CPP_TRITON=OFF",
    "-DVLLM_CPP_VULKAN=OFF",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
)

$targets = @(
    "server",
    "test_openai_api_server",
    "test_lmcache_client",
    "test_kv_offload_fs",
    "test_cpu_isa_x86",
    "test_ops_matmul_elem"
)
Invoke-Checked cmake (@("--build", $BuildDir, "--config", "Release", "--target") + $targets)

foreach ($test in @(
    "test_openai_api_server.exe",
    "test_lmcache_client.exe",
    "test_kv_offload_fs.exe",
    "test_cpu_isa_x86.exe"
)) {
    Invoke-Checked (Join-Path $BuildDir "tests/Release/$test") @()
}

$tierTest = Join-Path $BuildDir "tests/Release/test_ops_matmul_elem.exe"
$savedTier = $env:VT_CPU_MATMUL_TIER
try {
    $env:VT_CPU_MATMUL_TIER = "portable"
    Invoke-Checked $tierTest @()
    $env:VT_CPU_MATMUL_TIER = "avx2"
    Invoke-Checked $tierTest @()
    $env:VT_CPU_MATMUL_TIER = "amx"
    & $tierTest
    if ($LASTEXITCODE -eq 0) {
        throw "unsupported forced CPU tier 'amx' was silently accepted"
    }
} finally {
    $env:VT_CPU_MATMUL_TIER = $savedTier
}

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
Invoke-Checked cmake @(
    "--install", $BuildDir,
    "--config", "Release",
    "--prefix", $StageDir,
    "--component", "vllm-server"
)

$server = Join-Path $StageDir "bin/vllm-server.exe"
if (-not (Test-Path $server)) {
    throw "native install did not stage bin/vllm-server.exe"
}
Invoke-Checked $server @("--help")

# Python's Windows subprocess path calls CreateProcess with
# CREATE_NEW_PROCESS_GROUP, letting the smoke target one CTRL_BREAK_EVENT at the
# extracted server without broadcasting to the Actions runner's console.
$smokeHarness = Join-Path $BuildDir "windows_server_smoke.py"
@'
import json
import signal
import subprocess
import sys
import time
import urllib.request

server, model, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
proc = subprocess.Popen(
    [server, "--model", model, "--host", "127.0.0.1", "--port", str(port)],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
try:
    deadline = time.monotonic() + 60
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health check: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("/health did not return 200 within 60 seconds")
        time.sleep(0.1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/version", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"/version returned {response.status}")
        json.loads(response.read())
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    if proc.wait(timeout=20) != 0:
        raise RuntimeError(f"server did not stop cleanly: {proc.returncode}")
finally:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
'@ | Set-Content -LiteralPath $smokeHarness -Encoding utf8NoBOM

Invoke-Checked python @($smokeHarness, $server, $SmokeModel, "$SmokePort")
Write-Host "Windows native CPU build/stage/runtime gate OK: $server"
