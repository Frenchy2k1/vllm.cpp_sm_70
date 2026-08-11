[CmdletBinding()]
param(
    [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$BuildDir = (Join-Path $SourceDir "build-release-windows-cpu"),
    [string]$StageDir = (Join-Path $BuildDir "stage"),
    [string]$SmokeModel = (Join-Path $SourceDir "tests/vllm/models/fixtures/llama_embed_e2e"),
    [int]$SmokePort = 18080,
    [switch]$ContractTest
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

function Assert-CrtPolicy {
    param([Parameter(Mandatory)][string[]]$DirectiveOutput,
          [Parameter(Mandatory)][string[]]$ImportOutput)
    $directives = $DirectiveOutput -join "`n"
    $imports = $ImportOutput -join "`n"
    if ($directives -notmatch '(?im)DEFAULTLIB\s*:\s*"?LIBCMT"?') {
        throw "COFF CRT audit: no /DEFAULTLIB:LIBCMT static CRT directive found"
    }
    if ($directives -match '(?im)DEFAULTLIB\s*:\s*"?(?:MSVCRT|MSVCPRT|LIBCMTD)"?') {
        throw "COFF CRT audit: dynamic or debug CRT directive found"
    }
    if ($imports -match '(?im)^\s*(?:VCRUNTIME[^\s]*|MSVCP[^\s]*|CONCRT[^\s]*|UCRTBASED?|api-ms-win-crt-[^\s]*|MSVCR[^\s]*)\.dll\s*$') {
        throw "PE CRT audit: dynamic or debug CRT DLL import found"
    }
}

function Invoke-CrtAudit {
    param([Parameter(Mandatory)][string[]]$Artifacts,
          [Parameter(Mandatory)][string]$Server,
          [scriptblock]$DumpbinRunner = {
              param([string]$Mode, [string]$Path)
              $output = & dumpbin $Mode $Path 2>&1
              if ($LASTEXITCODE -ne 0) {
                  throw "dumpbin $Mode failed for $Path with status $LASTEXITCODE"
              }
              return @($output)
          })
    $directiveOutput = @()
    foreach ($artifact in $Artifacts) {
        $directiveOutput += & $DumpbinRunner "/directives" $artifact
    }
    $importOutput = @(& $DumpbinRunner "/imports" $Server)
    Assert-CrtPolicy -DirectiveOutput $directiveOutput -ImportOutput $importOutput
    Write-Host ($directiveOutput -join "`n")
    Write-Host ($importOutput -join "`n")
}

function Invoke-CrtContractTests {
    $good = {
        param([string]$Mode, [string]$Path)
        if ($Mode -eq "/directives") { return '/DEFAULTLIB:"LIBCMT"' }
        return @("$Path", "KERNEL32.dll", "WS2_32.dll")
    }
    Invoke-CrtAudit -Artifacts @("fake-vllm.lib", "fake-server.obj") `
        -Server "fake-vllm-server.exe" -DumpbinRunner $good
    foreach ($bad in @(
        { param($Mode, $Path) if ($Mode -eq "/directives") { '/DEFAULTLIB:"MSVCRT"' } else { 'KERNEL32.dll' } },
        { param($Mode, $Path) if ($Mode -eq "/directives") { '/DEFAULTLIB:"LIBCMT"' } else { 'UCRTBASE.dll' } }
    )) {
        $rejected = $false
        try {
            Invoke-CrtAudit -Artifacts @("fake.lib") -Server "fake.exe" `
                -DumpbinRunner $bad
        } catch {
            $rejected = $true
        }
        if (-not $rejected) { throw "injected bad dumpbin output was accepted" }
    }
}

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
    if ($probeExitCode -ne 1) {
        throw "unsupported forced CPU tier probe exited with status $probeExitCode instead of 1"
    }
    $diagnostic = $probeOutput -join "`n"
    if ($diagnostic -notmatch [regex]::Escape("unknown x86 ISA tier 'amx'")) {
        throw "unsupported forced CPU tier probe did not report the expected diagnostic"
    }
}

function Invoke-UnsupportedTierContractTests {
    $diagnostic = "unknown x86 ISA tier 'amx'"
    $good = {
        param([string]$Program, [string[]]$Arguments)
        [pscustomobject]@{ ExitCode = 1; Output = @($diagnostic) }
    }.GetNewClosure()
    Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" -Runner $good

    $badResults = @(
        [pscustomobject]@{ ExitCode = 0; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 134; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = -1073741819; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 3; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 2; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 1; Output = @("wrong diagnostic") }
    )
    foreach ($badResult in $badResults) {
        $runner = {
            param([string]$Program, [string[]]$Arguments)
            return $badResult
        }.GetNewClosure()
        $rejected = $false
        try {
            Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" `
                -Runner $runner
        } catch {
            $rejected = $true
        }
        if (-not $rejected) {
            throw "injected bad unsupported-tier result was accepted"
        }
    }
}

if ($ContractTest) {
    Invoke-CrtContractTests
    Invoke-UnsupportedTierContractTests
    Write-Host "Windows PowerShell/CRT contract tests OK"
    exit 0
}

if (-not (Test-Path (Join-Path $SmokeModel "config.json"))) {
    throw "Windows runtime smoke model is incomplete: $SmokeModel"
}

$queryDir = Join-Path $BuildDir ".cmake/api/v1/query"
New-Item -ItemType Directory -Force -Path $queryDir | Out-Null
New-Item -ItemType File -Force -Path (Join-Path $queryDir "codemodel-v2") | Out-Null

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
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/check-windows-portability.py"),
    "--root", $SourceDir,
    "--build-dir", $BuildDir
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
$crtArtifacts = @(
    Get-ChildItem -Path $BuildDir -Recurse -File -Include "*.obj", "vllm*.lib" |
        Where-Object { $_.FullName -notmatch '[\\/](?:_deps|third_party)[\\/]' } |
        ForEach-Object { $_.FullName }
)
if ($crtArtifacts.Count -eq 0) {
    throw "COFF CRT audit found no project objects or static libraries"
}
Invoke-CrtAudit -Artifacts $crtArtifacts -Server $server

Invoke-Checked $server @("--help")

$tierTest = Join-Path $BuildDir "tests/Release/test_ops_matmul_elem.exe"
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
