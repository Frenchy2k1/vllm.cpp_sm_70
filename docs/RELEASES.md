# Binary releases

Tagged releases are built from one immutable source commit. The release workflow
builds each host/backend tuple independently, validates the freshly extracted
archive, assembles a byte-bound handoff, generates the indexes from the embedded
manifests, attests the verified archives, and only then publishes them.

No release is created by pull requests or manual workflow runs. A tag must equal
the project version as `v<version>`, the release matrix must be marked ready, and
the protected `release` environment must approve publication.

## Primary downloads

| Artifact | Channel | Contents and runtime boundary |
|---|---|---|
| `linux-x86_64-glibc-cpu` | stable | One conservative-baseline adaptive x86-64 binary with portable/SSE2, F16C, AVX2, and AVX-512 tiers |
| `linux-aarch64-glibc-cpu` | stable | One adaptive arm64 binary with portable/NEON and independently gated DotProd/i8mm kernels |
| `linux-x86_64-musl-cpu-static` | experimental-preview | Literal-static, CPU-only feasibility bundle; not a glibc replacement |
| `linux-x86_64-glibc-cuda` | preview | One x86-64 binary containing all ten supported SMs and six exact-SM Triton AOT trees |
| `linux-aarch64-glibc-cuda` | preview | The same complete CUDA architecture inventory for the arm64 host ABI |
| `linux-x86_64-glibc-vulkan` | preview | Vulkan bundle; the loader, ICD, and device driver remain host dependencies |
| `macos-arm64-metal` | stable | Native Apple Silicon Metal bundle; system frameworks remain host dependencies |
| `macos-arm64-metal-mlx` | preview | Metal plus the exact redistributable MLX dylib/metallib and license |

The CUDA fat archives contain `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90a`,
`sm_100a`, `sm_103a`, `sm_110`, `sm_120a`, and `sm_121a`. NVIDIA's kernel and
driver ABI are never bundled. Optional per-SM diagnostic builds are not primary
downloads and cannot substitute for either fat-archive gate.

Models, tokenizer data, certificates, GPU drivers, Python, PyTorch, compilers,
and source trees are not included. Text serving does not require `ffmpeg`; video
features require a compatible `ffmpeg` executable on `PATH` or an explicit
`--video-ffmpeg` path.

## Verify a download

Each archive is published with a `.sha256` checksum and a
`.provenance.json` sidecar. The generated `release-index.json` and
`RELEASE_INDEX.md` enumerate the exact published bytes, channel, host ABI,
compiled CPU tiers or CUDA SMs, runtime boundary, and known limitations.

```sh
sha256sum --check vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz.sha256
tar -xzf vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz
./bin/vllm-server --version
./bin/vllm-server --help
```

The archive also contains `release-manifest.json`, `VERSION`, an SPDX JSON SBOM,
third-party notices, and required redistributed licenses.

## Retention

Intermediate GitHub Actions artifacts are retained for seven days. Published
GitHub release assets remain available until a maintainer explicitly deletes
the release or its assets. The same policy is machine-readable in
`release/release-matrix.json` and copied into every generated release index.

## Maintainer flow

1. Set the project version and ensure `release/release-matrix.json` remains
   publish-ready.
2. Run a manual `release` workflow on the intended commit. It is a non-publishing
   full-matrix dry run.
3. Create and push the exact `v<version>` tag only after the dry run is green.
4. Approve the protected `release` environment after build, verification, and
   attestation succeed. Publication enumerates only files authenticated by the
   verified handoff; shell globs and older workflow artifacts are rejected.
