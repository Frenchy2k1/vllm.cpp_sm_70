# Per-arch Triton-AOT GDN cubins (`CLAIM-TRITON-AOT-PER-ARCH`)

Owner claim: `CLAIM-TRITON-AOT-PER-ARCH`. Base `main` `308c312a`. Upstream pin
vLLM `555967922` (0.26.0.dev0). This spec closes the honest gap that the vendored
Triton-AOT GDN fast-path cubins existed for **only `sm_121a`**, so the codegen
win was runtime-real on GB10 alone and every other CUDA arch fell back to the
spilling hand kernel on GDN decode.

## The gap (verified in source)

`src/vt/cuda/cuda_gdn.cu:2308-2318` dispatches GDN packed decode through
`TryTritonPackedDecode`, which launches the vendored FLA cubin
(`gdn_decode_h48`/`gdn_decode_h32`). That fast-path is **codegen-bound** — dgx
phase-1 (2026-07-16) cuobjdump MEASURED the identical register-resident
`[BV=32,BK=128]` fp32 state tile at **REG:205 / 0-spill under Triton** vs
**REG:255 + STACK:48 (spills) as hand-CUDA**. The vendored cubins are compiled
per-arch and embedded as C byte arrays (see `cmake/TritonAOT.cmake`); a cubin is
loaded via `cuModuleLoadData` and the CUDA driver **rejects it on any SM other
than the one it was compiled for**. Before this work the only vendored tree was
`src/vt/cuda/triton_aot_vendored/sm_121a/`.

Consequence (now stated plainly): on `sm_80/86/89/90a/100a` a
`-DVLLM_CPP_TRITON=ON` single-arch build had **no vendored tree** — configuration
would FATAL with "regenerate for this arch". Every committed cross-family arch
build therefore used **`-DVLLM_CPP_TRITON=OFF`** (portable-kernels-only, see
`backend-matrix.md` `BACKEND-CUDA-SM080/…`), i.e. GDN decode ran the **spilling
hand kernel**. The Triton-AOT GDN-decode codegen win was `sm_121a`-runtime-only;
it was NOT at parity on any other arch. No committed record over-claimed
non-`sm_121` GDN-decode Triton parity (the `KERNEL-GDN-PACKED-DECODE` row is
explicitly "27B-only … `sm_121a`"), but `KERNEL-CUDA-DISPATCH-AOT` read "only
SM121 AOT artifacts are evidenced" — corrected here.

## The dispatch model (build-time, already additive)

There is NO runtime arch dispatch of cubins — a single-arch build embeds exactly
one arch's cubin. `cmake/TritonAOT.cmake::_triton_aot_arch_name` derives the
vendored subdir as `sm_${VLLM_CPP_CUDA_ARCHITECTURES}` (or the
`VLLM_CPP_TRITON_VENDORED_ARCH` override), and `_triton_aot_resolved_target`
derives the Triton target `cuda:<cc>:32` from the dir name (strips the `a`
suffix). So the "per-arch dispatch" is the **build-time selection of the vendored
tree**, and it is already additive: dropping a new `sm_XX/` tree makes a
`-DVLLM_CPP_CUDA_ARCHITECTURES=XX -DVLLM_CPP_TRITON=ON` build select it with zero
code change. `cuda_gdn.cu`'s `EnsureGdnPackedDecodeLoaded` / `load_gdn_decode_*`
loaders compile in whichever arch's cubin the build selected — mirrors how
`sm_121a` was selected, additive per-arch.

## The regen recipe (maintainer / GPU task)

Run on a box with Triton + ptxas (dgx GB10, `~/venvs/vllm-oracle` = Triton 3.6.0,
bundled ptxas CUDA 12.8 — supports sm_80/86/89/90a/100a/120a/121a). ptxas is a
cross-compiler: the **target GPU need not be present**, and AOT compile with a
pinned signature/num-warps/num-stages does no autotuning, so no GPU is touched.

Per arch (`sm_80 sm_86 sm_89 sm_90a sm_100a`):

```
cmake -S . -B build-regen \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_TRITON_REGEN=ON \
  -DVLLM_CPP_TRITON_PYTHON=$HOME/venvs/vllm-oracle/bin/python \
  -DVLLM_CPP_TRITON_VENDORED_ARCH=sm_XX
```

The regen runs at **configure time** (execute_process → `triton.tools.compile`
per specialization, `triton.tools.link` for the stable dispatcher), writes
`src/vt/cuda/triton_aot_vendored/sm_XX/` (57 artifacts + MANIFEST, matching the
`sm_121a` fileset), and `triton_aot_finalize()` records the MANIFEST (arch,
`triton_target cuda:XX:32`, generator/ptxas/toolkit versions, per-`.py` source
sha256, per-base params, per-artifact sha256). One configure regenerates the
WHOLE GDN AOT set for the arch (deltah, chunk_o, chunk_o-bf16, decode, kkt, tril,
wu — h48 + h32), so delta_h/chunk_o come with the decode kernel for free.

## Per-arch status (2026-07-28, base `308c312a`)

All five regenerated on dgx GB10 and vendored:

| Arch dir | Triton target | GDN-decode SASS (cuobjdump) | decode REG (h48/h32) | Signal |
|---|---|---|---|---|
| `sm_80` | `cuda:80:32` | `sm=80`, `EF_CUDA_SM80` | 212 / 217 | DERIVED+BUILD-VERIFIED |
| `sm_86` | `cuda:86:32` | `sm=86`, `EF_CUDA_SM86` | 209 / 209 | DERIVED+BUILD-VERIFIED |
| `sm_89` | `cuda:89:32` | `sm=89`, `EF_CUDA_SM89` | 209 / 209 | DERIVED+BUILD-VERIFIED |
| `sm_90a` | `cuda:90:32` | `sm=90`, `EF_CUDA_SM90` (wgmma-class) | (Hopper disasm) | DERIVED+BUILD-VERIFIED |
| `sm_100a` | `cuda:100:32` | `sm=100` | (Blackwell disasm) | DERIVED+BUILD-VERIFIED |
| `sm_121a` | `cuda:121:32` | `sm=121` (UNCHANGED, git byte-identical) | — | RUNTIME-VERIFIED (27B/35B SACRED) |

The decode register counts (209–217, 0-spill on the disassemblable Ampere/Ada
arches) are all well under the hand-CUDA REG:255+STACK:48 spill floor, so the
codegen-win rationale carries to the new arches — **build-verified, not
runtime-measured**.

## Honest signal (the derive-and-ship boundary)

**No non-`sm_121` board runs a GDN-hybrid model here.** The regenerated cubins
are **DERIVED+BUILD-VERIFIED**: they are real, valid SASS for the named target
(cuobjdump `sm=XX` per arch), the builder-path configure selects + integrity-
verifies each tree (`vendored … (no Python)`, MANIFEST hash/staleness/target
checks pass), and the generated C compiles clean. But GDN-decode **parity on
these arches is NOT runtime-measured** — the codegen win (TPOT/throughput) was
measured only on GB10 `sm_121a`. Do NOT read a build as a runtime decode-parity
claim on any arch no board ran. Runtime verification is welcome on
Ampere/Ada/Hopper/datacenter-Blackwell silicon.

## sm_121a untouched (SACRED gate proof)

This work only ADDS other-arch trees. `src/vt/cuda/triton_aot_vendored/sm_121a/`
is byte-identical to base `308c312a` (git status shows zero changes under it);
the `sm_121a` builder-path configure still validates its MANIFEST and selects it.
The 27B/35B GDN SACRED gate is therefore structurally unchanged (same cubin
bytes, same loaders, same dispatch). Verified: `check-triton-aot-drift.sh` rc=0
across all six arch trees.
