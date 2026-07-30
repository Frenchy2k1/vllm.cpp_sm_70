# ds4 vs ours — the Q8_0 decode GEMV, DIRECTLY PROFILED (the one open ds4-gap question)

Status: **MEASUREMENT-ONLY (no code changed). VERDICT: (A) PORTABLE LEVER FOUND.**
The DeepSeek-V4-Flash decode Q8_0 GEMV is **NOT** at an irreducible roofline — Brick 11's
"41 ms/step is the DRAM-bandwidth roofline, irreducible" conclusion is **REFUTED by direct
measurement**: on the *identical* 6.15 GiB Q8_0 weight tower, ds4 moves it at **~90% of the
GB10 roofline** while ours moves it at **~67%**. The gap is not bytes, not ALU, not alignment,
not lane-occupancy (Bricks 3/4/8/11 all correctly found those flat) — it is **launch
consolidation**: ds4 reads the tower in **259 big, paired/fused launches/step**, ours in
**646 tiny single-tensor launches/step**, and the many small launches under-subscribe DRAM.

Branch `ds4-q8-kernel-profile` off `d6419601` (records-only, NOT pushed).

---

## 0. What was asked, and the method actually used

Directly profile **ds4's** Q8_0 decode GEMV kernel AND **our** `QuantDotGemmQ8_0Kernel` on the
SAME DeepSeek-V4-Flash decode projections and DIFF: (1) achieved DRAM GB/s, (2) bytes/step,
(3) L2-hit + access pattern, (4) kernel count/shape. Decide (A) portable-lever vs (B) honest-ceiling.

**`ncu` IS installed on the DGX** (`/usr/local/cuda-13.0/bin/ncu`) — the earlier "no ncu"
memory was wrong about presence but right about usability: **every hardware-counter path is
permission-blocked**. `ncu` → `ERR_NVGPUCTRPERM`; `nsys --gpu-metrics-devices` → *"Insufficient
privilege … NVIDIA GB10 … ERR_NVGPUCTRPERM"*. No `sudo` on the box → the driver
`RestrictProfilingToAdminUsers` flag cannot be flipped. **So task items (1)/(2)/(4) are answered
via the counter-IMMUNE method** (the task's stated fallback): `nsys` per-kernel timing
(`cuda_gpu_kern_sum`, `--cuda-graph-trace=node`) + exact byte-accounting from the GGUF tensor
table. **Task item (3) — L2 hit-rate / raw sector access pattern — is UNMEASURABLE this
session** (needs the blocked counters); achieved-BW (bytes ÷ time) is the immune proxy and is
what decides the verdict.

**Rig discipline:** worker absent (`docker ps` empty), one model resident at a time, every run
under `flock $HOME/gpu.lock`, `--gpu-vram 90`, `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=28000`, all
runs foreground/synchronous. Prompt `"The capital of France is"`, greedy (`--temp 0`).

**Prefill-strip via the diff method:** every per-step number below is
`(Total_ns@-n60 − Total_ns@-n4) / 56` per kernel — the exact pure-decode-per-step time, with
the single prefill and its slow big-M launches removed by construction.

---

## 1. THE TWO-KERNEL MEASURED DIFF (headline deliverable)

**Model:** `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`, 80.76 GiB,
43 layers. Runs verified steady-state: ds4 **16.51 tok/s** under nsys (== the 16.5 claim),
ours **11.68 tok/s** (Lever-1 build `ds4lever1`, the current default `QuantDotGemmQ8_0Kernel`).

### Byte anchor (GGUF `--inspect`, exact)
`q8_0: 345 tensors, 6.15 GiB`. Under keep-quant device-resident decode the **entire Q8_0 tower
is read exactly once per decode step in BOTH engines** (weights resident; ds4's opt fp16-dequant
cache was **budget-EXHAUSTED this run** — log: *"q8 fp16 cache budget exhausted; using q8
kernels"* — so ds4 reads the **same raw 34-byte q8_0 blocks we do**, apples-to-apples).
Bytes/step = 6.15 GiB = **6.604 GB** (decimal). GB10 peak DRAM BW = **240.2 GB/s** (measured, Brick 0).

*Sanity that bytes are equal, not fewer:* reading >6.6 GB in our 40.93 ms would need >240 GB/s —
impossible — so our 646 launches are **row-splits summing to the same 6.15 GiB read once**, NOT
redundant re-reads. Both engines move an identical 6.604 GB. The time difference is therefore a
pure **effective-bandwidth** difference.

### The diff table (dense Q8_0 decode, per step)

| | our engine | ds4 (sum of its 4 Q8_0 kernels) |
|---|---|---|
| **dense-Q8_0 time / step** | **40.93 ms** | **30.38 ms** |
| **launches / step** | **646** | **259** |
| **weight bytes / step** | 6.604 GB (345 tensors ×1) | 6.604 GB (same 345 ×1) |
| **achieved DRAM BW** | **161.4 GB/s** | **217.4 GB/s** |
| **% of 240.2 roofline** | **67.2 %** | **90.5 %** |
| **L2 hit-rate** | *counter-blocked (ERR_NVGPUCTRPERM)* | *counter-blocked* |
| per-matmul-equiv (paired class) | 63.4 µs | **38.8 µs (1.63×)** |

**ds4 is 1.35× more bandwidth-efficient on the byte-identical tower** (−10.5 ms/step on Q8_0).

### ds4's Q8_0 decode is 4 structurally-distinct kernels (per-step, diff-verified)
| ds4 kernel | ms/step | launches/step | med µs | what it does differently |
|---|---|---|---|---|
| `matmul_q8_0_pair_preq_warp8_kernel` | 5.20 | 86 | 77.6 | **TWO weight matrices, ONE shared activation** per launch (2 `const unsigned char*` w) |
| `matmul_q8_0_hc_expand_preq_warp8_kernel` | 9.04 | 86 | 128.0 | **fuses the MHC hyper-connection expand INTO the matmul epilogue** (no separate glue kernel / HBM round-trip) |
| `grouped_q8_0_a_preq_warp8_kernel` | 6.88 | 43 | 158.4 | **groups** the MLA "a" (down) projections |
| `matmul_q8_0_preq_warp8_kernel` | 9.26 | 44 | 159.5 | plain warp-per-row (incl. lm_head); the ONE that is 1:1 with ours |
| **ds4 dense-Q8_0 total** | **30.38** | **259** | | |

Our side: **one** kernel, `QuantDotGemmQ8_0Kernel`, **646 single-tensor launches/step**, avg 63.4 µs.

### Whole-step reconciliation (all-kernel sum, diff /56)
| | GPU-active ms/step | → t/s (GPU) | wall t/s | host gap |
|---|---|---|---|---|
| ds4 | 56.80 | 17.6 | 16.51 | ~3.8 ms |
| ours | 82.62 | 12.1 | 11.68 | ~3.0 ms |

Whole-step GPU gap = **25.82 ms/step**; the dense-Q8_0 bucket is **10.54 ms (40.8 %)** of it.
(The rest is the MoE-expert + glue delta — a separate front.)

---

## 2. Source grounding (both kernels read, not inferred)

- **ds4 T=1 decode dispatch** `ds4_cuda.cu:12254` → for `n_tok==1` launches
  `matmul_q8_0_preq_warp8_kernel<<<(out_dim+7)/8, 256>>>` (`:4343`): 8 warps/block, warp-per-row,
  34-byte blocks (`d` uint16 + 32×int8 at offset 2), `dot_i8x32_dp4a` (`:4169`) with
  `load_i8x4_i32_unaligned`. The pair/hc_expand/grouped variants (`:4485` / `:4762` / `:5509`)
  are the fused/paired dispatch sites. **Note:** ds4's plain kernel and ours are STRUCTURALLY
  NEAR-IDENTICAL (warp-per-row, 8 warps/block, 34-B blocks, dp4a, unaligned i8×4) — confirming
  the gap is NOT in the inner dot (Bricks 3/4 were right) but in **how the launches are packed**.
- **Ours:** `src/vt/cuda/cuda_quant_dot.cu` `QuantDotGemmQ8_0Kernel` (Brick-3 dp4a `GetIntB2`,
  `dim3(32,kWarpsPerBlock)`), dispatched per-tensor from `MatmulQ8_0Cuda`.
- **MHC re-read is NOT the mechanism:** `deepseek_v4.cpp` MHC expands the residual to `[hc,H]`
  but the byte-impossibility proof (§1) shows we read the tower exactly once; our 646 vs 259 is
  finer row-splitting, not extra bytes.

---

## 3. ROOT CAUSE (evidence-first, brutally honest — 4 levers already failed)

**Why is ds4 at 90% and we at 67% on byte-identical reads?** Not the inner dot (identical dp4a).
Not bytes (proven equal). Not L2 (couldn't measure, but achieved-BW subsumes it). The measured
signature is **launch granularity**:

- ds4: **259 launches/step**, each big and long-lived — pairs two weight streams behind one
  activation load (`pair`, 38.8 µs/matmul-equiv vs our 63.4 = **1.63×**), grows the grid by
  grouping (`grouped_a`), and folds the residual epilogue in (`hc_expand`). Bigger grids stay
  resident long enough to keep DRAM **saturated → 90%**.
- ours: **646 launches/step**, each a single small tensor. A short warp-per-row GEMV spends a
  large fraction of its life in grid ramp-up / tail-drain where DRAM is **under-subscribed**, and
  we pay that 646×/step → **67%**.

This is exactly the axis Brick 8's "corrected next lever" and Brick 11 *named*
("multiple-outputs-per-warp / batched/persistent so one launch reuses the activation across rows")
but **never actually built** — Brick 11 built **sub-warp lane-splitting** (splitting a warp
*within* a launch), the WRONG structure, and correctly measured it flat. Brick 9-Step0's
`kWarpsPerBlock 4→8` was also *within-launch* occupancy → also flat. **Nobody had changed the
launch COUNT/SIZE** — the one thing ds4 does differently, and the one thing that moves the 67%→90%.

---

## 4. VERDICT — (A) PORTABLE LEVER FOUND. Re-opens the path.

**The Q8_0 GEMV is memory-bound but at 67% of roofline with ~1.35× measured headroom (proven
reachable — ds4 sits at 90% on the same bytes). Brick 11's "irreducible 41 ms roofline" is
REFUTED.**

### What to build (ranked; mirror ds4, ground each), for a follow-on lane
1. **Q8_0 projection PAIRING** — port `matmul_q8_0_pair_preq_warp8_kernel` (`ds4_cuda.cu:4485`):
   one kernel, two weight pointers, one shared pre-quantized activation → two outputs. Apply to
   dense Q8_0 projections that consume the same activation (MLA `q_a`+`kv_a`; shared-expert
   siblings). **Bit-exact** (each output's integer-dot order is preserved). Measured class win 1.63×.
2. **Fold the MHC/residual expand into the Q8_0 matmul epilogue** — port
   `matmul_q8_0_hc_expand_preq_warp8_kernel` (`:4762`): removes a separate glue kernel + its HBM
   round-trip and enlarges the launch. Residual reassociation ⇒ characterize as **near-tie**.
3. **Consolidate the 646 row-split launches** into fewer, bigger per-tensor launches (grow grid.x,
   mirror `grouped_q8_0_a`) so DRAM stays subscribed across the step — the general form of the fix.

### Honest expected gain (not over-sold)
- **Full capture** (reach ds4's 90%): 40.93 → **30.4 ms/step**, −10.5 ms → GPU-active
  82.6→72.1 ms → **~13.9 t/s** (from 12.1); wall ~11.7 → **~13.3**.
- **Half capture** (~78% BW, ~35 ms): −6 ms → **~13 t/s**.
- This lever **does NOT reach 16.5 alone** — the MoE-expert + glue residual is another ~15 ms/step
  (a separate front). But it is a **real, measured, portable +1.5–2 t/s**, it is the single
  biggest remaining Q8_0 win, and it **re-opens the 12.7→16.5 ladder** that Brick 11 had declared
  closed at ~13.

### Premise correction (honesty)
The campaign's "ds4's Q8_0 is ~19.8 ms/step faster" was an older whole-step-derived estimate. The
**direct** measurement is a **10.5 ms/step** Q8_0 gap (ds4 30.38 vs ours 40.93). Direction
confirmed; magnitude corrected; ds4's Q8_0 sits at 90% roofline (217 GB/s), not the whole-step
~139 GB/s average that mixed in our slow MoE.

---

## 5. Reproduce
```
# ds4 reference (repeat with -n 4 for the diff baseline):
cd ~/w8run/ds4 && flock $HOME/gpu.lock env DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=28000 \
  nsys profile --cuda-graph-trace=node -o /tmp/ds4_ref -f true \
  ./ds4 -m ds4flash.gguf --cuda --gpu-vram 90 -p "The capital of France is" -n 60 --temp 0
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/ds4_ref.nsys-rep
# ours (Lever-1 build; -n 60 and -n 4):
cd ~/w8run/ds4lever1/build-cuda && flock $HOME/gpu.lock \
  env VT_V4_RESIDENT_DECODE=1 VT_V4_DECODE_GRAPH=1 \
  nsys profile --cuda-graph-trace=node -o /tmp/ours60 -f true \
  ./examples/deepseek-v4-gen --gpu --model <gguf> --prompt "The capital of France is" --max-tokens 60 --kv-cache
# byte anchor:
./ds4 -m ds4flash.gguf --inspect   # → q8_0: 345 tensors, 6.15 GiB
```
Counter path (BLOCKED, recorded for the record): `ncu --metrics dram__throughput... ./ds4 …`
→ `ERR_NVGPUCTRPERM` (no admin; needs `NVreg_RestrictProfilingToAdminUsers=0` + reboot).
