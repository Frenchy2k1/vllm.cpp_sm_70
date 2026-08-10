# ROCm gfx1200 (RX 9060 XT) — M2/M4: real-oracle-verified correct on Qwen3-0.6B and Gemma-3-1B-it

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`).
**Claim:** unclaimed — this spec documents a closed investigation, not a landed
fix (there is nothing to fix — ROCm's output was correct all along; see
Outcome, corrected 2026-08-10 once a real vLLM-ROCm oracle became available).
**Issue:** [#269](https://github.com/mudler/vllm.cpp/issues/269).
**Origin:** [issue #41](https://github.com/mudler/vllm.cpp/issues/41), the
gfx1200 M0/M1 report, first M2 attempt on this board. Distinct defect from
#201 (`hipblasGemmEx` overload) and #132 (`-O0` teardown deadlock), neither of
which reproduced here (and both since fixed on main, `ce134e1d`, picked up
when this investigation rebased onto `origin/main` 2026-08-10).
**Board:** AMD Radeon RX 9060 XT (`gfx1200`, Navi 44, RDNA4, discrete — no
reference tier). ROCm 7.2.3 (nixpkgs `rocmPackages.*`), hipClang/Clang 22.0.0.
M0/M1 independently MET on this board (separate report to #41).

---

## Outcome (2026-08-10, corrected same day — see below)

**CORRECTION (2026-08-10, later same day): the conclusion below was wrong
about which side is right.** It was written without a real oracle available
(the "not attempted, hardware/environment-blocked" note further down was
literal). A real vLLM-ROCm oracle was subsequently stood up on this exact
board (§ Real-oracle verification) and it produces **`' 1000000'`
deterministically, 5/5 runs** — the same output our own ROCm backend
produces, and *not* the `Paris` output our own CPU backend produces. **ROCm
was correct. CPU is the outlier on this input.** The near-tie mechanism
described below (two backends landing on opposite sides of an extremely tight
logit cluster) is still the right *mechanism*; what was wrong was the
unstated assumption that CPU's answer was the ground truth to measure ROCm
against. It wasn't — vLLM-the-reference is, per this project's own standing
rule, and on this input vLLM agrees with ROCm. The paragraph below is kept
verbatim as the mechanism analysis; read it with CPU and ROCm's roles
reversed from how it frames them.

**A genuine, extremely tight near-tie that two backends resolve to opposite
sides, amplified into visible garbage because the losing token happens to be
content-free — original (partially superseded) framing follows:**

`Qwen3ForCausalLM` (Qwen3-0.6B bf16), prompt `'The capital of france is'`,
greedy, `--temperature 0`. Final-position top-5 logits:

| | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| CPU | `ĠParis` 15.1216 | `Ġlocated` 15.0541 | `Ġ` 15.0483 | `Ġthe` 14.9142 | `Ġin` 14.5216 |
| ROCm | `Ġ` 15.1009 | `ĠParis` 15.0907 | `Ġlocated` 15.0620 | `Ġthe` 14.9496 | `Ġin` 14.5309 |

Same five candidates, same magnitudes, clustered within ~0.07 logit units of
each other on both backends (max logit ≈15). CPU resolves the cluster toward
`Paris` by 0.0675 over its runner-up; ROCm resolves it toward `Ġ` (a bare
leading-space token — content-free) by just 0.0102 over `Paris`. Picking a
content-free token derails the rest of greedy decoding into out-of-distribution
territory, which is why the completion reads as total garbage (` 1000000`)
rather than a merely-different-but-plausible word — the mechanism is an
ordinary near-tie flip, not a broken computation.

**Per-layer activation drift** (L2 norm of the last token's hidden row after
every decoder layer, CPU vs ROCm, same prompt):

| Layer | CPU L2 | ROCm L2 | rel. diff |
|---|---|---|---|
| L0 | 3.876 | 3.886 | 0.271% |
| L1 | 4.867 | 4.869 | 0.035% |
| L2 | 4.597 | 4.599 | 0.039% |
| L3 | 6.253 | 6.277 | 0.381% |
| L4 | 5.542 | 5.536 | 0.109% |
| L5 | 5.952 | 5.942 | 0.155% |
| L6 | 8.451 | 8.422 | 0.334% |
| L7 | 11.707 | 11.715 | 0.070% |
| L8 | 9.796 | 9.787 | 0.101% |
| L9 | 10.276 | 10.247 | 0.279% |
| L10 | 14.790 | 14.715 | 0.503% |
| L11 | 11.852 | 11.822 | 0.248% |
| L12 | 12.502 | 12.430 | 0.582% |
| L13 | 12.488 | 12.415 | 0.582% |
| L14 | 12.563 | 12.564 | 0.009% |
| L15 | 16.946 | 16.993 | 0.275% |
| L16 | 20.044 | 20.022 | 0.109% |
| L17 | 31.010 | 31.030 | 0.064% |
| L18 | 30.977 | 31.031 | 0.173% |
| L19 | 43.472 | 43.698 | 0.521% |
| L20 | 53.452 | 53.369 | 0.155% |
| L21 | 66.609 | 66.510 | 0.148% |
| L22 | 79.621 | 80.029 | 0.513% |
| L23 | 85.998 | 85.555 | 0.515% |
| L24 | 127.735 | 128.331 | 0.466% |
| L25 | 104.664 | 104.763 | 0.095% |
| L26 | 112.977 | 112.578 | 0.353% |
| L27 (final) | 216.317 | 217.610 | 0.598% |

Divergence is already present at **layer 0** (0.271%) — rules out "something
breaks at layer N" entirely. It oscillates in a tight 0.01%–0.6% band for all
28 layers with no jump, no blow-up, sometimes CPU higher and sometimes ROCm
higher. This is the textbook signature of ordinary bf16 rounding +
reduction-order noise (hipBLASLt's GEMM accumulation order vs. CPU's fixed
sequential order accumulating independently at every op), not a localized
defect anywhere in the stack. It happens to be enough, by the final layer, to
tip an unusually thin 0.07%-of-max-logit tie the other way.

**What was ruled out along the way**, in order:
1. The `is_cuda()`-vs-`is_cpu()` host-pointer-aliasing defect at
   `dense_attn_block.h:181` (`ResidentWeight`) — already fixed generically,
   confirmed correct for `kROCM` in current source.
2. The embedding gather itself — pulled `model.embed_tokens.weight` directly
   from the safetensors file, ran `vt::Embedding` on CPU and ROCm at the real
   row indices 47587 ("france") and 9625 ("France"), both **bit-exact**
   against the CPU oracle (see `scratch_diag_embed.cpp`, not for landing).
3. A discrete localized op bug — the per-layer drift table above shows none;
   drift is present from layer 0 and grows uniformly.

**No longer pending — a real oracle was stood up same-day.** The
teacher-forcing decisive-measurement method
(`scripts/qwen3-neartie-gap-transformers.py`) still hasn't been run (it wants
the exact pinned vLLM commit `555967922`/0.26.0.dev0, and what's available
here is a newer prebuilt AMD image, 0.19.1 — see below), but a direct greedy
comparison against a genuinely independent, real vLLM-ROCm install on this
same board is strictly stronger evidence than eyeballing logit margins, and
it settles the question the logit-margin analysis above could only gesture
at: which side of the tie is *actually* right.

**Debug instrumentation used, then reverted** (not landed, not left in the
tree): two `VT_DEBUG_LOGITS=1` hooks in `sampler.cpp` (`greedy_argmax_host`
and the async `Sampler::forward` path — CPU and ROCm take different code
paths by default, both needed instrumenting for a fair comparison) and one
`VT_DEBUG_LAYERS=1` hook in `qwen3.cpp`'s `ForwardLayers` loop.

## Real-oracle verification (2026-08-10)

Docker was the practical path here — pip/PyTorch-vs-NixOS's non-FHS layout is
what made earlier oracle attempts finicky, and Docker sidesteps it entirely.
AMD ships a prebuilt image **explicitly targeting this arch family**:

```sh
docker pull rocm/vllm:rocm7.13.0_gfx120X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1   # 10.8 GB
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --shm-size=8g -v <model-dir>:/models/<name>:ro \
  rocm/vllm:rocm7.13.0_gfx120X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  python3 -c '<vllm.LLM(...) offline-inference script>'
```

`torch.cuda.get_device_properties(0).gcnArchName` inside the container reports
`gfx1200` and `current_platform` resolves to `RocmPlatform` — a real,
independent PyTorch/vLLM stack on the same silicon, not our own code twice.
**Caveat, stated plainly:** this is vLLM **0.19.1**, not this project's
pinned CUDA-oracle commit (0.26.0.dev0/`555967922`) — no gfx120X-targeted
image exists yet at that pin (RDNA4 is too new). Treated as a first-pass
"does a real oracle even agree with us at all" check, not a final SACRED gate;
building the exact pinned commit from source for gfx1200 (no prebuilt path
exists) is separate, unstarted follow-on work.

**Qwen3-0.6B, prompt `'The capital of france is'`, greedy, K=5:** oracle
outputs `' 1000000'` (`[220, 16, 15, 15, 15, 15, 15, 15]`) **every run,
identical** — deterministic, not itself near a tie at the sampling level.
Matches our ROCm backend exactly. Diverges from our CPU backend (`' Paris.
The capital of the United States'`).

**Gemma-3-1B-it, the same 6-prompt battery used above:** oracle output is
**byte-identical to both of our backends on every prompt** (` Paris.\n\nThe
largest city in France`, ` Jupiter, and it's a truly`, ` 100 degrees
Celsius.\n\n`, ` Au.\n\nThe chemical formula for gold`, ` George
Washington.\n\nThe second president of`, ` blue,\nI like to eat a`) — full
text and token ids, not just the last printed line.

## What this means for the row

**M4, not just M2, is now real for a narrow but genuine slice.** Gemma-3-1B-it
is real-oracle greedy-token-exact on gfx1200, 6/6 prompts. Qwen3-0.6B's ROCm
output matches the real oracle too, deterministically — it is **our CPU
backend**, not ROCm, that diverges from ground truth on this specific
near-tie prompt. That CPU-side divergence is a new, separate, small open
question (likely still ordinary bf16/reduction-order drift, just landing on
the wrong side of an extremely tight cluster on this box's CPU kernel path —
not established as a "bug" so much as an unresolved direction of a very
narrow tie) — worth a note on the CPU backend somewhere, but it is not a
ROCm-row defect and does not block this row.

## Broadening M2: `Gemma3ForCausalLM` (gemma-3-1b-it), 2026-08-10 — clean pass, no near-tie

Same board (gfx1200, RX 9060 XT), same `build-hip` (no rebuild needed — already
current), same `--device cpu` vs `--device auto` (`auto` resolves to
`device=5`=`kROCM`, confirmed, not a silent CPU-reference-tier fallback: every
op in the forward reports `selected=vt-native` under
`VT_OP_PROVIDER_STATS=1`). Model: `unsloth/gemma-3-1b-it` (ungated mirror of
the gated `google/gemma-3-1b-it` the SACRED gate uses; `model.safetensors`
SHA-256 `3d4ef8d7…8516b6` matches the upstream blob hash exactly — same
weights, different host). Config confirmed matching the sweep-gemma spec's
expected shape: `head_dim=256`, `final_logit_softcapping=null`,
`sliding_window=512`/pattern 6, dual rope theta (`rope_theta=1e6`,
`rope_local_base_freq=1e4`).

Greedy (`--temperature 0`, `--max-tokens 8`), the same 6-prompt battery the
SACRED Gemma-3 gate uses (`scripts/gemma3-oracle-capture.py`):

| Prompt | CPU vs ROCm |
|---|---|
| "The capital of France is" | identical (` Paris.\n\nThe largest city in France`) |
| "The largest planet in our solar system is" | identical |
| "Water boils at a temperature of" | identical |
| "The chemical symbol for gold is" | identical |
| "The first president of the United States was" | identical |
| "Roses are red, violets are" | identical |

**48/48 tokens identical, CPU vs ROCm.** This is the first real exercise of
the gemma `(1+w)` RmsNorm code path (`vt::RmsNorm{gemma=true}` — sandwich
norms + QK-norm, `rocm_rmsnorm.hip:69-110`) plus GeGLU
(`kGeluAndMul`/`rocm_ops.hip`) and the dual per-layer RoPE-theta routing on
this board, and unlike the Qwen3-0.6B M2 attempt above it hits **no near-tie
anywhere in the battery** — a clean unanimous match rather than a coin-flip on
an unusually tight cluster.

**Upgraded to a real-oracle result same day** (§ Real-oracle verification):
the independent vLLM-ROCm oracle matches all 6/6 prompts byte-identically too,
not just our own two backends agreeing with each other. This is the stronger
claim — M4, not just M2 — for this model on this board.

No kernel changes were needed or made; this is a validation run only, done
from a separate worktree against the already-built `build-hip` in the primary
checkout.

## Environment note (reproducibility)

This board is accessed through `nix develop .#rocm-shell` (local, uncommitted
`flake.nix` addition — see the M0/M1 report). `ROCM_PATH` is nixpkgs' `clr`
output (has `lib/cmake/hip-lang`); hipBLAS/hipBLASLt/hipblas-common are merged
into a small writable overlay at `~/.cache/vllm-cpp-rocm-overlay` because
nixpkgs ships them as separate store paths. None of this affects the finding —
it reproduces identically build-to-build, deterministic at `--temperature 0`.
