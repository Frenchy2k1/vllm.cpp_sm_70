# MUSE-GLIMMER-VISION-ATTN-FLASH — the perception encoder's 50 layers all ran on the correctness-grade kernel, on a path nothing calls yet

Row: `MUSE-GLIMMER-VISION-ATTN-FLASH`, against the model-matrix row
`MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`
([`muse-glimmer.md`](muse-glimmer.md)).
Issue: [#1545](https://github.com/mudler/vllm.cpp/issues/1545), an instance of
the class issue [#1544](https://github.com/mudler/vllm.cpp/issues/1544).
Sibling instance, on the only live victim:
row `LTX25-DIT-ATTN-FLASH` / [#1549](https://github.com/mudler/vllm.cpp/issues/1549),
whose spec `ltx25-dit-attn-flash.md` is not on `main` yet, which is why it is
named here rather than linked.

Every bare `file:line` anchor below is read at this row's base, **`04f1cead6`**,
which is `origin/main` at the claim.

## Now

`DONE`. The one production line is changed, a routing test that fails without it
is in the suite, and every existing Muse Glimmer golden is byte-for-byte
unchanged because the CPU dispatch for both ops is the same function pointer.
What this row does **not** carry is a CUDA measurement: no GPU lease was taken,
`dgx:gpu0` is held, and §5 records the CUDA numerics change and §8 records the
A/B as owed to whoever wires the encoder.

## 0. The one-sentence finding

`src/vllm/model_executor/models/muse_glimmer_vision.cpp:639` called
`vt::Attention` for every one of the perception encoder's 50 layers, that op is
frozen on the kernel whose own header at `src/vt/cuda/cuda_ops.cu:1456-1460`
calls itself "Correctness-grade (M0.9)", and the tower has no other attention
path, no knob, and no A/B rung to escape onto.

## 1. Why this row exists although nothing calls the code

This is the part that governs how the change lands, so it comes before the
arithmetic.

`MuseGlimmerVisionForward` is reached only from
`MuseGlimmerEncodePixelGroups` (`muse_glimmer_mm.cpp:191-203`) and
`MuseGlimmerGenerateGreedyViaRegistry` (`muse_glimmer_mm.cpp:276-345`). Neither
has a caller in `src/`, in `examples/`, or in `include/vllm.h`; the only callers
in the tree are `tests/vllm/models/test_muse_glimmer_wiring.cpp:724,804,805,864,883`
and `tests/vllm/models/test_muse_glimmer_vision.cpp:308,348`. The registry TU
states the same thing in prose at `muse_glimmer_registry.cpp:13-14`: *"The
perception encoder is still W3, so an image or video prompt is a pending
brick."*

So this row cannot prove reachability from a production entry point and does not
pretend to. It is a **latent-cost repair on an unreached staged slice**, and the
argument for doing it now rather than later is that the cost is not latent for
long: the moment W4 wires an image prompt into `ModelRegistry::Forward`, this
becomes a multi-second — on the shipped geometry, a multi-*minute*, see §3 —
time-to-first-token stall that will be discovered by whoever did the wiring
rather than by whoever caused it. Fixing it before the wiring costs one line.
Fixing it after costs a regression hunt through a change that did not touch
attention.

`.agents/reachability.md`'s staged-slice exception is taken deliberately and its
four obligations are all met: the commit body and the pull request body name what
is unreached, `## Owed` below names it, the owning row is
`MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`, and the issue
that tracks the wiring is [#1566](https://github.com/mudler/vllm.cpp/issues/1566),
filed by this row because the model's umbrella issue
[#268](https://github.com/mudler/vllm.cpp/issues/268) is CLOSED and nothing open
tracked the gap.

## 2. The call site, and why it maps onto the target op

`muse_glimmer_vision.cpp:611-641` is the whole attention block. Its properties,
all read at the base:

| property | value | anchor |
|---|---|---|
| ops called | exactly one, `vt::Attention` | `:639`, sole `vt::Attention` in the file |
| causal | never | `:632`, `AttentionArgs{scale, /*causal=*/false}` |
| shape | `[seg, nh, hd]`, `hk == hq`, square | `:606-608` + `RowSlice` at `:635-638` |
| head_dim | 96 | `hidden_size 1536 / num_attention_heads 16`, `muse_glimmer_vision.h:67-68,88` |
| layers | 50 | `muse_glimmer_vision.h:69` |
| knob | none | no `getenv` in the file |

The block-diagonal mask is **not** a mask in this implementation. It is already
expressed as a loop over `RowSlice` segments (`:633-641`), each of which is a
separate dense non-causal call over one image (full-attention layers) or one
`pos_emb_height x pos_emb_width` window (window layers). Nothing about that
structure interacts with the kernel choice, so the swap is confined to which op
the loop body names.

`vt::AttentionDenseFlash` (`include/vt/ops.h:3317-3318`, registered at
`src/vt/cuda/cuda_ops.cu:3839-3840`) accepts exactly this contract: rank-3
`[T,H,D]`, `key.shape[0] == query.shape[0]`, GQA broadcast with `hq % hk == 0`,
one shared float dtype, `args.causal` honoured (`cuda_ops.cu:3277-3296`). Every
`VT_CHECK` in `src/vt/ops.cpp:3083-3106` is satisfied by the tensors already
being passed.

`vt::AttentionDenseFa2` is **not** usable and was not considered further: its
fast path requires `head_dim == 64` (`src/vt/cuda/cuda_ops.cu:3395-3400`), and
at 96 it would fall straight through to `AttentionDenseFlash` anyway, so naming
it would only obscure which kernel runs.

## 3. The cost, with the resolution PINNED rather than assumed

[#1545](https://github.com/mudler/vllm.cpp/issues/1545) labelled its token count
as inferred and asked whoever took the row to pin the real resolution first,
because it moves the estimate quadratically. It was cheap to pin: the released
checkpoint is already on the NAS at
`/mnt/nas_share/checkpoints/muse-glimmer-30b/`, no download needed.

**Read from `config.json`** (`vision_config`), and it agrees with our defaults
key for key: `hidden_size 1536`, `num_attention_heads 16` (head_dim 96),
`num_hidden_layers 50`, `patch_size 14`, `merge_size 2`, `pos_emb_height 32`,
`pos_emb_width 32`.

**The 13/37 layer split is now READ, not inferred.** `layer_types` ships
explicitly, length 50, with `full_attention` at indices
`3, 7, 11, ..., 47, 49` — 13 full and 37 `window_attention`. That is exactly
what `muse_glimmer_weights.cpp:429-439` computes when the key is absent
("full every 4th layer AND on the last"), so the fallback rule is confirmed
correct against the shipped file rather than merely plausible.

**The window is 32x32 = 1024 patches**, from `pos_emb_height/width` through
`MuseGlimmerVisionSparsePermutation` (`muse_glimmer_vision.cpp:355-382`). This
is load-bearing for the cost model and #1545 did not have it: a window layer
does **not** attend over the whole image, it attends within one 1024-patch
block.

**The resolution.** `processor_config.json` sets
`image_processor.max_image_tokens = 4096`, and the checkpoint README's overview
table says "Max visual tokens per image | 4,096". Whether that counts
post-merge tokens (what the language model sees) or patch tokens (what the tower
runs on) is **not** determinable from the checkpoint alone — the tree has no
image processor for this model and `transformers` is not checked out here. Both
readings are carried:

| reading | patch tokens | grid | windows | naive cost per tower forward |
|---|---:|---|---:|---:|
| A: 4096 post-merge (merge 2, so x4) | 16,384 | 128x128 | 16 | **375 s** |
| B: 4096 patch tokens | 4,096 | 64x64 | 4 | **34 s** |
| #1545's illustrative 448x448 | 1,024 | 32x32 | 1 | 4.8 s |

Reading A is the more likely one, and the architecture is the argument:
`window_attention` over a 32x32 block does nothing at all unless the grid
exceeds 32x32. At 448x448 the grid *is* one window, so all 50 layers compute the
identical thing and 37 of the checkpoint's layer-type entries are inert. A tower
that ships 37 window layers is a tower meant to run at grids well past one
window.

Arithmetic, at `Q * H * K_eff` iterations and the 5.70 ns/iteration GB10
constant stated in [#1544](https://github.com/mudler/vllm.cpp/issues/1544) and
derived there from `.agents/specs/multimodal-speed.md:24-26`, which records 56 ms
per block over 784 patches and 16 heads rather than the constant itself
(`56e-3 / (784 * 16 * 784) = 5.694e-9`). The head_dim 64 and 72 provenance is in
that issue too. The tree holds the second anchor's underlying measurement at
`.agents/benchmark-record.md:4654` — the Voxtral Whisper encoder forward over
1500 frames and 32 layers at 8870 ms — and only the 8.21 s prediction and the
"within 7%" comparison are the issue's alone:

    reading A: 13 x (16384 x 16 x 16384)              = 5.583e10
             + 37 x (16 x (1024 x 16 x 1024))         = 9.932e9
                                                        6.576e10 x 5.70 ns = 375 s

    reading B: 13 x (4096 x 16 x 4096)                = 3.489e9
             + 37 x (4 x (1024 x 16 x 1024))          = 2.483e9
                                                        5.973e9 x 5.70 ns  = 34.0 s

The 5.70 ns constant is anchored at head_dim 64 and 72 and its transfer to 96 is
**inferred**, exactly as #1544 labels it. Nothing in this row depends on the
constant being right to better than a factor: at reading B the naive kernel
already costs half a minute per image, and at reading A it costs six minutes.

## 4. What changes

One line, `muse_glimmer_vision.cpp:639`:

```
-        vt::Attention(q, os, qs, ks, vs, aargs);
+        vt::AttentionDenseFlash(q, os, qs, ks, vs, aargs);
```

plus the comment above it that says why, mirroring the shape of
`qwen3_vl_vision.cpp:462-480` and `whisper_audio.cpp:310-322`, which
[#1544](https://github.com/mudler/vllm.cpp/issues/1544) names as the intended
pattern.

**No A/B knob is added, and that is deliberate.** Qwen3-VL and Whisper each keep
a `VT_*_EAGER` escape to the naive arm because each has a measured result to
defend and a live path to measure on. This tower has neither: it is unreached, so
a knob here would add a second dead arm and a second untested branch to a file
that already has no caller. When W4 wires the encoder and someone takes the A/B
in §8, adding the knob is a two-line change made *with* the measurement in hand.

**`src/vt/cuda/cuda_ops.cu` is deliberately NOT touched.**
[#1544](https://github.com/mudler/vllm.cpp/issues/1544) item 2 owes a repair to
`LaunchAttentionDenseFlash`'s advertised `head_dim <= 256` contract, and
row `LTX25-DIT-ATTN-FLASH` ([#1549](https://github.com/mudler/vllm.cpp/issues/1549))
§4.3 is implementing it.
Two rows editing one kernel launcher is the shared-file lock this repository's
records rules exist to avoid. §6 records why 96 does not need that repair to be
correct today.

## 5. Numerics: what is bit-identical and what is not

This has two answers and they must not be collapsed into one.

**On CPU the swap is byte-identical, by construction and not by tolerance.**
`src/vt/cpu/cpu_ops.cpp:3760-3761` registers `OpId::kAttentionDenseFlash` on
`DeviceType::kCPU` to `&AttentionKernel` — the *same function pointer*
`OpId::kAttention` is registered to at `:3750-3751`. The comment there says so:
"Flash-tiled dense attention is a CUDA shared-memory optimization;
byte-identical to kAttention on CPU." Every Muse Glimmer golden in the tree is a
CPU golden, so all of them are unchanged bit-for-bit, and that is what the gate
run in §7 observes.

**On CUDA the swap is NOT bit-identical, and no CUDA run was taken here.**
`AttentionDenseFlashKernel` (`cuda_ops.cu:3239-3325`) is one warp per
(query, head) with the head_dim split into per-lane strips reduced by
`__shfl_xor_sync`, carrying a running online-softmax `(m, l, acc)`.
`AttentionKernel` (`cuda_ops.cu:1463`) is a 256-thread block per (query, head)
with a shared-memory tree reduction per key. The arithmetic is the same f32
softmax; the **partial-sum grouping and the reduction order are not**, so the
two differ within the f32 envelope. `include/vt/ops.h:3304-3318` states exactly
this and says adoption is per token-exact gate.

Three things follow, and this row states all of them rather than the convenient
one:

1. The change **does** move CUDA numerics on this path.
2. It moves them onto the rung Whisper's encoder and the Qwen3-VL tower already
   run on by default, which is where the tree's other non-causal encoder towers
   already are.
3. It is bit-identical to `AttentionDenseFast` (`ops.h:3313-3314`: "the CUDA
   output is BIT-IDENTICAL to it (K/V bytes merely sourced from shared memory)").

No tolerance anywhere was widened to accommodate any of this. The existing
`rel_l2 < 1e-6` f32 bound and `rel_l2 < 2e-2` bf16 bound in
`tests/vllm/models/test_muse_glimmer_vision.cpp` are untouched and still pass at
the same numbers, because on CPU nothing moved at all.

The CUDA-side confirmation is **owed**, not claimed. It is recorded in `## Owed`
and in §8, and it is not fabricated as a pass.

## 6. Shared memory at head_dim 96, for both arms

`LaunchAttentionDenseFlash` requests `2 * kFlashBc * d * sizeof(Tin)` bytes of
dynamic shared memory (`cuda_ops.cu:3338`) with `kFlashBc = 64`
(`cuda_ops.cu:3236`), and `cuda_ops.cu` contains no `cudaFuncSetAttribute`, so
the real cap is the 48 KiB every architecture gives without opting in — the
defect [#1544](https://github.com/mudler/vllm.cpp/issues/1544) item 2 records.

The scope of that sentence is `cuda_ops.cu` and not `src/vt/cuda/`, deliberately.
Seven files under `src/vt/cuda/` do call `cudaFuncSetAttribute` — `cuda_gdn.cu`,
`cuda_marlin_repack.cu`, `cuda_mla_attn.cu`, `cuda_paged_attn.cu`, the two other
Marlin translation units, and `flash_attn/src/flash_fwd_launch_template.h`, which
is inside the very subtree `ops.h:3309-3310` says this kernel was ported from.
None of them is on this launch path, and the claim the argument needs is the
narrow one.

The kernel body also declares **no static `__shared__`** (`cuda_ops.cu:3239-3328`
holds one `extern __shared__` array and nothing else), so nothing competes with
the dynamic request. That check matters: one static byte would put the f32 arm
over the cap.

| arm | `sizeof(Tin)` | request | vs the 49,152 B cap |
|---|---:|---:|---|
| bf16 — the production dtype, `muse_glimmer_vision.h:86` | 2 | 24,576 B | 50% of it |
| f32 — the per-stage gate arm, `muse_glimmer_vision.h:84-86` | 4 | 49,152 B | **exactly on it** |

The default arm has a factor of two in hand. The f32 arm lands on the boundary
to the byte. It is a `<=` comparison and 49,152 B is the guaranteed default, so
it should launch; the kernel fails loud if it does not
(`Check(cudaGetLastError(), ...)` at `cuda_ops.cu:3352`), so the failure mode is
a diagnosable refusal and never a silent wrong answer. That claim is **not
measured here** — no GPU lease — and the f32 arm has never run on CUDA in this
tree in any case, since every f32 Muse Glimmer test is a CPU test. The
`LTX25-DIT-ATTN-FLASH` ([#1549](https://github.com/mudler/vllm.cpp/issues/1549))
§4.3 repair removes the
question entirely by opting in to the larger cap, and this row is a beneficiary
of it rather than a blocker on it.

## 7. Tests and gates

**The problem this row had to solve first:** no CPU test can detect the routing
change through its *output*, because §5's first paragraph makes the two ops the
same function on CPU. A test that compared numbers would pass before the change
and after it, which is a test that measures nothing
([[a-mutation-that-never-applied-reads-as-a-passing-test]]).

The instrument that does see it is the op-provider selection counter, and the
tree already uses it for exactly this question at
`tests/vllm/models/test_ltx2.cpp:665-686`. New case
`muse_glimmer_vision_tower_routes_attention_to_dense_flash` in
`tests/vllm/models/test_muse_glimmer_vision.cpp`:

- `vt::EnableOpProviderCallStats(true)`, read `selections` for
  `kAttentionDenseFlash` and `kAttention` on `kCPU` before and after one
  `MuseGlimmerVisionForward` over the existing fixture;
- assert `kAttentionDenseFlash` gained **exactly 12** and `kAttention` gained
  **exactly 0**.

12 is derived, not observed: the fixture is 3 layers `{window, full, window}`
over two images whose grids are 6x6 and 4x4 with a 4x4 window
(`test_muse_glimmer_vision.cpp:64-78,122-134`), so the window segmentation is
`[16, 8, 8, 4]` + `[16]` = 5 calls and the full segmentation is `[36]` + `[16]`
= 2 calls: `5 + 2 + 5 = 12`. Asserting the derived count rather than a
before/after inequality is what makes the case sensitive to a partial routing
change, and it pins the segmentation as a side effect.

**Red-first evidence, observed.** With the case in the suite and
`muse_glimmer_vision.cpp:639` still on `vt::Attention`:

    :395: MESSAGE: attention selections: dense-flash +0, naive +12
    :397: ERROR: CHECK( flash_after - flash_before == 12ull ) values: CHECK( 0 == 12 )
    :398: ERROR: CHECK( naive_after - naive_before == 0ull ) values: CHECK( 12 == 0 )
    [doctest] assertions: 5 | 3 passed | 2 failed | Status: FAILURE!

The exact inversion the derivation predicts, and 12 is confirmed by the run
rather than only by the arithmetic. After the one-line change, the whole file:

    :395: MESSAGE: attention selections: dense-flash +12, naive +0
    [doctest] test cases: 8 | 8 passed | 0 failed | 0 skipped
    [doctest] assertions: 103 | 103 passed | 0 failed | Status: SUCCESS!

`tests/vllm/models/test_muse_glimmer_wiring.cpp` is green on the same tree at
9/9 cases and 10,317 assertions.

**The byte-identity claim of §5 is measured, not only argued.** Both binaries
were built from one tree with only that line differing, and every `MESSAGE:` line
of the suite was captured from each and diffed. Exactly one line of fourteen
differs, and it is the routing counter:

    13c13
    < attention selections: dense-flash +0, naive +12
    > attention selections: dense-flash +12, naive +0

The other thirteen — `ln_pre` and `block 0` and the tower and the adapter in f32
at `rel_l2` 1.201e-07 to 2.983e-07, and the tower in bf16 at
`rel_l2=5.951e-03 max_abs=3.675e-02` — are byte-for-byte equal across the two
binaries. That is agreement to the four significant digits doctest prints, on top
of the structural argument in §5 that the CPU dispatch is one function pointer.

**Gate:** `scripts/agent-preflight.sh`, run by this row on its own tree. CPU
only. No GPU axis is claimed, opened, or needed for what this row asserts.

**No public document changes, and that is the correct answer rather than an
omission.** AGENTS.md's projection table triggers on a lifecycle change, a
measurement, or a user-visible surface. This row changes none of them: it takes
no measurement (§8), it moves no roadmap row's state, and the surface it touches
is not user-visible because nothing reaches it (§1). `docs/FEATURES.md:150`
already records the vision arm's real state — "vision: **no reference run of any
kind**" — and a line saying the unreachable tower now names a faster op would be
a claim about a capability no user can invoke.

## 8. The A/B, which is OWED and not taken

The confirming measurement — naive vs flash on the real geometry, same binary,
`dgx:gpu0` — is not takeable by this row: the box is held by the operator and
this row has no lease authority. It is also not *worth* taking until W4 lands,
because the tower has no production caller to measure through and a synthetic
harness would measure a kernel rather than a capability.

When it is taken it needs: the real image resolution resolved between §3's
readings A and B, both arms from one binary, and the tower forward timed end to
end rather than the kernel alone
([[component-speedup-is-not-system-speedup-fixed-serial-term]]). The recorded
precedent for the shape of the win is the Qwen3-VL tower at 14.3x
(2114 -> 148 ms/image, `bbaa182b6`), on a different geometry; it is cited as
precedent and is **not** this row's number.

## 9. Risks

| risk | handling |
|---|---|
| CUDA numerics move on an ungated path | Stated in §5 rather than hidden. Bit-identical to `AttentionDenseFast`, and the rung Whisper and Qwen3-VL default to. Confirmation owed. |
| f32 CUDA arm sits exactly on the 48 KiB cap | §6. Fails loud, never silent. Removed outright by the #1544 item-2 repair the sibling row carries. |
| The routing test's `12` goes stale if the fixture changes | It is derived in a comment beside the assertion, so a fixture change that breaks it says which number to re-derive and why. |
| Landing unreached code | §1. The staged-slice exception is taken explicitly with all four obligations met. |

## 10. Stop conditions

Stop and return `NEEDS_DECISION` if `AttentionDenseFlash` turns out not to
express the segmentation (it does — §2), if the swap moves a CPU golden (it
cannot — §5), or if a production caller for the tower appears (none exists —
§1). None of the three fired.

## Owed

- [#1566](https://github.com/mudler/vllm.cpp/issues/1566) — **the perception
  encoder has no production caller.** `MuseGlimmerEncodePixelGroups` and
  `MuseGlimmerGenerateGreedyViaRegistry` are reachable only from `tests/`, so an
  image or video prompt through `ModelRegistry::Forward` still hits the W3/W4
  brick at `muse_glimmer_registry.cpp:13-14`. This row's change lands *inside*
  that unreached slice and is the staged-slice exception in
  `.agents/reachability.md`. Owned by
  `MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`. Filed by this
  row because [#268](https://github.com/mudler/vllm.cpp/issues/268) is closed and
  nothing open tracked the gap.
- **The CUDA A/B is not taken** (§8). `dgx:gpu0` was held by the operator for
  the whole of this row and no lease was requested. Ready-to-measure, not
  measured: the recipe is in §8 and the geometry question it needs answered is
  in §3.
- **The f32 CUDA arm has never launched at 49,152 B** (§6). The bf16 default arm
  has a factor of two in hand; the f32 arm sits on the 48 KiB cap to the byte, and
  §6 argues it fits rather than measuring it. The argument is an inference about
  driver behaviour at the exact boundary and it does not account for any per-block
  driver shared-memory reservation. It is safe to carry today only because the arm
  is dead twice over — no production caller for the tower at all, and every Muse
  Glimmer f32 test is a CPU test — and because the failure mode is a loud
  `Check(cudaGetLastError(), ...)`. It stops being safe the moment W4 wires the
  encoder or anyone runs the f32 stage gate on CUDA, whichever comes first. Owed:
  one launch probe at head_dim 96 f32, or the `LTX25-DIT-ATTN-FLASH`
  ([#1549](https://github.com/mudler/vllm.cpp/issues/1549)) §4.3 opt-in landing
  first and removing the question. Recorded as debt rather than as a settled
  argument, because an inference quoted twice starts reading as a measurement.
- **The image resolution is still two readings, not one** (§3). Resolving it
  needs the upstream `MuseGlimmerImageProcessor`, which is neither in this tree
  nor in a local `transformers` checkout. It changes the size of the number and
  nothing about the correctness of the change.

## Outcome

Measured: nothing on CUDA, deliberately, and §8 says why. What was *established*
is the shape of the cost with the geometry pinned from the released checkpoint
rather than assumed (§3), which moved the headline from #1545's illustrative
4.8 s to 34 s or 375 s per image depending on a reading that is still open.

Rejected: `AttentionDenseFa2`, which refuses head_dim 96 and would silently be
`AttentionDenseFlash` anyway (§2). An A/B knob, because a second arm on a path
with no first caller is two dead arms (§4). Touching
`LaunchAttentionDenseFlash`'s shared-memory opt-in, because a sibling row is
already there and one kernel launcher edited by two rows is the exact
shared-file lock the records rules forbid (§4).

Why the default has its value: `AttentionDenseFlash` and not `AttentionDenseFast`
because the tower's whole problem is redundant global K/V re-reads over a
1024-to-16384-token non-causal context, which is the difference between those two
rungs; and unconditional rather than shape-gated because every shape this call
site can produce is inside the op's contract (§2).
