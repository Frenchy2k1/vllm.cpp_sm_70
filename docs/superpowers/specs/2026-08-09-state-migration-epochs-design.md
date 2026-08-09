# Structured-state migration epoch reconciliation

**Task:** `state-record-structure-1`

**Approved approach:** preserve the original migration and append exact source
epochs for legacy-state bytes that land on `main` before the structured cutover.

**Current reconciliation base:** `f921062ba4fbf3341b02d2ac826bb3d84cc91a64`

## Problem

The structured-state branch froze and migrated `.agents/state.md` at commit
`994cd8d4122ecf44f72d51fabd61c45adaaea9d3`. While the branch was under review,
`main` appended three more legacy checkpoints. The current file is 3,199,258
bytes: the frozen 3,191,283-byte source remains an exact prefix and 7,975 bytes
were appended.

Regenerating the original migration from the newer blob would rewrite the
provenance wrappers and migration metadata for 146 reviewed, immutable imports.
Ignoring the tail would make the cutover lossy. The reconciliation therefore
needs multiple exact source epochs while retaining one continuous byte stream.

The same upstream changes left the inherited environment-documentation gate
red. The developer approved classifying nine default-off implementation tactics
as kernel-internal and documenting three operational controls publicly.

## Source epochs

The migration has three independently pinned cumulative snapshots:

| Epoch | Commit | Blob | Bytes | SHA-256 |
|---|---|---|---:|---|
| Frozen | `994cd8d4122ecf44f72d51fabd61c45adaaea9d3` | `93a8d0da802a7ea7cbea4bee3bedffb4d90459f7` | 3,191,283 | `00c08e974724c19b5f79cce44df71c6fbfef4db32aa6acb545ef56546e3bb5e6` |
| Vulkan lm_head | `eee64f81812f97d87190d715c41dd004c1a89d9a` | `b8c38281f5e3ef54d0be30f8c2c234a8ba754fd5` | 3,195,928 | `ff9a66b618777f88bcb025a9636c04e87f9457d2c96606b6bd99eac151513a0f` |
| Combined Vulkan result | `161bf64b71413db3fbdffe27e3d8765cc2138d2f` | `48588f1f9154ea0bbc334b278d9d339217fa181f` | 3,199,258 | `53e41e89e6d5de1b57db20654296392946ff2484a607ba0cbd142cca5bcb043f` |

One code-owned ordered tuple is the independent authority for all four values
of every epoch. Each commit must resolve to the recorded blob, byte count, and
digest. Epoch commits must be ordered ancestors, and every later snapshot must
start with the complete bytes of its predecessor.

Verification against `origin/main` accepts a later commit only when its
`.agents/state.md` blob is byte-identical to the final configured epoch. New or
changed bytes make verification red until another reviewed epoch is added.

## Manifest and reconstruction

`state-migration-manifest.csv` keeps its typed schema. It contains one `source`
row before the event rows contributed by each epoch:

```text
source frozen
event rows covering 0..3191283
source eee64f81
event rows covering 3191283..3195928
source 161bf64b
event rows covering 3195928..3199258
```

Source rows describe cumulative snapshots, not independent files. Event ranges
remain absolute offsets in the final byte stream. At every source boundary the
payloads reconstructed so far must equal the preceding snapshot exactly; after
the following event rows they must reach the new snapshot exactly. Concatenating
all 149 imported payloads must reproduce the final 3,199,258-byte source.

The compatibility stub retains the permanent link to the original frozen blob
and adds the final imported snapshot as bounded provenance. Historical links do
not move when `origin/main` advances with identical state bytes.

## Reviewed tail ranges

The new bytes become three `legacy_import` events. Their absolute byte ranges
are explicit because the third legacy checkpoint placed its state anchor before
its heading, which the original heading-then-anchor heuristic deliberately does
not reinterpret.

| Event | Occurred at | Range | Epoch |
|---|---|---:|---|
| `STATE-20260809T083000-001` | `2026-08-09T08:30:00Z` | `3191283..3193748` | `eee64f81` |
| `STATE-20260809T110000-001` | `2026-08-09T11:00:00Z` | `3193748..3195928` | `eee64f81` |
| `STATE-20260809T130000-001` | `2026-08-09T13:00:00Z` | `3195928..3199258` | `161bf64b` |

The first and third ranges intentionally include the leading newline appended
by their source commit. Payload bytes, line endings, spelling, and malformed
legacy layout remain untouched. The new rows append to the writable August
index; existing index rows and all 146 imported event files remain byte-for-byte
unchanged.

## Environment classification

The public deployment surface gains exact documentation for:

- `VT_GEMMA4_EXPERT_VRAM_MB`: positive MiB values cap the device expert-cache
  LRU; unset or zero means unlimited.
- `VT_SERVER_MAX_PROMPT_CHARS`: defaults to 200,000 characters; zero disables
  the prompt-size guard.
- `VT_SERVER_MAX_NEW_TOKENS`: defaults to 4,096 and clamps requested generation;
  zero disables the cap.

The kernel-internal allowlist gains these default-off diagnostic, fallback, or
experimental implementation tactics:

- `VT_GEMMA4_BATCH_EXPERTS`
- `VT_GEMMA4_CUSTOM_EXPERT`
- `VT_GEMMA4_FP8_NATIVE`
- `VT_GEMMA4_FUSED_EXPERTS`
- `VT_GEMMA4_HOST_AXPY`
- `VT_GEMMA4_PROFILE`
- `VT_ROCM_GEMM_COMPUTE`
- `VT_ROCM_GEMV`
- `VT_ROCM_HIPBLASLT`

The classification changes no runtime default. The stale source comment that
claims a 12,288 MiB default is corrected to match the actual unlimited default.

## Implementation and failure handling

The migrator builds the frozen artifacts first, then adds only the configured
delta segments. Applying an epoch refuses to overwrite any existing event or
existing index-row bytes. It may append the three reviewed rows, add their three
event files, extend the migration manifest, and update the bounded stub.

Generation and verification fail when an epoch tuple disagrees with Git, an
epoch is not an ordered descendant, a later snapshot changes any prefix byte,
ranges overlap or leave a gap, a range crosses its epoch size, an existing
artifact would change, or the requested current source differs from the final
epoch.

After reconciliation, a strict structured checkpoint records the verified PR
state and refreshes `.agents/NOW.md` in the same change. It does not replace the
three imported legacy events or infer metadata for them.

## Test and delivery gates

Tests must first demonstrate these failures on the existing single-source
implementation, then bind the repair:

1. A current source with an exact frozen prefix and appended bytes is rejected
   before epoch support and accepted afterward.
2. Existing 146 event files and existing index rows retain exact hashes.
3. Each source tuple field, epoch order, prefix byte, range boundary, payload
   byte, and final-current byte comparison is mutation-bound.
4. The three explicit tail ranges reconstruct the two intermediate snapshots
   and final source byte-for-byte.
5. The direct environment checker is red before classification and green after;
   a focused test pins the approved three-public/nine-internal split and requires
   the two sets to remain disjoint.
6. Migration, structured-state, NOW, public-doc, policy, trailer, PR-size, and
   full role-aware preflight gates pass on the rebased branch.
7. A fresh reviewer performs static and scratch-mutation review on the immutable
   head, followed by the operator's independent declared gate.

The branch is then force-with-lease updated only if the remote PR head still
matches the inspected stale SHA.
