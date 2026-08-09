# Concurrent legacy-state merge reconciliation

**Task:** `state-record-structure-1`

**Approved direction:** continue the developer-selected preservation option when
`main` merges concurrently-authored legacy checkpoints into chronological
positions instead of appending one byte prefix.

## New target

The integration target is
`dc2139b3ff95f6de9b6a8ec8cae4bd5d40262dc7`. Its `.agents/state.md` is:

- blob `e79acd955e7042f9d028e18d41569ff5b67e32c3`
- 3,225,646 bytes
- SHA-256 `5c389a0cd84e6834263630af9e9d5a7a64f131ef06d479670dc6e3dc942ec202`

The first difference from both the original frozen source and the prior final
epoch is byte 3,172,950. Commits merged concurrent checkpoints at their legacy
timestamps and corrected the 13:00 anchor/heading order, so this snapshot is not
a cumulative prefix epoch. The append-epoch verifier must continue rejecting it;
the final cutover instead rebuilds live migration metadata from this exact merge
snapshot while retaining the frozen origin as historical provenance.

## Current-main append addendum

After that reconciliation was reviewed, `main` advanced to
`776c56f1c8b78ab69ea01e14759187b243b24d9e`. Its `.agents/state.md` is blob
`e266a8892401bc744955ccc1cb3bc75f64e4f399`, 3,231,342 bytes, with SHA-256
`913c76a2ab8303b1b5ad7dae3ec7876c5e29f0c319670b86eb646c4baa3119b6`.
The complete 3,225,646-byte `dc2139b3` source is its exact prefix, so this is a
strict append after the concurrent-merge snapshot rather than another rebuild.

All 156 prior legacy wrappers and their live raw rows remain byte-for-byte
unchanged. Their final row still ends at byte 3,225,646. The appended bytes form
one new legacy import, `STATE-20260809T150000-001`, whose payload owns the leading
newline and covers the half-open range `3225646..3231342`. The final live
manifest therefore contains 157 contiguous event rows and reconstructs all
3,231,342 bytes. The archived `f921` manifest and every original wrapper and
raw-row preservation inventory remain unchanged.

Verification independently pins the new raw-row hash, the wrapper hash, and the
extracted payload bytes. Mutating any of those authorities, removing the leading
newline, changing either range boundary, or altering the final provenance tuple
must fail while every prior preservation hash remains green. The resulting live
checkpoint reports 157 imports and 3,231,342 reconstructed bytes; the focused
test count is updated from measured implementation results rather than predicted
in this addendum.

## Preservation boundary

All 146 event files imported from the original frozen source remain
byte-for-byte unchanged. Their wrapper source links, offsets, payloads, spelling,
and hashes are historical evidence and are not regenerated against the newer
commit. The already-reviewed 08:30 and 11:00 event files also remain unchanged.

Canonical segmentation of the merge snapshot changes four payload boundaries:

- `STATE-20260809T001000-001` gains a trailing newline;
- `STATE-20260809T083000-001` loses the matching leading newline;
- `STATE-20260809T110000-001` gains a trailing newline;
- `STATE-20260809T130000-001` loses the matching leading newline and has its
  heading moved before the anchor.

The first three are represented with explicit boundary overrides so their
existing payload bytes remain unchanged and concatenation is still exact. The
13:00 file is local, unpushed migration output and is regenerated from the
corrected target bytes, including the boundary newline required after the
preserved 11:00 payload.

The prior live migration manifest is copied byte-for-byte to a new archival path
under `.agents/completed/` before the live manifest is rebuilt. This preserves
the reviewed prefix-epoch record without requiring its obsolete offsets to
describe the merge snapshot.

## Final-snapshot authority

Code independently pins two roles:

1. the original frozen provenance, retained permanently for historical wrapper
   links and archive validation; and
2. the exact `dc2139b3` final merge snapshot used by the live manifest and
   byte-exact reconstruction gate.

The compatibility stub links both. Live manifest event ranges are absolute in
the final merge snapshot and cover it contiguously. Existing wrappers are
accepted only when their extracted payload bytes equal the exact configured
segment; changing wrapper payload bytes, the archived manifest, a live raw CSV
row, either provenance tuple, any boundary override, or any final source byte is
red. Wrapper metadata remains historical and is not normalized.

## New concurrent events

The canonical source parser contributes exactly seven new legacy events:

- `STATE-20260808T210000-003`
- `STATE-20260808T220000-002`
- `STATE-20260808T230000-002`
- `STATE-20260808T233000-002`
- `STATE-20260808T234500-002`
- `STATE-20260809T140000-001`
- `STATE-20260809T140000-002`

Their rows are inserted in globally sorted order while every pre-existing index
row retains its original raw bytes. The strict PR 166 checkpoint remains a
post-cutover structured event and is not part of legacy reconstruction.

## Delivery gate

Tests must first show that the current prefix-epoch implementation rejects
`dc2139b3`, then prove:

1. all 146 original wrappers and the 08:30/11:00 wrappers retain exact hashes;
2. the archived prefix manifest retains its exact bytes;
3. explicit boundary overrides plus the regenerated 13:00 payload and seven new
   payloads reconstruct all 3,225,646 target bytes;
4. frozen-origin and final-snapshot tuple mutations, boundary mutations, payload
   mutations, raw-row mutations, and target-byte mutations fail;
5. valid post-cutover events and new shards still coexist with verification;
6. the branch is rebased onto current `origin/main`, current-main keyed records
   win conflicts, and the scoped NOW/env edits are reapplied;
7. focused, full, policy, trailer, PR-size, and role-aware preflight gates pass,
   followed by fresh static and scratch-mutation review.
