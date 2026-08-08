# Pre-cutover claim snapshot protocol

This is the evidence-preserving archive of the claim-view procedure retired by
the live-claim cutover. Feature lifecycle rows and claim narratives remain in
`.agents/coordination.md`; only this obsolete procedure and its empty generated
snapshot moved.

## Retired procedure

The former `scripts/claim-view.py --apply` queried open pull requests and wrote
a generated table into `.agents/coordination.md`. Offline `--check` accepted the
committed table for up to 14 days. An open `row/<ROW-ID>` PR rendered as a
reservation; merging or closing it removed the row only after another apply.

The last committed snapshot was generated on 2026-08-04 and contained no rows:

| Row | PR | State | Agent | Updated |
|---|---|---|---|---|
| _none_ | | | | |

## Why it was retired

The timestamp and TTL made remote state look locally authoritative between
refreshes. The replacement separates network-independent `--check-local` from
remote-authoritative `--check-live`; a failed query is `REMOTE_UNVERIFIED` and
cannot be interpreted as an unclaimed task.
