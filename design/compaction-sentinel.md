# Specification: Compaction Sentinel (recoverable compaction)

**Status:** Proposed
**Target:** supergraph 0.8.0+

Carried over from the April 2026 hardening plan. Three of that plan's four
priorities shipped - `RuntimeState`, the modular DSL handler registry, and the
EvolutionEngine. This one did not.

## 1. Problem

`compact_tombstones` renumbers every live slot across five storage layers:

| Layer | What moves |
|---|---|
| numpy arrays | `node_ids`, `node_kinds` |
| ColumnStore | every typed column and its presence bitmap |
| CSR matrices | edge endpoints |
| usearch | vector keys |
| sqlite | `documents`, `summaries`, `images`, `doc_metadata`, `doc_fts` |

`compact_tombstones_safe` wraps the sqlite half in an `ATTACH` plus
`BEGIN IMMEDIATE`, so a crash there rolls back. The in-memory half has no such
protection. A crash after the numpy arrays shift but before `id_to_slot` is
rebuilt leaves the store permanently inconsistent, and there is no intent log
to recover from.

## 2. The wider defect class

The same root cause produces a second, already-observable bug: code that
mutates store internals from outside `CoreStore` skips the dirty-flag protocol
that `checkpoint()` gates on.

`persistence/serializer.py` writes the tombstone set, `store_meta` and
`raw_edges` blobs only when `store._dirty_nodes` / `_dirty_edges` are set.
Two callers mutate `node_tombstones`, `id_to_slot`, `_count` and
`_edges_by_type` without setting either:

- `dsl/sys/lifecycle.py::_expire` (`SYS EXPIRE`)
- `core/optimizer.py::_evict_nodes` (`SYS EVICT`)

Both also bypass the WAL, because `SYS` statements are not appended to it.
Net effect: expired and evicted nodes return on the next open, with their
columns cleared and `_count` wrong.

## 3. Proposed fix

**Intent log.** Before compaction, write a `COMPACTION_START` row to a sqlite
metadata table holding the `old_to_new` remap. Clear it on success. On
`WAL.replay()`, a dangling `COMPACTION_START` triggers a forced full re-sync of
every layer, or a rollback to the last valid checkpoint.

**Close the flag gap.** Route `SYS EXPIRE` and `_evict_nodes` through
`CoreStore.delete_nodes_bulk`, which already sets every flag and invalidates
the live-mask cache. That deletes two hand-rolled copies of a 40-line invariant
rather than adding a third.

## 4. Success criteria

- A kill -9 at any point during `SYS OPTIMIZE COMPACT` leaves a store that
  opens and reports the same node and edge counts as before, or cleanly rolls
  back to the previous checkpoint.
- `SYS EXPIRE` followed by `close()` and reopen does not resurrect the expired
  nodes. There is currently no test asserting this.
