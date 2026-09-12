# Specification: Tiered Cognitive Storage (Hot/Cold Memory)

**Status:** Proposed  
**Version:** 1.0.0  
**Target:** supergraph 0.8.0+

## 1. Problem Statement
SuperGraph is designed for high-performance "Agentic Memory." Currently, all "live" nodes must reside in the `numpy`-backed **Hot Tier**. 
*   **RAM Inefficiency**: As an agent's memory grows to millions of nodes, the RAM footprint of the `numpy` arrays and `usearch` index scales linearly, even for memories that haven't been accessed in weeks.
*   **Binary Forgetting**: The only way to save RAM is `SYS EVICT` (deletion), which leads to permanent "knowledge loss." 
*   **Missing Intermediate State**: There is no "dormant" state where a memory is preserved but offloaded from expensive RAM to cheap disk.

## 2. Proposed Architecture: Two-Tier Storage

### 2.1 The Hot Tier (In-Memory)
*   **Backing**: Numpy arrays, Scipy CSR matrices, Usearch HNSW.
*   **Performance**: Microsecond-latency traversals and filtered counts.
*   **Lifecycle**: Active context, recently learned facts, and high-frequency associations.

### 2.2 The Cold Tier (Disk-Backed)
*   **Backing**: SQLite `nodes_cold` table (JSON attributes), FTS5 index.
*   **Performance**: Millisecond-latency point lookups and lexical search.
*   **Lifecycle**: Historical logs, archived documents, and low-confidence beliefs.

## 3. Implementation Details

### 3.1 Schema Changes (SQLite)
A new table to store the state of archived nodes:
```sql
CREATE TABLE nodes_cold (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    archived_at REAL NOT NULL
);
CREATE INDEX idx_nodes_cold_kind ON nodes_cold(kind);
```

### 3.2 Slot Mapping Evolution
The `id_to_slot` mapping in `CoreStore` currently maps `str_id -> int (0 to N)`. 
We will extend this to support **Sentinel Slot IDs**:
*   `slot >= 0`: Node is in **Hot Tier** (index into Numpy arrays).
*   `slot == -1`: Node is **Archived (Cold)**. Point lookup triggers a fetch from `nodes_cold`.

### 3.3 DSL Commands

#### `SYS ARCHIVE <where_clause>`
Moves matching nodes from `numpy` to SQLite.
1.  Materialize node attributes.
2.  Insert into `nodes_cold`.
3.  Set `id_to_slot[id] = -1`.
4.  Tombstone the node in `numpy` (freeing the slot for future compaction).
5.  Remove from `usearch` index.

#### `SYS PROMOTE <where_clause>`
Moves nodes from SQLite back to the Hot Tier.
1.  Read from `nodes_cold`.
2.  Allocate a new slot in `numpy`.
3.  Re-embed for the `usearch` index.
4.  Delete from `nodes_cold`.

### 3.4 Federated Retrieval
The `REMEMBER` and `SIMILAR TO` commands will be updated to:
1.  Search the Hot Tier (standard path).
2.  Search the Cold Tier (SQLite FTS or Vector-lite if a small Cold-HNSW exists).
3.  Merge results, marking cold hits with a `_tier: "cold"` attribute.

## 4. Metacognitive Integration (`EvolutionEngine`)
The `EvolutionEngine` will handle automatic tiering based on signals:
```sql
SYS EVOLVE RULE "auto-archive-old-memories"
  WHEN memory_pct > 80 AND node_count > 50000
  THEN RUN "SYS ARCHIVE WHERE __updated_at__ < (NOW() - 30d) AND importance < 0.5"
  COOLDOWN 3600
```

## 5. Success Criteria & Metrics
*   **RAM Savings**: Moving 50% of nodes to the Cold Tier should result in a ~40% reduction in `numpy` heap usage.
*   **Retrieval Integrity**: `REMEMBER` must return archived nodes if they are the most relevant context, albeit with a slight latency increase.
*   **Zero Data Loss**: Transitions between tiers must be atomic (using the WAL as the sentinel).

## 6. Compatibility & Risks
*   **Graph Traversals**: `TRAVERSE` and `PATH` will **not** cross into the Cold Tier for performance reasons. Archived nodes act as "terminator nodes" in the graph unless explicitly promoted.
*   **Secondary Indices**: SQLite indexes in the Cold Tier must be kept in sync with the Schema Registry.
