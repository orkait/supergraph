from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# --- Filter expressions ---
@dataclass(slots=True)
class Condition:
    field: str
    op: str      # "=", "!=", ">", "<", ">=", "<="
    value: Any   # str, int, float, or None (for NULL)

@dataclass(slots=True)
class ContainsCondition:
    field: str
    value: str

@dataclass(slots=True)
class LikeCondition:
    field: str
    pattern: str  # SQL-like: % = any, _ = single char
    _compiled_re: Any = field(default=None, compare=False, repr=False, hash=False)

@dataclass(slots=True)
class InCondition:
    field: str
    values: list

@dataclass(slots=True)
class SimilarCondition:
    query: str
    threshold: float
    field: str = "id"  # Usually similarity is on the node content, but we reference the node identifier

@dataclass(slots=True)
class DegreeCondition:
    degree_type: str   # "INDEGREE" or "OUTDEGREE"
    edge_kind: str | None  # optional edge kind filter
    op: str
    value: int | float

@dataclass(slots=True)
class NotExpr:
    operand: Any

@dataclass(slots=True)
class AndExpr:
    operands: list

@dataclass(slots=True)
class OrExpr:
    operands: list

@dataclass(slots=True)
class WhereClause:
    expr: Any

@dataclass(slots=True)
class LimitClause:
    value: int

@dataclass(slots=True)
class OffsetClause:
    value: int

@dataclass(slots=True)
class MaxDepthClause:
    """Optional MAX_DEPTH N clause on SHORTEST PATH / WEIGHTED variants.

    A sentinel container the transformer emits when the grammar's optional
    ``max_depth_clause?`` branch matches, letting query-site helpers pluck
    the value from ``args`` the same way they pluck WhereClause/LimitClause
    instead of positional index arithmetic.
    """
    value: int

@dataclass(slots=True)
class OrderClause:
    field: str
    direction: str = "ASC"  # "ASC" or "DESC"

@dataclass(slots=True)
class AggFunc:
    func: str       # "COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"
    field: str | None  # None for COUNT()

    def label(self) -> str:
        if self.field is None:
            return f"{self.func}()"
        return f"{self.func}({self.field})"

@dataclass(slots=True)
class AggregateQuery:
    where: WhereClause | None = None
    group_by: list[str] | None = None
    select: list[AggFunc] | None = None
    having: Any | None = None
    order_by: AggFunc | None = None
    order_desc: bool = False
    limit: LimitClause | None = None

# --- Read queries ---
@dataclass(slots=True)
class NodeQuery:
    id: str
    with_document: bool = False

@dataclass(slots=True)
class NodesQuery:
    where: WhereClause | None = None
    order: OrderClause | None = None
    limit: LimitClause | None = None
    offset: OffsetClause | None = None

@dataclass(slots=True)
class EdgesQuery:
    direction: str   # "FROM" or "TO"
    node_id: str
    where: WhereClause | None = None
    limit: LimitClause | None = None

@dataclass(slots=True)
class CountQuery:
    target: str  # "NODES" or "EDGES"
    where: WhereClause | None = None

@dataclass(slots=True)
class TraverseQuery:
    start_id: str
    depth: int
    where: WhereClause | None = None
    limit: LimitClause | None = None

@dataclass(slots=True)
class SubgraphQuery:
    start_id: str
    depth: int

@dataclass(slots=True)
class PathQuery:
    from_id: str
    to_id: str
    max_depth: int
    where: WhereClause | None = None

@dataclass(slots=True)
class PathsQuery:
    from_id: str
    to_id: str
    max_depth: int
    where: WhereClause | None = None

@dataclass(slots=True)
class ShortestPathQuery:
    from_id: str
    to_id: str
    where: WhereClause | None = None
    # Optional per-direction expansion cap. None means unbounded. Prior
    # behavior was a hardcoded 10 inside algos.graph.bidirectional_bfs which
    # silently truncated long paths — see bug #40.
    max_depth: int | None = None

@dataclass(slots=True)
class DistanceQuery:
    from_id: str
    to_id: str
    max_depth: int

@dataclass(slots=True)
class WeightedShortestPathQuery:
    from_id: str
    to_id: str
    where: WhereClause | None = None
    # Optional cap passed to Dijkstra's early-termination check. None means
    # unbounded; caller gets whatever Dijkstra finds or None if unreachable.
    max_depth: int | None = None

@dataclass(slots=True)
class WeightedDistanceQuery:
    from_id: str
    to_id: str
    max_depth: int | None = None

@dataclass(slots=True)
class AncestorsQuery:
    node_id: str
    depth: int
    where: WhereClause | None = None

@dataclass(slots=True)
class DescendantsQuery:
    node_id: str
    depth: int
    where: WhereClause | None = None

@dataclass(slots=True)
class CommonNeighborsQuery:
    node_a: str
    node_b: str
    where: WhereClause | None = None

# --- Pattern matching ---
@dataclass(slots=True)
class PatternStep:
    """A step in a MATCH pattern - either a bound ID or a variable with optional filter."""
    bound_id: str | None = None      # if this is a literal string ID
    variable: str | None = None      # if this is a variable binding
    where: Any | None = None         # optional filter for variable steps

@dataclass(slots=True)
class PatternArrow:
    """An edge constraint in a MATCH pattern."""
    expr: Any   # filter expression (typically kind = "something")

@dataclass(slots=True)
class MatchPattern:
    """A full MATCH pattern: step (arrow step)+"""
    steps: list[PatternStep]
    arrows: list[PatternArrow]

@dataclass(slots=True)
class MatchQuery:
    pattern: MatchPattern
    limit: LimitClause | None = None

# --- Write queries ---
@dataclass(slots=True)
class FieldPair:
    name: str
    value: Any

@dataclass(slots=True)
class CreateNode:
    id: str | None  # None for AUTO ID
    fields: list[FieldPair]
    auto_id: bool = False
    expires_in: tuple[int, str] | None = None   # (amount, unit) e.g. (30, "m")
    expires_at: str | None = None                # ISO-8601 string
    event_at: str | int | None = None            # ISO-8601 string or epoch ms
    vector: list[float] | None = None
    document: str | None = None

@dataclass(slots=True)
class VarAssign:
    variable: str  # e.g. "$fn1"
    statement: Any  # the write query that produces an ID

@dataclass(slots=True)
class UpdateNode:
    id: str
    fields: list[FieldPair]

@dataclass(slots=True)
class UpsertNode:
    id: str
    fields: list[FieldPair]
    expires_in: tuple[int, str] | None = None
    expires_at: str | None = None
    event_at: str | int | None = None
    vector: list[float] | None = None

@dataclass(slots=True)
class DeleteNode:
    id: str

@dataclass(slots=True)
class DeleteNodes:
    where: WhereClause

@dataclass(slots=True)
class CreateEdge:
    source: str  # literal ID or "$variable"
    target: str  # literal ID or "$variable"
    fields: list[FieldPair]

@dataclass(slots=True)
class UpdateEdge:
    source: str
    target: str
    fields: list[FieldPair]
    where: WhereClause | None = None

@dataclass(slots=True)
class DeleteEdge:
    source: str
    target: str
    where: WhereClause | None = None

@dataclass(slots=True)
class DeleteEdges:
    direction: str  # "FROM" or "TO"
    node_id: str
    where: WhereClause | None = None

@dataclass(slots=True)
class Increment:
    node_id: str
    field: str
    amount: int | float

@dataclass(slots=True)
class Batch:
    statements: list  # list of write queries

@dataclass(slots=True)
class AssertStmt:
    id: str
    fields: list[FieldPair]
    confidence: float | None = None
    source: str | None = None
    event_at: str | int | None = None

@dataclass(slots=True)
class RetractStmt:
    id: str
    reason: str | None = None

@dataclass(slots=True)
class UpdateNodes:
    where: WhereClause
    fields: list[FieldPair]

@dataclass(slots=True)
class MergeStmt:
    source_id: str
    target_id: str

@dataclass(slots=True)
class PropagateStmt:
    node_id: str
    field: str
    depth: int

@dataclass(slots=True)
class BindContext:
    name: str

@dataclass(slots=True)
class DiscardContext:
    name: str

@dataclass(slots=True)
class BindNamespace:
    name: str

@dataclass(slots=True)
class DiscardNamespace:
    name: str | None = None

@dataclass(slots=True)
class IngestStmt:
    file_path: str
    node_id: str | None = None
    kind: str | None = None
    using: str | None = None
    vision_model: str | None = None

# --- Read queries (intelligence) ---

@dataclass(slots=True)
class RecallQuery:
    node_id: str
    depth: int
    limit: LimitClause | None = None
    where: WhereClause | None = None

@dataclass(slots=True)
class CounterfactualQuery:
    node_id: str

@dataclass(slots=True)
class SimilarQuery:
    target_vector: list[float] | None = None
    target_text: str | None = None
    target_node_id: str | None = None
    limit: LimitClause | None = None
    where: WhereClause | None = None

# --- System queries ---
@dataclass(slots=True)
class SysStats:
    target: str | None = None  # "NODES", "EDGES", "MEMORY", "WAL", or None for all

@dataclass(slots=True)
class SysKinds:
    pass

@dataclass(slots=True)
class SysEdgeKinds:
    pass

@dataclass(slots=True)
class SysDescribe:
    entity_type: str  # "NODE" or "EDGE"
    name: str

@dataclass(slots=True)
class SysSlowQueries:
    since: str | None = None
    limit: LimitClause | None = None

@dataclass(slots=True)
class SysFrequentQueries:
    limit: LimitClause | None = None

@dataclass(slots=True)
class SysFailedQueries:
    limit: LimitClause | None = None

@dataclass(slots=True)
class SysExplain:
    query: Any  # the read query to explain

@dataclass(slots=True)
class SysRegisterNodeKind:
    kind: str
    required: list[tuple[str, str | None]]  # [(field_name, type_name_or_none), ...]
    optional: list[tuple[str, str | None]]
    embed_field: str | None = None

@dataclass(slots=True)
class SysRegisterEdgeKind:
    kind: str
    from_kinds: list[str]
    to_kinds: list[str]

@dataclass(slots=True)
class SysUnregister:
    entity_type: str  # "NODE" or "EDGE"
    kind: str

@dataclass(slots=True)
class SysCheckpoint:
    pass

@dataclass(slots=True)
class SysRebuild:
    pass

@dataclass(slots=True)
class SysClear:
    target: str  # "LOG" or "CACHE"

@dataclass(slots=True)
class SysWal:
    action: str  # "STATUS" or "REPLAY"

@dataclass(slots=True)
class SysExpire:
    where: WhereClause | None = None

@dataclass(slots=True)
class SysContradictions:
    where: WhereClause | None = None
    field: str = ""
    group_by: str = ""

@dataclass(slots=True)
class SysSnapshot:
    name: str

@dataclass(slots=True)
class SysRollback:
    name: str

@dataclass(slots=True)
class SysSnapshots:
    pass

@dataclass(slots=True)
class SysDuplicates:
    where: WhereClause | None = None
    threshold: float = 0.95

@dataclass(slots=True)
class SysEmbedders:
    pass

@dataclass(slots=True)
class SysConnect:
    where: WhereClause | None = None
    threshold: float = 0.85

@dataclass(slots=True)
class ConnectNode:
    node_id: str
    threshold: float = 0.8

@dataclass(slots=True)
class SysConsolidate:
    similarity_threshold: float = 0.7
    min_cluster_size: int = 2

@dataclass(slots=True)
class SysReembed:
    pass

@dataclass(slots=True)
class SysStatus:
    pass

# --- Vault queries ---
@dataclass(slots=True)
class VaultNew:
    title: str
    kind: str = "memory"
    tags: str | None = None  # comma-separated

@dataclass(slots=True)
class VaultRead:
    title: str

@dataclass(slots=True)
class VaultWrite:
    title: str
    section: str
    content: str

@dataclass(slots=True)
class VaultAppend:
    title: str
    section: str
    content: str

@dataclass(slots=True)
class VaultSearch:
    query: str
    limit: LimitClause | None = None
    where: WhereClause | None = None

@dataclass(slots=True)
class VaultBacklinks:
    title: str

@dataclass(slots=True)
class VaultList:
    where: WhereClause | None = None
    order: OrderClause | None = None
    limit: LimitClause | None = None

@dataclass(slots=True)
class VaultSync:
    pass

@dataclass(slots=True)
class VaultDaily:
    pass

@dataclass(slots=True)
class VaultArchive:
    title: str

# --- Lexical search ---
@dataclass(slots=True)
class LexicalSearchQuery:
    query: str
    limit: LimitClause | None = None
    where: WhereClause | None = None

@dataclass(slots=True)
class RememberQuery:
    query: str
    limit: LimitClause | None = None
    where: WhereClause | None = None
    tokens: int | None = None
    at: int | None = None  # epoch ms anchor for temporal scoring
    at_range: tuple[int, int] | None = None


@dataclass(slots=True)
class AnswerQuery:
    """ANSWER: REMEMBER + reader-LLM synthesis.

    Runs the same retrieval pipeline as REMEMBER internally, then hands the
    retrieved passages + the question to a configured reader callable. The
    reader produces free-form answer text; supergraph returns that answer
    alongside the citing slot ids and the usual ``meta["signals"]`` block.
    """
    query: str
    limit: LimitClause | None = None
    where: WhereClause | None = None
    tokens: int | None = None
    at: int | None = None
    at_range: tuple[int, int] | None = None
    using: str | None = None  # named reader from the SuperGraph reader registry


# --- Forget (hard delete blob + memory) ---
@dataclass(slots=True)
class ForgetNode:
    id: str

# --- Retention policy scan ---
@dataclass(slots=True)
class SysRetain:
    pass

# --- Self-balancing ---
@dataclass(slots=True)
class SysHealth:
    pass

@dataclass(slots=True)
class SysOptimize:
    target: str | None = None  # None=all, or "COMPACT","STRINGS","EDGES","VECTORS","BLOBS","CACHE"

# --- Log queries ---
@dataclass(slots=True)
class SysLog:
    where: WhereClause | None = None
    since: str | None = None
    trace_id: str | None = None
    limit: LimitClause | None = None

# --- Emergency eviction ---
@dataclass(slots=True)
class SysEvict:
    limit: LimitClause | None = None

# --- Cron management ---
@dataclass(slots=True)
class SysCronAdd:
    name: str
    schedule: str
    query: str

@dataclass(slots=True)
class SysCronDelete:
    name: str

@dataclass(slots=True)
class SysCronEnable:
    name: str

@dataclass(slots=True)
class SysCronDisable:
    name: str

@dataclass(slots=True)
class SysCronList:
    pass

@dataclass(slots=True)
class SysCronRun:
    name: str


# --- Evolution rule management (Layer 5: metacognitive memory) ---
@dataclass(slots=True)
class SysEvolveRule:
    """SYS EVOLVE RULE "name" WHEN ... THEN ... COOLDOWN n PRIORITY n"""
    name: str
    conditions: list   # list of Condition-like dicts
    actions: list      # list of Action-like dicts
    cooldown: int = 60
    priority: int = 5

@dataclass(slots=True)
class SysEvolveList:
    pass

@dataclass(slots=True)
class SysEvolveShow:
    name: str

@dataclass(slots=True)
class SysEvolveEnable:
    name: str

@dataclass(slots=True)
class SysEvolveDisable:
    name: str

@dataclass(slots=True)
class SysEvolveDelete:
    name: str

@dataclass(slots=True)
class SysEvolveHistory:
    limit: int = 10

@dataclass(slots=True)
class SysEvolveReset:
    pass
