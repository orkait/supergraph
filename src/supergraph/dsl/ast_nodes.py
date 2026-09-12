from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class Condition:
    field: str
    op: str
    value: Any

@dataclass(slots=True)
class ContainsCondition:
    field: str
    value: str

@dataclass(slots=True)
class LikeCondition:
    field: str
    pattern: str
    _compiled_re: Any = field(default=None, compare=False, repr=False, hash=False)

@dataclass(slots=True)
class InCondition:
    field: str
    values: list

@dataclass(slots=True)
class SimilarCondition:
    query: str
    threshold: float
    field: str = "id"

@dataclass(slots=True)
class DegreeCondition:
    degree_type: str
    edge_kind: str | None
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
    value: int

@dataclass(slots=True)
class OrderClause:
    field: str
    direction: str = "ASC"

@dataclass(slots=True)
class AggFunc:
    func: str
    field: str | None

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
    direction: str
    node_id: str
    where: WhereClause | None = None
    limit: LimitClause | None = None

@dataclass(slots=True)
class CountQuery:
    target: str
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

@dataclass(slots=True)
class PatternStep:
    bound_id: str | None = None
    variable: str | None = None
    where: Any | None = None

@dataclass(slots=True)
class PatternArrow:
    expr: Any

@dataclass(slots=True)
class MatchPattern:
    steps: list[PatternStep]
    arrows: list[PatternArrow]

@dataclass(slots=True)
class MatchQuery:
    pattern: MatchPattern
    limit: LimitClause | None = None

@dataclass(slots=True)
class FieldPair:
    name: str
    value: Any

@dataclass(slots=True)
class CreateNode:
    id: str | None
    fields: list[FieldPair]
    auto_id: bool = False
    expires_in: tuple[int, str] | None = None
    expires_at: str | None = None
    event_at: str | int | None = None
    vector: list[float] | None = None
    document: str | None = None

@dataclass(slots=True)
class VarAssign:
    variable: str
    statement: Any

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
    source: str
    target: str
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
    direction: str
    node_id: str
    where: WhereClause | None = None

@dataclass(slots=True)
class Increment:
    node_id: str
    field: str
    amount: int | float

@dataclass(slots=True)
class Batch:
    statements: list

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

@dataclass(slots=True)
class SysStats:
    target: str | None = None

@dataclass(slots=True)
class SysKinds:
    pass

@dataclass(slots=True)
class SysEdgeKinds:
    pass

@dataclass(slots=True)
class SysDescribe:
    entity_type: str
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
    query: Any

@dataclass(slots=True)
class SysRegisterNodeKind:
    kind: str
    required: list[tuple[str, str | None]]
    optional: list[tuple[str, str | None]]
    embed_field: str | None = None

@dataclass(slots=True)
class SysRegisterEdgeKind:
    kind: str
    from_kinds: list[str]
    to_kinds: list[str]

@dataclass(slots=True)
class SysUnregister:
    entity_type: str
    kind: str

@dataclass(slots=True)
class SysCheckpoint:
    pass

@dataclass(slots=True)
class SysRebuild:
    pass

@dataclass(slots=True)
class SysClear:
    target: str

@dataclass(slots=True)
class SysWal:
    action: str

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

@dataclass(slots=True)
class VaultNew:
    title: str
    kind: str = "memory"
    tags: str | None = None

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
    at: int | None = None
    at_range: tuple[int, int] | None = None


@dataclass(slots=True)
class AnswerQuery:
    query: str
    limit: LimitClause | None = None
    where: WhereClause | None = None
    tokens: int | None = None
    at: int | None = None
    at_range: tuple[int, int] | None = None
    using: str | None = None


@dataclass(slots=True)
class ForgetNode:
    id: str

@dataclass(slots=True)
class SysRetain:
    pass

@dataclass(slots=True)
class SysHealth:
    pass

@dataclass(slots=True)
class SysOptimize:
    target: str | None = None

@dataclass(slots=True)
class SysLog:
    where: WhereClause | None = None
    since: str | None = None
    trace_id: str | None = None
    limit: LimitClause | None = None

@dataclass(slots=True)
class SysEvict:
    limit: LimitClause | None = None

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


@dataclass(slots=True)
class SysEvolveRule:
    name: str
    conditions: list
    actions: list
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
