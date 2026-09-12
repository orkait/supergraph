import time as _time
from datetime import datetime, timedelta

from lark import Transformer, Token
from supergraph.dsl.ast_nodes import (
    Condition,
    ContainsCondition,
    LikeCondition,
    InCondition,
    SimilarCondition,
    DegreeCondition,
    NotExpr,
    AndExpr,
    OrExpr,
    WhereClause,
    LimitClause,
    MaxDepthClause,
    NodeQuery,
    NodesQuery,
    EdgesQuery,
    TraverseQuery,
    SubgraphQuery,
    PathQuery,
    PathsQuery,
    ShortestPathQuery,
    DistanceQuery,
    WeightedShortestPathQuery,
    WeightedDistanceQuery,
    AncestorsQuery,
    DescendantsQuery,
    CommonNeighborsQuery,
    PatternStep,
    PatternArrow,
    MatchPattern,
    MatchQuery,
    AggFunc,
    AggregateQuery,
    FieldPair,
    CreateNode,
    VarAssign,
    UpdateNode,
    UpsertNode,
    DeleteNode,
    DeleteNodes,
    CreateEdge,
    UpdateEdge,
    DeleteEdge,
    DeleteEdges,
    OffsetClause,
    OrderClause,
    CountQuery,
    Increment,
    Batch,
    AssertStmt,
    RetractStmt,
    UpdateNodes,
    MergeStmt,
    PropagateStmt,
    BindContext,
    DiscardContext,
    BindNamespace,
    DiscardNamespace,
    IngestStmt,
    RecallQuery,
    CounterfactualQuery,
    SimilarQuery,
    VaultNew,
    VaultRead,
    VaultWrite,
    VaultAppend,
    VaultSearch,
    VaultBacklinks,
    VaultList,
    VaultSync,
    VaultDaily,
    VaultArchive,
    SysStats,
    SysKinds,
    SysEdgeKinds,
    SysDescribe,
    SysSlowQueries,
    SysFrequentQueries,
    SysFailedQueries,
    SysExplain,
    SysRegisterNodeKind,
    SysRegisterEdgeKind,
    SysUnregister,
    SysCheckpoint,
    SysRebuild,
    SysClear,
    SysWal,
    SysExpire,
    SysContradictions,
    SysSnapshot,
    SysRollback,
    SysSnapshots,
    SysDuplicates,
    SysEmbedders,
    SysConnect,
    SysConsolidate,
    ConnectNode,
    SysReembed,
    SysStatus,
    LexicalSearchQuery,
    AnswerQuery,
    RememberQuery,
    ForgetNode,
    SysRetain,
    SysHealth,
    SysOptimize,
    SysEvict,
    SysLog,
    SysCronAdd,
    SysCronDelete,
    SysCronEnable,
    SysCronDisable,
    SysCronList,
    SysCronRun,
    SysEvolveRule,
    SysEvolveList,
    SysEvolveShow,
    SysEvolveEnable,
    SysEvolveDisable,
    SysEvolveDelete,
    SysEvolveHistory,
    SysEvolveReset,
)



class DSLTransformer(Transformer):
    def start(self, args):
        return args[0]

    def statement(self, args):
        return args[0]

    def user_query(self, args):
        return args[0]

    def read_query(self, args):
        return args[0]

    def write_query(self, args):
        return args[0]

    def system_query(self, args):
        return args[0]

    def sys_command(self, args):
        return args[0]

    # --- Values ---
    def val_string(self, args):
        s = str(args[0])
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"').replace('\\\\', '\\')

    def val_number(self, args):
        s = str(args[0])
        return float(s) if '.' in s else int(s)

    def val_null(self, args):
        return None

    # --- Time expressions ---
    def time_now(self, _items):
        return int(_time.time() * 1000)

    def time_offset(self, items):
        n = int(float(str(items[0])))
        unit = str(items[1])
        ms = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}[unit]
        return int(_time.time() * 1000) - n * ms

    def time_today(self, _items):
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight.timestamp() * 1000)

    def time_yesterday(self, _items):
        midnight = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight.timestamp() * 1000)

    def time_expr(self, items):
        return items[0]  # unwrap - the inner rule already returns the int value

    def STRING(self, token):
        return token

    def NUMBER(self, token):
        return token

    def IDENTIFIER(self, token):
        return str(token)

    # --- Read queries ---
    def node_q(self, args):
        node_id = self._str(args[0])
        with_document = any(isinstance(a, str) and a == "_with_doc" for a in args[1:])
        return NodeQuery(id=node_id, with_document=with_document)

    def with_doc(self, args):
        return "_with_doc"

    def nodes_q(self, args):
        where = self._find(args, WhereClause)
        order = self._find(args, OrderClause)
        limit = self._find(args, LimitClause)
        offset = self._find(args, OffsetClause)
        return NodesQuery(where=where, order=order, limit=limit, offset=offset)

    def edges_q(self, args):
        direction = args[0]  # "FROM" or "TO"
        node_id = self._str(args[1])
        where = self._find(args[2:], WhereClause)
        limit = self._find(args[2:], LimitClause)
        return EdgesQuery(direction=direction, node_id=node_id, where=where, limit=limit)

    def traverse_q(self, args):
        return TraverseQuery(
            start_id=self._str(args[0]),
            depth=self._num(args[1]),
            where=self._find(args[2:], WhereClause),
            limit=self._find(args[2:], LimitClause),
        )

    def subgraph_q(self, args):
        return SubgraphQuery(start_id=self._str(args[0]), depth=self._num(args[1]))

    def path_q(self, args):
        return PathQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=self._num(args[2]),
            where=self._find(args[3:], WhereClause),
        )

    def paths_q(self, args):
        return PathsQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=self._num(args[2]),
            where=self._find(args[3:], WhereClause),
        )

    def shortest_q(self, args):
        md = self._find(args[2:], MaxDepthClause)
        return ShortestPathQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
            max_depth=md.value if md is not None else None,
        )

    def distance_q(self, args):
        return DistanceQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=self._num(args[2]),
        )

    def weighted_sp_q(self, args):
        md = self._find(args[2:], MaxDepthClause)
        return WeightedShortestPathQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
            max_depth=md.value if md is not None else None,
        )

    def weighted_dist_q(self, args):
        md = self._find(args[2:], MaxDepthClause)
        return WeightedDistanceQuery(
            from_id=self._str(args[0]),
            to_id=self._str(args[1]),
            max_depth=md.value if md is not None else None,
        )

    def ancestors_q(self, args):
        return AncestorsQuery(
            node_id=self._str(args[0]),
            depth=self._num(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def descendants_q(self, args):
        return DescendantsQuery(
            node_id=self._str(args[0]),
            depth=self._num(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def common_q(self, args):
        return CommonNeighborsQuery(
            node_a=self._str(args[0]),
            node_b=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    # --- Direction ---
    def dir_from(self, args):
        return "FROM"

    def dir_to(self, args):
        return "TO"

    # --- Pattern matching ---
    def match_q(self, args):
        pattern = args[0]
        limit = self._find(args[1:], LimitClause)
        return MatchQuery(pattern=pattern, limit=limit)

    def pattern(self, args):
        steps = []
        arrows = []
        for a in args:
            if isinstance(a, PatternStep):
                steps.append(a)
            elif isinstance(a, PatternArrow):
                arrows.append(a)
        return MatchPattern(steps=steps, arrows=arrows)

    def bound_step(self, args):
        return PatternStep(bound_id=self._str(args[0]))

    def var_step(self, args):
        var_name = str(args[0])
        where = args[1] if len(args) > 1 else None
        return PatternStep(variable=var_name, where=where)

    def step_where(self, args):
        return args[0]

    def arrow(self, args):
        expr = args[0] if args else None
        return PatternArrow(expr=expr)

    # --- Writes ---
    def vector_clause(self, args):
        return ("vector", args[0])  # args[0] is the list from vector_literal

    def embed_clause(self, args):
        return ("embed", str(args[0]))

    def create_node(self, args):
        fields = args[1] if isinstance(args[1], list) else []
        vec = self._find_vector(args[2:])
        exp_in, exp_at = self._find_expires(args[2:])
        event_at = self._find_event_at(args[2:])
        doc = self._find_document(args[2:])
        return CreateNode(
            id=self._str(args[0]),
            fields=fields,
            expires_in=exp_in,
            expires_at=exp_at,
            event_at=event_at,
            vector=vec,
            document=doc,
        )

    def create_node_auto(self, args):
        fields = args[0] if isinstance(args[0], list) else []
        vec = self._find_vector(args[1:])
        exp_in, exp_at = self._find_expires(args[1:])
        event_at = self._find_event_at(args[1:])
        doc = self._find_document(args[1:])
        return CreateNode(
            id=None,
            fields=fields,
            auto_id=True,
            expires_in=exp_in,
            expires_at=exp_at,
            event_at=event_at,
            vector=vec,
            document=doc,
        )

    def document_clause(self, args):
        return ("document", self._str(args[0]))

    def var_assign(self, args):
        var_name = str(args[0])
        stmt = args[1]
        return VarAssign(variable=var_name, statement=stmt)

    def node_ref(self, args):
        """Return either a literal string or a $variable reference."""
        token = args[0]
        s = str(token)
        if s.startswith('$'):
            return s  # variable reference, kept as-is
        # Strip quotes from string literal
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return s

    def update_node(self, args):
        return UpdateNode(
            id=self._str(args[0]),
            fields=args[1] if isinstance(args[1], list) else [],
        )

    def upsert_node(self, args):
        fields = args[1] if isinstance(args[1], list) else []
        vec = self._find_vector(args[2:])
        exp_in, exp_at = self._find_expires(args[2:])
        event_at = self._find_event_at(args[2:])
        return UpsertNode(
            id=self._str(args[0]),
            fields=fields,
            expires_in=exp_in,
            expires_at=exp_at,
            event_at=event_at,
            vector=vec,
        )

    def expires_in(self, args):
        return ("expires_in", self._num(args[0]), str(args[1]))

    def expires_at(self, args):
        return ("expires_at", self._str(args[0]))

    def event_clause(self, args):
        return ("event_at", args[0])

    def assert_stmt(self, args):
        node_id = self._str(args[0])
        fields = args[1] if isinstance(args[1], list) else []
        confidence = None
        source = None
        event_at = None
        for a in args[2:]:
            if isinstance(a, tuple) and a[0] == "confidence":
                confidence = a[1]
            elif isinstance(a, tuple) and a[0] == "source":
                source = a[1]
            elif isinstance(a, tuple) and a[0] == "event_at":
                event_at = a[1]
        return AssertStmt(id=node_id, fields=fields, confidence=confidence, source=source, event_at=event_at)

    def confidence_clause(self, args):
        return ("confidence", self._num(args[0]))

    def source_clause(self, args):
        return ("source", self._str(args[0]))

    def retract_stmt(self, args):
        node_id = self._str(args[0])
        reason = None
        for a in args[1:]:
            if isinstance(a, tuple) and a[0] == "reason":
                reason = a[1]
        return RetractStmt(id=node_id, reason=reason)

    def reason_clause(self, args):
        return ("reason", self._str(args[0]))

    def update_nodes(self, args):
        where = args[0]
        fields = args[1] if isinstance(args[1], list) else []
        return UpdateNodes(where=where, fields=fields)

    def merge_stmt(self, args):
        return MergeStmt(source_id=self._str(args[0]), target_id=self._str(args[1]))

    def delete_node(self, args):
        return DeleteNode(id=self._str(args[0]))

    def delete_nodes(self, args):
        return DeleteNodes(where=args[0])

    def create_edge(self, args):
        return CreateEdge(
            source=args[0],  # node_ref returns str directly
            target=args[1],
            fields=args[2] if len(args) > 2 and isinstance(args[2], list) else [],
        )

    def delete_edge(self, args):
        return DeleteEdge(
            source=self._str(args[0]),
            target=self._str(args[1]),
            where=self._find(args[2:], WhereClause),
        )

    def delete_edges(self, args):
        direction = args[0]
        node_id = self._str(args[1])
        where = self._find(args[2:], WhereClause)
        return DeleteEdges(direction=direction, node_id=node_id, where=where)

    def increment(self, args):
        return Increment(
            node_id=self._str(args[0]),
            field=str(args[1]),
            amount=self._num(args[2]),
        )

    def batch(self, args):
        stmts = [a for a in args if not isinstance(a, Token) or a.type != "NEWLINE"]
        return Batch(statements=stmts)

    def field_pairs(self, args):
        pairs = []
        i = 0
        while i < len(args):
            name = str(args[i])
            value = args[i + 1]
            pairs.append(FieldPair(name=name, value=value))
            i += 2
        return pairs

    # --- Filters ---
    def where_clause(self, args):
        return WhereClause(expr=args[0])

    def expr(self, args):
        return args[0]

    def or_expr(self, args):
        # LALR left-recursive rule creates binary tree:
        #   or_expr(or_expr(a, b), c) for "a OR b OR c"
        # Transformer receives 2 args per reduction: [left, right]
        if len(args) == 1:
            return args[0]
        left, right = args[0], args[1]
        if isinstance(left, OrExpr) and isinstance(right, OrExpr):
            return OrExpr(operands=left.operands + right.operands)
        if isinstance(left, OrExpr):
            return OrExpr(operands=left.operands + [right])
        if isinstance(right, OrExpr):
            return OrExpr(operands=[left] + right.operands)
        return OrExpr(operands=[left, right])

    def and_expr(self, args):
        # LALR left-recursive rule creates binary tree:
        #   and_expr(and_expr(a, b), c) for "a AND b AND c"
        if len(args) == 1:
            return args[0]
        left, right = args[0], args[1]
        if isinstance(left, AndExpr) and isinstance(right, AndExpr):
            return AndExpr(operands=left.operands + right.operands)
        if isinstance(left, AndExpr):
            return AndExpr(operands=left.operands + [right])
        if isinstance(right, AndExpr):
            return AndExpr(operands=[left] + right.operands)
        return AndExpr(operands=[left, right])

    def not_term(self, args):
        return NotExpr(operand=args[0])

    def not_expr(self, args):
        return args[0]

    def condition(self, args):
        return Condition(field=self._field(args[0]), op=str(args[1]), value=args[2])

    def field_ref(self, args):
        """Return dot-notation field reference as string, e.g. 'x.kind'."""
        if len(args) == 1:
            return str(args[0])
        return f"{str(args[0])}.{str(args[1])}"

    def contains_cond(self, args):
        return ContainsCondition(field=self._field(args[0]), value=self._str(args[1]))

    def like_cond(self, args):
        return LikeCondition(field=self._field(args[0]), pattern=self._str(args[1]))

    def in_cond(self, args):
        field = self._field(args[0])
        values = args[1:]
        return InCondition(field=field, values=values)

    def similar_cond(self, args):
        return SimilarCondition(field=self._field(args[0]), query=self._str(args[1]), threshold=float(args[2]))

    def degree_condition(self, args):
        degree_type = args[0]
        rest = args[1:]
        if len(rest) == 3:
            return DegreeCondition(
                degree_type=degree_type,
                edge_kind=str(rest[0]),
                op=str(rest[1]),
                value=self._num(rest[2]),
            )
        else:
            return DegreeCondition(
                degree_type=degree_type,
                edge_kind=None,
                op=str(rest[0]),
                value=self._num(rest[1]),
            )

    def indegree(self, args):
        return "INDEGREE"

    def outdegree(self, args):
        return "OUTDEGREE"

    def limit_clause(self, args):
        return LimitClause(value=self._num(args[0]))

    def offset_clause(self, args):
        return OffsetClause(value=self._num(args[0]))

    def max_depth_clause(self, args):
        return MaxDepthClause(value=self._num(args[0]))

    def order_clause(self, args):
        field = str(args[0])
        direction = args[1] if len(args) > 1 else "ASC"
        return OrderClause(field=field, direction=direction)

    def order_asc(self, args):
        return "ASC"

    def order_desc(self, args):
        return "DESC"

    def count_q(self, args):
        target = args[0]
        where = self._find(args[1:], WhereClause)
        return CountQuery(target=target, where=where)

    def count_nodes(self, args):
        return "NODES"

    def count_edges(self, args):
        return "EDGES"

    # --- Aggregate queries ---
    def aggregate_q(self, items):
        where = None
        group_by = []
        select = []
        having = None
        order_by = None
        order_desc = False
        limit = None
        for item in items:
            if isinstance(item, WhereClause):
                where = item
            elif isinstance(item, list) and item and isinstance(item[0], str):
                group_by = item
            elif isinstance(item, list) and item and isinstance(item[0], AggFunc):
                select = item
            elif isinstance(item, LimitClause):
                limit = item
            elif isinstance(item, AggFunc):
                order_by = item
            elif isinstance(item, tuple) and len(item) == 2:
                order_by, order_desc = item
            # having is an expression (Condition, AndExpr, etc.)
            elif item is not None and not isinstance(item, (WhereClause, LimitClause, AggFunc, list)):
                having = item
        return AggregateQuery(where=where, group_by=group_by, select=select,
                              having=having, order_by=order_by, order_desc=order_desc, limit=limit)

    def group_clause(self, items):
        return [str(i) for i in items]

    def select_clause(self, items):
        return list(items)

    def having_clause(self, items):
        return items[0]  # the having_expr (a Condition)

    def having_expr(self, items):
        agg = items[0]  # AggFunc
        op = str(items[1])
        val = items[2]
        return Condition(field=agg.label(), op=op, value=val)

    def order_agg_clause(self, items):
        agg = items[0]
        desc = len(items) > 1 and items[1] == "DESC"
        return (agg, desc)

    def agg_func_ref(self, items):
        return items[0]

    def agg_count(self, _items):
        return AggFunc("COUNT", None)

    def agg_count_distinct(self, items):
        return AggFunc("COUNT_DISTINCT", str(items[0]))

    def agg_sum(self, items):
        return AggFunc("SUM", str(items[0]))

    def agg_avg(self, items):
        return AggFunc("AVG", str(items[0]))

    def agg_min(self, items):
        return AggFunc("MIN", str(items[0]))

    def agg_max(self, items):
        return AggFunc("MAX", str(items[0]))

    def update_edge(self, args):
        return UpdateEdge(
            source=self._str(args[0]),
            target=self._str(args[1]),
            fields=args[2] if len(args) > 2 and isinstance(args[2], list) else [],
            where=self._find(args[2:], WhereClause),
        )

    # --- Intelligence queries ---
    def recall_q(self, args):
        node_id = self._str(args[0])
        depth = self._num(args[1])
        limit = self._find(args[2:], LimitClause)
        where = self._find(args[2:], WhereClause)
        return RecallQuery(node_id=node_id, depth=depth, limit=limit, where=where)

    def counterfactual(self, args):
        return CounterfactualQuery(node_id=self._str(args[0]))

    # --- Similar queries ---
    def similar_q(self, args):
        target = args[0]  # SimilarQuery with target set
        limit = self._find(args[1:], LimitClause)
        where = self._find(args[1:], WhereClause)
        target.limit = limit
        target.where = where
        return target

    def similar_vector(self, args):
        vec = args[0]  # list of floats from vector_literal
        return SimilarQuery(target_vector=vec)

    def similar_text(self, args):
        return SimilarQuery(target_text=self._str(args[0]))

    def similar_node(self, args):
        return SimilarQuery(target_node_id=self._str(args[0]))

    def lexical_q(self, args):
        query = self._str(args[0])
        limit = self._find(args[1:], LimitClause)
        where = self._find(args[1:], WhereClause)
        return LexicalSearchQuery(query=query, limit=limit, where=where)

    def tokens_clause(self, args):
        return ("tokens", self._num(args[0]))

    def at_clause(self, args):
        val = args[0]
        if isinstance(val, (int, float)):
            ms = int(val)
            return ("at", ms, (ms, ms))
        from supergraph.core.temporal import parse_date, parse_date_range
        ms = parse_date(str(val))
        if ms is None:
            raise ValueError(f"Cannot parse AT date: {val!r}")
        date_range = parse_date_range(str(val))
        range_tuple = None
        if date_range is not None:
            range_tuple = (date_range.start_ms, date_range.end_ms)
        return ("at", ms, range_tuple)

    def remember_q(self, args):
        query = self._str(args[0])
        limit = self._find(args[1:], LimitClause)
        where = self._find(args[1:], WhereClause)
        tokens = None
        at = None
        at_range = None
        for a in args[1:]:
            if isinstance(a, tuple) and a[0] == "tokens":
                tokens = int(a[1])
            elif isinstance(a, tuple) and a[0] == "at":
                at = int(a[1])
                at_range = a[2] if len(a) > 2 else None
        return RememberQuery(query=query, limit=limit, where=where, tokens=tokens, at=at, at_range=at_range)

    def answer_q(self, args):
        query = self._str(args[0])
        limit = self._find(args[1:], LimitClause)
        where = self._find(args[1:], WhereClause)
        tokens = None
        at = None
        at_range = None
        using = None
        for a in args[1:]:
            if isinstance(a, tuple):
                if a[0] == "tokens":
                    tokens = int(a[1])
                elif a[0] == "at":
                    at = int(a[1])
                    at_range = a[2] if len(a) > 2 else None
                elif a[0] == "using_reader":
                    using = a[1]
        return AnswerQuery(query=query, limit=limit, where=where,
                           tokens=tokens, at=at, at_range=at_range, using=using)

    def using_reader(self, args):
        return ("using_reader", self._str(args[0]))

    def vector_literal(self, args):
        return [self._num(a) for a in args]

    def propagate_stmt(self, args):
        return PropagateStmt(
            node_id=self._str(args[0]),
            field=str(args[1]),
            depth=self._num(args[2]),
        )

    def bind_context(self, args):
        return BindContext(name=self._str(args[0]))

    def discard_context(self, args):
        return DiscardContext(name=self._str(args[0]))

    def bind_namespace(self, args):
        return BindNamespace(name=self._str(args[0]))

    def discard_namespace(self, args):
        return DiscardNamespace(name=self._str(args[0]) if args else None)

    def ingest_stmt(self, args):
        file_path = self._str(args[0])
        node_id = None
        kind = None
        using = None
        vision_model = None
        for a in args[1:]:
            if isinstance(a, tuple):
                if a[0] == "ingest_as":
                    node_id = a[1]
                elif a[0] == "ingest_kind":
                    kind = a[1]
                elif a[0] == "using_clause":
                    using = a[1]
                elif a[0] == "vision_clause":
                    vision_model = a[1]
        return IngestStmt(file_path=file_path, node_id=node_id, kind=kind,
                          using=using, vision_model=vision_model)

    def ingest_as(self, args):
        return ("ingest_as", self._str(args[0]))

    def ingest_kind(self, args):
        return ("ingest_kind", self._str(args[0]))

    def ingest_using(self, args):
        return args[0]  # pass through the using_clause or vision_clause tuple

    def using_clause(self, args):
        return ("using_clause", str(args[0]))

    def vision_clause(self, args):
        return ("vision_clause", self._str(args[0]))

    # --- Vault queries ---
    def vault_new(self, args):
        title = self._str(args[0])
        kind = "memory"
        tags = None
        for a in args[1:]:
            if isinstance(a, tuple) and a[0] == "vault_kind":
                kind = a[1]
            elif isinstance(a, tuple) and a[0] == "vault_tags":
                tags = a[1]
        return VaultNew(title=title, kind=kind, tags=tags)

    def vault_kind(self, args):
        return ("vault_kind", self._str(args[0]))

    def vault_tags(self, args):
        return ("vault_tags", self._str(args[0]))

    def vault_read(self, args):
        return VaultRead(title=self._str(args[0]))

    def vault_write(self, args):
        return VaultWrite(
            title=self._str(args[0]),
            section=self._str(args[1]),
            content=self._str(args[2]),
        )

    def vault_append(self, args):
        return VaultAppend(
            title=self._str(args[0]),
            section=self._str(args[1]),
            content=self._str(args[2]),
        )

    def vault_search(self, args):
        query = self._str(args[0])
        limit = self._find(args[1:], LimitClause)
        where = self._find(args[1:], WhereClause)
        return VaultSearch(query=query, limit=limit, where=where)

    def vault_backlinks(self, args):
        return VaultBacklinks(title=self._str(args[0]))

    def vault_list(self, args):
        where = self._find(args, WhereClause)
        order = self._find(args, OrderClause)
        limit = self._find(args, LimitClause)
        return VaultList(where=where, order=order, limit=limit)

    def vault_sync(self, args):
        return VaultSync()

    def vault_daily(self, args):
        return VaultDaily()

    def vault_archive(self, args):
        return VaultArchive(title=self._str(args[0]))

    # --- System queries ---
    def sys_stats(self, args):
        target = str(args[0]) if args else None
        return SysStats(target=target)

    def sys_kinds(self, args):
        return SysKinds()

    def sys_edge_kinds(self, args):
        return SysEdgeKinds()

    def sys_describe(self, args):
        entity_type = str(args[0])
        name = self._str(args[1])
        return SysDescribe(entity_type=entity_type, name=name)

    def sys_slow(self, args):
        since = None
        limit = None
        for a in args:
            if isinstance(a, str):
                since = a
            elif isinstance(a, LimitClause):
                limit = a
        return SysSlowQueries(since=since, limit=limit)

    def sys_frequent(self, args):
        return SysFrequentQueries(limit=self._find(args, LimitClause))

    def sys_failed(self, args):
        return SysFailedQueries(limit=self._find(args, LimitClause))

    def since_clause(self, args):
        return self._str(args[0])

    def sys_explain(self, args):
        return SysExplain(query=args[0])

    def sys_register_node_kind(self, args):
        kind = self._str(args[0])
        required = args[1]  # ident_list
        optional = []
        embed_field = None
        for a in args[2:]:
            if isinstance(a, list):
                optional = a  # from optional_clause
            elif isinstance(a, tuple) and a[0] == "embed":
                embed_field = a[1]
        return SysRegisterNodeKind(kind=kind, required=required, optional=optional,
                                   embed_field=embed_field)

    def sys_register_edge_kind(self, args):
        kind = self._str(args[0])
        from_kinds = args[1]
        to_kinds = args[2]
        return SysRegisterEdgeKind(kind=kind, from_kinds=from_kinds, to_kinds=to_kinds)

    def optional_clause(self, args):
        return args[0]

    def sys_unregister(self, args):
        entity_type = str(args[0])
        kind = self._str(args[1])
        return SysUnregister(entity_type=entity_type, kind=kind)

    def sys_checkpoint(self, args):
        return SysCheckpoint()

    def sys_rebuild(self, args):
        return SysRebuild()

    def sys_clear(self, args):
        target = str(args[0])
        return SysClear(target=target)

    def sys_wal(self, args):
        action = str(args[0])
        return SysWal(action=action)

    def sys_expire(self, args):
        where = self._find(args, WhereClause)
        return SysExpire(where=where)

    def sys_snapshot(self, args):
        return SysSnapshot(name=self._str(args[0]))

    def sys_rollback(self, args):
        return SysRollback(name=self._str(args[0]))

    def sys_snapshots(self, args):
        return SysSnapshots()

    def sys_duplicates(self, args):
        where = self._find(args, WhereClause)
        threshold = 0.95
        for a in args:
            if isinstance(a, tuple) and a[0] == "threshold":
                threshold = a[1]
        return SysDuplicates(where=where, threshold=threshold)

    def threshold_clause(self, args):
        return ("threshold", self._num(args[0]))

    def sys_embedders(self, args):
        return SysEmbedders()

    def sys_connect(self, args):
        where = self._find(args, WhereClause)
        threshold = 0.85
        for a in args:
            if isinstance(a, tuple) and a[0] == "threshold":
                threshold = a[1]
        return SysConnect(where=where, threshold=threshold)

    def sys_consolidate(self, args):
        threshold = 0.7
        min_size = 2
        for a in args:
            if isinstance(a, tuple) and a[0] == "threshold":
                threshold = a[1]
            elif isinstance(a, tuple) and a[0] == "min_cluster_size":
                min_size = int(a[1])
        return SysConsolidate(similarity_threshold=threshold, min_cluster_size=min_size)

    def min_cluster_clause(self, args):
        return ("min_cluster_size", self._num(args[0]))

    def connect_node(self, args):
        node_id = self._str(args[0])
        threshold = 0.8
        for a in args[1:]:
            if isinstance(a, tuple) and a[0] == "threshold":
                threshold = a[1]
        return ConnectNode(node_id=node_id, threshold=threshold)

    def forget_node(self, args):
        return ForgetNode(id=self._str(args[0]))

    def sys_reembed(self, args):
        return SysReembed()

    def sys_status(self, args):
        return SysStatus()

    def sys_retain(self, args):
        return SysRetain()

    def sys_health(self, args):
        return SysHealth()

    def sys_optimize(self, args):
        target = str(args[0]) if args else None
        return SysOptimize(target=target)

    def sys_evict(self, args):
        limit = self._find(args, LimitClause)
        return SysEvict(limit=limit)

    def sys_contradictions(self, args):
        where = self._find(args, WhereClause)
        # The last two args are IDENTIFIER tokens: field, group_by
        identifiers = [str(a) for a in args if isinstance(a, str) and not isinstance(a, Token)]
        # Actually, IDENTIFIER tokens are strings after transformation
        # Filter out WhereClause and collect remaining string identifiers
        idents = []
        for a in args:
            if isinstance(a, WhereClause):
                continue
            if isinstance(a, (str, Token)) and not isinstance(a, WhereClause):
                idents.append(str(a))
        # idents should be [field, group_by_field]
        field = idents[0] if len(idents) >= 1 else ""
        group_by = idents[1] if len(idents) >= 2 else ""
        return SysContradictions(where=where, field=field, group_by=group_by)

    def typed_ident_with_type(self, args):
        return (str(args[0]), str(args[1]))

    def typed_ident_bare(self, args):
        return (str(args[0]), None)

    def ident_list(self, args):
        return list(args)

    def string_list(self, args):
        return [self._str(a) for a in args]

    # --- Log queries ---
    def sys_log(self, args):
        where = None
        since = None
        trace_id = None
        limit = self._find(args, LimitClause)
        for a in args:
            if isinstance(a, WhereClause):
                where = a
            elif isinstance(a, tuple) and a[0] == "log_since":
                since = a[1]
            elif isinstance(a, tuple) and a[0] == "log_trace":
                trace_id = a[1]
        return SysLog(where=where, since=since, trace_id=trace_id, limit=limit)

    def log_where(self, args):
        return WhereClause(expr=args[0])

    def log_since(self, args):
        return ("log_since", self._str(args[0]))

    def log_trace(self, args):
        return ("log_trace", self._str(args[0]))

    # --- Cron commands ---
    def sys_cron(self, args):
        return args[0]

    def cron_command(self, args):
        return args[0]

    def cron_add(self, args):
        return SysCronAdd(
            name=self._str(args[0]),
            schedule=self._str(args[1]),
            query=self._str(args[2]),
        )

    def cron_delete(self, args):
        return SysCronDelete(name=self._str(args[0]))

    def cron_enable(self, args):
        return SysCronEnable(name=self._str(args[0]))

    def cron_disable(self, args):
        return SysCronDisable(name=self._str(args[0]))

    def cron_list(self, args):
        return SysCronList()

    def cron_run(self, args):
        return SysCronRun(name=self._str(args[0]))

    # --- Evolution commands ---
    def sys_evolve(self, args):
        return args[0]

    def evolve_rule(self, args):
        name = self._str(args[0])
        conditions = []
        actions = []
        cooldown = 60
        priority = 5
        for a in args[1:]:
            if isinstance(a, list) and a and isinstance(a[0], dict) and "signal" in a[0]:
                conditions = a
            elif isinstance(a, list) and a and isinstance(a[0], dict) and "kind" in a[0]:
                actions.extend(a)
            elif isinstance(a, tuple) and a[0] == "cooldown":
                cooldown = a[1]
            elif isinstance(a, tuple) and a[0] == "priority":
                priority = a[1]
        return SysEvolveRule(name=name, conditions=conditions, actions=actions,
                             cooldown=cooldown, priority=priority)

    def evolve_when_clause(self, args):
        return [a for a in args if isinstance(a, dict)]

    def evolve_condition(self, args):
        signal = str(args[0])
        op = str(args[1])
        value = self._num(args[2])
        return {"signal": signal, "operator": op, "value": value}

    def evolve_then_clause(self, args):
        return [a for a in args if isinstance(a, dict)]

    def evolve_action_set(self, args):
        param = str(args[0])
        value = args[1]
        return {"kind": "set", "param": param, "value": value, "delta": 0.0, "until": None}

    def evolve_action_adjust_until(self, args):
        param = str(args[0])
        delta = float(self._num(args[1]))
        until = float(self._num(args[2]))
        return {"kind": "adjust", "param": param, "value": None, "delta": delta, "until": until}

    def evolve_action_adjust(self, args):
        param = str(args[0])
        delta = float(self._num(args[1]))
        return {"kind": "adjust", "param": param, "value": None, "delta": delta, "until": None}

    def evolve_action_add(self, args):
        param = str(args[0])
        element = self._str(args[1])
        return {"kind": "add", "param": param, "value": element, "delta": 0.0, "until": None}

    def evolve_action_remove(self, args):
        param = str(args[0])
        element = self._str(args[1])
        return {"kind": "remove", "param": param, "value": element, "delta": 0.0, "until": None}

    def evolve_action_run(self, args):
        cmd = " ".join(str(a) for a in args)
        return {"kind": "run", "param": cmd, "value": None, "delta": 0.0, "until": None}

    def evolve_value_scalar(self, args):
        return self._num(args[0])

    def evolve_value_list(self, args):
        return [self._num(a) for a in args]

    def evolve_cooldown(self, args):
        return ("cooldown", int(self._num(args[0])))

    def evolve_priority(self, args):
        return ("priority", int(self._num(args[0])))

    def evolve_list(self, args):
        return SysEvolveList()

    def evolve_show(self, args):
        return SysEvolveShow(name=self._str(args[0]))

    def evolve_enable(self, args):
        return SysEvolveEnable(name=self._str(args[0]))

    def evolve_disable(self, args):
        return SysEvolveDisable(name=self._str(args[0]))

    def evolve_delete(self, args):
        return SysEvolveDelete(name=self._str(args[0]))

    def evolve_history(self, args):
        limit = 10
        for a in args:
            try:
                limit = int(self._num(a))
            except Exception:
                pass
        return SysEvolveHistory(limit=limit)

    def evolve_reset(self, args):
        return SysEvolveReset()

    # --- Helpers ---
    def _str(self, token) -> str:
        """Extract string value from token, stripping quotes."""
        s = str(token)
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return s

    def _num(self, token) -> int | float:
        s = str(token)
        return float(s) if '.' in s else int(s)

    def _find(self, args, cls):
        """Find first instance of cls in args."""
        for a in args:
            if isinstance(a, cls):
                return a
        return None

    def _find_expires(self, args):
        """Extract expires_in or expires_at from args. Returns (exp_in, exp_at)."""
        exp_in = None
        exp_at = None
        for a in args:
            if isinstance(a, tuple):
                if a[0] == "expires_in":
                    exp_in = (int(a[1]), a[2])
                elif a[0] == "expires_at":
                    exp_at = a[1]
        return exp_in, exp_at

    def _find_vector(self, args):
        """Extract vector from args. Returns list[float] or None."""
        for a in args:
            if isinstance(a, tuple) and a[0] == "vector":
                return a[1]
        return None

    def _find_document(self, args):
        """Extract document text from args. Returns str or None."""
        for a in args:
            if isinstance(a, tuple) and a[0] == "document":
                return a[1]
        return None

    def _field(self, arg) -> str:
        """Extract field reference string, handling dot-notation."""
        if isinstance(arg, str):
            return arg
        return str(arg)

    def _find_event_at(self, args):
        """Extract EVENT_AT from args. Returns raw value or None."""
        for a in args:
            if isinstance(a, tuple) and a[0] == "event_at":
                return a[1]
        return None
