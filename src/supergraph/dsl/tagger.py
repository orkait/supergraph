"""Auto-tag inference from AST node types for the intelligent log layer."""

from supergraph.dsl import ast_nodes

TAG_MAP = {
    ast_nodes.NodeQuery: "read",
    ast_nodes.NodesQuery: "read",
    ast_nodes.EdgesQuery: "read",
    ast_nodes.CountQuery: "read",
    ast_nodes.TraverseQuery: "read",
    ast_nodes.SubgraphQuery: "read",
    ast_nodes.PathQuery: "read",
    ast_nodes.PathsQuery: "read",
    ast_nodes.ShortestPathQuery: "read",
    ast_nodes.DistanceQuery: "read",
    ast_nodes.WeightedShortestPathQuery: "read",
    ast_nodes.WeightedDistanceQuery: "read",
    ast_nodes.AncestorsQuery: "read",
    ast_nodes.DescendantsQuery: "read",
    ast_nodes.CommonNeighborsQuery: "read",
    ast_nodes.MatchQuery: "read",
    ast_nodes.AggregateQuery: "read",
    ast_nodes.RecallQuery: "intelligence",
    ast_nodes.SimilarQuery: "intelligence",
    ast_nodes.LexicalSearchQuery: "intelligence",
    ast_nodes.RememberQuery: "intelligence",
    ast_nodes.CounterfactualQuery: "intelligence",
    ast_nodes.CreateNode: "write",
    ast_nodes.UpdateNode: "write",
    ast_nodes.UpsertNode: "write",
    ast_nodes.DeleteNode: "write",
    ast_nodes.DeleteNodes: "write",
    ast_nodes.UpdateNodes: "write",
    ast_nodes.Increment: "write",
    ast_nodes.MergeStmt: "write",
    ast_nodes.CreateEdge: "write",
    ast_nodes.UpdateEdge: "write",
    ast_nodes.DeleteEdge: "write",
    ast_nodes.DeleteEdges: "write",
    ast_nodes.ConnectNode: "write",
    ast_nodes.ForgetNode: "write",
    ast_nodes.Batch: "write",
    ast_nodes.AssertStmt: "belief",
    ast_nodes.RetractStmt: "belief",
    ast_nodes.PropagateStmt: "belief",
    ast_nodes.IngestStmt: "ingest",
    ast_nodes.BindContext: "write",
    ast_nodes.DiscardContext: "write",
    ast_nodes.VaultNew: "vault",
    ast_nodes.VaultRead: "vault",
    ast_nodes.VaultWrite: "vault",
    ast_nodes.VaultAppend: "vault",
    ast_nodes.VaultSearch: "vault",
    ast_nodes.VaultBacklinks: "vault",
    ast_nodes.VaultList: "vault",
    ast_nodes.VaultSync: "vault",
    ast_nodes.VaultDaily: "vault",
    ast_nodes.VaultArchive: "vault",
}

PHASE_MAP = {
    "read": "query",
    "intelligence": "query",
    "write": "mutation",
    "belief": "mutation",
    "ingest": "mutation",
    "vault": "mutation",
    "system": "system",
}


def infer_tag(ast) -> str:
    """Infer semantic tag from AST node type."""
    return TAG_MAP.get(type(ast), "system")


def infer_phase(tag: str) -> str:
    """Infer execution phase from tag."""
    return PHASE_MAP.get(tag, "system")
