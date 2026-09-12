from supergraph.core.types import Result
from supergraph.core.errors import SuperGraphError
from supergraph.dsl.ast_nodes import (
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
)


class VaultExecutor:
    def __init__(self, vault_manager, vault_sync, runtime):
        self._manager = vault_manager
        self._sync = vault_sync
        self._runtime = runtime

    @property
    def _store(self):
        return self._runtime.store

    @property
    def _embedder(self):
        return self._runtime.embedder

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    def dispatch(self, ast) -> Result:
        handlers = {
            VaultNew: self._new,
            VaultRead: self._read,
            VaultWrite: self._write,
            VaultAppend: self._append,
            VaultSearch: self._search,
            VaultBacklinks: self._backlinks,
            VaultList: self._list,
            VaultSync: self._sync_cmd,
            VaultDaily: self._daily,
            VaultArchive: self._archive,
        }
        handler = handlers.get(type(ast))
        if handler is None:
            raise SuperGraphError(f"Unknown vault command: {type(ast).__name__}")
        try:
            return handler(ast)
        except ValueError as e:
            raise SuperGraphError(f"invalid vault operation: {e}") from e

    def _new(self, q) -> Result:
        tags = q.tags.split(",") if q.tags else []
        slug = self._manager.new(q.title, kind=q.kind, tags=tags)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "file": f"{slug}.md"}, count=0)

    def _read(self, q) -> Result:
        content = self._manager.read(q.title)
        from supergraph.vault.parser import parse_frontmatter, parse_sections
        fm = parse_frontmatter(content)
        sections = parse_sections(content)
        return Result(kind="note", data={
            "content": content,
            "frontmatter": fm,
            "sections": sections,
        }, count=1)

    def _write(self, q) -> Result:
        self._manager.write_section(q.title, q.section, q.content)
        from supergraph.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "section": q.section}, count=0)

    def _append(self, q) -> Result:
        self._manager.append_section(q.title, q.section, q.content)
        from supergraph.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "section": q.section}, count=0)

    def _search(self, q) -> Result:
        if not self._embedder or not self._vector_store:
            results = []
            for slug in self._manager.list_files():
                node_id = f"note:{slug}"
                node = self._store.get_node(node_id)
                if node and q.query.lower() in node.get("summary", "").lower():
                    results.append(node)
            limit = q.limit.value if q.limit else 10
            return Result(kind="nodes", data=results[:limit], count=min(len(results), limit))

        import numpy as np
        query_vec = self._embedder.encode_queries([q.query])[0]
        n = self._store._next_slot
        mask = self._store.compute_live_mask(n)
        kind_mask = self._store.columns.get_mask("kind", "=", "note", n)
        if kind_mask is not None:
            mask = mask & kind_mask

        vs_mask = (self._vector_store._has_vector[:n]
                   if n <= self._vector_store._capacity
                   else np.zeros(n, dtype=bool))
        combined = mask & vs_mask[:n]

        limit = q.limit.value if q.limit else 10
        slots, dists = self._vector_store.search(query_vec, k=limit * 3, mask=combined)

        results = []
        for slot, dist in zip(slots, dists):
            slot = int(slot)
            node = self._store._materialize_slot(slot)
            if node:
                node["_similarity_score"] = round(1.0 - float(dist), 4)
                results.append(node)
                if len(results) >= limit:
                    break

        return Result(kind="nodes", data=results, count=len(results))

    def _backlinks(self, q) -> Result:
        from supergraph.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        node_id = f"note:{slug}"
        edges = self._store.get_edges_to(node_id, kind="links")
        return Result(kind="edges", data=edges, count=len(edges))

    def _list(self, q) -> Result:
        nodes = self._store.get_all_nodes(kind="note")

        if q.where:
            from supergraph.dsl.executor_base import ExecutorBase
            base = ExecutorBase(self._runtime)
            nodes = [n for n in nodes if base._eval_where(q.where.expr, n)]

        if q.order:
            field = q.order.field
            desc = q.order.direction == "DESC"
            nodes.sort(key=lambda n: n.get(field, ""), reverse=desc)

        if q.limit:
            nodes = nodes[:q.limit.value]

        return Result(kind="nodes", data=nodes, count=len(nodes))

    def _sync_cmd(self, q) -> Result:
        result = self._sync.sync_all()
        return Result(kind="ok", data=result, count=0)

    def _daily(self, q) -> Result:
        slug = self._manager.daily()
        self._sync.sync_file(slug)
        content = self._manager.read(slug)
        return Result(kind="note", data={"slug": slug, "file": f"{slug}.md", "content": content}, count=1)

    def _archive(self, q) -> Result:
        self._manager.archive(q.title)
        from supergraph.vault.parser import title_to_slug
        slug = title_to_slug(q.title)
        self._sync.sync_file(slug)
        return Result(kind="ok", data={"slug": slug, "status": "archived"}, count=0)
