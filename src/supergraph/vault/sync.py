import logging

logger = logging.getLogger(__name__)

from supergraph.vault.parser import (
    parse_frontmatter, parse_sections, extract_wikilinks,
)
from supergraph.vault.manager import VaultManager


class VaultSync:

    def __init__(self, manager: VaultManager, runtime,
                 summary_max_length: int = 200):
        self._manager = manager
        self._runtime = runtime
        self._summary_max_length = summary_max_length

    @property
    def _store(self):
        return self._runtime.store

    @property
    def _schema(self):
        return self._runtime.schema

    @property
    def _embedder(self):
        return self._runtime.embedder

    @property
    def _vector_store(self):
        return self._runtime.vector_store

    @property
    def _document_store(self):
        return self._runtime.document_store

    def sync_all(self) -> dict:
        synced = 0
        skipped = 0
        errors = 0

        slugs = self._manager.list_files()
        synced_slugs = []

        for slug in slugs:
            try:
                node_id = f"note:{slug}"
                str_id = (self._store.string_table.intern(node_id)
                          if node_id in self._store.string_table else None)
                if str_id is not None:
                    slot = self._store.id_to_slot.get(str_id)
                    if slot is not None and slot not in self._store.node_tombstones:
                        file_mtime = self._manager.get_mtime(slug)
                        node_mtime = self._read_file_mtime(slot)
                        if node_mtime > 0 and file_mtime <= node_mtime:
                            skipped += 1
                            continue

                self._sync_node(slug)
                synced_slugs.append(slug)
                synced += 1
            except Exception as e:
                logger.debug("vault node sync failed for %r: %s", slug, e, exc_info=True)
                errors += 1

        for slug in synced_slugs:
            try:
                self._sync_edges(slug)
            except Exception as e:
                logger.debug("vault edge sync failed for %r: %s", slug, e, exc_info=True)

        newly_created_nodes = set()
        for slug in synced_slugs:
            node_id = f"note:{slug}"
            if node_id in self._store.string_table:
                newly_created_nodes.add(slug)
        if newly_created_nodes:
            all_slugs = self._manager.list_files()
            for slug in all_slugs:
                if slug in synced_slugs:
                    continue
                try:
                    content = self._manager.read(slug)
                    wikilinks = set(extract_wikilinks(content))
                    if wikilinks & newly_created_nodes:
                        self._sync_edges(slug)
                except Exception as e:
                    logger.debug(
                        "vault pass-3 edge resync failed for %r: %s",
                        slug, e, exc_info=True,
                    )

        return {"synced": synced, "skipped": skipped, "errors": errors}

    def sync_file(self, slug: str) -> str:
        self._sync_node(slug)
        self._sync_edges(slug)
        return f"note:{slug}"

    def _read_file_mtime(self, slot: int) -> float:
        cols = self._store.columns
        field = "__file_mtime__"
        if not cols.has_column(field):
            return 0.0
        if not cols._presence[field][slot]:
            return 0.0
        return float(cols._columns[field][slot])

    def _sync_node(self, slug: str) -> int:
        content = self._manager.read(slug)
        fm = parse_frontmatter(content)
        sections = parse_sections(content)

        node_id = f"note:{slug}"

        fields = {
            "note_kind": fm.get("kind", "memory"),
            "status": fm.get("status", "active"),
            "title": slug,
            "file": f"{slug}.md",
        }

        tags = fm.get("tags", [])
        if isinstance(tags, list):
            fields["tags"] = ",".join(str(t) for t in tags)
        elif isinstance(tags, str):
            fields["tags"] = tags

        if fm.get("agent"):
            fields["agent"] = fm["agent"]

        summary = sections.get("summary", "")
        if summary:
            fields["summary"] = summary[:self._summary_max_length]

        self._store.upsert_node(node_id, "note", fields)

        str_id = self._store.string_table.intern(node_id)
        slot = self._store.id_to_slot.get(str_id)

        if slot is not None:
            file_mtime = self._manager.get_mtime(slug)
            self._store.columns.set_reserved(slot, "__file_mtime__", file_mtime)

        if self._document_store and slot is not None:
            self._document_store.put_document(slot, content.encode("utf-8"), "text/markdown")
            if summary:
                self._document_store.put_summary(slot, summary[:self._summary_max_length], heading=None,
                                                  page=None, chunk_index=0, doc_slot=slot)

        if self._embedder and self._vector_store and slot is not None:
            body = sections.get("body", "")
            embed_text = f"{slug}: {summary} {body}" if summary else body
            if embed_text.strip():
                vec = self._embedder.encode_documents([embed_text])[0]
                if self._vector_store is not None:
                    self._vector_store.add(slot, vec)

        if fm.get("kind") == "fact" and slot is not None:
            confidence = fm.get("confidence", 1.0)
            source = fm.get("source", "vault")
            if isinstance(confidence, (int, float)):
                self._store.columns.set_reserved(slot, "__confidence__", float(confidence))
            self._store.columns.set_reserved(slot, "__source__", str(source))
            self._store.columns.set_reserved(slot, "__retracted__", 0)

        return slot

    def _sync_edges(self, slug: str) -> None:
        content = self._manager.read(slug)
        wikilinks = extract_wikilinks(content)

        node_id = f"note:{slug}"
        if node_id not in self._store.string_table:
            return

        existing_link_edges = self._store.get_edges_from(node_id, kind="links")
        if existing_link_edges:
            try:
                self._store.delete_edges_bulk([
                    (edge["source"], edge["target"], "links")
                    for edge in existing_link_edges
                ])
            except Exception as e:
                logger.debug("vault link edge bulk-delete failed: %s", e, exc_info=True)

        for target_slug in wikilinks:
            target_id = f"note:{target_slug}"
            if target_id in self._store.string_table:
                target_str_id = self._store.string_table.intern(target_id)
                if target_str_id in self._store.id_to_slot:
                    target_slot = self._store.id_to_slot[target_str_id]
                    if target_slot not in self._store.node_tombstones:
                        try:
                            self._store.put_edge(node_id, target_id, "links")
                        except Exception as e:
                            logger.debug("vault link edge creation failed: %s", e, exc_info=True)
