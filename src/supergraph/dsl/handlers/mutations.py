
import logging
import time

logger = logging.getLogger(__name__)

import numpy as np

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import (
    Batch,
    CreateEdge,
    CreateNode,
    DeleteNode,
    DeleteNodes,
    ForgetNode,
    Increment,
    MergeStmt,
    UpdateNode,
    UpdateNodes,
    UpsertNode,
    VarAssign,
)
from supergraph.core.errors import BatchRollback, SuperGraphError, NodeNotFound
from supergraph.core.types import Result

_ner_unavailable_warned = False


def _warn_ner_unavailable(exc: Exception) -> None:
    global _ner_unavailable_warned
    if not _ner_unavailable_warned:
        _ner_unavailable_warned = True
        logger.warning(
            "entity extraction (NER) unavailable; nodes are stored without "
            "extracted entities. Install onnxruntime + tokenizers and the NER "
            "model to enable it. (%s: %s)",
            type(exc).__name__, exc,
        )


class MutationHandlers:

    def _generate_auto_id(self, kind: str, data: dict) -> str:
        import hashlib
        import json
        parts = [f"kind={kind}"]
        for k in sorted(data.keys()):
            v = data[k]
            if isinstance(v, (str, int, float, bool)) or v is None:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={json.dumps(v, sort_keys=True, default=str)}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _handle_vector(self, slot: int, kind: str, data: dict, explicit_vector: list[float] | None) -> bool:
        if explicit_vector is not None and self._vector_store is not None:
            vec = np.array(explicit_vector, dtype=np.float32)
            self._vector_store.add(slot, vec)
            if self._batch_vector_record is not None:
                self._batch_vector_record.append(slot)
            return True
        if explicit_vector is not None and self._vector_store is None:
            if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                vec = np.array(explicit_vector, dtype=np.float32)
                self._ensure_vector_store_cb(len(vec))
                self._vector_store.add(slot, vec)
                if self._batch_vector_record is not None:
                    self._batch_vector_record.append(slot)
                return True
            return False
        if self._embedder:
            kind_def = self.schema.describe_node_kind(kind)
            if kind_def and kind_def.get("embed_field"):
                embed_field = kind_def["embed_field"]
                text = data.get(embed_field)
                if text and isinstance(text, str):
                    return self._embed_and_store(slot, text)
        return False

    def _try_auto_embed(self, slot: int, kind: str, data: dict) -> bool:
        kind_def = self.schema.describe_node_kind(kind)
        if kind_def and kind_def.get("embed_field"):
            embed_field = kind_def["embed_field"]
            text = data.get(embed_field)
            if text and isinstance(text, str):
                return self._embed_and_store(slot, text)
        return False

    def _embed_and_store(self, slot: int, text: str) -> bool:
        if not self._embedder:
            return False
        if self._vector_store is None:
            if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                self._ensure_vector_store_cb(self._embedder.dims)
            else:
                return False
        if getattr(self, "_defer_embeddings", False):
            self._pending_embeddings.append((slot, text))
            if len(self._pending_embeddings) >= self._embed_batch_size:
                self.flush_pending_embeddings()
            return True
        vec = self._embedder.encode_documents([text])[0]
        self._vector_store.add(slot, vec)
        if self._batch_vector_record is not None:
            self._batch_vector_record.append(slot)
        return True

    def _batch_embed_and_store(self, items: list[tuple[int, str]]) -> None:
        if not self._embedder or not items:
            return
        slots, texts = zip(*items)
        if self._vector_store is None:
            if hasattr(self, '_ensure_vector_store_cb') and self._ensure_vector_store_cb:
                self._ensure_vector_store_cb(self._embedder.dims)
            else:
                return
        vecs = self._embedder.encode_documents(list(texts))
        for slot, vec in zip(slots, vecs):
            self._vector_store.add(slot, vec)
            if self._batch_vector_record is not None:
                self._batch_vector_record.append(slot)

    def flush_pending_embeddings(self) -> None:
        pending = getattr(self, "_pending_embeddings", None)
        if not pending:
            return
        self._batch_embed_and_store(pending)
        pending.clear()

    def _collect_doc_children(self, doc_id: str) -> list[tuple[str, int]]:
        doc_slot = self._resolve_slot(doc_id)
        if doc_slot is None:
            return []
        em = self.store.edge_matrices
        tombstones = self.store.node_tombstones
        slot_to_id = self.store._slot_to_id
        children: list[tuple[str, int]] = []

        def _emit(slot: int) -> None:
            if slot in tombstones:
                return
            nid = slot_to_id(slot)
            if nid is not None:
                children.append((nid, slot))

        for etype in ("has_chunk", "has_image", "has_section"):
            for nb in em.neighbors_out(doc_slot, etype):
                t = int(nb)
                _emit(t)
                if etype == "has_section":
                    for nb2 in em.neighbors_out(t, "has_chunk"):
                        _emit(int(nb2))
        return children

    @staticmethod
    def _parse_event_at(val) -> int | None:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        from supergraph.core.temporal import parse_date
        ms = parse_date(str(val))
        if ms is None:
            raise ValueError(f"Cannot parse EVENT_AT date: {val!r}")
        return ms

    def _apply_event_at(self, node_id: str, event_at) -> None:
        ms = self._parse_event_at(event_at)
        if ms is not None:
            str_id = self.store.string_table.intern(node_id)
            slot = self.store.id_to_slot[str_id]
            self.store.columns.set_reserved(slot, "__event_at__", ms)

    @handles(CreateNode, write=True)
    def _create_node(self, q: CreateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        if q.auto_id:
            node_id = self._generate_auto_id(kind, data)
        else:
            node_id = q.id
        self.store.put_node(node_id, kind, data)
        self._apply_ttl(node_id, q.expires_in, q.expires_at)
        self._apply_event_at(node_id, getattr(q, 'event_at', None))
        if self.store._active_context:
            str_id = self.store.string_table.intern(node_id)
            slot = self.store.id_to_slot[str_id]
            self.store.columns.set_reserved(slot, "__context__", self.store._active_context)

        is_sub_node = kind in ("sentence", "entity")
        
        from supergraph.algos.sentence_split import split_sentences
        from supergraph.ingest.entity_extract import (
            extract_batch, CoReferenceResolver, slug as _ent_slug,
        )

        resolver = CoReferenceResolver()
        entity_model_dir = getattr(self, '_entity_model_dir', None)
        entity_score_threshold = getattr(self, '_entity_score_threshold', 0.6)
        entity_max_length = getattr(self, '_entity_max_length', 256)

        kind_def = self.schema.describe_node_kind(kind)
        embed_field = kind_def.get("embed_field") if kind_def else None
        text_to_split = data.get(embed_field) if embed_field else data.get("content", "")

        enable_sentences = getattr(self, '_enable_sentence_nodes', True)

        sentences = []
        if enable_sentences and not is_sub_node and text_to_split and isinstance(text_to_split, str):
            sentences = split_sentences(text_to_split)

        need_sentence_nodes = enable_sentences and not is_sub_node and len(sentences) >= 1
        if need_sentence_nodes:
            parent_event_ms = self._parse_event_at(getattr(q, 'event_at', None))

            if entity_model_dir:
                try:
                    all_ents = extract_batch(
                        sentences, model_dir=entity_model_dir,
                        score_threshold=entity_score_threshold,
                        max_length=entity_max_length,
                    )
                except (ImportError, FileNotFoundError, OSError) as e:
                    _warn_ner_unavailable(e)
                    all_ents = [[] for _ in sentences]
            else:
                all_ents = [[] for _ in sentences]

            batch_embed_sentences = (
                self._embedder is not None
                and not getattr(self, '_defer_embeddings', False)
            )
            sent_slots: list[int] = []

            for i, sent_text in enumerate(sentences):
                sent_id = f"{node_id}:s{i}"
                sent_slot = self.store.put_node(sent_id, "sentence", {
                    "text": sent_text,
                    "parent_node": node_id,
                })
                self.store.columns.set_reserved(sent_slot, "__blob_state__", "warm")
                if parent_event_ms is not None:
                    self.store.columns.set_reserved(sent_slot, "__event_at__", parent_event_ms)
                self.store.put_edge(node_id, sent_id, "has_sentence")
                sent_slots.append(sent_slot)

                if self._embedder and getattr(self, '_defer_embeddings', False):
                    self._pending_embeddings.append((sent_slot, sent_text))
                    if len(self._pending_embeddings) >= self._embed_batch_size:
                        self.flush_pending_embeddings()

                if entity_model_dir:
                    ents = all_ents[i]
                    sentence_entities: dict[str, str] = {}
                    resolved = resolver.resolve(sent_text)
                    for name in resolved:
                        s = _ent_slug(name)
                        if s:
                            sentence_entities[s] = name
                    for ent in ents:
                        s = _ent_slug(ent.text)
                        if s:
                            sentence_entities[s] = ent.text
                            if ent.label == "PER":
                                resolver.update_context(ent.text)

                    for ent_slug_val, ent_display in sentence_entities.items():
                        ent_id = f"ent:{ent_slug_val}"
                        try:
                            self.store.put_node(ent_id, "entity", {"name": ent_display})
                        except Exception as err:
                            logger.debug("put_node(%s) skipped during mutation entity link: %s", ent_id, err)
                        try:
                            self.store.put_edge(sent_id, ent_id, "mentions")
                        except Exception as err:
                            logger.debug("put_edge(%s -> %s) skipped during mutation entity link: %s", sent_id, ent_id, err)

            if batch_embed_sentences:
                self._batch_embed_and_store(list(zip(sent_slots, sentences)))

        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot[str_id]

        embedded = self._handle_vector(slot, kind, data, q.vector)
            
        if q.document and self._document_store:
            self._document_store.put_document(slot, q.document.encode("utf-8"), "text/plain")
            if not need_sentence_nodes and not embedded and q.vector is None and self._embedder:
                self._embed_and_store(slot, q.document)
        node = self.store.get_node(node_id)
        return Result(kind="node", data=node, count=1)

    @handles(UpdateNode, write=True)
    def _update_node(self, q: UpdateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        self.store.update_node(q.id, data)

        if self._embedder:
            str_id = self.store.string_table.intern(q.id)
            slot = self.store.id_to_slot.get(str_id)
            if slot is not None:
                node_data = self.store.get_node(q.id)
                if node_data:
                    kind = node_data.get("kind", "default")
                    kind_def = self.schema.describe_node_kind(kind)
                    if kind_def and kind_def.get("embed_field"):
                        embed_field = kind_def["embed_field"]
                        if embed_field in data:
                            text = data[embed_field]
                            if text and isinstance(text, str):
                                self._embed_and_store(slot, text)

        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    @handles(UpsertNode, write=True)
    def _upsert_node(self, q: UpsertNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        kind = data.pop("kind", "default")
        self.schema.validate_node(kind, data)
        self.store.upsert_node(q.id, kind, data)
        self._apply_ttl(q.id, q.expires_in, q.expires_at)
        self._apply_event_at(q.id, getattr(q, 'event_at', None))
        str_id = self.store.string_table.intern(q.id)
        slot = self.store.id_to_slot[str_id]
        self._handle_vector(slot, kind, data, q.vector)
        node = self.store.get_node(q.id)
        return Result(kind="node", data=node, count=1)

    @handles(DeleteNode, write=True)
    def _delete_node(self, q: DeleteNode) -> Result:
        slot = self._resolve_slot(q.id)

        if slot is not None:
            kind_str_id = int(self.store.node_kinds[slot])
            kind_name = self.store.string_table.lookup(kind_str_id)
            if kind_name == "document":
                for child_id, child_slot in self._collect_doc_children(q.id):
                    if self._vector_store:
                        self._vector_store.remove(child_slot)
                    if self._document_store:
                        self._document_store.delete_document(child_slot)
                    try:
                        self.store.delete_node(child_id)
                    except NodeNotFound:
                        pass
                if self._document_store:
                    self._document_store.delete_all_for_doc(slot)

        if self._vector_store and slot is not None:
            self._vector_store.remove(slot)
        if self._document_store and slot is not None:
            try:
                self._document_store.delete_document(slot)
            except Exception as _doc_err:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "DocumentStore.delete_document failed for slot=%s: %s", slot, _doc_err
                )
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"id": q.id}, count=1)

    @handles(DeleteNodes, write=True)
    def _delete_nodes(self, q: DeleteNodes) -> Result:
        kind_filter = self._extract_kind_from_where(q.where)
        remaining = self._strip_kind_from_expr(q.where.expr)

        ids_to_delete = None

        if remaining is not None:
            col_ids = self._try_column_delete_ids(remaining, kind_filter)
            if col_ids is not None:
                ids_to_delete = col_ids

        if ids_to_delete is None:
            if remaining is not None:
                raw_pred = self._make_raw_predicate(remaining)
            else:
                raw_pred = None

            if raw_pred is not None or remaining is None:
                ids_to_delete = self.store.query_node_ids(kind=kind_filter, predicate=raw_pred)
            else:
                nodes = self.store.get_all_nodes(kind=kind_filter)
                ids_to_delete = [n["id"] for n in nodes if self._eval_where(q.where.expr, n)]

        deleted_ids = self.store.delete_nodes_bulk(
            ids_to_delete,
            vector_store=self._vector_store,
            document_store=self._document_store,
        )
        return Result(kind="nodes", data=[{"id": i} for i in deleted_ids], count=len(deleted_ids))

    @handles(UpdateNodes, write=True)
    def _update_nodes(self, q: UpdateNodes) -> Result:
        n = self.store._next_slot
        if n == 0:
            return Result(kind="ok", data={"updated": 0}, count=0)

        mask = self._compute_live_mask(n)

        kind_filter = self._extract_kind_from_where(q.where)
        if kind_filter:
            kind_mask = self.store._live_mask(kind_filter)
            mask = mask & kind_mask
            remaining = self._strip_kind_from_expr(q.where.expr)
            if remaining is not None:
                col_mask = self._try_column_filter(remaining, mask, n)
                if col_mask is not None:
                    mask = col_mask
                else:
                    fallback_mask = np.zeros(n, dtype=bool)
                    for slot_idx in np.nonzero(mask)[0]:
                        node = self.store._materialize_slot(int(slot_idx))
                        if node and self._eval_where(q.where.expr, node):
                            fallback_mask[int(slot_idx)] = True
                    mask = fallback_mask
        else:
            col_mask = self._try_column_filter(q.where.expr, mask, n)
            if col_mask is not None:
                mask = col_mask
            else:
                fallback_mask = np.zeros(n, dtype=bool)
                for slot_idx in np.nonzero(mask)[0]:
                    node = self.store._materialize_slot(int(slot_idx))
                    if node and self._eval_where(q.where.expr, node):
                        fallback_mask[int(slot_idx)] = True
                mask = fallback_mask

        matching_slots = np.nonzero(mask)[0]
        now_ms = int(time.time() * 1000)

        if len(matching_slots) > 0:
            store = self.store
            for fp in q.fields:
                field = fp.name
                value = fp.value
                if store.columns.has_column(field):
                    dtype_str = store.columns._dtypes[field]
                    if dtype_str == "int32_interned" and isinstance(value, str):
                        raw_val = store.string_table.intern(value)
                        store.columns._columns[field][matching_slots] = raw_val
                        store.columns._presence[field][matching_slots] = True
                    elif dtype_str == "int64" and isinstance(value, int):
                        store.columns._columns[field][matching_slots] = int(value)
                        store.columns._presence[field][matching_slots] = True
                    elif dtype_str == "float64" and isinstance(value, (int, float)) and not isinstance(value, bool):
                        store.columns._columns[field][matching_slots] = float(value)
                        store.columns._presence[field][matching_slots] = True
                    else:
                        for slot_idx in matching_slots:
                            store.columns.set(int(slot_idx), {field: value})
                else:
                    for slot_idx in matching_slots:
                        store.columns.set(int(slot_idx), {field: value})

            if store.columns.has_column("__updated_at__"):
                store.columns._columns["__updated_at__"][matching_slots] = now_ms
                store.columns._presence["__updated_at__"][matching_slots] = True
            else:
                for slot_idx in matching_slots:
                    store.columns.set_reserved(int(slot_idx), "__updated_at__", now_ms)

        for fp in q.fields:
            if fp.name in self.store._indexed_fields:
                self.store.reindex_slots(fp.name, matching_slots)

        if self._embedder and self._vector_store is not None:
            updated_fields = {fp.name for fp in q.fields}
            reembed_batch: list[tuple[int, str]] = []
            for slot_idx in matching_slots:
                slot = int(slot_idx)
                node = self.store._materialize_slot(slot)
                if node is None:
                    continue
                kind = node.get("kind", "default")
                kind_def = self.schema.describe_node_kind(kind)
                if kind_def and kind_def.get("embed_field"):
                    embed_field = kind_def["embed_field"]
                    if embed_field in updated_fields:
                        text = node.get(embed_field)
                        if text and isinstance(text, str):
                            reembed_batch.append((slot, text))
            if reembed_batch:
                self._batch_embed_and_store(reembed_batch)

        updated = len(matching_slots)
        return Result(kind="ok", data={"updated": updated}, count=updated)

    @handles(Increment, write=True)
    def _increment(self, q: Increment) -> Result:
        self.store.increment_field(q.node_id, q.field, q.amount)
        return Result(kind="ok", data=None, count=0)

    @handles(MergeStmt, write=True)
    def _merge(self, q: MergeStmt) -> Result:
        if q.source_id not in self.store.string_table:
            raise NodeNotFound(q.source_id)
        src_str = self.store.string_table.intern(q.source_id)
        src_slot = self.store.id_to_slot.get(src_str)
        if src_slot is None or src_slot in self.store.node_tombstones:
            raise NodeNotFound(q.source_id)

        if q.target_id not in self.store.string_table:
            raise NodeNotFound(q.target_id)
        tgt_str = self.store.string_table.intern(q.target_id)
        tgt_slot = self.store.id_to_slot.get(tgt_str)
        if tgt_slot is None or tgt_slot in self.store.node_tombstones:
            raise NodeNotFound(q.target_id)

        if src_slot == tgt_slot:
            raise SuperGraphError(
                f"MERGE source and target resolve to the same slot: "
                f"{q.source_id!r} == {q.target_id!r}"
            )

        snap = self.store.make_snapshot()

        try:
            fields_merged = 0
            for field in list(self.store.columns._columns.keys()):
                if field.startswith("__") and field.endswith("__"):
                    continue
                if not self.store.columns._presence[field][src_slot]:
                    continue
                if self.store.columns._presence[field][tgt_slot]:
                    continue
                raw = self.store.columns._columns[field][src_slot]
                self.store.columns._columns[field][tgt_slot] = raw
                self.store.columns._presence[field][tgt_slot] = True
                fields_merged += 1

            from supergraph.algos.edges_ops import (
                rewire_edges_source_target,
                dedupe_edges_by_src_tgt,
                rebuild_edge_keys_set,
            )
            rewired_edges, edges_rewired = rewire_edges_source_target(
                self.store._edges_by_type, src_slot, tgt_slot,
            )
            self.store._edges_by_type = dedupe_edges_by_src_tgt(rewired_edges)
            self.store._edge_keys = rebuild_edge_keys_set(self.store._edges_by_type)
            self.store._edges_dirty = True
            self.store._ensure_edges_built()

            self.store.delete_node(q.source_id)

            now_ms = int(time.time() * 1000)
            self.store.columns.set_reserved(tgt_slot, "__updated_at__", now_ms)

            return Result(
                kind="ok",
                data={
                    "merged_into": q.target_id,
                    "fields_merged": fields_merged,
                    "edges_rewired": edges_rewired,
                },
                count=1,
            )
        except Exception:
            self.store.restore_snapshot(snap)
            self.store._rebuild_edges()
            raise

    @handles(ForgetNode, write=True)
    def _forget(self, q: ForgetNode) -> Result:
        slot = self._resolve_slot(q.id)
        if slot is None:
            raise NodeNotFound(q.id)

        kind_str_id = int(self.store.node_kinds[slot])
        kind_name = self.store.string_table.lookup(kind_str_id)
        if kind_name == "document":
            for child_id, child_slot in self._collect_doc_children(q.id):
                if self._vector_store:
                    self._vector_store.remove(child_slot)
                if self._document_store:
                    self._document_store.delete_document(child_slot)
                try:
                    self.store.delete_node(child_id)
                except NodeNotFound:
                    pass
            if self._document_store:
                self._document_store.delete_all_for_doc(slot)

        if self._vector_store:
            self._vector_store.remove(slot)
        if self._document_store:
            self._document_store.delete_document(slot)
            try:
                self._document_store._conn.execute("DELETE FROM doc_fts WHERE rowid = ?", (slot,))
                self._document_store._conn.commit()
            except Exception as _fts_err:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "FORGET: doc_fts delete failed for slot=%s: %s", slot, _fts_err
                )
        self.store.delete_node(q.id)
        return Result(kind="ok", data={"forgotten": q.id}, count=1)

    @handles(Batch, write=True)
    def _batch(self, q: Batch) -> Result:
        enable_rollback = getattr(self, '_enable_rollback', True)

        prev_record = self._batch_vector_record
        self._batch_vector_record = [] if enable_rollback else None

        snap = self.store.make_snapshot() if enable_rollback else None

        try:
            variables: dict[str, str] = {}
            for stmt in q.statements:
                if isinstance(stmt, VarAssign):
                    result = self._dispatch(stmt.statement)
                    if result.data and isinstance(result.data, dict) and "id" in result.data:
                        variables[stmt.variable] = result.data["id"]
                    else:
                        raise SuperGraphError(
                            f"Variable {stmt.variable}: statement did not return an ID"
                        )
                else:
                    if isinstance(stmt, CreateEdge):
                        src = variables.get(stmt.source, stmt.source) if stmt.source.startswith("$") else stmt.source
                        tgt = variables.get(stmt.target, stmt.target) if stmt.target.startswith("$") else stmt.target
                        if src.startswith("$"):
                            raise SuperGraphError(f"Unresolved variable: {src}")
                        if tgt.startswith("$"):
                            raise SuperGraphError(f"Unresolved variable: {tgt}")
                        resolved = CreateEdge(source=src, target=tgt, fields=stmt.fields)
                        self._dispatch(resolved)
                    else:
                        self._dispatch(stmt)
            self.store._ensure_edges_built()
            return Result(kind="ok", data=None, count=0)
        except Exception as e:
            if enable_rollback and self._batch_vector_record:
                vs = self._vector_store
                if vs is not None:
                    for slot in self._batch_vector_record:
                        try:
                            vs.remove(slot)
                        except Exception:
                            pass
            if enable_rollback and snap is not None:
                self.store.restore_snapshot(snap)
                self.store._rebuild_edges()
            raise BatchRollback(
                failed_statement=str(type(e).__name__), error=str(e)
            )
        finally:
            self._batch_vector_record = prev_record
