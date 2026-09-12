"""Mutation handlers for the DSL executor (create, update, delete, merge, batch)."""

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

import numpy as np
from scipy.sparse import csr_matrix

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import (
    Batch,
    ConnectNode,
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
    """Log once that NER is unavailable. Entity extraction is opt-in (onnxruntime
    + tokenizers + model files); when it is missing, writes proceed without
    entities instead of failing."""
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
        """Generate a deterministic content-hash ID from kind + sorted fields.

        Serializes each field value through ``json.dumps(..., sort_keys=True)``
        so that nested dicts, lists, and scalar values all produce a stable
        canonical form. Pre-fix, ``str(value)`` was used which produces
        dict reprs whose key ordering depends on insertion order — two
        callers building semantically-equal dicts via different code paths
        could hash to different auto-IDs (bug #30). The ``default=str``
        fallback catches non-JSON-serializable types (datetimes, sets) by
        rendering them via their string repr, which is still deterministic
        within a single process.
        """
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
        """Handle explicit VECTOR clause or auto-embed from schema EMBED field.

        Returns True if a vector was stored or queued for this slot.
        """
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
        """Auto-embed if schema has EMBED field defined for this kind."""
        kind_def = self.schema.describe_node_kind(kind)
        if kind_def and kind_def.get("embed_field"):
            embed_field = kind_def["embed_field"]
            text = data.get(embed_field)
            if text and isinstance(text, str):
                return self._embed_and_store(slot, text)
        return False

    def _embed_and_store(self, slot: int, text: str) -> bool:
        """Embed text and store vector at slot.

        In deferred mode (set via deferred_embeddings context), appends to
        pending queue and auto-flushes when the batch size is reached.
        Otherwise embeds immediately. Handles lazy vector store init.

        Returns True if the embedding was stored or queued.
        """
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
        """Batch-embed multiple (slot, text) pairs in one model call."""
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
        """Flush any pending embeddings queued in deferred mode."""
        pending = getattr(self, "_pending_embeddings", None)
        if not pending:
            return
        self._batch_embed_and_store(pending)
        pending.clear()

    def _collect_doc_children(self, doc_id: str) -> list[tuple[str, int]]:
        """Collect all (node_id, slot) pairs for a document's children.

        Uses built CSR matrices instead of scanning raw edge lists - O(fanout)
        per edge type instead of O(|E|) per type, and section -> chunk descent
        becomes two neighbor lookups rather than N*M list scans.
        """
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
        """Parse EVENT_AT value to epoch ms using the core temporal parser."""
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
        """Set __event_at__ reserved column from EVENT_AT clause."""
        ms = self._parse_event_at(event_at)
        if ms is not None:
            str_id = self.store.string_table.intern(node_id)
            slot = self.store.id_to_slot[str_id]
            self.store.columns.set_reserved(slot, "__event_at__", ms)

    @handles(CreateNode, write=True)
    def _create_node(self, q: CreateNode) -> Result:
        """Create a node. Text content has four distinct landing pads:

        | Clause                  | ColumnStore | DocumentStore blob | doc_fts BM25 | Embedded |
        |-------------------------|-------------|--------------------|--------------|----------|
        | Regular field (text=X)  | yes         | no                 | no           | iff schema has EMBED <field> |
        | DOCUMENT "..."          | no          | yes                | yes (text/*) | fallback when no sentences + no VECTOR |
        | VECTOR [...]            | no          | no                 | no           | yes (direct) |
        | put_summary / INGEST    | no          | yes (via ingest)   | yes          | per-sentence via pipeline |

        Pick the one that matches what you want REMEMBER to find:
          - Column field with EMBED  -> semantic (vec) search
          - DOCUMENT "plaintext"     -> both vec (fallback) and BM25 work
          - INGEST file.pdf          -> full pipeline, sentence-level vectors + BM25
        """
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

        # ── Sentence splitting + entity extraction ──────────────
        # Only split if this is a primary node, not a sentence/entity already
        is_sub_node = kind in ("sentence", "entity")
        
        from supergraph.algos.sentence_split import split_sentences
        from supergraph.ingest.entity_extract import (
            extract_batch, CoReferenceResolver, slug as _ent_slug,
        )

        resolver = CoReferenceResolver()
        entity_model_dir = getattr(self, '_entity_model_dir', None)
        entity_score_threshold = getattr(self, '_entity_score_threshold', 0.6)
        entity_max_length = getattr(self, '_entity_max_length', 256)

        # Get text to split from embed field or content
        kind_def = self.schema.describe_node_kind(kind)
        embed_field = kind_def.get("embed_field") if kind_def else None
        text_to_split = data.get(embed_field) if embed_field else data.get("content", "")

        enable_sentences = getattr(self, '_enable_sentence_nodes', True)

        sentences = []
        if enable_sentences and not is_sub_node and text_to_split and isinstance(text_to_split, str):
            sentences = split_sentences(text_to_split)

        need_sentence_nodes = enable_sentences and not is_sub_node and len(sentences) >= 1
        if need_sentence_nodes:
            # Parse parent's EVENT_AT for inheritance by sentence nodes
            parent_event_ms = self._parse_event_at(getattr(q, 'event_at', None))

            # Batch NER across all sentences (1 ONNX run instead of N).
            # NER is opt-in: degrade to no entities rather than failing the
            # write when onnxruntime/tokenizers or the model files are absent.
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

            # Batch sentence embeddings when embedder supports it and not deferred
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

                # Embed (deferred path only; batched path runs after the loop)
                if self._embedder and getattr(self, '_defer_embeddings', False):
                    self._pending_embeddings.append((sent_slot, sent_text))
                    if len(self._pending_embeddings) >= self._embed_batch_size:
                        self.flush_pending_embeddings()

                # Entity linking (use pre-computed batch result)
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

        # ── Standard embedding / document handling ──────────────
        # Sentences supplement the parent for fine-grained retrieval.
        # Parent is still embedded for direct retrieval and update-re-embed.
        str_id = self.store.string_table.intern(node_id)
        slot = self.store.id_to_slot[str_id]

        embedded = self._handle_vector(slot, kind, data, q.vector)
            
        if q.document and self._document_store:
            self._document_store.put_document(slot, q.document.encode("utf-8"), "text/plain")
            # Only fallback-embed parent if no sentence nodes and no explicit vector
            if not need_sentence_nodes and not embedded and q.vector is None and self._embedder:
                self._embed_and_store(slot, q.document)
        node = self.store.get_node(node_id)
        return Result(kind="node", data=node, count=1)

    @handles(UpdateNode, write=True)
    def _update_node(self, q: UpdateNode) -> Result:
        data = {fp.name: fp.value for fp in q.fields}
        self.store.update_node(q.id, data)

        # Auto re-embed if an embed field was updated
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
        """UPDATE NODES WHERE ... SET ...: bulk column update."""
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

        update_data = {fp.name: fp.value for fp in q.fields}
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
                # Incremental reindex of only the affected slots instead
                # of a full table rescan. For a single-slot UPDATE on a
                # 1M-row table this is 1M× faster (bug #32).
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
        """MERGE NODE src INTO tgt: copy fields, rewire edges, tombstone source.

        Snapshotted for atomic rollback on failure.
        """
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

        # Centralized snapshot captures everything including the fields the
        # pre-fix hand-rolled version missed: string_table, secondary_indices,
        # _indexed_fields, _edge_data_idx (bug #36).
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
                dtype = self.store.columns._dtypes[field]
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
            # Single restore undoes every field the centralized primitive
            # captured, including the index structures the old rollback
            # missed. _rebuild_edges still runs so the CSR matrices match
            # the restored _edges_by_type.
            self.store.restore_snapshot(snap)
            self.store._rebuild_edges()
            raise

    @handles(ForgetNode, write=True)
    def _forget(self, q: ForgetNode) -> Result:
        """FORGET NODE: hard delete blob + vector + memory (irreversible)."""
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
        """Execute batch with rollback on failure."""
        enable_rollback = getattr(self, '_enable_rollback', True)

        prev_record = self._batch_vector_record
        self._batch_vector_record = [] if enable_rollback else None

        # Only pay the snapshot cost when rollback is actually requested.
        # This preserves the pre-fix behavior where BATCH ... NOROLLBACK
        # (the non-transactional fast path) skipped the store-wide copy.
        # Centralized primitive captures everything the 9-line hand-rolled
        # version did, plus the fields it was missing — string_table,
        # secondary_indices, _indexed_fields, _edge_data_idx (bug #8).
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
            # Vector state: BATCH tracks mid-batch VECTOR assignments so we
            # can unwind them out-of-band. The centralized primitive doesn't
            # touch runtime.vector_store on purpose.
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
