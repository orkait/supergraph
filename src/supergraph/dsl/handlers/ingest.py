"""Ingest and connect handlers for the DSL executor."""

import hashlib
import logging
from pathlib import Path as _Path


logger = logging.getLogger(__name__)

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import IngestStmt, ConnectNode
from supergraph.core.errors import SuperGraphError
from supergraph.core.types import Result


class IngestHandlers:

    @staticmethod
    def _infer_event_at_from_metadata(metadata: dict) -> int | None:
        """Best-effort event time extraction from parser metadata."""
        from supergraph.core.temporal import parse_date

        for key in ("event_at", "event_time", "date", "published_at", "published", "timestamp"):
            value = metadata.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return int(value)
            ms = parse_date(str(value))
            if ms is not None:
                return ms
        return None

    @handles(IngestStmt, write=True)
    def _ingest(self, q: IngestStmt) -> Result:
        """INGEST: parse file, chunk, create graph nodes + edges, store documents."""
        from supergraph.ingest.router import ingest_file
        import os as _os

        resolved = _Path(q.file_path).resolve()
        # Also resolve the real path following symlinks to prevent symlink traversal
        real_resolved = _Path(_os.path.realpath(resolved))
        if self._ingest_root:
            root = _Path(self._ingest_root).resolve()
            real_root = _Path(_os.path.realpath(root))
            root_str = str(real_root)
            # Use os.sep to prevent prefix collision (/data vs /data2)
            if str(real_resolved) != root_str and not str(real_resolved).startswith(root_str + _os.sep):
                raise SuperGraphError(
                    f"Path traversal not allowed: {q.file_path} "
                    f"is outside ingest root {self._ingest_root}"
                )
        if not resolved.exists():
            raise SuperGraphError(f"File not found: {q.file_path}")

        safe_path = str(resolved)
        ext = resolved.suffix.lstrip(".").lower()

        image_exts = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}
        if ext in image_exts and q.vision_model:
            return self._ingest_image_with_vision(q, safe_path, ext)

        if self._ingestor_registry is not None:
            ingestor = self._ingestor_registry.resolve(safe_path, using=q.using)
            result = ingestor.convert(safe_path)
        else:
            result = ingest_file(safe_path, using=q.using)

        logger.info("[ingest] %s → %s (%d chars, parser=%s, %.1fs)",
                     resolved.name, safe_path, len(result.markdown),
                     result.parser_used, result.elapsed if hasattr(result, 'elapsed') else 0)

        chunk_size = getattr(self, '_chunk_max_size', 2000)
        summary_len = getattr(self, '_summary_max_length', 200)
        chunk_overlap = getattr(self, '_chunk_overlap', 50)

        # Ingestors may pre-chunk (e.g. pymupdf4llm preserves per-page
        # provenance by returning page-scoped chunks with .page set). When
        # IngestResult.chunks is already populated, skip re-chunking - we'd
        # lose the page numbers.
        if result.chunks:
            chunks = result.chunks
            logger.info("[ingest] using %d pre-chunked segments from %s",
                        len(chunks), result.parser_used)
        elif self._chunker is not None:
            chunks = self._chunker.chunk(
                result.markdown,
                max_chunk_size=chunk_size,
                summary_max_len=summary_len,
                overlap=chunk_overlap,
            )
        else:
            from supergraph.ingest.chunker import chunk_by_heading
            logger.info("[ingest] chunking %d chars (size=%d, overlap=%d) ...",
                        len(result.markdown), chunk_size, chunk_overlap)
            chunks = chunk_by_heading(
                result.markdown,
                max_chunk_size=chunk_size,
                summary_max_len=summary_len,
                overlap=chunk_overlap,
            )

        parent_id = q.node_id
        if not parent_id:
            h = hashlib.sha256(q.file_path.encode()).hexdigest()[:12]
            parent_id = f"doc:{h}"

        parent_kind = q.kind or "document"

        existing = self.store.get_node(parent_id)
        if existing is not None:
            raise SuperGraphError(f"Node already exists: {parent_id}")

        metadata_fields = {
            "source": q.file_path,
            "parser": result.parser_used,
            "confidence": result.confidence,
        }
        metadata_fields.update({
            k: v for k, v in result.metadata.items()
            if isinstance(v, (str, int, float)) and k not in ("source",)
        })

        # Snapshot before any mutation so a mid-INGEST failure (embedding
        # crash, disk full, VLM timeout) can roll the graph back to the
        # exact pre-INGEST state. Pre-fix, a failure after some chunks
        # were created left the parent node, document-store row, and
        # partial section/chunk nodes persisted with no indication of
        # partial state (bug #48). DocumentStore writes are tracked
        # separately since they live outside the StoreSnapshot primitive.
        store_snap = self.store.make_snapshot()
        doc_slots_written: list[int] = []
        try:
            return self._ingest_body(
                q, result, chunks, parent_id, parent_kind,
                metadata_fields, doc_slots_written,
            )
        except Exception:
            # Roll back the store to its pre-INGEST state. Then unwind
            # DocumentStore writes we tracked — restore_snapshot doesn't
            # touch DocumentStore since documents live outside the store.
            self.store.restore_snapshot(store_snap)
            self.store._rebuild_edges()
            if self._document_store:
                for slot in doc_slots_written:
                    try:
                        self._document_store.delete_document(slot)
                    except Exception:
                        logger.debug(
                            "INGEST rollback: doc delete for slot=%s failed",
                            slot, exc_info=True,
                        )
            raise

    def _ingest_body(self, q, result, chunks, parent_id, parent_kind,
                     metadata_fields, doc_slots_written: list[int]) -> Result:
        """Internal ingest body, extracted so the top-level _ingest can wrap
        the whole thing in a StoreSnapshot + DocumentStore rollback (bug #48)."""
        import time
        parent_slot = self.store.put_node(parent_id, parent_kind, metadata_fields)
        self.store.columns.set_reserved(parent_slot, "__blob_state__", "warm")
        event_at_ms = self._infer_event_at_from_metadata(result.metadata)
        if event_at_ms is not None:
            self.store.columns.set_reserved(parent_slot, "__event_at__", event_at_ms)

        if self._document_store:
            self._document_store.put_document(
                parent_slot, result.markdown.encode("utf-8"), "text/markdown")
            doc_slots_written.append(parent_slot)
            self._document_store.put_metadata(parent_slot, {
                "source_path": q.file_path,
                "pages": result.metadata.get("pages"),
                "author": result.metadata.get("author"),
                "title": result.metadata.get("title"),
                "parser_used": result.parser_used,
                "confidence": result.confidence,
                "ingested_at": int(time.time() * 1000),
            })

        embed_batch: list[tuple[int, str]] = []
        sections: dict[str, str] = {}
        section_slots: dict[str, int] = {}
        set_reserved = self.store.columns.set_reserved
        for chunk in chunks:
            if chunk.heading and chunk.heading not in sections:
                section_id = f"{parent_id}:section:{len(sections)}"
                sec_slot = self.store.put_node(section_id, "section", {
                    "heading": chunk.heading,
                    "summary": chunk.summary[:200],
                })
                set_reserved(sec_slot, "__confidence__", 0.6)
                set_reserved(sec_slot, "__blob_state__", "warm")
                self.store.put_edge(parent_id, section_id, "has_section")
                sections[chunk.heading] = section_id
                section_slots[chunk.heading] = sec_slot

        chunk_ids = []
        ds = self._document_store
        embed_batch: list[tuple[int, str]] = []
        logger.info("[ingest] creating %d chunks for %s ...", len(chunks), parent_id)

        # Entity extraction config
        entity_model_dir = getattr(self, '_entity_model_dir', None)
        entity_score_threshold = getattr(self, '_entity_score_threshold', 0.6)
        entity_max_length = getattr(self, '_entity_max_length', 256)

        entity_seen: dict[str, str] = {}  # slug -> display_name

        # Pre-batch NER across all chunks (1 ONNX run instead of N)
        chunk_ents: list[list] = [[] for _ in chunks]
        if entity_model_dir:
            from supergraph.ingest.entity_extract import extract_batch, slug as _ent_slug
            try:
                chunk_ents = extract_batch(
                    [c.text for c in chunks], model_dir=entity_model_dir,
                    score_threshold=entity_score_threshold, max_length=entity_max_length,
                )
            except (ImportError, FileNotFoundError, OSError) as e:
                # NER is opt-in: skip entities rather than failing ingest when
                # onnxruntime/tokenizers or the model files are unavailable.
                logger.warning(
                    "entity extraction (NER) unavailable during ingest; chunks "
                    "stored without entities (%s: %s)", type(e).__name__, e,
                )

        for i, chunk in enumerate(chunks):
            chunk_id = f"{parent_id}:chunk:{chunk.index}"
            chunk_fields = {"summary": chunk.summary}
            if chunk.heading:
                chunk_fields["heading"] = chunk.heading
            if chunk.page is not None:
                chunk_fields["page"] = chunk.page

            chunk_slot = self.store.put_node(chunk_id, "chunk", chunk_fields)
            set_reserved(chunk_slot, "__blob_state__", "warm")
            chunk_ids.append(chunk_id)

            if ds:
                ds._conn.execute(
                    "INSERT OR REPLACE INTO documents (slot, content, content_type, size) VALUES (?, ?, ?, ?)",
                    (chunk_slot, chunk.text.encode("utf-8"), "text/markdown", len(chunk.text)))
                ds._conn.execute(
                    "INSERT OR REPLACE INTO summaries VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_slot, chunk.summary, chunk.heading, chunk.page, chunk.index, parent_slot))
                fts_text = chunk.text if getattr(self, '_fts_full_text', True) else chunk.summary
                ds._conn.execute(
                    "INSERT OR REPLACE INTO doc_fts (rowid, summary) VALUES (?, ?)",
                    (chunk_slot, fts_text))

            embed_text = f"{chunk.heading}: {chunk.text}" if chunk.heading else chunk.text

            if chunk.heading and chunk.heading in sections:
                self.store.put_edge(sections[chunk.heading], chunk_id, "has_chunk")
            else:
                self.store.put_edge(parent_id, chunk_id, "has_chunk")

            # Entity linking (use pre-computed batch result)
            if entity_model_dir:
                for ent in chunk_ents[i]:
                    s = _ent_slug(ent.text)
                    if not s:
                        continue
                    if s not in entity_seen:
                        entity_seen[s] = ent.text
                    ent_id = f"ent:{s}"
                    try:
                        self.store.put_node(ent_id, "entity", {"name": ent.text})
                    except Exception as err:
                        logger.debug("put_node(%s) skipped during ingest entity link: %s", ent_id, err)
                    try:
                        self.store.put_edge(chunk_id, ent_id, "mentions")
                    except Exception as err:
                        logger.debug("put_edge(%s -> %s) skipped during ingest entity link: %s", chunk_id, ent_id, err)

            # Embed chunk text for vector retrieval
            embed_batch.append((chunk_slot, embed_text))
            
        if ds:
            ds._conn.commit()

        image_count = 0
        vision_handler = None
        if q.vision_model:
            try:
                from supergraph.ingest.vision import VisionHandler
                vision_handler = VisionHandler(
                    model=q.vision_model,
                    base_url=getattr(self, '_vision_base_url', None),
                    max_tokens=getattr(self, '_vision_max_tokens', 512),
                )
            except Exception as e:
                logger.debug("vision handler init failed: %s", e, exc_info=True)

        for i, img in enumerate(result.images):
            img_id = f"{parent_id}:image:{i}"
            img_fields = {}
            if img.page is not None:
                img_fields["page"] = img.page

            if not img.description and vision_handler:
                try:
                    img.description = vision_handler.describe(img.data, img.mime_type)
                except Exception as e:
                    logger.debug("image description failed: %s", e, exc_info=True)

            if img.description:
                img_fields["summary"] = img.description

            img_slot = self.store.put_node(img_id, "image", img_fields)
            set_reserved(img_slot, "__blob_state__", "warm")

            if ds:
                ds.put_image(img_slot, img.data, img.mime_type, img.page, img.description)

            if img.description:
                embed_batch.append((img_slot, img.description))

            self.store.put_edge(parent_id, img_id, "has_image")
            image_count += 1

        self._batch_embed_and_store(embed_batch)

        logger.info("[ingest] %s done: %d sections, %d chunks, %d images",
                     parent_id, len(sections), len(chunks), image_count)

        # Surface ingestor-level warnings (e.g. scanned PDF with near-zero
        # text extraction) to the caller. Without this, users only discover
        # a low-confidence ingest when REMEMBER returns nothing against the
        # new doc.
        meta: dict = {}
        warnings: list[str] = []
        ing_warning = result.metadata.get("warning")
        if ing_warning:
            warnings.append(ing_warning)
        if result.confidence < 0.5:
            warnings.append(
                f"Ingest confidence {result.confidence:.2f} (parser={result.parser_used}). "
                f"Text quality may be insufficient for retrieval."
            )
        if warnings:
            meta["warnings"] = warnings

        return Result(kind="ok", data={
            "doc_id": parent_id,
            "chunks": len(chunks),
            "sections": len(sections),
            "images": image_count,
            "parser": result.parser_used,
            "confidence": result.confidence,
        }, count=len(chunks), meta=meta)

    def _ingest_image_with_vision(self, q: IngestStmt, safe_path: str, ext: str) -> Result:
        """Handle standalone image ingest with VLM description."""
        from supergraph.ingest.vision import VisionHandler

        with open(safe_path, "rb") as f:
            image_bytes = f.read()

        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
                    "tiff": "image/tiff"}
        mime_type = mime_map.get(ext, "image/png")

        vh = VisionHandler(
            model=q.vision_model,
            base_url=getattr(self, '_vision_base_url', None),
            max_tokens=getattr(self, '_vision_max_tokens', 512),
        )
        description = vh.describe(image_bytes, mime_type)

        node_id = q.node_id
        if not node_id:
            h = hashlib.sha256(q.file_path.encode()).hexdigest()[:12]
            node_id = f"img:{h}"

        existing = self.store.get_node(node_id)
        if existing is not None:
            raise SuperGraphError(f"Node already exists: {node_id}")

        node_kind = q.kind or "image"
        fields = {
            "summary": description,
            "source": q.file_path,
            "mime_type": mime_type,
        }
        slot = self.store.put_node(node_id, node_kind, fields)
        self.store.columns.set_reserved(slot, "__blob_state__", "warm")

        if self._document_store:
            self._document_store.put_image(slot, image_bytes, mime_type, description=description)
            self._document_store.put_summary(slot, description)

        self._embed_and_store(slot, description)

        return Result(kind="ok", data={
            "doc_id": node_id,
            "chunks": 0,
            "sections": 0,
            "images": 1,
            "parser": "vision",
            "confidence": 0.8,
        }, count=1)

    @handles(ConnectNode, write=True)
    def _connect_node(self, q: ConnectNode) -> Result:
        """CONNECT NODE: wire one node to similar neighbors via vector similarity."""
        from supergraph.ingest.connector import connect_node as _connect_node_fn
        return _connect_node_fn(
            self.store, self._vector_store,
            q.node_id, threshold=q.threshold,
        )
