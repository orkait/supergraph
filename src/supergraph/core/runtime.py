
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

    from supergraph.core.store import CoreStore
    from supergraph.core.schema import SchemaRegistry
    from supergraph.vector.store import VectorStore
    from supergraph.document.store import DocumentStore
    from supergraph.embedding.base import Embedder


@dataclass
class RuntimeState:
    store: "CoreStore"
    schema: "SchemaRegistry"
    vector_store: "VectorStore | None" = None
    document_store: "DocumentStore | None" = None
    embedder: "Embedder | None" = None
    conn: "sqlite3.Connection | None" = None
    similarity_buffer: Any = field(default_factory=lambda: deque(maxlen=100))
