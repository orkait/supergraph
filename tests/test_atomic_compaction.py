import pytest
import sqlite3
import numpy as np
from unittest.mock import patch
from supergraph.core.store import CoreStore
from supergraph.core.schema import SchemaRegistry
from supergraph.document.store import DocumentStore
from supergraph.vector.store import VectorStore
from supergraph.core.optimizer import compact_tombstones_safe
from supergraph.persistence.database import open_database

def test_atomic_compaction_rollback_on_failure(tmp_path):
    db_path = tmp_path / "supergraph.db"
    docs_path = tmp_path / "documents.db"
    
    conn = open_database(str(db_path))
    store = CoreStore()
    schema = SchemaRegistry()
    doc_store = DocumentStore(str(docs_path))
    vec_store = VectorStore(dims=3)
    
    slot0 = store._alloc_slot()
    slot1 = store._alloc_slot()
    slot2 = store._alloc_slot()
    
    store.node_ids[slot0] = 100
    store.node_ids[slot1] = 101
    store.node_ids[slot2] = 102
    
    doc_store.put_document(slot0, b"doc0", "text/plain")
    doc_store.put_document(slot1, b"doc1", "text/plain")
    doc_store.put_document(slot2, b"doc2", "text/plain")
    
    vec_store.add(slot0, np.array([0.1, 0.2, 0.3]))
    vec_store.add(slot1, np.array([0.4, 0.5, 0.6]))
    vec_store.add(slot2, np.array([0.7, 0.8, 0.9]))
    
    store.node_tombstones.add(slot1)
    
    def failing_checkpoint(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated disk full or I/O error during final checkpoint!")

    with patch('supergraph.persistence.serializer.checkpoint', side_effect=failing_checkpoint):
        with pytest.raises(sqlite3.OperationalError, match="Simulated disk full"):
            compact_tombstones_safe(store, schema, conn, vec_store, doc_store)
            
    
    conn.close()
    doc_store.close()
    
    conn2 = open_database(str(db_path))
    doc_store2 = DocumentStore(str(docs_path))
    
    assert doc_store2.has_document(slot0)
    assert doc_store2.has_document(slot1)
    assert doc_store2.has_document(slot2)
    
    conn2.close()
    doc_store2.close()

