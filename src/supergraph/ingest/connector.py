"""Auto-wire cross-document relationships via vector similarity."""
import logging
from supergraph.core.types import Result

logger = logging.getLogger(__name__)


def connect_all(store, vector_store, threshold=0.85, where_expr=None, executor=None,
                cancel_event=None, progress_callback=None):
    """Find and wire similar chunks across documents.

    Runs an O(N * log N) scan: one k-NN query per live vector-bearing
    slot. On a 100K-node graph, this means 100K HNSW queries — easily
    multi-minute territory (bug #60). Callers should pass a ``cancel_event``
    (a ``threading.Event``) to allow aborting via SYS CRON STOP or similar
    lifecycle machinery. The check is cheap enough to run every iteration.

    ``progress_callback`` receives ``(checked_count, total)`` every 100
    iterations so long-running operations can report back to a status log.
    """
    if vector_store is None or vector_store.count() == 0:
        return Result(kind="ok", data={"edges_created": 0}, count=0)

    n = store._next_slot
    live = store.compute_live_mask(n)

    edges_created = 0
    checked = set()

    for slot in range(n):
        # Cancellation check — cheap boolean read.
        if cancel_event is not None and cancel_event.is_set():
            logger.info(
                "connect_all cancelled at slot %d/%d, %d edges created",
                slot, n, edges_created,
            )
            break

        if progress_callback is not None and slot % 100 == 0:
            try:
                progress_callback(slot, n)
            except Exception as err:
                logger.debug("user progress_callback raised (ignored): %s", err)

        if not live[slot] or not vector_store.has_vector(slot):
            continue

        vec = vector_store.get_vector(slot)
        # Find top-10 similar (oversample to filter self + same-doc)
        results_slots, dists = vector_store.search(vec, k=10, mask=live)

        for other_slot, dist in zip(results_slots, dists):
            other_slot = int(other_slot)
            if other_slot == slot:
                continue

            similarity = 1.0 - float(dist)
            if similarity < threshold:
                continue

            pair = (min(slot, other_slot), max(slot, other_slot))
            if pair in checked:
                continue
            checked.add(pair)

            # Check if edge already exists
            src_id = store._slot_to_id(slot)
            tgt_id = store._slot_to_id(other_slot)
            if src_id and tgt_id:
                edge_key = (slot, other_slot, "similar_to")
                if edge_key not in store._edge_keys:
                    try:
                        store.put_edge(src_id, tgt_id, "similar_to", {"similarity": round(similarity, 4)})
                        edges_created += 1
                    except Exception as e:
                        logger.debug("similar_to edge creation skipped: %s", e, exc_info=True)

    return Result(kind="ok", data={"edges_created": edges_created}, count=edges_created)


def connect_node(store, vector_store, node_id, threshold=0.8):
    """Wire one node to its nearest similar neighbors."""
    if vector_store is None:
        return Result(kind="ok", data={"edges_created": 0}, count=0)

    str_id = store.string_table.intern(node_id) if node_id in store.string_table else None
    slot = store.id_to_slot.get(str_id) if str_id is not None else None
    if slot is None:
        from supergraph.core.errors import NodeNotFound
        raise NodeNotFound(node_id)

    if not vector_store.has_vector(slot):
        from supergraph.core.errors import VectorError
        raise VectorError(f"Node '{node_id}' has no vector")

    vec = vector_store.get_vector(slot)
    n = store._next_slot
    live = store.compute_live_mask(n)

    results_slots, dists = vector_store.search(vec, k=10, mask=live)

    edges_created = 0
    for other_slot, dist in zip(results_slots, dists):
        other_slot = int(other_slot)
        if other_slot == slot:
            continue
        similarity = 1.0 - float(dist)
        if similarity < threshold:
            continue
        tgt_id = store._slot_to_id(other_slot)
        if tgt_id:
            edge_key = (slot, other_slot, "similar_to")
            if edge_key not in store._edge_keys:
                try:
                    store.put_edge(node_id, tgt_id, "similar_to", {"similarity": round(similarity, 4)})
                    edges_created += 1
                except Exception as e:
                    logger.debug("similar_to edge creation skipped: %s", e, exc_info=True)

    return Result(kind="ok", data={"edges_created": edges_created}, count=edges_created)
