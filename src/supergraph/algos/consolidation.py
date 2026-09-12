"""Memory consolidation: cluster episodic memories into observations.

TSM-inspired durative memory construction WITHOUT LLM:
    1. Group messages by entity (via graph edges)
    2. Cluster by cosine similarity within entity groups
    3. Pick most representative message per cluster as observation text
    4. Track evidence (source message count, recency)

This is the "sleep-time consolidation" step from the TSM paper,
implemented as a pure numpy/scipy operation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Observation:
    """A consolidated memory - one cluster of related episodic facts."""
    entity: str
    text: str
    evidence_slots: List[int]
    evidence_count: int
    centroid: Optional[np.ndarray] = None
    event_at_ms: Optional[int] = None
    confidence: float = 0.0


def cluster_by_entity(
    entity_to_slots: Dict[str, List[int]],
    vectors: Optional[np.ndarray],
    has_vector: Optional[np.ndarray],
    texts: Dict[int, str],
    event_times: Optional[Dict[int, int]] = None,
    similarity_threshold: float = 0.7,
    min_cluster_size: int = 1,
    cancel_event=None,
) -> List[Observation]:
    """Cluster messages by entity, then by semantic similarity.

    Args:
        entity_to_slots: entity_name -> list of message slot indices
        vectors: full vector array (slot-indexed), or None if no embedder
        has_vector: boolean mask of which slots have vectors
        texts: slot -> message text content
        event_times: slot -> __event_at__ epoch ms (optional)
        similarity_threshold: cosine sim threshold for same-cluster
        min_cluster_size: minimum messages to form an observation
        cancel_event: optional threading.Event; checked between entity
            groups so long consolidation runs can be aborted (bug #85).

    Returns list of Observation objects, one per cluster.
    """
    observations: list[Observation] = []

    for entity, slots in entity_to_slots.items():
        if not slots:
            continue

        # If no vectors, each message is its own observation
        if vectors is None or has_vector is None:
            for s in slots:
                text = texts.get(s, "")
                if not text:
                    continue
                evt = event_times.get(s) if event_times else None
                observations.append(Observation(
                    entity=entity, text=text,
                    evidence_slots=[s], evidence_count=1,
                    event_at_ms=evt,
                    confidence=1.0,
                ))
            continue

        # Get slots that have vectors. Vectorised against has_vector bitmap
        # instead of a Python comprehension per slot.
        slots_arr = np.asarray(slots, dtype=np.int64)
        in_range = slots_arr < len(has_vector)
        live = np.zeros_like(in_range)
        live[in_range] = has_vector[slots_arr[in_range]]
        vec_slots = slots_arr[live].tolist()
        non_vec_slots = slots_arr[~live].tolist()

        if not vec_slots:
            # No vectors - each message standalone
            for s in slots:
                text = texts.get(s, "")
                if text:
                    evt = event_times.get(s) if event_times else None
                    observations.append(Observation(
                        entity=entity, text=text,
                        evidence_slots=[s], evidence_count=1,
                        event_at_ms=evt, confidence=1.0,
                    ))
            continue

        # Cluster by cosine similarity (greedy single-linkage)
        slot_vecs = vectors[vec_slots]
        # Normalize for cosine
        norms = np.linalg.norm(slot_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        slot_vecs_norm = slot_vecs / norms

        # Optional cancellation check — cheap read, coarse granularity.
        # Checked per entity group so a 100K-entity consolidation can abort
        # in bounded time (bug #85).
        if cancel_event is not None and cancel_event.is_set():
            break

        assigned = np.zeros(len(vec_slots), dtype=bool)
        clusters: list[list[int]] = []

        # Vectorized single-linkage-ish clustering. For each unassigned
        # seed, compute similarity to every other unassigned vector in
        # one matrix-vector multiply, then greedily merge above-threshold
        # matches. Pre-fix this loop was an O(N²) Python-level double
        # for-loop with per-pair np.dot calls (bug #85); on a 10K-entity
        # group that cost ~100M Python function calls.
        for i in range(len(vec_slots)):
            if assigned[i]:
                continue
            # Running SUM of cluster vectors (not mean). We keep the sum
            # until the cluster is closed, then normalize once. Pre-fix,
            # the code tried to maintain a running normalized mean by
            # renormalizing after every add, which broke the invariant
            # that ``centroid == mean(members)`` for clusters >2 (bug
            # #86). Any subsequent similarity test used a distorted
            # centroid and mis-assigned downstream members.
            cluster_sum = slot_vecs_norm[i].copy()
            cluster = [i]
            assigned[i] = True

            while True:
                # Similarity of normalized cluster mean to every remaining
                # unassigned candidate. ``cluster_sum / len(cluster)`` is
                # the arithmetic mean; we skip an explicit mean computation
                # since cosine similarity is scale-invariant — dot with
                # the sum and divide by its norm at the end.
                centroid = cluster_sum / np.linalg.norm(cluster_sum)
                candidates = np.where(~assigned)[0]
                if len(candidates) == 0:
                    break
                sims = slot_vecs_norm[candidates] @ centroid
                # Find the highest-similarity above-threshold candidate.
                above = np.where(sims >= similarity_threshold)[0]
                if len(above) == 0:
                    break
                # Add all above-threshold candidates in this round so the
                # loop converges in O(log N) rounds rather than N.
                # Batched update: one mask assign + one vector sum instead
                # of a Python loop over new_indices.
                new_indices = candidates[above]
                assigned[new_indices] = True
                cluster.extend(int(i) for i in new_indices)
                cluster_sum = cluster_sum + slot_vecs_norm[new_indices].sum(axis=0)

            clusters.append(cluster)

        # Build observations from clusters
        for cluster_indices in clusters:
            if len(cluster_indices) < min_cluster_size:
                continue

            cluster_slots = [vec_slots[i] for i in cluster_indices]
            cluster_texts = [(s, texts.get(s, "")) for s in cluster_slots]
            cluster_texts = [(s, t) for s, t in cluster_texts if t]

            if not cluster_texts:
                continue

            # Pick most representative: longest text (most informative)
            best_slot, best_text = max(cluster_texts, key=lambda x: len(x[1]))

            # Compute centroid
            centroid = slot_vecs_norm[cluster_indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            # Earliest event time
            evt = None
            if event_times:
                evts = [event_times[s] for s in cluster_slots if s in event_times]
                if evts:
                    evt = min(evts)

            # Confidence based on evidence count
            confidence = min(1.0, len(cluster_slots) / 5.0)

            observations.append(Observation(
                entity=entity,
                text=best_text,
                evidence_slots=cluster_slots,
                evidence_count=len(cluster_slots),
                centroid=centroid.astype(np.float32),
                event_at_ms=evt,
                confidence=confidence,
            ))

        # Non-vectorized slots as standalone observations
        for s in non_vec_slots:
            text = texts.get(s, "")
            if text:
                evt = event_times.get(s) if event_times else None
                observations.append(Observation(
                    entity=entity, text=text,
                    evidence_slots=[s], evidence_count=1,
                    event_at_ms=evt, confidence=0.5,
                ))

    return observations
