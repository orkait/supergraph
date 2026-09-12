
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Observation:
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
    observations: list[Observation] = []

    for entity, slots in entity_to_slots.items():
        if not slots:
            continue

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

        slots_arr = np.asarray(slots, dtype=np.int64)
        in_range = slots_arr < len(has_vector)
        live = np.zeros_like(in_range)
        live[in_range] = has_vector[slots_arr[in_range]]
        vec_slots = slots_arr[live].tolist()
        non_vec_slots = slots_arr[~live].tolist()

        if not vec_slots:
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

        slot_vecs = vectors[vec_slots]
        norms = np.linalg.norm(slot_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        slot_vecs_norm = slot_vecs / norms

        if cancel_event is not None and cancel_event.is_set():
            break

        assigned = np.zeros(len(vec_slots), dtype=bool)
        clusters: list[list[int]] = []

        for i in range(len(vec_slots)):
            if assigned[i]:
                continue
            cluster_sum = slot_vecs_norm[i].copy()
            cluster = [i]
            assigned[i] = True

            while True:
                centroid = cluster_sum / np.linalg.norm(cluster_sum)
                candidates = np.where(~assigned)[0]
                if len(candidates) == 0:
                    break
                sims = slot_vecs_norm[candidates] @ centroid
                above = np.where(sims >= similarity_threshold)[0]
                if len(above) == 0:
                    break
                new_indices = candidates[above]
                assigned[new_indices] = True
                cluster.extend(int(i) for i in new_indices)
                cluster_sum = cluster_sum + slot_vecs_norm[new_indices].sum(axis=0)

            clusters.append(cluster)

        for cluster_indices in clusters:
            if len(cluster_indices) < min_cluster_size:
                continue

            cluster_slots = [vec_slots[i] for i in cluster_indices]
            cluster_texts = [(s, texts.get(s, "")) for s in cluster_slots]
            cluster_texts = [(s, t) for s, t in cluster_texts if t]

            if not cluster_texts:
                continue

            best_slot, best_text = max(cluster_texts, key=lambda x: len(x[1]))

            centroid = slot_vecs_norm[cluster_indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            evt = None
            if event_times:
                evts = [event_times[s] for s in cluster_slots if s in event_times]
                if evts:
                    evt = min(evts)

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
