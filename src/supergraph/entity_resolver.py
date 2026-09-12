from __future__ import annotations

import logging
import re
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

_RESOLVER_LOCK = threading.Lock()

_NAME_TO_ENTITY_CACHE: dict[str, str] = {}


def reset_resolver_cache_for_tests() -> None:
    _NAME_TO_ENTITY_CACHE.clear()

DEFAULT_HIGH_THRESHOLD = 0.85

KIND_MENTION = "mention"
KIND_ENTITY = "entity"
EDGE_REFERS_TO = "refers_to"


@dataclass(frozen=True)
class ResolvedMention:

    entity_id: str
    confidence: float
    is_new_entity: bool
    canonical_name: str
    candidates_seen: int
    notes: list[str]


_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name).lower()
    return _NAME_NORMALIZE_RE.sub("", decomposed)


def make_entity_id(prefix: str = "entity") -> str:
    return f"{prefix}:{uuid.uuid4().hex[:12]}"


_CANONICAL_ENTITY_PREFIX = "entity:"


def _candidates_by_name(gs: Any, surface_name: str) -> list[dict]:
    target = normalize_name(surface_name)
    if not target:
        return []
    try:
        result = gs.execute(f'NODES WHERE kind = "{KIND_ENTITY}" LIMIT 5000')
    except Exception as e:
        _log.warning("entity_resolver: NODES query failed (%s)", e)
        return []
    nodes = result.data if hasattr(result, "data") else []
    if not isinstance(nodes, list):
        return []
    out: list[dict] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_id = n.get("id", "")
        if not isinstance(node_id, str) or not node_id.startswith(_CANONICAL_ENTITY_PREFIX):
            continue
        cn = n.get("canonical_name") or n.get("name") or ""
        if normalize_name(cn) == target:
            out.append(n)
    return out


def _embed_text(gs: Any, text: str) -> list[float] | None:
    embedder = getattr(gs, "_embedder", None)
    if embedder is None:
        return None
    try:
        vecs = embedder.encode_documents([text])
        if vecs is None or len(vecs) == 0:
            return None
        return list(vecs[0])
    except Exception as e:
        _log.warning("entity_resolver: embed failed (%s)", e)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def resolve_mention(
    gs: Any,
    surface_name: str,
    context: str,
    threshold_high: float = DEFAULT_HIGH_THRESHOLD,
) -> ResolvedMention:
    notes: list[str] = []

    candidates = _candidates_by_name(gs, surface_name)
    notes.append(f"name match candidates: {len(candidates)}")

    if not candidates:
        return ResolvedMention(
            entity_id=make_entity_id(),
            confidence=1.0,
            is_new_entity=True,
            canonical_name=surface_name,
            candidates_seen=0,
            notes=notes + ["no existing entity with this name; minting new"],
        )

    if len(candidates) == 1:
        return ResolvedMention(
            entity_id=candidates[0]["id"],
            confidence=1.0,
            is_new_entity=False,
            canonical_name=surface_name,
            candidates_seen=1,
            notes=notes + ["single name match; linking with confidence=1.0"],
        )

    new_vec = _embed_text(gs, f"{surface_name}. {context}")
    if new_vec is None:
        notes.append("no embedder; falling back to most-mentioned entity")
        best = max(candidates,
                   key=lambda n: int(n.get("mention_count", 0)))
        return ResolvedMention(
            entity_id=best["id"],
            confidence=0.5,
            is_new_entity=False,
            canonical_name=surface_name,
            candidates_seen=len(candidates),
            notes=notes,
        )

    best_id: str | None = None
    best_score = -1.0
    for cand in candidates:
        cand_id = cand.get("id", "")
        cand_text = " ".join([
            str(cand.get("canonical_name") or cand.get("name") or ""),
            str(cand.get("context", "")),
        ]).strip()
        cand_vec = _embed_text(gs, cand_text)
        if cand_vec is None:
            continue
        score = _cosine(new_vec, cand_vec)
        if score > best_score:
            best_score = score
            best_id = cand_id

    if best_id is not None and best_score >= threshold_high:
        return ResolvedMention(
            entity_id=best_id,
            confidence=float(best_score),
            is_new_entity=False,
            canonical_name=surface_name,
            candidates_seen=len(candidates),
            notes=notes + [
                f"best candidate {best_id} cosine={best_score:.3f} "
                f">= threshold {threshold_high:.2f}; linking"
            ],
        )

    return ResolvedMention(
        entity_id=make_entity_id(),
        confidence=1.0,
        is_new_entity=True,
        canonical_name=surface_name,
        candidates_seen=len(candidates),
        notes=notes + [
            f"best candidate cosine={best_score:.3f} < threshold "
            f"{threshold_high:.2f}; minting new entity"
        ],
    )


def make_mention_id(msg_id: str, slug: str, occurrence: int = 0) -> str:
    return f"mention:{msg_id}:{slug}:{occurrence}"


def resolve_and_create_entity(
    gs: Any,
    surface_name: str,
    context: str,
    threshold_high: float = DEFAULT_HIGH_THRESHOLD,
) -> tuple[str, float, bool]:
    with _RESOLVER_LOCK:
        nkey = normalize_name(surface_name)
        cached = _NAME_TO_ENTITY_CACHE.get(nkey) if nkey else None
        if cached is not None:
            return cached, 1.0, False

        result = resolve_mention(gs, surface_name, context, threshold_high)
        if result.is_new_entity:
            esc_id = result.entity_id.replace("\\", "\\\\").replace('"', '\\"')
            esc_name = surface_name.replace("\\", "\\\\").replace('"', '\\"')
            esc_ctx = context.replace("\\", "\\\\").replace('"', '\\"')
            try:
                gs.execute(
                    f'CREATE NODE "{esc_id}" kind = "{KIND_ENTITY}" '
                    f'canonical_name = "{esc_name}" mention_count = 0 '
                    f'context = "{esc_ctx}" '
                    f'DOCUMENT "{esc_name} | {esc_ctx}"'
                )
            except Exception as e:
                _log.warning(
                    "resolve_and_create_entity: CREATE NODE %s failed (%s); "
                    "caller may receive duplicate-key error",
                    result.entity_id, e,
                )
        if nkey:
            _NAME_TO_ENTITY_CACHE[nkey] = result.entity_id
        return result.entity_id, result.confidence, result.is_new_entity
