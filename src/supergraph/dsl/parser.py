from pathlib import Path
from collections import OrderedDict
from lark import Lark
from supergraph.dsl.transformer import DSLTransformer
from supergraph.core.errors import QueryError

_grammar_path = Path(__file__).parent / "grammar.lark"
_grammar = _grammar_path.read_text()
_parser = Lark(_grammar, parser="lalr", start="start")

# Keywords that resolve to the current time - must bypass LRU cache
_TIME_KEYWORDS = {"NOW()", "TODAY", "YESTERDAY"}


def _normalize_cache_key(query: str) -> str:
    """Build a cache key that normalizes whitespace OUTSIDE quoted strings.

    The naive ``" ".join(query.split())`` collapses runs of whitespace
    globally, which corrupts multi-space payloads embedded in string
    literals: ``msg="a  b"`` and ``msg="a b"`` would hash to the same key
    and the first-cached AST wins regardless of the actual stored value
    (bug #20).

    This implementation walks the query tracking quote state and only
    collapses whitespace outside a quoted region. It handles backslash
    escaping inside quotes so ``"he said \"hi\""`` stays quoted. Both
    ``"`` and ``'`` are recognized as quote characters.
    """
    out: list[str] = []
    in_quote: str | None = None
    prev_ws = False
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if in_quote is None:
            if ch in ('"', "'"):
                in_quote = ch
                out.append(ch)
                prev_ws = False
            elif ch.isspace():
                # Collapse runs of whitespace into a single space, and skip
                # leading whitespace so different indentation hashes identically.
                if not prev_ws and out:
                    out.append(" ")
                    prev_ws = True
            else:
                out.append(ch)
                prev_ws = False
        else:
            # Inside a quoted literal: preserve bytes verbatim including
            # whitespace. Respect ``\"`` / ``\'`` escapes so the closing
            # quote check does not fire on them.
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(query[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
        i += 1
    # Trailing whitespace stripped by not appending it after the last
    # non-space char; leading already skipped.
    result = "".join(out)
    if result.endswith(" "):
        result = result[:-1]
    return result


class PlanCache:
    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get_or_parse(self, query: str):
        # Bypass cache for queries with time expressions so timestamps are fresh
        if any(kw in query for kw in _TIME_KEYWORDS):
            return _parse_internal(query)
        key = _normalize_cache_key(query)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ast = _parse_internal(query)
        self._cache[key] = ast
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return ast

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


def _parse_internal(query: str):
    try:
        tree = _parser.parse(query)
        return DSLTransformer().transform(tree)
    except Exception as e:
        raise QueryError(message=str(e), query=query)


# Module-level cache
_plan_cache = PlanCache()


def set_cache_size(maxsize: int):
    """Resize the plan cache (clears existing entries)."""
    _plan_cache._maxsize = maxsize
    _plan_cache.clear()


def parse(query: str):
    """Parse a DSL query string into an AST node. Uses plan cache."""
    return _plan_cache.get_or_parse(query)


def parse_uncached(query: str):
    """Parse without caching."""
    return _parse_internal(query)


def clear_cache():
    """Clear the plan cache."""
    _plan_cache.clear()


# Register plan-cache accessors with core so optimizer can read/clear this
# cache without importing dsl - enforces top-down layering.
from supergraph.core import plan_cache as _core_plan_cache
_core_plan_cache.register(lambda: len(_plan_cache), clear_cache)
