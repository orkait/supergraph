from pathlib import Path
from collections import OrderedDict
from lark import Lark
from supergraph.dsl.transformer import DSLTransformer
from supergraph.core.errors import QueryError

_grammar_path = Path(__file__).parent / "grammar.lark"
_grammar = _grammar_path.read_text()
_parser = Lark(_grammar, parser="lalr", start="start")

_TIME_KEYWORDS = {"NOW()", "TODAY", "YESTERDAY"}


def _normalize_cache_key(query: str) -> str:
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
                if not prev_ws and out:
                    out.append(" ")
                    prev_ws = True
            else:
                out.append(ch)
                prev_ws = False
        else:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(query[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
        i += 1
    result = "".join(out)
    if result.endswith(" "):
        result = result[:-1]
    return result


class PlanCache:
    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get_or_parse(self, query: str):
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


_plan_cache = PlanCache()


def set_cache_size(maxsize: int):
    _plan_cache._maxsize = maxsize
    _plan_cache.clear()


def parse(query: str):
    return _plan_cache.get_or_parse(query)


def parse_uncached(query: str):
    return _parse_internal(query)


def clear_cache():
    _plan_cache.clear()


from supergraph.core import plan_cache as _core_plan_cache
_core_plan_cache.register(lambda: len(_plan_cache), clear_cache)
