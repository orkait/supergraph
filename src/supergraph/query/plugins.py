from __future__ import annotations

from typing import Callable

from supergraph.query.runtime import Query


_REGISTRY: dict[str, Callable[..., Query]] = {}


def register_verb(name: str) -> Callable[[Callable[..., Query]], Callable[..., Query]]:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"verb name must be a valid Python identifier, got {name!r}")

    def _decorator(fn: Callable[..., Query]) -> Callable[..., Query]:
        _REGISTRY[name] = fn
        return fn

    return _decorator


def _lookup(name: str) -> Callable[..., Query] | None:
    return _REGISTRY.get(name)


def _registered_names() -> list[str]:
    return sorted(_REGISTRY)
