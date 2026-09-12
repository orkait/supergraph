"""Auto-dispatch registry for SYS command handlers.

Mirrors supergraph.dsl.handlers._registry. Mixin methods register
themselves via @handles_sys(AstType) at import time; SystemExecutor._dispatch
looks them up instead of maintaining a giant switch dict.
"""

SYS_DISPATCH: dict[type, callable] = {}


def handles_sys(*ast_types):
    def decorator(fn):
        for t in ast_types:
            SYS_DISPATCH[t] = fn
        return fn
    return decorator
