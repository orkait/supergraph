"""Auto-dispatch registry for DSL handlers.

Eliminates manual dispatch dicts and is_write tuples.
Handlers self-register via the @handles decorator at import time.
"""

DISPATCH: dict[type, callable] = {}
WRITE_OPS: set[type] = set()


def handles(*ast_types, write=False):
    """Register a method as the handler for one or more AST node types."""
    def decorator(fn):
        for t in ast_types:
            DISPATCH[t] = fn
            if write:
                WRITE_OPS.add(t)
        return fn
    return decorator


def is_write_op(ast) -> bool:
    """Check if an AST node is a write operation."""
    return type(ast) in WRITE_OPS
