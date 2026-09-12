
DISPATCH: dict[type, callable] = {}
WRITE_OPS: set[type] = set()


def handles(*ast_types, write=False):
    def decorator(fn):
        for t in ast_types:
            DISPATCH[t] = fn
            if write:
                WRITE_OPS.add(t)
        return fn
    return decorator


def is_write_op(ast) -> bool:
    return type(ast) in WRITE_OPS
