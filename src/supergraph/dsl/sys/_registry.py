
SYS_DISPATCH: dict[type, callable] = {}


def handles_sys(*ast_types):
    def decorator(fn):
        for t in ast_types:
            SYS_DISPATCH[t] = fn
        return fn
    return decorator
