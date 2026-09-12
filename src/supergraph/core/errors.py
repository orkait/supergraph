

class SuperGraphError(Exception):
    pass


class QueryError(SuperGraphError):

    def __init__(self, message: str, position: int = -1, query: str = "") -> None:
        self.message = message
        self.position = position
        self.query = query
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.position >= 0:
            parts.append(f"at position {self.position}")
        if self.query:
            parts.append(f"in query: {self.query!r}")
        return " ".join(parts)


class NodeNotFound(SuperGraphError):

    def __init__(self, id: str) -> None:
        self.id = id
        super().__init__(f"Node not found: {id!r}")


class NodeExists(SuperGraphError):

    def __init__(self, id: str) -> None:
        self.id = id
        super().__init__(f"Node already exists: {id!r}")


class CeilingExceeded(SuperGraphError):

    def __init__(self, current_mb: int, ceiling_mb: int, operation: str) -> None:
        self.current_mb = current_mb
        self.ceiling_mb = ceiling_mb
        self.operation = operation
        super().__init__(
            f"Memory ceiling exceeded: {current_mb} MB / {ceiling_mb} MB "
            f"during {operation!r}"
        )


class VersionMismatch(SuperGraphError):

    def __init__(self, found: str | None, expected: int) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"Version mismatch: found {found!r}, expected {expected}"
        )


class SchemaError(SuperGraphError):

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AggregationError(SchemaError):
    pass


class CostThresholdExceeded(SuperGraphError):

    def __init__(self, estimated_frontier: float, threshold: float) -> None:
        self.estimated_frontier = estimated_frontier
        self.threshold = threshold
        super().__init__(
            f"Cost threshold exceeded: estimated frontier {estimated_frontier} "
            f"> threshold {threshold}"
        )


class BatchRollback(SuperGraphError):

    def __init__(self, failed_statement: str, error: str) -> None:
        self.failed_statement = failed_statement
        self.error = error
        super().__init__(
            f"Batch rollback on statement {failed_statement!r}: {error}"
        )


class VectorError(SuperGraphError):
    pass


class EmbedderRequired(VectorError):
    pass


class VectorNotFound(VectorError):
    pass


class OptimizationInProgress(SuperGraphError):

    def __init__(self) -> None:
        super().__init__("Self-balance in progress. Retry after optimization completes.")


class StoreInUse(SuperGraphError):

    def __init__(self, lock_path: str) -> None:
        super().__init__(
            f"Another process holds {lock_path!r}. "
            "SuperGraph is single-owner per path. "
            "Close the other instance or open a different path."
        )
        self.lock_path = lock_path
