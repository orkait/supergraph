
from __future__ import annotations

from typing import Any

import numpy as np

from supergraph.core.strings import StringTable
from supergraph.algos.column_ops import (
    INT64_SENTINEL,
    STR_SENTINEL,
    NUM_OPS as _NUM_OPS,
    eval_mask,
    eval_mask_in,
)


class ColumnStore:

    INT64_SENTINEL = INT64_SENTINEL
    STR_SENTINEL = STR_SENTINEL
    _NUM_OPS = _NUM_OPS

    def __init__(self, string_table: StringTable, capacity: int = 1024):
        self._columns: dict[str, np.ndarray] = {}
        self._presence: dict[str, np.ndarray] = {}
        self._dtypes: dict[str, str] = {}
        self._string_table = string_table
        self._capacity = capacity
        self.dirty: bool = False

    def set(self, slot: int, data: dict) -> None:
        for field, value in data.items():
            if field not in self._dtypes:
                dtype_str = self._infer_dtype(value)
                if dtype_str is None:
                    continue
                self._create_column(field, dtype_str)

            dtype_str = self._dtypes[field]
            if dtype_str == "int64":
                if isinstance(value, int):
                    self._columns[field][slot] = int(value)
                    self._presence[field][slot] = True
                else:
                    self._columns[field][slot] = self.INT64_SENTINEL
                    self._presence[field][slot] = False
            elif dtype_str == "float64":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._columns[field][slot] = float(value)
                    self._presence[field][slot] = True
                else:
                    self._columns[field][slot] = np.nan
                    self._presence[field][slot] = False
            elif dtype_str == "int32_interned":
                if isinstance(value, str):
                    self._columns[field][slot] = self._string_table.intern(value)
                    self._presence[field][slot] = True
                else:
                    self._columns[field][slot] = self.STR_SENTINEL
                    self._presence[field][slot] = False
        self.dirty = True

    def clear(self, slot: int) -> None:
        self.clear_slots((slot,))

    def clear_slots(self, slots) -> None:
        if not len(slots) if hasattr(slots, "__len__") else slots is None:
            return
        idx = np.asarray(list(slots), dtype=np.int64) if not isinstance(slots, np.ndarray) else slots
        if idx.size == 0:
            return
        for field in self._columns:
            dtype_str = self._dtypes[field]
            if dtype_str == "int64":
                self._columns[field][idx] = self.INT64_SENTINEL
            elif dtype_str == "float64":
                self._columns[field][idx] = np.nan
            elif dtype_str == "int32_interned":
                self._columns[field][idx] = self.STR_SENTINEL
            self._presence[field][idx] = False
        self.dirty = True

    def grow(self, new_capacity: int) -> None:
        for field in list(self._columns):
            old_col = self._columns[field]
            new_col = self._make_sentinel_array(self._dtypes[field], new_capacity)
            new_col[: len(old_col)] = old_col
            self._columns[field] = new_col

            old_pres = self._presence[field]
            new_pres = np.zeros(new_capacity, dtype=bool)
            new_pres[: len(old_pres)] = old_pres
            self._presence[field] = new_pres
        self._capacity = new_capacity
        self.dirty = True

    def get_mask(self, field: str, op: str, value: Any, n: int) -> np.ndarray | None:
        if field not in self._columns:
            return None
        return eval_mask(
            col=self._columns[field],
            presence=self._presence[field],
            dtype_str=self._dtypes[field],
            op=op,
            value=value,
            n=n,
            intern_lookup=self._string_table.intern,
            has_string=lambda v: v in self._string_table,
        )

    def get_mask_in(self, field: str, values: list, n: int) -> np.ndarray | None:
        if field not in self._columns:
            return None
        return eval_mask_in(
            col=self._columns[field],
            presence=self._presence[field],
            dtype_str=self._dtypes[field],
            values=values,
            n=n,
            intern_lookup=self._string_table.intern,
            has_string=lambda v: v in self._string_table,
        )

    def get_presence(self, field: str, n: int) -> np.ndarray | None:
        if field not in self._presence:
            return None
        return self._presence[field][:n]

    def has_column(self, field: str) -> bool:
        return field in self._columns

    def get_column(self, field: str, n: int) -> tuple[np.ndarray, np.ndarray, str] | None:
        if field not in self._columns:
            return None
        return self._columns[field][:n], self._presence[field][:n], self._dtypes[field]

    def declare_column(self, field: str, dtype_str: str) -> None:
        if field not in self._dtypes:
            self._create_column(field, dtype_str)
            self.dirty = True

    def _ensure_column(self, field: str, dtype_str: str) -> None:
        self.declare_column(field, dtype_str)

    def set_reserved(self, slot: int, field: str, value) -> None:
        if isinstance(value, str):
            self._ensure_column(field, "int32_interned")
            self._columns[field][slot] = self._string_table.intern(value)
        elif isinstance(value, float):
            self._ensure_column(field, "float64")
            self._columns[field][slot] = value
        else:
            self._ensure_column(field, "int64")
            self._columns[field][slot] = int(value)
        self._presence[field][slot] = True
        self.dirty = True

    def set_field(self, slot: int, field: str, value) -> None:
        self.set(slot, {field: value})

    def snapshot_arrays(self) -> dict[str, tuple]:
        snap: dict[str, tuple] = {}
        for field in self._columns:
            snap[field] = (
                self._columns[field].copy(),
                self._presence[field].copy(),
                self._dtypes[field],
            )
        return snap

    def restore_arrays(self, snap: dict[str, tuple]) -> None:
        self._columns.clear()
        self._presence.clear()
        self._dtypes.clear()
        for field, (col, pres, dtype_str) in snap.items():
            self._columns[field] = col
            self._presence[field] = pres
            self._dtypes[field] = dtype_str
        self.dirty = True

    @property
    def memory_bytes(self) -> int:
        total = 0
        for field in self._columns:
            total += self._columns[field].nbytes
            total += self._presence[field].nbytes
        return total


    def _infer_dtype(self, value) -> str | None:
        if isinstance(value, bool):
            return "int64"
        if isinstance(value, int):
            return "int64"
        if isinstance(value, float):
            return "float64"
        if isinstance(value, str):
            return "int32_interned"
        return None

    def _create_column(self, field: str, dtype_str: str) -> None:
        self._columns[field] = self._make_sentinel_array(dtype_str, self._capacity)
        self._presence[field] = np.zeros(self._capacity, dtype=bool)
        self._dtypes[field] = dtype_str
        self.dirty = True

    def _make_sentinel_array(self, dtype_str: str, size: int) -> np.ndarray:
        if dtype_str == "int64":
            return np.full(size, self.INT64_SENTINEL, dtype=np.int64)
        elif dtype_str == "float64":
            return np.full(size, np.nan, dtype=np.float64)
        elif dtype_str == "int32_interned":
            return np.full(size, self.STR_SENTINEL, dtype=np.int32)
        raise ValueError(f"Unknown column dtype: {dtype_str}")
