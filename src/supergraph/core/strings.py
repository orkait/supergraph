"""Bidirectional mapping between strings and int32 IDs.

Every string that enters the store (node IDs, field names, field values,
edge types, file paths) is interned once and referred to by a compact
integer thereafter.
"""

from __future__ import annotations


class StringTable:
    """Intern table: str <-> int, with sequential IDs starting at 0."""

    __slots__ = ("_id_to_str", "_str_to_id")

    def __init__(self) -> None:
        self._id_to_str: list[str] = []
        self._str_to_id: dict[str, int] = {}

    # -- public API ----------------------------------------------------------

    def intern(self, s: str) -> int:
        """Return the integer ID for *s*, assigning the next sequential ID
        if *s* has not been seen before."""
        try:
            return self._str_to_id[s]
        except KeyError:
            idx = len(self._id_to_str)
            self._id_to_str.append(s)
            self._str_to_id[s] = idx
            return idx

    def lookup(self, i: int) -> str:
        """Return the string for integer ID *i*.

        Raises ``KeyError`` if *i* is not a valid ID.
        """
        try:
            return self._id_to_str[i]
        except IndexError:
            raise KeyError(i) from None

    def __len__(self) -> int:
        return len(self._id_to_str)

    def __contains__(self, s: object) -> bool:
        return s in self._str_to_id

    # -- serialization helpers -----------------------------------------------

    def to_list(self) -> list[str]:
        """Export the table as a plain list (ordered by ID) for serialization."""
        return list(self._id_to_str)

    @classmethod
    def from_list(cls, strings: list[str]) -> StringTable:
        """Rebuild a ``StringTable`` from a previously exported list."""
        table = cls()
        table._id_to_str = list(strings)
        table._str_to_id = {s: i for i, s in enumerate(strings)}
        return table
