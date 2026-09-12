
from __future__ import annotations

from dataclasses import dataclass, field

from supergraph.core.errors import SchemaError


@dataclass
class NodeKindDef:
    required: set[str]
    optional: set[str]
    field_types: dict[str, str] = field(default_factory=dict)
    embed_field: str | None = None


@dataclass
class EdgeKindDef:
    from_kinds: set[str]
    to_kinds: set[str]


class SchemaRegistry:
    def __init__(self):
        self._node_kinds: dict[str, NodeKindDef] = {}
        self._edge_kinds: dict[str, EdgeKindDef] = {}

    @property
    def has_node_kinds(self) -> bool:
        return bool(self._node_kinds)

    @property
    def has_edge_kinds(self) -> bool:
        return bool(self._edge_kinds)

    def register_node_kind(self, kind: str, required: list, optional: list | None = None,
                           embed_field: str | None = None):
        req_fields = set()
        field_types = {}
        for item in required:
            if isinstance(item, tuple):
                name, type_name = item
                req_fields.add(name)
                if type_name:
                    field_types[name] = type_name
            else:
                req_fields.add(item)

        opt_fields = set()
        for item in (optional or []):
            if isinstance(item, tuple):
                name, type_name = item
                opt_fields.add(name)
                if type_name:
                    field_types[name] = type_name
            else:
                opt_fields.add(item)

        self._node_kinds[kind] = NodeKindDef(
            required=req_fields, optional=opt_fields, field_types=field_types,
            embed_field=embed_field,
        )

    def register_edge_kind(self, kind: str, from_kinds: list[str], to_kinds: list[str]):
        self._edge_kinds[kind] = EdgeKindDef(
            from_kinds=set(from_kinds),
            to_kinds=set(to_kinds),
        )

    def unregister_node_kind(self, kind: str):
        if kind in self._node_kinds:
            del self._node_kinds[kind]

    def unregister_edge_kind(self, kind: str):
        if kind in self._edge_kinds:
            del self._edge_kinds[kind]

    def get_node_kind(self, kind: str) -> NodeKindDef | None:
        return self._node_kinds.get(kind)

    def get_edge_kind(self, kind: str) -> EdgeKindDef | None:
        return self._edge_kinds.get(kind)

    def list_node_kinds(self) -> list[str]:
        return list(self._node_kinds.keys())

    def list_edge_kinds(self) -> list[str]:
        return list(self._edge_kinds.keys())

    def describe_node_kind(self, kind: str) -> dict | None:
        defn = self._node_kinds.get(kind)
        if defn is None:
            return None
        result = {
            "kind": kind,
            "required": sorted(defn.required),
            "optional": sorted(defn.optional),
        }
        if defn.field_types:
            result["field_types"] = defn.field_types
        if defn.embed_field:
            result["embed_field"] = defn.embed_field
        return result

    def describe_edge_kind(self, kind: str) -> dict | None:
        defn = self._edge_kinds.get(kind)
        if defn is None:
            return None
        return {
            "kind": kind,
            "from_kinds": sorted(defn.from_kinds),
            "to_kinds": sorted(defn.to_kinds),
        }

    def validate_node(self, kind: str, data: dict):
        if kind not in self._node_kinds:
            return
        defn = self._node_kinds[kind]
        missing = defn.required - set(data.keys())
        if missing:
            raise SchemaError(f"kind '{kind}' requires fields: {sorted(missing)}")
        for field_name, value in data.items():
            if field_name in defn.field_types:
                expected = defn.field_types[field_name]
                if not self._type_matches(value, expected):
                    raise SchemaError(
                        f"Field '{field_name}' expects type '{expected}', "
                        f"got {type(value).__name__}"
                    )

    @staticmethod
    def _type_matches(value, type_name: str) -> bool:
        if type_name == "string":
            return isinstance(value, str)
        if type_name == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return True

    def validate_edge(self, kind: str, source_kind: str, target_kind: str):
        if kind not in self._edge_kinds:
            return
        defn = self._edge_kinds[kind]
        if source_kind not in defn.from_kinds:
            raise SchemaError(
                f"edge '{kind}' cannot originate from kind '{source_kind}', "
                f"allowed: {sorted(defn.from_kinds)}"
            )
        if target_kind not in defn.to_kinds:
            raise SchemaError(
                f"edge '{kind}' cannot target kind '{target_kind}', "
                f"allowed: {sorted(defn.to_kinds)}"
            )

    def to_dict(self) -> dict:
        node_kinds = {}
        for k, v in self._node_kinds.items():
            d = {
                "required": sorted(v.required),
                "optional": sorted(v.optional),
                "field_types": v.field_types,
            }
            if v.embed_field:
                d["embed_field"] = v.embed_field
            node_kinds[k] = d
        return {
            "node_kinds": node_kinds,
            "edge_kinds": {
                k: {"from_kinds": sorted(v.from_kinds), "to_kinds": sorted(v.to_kinds)}
                for k, v in self._edge_kinds.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> SchemaRegistry:
        registry = cls()
        for kind, defn in data.get("node_kinds", {}).items():
            field_types = defn.get("field_types", {})
            required = [(f, field_types.get(f)) for f in defn["required"]]
            optional = [(f, field_types.get(f)) for f in defn.get("optional", [])]
            embed_field = defn.get("embed_field")
            registry.register_node_kind(kind, required, optional, embed_field=embed_field)
        for kind, defn in data.get("edge_kinds", {}).items():
            registry.register_edge_kind(kind, defn["from_kinds"], defn["to_kinds"])
        return registry
