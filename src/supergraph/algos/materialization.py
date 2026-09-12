
import numpy as np

__all__ = ["materialize_bulk"]


def _is_reserved(field: str) -> bool:
    return len(field) >= 2 and field[0] == "_" and field[-1] == "_"


def materialize_bulk(
    slots: np.ndarray,
    node_ids: np.ndarray,
    node_kinds: np.ndarray,
    id_to_str: list,
    columns: dict,
    presence: dict,
    dtypes: dict,
) -> list[dict]:
    if len(slots) == 0:
        return []

    str_id_list = node_ids[slots].tolist()
    kind_id_list = node_kinds[slots].tolist()
    n = len(slots)

    result: list[dict] = [
        {"id": id_to_str[str_id_list[i]], "kind": id_to_str[kind_id_list[i]]}
        for i in range(n)
    ]

    for field, col in columns.items():
        if _is_reserved(field):
            continue
        dtype = dtypes[field]
        field_pres = presence[field][slots]
        field_vals = col[slots]
        vals_list = field_vals.tolist()
        if field_pres.all():
            if dtype == "int32_interned":
                for i, raw in enumerate(vals_list):
                    result[i][field] = id_to_str[raw]
            else:
                for i, raw in enumerate(vals_list):
                    result[i][field] = raw
            continue
        present_idx = np.where(field_pres)[0]
        if len(present_idx) == 0:
            continue
        if dtype == "int32_interned":
            for i in present_idx:
                result[i][field] = id_to_str[vals_list[i]]
        else:
            for i in present_idx:
                result[i][field] = vals_list[i]

    return result
