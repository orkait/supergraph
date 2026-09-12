
import numpy as np

from supergraph.dsl.handlers._registry import handles
from supergraph.dsl.ast_nodes import AggregateQuery
from supergraph.core.types import Result
from supergraph.core.errors import AggregationError


class AggregationHandlers:

    @handles(AggregateQuery)
    def _aggregate(self, query: AggregateQuery) -> Result:
        n = self.store._next_slot
        if n == 0:
            group_by = query.group_by or []
            select = query.select or []
            if group_by:
                return Result(kind="aggregate", data=[], count=0)
            else:
                row = {}
                for func in select:
                    if func.func == "COUNT" or func.func == "COUNT_DISTINCT":
                        row[func.label()] = 0
                    elif func.func == "AVG":
                        row[func.label()] = None
                    else:
                        row[func.label()] = 0
                return Result(kind="aggregate", data=[row], count=1)

        mask = self._compute_live_mask(n)

        if query.where:
            kind_filter = self._extract_kind_from_where(query.where)
            if kind_filter:
                kind_mask = self.store._live_mask(kind_filter)
                mask = mask & kind_mask
                remaining = self._strip_kind_from_expr(query.where.expr)
                if remaining is not None:
                    where_mask = self._try_column_filter(remaining, mask, n)
                    if where_mask is None:
                        raise AggregationError("AGGREGATE WHERE fields must be columnarized")
                    mask = where_mask
            else:
                where_mask = self._try_column_filter(query.where.expr, mask, n)
                if where_mask is None:
                    raise AggregationError("AGGREGATE WHERE fields must be columnarized")
                mask = where_mask

        group_by = query.group_by or []
        select = query.select or []
        for field in group_by:
            if not self.store.columns.has_column(field):
                raise AggregationError(f"GROUP BY field '{field}' is not columnarized")
        for func in select:
            if func.field and not self.store.columns.has_column(func.field):
                raise AggregationError(f"Aggregate field '{func.field}' is not columnarized")

        filtered_count = int(np.sum(mask))
        if filtered_count == 0:
            if group_by:
                return Result(kind="aggregate", data=[], count=0)
            else:
                row = {}
                for func in select:
                    if func.func == "COUNT" or func.func == "COUNT_DISTINCT":
                        row[func.label()] = 0
                    elif func.func == "AVG":
                        row[func.label()] = None
                    else:
                        row[func.label()] = 0
                return Result(kind="aggregate", data=[row], count=1)

        from supergraph.algos.aggregate import (
            group_assign_single,
            group_assign_multi,
            group_count,
            group_sum,
            group_avg,
            group_min,
            group_max,
            group_count_distinct,
        )

        if group_by:
            group_cols = [self.store.columns._columns[f][:n][mask] for f in group_by]
            if len(group_cols) == 1:
                unique_keys, inverse = group_assign_single(group_cols[0])
            else:
                unique_keys, inverse = group_assign_multi(group_cols)
        else:
            unique_keys = np.array([0])
            inverse = np.zeros(filtered_count, dtype=np.intp)

        num_groups = len(unique_keys)

        results = {}
        for func in select:
            label = func.label()
            if func.func == "COUNT":
                results[label] = group_count(inverse, num_groups)
            elif func.func == "COUNT_DISTINCT":
                col = self.store.columns._columns[func.field][:n][mask]
                results[label] = group_count_distinct(col, inverse, num_groups)
            elif func.func == "SUM":
                col = self.store.columns._columns[func.field][:n][mask]
                results[label] = group_sum(col, inverse, num_groups)
            elif func.func == "AVG":
                col = self.store.columns._columns[func.field][:n][mask]
                results[label] = group_avg(col, inverse, num_groups)
            elif func.func == "MIN":
                col = self.store.columns._columns[func.field][:n][mask]
                results[label] = group_min(col, inverse, num_groups)
            elif func.func == "MAX":
                col = self.store.columns._columns[func.field][:n][mask]
                results[label] = group_max(col, inverse, num_groups)

        group_dicts = []
        for i in range(num_groups):
            d = {}
            if group_by:
                for j, field in enumerate(group_by):
                    if len(group_by) == 1:
                        raw = unique_keys[i]
                    else:
                        raw = unique_keys[i][j]
                    dtype = self.store.columns._dtypes[field]
                    if dtype == "int32_interned":
                        d[field] = self.store.string_table.lookup(int(raw))
                    elif dtype == "float64":
                        d[field] = float(raw)
                    elif dtype == "int64":
                        d[field] = int(raw)
            for label, arr in results.items():
                val = arr[i]
                if np.isnan(val) or np.isinf(val):
                    d[label] = None if np.isnan(val) else float(val)
                else:
                    d[label] = int(val) if float(val) == int(float(val)) else float(val)
            group_dicts.append(d)

        if query.having:
            group_dicts = [d for d in group_dicts if self._eval_where(query.having, d)]

        if query.order_by:
            sort_key = query.order_by.label()
            group_dicts.sort(key=lambda d: d.get(sort_key, 0), reverse=query.order_desc)

        if query.limit:
            group_dicts = group_dicts[:query.limit.value]

        return Result(kind="aggregate", data=group_dicts, count=len(group_dicts))
