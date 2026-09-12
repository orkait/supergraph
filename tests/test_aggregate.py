import pytest
from supergraph import SuperGraph
from supergraph.core.errors import AggregationError


def make_graph():
    g = SuperGraph(ceiling_mb=256)
    g.execute('SYS REGISTER NODE KIND "memory" REQUIRED topic:string, importance:float, score:int')
    for i in range(100):
        topic = f"topic_{i % 5}"
        g.execute(f'CREATE NODE "m{i}" kind = "memory" topic = "{topic}" importance = {(i % 10) * 0.1} score = {i}')
    return g


class TestAggregateGroupBy:
    def test_count_by_topic(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT()')
        assert result.kind == "aggregate"
        assert result.count == 5
        for row in result.data:
            assert row["COUNT()"] == 20

    def test_avg_importance(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT AVG(importance)')
        assert result.count == 5
        for row in result.data:
            assert "AVG(importance)" in row
            assert isinstance(row["AVG(importance)"], (int, float))

    def test_sum_score(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT SUM(score)')
        assert result.count == 5
        total = sum(row["SUM(score)"] for row in result.data)
        assert total == sum(range(100))

    def test_min_max(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT MIN(score), MAX(score)')
        assert result.count == 5
        for row in result.data:
            assert "MIN(score)" in row
            assert "MAX(score)" in row
            assert row["MIN(score)"] <= row["MAX(score)"]

    def test_multiple_agg_funcs(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES WHERE kind = "memory" GROUP BY topic SELECT COUNT(), SUM(score), AVG(importance)')
        assert result.count == 5
        for row in result.data:
            assert "COUNT()" in row
            assert "SUM(score)" in row
            assert "AVG(importance)" in row


class TestAggregateGlobal:
    def test_no_group_by(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES WHERE kind = "memory" SELECT COUNT(), SUM(score)')
        assert result.count == 1
        assert result.data[0]["COUNT()"] == 100
        assert result.data[0]["SUM(score)"] == sum(range(100))

    def test_empty_result(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('SYS REGISTER NODE KIND "x" REQUIRED score:int')
        result = g.execute('AGGREGATE NODES WHERE kind = "x" SELECT COUNT()')
        assert result.count == 1
        assert result.data[0]["COUNT()"] == 0


class TestAggregateHaving:
    def test_having_filters_groups(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES GROUP BY topic SELECT COUNT() HAVING COUNT() > 10')
        assert all(row["COUNT()"] > 10 for row in result.data)

    def test_having_eliminates_all(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES GROUP BY topic SELECT COUNT() HAVING COUNT() > 100')
        assert result.count == 0


class TestAggregateOrderLimit:
    def test_order_by_desc(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES GROUP BY topic SELECT SUM(score) ORDER BY SUM(score) DESC')
        sums = [row["SUM(score)"] for row in result.data]
        assert sums == sorted(sums, reverse=True)

    def test_limit(self):
        g = make_graph()
        result = g.execute('AGGREGATE NODES GROUP BY topic SELECT COUNT() LIMIT 3')
        assert result.count == 3


class TestAggregateErrors:
    def test_non_columnarized_group_by_raises(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "n1" kind = "test" name = "x"')
        with pytest.raises(AggregationError):
            g.execute('AGGREGATE NODES GROUP BY nonexistent SELECT COUNT()')

    def test_non_columnarized_agg_field_raises(self):
        g = SuperGraph(ceiling_mb=256)
        g.execute('CREATE NODE "n1" kind = "test" name = "x"')
        with pytest.raises(AggregationError):
            g.execute('AGGREGATE NODES SELECT SUM(nonexistent)')
