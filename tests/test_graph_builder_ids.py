"""Regression tests: graph edge ids must never collide. _generate_id
previously hashed str(time.time()), which returns the same value for
calls inside one clock tick, so add_edge silently overwrote siblings."""
from analysis.graph_builder import DynamicEdge, GraphBuilder, KnowledgeGraph


def _bare_builder() -> GraphBuilder:
    # __new__ skips __init__ on purpose: _generate_id is stateless and
    # __init__ would construct a live LLM client this unit never calls
    return GraphBuilder.__new__(GraphBuilder)


def test_generate_ids_unique_in_tight_loop():
    b = _bare_builder()
    ids = [b._generate_id("edge") for _ in range(1000)]
    assert len(set(ids)) == 1000, (
        f"id collision: {len(set(ids))} distinct of 1000")


def test_distinct_edges_all_survive_apply_scale():
    b = _bare_builder()
    kg = KnowledgeGraph(name="g", focus="f")
    for i in range(120):
        edge = DynamicEdge(
            id=b._generate_id("edge"),
            type="calls",
            source_id=f"fn_{i}",
            target_id=f"fn_{i + 1}",
        )
        assert kg.add_edge(edge) is True
    assert len(kg.edges) == 120, (
        f"{len(kg.edges)} of 120 distinct-triple edges survived")
