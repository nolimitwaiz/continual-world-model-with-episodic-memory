from types import SimpleNamespace

import pytest

from cmwm.evaluation.conformance import (
    EndpointRow,
    effective_query_lag,
    summarize_endpoint_rows,
)
from cmwm.schemas import QueryExample, QueryKind, StructuredQuery


def _example(kind: QueryKind, *, ask: int, reference: int, answer: object) -> QueryExample:
    return QueryExample(
        StructuredQuery(
            query_id=f"query-{kind.value}",
            kind=kind,
            ask_step=ask,
            reference_step=reference,
        ),
        answer,
    )


def test_effective_query_lag_uses_reference_for_noncausal_query() -> None:
    episode = SimpleNamespace(steps=())
    example = _example(QueryKind.DELAYED_STATE, ask=160, reference=32, answer=None)
    assert effective_query_lag(episode, example) == 128


def test_effective_query_lag_recovers_hidden_causal_source_step() -> None:
    source_edge = SimpleNamespace(source="action:32")
    episode = SimpleNamespace(
        steps=(SimpleNamespace(step_index=32, causal_edges=(source_edge,)),)
    )
    example = _example(
        QueryKind.CAUSAL_PREDECESSOR,
        ask=160,
        reference=160,
        answer="action:32",
    )
    assert effective_query_lag(episode, example) == 128


def test_effective_query_lag_rejects_ambiguous_causal_source() -> None:
    source_edge = SimpleNamespace(source="action:x")
    episode = SimpleNamespace(
        steps=(
            SimpleNamespace(step_index=10, causal_edges=(source_edge,)),
            SimpleNamespace(step_index=20, causal_edges=(source_edge,)),
        )
    )
    example = _example(
        QueryKind.CAUSAL_PREDECESSOR,
        ask=160,
        reference=160,
        answer="action:x",
    )
    with pytest.raises(ValueError, match="exactly one"):
        effective_query_lag(episode, example)


def test_endpoint_coverage_fails_closed_on_a_missing_cell() -> None:
    rows = []
    index = 0
    for task in (kind.value for kind in QueryKind):
        for lag in (64, 128):
            if (task, lag) == (QueryKind.CAUSAL_PREDECESSOR.value, 128):
                continue
            rows.append(EndpointRow(str(index), f"e{index}", f"q{index}", task, lag))
            index += 1
    summary = summarize_endpoint_rows(rows)
    assert not summary["endpoint_computable"]
    assert summary["missing_required_cells"] == ["causal_predecessor@128"]


def test_endpoint_coverage_tracks_repeated_queries_without_losing_occurrences() -> None:
    rows = []
    index = 0
    for task in (kind.value for kind in QueryKind):
        for lag in (64, 128):
            query_id = "repeated" if index < 2 else f"q{index}"
            rows.append(EndpointRow(str(index), f"e{index}", query_id, task, lag))
            index += 1
    summary = summarize_endpoint_rows(rows)
    assert summary["endpoint_computable"]
    assert summary["row_count"] == 8
    assert summary["unique_query_count"] == 7
    assert summary["duplicate_query_occurrences"] == 1
    assert summary["repeated_queries"] == {"repeated": 2}
