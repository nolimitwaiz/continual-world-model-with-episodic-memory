"""Auditable task/lag metadata for the preregistered CMWM endpoint.

The training-time tensor stores ``ask_step - reference_step`` as query lag.  That
is correct for state and occlusion queries, but a causal query intentionally uses
the ask step as its reference step so that the source record is not disclosed to
the model.  Evaluation must therefore recover a causal query's effective lag
from the oracle source action, without changing the model input.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from cmwm.schemas import Episode, QueryExample, QueryKind
from cmwm.synthetic.queries import effective_query_lag


PRIMARY_TASKS = tuple(kind.value for kind in QueryKind)
PRIMARY_LAGS = (64, 128)


@dataclass(frozen=True)
class EndpointRow:
    """Condition-blind identity and stratum for one evaluation occurrence."""

    occurrence_id: str
    episode_id: str
    query_id: str
    task: str
    effective_lag: int


def endpoint_rows(
    batches: Sequence[Sequence[tuple[Episode, QueryExample]]],
) -> tuple[EndpointRow, ...]:
    """Materialize stable occurrence identities for a frozen validation cohort."""

    rows: list[EndpointRow] = []
    for batch_index, pairs in enumerate(batches):
        for row_index, (episode, example) in enumerate(pairs):
            rows.append(
                EndpointRow(
                    occurrence_id=f"batch-{batch_index:03d}:row-{row_index:03d}",
                    episode_id=episode.episode_id,
                    query_id=example.query.query_id,
                    task=example.query.kind.value,
                    effective_lag=effective_query_lag(episode, example),
                )
            )
    return tuple(rows)


def summarize_endpoint_rows(
    rows: Iterable[EndpointRow],
    *,
    primary_tasks: Sequence[str] = PRIMARY_TASKS,
    primary_lags: Sequence[int] = PRIMARY_LAGS,
) -> dict[str, object]:
    """Summarize endpoint coverage and fail closed when a required cell is absent."""

    materialized = tuple(rows)
    occurrence_ids = [row.occurrence_id for row in materialized]
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError("occurrence identifiers must be unique")

    query_counts = Counter(row.query_id for row in materialized)
    task_counts = Counter(row.task for row in materialized)
    lag_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in materialized:
        lag_counts[row.task][row.effective_lag] += 1

    required_cells = {
        f"{task}@{lag}": int(lag_counts[task][int(lag)])
        for task in primary_tasks
        for lag in primary_lags
    }
    missing_cells = sorted(key for key, value in required_cells.items() if value == 0)
    repeated_queries = {
        query_id: count for query_id, count in sorted(query_counts.items()) if count > 1
    }
    return {
        "row_count": len(materialized),
        "unique_query_count": len(query_counts),
        "duplicate_query_occurrences": sum(count - 1 for count in query_counts.values()),
        "repeated_queries": repeated_queries,
        "task_counts": dict(sorted(task_counts.items())),
        "effective_lag_histograms": {
            task: {str(lag): count for lag, count in sorted(counts.items())}
            for task, counts in sorted(lag_counts.items())
        },
        "required_cells": required_cells,
        "missing_required_cells": missing_cells,
        "endpoint_computable": not missing_cells,
        "rows": [asdict(row) for row in materialized],
    }
