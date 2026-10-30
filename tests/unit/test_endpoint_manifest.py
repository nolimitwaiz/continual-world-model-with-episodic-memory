from copy import deepcopy

import pytest

from cmwm.evaluation.endpoint_manifest import (
    EndpointCandidate,
    build_balanced_endpoint_manifest,
    validate_development_endpoint_manifest,
)


def _candidates() -> list[EndpointCandidate]:
    rows = []
    index = 0
    for task in ("a", "b"):
        for lag in (64, 128):
            for item in range(4):
                rows.append(
                    EndpointCandidate(
                        episode_index=index,
                        episode_id=f"episode-{index}",
                        query_id=f"query-{index}",
                        task=task,
                        effective_lag=lag,
                        oracle_answer_sha256=f"answer-{item}",
                    )
                )
                index += 1
    return rows


def _build(rows: list[EndpointCandidate]) -> dict[str, object]:
    return build_balanced_endpoint_manifest(
        rows,
        tasks=("a", "b"),
        lags=(64, 128),
        examples_per_cell=2,
        dataset_source_sha256="dataset",
        config_sha256="config",
        generator_policy="exact_task_lag_v2",
    )


def test_balanced_manifest_is_deterministic_and_exact() -> None:
    first = _build(_candidates())
    second = _build(list(reversed(_candidates())))
    assert first == second
    assert first["selected_count"] == 8
    assert first["test_accessed"] is False
    selected = first["selected"]
    assert isinstance(selected, list)
    assert len({row["query_id"] for row in selected}) == 8


def test_balanced_manifest_fails_on_sparse_cell() -> None:
    rows = [
        row
        for row in _candidates()
        if not (row.task == "b" and row.effective_lag == 128)
    ]
    with pytest.raises(ValueError, match="insufficient endpoint cells"):
        _build(rows)


def test_balanced_manifest_rejects_duplicate_query_ids() -> None:
    rows = _candidates()
    rows[1] = EndpointCandidate(
        episode_index=rows[1].episode_index,
        episode_id=rows[1].episode_id,
        query_id=rows[0].query_id,
        task=rows[1].task,
        effective_lag=rows[1].effective_lag,
        oracle_answer_sha256=rows[1].oracle_answer_sha256,
    )
    with pytest.raises(ValueError, match="duplicate query IDs"):
        _build(rows)


def test_sealed_manifest_validation_detects_tampering() -> None:
    manifest = _build(_candidates())
    validate_development_endpoint_manifest(
        manifest,
        expected_dataset_source_sha256="dataset",
        expected_generator_policy="exact_task_lag_v2",
    )

    tampered = deepcopy(manifest)
    tampered["selected"][0]["task"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        validate_development_endpoint_manifest(
            tampered,
            expected_dataset_source_sha256="dataset",
            expected_generator_policy="exact_task_lag_v2",
        )
