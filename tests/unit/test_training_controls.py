from __future__ import annotations

from copy import deepcopy

import pytest

from cmwm.models.cmwm import MemoryCondition
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.training import (
    ENGINEERING_INTERRUPT_ENV,
    EvaluationSummary,
    TrainingContractError,
    ValidationCohort,
    _build_validation_cohort,
    _engineering_interrupt_step,
    _prediction_metrics,
)


def _config() -> dict:
    return {
        "data": {
            "validation_episodes": 20,
            "query_lags": [16, 32],
        },
        "training": {"batch_size": 2},
        "experiment": {"run_class": "engineering"},
    }


def test_validation_cohort_is_fixed_and_content_addressed() -> None:
    generator = SyntheticEpisodeGenerator(base_seed=1729, episode_steps=40)
    first = _build_validation_cohort(generator=generator, config=_config(), seed=100_007)
    repeated = _build_validation_cohort(
        generator=generator, config=deepcopy(_config()), seed=100_007
    )
    changed = _build_validation_cohort(generator=generator, config=_config(), seed=100_008)

    first_ids = [
        example.query.query_id for batch in first.batches for _, example in batch
    ]
    repeated_ids = [
        example.query.query_id for batch in repeated.batches for _, example in batch
    ]
    assert first_ids == repeated_ids
    assert first.sha256 == repeated.sha256
    assert first.sha256 != changed.sha256


def test_metrics_are_condition_aware() -> None:
    summary = EvaluationSummary(8, 4, 4, 2, 1, 3, 2, 1, 1)

    context = summary.to_dict(condition=MemoryCondition.CONTEXT_ONLY)
    learned = summary.to_dict(condition=MemoryCondition.LEARNED_EVENTS)

    assert context["retrieval_recall_at_four"] is None
    assert context["boundary_f1"] is None
    assert learned["retrieval_recall_at_four"] == pytest.approx(3 / 8)
    assert learned["boundary_f1"] == pytest.approx(2 / 3)


def test_primary_endpoint_is_equal_weight_task_by_lag_macro() -> None:
    cohort = ValidationCohort(
        seed=0,
        sha256="sealed",
        batches=(),
        tasks=("task-a", "task-b"),
        lags=(64, 128),
        examples_per_cell=2,
    )
    correctness = {
        ("task-a", 64): (True, True),
        ("task-a", 128): (False, False),
        ("task-b", 64): (True, False),
        ("task-b", 128): (True, True),
    }
    rows = []
    for (task, lag), values in correctness.items():
        for index, correct in enumerate(values):
            rows.append(
                {
                    "query_id": f"{task}-{lag}-{index}",
                    "task": task,
                    "effective_lag": lag,
                    "correct": correct,
                }
            )

    metrics = _prediction_metrics(rows, cohort=cohort)

    assert metrics["paired_prediction_count"] == 8
    assert metrics["macro_task_lag_accuracy"] == pytest.approx(0.625)
    assert metrics["task_lag_accuracy"]["task-a@128"] == 0.0


def test_engineering_interrupt_must_be_checkpoint_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENGINEERING_INTERRUPT_ENV, "10")
    assert _engineering_interrupt_step(
        _config(), checkpoint_every=5, max_steps=20
    ) == 10

    monkeypatch.setenv(ENGINEERING_INTERRUPT_ENV, "11")
    with pytest.raises(TrainingContractError, match="checkpoint step"):
        _engineering_interrupt_step(_config(), checkpoint_every=5, max_steps=20)


def test_confirmatory_run_cannot_enable_engineering_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["experiment"]["run_class"] = "confirmatory"
    monkeypatch.setenv(ENGINEERING_INTERRUPT_ENV, "10")
    with pytest.raises(TrainingContractError, match="only for engineering"):
        _engineering_interrupt_step(config, checkpoint_every=5, max_steps=20)
