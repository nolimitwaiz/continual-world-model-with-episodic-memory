from __future__ import annotations

import pytest

from cmwm.evaluation.gates import (
    Budget,
    GateThresholds,
    PairedEffect,
    evaluate_detector_gate,
    evaluate_oracle_gate,
    verify_matched_budgets,
)
from cmwm.evaluation.metrics import boundary_metrics


def test_boundary_metrics_use_tolerance_aware_one_to_one_matching() -> None:
    metrics = boundary_metrics(
        true_steps=[10, 20],
        predicted_steps=[9, 11, 22],
        episode_length=30,
        tolerance=2,
    )

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.false_boundaries_per_episode == 1.0
    assert metrics.mean_detection_latency == pytest.approx(0.5)


def test_oracle_gate_requires_effect_confidence_and_seeds() -> None:
    passing = PairedEffect(
        baseline_mean=0.60,
        candidate_mean=0.68,
        mean_delta=0.08,
        ci_low=0.03,
        ci_high=0.13,
        seeds=5,
    )
    failing = PairedEffect(
        baseline_mean=0.60,
        candidate_mean=0.68,
        mean_delta=0.08,
        ci_low=-0.01,
        ci_high=0.17,
        seeds=5,
    )

    assert evaluate_oracle_gate(passing).passed
    assert not evaluate_oracle_gate(failing).passed


def test_detector_gate_measures_oracle_gain_recovery() -> None:
    oracle = PairedEffect(0.50, 0.60, 0.10, 0.05, 0.15, 5)
    learned = PairedEffect(0.50, 0.58, 0.08, 0.03, 0.13, 5)

    result = evaluate_detector_gate(
        learned_effect=learned,
        oracle_effect=oracle,
        thresholds=GateThresholds(learned_oracle_gain_recovery=0.75),
    )

    assert result.passed


def test_budget_gate_rejects_unfair_storage() -> None:
    baseline = Budget(4096, 4, 32, 1000, 10_000, 1e8)
    candidate = Budget(8192, 4, 32, 1000, 10_000, 1e8)

    result = verify_matched_budgets(baseline, candidate)

    assert not result.passed
    assert not result.checks["storage_bytes"]
