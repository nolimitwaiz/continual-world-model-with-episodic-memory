"""Pre-registered go/no-go gates for the first memory experiment.

The functions in this module deliberately contain no training code.  They turn
paired experimental summaries into explicit decisions and prevent post-hoc
changes to what counts as success.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping


@dataclass(frozen=True)
class Budget:
    """Resources that must be matched across segmentation conditions."""

    storage_bytes: int
    retrieved_items: int
    observation_history_steps: int
    train_examples: int
    parameter_count: int
    train_flops: float | None = None
    tuning_trials: int = 0
    hardware_class: str = ""

    def __post_init__(self) -> None:
        integer_fields = (
            self.storage_bytes,
            self.retrieved_items,
            self.observation_history_steps,
            self.train_examples,
            self.parameter_count,
            self.tuning_trials,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("budget values must be non-negative")
        if self.train_flops is not None and self.train_flops < 0:
            raise ValueError("train_flops must be non-negative")


@dataclass(frozen=True)
class PairedEffect:
    """Paired candidate-minus-baseline effect for a higher-is-better metric."""

    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci_low: float
    ci_high: float
    seeds: int

    def __post_init__(self) -> None:
        if self.seeds < 1:
            raise ValueError("seeds must be positive")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low cannot exceed ci_high")
        observed_delta = self.candidate_mean - self.baseline_mean
        if not isclose(observed_delta, self.mean_delta, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("mean_delta must equal candidate_mean - baseline_mean")

    @property
    def relative_error_reduction(self) -> float:
        """Return relative error reduction when scores lie in [0, 1]."""

        if not 0.0 <= self.baseline_mean <= 1.0:
            raise ValueError("relative error reduction requires a [0, 1] metric")
        if not 0.0 <= self.candidate_mean <= 1.0:
            raise ValueError("relative error reduction requires a [0, 1] metric")
        baseline_error = 1.0 - self.baseline_mean
        if baseline_error == 0.0:
            return 0.0
        return (baseline_error - (1.0 - self.candidate_mean)) / baseline_error


@dataclass(frozen=True)
class GateThresholds:
    """Prototype 1 thresholds frozen before confirmatory cluster runs."""

    minimum_seeds: int = 5
    oracle_absolute_gain: float = 0.05
    oracle_relative_error_reduction: float = 0.20
    learned_oracle_gain_recovery: float = 0.75
    confidence_lower_bound: float = 0.0
    parameter_relative_tolerance: float = 0.02
    flops_relative_tolerance: float = 0.05


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    checks: Mapping[str, bool]
    reason: str


def verify_matched_budgets(
    baseline: Budget,
    candidate: Budget,
    *,
    parameter_relative_tolerance: float = 0.02,
    flops_relative_tolerance: float = 0.05,
) -> GateResult:
    """Check fairness-critical resources before comparing outcome metrics."""

    if parameter_relative_tolerance < 0 or flops_relative_tolerance < 0:
        raise ValueError("relative tolerances must be non-negative")

    def relative_match(left: float, right: float, tolerance: float) -> bool:
        return abs(right - left) / max(abs(left), 1.0) <= tolerance

    exact_fields = {
        "storage_bytes": baseline.storage_bytes == candidate.storage_bytes,
        "retrieved_items": baseline.retrieved_items == candidate.retrieved_items,
        "observation_history_steps": (
            baseline.observation_history_steps == candidate.observation_history_steps
        ),
        "train_examples": baseline.train_examples == candidate.train_examples,
        "parameter_count": relative_match(
            baseline.parameter_count,
            candidate.parameter_count,
            parameter_relative_tolerance,
        ),
        "tuning_trials": baseline.tuning_trials == candidate.tuning_trials,
        "hardware_class": baseline.hardware_class == candidate.hardware_class,
    }
    if baseline.train_flops is None or candidate.train_flops is None:
        flops_match = baseline.train_flops is candidate.train_flops
    else:
        flops_match = relative_match(
            baseline.train_flops,
            candidate.train_flops,
            flops_relative_tolerance,
        )
    checks = {**exact_fields, "train_flops": flops_match}
    passed = all(checks.values())
    return GateResult(
        name="matched_budget",
        passed=passed,
        checks=checks,
        reason=(
            "budgets match"
            if passed
            else "candidate and baseline budgets are not matched"
        ),
    )


def evaluate_oracle_gate(
    effect: PairedEffect,
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> GateResult:
    """Decide whether semantically correct segmentation has downstream value."""

    effect_size = (
        effect.mean_delta >= thresholds.oracle_absolute_gain
        or effect.relative_error_reduction
        >= thresholds.oracle_relative_error_reduction
    )
    checks = {
        "minimum_seeds": effect.seeds >= thresholds.minimum_seeds,
        "positive_confidence_interval": (
            effect.ci_low > thresholds.confidence_lower_bound
        ),
        "minimum_effect_size": effect_size,
    }
    passed = all(checks.values())
    return GateResult(
        name="oracle_value",
        passed=passed,
        checks=checks,
        reason=(
            "oracle segmentation demonstrates downstream value"
            if passed
            else "stop or redesign: oracle segmentation did not clear its gate"
        ),
    )


def evaluate_detector_gate(
    *,
    learned_effect: PairedEffect,
    oracle_effect: PairedEffect,
    thresholds: GateThresholds = GateThresholds(),
) -> GateResult:
    """Decide whether learned boundaries recover enough of the oracle benefit."""

    if oracle_effect.mean_delta <= 0:
        recovery = 0.0
    else:
        recovery = learned_effect.mean_delta / oracle_effect.mean_delta
    checks = {
        "minimum_seeds": learned_effect.seeds >= thresholds.minimum_seeds,
        "positive_confidence_interval": (
            learned_effect.ci_low > thresholds.confidence_lower_bound
        ),
        "oracle_gain_recovery": recovery >= thresholds.learned_oracle_gain_recovery,
    }
    passed = all(checks.values())
    return GateResult(
        name="learned_detector",
        passed=passed,
        checks=checks,
        reason=(
            f"learned detector recovered {recovery:.1%} of oracle gain"
            if passed
            else f"stop or revise detector: recovered {recovery:.1%} of oracle gain"
        ),
    )
