"""Dependency-free boundary metrics for deterministic benchmark validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float
    average_precision: float
    false_boundaries_per_episode: float
    missed_boundary_rate: float
    mean_detection_latency: float | None


def _validate_steps(steps: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(sorted(steps))
    if any(step < 0 for step in result):
        raise ValueError(f"{name} steps must be non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} steps must be unique")
    return result


def _match_boundaries(
    truth: Sequence[int], predicted: Sequence[int], tolerance: int
) -> tuple[int, tuple[int, ...]]:
    unmatched = set(truth)
    latencies: list[int] = []
    matches = 0
    for candidate in predicted:
        possible = [step for step in unmatched if abs(candidate - step) <= tolerance]
        if not possible:
            continue
        target = min(possible, key=lambda step: (abs(candidate - step), step))
        unmatched.remove(target)
        matches += 1
        latencies.append(candidate - target)
    return matches, tuple(latencies)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_at_positive: list[float] = []
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_at_positive.append(true_positives / rank)
    return fsum(precision_at_positive) / positives


def boundary_metrics(
    *,
    true_steps: Iterable[int],
    predicted_steps: Iterable[int],
    episode_length: int,
    tolerance: int = 0,
    probabilities: Sequence[float] | None = None,
) -> BoundaryMetrics:
    """Evaluate sparse boundaries with one-to-one tolerance-aware matching.

    ``probabilities`` must contain one boundary probability for every episode
    step.  Average precision is intentionally reported instead of ordinary
    accuracy because boundaries are sparse.
    """

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    truth = _validate_steps(true_steps, name="true")
    predicted = _validate_steps(predicted_steps, name="predicted")
    if any(step >= episode_length for step in (*truth, *predicted)):
        raise ValueError("boundary step falls outside the episode")

    matches, latencies = _match_boundaries(truth, predicted, tolerance)
    precision = matches / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = matches / len(truth) if truth else (1.0 if not predicted else 0.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    labels = [1 if step in set(truth) else 0 for step in range(episode_length)]
    if probabilities is None:
        predicted_set = set(predicted)
        probability_values = [
            1.0 if step in predicted_set else 0.0 for step in range(episode_length)
        ]
    else:
        if len(probabilities) != episode_length:
            raise ValueError("probabilities must match episode_length")
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("boundary probabilities must lie in [0, 1]")
        probability_values = list(probabilities)

    return BoundaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=_average_precision(labels, probability_values),
        false_boundaries_per_episode=float(len(predicted) - matches),
        missed_boundary_rate=(len(truth) - matches) / len(truth) if truth else 0.0,
        mean_detection_latency=(
            fsum(latencies) / len(latencies) if latencies else None
        ),
    )
