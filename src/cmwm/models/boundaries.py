"""Boundary baselines with explicit online semantics.

A boundary value is a split position between sequence items. For one-based
``StepRecord.step_index`` values, a boundary emitted after transition ``t`` is
passed as split position ``t``: segment ``[start, t)`` therefore contains
zero-based records ``0..t-1`` (world transitions ``1..t``), including the
transition and outcome that triggered the boundary. The next record begins the
new segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Segment:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("segments must satisfy 0 <= start < end")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class BoundaryFeatures:
    """Causal, online features available when deciding whether step t closes an event."""

    normalized_surprise: float
    semantic_change: float
    action_change: float
    goal_change: float
    delta_t: float
    context_change: float = 0.0


def heuristic_boundary_probability(
    features: BoundaryFeatures,
    *,
    weights: BoundaryFeatures = BoundaryFeatures(1.0, 1.0, 0.5, 1.0, 0.0, 1.0),
    bias: float = -2.0,
) -> float:
    """Return a calibrated-style heuristic score without using future frames.

    ``normalized_surprise`` should be a negative log likelihood or a residual
    normalized by predictive uncertainty.  Raw latent distance is intentionally
    not accepted or computed here.
    """

    logit = bias + sum(
        value * weight
        for value, weight in zip(
            (
                features.normalized_surprise,
                features.semantic_change,
                features.action_change,
                features.goal_change,
                features.delta_t,
                features.context_change,
            ),
            (
                weights.normalized_surprise,
                weights.semantic_change,
                weights.action_change,
                weights.goal_change,
                weights.delta_t,
                weights.context_change,
            ),
        )
    )
    if logit >= 0:
        return 1.0 / (1.0 + exp(-logit))
    exp_logit = exp(logit)
    return exp_logit / (1.0 + exp_logit)


def _validate_boundaries(length: int, boundaries: Iterable[int]) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("length must be positive")
    result = tuple(sorted(boundaries))
    if len(set(result)) != len(result):
        raise ValueError("boundary steps must be unique")
    if any(step <= 0 or step >= length for step in result):
        raise ValueError("boundary steps must lie strictly inside the sequence")
    return result


def segments_from_boundaries(length: int, boundaries: Iterable[int]) -> tuple[Segment, ...]:
    boundaries_tuple = _validate_boundaries(length, boundaries)
    endpoints = (0, *boundaries_tuple, length)
    return tuple(
        Segment(start, end) for start, end in zip(endpoints, endpoints[1:])
    )


@dataclass(frozen=True)
class FixedWindowSegmenter:
    width: int

    def boundaries(self, *, length: int) -> tuple[int, ...]:
        if self.width <= 0:
            raise ValueError("width must be positive")
        return tuple(range(self.width, length, self.width))


@dataclass(frozen=True)
class RandomMatchedSegmenter:
    """Random null preserving the complete reference segment-length multiset.

    Matching only the number of boundaries changes the writer's compression
    distribution and can create a storage or optimization advantage. This null
    permutes the reference lengths while preserving every length.
    """

    reference_segment_lengths: tuple[int, ...]
    seed: int

    def boundaries(self, *, length: int) -> tuple[int, ...]:
        if not self.reference_segment_lengths:
            raise ValueError("reference_segment_lengths cannot be empty")
        if any(item <= 0 for item in self.reference_segment_lengths):
            raise ValueError("reference segment lengths must be positive")
        if sum(self.reference_segment_lengths) != length:
            raise ValueError("reference segment lengths must cover the sequence exactly")
        randomized = list(self.reference_segment_lengths)
        Random(self.seed).shuffle(randomized)
        boundaries: list[int] = []
        cursor = 0
        for segment_length in randomized[:-1]:
            cursor += segment_length
            boundaries.append(cursor)
        return tuple(boundaries)


@dataclass(frozen=True)
class SurpriseThresholdSegmenter:
    threshold: float
    refractory_steps: int = 0

    def boundaries(self, *, probabilities: Sequence[float]) -> tuple[int, ...]:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if self.refractory_steps < 0:
            raise ValueError("refractory_steps must be non-negative")
        selected: list[int] = []
        last = -self.refractory_steps - 1
        for step, probability in enumerate(probabilities):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("probabilities must lie in [0, 1]")
            if step == 0:
                continue
            if probability >= self.threshold and step - last > self.refractory_steps:
                selected.append(step)
                last = step
        return tuple(selected)


@dataclass(frozen=True)
class OracleSegmenter:
    """Diagnostic upper bound supplied by the simulator event ontology."""

    true_boundary_steps: tuple[int, ...]

    def boundaries(self, *, length: int) -> tuple[int, ...]:
        return _validate_boundaries(length, self.true_boundary_steps)
