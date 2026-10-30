from __future__ import annotations

import pytest

from cmwm.models.boundaries import (
    BoundaryFeatures,
    FixedWindowSegmenter,
    RandomMatchedSegmenter,
    Segment,
    SurpriseThresholdSegmenter,
    heuristic_boundary_probability,
    segments_from_boundaries,
)


def test_split_position_closes_the_prior_event() -> None:
    assert segments_from_boundaries(10, [3, 7]) == (
        Segment(0, 3),
        Segment(3, 7),
        Segment(7, 10),
    )


def test_random_null_matches_full_segment_length_distribution() -> None:
    strategy = RandomMatchedSegmenter(
        reference_segment_lengths=(1, 2, 3, 14), seed=17
    )
    first = strategy.boundaries(length=20)
    second = strategy.boundaries(length=20)

    assert first == second
    assert sorted(segment.length for segment in segments_from_boundaries(20, first)) == [
        1,
        2,
        3,
        14,
    ]


def test_threshold_respects_refractory_period() -> None:
    strategy = SurpriseThresholdSegmenter(threshold=0.7, refractory_steps=2)

    assert strategy.boundaries(probabilities=[0.9, 0.8, 0.9, 0.1, 0.8]) == (1, 4)


def test_heuristic_boundary_probability_is_bounded_and_monotonic() -> None:
    low = heuristic_boundary_probability(BoundaryFeatures(0, 0, 0, 0, 1))
    high = heuristic_boundary_probability(BoundaryFeatures(4, 1, 1, 1, 1))

    assert 0.0 <= low < high <= 1.0


def test_internal_context_change_can_raise_boundary_probability() -> None:
    stable = BoundaryFeatures(0, 0, 0, 0, 1, context_change=0)
    shifted = BoundaryFeatures(0, 0, 0, 0, 1, context_change=2)

    assert heuristic_boundary_probability(stable) < heuristic_boundary_probability(shifted)


def test_fixed_windows_reject_zero_width() -> None:
    with pytest.raises(ValueError):
        FixedWindowSegmenter(width=0).boundaries(length=10)
