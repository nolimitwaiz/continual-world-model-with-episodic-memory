from __future__ import annotations

import pytest

from cmwm.evaluation.replication import paired_development_effect


def test_paired_development_effect_uses_seed_differences() -> None:
    effect = paired_development_effect(
        [0.70, 0.72, 0.74, 0.76, 0.78],
        [0.76, 0.78, 0.80, 0.82, 0.84],
    )

    assert effect["mean_delta"] == pytest.approx(0.06)
    assert effect["positive_seed_count"] == 5
    assert effect["paired_t_ci_low"] == pytest.approx(0.06)
    assert effect["relative_error_reduction"] == pytest.approx(0.2307692308)


def test_paired_development_effect_requires_five_pairs() -> None:
    with pytest.raises(ValueError, match="five paired seeds"):
        paired_development_effect([0.5], [0.6])
