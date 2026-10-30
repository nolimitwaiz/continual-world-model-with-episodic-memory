from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "submit_slurm.py"
SPEC = importlib.util.spec_from_file_location("submit_slurm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_seed_expansion_is_condition_major_and_does_not_mutate_inputs() -> None:
    path = Path("fixture.yaml")
    original = [(path, {"experiment": {"name": "c0", "seed": 0}})]

    expanded = MODULE._expand_seeds(original, (11, 23, 47))

    assert [config["experiment"]["seed"] for _, config in expanded] == [11, 23, 47]
    assert original[0][1]["experiment"]["seed"] == 0


@pytest.mark.parametrize("raw", ["", "11,11", "-1,23", "one,23"])
def test_seed_parser_fails_closed(raw: str) -> None:
    with pytest.raises(ValueError, match="seeds"):
        MODULE._parse_seeds(raw)
