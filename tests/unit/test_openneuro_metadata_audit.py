from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_openneuro_metadata.py"
SPEC = importlib.util.spec_from_file_location("audit_openneuro_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _snapshot(dataset_id: str, tag: str) -> dict:
    return {
        "id": f"{dataset_id}:{tag}",
        "tag": tag,
        "description": {
            "Name": "Fixture",
            "DatasetDOI": "10.example/fixture",
            "Authors": ["A. Researcher"],
            "License": "CC0",
        },
        "top": [
            {"filename": "sub-02", "size": 0, "directory": True, "annexed": False},
            {"filename": "README", "size": 5, "directory": False, "annexed": False},
            {"filename": "sub-01", "size": 0, "directory": True, "annexed": False},
        ],
        "recursive": [
            {"filename": "sub-01", "size": 0, "directory": True, "annexed": False},
            {"filename": "bold.nii.gz", "size": 100, "directory": False, "annexed": True},
            {"filename": "events.tsv", "size": 20, "directory": False, "annexed": False},
        ],
    }


def test_summarize_snapshot_counts_without_content() -> None:
    summary = MODULE.summarize_snapshot("fixture", "ds000001", "1.0.0", _snapshot("ds000001", "1.0.0"))

    assert summary["top_level_subject_directories"] == ["sub-01", "sub-02"]
    assert summary["top_level_subject_count"] == 2
    assert summary["recursive_file_count"] == 2
    assert summary["declared_bytes"] == 120
    assert summary["annexed_file_count"] == 1
    assert summary["extension_counts"] == {".nii.gz": 1, ".tsv": 1}


def test_snapshot_identity_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="expected immutable snapshot"):
        MODULE.summarize_snapshot("fixture", "ds000001", "1.0.0", _snapshot("ds000001", "1.0.1"))


def test_build_audit_requires_every_declared_snapshot() -> None:
    payload = {"data": {}}
    with pytest.raises(RuntimeError, match="omitted filmfest"):
        MODULE.build_audit(payload, queried_at="2026-08-16T00:00:00+00:00")
