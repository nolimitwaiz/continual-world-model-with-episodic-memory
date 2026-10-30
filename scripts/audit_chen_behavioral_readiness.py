#!/usr/bin/env python3
"""Audit the existing lightweight FilmFest/Sherlock behavioral products.

This is a provenance/readiness check. It never downloads data, treats predicted
causal edges as human labels, or makes a model-performance claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


EXPECTED = {
    "filmfest_events": 202,
    "filmfest_stories": 10,
    "filmfest_recall_rows": 3030,
    "filmfest_published_cohort_rows": 2778,
    "sherlock_scenes": 50,
    "sherlock_subjects": 17,
    "sherlock_recall_occurrences": 616,
    "sherlock_scene_recall_rows": 850,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required) - fields)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return list(reader)


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            count += 1
    return count


def audit(source_project: Path) -> dict[str, object]:
    source_project = source_project.expanduser().resolve()
    filmfest = source_project / "data/filmfest"
    sherlock = source_project / "data/sherlock"

    filmfest_events_path = filmfest / "normalized_events.csv"
    filmfest_recall_path = filmfest / "recall_matrix.csv"
    predicted_edges_path = (
        filmfest
        / "causal_annotation_v2/predicted_graph_v2/predicted_directed_edges.score_ge_2.csv"
    )
    sherlock_scenes_path = sherlock / "normalized_scenes.csv"
    sherlock_occurrences_path = sherlock / "recall_occurrences.csv"
    sherlock_matrix_path = sherlock / "scene_recall_matrix.csv"
    sherlock_events_path = sherlock / "events.jsonl"

    filmfest_events = _csv_rows(
        filmfest_events_path,
        ("event_id", "story_id", "event_description", "causal_edges", "provenance"),
    )
    filmfest_recall = _csv_rows(
        filmfest_recall_path,
        ("subject_id", "event_id", "included_in_published_event_cohort"),
    )
    sherlock_scenes = _csv_rows(
        sherlock_scenes_path,
        ("event_id", "primary_start_sec_part", "primary_end_sec_part", "boundary_source"),
    )
    sherlock_occurrences = _csv_rows(
        sherlock_occurrences_path,
        ("source_subject_id", "event_id", "recall_start_sec", "recall_end_sec"),
    )
    sherlock_matrix = _csv_rows(
        sherlock_matrix_path,
        ("source_subject_id", "event_id", "recalled"),
    )

    observed = {
        "filmfest_events": len(filmfest_events),
        "filmfest_stories": len({row["story_id"] for row in filmfest_events}),
        "filmfest_recall_rows": len(filmfest_recall),
        "filmfest_published_cohort_rows": sum(
            row["included_in_published_event_cohort"].strip().lower() == "true"
            for row in filmfest_recall
        ),
        "sherlock_scenes": len(sherlock_scenes),
        "sherlock_subjects": len(
            {row["source_subject_id"] for row in sherlock_matrix}
        ),
        "sherlock_recall_occurrences": len(sherlock_occurrences),
        "sherlock_scene_recall_rows": len(sherlock_matrix),
    }
    checks = {
        key: {"expected": expected, "observed": observed[key], "passed": observed[key] == expected}
        for key, expected in EXPECTED.items()
    }
    checks["filmfest_unique_event_ids"] = {
        "expected": len(filmfest_events),
        "observed": len({row["event_id"] for row in filmfest_events}),
        "passed": len({row["event_id"] for row in filmfest_events}) == len(filmfest_events),
    }
    checks["sherlock_unique_event_ids"] = {
        "expected": len(sherlock_scenes),
        "observed": len({row["event_id"] for row in sherlock_scenes}),
        "passed": len({row["event_id"] for row in sherlock_scenes}) == len(sherlock_scenes),
    }
    checks["sherlock_event_jsonl"] = {
        "expected": EXPECTED["sherlock_scenes"],
        "observed": _jsonl_count(sherlock_events_path),
        "passed": _jsonl_count(sherlock_events_path) == EXPECTED["sherlock_scenes"],
    }

    paths = (
        filmfest_events_path,
        filmfest_recall_path,
        sherlock_scenes_path,
        sherlock_occurrences_path,
        sherlock_matrix_path,
        sherlock_events_path,
    )
    predicted_graph_present = predicted_edges_path.is_file()
    return {
        "artifact_type": "cmwm_chen_behavioral_readiness_v1",
        "claim_eligible": False,
        "source_project": str(source_project),
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "checks": checks,
        "files": {
            str(path.relative_to(source_project)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        },
        "readiness": {
            "filmfest_graph_free_behavioral": True,
            "sherlock_graph_free_behavioral": True,
            "filmfest_human_directed_causal_graph": False,
            "sherlock_human_directed_causal_graph": False,
            "raw_fmri_acquired": False,
        },
        "exploratory_predicted_graph": {
            "present": predicted_graph_present,
            "path": str(predicted_edges_path) if predicted_graph_present else None,
            "sha256": _sha256(predicted_edges_path) if predicted_graph_present else None,
            "permitted_role": "exploratory_only",
        },
        "limitations": [
            "No raw fMRI is checked or acquired by this audit.",
            "Predicted causal edges are not independent human directed edges.",
            "Readiness is not model-performance evidence.",
        ],
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source_project)
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
