#!/usr/bin/env python3
"""Check the research-facing numbers against stored artifacts; no model execution."""
from pathlib import Path
from math import isclose
from statistics import mean
import json

from audit_saved_endpoints import audit

ROOT = Path(__file__).resolve().parents[1]
AUDIT = "results/analysis/saved_endpoint_sensitivity_2026-11-11/sensitivity.json"


def main():
    saved = json.loads((ROOT / AUDIT).read_text())
    recalculated = json.loads(json.dumps(audit(ROOT)))
    assert saved == recalculated, "Saved sensitivity report differs from source artifacts"
    assert saved["new_training_runs"] == 0 and saved["test_accessed"] is False
    texts = {name: (ROOT / name).read_text() for name in
             ("README.md", "docs/results.md")}
    for study in ("E2", "E3"):
        for condition in ("fixed_chunks", "random_matched", "oracle_events"):
            expected = f'{saved[study]["groups"]["original_composite"]["means"][condition] * 100:.2f}%'
            for name, content in texts.items():
                assert expected in content, f"Missing verified score {expected}: {name}"
    sherlock = json.loads((ROOT / "results/cluster/1764075/result.json").read_text())
    metric = "subject_macro_best_iou_hit_at_5"
    random_rows = [row for row in sherlock["summaries"] if row["condition"] == "random_matched"]
    assert len(random_rows) == 10
    random_mean = mean(row[metric] for row in random_rows)
    assert isclose(random_mean, sherlock["random_matched_mean"][metric], abs_tol=1e-12)
    human = next(row for row in sherlock["summaries"] if row["condition"] == "human_events")
    assert human["queries"] == 401 and human["subjects"] == 12
    assert human[metric] < random_mean
    for value in (human[metric], random_mean):
        for name, content in texts.items():
            assert f"{value * 100:.2f}%" in content, name
    filmfest = json.loads((ROOT / "archive/filmfest/results/filmfest_reproduction/reproduction_evaluation.json").read_text())
    assert filmfest["tolerance"] == .015 and len(filmfest["checks"]) == 6
    for item in filmfest["checks"]:
        difference = abs(item["published"] - item["reproduced"])
        assert isclose(difference, item["absolute_difference"], abs_tol=1e-12)
        assert difference <= filmfest["tolerance"] and item["pass"] is True
        assert f'{item["reproduced"]:.6f}' in texts["docs/results.md"]
    print(json.dumps({
        "passed": True,
        "saved_sensitivity_recomputed": True,
        "e2_prediction_rows_checked": 12800,
        "e3_checked_from_summary_not_predictions": True,
        "sherlock_random_groupings_checked": 10,
        "filmfest_reproduction_checks": 6,
        "training_run": False,
        "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
