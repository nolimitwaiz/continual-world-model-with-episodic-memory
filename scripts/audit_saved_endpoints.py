#!/usr/bin/env python3
"""Post-hoc sensitivity audit of saved development endpoints, never training.

Uses E2 final prediction rows and E3 aggregate task/seed scores. Does not invent
query-level E3 results. Output goes to a new directory and is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 23, 47, 89, 131)
CONDITIONS = ("context_only", "frame_memory", "fixed_chunks", "random_matched", "oracle_events")
TASKS = ("delayed_state", "object_location_history", "occlusion_continuity", "causal_predecessor")
CELLS = tuple(f"{task}@{lag}" for task in TASKS for lag in (64, 128))
GROUPS = {
    "original_composite": CELLS,
    "answer_tasks_only": tuple(cell for cell in CELLS if not cell.startswith("causal_predecessor@")),
    "pointer_task_only": tuple(cell for cell in CELLS if cell.startswith("causal_predecessor@")),
}
T_DF4 = 2.7764451051977987


def paired_contrast(candidate, baseline):
    if len(candidate) != len(SEEDS) or len(baseline) != len(SEEDS):
        raise ValueError("Require the five declared paired seeds")
    if not all(isfinite(x) and 0 <= x <= 1 for x in [*candidate, *baseline]):
        raise ValueError("Scores must be finite fractions in [0, 1]")
    deltas = [a - b for a, b in zip(candidate, baseline)]
    center = mean(deltas)
    half = T_DF4 * stdev(deltas) / sqrt(len(SEEDS))
    return {
        "mean_delta": center, "paired_seed_95_t_interval": [center - half, center + half],
        "deltas_in_seed_order": deltas, "positive_seeds": sum(x > 0 for x in deltas),
        "leave_one_seed_out_mean_deltas": {
            str(seed): mean(deltas[:index] + deltas[index + 1:])
            for index, seed in enumerate(SEEDS)
        },
    }


def validate_e2_run(artifact):
    if artifact["step"] != 1000 or artifact["claim_eligible"] is not False:
        raise ValueError("Only fixed-step development endpoints are eligible for this audit")
    rows = artifact["predictions"]
    if len(rows) != 512 or len({row["query_id"] for row in rows}) != 512:
        raise ValueError("Expected 512 unique query rows")
    cells = defaultdict(list)
    for row in rows:
        available = row["target"] != -100
        if row["target_available"] != available:
            raise ValueError("Stored target-availability flag is inconsistent")
        if row["correct"] != (available and row["prediction"] == row["target"]):
            raise ValueError("Stored correctness is inconsistent")
        cells[f"{row['task']}@{row['effective_lag']}"].append(row)
    if set(cells) != set(CELLS) or any(len(value) != 64 for value in cells.values()):
        raise ValueError("Expected eight cells of 64 queries")
    identity = sorted((row["query_id"], row["episode_id"], row["task"], row["effective_lag"],
                       row["target"] if row["task"] != "causal_predecessor" else None) for row in rows)
    return {
        "identity": identity,
        "cell_scores": {key: mean(row["correct"] for row in value) for key, value in cells.items()},
        "eligible_targets_by_cell": {key: sum(row["target_available"] for row in value) for key, value in cells.items()},
        "prediction_types": dict(Counter(row["prediction_type"] for row in rows)),
        "unique_episodes": len({row["episode_id"] for row in rows}),
    }


def summarize_cells(cells_by_condition):
    for condition in CONDITIONS:
        cells = cells_by_condition[condition]
        if set(cells) != set(CELLS) or any(len(values) != 5 for values in cells.values()):
            raise ValueError("Incomplete task/lag/seed matrix")
        if not all(isfinite(x) and 0 <= x <= 1 for values in cells.values() for x in values):
            raise ValueError("Invalid task-cell score")
    groups = {}
    for group_name, included in GROUPS.items():
        scores = {condition: [mean(cells_by_condition[condition][cell][i] for cell in included)
                              for i in range(5)] for condition in CONDITIONS}
        groups[group_name] = {
            "cells": included, "scores_in_seed_order": scores,
            "means": {condition: mean(values) for condition, values in scores.items()},
            "oracle_minus_fixed": paired_contrast(scores["oracle_events"], scores["fixed_chunks"]),
            "oracle_minus_random": paired_contrast(scores["oracle_events"], scores["random_matched"]),
        }
    return groups


def audit(root=ROOT):
    sources = []
    e2_runs = {condition: {} for condition in CONDITIONS}
    cohort = None
    unique_episodes = None
    for path in sorted((root / "results/cluster/1779882_e2_final").glob("*/seed-*/predictions/*.json")):
        obj = json.loads(path.read_text())
        condition, seed = obj["condition"], obj["seed"]
        if condition not in CONDITIONS or seed not in SEEDS or seed in e2_runs[condition]:
            raise ValueError("Unexpected or duplicate run identity")
        checked = validate_e2_run(obj)
        if cohort is None:
            cohort, unique_episodes = checked["identity"], checked["unique_episodes"]
        if checked.pop("identity") != cohort:
            raise ValueError("Endpoint questions or answer targets differ across runs")
        e2_runs[condition][seed] = checked
        sources.append(path)
    if any(set(runs) != set(SEEDS) for runs in e2_runs.values()):
        raise ValueError("Missing condition/seed results")
    e2_cells = {condition: {cell: [runs[seed]["cell_scores"][cell] for seed in SEEDS]
                            for cell in CELLS} for condition, runs in e2_runs.items()}
    e3_path = root / "results/cluster/1806501_e3_final/summary.json"
    e3 = json.loads(e3_path.read_text())
    if tuple(e3["seeds"]) != SEEDS or e3["primary_checkpoint"] != 1000:
        raise ValueError("E3 seed/endpoint mismatch")
    if e3["claim_eligible"] is not False or e3["test_accessed"] is not False:
        raise ValueError("Do not relabel development as confirmation")
    e3_groups = summarize_cells(e3["task_lag_scores_by_condition"])
    for condition in CONDITIONS:
        calculated = e3_groups["original_composite"]["scores_in_seed_order"][condition]
        if not all(isclose(a, b, abs_tol=1e-12) for a, b in zip(calculated, e3["scores_by_condition"][condition])):
            raise ValueError("E3 task means do not reconstruct saved original endpoint")
    sources.append(e3_path)
    return {
        "artifact_type": "saved_development_endpoint_sensitivity_v1", "analysis_date": "2026-11-11",
        "exploratory_post_hoc": True, "original_primary_endpoint_replaced": False,
        "new_training_runs": 0, "test_accessed": False, "seeds": SEEDS,
        "units": "All scores, deltas and confidence intervals are fractions.",
        "E2": {"groups": summarize_cells(e2_cells), "task_lag_scores": e2_cells,
               "run_audits": e2_runs, "unique_endpoint_queries": 512,
               "unique_endpoint_episodes": unique_episodes, "prediction_rows_checked": 12800},
        "E3": {"groups": e3_groups, "task_lag_scores": e3["task_lag_scores_by_condition"],
               "query_level_audit_available": False},
        "limitations": [
            "Post-hoc development sensitivity analysis, not independent replication or a new training experiment.",
            "Intervals describe paired training seeds on a reused cohort, not independent-world uncertainty.",
            "Task and leave-one-out analyses are exploratory; no multiple-comparison-adjusted claim is made.",
            "Pointer target eligibility is not a measured source-retention audit.",
            "Original E3 per-query outputs and checkpoint tensors are not included locally.",
        ],
        "sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
    }


def markdown_report(result):
    lines = ["# Saved endpoint sensitivity analysis", "", "11 November 2026. Exploratory reanalysis; no new training or test access.", "",
             "The original composite endpoint remains primary. Answer-only scores exclude causal-pointer retrieval because its target space depends on grouping; this is a post-hoc diagnostic, not a new confirmatory endpoint.", ""]
    for study in ("E2", "E3"):
        lines += [f"## {study} scores", "", "| Condition | Original composite | Answer tasks only | Pointer task only |", "|---|---:|---:|---:|"]
        groups = result[study]["groups"]
        for condition in CONDITIONS:
            values = " | ".join(f"{groups[group]['means'][condition] * 100:.2f}%" for group in GROUPS)
            lines.append(f"| {condition} | {values} |")
        lines += ["", "| Diagnostic | Contrast | Mean difference in points | Paired seed 95% interval | Positive seeds | Leave one seed out range in points |", "|---|---|---:|---|---:|---|"]
        for group_name, group in groups.items():
            for key in ("oracle_minus_fixed", "oracle_minus_random"):
                item = group[key]
                low, high = item["paired_seed_95_t_interval"]
                loo = list(item["leave_one_seed_out_mean_deltas"].values())
                lines.append(f"| {group_name} | {key} | {item['mean_delta'] * 100:+.2f} | [{low * 100:.2f}, {high * 100:.2f}] | {item['positive_seeds']}/5 | [{min(loo) * 100:.2f}, {max(loo) * 100:.2f}] |")
        lines.append("")
    lines += ["## Data checks", "", f"E2: 25 runs, 12,800 endpoint prediction rows, 512 unique questions from {result['E2']['unique_endpoint_episodes']} unique episodes. Every run has eight task/lag cells of 64 rows. Identity, correctness and target-eligibility checks passed.", "",
              "E3 task/seed means reconstruct the saved composite. Its original per-query outputs are not available here, so no E3 query-level leakage, retention or episode-resampling claim follows.", "",
              "Full task cells, seed differences, leave-one-out values, E2 target eligibility and source hashes are in `sensitivity.json`.", "", "## Interpretation limits", ""]
    lines += [f"- {item}" for item in result["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit directory")
    result = audit()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(markdown_report(result))
    print(json.dumps({"output": str(args.output.resolve()), "passed": True, "new_training_runs": 0,
                      "E2": result["E2"]["groups"]["answer_tasks_only"],
                      "E3": result["E3"]["groups"]["answer_tasks_only"]}, indent=2))


if __name__ == "__main__":
    main()
