"""Fail-closed aggregation for the five-seed development replication."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence


EXPECTED_RUNS = {
    "context_only": "replication_c0_context",
    "frame_memory": "replication_c1_frame",
    "fixed_chunks": "replication_c2_fixed",
    "random_matched": "replication_c3_random_matched",
    "oracle_events": "replication_c4_oracle",
}
EXPECTED_SEEDS = (11, 23, 47, 89, 131)
T_CRITICAL_95_DF4 = 2.7764451051977987


def paired_development_effect(
    baseline: Sequence[float], candidate: Sequence[float]
) -> dict[str, float | int | list[float]]:
    if len(baseline) != len(candidate) or len(baseline) != 5:
        raise ValueError("paired development effects require exactly five paired seeds")
    deltas = [float(right) - float(left) for left, right in zip(baseline, candidate)]
    mean_delta = fmean(deltas)
    margin = T_CRITICAL_95_DF4 * stdev(deltas) / math.sqrt(len(deltas))
    baseline_mean = fmean(baseline)
    candidate_mean = fmean(candidate)
    baseline_error = 1.0 - baseline_mean
    relative_error_reduction = (
        mean_delta / baseline_error if baseline_error > 0.0 else 0.0
    )
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_delta": mean_delta,
        "paired_t_ci_low": mean_delta - margin,
        "paired_t_ci_high": mean_delta + margin,
        "relative_error_reduction": relative_error_reduction,
        "positive_seed_count": sum(delta > 0.0 for delta in deltas),
        "deltas_by_seed_order": deltas,
    }


def aggregate_development_replication(run_root: str | Path) -> dict[str, object]:
    root = Path(run_root).expanduser().resolve()
    results: dict[str, dict[int, dict]] = {}
    for condition, run_name in EXPECTED_RUNS.items():
        results[condition] = {}
        for seed in EXPECTED_SEEDS:
            path = root / run_name / f"seed-{seed}" / "result.json"
            if not path.is_file():
                raise ValueError(f"missing replication result: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("claim_eligible") is not False:
                raise ValueError(f"development result became claim eligible: {path}")
            if int(result.get("seed", -1)) != seed or result.get("condition") != condition:
                raise ValueError(f"result identity mismatch: {path}")
            if int(result.get("steps", -1)) != 1000:
                raise ValueError(f"replication result has the wrong step count: {path}")
            validation = result.get("validation", {})
            if int(validation.get("examples", -1)) != 512:
                raise ValueError(f"replication result has the wrong cohort size: {path}")
            results[condition][seed] = result

    identity_fields = (
        "source_sha256",
        "dataset_source_sha256",
        "g0_manifest_sha256",
        "g0_seal_sha256",
        "torch_version",
    )
    reference = results["context_only"][EXPECTED_SEEDS[0]]
    for condition_results in results.values():
        for result in condition_results.values():
            for field in identity_fields:
                if result.get(field) != reference.get(field):
                    raise ValueError(f"replication identity mismatch for {field}")
    for seed in EXPECTED_SEEDS:
        cohort_hashes = {
            results[condition][seed].get("validation_cohort_sha256")
            for condition in EXPECTED_RUNS
        }
        if len(cohort_hashes) != 1 or None in cohort_hashes:
            raise ValueError(f"validation cohort is not paired for seed {seed}")

    final_scores = {
        condition: [
            float(results[condition][seed]["validation"]["long_delay_memory_score"])
            for seed in EXPECTED_SEEDS
        ]
        for condition in EXPECTED_RUNS
    }
    baseline = "fixed_chunks"
    oracle_effect = paired_development_effect(
        final_scores[baseline], final_scores["oracle_events"]
    )
    random_effect = paired_development_effect(
        final_scores["random_matched"], final_scores["oracle_events"]
    )
    return {
        "artifact_type": "cmwm_development_replication_summary_v1",
        "claim_eligible": False,
        "evidence_class": "development_replication",
        "seeds": list(EXPECTED_SEEDS),
        "primary_checkpoint": 1000,
        "development_selected_baseline": baseline,
        "selection_source": "independent seed-0 pilot 1764055",
        "scores_by_condition": final_scores,
        "means_by_condition": {
            condition: fmean(values) for condition, values in final_scores.items()
        },
        "oracle_vs_fixed": oracle_effect,
        "oracle_vs_random_matched": random_effect,
        "development_diagnostics": {
            "oracle_vs_fixed_positive_ci": oracle_effect["paired_t_ci_low"] > 0.0,
            "oracle_vs_fixed_practical_effect": (
                oracle_effect["mean_delta"] >= 0.05
                or oracle_effect["relative_error_reduction"] >= 0.20
            ),
            "oracle_vs_fixed_four_of_five_positive": (
                oracle_effect["positive_seed_count"] >= 4
            ),
        },
        "g1_decision": "not_evaluated_on_development_data",
        "confirmatory_blockers": [
            "No HF-A/HF-B held-out evaluation was run.",
            "Current validation LDMS is pooled accuracy rather than the frozen task/lag/family macro-average.",
            "Hierarchical episode-within-seed confidence intervals require saved paired predictions.",
        ],
        "identities": {field: reference[field] for field in identity_fields},
        "hardware_by_condition_and_seed": {
            condition: {
                str(seed): results[condition][seed].get("cuda_device")
                for seed in EXPECTED_SEEDS
            }
            for condition in EXPECTED_RUNS
        },
    }
