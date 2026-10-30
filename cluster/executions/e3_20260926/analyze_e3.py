"""Fail-closed E3 development analysis, kept outside the frozen training tree."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import fmean

from cmwm.config import load_config
from cmwm.evaluation.endpoint_manifest import validate_development_endpoint_manifest
from cmwm.evaluation.replication import paired_development_effect
from cmwm.infrastructure.artifacts import atomic_write_json, atomic_write_text
from cmwm.synthetic.manifest import content_sha256

SEEDS = (11, 23, 47, 89, 131)
CONDITIONS = {
    "context_only": "c0_context", "frame_memory": "c1_frame",
    "fixed_chunks": "c2_fixed", "random_matched": "c3_random_matched",
    "oracle_events": "c4_oracle",
}
SOURCE = "067f49cc162929d935b7c1b497eb3c8131d7a1662358ceb1360a11ff623b3cec"
DATASET = "17cf309a2d67355981164d23d7d5d745be548a02fe16bea38622dd4022045e2f"
ENDPOINT = "3e7aa3a19d0beb9dce24f9c899405047926d966090f826d1adf4187a2208924c"
STEPS = tuple(range(100, 1001, 100))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_predictions(artifact, manifest, identity, step):
    require(artifact["artifact_type"] == "cmwm_endpoint_predictions_v1", "prediction type")
    require(artifact["claim_eligible"] is False and artifact["step"] == step, "prediction scope")
    for key in ("condition", "seed", "endpoint_manifest_sha256",
                "endpoint_manifest_seal_sha256", "validation_cohort_sha256"):
        require(artifact[key] == identity[key], f"prediction identity {key}")
    rows = artifact["predictions"]
    require(len(rows) == 512 == len({r["query_id"] for r in rows}), "prediction coverage")
    cells = defaultdict(list)
    for row, selected in zip(rows, manifest["selected"]):
        require(all(row[k] == selected[k] for k in
                    ("episode_id", "query_id", "task", "effective_lag")), "prediction order/labels")
        require(type(row["correct"]) is bool and type(row["target_available"]) is bool,
                "prediction boolean types")
        require(row["correct"] == (row["target_available"] and row["prediction"] == row["target"]),
                "prediction correctness flag")
        cells[f'{row["task"]}@{row["effective_lag"]}'].append(row["correct"])
    require(len(cells) == 8 and all(len(v) == 64 for v in cells.values()), "cell balance")
    accuracy = {k: fmean(v) for k, v in cells.items()}
    summary = artifact["summary"]
    require(summary["examples"] == summary["paired_prediction_count"] == 512, "summary count")
    require(content_sha256(rows) == summary["prediction_identity_sha256"], "prediction digest")
    require(summary["task_lag_accuracy"] == accuracy, "cell scores")
    score = fmean(accuracy.values())
    for key in ("long_delay_memory_score", "macro_task_lag_accuracy"):
        require(math.isclose(summary[key], score, abs_tol=1e-12), f"summary score {key}")
    require(finite(artifact), "nonfinite prediction artifact")
    return score, accuracy


def summarize(scores, cell_scores, curves, resources):
    primary = paired_development_effect(scores["random_matched"], scores["oracle_events"])
    secondary = paired_development_effect(scores["fixed_chunks"], scores["oracle_events"])
    gate = ((primary["mean_delta"] >= .05 or primary["relative_error_reduction"] >= .20)
            and primary["paired_t_ci_low"] > 0 and primary["positive_seed_count"] >= 4)
    cell_effects = {
        cell: {
            "oracle_vs_random": paired_development_effect(
                cell_scores["random_matched"][cell], cell_scores["oracle_events"][cell]),
            "oracle_vs_fixed": paired_development_effect(
                cell_scores["fixed_chunks"][cell], cell_scores["oracle_events"][cell]),
        } for cell in cell_scores["oracle_events"]
    }
    return {
        "artifact_type": "cmwm_e3_development_summary_v1", "claim_eligible": False,
        "test_accessed": False, "decision": "D-039", "seeds": list(SEEDS),
        "primary_checkpoint": 1000, "runs": 25, "endpoint_queries": 512,
        "primary_contrast": "oracle_events minus random_matched",
        "scores_by_condition": scores,
        "means_by_condition": {k: fmean(v) for k, v in scores.items()},
        "oracle_vs_random_matched": primary, "oracle_vs_fixed": secondary,
        "development_gate_passed": gate,
        "task_lag_scores_by_condition": cell_scores,
        "descriptive_cell_effects_unadjusted": cell_effects,
        "learning_curves_by_condition_and_seed": curves, "resources": resources,
        "limitations": [
            "Post-E2 development benchmark revision; not a preregistered confirmatory test.",
            "Five paired seeds; t intervals describe seed variability, not unseen-world uncertainty.",
            "Cellwise intervals are descriptive and unadjusted for multiple comparisons.",
            "Causal endpoint selects a source-containing retrieved record, not an exact causal action.",
            "Context/frame controls can lack source records by design; broad reasoning superiority does not follow.",
            "Oracle boundaries are supplied; learned segmentation, fast weights and raw fMRI are not tested.",
            "No evidence of superiority over all RAG systems or a complete multimodal world model.",
        ],
    }


def analyze(run_root, project, manifest_path, g0_path):
    manifest, g0 = read(manifest_path), read(g0_path)
    validate_development_endpoint_manifest(manifest, expected_dataset_source_sha256=DATASET,
                                          expected_generator_policy="confusable_events_v3")
    require(manifest["manifest_sha256"] == ENDPOINT, "unfrozen endpoint")
    require(g0["passed"] is True and g0["test_accessed"] is False, "G0 scope")
    require(g0["dataset_source_sha256"] == DATASET and not g0["errors"], "G0 source/errors")
    require(content_sha256(g0["manifest_seal"]) == g0["seal_sha256"], "G0 seal digest")
    require(g0["manifest_seal"]["manifest_sha256"] == g0["manifest_sha256"], "G0 identity")
    expected_ids = {
        "source_sha256": SOURCE, "dataset_source_sha256": DATASET,
        "g0_manifest_sha256": g0["manifest_sha256"], "g0_seal_sha256": g0["seal_sha256"],
        "endpoint_manifest_sha256": ENDPOINT, "endpoint_manifest_seal_sha256": manifest["seal_sha256"],
        "torch_version": "2.5.1+cu121",
    }
    scores, cell_scores, curves, resources = {}, {}, {}, {}
    cohort = None
    state_targets = None
    for condition, suffix in CONDITIONS.items():
        scores[condition], cell_scores[condition] = [], defaultdict(list)
        curves[condition], resources[condition] = {}, {}
        base_config = load_config(project / "configs/e3" / f"{suffix}.yaml")
        for seed in SEEDS:
            folder = run_root / f"e3_{suffix}" / f"seed-{seed}"
            result, status = read(folder / "result.json"), read(folder / "run_status.json")
            expected_config = deepcopy(base_config)
            expected_config["experiment"]["seed"] = seed
            config = read(folder / "resolved_config.json")
            require(config == expected_config, f"unexpected config {folder}")
            require(config["data"]["test_episodes"] == 0, "test data forbidden")
            require(result["config_sha256"] == content_sha256(config), "config digest")
            require(result["condition"] == condition and result["seed"] == seed, "run identity")
            require(result["steps"] == 1000 and result["claim_eligible"] is False, "run scope")
            require(status["state"] == "completed" and status["steps"] == 1000, "incomplete run")
            for key, expected in expected_ids.items():
                require(result[key] == status[key] == expected, f"run identity {key}")
            require(status["config_sha256"] == result["config_sha256"], "status config")
            if cohort is None:
                cohort = result["validation_cohort_sha256"]
            require(result["validation_cohort_sha256"] == cohort, "unpaired cohort")
            require(finite(result), "nonfinite result")
            history, checkpoints = read(folder / "metrics_history.json"), read(folder / "checkpoint_index.json")
            require([r["step"] for r in history] == list(STEPS), "missing learning-curve step")
            require([r["step"] for r in checkpoints] == list(STEPS), "missing checkpoint index")
            require(result["checkpoint_count"] == 10, "checkpoint count")
            curves[condition][str(seed)] = []
            for metric, checkpoint in zip(history, checkpoints):
                step = metric["step"]
                require(finite(metric), "nonfinite history")
                path = folder / "checkpoints" / f"step-{step:08d}.pt"
                require(path.is_file() and path.stat().st_size > 0, "missing checkpoint")
                require(Path(checkpoint["path"]).resolve() == path.resolve(), "checkpoint path")
                artifact = read(folder / "predictions" / f"step-{step:08d}.json")
                score, cells = validate_predictions(artifact, manifest, result, step)
                require(artifact["summary"] == metric["validation"] == checkpoint["latest_validation"],
                        "metric/checkpoint/prediction disagreement")
                curves[condition][str(seed)].append({"step": step, "score": score,
                                                      "elapsed_seconds": metric["elapsed_seconds"]})
            require(result["validation"] == history[-1]["validation"], "final result disagreement")
            targets = [(r["query_id"], r["target"]) for r in artifact["predictions"]
                       if r["prediction_type"] == "answer_vocabulary_index"]
            if state_targets is None:
                state_targets = targets
            require(targets == state_targets, "state targets differ across runs")
            scores[condition].append(score)
            for cell, value in cells.items():
                cell_scores[condition][cell].append(value)
            resources[condition][str(seed)] = {
                k: result[k] for k in ("elapsed_seconds", "peak_cuda_memory_bytes",
                                      "cuda_device", "hostname", "slurm_job_id")}
    report = summarize(scores, cell_scores, curves, resources)
    report["identities"] = {**expected_ids, "validation_cohort_sha256": cohort}
    report["prediction_rows_audited"] = 25 * 10 * 512
    report["checkpoint_files_indexed"] = 250
    report["checkpoint_tensors_reloaded"] = False
    report["hardware_homogeneous"] = len({r["cuda_device"] for c in resources.values()
                                           for r in c.values()}) == 1
    return report


def markdown(report):
    primary, fixed = report["oracle_vs_random_matched"], report["oracle_vs_fixed"]
    lines = ["# E3 complete development comparison", "", "Development evidence only; no test data accessed.",
             "", "| Memory | Mean score |", "| --- | ---: |"]
    lines += [f"| {k} | {v:.6f} |" for k, v in report["means_by_condition"].items()]
    lines += ["", f"D-039 development gate: {'PASS' if report['development_gate_passed'] else 'NOT MET'}.", ""]
    for name, effect in [("Primary: oracle minus random-matched", primary), ("Secondary: oracle minus fixed", fixed)]:
        lines.append(f"{name}: {effect['mean_delta']:+.6f}; 95% paired-seed t interval "
                     f"[{effect['paired_t_ci_low']:+.6f}, {effect['paired_t_ci_high']:+.6f}]; "
                     f"positive in {effect['positive_seed_count']}/5 seeds.")
    lines += ["", "## Limits", ""] + [f"- {x}" for x in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    for name in ("run-root", "project", "manifest", "g0", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "output exists; refusing to overwrite")
    report = analyze(args.run_root, args.project, args.manifest, args.g0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_dir / "summary.json", report)
    atomic_write_text(args.output_dir / "REPORT.md", markdown(report))
    print(json.dumps({"passed": True, "runs": 25, "output_dir": str(args.output_dir),
                      "development_gate_passed": report["development_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
