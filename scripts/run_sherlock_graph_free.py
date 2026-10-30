#!/usr/bin/env python3
"""Run the frozen development-only Sherlock graph-free BM25 matrix."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from cmwm.chen.sherlock_graph_free import (
    atomic_timing_diagnostics,
    build_memory_records,
    build_recall_queries,
    evaluate_bm25,
    load_atomic_segments,
    scene_spans_from_segments,
    sha256,
)


SEALED_TEST_ENV = "CMWM_SHERLOCK_TEST_AUTHORIZED"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("claim_eligible") is not False:
        raise ValueError("the first Sherlock run must remain development-only")
    return value


def _subject_splits(config: dict) -> dict[str, str]:
    splits = config["splits"]
    development = list(splits["development_subjects"])
    sealed = list(splits["sealed_test_subjects"])
    if len(development) != 12 or len(sealed) != 5 or set(development) & set(sealed):
        raise ValueError("Sherlock subject split must be disjoint 12/5")
    expected = {f"s{index}" for index in range(1, 18)}
    if set(development) | set(sealed) != expected:
        raise ValueError("Sherlock subject split must cover s1..s17")
    return {**{item: "development" for item in development}, **{item: "test" for item in sealed}}


def _mean_numeric(summaries: list[dict]) -> dict[str, float]:
    keys = sorted(
        key
        for key, value in summaries[0].items()
        if isinstance(value, (int, float)) and key not in {"random_seed"}
    )
    return {
        key: sum(float(summary[key]) for summary in summaries) / len(summaries)
        for key in keys
    }


def run(config_path: Path, source_project: Path, output_directory: Path, split: str) -> dict:
    config_path = config_path.expanduser().resolve()
    source_project = source_project.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if split not in {"development", "test"}:
        raise ValueError("split must be development or test")
    config = _load_config(config_path)
    if split == "test" and (
        config["splits"].get("sealed_test_authorized") is not True
        or os.environ.get(SEALED_TEST_ENV) != "explicit-post-development-authorization"
    ):
        raise PermissionError(
            "sealed Sherlock test refused: protocol and environment both require explicit authorization"
        )
    if split != config["splits"]["active_split"]:
        raise ValueError("requested split disagrees with the frozen active split")

    dataset = config["dataset"]
    labels = source_project / dataset["atomic_labels"]
    tokens = source_project / dataset["recall_tokens"]
    occurrences = source_project / dataset["recall_occurrences"]
    segments = load_atomic_segments(
        labels,
        expected_segments=int(dataset["expected_atomic_segments"]),
        expected_scenes=int(dataset["expected_human_events"]),
    )
    timing_diagnostics = atomic_timing_diagnostics(segments)
    if timing_diagnostics["zero_duration_segments"] != int(
        dataset["expected_zero_duration_segments"]
    ):
        raise ValueError("zero-duration segment count disagrees with frozen protocol")
    if timing_diagnostics["start_time_resets"] != int(dataset["expected_start_time_resets"]):
        raise ValueError("start-time reset count disagrees with frozen protocol")
    queries, diagnostics = build_recall_queries(
        tokens, occurrences, subject_splits=_subject_splits(config)
    )
    if diagnostics["raw_token_rows"] != int(dataset["expected_recall_token_rows"]):
        raise ValueError("recall token count disagrees with frozen protocol")
    if diagnostics["recall_occurrences"] != int(dataset["expected_recall_occurrences"]):
        raise ValueError("recall occurrence count disagrees with frozen protocol")

    retrieval = config["retrieval"]
    scene_spans = scene_spans_from_segments(segments)
    ks = tuple(int(value) for value in retrieval["cutoffs"])
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for condition in ("fine_units", "fixed_chunks", "human_events"):
        records = build_memory_records(
            segments,
            condition=condition,
            fixed_size=int(config["conditions"]["fixed_chunks"]["fixed_size"]),
        )
        rows, summary = evaluate_bm25(
            records,
            queries,
            split=split,
            scene_spans=scene_spans,
            ks=ks,
            k1=float(retrieval["k1"]),
            b=float(retrieval["b"]),
        )
        for row in rows:
            row["random_seed"] = None
        summary["random_seed"] = None
        all_rows.extend(rows)
        summaries.append(summary)

    random_summaries: list[dict[str, object]] = []
    for seed in config["conditions"]["random_matched"]["seeds"]:
        records = build_memory_records(
            segments, condition="random_matched", random_seed=int(seed)
        )
        rows, summary = evaluate_bm25(
            records,
            queries,
            split=split,
            scene_spans=scene_spans,
            ks=ks,
            k1=float(retrieval["k1"]),
            b=float(retrieval["b"]),
        )
        for row in rows:
            row["random_seed"] = int(seed)
        summary["random_seed"] = int(seed)
        all_rows.extend(rows)
        random_summaries.append(summary)
        summaries.append(summary)

    primary_key = config["evaluation"].get(
        "primary_metric_key",
        f"subject_macro_hit_at_{int(retrieval['primary_cutoff'])}",
    )
    human = next(summary for summary in summaries if summary["condition"] == "human_events")
    random_values = [float(summary[primary_key]) for summary in random_summaries]
    report = {
        "artifact_type": config.get(
            "result_artifact_type", "cmwm_sherlock_graph_free_bm25_development_v1"
        ),
        "claim_eligible": False,
        "split": split,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_project": str(source_project),
        "source_sha256": {
            str(labels.relative_to(source_project)): sha256(labels),
            str(tokens.relative_to(source_project)): sha256(tokens),
            str(occurrences.relative_to(source_project)): sha256(occurrences),
        },
        "query_diagnostics": diagnostics,
        "atomic_timing_diagnostics": timing_diagnostics,
        "summaries": summaries,
        "random_matched_mean": _mean_numeric(random_summaries),
        "primary_development_contrast": {
            "metric": primary_key,
            "human_events": float(human[primary_key]),
            "random_matched_mean": sum(random_values) / len(random_values),
            "human_minus_random_matched": float(human[primary_key])
            - sum(random_values) / len(random_values),
            "random_seed_min": min(random_values),
            "random_seed_max": max(random_values),
        },
        "limitations": config["limitations"],
        "sealed_test_accessed": split == "test",
    }

    result_path = output_directory / "result.json"
    rows_path = output_directory / "rankings.jsonl"
    if result_path.exists() or rows_path.exists():
        raise FileExistsError("refusing to overwrite an existing Sherlock result")
    _atomic_text(result_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_text(
        rows_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    args = parser.parse_args()
    report = run(args.config, args.source_project, args.output_directory, args.split)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
