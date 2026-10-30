from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cmwm.chen.sherlock_graph_free import (
    AtomicSegment,
    RecallQuery,
    atomic_timing_diagnostics,
    build_memory_records,
    build_recall_queries,
    evaluate_bm25,
    load_atomic_segments,
    scene_spans_from_segments,
    subject_to_transcript_alias,
)


def _segments() -> list[AtomicSegment]:
    scene_numbers = [1, 1, 2, 3, 3, 3]
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    return [
        AtomicSegment(index, scene, float(index - 1), float(index), word)
        for index, (scene, word) in enumerate(zip(scene_numbers, words), start=1)
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_atomic_segments_requires_ordered_complete_scene_groups(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    rows = []
    for segment in _segments():
        rows.append(
            {
                "values": {
                    "Segment Number": segment.segment_number,
                    "Start Time (s)": segment.start_sec,
                    "End Time (s)": segment.start_sec if segment.segment_number == 1 else segment.end_sec,
                    "Scene Segments": (
                        f"{segment.scene_number}. Scene {segment.scene_number}"
                        if segment.segment_number in {1, 3, 4}
                        else ""
                    ),
                    "Scene Details - A Level": segment.text,
                }
            }
        )
    _write_jsonl(path, rows)

    loaded = load_atomic_segments(path, expected_segments=6, expected_scenes=3)

    assert [item.scene_number for item in loaded] == [1, 1, 2, 3, 3, 3]
    assert atomic_timing_diagnostics(loaded) == {
        "zero_duration_segments": 1,
        "negative_duration_segments": 0,
        "start_time_resets": 0,
    }
    assert [item.text for item in loaded] == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
    ]


def test_random_memory_matches_human_length_multiset_without_copying_boundaries() -> None:
    human = build_memory_records(_segments(), condition="human_events")
    random_records = build_memory_records(_segments(), condition="random_matched", random_seed=11)

    human_lengths = [record.end_segment - record.start_segment + 1 for record in human]
    random_lengths = [record.end_segment - record.start_segment + 1 for record in random_records]
    assert sorted(random_lengths) == sorted(human_lengths) == [1, 2, 3]
    assert random_lengths != human_lengths
    assert [record.start_segment for record in random_records] == [1, 3, 6]


def test_build_recall_queries_uses_published_alias_crosswalk_and_aggregates_occurrences(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "tokens.jsonl"
    occurrence_path = tmp_path / "occurrences.csv"
    _write_jsonl(
        token_path,
        [
            {
                "transcript_alias": "NN01",
                "source_row": 1,
                "word_value_type": "string",
                "word_raw": "alpha",
                "princeton_word_onset_sec": 0.5,
            },
            {
                "transcript_alias": "NN01",
                "source_row": 2,
                "word_value_type": "numeric",
                "word_raw": 7,
                "princeton_word_onset_sec": 1.0,
            },
            {
                "transcript_alias": "NN01",
                "source_row": 3,
                "word_value_type": "string",
                "word_raw": "bravo",
                "princeton_word_onset_sec": 2.5,
            },
        ],
    )
    with occurrence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_subject_id",
                "scene_number",
                "recall_start_sec",
                "recall_end_sec",
                "recall_sequence_index",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_subject_id": "s1",
                    "scene_number": 1,
                    "recall_start_sec": 0,
                    "recall_end_sec": 2,
                    "recall_sequence_index": 1,
                },
                {
                    "source_subject_id": "s1",
                    "scene_number": 1,
                    "recall_start_sec": 2,
                    "recall_end_sec": 3,
                    "recall_sequence_index": 2,
                },
            ]
        )

    queries, diagnostics = build_recall_queries(
        token_path, occurrence_path, subject_splits={"s1": "development"}
    )

    assert subject_to_transcript_alias("s1") == "NN01"
    assert len(queries) == 1
    assert queries[0].text == "alpha bravo"
    assert queries[0].occurrence_count == 2
    assert diagnostics["non_string_token_rows_excluded"] == 1


def test_bm25_reports_micro_and_subject_macro_metrics() -> None:
    records = build_memory_records(_segments(), condition="human_events")
    queries = [
        RecallQuery("q1", "s1", "NN01", 1, "alpha", 1, "development"),
        RecallQuery("q2", "s2", "NN02", 3, "foxtrot", 1, "development"),
    ]

    rows, summary = evaluate_bm25(
        records,
        queries,
        split="development",
        scene_spans=scene_spans_from_segments(_segments()),
        ks=(1, 2),
    )

    assert len(rows) == 2
    assert summary["micro_hit_at_1"] == 1.0
    assert summary["subject_macro_hit_at_1"] == 1.0
    assert summary["subject_macro_best_iou_hit_at_1"] == 1.0
    assert summary["subject_macro_iou_ndcg_at_1"] == 1.0
    assert summary["records"] == 3


def test_alias_crosswalk_rejects_out_of_cohort_subjects() -> None:
    with pytest.raises(ValueError, match="invalid Sherlock source subject"):
        subject_to_transcript_alias("s18")
