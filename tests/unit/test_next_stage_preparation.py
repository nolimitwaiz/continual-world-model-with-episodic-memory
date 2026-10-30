"""Tiny invented fixtures only: no real data, model execution or scientific result."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cmwm.chen.sherlock_dense import (
    build_dense_records, encoder_text_inputs, evaluate_dense_development,
    normalized_float32, rank_dense,
)
from cmwm.chen.sherlock_graph_free import AtomicSegment, RecallQuery, scene_spans_from_segments
from cmwm.evaluation.confirmation import (
    CONDITIONS, FAMILIES, LAGS, ConfirmationPrediction, ConfirmationQuery,
    summarize_confirmation, validate_cohort, validate_heldout_episode_structure,
)
from cmwm.evaluation.conformance import PRIMARY_TASKS
from cmwm.evaluation.replication import EXPECTED_SEEDS
from cmwm.schemas import DatasetSplit, EntityKind


def fixture():
    cohort = tuple(
        ConfirmationQuery(f"{f}-{lag}-{i}-{t}", f"{f}-{lag}-{i}", f, t, lag, "a" * 64)
        for f in FAMILIES for lag in LAGS for i in range(2) for t in PRIMARY_TASKS
    )
    rows = tuple(
        ConfirmationPrediction(seed, condition, q, condition == "oracle_events")
        for seed in EXPECTED_SEEDS for condition in CONDITIONS for q in cohort
    )
    return cohort, rows


def summarize(rows, cohort):
    return summarize_confirmation(rows, cohort=cohort, examples_per_cell=2, bootstrap_replicates=12)


class ConfirmationTests(unittest.TestCase):
    def test_legacy_heldout_structure_refused(self):
        templates = {"HF-A": ("invented_fixture_A",), "HF-B": ("invented_fixture_B",)}
        meta = SimpleNamespace(dataset_split=DatasetSplit.TEST, held_out_family="HF-A",
                               template_id="invented_fixture_A")
        kinds = [EntityKind.OBJECT, EntityKind.CONTAINER] * 2
        entities = tuple(SimpleNamespace(entity_id=str(i), kind=kind) for i, kind in enumerate(kinds))
        episode = SimpleNamespace(split_metadata=meta, initial_state=SimpleNamespace(entities=entities))
        validate_heldout_episode_structure(episode, approved_templates=templates)
        episode.initial_state.entities = entities[:2]
        with self.assertRaises(ValueError):
            validate_heldout_episode_structure(episode, approved_templates=templates)
        episode.initial_state.entities = entities
        meta.template_id = "held_out_persistence_composition_v1"
        with self.assertRaises(ValueError):
            validate_heldout_episode_structure(episode, approved_templates=templates)
        with self.assertRaises(ValueError):
            validate_heldout_episode_structure(episode, approved_templates={})

    def test_perfect_fixture_and_no_gate_authority(self):
        cohort, rows = fixture()
        result = summarize(rows, cohort)
        self.assertEqual(result["unique_queries"], 32)
        self.assertEqual(result["unique_episodes"], 8)
        self.assertEqual(result["scores_by_condition"]["oracle_events"], [1.] * 5)
        self.assertEqual(result["contrasts"]["fixed_chunks"]["paired_hierarchical_ci_95"], [1., 1.])
        self.assertTrue(all(result["contrasts"]["fixed_chunks"]["numerical_checks_only"].values()))
        self.assertFalse(result["claim_eligible"])
        self.assertTrue(result["g1_decision"].startswith("not_assessed"))

    def test_order_invariant_and_deterministic(self):
        cohort, rows = fixture()
        mixed = tuple(replace(r, correct=(r.query.episode_id.endswith("1") and r.seed != 89)
                              if r.condition == "oracle_events" else False) for r in rows)
        self.assertEqual(summarize(mixed, cohort), summarize(reversed(mixed), tuple(reversed(cohort))))

    def test_null_does_not_pass_numerical_checks(self):
        cohort, rows = fixture()
        result = summarize(tuple(replace(r, correct=False) for r in rows), cohort)
        checks = result["contrasts"]["fixed_chunks"]["numerical_checks_only"]
        self.assertFalse(any(checks.values()))

    def test_family_reversal_visible(self):
        cohort, rows = fixture()
        mixed = tuple(replace(r, correct=(r.query.family == "HF-A") if r.condition == "oracle_events"
                              else r.query.family == "HF-B") for r in rows)
        effect = summarize(mixed, cohort)["contrasts"]["fixed_chunks"]
        self.assertEqual(effect["family_deltas"], {"HF-A": 1., "HF-B": -1.})
        self.assertFalse(effect["numerical_checks_only"]["both_families_positive"])

    def test_incomplete_and_duplicate_predictions_rejected(self):
        cohort, rows = fixture()
        for invalid in (rows[:-1], rows + (rows[0],)):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                summarize(invalid, cohort)

    def test_prediction_corruption_rejected(self):
        cohort, rows = fixture()
        variants = [replace(rows[0], seed=999), replace(rows[0], condition="new"),
                    replace(rows[0], checkpoint_step=900), replace(rows[0], correct=math.nan),
                    replace(rows[0], query=replace(rows[0].query, answer_sha256="b" * 64))]
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize((bad,) + rows[1:], cohort)

    def test_bad_cohorts_rejected(self):
        cohort, _ = fixture()
        variants = [cohort[:-1], cohort + (cohort[0],),
                    (replace(cohort[0], answer_sha256="invalid"),) + cohort[1:],
                    (replace(cohort[0], episode_id="cross-episode"),) + cohort[1:]]
        for bad in variants:
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                validate_cohort(bad, examples_per_cell=2)


class DenseTests(unittest.TestCase):
    def setUp(self):
        self.segments = tuple(AtomicSegment(i, 1 if i <= 2 else 2, float(i), float(i + 1),
                                            f"observation {i}") for i in range(1, 7))
        self.vectors = {i: (1., 0.) if i <= 2 else (0., 1.) for i in range(1, 7)}
        self.queries = (RecallQuery("q1", "s1", "NN01", 1, "first thing", 1, "development"),
                        RecallQuery("q2", "s1", "NN01", 2, "second thing", 1, "development"))

    def records(self, condition="human_events", vectors=None):
        return build_dense_records(self.segments, self.vectors if vectors is None else vectors,
                                   condition=condition, expected_records=2, fixed_size=3)

    def test_matched_record_bytes_and_length_multiset(self):
        conditions = {name: self.records(name) for name in ("fixed_chunks", "random_matched", "human_events")}
        self.assertEqual({sum(r.logical_payload_bytes for r in group) for group in conditions.values()}, {32})
        lengths = lambda group: sorted(r.end_segment - r.start_segment + 1 for r in group)
        self.assertEqual(lengths(conditions["human_events"]), lengths(conditions["random_matched"]))
        self.assertTrue(all(not hasattr(r, "text") for group in conditions.values() for r in group))

    def test_text_only_encoding_boundary(self):
        texts, queries = encoder_text_inputs(self.segments, self.queries)
        self.assertEqual(texts, tuple(f"observation {i}" for i in range(1, 7)))
        self.assertEqual(queries, ("first thing", "second thing"))

    def test_rank_and_development_score(self):
        records = self.records()
        self.assertEqual(rank_dense(records, (1., 0.)), (0, 1))
        result = evaluate_dense_development(records, self.queries, {"q1": (1., 0.), "q2": (0., 1.)},
                                            scene_spans_from_segments(self.segments), cutoff=1)
        self.assertEqual(result["subject_macro_best_iou_hit"], 1.)
        self.assertFalse(result["claim_eligible"])
        self.assertFalse(result["sealed_test_accessed"])
        self.assertEqual([r["returned_atomic_units"] for r in result["rows"]], [2, 4])

    def test_tie_break_never_uses_scene_answer(self):
        self.assertEqual(rank_dense(self.records(), (1., 1.)), (0, 1))

    def test_sealed_subject_and_test_split_refused(self):
        for query in (replace(self.queries[0], split="test"), replace(self.queries[0], source_subject_id="s11")):
            with self.subTest(query=query), self.assertRaises(PermissionError):
                encoder_text_inputs(self.segments, (query,))
            with self.assertRaises(PermissionError):
                evaluate_dense_development(self.records(), (query,), {"q1": (1., 0.)}, {1: (1, 2)}, cutoff=1)

    def test_vector_errors_refused(self):
        for value in ((), (0., 0.), (math.nan, 1.), (math.inf, 1.)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalized_float32(value)
        for vectors in ({1: (1., 0.)}, {**self.vectors, 1: (1.,)}, {**self.vectors, 1: (0., 0.)}):
            with self.assertRaises(ValueError):
                self.records(vectors=vectors)

    def test_count_identity_and_span_errors_refused(self):
        with self.assertRaises(ValueError):
            build_dense_records(self.segments, self.vectors, condition="human_events", expected_records=3)
        with self.assertRaises(ValueError):
            self.records("fine_units")
        with self.assertRaises(ValueError):
            rank_dense(self.records(), (1.,))
        with self.assertRaises(ValueError):
            rank_dense((replace(self.records()[0], vector=(2., 0.)),), (1., 0.))
        bad = (replace(self.records()[0], end_segment=4), self.records()[1])
        with self.assertRaises(ValueError):
            evaluate_dense_development(bad, self.queries, {"q1": (1., 0.), "q2": (0., 1.)},
                                       {1: (1, 2), 2: (3, 6)}, cutoff=1)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("prepare_next_experiments", root / "scripts/prepare_next_experiments.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.config = json.loads((root / "configs/next_stage/preparation.json").read_text())

    def test_inventory_is_inert(self):
        result = self.module.preparation_inventory(self.config)
        self.assertEqual(result["submitted_jobs"], 0)
        self.assertEqual(result["training_jobs"], 0)
        self.assertEqual(result["proposed_synthetic_checkpoint_evaluations"], 25)
        self.assertEqual(result["proposed_unique_synthetic_queries"], 1024)
        self.assertEqual(result["proposed_sherlock_groupings"], 12)

    def test_authorization_flags_cannot_turn_preparer_into_launcher(self):
        for flag in ("protocol_frozen", "execution_authorized", "test_access_authorized", "training_authorized"):
            with self.subTest(flag=flag), self.assertRaises(PermissionError):
                self.module.preparation_inventory({**self.config, flag: True})


if __name__ == "__main__":
    unittest.main()
