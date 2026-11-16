"""Small arithmetic/validation fixtures; no training, model imports or data generation."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("saved_audit", ROOT / "scripts/audit_saved_endpoints.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    rows = []
    for task in audit.TASKS:
        for lag in (64, 128):
            for index in range(64):
                rows.append({"query_id": f"{task}/{lag}/{index}", "episode_id": f"episode-{index}",
                             "task": task, "effective_lag": lag, "target": 1, "prediction": 1,
                             "correct": True, "target_available": True,
                             "prediction_type": "fixture"})
    return {"step": 1000, "claim_eligible": False, "predictions": rows}


class SavedEndpointSensitivityTests(unittest.TestCase):
    def test_valid_fixture(self):
        result = audit.validate_e2_run(fixture())
        self.assertEqual(result["unique_episodes"], 64)
        self.assertEqual(set(result["cell_scores"].values()), {1})

    def test_duplicate_query_rejected(self):
        obj = fixture()
        obj["predictions"][1]["query_id"] = obj["predictions"][0]["query_id"]
        with self.assertRaises(ValueError):
            audit.validate_e2_run(obj)

    def test_bad_correctness_rejected(self):
        obj = fixture()
        obj["predictions"][0]["prediction"] = 2
        with self.assertRaises(ValueError):
            audit.validate_e2_run(obj)

    def test_missing_target_remains_wrong(self):
        obj = fixture()
        obj["predictions"][0].update(target=-100, target_available=False, correct=False)
        result = audit.validate_e2_run(obj)
        self.assertEqual(result["cell_scores"]["delayed_state@64"], 63 / 64)

    def test_eligibility_mismatch_rejected(self):
        obj = fixture()
        obj["predictions"][0]["target_available"] = False
        with self.assertRaises(ValueError):
            audit.validate_e2_run(obj)

    def test_contrast_is_paired_and_signed(self):
        result = audit.paired_contrast([.9, .8, .7, .6, .5], [.8, .7, .6, .7, .4])
        self.assertAlmostEqual(result["mean_delta"], .06)
        self.assertEqual(result["positive_seeds"], 4)
        self.assertAlmostEqual(result["leave_one_seed_out_mean_deltas"]["89"], .1)

    def test_wrong_seed_count_rejected(self):
        with self.assertRaises(ValueError):
            audit.paired_contrast([.5] * 4, [.5] * 4)

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            audit.paired_contrast([float("nan")] * 5, [.5] * 5)

    def test_answer_diagnostic_excludes_pointer(self):
        cells = {condition: {cell: [0.0 if cell.startswith("causal_predecessor@") else 1.0] * 5
                             for cell in audit.CELLS} for condition in audit.CONDITIONS}
        groups = audit.summarize_cells(cells)
        self.assertEqual(groups["original_composite"]["means"]["oracle_events"], .75)
        self.assertEqual(groups["answer_tasks_only"]["means"]["oracle_events"], 1)
        self.assertEqual(groups["pointer_task_only"]["means"]["oracle_events"], 0)


if __name__ == "__main__":
    unittest.main()
