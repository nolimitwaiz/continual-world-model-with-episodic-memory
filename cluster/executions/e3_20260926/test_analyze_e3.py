"""CPU-only validation fixtures, not model experiments or scientific results."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

import analyze_e3 as analysis


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(os.environ["CMWM_PROJECT_ROOT"])
        cls.manifest_path = Path(os.environ["CMWM_DEVELOPMENT_ENDPOINT_MANIFEST"])
        cls.g0_path = Path(os.environ["CMWM_G0_ATTESTATION"])
        cls.manifest = analysis.read(cls.manifest_path)
        cls.g0 = analysis.read(cls.g0_path)
        cls.smoke = Path(os.environ["CMWM_RUN_ROOT"]) / "e3_c4_engineering_20260926/seed-0"
        cls.result = analysis.read(cls.smoke / "result.json")
        cls.prediction = analysis.read(cls.smoke / "predictions/step-00000020.json")

    def validate(self, artifact):
        return analysis.validate_predictions(artifact, self.manifest, self.result, 20)

    def test_real_smoke_prediction_validates(self):
        score, cells = self.validate(self.prediction)
        self.assertEqual(len(cells), 8)
        self.assertEqual(score, self.result["validation"]["long_delay_memory_score"])

    def test_prediction_corruption_rejected(self):
        mutations = [
            lambda a: a["predictions"].pop(),
            lambda a: a["predictions"].reverse(),
            lambda a: a["predictions"][0].update(query_id="wrong"),
            lambda a: a["predictions"][0].update(correct=not a["predictions"][0]["correct"]),
            lambda a: a.update(seed=999),
            lambda a: a.update(endpoint_manifest_sha256="wrong"),
            lambda a: a.update(claim_eligible=True),
            lambda a: a["summary"].update(prediction_identity_sha256="wrong"),
            lambda a: a["summary"].update(long_delay_memory_score=float("nan")),
            lambda a: a["summary"].update(examples=511),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                artifact = deepcopy(self.prediction)
                mutate(artifact)
                with self.assertRaises(ValueError):
                    self.validate(artifact)

    def test_gate_is_random_primary_not_fixed(self):
        scores = {k: [.5] * 5 for k in analysis.CONDITIONS}
        scores["oracle_events"] = [.6] * 5
        scores["fixed_chunks"] = [.65] * 5
        cells = {k: {"fixture@64": v} for k, v in scores.items()}
        report = analysis.summarize(scores, cells, {}, {})
        self.assertTrue(report["development_gate_passed"])
        self.assertLess(report["oracle_vs_fixed"]["mean_delta"], 0)
        scores["oracle_events"] = [.51] * 5
        report = analysis.summarize(scores, cells, {}, {})
        self.assertFalse(report["development_gate_passed"])

    def test_full_matrix_fixture_and_missing_run(self):
        # Explicitly fabricated file fixtures. Never use these as experiment output.
        with tempfile.TemporaryDirectory(prefix="cmwm-e3-analysis-test-") as temp:
            root = Path(temp)
            for condition, suffix in analysis.CONDITIONS.items():
                base_config = analysis.load_config(self.project / "configs/e3" / f"{suffix}.yaml")
                for seed in analysis.SEEDS:
                    folder = root / f"e3_{suffix}" / f"seed-{seed}"
                    (folder / "checkpoints").mkdir(parents=True)
                    (folder / "predictions").mkdir()
                    config = deepcopy(base_config)
                    config["experiment"]["seed"] = seed
                    result = deepcopy(self.result)
                    result.update(condition=condition, seed=seed, steps=1000, checkpoint_count=10,
                                  config_sha256=analysis.content_sha256(config))
                    history, index = [], []
                    for step in analysis.STEPS:
                        artifact = deepcopy(self.prediction)
                        artifact.update(condition=condition, seed=seed, step=step)
                        checkpoint = folder / "checkpoints" / f"step-{step:08d}.pt"
                        checkpoint.write_bytes(b"TEST FIXTURE NOT A CHECKPOINT")
                        (folder / "predictions" / f"step-{step:08d}.json").write_text(json.dumps(artifact))
                        history.append({"step": step, "validation": artifact["summary"],
                                        "elapsed_seconds": float(step), "loss": {"total": 1.0}})
                        index.append({"step": step, "latest_validation": artifact["summary"],
                                      "path": str(checkpoint)})
                    for name, value in [("result", result), ("resolved_config", config),
                                        ("run_status", {**result, "state": "completed"}),
                                        ("metrics_history", history), ("checkpoint_index", index)]:
                        (folder / f"{name}.json").write_text(json.dumps(value))
            report = analysis.analyze(root, self.project, self.manifest_path, self.g0_path)
            self.assertEqual(report["runs"], 25)
            self.assertEqual(report["prediction_rows_audited"], 128000)
            self.assertFalse(report["development_gate_passed"])
            self.assertFalse(report["checkpoint_tensors_reloaded"])
            self.assertIn("Primary: oracle minus random", analysis.markdown(report))
            missing = root / "e3_c4_oracle/seed-131/result.json"
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                analysis.analyze(root, self.project, self.manifest_path, self.g0_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
