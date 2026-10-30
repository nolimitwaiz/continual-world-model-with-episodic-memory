"""Read-only validation of our trusted cluster smoke artifacts; no optimization."""
import json
import math
import os
from pathlib import Path

import torch
from cmwm.synthetic.manifest import content_sha256
from cmwm.training import _source_tree_sha256


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    return True


root = Path(os.environ["CMWM_RUN_ROOT"]) / "e3_c4_engineering_20260926/seed-0"
result = read(root / "result.json")
status = read(root / "run_status.json")
config = read(root / "resolved_config.json")
manifest = read(os.environ["CMWM_DEVELOPMENT_ENDPOINT_MANIFEST"])
g0 = read(os.environ["CMWM_G0_ATTESTATION"])
history = read(root / "metrics_history.json")
index = read(root / "checkpoint_index.json")
require(status["state"] == "completed" and result["steps"] == 20, "incomplete smoke")
require(result["slurm_job_id"] == "1806489", "unexpected smoke job")
require(result["checkpoint_count"] == 4, "wrong checkpoint count")
require(result["claim_eligible"] is False, "engineering claim boundary changed")
require(config["data"]["test_episodes"] == 0, "test access is forbidden")
require(content_sha256(config) == result["config_sha256"], "config digest mismatch")
require(result["source_sha256"] == _source_tree_sha256() ==
        "067f49cc162929d935b7c1b497eb3c8131d7a1662358ceb1360a11ff623b3cec", "source changed")
require(result["dataset_source_sha256"] == g0["dataset_source_sha256"], "dataset mismatch")
require(result["g0_manifest_sha256"] == g0["manifest_sha256"], "G0 mismatch")
require(result["g0_seal_sha256"] == g0["seal_sha256"], "G0 seal mismatch")
require(result["endpoint_manifest_sha256"] == manifest["manifest_sha256"], "endpoint mismatch")
require(result["endpoint_manifest_seal_sha256"] == manifest["seal_sha256"], "endpoint seal mismatch")
require([r["step"] for r in history] == [5, 10, 15, 20], "missing metric checkpoint")
require([r["step"] for r in index] == [5, 10, 15, 20], "missing checkpoint index")
require(finite_tree(history) and finite_tree(result), "nonfinite metrics")
identity_fields = ["source_sha256", "dataset_source_sha256", "config_sha256",
                   "g0_manifest_sha256", "g0_seal_sha256", "endpoint_manifest_sha256",
                   "endpoint_manifest_seal_sha256", "validation_cohort_sha256",
                   "condition", "seed", "experiment"]
for metric, checkpoint_row in zip(history, index):
    step = metric["step"]
    artifact = read(root / "predictions" / f"step-{step:08d}.json")
    rows = artifact["predictions"]
    require(artifact["step"] == step, "prediction step mismatch")
    require(artifact["claim_eligible"] is False, "prediction claim flag")
    for field in ["endpoint_manifest_sha256", "endpoint_manifest_seal_sha256",
                  "validation_cohort_sha256", "condition", "seed"]:
        require(artifact[field] == result[field], f"prediction identity: {field}")
    require(len(rows) == 512 == len({r["query_id"] for r in rows}), "prediction count")
    for row, selected in zip(rows, manifest["selected"]):
        require(all(row[k] == selected[k] for k in
                    ["query_id", "episode_id", "task", "effective_lag"]), "query order/identity")
        require(row["correct"] == (row["target_available"] and
                row["prediction"] == row["target"]), "incorrect correctness flag")
    digest = content_sha256(rows)
    require(digest == artifact["summary"]["prediction_identity_sha256"] ==
            metric["validation"]["prediction_identity_sha256"], "prediction digest")
    accuracy = sum(r["correct"] for r in rows) / 512
    require(math.isclose(accuracy, metric["validation"]["long_delay_memory_score"],
                        abs_tol=1e-12), "score mismatch")
    # Only load checkpoints authored by this project's known smoke job.
    path = root / "checkpoints" / f"step-{step:08d}.pt"
    require(Path(checkpoint_row["path"]).resolve() == path.resolve(), "checkpoint path")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint["step"] == step and checkpoint["max_steps"] == 20, "checkpoint step")
    require(all(checkpoint[k] == result[k] for k in identity_fields), "checkpoint identities")
    require(finite_tree(checkpoint["model"]) and finite_tree(checkpoint["optimizer"]),
            "nonfinite model or optimizer")
    require(checkpoint["latest_evaluation"] == metric["validation"], "checkpoint evaluation")
    require(checkpoint["latest_loss"] == metric["loss"], "checkpoint loss")
last = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
require(last["step"] == 20, "last checkpoint step")
require(all(torch.equal(last["model"][k], v) for k, v in checkpoint["model"].items()),
        "last checkpoint differs from final named checkpoint")
require(result["validation"] == history[-1]["validation"], "final result mismatch")
print(json.dumps({"passed": True, "job": "1806489", "checkpoints_checked": 4,
                  "prediction_rows_checked": 2048, "finite_model_optimizer_metrics": True,
                  "final_smoke_ldms": result["validation"]["long_delay_memory_score"],
                  "elapsed_seconds": result["elapsed_seconds"],
                  "source_sha256": result["source_sha256"],
                  "claim_eligible": False}, indent=2))
