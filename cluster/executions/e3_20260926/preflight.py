"""Read-only verification of the frozen E3 artifacts; never runs training."""
import json
import os
from collections import Counter
from pathlib import Path

from cmwm.config import load_config
from cmwm.training import (
    _require_development_endpoint_manifest,
    _require_g0_attestation,
    _source_tree_sha256,
)

project = Path(os.environ["CMWM_PROJECT_ROOT"])
expected_source = "17cf309a2d67355981164d23d7d5d745be548a02fe16bea38622dd4022045e2f"
expected_endpoint = "3e7aa3a19d0beb9dce24f9c899405047926d966090f826d1adf4187a2208924c"
conditions = ["c0_context", "c1_frame", "c2_fixed", "c3_random_matched", "c4_oracle"]
for condition in conditions:
    config = load_config(project / "configs/e3" / f"{condition}.yaml")
    attestation = _require_g0_attestation(config)
    manifest = _require_development_endpoint_manifest(config, attestation=attestation)
    assert attestation["dataset_source_sha256"] == expected_source
    assert attestation["full_declared_development"] is True
    assert attestation["test_accessed"] is False and not attestation["errors"]
    assert manifest["manifest_sha256"] == expected_endpoint
    rows = manifest["selected"]
    counts = Counter((r["task"], r["effective_lag"]) for r in rows)
    assert len(rows) == len({r["query_id"] for r in rows}) == 512
    assert len(counts) == 8 and set(counts.values()) == {64}
    assert config["training"]["max_steps"] == 1000
print(json.dumps({
    "passed": True,
    "conditions": conditions,
    "episodes": attestation["episode_count"],
    "queries": attestation["query_count"],
    "endpoint_queries": len(rows),
    "source_sha256": _source_tree_sha256(),
    "dataset_source_sha256": expected_source,
    "endpoint_manifest_sha256": expected_endpoint,
    "test_accessed": False,
}, indent=2))
