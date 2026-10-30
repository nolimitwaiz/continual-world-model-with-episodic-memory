"""Condition-blind construction of a balanced development endpoint manifest."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

from cmwm.synthetic.manifest import content_sha256


ENDPOINT_MANIFEST_SCHEMA = "cmwm.development-endpoint-manifest.v1"
ENDPOINT_SELECTION_SALT = "cmwm-e2-development-endpoint-v1"
ENDPOINT_ARTIFACT_TYPE = "cmwm_development_endpoint_manifest_v1"


@dataclass(frozen=True)
class EndpointCandidate:
    episode_index: int
    episode_id: str
    query_id: str
    task: str
    effective_lag: int
    oracle_answer_sha256: str


def _selection_rank(candidate: EndpointCandidate, *, salt: str) -> str:
    identity = (
        f"{salt}\0{candidate.episode_id}\0{candidate.query_id}\0"
        f"{candidate.task}\0{candidate.effective_lag}"
    )
    return sha256(identity.encode("utf-8")).hexdigest()


def build_balanced_endpoint_manifest(
    candidates: Iterable[EndpointCandidate],
    *,
    tasks: Sequence[str],
    lags: Sequence[int],
    examples_per_cell: int,
    dataset_source_sha256: str,
    config_sha256: str,
    generator_policy: str,
    salt: str = ENDPOINT_SELECTION_SALT,
) -> dict[str, object]:
    """Select an exact balanced cohort without labels or model predictions."""

    if examples_per_cell <= 0:
        raise ValueError("examples_per_cell must be positive")
    materialized = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.task,
                candidate.effective_lag,
                candidate.episode_id,
                candidate.query_id,
            ),
        )
    )
    query_ids = [candidate.query_id for candidate in materialized]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("endpoint candidates contain duplicate query IDs")

    grouped: dict[tuple[str, int], list[EndpointCandidate]] = {
        (task, int(lag)): [] for task in tasks for lag in lags
    }
    for candidate in materialized:
        key = (candidate.task, candidate.effective_lag)
        if key in grouped:
            grouped[key].append(candidate)

    candidate_counts = {
        f"{task}@{lag}": len(grouped[(task, int(lag))])
        for task in tasks
        for lag in lags
    }
    insufficient = {
        key: count
        for key, count in candidate_counts.items()
        if count < examples_per_cell
    }
    if insufficient:
        raise ValueError(f"insufficient endpoint cells: {insufficient}")

    selected: list[dict[str, object]] = []
    for task in tasks:
        for lag_value in lags:
            lag = int(lag_value)
            ranked = sorted(
                grouped[(task, lag)],
                key=lambda candidate: (
                    _selection_rank(candidate, salt=salt),
                    candidate.episode_id,
                    candidate.query_id,
                ),
            )
            for candidate in ranked[:examples_per_cell]:
                selected.append(
                    {
                        **asdict(candidate),
                        "selection_rank_sha256": _selection_rank(
                            candidate, salt=salt
                        ),
                    }
                )

    selected_query_ids = [str(row["query_id"]) for row in selected]
    if len(set(selected_query_ids)) != len(selected_query_ids):
        raise ValueError("balanced endpoint selected a query more than once")
    selected_counts = Counter(
        (str(row["task"]), int(row["effective_lag"])) for row in selected
    )
    expected = examples_per_cell
    if any(
        selected_counts[(task, int(lag))] != expected
        for task in tasks
        for lag in lags
    ):
        raise AssertionError("balanced endpoint construction lost cell balance")

    core: dict[str, object] = {
        "schema_version": ENDPOINT_MANIFEST_SCHEMA,
        "claim_eligible": False,
        "split": "validation",
        "test_accessed": False,
        "generator_policy": generator_policy,
        "dataset_source_sha256": dataset_source_sha256,
        "config_sha256": config_sha256,
        "selection_algorithm": "sha256_rank_without_replacement_v1",
        "selection_salt": salt,
        "tasks": list(tasks),
        "lags": [int(lag) for lag in lags],
        "examples_per_cell": examples_per_cell,
        "candidate_counts": candidate_counts,
        "candidate_identity_sha256": content_sha256(
            [asdict(candidate) for candidate in materialized]
        ),
        "selected_count": len(selected),
        "selected_query_identity_sha256": content_sha256(selected),
        "selected": selected,
    }
    manifest_sha256 = content_sha256(core)
    return {
        "artifact_type": ENDPOINT_ARTIFACT_TYPE,
        **core,
        "manifest_sha256": manifest_sha256,
        "seal_sha256": content_sha256(
            {
                "schema_version": ENDPOINT_MANIFEST_SCHEMA,
                "manifest_sha256": manifest_sha256,
                "selected_query_identity_sha256": core[
                    "selected_query_identity_sha256"
                ],
            }
        ),
    }


def validate_development_endpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_dataset_source_sha256: str,
    expected_generator_policy: str,
) -> None:
    """Fail closed when a frozen endpoint artifact is incomplete or altered."""

    if manifest.get("artifact_type") != ENDPOINT_ARTIFACT_TYPE:
        raise ValueError("endpoint manifest has the wrong artifact_type")
    if manifest.get("schema_version") != ENDPOINT_MANIFEST_SCHEMA:
        raise ValueError("endpoint manifest has the wrong schema version")
    if manifest.get("claim_eligible") is not False:
        raise ValueError("development endpoint must not be claim eligible")
    if manifest.get("split") != "validation" or manifest.get("test_accessed") is not False:
        raise ValueError("development endpoint must use validation without test access")
    if manifest.get("dataset_source_sha256") != expected_dataset_source_sha256:
        raise ValueError("endpoint dataset source does not match the current generator")
    if manifest.get("generator_policy") != expected_generator_policy:
        raise ValueError("endpoint generator policy does not match the run")

    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"artifact_type", "manifest_sha256", "seal_sha256"}
    }
    observed_manifest_sha256 = manifest.get("manifest_sha256")
    if not isinstance(observed_manifest_sha256, str) or content_sha256(core) != observed_manifest_sha256:
        raise ValueError("endpoint manifest digest is invalid")
    expected_seal = content_sha256(
        {
            "schema_version": ENDPOINT_MANIFEST_SCHEMA,
            "manifest_sha256": observed_manifest_sha256,
            "selected_query_identity_sha256": manifest.get(
                "selected_query_identity_sha256"
            ),
        }
    )
    if manifest.get("seal_sha256") != expected_seal:
        raise ValueError("endpoint manifest seal is invalid")

    selected = manifest.get("selected")
    tasks = manifest.get("tasks")
    lags = manifest.get("lags")
    examples_per_cell = manifest.get("examples_per_cell")
    if not isinstance(selected, list) or not isinstance(tasks, list) or not isinstance(lags, list):
        raise ValueError("endpoint manifest is missing selected rows or strata")
    if not isinstance(examples_per_cell, int) or examples_per_cell <= 0:
        raise ValueError("endpoint examples_per_cell must be positive")
    if manifest.get("selected_count") != len(selected):
        raise ValueError("endpoint selected_count does not match selected rows")
    if content_sha256(selected) != manifest.get("selected_query_identity_sha256"):
        raise ValueError("endpoint selected-query identity is invalid")

    query_ids: list[str] = []
    counts: Counter[tuple[str, int]] = Counter()
    salt = manifest.get("selection_salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("endpoint selection salt is missing")
    required = {
        "episode_index",
        "episode_id",
        "query_id",
        "task",
        "effective_lag",
        "oracle_answer_sha256",
        "selection_rank_sha256",
    }
    for row in selected:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("endpoint selected row is malformed")
        candidate = EndpointCandidate(
            episode_index=int(row["episode_index"]),
            episode_id=str(row["episode_id"]),
            query_id=str(row["query_id"]),
            task=str(row["task"]),
            effective_lag=int(row["effective_lag"]),
            oracle_answer_sha256=str(row["oracle_answer_sha256"]),
        )
        if row["selection_rank_sha256"] != _selection_rank(candidate, salt=salt):
            raise ValueError("endpoint selected row has an invalid selection rank")
        query_ids.append(candidate.query_id)
        counts[(candidate.task, candidate.effective_lag)] += 1
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("endpoint selected rows contain duplicate query IDs")
    expected_cells = {
        (str(task), int(lag)) for task in tasks for lag in lags
    }
    if set(counts) != expected_cells or any(
        counts[cell] != examples_per_cell for cell in expected_cells
    ):
        raise ValueError("endpoint selected rows are not exactly cell balanced")
