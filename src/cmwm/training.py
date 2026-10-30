"""Guarded cluster trainer for the first integrated symbolic CMWM experiment.

This module is executable only after the CLI's Slurm/CUDA guard passes and a
separate G0 dataset attestation is supplied.  It intentionally cannot turn the
current scaffold generator into a scientific run by accident.
"""

from __future__ import annotations

import json
import os
import random
import socket
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from cmwm.data import AnswerVocabulary, CMWMBatch, tensorize_examples
from cmwm.evaluation.endpoint_manifest import (
    validate_development_endpoint_manifest,
)
from cmwm.g0 import dataset_source_sha256
from cmwm.infrastructure.artifacts import atomic_write_json
from cmwm.infrastructure.reproducibility import seed_everything
from cmwm.models.cmwm import CMWM, CMWMLoss, CMWMOutput, MemoryCondition
from cmwm.schemas import DatasetSplit, Episode, QueryExample, QueryKind
from cmwm.synthetic.generator import SyntheticEpisodeGenerator
from cmwm.synthetic.manifest import content_sha256
from cmwm.synthetic.queries import effective_query_lag, generate_queries


class TrainingContractError(RuntimeError):
    """Raised when a run lacks a prerequisite scientific-control artifact."""


CHECKPOINT_SCHEMA_VERSION = "cmwm.training-checkpoint.v1"
ENGINEERING_INTERRUPT_ENV = "CMWM_ENGINEERING_INTERRUPT_AFTER_STEP"


class EngineeringInterruption(RuntimeError):
    """Intentional post-checkpoint stop used to validate exact cluster resume."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_tree_sha256() -> str:
    """Hash executable CMWM Python sources without including result artifacts."""

    project_root = Path(__file__).resolve().parents[2]
    digest = sha256()
    files = sorted((project_root / "src").rglob("*.py"))
    files.extend(
        sorted(
            path
            for path in (project_root / "scripts").glob("*.py")
            if path.is_file()
        )
    )
    for path in files:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace a checkpoint without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def record_training_failure(config: Mapping[str, Any], error: Exception) -> None:
    """Persist a terminal failure state without obscuring the original error."""

    run_root_value = os.environ.get(str(config["paths"]["run_root_env"]))
    if not run_root_value:
        return
    seed = int(config["experiment"]["seed"])
    status_path = (
        Path(run_root_value).expanduser().resolve()
        / str(config["experiment"]["name"])
        / f"seed-{seed}"
        / "run_status.json"
    )
    previous: dict[str, Any] = {}
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    previous.update(
        {
            "artifact_type": "cmwm_training_status",
            "state": "failed",
            "failed_at": _utc_now(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    )
    atomic_write_json(status_path, previous)


@dataclass(frozen=True)
class EvaluationSummary:
    examples: int
    state_examples: int
    causal_examples: int
    state_correct: int
    causal_correct: int
    retrieval_target_available: int
    boundary_true_positives: int
    boundary_false_positives: int
    boundary_false_negatives: int

    @property
    def state_accuracy(self) -> float:
        return self.state_correct / self.state_examples if self.state_examples else 0.0

    @property
    def causal_accuracy(self) -> float:
        return self.causal_correct / self.causal_examples if self.causal_examples else 0.0

    @property
    def long_delay_memory_score(self) -> float:
        correct = self.state_correct + self.causal_correct
        return correct / self.examples if self.examples else 0.0

    @property
    def retrieval_recall_at_four(self) -> float:
        return (
            self.retrieval_target_available / self.examples if self.examples else 0.0
        )

    @property
    def boundary_f1(self) -> float:
        denominator = (
            2 * self.boundary_true_positives
            + self.boundary_false_positives
            + self.boundary_false_negatives
        )
        return 2 * self.boundary_true_positives / denominator if denominator else 0.0

    def to_dict(
        self, *, condition: MemoryCondition | str | None = None
    ) -> dict[str, int | float | None]:
        values = asdict(self)
        values.update(
            {
                "state_accuracy": self.state_accuracy,
                "causal_accuracy": self.causal_accuracy,
                "long_delay_memory_score": self.long_delay_memory_score,
                "retrieval_recall_at_four": self.retrieval_recall_at_four,
                "boundary_f1": self.boundary_f1,
            }
        )
        if condition is not None:
            resolved = MemoryCondition(condition)
            if resolved is MemoryCondition.CONTEXT_ONLY:
                values["retrieval_recall_at_four"] = None
                values["retrieval_target_available"] = None
            if resolved not in (
                MemoryCondition.LEARNED_EVENTS,
                MemoryCondition.PREDICTION_ERROR,
            ):
                values["boundary_f1"] = None
                values["boundary_true_positives"] = None
                values["boundary_false_positives"] = None
                values["boundary_false_negatives"] = None
        return values


@dataclass(frozen=True)
class ValidationCohort:
    """A fixed engineering cohort reused for every checkpoint in one run."""

    seed: int
    sha256: str
    batches: tuple[tuple[tuple[Episode, QueryExample], ...], ...]
    manifest_sha256: str | None = None
    manifest_seal_sha256: str | None = None
    tasks: tuple[str, ...] = ()
    lags: tuple[int, ...] = ()
    examples_per_cell: int | None = None


def condition_from_config(config: Mapping[str, Any]) -> MemoryCondition:
    prototype = config["prototype1"]
    memory_name = prototype["memory"]["name"]
    boundary_name = prototype["boundary"]["name"]
    if memory_name == "none":
        return MemoryCondition.CONTEXT_ONLY
    if memory_name == "frame":
        return MemoryCondition.FRAME
    mapping = {
        "fixed": MemoryCondition.FIXED,
        "random_matched": MemoryCondition.RANDOM_MATCHED,
        "oracle": MemoryCondition.ORACLE_EVENTS,
        "learned": MemoryCondition.LEARNED_EVENTS,
        "prediction_error": MemoryCondition.PREDICTION_ERROR,
    }
    try:
        return mapping[boundary_name]
    except KeyError as error:
        raise TrainingContractError(
            f"unsupported memory/boundary configuration: {memory_name}/{boundary_name}"
        ) from error


def _require_g0_attestation(config: Mapping[str, Any]) -> dict[str, Any]:
    variable = str(config["paths"].get("g0_attestation_env", "CMWM_G0_ATTESTATION"))
    value = os.environ.get(variable)
    if not value:
        raise TrainingContractError(
            f"{variable} must point to a passing, sealed G0 attestation"
        )
    path = Path(value).expanduser().resolve()
    try:
        attestation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"cannot read G0 attestation: {path}") from error
    if attestation.get("artifact_type") != "cmwm_g0_attestation":
        raise TrainingContractError("G0 artifact has the wrong artifact_type")
    if attestation.get("passed") is not True:
        raise TrainingContractError("G0 attestation does not pass")
    if not attestation.get("manifest_sha256") or not attestation.get("seal_sha256"):
        raise TrainingContractError("G0 attestation is missing manifest/seal identities")
    seal = attestation.get("manifest_seal")
    if not isinstance(seal, dict):
        raise TrainingContractError("G0 attestation is missing its manifest seal")
    if seal.get("manifest_sha256") != attestation["manifest_sha256"]:
        raise TrainingContractError("G0 manifest identity and seal disagree")
    if content_sha256(seal) != attestation["seal_sha256"]:
        raise TrainingContractError("G0 manifest seal digest is invalid")
    if attestation.get("dataset_source_sha256") != dataset_source_sha256():
        raise TrainingContractError(
            "dataset-generating source changed after G0; regenerate the attestation"
        )
    expected = {
        "generator_policy": str(
            config["data"].get("generator_policy", "random_v1")
        ),
        "split_seed": int(config["data"]["split_seed"]),
        "train_episodes": int(config["data"]["train_episodes"]),
        "validation_episodes": int(config["data"]["validation_episodes"]),
        "test_episodes": int(config["data"]["test_episodes"]),
    }
    observed = attestation.get("dataset_contract")
    if observed != expected:
        raise TrainingContractError(
            "G0 attestation dataset contract does not match the training config"
        )
    return attestation


def _require_development_endpoint_manifest(
    config: Mapping[str, Any], *, attestation: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Load a sealed development endpoint; legacy E1 engineering runs remain supported."""

    policy = str(config["data"].get("generator_policy", "random_v1"))
    if policy not in {"exact_task_lag_v2", "confusable_events_v3"}:
        return None
    variable = str(
        config["paths"].get(
            "development_endpoint_manifest_env",
            "CMWM_DEVELOPMENT_ENDPOINT_MANIFEST",
        )
    )
    value = os.environ.get(variable)
    if not value:
        raise TrainingContractError(
            f"{variable} must point to the sealed development endpoint manifest"
        )
    path = Path(value).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(
            f"cannot read development endpoint manifest: {path}"
        ) from error
    try:
        validate_development_endpoint_manifest(
            manifest,
            expected_dataset_source_sha256=str(attestation["dataset_source_sha256"]),
            expected_generator_policy=policy,
        )
    except (TypeError, ValueError) as error:
        raise TrainingContractError(
            f"development endpoint manifest is invalid: {error}"
        ) from error
    return manifest


def _query_examples(
    episode: Episode, config: Mapping[str, Any]
) -> tuple[QueryExample, ...]:
    return generate_queries(episode, lags=config["data"]["query_lags"])


def _sample_examples(
    *,
    generator: SyntheticEpisodeGenerator,
    split: DatasetSplit,
    episode_count: int,
    batch_size: int,
    rng: random.Random,
    config: Mapping[str, Any],
) -> list[tuple[Episode, QueryExample]]:
    if episode_count <= 0:
        raise TrainingContractError(f"{split.value} episode count must be positive")
    selected: list[tuple[Episode, QueryExample]] = []
    query_kinds = tuple(QueryKind)
    attempts = 0
    while len(selected) < batch_size:
        attempts += 1
        if attempts > batch_size * 100:
            raise TrainingContractError("unable to construct a task-balanced batch")
        episode = generator.generate_episode(split, rng.randrange(episode_count))
        queries = _query_examples(episode, config)
        desired_kind = query_kinds[len(selected) % len(query_kinds)]
        eligible = [query for query in queries if query.query.kind is desired_kind]
        if not eligible:
            eligible = list(queries)
        if eligible:
            selected.append((episode, eligible[rng.randrange(len(eligible))]))
    return selected


def _model_from_config(config: Mapping[str, Any], answer_vocabulary: Any) -> CMWM:
    adapter = config["prototype1"]["state_adapter"]
    boundary = config["prototype1"]["boundary"]
    return CMWM(
        max_entities=int(adapter["max_entities"]),
        grid_width=int(adapter["grid_width"]),
        grid_height=int(adapter["grid_height"]),
        context_steps=int(config["data"]["context_steps"]),
        answer_vocabulary=answer_vocabulary,
        fixed_chunk_steps=int(boundary.get("window_steps", 8)),
        boundary_threshold=float(boundary.get("threshold") or 0.5),
    )


def _loss_dict(loss: CMWMLoss) -> dict[str, float]:
    return {
        "total": float(loss.total.detach()),
        "answer": float(loss.answer.detach()),
        "retrieval": float(loss.retrieval.detach()),
        "causal_pointer": float(loss.causal_pointer.detach()),
        "boundary": float(loss.boundary.detach()),
        "prediction": float(loss.prediction.detach()),
    }


def _batch_counts(
    *,
    model: CMWM,
    batch: CMWMBatch,
    condition: MemoryCondition,
    random_seed: int,
) -> EvaluationSummary:
    output = model(batch, condition=condition, random_seed=random_seed)
    return _counts_from_output(model=model, batch=batch, output=output)


def _counts_from_output(
    *, model: CMWM, batch: CMWMBatch, output: CMWMOutput
) -> EvaluationSummary:
    state_mask = batch.answer_targets >= 0
    causal_mask = batch.causal_source_steps >= 0
    state_predictions = output.answer_logits.argmax(dim=-1)
    candidate_predictions = output.candidate_logits.argmax(dim=-1)
    state_correct = int(
        ((state_predictions == batch.answer_targets) & state_mask).sum().item()
    )
    causal_correct = int(
        (
            (candidate_predictions == output.candidate_targets)
            & causal_mask
            & (output.candidate_targets >= 0)
        )
        .sum()
        .item()
    )
    valid_boundaries = batch.sequence.step_mask.clone()
    valid_boundaries[:, 0] = False
    predicted_boundaries = output.boundary_probabilities >= model.boundary_threshold
    truth = batch.oracle_boundaries
    true_positives = int((predicted_boundaries & truth & valid_boundaries).sum().item())
    false_positives = int(
        (predicted_boundaries & ~truth & valid_boundaries).sum().item()
    )
    false_negatives = int(
        (~predicted_boundaries & truth & valid_boundaries).sum().item()
    )
    return EvaluationSummary(
        examples=batch.batch_size,
        state_examples=int(state_mask.sum().item()),
        causal_examples=int(causal_mask.sum().item()),
        state_correct=state_correct,
        causal_correct=causal_correct,
        retrieval_target_available=int((output.candidate_targets >= 0).sum().item()),
        boundary_true_positives=true_positives,
        boundary_false_positives=false_positives,
        boundary_false_negatives=false_negatives,
    )


def _prediction_rows(
    *,
    pairs: Sequence[tuple[Episode, QueryExample]],
    batch: CMWMBatch,
    output: CMWMOutput,
) -> list[dict[str, Any]]:
    """Emit one auditable prediction row for every frozen endpoint query."""

    state_predictions = output.answer_logits.argmax(dim=-1)
    candidate_predictions = output.candidate_logits.argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    for index, (episode, example) in enumerate(pairs):
        is_causal = int(batch.causal_source_steps[index]) >= 0
        if is_causal:
            target = int(output.candidate_targets[index])
            prediction = int(candidate_predictions[index])
            target_available = target >= 0
            correct = target_available and prediction == target
            prediction_type = "retrieved_record_index"
        else:
            target = int(batch.answer_targets[index])
            prediction = int(state_predictions[index])
            target_available = True
            correct = prediction == target
            prediction_type = "answer_vocabulary_index"
        rows.append(
            {
                "occurrence_id": example.query.query_id,
                "episode_id": episode.episode_id,
                "query_id": example.query.query_id,
                "task": example.query.kind.value,
                "effective_lag": effective_query_lag(episode, example),
                "prediction_type": prediction_type,
                "prediction": prediction,
                "target": target,
                "target_available": target_available,
                "correct": correct,
            }
        )
    return rows


def _prediction_metrics(
    rows: Sequence[Mapping[str, Any]], *, cohort: ValidationCohort
) -> dict[str, Any]:
    """Compute the primary equal-weight task-by-lag macro endpoint."""

    grouped: dict[tuple[str, int], list[bool]] = defaultdict(list)
    query_ids: list[str] = []
    for row in rows:
        query_ids.append(str(row["query_id"]))
        grouped[(str(row["task"]), int(row["effective_lag"]))].append(
            bool(row["correct"])
        )
    if len(query_ids) != len(set(query_ids)):
        raise TrainingContractError("endpoint predictions contain duplicate query IDs")
    required_cells = {
        (task, lag) for task in cohort.tasks for lag in cohort.lags
    }
    if required_cells and set(grouped) != required_cells:
        raise TrainingContractError(
            "endpoint predictions do not cover the frozen task-by-lag cells"
        )
    if cohort.examples_per_cell is not None and any(
        len(grouped[cell]) != cohort.examples_per_cell for cell in required_cells
    ):
        raise TrainingContractError(
            "endpoint prediction cells do not match examples_per_cell"
        )
    cell_accuracy = {
        f"{task}@{lag}": sum(grouped[(task, lag)]) / len(grouped[(task, lag)])
        for task, lag in sorted(grouped)
    }
    macro = (
        sum(cell_accuracy.values()) / len(cell_accuracy) if cell_accuracy else 0.0
    )
    pooled = sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0
    return {
        "task_lag_accuracy": cell_accuracy,
        "macro_task_lag_accuracy": macro,
        "pooled_accuracy_from_predictions": pooled,
        "paired_prediction_count": len(rows),
        "prediction_identity_sha256": content_sha256(list(rows)),
    }


def _combine_summaries(items: Sequence[EvaluationSummary]) -> EvaluationSummary:
    if not items:
        return EvaluationSummary(*(0 for _ in range(9)))
    values = [asdict(item) for item in items]
    return EvaluationSummary(
        **{key: sum(int(value[key]) for value in values) for key in values[0]}
    )


def _build_validation_cohort(
    *,
    generator: SyntheticEpisodeGenerator,
    config: Mapping[str, Any],
    seed: int,
) -> ValidationCohort:
    """Select and identify the bounded validation examples exactly once."""

    rng = random.Random(seed)
    episode_count = int(config["data"]["validation_episodes"])
    batches = min(16, max(1, episode_count))
    batch_size = min(int(config["training"]["batch_size"]), 64)
    selected_batches: list[tuple[tuple[Episode, QueryExample], ...]] = []
    identity: list[dict[str, str]] = []
    for _ in range(batches):
        pairs = tuple(
            _sample_examples(
                generator=generator,
                split=DatasetSplit.VALIDATION,
                episode_count=episode_count,
                batch_size=batch_size,
                rng=rng,
                config=config,
            )
        )
        selected_batches.append(pairs)
        identity.extend(
            {
                "episode_id": episode.episode_id,
                "query_id": example.query.query_id,
            }
            for episode, example in pairs
        )
    return ValidationCohort(
        seed=seed,
        sha256=content_sha256(identity),
        batches=tuple(selected_batches),
    )


def _build_manifest_validation_cohort(
    *,
    generator: SyntheticEpisodeGenerator,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> ValidationCohort:
    """Reconstruct every sealed episode/query pair and verify its oracle identity."""

    selected = manifest["selected"]
    if not isinstance(selected, list):
        raise TrainingContractError("endpoint selected rows must be a list")
    episode_cache: dict[int, Episode] = {}
    pairs: list[tuple[Episode, QueryExample]] = []
    for row in selected:
        if not isinstance(row, dict):
            raise TrainingContractError("endpoint selected row must be a mapping")
        episode_index = int(row["episode_index"])
        episode = episode_cache.get(episode_index)
        if episode is None:
            episode = generator.generate_episode(
                DatasetSplit.VALIDATION, episode_index
            )
            episode_cache[episode_index] = episode
        if episode.episode_id != row["episode_id"]:
            raise TrainingContractError(
                f"endpoint episode identity changed at index {episode_index}"
            )
        matching = [
            example
            for example in _query_examples(episode, config)
            if example.query.query_id == row["query_id"]
        ]
        if len(matching) != 1:
            raise TrainingContractError(
                f"endpoint query cannot be reconstructed exactly: {row['query_id']}"
            )
        example = matching[0]
        observed = {
            "task": example.query.kind.value,
            "effective_lag": effective_query_lag(episode, example),
            "oracle_answer_sha256": content_sha256(example.oracle_answer),
        }
        expected = {key: row[key] for key in observed}
        if observed != expected:
            raise TrainingContractError(
                f"endpoint query content changed for {row['query_id']}: "
                f"observed={observed}, expected={expected}"
            )
        pairs.append((episode, example))

    batch_size = min(int(config["training"]["batch_size"]), 64)
    batches = tuple(
        tuple(pairs[start : start + batch_size])
        for start in range(0, len(pairs), batch_size)
    )
    return ValidationCohort(
        seed=0,
        sha256=str(manifest["selected_query_identity_sha256"]),
        batches=batches,
        manifest_sha256=str(manifest["manifest_sha256"]),
        manifest_seal_sha256=str(manifest["seal_sha256"]),
        tasks=tuple(str(task) for task in manifest["tasks"]),
        lags=tuple(int(lag) for lag in manifest["lags"]),
        examples_per_cell=int(manifest["examples_per_cell"]),
    )


def _engineering_interrupt_step(
    config: Mapping[str, Any], *, checkpoint_every: int, max_steps: int
) -> int | None:
    """Resolve an intentional, checkpoint-aligned engineering interruption."""

    raw = os.environ.get(ENGINEERING_INTERRUPT_ENV)
    if raw is None:
        return None
    if config["experiment"]["run_class"] != "engineering":
        raise TrainingContractError(
            f"{ENGINEERING_INTERRUPT_ENV} is permitted only for engineering runs"
        )
    try:
        step = int(raw)
    except ValueError as error:
        raise TrainingContractError(
            f"{ENGINEERING_INTERRUPT_ENV} must be an integer"
        ) from error
    if step <= 0 or step >= max_steps or step % checkpoint_every != 0:
        raise TrainingContractError(
            f"{ENGINEERING_INTERRUPT_ENV} must be a positive checkpoint step "
            "strictly before max_steps"
        )
    return step


@torch.no_grad()
def _evaluate(
    *,
    model: CMWM,
    cohort: ValidationCohort,
    condition: MemoryCondition,
    device: torch.device,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    summaries: list[EvaluationSummary] = []
    predictions: list[dict[str, Any]] = []
    for batch_index, pairs in enumerate(cohort.batches):
        batch, _ = tensorize_examples(
            pairs,
            max_entities=model.max_entities,
            device=device,
        )
        output = model(
            batch,
            condition=condition,
            random_seed=random_seed + batch_index,
        )
        summaries.append(_counts_from_output(model=model, batch=batch, output=output))
        predictions.extend(_prediction_rows(pairs=pairs, batch=batch, output=output))
    model.train()
    summary = _combine_summaries(summaries).to_dict(condition=condition)
    prediction_metrics = _prediction_metrics(predictions, cohort=cohort)
    summary["pooled_long_delay_memory_score"] = summary[
        "long_delay_memory_score"
    ]
    summary["long_delay_memory_score"] = prediction_metrics[
        "macro_task_lag_accuracy"
    ]
    summary.update(prediction_metrics)
    return summary, predictions


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    """Train one guarded condition with durable checkpoints and exact resume."""

    attestation = _require_g0_attestation(config)
    endpoint_manifest = _require_development_endpoint_manifest(
        config, attestation=attestation
    )
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, deterministic=bool(config["execution"]["deterministic"]))
    device = torch.device("cuda")
    condition = condition_from_config(config)
    config_sha256 = content_sha256(config)
    source_sha256 = _source_tree_sha256()
    run_root_value = os.environ.get(str(config["paths"]["run_root_env"]))
    if not run_root_value:
        raise TrainingContractError("run-root environment variable is missing")
    run_directory = (
        Path(run_root_value).expanduser().resolve()
        / str(config["experiment"]["name"])
        / f"seed-{seed}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    last_checkpoint = run_directory / "last.pt"
    metrics_path = run_directory / "metrics_history.json"
    checkpoint_index_path = run_directory / "checkpoint_index.json"
    status_path = run_directory / "run_status.json"
    result_path = run_directory / "result.json"
    atomic_write_json(run_directory / "resolved_config.json", config)

    generator = SyntheticEpisodeGenerator(
        base_seed=int(config["data"]["split_seed"]),
        episode_steps=int(config["data"]["max_episode_steps"]),
        schedule_policy=str(config["data"].get("generator_policy", "random_v1")),
    )
    if endpoint_manifest is None:
        validation_cohort = _build_validation_cohort(
            generator=generator,
            config=config,
            seed=seed + 100_000,
        )
    else:
        validation_cohort = _build_manifest_validation_cohort(
            generator=generator,
            config=config,
            manifest=endpoint_manifest,
        )
    rng = random.Random(seed)
    adapter = config["prototype1"]["state_adapter"]
    answer_vocabulary = AnswerVocabulary(
        grid_width=int(adapter["grid_width"]),
        grid_height=int(adapter["grid_height"]),
        max_entities=int(adapter["max_entities"]),
    )
    model = _model_from_config(config, answer_vocabulary).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    max_steps = int(config["training"]["max_steps"])
    eval_every = int(config["training"]["eval_every_steps"])
    checkpoint_every = int(config["training"]["checkpoint_every_steps"])
    interrupt_after_step = _engineering_interrupt_step(
        config, checkpoint_every=checkpoint_every, max_steps=max_steps
    )
    gradient_clip = float(config["training"]["gradient_clip_norm"])
    resume = bool(config["training"].get("resume", False))
    save_best = bool(config.get("artifacts", {}).get("save_best_checkpoint", True))
    save_predictions = bool(
        config.get("artifacts", {}).get("save_predictions", True)
    )
    history: list[dict[str, Any]] = []
    checkpoint_index: list[dict[str, Any]] = []
    latest_loss: dict[str, float] = {}
    latest_evaluation: dict[str, int | float | None] = {}
    best_score = float("-inf")
    best_step: int | None = None
    start_step = 0
    elapsed_before = 0.0
    previous_peak_memory = 0
    resumed_from: str | None = None
    latest_predictions_path: str | None = None

    if last_checkpoint.exists():
        if not resume:
            raise TrainingContractError(
                f"checkpoint exists but training.resume is false: {last_checkpoint}"
            )
        payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "condition": condition.value,
            "seed": seed,
            "config_sha256": config_sha256,
            "source_sha256": source_sha256,
            "g0_manifest_sha256": attestation["manifest_sha256"],
            "endpoint_manifest_sha256": validation_cohort.manifest_sha256,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise TrainingContractError(
                f"checkpoint identity does not match this run: {mismatches}"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        rng.setstate(payload["sampler_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_state_all"]]
        )
        start_step = int(payload["step"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        previous_peak_memory = int(payload.get("peak_cuda_memory_bytes", 0))
        history = list(payload.get("metrics_history", []))
        checkpoint_index = list(payload.get("checkpoint_index", []))
        latest_loss = dict(payload.get("latest_loss", {}))
        latest_evaluation = dict(payload.get("latest_evaluation", {}))
        best_score = float(payload.get("best_score", float("-inf")))
        best_step = payload.get("best_step")
        latest_predictions_path = payload.get("latest_predictions_path")
        resumed_from = str(last_checkpoint)

    if start_step > max_steps:
        raise TrainingContractError(
            f"checkpoint step {start_step} exceeds configured max_steps {max_steps}"
        )
    if start_step == max_steps and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    torch.cuda.reset_peak_memory_stats(device)
    segment_started = time.perf_counter()
    run_started_at = _utc_now()
    common_identity = {
        "experiment": str(config["experiment"]["name"]),
        "condition": condition.value,
        "seed": seed,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "dataset_source_sha256": attestation["dataset_source_sha256"],
        "g0_manifest_sha256": attestation["manifest_sha256"],
        "g0_seal_sha256": attestation["seal_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "validation_cohort_seed": validation_cohort.seed,
        "validation_cohort_sha256": validation_cohort.sha256,
        "endpoint_manifest_sha256": validation_cohort.manifest_sha256,
        "endpoint_manifest_seal_sha256": validation_cohort.manifest_seal_sha256,
    }
    atomic_write_json(
        status_path,
        {
            "artifact_type": "cmwm_training_status",
            "state": "running",
            "started_at": run_started_at,
            "resumed_from": resumed_from,
            "start_step": start_step,
            "max_steps": max_steps,
            **common_identity,
        },
    )

    def elapsed_seconds() -> float:
        return elapsed_before + (time.perf_counter() - segment_started)

    def checkpoint_payload(step: int) -> dict[str, Any]:
        torch.cuda.synchronize(device)
        peak_memory = max(
            previous_peak_memory, int(torch.cuda.max_memory_allocated(device))
        )
        return {
            "artifact_type": "cmwm_training_checkpoint",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "max_steps": max_steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "elapsed_seconds": elapsed_seconds(),
            "peak_cuda_memory_bytes": peak_memory,
            "metrics_history": history,
            "checkpoint_index": checkpoint_index,
            "latest_loss": latest_loss,
            "latest_evaluation": latest_evaluation,
            "best_score": best_score,
            "best_step": best_step,
            "latest_predictions_path": latest_predictions_path,
            **common_identity,
        }

    for step in range(start_step + 1, max_steps + 1):
        pairs = _sample_examples(
            generator=generator,
            split=DatasetSplit.TRAIN,
            episode_count=int(config["data"]["train_episodes"]),
            batch_size=int(config["training"]["batch_size"]),
            rng=rng,
            config=config,
        )
        batch, _ = tensorize_examples(
            pairs,
            max_entities=model.max_entities,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, condition=condition, random_seed=seed + step)
            loss = model.loss(output, batch, condition=condition)
        if not torch.isfinite(loss.total):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        )
        optimizer.step()
        latest_loss = _loss_dict(loss)
        evaluation_due = step % eval_every == 0 or step == max_steps
        if evaluation_due:
            latest_evaluation, predictions = _evaluate(
                model=model,
                cohort=validation_cohort,
                condition=condition,
                device=device,
                random_seed=seed + 100_000,
            )
            if save_predictions:
                predictions_path = (
                    run_directory
                    / "predictions"
                    / f"step-{step:08d}.json"
                )
                atomic_write_json(
                    predictions_path,
                    {
                        "artifact_type": "cmwm_endpoint_predictions_v1",
                        "claim_eligible": bool(config["scale"]["claim_eligible"]),
                        "step": step,
                        "condition": condition.value,
                        "seed": seed,
                        "endpoint_manifest_sha256": validation_cohort.manifest_sha256,
                        "endpoint_manifest_seal_sha256": (
                            validation_cohort.manifest_seal_sha256
                        ),
                        "validation_cohort_sha256": validation_cohort.sha256,
                        "summary": latest_evaluation,
                        "predictions": predictions,
                    },
                )
                latest_predictions_path = str(predictions_path)
            score = float(latest_evaluation["long_delay_memory_score"])
            row = {
                "step": step,
                "elapsed_seconds": elapsed_seconds(),
                "gradient_norm_before_clip": gradient_norm,
                "loss": latest_loss,
                "validation": latest_evaluation,
                "predictions": latest_predictions_path,
            }
            history.append(row)
            atomic_write_json(metrics_path, history)
            if score > best_score:
                best_score = score
                best_step = step
                if save_best:
                    _atomic_torch_save(run_directory / "best.pt", checkpoint_payload(step))

        checkpoint_due = step % checkpoint_every == 0 or step == max_steps
        if checkpoint_due:
            named_checkpoint = checkpoint_directory / f"step-{step:08d}.pt"
            checkpoint_index.append(
                {
                    "step": step,
                    "path": str(named_checkpoint),
                    "elapsed_seconds": elapsed_seconds(),
                    "latest_validation": latest_evaluation,
                }
            )
            payload = checkpoint_payload(step)
            _atomic_torch_save(named_checkpoint, payload)
            _atomic_torch_save(last_checkpoint, payload)
            atomic_write_json(checkpoint_index_path, checkpoint_index)
            if interrupt_after_step == step:
                atomic_write_json(
                    status_path,
                    {
                        "artifact_type": "cmwm_training_status",
                        "state": "interrupted_for_resume_test",
                        "interrupted_at": _utc_now(),
                        "step": step,
                        "checkpoint": str(last_checkpoint),
                        **common_identity,
                    },
                )
                raise EngineeringInterruption(
                    f"intentional engineering interruption after checkpoint {step}"
                )

    torch.cuda.synchronize(device)
    total_elapsed = elapsed_seconds()
    peak_memory = max(
        previous_peak_memory, int(torch.cuda.max_memory_allocated(device))
    )
    result: dict[str, Any] = {
        "artifact_type": "cmwm_engineering_training_run",
        "claim_eligible": bool(config["scale"]["claim_eligible"]),
        "steps": max_steps,
        "loss": latest_loss,
        "validation": latest_evaluation,
        "best_validation_long_delay_memory_score": best_score,
        "best_step": best_step,
        "final_predictions": latest_predictions_path,
        "elapsed_seconds": total_elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "checkpoint": str(last_checkpoint),
        "checkpoint_count": len(checkpoint_index),
        "metrics_history": str(metrics_path),
        "completed_at": _utc_now(),
        **common_identity,
    }
    atomic_write_json(result_path, result)
    atomic_write_json(
        status_path,
        {
            "artifact_type": "cmwm_training_status",
            "state": "completed",
            "completed_at": result["completed_at"],
            "steps": max_steps,
            "elapsed_seconds": total_elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            **common_identity,
        },
    )
    return result
