"""Small configuration loader used by local-safe validation and cluster jobs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration violates the Prototype 1 contract."""


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config and recursively resolve a relative ``extends`` field."""

    config_path = Path(path).expanduser().resolve()
    if config_path in _seen:
        raise ConfigError(f"configuration inheritance cycle at {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"configuration does not exist: {config_path}")
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ConfigError("configuration root must be a mapping")
    parent = parsed.pop("extends", None)
    if parent is None:
        resolved = parsed
    else:
        parent_path = (config_path.parent / str(parent)).resolve()
        resolved = _merge(load_config(parent_path, _seen | {config_path}), parsed)
    validate_config(resolved)
    return resolved


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate invariants that must hold before any job can be submitted."""

    required_sections = {
        "experiment",
        "execution",
        "paths",
        "data",
        "prototype1",
        "training",
        "evaluation",
        "gates",
    }
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ConfigError(f"missing configuration sections: {', '.join(missing)}")

    execution = config["execution"]
    if not execution.get("require_slurm") or not execution.get("require_cuda"):
        raise ConfigError("Prototype 1 training must require both Slurm and CUDA")
    if execution.get("device") != "cuda":
        raise ConfigError("Prototype 1 experiment device must be cuda")

    paths = config["paths"]
    for key in (
        "data_root_env",
        "run_root_env",
        "g0_attestation_env",
        "development_endpoint_manifest_env",
    ):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ConfigError(f"paths.{key} must name a non-empty environment variable")

    data = config["data"]
    if data.get("observation_source") != "oracle_symbolic_state":
        raise ConfigError("Prototype 1A must use oracle_symbolic_state")
    if data.get("generator_policy", "random_v1") not in {
        "random_v1",
        "exact_task_lag_v2",
        "confusable_events_v3",
    }:
        raise ConfigError("unknown synthetic generator policy")
    if int(data.get("context_steps", 0)) <= 0:
        raise ConfigError("context_steps must be positive")
    lags = data.get("query_lags")
    if not isinstance(lags, list) or not lags or any(int(lag) <= 0 for lag in lags):
        raise ConfigError("query_lags must be a non-empty list of positive integers")

    prototype = config["prototype1"]
    memory = prototype.get("memory", {})
    if int(memory.get("capacity_records", 0)) != 64:
        raise ConfigError("Prototype 1 memory capacity must remain 64 records")
    if int(memory.get("retrieval_top_k", 0)) != 4:
        raise ConfigError("Prototype 1 retrieval_top_k must remain 4")
    if not memory.get("metadata_bytes_counted"):
        raise ConfigError("memory metadata must be included in byte accounting")
    layout = memory.get("record_layout", {})
    expected_layout = {
        "vector_bytes": 1024,
        "metadata_bytes": 40,
        "record_bytes": 1064,
        "capacity_bytes": 68096,
    }
    if layout != expected_layout:
        raise ConfigError("Prototype 1 memory must use the fixed byte-accounting layout")
    if prototype.get("temporal", {}).get("name") != "gru":
        raise ConfigError("Prototype 1 primary temporal model must remain GRU")
    if int(prototype.get("temporal", {}).get("hidden_dim", 0)) != 128:
        raise ConfigError("Prototype 1 GRU hidden_dim must remain 128")
    boundary_name = prototype.get("boundary", {}).get("name")
    memory_name = memory.get("name")
    allowed_boundaries = {
        "none",
        "fixed",
        "random_matched",
        "oracle",
        "learned",
        "prediction_error",
    }
    if boundary_name not in allowed_boundaries:
        raise ConfigError("unknown Prototype 1 boundary condition")
    if memory_name not in {"none", "frame", "event"}:
        raise ConfigError("unknown Prototype 1 memory condition")
    if memory_name in {"none", "frame"} and boundary_name != "none":
        raise ConfigError("context/frame conditions cannot enable event boundaries")
    if memory_name == "event" and boundary_name == "none":
        raise ConfigError("event memory requires a declared boundary policy")

    training = config["training"]
    for key in (
        "max_steps",
        "batch_size",
        "eval_every_steps",
        "checkpoint_every_steps",
    ):
        if int(training.get(key, 0)) <= 0:
            raise ConfigError(f"training.{key} must be positive")
    if not isinstance(training.get("resume"), bool):
        raise ConfigError("training.resume must be boolean")
    if int(training["eval_every_steps"]) > int(training["max_steps"]):
        raise ConfigError("training.eval_every_steps cannot exceed max_steps")
    if int(training["checkpoint_every_steps"]) > int(training["max_steps"]):
        raise ConfigError("training.checkpoint_every_steps cannot exceed max_steps")

    evaluation = config["evaluation"]
    if evaluation.get("primary_metric") != "long_delay_memory_score":
        raise ConfigError("the preregistered primary metric is long_delay_memory_score")

    run_class = config["experiment"].get("run_class")
    if run_class not in {"engineering", "development", "confirmatory"}:
        raise ConfigError("experiment.run_class must declare an allowed evidence class")
    if run_class != "confirmatory" and config.get("scale", {}).get("claim_eligible") is not False:
        raise ConfigError("non-confirmatory runs must set scale.claim_eligible to false")


def dump_config(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=True)
