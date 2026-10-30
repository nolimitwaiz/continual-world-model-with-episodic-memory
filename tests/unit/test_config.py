from __future__ import annotations

from pathlib import Path

import pytest

from cmwm.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name",
    [
        "context_only",
        "frame_memory",
        "event_fixed",
        "event_random_matched",
        "event_prediction_error",
        "event_oracle",
        "event_learned",
        "smoke",
    ],
)
def test_prototype_configs_resolve_and_preserve_preregistered_invariants(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs/prototype1" / f"{name}.yaml")

    assert config["execution"]["require_slurm"] is True
    assert config["execution"]["require_cuda"] is True
    assert config["data"]["observation_source"] == "oracle_symbolic_state"
    assert config["prototype1"]["temporal"]["hidden_dim"] == 128
    assert config["evaluation"]["primary_metric"] == "long_delay_memory_score"


def test_config_rejects_local_training_policy(tmp_path: Path) -> None:
    config = (PROJECT_ROOT / "configs/defaults.yaml").read_text(encoding="utf-8")
    config = config.replace("require_slurm: true", "require_slurm: false")
    path = tmp_path / "unsafe.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ConfigError, match="Slurm"):
        load_config(path)


@pytest.mark.parametrize(
    "name",
    ["c0_context", "c1_frame", "c2_fixed", "c3_random_matched", "c4_oracle"],
)
def test_engineering_configs_match_g0_contract_and_checkpoint_frequently(
    name: str,
) -> None:
    config = load_config(PROJECT_ROOT / "configs/engineering" / f"{name}.yaml")

    assert config["data"]["train_episodes"] == 10_000
    assert config["data"]["validation_episodes"] == 1_000
    assert config["data"]["test_episodes"] == 1_000
    assert config["training"]["max_steps"] == 20
    assert config["training"]["eval_every_steps"] == 5
    assert config["training"]["checkpoint_every_steps"] == 5
    assert config["training"]["resume"] is True


@pytest.mark.parametrize(
    "name",
    ["c0_context", "c1_frame", "c2_fixed", "c3_random_matched", "c4_oracle"],
)
def test_development_pilot_configs_are_matched_and_not_claim_eligible(
    name: str,
) -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot" / f"{name}.yaml")

    assert config["experiment"]["run_class"] == "development"
    assert config["experiment"]["seed"] == 0
    assert config["scale"]["active_profile"] == "development_pilot_1000x32"
    assert config["scale"]["claim_eligible"] is False
    assert config["data"]["test_episodes"] == 1_000
    assert config["training"]["max_steps"] == 1_000
    assert config["training"]["batch_size"] == 32
    assert config["training"]["eval_every_steps"] == 100
    assert config["training"]["checkpoint_every_steps"] == 100
    assert config["training"]["resume"] is True

    condition = {
        "c0_context": ("none", "none"),
        "c1_frame": ("none", "frame"),
        "c2_fixed": ("fixed", "event"),
        "c3_random_matched": ("random_matched", "event"),
        "c4_oracle": ("oracle", "event"),
    }[name]
    assert config["prototype1"]["boundary"]["name"] == condition[0]
    assert config["prototype1"]["memory"]["name"] == condition[1]


def test_config_rejects_checkpoint_interval_after_run_end(tmp_path: Path) -> None:
    config = (PROJECT_ROOT / "configs/defaults.yaml").read_text(encoding="utf-8")
    config = config.replace("checkpoint_every_steps: 1000", "checkpoint_every_steps: 20001")
    path = tmp_path / "unsafe-checkpoint.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ConfigError, match="checkpoint_every_steps"):
        load_config(path)


@pytest.mark.parametrize(
    "name",
    ["c0_context", "c1_frame", "c2_fixed", "c3_random_matched", "c4_oracle"],
)
def test_replication_configs_remain_development_only(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs/replication" / f"{name}.yaml")

    assert config["experiment"]["run_class"] == "development"
    assert config["scale"]["active_profile"] == "development_replication_5seed_1000x32"
    assert config["scale"]["claim_eligible"] is False
    assert config["training"]["max_steps"] == 1_000
    assert config["training"]["batch_size"] == 32
