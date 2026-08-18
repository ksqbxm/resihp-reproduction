"""Tests for configuration and fail-stop event validation."""

import json
from pathlib import Path

import pytest

from resihp.config import ConfigError, load_config


VALID_CONFIG = {
    "model_dim": 128,
    "num_layers": 4,
    "num_heads": 8,
    "batch_size": 8,
    "micro_batch_size": 2,
    "seed": 1234,
    "tp": 2,
    "pp": 2,
    "dp": 2,
    "iterations": 6,
}
VALID_FAILURES = {
    "events": [
        {"after_iteration": 2, "failed_rank": 1},
        {"after_iteration": 4, "failed_rank": 5},
    ]
}


def write_inputs(tmp_path, config=None, failures=None):
    config_path = tmp_path / "train.json"
    failures_path = tmp_path / "failures.json"
    config_path.write_text(json.dumps(config or VALID_CONFIG), encoding="utf-8")
    failures_path.write_text(json.dumps(failures or VALID_FAILURES), encoding="utf-8")
    return config_path, failures_path


def test_valid_files_load(tmp_path):
    config_path, failures_path = write_inputs(tmp_path)

    parsed = load_config(config_path, failures_path)

    assert parsed.train.model_dim == 128
    assert parsed.train.world_size == 8
    assert parsed.failures[1].after_iteration == 4


@pytest.mark.parametrize("missing", ["model_dim", "tp", "iterations"])
def test_missing_train_field_is_rejected(tmp_path, missing):
    config = {key: value for key, value in VALID_CONFIG.items() if key != missing}
    config_path, failures_path = write_inputs(tmp_path, config=config)

    with pytest.raises(ConfigError, match=missing):
        load_config(config_path, failures_path)


def test_failure_iterations_must_increase(tmp_path):
    failures = {
        "events": [
            {"after_iteration": 2, "failed_rank": 1},
            {"after_iteration": 2, "failed_rank": 5},
        ]
    }
    config_path, failures_path = write_inputs(tmp_path, failures=failures)

    with pytest.raises(ConfigError, match="after_iteration"):
        load_config(config_path, failures_path)


@pytest.mark.parametrize(
    "events, message",
    [
        ([{"after_iteration": 2, "failed_rank": 8}], "failed_rank"),
        ([{"after_iteration": 2, "failed_rank": 1}, {"after_iteration": 4, "failed_rank": 1}], "failed_rank"),
    ],
)
def test_failure_rank_is_valid_and_unique(tmp_path, events, message):
    config_path, failures_path = write_inputs(tmp_path, failures={"events": events})

    with pytest.raises(ConfigError, match=message):
        load_config(config_path, failures_path)


@pytest.mark.parametrize("extra_field", ["speed", "p_i", "detector"])
def test_forbidden_failure_field_is_rejected_by_name(tmp_path, extra_field):
    failures = {"events": [{"after_iteration": 2, "failed_rank": 1, extra_field: 0.5}]}
    config_path, failures_path = write_inputs(tmp_path, failures=failures)

    with pytest.raises(ConfigError, match=extra_field):
        load_config(config_path, failures_path)
