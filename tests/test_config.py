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


# --- the optional analytical memory budget ----------------------------------------


def test_memory_budget_defaults_to_no_artificial_limit(tmp_path):
    """Absent means "no ceiling", which is what the shipped default config says."""
    loaded = load_config(*write_inputs(tmp_path))

    assert loaded.train.memory_budget_bytes is None


def test_explicit_null_memory_budget_means_no_limit(tmp_path):
    config = dict(VALID_CONFIG, memory_budget_bytes=None)
    loaded = load_config(*write_inputs(tmp_path, config=config))

    assert loaded.train.memory_budget_bytes is None


def test_positive_memory_budget_turns_the_feasibility_gate_on(tmp_path):
    config = dict(VALID_CONFIG, memory_budget_bytes=64 * 1024 * 1024)
    loaded = load_config(*write_inputs(tmp_path, config=config))

    assert loaded.train.memory_budget_bytes == 64 * 1024 * 1024


@pytest.mark.parametrize("value", [0, -1, "8GB", 1.5, True])
def test_a_malformed_memory_budget_is_rejected_rather_than_read_as_unlimited(tmp_path, value):
    """A typo must not silently disable the gate it was meant to switch on."""
    config = dict(VALID_CONFIG, memory_budget_bytes=value)
    with pytest.raises(ConfigError, match="memory_budget_bytes"):
        load_config(*write_inputs(tmp_path, config=config))


def test_the_shipped_default_config_loads_and_leaves_the_budget_unset():
    """The acceptance command's own inputs, parsed by the same validator."""
    loaded = load_config(Path("configs/train.json"), Path("configs/failures.json"))

    assert loaded.train.memory_budget_bytes is None
    assert loaded.train.num_layers == 6  # enough layers for a repartition to move one
    events = [(event.after_iteration, event.failed_rank) for event in loaded.failures]
    assert events == [(2, 1), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert [event[0] for event in events] == sorted({event[0] for event in events})
    assert len({event[1] for event in events}) == len(events)  # never the same rank twice
    assert max(event[0] for event in events) < loaded.train.iterations  # training continues after
