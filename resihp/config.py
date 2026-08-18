"""Validated configuration and fail-stop event loading."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a training or failure configuration is invalid."""


@dataclass(frozen=True)
class TrainConfig:
    model_dim: int
    num_layers: int
    num_heads: int
    batch_size: int
    micro_batch_size: int
    seed: int
    tp: int
    pp: int
    dp: int
    iterations: int

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp


@dataclass(frozen=True)
class FailureEvent:
    after_iteration: int
    failed_rank: int


@dataclass(frozen=True)
class LoadedConfig:
    train: TrainConfig
    failures: tuple[FailureEvent, ...]


_TRAIN_FIELDS = {
    "model_dim",
    "num_layers",
    "num_heads",
    "batch_size",
    "micro_batch_size",
    "seed",
    "tp",
    "pp",
    "dp",
    "iterations",
}
_EVENT_FIELDS = {"after_iteration", "failed_rank"}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"无法读取{label}文件 {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"{label}文件必须是 JSON 对象: {path}")
    return value


def _positive_int(values: dict[str, Any], field: str, label: str) -> int:
    value = values.get(field)
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{label}字段 {field} 必须是正整数")
    return value


def _parse_train(raw: dict[str, Any]) -> TrainConfig:
    missing = _TRAIN_FIELDS - raw.keys()
    if missing:
        raise ConfigError(f"训练配置缺少字段: {sorted(missing)}")
    unknown = raw.keys() - _TRAIN_FIELDS
    if unknown:
        raise ConfigError(f"训练配置包含未知字段: {sorted(unknown)}")
    values = {field: _positive_int(raw, field, "训练配置") for field in _TRAIN_FIELDS}
    if values["model_dim"] % values["num_heads"]:
        raise ConfigError("训练配置字段 model_dim 必须能被 num_heads 整除")
    if values["batch_size"] % values["micro_batch_size"]:
        raise ConfigError("训练配置字段 batch_size 必须能被 micro_batch_size 整除")
    return TrainConfig(**values)


def _parse_failures(raw: dict[str, Any], train: TrainConfig) -> tuple[FailureEvent, ...]:
    if set(raw) != {"events"} or not isinstance(raw.get("events"), list):
        raise ConfigError("故障配置必须只包含 events 数组")
    events = []
    previous_iteration = 0
    seen_ranks = set()
    for index, event in enumerate(raw["events"]):
        if not isinstance(event, dict):
            raise ConfigError(f"故障事件 {index} 必须是 JSON 对象")
        unknown = event.keys() - _EVENT_FIELDS
        if unknown:
            field = sorted(unknown)[0]
            raise ConfigError(f"故障事件包含禁止字段: {field}")
        missing = _EVENT_FIELDS - event.keys()
        if missing:
            raise ConfigError(f"故障事件缺少字段: {sorted(missing)}")
        after_iteration = _positive_int(event, "after_iteration", "故障事件")
        failed_rank = event["failed_rank"]
        if type(failed_rank) is not int or not 0 <= failed_rank < train.world_size:
            raise ConfigError(f"故障事件 failed_rank 越界: {failed_rank}")
        if after_iteration <= previous_iteration:
            raise ConfigError("故障事件 after_iteration 必须严格递增")
        if failed_rank in seen_ranks:
            raise ConfigError(f"故障事件 failed_rank 重复: {failed_rank}")
        previous_iteration = after_iteration
        seen_ranks.add(failed_rank)
        events.append(FailureEvent(after_iteration, failed_rank))
    return tuple(events)


def load_config(config_path: str | Path, failures_path: str | Path) -> LoadedConfig:
    train = _parse_train(_read_object(Path(config_path), "训练配置"))
    failures = _parse_failures(_read_object(Path(failures_path), "故障配置"), train)
    return LoadedConfig(train=train, failures=failures)
