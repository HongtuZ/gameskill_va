"""YAML configuration loading, overrides, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a training configuration is incomplete or invalid."""


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    config = deepcopy(loaded)
    for override in overrides or []:
        apply_override(config, override)
    validate_config(config)
    return config


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must use key=value syntax: {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    if not all(keys):
        raise ConfigError(f"invalid override key: {dotted_key!r}")
    value = yaml.safe_load(raw_value)
    node: dict[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            raise ConfigError(f"override path does not exist: {dotted_key!r}")
        node = child
    if keys[-1] not in node:
        raise ConfigError(f"override key does not exist: {dotted_key!r}")
    node[keys[-1]] = value


def validate_config(config: dict[str, Any]) -> None:
    sections = {"data", "model", "algorithm", "optimizer", "training", "logging"}
    missing = sections.difference(config)
    if missing:
        raise ConfigError(f"missing configuration sections: {sorted(missing)}")

    data = config["data"]
    model = config["model"]
    algorithm = config["algorithm"]
    training = config["training"]
    optimizer = config["optimizer"]

    if data.get("train_batch_size") is not None:
        _positive(data, "train_batch_size")
    elif data.get("batch_size") is not None:
        _positive(data, "batch_size")
    else:
        raise ConfigError("data.train_batch_size must be positive")
    _positive(data, "sequence_length")
    _non_negative(data, "num_workers")
    if data.get("precompute_batch_size") is not None:
        _positive(data, "precompute_batch_size")
    if data.get("precompute_num_workers") is not None:
        _non_negative(data, "precompute_num_workers")
    if data.get("max_samples") is not None:
        _positive(data, "max_samples")
    _positive(model, "state_dim")
    _positive(model, "vision_feature_dim")
    _positive(model, "num_views")
    _positive(model, "temporal_num_layers")
    _positive(algorithm, "flow_steps")
    _positive(training, "total_steps")
    if training.get("epochs") is not None:
        _positive(training, "epochs")
    _positive(training, "gradient_accumulation_steps")
    _positive(training, "max_grad_norm")
    _positive(optimizer, "learning_rate")
    _positive(optimizer, "vision_learning_rate")

    discount = float(algorithm["discount"])
    tau = float(algorithm["tau"])
    if not 0.0 <= discount <= 1.0:
        raise ConfigError("algorithm.discount must be in [0, 1]")
    if not 0.0 < tau <= 1.0:
        raise ConfigError("algorithm.tau must be in (0, 1]")
    if algorithm["q_aggregation"] not in {"mean", "min"}:
        raise ConfigError("algorithm.q_aggregation must be 'mean' or 'min'")
    if int(data["sequence_length"]) > 10:
        raise ConfigError("data.sequence_length cannot exceed the dataset history length 10")
    if bool(model["use_precomputed_features"]) and not data.get("feature_cache_path"):
        raise ConfigError(
            "data.feature_cache_path is required when model.use_precomputed_features=true"
        )
    if bool(model["use_precomputed_features"]) and not bool(
        model["freeze_vision_encoder"]
    ):
        raise ConfigError(
            "precomputed features require model.freeze_vision_encoder=true"
        )
    if int(model["num_views"]) != 2:
        raise ConfigError("this policy currently requires model.num_views=2")
    crop_scale = float(model["center_crop_scale"])
    if not 0.0 < crop_scale <= 1.0:
        raise ConfigError("model.center_crop_scale must be in (0, 1]")


def _positive(section: dict[str, Any], key: str) -> None:
    if key not in section or float(section[key]) <= 0:
        raise ConfigError(f"{key} must be positive")


def _non_negative(section: dict[str, Any], key: str) -> None:
    if key not in section or float(section[key]) < 0:
        raise ConfigError(f"{key} must be non-negative")


__all__ = ["ConfigError", "apply_override", "load_config", "validate_config"]
