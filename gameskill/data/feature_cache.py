"""On-disk cache for frozen, concatenated dual-view DINOv2 features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def normalize_frame_name(name: str | Path) -> str:
    return Path(name).as_posix()


@dataclass
class FrozenFeatureCache:
    features: Tensor
    filenames: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(
                f"cached features must be [N,D], got {tuple(self.features.shape)}"
            )
        if self.features.shape[0] != len(self.filenames):
            raise ValueError("cache filenames and feature rows have different lengths")
        normalized = tuple(normalize_frame_name(name) for name in self.filenames)
        if len(set(normalized)) != len(normalized):
            raise ValueError("cache contains duplicate frame filenames")
        self.filenames = normalized
        self._index = {name: index for index, name in enumerate(normalized)}

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def get_sequence(self, filenames: list[str] | tuple[str, ...]) -> Tensor:
        missing = [
            normalize_frame_name(name)
            for name in filenames
            if normalize_frame_name(name) not in self._index
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(
                f"{len(missing)} frame(s) are absent from the feature cache: {preview}. "
                "Re-run scripts/precompute_features.py for this dataset."
            )
        indices = torch.tensor(
            [self._index[normalize_frame_name(name)] for name in filenames],
            dtype=torch.long,
        )
        # The cache is stored as fp16 to reduce disk/RAM usage. Converting only
        # the selected sequence keeps the model and GRU computation in fp32.
        return self.features.index_select(0, indices).float()

    def validate_model_config(self, model_config: dict[str, Any]) -> None:
        expected_dim = int(model_config["vision_feature_dim"]) * int(
            model_config["num_views"]
        )
        if self.feature_dim != expected_dim:
            raise ValueError(
                f"feature cache dimension is {self.feature_dim}, expected {expected_dim}"
            )
        expected = {
            "vision_encoder_name": str(model_config["vision_encoder_name"]),
            "num_views": int(model_config["num_views"]),
            "center_crop_scale": float(model_config["center_crop_scale"]),
        }
        for key, value in expected.items():
            cached = self.metadata.get(key)
            if cached is None:
                continue
            if isinstance(value, float):
                matches = abs(float(cached) - value) < 1e-9
            else:
                matches = cached == value
            if not matches:
                raise ValueError(
                    f"feature cache {key}={cached!r}, but config requests {value!r}"
                )


def load_feature_cache(path: str | Path) -> FrozenFeatureCache:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"frozen feature cache not found: {cache_path}. Run "
            "`uv run python scripts/precompute_features.py` first."
        )
    try:
        payload = torch.load(
            cache_path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    required = {"features", "filenames", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"invalid feature cache format: {cache_path}")
    return FrozenFeatureCache(
        features=torch.as_tensor(payload["features"]),
        filenames=tuple(str(name) for name in payload["filenames"]),
        metadata=dict(payload["metadata"]),
    )


def save_feature_cache(
    path: str | Path,
    features: Tensor,
    filenames: list[str] | tuple[str, ...],
    metadata: dict[str, Any],
) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.detach().cpu().contiguous().half(),
            "filenames": [normalize_frame_name(name) for name in filenames],
            "metadata": dict(metadata),
        },
        cache_path,
    )
    return cache_path


__all__ = [
    "FrozenFeatureCache",
    "load_feature_cache",
    "normalize_frame_name",
    "save_feature_cache",
]
