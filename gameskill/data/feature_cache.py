"""On-disk cache for frozen DINOv3 visual and EAT audio features."""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    audio_features: Tensor | None = None
    audio_frame_indices: Tensor | None = None

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
        if (self.audio_features is None) != (self.audio_frame_indices is None):
            raise ValueError("audio features and frame indices must both be present")
        self._audio_index: dict[int, int] = {}
        if self.audio_features is not None and self.audio_frame_indices is not None:
            if self.audio_features.ndim != 2:
                raise ValueError(
                    "cached audio features must be [N,D], got "
                    f"{tuple(self.audio_features.shape)}"
                )
            self.audio_frame_indices = self.audio_frame_indices.long().flatten()
            if self.audio_features.shape[0] != self.audio_frame_indices.numel():
                raise ValueError("audio feature rows and frame indices differ")
            frame_indices = [int(value) for value in self.audio_frame_indices.tolist()]
            if len(set(frame_indices)) != len(frame_indices):
                raise ValueError("cache contains duplicate audio frame indices")
            self._audio_index = {
                frame_index: row for row, frame_index in enumerate(frame_indices)
            }

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def audio_feature_dim(self) -> int | None:
        if self.audio_features is None:
            return None
        return int(self.audio_features.shape[1])

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

    def validate_audio_config(
        self, audio_config: dict[str, Any], model_config: dict[str, Any]
    ) -> None:
        if not bool(audio_config.get("enabled", False)):
            return
        if self.audio_features is None or self.audio_frame_indices is None:
            raise ValueError(
                "feature cache has no EAT audio features; regenerate it with "
                "scripts/precompute_features.py --overwrite"
            )
        expected_dim = int(model_config["audio_feature_dim"])
        if self.audio_feature_dim != expected_dim:
            raise ValueError(
                f"audio cache dimension is {self.audio_feature_dim}, expected {expected_dim}"
            )
        expected = {
            "audio_encoder_name": str(audio_config["encoder_name"]),
            "audio_encoder_revision": str(audio_config.get("revision") or ""),
            "audio_sample_rate": int(audio_config["sample_rate"]),
            "audio_window_samples": int(audio_config["window_samples"]),
        }
        for key, value in expected.items():
            cached = self.metadata.get(key)
            if cached != value:
                raise ValueError(
                    f"feature cache {key}={cached!r}, but config requests {value!r}"
                )

    def get_audio_features(self, frame_indices: list[int]) -> Tensor:
        if self.audio_features is None:
            raise ValueError("feature cache does not contain audio features")
        missing = [index for index in frame_indices if index not in self._audio_index]
        if missing:
            raise KeyError(
                f"{len(missing)} row(s) are absent from the audio cache; "
                f"first frame_index={missing[0]}"
            )
        rows = torch.tensor(
            [self._audio_index[index] for index in frame_indices], dtype=torch.long
        )
        return self.audio_features.index_select(0, rows).float()


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
        audio_features=(
            torch.as_tensor(payload["audio_features"])
            if "audio_features" in payload
            else None
        ),
        audio_frame_indices=(
            torch.as_tensor(payload["audio_frame_indices"])
            if "audio_frame_indices" in payload
            else None
        ),
    )


def save_feature_cache(
    path: str | Path,
    features: Tensor,
    filenames: list[str] | tuple[str, ...],
    metadata: dict[str, Any],
    *,
    audio_features: Tensor | None = None,
    audio_frame_indices: Tensor | None = None,
) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "features": features.detach().cpu().contiguous().half(),
        "filenames": [normalize_frame_name(name) for name in filenames],
        "metadata": dict(metadata),
    }
    if (audio_features is None) != (audio_frame_indices is None):
        raise ValueError("audio_features and audio_frame_indices must be saved together")
    if audio_features is not None and audio_frame_indices is not None:
        payload["audio_features"] = audio_features.detach().cpu().contiguous().half()
        payload["audio_frame_indices"] = (
            audio_frame_indices.detach().cpu().contiguous().long()
        )
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, cache_path)
    return cache_path


__all__ = [
    "FrozenFeatureCache",
    "load_feature_cache",
    "normalize_frame_name",
    "save_feature_cache",
]
