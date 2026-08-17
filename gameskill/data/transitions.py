"""Hugging Face imagefolder to offline-RL transition conversion."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from gameskill.constants import DEFAULT_KEYBOARD_ACTIONS
from gameskill.data.action_codec import HybridActionCodec
from gameskill.data.feature_cache import (
    FrozenFeatureCache,
    load_feature_cache,
    normalize_frame_name,
)


class _TransitionDatasetBase(TorchDataset[dict[str, Tensor]]):
    """Shared transition, reward, action, and episode-boundary handling."""

    def __init__(
        self,
        dataset: Dataset,
        action_codec: HybridActionCodec,
        reward_column: str,
        done_column: str | None,
        allow_missing_reward: bool,
        missing_reward_value: float,
        reward_scale: float,
        reward_bias: float,
    ) -> None:
        self.dataset = dataset
        self.action_codec = action_codec
        self.reward_column = reward_column
        self.done_column = done_column
        self.allow_missing_reward = allow_missing_reward
        self.missing_reward_value = float(missing_reward_value)
        self.reward_scale = float(reward_scale)
        self.reward_bias = float(reward_bias)

        required = {"mouse_move", "segment_id", *DEFAULT_KEYBOARD_ACTIONS}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
        if reward_column not in dataset.column_names and not allow_missing_reward:
            raise ValueError(
                f"dataset has no {reward_column!r} column. Flow Q-Learning requires "
                "a real scalar reward for every transition. Add it to metadata.jsonl, "
                "or set data.allow_missing_reward=true only for a code-path smoke test."
            )
        if done_column and done_column not in dataset.column_names:
            done_column = None
        self.done_column = done_column
        self.segment_ids = list(dataset["segment_id"])

    def __len__(self) -> int:
        return len(self.dataset)

    def _is_segment_end(self, index: int) -> bool:
        return (
            index + 1 >= len(self.dataset)
            or self.segment_ids[index + 1] != self.segment_ids[index]
        )

    def _observation(self, index: int) -> Tensor:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = self.dataset[index]
        segment_end = self._is_segment_end(index)
        explicit_done = bool(sample[self.done_column]) if self.done_column else False
        done = segment_end or explicit_done
        next_index = index if segment_end else index + 1
        raw_reward = (
            sample[self.reward_column]
            if self.reward_column in sample
            else self.missing_reward_value
        )
        reward = (float(raw_reward) + self.reward_bias) * self.reward_scale
        return {
            "observations": self._observation(index),
            "actions": self.action_codec.encode_sample(sample),
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "next_observations": self._observation(next_index),
            "masks": torch.tensor(0.0 if done else 1.0, dtype=torch.float32),
        }


class ImagefolderTransitionDataset(_TransitionDatasetBase):
    """Pixel-backed transitions yielding ``[T,2,3,H,W]`` observations."""

    def __init__(
        self,
        dataset: Dataset,
        image_transform: Callable[[Any], Tensor],
        sequence_length: int,
        **kwargs: Any,
    ) -> None:
        if "images" not in dataset.column_names:
            raise ValueError("dataset is missing required 'images' column")
        super().__init__(dataset=dataset, **kwargs)
        self.image_transform = image_transform
        self.sequence_length = sequence_length

    def _observation(self, index: int) -> Tensor:
        frames = self.dataset[index]["images"]
        if not frames:
            raise ValueError("dataset sample contains no frames")
        if len(frames) < self.sequence_length:
            frames = [frames[0]] * (self.sequence_length - len(frames)) + frames
        else:
            frames = frames[-self.sequence_length :]
        return torch.stack([self.image_transform(frame) for frame in frames])


class CachedFeatureTransitionDataset(_TransitionDatasetBase):
    """Transitions that never decode images and directly index frozen features.

    All observations and actions are materialized into contiguous tensors at
    construction time, so ``__getitem__`` is a pure row slice. This keeps the
    DataLoader cheap enough to feed large training batches without starving the
    GPU.
    """

    def __init__(
        self,
        dataset: Dataset,
        filename_sequences: list[list[str]],
        feature_cache: FrozenFeatureCache,
        sequence_length: int,
        **kwargs: Any,
    ) -> None:
        if len(filename_sequences) != len(dataset):
            raise ValueError(
                "metadata row count does not match Hugging Face dataset row count"
            )
        # Removing the Image feature is important: datasets would otherwise
        # decode ten PIL images whenever a transition row is accessed.
        if "images" in dataset.column_names:
            dataset = dataset.remove_columns(["images"])
        super().__init__(dataset=dataset, **kwargs)
        self.filename_sequences = [
            _pad_or_trim(names, sequence_length) for names in filename_sequences
        ]
        self.feature_cache = feature_cache
        self._build_tensors(sequence_length)

    def _build_tensors(self, sequence_length: int) -> None:
        dataset = self.dataset
        count = len(dataset)
        # Resolve every frame name to a cache row once; per-sample dict lookups
        # in the training loop were the dominant CPU cost.
        index_map = self.feature_cache._index
        frame_indices = torch.empty(count, sequence_length, dtype=torch.long)
        for row, names in enumerate(self.filename_sequences):
            for column, name in enumerate(names):
                key = normalize_frame_name(name)
                frame_index = index_map.get(key)
                if frame_index is None:
                    raise KeyError(
                        f"frame {key!r} is absent from the feature cache. "
                        "Re-run scripts/precompute_features.py for this dataset."
                    )
                frame_indices[row, column] = frame_index
        # [N, T, num_views * feature_dim]; fp16 on disk, fp32 for the model.
        flat = frame_indices.reshape(-1)
        self._observations = (
            self.feature_cache.features.index_select(0, flat)
            .float()
            .view(count, sequence_length, -1)
        )

        columns = ["mouse_move", "segment_id", *self.action_codec.keyboard_actions]
        if self.reward_column in dataset.column_names:
            columns.append(self.reward_column)
        if self.done_column:
            columns.append(self.done_column)
        table = dataset.select_columns(columns)[:]

        mouse = torch.as_tensor(table["mouse_move"], dtype=torch.float32)
        normalized_mouse = (
            (mouse - self.action_codec.mouse_center)
            / self.action_codec.mouse_scale
        ).clamp(-1.0, 1.0)
        keyboard = torch.empty(
            count, len(self.action_codec.keyboard_actions), dtype=torch.float32
        )
        for column, action in enumerate(self.action_codec.keyboard_actions):
            keyboard[:, column] = torch.as_tensor(
                [1.0 if pressed else -1.0 for pressed in table[action]],
                dtype=torch.float32,
            )
        self._actions = torch.cat((normalized_mouse, keyboard), dim=1)

        segment_ids = table["segment_id"]
        segment_end = torch.tensor(
            [
                row + 1 >= count or segment_ids[row + 1] != segment_ids[row]
                for row in range(count)
            ]
        )
        explicit_done = (
            torch.as_tensor(table[self.done_column], dtype=torch.bool)
            if self.done_column
            else torch.zeros(count, dtype=torch.bool)
        )
        done = segment_end | explicit_done
        self._masks = torch.where(done, torch.zeros(()), torch.ones(())).float()

        raw_rewards = (
            torch.as_tensor(table[self.reward_column], dtype=torch.float32)
            if self.reward_column in table
            else torch.full((count,), self.missing_reward_value, dtype=torch.float32)
        )
        self._rewards = (raw_rewards + self.reward_bias) * self.reward_scale

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        next_index = index if bool(self._masks[index] == 0.0) else index + 1
        return {
            "observations": self._observations[index],
            "actions": self._actions[index],
            "rewards": self._rewards[index],
            "next_observations": self._observations[next_index],
            "masks": self._masks[index],
        }

    def to(self, device: torch.device) -> "CachedFeatureTransitionDataset":
        """Move every materialized tensor to ``device`` for GPU-resident training."""
        self._observations = self._observations.to(device)
        self._actions = self._actions.to(device)
        self._rewards = self._rewards.to(device)
        self._masks = self._masks.to(device)
        return self

    def gather(self, indices: Tensor) -> dict[str, Tensor]:
        """Batch lookup fully on-device; used by GPU-resident training."""
        next_indices = torch.where(self._masks[indices] == 0.0, indices, indices + 1)
        return {
            "observations": self._observations[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_observations": self._observations[next_indices],
            "masks": self._masks[indices],
        }


def _pad_or_trim(names: list[str], sequence_length: int) -> list[str]:
    if not names:
        raise ValueError("metadata row contains an empty file_names list")
    if len(names) < sequence_length:
        return [names[0]] * (sequence_length - len(names)) + names
    return names[-sequence_length:]


def load_filename_sequences(
    data_config: dict[str, Any], expected_rows: int
) -> list[list[str]]:
    metadata_path = (
        Path(data_config["path"])
        / str(data_config["split"])
        / "metadata.jsonl"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata file required for cached training was not found: {metadata_path}"
        )
    sequences: list[list[str]] = []
    with metadata_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            names = row.get("file_names")
            if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names
            ):
                raise ValueError(
                    f"invalid file_names at {metadata_path}:{line_number}"
                )
            sequences.append(names)
            if len(sequences) == expected_rows:
                break
    if len(sequences) != expected_rows:
        raise ValueError(
            f"read {len(sequences)} metadata rows but expected {expected_rows}"
        )
    return sequences


def build_transition_dataset(
    dataset: Dataset,
    image_transform: Callable[[Any], Tensor] | None,
    action_codec: HybridActionCodec,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
) -> _TransitionDatasetBase:
    common = {
        "action_codec": action_codec,
        "reward_column": str(data_config["reward_column"]),
        "done_column": data_config.get("done_column"),
        "allow_missing_reward": bool(data_config["allow_missing_reward"]),
        "missing_reward_value": float(data_config["missing_reward_value"]),
        "reward_scale": float(data_config["reward_scale"]),
        "reward_bias": float(data_config["reward_bias"]),
    }
    sequence_length = int(data_config["sequence_length"])
    if bool(model_config["use_precomputed_features"]):
        cache = load_feature_cache(data_config["feature_cache_path"])
        cache.validate_model_config(model_config)
        sequences = load_filename_sequences(data_config, len(dataset))
        return CachedFeatureTransitionDataset(
            dataset=dataset,
            filename_sequences=sequences,
            feature_cache=cache,
            sequence_length=sequence_length,
            **common,
        )
    if image_transform is None:
        raise ValueError("image_transform is required for pixel-backed training")
    return ImagefolderTransitionDataset(
        dataset=dataset,
        image_transform=image_transform,
        sequence_length=sequence_length,
        **common,
    )


__all__ = [
    "CachedFeatureTransitionDataset",
    "ImagefolderTransitionDataset",
    "build_transition_dataset",
    "load_filename_sequences",
]
