"""Bounded action encoding for FQL's continuous flow policy.

Mouse deltas are robustly normalized to [-1, 1].  Boolean keyboard/mouse-button
actions are relaxed from {False, True} to {-1, +1}.  The result is a 33-D
continuous vector compatible with the original continuous-action FQL objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from torch import Tensor

from gameskill.constants import DEFAULT_KEYBOARD_ACTIONS, HYBRID_ACTION_DIM


@dataclass
class HybridActionCodec:
    mouse_center: Tensor
    mouse_scale: Tensor
    keyboard_actions: tuple[str, ...] = DEFAULT_KEYBOARD_ACTIONS

    @property
    def action_dim(self) -> int:
        return 2 + len(self.keyboard_actions)

    def encode_sample(self, sample: dict[str, Any]) -> Tensor:
        mouse = torch.as_tensor(sample["mouse_move"], dtype=torch.float32)
        normalized_mouse = ((mouse - self.mouse_center) / self.mouse_scale).clamp(
            -1.0, 1.0
        )
        keyboard = torch.tensor(
            [1.0 if sample[action] else -1.0 for action in self.keyboard_actions],
            dtype=torch.float32,
        )
        action = torch.cat((normalized_mouse, keyboard))
        if action.shape != (HYBRID_ACTION_DIM,):
            raise RuntimeError(f"unexpected encoded action shape: {tuple(action.shape)}")
        return action

    def decode(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        bounded = actions.clamp(-1.0, 1.0)
        center = self.mouse_center.to(device=actions.device, dtype=actions.dtype)
        scale = self.mouse_scale.to(device=actions.device, dtype=actions.dtype)
        mouse_move = bounded[..., :2] * scale + center
        keyboard_probabilities = ((bounded[..., 2:] + 1.0) * 0.5).clamp(0.0, 1.0)
        return mouse_move, keyboard_probabilities

    def state_dict(self) -> dict[str, Any]:
        return {
            "mouse_center": self.mouse_center.cpu(),
            "mouse_scale": self.mouse_scale.cpu(),
            "keyboard_actions": self.keyboard_actions,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HybridActionCodec:
        return cls(
            mouse_center=torch.as_tensor(state["mouse_center"], dtype=torch.float32),
            mouse_scale=torch.as_tensor(state["mouse_scale"], dtype=torch.float32),
            keyboard_actions=tuple(state["keyboard_actions"]),
        )

    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        quantile: float = 0.995,
        minimum_scale: float = 1.0,
    ) -> HybridActionCodec:
        if not 0.5 <= quantile <= 1.0:
            raise ValueError("mouse scale quantile must be in [0.5, 1.0]")
        chunks: list[Tensor] = []
        for batch in dataset.select_columns(["mouse_move"]).iter(batch_size=4096):
            chunks.append(torch.as_tensor(batch["mouse_move"], dtype=torch.float32))
        if not chunks:
            raise ValueError("cannot fit action codec on an empty dataset")
        mouse_moves = torch.cat(chunks, dim=0)
        center = mouse_moves.median(dim=0).values
        scale = torch.quantile((mouse_moves - center).abs(), quantile, dim=0)
        scale = scale.clamp_min(minimum_scale)
        return cls(mouse_center=center, mouse_scale=scale)


__all__ = ["HybridActionCodec"]
