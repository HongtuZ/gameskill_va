"""Standalone dual-head vision-action policy."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from gameskill.constants import DEFAULT_KEYBOARD_ACTIONS, MOUSE_ACTION_DIM
from gameskill.models.vision import VisionStateEncoder


class GameSkillVisionPolicy(nn.Module):
    """DINOv3 + EAT feature-fusion policy with separate action heads.

    This class is useful for supervised policy evaluation and inference. The
    FQL trainer uses the same :class:`VisionStateEncoder`; its one-step actor
    also has named mouse/keyboard output heads before concatenating the hybrid
    action for the flow-matching objective.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.vision_encoder = VisionStateEncoder(model_config)
        state_dim = int(model_config["state_dim"])
        hidden_dims = tuple(int(dim) for dim in model_config["actor_hidden_dims"])
        layers: list[nn.Module] = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
            current_dim = hidden_dim
        self.policy_trunk = nn.Sequential(*layers)
        self.mouse_policy = nn.Linear(current_dim, MOUSE_ACTION_DIM)
        self.keyboard_policy = nn.Linear(current_dim, len(DEFAULT_KEYBOARD_ACTIONS))

    def forward(
        self, observations: Tensor, audio_features: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        states = self.vision_encoder(observations, audio_features)
        hidden = self.policy_trunk(states)
        mouse_actions = self.mouse_policy(hidden).tanh()
        keyboard_logits = self.keyboard_policy(hidden)
        return mouse_actions, keyboard_logits

    def predict(
        self, observations: Tensor, audio_features: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        mouse_actions, keyboard_logits = self(observations, audio_features)
        return mouse_actions, keyboard_logits.sigmoid()


__all__ = ["GameSkillVisionPolicy"]
