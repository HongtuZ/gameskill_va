"""Actor and critic networks used by Flow Q-Learning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    layer_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(nn.GELU())
        current_dim = int(hidden_dim)
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class FlowVectorField(nn.Module):
    """Conditional rectified-flow velocity v(s, x_t, t)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.network = _mlp(
            state_dim + action_dim + 1,
            hidden_dims,
            action_dim,
            layer_norm,
        )

    def forward(self, states: Tensor, actions: Tensor, times: Tensor) -> Tensor:
        if times.ndim == 1:
            times = times.unsqueeze(-1)
        return self.network(torch.cat((states, actions, times), dim=-1))


class OneStepPolicy(nn.Module):
    """One-step noise-to-action policy μ(s, z) with hybrid-action heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(current_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.mouse_policy = nn.Linear(current_dim, 2)
        self.keyboard_policy = nn.Linear(current_dim, action_dim - 2)

    def forward(self, states: Tensor, noises: Tensor) -> Tensor:
        hidden = self.network(torch.cat((states, noises), dim=-1))
        return torch.cat(
            (self.mouse_policy(hidden), self.keyboard_policy(hidden)), dim=-1
        )


class TwinQ(nn.Module):
    """Independent double-Q networks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = _mlp(input_dim, hidden_dims, 1, layer_norm)
        self.q2 = _mlp(input_dim, hidden_dims, 1, layer_norm)

    def forward(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((states, actions), dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


__all__ = ["FlowVectorField", "OneStepPolicy", "TwinQ"]
