"""PyTorch implementation of Flow Q-Learning (Park, Li, Levine, 2025)."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gameskill.constants import HYBRID_ACTION_DIM
from gameskill.models.fql_networks import FlowVectorField, OneStepPolicy, TwinQ
from gameskill.models.vision import VisionStateEncoder


class FlowQLearning(nn.Module):
    """FQL with a shared temporal DINOv2 observation encoder.

    The loss matches the official algorithm: twin TD critics, behavior-cloning
    flow matching, Euler-flow distillation into a one-step actor, and a
    reparameterized Q-maximization loss on that one-step actor.
    """

    def __init__(self, model_config: dict[str, Any], algorithm_config: dict[str, Any]) -> None:
        super().__init__()
        self.algorithm_config = dict(algorithm_config)
        self.action_dim = HYBRID_ACTION_DIM
        state_dim = int(model_config["state_dim"])
        actor_hidden_dims = tuple(model_config["actor_hidden_dims"])
        critic_hidden_dims = tuple(model_config["critic_hidden_dims"])

        self.state_encoder = VisionStateEncoder(model_config)
        self.actor_bc_flow = FlowVectorField(
            state_dim,
            self.action_dim,
            actor_hidden_dims,
            layer_norm=bool(model_config["actor_layer_norm"]),
        )
        self.actor_onestep_flow = OneStepPolicy(
            state_dim,
            self.action_dim,
            actor_hidden_dims,
            layer_norm=bool(model_config["actor_layer_norm"]),
        )
        self.critic = TwinQ(
            state_dim,
            self.action_dim,
            critic_hidden_dims,
            layer_norm=bool(model_config["critic_layer_norm"]),
        )
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.target_critic.eval()

    @property
    def flow_steps(self) -> int:
        return int(self.algorithm_config["flow_steps"])

    @torch.no_grad()
    def compute_flow_actions(self, states: Tensor, noises: Tensor) -> Tensor:
        actions = noises
        step_size = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            times = torch.full(
                (states.shape[0], 1),
                step / self.flow_steps,
                device=states.device,
                dtype=states.dtype,
            )
            actions = actions + self.actor_bc_flow(states, actions, times) * step_size
            actions = actions.clamp(-1.0, 1.0)
        return actions

    def sample_actions_from_states(self, states: Tensor, noises: Tensor) -> Tensor:
        return self.actor_onestep_flow(states, noises).clamp(-1.0, 1.0)

    def encode(self, observations: Tensor) -> Tensor:
        return self.state_encoder(observations)

    def act(self, observations: Tensor, noises: Tensor) -> Tensor:
        return self.sample_actions_from_states(self.encode(observations), noises)

    def train(self, mode: bool = True) -> FlowQLearning:
        super().train(mode)
        self.target_critic.eval()
        return self

    def _aggregate_q(self, q1: Tensor, q2: Tensor) -> Tensor:
        if self.algorithm_config["q_aggregation"] == "min":
            return torch.minimum(q1, q2)
        return 0.5 * (q1 + q2)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        masks = batch["masks"]
        batch_size = observations.shape[0]

        states = self.state_encoder(observations)
        with torch.no_grad():
            next_states = self.state_encoder(batch["next_observations"])
            next_noises = torch.randn(
                batch_size,
                self.action_dim,
                device=states.device,
                dtype=states.dtype,
            )
            next_actions = self.sample_actions_from_states(next_states, next_noises)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            next_q = self._aggregate_q(target_q1, target_q2)
            target_q = rewards + float(self.algorithm_config["discount"]) * masks * next_q

        q1, q2 = self.critic(states, actions)
        critic_loss = 0.5 * (
            F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        )

        flow_noises = torch.randn_like(actions)
        times = torch.rand(batch_size, 1, device=states.device, dtype=states.dtype)
        interpolated_actions = (1.0 - times) * flow_noises + times * actions
        target_velocities = actions - flow_noises
        predicted_velocities = self.actor_bc_flow(
            states, interpolated_actions, times
        )
        bc_flow_loss = F.mse_loss(predicted_velocities, target_velocities)

        distill_noises = torch.randn_like(actions)
        target_flow_actions = self.compute_flow_actions(
            states.detach(), distill_noises
        )
        actor_actions_unclipped = self.actor_onestep_flow(states, distill_noises)
        distill_loss = F.mse_loss(actor_actions_unclipped, target_flow_actions)
        actor_actions = actor_actions_unclipped.clamp(-1.0, 1.0)

        critic_requires_grad = [parameter.requires_grad for parameter in self.critic.parameters()]
        self.critic.requires_grad_(False)
        try:
            actor_q1, actor_q2 = self.critic(states, actor_actions)
        finally:
            for parameter, requires_grad in zip(
                self.critic.parameters(), critic_requires_grad
            ):
                parameter.requires_grad_(requires_grad)
        actor_q = 0.5 * (actor_q1 + actor_q2)
        q_loss = -actor_q.mean()
        if bool(self.algorithm_config["normalize_q_loss"]):
            q_scale = actor_q.detach().abs().mean().clamp_min(1e-6).reciprocal()
            q_loss = q_scale * q_loss

        actor_loss = (
            bc_flow_loss
            + float(self.algorithm_config["alpha"]) * distill_loss
            + q_loss
        )
        total_loss = (
            float(self.algorithm_config["critic_loss_weight"]) * critic_loss
            + actor_loss
        )
        with torch.no_grad():
            policy_mse = F.mse_loss(actor_actions, actions)
        return {
            "loss": total_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q_mean": 0.5 * (q1.mean() + q2.mean()),
            "target_q_mean": target_q.mean(),
            "policy_mse": policy_mse,
            "reward_mean": rewards.mean(),
            "mask_mean": masks.mean(),
        }

    @torch.no_grad()
    def soft_update_target(self) -> None:
        tau = float(self.algorithm_config["tau"])
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, tau)


__all__ = ["FlowQLearning"]
