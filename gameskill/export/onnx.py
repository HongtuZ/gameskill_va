"""Export the one-step FQL policy to ONNX and optionally verify it."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from gameskill.algorithms.flow_q_learning import FlowQLearning
from gameskill.data.action_codec import HybridActionCodec


class OnnxGameSkillPolicy(nn.Module):
    """Deployment wrapper returning decoded mouse and keyboard probabilities."""

    def __init__(self, agent: FlowQLearning, codec: HybridActionCodec) -> None:
        super().__init__()
        self.state_encoder = agent.state_encoder
        self.actor = agent.actor_onestep_flow
        self.register_buffer("mouse_center", codec.mouse_center.float())
        self.register_buffer("mouse_scale", codec.mouse_scale.float())

    def forward(
        self, images: Tensor, audio_features: Tensor, noises: Tensor
    ) -> tuple[Tensor, Tensor]:
        states = self.state_encoder(images, audio_features)
        normalized_actions = self.actor(states, noises).clamp(-1.0, 1.0)
        mouse_move = normalized_actions[:, :2] * self.mouse_scale + self.mouse_center
        keyboard_probabilities = (
            (normalized_actions[:, 2:] + 1.0) * 0.5
        ).clamp(0.0, 1.0)
        return mouse_move, keyboard_probabilities


def export_onnx_policy(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset_version: int = 18,
    batch_size: int = 1,
    verify: bool = True,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: dict[str, Any] = checkpoint["config"]
    model_config = deepcopy(config["model"])
    cached_checkpoint = bool(model_config.get("use_precomputed_features", False))
    # Deployment accepts pixels, so a cache-trained temporal policy must have
    # the locally cached frozen DINOv3 backbone attached again for export.
    model_config["use_precomputed_features"] = False
    if not cached_checkpoint:
        # Pixel-trained checkpoints already contain every DINOv3 parameter.
        model_config["pretrained"] = False
    elif not bool(model_config.get("pretrained", False)):
        raise ValueError(
            "a feature-cache checkpoint needs model.pretrained=true so the frozen "
            "DINOv3 weights can be reattached during ONNX export"
        )
    agent = FlowQLearning(model_config, config["algorithm"])
    incompatible = agent.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not (cached_checkpoint and key.startswith("state_encoder.vision_encoder."))
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    codec = HybridActionCodec.from_state_dict(checkpoint["action_codec"])
    wrapper = OnnxGameSkillPolicy(agent, codec).eval()

    sequence_length = int(config["data"]["sequence_length"])
    data_config = agent.state_encoder.vision_encoder.pretrained_cfg
    input_height, input_width = data_config["input_size"][-2:]
    images = torch.randn(
        batch_size,
        sequence_length,
        int(model_config["num_views"]),
        3,
        input_height,
        input_width,
        dtype=torch.float32,
    )
    noises = torch.zeros(batch_size, agent.action_dim, dtype=torch.float32)
    audio_features = torch.zeros(
        batch_size, int(model_config["audio_feature_dim"]), dtype=torch.float32
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference_outputs = wrapper(images, audio_features, noises)
    torch.onnx.export(
        wrapper,
        (images, audio_features, noises),
        output_path,
        input_names=["images", "audio_features", "noises"],
        output_names=["mouse_move", "keyboard_probabilities"],
        dynamic_axes={
            "images": {0: "batch"},
            "audio_features": {0: "batch"},
            "noises": {0: "batch"},
            "mouse_move": {0: "batch"},
            "keyboard_probabilities": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    if verify:
        _verify_onnx(
            output_path, images, audio_features, noises, reference_outputs
        )
    return output_path


def _verify_onnx(
    output_path: Path,
    images: Tensor,
    audio_features: Tensor,
    noises: Tensor,
    reference_outputs: tuple[Tensor, Tensor],
) -> None:
    import onnx
    import onnxruntime as ort

    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    test_cases = [(images, audio_features, noises, reference_outputs)]
    if images.shape[0] == 1:
        batch_two_images = images.repeat(2, 1, 1, 1, 1, 1)
        batch_two_noises = noises.repeat(2, 1)
        batch_two_audio = audio_features.repeat(2, 1)
        with torch.no_grad():
            batch_two_outputs = (
                torch.from_numpy(reference_outputs[0].detach().numpy()).repeat(2, 1),
                torch.from_numpy(reference_outputs[1].detach().numpy()).repeat(2, 1),
            )
        test_cases.append(
            (batch_two_images, batch_two_audio, batch_two_noises, batch_two_outputs)
        )

    for case_images, case_audio, case_noises, expected_outputs in test_cases:
        actual_outputs = session.run(
            None,
            {
                "images": case_images.numpy(),
                "audio_features": case_audio.numpy(),
                "noises": case_noises.numpy(),
            },
        )
        for name, actual, expected in zip(
            ("mouse_move", "keyboard_probabilities"),
            actual_outputs,
            expected_outputs,
        ):
            np.testing.assert_allclose(
                actual,
                expected.detach().numpy(),
                rtol=2e-3,
                atol=2e-4,
                err_msg=(
                    f"ONNX verification failed for {name} "
                    f"at batch={case_images.shape[0]}"
                ),
            )


__all__ = ["OnnxGameSkillPolicy", "export_onnx_policy"]
