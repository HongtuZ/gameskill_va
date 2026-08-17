"""Self-contained deployment policy for GameSkill FQL checkpoints.

This file is fully standalone: ship only ``policy.py`` together with the
trained ``final.pt`` checkpoint. No other project code is required.

Dependencies:
    pip install torch torchvision timm pillow numpy
    # Audio-capable checkpoints additionally need:
    pip install torchaudio transformers
    # The demo script additionally needs:
    pip install soundfile

Quick start:
    from policy import GameSkillPolicy

    policy = GameSkillPolicy.from_checkpoint("final.pt", device="cuda")
    action = policy.infer(frames, audio_waveform=waveform)
    # frames: 10 consecutive screenshots (PIL images or HWC numpy arrays)
    # waveform: 3 s of 16 kHz mono audio, tensor of shape [48000]
    # action == {
    #     "mouse_move": {"dx": ..., "dy": ...},           # mouse delta in pixels
    #     "keyboard_probabilities": {"Fire": 0.98, ...},  # per-key press probability
    #     "pressed_keys": ["Fire", ...],                  # keys above the threshold
    # }

Notes:
- The checkpoint was trained on cached DINOv3 features, so this module
  re-attaches the frozen DINOv3 backbone from timm. The first run downloads
  it; offline machines need a pre-populated Hugging Face cache or must set
  HF_HUB_OFFLINE=1 with cached weights in place.
- Whether audio input is required is detected automatically from the weights.
  Audio checkpoints accept either a raw 3 s waveform (the EAT encoder is
  loaded lazily on first use) or a precomputed [768] feature vector.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import timm
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

MOUSE_ACTION_DIM = 2

FrameLike = Any  # PIL.Image or HWC uint8 numpy array


class DualViewTransform:
    """Create a whole-frame view and a centered local view.

    Both views are resized to the DINOv3 input size. The whole-frame resize is
    deliberately non-aspect-preserving so that no source pixels are discarded.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        center_crop_scale: float,
    ) -> None:
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.center_crop_scale = float(center_crop_scale)

    def _prepare(self, image: Image.Image) -> Tensor:
        resized = TF.resize(
            image,
            list(self.input_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        tensor = TF.to_tensor(resized)
        return TF.normalize(tensor, self.mean, self.std)

    def __call__(self, image: Image.Image) -> Tensor:
        image = image.convert("RGB")
        width, height = image.size
        crop_side = max(1, round(min(width, height) * self.center_crop_scale))
        left = (width - crop_side) // 2
        top = (height - crop_side) // 2
        center = TF.crop(image, top, left, crop_side, crop_side)
        return torch.stack((self._prepare(image), self._prepare(center)), dim=0)


class _StateEncoder(nn.Module):
    """Dual-view DINOv3 temporal encoder with frozen EAT feature fusion."""

    def __init__(self, config: dict[str, Any], vision_backbone: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_backbone
        self.num_views = int(config["num_views"])
        self.vision_feature_dim = int(self.vision_encoder.num_features)
        configured_dim = int(config["vision_feature_dim"])
        if configured_dim != self.vision_feature_dim:
            raise ValueError(
                f"model.vision_feature_dim={configured_dim} but the backbone "
                f"produces {self.vision_feature_dim} features"
            )
        self.frame_feature_dim = self.vision_feature_dim * self.num_views
        self.use_audio = bool(config.get("use_audio", True))
        temporal_dim = int(config["temporal_hidden_dim"] or self.vision_feature_dim)
        temporal_layers = int(config["temporal_num_layers"])
        dropout = float(config["dropout"])
        self.temporal_encoder = nn.GRU(
            input_size=self.frame_feature_dim,
            hidden_size=temporal_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, int(config["state_dim"])),
            nn.GELU(),
            nn.LayerNorm(int(config["state_dim"])),
        )
        if self.use_audio:
            self.audio_feature_dim = int(config["audio_feature_dim"])
            audio_hidden_dim = int(config["audio_hidden_dim"])
            self.audio_projection = nn.Sequential(
                nn.LayerNorm(self.audio_feature_dim),
                nn.Linear(self.audio_feature_dim, audio_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(audio_hidden_dim),
            )
            self.fusion = nn.Sequential(
                nn.Linear(
                    int(config["state_dim"]) + audio_hidden_dim,
                    int(config["state_dim"]),
                ),
                nn.GELU(),
                nn.LayerNorm(int(config["state_dim"])),
            )
        else:
            self.audio_feature_dim = 0
            self.audio_projection = None
            self.fusion = None

    @torch.no_grad()
    def _encode_views(self, images: Tensor) -> Tensor:
        batch, steps, views, channels, height, width = images.shape
        if views != self.num_views or channels != 3:
            raise ValueError(
                f"expected [B,T,{self.num_views},3,H,W] dual-view images, "
                f"got {tuple(images.shape)}"
            )
        flattened = images.reshape(batch * steps * views, channels, height, width)
        features = self.vision_encoder(flattened)
        return features.reshape(batch, steps, views * self.vision_feature_dim)

    def forward(self, images: Tensor, audio_features: Tensor | None) -> Tensor:
        frame_features = self._encode_views(images)
        batch = frame_features.shape[0]
        initial_state = frame_features.new_zeros(
            self.temporal_encoder.num_layers,
            batch,
            self.temporal_encoder.hidden_size,
        )
        temporal_features, _ = self.temporal_encoder(frame_features, initial_state)
        visual_state = self.projection(temporal_features[:, -1])
        if not self.use_audio:
            return visual_state
        if audio_features is None:
            raise ValueError("audio_features are required for this checkpoint")
        if audio_features.ndim != 2 or audio_features.shape != (
            batch, self.audio_feature_dim
        ):
            raise ValueError(
                f"audio_features must be [B,{self.audio_feature_dim}], got "
                f"{tuple(audio_features.shape)}"
            )
        assert self.audio_projection is not None and self.fusion is not None
        audio_state = self.audio_projection(audio_features)
        return self.fusion(torch.cat((visual_state, audio_state), dim=-1))


class _OneStepActor(nn.Module):
    """One-step noise-to-action policy with hybrid mouse/keyboard heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(current_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.mouse_policy = nn.Linear(current_dim, MOUSE_ACTION_DIM)
        self.keyboard_policy = nn.Linear(current_dim, action_dim - MOUSE_ACTION_DIM)

    def forward(self, states: Tensor, noises: Tensor) -> Tensor:
        hidden = self.network(torch.cat((states, noises), dim=-1))
        return torch.cat(
            (self.mouse_policy(hidden), self.keyboard_policy(hidden)), dim=-1
        )


class GameSkillPolicy(nn.Module):
    """Deployment-ready FQL policy producing semantic action dictionaries."""

    def __init__(
        self,
        model_config: dict[str, Any],
        keyboard_actions: tuple[str, ...],
        mouse_center: Tensor,
        mouse_scale: Tensor,
        *,
        vision_pretrained: bool = True,
        press_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= press_threshold <= 1.0:
            raise ValueError("press_threshold must be in [0, 1]")
        self.press_threshold = float(press_threshold)
        self.keyboard_actions = tuple(keyboard_actions)
        self.action_dim = MOUSE_ACTION_DIM + len(self.keyboard_actions)
        self.register_buffer(
            "mouse_center", mouse_center.detach().float(), persistent=False
        )
        self.register_buffer(
            "mouse_scale", mouse_scale.detach().float(), persistent=False
        )
        self.vision_backbone_name = str(model_config["vision_encoder_name"])
        self.center_crop_scale = float(model_config["center_crop_scale"])
        vision_backbone = timm.create_model(
            self.vision_backbone_name,
            pretrained=bool(vision_pretrained),
            num_classes=0,
        )
        self.state_encoder = _StateEncoder(model_config, vision_backbone)
        self.actor_onestep_flow = _OneStepActor(
            int(model_config["state_dim"]),
            self.action_dim,
            tuple(model_config["actor_hidden_dims"]),
            layer_norm=bool(model_config["actor_layer_norm"]),
        )
        self.use_audio = self.state_encoder.use_audio
        self.audio_feature_dim = self.state_encoder.audio_feature_dim
        self.sequence_length = 10
        self._audio_backbone: nn.Module | None = None
        self._audio_config: dict[str, Any] | None = None
        self._transform: DualViewTransform | None = None

    # ------------------------------------------------------------- checkpoint
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
        press_threshold: float = 0.5,
    ) -> "GameSkillPolicy":
        """Build the policy from a training checkpoint and move it to device."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("model", "config", "action_codec"):
            if key not in checkpoint:
                raise ValueError(f"checkpoint is missing the '{key}' section")
        model_config = deepcopy(checkpoint["config"]["model"])
        codec_state = checkpoint["action_codec"]
        state_dict = checkpoint["model"]

        has_audio_weights = any("audio_projection" in k for k in state_dict)
        use_audio = bool(model_config.get("use_audio", has_audio_weights))
        if use_audio != has_audio_weights:
            raise ValueError(
                f"checkpoint config says use_audio={use_audio} but the weights "
                f"{'contain' if has_audio_weights else 'lack'} audio layers"
            )
        model_config["use_audio"] = use_audio
        has_vision_weights = any(
            k.startswith("state_encoder.vision_encoder.") for k in state_dict
        )
        policy = cls(
            model_config,
            tuple(codec_state["keyboard_actions"]),
            torch.as_tensor(codec_state["mouse_center"], dtype=torch.float32),
            torch.as_tensor(codec_state["mouse_scale"], dtype=torch.float32),
            # Cache-trained checkpoints omit the backbone; fetch it from timm.
            vision_pretrained=not has_vision_weights,
            press_threshold=press_threshold,
        )
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("state_encoder.", "actor_onestep_flow."))
        }
        incompatible = policy.load_state_dict(filtered, strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not (
                not has_vision_weights
                and key.startswith("state_encoder.vision_encoder.")
            )
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint mismatch: "
                f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
            )
        policy._audio_config = checkpoint["config"].get("audio") or {}
        policy.sequence_length = int(checkpoint["config"]["data"]["sequence_length"])
        policy.eval().to(policy._resolve_device(device))
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        return policy

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -------------------------------------------------------------- transform
    def get_transform(self) -> DualViewTransform:
        if self._transform is None:
            data_config = timm.data.resolve_model_data_config(
                self.state_encoder.vision_encoder
            )
            self._transform = DualViewTransform(
                input_size=tuple(int(v) for v in data_config["input_size"][-2:]),
                mean=tuple(float(v) for v in data_config["mean"]),
                std=tuple(float(v) for v in data_config["std"]),
                center_crop_scale=self.center_crop_scale,
            )
        return self._transform

    # ------------------------------------------------------------------ audio
    def encode_audio_waveform(self, waveforms: Tensor) -> Tensor:
        """Encode ``[B,S]`` 16 kHz mono waveforms (3 s) into ``[B,D]`` features."""
        if not self.use_audio:
            raise RuntimeError("this checkpoint was trained without audio input")
        if self._audio_config is None:
            raise RuntimeError("checkpoint does not embed an audio configuration")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if self._audio_backbone is None:
            from transformers import AutoModel

            kwargs: dict[str, Any] = {
                "trust_remote_code": bool(
                    self._audio_config.get("trust_remote_code", True)
                )
            }
            if self._audio_config.get("revision"):
                kwargs["revision"] = str(self._audio_config["revision"])
            self._audio_backbone = (
                AutoModel.from_pretrained(
                    str(self._audio_config["encoder_name"]), **kwargs
                )
                .eval()
                .to(self.device)
            )
            self._audio_backbone.requires_grad_(False)
        fbanks = self._waveforms_to_eat_fbank(waveforms).to(self.device)
        with torch.inference_mode():
            tokens = self._audio_backbone.extract_features(fbanks)
        return tokens[:, 0].float()

    def _waveforms_to_eat_fbank(self, waveforms: Tensor) -> Tensor:
        """Convert ``[B,S]`` 16 kHz waveforms to normalized ``[B,1,T,128]`` fbanks."""
        import torch.nn.functional as F
        import torchaudio

        config = self._audio_config
        assert config is not None
        sample_rate = int(config["sample_rate"])
        target_length = int(config["mel_target_length"])
        mel_bins = int(config["mel_bins"])
        fbanks: list[Tensor] = []
        for waveform in waveforms.float().cpu():
            waveform = waveform - waveform.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=mel_bins,
                dither=0.0,
                frame_shift=10,
            )
            if mel.shape[0] < target_length:
                mel = F.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
            else:
                mel = mel[:target_length]
            mel = (mel - float(config["norm_mean"])) / (
                float(config["norm_std"]) * 2.0
            )
            fbanks.append(mel)
        return torch.stack(fbanks, dim=0).unsqueeze(1)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        images: Tensor,
        audio_features: Tensor | None = None,
        noises: Tensor | None = None,
    ) -> Tensor:
        """Return bounded ``[B,action_dim]`` actions from pixel views."""
        states = self.state_encoder(images, audio_features)
        if noises is None:
            noises = states.new_zeros(states.shape[0], self.action_dim)
        return self.actor_onestep_flow(states, noises).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ infer
    @torch.inference_mode()
    def infer(
        self,
        frames: Sequence[FrameLike],
        *,
        audio_features: Tensor | None = None,
        audio_waveform: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> dict[str, Any]:
        """Run one policy step over consecutive frames.

        Returns a dictionary with:

        - ``mouse_move``: ``{"dx": float, "dy": float}`` mouse delta in pixels.
        - ``keyboard_probabilities``: press probability per semantic key name.
        - ``pressed_keys``: key names whose probability exceeds
          ``press_threshold``.
        """
        if len(frames) != self.sequence_length:
            raise ValueError(
                f"this policy consumes {self.sequence_length} consecutive "
                f"frames, got {len(frames)}"
            )
        transform = self.get_transform()
        views = torch.stack(
            [
                transform(
                    frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
                )
                for frame in frames
            ]
        )
        images = views.unsqueeze(0).to(self.device, non_blocking=True)

        if self.use_audio:
            if audio_waveform is not None:
                audio_features = self.encode_audio_waveform(audio_waveform)
            if audio_features is None:
                raise ValueError(
                    "this checkpoint requires audio: pass audio_features "
                    f"[1,{self.audio_feature_dim}] or audio_waveform [1,S] "
                    "(3 s of 16 kHz mono)"
                )
            audio_features = audio_features.to(
                device=self.device, dtype=images.dtype
            )
            if audio_features.ndim == 1:
                audio_features = audio_features.unsqueeze(0)
        elif audio_features is not None or audio_waveform is not None:
            raise ValueError("this checkpoint was trained without audio input")

        if noise is not None:
            noise = noise.to(device=self.device, dtype=images.dtype)
            if noise.ndim == 1:
                noise = noise.unsqueeze(0)
        normalized_actions = self.forward(images, audio_features, noise)
        return self.to_semantic_action(normalized_actions)

    def to_semantic_action(self, normalized_actions: Tensor) -> dict[str, Any]:
        """Convert bounded ``[B,action_dim]`` policy outputs into semantic actions."""
        bounded = normalized_actions.clamp(-1.0, 1.0)
        mouse_move = (
            bounded[..., :MOUSE_ACTION_DIM] * self.mouse_scale + self.mouse_center
        )
        keyboard_probabilities = (
            (bounded[..., MOUSE_ACTION_DIM:] + 1.0) * 0.5
        ).clamp(0.0, 1.0)
        mouse_move = mouse_move.detach().float().cpu()
        keyboard_probabilities = keyboard_probabilities.detach().float().cpu()
        results: list[dict[str, Any]] = []
        for sample_index in range(normalized_actions.shape[0]):
            dx, dy = mouse_move[sample_index].tolist()
            probability_map = {
                name: float(probability)
                for name, probability in zip(
                    self.keyboard_actions, keyboard_probabilities[sample_index]
                )
            }
            results.append(
                {
                    "mouse_move": {"dx": float(dx), "dy": float(dy)},
                    "keyboard_probabilities": probability_map,
                    "pressed_keys": [
                        name
                        for name, probability in probability_map.items()
                        if probability >= self.press_threshold
                    ],
                }
            )
        return results[0] if len(results) == 1 else {"batch": results}


def load_policy(
    checkpoint_path: str,
    device: str = "auto",
    press_threshold: float = 0.5,
) -> GameSkillPolicy:
    """Convenience alias for :meth:`GameSkillPolicy.from_checkpoint`."""
    return GameSkillPolicy.from_checkpoint(
        checkpoint_path, device=device, press_threshold=press_threshold
    )


__all__ = ["DualViewTransform", "GameSkillPolicy", "load_policy"]
