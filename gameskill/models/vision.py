"""Dual-view DINOv3 temporal encoder with frozen EAT feature fusion."""

from __future__ import annotations

from typing import Any

import timm
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def to_timm_model_name(name: str) -> str:
    """Translate a Hugging Face timm id into ``timm.create_model`` syntax."""
    return f"hf_hub:{name}" if name.startswith("timm/") else name


def create_vision_backbone(
    config: dict[str, Any], *, pretrained: bool | None = None
) -> nn.Module:
    return timm.create_model(
        to_timm_model_name(str(config["vision_encoder_name"])),
        pretrained=bool(config["pretrained"] if pretrained is None else pretrained),
        num_classes=0,
    )


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


def build_dual_view_transform(
    backbone: nn.Module, center_crop_scale: float
) -> DualViewTransform:
    data_config = timm.data.resolve_model_data_config(backbone)
    input_size = tuple(int(value) for value in data_config["input_size"][-2:])
    return DualViewTransform(
        input_size=input_size,
        mean=tuple(float(value) for value in data_config["mean"]),
        std=tuple(float(value) for value in data_config["std"]),
        center_crop_scale=center_crop_scale,
    )


class VisionStateEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.vision_encoder_name = str(config["vision_encoder_name"])
        self.num_views = int(config["num_views"])
        self.center_crop_scale = float(config["center_crop_scale"])
        self.use_precomputed_features = bool(config["use_precomputed_features"])
        self._vision_frozen = bool(config["freeze_vision_encoder"])
        self.vision_encoder: nn.Module | None
        if self.use_precomputed_features:
            self.vision_encoder = None
            vision_dim = int(config["vision_feature_dim"])
        else:
            self.vision_encoder = create_vision_backbone(config)
            vision_dim = int(self.vision_encoder.num_features)
            configured_dim = int(config["vision_feature_dim"])
            if configured_dim != vision_dim:
                raise ValueError(
                    f"model.vision_feature_dim={configured_dim} but "
                    f"{self.vision_encoder_name} produces {vision_dim} features"
                )
        self.vision_feature_dim = vision_dim
        self.frame_feature_dim = vision_dim * self.num_views
        self.use_audio = bool(config.get("use_audio", True))
        temporal_dim = int(config["temporal_hidden_dim"] or vision_dim)
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
                nn.Linear(int(config["state_dim"]) + audio_hidden_dim, int(config["state_dim"])),
                nn.GELU(),
                nn.LayerNorm(int(config["state_dim"])),
            )
        else:
            self.audio_feature_dim = 0
            self.audio_projection = None
            self.fusion = None
        if self.vision_encoder is not None:
            self.set_vision_trainable(not bool(config["freeze_vision_encoder"]))

    @property
    def state_dim(self) -> int:
        return int(self.projection[-1].normalized_shape[0])

    def set_vision_trainable(self, trainable: bool) -> None:
        if self.vision_encoder is None:
            if trainable:
                raise ValueError("precomputed-feature mode has no vision encoder")
            self._vision_frozen = True
            return
        self._vision_frozen = not trainable
        self.vision_encoder.requires_grad_(trainable)
        self.vision_encoder.train(self.training and trainable)

    def train(self, mode: bool = True) -> VisionStateEncoder:
        super().train(mode)
        if self._vision_frozen and self.vision_encoder is not None:
            self.vision_encoder.eval()
        return self

    @property
    def has_vision_encoder(self) -> bool:
        return self.vision_encoder is not None

    def get_dual_view_transform(self) -> DualViewTransform:
        if self.vision_encoder is None:
            raise RuntimeError(
                "the DINOv3 backbone is omitted in precomputed-feature mode"
            )
        return build_dual_view_transform(
            self.vision_encoder, self.center_crop_scale
        )

    def encode_frame_views(self, images: Tensor) -> Tensor:
        if self.vision_encoder is None:
            raise RuntimeError("cannot encode pixels without a DINOv3 backbone")
        if images.ndim != 6:
            raise ValueError(
                "dual-view images must be [B,T,V,C,H,W], "
                f"got {tuple(images.shape)}"
            )
        batch, steps, views, channels, height, width = images.shape
        if views != self.num_views or channels != 3 or steps < 1:
            raise ValueError(
                f"expected at least one RGB frame with {self.num_views} views"
            )
        # All whole-frame and center-crop views share one DINOv3 call. On GPU,
        # this executes as a single parallel batch rather than two serial passes.
        flattened = images.reshape(batch * steps * views, channels, height, width)
        if self._vision_frozen:
            with torch.no_grad():
                features = self.vision_encoder(flattened)
        else:
            features = self.vision_encoder(flattened)
        return features.reshape(batch, steps, views * self.vision_feature_dim)

    def forward(
        self, observations: Tensor, audio_features: Tensor | None = None
    ) -> Tensor:
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim == 3:
            frame_features = observations
            if frame_features.shape[-1] != self.frame_feature_dim:
                raise ValueError(
                    f"cached features must end in {self.frame_feature_dim}, "
                    f"got {tuple(frame_features.shape)}"
                )
        elif observations.ndim == 6:
            frame_features = self.encode_frame_views(observations)
        else:
            raise ValueError(
                "observations must be cached [B,T,V*D] features or pixel "
                f"[B,T,V,C,H,W] views, got {tuple(observations.shape)}"
            )
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
            raise ValueError("audio_features are required when model.use_audio=true")
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


__all__ = [
    "DualViewTransform",
    "VisionStateEncoder",
    "build_dual_view_transform",
    "create_vision_backbone",
    "to_timm_model_name",
]
