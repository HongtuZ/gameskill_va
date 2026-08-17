"""Precompute frozen dual-view DINOv2 features for all unique frames."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gameskill.data.feature_cache import save_feature_cache
from gameskill.models.vision import (
    DualViewTransform,
    build_dual_view_transform,
    create_vision_backbone,
)


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class UniqueFrameDataset(Dataset[tuple[str, Tensor]]):
    def __init__(
        self,
        split_directory: str | Path,
        transform: DualViewTransform,
        max_frames: int | None = None,
    ) -> None:
        self.split_directory = Path(split_directory)
        self.transform = transform
        self.paths = sorted(
            path
            for path in self.split_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if max_frames is not None:
            self.paths = self.paths[:max_frames]
        if not self.paths:
            raise ValueError(f"no image files found under {self.split_directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            views = self.transform(image)
        return path.relative_to(self.split_directory).as_posix(), views


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def precompute_features(
    config: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "auto",
    amp: bool = True,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> Path:
    model_config = config["model"]
    data_config = config["data"]
    destination = Path(output_path or data_config["feature_cache_path"])
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"feature cache already exists: {destination}; pass --overwrite to replace it"
        )
    target_device = resolve_device(device)
    backbone = create_vision_backbone(model_config).eval().to(target_device)
    backbone.requires_grad_(False)
    transform = build_dual_view_transform(
        backbone, float(model_config["center_crop_scale"])
    )
    split_directory = Path(data_config["path"]) / str(data_config["split"])
    frame_dataset = UniqueFrameDataset(split_directory, transform, max_frames)
    loader = DataLoader(
        frame_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    per_view_dim = int(backbone.num_features)
    num_views = int(model_config["num_views"])
    if num_views != 2:
        raise ValueError("dual-view precomputation requires model.num_views=2")
    all_features = torch.empty(
        len(frame_dataset), num_views * per_view_dim, dtype=torch.float16
    )
    all_filenames: list[str] = []
    offset = 0
    start = time.perf_counter()
    amp_enabled = amp and target_device.type in {"cuda", "mps"}
    amp_dtype = torch.float16
    for batch_index, (filenames, views) in enumerate(loader, start=1):
        views = views.to(target_device, non_blocking=True)
        batch, actual_views, channels, height, width = views.shape
        # Whole-frame and crop views are flattened into one large batch. This is
        # one backbone invocation, so both views are processed in parallel.
        with torch.autocast(
            device_type=target_device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            encoded = backbone(
                views.reshape(batch * actual_views, channels, height, width)
            )
        encoded = encoded.reshape(batch, actual_views * per_view_dim)
        all_features[offset : offset + batch].copy_(encoded.cpu().half())
        all_filenames.extend(filenames)
        offset += batch
        if batch_index == 1 or batch_index % 20 == 0 or offset == len(frame_dataset):
            elapsed = time.perf_counter() - start
            print(
                f"encoded={offset}/{len(frame_dataset)} "
                f"frames_per_second={offset / max(elapsed, 1e-6):.2f}"
            )

    metadata = {
        "format_version": 1,
        "vision_encoder_name": str(model_config["vision_encoder_name"]),
        "vision_feature_dim": per_view_dim,
        "num_views": num_views,
        "center_crop_scale": float(model_config["center_crop_scale"]),
        "input_size": list(transform.input_size),
        "feature_dtype": "float16",
        "frame_count": len(frame_dataset),
        "complete": max_frames is None,
    }
    return save_feature_cache(
        destination, all_features, all_filenames, metadata
    )


__all__ = ["UniqueFrameDataset", "precompute_features", "resolve_device"]
