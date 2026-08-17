#!/usr/bin/env python3
"""Smoke-test dual-view preprocessing, parallel DINOv2, and policy heads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gameskill.config import load_config  # noqa: E402
from gameskill.models import GameSkillVisionPolicy  # noqa: E402
from gameskill.models.vision import (  # noqa: E402
    build_dual_view_transform,
    create_vision_backbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fql_game.yaml"))
    parser.add_argument(
        "--image", type=Path, default=Path("dataset/train/00000000.jpg")
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    backbone = create_vision_backbone(config["model"]).eval().to(device)
    transform = build_dual_view_transform(
        backbone, float(config["model"]["center_crop_scale"])
    )
    with Image.open(args.image) as image:
        source_size = image.size
        views = transform(image)

    # Shape [2,3,H,W] means whole image and center crop enter one DINO call as
    # a batch. They are not processed by two serial Python forward calls.
    per_view_features = backbone(views.to(device))
    concatenated = per_view_features.reshape(1, 1, -1).cpu()
    policy = GameSkillVisionPolicy(config["model"]).eval()
    sequence = concatenated.repeat(1, int(config["data"]["sequence_length"]), 1)
    mouse, keyboard_logits = policy(sequence)
    print(f"source_size={source_size}")
    print(f"dual_views={tuple(views.shape)}")
    print(f"single_dinov2_batch={tuple(per_view_features.shape)}")
    print(f"concatenated_features={tuple(concatenated.shape)}")
    print(f"mouse_policy={tuple(mouse.shape)}")
    print(f"keyboard_policy={tuple(keyboard_logits.shape)}")


if __name__ == "__main__":
    main()
