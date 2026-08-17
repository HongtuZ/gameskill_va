#!/usr/bin/env python3
"""Cache dual-view DINOv3 and aligned 3-second EAT audio features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gameskill.config import load_config  # noqa: E402
from gameskill.data.precompute import precompute_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fql_game.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Source frames per batch; DINOv3 receives twice this many views.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--audio-batch-size",
        type=int,
        default=None,
        help="Audio windows per EAT batch; defaults to audio.precompute_batch_size.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Debug-only partial cache."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.audio_batch_size is not None and args.audio_batch_size <= 0:
        raise ValueError("audio-batch-size must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    path = precompute_features(
        load_config(args.config, args.overrides),
        output_path=args.output,
        batch_size=args.batch_size,
        audio_batch_size=args.audio_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    print(f"Saved frozen dual-view feature cache to {path}")


if __name__ == "__main__":
    main()
