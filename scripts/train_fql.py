#!/usr/bin/env python3
"""Configuration-driven Flow Q-Learning training entry point.

When the vision encoder is frozen, this script first checks whether the
precomputed dual-view feature cache exists locally. If it does, training
reuses that cache directly; otherwise the cache is generated from the raw
dataset before full training starts. Training metrics are recorded with
TensorBoard.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gameskill.config import load_config
from gameskill.data.precompute import precompute_features
from gameskill.training import train

# Non-main ranks wait this long for rank 0 to finish building the cache.
_CACHE_WAIT_TIMEOUT_SECONDS = 6 * 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fql_game.yaml"))
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; may be repeated.",
    )
    return parser.parse_args()


def _wait_for_cache(cache_path: Path, rank: int) -> None:
    deadline = time.monotonic() + _CACHE_WAIT_TIMEOUT_SECONDS
    while not cache_path.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f"rank {rank} timed out waiting for feature cache: {cache_path}")
        time.sleep(10)


def ensure_feature_cache(config: dict[str, Any]) -> None:
    """Prepare the frozen feature cache when the vision encoder is frozen."""
    model_config = config["model"]
    rank = int(os.environ.get("RANK", "0"))
    if not bool(model_config["freeze_vision_encoder"]):
        if rank == 0:
            print("Vision encoder is trainable; skipping precomputed feature cache.")
        model_config["use_precomputed_features"] = False
        return

    cache_path = Path(str(config["data"]["feature_cache_path"]))
    if cache_path.is_file():
        if rank == 0:
            print(f"Reusing existing feature cache: {cache_path}")
    elif rank == 0:
        print(f"Feature cache not found at {cache_path}; generating it from the dataset ...")
        saved_path = precompute_features(
            config,
            batch_size=int(config["data"].get("precompute_batch_size") or 8),
            num_workers=int(config["data"].get("precompute_num_workers") or 0),
            device=str(config["training"].get("device", "auto")),
        )
        print(f"Saved frozen dual-view feature cache to {saved_path}")
    else:
        print(f"rank {rank}: waiting for rank 0 to build the feature cache ...")
        _wait_for_cache(cache_path, rank)
    model_config["use_precomputed_features"] = True


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    ensure_feature_cache(config)
    train(config)


if __name__ == "__main__":
    main()
