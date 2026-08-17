#!/usr/bin/env python3
"""Export a GameSkill FQL checkpoint as a one-step ONNX policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gameskill.export import export_onnx_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("exports/game_skill_fql.onnx"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = export_onnx_policy(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        batch_size=args.batch_size,
        verify=not args.no_verify,
    )
    print(f"Exported and verified ONNX policy: {path}")


if __name__ == "__main__":
    main()
