#!/usr/bin/env python3
"""Demo: run one GameSkill FQL policy step from screenshots and audio.

Ship together with ``policy.py`` and ``final.pt``:

    python demo_infer.py --checkpoint final.pt \
        --frames ./frames_dir \
        --audio ./audio_16k_mono.flac

- ``--frames`` is a folder of consecutive screenshots; the first
  ``sequence_length`` (10) images, sorted by name, are used.
- ``--audio`` is any 16 kHz mono audio file; a 3 s window starting at
  ``--audio-start-seconds`` is fed to the policy (required by audio-trained
  checkpoints). Omit it only for vision-only checkpoints.

The predicted semantic action is printed as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from policy import GameSkillPolicy

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_frames(directory: Path, count: int) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if len(paths) < count:
        raise ValueError(f"{directory} has only {len(paths)} images, need {count}")
    return paths[:count]


def load_audio_window(audio_path: Path, start_seconds: float) -> torch.Tensor:
    """Return a [48000] float32 waveform (3 s at 16 kHz, zero-padded)."""
    import soundfile as sf

    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if int(sample_rate) != 16000:
        raise ValueError(f"--audio must be 16 kHz mono, got {sample_rate} Hz")
    waveform = torch.from_numpy(samples[:, 0].copy())
    start = int(start_seconds * 16000)
    window = waveform[start : start + 48000]
    if window.numel() < 48000:
        window = torch.nn.functional.pad(window, (0, 48000 - window.numel()))
    return window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="folder of consecutive screenshots",
    )
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--audio-start-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--press-threshold",
        type=float,
        default=0.5,
        help="keyboard probability above which a key counts as pressed",
    )
    args = parser.parse_args()

    policy = GameSkillPolicy.from_checkpoint(
        args.checkpoint, device=args.device, press_threshold=args.press_threshold
    )
    print(f"device: {policy.device} | audio_required: {policy.use_audio}")

    frame_paths = collect_frames(args.frames, policy.sequence_length)
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    print(f"frames: {[path.name for path in frame_paths]}")

    audio_waveform = None
    if args.audio is not None:
        audio_waveform = load_audio_window(args.audio, args.audio_start_seconds)

    action = policy.infer(frames, audio_waveform=audio_waveform)
    print(
        json.dumps(
            {
                "mouse_move": action["mouse_move"],
                "pressed_keys": action["pressed_keys"],
                "keyboard_probabilities": action["keyboard_probabilities"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
