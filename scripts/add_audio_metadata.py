#!/usr/bin/env python3
"""Attach one fixed-length pre-frame audio window to each metadata row.

The audio window belongs to the row itself (the final/current frame), not to
every image in ``file_names``.  Its interval is ``[ts_ns-context, ts_ns)``.
Samples outside the recorded audio stream are represented by left/right zero
padding so every row always describes exactly the same number of samples.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("dataset/train/metadata.jsonl"))
    parser.add_argument("--audio-meta", type=Path, default=Path("audio_meta.json"))
    parser.add_argument(
        "--audio-path",
        default="audio/audio_16k_mono.flac",
        help="Path stored in metadata, relative to the dataset split directory.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--context-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace --metadata after creating a backup.",
    )
    parser.add_argument("--backup-suffix", default=".before_audio")
    return parser.parse_args()


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round an exact rational to the nearest integer, symmetric around zero."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        return -_round_ratio(-numerator, denominator)
    return (numerator + denominator // 2) // denominator


def _load_alignment(path: Path, sample_rate: int) -> tuple[list[dict[str, int]], int]:
    with path.open(encoding="utf-8") as file:
        audio_meta = json.load(file)
    source_rate = int(audio_meta["sample_rate_hz"])
    accounting = audio_meta["sample_accounting"]
    anchors = sorted(
        (
            {
                "capture_ns": int(anchor["capture_ns"]),
                "device_position_frames": int(anchor["device_position_frames"]),
            }
            for anchor in accounting["anchors"]
        ),
        key=lambda item: item["capture_ns"],
    )
    if len(anchors) < 2:
        raise ValueError("audio_meta.json must contain at least two timing anchors")
    if any(
        right["capture_ns"] <= left["capture_ns"]
        for left, right in zip(anchors, anchors[1:])
    ):
        raise ValueError("audio timing anchors must have strictly increasing timestamps")
    delivered_frames = int(accounting["delivered_device_frames"])
    total_samples = _round_ratio(delivered_frames * sample_rate, source_rate)
    for anchor in anchors:
        anchor["source_rate"] = source_rate
    return anchors, total_samples


def _timestamp_to_sample(ts_ns: int, anchors: list[dict[str, int]], sample_rate: int) -> int:
    pair = (anchors[0], anchors[1])
    for left, right in zip(anchors, anchors[1:]):
        pair = (left, right)
        if ts_ns <= right["capture_ns"]:
            break
    else:
        pair = (anchors[-2], anchors[-1])
    left, right = pair
    delta_time = right["capture_ns"] - left["capture_ns"]
    delta_frames = right["device_position_frames"] - left["device_position_frames"]
    # Interpolate in the original device-frame clock and resample the exact
    # rational position in one operation to avoid cumulative rounding drift.
    source_position_numerator = (
        left["device_position_frames"] * delta_time
        + (ts_ns - left["capture_ns"]) * delta_frames
    )
    denominator = delta_time * left["source_rate"]
    return _round_ratio(source_position_numerator * sample_rate, denominator)


def _window_fields(
    ts_ns: int,
    anchors: list[dict[str, int]],
    sample_rate: int,
    window_samples: int,
    total_samples: int,
    audio_path: str,
) -> dict[str, int | str]:
    raw_end = _timestamp_to_sample(ts_ns, anchors, sample_rate)
    raw_start = raw_end - window_samples
    source_start = min(max(raw_start, 0), total_samples)
    source_end = min(max(raw_end, 0), total_samples)
    source_length = max(0, source_end - source_start)
    pad_left = min(window_samples, max(0, -raw_start))
    pad_right = window_samples - pad_left - source_length
    if pad_right < 0:
        raise AssertionError("audio window accounting produced negative padding")
    return {
        "audio_path": audio_path,
        "audio_sample_rate": sample_rate,
        "audio_window_samples": window_samples,
        "audio_start_sample": source_start,
        "audio_end_sample": source_end,
        "audio_pad_left_samples": pad_left,
        "audio_pad_right_samples": pad_right,
    }


def add_audio_metadata(
    metadata_path: Path,
    audio_meta_path: Path,
    output_path: Path,
    *,
    audio_path: str,
    sample_rate: int,
    context_seconds: float,
) -> tuple[int, int, int]:
    if sample_rate <= 0 or context_seconds <= 0:
        raise ValueError("sample rate and context duration must be positive")
    window_samples = round(sample_rate * context_seconds)
    anchors, total_samples = _load_alignment(audio_meta_path, sample_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    partial_count = 0
    silence_count = 0
    with metadata_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            if "ts_ns" not in row:
                raise ValueError(f"missing ts_ns at {metadata_path}:{line_number}")
            fields = _window_fields(
                int(row["ts_ns"]),
                anchors,
                sample_rate,
                window_samples,
                total_samples,
                audio_path,
            )
            row.update(fields)
            valid_samples = int(fields["audio_end_sample"]) - int(
                fields["audio_start_sample"]
            )
            padding = int(fields["audio_pad_left_samples"]) + int(
                fields["audio_pad_right_samples"]
            )
            if valid_samples + padding != window_samples:
                raise AssertionError(f"invalid window accounting at row {line_number}")
            partial_count += int(0 < valid_samples < window_samples)
            silence_count += int(valid_samples == 0)
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            destination.write("\n")
            row_count += 1
        destination.flush()
        os.fsync(destination.fileno())
    return row_count, partial_count, silence_count


def main() -> None:
    args = parse_args()
    if args.in_place == (args.output is not None):
        raise ValueError("choose exactly one of --in-place or --output")
    metadata_path = args.metadata.resolve()
    if args.in_place:
        backup_path = metadata_path.with_name(metadata_path.name + args.backup_suffix)
        if backup_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing metadata backup: {backup_path}"
            )
        shutil.copy2(metadata_path, backup_path)
        temporary = tempfile.NamedTemporaryFile(
            mode="w", prefix=metadata_path.name + ".", suffix=".tmp",
            dir=metadata_path.parent, delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            counts = add_audio_metadata(
                metadata_path,
                args.audio_meta.resolve(),
                temporary_path,
                audio_path=args.audio_path,
                sample_rate=args.sample_rate,
                context_seconds=args.context_seconds,
            )
            os.replace(temporary_path, metadata_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        output_path = metadata_path
        print(f"Backup saved to {backup_path}")
    else:
        output_path = args.output.resolve()
        counts = add_audio_metadata(
            metadata_path,
            args.audio_meta.resolve(),
            output_path,
            audio_path=args.audio_path,
            sample_rate=args.sample_rate,
            context_seconds=args.context_seconds,
        )
    rows, partial, silence = counts
    print(
        f"Updated {rows} rows at {output_path}; partial={partial}, "
        f"silence={silence}, window={round(args.sample_rate * args.context_seconds)} samples"
    )


if __name__ == "__main__":
    main()
