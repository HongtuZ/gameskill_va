"""Precompute frozen dual-view DINOv3 and per-transition EAT features."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
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
from gameskill.models.audio import (
    create_audio_backbone,
    extract_audio_cls,
    waveforms_to_eat_fbank,
)


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class AudioWindow:
    frame_index: int
    start: int
    end: int
    pad_left: int
    pad_right: int


def _load_audio_windows(
    split_directory: Path, max_rows: int | None
) -> tuple[Path, list[AudioWindow], int, int]:
    metadata_path = split_directory / "metadata.jsonl"
    windows: list[AudioWindow] = []
    stored_audio_path: str | None = None
    sample_rate: int | None = None
    window_samples: int | None = None
    with metadata_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "frame_index", "audio_path", "audio_sample_rate",
                "audio_window_samples", "audio_start_sample", "audio_end_sample",
                "audio_pad_left_samples", "audio_pad_right_samples",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"missing audio metadata {sorted(missing)} at "
                    f"{metadata_path}:{line_number}; run scripts/add_audio_metadata.py"
                )
            current_path = str(row["audio_path"])
            current_rate = int(row["audio_sample_rate"])
            current_window = int(row["audio_window_samples"])
            if stored_audio_path is None:
                stored_audio_path = current_path
                sample_rate = current_rate
                window_samples = current_window
            elif (current_path, current_rate, current_window) != (
                stored_audio_path, sample_rate, window_samples
            ):
                raise ValueError("all metadata rows must use one audio file/rate/window")
            window = AudioWindow(
                frame_index=int(row["frame_index"]),
                start=int(row["audio_start_sample"]),
                end=int(row["audio_end_sample"]),
                pad_left=int(row["audio_pad_left_samples"]),
                pad_right=int(row["audio_pad_right_samples"]),
            )
            if window.start < 0 or window.end < window.start:
                raise ValueError(f"invalid audio slice at {metadata_path}:{line_number}")
            if window.pad_left + (window.end - window.start) + window.pad_right != current_window:
                raise ValueError(f"audio window is not fixed length at {metadata_path}:{line_number}")
            windows.append(window)
            if max_rows is not None and len(windows) >= max_rows:
                break
    if not windows or stored_audio_path is None or sample_rate is None or window_samples is None:
        raise ValueError(f"no audio-aligned rows found in {metadata_path}")
    return split_directory / stored_audio_path, windows, sample_rate, window_samples


def _load_mono_audio(path: Path, expected_rate: int) -> Tensor:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError("audio precomputation requires soundfile; run `uv sync`") from error
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != expected_rate:
        raise ValueError(
            f"audio sample rate is {sample_rate}, expected {expected_rate}: {path}"
        )
    if samples.shape[1] != 1:
        raise ValueError(f"precomputed audio must be mono, got {samples.shape[1]} channels")
    return torch.from_numpy(samples[:, 0].copy())


@torch.inference_mode()
def _precompute_audio_features(
    split_directory: Path,
    audio_config: dict[str, Any],
    target_device: torch.device,
    batch_size: int,
    amp: bool,
    max_rows: int | None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    audio_path, windows, sample_rate, window_samples = _load_audio_windows(
        split_directory, max_rows
    )
    if sample_rate != int(audio_config["sample_rate"]):
        raise ValueError("metadata audio sample rate differs from audio.sample_rate")
    if window_samples != int(audio_config["window_samples"]):
        raise ValueError("metadata audio window differs from audio.window_samples")
    waveform = _load_mono_audio(audio_path, sample_rate)
    if max(window.end for window in windows) > waveform.numel():
        raise ValueError("metadata references samples beyond the decoded audio file")
    backbone = create_audio_backbone(audio_config).eval().to(target_device)
    backbone.requires_grad_(False)
    feature_dim = int(audio_config["feature_dim"])
    all_features = torch.empty(len(windows), feature_dim, dtype=torch.float16)
    all_frame_indices = torch.tensor(
        [window.frame_index for window in windows], dtype=torch.long
    )
    amp_enabled = amp and target_device.type in {"cuda", "mps"}
    start_time = time.perf_counter()
    for offset in range(0, len(windows), batch_size):
        batch_windows = windows[offset : offset + batch_size]
        batch_waveforms = torch.zeros(
            len(batch_windows), window_samples, dtype=torch.float32
        )
        for row, window in enumerate(batch_windows):
            valid = waveform[window.start : window.end]
            batch_waveforms[row, window.pad_left : window.pad_left + valid.numel()] = valid
        fbanks = waveforms_to_eat_fbank(batch_waveforms, audio_config).to(
            target_device, non_blocking=True
        )
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            encoded = extract_audio_cls(backbone, fbanks)
        if encoded.shape != (len(batch_windows), feature_dim):
            raise ValueError(
                f"EAT returned {tuple(encoded.shape)}, expected "
                f"({len(batch_windows)}, {feature_dim})"
            )
        all_features[offset : offset + len(batch_windows)].copy_(
            encoded.cpu().half()
        )
        completed = offset + len(batch_windows)
        if offset == 0 or completed % (batch_size * 20) == 0 or completed == len(windows):
            elapsed = time.perf_counter() - start_time
            print(
                f"audio_encoded={completed}/{len(windows)} "
                f"windows_per_second={completed / max(elapsed, 1e-6):.2f}"
            )
    metadata = {
        "audio_encoder_name": str(audio_config["encoder_name"]),
        "audio_encoder_revision": str(audio_config.get("revision") or ""),
        "audio_feature_dim": feature_dim,
        "audio_sample_rate": sample_rate,
        "audio_window_samples": window_samples,
        "audio_mel_bins": int(audio_config["mel_bins"]),
        "audio_mel_target_length": int(audio_config["mel_target_length"]),
        "audio_row_count": len(windows),
        "audio_source_path": str(audio_path),
    }
    return all_features, all_frame_indices, metadata


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
    audio_batch_size: int | None = None,
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
        "format_version": 2,
        "vision_encoder_name": str(model_config["vision_encoder_name"]),
        "vision_feature_dim": per_view_dim,
        "num_views": num_views,
        "center_crop_scale": float(model_config["center_crop_scale"]),
        "input_size": list(transform.input_size),
        "feature_dtype": "float16",
        "frame_count": len(frame_dataset),
        "complete": max_frames is None,
    }
    # The two frozen encoders run sequentially. Release DINOv3 before loading
    # EAT so peak precomputation memory is the larger encoder, not their sum.
    del backbone
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    audio_features: Tensor | None = None
    audio_frame_indices: Tensor | None = None
    audio_config = config.get("audio", {})
    if bool(audio_config.get("enabled", False)):
        audio_features, audio_frame_indices, audio_metadata = _precompute_audio_features(
            split_directory,
            audio_config,
            target_device,
            int(audio_batch_size or audio_config.get("precompute_batch_size") or batch_size),
            amp,
            max_frames,
        )
        metadata.update(audio_metadata)
    return save_feature_cache(
        destination,
        all_features,
        all_filenames,
        metadata,
        audio_features=audio_features,
        audio_frame_indices=audio_frame_indices,
    )


__all__ = ["UniqueFrameDataset", "precompute_features", "resolve_device"]
