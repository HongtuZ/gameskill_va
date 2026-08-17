"""Frozen EAT audio encoder and its official 16 kHz fbank preprocessing."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def create_audio_backbone(config: dict[str, Any]) -> nn.Module:
    """Load the Hugging Face EAT model only when precomputation is requested."""
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise ImportError(
            "EAT precomputation requires transformers. Run `uv sync` first."
        ) from error
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", True))
    }
    if config.get("revision"):
        kwargs["revision"] = str(config["revision"])
    return AutoModel.from_pretrained(str(config["encoder_name"]), **kwargs)


def waveforms_to_eat_fbank(waveforms: Tensor, config: dict[str, Any]) -> Tensor:
    """Convert ``[B,S]`` 16 kHz waveforms to normalized ``[B,1,T,128]`` fbanks."""
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError(
            "EAT preprocessing requires torchaudio. Run `uv sync` first."
        ) from error
    if waveforms.ndim != 2:
        raise ValueError(f"waveforms must be [B,S], got {tuple(waveforms.shape)}")
    sample_rate = int(config["sample_rate"])
    target_length = int(config["mel_target_length"])
    mel_bins = int(config["mel_bins"])
    fbanks: list[Tensor] = []
    # torchaudio's Kaldi-compatible fbank operates on one waveform at a time.
    # It is deliberately kept on CPU; the frozen EAT forward is batched on GPU.
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


def extract_audio_cls(backbone: nn.Module, fbanks: Tensor) -> Tensor:
    """Return EAT's utterance-level CLS representation as ``[B,D]``."""
    if not hasattr(backbone, "extract_features"):
        raise TypeError("configured audio model does not expose extract_features")
    tokens = backbone.extract_features(fbanks)
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError(
            "EAT extract_features must return [B,tokens,D], got "
            f"{type(tokens).__name__} {getattr(tokens, 'shape', None)}"
        )
    return tokens[:, 0]


__all__ = [
    "create_audio_backbone",
    "extract_audio_cls",
    "waveforms_to_eat_fbank",
]
