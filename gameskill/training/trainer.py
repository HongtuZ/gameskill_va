"""Configuration-driven DDP trainer for GameSkill FQL."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from gameskill.algorithms.flow_q_learning import FlowQLearning
from gameskill.data.action_codec import HybridActionCodec
from gameskill.data.transitions import CachedFeatureTransitionDataset, build_transition_dataset
from gameskill.training.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    reduce_metrics,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional; training still works without it.
    SummaryWriter = None


def _load_hf_dataset(config: dict[str, Any], context: DistributedContext) -> Dataset:
    data_config = config["data"]

    def load() -> Dataset:
        return load_dataset(
            "imagefolder",
            data_dir=str(data_config["path"]),
            split=str(data_config["split"]),
            cache_dir=data_config.get("cache_dir"),
        )

    if context.is_main:
        dataset = load()
    barrier(context)
    if not context.is_main:
        dataset = load()
    barrier(context)
    max_samples = data_config.get("max_samples")
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    return dataset


def _build_optimizer(model: FlowQLearning, config: dict[str, Any]) -> AdamW:
    optimizer_config = config["optimizer"]
    vision_encoder = model.state_encoder.vision_encoder
    vision_parameter_ids = (
        {id(parameter) for parameter in vision_encoder.parameters()} if vision_encoder is not None else set()
    )
    vision_parameters = (
        [parameter for parameter in vision_encoder.parameters() if parameter.requires_grad]
        if vision_encoder is not None
        else []
    )
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in vision_parameter_ids
    ]
    groups: list[dict[str, Any]] = [
        {
            "params": other_parameters,
            "lr": float(optimizer_config["learning_rate"]),
        }
    ]
    if vision_parameters:
        groups.append(
            {
                "params": vision_parameters,
                "lr": float(optimizer_config["vision_learning_rate"]),
            }
        )
    return AdamW(
        groups,
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def _build_scheduler(optimizer: AdamW, config: dict[str, Any], total_steps: int) -> LambdaLR:
    warmup_steps = int(config["optimizer"]["warmup_steps"])

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def _unwrap(model: nn.Module) -> FlowQLearning:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _train_batch_size(config: dict[str, Any]) -> int:
    """Action-training batch size per rank; legacy configs use data.batch_size."""
    data_config = config["data"]
    return int(data_config.get("train_batch_size") or data_config["batch_size"])


def _uses_gpu_dataset(config: dict[str, Any], transitions: Any) -> bool:
    """GPU-resident training needs the fully materialized cached-feature dataset."""
    return bool(config["data"].get("gpu_dataset")) and isinstance(
        transitions, CachedFeatureTransitionDataset
    )


class _GpuEpochBatchIterator:
    """Yields batches via on-GPU index gather, with per-epoch reshuffling.

    Keeping the whole transition dataset on the device removes DataLoader
    collate and host-to-device copies from the training loop, which otherwise
    starve the GPU when per-step compute is tiny.
    """

    def __init__(
        self,
        transitions: CachedFeatureTransitionDataset,
        batch_size: int,
        seed: int,
    ) -> None:
        self._transitions = transitions
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0
        self._position = 0
        self._order: Tensor | None = None

    def __iter__(self) -> "_GpuEpochBatchIterator":
        device = self._transitions._observations.device
        generator_device = device if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self._seed + self._epoch)
        order = torch.randperm(
            len(self._transitions), generator=generator, device=generator_device
        )
        self._order = order.to(device)
        self._position = 0
        self._epoch += 1
        return self

    def __next__(self) -> dict[str, Tensor]:
        if self._order is None or self._position >= len(self._order):
            raise StopIteration
        indices = self._order[self._position : self._position + self._batch_size]
        self._position += self._batch_size
        return self._transitions.gather(indices)


def _resolve_total_steps(
    config: dict[str, Any], num_samples: int, world_size: int, gpu_dataset: bool
) -> int:
    """Effective optimizer-step budget; epochs override total_steps when set."""
    epochs = config["training"].get("epochs")
    if not epochs:
        return int(config["training"]["total_steps"])
    batch_size = _train_batch_size(config)
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    # GPU-resident sampling draws without replacement from the full dataset on
    # every rank, so epochs consume samples world_size times faster per step
    # than with a DistributedSampler.
    per_rank_samples = (
        num_samples
        if gpu_dataset or world_size <= 1
        else math.ceil(num_samples / world_size)
    )
    batches_per_epoch = math.ceil(per_rank_samples / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    return max(1, steps_per_epoch * int(epochs))


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    codec: HybridActionCodec,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "action_codec": codec.state_dict(),
            "global_step": global_step,
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    model: FlowQLearning,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["global_step"])


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train(config: dict[str, Any]) -> None:
    context = initialize_distributed(str(config["training"]["device"]))
    try:
        _train(config, context)
    finally:
        cleanup_distributed(context)


def _train(config: dict[str, Any], context: DistributedContext) -> None:
    seed = int(config["seed"]) + context.rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    if context.is_main:
        print(f"Loading imagefolder dataset from {config['data']['path']} ...")
    hf_dataset = _load_hf_dataset(config, context)
    reward_column = str(config["data"]["reward_column"])
    if reward_column not in hf_dataset.column_names and not bool(config["data"]["allow_missing_reward"]):
        raise ValueError(
            f"dataset has no {reward_column!r} column. FQL cannot train a critic "
            "without real rewards. Add the column to metadata.jsonl; use "
            "data.allow_missing_reward=true only for smoke testing."
        )
    codec = HybridActionCodec.fit(
        hf_dataset,
        quantile=float(config["data"]["mouse_scale_quantile"]),
        minimum_scale=float(config["data"]["minimum_mouse_scale"]),
    )
    raw_model = FlowQLearning(config["model"], config["algorithm"])
    image_transform = (
        raw_model.state_encoder.get_dual_view_transform() if raw_model.state_encoder.has_vision_encoder else None
    )
    transitions = build_transition_dataset(
        hf_dataset,
        image_transform,
        codec,
        config["data"],
        config["model"],
    )

    gpu_dataset = _uses_gpu_dataset(config, transitions)
    if gpu_dataset:
        transitions.to(context.device)
        sampler = None
        iterator_source: Any = _GpuEpochBatchIterator(
            transitions, _train_batch_size(config), seed
        )
        if context.is_main:
            print(
                "gpu_dataset enabled: transitions reside on-device; DataLoader skipped"
            )
    else:
        if context.distributed:
            sampler = DistributedSampler(
                transitions,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=int(config["seed"]),
                drop_last=False,
            )
        else:
            sampler = RandomSampler(transitions)
        loader = DataLoader(
            transitions,
            batch_size=_train_batch_size(config),
            sampler=sampler,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=context.device.type == "cuda",
            persistent_workers=int(config["data"]["num_workers"]) > 0,
            drop_last=False,
        )
        iterator_source = loader

    raw_model.to(context.device)
    optimizer = _build_optimizer(raw_model, config)
    total_steps = _resolve_total_steps(
        config, len(transitions), context.world_size, gpu_dataset
    )
    scheduler = _build_scheduler(optimizer, config, total_steps)
    amp_enabled = bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and context.device.type == "cuda")
    global_step = 0
    resume_path = config["training"].get("resume")
    if resume_path:
        global_step = _load_checkpoint(resume_path, raw_model, optimizer, scheduler, scaler)

    if context.distributed:
        ddp_kwargs: dict[str, Any] = {}
        if context.device.type == "cuda":
            ddp_kwargs = {
                "device_ids": [context.local_rank],
                "output_device": context.local_rank,
            }
        model: nn.Module = DistributedDataParallel(raw_model, **ddp_kwargs)
    else:
        model = raw_model

    output_dir = Path(config["training"]["output_dir"])
    writer = None
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.jsonl"
        tensorboard_dir = config["logging"].get("tensorboard_dir") or str(output_dir / "tensorboard")
        if SummaryWriter is not None:
            writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard logging to {tensorboard_dir}")
        else:
            print("tensorboard is not installed; skipping TensorBoard logging")
        parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
        trainable_count = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
        print(
            f"world_size={context.world_size} device={context.device} "
            f"samples={len(transitions)} batch_size_per_rank={_train_batch_size(config)} "
            f"total_steps={total_steps}"
        )
        print(
            f"action_dim={codec.action_dim} mouse_center={codec.mouse_center.tolist()} "
            f"mouse_scale={codec.mouse_scale.tolist()}"
        )
        print(f"parameters: total={parameter_count:,} trainable={trainable_count:,}")
    barrier(context)

    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    log_every = int(config["logging"]["log_every"])
    save_every = int(config["training"]["save_every"])
    no_save = bool(config["training"]["no_save"])
    amp_dtype = torch.float16 if context.device.type in {"cuda", "mps"} else torch.bfloat16
    epoch = 0
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
    iterator = iter(iterator_source)
    model.train()
    start_time = time.perf_counter()

    while global_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_metrics: dict[str, Tensor] = {}
        for micro_step in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
                iterator = iter(iterator_source)
                batch = next(iterator)
            batch = _move_batch(batch, context.device)
            synchronize = micro_step + 1 == accumulation_steps
            sync_context = (
                nullcontext() if synchronize or not isinstance(model, DistributedDataParallel) else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    metrics = model(batch)
                    loss = metrics["loss"] / accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            for key, value in metrics.items():
                accumulated_metrics[key] = (
                    accumulated_metrics.get(key, torch.zeros_like(value.detach())) + value.detach() / accumulation_steps
                )

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), float(config["training"]["max_grad_norm"]))
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        raw_model.soft_update_target()
        global_step += 1

        accumulated_metrics["gradient_norm"] = torch.as_tensor(gradient_norm, device=context.device)
        if global_step == 1 or global_step % log_every == 0:
            reduced = reduce_metrics(accumulated_metrics, context)
            reduced["step"] = global_step
            reduced["learning_rate"] = scheduler.get_last_lr()[0]
            reduced["elapsed_seconds"] = time.perf_counter() - start_time
            if context.is_main:
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={reduced['loss']:.6f} "
                    f"critic={reduced['critic_loss']:.6f} "
                    f"flow={reduced['bc_flow_loss']:.6f} "
                    f"distill={reduced['distill_loss']:.6f} "
                    f"q={reduced['q_mean']:.4f} "
                    f"reward={reduced['reward_mean']:.4f}"
                )
                with metrics_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(reduced) + "\n")
                if writer is not None:
                    for key, value in reduced.items():
                        if key == "step":
                            continue
                        writer.add_scalar(f"train/{key}", float(value), global_step)

        if not no_save and save_every > 0 and global_step % save_every == 0:
            if context.is_main:
                _save_checkpoint(
                    output_dir / f"step_{global_step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    codec,
                    global_step,
                )
            barrier(context)

    if context.is_main:
        if not no_save:
            _save_checkpoint(
                output_dir / "final.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                codec,
                global_step,
            )
            print(f"Saved final checkpoint to {output_dir / 'final.pt'}")
        if writer is not None:
            writer.close()
        print(f"Training completed in {time.perf_counter() - start_time:.1f}s")


__all__ = ["train"]
