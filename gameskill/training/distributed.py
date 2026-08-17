"""Small torch.distributed runtime helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(requested_device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank if distributed else 0)
        elif torch.backends.mps.is_available() and not distributed:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device("cuda", local_rank if distributed else 0)
    elif requested_device == "mps":
        if distributed:
            raise RuntimeError("multi-process MPS training is not supported; use CUDA or CPU")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device, distributed)


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def reduce_metrics(
    metrics: dict[str, Tensor], context: DistributedContext
) -> dict[str, float]:
    keys = sorted(metrics)
    values = torch.stack([metrics[key].detach().float() for key in keys])
    if context.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= context.world_size
    return {key: value.item() for key, value in zip(keys, values)}


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "DistributedContext",
    "barrier",
    "cleanup_distributed",
    "initialize_distributed",
    "reduce_metrics",
]
