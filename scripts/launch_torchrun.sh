#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONFIG_PATH="${CONFIG_PATH:-configs/fql_game.yaml}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

uv run torchrun \
  --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  scripts/train_fql.py \
  --config "${CONFIG_PATH}" \
  "$@"
