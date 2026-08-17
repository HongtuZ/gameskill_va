# GameSkill Flow Q-Learning

This package trains the DINOv2 game policy with Flow Q-Learning (FQL). Network
definitions, algorithm losses, dataset conversion, distributed training, and
deployment export are intentionally kept in separate directories.

## Layout

```text
configs/fql_game.yaml                   Formal training configuration
gameskill/data/                         HF transitions, action codec, feature cache
gameskill/models/                       DINOv2, GRU, flow actor, one-step actor, twin Q
gameskill/algorithms/flow_q_learning.py FQL objectives and target update
gameskill/training/                     DDP runtime, checkpoints, metrics
gameskill/export/                       ONNX deployment wrapper
scripts/train_fql.py                    Python/torchrun training entry
scripts/launch_torchrun.sh              Convenient torchrun launcher
scripts/precompute_features.py          Frozen dual-view DINOv2 precomputation
scripts/test_dual_view_pipeline.py      One-image shape and forward smoke test
scripts/export_onnx.py                  ONNX export entry
```

The standalone two-head network is
`gameskill.models.GameSkillVisionPolicy`: `mouse_policy` emits two bounded
continuous values, while `keyboard_policy` emits 31 discrete-action logits.
The FQL agent shares the same dual-view `VisionStateEncoder`; its deployable
one-step flow actor also has separate `mouse_policy` and `keyboard_policy`
output layers before concatenating the 33-D hybrid action.

## Dual-view vision input and frozen-feature cache

Every source frame is converted into two normalized `518 x 518` views:

1. the complete frame resized to `518 x 518`;
2. a centered square crop, with side length equal to 75% of the source image's
   shorter edge, resized to `518 x 518`.

The crop ratio is configurable with `model.center_crop_scale`. Both views are
flattened into the same leading batch dimension and passed through DINOv2 in a
single call. Their two 384-D outputs are concatenated into one 768-D feature.

Because DINOv2 is frozen, generate the feature cache once before training:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/precompute_features.py \
  --config configs/fql_game.yaml \
  --batch-size 8 \
  --device cuda
```

Use `--device mps` on Apple Silicon, or omit it for automatic selection. Tune
the precompute batch size independently from the RL training batch size. The
cache stores one FP16 row per unique source image rather than one copy per
10-frame sliding window. For 18,621 frames, its raw feature tensor is about
27.3 MiB (`18621 x 768 x 2` bytes), plus a small filename index.

The default config enables `model.use_precomputed_features`. In this mode the
training process does not instantiate DINOv2 and does not decode the image
column; it directly supplies `[B, 10, 768]` tensors to the temporal GRU. To test
end-to-end pixel training instead, set `model.use_precomputed_features=false`.

Verify the two views, concatenated feature, and both policy heads with the local
DINOv2 weights:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/test_dual_view_pipeline.py
```

## Reward data

Offline Q-learning needs transitions containing `(state, action, reward,
next_state, done)`. State, action, next state, and episode boundaries can be
derived from the existing rows and `segment_id`. Every metadata row must carry
a scalar reward, for example:

```json
{"reward": 0.5, "done": false}
```

`done` is optional because segment boundaries are automatically terminal.
Reward must reflect the actual game objective; a fabricated constant reward
cannot teach the critic a useful policy.

## Training

Single process:

```bash
uv run python scripts/train_fql.py --config configs/fql_game.yaml
```

Distributed CUDA training, where `data.batch_size` is per GPU:

```bash
NPROC_PER_NODE=4 bash scripts/launch_torchrun.sh
```

Any YAML value can be overridden without editing the file:

```bash
uv run torchrun --standalone --nproc-per-node=2 scripts/train_fql.py \
  --config configs/fql_game.yaml \
  --set data.batch_size=1 \
  --set training.amp=true
```

For a short cache-backed smoke test:

```bash
HF_HUB_OFFLINE=1 uv run torchrun --standalone --nproc-per-node=1 \
  scripts/train_fql.py --config configs/fql_game.yaml \
  --set data.max_samples=100 \
  --set training.total_steps=1 \
  --set training.no_save=true
```

## ONNX export

The exported model reattaches the locally downloaded frozen DINOv2 weights and
takes already transformed, normalized dual-view images
`[B, 10, 2, 3, 518, 518]` plus Gaussian noise `[B, 33]`. View index 0 is the
whole-frame resize and view index 1 is the center crop. It returns de-normalized
mouse movement `[B, 2]` and keyboard probabilities `[B, 31]`. Use zero noise
for a deterministic policy or sampled normal noise for stochastic actions.

```bash
uv run python scripts/export_onnx.py \
  --checkpoint checkpoints/fql_game/final.pt \
  --output exports/game_skill_fql.onnx
```
