# GameSkill Flow Q-Learning

This package trains a DINOv3 + EAT game policy with Flow Q-Learning (FQL). Network
definitions, algorithm losses, dataset conversion, distributed training, and
deployment export are intentionally kept in separate directories.

## Layout

```text
configs/fql_game.yaml                   Formal training configuration
gameskill/data/                         HF transitions, action codec, multimodal cache
gameskill/models/                       DINOv3, EAT, fusion, flow actor, twin Q
gameskill/algorithms/flow_q_learning.py FQL objectives and target update
gameskill/training/                     DDP runtime, checkpoints, metrics
gameskill/export/                       ONNX deployment wrapper
scripts/train_fql.py                    Python/torchrun training entry
scripts/launch_torchrun.sh              Convenient torchrun launcher
scripts/add_audio_metadata.py           Timestamp-to-audio-window alignment
scripts/precompute_features.py          Frozen DINOv3 + EAT precomputation
scripts/test_dual_view_pipeline.py      One-image shape and forward smoke test
scripts/export_onnx.py                  ONNX export entry
```

The standalone two-head network is
`gameskill.models.GameSkillVisionPolicy`: `mouse_policy` emits two bounded
continuous values, while `keyboard_policy` emits 31 discrete-action logits.
The FQL agent shares the same multimodal `VisionStateEncoder`; its deployable
one-step flow actor also has separate `mouse_policy` and `keyboard_policy`
output layers before concatenating the 33-D hybrid action.

## Audio alignment

Each metadata row receives exactly one audio window: the 3 seconds immediately
before that row's `ts_ns`. It is associated with the current/final frame only;
it is not expanded once for every image in `file_names`. The waveform is always
48,000 mono samples at 16 kHz. Time outside the recorded stream is zero-padded,
including rows whose entire window has no recorded audio. There is no validity
ratio or mask: a silent/padded waveform is processed normally by EAT.

Prepare the dataset audio and update `metadata.jsonl`:

```bash
mkdir -p dataset/train/audio
ffmpeg -i audio.mka -map 0:a:0 -ac 1 -ar 16000 -c:a flac \
  dataset/train/audio/audio_16k_mono.flac
cp audio_meta.json dataset/train/audio/audio_meta.json
uv run python scripts/add_audio_metadata.py \
  --metadata dataset/train/metadata.jsonl \
  --audio-meta audio_meta.json \
  --audio-path audio/audio_16k_mono.flac \
  --in-place
```

In-place mode first creates `metadata.jsonl.before_audio` and refuses to
overwrite an existing backup. Each row stores the audio path, clipped source
sample interval, and left/right zero-padding counts.

## Dual-view vision input and frozen multimodal cache

Every source frame is converted into two normalized `256 x 256` views:

1. the complete frame resized to `256 x 256`;
2. a centered square crop, with side length equal to 75% of the source image's
   shorter edge, resized to `256 x 256`.

The crop ratio is configurable with `model.center_crop_scale`. Both views are
flattened into the same leading batch dimension and passed through DINOv3 in a
single call. Their two 384-D outputs are concatenated into one 768-D feature.

Because DINOv3 and EAT are frozen, generate both feature types once:

```bash
uv run python scripts/precompute_features.py \
  --config configs/fql_game.yaml \
  --batch-size 8 \
  --audio-batch-size 32 \
  --device cuda
```

Use `--device mps` on Apple Silicon, or omit it for automatic selection. Tune
the precompute batch size independently from the RL training batch size. The
cache stores one FP16 visual row per unique source image and one FP16 768-D EAT
CLS feature per metadata row. For 18,621 rows, the raw visual and audio tensors
are about 54.6 MiB combined, plus small filename/frame indices.

The default config enables `model.use_precomputed_features`. In this mode the
training process instantiates neither frozen encoder and does not decode images
or audio. It supplies visual `[B, 10, 768]` and audio `[B, 768]` features to the
trainable GRU/fusion network. The preprocessing follows EAT's official 16 kHz,
128-bin Kaldi fbank normalization, using the utterance-level CLS token.
Because this Hugging Face model executes custom code, the configuration pins
the exact EAT repository revision tested by this project.

Verify the two views, fusion input, and both policy heads with local DINOv3:

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

Distributed CUDA training, where `data.train_batch_size` is per GPU:

```bash
NPROC_PER_NODE=4 bash scripts/launch_torchrun.sh
```

Any YAML value can be overridden without editing the file:

```bash
NPROC_PER_NODE=2 bash scripts/launch_torchrun.sh \
  --set data.train_batch_size=1 \
  --set training.amp=true
```

For a short cache-backed smoke test:

```bash
HF_HUB_OFFLINE=1 NPROC_PER_NODE=1 bash scripts/launch_torchrun.sh \
  --set data.max_samples=100 \
  --set training.total_steps=1 \
  --set training.no_save=true
```

## ONNX export

The exported policy reattaches the locally downloaded frozen DINOv3 weights and
takes normalized dual-view images `[B, 10, 2, 3, 256, 256]`, an externally
precomputed EAT CLS feature `[B, 768]`, and Gaussian noise `[B, 33]`. Keeping
EAT outside this ONNX graph makes the online audio ring-buffer and fbank stage
explicit. View index 0 is the whole-frame resize and view index 1 is the center
crop. The outputs are mouse movement `[B, 2]` and keyboard probabilities
`[B, 31]`.

```bash
uv run python scripts/export_onnx.py \
  --checkpoint checkpoints/fql_game/final.pt \
  --output exports/game_skill_fql.onnx
```
