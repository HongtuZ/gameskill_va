# GameSkill FQL 策略部署说明

本目录包含部署训练好的 GameSkill FQL 策略所需的全部文件，与 `final.pt` 一起分发：

| 文件 | 说明 |
| --- | --- |
| `policy.py` | 自包含的策略模块（模型结构 + 预处理 + 动作解码），不依赖训练仓库 |
| `demo_infer.py` | 命令行推理演示脚本 |
| `final.pt` | 训练 checkpoint（内嵌模型权重、训练配置、动作编解码参数） |

## 1. 环境依赖

```bash
pip install torch torchvision timm pillow numpy
# 本 checkpoint 训练时启用了音频融合，还需要：
pip install torchaudio transformers
# demo_infer.py 读取音频文件额外需要：
pip install soundfile
```

首次运行会自动下载两个预训练骨干：

- 视觉编码器 DINOv3（`vit_small_patch16_dinov3.lvd1689m`，经 timm/Hugging Face）
- 音频编码器 EAT（`worstchan/EAT-base_epoch30_finetune_AS2M`，经 Hugging Face）

离线机器需预先填充 Hugging Face 缓存，并设置 `HF_HUB_OFFLINE=1`。

## 2. 快速开始

```python
from policy import GameSkillPolicy

policy = GameSkillPolicy.from_checkpoint("final.pt", device="cuda")  # 或 "cpu" / "auto"

action = policy.infer(frames, audio_waveform=waveform)
print(action["mouse_move"])       # {"dx": -7.91, "dy": 0.87}
print(action["pressed_keys"])     # ["MoveForward", ...]
```

命令行演示：

```bash
python demo_infer.py \
    --checkpoint final.pt \
    --frames ./frames_dir \
    --audio ./audio_16k_mono.flac \
    --audio-start-seconds 12.5
```

## 3. 输入格式

### 3.1 视频帧 `frames`

- **数量**：`sequence_length = 10` 张**连续**截图，按时间顺序传入（列表/元组）。
  数量不符会直接报错；`policy.sequence_length` 可查询所需帧数。
- **格式**：任意分辨率的 RGB 图像，支持 `PIL.Image` 或 HWC 布局的 `numpy` 数组
  （`uint8`，shape `[H, W, 3]`）。
- **内部处理**（无需手动做）：每帧自动生成双视图——
  视图 0 为整帧缩放到 256×256，视图 1 为居中裁剪（短边的 75%）后缩放到 256×256，
  再做 ImageNet 均值/方差归一化。

### 3.2 音频 `audio_waveform`

此 checkpoint 训练时启用了音频融合，**推理必须提供音频**。

- **采样率**：必须重采样到 **16 kHz**。
- **声道**：**单声道**（多声道请先混音为 mono）。
- **长度**：**3 秒窗口，即 48000 个采样点**，`float32` 张量，形状 `[48000]`
  或 `[1, 48000]`；幅值范围建议 `[-1, 1]`。
- **时间对齐**：窗口应为**截至当前决策时刻的最近 3 秒**
  （窗口结束时刻 ≈ 最后一帧截图的时刻），与训练时的对齐方式一致。
  不足 3 秒（如启动初期）可在前端补零。

重采样示例：

```bash
# ffmpeg：任意音源 → 16 kHz 单声道 FLAC
ffmpeg -i source.mka -ar 16000 -ac 1 audio_16k_mono.flac
```

```python
# Python：从音频文件读取并重采样为 16 kHz 单声道
import torch, torchaudio

waveform, sr = torchaudio.load("source.mka")          # [C, S]
waveform = waveform.mean(dim=0, keepdim=True)          # 混单声道
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
window = waveform[0, -48000:]                          # 取最近 3 秒
if window.numel() < 48000:                             # 不足补零
    window = torch.nn.functional.pad(window, (48000 - window.numel(), 0))
```

> 替代方案：若你已离线预计算好 EAT 的 768 维 CLS 特征，可跳过波形输入，
> 直接传 `policy.infer(frames, audio_features=features)`，
> 其中 `features` 为形状 `[768]` 或 `[1, 768]` 的 `float` 张量。

## 4. 输出格式

`policy.infer(...)` 返回一个 `dict`，即**语义操作**：

```jsonc
{
  // 鼠标位移（像素），由 [-1,1] 归一化输出经 checkpoint 内置的
  // center/scale 解码而来，与训练数据的 mouse_move 同分布
  "mouse_move": { "dx": -7.912726, "dy": 0.870892 },

  // 31 个语义按键的按下概率，取值 [0, 1]
  "keyboard_probabilities": {
    "Fire": 0.1958,
    "MoveForward": 0.4588,
    "Jump": 0.1666,
    // ... 共 31 个键
  },

  // 概率 >= press_threshold（默认 0.5，构造时可配）的按键列表
  "pressed_keys": ["MoveForward"]
}
```

按键名列表（与训练标注一致，顺序即键盘输出维度顺序）：

```
Fire, AltFire, Ping, CyclePrimaryWeaponNext, CyclePrimaryWeaponPrev,
MoveForward, MoveBackward, MoveLeft, MoveRight, Jump, Crouch, Walk,
Activate_Ability1, Activate_Ability2, Activate_Ability3, Activate_Ultimate,
UseObject, Reload, EquipPrimaryWeapon, EquipSecondaryWeapon, EquipMeleeWeapon,
EquipSpike, DropEquippable, InspectWeapon, OpenWheel, RadioCommsMenu,
PushToTalk, OpenMap, ToggleScoreboard, OpenShop, Esc
```

部署侧执行建议：

- `mouse_move` 直接作为相对鼠标位移下发（正负方向与训练数据采集约定一致）。
- 键盘为多标签输出，`pressed_keys` 中多个键可同时按下；
  阈值可按需在 `from_checkpoint(..., press_threshold=...)` 调整。

## 5. API 参考

### `GameSkillPolicy.from_checkpoint(checkpoint_path, device="auto", press_threshold=0.5)`

加载 checkpoint 并构建策略实例：自动重建模型结构、挂载预训练 DINOv3 骨干、
严格校验权重匹配、恢复动作解码参数与 `sequence_length`，并置为 eval 模式。

### `policy.infer(frames, *, audio_features=None, audio_waveform=None, noise=None) -> dict`

执行一步推理，输入/输出见第 3、4 节。`noise` 默认取零向量（确定性输出，
与 ONNX 导出一致）；如需随机策略可传入形状 `[33]`/`[1, 33]` 的噪声张量。

### `policy.encode_audio_waveform(waveforms) -> Tensor`

将 `[B, 48000]` 的 16 kHz 波形编码为 `[B, 768]` EAT 特征。
EAT 编码器在首次调用时懒加载，之后复用。

### `policy.to_semantic_action(normalized_actions) -> dict`

将任意来源的 `[B, 33]` 归一化动作（前 2 维鼠标、后 31 维键盘）
转换为第 4 节的语义 dict。

## 6. 实时部署注意事项

- 每次决策需要维护一个长度为 10 的滑动帧窗口（按推理频率滚动更新），
  以及最近 3 秒的音频环形缓冲。
- 首次 `infer` 包含 EAT 编码器加载与 CUDA kernel 预热，耗时较长；
  建议部署后先用一批假数据 warmup 再进入控制回路。
- 策略为单步决策（无内部隐状态缓存），GRU 时序由每次传入的 10 帧窗口
  从头计算，天然支持任意时刻重启推理。
