"""按 dataset/train/metadata.jsonl 顺序逐帧显示图片，并为每条数据打 reward 标签。

用法:
    python3 show_frames.py [--dataset-dir dataset/train] [--start 0] [--delay 0]

标签规则:
    - reward 直接写回 metadata.jsonl 的每一行（新增/更新 "reward" 字段）；
    - 每条数据默认 reward=0；当前帧按 Space 在 0 → 1 → -1 → 0 间循环切换，
      切换后的奖励会应用到当前帧及其后所有帧，直到再次按 Space 切换；
    - 保存采用临时文件 + 原子替换，避免写坏原文件；重开自动继承已有标签。

交互:
    →/d/n   下一帧        ←/a/p   上一帧
    ]       前进 100 帧    [       后退 100 帧
    Space   循环切换 reward 0 → 1 → -1，并应用到当前帧及之后所有帧
    p       播放/暂停      +/-     播放间隔 ±10ms
    g       跳转到指定帧    s       立即保存标签    Esc/q  保存并退出
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

SKIP_KEYS = {"frame_index", "segment_id", "ts_ns", "file_names", "mouse_move", "reward"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fmt_time(ns: int) -> str:
    """纳秒时长 → HH:MM:SS.mmm 可读格式。"""
    total_ms, ms_rem = divmod(max(0, ns) // 1_000_000, 1000)
    total_s, s_rem = divmod(total_ms, 1000)
    h, m = divmod(total_s, 60)
    return f"{h:02d}:{m:02d}:{s_rem:02d}.{ms_rem:03d}"


def load_metadata(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r.setdefault("reward", 0)  # 继承已有标签，缺省 0
                rows.append(r)
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def save_metadata(path: Path, rows: list[dict]) -> None:
    """将带 reward 的全部行写回 metadata.jsonl（临时文件 + 原子替换）。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    os.replace(tmp, path)


def render_overlay(img, row: dict, idx: int, total: int, playing: bool, delay: int, t0: int):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 186), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    scale, thick = max(0.6, w / 2200), 2
    y = 34

    def put(text: str, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(img, text, (16, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += int(36 * scale + 8)

    state = "PLAY" if playing else "LABEL"
    put(f"[{idx + 1}/{total}]  {state}  delay={delay}ms", (0, 220, 255))

    fi = int(row["frame_index"])
    put(f"frame_index={fi}  ts_ns={row['ts_ns']}  time={fmt_time(int(row['ts_ns']) - t0)}", (255, 255, 255))

    # reward 状态：1 绿色，-1 红色，0 灰色
    rv = int(row.get("reward", 0))
    if rv == 1:
        put("reward=1", (0, 255, 0))
    elif rv == -1:
        put("reward=-1", (0, 0, 255))
    else:
        put("reward=0   (Space to toggle)", (170, 170, 170))

    dx, dy = row["mouse_move"]
    active = [k for k, v in row.items() if k not in SKIP_KEYS and v]
    tail = f"move=({dx}, {dy})"
    if active:
        tail += "  active: " + ", ".join(active)
    cv2.putText(img, tail, (16, h - 16), FONT, scale * 0.9, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="reward labeling viewer")
    ap.add_argument("--dataset-dir", default="dataset/train", help="含 metadata.jsonl 与图片的目录")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--delay", type=int, default=0, help="播放模式每帧停留毫秒，0=手动标注")
    args = ap.parse_args()

    ds_dir = Path(args.dataset_dir)
    meta_path = ds_dir / "metadata.jsonl"
    if not meta_path.is_file():
        raise SystemExit(f"metadata.jsonl not found in {ds_dir}")

    rows = load_metadata(meta_path)
    total = len(rows)
    if total == 0:
        raise SystemExit("metadata.jsonl is empty")
    t0 = int(rows[0]["ts_ns"])

    idx = max(0, min(args.start, total - 1))
    delay = args.delay
    playing = delay > 0
    n_pos = sum(1 for r in rows if int(r.get("reward", 0)) == 1)
    n_neg = sum(1 for r in rows if int(r.get("reward", 0)) == -1)
    print(f"loaded {total} rows (existing reward=1: {n_pos}, reward=-1: {n_neg})")
    print(f"labels will be written back to: {meta_path}")

    win = "reward labeling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def current_file(row: dict) -> str:
        # file_names 最后一个为当前帧图片
        return row["file_names"][-1]

    while True:
        row = rows[idx]
        img = cv2.imread(str(ds_dir / current_file(row)))
        if img is None:
            print(f"[warn] cannot read {current_file(row)}, skip")
            idx = (idx + 1) % total
            continue
        fi = int(row["frame_index"])
        render_overlay(img, row, idx, total, playing, delay, t0)
        cv2.imshow(win, img)

        key = cv2.waitKey(delay if playing else 0) & 0xFFFFFF
        if key in (27, ord("q")):  # Esc / q → 保存退出
            break
        elif key == ord(" "):  # Space: 循环切换，并应用到当前帧及之后所有帧
            new_val = {0: 1, 1: -1, -1: 0}[int(row.get("reward", 0))]
            for r in rows[idx:]:
                r["reward"] = new_val
            print(f"label frame {fi}..{int(rows[-1]['frame_index'])} -> reward={new_val}")
        elif key in (ord("d"), ord("n"), 3, 63235):  # → / d / n
            idx = min(idx + 1, total - 1)
        elif key in (ord("a"), ord("p"), 2, 63234):  # ← / a / p
            idx = max(idx - 1, 0)
        elif key in (ord("]"), 21):  # ] / PgDn
            idx = min(idx + 100, total - 1)
        elif key in (ord("["), 19):  # [ / PgUp
            idx = max(idx - 100, 0)
        elif key in (ord("p"), 80):  # p 播放/暂停（Space 已用于打标）
            playing = not playing
            if playing and delay == 0:
                delay = 100
        elif key in (ord("+"), ord("=")):
            delay = min(delay + 10, 1000)
        elif key == ord("-"):
            delay = max(delay - 10, 10)
        elif key == ord("s"):  # 手动保存
            save_metadata(meta_path, rows)
            print(f"saved {total} rows -> {meta_path}")
        elif key == ord("g"):
            print(f"\ngoto frame index [0-{total - 1}]: ", end="", flush=True)
            try:
                target = int(input())
                idx = max(0, min(target, total - 1))
            except ValueError:
                pass

        if playing:
            idx = (idx + 1) % total

    save_metadata(meta_path, rows)
    n_pos = sum(1 for r in rows if int(r.get("reward", 0)) == 1)
    n_neg = sum(1 for r in rows if int(r.get("reward", 0)) == -1)
    print(f"saved {total} rows -> {meta_path} (reward=1: {n_pos}, reward=-1: {n_neg})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
