# -*- coding: utf-8 -*-
"""quality_gate.py — 红队语料质量门禁（极致化计划书 §四「质量门禁」）

对 factory meta.csv 中的每条 wav 做：
  1. 物理检查：存在性、时长（详册 §7.2 目标 3~6s，允许 1~10s 容差）、
     削波比例、静音占比；
  2. --reverse-screen：用通道①（AASIST）反向筛选——防伪模型高置信判真
     的样本是最有价值的难例（hard=True，优先入评估集）；
  3. 产出 quality_report.json + quarantine.csv（物理不合格隔离清单）。

用法：
  python quality_gate.py                      # 仅物理检查（零依赖重活）
  python quality_gate.py --reverse-screen     # 加通道①反向筛选（需 GPU/模型）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

META = ROOT / "data" / "redteam" / "factory" / "meta.csv"
REPORT = ROOT / "data" / "redteam" / "factory" / "quality_report.json"
QUAR = ROOT / "data" / "redteam" / "factory" / "quarantine.csv"

DUR_MIN, DUR_MAX = 1.0, 20.0       # 母本容差区间（详册 3~6s 是后续切分目标，非母本标准）
CLIP_RATIO_MAX = 0.01              # 削波样本占比上限
SILENCE_RATIO_MAX = 0.6            # 静音帧占比上限
HARD_SCORE_MAX = 0.3               # 通道① spoof 分 <0.3 = 高置信漏检难例


def read_wav_np(path: Path):
    """读 wav 为 float32 单声道 @16k（audio_util.read_wav 只回样本，采样率即目标值）。"""
    from server.audio_util import read_wav
    return read_wav(str(path), 16000), 16000


def physical_check(path: Path) -> dict:
    r = {"exists": path.exists()}
    if not r["exists"]:
        return r
    samples, sr = read_wav_np(path)
    x = np.asarray(samples, dtype=np.float32)
    if np.max(np.abs(x)) > 1.5:  # int16 量程
        x = x / 32768.0
    dur = len(x) / sr
    r["duration_s"] = round(dur, 2)
    r["duration_ok"] = DUR_MIN <= dur <= DUR_MAX
    r["clip_ratio"] = float(np.mean(np.abs(x) > 0.999)) if len(x) else 1.0
    r["clip_ok"] = r["clip_ratio"] <= CLIP_RATIO_MAX
    # 帧级 RMS 静音占比
    frame = sr // 50
    n = len(x) // frame
    if n:
        rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1))
        r["silence_ratio"] = float(np.mean(rms < 0.005))
    else:
        r["silence_ratio"] = 1.0
    r["silence_ok"] = r["silence_ratio"] <= SILENCE_RATIO_MAX
    r["pass"] = r["duration_ok"] and r["clip_ok"] and r["silence_ok"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reverse-screen", action="store_true")
    ap.add_argument("--screen-limit", type=int, default=200)
    args = ap.parse_args()

    if not META.exists():
        sys.exit(f"未找到 {META}，先跑 run_factory.py")

    with open(META, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"meta 共 {len(rows)} 条")

    acoustic = None
    if args.reverse_screen:
        from fusion.acoustic_channel import AcousticChannel
        acoustic = AcousticChannel()

    results, quarantine, n_hard = [], [], 0
    for i, row in enumerate(rows):
        p = ROOT / row["path"]
        chk = physical_check(p)
        rec = {**row, **chk}
        if not chk.get("pass"):
            quarantine.append(rec)
        elif acoustic is not None and row["channel"] == "clean" and i < args.screen_limit:
            # 反向筛选只对 clean 母本打标（信道版继承母本标签）
            try:
                r = acoustic.analyze(str(p), unload_after=False)
                rec["acoustic_spoof_score"] = round(float(r.score), 4)
                rec["hard"] = r.score < HARD_SCORE_MAX
                if rec["hard"]:
                    n_hard += 1
            except Exception as e:  # noqa: BLE001
                rec["screen_error"] = f"{type(e).__name__}: {e}"
        results.append(rec)

    n_pass = sum(1 for r in results if r.get("pass"))
    report = {
        "total": len(results),
        "physical_pass": n_pass,
        "quarantined": len(quarantine),
        "hard_examples": n_hard,
        "hard_note": f"通道① spoof 分 <{HARD_SCORE_MAX} 的高置信漏检难例（优先入评估集）",
        "thresholds": {"dur": [DUR_MIN, DUR_MAX],
                       "clip_ratio_max": CLIP_RATIO_MAX,
                       "silence_ratio_max": SILENCE_RATIO_MAX},
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if quarantine:
        with open(QUAR, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(quarantine[0].keys()))
            w.writeheader()
            w.writerows(quarantine)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告 -> {REPORT}" + (f"，隔离清单 -> {QUAR}" if quarantine else ""))


if __name__ == "__main__":
    main()
