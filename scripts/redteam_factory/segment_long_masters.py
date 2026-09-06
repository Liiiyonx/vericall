#!/usr/bin/env python
"""segment_long_masters.py — 超长隔离母本静音感知切段（2026-09-06）

把 quarantine 隔离的 clean 超长母本（>20s）切成 3~6s 语义段：
  1. 静音检测（rms 阈值）找语音活跃区；
  2. 贪婪拼接活跃帧成 3~6s 段，段界落在静音处（不切句中）；
  3. 输出到 data/redteam/factory/segments/<engine>/<dialect>/seg_<母本名>_<i>.wav；
  4. 更新 meta（engine/dialect/script_id 继承母本）。

用途：扩大红队评测样本池（红队禁入训练管线，段仅作评测/难度分析）。
用法：python -u scripts/redteam_factory/segment_long_masters.py [--min-sec 3] [--max-sec 6] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QUAR = ROOT / "data/redteam/factory/quarantine.csv"
META = ROOT / "data/redteam/factory/meta.csv"
OUT_ROOT = ROOT / "data/redteam/factory/segments"

SIL_RMS = 0.01     # 静音 rms 阈值（母本已过门禁 silence_ratio<0.6，0.01 保守）
HOP_MS = 20        # 帧长


def _rms_frames(wav, sr, hop):
    n = len(wav)
    idx = np.arange(0, max(1, n - hop), hop)
    frames = np.array([wav[i:i + hop] for i in idx])
    return np.sqrt((frames ** 2).mean(axis=1)), idx


def split_by_silence(wav, sr, min_sec, max_sec, hop):
    """返回段边界列表 [(start_idx, end_idx), ...]（采样点）。"""
    rms, idx = _rms_frames(wav, sr, hop)
    voice = rms >= SIL_RMS
    max_frames = int(max_sec * sr / hop)
    min_frames = int(min_sec * sr / hop)

    segments = []
    cur_start = None
    cur_voice = 0
    i = 0
    nf = len(voice)
    while i < nf:
        if voice[i]:
            if cur_start is None:
                cur_start = i
            cur_voice += 1
            # 达到 max → 从最近的静音处截断；无静音则强制截
            if cur_voice >= max_frames:
                seg_end = i + 1
                # 往回找静音界（最近 1s 内的静音点）
                lookback = max(cur_start, i - int(sr / hop))
                cut = i + 1
                for j in range(i, lookback - 1, -1):
                    if not voice[j]:
                        cut = j
                        break
                if cut - cur_start >= min_frames:
                    segments.append((idx[cur_start], idx[cut]))
                    cur_start = None
                    cur_voice = 0
                    i = cut
                    continue
        else:
            # 静音：若已攒够 min 且静音>=3帧，收段
            if cur_start is not None and cur_voice >= min_frames:
                segments.append((idx[cur_start], idx[i]))
                cur_start = None
                cur_voice = 0
        i += 1
    # 收尾
    if cur_start is not None and cur_voice >= min_frames:
        segments.append((idx[cur_start], idx[min(nf - 1, cur_start + cur_voice)]))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    import librosa

    quar = [r for r in csv.DictReader(open(QUAR, encoding="utf-8-sig"))
            if r["channel"] == "clean"]
    if args.limit:
        quar = quar[: args.limit]
    print(f"隔离 clean 母本 {len(quar)} 条", flush=True)

    new_rows, n_seg, n_err = [], 0, 0
    for i, r in enumerate(quar):
        p = ROOT / r["path"]
        if not p.exists():
            n_err += 1
            continue
        try:
            wav, sr = sf.read(str(p))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            # 重采样到 16k（统一评测采样率）
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                sr = 16000
            segs = split_by_silence(wav, sr, args.min_sec, args.max_sec,
                                    int(HOP_MS * sr / 1000))
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {r['path']}: {type(e).__name__}", flush=True)
            n_err += 1
            continue

        for si, (a, b) in enumerate(segs):
            if b - a < int(args.min_sec * sr):
                continue
            sub = wav[a:b]
            ddir = OUT_ROOT / r["engine"] / r["dialect"]
            ddir.mkdir(parents=True, exist_ok=True)
            stem = Path(r["path"]).stem
            out = ddir / f"seg_{stem}_{si:02d}.wav"
            sf.write(str(out), sub, sr)
            dur = (b - a) / sr
            new_rows.append({"path": str(out.relative_to(ROOT)).replace("\\", "/"),
                             "label": "spoof", "attack_id": r["attack_id"],
                             "channel": "clean", "source": r["source"],
                             "license": r["license"], "engine": r["engine"],
                             "dialect": r["dialect"], "script_id": r["script_id"],
                             "speaker_ref": r["speaker_ref"],
                             "duration_s": f"{dur:.1f}", "seg": True,
                             "parent": r["path"]})
            n_seg += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(quar)}] 累计 {n_seg} 段", flush=True)

    # 追加到 meta.csv（表头不同需合并）
    meta_path = META
    if new_rows:
        import json
        with open(meta_path, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            existing = list(rd)
            hdr = rd.fieldnames if rd.fieldnames else []
        fields = list(hdr)
        for k in new_rows[0]:
            if k not in fields:
                fields.append(k)
        with open(meta_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in existing:
                w.writerow(row)
            for row in new_rows:
                w.writerow(row)
    print(f"\n切段完成：{n_seg} 段（{n_err} 条失败），meta 已追加")
    print(f"输出 -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
