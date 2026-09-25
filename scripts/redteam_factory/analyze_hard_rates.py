#!/usr/bin/env python
"""analyze_hard_rates.py — reverse-screen 难例率诊断（各引擎 hard 占比 + score 分布）

背景：quality_gate --reverse-screen 全量 hard_examples=2546 偏高（82%），
需确认是否 AASIST 真把红队合成音频判为真人（=高价值难例）还是流程问题。
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fusion.acoustic_channel import AcousticChannel  # noqa: E402


def main():
    ac = AcousticChannel()
    ac.load()
    rows = list(csv.DictReader(open(ROOT / "data/redteam/factory/meta.csv", encoding="utf-8")))
    clean = [r for r in rows if r["channel"] == "clean"]
    print(f"clean 母本 {len(clean)}")

    by_eng = defaultdict(list)
    for r in clean:
        by_eng[r["engine"]].append(r)

    per_eng = {}
    all_hard = []
    for eng, recs in by_eng.items():
        sample = random.Random(9).sample(recs, min(12, len(recs)))
        hard = 0
        scores = []
        for r in sample:
            p = ROOT / r["path"]
            try:
                v = ac.analyze(str(p), unload_after=False)
                sc = float(v.score)
                scores.append(sc)
                is_hard = sc < 0.3
                hard += is_hard
                if is_hard:
                    all_hard.append((eng, r["script_id"], r["path"], round(sc, 3)))
            except Exception as e:  # noqa: BLE001
                print(f"  [err] {r['path']}: {type(e).__name__}")
        n = len(sample)
        per_eng[eng] = {
            "n": n, "hard": hard, "rate": hard / max(1, n),
            "score_mean": sum(scores) / max(1, len(scores)),
            "scores": [round(s, 2) for s in scores],
        }
        print(f"{eng}: 抽 {n} 条 | hard {hard} ({hard/max(1,n)*100:.0f}%) | "
              f"score均值 {per_eng[eng]['score_mean']:.2f} | 分布 {per_eng[eng]['scores']}")

    # 方言维度
    by_dial = defaultdict(list)
    for eng, recs in by_eng.items():
        sample = random.Random(9).sample(recs, min(12, len(recs)))
        for r in sample:
            by_dial[r["dialect"]].append(recs[0])  # 简化占位
    print("\nhard 样例(前10):")
    for eng, sid, path, sc in all_hard[:10]:
        print(f"  hard {sc:.3f} [{eng}] {sid} {Path(path).name[:60]}")

    ac.unload()


if __name__ == "__main__":
    main()
