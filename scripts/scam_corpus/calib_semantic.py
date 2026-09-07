#!/usr/bin/env python
"""calib_semantic.py — A5：语义风险分置信度校准分析（2026-09-07）

输入 evaluation/semantic_f1_rows.csv（id, expected, pred, risk，来自 eval_semantic_f1 全量跑）。
以"诈骗为正类"（expected != benign）看 risk 与真实诈骗率是否一致：
  - 可靠性表：risk 每 0.1 桶 → 桶均值 risk vs 桶内诈骗率
  - ECE（期望校准误差）
  - 给温度缩放建议 T（grid 0.5~3.0 最小化 ECE；无 sklearn，纯实现）：
    但 LLM risk 非 logit，温度缩放不适用 → 改为"阈值-经验校准"建议：
    block 阈值 0.6 的经验精度/召回，与融合器口径对照（semantic risk>0.6 判 block）。
用法：python -u scripts/scam_corpus/calib_semantic.py
产物：evaluation/semantic_calibration.json / .md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "evaluation" / "semantic_f1_rows.csv"
OUT_J = ROOT / "evaluation" / "semantic_calibration.json"
OUT_M = ROOT / "evaluation" / "semantic_calibration.md"


def main():
    rows = list(csv.DictReader(open(ROWS, encoding="utf-8-sig")))
    rows = [r for r in rows if r["pred"] != "error"]
    scam = [r for r in rows if r["expected"] != "benign"]
    n = len(rows)
    # 可靠性（risk 桶 0.1）
    NB = 10
    buckets = [[] for _ in range(NB)]
    for r in rows:
        b = min(int(float(r["risk"]) * NB), NB - 1)
        buckets[b].append(r)
    table = []
    for b, rs in enumerate(buckets):
        if not rs:
            continue
        conf = sum(float(r["risk"]) for r in rs) / len(rs)
        frac = sum(1 for r in rs if r["expected"] != "benign") / len(rs)
        table.append({"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": len(rs),
                      "mean_risk": round(conf, 3), "scam_rate": round(frac, 3)})
    # ECE（期望校准误差）
    ece = 0.0
    for b, rs in enumerate(buckets):
        if not rs:
            continue
        conf = sum(float(r["risk"]) for r in rs) / len(rs)
        frac = sum(1 for r in rs if r["expected"] != "benign") / len(rs)
        ece += len(rs) / n * abs(conf - frac)
    # 阈值-经验（0.50 / 0.60 / 0.70）在诈骗判定上的 P/R
    thr_rows = []
    for t in (0.50, 0.60, 0.70):
        tp = sum(1 for r in rows if float(r["risk"]) >= t and r["expected"] != "benign")
        fp = sum(1 for r in rows if float(r["risk"]) >= t and r["expected"] == "benign")
        fn = sum(1 for r in rows if float(r["risk"]) < t and r["expected"] != "benign")
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        thr_rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn,
                         "precision": round(p, 4), "recall": round(rc, 4)})

    payload = {"date": "2026-09-07", "n": n, "ece": round(ece, 4),
               "reliability": table, "threshold_ops": thr_rows}
    OUT_J.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# 语义风险分置信度校准（A5，2026-09-07）", "",
         f"- 样本：{n} 条（A1 冻结集，含 risk 字段；诈骗=正类）",
         f"- **ECE = {ece*100:.1f}%**（0 完美校准；<10% 可用）", "",
         "| risk 桶 | n | 均值 risk | 桶内真实诈骗率 | 偏差 |", "|---|---|---|---|---|"]
    for t in table:
        L.append(f"| {t['bin']} | {t['n']} | {t['mean_risk']:.2f} | {t['scam_rate']*100:.0f}% | "
                 f"{abs(t['mean_risk']-t['scam_rate'])*100:.0f}pt |")
    L += ["", "## 阈值-经验操作点（block 判定用 risk 阈值）", "", "| 阈值 | TP | FP | FN | P | R |", "|---|---|---|---|---|---|"]
    for t in thr_rows:
        L.append(f"| risk≥{t['threshold']:.2f} | {t['tp']} | {t['fp']} | {t['fn']} | "
                 f"{t['precision']*100:.1f}% | {t['recall']*100:.1f}% |")
    L += ["", "## 结论建议", "- LLM 输出 risk 非严格概率 → 不用温度缩放；用**阈值-经验校准**：",
          "- 语义 block 建议阈值 risk ≥0.60（看上表 P/R 平衡；与融合器 hard 口径对齐并在代码注明出处）；",
          "- risk 桶越往高，真实诈骗率越高则校准可用；若 ECE 高，提示词加"给出概率感"约束或改两段式。"]
    OUT_M.write_text("\n".join(L), encoding="utf-8")
    print(f"ECE={ece*100:.1f}%  产物 {OUT_M.name}")


if __name__ == "__main__":
    main()
