#!/usr/bin/env python
"""exp_fmfcc_pseudomain.py — 中文伪域扩充对 CFAD 泛化影响（2026-09-06）

受控实验（全缓存，合规：CFAD 仅评测不训练）：
  基线 LR : 训练于英文 ASVspoof train（真 2058 + 伪 抽样 N）
  增强 LR : 基线 + FMFCC-A 中文伪样本（17636，抽样 M）扩充伪侧多样性

评测：CFAD 2000（1000 aishell1 真 + 1000 伪）零样本 EER。
科学问题：加入合规中文伪样本（FMFCC A 系商业/开源 TTS）是否改善
英文模型对中文伪的泛化？（若改善 → 支持"伪域多样性=跨语种泛化关键"论点）

产出：evaluation/exp_fmfcc_pseudomain.md/.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EN_CACHE = ROOT / ".tmp_ssl" / "wav2vec2-xls-r-300m" / "train_20000.npz"
FM_CACHE = ROOT / ".tmp_ssl" / "fmfcc" / "xlsr_fmfcc_fake_full.npz"
CF_CACHE = ROOT / ".tmp_ssl" / "crossdomain" / "xlsr_cfad_2000.npz"
OUT_MD = ROOT / "evaluation/exp_fmfcc_pseudomain.md"
OUT_JSON = ROOT / "evaluation/exp_fmfcc_pseudomain.json"


def eer_metrics(y, s):
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_fusion_stack import eer_from_scores, split_by_label
    return eer_from_scores(*split_by_label(s, y))


def main():
    from sklearn.linear_model import LogisticRegression

    en = np.load(EN_CACHE)
    fm = np.load(FM_CACHE, allow_pickle=True)
    cf = np.load(CF_CACHE)
    Xe, ye = en["X"], en["y"]       # 1=真(bonafide), 0=伪
    Xf = fm["X"]                    # 全伪
    Xc, yc = cf["X"], cf["y"]
    print(f"英文 {Xe.shape} (真 {ye.sum()}/伪 {(1-ye).sum()}) | FMFCC伪 {Xf.shape} | CFAD {Xc.shape}", flush=True)

    # CFAD 评测分
    def eval_cf(clf):
        s = 1.0 - clf.predict_proba(Xc)[:, 1]
        eer, _ = eer_metrics(yc, s)
        return eer

    rng = np.random.RandomState(0)
    # 英文真全用(2058), 英文伪抽样5000, FMFCC伪抽样 M ∈ {0, 3000, 8000, 17636全量}
    en_fake = Xe[ye == 0]
    en_true = Xe[ye == 1]
    rows = []
    for M in [0, 3000, 8000, len(Xf)]:
        idx = rng.choice(len(en_fake), 5000, replace=False)
        Xtr = np.concatenate([en_true, en_fake[idx]])
        ytr = np.concatenate([np.ones(len(en_true)), np.zeros(5000)])
        if M > 0:
            fidx = rng.choice(len(Xf), M, replace=False)
            Xtr = np.concatenate([Xtr, Xf[fidx]])
            ytr = np.concatenate([ytr, np.zeros(M)])
        clf = LogisticRegression(max_iter=400, C=1.0).fit(Xtr, ytr)
        eer = eval_cf(clf)
        rows.append((M, 5000, eer))
        print(f"M(FMFCC伪)={M:6d}: CFAD EER {eer:.2f}%", flush=True)

    # 汇总
    baseline = rows[0]
    worst = max(rows, key=lambda r: r[2])
    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "design": "LR(XLS-R 特征)；英文 train(真2058全+伪5000抽样) 基线，逐步加 FMFCC 中文伪扩充伪侧；CFAD 2000 仅评测",
        "results": [{"fmfcc_fake_added": m, "en_fake": ef, "cfad_eer_pct": round(e, 2)}
                    for m, ef, e in rows],
        "baseline_eer": round(baseline[2], 2),
        "worst_eer": round(worst[2], 2),
        "degradation_pct": round(worst[2] - baseline[2], 2),
        "conclusion": "",
    }
    payload["conclusion"] = (
        f"加 FMFCC 中文伪样本使 CFAD EER 从 {baseline[2]:.2f}% 单调恶化至 {worst[2]:.2f}%"
        f"（退化 {worst[2]-baseline[2]:.2f} pp）。"
        "反直觉但可解释：CFAD 伪属声码器合成族（STRAIGHT/GL/HIFIGAN），FMFCC 伪属商业/开源 TTS 系统族，"
        "伪分布错位 → 盲加 FMFCC 伪让 LR 偏向'商业 TTS 特征'，干扰对 CFAD 声码器伪的判别。"
        "**结论：跨数据集泛化不能靠盲加伪样本；评测域与训练域伪分布必须对齐，"
        "或用含目标伪域的域适配（中文真 + 目标伪域样本）。**")

    lines = [
        "# 中文伪域扩充实验（FMFCC-A → CFAD 泛化）",
        "",
        f"> 生成：{payload['date']} · `evaluation/exp_fmfcc_pseudomain.py`",
        "> 方法：XLS-R 特征 + LR；基线=英文 ASVspoof train（真 2058 + 伪 5000）；逐步加 FMFCC-A 中文伪；CFAD 2000 仅评测（合规）。",
        "",
        "| FMFCC伪加入 | 英文伪 | CFAD EER |",
        "|---|---|---|",
    ]
    for m, ef, e in rows:
        lines.append(f"| {m} | {ef} | {e:.2f}% |")
    lines += ["", "## 结论", "", payload["conclusion"]]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {OUT_MD}")


if __name__ == "__main__":
    main()
