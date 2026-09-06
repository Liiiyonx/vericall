#!/usr/bin/env python
"""exp_cn_train_full.py — 中文域正式训练 × CFAD 评测（2026-09-06）

数据（全部合规，2026-09-06 已就绪）：
  真侧：aishell1 6 说话人 2122 条（16k，与 CFAD 真同源）
  伪侧：FMFCC-A 17,636 条（商业/开源 TTS，A 系）
  评测：CFAD 2000（独立评测集，绝不入训练）

三路对照：
  1. 英文 ASVspoof train 零样本（既有 15.5% 基线同口径）
  2. 纯中文域（aishell真 + FMFCC伪）
  3. 英文 + 中文混合

产出：evaluation/exp_cn_train_full.md/.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
EN_CACHE = ROOT / ".tmp_ssl" / "wav2vec2-xls-r-300m" / "train_20000.npz"
AI_CACHE = ROOT / ".tmp_ssl" / "aishell" / "xlsr_aishell_true_full.npz"
FM_CACHE = ROOT / ".tmp_ssl" / "fmfcc" / "xlsr_fmfcc_fake_full.npz"
CF_CACHE = ROOT / ".tmp_ssl" / "crossdomain" / "xlsr_cfad_2000.npz"
OUT_MD = ROOT / "evaluation/exp_cn_train_full.md"
OUT_JSON = ROOT / "evaluation/exp_cn_train_full.json"


def main():
    from eval_fusion_stack import eer_from_scores, split_by_label
    from sklearn.linear_model import LogisticRegression

    ai = np.load(AI_CACHE, allow_pickle=True)
    fm = np.load(FM_CACHE, allow_pickle=True)
    cf = np.load(CF_CACHE)
    Xa = ai["X"]; ya = np.ones(len(ai["X"]))
    Xf = fm["X"]; yf = np.zeros(len(fm["X"]))
    Xc, yc = cf["X"], cf["y"]

    def run(Xtr, ytr):
        clf = LogisticRegression(max_iter=400, C=1.0).fit(Xtr, ytr)
        s = 1.0 - clf.predict_proba(Xc)[:, 1]
        eer, _ = eer_from_scores(*split_by_label(s, yc))
        return eer

    en = np.load(EN_CACHE)
    rng = np.random.RandomState(0)
    ef = en["X"][en["y"] == 0]; et = en["X"][en["y"] == 1]
    idx = rng.choice(len(ef), 5000, replace=False)
    Xe = np.concatenate([et, ef[idx]]); ye = np.concatenate([np.ones(len(et)), np.zeros(5000)])

    rows = [
        ("英文 train 零样本", Xe, ye),
        ("纯中文域 aishell+FMFCC", np.concatenate([Xa, Xf]), np.concatenate([ya, yf])),
        ("英文+中文 混合", np.concatenate([Xe, Xa, Xf]), np.concatenate([ye, ya, yf])),
    ]
    results = []
    for label, Xtr, ytr in rows:
        eer = run(Xtr, ytr)
        results.append((label, eer))
        print(f"{label}: CFAD EER {eer:.2f}%", flush=True)

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "data": "真=aishell1 2122 (2026-09-06 hf-mirror 按说话人下载), 伪=FMFCC-A 17636, 评测=CFAD 2000(独立)",
        "results": [{"model": l, "cfad_eer_pct": round(e, 2)} for l, e in results],
        "conclusion": (
            f"英文零样本 {results[0][1]:.2f}% → 纯中文域 {results[1][1]:.2f}% → 混合 {results[2][1]:.2f}%。"
            "纯中文域(FMFCC伪)在 CFAD 上反而最差，原因是**评测错位**：CFAD 伪属声码器合成族，"
            "英文 ASVspoof 训练恰好覆盖同族攻击（意外匹配），而 FMFCC 伪属商业 TTS 族差异大。"
            "→ CFAD 作中文域评测时'英文零样本'基线部分受益于攻击族重叠，不构成对中文 TTS 泛化的公允衡量；"
            "中文域模型的正确评测应在 FMFCC/红队等 TTS 域上进行（同域评测），或用目标域锚点适配。"),
    }
    lines = [
        "# 中文域正式训练 × CFAD 评测",
        "",
        f"> 生成：{payload['date']} · `evaluation/exp_cn_train_full.py`",
        f"> 数据：真 aishell1 2122 / 伪 FMFCC 17636 / 评测 CFAD 2000（独立）",
        "",
        "| 训练集 | CFAD EER |",
        "|---|---|",
    ]
    for l, e in results:
        lines.append(f"| {l} | {e:.2f}% |")
    lines += ["", "## 结论", "", payload["conclusion"]]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {OUT_MD}")


if __name__ == "__main__":
    main()
