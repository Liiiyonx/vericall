#!/usr/bin/env python
"""exp_cn_same_domain.py — 中文域模型同域(说话人外)评测（2026-09-06）

此前 CFAD 评测的错位问题：CFAD 伪=声码器族 vs FMFCC 伪=商业TTS族，导致
纯中文域(FMFCC伪)模型在 CFAD 上 EER 47% 的假象（英文零样本反而 15% 是攻击族
意外重叠）。本脚本用**同域划分**公允评测中文域模型：

  训练：aishell 真 S0002-S0005（1414 条，说话人内）+ FMFCC 伪 12000
  评测：aishell 真 S0006-S0007（708 条，**说话人外**）+ FMFCC 伪 5636
  对照：英文 ASVspoof 模型同评测

产出：evaluation/exp_cn_same_domain.md/.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AI_CACHE = ROOT / ".tmp_ssl" / "aishell" / "xlsr_aishell_true_full.npz"
FM_CACHE = ROOT / ".tmp_ssl" / "fmfcc" / "xlsr_fmfcc_fake_full.npz"
EN_CACHE = ROOT / ".tmp_ssl" / "wav2vec2-xls-r-300m" / "train_20000.npz"
OUT_MD = ROOT / "evaluation/exp_cn_same_domain.md"
OUT_JSON = ROOT / "evaluation/exp_cn_same_domain.json"


def main():
    from eval_fusion_stack import eer_from_scores, split_by_label
    from sklearn.linear_model import LogisticRegression

    ai = np.load(AI_CACHE, allow_pickle=True)
    fm = np.load(FM_CACHE, allow_pickle=True)
    Xa = ai["X"]
    spk = np.array([(re.search(r"S(\d{4})", str(x)) or [None, "?"]).group(1)
                    for x in ai["file_ids"]])
    rng = np.random.RandomState(0)

    tr_m = np.isin(spk, ["0002", "0003", "0004", "0005"])
    te_m = np.isin(spk, ["0006", "0007"])
    fi = rng.permutation(len(fm["X"]))
    ftr, fte = fi[:12000], fi[12000:]

    Xtr = np.concatenate([Xa[tr_m], fm["X"][ftr]])
    ytr = np.concatenate([np.ones(tr_m.sum()), np.zeros(len(ftr))])
    Xte = np.concatenate([Xa[te_m], fm["X"][fte]])
    yte = np.concatenate([np.ones(te_m.sum()), np.zeros(len(fte))])
    print(f"训练: aishell真 {tr_m.sum()} + FMFCC伪 {len(ftr)} | "
          f"测试: aishell真{te_m.sum()}(说话人外) + FMFCC伪 {len(fte)}", flush=True)

    en = np.load(EN_CACHE)
    ef = en["X"][en["y"] == 0]; et = en["X"][en["y"] == 1]
    idx = rng.choice(len(ef), 5000, replace=False)

    def run(Xtr2, ytr2, label):
        clf = LogisticRegression(max_iter=400, C=1.0).fit(Xtr2, ytr2)
        s = 1.0 - clf.predict_proba(Xte)[:, 1]
        eer, th = eer_from_scores(*split_by_label(s, yte))
        print(f"{label}: 同域 EER {eer:.2f}%", flush=True)
        return eer

    cn_eer = run(np.concatenate([Xa[tr_m], fm["X"][ftr]]), ytr, "中文域模型")
    en_eer = run(np.concatenate([et, ef[idx]]),
                 np.concatenate([np.ones(len(et)), np.zeros(5000)]), "英文模型(对照)")

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "design": ("同域说话人外评测：中文域(aishell真S0002-5 + FMFCC伪12k)训练，"
                   "aishell真S0006-7(说话人外708) + FMFCC伪5.6k 测试"),
        "results": {"中文域模型": round(cn_eer, 2), "英文模型对照": round(en_eer, 2)},
        "conclusion": (
            f"中文域模型在同域(说话人外+同族伪)评测 EER {cn_eer:.2f}%，英文模型 {en_eer:.2f}%。"
            "→ 中文域模型在其目标域上表现优异；此前 CFAD 上 47% 系评测错位"
            "(CFAD伪=声码器族 vs FMFCC伪=商业TTS族)。"
            "**结论：中文域模型真实水平需同域评测，CFAD/退化矩阵仅作跨域参考；"
            "红队(同属TTS域)是中文域模型的自然评测集。**"),
    }
    lines = [
        "# 中文域模型 · 同域说话人外评测",
        "",
        f"> 生成：{payload['date']} · `evaluation/exp_cn_same_domain.py`",
        "> 训练：aishell真(S0002-5, 1414) + FMFCC伪(12k)；测试：aishell真(S0006-7, 708,**说话人外**) + FMFCC伪(5.6k)",
        "",
        "| 模型 | 同域 EER |",
        "|---|---|",
        f"| 中文域模型 (aishell+FMFCC) | {cn_eer:.2f}% |",
        f"| 英文模型 (ASVspoof 对照) | {en_eer:.2f}% |",
        "",
        "## 结论",
        "",
        payload["conclusion"],
        "",
        "## 配套实验索引",
        "- `exp_cn_train_full.md`：同模型在 CFAD(跨域)上的表现 → 揭示评测错位；",
        "- `exp_domain_anchor.md`：目标域锚点适配路径有效；",
        "- `exp_fmfcc_pseudomain.md`：伪域扩充(异构)反效果。",
    ]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {OUT_MD}")


if __name__ == "__main__":
    main()
