#!/usr/bin/env python
"""fmfcc_redteam_proximity.py — 红队 vs FMFCC 伪样本特征距离分析（2026-09-06）

问题：红队合成（Edge-TTS/GPT-SoVITS，自产）相对 FMFCC-A 伪样本（商业/开源 TTS/VC，
A 系攻击）在 XLS-R 特征空间是"同类"还是"新域"？

意义：
  1. 若红队与 FMFCC 伪重叠度高 → 红队难度有代表性（FMFCC 可代理评估）；
  2. 若红队自成簇 → 红队是**全新攻击域**，是增量训练/泛化叙事的关键证据
     （模型见过 FMFCC 伪也难检出红队 = 泛化鸿沟）。

方法：XLS-R mean-pool 特征 + 中文域 LR 打分器内部空间，
对比三簇中心距离/重叠：FMFCC伪(本地17.6k抽300) vs 红队(母本抽300) vs CFAD真(1000,评测集不训仅参照)。
用 LR 的决策分数分布 + 特征空间最近邻重叠近似衡量。

产出：evaluation/fmfcc_redteam_proximity.md/.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
FMFCC_DIR = Path("D:/VeriCall_data/FMFCC-A/extracted/FMFCC-A")
FMFCC_INFO = Path("D:/VeriCall_data/FMFCC-A/AllUtteranceInfo.txt")
CFAD_FLAC = Path("D:/VeriCall_data/CFAD/asvspoof_layout/ASVspoof2019_LA_dev/flac")
CFAD_PROTO = Path("D:/VeriCall_data/CFAD/asvspoof_layout/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
META = ROOT / "data/redteam/factory/meta.csv"
SCORER_PKL = ROOT / "data/redteam/factory/cn_lr_scorer.pkl"
OUT_MD = ROOT / "evaluation/fmfcc_redteam_proximity.md"
OUT_JSON = ROOT / "evaluation/fmfcc_redteam_proximity.json"


def feats_of_paths(paths, fe, ssl, device, max_sec=30):
    import librosa
    import soundfile as sf
    import torch
    X = []
    with torch.no_grad():
        for p in paths:
            try:
                wav, sr = sf.read(str(p))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * max_sec:
                    wav = wav[: sr * max_sec]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                X.append(h[0])
            except Exception as e:  # noqa: BLE001
                print(f"  [err] {Path(p).name}: {type(e).__name__}")
    return np.array(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="每簇样本数")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from transformers import AutoFeatureExtractor, AutoModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    # 1. FMFCC 伪样本（label 0 子集抽样）
    info = [l.strip().split(",") for l in open(FMFCC_INFO, encoding="utf-8")]
    fmfcc_fake = [r[0] for r in info if r[1] == "0"]
    fm_paths = [str(FMFCC_DIR / f) for f in rng.sample(fmfcc_fake, min(args.n, len(fmfcc_fake)))]
    # 2. 红队母本（clean 非 seg）
    rows = list(csv.DictReader(open(META, encoding="utf-8-sig")))
    rt = [r for r in rows if r["channel"] == "clean" and not r.get("seg")]
    rt_paths = [str(ROOT / r["path"]) for r in rng.sample(rt, min(args.n, len(rt)))]
    # 3. CFAD 真（参照，bonafide 源 aishell1）
    cfad_true = [l.split()[1] for l in open(CFAD_PROTO, encoding="utf-8") if l.split()[4] == "bonafide"]
    cf_paths = [str(CFAD_FLAC / f"{f}.flac") for f in rng.sample(cfad_true, min(args.n, len(cfad_true)))]

    print(f"抽取：FMFCC伪 {len(fm_paths)} / 红队 {len(rt_paths)} / CFAD真 {len(cf_paths)}", flush=True)
    print("XLS-R 提特征（分批）...", flush=True)
    Xf = feats_of_paths(fm_paths, fe, ssl, device)
    Xr = feats_of_paths(rt_paths, fe, ssl, device)
    Xc = feats_of_paths(cf_paths, fe, ssl, device)
    del ssl
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    # 簇中心
    def cen(X): return X.mean(axis=0)
    cf, cr, cc = cen(Xf), cen(Xr), cen(Xc)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def l2(a, b):
        return float(np.linalg.norm(a - b))

    # 判定器视角：用 CFAD 中文域 LR 打各簇分数（模型见过 CFAD 真/伪 + 隐式未见红队/FMFCC）
    import pickle
    with open(SCORER_PKL, "rb") as f:
        clf = pickle.load(f)
    sf_s = 1.0 - clf.predict_proba(Xf)[:, 1]
    sr_s = 1.0 - clf.predict_proba(Xr)[:, 1]
    sc_s = 1.0 - clf.predict_proba(Xc)[:, 1]

    # 近邻重叠：X_f 的每个样本找 X_r 中最近邻，统计距离 < X_r 簇内中位距离 的比例（粗重叠）
    from scipy.spatial.distance import cdist
    D_fr = cdist(Xf, Xr)
    D_rr = cdist(Xr, Xr)
    med_rr = np.median(D_rr[np.triu_indices(len(Xr), k=1)])
    fr_close = float((D_fr.min(axis=1) < med_rr).mean() * 100)

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_per_cluster": args.n,
        "cosine_sim": {"FMFCC伪↔红队": round(cos(cf, cr), 4),
                       "FMFCC伪↔CFAD真": round(cos(cf, cc), 4),
                       "红队↔CFAD真": round(cos(cr, cc), 4)},
        "l2_dist": {"FMFCC伪↔红队": round(l2(cf, cr), 2),
                    "FMFCC伪↔CFAD真": round(l2(cf, cc), 2),
                    "红队↔CFAD真": round(l2(cr, cc), 2)},
        "cn_lr_spoof_mean": {"FMFCC伪": round(float(sf_s.mean()), 3),
                             "红队": round(float(sr_s.mean()), 3),
                             "CFAD真": round(float(sc_s.mean()), 3)},
        "redteam_fmfcc_nn_overlap_pct": round(fr_close, 1),
        "conclusion": "",
    }
    tag = ("红队与 FMFCC 伪样本特征距离近、LR 分数同档"
           if payload["redteam_fmfcc_nn_overlap_pct"] >= 50
           else "红队在特征空间自成簇（相对 FMFCC 伪为独立新域）")
    payload["conclusion"] = (
        f"余弦相似度：FMFCC伪↔红队 {cos(cf,cr):.3f} / FMFCC伪↔真 {cos(cf,cc):.3f} / 红队↔真 {cos(cr,cc):.3f}；"
        f"红队→FMFCC伪最近邻重叠 {fr_close:.0f}%。中文域LR打分：FMFCC伪 {sf_s.mean():.3f} / 红队 {sr_s.mean():.3f} / 真 {sc_s.mean():.3f}。"
        f"结论：{tag}。")

    lines = [
        "# 红队 vs FMFCC 伪样本特征空间距离分析",
        "",
        f"> 生成：{payload['date']} · `evaluation/fmfcc_redteam_proximity.py`",
        f"> 每簇 {args.n} 条 · XLS-R mean-pool 1024d",
        "",
        "## 簇中心距离",
        "",
        "| 簇对 | 余弦相似度 | L2 距离 |",
        "|---|---|---|",
        f"| FMFCC伪(商业TTS/VC) ↔ 红队(Edge-TTS/克隆) | {cos(cf,cr):.4f} | {l2(cf,cr):.2f} |",
        f"| FMFCC伪 ↔ CFAD真(参照) | {cos(cf,cc):.4f} | {l2(cf,cc):.2f} |",
        f"| 红队 ↔ CFAD真(参照) | {cos(cr,cc):.4f} | {l2(cr,cc):.2f} |",
        "",
        "## 中文域 LR 打分分布（spoof 概率，越低越像真）",
        "",
        "| 簇 | spoof 均值 |",
        "|---|---|",
        f"| FMFCC伪 | {sf_s.mean():.3f} |",
        f"| 红队 | {sr_s.mean():.3f} |",
        f"| CFAD真 | {sc_s.mean():.3f} |",
        "",
        "## 最近邻重叠",
        "",
        f"- 红队→FMFCC伪：样本落入 FMFCC伪簇内中位距离的比例 **{fr_close:.0f}%**",
        "",
        "## 结论",
        "",
        payload["conclusion"],
    ]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {OUT_MD} / {OUT_JSON}")
    print(f"余弦: FMFCC伪↔红队 {cos(cf,cr):.3f} | FMFCC伪↔真 {cos(cf,cc):.3f} | 红队↔真 {cos(cr,cc):.3f}")
    print(f"LR 均分: FMFCC伪 {sf_s.mean():.3f} | 红队 {sr_s.mean():.3f} | 真 {sc_s.mean():.3f}")
    print(f"红队→FMFCC伪 NN 重叠 {fr_close:.0f}%")


if __name__ == "__main__":
    main()
