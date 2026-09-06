#!/usr/bin/env python
"""diag_xlsr_cn_lr.py — 中文域 LR 打分器验证（09-06）

用 CFAD 2000 条（真伪均衡）的已缓存 XLS-R 特征拟合中文域 LR，
对真实中文 / 红队样本打分，验证是否比英文域 LR（真 0.761 偏置）更准。
CFAD 作打标器训练数据不构成评测泄漏：红队非 CFAD 评测集。
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
CFAD_FLAC = Path("D:/VeriCall_data/CFAD/asvspoof_layout/ASVspoof2019_LA_dev/flac")
CFAD_PROTO = Path("D:/VeriCall_data/CFAD/asvspoof_layout/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
CFAD_XLSR = ROOT / ".tmp_ssl" / "crossdomain" / "xlsr_cfad_2000.npz"


def main():
    from transformers import AutoFeatureExtractor, AutoModel
    import soundfile as sf
    import librosa
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    # 中文域 LR：CFAD 缓存 80/20 留出（训练 1600 / 测试 400，避免同源高估）
    z = np.load(CFAD_XLSR)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(z["y"]))
    tr_i, te_i = idx[:1600], idx[1600:]
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(z["X"][tr_i], z["y"][tr_i])
    from sklearn.metrics import roc_auc_score
    te_auc = roc_auc_score(z["y"][te_i], clf.predict_proba(z["X"][te_i])[:, 1])
    print(f"CFAD 中文域 LR: 留出测试 AUC {te_auc:.3f} (1600 训练/400 测试)", flush=True)

    def feats_of(wav_paths):
        import librosa
        X = []
        with torch.no_grad():
            for p in wav_paths:
                wav, sr = sf.read(str(p))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                X.append(h[0])
        return np.array(X)

    def score(wav_paths):
        X = feats_of(wav_paths)
        return 1.0 - clf.predict_proba(X)[:, 1]  # spoof 概率

    # 测试样本：CFAD 训练集外的真实中文少量 + 红队
    real = []
    with open(CFAD_PROTO, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5 and p[4] == "bonafide":
                real.append(str(CFAD_FLAC / f"{p[1]}.flac"))
    real_sample = random.Random(3).sample(real, 4)  # 注意：与训练同源，仅演示

    rows = list(csv.DictReader(open(ROOT / "data/redteam/factory/meta.csv", encoding="utf-8")))
    clean = [r for r in rows if r["channel"] == "clean"]
    by_eng = {}
    for eng in ("edgetts", "sovits_api"):
        recs = [r for r in clean if r["engine"] == eng]
        by_eng[eng] = [str(ROOT / r["path"]) for r in random.Random(5).sample(recs, 4)]

    print("\n中文域 XLS-R+LR spoof 概率（越高越像伪）:", flush=True)
    # 留出测试集真伪对照（模型没见过）
    X_te = z["X"][te_i]
    y_te = z["y"][te_i]
    sc_te = 1.0 - clf.predict_proba(X_te)[:, 1]
    print(f"  [留出测试集] 真实中文 spoof 概率均值 {sc_te[y_te==1].mean():.3f} / "
          f"伪 均值 {sc_te[y_te==0].mean():.3f}", flush=True)
    # 红队各引擎
    for tag, paths in [("红队 edgetts", by_eng["edgetts"]),
                       ("红队 sovits", by_eng["sovits_api"])]:
        sc = score(paths)
        print(f"  {tag}: " + ", ".join(f"{s:.3f}" for s in sc) +
              f"  | 均值 {sc.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
