#!/usr/bin/env python
"""diag_xlsr_redteam.py — 验证 XLS-R+LR 语义通道对红队音频的区分度（09-06 跟进）

背景：AASIST（acoustic）对中文红队全量漏检（score~0），reverse-screen 需换语义通道。
本脚本用英文 train 拟合的 XLS-R+LR 给三类样本打分：
  1. 真实中文语音（CFAD bonafide aishell1）
  2. 红队 edgetts 合成
  3. 红队 sovits 克隆
看 score（spoof 概率）分布是否拉开，决定语义通道改造是否可行。
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
EN_CACHE = ROOT / ".tmp_ssl" / "wav2vec2-xls-r-300m" / "train_20000.npz"


def main():
    from transformers import AutoFeatureExtractor, AutoModel
    import soundfile as sf
    import torch
    from sklearn.linear_model import LogisticRegression

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    # 拟合 LR（英文 train，与 CFAD 复验同口径）
    z = np.load(EN_CACHE)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(z["X"], z["y"])
    print(f"LR 拟合: {z['X'].shape} (真 {z['y'].sum()} / 伪 {(1-z['y']).sum()})", flush=True)

    def feats_of(wav_paths):
        """逐条 XLS-R mean-pool 特征（无缓存，仅诊断用小批）。16k 重采样统一。"""
        import librosa
        X = []
        with torch.no_grad():
            for p in wav_paths:
                wav, sr = sf.read(str(p))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:      # 超长截 30s 控制耗时
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                X.append(h[0])
        return np.array(X)

    def score(wav_paths):
        X = feats_of(wav_paths)
        return 1.0 - clf.predict_proba(X)[:, 1]  # spoof 概率（对齐 CFAD 口径）

    # 1. 真实中文（CFAD bonafide，aishell1 真语音）
    real = []
    with open(CFAD_PROTO, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5 and p[4] == "bonafide":
                real.append(str(CFAD_FLAC / f"{p[1]}.flac"))
    real_sample = random.Random(3).sample(real, 4)

    # 2/3. 红队 edgetts / sovits clean 母本
    rows = list(csv.DictReader(open(ROOT / "data/redteam/factory/meta.csv", encoding="utf-8")))
    clean = [r for r in rows if r["channel"] == "clean"]
    by_eng = {}
    for eng in ("edgetts", "sovits_api"):
        recs = [r for r in clean if r["engine"] == eng]
        by_eng[eng] = [str(ROOT / r["path"]) for r in
                       random.Random(5).sample(recs, 4)]

    print("\nXLS-R+LR spoof 概率（越高越像伪）:", flush=True)
    for tag, paths in [("真实中文 aishell1", real_sample),
                       ("红队 edgetts", by_eng["edgetts"]),
                       ("红队 sovits", by_eng["sovits_api"])]:
        sc = score(paths)
        print(f"  {tag}: " + ", ".join(f"{s:.3f}" for s in sc) +
              f"  | 均值 {sc.mean():.3f}", flush=True)
        # 对照：若通道有效，真实语音应 score 低（判真），红队应 score 高（判伪）


if __name__ == "__main__":
    main()
