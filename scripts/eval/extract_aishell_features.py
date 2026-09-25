#!/usr/bin/env python
"""extract_aishell_features.py — aishell1 真样本 XLS-R 特征提取（2026-09-06）

对本地 aishell1 子集（6 说话人，2122 条真语音，16k）提取 XLS-R mean-pool 特征，
作为中文域增量训练的真侧特征缓存（与 .tmp_ssl/fmfcc/ 伪侧对称）。

产出：.tmp_ssl/aishell/xlsr_aishell_true_full.npz（X, file_ids）
用法：python -u scripts/eval/extract_aishell_features.py [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

AISHELL_DIR = Path("D:/VeriCall_data/aishell1_sub")
SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
OUT_DIR = ROOT / ".tmp_ssl" / "aishell"
OUT_NPZ = OUT_DIR / "xlsr_aishell_true_full.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"XLS-R {SSL_LOCAL} -> {device}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    # 全部 aishell wav（train 目录）
    wavs = sorted(str(p) for p in AISHELL_DIR.rglob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"aishell 真样本 wav: {len(wavs)} 条", flush=True)

    Xs, ids = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(wavs):
            try:
                wav, sr = sf.read(p)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                Xs.append(h[0])
                ids.append(Path(p).stem)
            except Exception as e:  # noqa: BLE001
                print(f"  [err] {Path(p).name}: {type(e).__name__}")
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(wavs)}] {time.time()-t0:.0f}s", flush=True)

    Xa = np.array(Xs)
    np.savez(OUT_NPZ, X=Xa, file_ids=np.array(ids))
    print(f"\n完成 {len(Xa)} 条 -> {OUT_NPZ} ({OUT_NPZ.stat().st_size//1024//1024}MB)")


if __name__ == "__main__":
    main()
