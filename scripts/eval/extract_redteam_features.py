#!/usr/bin/env python
"""extract_redteam_features.py — 红队母本 XLS-R 特征（2026-09-06）

提取红队 clean 母本（3117 条，非 seg）XLS-R 特征，供训练模型评测击穿率。
产出：.tmp_ssl/redteam/xlsr_rt_masters.npz（X, file_ids, dialect, engine）
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
META = ROOT / "data/redteam/factory/meta.csv"
SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
OUT_NPZ = ROOT / ".tmp_ssl" / "redteam" / "xlsr_rt_masters.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    rows = [r for r in csv.DictReader(open(META, encoding="utf-8-sig"))
            if r["channel"] == "clean" and not r.get("seg")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"红队 clean 母本: {len(rows)}", flush=True)

    Xs, ids, dials, engs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            p = ROOT / r["path"]
            try:
                wav, sr = sf.read(str(p))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                Xs.append(h[0]); ids.append(Path(p).stem)
                dials.append(r["dialect"]); engs.append(r["engine"])
            except Exception as e:  # noqa: BLE001
                print(f"  [err] {r['path']}: {type(e).__name__}")
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, X=np.array(Xs), file_ids=np.array(ids),
             dialects=np.array(dials), engines=np.array(engs))
    print(f"\n完成 {len(Xs)} 条 -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
