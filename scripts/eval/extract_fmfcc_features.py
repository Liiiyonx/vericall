#!/usr/bin/env python
"""extract_fmfcc_features.py — FMFCC 伪样本 XLS-R 特征预提取（2026-09-06）

为「中文域增量训练」预提取 FMFCC-A 伪样本（17,636 条，label 0/A 系）的
XLS-R mean-pool 特征，存 npz。真样本（N01）到位后直接复用本缓存训练，
省去 ~45min 特征提取。

设计：
  - 从 AllUtteranceInfo.txt 读 label（0=伪），只提 label 0；
  - 分块提取，每块落盘临时 npz（断点续跑友好）；
  - 输出 .tmp_ssl/fmfcc/xlsr_fmfcc_fake_full.npz（X, file_ids, attack_ids）；
  - 失败文件记 skips，不中断。

用法：python -u scripts/eval/extract_fmfcc_features.py [--limit N] [--resume]
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

FMFCC_DIR = Path("D:/VeriCall_data/FMFCC-A/extracted/FMFCC-A")
FMFCC_INFO = Path("D:/VeriCall_data/FMFCC-A/AllUtteranceInfo.txt")
SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
OUT_DIR = ROOT / ".tmp_ssl" / "fmfcc"
OUT_NPZ = OUT_DIR / "xlsr_fmfcc_fake_full.npz"
CHUNK = 500  # 每批落盘大小


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for c in OUT_DIR.glob("chunk_*.npz"):
            c.unlink()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"XLS-R {SSL_LOCAL} -> {device}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    # 待提取清单：磁盘实际存在的 label 0（伪）——标注顺序前段文件可能不在磁盘（仅部分下载）
    disk_files = set(os.listdir(FMFCC_DIR))
    info = [l.strip().split(",") for l in open(FMFCC_INFO, encoding="utf-8")]
    fake = [(r[0], r[2]) for r in info
            if len(r) >= 3 and r[1] == "0" and r[0] in disk_files]
    print(f"磁盘∩标注 label0(伪): {len(fake)} 条", flush=True)
    if args.limit:
        fake = fake[: args.limit]

    # 断点：读已有 chunk 的完成情况
    done_ids = set()
    if args.resume:
        for c in OUT_DIR.glob("chunk_*.npz"):
            d = np.load(c, allow_pickle=True)
            for fid in d["file_ids"]:
                done_ids.add(str(fid))
        print(f"续跑：已完 {len(done_ids)} 条", flush=True)
        fake = [f for f in fake if f[0] not in done_ids]

    X_all, ids_all, att_all = [], [], []
    t0 = time.time()
    batch = []
    n_skip = 0
    with torch.no_grad():
        for i, (fname, att) in enumerate(fake):
            p = FMFCC_DIR / fname
            try:
                wav, sr = sf.read(str(p))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                batch.append((fname, att, h[0]))
            except Exception:  # noqa: BLE001
                n_skip += 1
            if len(batch) >= CHUNK or i + 1 == len(fake):
                if batch:
                    ck = len([1 for _ in OUT_DIR.glob("chunk_*.npz")])
                    chunk_path = OUT_DIR / f"chunk_{ck:04d}.npz"
                    np.savez(chunk_path,
                             X=np.array([b[2] for b in batch]),
                             file_ids=np.array([b[0] for b in batch]),
                             attack_ids=np.array([b[1] for b in batch]))
                    print(f"  chunk@{i+1} 落盘 {chunk_path.name} ({time.time()-t0:.0f}s, "
                          f"{(time.time()-t0)/max(1,i+1):.2f}s/条 累计skip {n_skip})", flush=True)
                    batch = []
        # 合并全部 chunk（含既有，若 resume）
        Xs, ids, atts = [], [], []
        for c in sorted(OUT_DIR.glob("chunk_*.npz")):
            d = np.load(c, allow_pickle=True)
            Xs.append(d["X"]); ids.extend(d["file_ids"]); atts.extend(d["attack_ids"])
        if Xs:
            X_all = np.concatenate(Xs)
            # 按 file_ids 去重排序保序
            uniq = {}
            for x, fid, a in zip(X_all, ids, atts):
                uniq[str(fid)] = (x, str(a))
            Xs2 = np.array([v[0] for v in uniq.values()])
            ids2 = np.array([k for k in uniq.keys()])
            atts2 = np.array([v[1] for v in uniq.values()])
            np.savez(OUT_NPZ, X=Xs2, file_ids=ids2, attack_ids=atts2)
            print(f"\n完成 {len(Xs2)} 条（skip {n_skip}）-> {OUT_NPZ}")
        else:
            print("无新特征产生（resume 且全完成？）")


if __name__ == "__main__":
    main()
