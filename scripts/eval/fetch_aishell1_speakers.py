#!/usr/bin/env python
"""fetch_aishell1_speakers.py — AISHELL-1 按说话人下载（2026-09-06 固化）

从 hf-mirror 的 AISHELL/AISHELL-1 仓库按说话人下载（无需 15.6G 完整包）：
  HF 直连 huggingface.co 不通(000) 但 hf-mirror.com API+文件全通；
  每个说话人一个 tar.gz（S0002-S0101，25-53MB，共 100 个 / 3.5GB）；
  URL: https://hf-mirror.com/datasets/AISHELL/AISHELL-1/resolve/main/data_aishell/wav/S0002.tar.gz
  转录: .../resolve/main/data_aishell/transcript/aishell_transcript_v0.8.txt

用法：
  python scripts/eval/fetch_aishell1_speakers.py --out D:/VeriCall_data/aishell1_sub \
      --start 2 --end 101 --extract --transcript
说明：下载 tar.gz → 可选解压(每说话人到 train/Sxxxx/) → 可选存转录 txt。
      全部 100 说话人 = 官方 train 集（34,716 wav，16k/真人/与 CFAD 真同源）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = "https://hf-mirror.com/datasets/AISHELL/AISHELL-1/resolve/main"
WAV_DIR = f"{BASE}/data_aishell/wav"
TXT = f"{BASE}/data_aishell/transcript/aishell_transcript_v0.8.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/VeriCall_data/aishell1_sub")
    ap.add_argument("--start", type=int, default=2, help="起始说话人号(默认2=S0002)")
    ap.add_argument("--end", type=int, default=101, help="结束说话人号(默认101=S0101)")
    ap.add_argument("--extract", action="store_true", help="下载后自动解压")
    ap.add_argument("--transcript", action="store_true", help="同时下载转录文本")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.transcript:
        tgt = out / "aishell_transcript.txt"
        if not tgt.exists():
            print(f"下载转录 -> {tgt}")
            subprocess.run(["curl", "-sL", "-m", "60", "-o", str(tgt), TXT], check=True)
        else:
            print(f"转录已存在: {tgt}")

    n_ok = 0
    for i in range(args.start, args.end + 1):
        s = f"S{i:04d}"
        gz = out / f"{s}.tar.gz"
        if gz.exists():
            print(f"[跳过] {gz.name} 已存在")
            n_ok += 1
        else:
            url = f"{WAV_DIR}/{s}.tar.gz"
            print(f"[下载] {s} ...", end="", flush=True)
            r = subprocess.run(["curl", "-sL", "-m", "300", "-o", str(gz), url])
            if r.returncode != 0 or gz.stat().st_size < 1_000_000:
                print(" FAIL"); gz.unlink(missing_ok=True)
                continue
            print(f" {gz.stat().st_size//1024//1024}MB")
            n_ok += 1
        if args.extract:
            d = out / "train" / s
            if not d.is_dir():
                subprocess.run(["tar", "xzf", str(gz), "-C", str(out)], check=True)
    print(f"\n完成 {n_ok} 个说话人包 -> {out}")


if __name__ == "__main__":
    main()
