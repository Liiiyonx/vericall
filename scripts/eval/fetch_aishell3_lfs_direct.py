#!/usr/bin/env python
"""fetch_aishell3_lfs_direct.py — AISHELL-3 音频直链下载（绕过 hf-mirror git-lfs）

背景（2026-09-07）：hf-mirror 的 git-lfs smudge 端点对 AISHELL-3 返回失败
（"external filter 'git-lfs filter-process' failed"），但 **resolve 直链可用**
（https://hf-mirror.com/datasets/AISHELL/AISHELL-3/resolve/main/<path>，HTTP 200 字节数一致）。

本脚本：从已克隆仓库（含全部 LFS 指针的 git 对象）用 `git ls-tree` 枚举 train/wav 全路径，
跳过 git-lfs，直接用 resolve URL 并发下载真实音频，RIFF 头校验。
用法：
  python -u scripts/eval/fetch_aishell3_lfs_direct.py [--limit 0] [--threads 12]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("D:/VeriCall_data/aishell3/repo")
OUT = Path("D:/VeriCall_data/aishell3/data")
BASE = "https://hf-mirror.com/datasets/AISHELL/AISHELL-3/resolve/main"
PREFIX = "train/wav/"


def enumerate_wavs(limit: int) -> list[str]:
    r = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", "HEAD", PREFIX],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        sys.exit(f"git ls-tree 失败: {r.stderr}")
    paths = [ln for ln in r.stdout.splitlines() if ln.startswith(PREFIX) and ln.endswith(".wav")]
    return paths if not limit else paths[:limit]


def fetch(path: str) -> tuple[str, str]:
    """下载单文件；返回 (path, ok|err)。"""
    dst = OUT / path
    if dst.is_file() and dst.stat().st_size > 4096:
        return path, "skip(exists)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{path}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as f:
                data = resp.read()
                f.write(data)
            if len(data) < 4096 or data[:4] != b"RIFF":
                dst.unlink(missing_ok=True)
                raise ValueError(f"非 wav(RIFF) 或过小: {len(data)}B")
            return path, f"ok({len(data)})"
        except Exception as e:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return path, f"FAIL:{type(e).__name__}"  # noqa: F821


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help=">0 仅前 N 条(冒烟)")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    paths = enumerate_wavs(args.limit)
    print(f"[枚举] {len(paths)} 条 wav（limit={args.limit or '全量'}）", flush=True)
    t0, done, fail = time.time(), 0, []
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        for i, (p, st) in enumerate(ex.map(fetch, paths), 1):
            done += 1
            if st.startswith("FAIL"):
                fail.append((p, st))
            if i % 500 == 0 or st.startswith("FAIL"):
                print(f"  [{i}/{len(paths)}] {st}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n完成 {done}/{len(paths)}，失败 {len(fail)}，耗时 {time.time()-t0:.0f}s")
    for p, st in fail[:10]:
        print("  FAIL", p, st)
    print(f"产物: {OUT}")


if __name__ == "__main__":
    main()
