#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 clean（62,999）AASIST 家族打分执行器。

物化完成后跑这个：对 5 个 AASIST 系权重各打一遍 CFAD_full/asvspoof_layout，
产出 cfad_scores_clean_<tag>_full.jsonl（格式与 2000 版完全一致，可被
cfad_crosspair.py / eerstar_table.py 直接复用）。

用法：<vericall python> scripts/run_full_clean_scores.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"D:/VeriCall_data/conda_envs/vericall/python.exe"
FULL_ROOT = "D:/VeriCall_data/CFAD_full"

JOBS = [
    # tag -> 论文 Table 1 对应列的权重（2026-09-15 逐列核对子样 cfad_breakdown_*.json 的 meta.ckpt）
    ("en0",         "external/aasist/models/weights/AASIST.pth"),                                              # M0
    ("cn_ft",       "external/aasist/exp_result/LA_AASIST_CN_ep8_bs16_cn_ft/weights/best.pth"),                 # M1
    ("m1lp",        "external/aasist/exp_result/LA_AASIST_CN_lp8_bs16/weights/best.pth"),                       # M1'
    # LP 列必须用 *_lp_strict_*（子样 cfad_breakdown_aasist_lp_strict_20260914_230440.json 的 meta.ckpt）；
    # 此前误写为 LA_AASIST_CN_lp_lr1e3_ep8_bs16_lp_lr1e3，那是另一个权重，与论文 Table 1 的 LP 不符。
    ("lp",          "external/aasist/exp_result/LA_AASIST_CN_lp_strict_lr1e3_ep8_bs16_lp_strict/weights/best.pth"),
    ("aasist_lpft", "external/aasist/exp_result/LA_AASIST_CN_lpft_lr5e5_ep8_bs16_lpft_lr5e5/weights/best.pth"),  # LP-FT
]


def main():
    env = dict(os.environ, VERICALL_CFAD_ROOT=FULL_ROOT)
    for tag, ckpt in JOBS:
        ck = ROOT / ckpt
        if not ck.is_file():
            print(f"[skip] 缺权重 {ck}", flush=True)
            continue
        out = ROOT / "evaluation" / "reports" / f"cfad_scores_clean_{tag}_full.jsonl"
        if out.is_file() and out.stat().st_size > 1_000_000:
            print(f"[skip] 已存在 {out.name}", flush=True)
            continue
        cmd = [PY, "evaluation/cfad_breakdown.py", "--ckpt", ckpt,
               "--out-tag", f"{tag}_full", "--batch_size", "64", "--domains", "clean"]
        print(f"\n===== RUN {tag} =====\n{' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        print(f"===== EXIT {tag} rc={r.returncode} {time.time()-t0:.0f}s =====\n", flush=True)


if __name__ == "__main__":
    main()
