#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 clean（62,999）**剩余 5 列**打分编排：RawNet2 x3 + XLS-R(M2/M3)。

为什么需要：Table 1/4 共 10 列。scripts/run_full_clean_scores.py 只覆盖 AASIST 五列
（M0/M1/M1'/LP/LP-FT）；若只跑那五列就写"全量复现"，会造成**表内部分全量、部分子样**
的不一致，反而是新把柄。本脚本补齐余下五列。

列 -> 权重/构造器（2026-09-15 逐列核对子样 meta 得出）：
  rn2_en0 -> models/weights/rawnet2_en_pretrained.pth           (RN2-EN, zero-shot)
  rn2_cn  -> models/weights/rawnet2_cn_ft_best.pth              (RN2-CN, 全量微调)
  rn2_lp  -> external/aasist/exp_result/LA_RawNet2_CN_lp_strict_lr1e3_ep8_bs16_rn2_lp_strict/weights/best.pth
  M2      -> XlsrCnChannel(scorer='v4')   冻结 XLS-R + LR
  M3      -> XlsrCnChannel(scorer='wide')

用法：<vericall python> scripts/run_full_remaining_scores.py [--only rn2_en0,...] [--skip-m2m3]
设计：单进程串行（GPU 争抢是上一轮失败的主因），每列独立 try，失败不阻断后续；
      已存在且 >1MB 的输出自动跳过，可反复运行续跑。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"D:/VeriCall_data/conda_envs/vericall/python.exe"
FULL_ROOT = "D:/VeriCall_data/CFAD_full"
REP = ROOT / "evaluation" / "reports"

# (tag, 说明, 类型, 权重相对路径)
AASIST_STYLE = [
    ("rn2_en0", "RN2-EN zero-shot", "cuda",
     "models/weights/rawnet2_en_pretrained.pth"),
    ("rn2_cn", "RN2-CN 全量微调", "cuda",
     "models/weights/rawnet2_cn_ft_best.pth"),
    ("rn2_lp", "RN2-LP 冻结躯干+MLP", "cuda",
     "external/aasist/exp_result/LA_RawNet2_CN_lp_strict_lr1e3_ep8_bs16_rn2_lp_strict/weights/best.pth"),
]

# RawNet2 的 config.conf 在其实验目录内；rn2_en0/rn2_cn 的权重在 models/weights/，
# 没有配套实验目录，需借用 rn2_lp 实验目录的 config（同架构，超参差异不影响推理）。
RN2_CONFIG_FALLBACK = ("external/aasist/exp_result/"
                       "LA_RawNet2_CN_lp_strict_lr1e3_ep8_bs16_rn2_lp_strict")

XLSR = [
    ("v4", "M2 冻结XLS-R+LR"),
    ("wide", "M3 XLS-R wide(见过测试域)"),
]


def run(cmd, env=None, tag=""):
    print(f"\n===== RUN {tag} =====\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print(f"===== {'OK' if r.returncode == 0 else 'FAIL'} {tag} "
          f"rc={r.returncode} {time.time()-t0:.0f}s =====\n", flush=True)
    return r.returncode == 0


def done(path: Path, min_bytes: int = 1_000_000) -> bool:
    return path.is_file() and path.stat().st_size > min_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔，只跑指定 tag")
    ap.add_argument("--skip-m2m3", action="store_true", help="跳过 XLS-R 两列")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else None

    env = dict(os.environ, VERICALL_CFAD_ROOT=FULL_ROOT)
    results = []
    return run_all(args, want, env, results)


def run_all(args, want, env, results):
    # ---- RawNet2 三列：走 cfad_breakdown.py，显式 --arch RawNet2Spoof ----
    for tag, desc, dev, ckpt in AASIST_STYLE:
        if want and tag not in want:
            continue
        ck = ROOT / ckpt
        out = REP / f"cfad_scores_clean_{tag}_full.jsonl"
        if not ck.is_file():
            print(f"[skip] {tag}: 缺权重 {ck}", flush=True)
            results.append((tag, "MISSING_WEIGHT"))
            continue
        if done(out):
            print(f"[skip] {tag}: 已完成 {out.name}", flush=True)
            results.append((tag, "ALREADY_DONE"))
            continue
        ok = run([PY, "evaluation/cfad_breakdown.py", "--ckpt", ckpt,
                  "--config", RN2_CONFIG_FALLBACK,
                  "--arch", "RawNet2Spoof", "--out-tag", f"{tag}_full",
                  "--batch_size", "32", "--device", "cuda", "--domains", "clean"],
                 env=env, tag=tag)
        results.append((tag, "OK" if ok else "FAIL"))

    # ---- XLS-R 两列：走 eval_xlsr_cfad.py，但根目录须指向全量 ----
    if not args.skip_m2m3:
        for scorer, desc in XLSR:
            tag = f"xlsr_{scorer}"
            if want and tag not in want:
                continue
            out = REP / f"cfad_scores_xlsr_{scorer}_clean_full.jsonl"
            if done(out):
                print(f"[skip] {tag}: 已完成 {out.name}", flush=True)
                results.append((tag, "ALREADY_DONE"))
                continue
            # eval_xlsr_cfad.py 的 DOMAINS 硬编码 2000 版路径；用环境变量覆盖根目录
            env2 = dict(env, VERICALL_CFAD_FULL_ONLY="1")
            ok = run([PY, "evaluation/eval_xlsr_full.py", "--scorer", scorer],
                     env=env2, tag=tag)
            results.append((tag, "OK" if ok else "FAIL"))

    print("\n" + "=" * 56)
    print("全量剩余列汇总")
    print("=" * 56)
    for tag, st in results:
        print(f"  {tag:12s} {st}")
    print()
    print("提示：跑完后执行 evaluation/verify_full_vs_subset.py 做子样 vs 全量一致性核验。")
    return 0 if all(s in ("OK", "ALREADY_DONE") for _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
