#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICASSP2027 投稿 · 通宵实验总队列（Tier B 全部补齐）
================================================================
按**重要性降序**执行，跑不完也不亏（前面的都是高价值的）。

  1. aasist_lp_lr1e3   AASIST 真·线性探测（冻结骨干，仅 out_layer）  ← 补 2×2 缺的那格
  2. lr5e6             全微调 lr=5e-6                                ← 排除"训过拟合"
  3. lr1e5             全微调 lr=1e-5
  4. aasist_lp_lr1e4   AASIST 线性探测 lr=1e-4（第二档，可选）
  5. s42_lr5e5         全微调 seed=42                                ← 多种子
  6. s7_lr5e5          全微调 seed=7

设计要点：
  - 每格独立实验目录，互不覆盖
  - --resume auto 续训：会话被杀后**重跑同一条命令**即可接着跑
  - 已完成（weights/best.pth 存在）的格子自动跳过
  - 冻结骨干通过 config 的 "freeze_backbone": "True" 生效
    （main.py 已支持：只把 out_layer 的 requires_grad 置 True）

用法：
    python scripts/night_run.py                 # 跑全部
    python scripts/night_run.py --dry-run       # 只看计划
    python scripts/night_run.py --only lp       # 只跑含 lp 的格子
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AASIST = ROOT / "external" / "aasist"
BASE_CONF = ROOT / "configs" / "AASIST_CN.conf"
EXP_ROOT = AASIST / "exp_result"
PY = sys.executable

# (tag, base_lr, seed, epochs, freeze, 说明)
GRID: list[tuple[str, float, int, int, bool, str]] = [
    ("lp_lr1e3", 1e-3, 1234, 8, True, "AASIST 真线性探测 lr1e-3"),
    ("lr5e6", 5e-6, 1234, 8, False, "全微调 lr5e-6"),
    ("lr1e5", 1e-5, 1234, 8, False, "全微调 lr1e-5"),
    ("lp_lr1e4", 1e-4, 1234, 8, True, "AASIST 真线性探测 lr1e-4"),
    ("s42_lr5e5", 5e-5, 42, 8, False, "全微调 seed42"),
    ("s7_lr5e5", 5e-5, 7, 8, False, "全微调 seed7"),
]


def make_conf(tag: str, base_lr: float, epochs: int, freeze: bool) -> Path:
    cfg = json.loads(BASE_CONF.read_text(encoding="utf-8"))
    cfg["num_epochs"] = epochs
    cfg["optim_config"]["base_lr"] = base_lr
    cfg["optim_config"]["lr_min"] = base_lr / 10.0
    cfg["eval_all_best"] = "False"          # 扫描阶段不做全量 eval，省一半墙钟
    cfg["freeze_backbone"] = "True" if freeze else "False"
    out = ROOT / "configs" / f"AASIST_CN_{tag}.conf"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")
    return out


def done(exp_dir: Path) -> bool:
    return (exp_dir / "weights" / "best.pth").is_file()


def run_one(tag, base_lr, seed, epochs, freeze, note, resume, dry) -> None:
    # 目录名由 main.py 依 config 的 database_path 等自动生成，前缀固定为
    # LA_AASIST_CN_{tag}_ep{epochs}_bs16_{tag}
    exp_dir = EXP_ROOT / f"LA_AASIST_CN_{tag}_ep{epochs}_bs16_{tag}"
    if done(exp_dir):
        print(f"[跳过] {tag:12s} 已完成")
        return
    conf = make_conf(tag, base_lr, epochs, freeze)
    cmd = [PY, "-u", "main.py",
           "--config", str(conf),
           "--output_dir", str(EXP_ROOT),
           "--seed", str(seed),
           "--num_epochs", str(epochs),
           "--comment", tag]
    if resume:
        cmd += ["--resume", "auto"]
    print(f"\n[启动] {tag:12s} lr={base_lr:g} seed={seed} freeze={freeze}  ({note})")
    if dry:
        print("       " + " ".join(cmd))
        return
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(AASIST), check=False)
    dt = (time.time() - t0) / 60.0
    ok = "完成" if proc.returncode == 0 else f"失败(rc={proc.returncode})"
    print(f"[{ok}] {tag:12s} 用时 {dt:.1f} 分钟")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑 tag 含该子串的格子")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BASE_CONF.is_file():
        print(f"缺基础配置: {BASE_CONF}")
        sys.exit(1)

    grid = [g for g in GRID if args.only is None or args.only in g[0]]
    print("=" * 72)
    print(f"通宵队列：{len(grid)} 格，按重要性降序。预计 {len(grid)*1.1:.1f} 小时。")
    print("被中断后重跑本命令即可续训（--resume auto）。")
    print("=" * 72)
    for i, (tag, lr, seed, ep, fr, note) in enumerate(grid, 1):
        print(f"  {i}. {tag:12s} lr={lr:g} seed={seed} freeze={str(fr):5s} — {note}")

    for g in grid:
        run_one(*g, resume=args.resume, dry=args.dry_run)

    print("\n" + "=" * 72)
    print("队列结束。下一步统一评测：")
    print("  python evaluation/sweep_eval_cfad.py")
    print("（它会评测所有 sweep/lp 格子在 CFAD 五条件 + LA dev 上的 EER*/AUC）")


if __name__ == "__main__":
    main()
