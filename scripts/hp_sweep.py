#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICASSP2027 投稿 · 超参扫描 + 多种子（排除"训练过拟合"替代解释）
================================================================
原论文只有 lr=5e-5 / 8 epoch / 8k+8k 一组配置。审稿人会问：
"你怎么知道不是你把模型训过拟合了？" 本脚本扫描学习率并换种子，
为"单域微调损害 OOD 判别力"这一结论提供鲁棒性证据。

网格：
    lr = 5e-6, 1e-5          （5e-5 已在 LA_AASIST_CN_ep8_bs16_cn_ft 跑过，复用）
    seed = 42, 7             （默认 1234 已跑过）
共 4 次训练，每次 8 epoch @ bs16 ≈ 1~1.2 h（RTX 5060 8GB）

设计要点：
  - 每个格子独立实验目录，互不覆盖
  - 支持 --resume auto 续训（会话被杀后重跑本脚本即可接着跑）
  - 已跑完（存在 best.pth）的格子自动跳过
  - lr_min 按 base_lr 的 1/10 同步缩放，保持 cosine 调度形状一致

用法：
    python scripts/hp_sweep.py                 # 跑全部
    python scripts/hp_sweep.py --dry-run       # 只看会跑什么
    python scripts/hp_sweep.py --lr 1e-5       # 只跑指定 lr
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AASIST = ROOT / "external" / "aasist"
BASE_CONF = ROOT / "configs" / "AASIST_CN.conf"
EXP_ROOT = AASIST / "exp_result"
PY = sys.executable

# (tag, base_lr, seed, epochs)
# 只扫学习率：seed 稳定性已由已有的第二次独立全微调（cfad_scores_*_m1lp.jsonl）提供证据
# —— 两次独立运行 clean AUC 0.453 / 0.437，均反转、均远低于零样本 0.593。
# 因此本轮只需回答"是不是我把模型训过拟合了"，即 2 个 lr 格子（约 2.4 h）。
GRID: list[tuple[str, float, int, int]] = [
    ("lr5e6_s1234", 5e-6, 1234, 8),
    ("lr1e5_s1234", 1e-5, 1234, 8),
]


def make_conf(tag: str, base_lr: float, epochs: int) -> Path:
    cfg = json.loads(BASE_CONF.read_text(encoding="utf-8"))
    cfg["num_epochs"] = epochs
    cfg["optim_config"]["base_lr"] = base_lr
    cfg["optim_config"]["lr_min"] = base_lr / 10.0
    # 扫描阶段不做全量 eval，省一半墙钟（skill 4c）
    cfg["eval_all_best"] = "False"
    out = ROOT / "configs" / f"AASIST_CN_{tag}.conf"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")
    return out


def already_done(exp_dir: Path) -> bool:
    return (exp_dir / "weights" / "best.pth").is_file()


def run_one(tag: str, base_lr: float, seed: int, epochs: int,
            resume: bool, dry: bool) -> None:
    exp_dir = EXP_ROOT / f"LA_AASIST_CN_{tag}"
    conf = make_conf(tag, base_lr, epochs)
    if already_done(exp_dir):
        print(f"[跳过] {tag} 已完成（{exp_dir}/weights/best.pth 存在）")
        return
    cmd = [PY, "-u", "main.py",
           "--config", str(conf),
           "--output_dir", str(EXP_ROOT),
           "--seed", str(seed),
           "--num_epochs", str(epochs),
           "--comment", tag]
    if resume:
        cmd += ["--resume", "auto"]
    print(f"[启动] {tag}  lr={base_lr:g} seed={seed} epochs={epochs}")
    print(f"       {' '.join(cmd)}")
    if dry:
        return
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(AASIST), check=False)
    dt = (time.time() - t0) / 60.0
    status = "完成" if proc.returncode == 0 else f"失败(rc={proc.returncode})"
    print(f"[{status}] {tag}  用时 {dt:.1f} 分钟")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=None, help="只跑指定 lr")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BASE_CONF.is_file():
        print(f"缺基础配置: {BASE_CONF}")
        sys.exit(1)

    grid = [g for g in GRID if args.lr is None or abs(g[1] - args.lr) < 1e-12]
    print(f"将跑 {len(grid)} 个配置（总计约 {len(grid)*1.2:.1f} 小时）")
    for tag, lr, seed, ep in grid:
        run_one(tag, lr, seed, ep, args.resume, args.dry_run)
    print("\n全部完成。下一步：")
    print("  python evaluation/sweep_eval_cfad.py   # 统一评测各格子在 CFAD 上的 EER/AUC")


if __name__ == "__main__":
    main()
