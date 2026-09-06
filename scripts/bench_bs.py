#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — AASIST 训练步速基准（选址 batch_size / nb_samp）。

背景:
  全量 25380 条、bs=28 时一个 epoch 跑了 30 分钟还没完, 30 epoch 就要 15 小时。
  本脚本直接测 "一个 train step(前向+反向+更新)" 的耗时与峰值显存,
  用数据决定 batch_size 与 nb_samp, 而不是拍脑袋调参。

测量项(每个 (bs, nb_samp) 组合):
  - warmup 3 步后, 测 10 步的平均耗时 (ms/step)
  - 峰值显存 (torch.cuda.max_memory_allocated)
  - 折算 "每千条样本耗时" = ms/step / bs * 1000, 用于横向比较吞吐

用法:
  python scripts/bench_bs.py                       # 默认扫 bs=8..32, nb_samp=48000
  python scripts/bench_bs.py --nb 48000 32000      # 同时比较两种切片长度
  python scripts/bench_bs.py --bs 8 16 24 32
"""
import os
import sys
import time
import argparse

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
AASIST = os.path.abspath(os.path.join(ROOT, "..", "external", "aasist"))
sys.path.insert(0, AASIST)

from importlib import import_module  # noqa: E402


def build(nb_samp):
    cfg = {
        "architecture": "AASIST",
        "nb_samp": nb_samp,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }
    module = import_module("models.{}".format(cfg["architecture"]))
    model = getattr(module, "Model")(cfg).cuda()
    return model


args_verbose = False


def bench(bs, nb_samp, steps=10, warmup=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build(nb_samp)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).cuda())

    x = torch.randn(bs, nb_samp).cuda()
    y = torch.randint(0, 2, (bs,)).cuda()

    # 注意: loss 必须算在 autocast 内部(与 main.py 一致)。放到外部会因 out 是
    # fp16 而 criterion 的 weight 是 fp32 触发 "expected Half but found Float"。
    def one_step():
        with torch.amp.autocast("cuda", enabled=True):
            _, out = model(x)
            loss = criterion(out, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        for _ in range(warmup):
            one_step()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(steps):
            one_step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    except RuntimeError as e:
        if args_verbose:
            import traceback
            traceback.print_exc()
        return None, None, str(e)
    finally:
        del model, opt, scaler, criterion, x, y
        torch.cuda.empty_cache()
    return dt * 1000, peak, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, nargs="+", default=[8, 16, 24, 28, 32])
    ap.add_argument("--nb", type=int, nargs="+", default=[48000])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    global args_verbose
    args_verbose = args.verbose

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MiB")
    print()
    print(f"{'bs':>4} {'nb_samp':>8} {'ms/step':>9} {'峰值MiB':>9} "
          f"{'秒/千样本':>10}  备注")
    print("-" * 62)
    best = None
    for nb in args.nb:
        for bs in args.bs:
            ms, peak, err = bench(bs, nb, args.steps)
            if err:
                print(f"{bs:>4} {nb:>8} {'—':>9} {'—':>9} {'—':>10}  OOM/失败")
                continue
            per1k = ms / bs * 1000 / 1000  # 秒/千样本
            note = ""
            if peak > 7000:
                note = "⚠ 显存吃紧"
            print(f"{bs:>4} {nb:>8} {ms:>9.1f} {peak:>9.0f} {per1k:>10.1f}  {note}")
            if best is None or per1k < best[0]:
                best = (per1k, bs, nb, ms, peak)
            sys.stdout.flush()
    if best:
        per1k, bs, nb, ms, peak = best
        print("-" * 62)
        print(f"最快吞吐: bs={bs} nb_samp={nb}  ({ms:.0f} ms/step, {peak:.0f} MiB)")
        n = 25380
        ep_min = ms / 1000 * (n / bs) / 60
        print(f"折算训练: {n} 条 -> {ep_min:.1f} 分钟/epoch, "
              f"30 epoch ≈ {ep_min*30/60:.1f} 小时")


if __name__ == "__main__":
    main()
