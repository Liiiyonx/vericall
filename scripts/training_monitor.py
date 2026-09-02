#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — AASIST 训练实时进度看板。

解决的问题:
  训练日志只有一行行 "epoch000 loss=0.1234", 既看不到速度也看不到 ETA,
  更看不到还剩几个 epoch 出 best.pth。本脚本把日志 + 权重目录 + GPU 状态
  汇总成一块看板, 并支持 --watch 持续刷新。

数据源:
  1. 训练日志 (默认 <VERICALL_DATA_ROOT>/training_ep30.log, 见 src/paths.py)
     关键行: "Start training epoch000" / "epoch{:03d} loss={:.4f}"
  2. 实验目录 exp_result/XXX/weights/ 下的 checkpoint
     - epoch_{N}.pth       周期性保底 (ckpt_every)
     - epoch_{N}_{EER}.pth dev 集刷新最佳时落盘
     - swa.pth / best.pth  训练结束时落盘
  3. nvidia-smi 的 GPU 利用率与显存

跨会话计时:
  每个 epoch 行的"首次出现时间"持久化到 .monitor_state.json,
  因此关掉看板再打开, ETA 依然准确(不依赖看板自身的运行时长)。

用法:
  python scripts/training_monitor.py                 # 打印一次
  python scripts/training_monitor.py --watch         # 每 30s 刷新
  python scripts/training_monitor.py --watch -i 10   # 每 10s 刷新
  python scripts/training_monitor.py --log <path>    # 指定日志
"""
import os
import re
import sys
import json
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "src")))
from paths import TRAIN_LOG, EXP_RESULT_DIR  # noqa: E402

DEFAULT_LOG = str(TRAIN_LOG)
EXP_ROOT = str(EXP_RESULT_DIR)

RE_EPOCH_LINE = re.compile(r"epoch(\d+)\s+loss=([\d.]+)")
RE_START = re.compile(r"Start training epoch(\d+)")


def _human_sec(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m{s%60:02d}s"


def _gpu():
    """返回 (util%, used MiB, total MiB); 失败返回 None"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        u, used, total = out.stdout.strip().split(",")
        return int(u.strip()), int(used.strip()), int(total.strip())
    except Exception:
        return None


def _latest_exp():
    """取 mtime 最新的实验目录"""
    if not os.path.isdir(EXP_ROOT):
        return None
    subs = [os.path.join(EXP_ROOT, d) for d in os.listdir(EXP_ROOT)]
    subs = [d for d in subs if os.path.isdir(d)]
    if not subs:
        return None
    return max(subs, key=os.path.getmtime)


def _scan_weights(exp_dir):
    """扫描权重目录, 返回 (周期ckpt列表, best/dev列表, 结束态列表)"""
    wdir = os.path.join(exp_dir, "weights")
    periodic, dev_best, final = [], [], []
    if not os.path.isdir(wdir):
        return periodic, dev_best, final
    for f in os.listdir(wdir):
        p = os.path.join(wdir, f)
        if not f.endswith(".pth"):
            continue
        item = (f, os.path.getmtime(p), os.path.getsize(p))
        if f in ("best.pth", "swa.pth"):
            final.append(item)
        elif re.fullmatch(r"epoch_\d+\.pth", f):
            periodic.append(item)
        elif re.fullmatch(r"epoch_\d+_[\d.]+\.pth", f):
            dev_best.append(item)
    periodic.sort(key=lambda x: x[0])
    dev_best.sort(key=lambda x: float(re.findall(r"_([\d.]+)\.pth", x[0])[0]))
    return periodic, dev_best, final


def _read_log(path):
    if not os.path.exists(path):
        return [], None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    epochs = []
    start_ep = None
    for ln in lines:
        m = RE_START.search(ln)
        if m:
            start_ep = int(m.group(1))
        m = RE_EPOCH_LINE.search(ln)
        if m:
            epochs.append((int(m.group(1)), float(m.group(2))))
    return epochs, start_ep


def _load_state(state_path):
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state_path, state):
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def render(log_path, exp_dir, num_epochs=None):
    epochs, start_ep = _read_log(log_path)
    state_path = os.path.join(exp_dir, ".monitor_state.json") if exp_dir else None
    state = _load_state(state_path) if state_path else {}
    now = time.time()

    # 记录每个 epoch 完成行的首次出现时间(跨会话持久化)
    changed = False
    for ep, loss in epochs:
        k = f"epoch_{ep}"
        if k not in state:
            state[k] = {"t": now, "loss": loss}
            changed = True
    if changed and state_path:
        _save_state(state_path, state)

    print("=" * 62)
    print(" 谛听 VeriCall · AASIST 训练看板   ", time.strftime("%H:%M:%S"))
    print("=" * 62)

    if not os.path.exists(log_path):
        print(f"[!] 日志文件不存在: {log_path}")
        return

    log_mtime = os.path.getmtime(log_path)
    stale = now - log_mtime
    print(f"日志      : {log_path}")
    print(f"最后写入  : {stale:.0f}s 前" + ("   (活跃)" if stale < 180 else "   ⚠ 疑似停滞"))

    g = _gpu()
    if g:
        print(f"GPU       : {g[0]}%  |  显存 {g[1]}/{g[2]} MiB")
    if exp_dir:
        print(f"实验目录  : {os.path.basename(exp_dir)}")

    if not epochs:
        print("-" * 62)
        print("尚未完成任何 epoch(首 epoch 需加载并解码全量 25380 条音频)。")
        if start_ep is not None:
            print(f"已启动: epoch{start_ep:03d} 训练中...")
        return

    # epoch 计时: 用相邻 epoch 完成时间差
    done = sorted(set(ep for ep, _ in epochs))
    times = []
    for ep in done:
        t = state.get(f"epoch_{ep}", {}).get("t")
        if t:
            times.append((ep, t))
    times.sort()

    per_epoch = None
    if len(times) >= 2:
        span = times[-1][1] - times[0][1]
        per_epoch = span / (times[-1][0] - times[0][0])
    elif len(times) == 1 and times[0][0] > 0:
        # 只有一条记录且不是 epoch0, 无法估算
        per_epoch = None

    last_ep, last_loss = epochs[-1]
    print("-" * 62)
    print(f"已完成    : epoch{last_ep:03d}   loss={last_loss:.4f}")
    if len(epochs) >= 2:
        prev = epochs[-2][1]
        delta = last_loss - prev
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        print(f"损失变化  : {prev:.4f} → {last_loss:.4f}  {arrow}{abs(delta):.4f}")

    if num_epochs:
        pct = min(100.0, 100.0 * (last_ep + 1) / num_epochs)
        bar_n = int(pct / 2.5)
        print(f"总进度    : [{('#' * bar_n).ljust(40, '.')}] {pct:5.1f}%  "
              f"({last_ep + 1}/{num_epochs})")
        if per_epoch:
            remain = (num_epochs - last_ep - 1) * per_epoch
            print(f"单 epoch  : ~{_human_sec(per_epoch)}    预计剩余: ~{_human_sec(remain)}")

    periodic, dev_best, final = _scan_weights(exp_dir)
    print("-" * 62)
    if periodic:
        print(f"周期存档  : {len(periodic)} 个, 最新 {periodic[-1][0]} "
              f"({_human_sec(now - periodic[-1][1])}前)")
    else:
        print("周期存档  : 无 (ckpt_every 到期后才落盘)")
    if dev_best:
        best = dev_best[0]
        eer = float(re.findall(r"_([\d.]+)\.pth", best[0])[0])
        print(f"dev 最佳  : {best[0]}   dev-EER = {eer:.3f}%")
    else:
        print("dev 最佳  : 无 (skip_per_epoch_eval=True 时不逐 epoch 评估)")
    if final:
        names = ", ".join(f[0] for f in final)
        print(f"结束态    : ✅ {names}  —— 训练已完成, 可跑 EER 评估")
    else:
        print("结束态    : 未落盘 (训练结束时才生成 swa.pth / best.pth)")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--exp", default=None, help="实验目录, 默认取最新的")
    ap.add_argument("--epochs", type=int, default=None, help="总 epoch 数, 用于算进度与 ETA")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("-i", "--interval", type=int, default=30)
    args = ap.parse_args()

    exp = args.exp or _latest_exp()
    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                render(args.log, exp, args.epochs)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[monitor] 已停止")
    else:
        render(args.log, exp, args.epochs)


if __name__ == "__main__":
    main()
