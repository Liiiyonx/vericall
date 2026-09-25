#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 脱离会话的 AASIST 训练启动器（Windows）。

为什么需要它:
  用普通方式起的后台训练，会随本次会话结束被一起 kill。30 个 epoch 要跑 3 小时，
  不可能一直开着会话等。本脚本用 DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP
  启动训练，并让启动器自身立刻退出 —— 训练进程随即被操作系统"收养"（父进程消失，
  重新挂到系统根进程），因此按进程树清理也找不到它，能真正跑完。

  配合 main.py 的 --resume 与 ckpt_every 周期存档：
    即使中途断电/被杀，下次 `python scripts/launch_training.py --resume auto`
    也能接着上次的 epoch 继续，进度不丢。

用法:
  python scripts/launch_training_detached.py                 # 启动（默认带 --resume auto）
  python scripts/launch_training_detached.py --check         # 只看训练是否还活着
  python scripts/launch_training_detached.py --stop          # 停止训练

运行时信息写在 <VERICALL_DATA_ROOT>/train_pid.txt（PID）与 training_ep30.log（日志）。
路径由 VERICALL_DATA_ROOT 或项目根 .env 控制，见 src/paths.py。
"""
import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "src")))
from paths import AASIST_DIR, CONFIG_DIR, TRAIN_LOG, PID_FILE  # noqa: E402

AASIST_DIR = str(AASIST_DIR)
CONFIG = str(CONFIG_DIR / "AASIST_5060.conf")
LOG = str(TRAIN_LOG)
PID_FILE = str(PID_FILE)

# Windows: 脱离控制台 + 独立进程组
DETACHED = 0x00000008        # DETACHED_PROCESS
NEW_GROUP = 0x00000200       # CREATE_NEW_PROCESS_GROUP


def _pid_alive(pid):
    """用 tasklist 判断 PID 是否存活(不依赖第三方库)。

    ⚠ 中文 Windows 的 tasklist 输出是 GBK 编码, 用默认 utf-8 解码会抛
    UnicodeDecodeError; 若在这里被 except 吞掉就会"一律判为已退出", 造成
    明明在训练却显示 ❌ 的假阴性。必须显式忽略解码错误。
    """
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, timeout=15)
        txt = out.stdout.decode("utf-8", errors="ignore")
        return str(pid) in txt
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 存活检查失败: {e}")
        return False


def _read_pid():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def check():
    pid = _read_pid()
    if pid is None:
        print("[check] 没有记录到训练 PID")
        return 1
    alive = _pid_alive(pid)
    print(f"[check] 训练 PID={pid}  {'存活 ✅' if alive else '已退出 ❌'}")
    if os.path.exists(LOG):
        mt = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(LOG)))
        print(f"[check] 日志最后写入: {mt}  -> {LOG}")
        with open(LOG, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l for l in f.readlines()
                     if l.strip() and "Warning" not in l and "warn" not in l]
        for l in lines[-3:]:
            print("        " + l.rstrip())
    return 0 if alive else 2


def stop():
    pid = _read_pid()
    if pid is None:
        print("[stop] 没有记录到训练 PID")
        return 1
    if not _pid_alive(pid):
        print(f"[stop] PID {pid} 已不存在")
        return 0
    subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                   capture_output=True, text=True, timeout=30)
    time.sleep(3)
    print(f"[stop] 已请求终止 PID {pid}, 存活={_pid_alive(pid)}")
    return 0


def launch(extra_args):
    os.makedirs(DATA_DIR, exist_ok=True)
    cmd = [sys.executable, "-u", "main.py", "--config", CONFIG] + extra_args
    logf = open(LOG, "a", encoding="utf-8")
    logf.write("\n" + "=" * 60 + "\n")
    logf.write("[detached] {} launch: {}\n".format(
        time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
    logf.flush()

    p = subprocess.Popen(
        cmd, cwd=AASIST_DIR, stdout=logf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, close_fds=True,
        creationflags=DETACHED | NEW_GROUP)
    with open(PID_FILE, "w") as f:
        f.write(str(p.pid))
    print(f"[launch] 训练已脱离启动 (PID={p.pid})")
    print(f"[launch] 日志: {LOG}")
    print(f"[launch] 查看进度: python scripts/training_monitor.py --epochs <总epoch>")
    print(f"[launch] 检查存活: python scripts/launch_training_detached.py --check")
    # 启动器立刻退出, 训练进程被系统收养, 不随本进程/会话结束而终止
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="强制从头训练")
    ap.add_argument("--epochs", type=int, default=None)
    args, extra = ap.parse_known_args()

    if args.check:
        return check()
    if args.stop:
        return stop()

    extra_args = list(extra)
    if not args.no_resume and "--resume" not in extra_args:
        extra_args += ["--resume", "auto"]
    if args.epochs:
        extra_args += ["--num_epochs", str(args.epochs)]
    return launch(extra_args)


if __name__ == "__main__":
    sys.exit(main())
