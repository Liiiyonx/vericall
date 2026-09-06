#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — AASIST 正式训练一键启动器。

职责：
  1. 检查 LA.zip 是否已下载完整（实际约 7.6GB，datashare 落地页标注 6.32GB 偏小）；
  2. 若尚未解压，解压到 <VERICALL_DATA_ROOT>/ASVspoof2019_LA/，
     得到 <VERICALL_DATA_ROOT>/ASVspoof2019_LA/LA/... 目录；
  3. 调用 external/aasist/main.py 用 configs/AASIST_5060.conf 开训。

用法：
  python scripts/launch_training.py            # 默认 VERICALL_DATA_ROOT
  python scripts/launch_training.py --check    # 只检查数据是否就绪，不训练

说明：
  - 解压用 Python 标准库 zipfile（无需 7z）；7.6GB 解压约 4-6 分钟。
  - 训练用 configs/AASIST_5060.conf 指定的 vericall/python 解释器在 main.py 内通过
    当前解释器运行，故请用 vericall 环境的 python 调本脚本。
  - 数据根目录由 VERICALL_DATA_ROOT 环境变量或项目根 .env 控制，见 src/paths.py。
"""
import os
import sys
import zipfile
import argparse
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "src")))
from paths import DATA_ROOT, LA_ZIP, AASIST_DIR, CONFIG_DIR  # noqa: E402

DATA_DIR = str(DATA_ROOT)
LA_ZIP = str(LA_ZIP)
EXPECTED_ZIP_BYTES = 6_320_000_000  # 约 6.32GB，做下界校验
AASIST_DIR = str(AASIST_DIR)
CONFIG = str(CONFIG_DIR / "AASIST_5060.conf")


def _human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _zip_complete(path):
    """轻量完整性判定: 能读到 central directory 即视为完整(截断文件会抛异常)"""
    try:
        with zipfile.ZipFile(path) as z:
            _ = z.namelist()
        return True
    except Exception:
        return False


def check_ready(verbose=True):
    """返回 (zip_ok, extracted_ok)"""
    # 真实 LA.zip 约 7.6GB(datashare 落地页标注的 6.32GB 偏小)。以 zip 完整性为准,
    # 不依赖精确字节阈值; 同时设 4.8GB 下界, 防止半截文件被误判为就绪。
    zip_ok = (os.path.exists(LA_ZIP)
              and os.path.getsize(LA_ZIP) >= 4_800_000_000
              and _zip_complete(LA_ZIP))
    la_dir = os.path.join(DATA_DIR, "ASVspoof2019_LA", "LA")
    proto = os.path.join(la_dir, "ASVspoof2019_LA_cm_protocols")
    train_flac = os.path.join(la_dir, "ASVspoof2019_LA_train", "flac")
    extracted_ok = os.path.isdir(proto) and os.path.isdir(train_flac)
    if verbose:
        z = os.path.getsize(LA_ZIP) if os.path.exists(LA_ZIP) else 0
        print(f"[check] LA.zip: {_human(z)}  ({'OK' if zip_ok else '未完成/缺失'})")
        print(f"[check] 已解压目录: {'OK' if extracted_ok else '未解压'}")
    return zip_ok, extracted_ok


def extract():
    # zip 顶层为 LA/, 解压到 ASVspoof2019_LA/ 得到 ASVspoof2019_LA/LA/...,
    # 与 configs/AASIST_5060.conf 的 database_path 严格对齐。
    extract_root = os.path.join(DATA_DIR, "ASVspoof2019_LA")
    os.makedirs(extract_root, exist_ok=True)
    print(f"[extract] 解压 {LA_ZIP} -> {extract_root} ...")
    t0 = time.time()
    with zipfile.ZipFile(LA_ZIP, "r") as z:
        total = len(z.namelist())
        for i, name in enumerate(z.namelist(), 1):
            z.extract(name, extract_root)
            if i % 500 == 0:
                print(f"  {i}/{total} ({100*i//total}%)")
    print(f"[extract] 完成, 用时 {time.time()-t0:.0f}s")


def main():
    global DATA_DIR, LA_ZIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--check", action="store_true", help="只检查数据就绪, 不训练")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 num_epochs")
    ap.add_argument("--resume", default="auto",
                    help="断点续训: auto(默认, 挑本实验最新 epoch_{N}.pth) / "
                         "路径 / none(强制从头训练)")
    args = ap.parse_args()

    DATA_DIR = args.data_dir
    LA_ZIP = os.path.join(DATA_DIR, "LA.zip")

    zip_ok, extracted_ok = check_ready()
    if args.check:
        if zip_ok and extracted_ok:
            print("[check] ✅ 数据就绪, 可开始训练")
            return 0
        print("[check] ❌ 数据未就绪")
        return 1

    if not zip_ok:
        print("[launch] ❌ LA.zip 尚未下载完整, 请先下载 (约 6.32GB)")
        return 1
    if not extracted_ok:
        extract()
        _, extracted_ok = check_ready(verbose=False)
        if not extracted_ok:
            print("[launch] ❌ 解压后目录结构异常, 请检查 LA.zip 完整性")
            return 1

    # -u: 无缓冲, 训练进度(epoch/损失)实时写入日志, 方便用户查看
    cmd = [sys.executable, "-u", "main.py", "--config", CONFIG]
    if args.epochs:
        cmd += ["--num_epochs", str(args.epochs)]
    # 断点续训: 后台训练常被会话结束 kill, 续训避免每次从 epoch0 重来
    if args.resume and args.resume.lower() != "none":
        cmd += ["--resume", args.resume]
    print(f"[launch] 启动训练: {' '.join(cmd)} (cwd={AASIST_DIR})")
    rc = subprocess.call(cmd, cwd=AASIST_DIR)
    print(f"[launch] main.py 退出码 = {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
