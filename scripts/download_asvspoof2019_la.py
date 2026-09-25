# -*- coding: utf-8 -*-
"""
下载 ASVspoof 2019 LA 数据集（爱丁堡 DataShare 开放获取，无需注册）。

用法:
    python download_asvspoof2019_la.py --dest ../data/raw

说明:
    - LA.zip 约 25 GB（flac 压缩），支持断点续传，中断后重跑即可。
    - 下载完成后自动校验并解压到 data/raw/ASVspoof2019_LA/。
    - PA.zip（物理访问）本项目暂不需要，默认不下载。
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

LA_URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
PAGE = "https://datashare.ed.ac.uk/handle/10283/3336"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent / "data" / "raw"))
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "LA.zip"

    if not zip_path.exists():
        print(f"[1/2] 下载 LA.zip -> {zip_path}")
        print(f"      约 25GB，支持断点续传；页面: {PAGE}")
        # curl -L 跟随重定向, -C - 断点续传, --retry 容错
        rc = subprocess.call([
            "curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10",
            "--progress-bar", "-o", str(zip_path), LA_URL,
        ])
        if rc != 0:
            sys.exit(f"下载中断 (exit {rc})，重新运行本脚本即可续传。")
    else:
        print(f"[1/2] 已存在 {zip_path}，跳过下载（如需重下请先删除）")

    out_dir = dest / "ASVspoof2019_LA"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[2/2] {out_dir} 已解压，跳过")
        return

    print(f"[2/2] 解压到 {out_dir}（约需 10-20 分钟）...")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            sys.exit(f"zip 校验失败: {bad}，请删除 LA.zip 后重新下载。")
        zf.extractall(dest)

    flac_count = len(list(out_dir.rglob("*.flac")))
    print(f"完成: {flac_count} 个 flac 文件。协议文件在 {out_dir}/ASVspoof2019_LA_cm_protocols/")


if __name__ == "__main__":
    main()
