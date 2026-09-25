# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 演示音频本地准备脚本（任务书 P0-5 合规）
========================================================
ASVspoof 许可**禁止再分发**数据集里的 flac，demo 不能打包进仓库。

本脚本：
  - 本机有 LA 数据（VERICALL_ASVSPOOF_LA）时，把三场景 demo 音频拷贝到
    assets/demo_audio/（仅本地用，已被 .gitignore 排除，不会入库）；
  - 没有数据集时，打印提示：用 GPT-SoVITS 自生成（红队集 P2-1 建成后切换为
    完全自产，彻底无版权风险），并给出 assets/demo_audio/ 的目标文件名。

用法：
    python scripts/prepare_demo_audio.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import DEV_FLAC, ROOT  # noqa: E402

# 三场景 demo 音频（与 src/api/server.py 的 DEMO_SCENARIOS 一一对应）
SCENARIOS = {
    "A": ("LA_D_1105538.flac", "家人本人来电（真声，音色匹配）"),
    "B": ("LA_D_1002910.flac", "AIGC 冒充家人（同音色伪造声）"),
    "C": ("LA_D_1090286.flac", "陌生人来电（真声，音色不匹配）"),
}

DEST = ROOT / "assets" / "demo_audio"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if not DEV_FLAC.is_dir():
        print("[提示] 未找到 ASVspoof LA 数据（VERICALL_ASVSPOOF_LA）。")
        print("[提示] 演示音频合规方案：用 GPT-SoVITS 少样本克隆自生成，")
        print(f"       放到 {DEST} 下，文件名建议：")
        for k, (fname, title) in SCENARIOS.items():
            print(f"         {k}: {fname}   # {title}")
        print("[提示] 红队方言集（P2-1）建成后，demo 将完全切换为自产音频，彻底无版权风险。")
        return 0

    copied = 0
    for k, (fname, title) in SCENARIOS.items():
        src = DEV_FLAC / fname
        if not src.is_file():
            print(f"[跳过] {k} 缺失: {src}")
            continue
        dst = DEST / fname
        shutil.copyfile(src, dst)
        print(f"[拷贝] {k} {title} -> {dst}")
        copied += 1
    print(f"\n完成：{copied}/3 个 demo 音频已准备到 {DEST}（已被 .gitignore 排除，不入库）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
