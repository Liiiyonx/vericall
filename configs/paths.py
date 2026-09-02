# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 全局路径配置（唯一路径来源）
=============================================
所有"数据/模型在 D 盘"的路径集中在这里，代码里不再出现任何写死的盘符。
评委/队友机器上只需设置环境变量 VERICALL_DATA 指向数据根目录即可。

目录约定（默认）：
    D:/Vericall_data/ 或 %VERICALL_DATA%/
    ├── ASVspoof2019_LA/LA/        # 通道① 训练/评测数据
    └── models/SenseVoiceSmall/    # 通道③ ASR 权重（modelscope 下载）

项目内路径（随仓库走，不在此配置）：
    data/voiceprints/              # 声纹向量落盘（P0-6）
    data/cache/transcripts.json    # ASR 转写缓存（P0-4 离线降级）
    assets/demo_audio/             # 演示音频（本地生成，不入库）
"""
from __future__ import annotations

import os
from pathlib import Path


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default))


# ---------------------------------------------------------------------- #
# 外部数据根（本机默认 D 盘，其他机器用环境变量覆盖）
# ---------------------------------------------------------------------- #
VERICALL_DATA = _p("VERICALL_DATA", "D:/VeriCall_data")

# SenseVoice ASR 权重目录（含 model.pt）
SENSEVOICE_DIR = _p("VERICALL_SENSEVOICE", str(VERICALL_DATA / "models" / "SenseVoiceSmall"))

# ASVspoof2019 LA 数据集根（含 *_cm_protocols / *_train / *_dev / *_eval）
ASVSPOOF2019_LA = _p("VERICALL_ASVSPOOF_LA", str(VERICALL_DATA / "ASVspoof2019_LA" / "LA"))

# 训练日志（训练看板用）
TRAINING_LOG = _p("VERICALL_TRAINING_LOG", str(VERICALL_DATA / "training_ep30.log"))

# ---------------------------------------------------------------------- #
# 项目内路径（相对仓库根）
# ---------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VOICEPRINT_DIR = DATA_DIR / "voiceprints"        # 声纹落盘
TRANSCRIPT_CACHE = DATA_DIR / "cache" / "transcripts.json"
DEMO_AUDIO_DIR = PROJECT_ROOT / "assets" / "demo_audio"

# ---------------------------------------------------------------------- #
# 运行模式开关
# ---------------------------------------------------------------------- #
def offline() -> bool:
    """VERICALL_OFFLINE=1 时启用演示降级（规则评分器 + 转写缓存）。"""
    return os.environ.get("VERICALL_OFFLINE", "0") == "1"


def device() -> str:
    """推理设备：VERICALL_DEVICE 可强制覆盖，默认有卡用卡。"""
    env = os.environ.get("VERICALL_DEVICE")
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
