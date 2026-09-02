# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 统一路径配置
================================================================
项目里所有绝对路径集中在此处定义，业务代码禁止再写死盘符。

优先级（高 → 低）：
    1. 进程环境变量          例：set VERICALL_DATA_ROOT=E:/vc_data
    2. 项目根 .env 文件      复制 .env.example 改名为 .env
    3. 代码内默认值          本机开发习惯（D:/VeriCall_data）

用法：
    # src 下的模块
    from paths import DEV_FLAC, SENSEVOICE_DIR

    # scripts 下的脚本（需先补 sys.path）
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    from paths import DATA_ROOT, ASVSPOOF_LA
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 根目录
ROOT = Path(__file__).resolve().parent.parent          # 项目根 src/ 的上一级
SRC_DIR = ROOT / "src"
WEB_DIR = ROOT / "web"
DOCS_DIR = ROOT / "docs"
CONFIG_DIR = ROOT / "configs"

# ---------------------------------------------------------------- .env
def _load_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 解析（零依赖）：KEY=VALUE 每行一条，# 开头为注释。"""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


_DOTENV = _load_dotenv(ROOT / ".env")


def env(key: str, default: str | None = None) -> str | None:
    """读配置：环境变量 > .env > 默认值。"""
    return os.environ.get(key) or _DOTENV.get(key) or default


def _p(key: str, default: str) -> Path:
    """读路径型配置，统一展开 ~ 并转成绝对路径。"""
    return Path(env(key, default) or default).expanduser().resolve()


# ---------------------------------------------------------------- 数据区
# 大文件（数据集 / 模型 / 日志）默认放 D 盘，C 盘空间不够。
DATA_ROOT = _p("VERICALL_DATA_ROOT", "D:/VeriCall_data")
MODELS_ROOT = _p("VERICALL_MODELS_ROOT", str(DATA_ROOT / "models"))

# ASVspoof2019 LA：解压后形如 ASVspoof2019_LA/LA/{flac, protocol...}
ASVSPOOF_LA = _p("VERICALL_ASVSPOOF_LA", str(DATA_ROOT / "ASVspoof2019_LA" / "LA"))
DEV_FLAC = ASVSPOOF_LA / "ASVspoof2019_LA_dev" / "flac"
EVAL_FLAC = ASVSPOOF_LA / "ASVspoof2019_LA_eval" / "flac"
TRAIN_FLAC = ASVSPOOF_LA / "ASVspoof2019_LA_train" / "flac"

LA_ZIP = _p("VERICALL_LA_ZIP", str(DATA_ROOT / "LA.zip"))
PARQUET_DIR = _p("VERICALL_PARQUET_DIR", str(DATA_ROOT / "parquet"))

# 冒烟测试用的迷你数据集
SMOKE_ROOT = _p("VERICALL_SMOKE_ROOT", str(DATA_ROOT / "smoke_ASVspoof2019_LA"))
SMOKE_LA = SMOKE_ROOT / "LA"

# ---------------------------------------------------------------- 模型
SENSEVOICE_DIR = _p("VERICALL_SENSEVOICE_DIR", str(MODELS_ROOT / "SenseVoiceSmall"))
SV_EXAMPLE = SENSEVOICE_DIR / "example"          # 自带 zh.mp3 / en.mp3，用于自测
OLLAMA_HOST = env("VERICALL_OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434"
OLLAMA_MODEL = env("VERICALL_OLLAMA_MODEL", "deepseek-r1:8b") or "deepseek-r1:8b"

# AASIST（clovaai/aasist 需自行 clone 到 external/，见 README）
AASIST_DIR = _p("VERICALL_AASIST_DIR", str(ROOT / "external" / "aasist"))
AASIST_EXP = env("VERICALL_AASIST_EXP", "LA_AASIST_5060_ep24_bs16") or "LA_AASIST_5060_ep24_bs16"
EXP_DIR = AASIST_DIR / "exp_result" / AASIST_EXP
EXP_RESULT_DIR = AASIST_DIR / "exp_result"

# ---------------------------------------------------------------- 项目内数据
DATA_DIR = ROOT / "data"
VOICE_DIR = DATA_DIR / "raw" / "family_voice"     # 家人声纹库
VOICEPRINT_DIR = DATA_DIR / "voiceprints"         # 声纹向量持久化（*.npy）
UPLOAD_DIR = DATA_DIR / "uploads"                 # API 上传暂存
HIST_FILE = DATA_DIR / "history.jsonl"            # 检测历史
REDTEAM_DIR = DATA_DIR / "redteam"                # 红队方言样本
DEMO_CACHE = DATA_DIR / "demo_cache.json"         # 离线演示缓存（A/B/C 三场景）

# 离线降级：VERICALL_OFFLINE=1 时跳过 Ollama/SenseVoice/torch，用缓存跑演示
# （答辩现场 GPU/Ollama/网络任一挂掉都不致命——这是性价比最高的兜底）
OFFLINE = (env("VERICALL_OFFLINE", "0") or "0") == "1"

# ---------------------------------------------------------------- 日志
TRAIN_LOG = _p("VERICALL_TRAIN_LOG", str(DATA_ROOT / "training_ep30.log"))
BENCH_LOG = _p("VERICALL_BENCH_LOG", str(DATA_ROOT / "bench.log"))
PID_FILE = _p("VERICALL_PID_FILE", str(DATA_ROOT / "train_pid.txt"))

# ---------------------------------------------------------------- 设备
DEVICE = env("VERICALL_DEVICE", "cuda") or "cuda"
API_PORT = int(env("VERICALL_PORT", "8000") or "8000")


def latest_exp_dir() -> Path:
    """返回 exp_result 下最近修改的实验目录；没有则返回配置的 EXP_DIR。"""
    if not EXP_RESULT_DIR.is_dir():
        return EXP_DIR
    cands = [d for d in EXP_RESULT_DIR.iterdir() if d.is_dir()]
    if not cands:
        return EXP_DIR
    return max(cands, key=lambda d: d.stat().st_mtime)


def ensure_data_dirs() -> None:
    """创建项目内数据目录（数据集目录不自动建，可能不存在）。"""
    for d in (VOICE_DIR, VOICEPRINT_DIR, UPLOAD_DIR, REDTEAM_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def report() -> str:
    """打印当前生效配置，用于排障。"""
    rows = [
        ("项目根 ROOT", ROOT),
        ("数据区 DATA_ROOT", DATA_ROOT),
        ("ASVspoof LA", ASVSPOOF_LA),
        ("SenseVoice", SENSEVOICE_DIR),
        ("AASIST", AASIST_DIR),
        ("实验目录", EXP_DIR),
        ("声纹库", VOICE_DIR),
        ("声纹向量", VOICEPRINT_DIR),
        ("离线演示缓存", DEMO_CACHE),
        ("Ollama", f"{OLLAMA_HOST} @ {OLLAMA_MODEL}"),
        ("推理设备", DEVICE),
        ("离线降级模式", "已开启" if OFFLINE else "关闭"),
    ]
    width = max(len(k) for k, _ in rows)
    # 非路径项/开关项不判定存在性
    skip = {"Ollama", "推理设备", "离线降级模式"}

    def mark(k: str, v) -> str:
        if k in skip:
            return ""
        return "存在" if Path(v).exists() else "缺失"

    return "\n".join(
        f"{k.ljust(width)}  {v}   [{mark(k, v)}]" for k, v in rows
    )


if __name__ == "__main__":
    print(report())
