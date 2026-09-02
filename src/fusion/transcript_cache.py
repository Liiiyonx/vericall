# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 转写文本缓存（通道③ ASR 降级）
============================================
以音频文件 md5 为 key，缓存 SenseVoice 的历史转写文本到 data/cache/transcripts.json。
- 在线跑过一次后，断网/无 SenseVoice 时也能复用转写（配合规则评分器兜底）。
- 任务书 P0-4 通道③ ASR 降级策略的实现：demo 音频首次跑完即有缓存。

零依赖（仅标准库 + numpy 用于 md5 读取）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paths import DATA_DIR

CACHE_FILE = DATA_DIR / "cache" / "transcripts.json"


def _md5(audio_path: str) -> str:
    """对音频文件内容取 md5（小文件直接读；大文件分块）。"""
    h = hashlib.md5()
    try:
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        # 读不到文件就退化用路径字符串做 key（仍能跨进程命中同一路径）
        return hashlib.md5(str(audio_path).encode("utf-8")).hexdigest()
    return h.hexdigest()


def _load() -> dict:
    if not CACHE_FILE.is_file():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def get(audio_path: str) -> str | None:
    """返回该音频的缓存转写文本；未命中返回 None。"""
    key = _md5(audio_path)
    return _load().get(key)


def put(audio_path: str, transcript: str) -> None:
    """写入该音频的转写文本。"""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    store = _load()
    store[_md5(audio_path)] = transcript
    try:
        CACHE_FILE.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
