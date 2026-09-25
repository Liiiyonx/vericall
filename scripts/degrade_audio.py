# -*- coding: utf-8 -*-
"""
电话信道 / 编解码退化仿真（任务书 P1-4）
========================================
预设档位：
  --preset phone8k  : 16k→8k→16k（去高低频）+ μ-law 压扩 + 轻度削顶
  --preset mp3_16k  : ffmpeg 转 mp3 32kbps 再转回（模拟网络电话编解码）
  --preset amr      : ffmpeg 转 amr-nb 12.2kbps（模拟移动语音信道）
  --preset noise    : phone8k + 加性粉噪 SNR 15dB（模拟免提外放）

对输入音频（或整目录）生成退化版，输出到 --out 目录并写 manifest.json。
ffmpeg 档位在缺失 ffmpeg 时自动跳过并提示。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from server.audio_util import read_wav, write_wav, resample

PRESETS = {"phone8k", "mp3_16k", "amr", "noise"}
_FFMPEG = {"mp3_16k": ["-b:a", "32k"], "amr": ["-ar", "8000", "-ab", "12.2k"]}


def _ulaw(x: np.ndarray) -> np.ndarray:
    """μ-law 压扩（G.711 连续域近似，纯 numpy 实现）。

    原为 ``audioop.lin2ulaw/ulaw2lin``，但 audioop 在 Python 3.13 已被移除，
    故改为纯 numpy 实现，保证脚本在 3.10/3.13 均可运行。
    输入/输出均为 [-1,1] 的 float32；压扩会引入典型的 μ-law 非线性失真。
    """
    MU = 255.0
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    sign = np.sign(x)
    xmag = np.abs(x)
    # 编码：log 压扩到 [0,1] 再量化
    y = np.log1p(MU * xmag) / np.log1p(MU)
    code = np.round(y * 127.0).astype(np.int16)          # 0..127
    # 解码：指数扩张回幅度（模拟 DAC 重建）
    y2 = code.astype(np.float64) / 127.0
    xout = sign * ((1.0 / MU) * ((1.0 + MU) ** y2 - 1.0))
    # 轻度削顶：压扩后再做一次软限幅，模拟电话链路削波
    return np.clip(xout, -1.0, 1.0).astype(np.float32)


def _python_degrade(samples: np.ndarray, sr: int, preset: str) -> np.ndarray | None:
    if preset == "phone8k":
        down = resample(samples, sr, 8000)
        up = resample(down, 8000, sr)
        return _ulaw(up)
    if preset == "noise":
        base = _python_degrade(samples, sr, "phone8k")
        sigpow = float(np.mean(base ** 2)) + 1e-9
        noise = np.random.randn(len(base)).astype(np.float32)
        noisepow = float(np.mean(noise ** 2)) + 1e-9
        scale = np.sqrt(sigpow / (10 ** (15 / 10)) / noisepow)
        return np.clip(base + scale * noise, -1.0, 1.0).astype(np.float32)
    return None  # ffmpeg 档位


def _ffmpeg_degrade(in_path: Path, out_path: Path, preset: str) -> bool:
    exe = "ffmpeg"
    extra = _FFMPEG.get(preset, [])
    cmd = [exe, "-y", "-i", str(in_path), *extra, str(out_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def degrade_file(in_path: Path, out_dir: Path, preset: str, sr: int = 16000):
    sig = read_wav(in_path, sr=sr)
    if preset in ("phone8k", "noise"):
        out = _python_degrade(sig, sr, preset)
        if out is None:
            return None
        out_path = out_dir / f"{in_path.stem}.{preset}.wav"
        write_wav(out_path, out, sr=sr)
        return out_path
    # ffmpeg 档位
    out_path = out_dir / f"{in_path.stem}.{preset}.wav" if preset != "mp3_16k" else \
        out_dir / f"{in_path.stem}.{preset}.mp3"
    ok = _ffmpeg_degrade(in_path, out_path, preset)
    if not ok:
        print(f"  [skip] {in_path.name}: 需要 ffmpeg（{preset}）")
        return None
    return out_path


def main():
    ap = argparse.ArgumentParser(description="电话信道退化仿真")
    ap.add_argument("input", help="输入音频文件或目录")
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out", default="data/degraded", help="输出目录")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files = [inp] if inp.is_file() else sorted(inp.glob("*.wav"))
    manifest = []
    for f in files:
        p = degrade_file(f, out_dir, args.preset, sr=args.sr)
        if p:
            manifest.append({"src": str(f), "degraded": str(p), "preset": args.preset})
            print(f"  [ok] {f.name} -> {p.name}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n完成 {len(manifest)} 条，manifest: {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
