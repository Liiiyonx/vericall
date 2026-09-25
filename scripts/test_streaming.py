# -*- coding: utf-8 -*-
"""
流式回放回归测试（任务书 P1-1e）
================================
无麦克风也能回归：回放一段音频（或合成 60s 演示信号），跑完整流式时间线，
输出每窗三通道/融合/状态日志，并画一张时间线图（直接进参赛 PPT）。

用法：
  python scripts/test_streaming.py --replay 诈骗录音.wav
  python scripts/test_streaming.py            # 无参数 → 合成 60s（前30s正常/后30s诈骗）
  python scripts/test_streaming.py --plot out.png

依赖：numpy（画图需 matplotlib，缺失则仅输出日志）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion.fusion_orchestrator import ChannelVerdict
from server.scheduler import make_mock_backends
from server.stream_pipeline import StreamProcessor


def synth_scam_signal(sr: int = 16000, dur: float = 60.0) -> np.ndarray:
    """合成演示信号：前 30s 家人闲聊（低危），后 30s 诈骗（高 spoof + 话术）。"""
    t = np.arange(int(sr * dur)) / sr
    sig = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)
    # 后 30s 叠加高频异常，模拟合成语音的频谱痕迹
    seg = int(sr * 30)
    sig[seg:] += (0.2 * np.sin(2 * np.pi * 3000 * t[seg:])).astype(np.float32)
    return sig


def mock_backends():
    """确定性后端：后 30s 诈骗分数渐起（中危→硬命中），验证绿→黄→红。"""
    def score(t):
        if t < 30:
            return 0.05
        if t == 30:
            return 0.5
        if t in (31, 32):
            return 0.6
        return 0.9
    def sem_score(t):
        if t < 30:
            return 0.05
        if t == 30:
            return 0.5
        if t in (31, 32):
            return 0.6
        return 0.85
    def ac(w):
        t = round(w.t_rel)
        s = score(t)
        return ChannelVerdict("acoustic", s, "spoof" if s >= 0.5 else "bonafide",
                              "AASIST", 0.95)
    def vp(w):
        return ChannelVerdict("voiceprint", 0.1, "match", "声纹一致", 0.9)
    def sem(w, prev):
        t = round(w.t_rel)
        s = sem_score(t)
        if s >= 0.5:
            return ChannelVerdict("semantic", s, "money_request",
                                  "冒充熟人要钱", 0.9), "妈我是同学号急用五万块"
        return ChannelVerdict("semantic", 0.05, "normal", "正常闲聊", 0.9), ""
    return make_mock_backends(ac, vp, sem)


def real_backends(offline: bool):
    """线上后端：真实三通道（需模型/GPU，懒加载）。"""
    from fusion.pipeline import VeriCallPipeline
    from server.scheduler import make_real_backends
    pipe = VeriCallPipeline()
    return make_real_backends(pipe, offline=offline)


def run(samples: np.ndarray, backends, sr: int, plot: str | None):
    proc = StreamProcessor(backends=backends)
    events = proc.process_signal(samples)

    windows = [e for e in events if e["type"] == "window"]
    alerts = [e for e in events if e["type"] == "alert"]
    print(f"\n=== 流式时间线（{len(windows)} 窗，{len(alerts)} 次告警）===")
    for e in windows:
        c = e["channels"]
        print(f"  t={e['t_rel']:5.1f}s  fused={e['fused']:.2f}  "
              f"state={e['state']:6s}  "
              f"A={c['acoustic']['score']:.2f} V={c['voiceprint']['score']:.2f} "
              f"S={c['semantic']['score']:.2f} [{c['semantic']['label']}]")
    for a in alerts:
        print(f"  ⚠ ALERT @ {a.get('state')}: {a.get('advice')}")

    states = [e["state"] for e in windows]
    print(f"\n状态分布: GREEN={states.count('GREEN')} "
          f"YELLOW={states.count('YELLOW')} RED={states.count('RED')}")

    if plot:
        try:
            _draw(windows, plot)
            print(f"时间线图已保存: {plot}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 绘图跳过（matplotlib 缺失？）: {e}")
    return events


def _draw(windows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [e["t_rel"] for e in windows]
    fused = [e["fused"] for e in windows]
    cmap = {"GREEN": "#8fb996", "YELLOW": "#d4a04a", "RED": "#c96a5e"}
    colors = [cmap[e["state"]] for e in windows]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    # 状态色块
    for x, c in zip(xs, colors):
        ax.axvspan(x - 0.5, x + 0.5, color=c, alpha=0.35)
    # 融合分曲线
    ax.plot(xs, fused, color="#d4b483", lw=1.8, marker="o", ms=3, zorder=5)
    ax.axhline(0.5, color="#d4a04a", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0.7, color="#c96a5e", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylim(0, 1); ax.set_xlabel("时间 (s)"); ax.set_ylabel("融合可疑度")
    ax.set_title("谛听 VeriCall · 流式实时风险时间线")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="流式回放回归测试")
    ap.add_argument("--replay", help="回放音频文件（wav/flac/mp3）")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--plot", default="stream_timeline.png", help="时间线图输出路径")
    ap.add_argument("--real", action="store_true", help="使用真实三通道（需模型/GPU）")
    ap.add_argument("--offline", action="store_true", help="真实后端但语义走规则兜底")
    args = ap.parse_args()

    if args.real:
        backends = real_backends(offline=args.offline)
        if args.replay and Path(args.replay).is_file():
            from server.audio_util import read_wav
            sig = read_wav(args.replay, sr=args.sr)
        else:
            print("[warn] --real 需配合 --replay 真实音频"); sys.exit(1)
    else:
        if args.replay and Path(args.replay).is_file():
            from server.audio_util import read_wav
            sig = read_wav(args.replay, sr=args.sr)
            backends = mock_backends()
            print(f"回放: {args.replay}")
        else:
            print("未提供可用音频，使用合成 60s 演示信号（前30s正常 / 后30s诈骗）")
            sig = synth_scam_signal(args.sr)
            backends = mock_backends()

    run(sig, backends, args.sr, args.plot)


if __name__ == "__main__":
    main()
