# -*- coding: utf-8 -*-
"""
家庭声纹采集助手 —— 红队方言评测集 · 真实样本采集。

采集规范（每人约 15 分钟）:
    - 普通话朗读 5 分钟（自然语速，覆盖数字/金额/称谓——诈骗高频词）
    - 方言自由说 5 分钟
    - 三种信道: 安静手机近讲 / 手机免提外放 / 电话模式(如有条件)

用法:
    python record_family_voice.py --name 张三 --session putonghua --duration 300

产出:
    data/raw/family_voice/<name>/<session>_<timestamp>.wav (16kHz mono)
    以及自动生成的采集清单 manifest.csv

依赖:
    pip install sounddevice   （需要麦克风；也可直接用手机录音再导入）
"""
import argparse
import csv
import datetime
import sys
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("缺少 sounddevice / soundfile，请先: pip install sounddevice soundfile")
    sys.exit(1)

SR = 16000
TEXTS = {
    "putonghua": "请朗读：妈，是我。我在外面出了点事，急需五万块钱周转，先别告诉我爸。"
                 "银行卡号是 6222 开头。妈你听出来是我了吧？（——覆盖称谓/金额/紧迫话术的诈骗高频模式）",
    "fangyan": "请用方言自然聊天 5 分钟（讲讲今天做了什么即可，无需朗读）。",
    "readback": "请复述屏幕上的随机数字：58 302 1174 90（用于声纹确认测试）",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="采集对象姓名（拼音或化名）")
    ap.add_argument("--session", default="putonghua", choices=list(TEXTS.keys()))
    ap.add_argument("--duration", type=int, default=300, help="秒，默认 300")
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "family_voice"))
    args = ap.parse_args()

    dest = Path(args.dest) / args.name
    dest.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print(f"采集对象: {args.name} | 场景: {args.session} | 时长: {args.duration}s")
    print(f"提示语: {TEXTS[args.session]}")
    print("=" * 56)
    input("按回车开始录音（Ctrl+C 取消）...")

    print("录音中...")
    audio = sd.rec(int(args.duration * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dest / f"{args.session}_{ts}.wav"
    sf.write(out, audio, SR)

    manifest = Path(args.dest) / "manifest.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "session", "file", "duration_s", "date"])
        w.writerow([args.name, args.session, str(out), args.duration, ts])

    print(f"已保存: {out}")
    print(f"清单: {manifest}")
    print("提醒: 请确认采集对象已签署 docs/知情同意书模板.md 对应的知情同意书！")


if __name__ == "__main__":
    main()
