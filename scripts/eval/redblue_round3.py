#!/usr/bin/env python
"""redblue_round3.py — 红蓝第 3 轮（语料扩产后复测，2026-09-07）

defender：cn_lr_scorer_wide（与第 1/2 轮同款不变）。
SET-C：wide 重扫后极难母本（<0.3，round3_setC_extreme.json，由 score_redteam_xlsr --scorer wide 产出）。
SET-D：SET-C × {phone8k, mp3_16k, amr, noise} 退化（degrade_audio.py 产物）。
输出 evaluation/round3_scores.json + 击穿统计。
用法：python -u scripts/eval/redblue_round3.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "redteam_factory"))
import torch  # noqa: E402
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
from transformers import AutoFeatureExtractor, AutoModel  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "srx", ROOT / "scripts/redteam_factory/score_redteam_xlsr.py")
srx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srx)  # type: ignore[union-attr]
SSL = srx.SSL_LOCAL


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL)
    ssl = AutoModel.from_pretrained(SSL).to(device).eval()
    clf = pickle.load(open(ROOT / "data/redteam/factory/cn_lr_scorer_wide.pkl", "rb"))

    def score(path: str) -> float:
        wav, sr = sf.read(str(path))
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        if len(wav) > sr * 30:
            wav = wav[: sr * 30]
        with torch.no_grad():
            inp = fe([wav], sampling_rate=16000, return_tensors="pt").to(device)
            h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
        return float((1.0 - clf.predict_proba(h)[:, 1])[0])

    setc = json.load(open(ROOT / "evaluation/round3_setC_extreme.json", encoding="utf-8"))
    out = {"set_c": [], "set_d": {}}
    for p in setc:
        out["set_c"].append({"path": p, "prob": round(score(str(ROOT / p)), 4)})
    for ch in ["phone8k", "mp3_16k", "amr", "noise"]:
        out["set_d"][ch] = []
        for p in sorted(glob.glob(str(ROOT / f"data/redteam/factory/round3_setd/{ch}/*.wav"))):
            out["set_d"][ch].append({"path": p, "prob": round(score(p), 4)})
    out_f = ROOT / "evaluation" / "round3_scores.json"
    out_f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    def stat(items):
        n = len(items)
        bt = sum(1 for x in items if x["prob"] < 0.3)
        return n, bt, round(bt / n * 100, 2) if n else 0

    cn, cb, cp = stat(out["set_c"])
    print(f"[SET-C clean] n={cn} 击穿<0.3 {cb} 击穿率 {cp}%")
    for ch, items in out["set_d"].items():
        n, bt, p = stat(items)
        print(f"[SET-D {ch}] n={n} 击穿 {bt} ({p}%)")
    print(f"产物 {out_f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
