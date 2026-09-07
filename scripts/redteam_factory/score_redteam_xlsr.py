#!/usr/bin/env python
"""score_redteam_xlsr.py — 红队全量难度打分（XLS-R + 中文域 LR，09-06）

背景：AASIST(acoustic) 对中文红队全漏（score~0 无区分），英文域 LR 有偏置。
本工具用 CFAD 拟合的中文域 LR（AUC 0.950，data/redteam/factory/cn_lr_scorer.pkl）
给红队全部 clean 母本打 spoof 概率：值越低 = 检测器越判"真" = 红队越隐蔽（越难检）。

设计（避免评测泄漏）：CFAD 只用于训练红队打标器；红队非任何评测集，打分结果仅用于
增量训练选样与质量报告，不参与 CFAD/FMFCC 评测。

产出：data/redteam/factory/redteam_difficulty.csv（全 clean 母本，含难度分/排序）
      data/redteam/factory/redteam_difficulty_summary.md
用法：python -u scripts/redteam_factory/score_redteam_xlsr.py [--limit N] [--resume]
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
SCORER_PKL = ROOT / "data/redteam/factory/cn_lr_scorer.pkl"
META = ROOT / "data/redteam/factory/meta.csv"
OUT_CSV = ROOT / "data/redteam/factory/redteam_difficulty.csv"
OUT_MD = ROOT / "data/redteam/factory/redteam_difficulty_summary.md"
SEG_CSV = ROOT / "data/redteam/factory/redteam_seg_difficulty.csv"
SEG_MD = ROOT / "data/redteam/factory/redteam_seg_difficulty_summary.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help=">0 仅前 N 条（冒烟）")
    ap.add_argument("--resume", action="store_true", help="断点续跑（跳过已有结果）")
    ap.add_argument("--seg-only", action="store_true", help="只对切段(seg=True)打分，输出到独立文件")
    ap.add_argument("--scorer", choices=["cfad", "full", "wide"], default="cfad", help="打分器：cfad=CFAD拟合(默认) / full=aishell+FMFCC中文域正式版 / wide=生产口径(第2轮同款)")
    args = ap.parse_args()

    global OUT_CSV, OUT_MD, SEG_CSV, SEG_MD
    if args.scorer in ("full", "wide"):
        tag = "_full" if args.scorer == "full" else "_wide"
        OUT_CSV = OUT_CSV.with_name(f"redteam_difficulty{tag}.csv")
        OUT_MD = OUT_MD.with_name(f"redteam_difficulty{tag}_summary.md")
        SEG_CSV = SEG_CSV.with_name(f"redteam_seg_difficulty{tag}.csv")
        SEG_MD = SEG_MD.with_name(f"redteam_seg_difficulty{tag}_summary.md")


    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"XLS-R 加载 {SSL_LOCAL} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()

    _pkl = SCORER_PKL if args.scorer == "cfad" else (SCORER_PKL.with_name("cn_lr_scorer_full.pkl") if args.scorer == "full" else SCORER_PKL.with_name("cn_lr_scorer_wide.pkl"))
    with open(_pkl, "rb") as f:
        clf = pickle.load(f)
    print(f"中文域 LR 打分器加载 OK", flush=True)

    rows = list(csv.DictReader(open(META, encoding="utf-8-sig")))
    clean = [r for r in rows if r["channel"] == "clean" and not r.get("seg")]
    if args.seg_only:
        clean = [r for r in rows if r.get("seg") == "True"]
    if args.limit:
        clean = clean[: args.limit]
    print(f"待打分 clean 母本: {len(clean)}", flush=True)

    # 断点：读已有结果集
    done = {}
    if args.resume and OUT_CSV.exists():
        for r in csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")):
            done[r["path"]] = True
        todo = [r for r in clean if r["path"] not in done]
        print(f"续跑：跳过 {len(done)}，剩 {len(todo)}", flush=True)
    else:
        todo = clean

    out_rows = []
    if args.resume and OUT_CSV.exists():
        # 保持原表头顺序：先读回
        pass

    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(todo):
            p = ROOT / r["path"]
            try:
                wav, sr = sf.read(str(p))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:      # 超长截 30s（质量门禁已隔离 >20s，安全）
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                spoof_prob = float((1.0 - clf.predict_proba(h)[:, 1])[0])
                out_rows.append({
                    "path": r["path"], "engine": r["engine"], "dialect": r["dialect"],
                    "script_id": r["script_id"], "speaker_ref": r["speaker_ref"],
                    "difficulty_spoof_prob": round(spoof_prob, 4),
                    "verdict": "隐蔽(难检)" if spoof_prob < 0.3 else ("边界" if spoof_prob < 0.5 else "可检"),
                })
            except Exception as e:  # noqa: BLE001
                out_rows.append({"path": r["path"], "engine": r.get("engine", ""),
                                 "dialect": r.get("dialect", ""), "script_id": r.get("script_id", ""),
                                 "speaker_ref": r.get("speaker_ref", ""),
                                 "difficulty_spoof_prob": "ERR", "verdict": f"{type(e).__name__}"})
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                dt = time.time() - t0
                print(f"  [{i+1}/{len(todo)}] {dt:.0f}s ({dt/max(1,i+1):.2f}s/条)", flush=True)

    # 合并已有 + 新结果（resume 时）
    all_rows = []
    if args.resume and OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8-sig", newline="") as f:
            all_rows += list(csv.DictReader(f))
    all_rows += out_rows
    # 排序：spoof 概率升序 = 最隐蔽在前
    def keyf(r):
        try:
            return float(r["difficulty_spoof_prob"])
        except Exception:
            return 1.0
    all_rows.sort(key=keyf)

    _csv_out, _md_out = (SEG_CSV, SEG_MD) if args.seg_only else (OUT_CSV, OUT_MD)
    with open(_csv_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "engine", "dialect", "script_id",
                                          "speaker_ref", "difficulty_spoof_prob", "verdict"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"难度分表 -> {_csv_out}（{len(all_rows)} 条）", flush=True)

    # 汇总
    from collections import Counter
    n_ok = [r for r in all_rows if r["difficulty_spoof_prob"] != "ERR"]
    eng = Counter(r["engine"] for r in n_ok)
    dial = Counter(r["dialect"] for r in n_ok)
    verd = Counter(r["verdict"] for r in n_ok)
    probs = [float(r["difficulty_spoof_prob"]) for r in n_ok]
    lines = [
        "# 红队音频难度分汇总（XLS-R + 中文域 LR）",
        "",
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M')} · `score_redteam_xlsr.py`",
        f"> 打分器：CFAD 拟合中文域 LR（AUC 0.950）· 共 {len(n_ok)} 条 clean 母本",
        "",
        "## 判定分布（spoof 概率越低 = 检测器越判真 = 越隐蔽难检）",
        f"- **隐蔽(难检) <0.3**：{verd.get('隐蔽(难检)', 0)} 条",
        f"- **边界 0.3~0.5**：{verd.get('边界', 0)} 条",
        f"- **可检 >=0.5**：{verd.get('可检', 0)} 条",
        f"- 全距：{min(probs):.3f} ~ {max(probs):.3f}（均值 {np.mean(probs):.3f}）",
        "",
        "| 引擎 | 条数 | 难度分均值 |", "|---|---|---|",
    ]
    for e, n in eng.most_common():
        ps = [float(r["difficulty_spoof_prob"]) for r in n_ok if r["engine"] == e]
        lines.append(f"| {e} | {n} | {np.mean(ps):.3f} |")
    lines += ["", "| 方言 | 条数 |", "|---|---|"]
    for d, n in dial.most_common():
        lines.append(f"| {d} | {n} |")
    lines += ["", "## 用途", "- 增量训练选样：隐蔽(难检) 段优先入训练集（spoof 角色）；",
              "- 质量报告：红队难度分布佐证数据价值；",
              "- 阈值 0.3/0.5 可按模型更新后重标定。"]
    Path(_md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总 -> {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
