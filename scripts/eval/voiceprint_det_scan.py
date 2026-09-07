#!/usr/bin/env python
"""voiceprint_det_scan.py — 声纹 DET 扫描 + 阈值科学标定（线 B，2026-09-07）

底库：AISHELL-3 train/wav（218 说话人 SSBxxxx，hf-mirror 全量物化 ~10GB）。
方法：
  1. 每说话人抽 K 条话语 → CAMPPlus（funasr, 与 VoiceprintChannel 同款）提 192 维 L2 归一化向量；
  2. 全体两两余弦：同人(genuine)对 / 异人(impostor)对（异人抽样上限 M 防爆）;
  3. DET 曲线 + EER；按「误拒家人代价 >> 误受陌生人」做 κ 敏感性分析：
       κ = C_miss/C_fa ∈ {1, 3, 1/3} 三表并列，给出工作阈值建议；
  4. 产出 evaluation/voiceprint_calibration.json/.md，阈值替换参考（0.55/0.35 拍脑袋值出处）。

用法（GPU 环境，python -u）：
  python -u scripts/eval/voiceprint_det_scan.py --wav D:/VeriCall_data/aishell3/repo/train/wav \
      --limit-speakers 0 --k 12 --max-imp 60000 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SV_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"


def list_speakers(wav_root: Path, limit: int = 0) -> list[Path]:
    spk = sorted(p for p in wav_root.iterdir() if p.is_dir())
    return spk if not limit else spk[:limit]def pick_utterances(spk_dir: Path, k: int, rng: random.Random) -> list[Path]:
    wavs = sorted(spk_dir.glob("*.wav"))
    if not wavs:
        return []
    # 均匀取 k 条（覆盖全时长分布）
    idx = sorted(rng.sample(range(len(wavs)), min(k, len(wavs))))
    return [wavs[i] for i in idx]


def embed_all(paths: list[tuple[str, Path]]):
    """paths: [(uid, wav)] -> {uid: vec}；复用 funasr CAMPPlus。"""
    from funasr import AutoModel
    model = AutoModel(model=SV_MODEL_ID, device="cuda:0" if _cuda() else "cpu",
                      disable_update=True)
    vecs, t0 = {}, time.time()
    for i, (uid, p) in enumerate(paths):
        res = model.generate(input=str(p), batch_size_s=60)
        raw = None
        for key in ("spk_embedding", "embedding", "spk_feature"):
            if res and isinstance(res[0], dict) and res[0].get(key) is not None:
                raw = res[0][key]
                break
        if raw is None:
            print(f"  !! {uid} 无向量，跳过", flush=True)
            continue
        if hasattr(raw, "cpu"):
            raw = raw.detach().cpu().numpy()
        v = np.asarray(raw, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        if n < 1e-8:
            continue
        vecs[uid] = v / n
        if (i + 1) % 50 == 0:
            print(f"  嵌入 {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    return vecs


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="D:/VeriCall_data/aishell1_sub/train,"
                    "D:/VeriCall_data/aishell3/data/train/wav",
                    help="底库根（逗号分隔多源：AISHELL-1 + AISHELL-3）")
    ap.add_argument("--limit-speakers", type=int, default=0, help=">0 每根仅前 N 说话人(冒烟)")
    ap.add_argument("--k", type=int, default=12, help="每说话人抽取话语数")
    ap.add_argument("--max-imp", type=int, default=60000, help="异人对抽样上限")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    # 多底库根（逗号分隔）：AISHELL-1 + AISHELL-3 合库
    # 支持多底库根（逗号分隔）：AISHELL-1 + AISHELL-3 合库
    roots = [Path(x.strip()) for x in args.wav.split(",") if x.strip()]
    rng = random.Random(args.seed)
    uid2spk, paths = {}, []
    spk_dirs_all = []
    for wav_root in roots:
        if not wav_root.is_dir():
            print(f"[跳过] 底库根不存在: {wav_root}")
            continue
        spk_dirs = list_speakers(wav_root, args.limit_speakers)
        spk_dirs_all += spk_dirs
        tag = wav_root.name
        for sp in spk_dirs:
            for p in pick_utterances(sp, args.k, rng):
                uid = f"{tag}:{sp.name}/{p.stem}"
                uid2spk[uid] = f"{tag}:{sp.name}"
                paths.append((uid, p))
    spk_dirs = spk_dirs_all
    print(f"[底库] 根 {len(roots)} 个，说话人 {len(spk_dirs)} @ "
          f"{[str(r) for r in roots]}")
    print(f"[样本] 共 {len(paths)} 条话语（≈{len(paths)/max(1,len(spk_dirs)):.0f} 条/人）")

    print("[嵌入] CAMPPlus ...", flush=True)
    t0 = time.time()
    vecs = embed_all(paths)
    print(f"[嵌入] 完成 {len(vecs)}/{len(paths)} ({time.time()-t0:.0f}s)")

    ids = list(vecs)
    spk_of = {i: uid2spk[i] for i in ids}
    M = np.stack([vecs[i] for i in ids])          # N x 192 (行已 L2)
    sim = M @ M.T                                   # N x N 余弦
    N = len(ids)

    # 同人对（genuine）：同一说话人内部不同话语
    gen_scores = []
    by_spk = defaultdict(list)
    for i, uid in enumerate(ids):
        by_spk[spk_of[uid]].append(i)
    for spk, idxs in by_spk.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                gen_scores.append(float(sim[idxs[a], idxs[b]]))
    gen_scores = np.array(gen_scores)

    # 异人对（impostor）：随机抽样上限 max_imp
    rng2 = random.Random(args.seed + 1)
    imp_scores = []
    tried = 0
    while len(imp_scores) < args.max_imp and tried < args.max_imp * 20:
        i = rng2.randrange(N)
        j = rng2.randrange(N)
        tried += 1
        if spk_of[ids[i]] == spk_of[ids[j]]:
            continue
        imp_scores.append(float(sim[i, j]))
    imp_scores = np.array(imp_scores)
    print(f"[得分] genuine {len(gen_scores)} / impostor {len(imp_scores)}")

    # DET / EER：阈值向下扫（分数>t 判同人）
    lo = min(float(gen_scores.min()), float(imp_scores.min())) - 0.01
    hi = max(float(gen_scores.max()), float(imp_scores.max())) + 0.01
    grid = np.linspace(lo, hi, 2001)
    far = np.array([(imp_scores > t).mean() for t in grid])
    frr = np.array([(gen_scores <= t).mean() for t in grid])
    d = np.abs(far - frr)
    i_eer = int(np.argmin(d))
    eer_t, eer = float(grid[i_eer]), float((far[i_eer] + frr[i_eer]) / 2)
    print(f"[EER] {eer:.4f} @ 阈值 {eer_t:.3f}")

    # κ 敏感性：κ = 误拒代价/误受代价 ∈ {1, 1/3, 3}
    # 代价 = κ·FRR + FAR（在 genuine/impostor 等先验下）→ 找最小代价阈值
    kappas = {"1:1": 1.0, "误拒3x贵(3:1)": 3.0, "误受3x贵(1:3)": 1 / 3}
    work = {}
    for label, kap in kappas.items():
        cost = kap * frr + far
        i_best = int(np.argmin(cost))
        work[label] = {"threshold": float(grid[i_best]),
                       "FAR": float(far[i_best]), "FRR": float(frr[i_best]),
                       "cost": float(cost[i_best])}
        print(f"  [{label}] 工作阈值 {grid[i_best]:.3f}  FAR {far[i_best]:.4f} FRR {frr[i_best]:.4f}")

    out_json = Path(__file__).resolve().parents[2] / "evaluation" / "voiceprint_calibration.json"
    payload = {
        "date": "2026-09-07", "model": SV_MODEL_ID,
        "speakers": len(spk_dirs), "utterances": len(vecs), "k": args.k,
        "genuine_pairs": len(gen_scores), "impostor_pairs": len(imp_scores),
        "eer": eer, "eer_threshold": eer_t,
        "working_points": work,
        "note": "阈值用于 VoiceprintChannel 0.55/0.35 拍脑袋值出处；det 数组未存(json大), 重跑可得",
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[产物] {out_json}")
    # 快速 md
    lines = ["# 声纹阈值科学标定（线 B2/B3）", "",
             f"- 底库：AISHELL-3 {len(spk_dirs)} 人 × ~{args.k} 条 = {len(vecs)} 条（CAMPPlus 192维）",
             f"- 同人对 {len(gen_scores)} / 异人对 {len(imp_scores)}",
             f"- **EER {eer*100:.2f}%** @ 阈值 {eer_t:.3f}", "", "| 工作点(κ) | 阈值 | FAR | FRR |", "|---|---|---|---|"]
    for label, wp in work.items():
        lines.append(f"| {label} | {wp['threshold']:.3f} | {wp['FAR']*100:.2f}% | {wp['FRR']*100:.2f}% |")
    (out_json.with_suffix(".md")).write_text("\n".join(lines), encoding="utf-8")
    print(f"[产物] {out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
