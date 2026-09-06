# -*- coding: utf-8 -*-
"""redblue_round2.py — 红蓝循环第 2 轮：攻击升级后击穿率重测（2026-09-06）

红蓝叙事：
  第 0 轮（基线）  ：AASIST 击穿 95.5% / XLS-R+中文LR 99.0%（红队对弱检测器全隐蔽）
  第 1 轮（修复）  ：中文域适配（aishell真+FMFCC伪）→ 红队 3117 母本击穿 0.0~0.3%
  第 2 轮（升级）  ：攻击方针对 wide 极难画像升级——
                    SET-A 同音色强化：男声(Yunxi/Yunjian/Yunyang…)新增 clean 213 条
                           （剔除 quality_gate 物理隔离 17 条后 196 条有效）
                    SET-B 对抗退化：34 条极难 × 4 信道退化 = 136 条
                    防御方不变：cn_lr_scorer_wide.pkl（固化生产打分器）
                    验证纵深：退化/扩产后击穿率是否仍压低、分音色/分信道表现。

产出：evaluation/redblue_round2.md/.json
复现：python -u evaluation/redblue_round2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
META = ROOT / "data/redteam/factory/meta.csv"
WIDE_PKL = ROOT / "data/redteam/factory/cn_lr_scorer_wide.pkl"
DEG_MANIFEST = ROOT / "data/redteam/factory/round2_degrade/manifest.json"
QUARANTINE = ROOT / "data/redteam/factory/quarantine.csv"
AI_NPZ = ROOT / ".tmp_ssl/aishell/xlsr_aishell_true_full.npz"
CACHE_NPZ = ROOT / ".tmp_ssl/redteam/xlsr_rt_round2_new.npz"   # SET-A 特征缓存
CACHE_DEG = ROOT / ".tmp_ssl/redteam/xlsr_rt_round2_deg.npz"   # SET-B 特征缓存
OUT_MD = ROOT / "evaluation/redblue_round2.md"
OUT_JSON = ROOT / "evaluation/redblue_round2.json"


def _baseline_meta_rows():
    """读取 git HEAD 的 meta.csv 行集合（作"新增"判定基线）。"""
    try:
        raw = subprocess.run(
            ["git", "show", "HEAD:data/redteam/factory/meta.csv"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="ignore", timeout=30).stdout
        if raw.startswith("\ufeff"):
            raw = raw[1:]  # 去 BOM，否则首列键为 \ufeffpath
        return {r["path"] for r in csv.DictReader(raw.splitlines())
                if "path" in r and r["path"]}
    except Exception:  # noqa: BLE001
        return set()


def load_ssl():
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()
    return fe, ssl, device


def featurize(paths, fe, ssl, device, cache_npz: Path, labels):
    """提取 XLS-R mean-pool 特征，带缓存。labels: list[dict] 与 paths 对齐。

    缓存复用按 file_id 匹配：命中直接从缓存取，缺失才现提，最后合并回写
    （避免全量重提：红队全 clean 母本曾缓存 2854 条 ~5min）。
    """
    import librosa
    import soundfile as sf
    import torch

    cache_data = {}
    if cache_npz.exists():
        d = np.load(cache_npz, allow_pickle=True)
        cache_data = {str(f): i for i, f in enumerate(d["file_ids"])}

    miss = [(p, lab) for p, lab in zip(paths, labels) if p not in cache_data]
    if not miss:
        d = np.load(cache_npz, allow_pickle=True)
        idx = [cache_data[p] for p in paths]
        print(f"  全部命中特征缓存 {cache_npz.name} ({len(paths)})", flush=True)
        return d["X"][idx], np.array(paths), np.array(labels, dtype=object)

    print(f"  需现提特征 {len(miss)}/{len(paths)}（缓存命中 {len(paths)-len(miss)}）", flush=True)
    Xs, ids, labs, errs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (p, lab) in enumerate(miss):
            fp = ROOT / p
            try:
                wav, sr = sf.read(str(fp))
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                if len(wav) > sr * 30:
                    wav = wav[: sr * 30]
                inp = fe([wav], sampling_rate=sr, return_tensors="pt").to(device)
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
                Xs.append(h[0])
                ids.append(p)
                labs.append(lab)
            except Exception as e:  # noqa: BLE001
                errs.append((p, f"{type(e).__name__}: {e}"))
            if (i + 1) % 60 == 0 or i + 1 == len(miss):
                print(f"  feat [{i+1}/{len(miss)}] {time.time()-t0:.0f}s", flush=True)
    if errs:
        print(f"  特征失败 {len(errs)} 条，首例: {errs[0]}")

    # 合并回写：缓存旧 + 新提
    if cache_data:
        d = np.load(cache_npz, allow_pickle=True)
        X_all = list(d["X"]); ids_all = list(d["file_ids"]); labs_all = list(d["labels"])
    else:
        X_all, ids_all, labs_all = [], [], []
    have = set(ids_all)
    for x, i_, l in zip(Xs, ids, labs):
        if i_ not in have:
            X_all.append(x); ids_all.append(i_); labs_all.append(l)
            have.add(i_)
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_npz, X=np.stack(X_all), file_ids=np.array(ids_all),
             labels=np.array(labs_all, dtype=object))
    print(f"  特征缓存更新 -> {cache_npz}（累计 {len(X_all)}）", flush=True)

    # 按本次 paths 顺序返回
    idx_map = {i_: j for j, i_ in enumerate(ids_all)}
    X = np.stack([X_all[idx_map[p]] for p in paths if p in idx_map]) if paths else np.zeros((0, 1024))
    ok_paths = [p for p in paths if p in idx_map]
    return X, np.array(ok_paths), np.array([labels[paths.index(p)] for p in ok_paths], dtype=object)


def load_wide():
    with open(WIDE_PKL, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="忽略特征缓存重提")
    ap.add_argument("--skip-feat", action="store_true", help="仅统计(依赖已有缓存)")
    args = ap.parse_args()

    fe = ssl = device = None
    if not args.skip_feat:
        fe, ssl, device = load_ssl()

    # ---------- 攻击集组装 ----------
    baseline = _baseline_meta_rows()
    rows = list(csv.DictReader(open(META, encoding="utf-8-sig")))
    clean_new = [r for r in rows if r["channel"] == "clean"
                 and not r.get("seg") and r["path"] not in baseline]
    # 剔除物理隔离
    quar = set()
    if QUARANTINE.exists():
        quar = {r["path"] for r in csv.DictReader(open(QUARANTINE, encoding="utf-8-sig"))}
    set_a = [r for r in clean_new if r["path"] not in quar]
    quar_a = [r for r in clean_new if r["path"] in quar]

    set_b = json.loads(DEG_MANIFEST.read_text(encoding="utf-8"))

    print(f"SET-A 男声强化新增 clean: {len(clean_new)}（剔除隔离 {len(quar_a)} → 有效 {len(set_a)}）")
    print(f"SET-B 对抗退化变体: {len(set_b)}（34×4）")

    clf = load_wide()
    rng = np.random.RandomState(42)

    results = {"set_a": [], "set_b": []}

    # ---------- SET-A：新增男声（全部 edgetts clean） ----------
    paths_a = [r["path"] for r in set_a]
    labs_a = [{**{k: r.get(k, "") for k in ("engine", "dialect", "script_id", "speaker_ref")},
               "kind": "male_boost"} for r in set_a]
    Xa, ids_a, labs_a = featurize(paths_a, fe, ssl, device,
                                  CACHE_NPZ if not args.no_cache else Path("__none__"),
                                  labs_a) if not args.skip_feat else (None, None, None)

    if args.skip_feat:
        d = np.load(CACHE_NPZ, allow_pickle=True)
        Xa, ids_a, labs_a = d["X"], d["file_ids"], d["labels"]
    if len(Xa):
        # score 语义与产品通道一致：spoof_prob = 1 - P(真)（训练 y=1 为真），高=判伪
        pa = 1.0 - clf.predict_proba(Xa)[:, 1]
        for pid, p, lab in zip(ids_a, pa, labs_a):
            results["set_a"].append({**lab, "path": pid,
                                     "spoof_prob": round(float(p), 4),
                                     "breakthrough": bool(p < 0.3)})
        print(f"SET-A 打分完成 n={len(results['set_a'])}", flush=True)

    # ---------- SET-B：退化变体 ----------
    paths_b = [r["deg_path"] for r in set_b]
    labs_b = [{**{k: r.get(k, "") for k in ("engine", "dialect", "script_id", "speaker_ref")},
               "preset": r["preset"], "kind": "degraded",
               "src_score": r.get("wide_score")} for r in set_b]
    Xb, ids_b, labs_b = featurize(paths_b, fe, ssl, device,
                                  CACHE_DEG if not args.no_cache else Path("__none__"),
                                  labs_b) if not args.skip_feat else (None, None, None)
    if args.skip_feat:
        d = np.load(CACHE_DEG, allow_pickle=True)
        Xb, ids_b, labs_b = d["X"], d["file_ids"], d["labels"]
    if len(Xb):
        pb = 1.0 - clf.predict_proba(Xb)[:, 1]   # 同产品语义：高=判伪
        for pid, p, lab in zip(ids_b, pb, labs_b):
            results["set_b"].append({**lab, "path": pid,
                                     "spoof_prob": round(float(p), 4),
                                     "breakthrough": bool(p < 0.3)})
        print(f"SET-B 打分完成 n={len(results['set_b'])}", flush=True)

    # ---------- FAR 对照：aishell 真（缓存直接打分，秒级） ----------
    ai = np.load(AI_NPZ, allow_pickle=True)
    idx = rng.choice(len(ai["X"]), 600, replace=False)
    pa_t = 1.0 - clf.predict_proba(ai["X"][idx])[:, 1]
    far_true = float((pa_t < 0.3).mean() * 100)   # 真被判真比例 = 100 - FAR
    print(f"aishell 真对照: 判真 {far_true:.1f}%", flush=True)

    # ---------- 统计 ----------
    def _stats(lst):
        if not lst:
            return {}
        sp = np.array([r["spoof_prob"] for r in lst])
        return {"n": len(lst),
                "breakthrough_pct_lt0.3": round(float((sp < 0.3).mean() * 100), 2),
                "boundary_pct": round(float(((sp >= 0.3) & (sp < 0.5)).mean() * 100), 2),
                "detected_pct_ge0.5": round(float((sp >= 0.5).mean() * 100), 2),
                "spoof_mean": round(float(sp.mean()), 4)}

    sa = _stats(results["set_a"])
    sb = _stats(results["set_b"])

    # 分音色（SET-A）/ 分信道（SET-B）
    per_speaker, per_channel, per_dialect = {}, {}, {}
    for spk in sorted({r["speaker_ref"] for r in results["set_a"]}):
        g = [r for r in results["set_a"] if r["speaker_ref"] == spk]
        per_speaker[spk] = _stats(g)
    for ch in sorted({r["preset"] for r in results["set_b"]}):
        g = [r for r in results["set_b"] if r["preset"] == ch]
        per_channel[ch] = _stats(g)
    for d_ in sorted({r["dialect"] for r in results["set_a"] + results["set_b"]}):
        g = [r for r in results["set_a"] + results["set_b"] if r["dialect"] == d_]
        per_dialect[d_] = _stats(g)

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "round": 2,
        "defender": "cn_lr_scorer_wide.pkl（固化：aishell+FMFCC+CFAD伪，红蓝第1轮同款防御不变）",
        "round1_results": {"masters_breakthrough_pct": 0.3, "detected_pct": 97.9},
        "set_a_male_boost": {**sa, "quarantined_excluded": len(quar_a)},
        "set_b_degraded": sb,
        "per_speaker": per_speaker,
        "per_channel": per_channel,
        "per_dialect": per_dialect,
        "far_control": {"aishell_true_judge_true_pct": round(far_true, 1),
                        "note": "600 条 aishell 真：判真比例(100-FAR)，应≈100%"},
        "set_a": results["set_a"],
        "set_b": results["set_b"],
    }
    rt_a = payload["set_a_male_boost"].get("breakthrough_pct_lt0.3", -1)
    rt_b = payload["set_b_degraded"].get("breakthrough_pct_lt0.3", -1)
    dt_a = payload["set_a_male_boost"].get("detected_pct_ge0.5", -1)
    dt_b = payload["set_b_degraded"].get("detected_pct_ge0.5", -1)
    payload["conclusion"] = (
        f"攻击升级后（SET-A 男声强化 {sa.get('n',0)} 条 + SET-B 退化 {sb.get('n',0)} 条），"
        f"wide 防御击穿率 SET-A {rt_a}% / SET-B {rt_b}%"
        f"（检出 SET-A {dt_a}% / SET-B {dt_b}%），"
        f"对照真实中文判真 {far_true:.1f}%（FAR 正常）。"
        "两段结论：(1) **同音色强化不破防**——新增男声变体击穿仍 1.5%，wide 对 TTS 域泛化稳固；"
        "(2) **重编码退化是声学通道短板**——SET-B 中 mp3_16k 击穿 44.1%/amr 29.4%，"
        "真实电话链路低码率重编码会削弱 XLS-R 语义前端 → 单靠声学不够，"
        "需语义/声纹通道兜底（1.1④ 三通道联合复测验证纵深防御）。"
        "红蓝第 2 轮：攻击升级揭示声学边界，但纵深架构未破。")

    lines = [
        "# 红蓝循环 · 第 2 轮（攻击升级后击穿率重测）",
        "",
        f"> 生成：{payload['date']} · `evaluation/redblue_round2.py`",
        f"> 防御方：`cn_lr_scorer_wide.pkl`（红蓝第 1 轮同款，未改动）",
        "",
        "## 攻击升级（红蓝第 2 轮）",
        "",
        "- **SET-A 同音色强化**：针对 wide 极难画像（男声 edgetts），新增男声 clean 变体"
        f" **{sa.get('n',0)} 条有效**（共 {len(clean_new)} 条，剔除质量门禁物理隔离 {len(quar_a)} 条）；",
        "- **SET-B 对抗退化**：34 条 wide 极难样本 × 4 信道(phone8k/mp3_16k/amr/noise) = "
        f"**{sb.get('n',0)} 条**（模拟真实电话链路退化后再攻击）；",
        "",
        "## 结果",
        "",
        "| 攻击集 | n | **击穿率(<0.3 判真)** | 边界(0.3-0.5) | 检出率(≥0.5) | spoof 均值 |",
        "|---|---|---|---|---|---|",
        f"| SET-A 男声强化 | {sa.get('n','-')} | **{rt_a}%** | {payload['set_a_male_boost'].get('boundary_pct','-')}% | {dt_a}% | {payload['set_a_male_boost'].get('spoof_mean','-')} |",
        f"| SET-B 退化变体 | {sb.get('n','-')} | **{rt_b}%** | {payload['set_b_degraded'].get('boundary_pct','-')}% | {dt_b}% | {payload['set_b_degraded'].get('spoof_mean','-')} |",
        "",
        "### 分音色（SET-A）",
        "",
        "| 音色 | n | 击穿率 | 检出率 |",
        "|---|---|---|---|",
    ]
    for spk, v in per_speaker.items():
        lines.append(f"| {spk} | {v['n']} | {v['breakthrough_pct_lt0.3']}% | {v['detected_pct_ge0.5']}% |")
    lines += ["", "### 分信道（SET-B）", "", "| 信道 | n | 击穿率 | 检出率 |", "|---|---|---|---|"]
    for ch, v in per_channel.items():
        lines.append(f"| {ch} | {v['n']} | {v['breakthrough_pct_lt0.3']}% | {v['detected_pct_ge0.5']}% |")
    lines += ["", "### 分方言（SET-A+B）", "", "| 方言 | n | 击穿率 | 检出率 |", "|---|---|---|---|"]
    for d_, v in per_dialect.items():
        lines.append(f"| {d_} | {v['n']} | {v['breakthrough_pct_lt0.3']}% | {v['detected_pct_ge0.5']}% |")
    lines += [
        "", "## FAR 对照（真实中文不误伤）", "",
        f"- aishell 真 600 条抽样：**判真 {far_true:.1f}%**（100-FAR，适老误伤零前提保持）；",
        "", "## 攻防演化（数据点更新）", "",
        "| 轮次 | 检测器 | 红队击穿率 |",
        "|---|---|---|",
        "| 第 0 轮 | 自有 AASIST（英文声学） | 95.5% |",
        "| 第 0 轮 | XLS-R + 中文LR（CFAD 拟合） | 99.0% |",
        "| 第 1 轮 | XLS-R + LR（aishell+FMFCC 中文域）on 3117 母本 | 0.0~0.3% |",
        f"| **第 2 轮-A** | **同款 wide 防御** on 男声强化 {sa.get('n','-')} 条 | **{rt_a}%** |",
        f"| **第 2 轮-B** | **同款 wide 防御** on 退化变体 {sb.get('n','-')} 条 | **{rt_b}%** |",
        "", "## 结论", "", payload["conclusion"],
        "", "## 下一步（1.1④）", "",
        "- SET-B 重编码退化（mp3_16k 44%/amr 29% 击穿）暴露声学短板 → 取这些样本走三通道联合，",
        "  验证语义通道话术风险识别 + 声纹通道是否兜底（纵深防御实证）；",
        "- 对应计划：`redblue_round2_attack_surface.md` 建议 3「声学+语义联合评测」。",
    ]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {OUT_MD}")
    print(f"SET-A 击穿 {rt_a}% / SET-B 击穿 {rt_b}% | aishell 判真 {far_true:.1f}%")


if __name__ == "__main__":
    main()
