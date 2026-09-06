# -*- coding: utf-8 -*-
"""eval_fusion_crossdomain.py — 融合模型的中文域零样本复验（极致化计划书 §一验收项）

背景：
  fusion_stack.md 证明 AASIST+XLS-R stacking 在英文 dev 上 EER 0.354%（-39.8%）。
  本实验把**在 ASVspoof19LA(英文) 上拟合的融合器原样**应用到中文集，零样本跨域：
    - CFAD（2000 条，1000 真 / 1000 伪，含 STRAIGHT/PARTIALLYFAKE 等多攻击）
  这是计划书 §一「中文域 EER<5%」验收口径的直接证据。

协议要点：
  - 融合器（SSL-LR 与 stack-LR）**不在中文集上重新拟合**，避免泄漏，成绩即零样本泛化；
  - XLS-R 特征与 AASIST 分数在 CFAD 上重新提取（缓存到 .tmp_ssl/crossdomain/）。

输出：evaluation/fusion_crossdomain.md / .json

用法：
  python scripts/eval_fusion_crossdomain.py              # CFAD 全量 2000 条
  python scripts/eval_fusion_crossdomain.py --limit 400  # 冒烟
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "external" / "aasist"))

import eval_fusion_stack as fs  # 复用：aasist_scores / eer_from_scores / split_by_label

REPORT_MD = ROOT / "evaluation" / "fusion_crossdomain.md"
REPORT_JSON = ROOT / "evaluation" / "fusion_crossdomain.json"
CACHE_DIR = ROOT / ".tmp_ssl" / "crossdomain"

CFAD = Path("D:/VeriCall_data/CFAD/asvspoof_layout")
SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"
TRAIN_CAP = 20000  # 与融合实验一致的拟合规模


def extract_ssl_features(items, flac_dir: Path, tag: str, batch_size=4):
    """XLS-R 冻结特征（mean-pool），带缓存。与 eval_ssl_frontend 同口径。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"xlsr_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"  [{tag}] XLS-R 特征命中缓存 ({len(z['y'])} 条)", flush=True)
        return z["X"], z["y"]
    from transformers import AutoFeatureExtractor, AutoModel
    import soundfile as sf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
    ssl = AutoModel.from_pretrained(SSL_LOCAL).to(device).eval()
    subset = sorted(items, key=lambda t: (flac_dir / f"{t[0]}.flac").stat().st_size)
    X, y = [], []
    for st in range(0, len(subset), batch_size):
        chunk = subset[st:st + batch_size]
        wavs, labs = [], []
        for utt, lab in chunk:
            wav, sr = sf.read(str(flac_dir / f"{utt}.flac"))
            wavs.append(wav)
            labs.append(lab)
        inp = fe(wavs, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
        X.append(h)
        y.extend(labs)
        done = min(st + batch_size, len(subset))
        if done % 200 == 0 or done == len(subset):
            print(f"  [{tag}] XLS-R {done}/{len(subset)}", flush=True)
    Xa, ya = np.concatenate(X), np.array(y)
    np.savez(cache, X=Xa, y=ya)
    return Xa, ya


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--deg", default="", help="退化信道名(amr/mp3_16k/noise/phone8k)或留空=原布局")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from paths import EXP_RESULT_DIR  # noqa: E402

    # ---- 0. 布局选择（退化信道矩阵）
    deg_name = args.deg.strip()
    layout = CFAD if not deg_name else CFAD.parent / f"asvspoof_layout_deg_{deg_name}"
    report_md = REPORT_MD if not deg_name else REPORT_MD.with_name(f"fusion_crossdomain_deg_{deg_name}.md")
    report_json = REPORT_JSON if not deg_name else REPORT_JSON.with_name(f"fusion_crossdomain_deg_{deg_name}.json")
    tag = f"cfad{'_deg_'+deg_name if deg_name else ''}_{args.limit}"
    print(f"[0/4] 布局: {layout.name} (tag={tag})", flush=True)

    # ---- 1. CFAD 样本（零样本评估集）
    proto = layout / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    flac_dir = layout / "ASVspoof2019_LA_dev" / "flac"
    items = []
    with open(proto, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                items.append((p[1], 1 if p[4] == "bonafide" else 0))
    items = items[: args.limit]
    print(f"[1/4] CFAD 样本 {len(items)} 条（真 {sum(l for _, l in items)} / 伪 {sum(1 - l for _, l in items)}）", flush=True)

    # ---- 2. 在英文训练缓存上拟合 SSL-LR 与 stack-LR（不碰中文集）
    print("[2/4] 用 ASVspoof 英文 train 缓存拟合融合器（不碰中文集）...", flush=True)
    en_cache = ROOT / ".tmp_ssl" / "wav2vec2-xls-r-300m"
    ztr = np.load(en_cache / f"train_{TRAIN_CAP}.npz")
    ssl_clf = LogisticRegression(max_iter=300, C=1.0).fit(ztr["X"], ztr["y"])
    aa_tr = np.load(en_cache / f"aasist_scores_train_{TRAIN_CAP}.npz")["s"]
    ssl_tr = 1.0 - ssl_clf.predict_proba(ztr["X"])[:, 1]
    stack = LogisticRegression(max_iter=300).fit(np.stack([aa_tr, ssl_tr], axis=1), ztr["y"])

    # ---- 3. CFAD 上提取两路分数
    print("[3/4] CFAD 提取 XLS-R 特征与 AASIST 分数 ...", flush=True)
    Xc, yc = extract_ssl_features(items, flac_dir, tag)
    exp_dir = Path(EXP_RESULT_DIR) / fs.MAIN_EXP
    cfg = json.loads((exp_dir / "config.conf").read_text(encoding="utf-8"))
    module = __import__("models." + cfg["model_config"]["architecture"], fromlist=["Model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = module.Model(cfg["model_config"]).to(device)
    model.load_state_dict(torch.load(exp_dir / "weights" / "best.pth", map_location=device))
    model.eval()
    # AASIST 打分需与特征同序：extract 内部按大小排序，这里复用同一排序
    items_sorted = sorted(items, key=lambda t: (flac_dir / f"{t[0]}.flac").stat().st_size)
    aa_c, _ = fs.aasist_scores(model, items_sorted, flac_dir, device, args.batch_size,
                               "cfad", CACHE_DIR / f"aasist_{tag}.npz")
    if not np.array_equal(yc, np.array([l for _, l in items_sorted])):
        raise SystemExit("CFAD 标签对齐失败")

    ssl_c = 1.0 - ssl_clf.predict_proba(Xc)[:, 1]
    avg_c = (aa_c + ssl_c) / 2.0
    st_c = 1.0 - stack.predict_proba(np.stack([aa_c, ssl_c], axis=1))[:, 1]

    # ---- 4. EER 汇总
    rows = []
    for name, s in [("AASIST 单模型", aa_c), ("XLS-R 冻结 + LR", ssl_c),
                    ("等权平均融合", avg_c), ("LR stacking 融合", st_c)]:
        eer, _ = fs.eer_from_scores(*fs.split_by_label(s, yc))
        rows.append((name, eer))

    best = min(rows, key=lambda r: r[1])
    target = 5.0  # 计划书 §一：中文域 EER < 5%
    conclusion = (f"中文域（CFAD 零样本）最优 **{best[0]} EER {best[1]:.2f}%**，"
                  f"{'✅ 达到' if best[1] < target else '⚠️ 未达到'}计划书 §一「中文域 EER<5%」验收线。"
                  "融合相对单模型的增益见上表。")

    payload = {"date": dt.datetime.now().isoformat(timespec="seconds"),
               "dataset": f"CFAD {layout.name} (zero-shot{' + '+deg_name if deg_name else ''})",
               "n": int(len(yc)),
               "results": [{"name": n, "eer_pct": e} for n, e in rows],
               "conclusion": conclusion}
    lines = ["# 融合模型中文域零样本复验（CFAD）",
             "",
             f"> 生成：{payload['date']} · 脚本 `scripts/eval_fusion_crossdomain.py`",
             f"> CFAD {len(yc)} 条 · 布局 `{layout.name}` · 融合器在英文 ASVspoof train 上拟合，零样本应用",
             "",
             "| 方案 | CFAD EER |",
             "|---|---|"]
    lines += [f"| {n} | {e:.2f}% |" for n, e in rows]
    lines += ["", "## 结论", "", conclusion]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"报告 -> {REPORT_MD}")
    for n, e in rows:
        print(f"  {n}: {e:.2f}%")
    print(conclusion)


if __name__ == "__main__":
    main()
