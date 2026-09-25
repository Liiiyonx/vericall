# -*- coding: utf-8 -*-
"""eval_fusion_stack.py — AASIST + XLS-R 分数级融合复评实验（极致化计划书 §一.1）

背景：
  SSL 选型实验（evaluation/ssl_selection.md）结论：XLS-R 冻结+LR 单模型 dev 1.61%
  未超 AASIST dev 0.745%，B3 暂不上马。但计划书 §一.1 的「三路分数级融合」
  不要求单路最优——本实验检验 stacking 融合是否超过 AASIST 单模型。

设计：
  - 与 eval_ssl_frontend.py 完全相同的 train/dev 子集（同 shuffle 种子 + 同大小排序），
    直接复用其 .tmp_ssl/ 特征缓存（XLS-R 部分零重算）。
  - AASIST best.pth 对相同样本逐条打分（softmax 第 0 列 = spoof 概率），
    分数缓存到 .tmp_ssl/aasist_scores_*.npz，重跑秒级。
  - 融合器：train 上拟合 LogisticRegression（输入 [aasist, ssl] 二维分数），
    dev 上算 EER；同时给一个等权平均基线作对照。

自检：AASIST 单独在 dev 的 EER 应 ≈0.745%（model_quality.json 核验值），
      偏差 >0.1 个百分点则报警（说明对齐或权重有误）。

输出：evaluation/fusion_stack.md / .json

用法：
  python scripts/eval_fusion_stack.py                 # 全量（train 2 万 + dev 24844）
  python scripts/eval_fusion_stack.py --limit 2000 --train-cap 4000   # 冒烟
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "external" / "aasist"))

REPORT_MD = ROOT / "evaluation" / "fusion_stack.md"
REPORT_JSON = ROOT / "evaluation" / "fusion_stack.json"
CACHE = ROOT / ".tmp_ssl" / "wav2vec2-xls-r-300m"
MAIN_EXP = "LA_AASIST_5060_ep24_bs16"


def load_items(proto: Path, cap: int) -> list[tuple[str, int]]:
    """与 eval_ssl_frontend 完全同序：shuffle(42) → cap → 按文件大小排序。"""
    items = []
    with open(proto, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                items.append((p[1], 1 if p[4] == "bonafide" else 0))
    random.Random(42).shuffle(items)
    return items[:cap]


def sort_by_size(items, flac_dir: Path):
    return sorted(items, key=lambda t: (flac_dir / f"{t[0]}.flac").stat().st_size)


@torch.no_grad()
def aasist_scores(model, items, flac_dir: Path, device, batch_size, tag, cache: Path):
    """逐条 AASIST spoof 概率，带缓存。返回与 items 同序的 np 数组。"""
    if cache.exists():
        z = np.load(cache)
        print(f"  [{tag}] AASIST 分数命中缓存 ({len(z['y'])} 条)", flush=True)
        return z["s"], z["y"]
    import soundfile as sf
    scores, labels = [], []
    cut = 64600
    for st in range(0, len(items), batch_size):
        chunk = items[st:st + batch_size]
        wavs = []
        for utt, lab in chunk:
            wav, sr = sf.read(str(flac_dir / f"{utt}.flac"))
            if len(wav) >= cut:
                wav = wav[:cut]
            else:
                reps = int(np.ceil(cut / max(1, len(wav))))
                wav = np.tile(wav, reps)[:cut]
            wavs.append(wav)
            labels.append(lab)
        x = torch.tensor(np.array(wavs), dtype=torch.float32).to(device)
        _, out = model(x)
        prob = torch.softmax(out.float(), dim=-1)[:, 0].cpu().numpy()
        scores.append(prob)
        done = min(st + batch_size, len(items))
        if done % 2000 == 0 or done == len(items):
            print(f"  [{tag}] AASIST {done}/{len(items)}", flush=True)
    s = np.concatenate(scores)
    y = np.array(labels)
    np.savez(cache, s=s, y=y)
    return s, y


def eer_from_scores(bon_scores, spo_scores) -> tuple[float, float]:
    """项目口径：伪造概率越高越可疑。compute_eer 返回 (EER百分数, 阈值)。"""
    from metrics import compute_eer
    eer_pct, thr = compute_eer(list(bon_scores), list(spo_scores))
    return eer_pct, thr


def split_by_label(scores, labels):
    bon = scores[labels == 1]
    spo = scores[labels == 0]
    return bon, spo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24844, help="dev 条数")
    ap.add_argument("--train-cap", type=int, default=20000, help="train 条数")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression

    from paths import ASVSPOOF_LA, EXP_RESULT_DIR  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    la = Path(ASVSPOOF_LA)
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---- 1. 重建与 SSL 实验同序的样本列表
    print("[1/5] 重建样本列表（协议 shuffle + 大小排序）...", flush=True)
    train_items = sort_by_size(load_items(proto_dir / "ASVspoof2019.LA.cm.train.trn.txt",
                                          args.train_cap),
                               la / "ASVspoof2019_LA_train" / "flac")
    dev_items = sort_by_size(load_items(proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
                                        args.limit),
                             la / "ASVspoof2019_LA_dev" / "flac")
    y_dev = np.array([lab for _, lab in dev_items])

    # ---- 2. XLS-R 冻结特征 + LR（复用缓存）
    print("[2/5] 加载 XLS-R 特征缓存并拟合 LR ...", flush=True)
    ztr = np.load(CACHE / f"train_{args.train_cap}.npz")
    zdv = np.load(CACHE / f"dev_{args.limit}.npz")
    if not np.array_equal(ztr["y"], np.array([l for _, l in train_items])) or \
       not np.array_equal(zdv["y"], y_dev):
        raise SystemExit("特征缓存与协议顺序不一致——不要用旧缓存，先重跑 eval_ssl_frontend.py")
    ssl_clf = LogisticRegression(max_iter=300, C=1.0).fit(ztr["X"], ztr["y"])
    print("      LR 拟合完成", flush=True)
    ssl_tr = 1.0 - ssl_clf.predict_proba(ztr["X"])[:, 1]   # spoof 概率
    ssl_dv = 1.0 - ssl_clf.predict_proba(zdv["X"])[:, 1]

    # ---- 3. AASIST 逐条打分（缓存）
    exp_dir = Path(EXP_RESULT_DIR) / MAIN_EXP
    cfg = json.loads((exp_dir / "config.conf").read_text(encoding="utf-8"))
    mcfg = cfg["model_config"]
    module = __import__("models." + mcfg["architecture"], fromlist=["Model"])
    model = module.Model(mcfg).to(device)
    model.load_state_dict(torch.load(exp_dir / "weights" / "best.pth", map_location=device))
    model.eval()
    aa_tr, _ = aasist_scores(model, train_items, la / "ASVspoof2019_LA_train" / "flac",
                             device, args.batch_size, "train",
                             CACHE / f"aasist_scores_train_{args.train_cap}.npz")
    aa_dv, _ = aasist_scores(model, dev_items, la / "ASVspoof2019_LA_dev" / "flac",
                             device, args.batch_size, "dev",
                             CACHE / f"aasist_scores_dev_{args.limit}.npz")

    # ---- 4. 单模型 EER
    rows = []
    eer_aa, _ = eer_from_scores(*split_by_label(aa_dv, y_dev))
    rows.append(("AASIST 单模型", eer_aa, "应与 dev 核验值 0.745% 一致"))
    eer_ssl, _ = eer_from_scores(*split_by_label(ssl_dv, y_dev))
    rows.append(("XLS-R 冻结 + LR", eer_ssl, "SSL 选型实验同口径"))

    # ---- 5. 融合：等权平均基线 + LR stacking
    avg_dv = (aa_dv + ssl_dv) / 2.0
    eer_avg, _ = eer_from_scores(*split_by_label(avg_dv, y_dev))
    rows.append(("等权平均融合", eer_avg, "无学习基线"))

    Ztr = np.stack([aa_tr, ssl_tr], axis=1)
    Zdv = np.stack([aa_dv, ssl_dv], axis=1)
    stack = LogisticRegression(max_iter=1000).fit(Ztr, ztr["y"])
    st_dv = 1.0 - stack.predict_proba(Zdv)[:, 1]
    eer_st, thr_st = eer_from_scores(*split_by_label(st_dv, y_dev))
    w = stack.coef_[0].tolist()
    rows.append(("LR stacking 融合", eer_st,
                 f"权重 aasist={w[0]:.2f}, ssl={w[1]:.2f}, b={stack.intercept_[0]:.2f}"))

    # ---- 6. 自检与结论
    warn = ""
    if abs(eer_aa - 0.745) > 0.1:
        warn = (f"⚠️ AASIST dev EER {eer_aa:.3f}% 与核验值 0.745% 偏差超 0.1pp，"
                "请检查权重/对齐！")
    best = min(rows, key=lambda r: r[1])
    gain = (eer_aa - best[1]) / eer_aa * 100
    conclusion = (
        f"最优为 **{best[0]}（dev EER {best[1]:.3f}%）**，相对 AASIST 单模型 "
        f"{'提升' if gain > 0 else '退化'}{abs(gain):.1f}%。"
        + ("融合有效 → 建议将 XLS-R 支路纳入三路融合架构（计划书 §一.1），"
           "下一步在 CFAD/FMFCC-A 中文集上复验。" if gain > 5 else
           "融合收益不显著（≤5%）→ XLS-R 支路暂缓，优先中文域数据扩充。")
    )

    # ---- 7. 报告
    payload = {"date": dt.datetime.now().isoformat(timespec="seconds"),
               "dev_n": int(len(y_dev)), "train_n": int(len(ztr["y"])),
               "results": [{"name": n, "eer_pct": e, "note": t} for n, e, t in rows],
               "warning": warn, "conclusion": conclusion}
    lines = [
        "# AASIST + XLS-R 分数级融合复评（计划书 §一.1 三路融合第一块拼图）",
        "",
        f"> 生成：{payload['date']} · 脚本 `scripts/eval_fusion_stack.py`",
        f"> dev {payload['dev_n']} 条 / train {payload['train_n']} 条，"
        "与 SSL 选型实验同集同序（特征/分数全部缓存可复算）",
        "",
        "| 方案 | dev EER | 备注 |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {e:.3f}% | {t} |" for n, e, t in rows]
    lines += ["", "## 结论", "", conclusion]
    if warn:
        lines += ["", f"**自检告警**：{warn}"]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"报告 -> {REPORT_MD}")
    for n, e, t in rows:
        print(f"  {n}: {e:.3f}%  ({t})")
    print(conclusion)


if __name__ == "__main__":
    main()
