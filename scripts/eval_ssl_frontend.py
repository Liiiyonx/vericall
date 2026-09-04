# -*- coding: utf-8 -*-
"""eval_ssl_frontend.py — SSL 前端选型对比实验（极致化计划书 §一）

对比两条技术路线在 ASVspoof2019 LA 上的表现：
  A. 现有 AASIST（原始波形 → AASIST，已训练，EER 证据链见 exp_result）
  B. SSL 前端（wav2vec2-XLS-R-300M / WavLM 冻结特征 → 轻量分类头）
     —— 近年 ASVspoof 冠军方案的标准结构

选型实验设计（快速但有效）：
  SSL 冻结提特征 + sklearn LogisticRegression 头（train 子集训练、dev 评估），
  若 B 明显优于 A，则立项做「SSL+AASIST 后端」完整训练（多卡，见计划书）。

输出：evaluation/ssl_selection.md（对比报告）+ ssl_selection.json（原始数据）

降级策略：缺 transformers/torch/SSL 权重/GPU 任一项 → 生成骨架报告
（含实验矩阵与执行步骤），退出码 0，不阻塞 CI。

用法：
  python scripts/eval_ssl_frontend.py --limit 2000          # dev 冒烟
  python scripts/eval_ssl_frontend.py --limit 20000 --ssl facebook/wav2vec2-xls-r-300m
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

REPORT_MD = ROOT / "evaluation" / "ssl_selection.md"
REPORT_JSON = ROOT / "evaluation" / "ssl_selection.json"

EXPERIMENT_MATRIX = [
    ("A0", "AASIST（现有基线）", "原始波形", "已训练 best.pth", "证据链已存在，直接引用"),
    ("B1", "wav2vec2-XLS-R-300M + LR 头", "SSL 冻结特征", "train 子集 2h 训练", "选型主力"),
    ("B2", "WavLM-Base+ + LR 头", "SSL 冻结特征", "同上", "备选，域偏置更小"),
    ("B3", "XLS-R + AASIST 后端（完整训练）", "SSL+AASIST", "多卡 1-2 天", "B1 胜出后立项"),
]

SKELETON_STEPS = [
    "pip install transformers torch（训练用 python 环境）",
    "huggingface-cli download facebook/wav2vec2-xls-r-300m（境内：HF_ENDPOINT=https://hf-mirror.com）",
    "python scripts/eval_ssl_frontend.py --limit 2000   # 冒烟",
    "python scripts/eval_ssl_frontend.py                # 全量 dev（24844 条）",
    "B1 显著优于 A0 → 立项 B3 完整训练（configs/ 新增 SSL 配置）",
]


def try_import_ssl():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, str(e)


def write_report(payload: dict):
    lines = [
        "# SSL 前端选型报告（通道①升级实验）",
        "",
        f"> 生成：{payload['date']} · 脚本 `scripts/eval_ssl_frontend.py`",
        "> 对应：极致化计划书 §一「模型矩阵」增量 1",
        "",
        "## 实验矩阵",
        "",
        "| 编号 | 方案 | 特征 | 训练成本 | 备注 |",
        "|---|---|---|---|---|",
    ]
    for row in EXPERIMENT_MATRIX:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 结果", ""]
    if payload.get("results"):
        lines += ["| 方案 | dev EER | FAR@EER | 样本数 | 备注 |",
                  "|---|---|---|---|---|"]
        for r in payload["results"]:
            lines.append(f"| {r['name']} | {r['eer']:.2%} | {r.get('far', 0):.2%} "
                         f"| {r['n']} | {r.get('note', '')} |")
        lines += ["", "## 选型结论", "", payload.get("conclusion", "（待填）")]
    else:
        lines += [f"**骨架报告**：{payload.get('reason', '依赖未就绪')}，实验未执行。",
                  "", "### 执行步骤", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(SKELETON_STEPS, 1)]
        lines += ["", "### 选型标准（评审维度）", "",
                  "1. **dev EER**：与 A0 基线（eval 3.49% / dev 0.745%）对比，提升 <30% 相对则不上马；",
                  "2. **跨域保持**：CFAD/FMFCC-A 子集 EER 不得显著劣于 A0；",
                  "3. **流式可行性**：300M 级 SSL 单窗（3s）GPU 推理 <100ms 才允许进实时线，否则只进离线深评；",
                  "4. **显存预算**：与 SenseVoice/CAMPPlus 常驻合计 <6GB（P1-1 阈值纪律）。"]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"报告 -> {REPORT_MD}")


def load_protocol(protocol: Path):
    """ASVspoof19 LA 协议 → [(utt_id, label)]，label: 1=bonafide 0=spoof"""
    items = []
    with open(protocol, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                items.append((parts[1], 1 if parts[4] == "bonafide" else 0))
    return items


def run_ssl_branch(ssl_model: str, limit: int, train_cap: int) -> dict:
    """SSL 冻结特征 + LR 头。返回结果 dict 或抛异常由调用方降级。"""
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    from sklearn.linear_model import LogisticRegression
    from metrics import compute_eer
    from paths import ASVSPOOF_LA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_ext = AutoFeatureExtractor.from_pretrained(ssl_model)
    ssl = AutoModel.from_pretrained(ssl_model).to(device).eval()

    la = Path(ASVSPOOF_LA)
    # 协议文件可能在 LA 根目录或 ASVspoof2019_LA_cm_protocols/ 子目录
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    if not (proto_dir / "ASVspoof2019.LA.cm.train.trn.txt").exists():
        proto_dir = la
    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    flac_dirs = {"train": la / "ASVspoof2019_LA_train" / "flac",
                 "dev": la / "ASVspoof2019_LA_dev" / "flac"}

    def extract(items, flac_dir, cap, batch_size=8):
        X, y = [], []
        import soundfile as sf
        subset = items[:cap]
        # 按文件大小（时长代理）排序后组批：批次内长度一致，避免显存碎片 OOM
        subset = sorted(subset, key=lambda t: (flac_dir / f"{t[0]}.flac").stat().st_size)
        for st in range(0, len(subset), batch_size):
            chunk = subset[st:st + batch_size]
            wavs, labs = [], []
            for utt, lab in chunk:
                wav, sr = sf.read(str(flac_dir / f"{utt}.flac"))
                wavs.append(wav)
                labs.append(lab)
            inp = feat_ext(wavs, sampling_rate=sr, return_tensors="pt",
                           padding=True).to(device)
            with torch.no_grad():
                h = ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
            X.append(h)
            y.extend(labs)
            done = min(st + batch_size, len(subset))
            if done % (batch_size * 100) == 0:
                torch.cuda.empty_cache() if device == "cuda" else None
            if done % (batch_size * 50) == 0 or done == len(subset):
                print(f"  [extract] {done}/{len(subset)}", flush=True)
        return np.concatenate(X), np.array(y)

    train_items = load_protocol(train_proto)
    dev_items = load_protocol(dev_proto)
    # 协议文件按类别聚块排列（bonafide 在前），截取前必须打乱，否则单类无法训练
    import random as _rnd
    _rnd.Random(42).shuffle(train_items)
    _rnd.Random(42).shuffle(dev_items)
    Xtr, ytr = extract(train_items, flac_dirs["train"], min(train_cap, len(train_items)))
    Xdv, ydv = extract(dev_items, flac_dirs["dev"], min(limit, len(dev_items)))

    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
    spoof_scores = 1.0 - clf.predict_proba(Xdv)[:, 1]  # 伪造概率（项目口径）
    bon = [s for s, lab in zip(spoof_scores, ydv) if lab == 1]
    spo = [s for s, lab in zip(spoof_scores, ydv) if lab == 0]
    eer_pct, thr = compute_eer(bon, spo)  # compute_eer 返回 (EER百分数, 阈值)
    far = float((np.asarray(bon) >= thr).mean())  # EER 工作点的误拦率
    return {"name": f"B1 {ssl_model.split('/')[-1]} + LR",
            "eer": eer_pct / 100.0, "far": far, "threshold": float(thr), "n": len(ydv),
            "note": f"train {len(ytr)} 条, device={device}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="dev 评估条数上限")
    ap.add_argument("--train-cap", type=int, default=20000, help="LR 头训练条数上限（冒烟可调小）")
    ap.add_argument("--ssl", default="facebook/wav2vec2-xls-r-300m")
    args = ap.parse_args()

    payload = {"date": dt.datetime.now().isoformat(timespec="seconds"),
               "results": [], "matrix": EXPERIMENT_MATRIX}

    ok, why = try_import_ssl()
    if not ok:
        payload["reason"] = f"缺少依赖：{why}"
        write_report(payload)
        return

    # A0 基线：引用已训练的 EER 证据链（不重复跑 7 万条）
    # 优先选含 eval 集核验字段的主实验文件；model_quality.json 中 EER 单位为百分数
    mq = sorted(ROOT.glob("external/aasist/exp_result/**/model_quality.json"),
                key=lambda p: 0 if "eval_eer_best_verified" in p.read_text(encoding="utf-8") else 1)
    if mq:
        with open(mq[0], encoding="utf-8") as f:
            q = json.load(f)
        raw_eer = q.get("eval_eer_best_verified") or q.get("eval_eer") or q.get("dev_eer", 0)
        eer = raw_eer / 100.0 if raw_eer > 1 else raw_eer  # 百分数 → 小数
        payload["results"].append({
            "name": "A0 AASIST（现有基线）",
            "eer": eer,
            "far": 0, "n": 0,
            "note": f"引用 {mq[0].parent.name}/model_quality.json（eval 集核验值，不重复推理）"})

    try:
        payload["results"].append(run_ssl_branch(args.ssl, args.limit, args.train_cap))
    except Exception as e:  # noqa: BLE001
        payload["reason"] = f"SSL 实验执行失败：{type(e).__name__}: {e}"
        if not payload["results"]:
            write_report(payload)
            return

    if len(payload["results"]) >= 2:
        a0, b1 = payload["results"][0], payload["results"][-1]
        if a0["eer"] and b1["eer"] < a0["eer"] * 0.7:
            payload["conclusion"] = (f"**B1 显著优于 A0**（{b1['eer']:.2%} vs {a0['eer']:.2%}，"
                                     "相对提升 >30%）→ 建议立项 B3「XLS-R + AASIST 后端」完整训练。")
        else:
            payload["conclusion"] = (f"B1（{b1['eer']:.2%}）相对 A0（{a0['eer']:.2%}）提升不足 30%，"
                                     "维持 AASIST 主力线，SSL 转离线深评候选。")
    write_report(payload)


if __name__ == "__main__":
    main()
