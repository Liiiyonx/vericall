#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — AASIST 通道① 离线 EER 评估器。

用途:
  训练结束后(或中途拿到周期存档时)量化通道①的真实鉴伪水平, 得到:
    - EER (等错误率): 论文/比赛通用指标。ASVspoof2019 LA 上 AASIST 基线
      约 0.8%~1.2%; 本项目的合格线定为 ≤1.5%。
    - EER 处阈值: 用于校准 fusion 里 0.85 硬拦截阈值是否合理。
    - 分段统计: bonafide / spoof 两类分数的均值与分位数, 用来判断
      融合层的 0.5 决策边界是否偏。

数据源:
  - dev  集: 24844 条 (2548 bonafide / 22296 spoof) —— 快速体检
  - eval 集: 71237 条 (7355 bonafide / 63882 spoof) —— 最终成绩
  本机 ASVspoof2019.LA.cm.eval.trl.txt 带标签, 故 eval EER 可直接算。

注意:
  - 推理走 GPU, 但显存占用不大(约 1.5GB)。若训练仍在跑, 用 --device cpu
    会非常慢(71237 条 × 数秒), 建议等训练结束再跑 eval。
  - num_workers=0: Windows 上多进程 DataLoader 会触发共享内存错误 1455。

用法:
  python scripts/eval_aasist_eer.py                    # dev 集, 自动挑最新 best.pth
  python scripts/eval_aasist_eer.py --split eval       # eval 集
  python scripts/eval_aasist_eer.py --split both
  python scripts/eval_aasist_eer.py --ckpt <path>      # 指定权重
  python scripts/eval_aasist_eer.py --limit 2000       # 只测前 N 条(快速冒烟)
"""
import os
import re
import sys
import glob
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "src")))
from paths import AASIST_DIR, ASVSPOOF_LA, EXP_RESULT_DIR  # noqa: E402

sys.path.insert(0, str(AASIST_DIR))

from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list  # noqa: E402
from evaluation import compute_eer  # noqa: E402

DATABASE_PATH = str(ASVSPOOF_LA)
EXP_ROOT = str(EXP_RESULT_DIR)


def _latest_exp():
    subs = [os.path.join(EXP_ROOT, d) for d in os.listdir(EXP_ROOT)]
    subs = [d for d in subs if os.path.isdir(d)]
    return max(subs, key=os.path.getmtime)


def resolve_ckpt(exp_dir, explicit=None):
    """权重优先级: best.pth > swa.pth > epoch_{N}_{EER}.pth > epoch_{N}.pth"""
    if explicit:
        return explicit
    wdir = os.path.join(exp_dir, "weights")
    for name in ("best.pth", "swa.pth"):
        p = os.path.join(wdir, name)
        if os.path.exists(p):
            return p
    cand = glob.glob(os.path.join(wdir, "epoch_*_*.pth"))
    if cand:
        return min(cand, key=lambda p: float(re.findall(r"_([\d.]+)\.pth", p)[0]))
    cand = glob.glob(os.path.join(wdir, "epoch_*.pth"))
    if cand:
        return max(cand, key=lambda p: (int(re.findall(r"epoch_(\d+)\.pth", p)[0]),
                                        os.path.getmtime(p)))
    raise FileNotFoundError(f"实验目录无可用权重: {wdir}")


def _parse_conf(path):
    """极简 JSON 解析(AASIST 的 conf 是 JSON 格式)"""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(exp_dir, device):
    cfg_path = os.path.join(exp_dir, "config.conf")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"缺少实验配置: {cfg_path}")
    cfg = _parse_conf(cfg_path)
    mcfg = cfg["model_config"]
    arch = mcfg["architecture"]
    module = __import__("models." + arch, fromlist=["Model"])
    model = getattr(module, "Model")(mcfg).to(device)
    return model, mcfg


def _dataloader(split, mcfg, batch_size, limit=None):
    """构建 dev / eval 的 DataLoader 与标签表。

    genSpoof_list 的 is_eval=True 分支不返回标签(官方 eval 标签未公开), 但本机
    eval.trl.txt 第 5 列是带标签的, 故 eval 分支自己解析标签, 不依赖 genSpoof_list。
    """
    db = DATABASE_PATH
    if split == "dev":
        trl = os.path.join(db, "ASVspoof2019_LA_cm_protocols",
                           "ASVspoof2019.LA.cm.dev.trl.txt")
        base = os.path.join(db, "ASVspoof2019_LA_dev")
        d_meta, keys = genSpoof_list(dir_meta=trl, is_train=False, is_eval=False)
    else:
        trl = os.path.join(db, "ASVspoof2019_LA_cm_protocols",
                           "ASVspoof2019.LA.cm.eval.trl.txt")
        base = os.path.join(db, "ASVspoof2019_LA_eval")
        d_meta, keys = {}, []
        with open(trl, "r") as f:
            for line in f:
                _, key, _, _, label = line.strip().split(" ")
                keys.append(key)
                d_meta[key] = 1 if label == "bonafide" else 0

    if limit:
        keys = keys[:limit]
    cut = int(mcfg.get("nb_samp", 64600))
    # data_utils 内部用 pathlib 拼路径, base_dir 必须是 Path 而非 str
    ds = Dataset_ASVspoof2019_devNeval(list_IDs=keys, base_dir=Path(base), cut=cut)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=False), d_meta


@torch.no_grad()
def run_split(model, split, mcfg, device, batch_size, limit, use_amp):
    loader, d_meta = _dataloader(split, mcfg, batch_size, limit)
    model.eval()
    bon, spo = [], []
    t0 = time.time()
    total = len(loader)
    for i, (batch_x, utt_ids) in enumerate(loader, 1):
        batch_x = batch_x.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, out = model(batch_x)
        # 原版 AASIST 标签约定: bonafide=1, spoof=0 —— softmax 第 0 列才是 spoof 概率
        prob = torch.softmax(out.float(), dim=-1)[:, 0].cpu().numpy()
        for s, u in zip(prob, utt_ids):
            (bon if d_meta[u] == 1 else spo).append(float(s))
        if i % 200 == 0 or i == total:
            el = time.time() - t0
            eta = el / i * (total - i)
            print(f"  [{split}] {i}/{total}  {el:.0f}s  ETA {eta:.0f}s",
                  flush=True)

    bon = np.asarray(bon)
    spo = np.asarray(spo)
    # compute_eer(target=bonafide, nontarget=spoof): 分数越高越像 bonafide 才对,
    # 这里用的是 spoof 概率, 故把 spoof 当 "target" 传入, 语义等价。
    eer, thr = compute_eer(spo, bon)
    return {
        "split": split,
        "n_bonafide": len(bon),
        "n_spoof": len(spo),
        "eer": eer * 100.0,
        "thr": float(thr),
        "bonafide_mean": float(bon.mean()),
        "spoof_mean": float(spo.mean()),
        "bonafide_p95": float(np.percentile(bon, 95)),
        "spoof_p5": float(np.percentile(spo, 5)),
        "elapsed": time.time() - t0,
    }


def report(r):
    print("-" * 60)
    print(f"[{r['split']}] 样本: {r['n_bonafide']} bonafide / {r['n_spoof']} spoof"
          f"   用时 {r['elapsed']:.0f}s")
    print(f"  EER            = {r['eer']:.3f}%   (阈值 {r['thr']:.4f})")
    print(f"  bonafide 分数  均值 {r['bonafide_mean']:.4f}  p95 {r['bonafide_p95']:.4f}")
    print(f"  spoof    分数  均值 {r['spoof_mean']:.4f}  p5  {r['spoof_p5']:.4f}")
    ok = "✅ 达标" if r["eer"] <= 1.5 else "⚠️ 未达合格线(1.5%)"
    print(f"  判定: {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "eval", "both"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    exp = args.exp or _latest_exp()
    ckpt = resolve_ckpt(exp, args.ckpt)
    print(f"实验目录: {exp}")
    print(f"权重    : {os.path.basename(ckpt)}")

    device = torch.device(args.device)
    model, mcfg = build_model(exp, device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    print(f"模型    : {mcfg.get('architecture')}  nb_samp={mcfg.get('nb_samp')}"
          f"  device={device}")

    use_amp = (device.type == "cuda")
    splits = ["dev", "eval"] if args.split == "both" else [args.split]
    results = []
    for sp in splits:
        results.append(run_split(model, sp, mcfg, device, args.batch_size,
                                 args.limit, use_amp))
    print("=" * 60)
    for r in results:
        report(r)


if __name__ == "__main__":
    main()
