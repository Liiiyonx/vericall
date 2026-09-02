#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P2-2 跨域评测矩阵 · 即插即用封装
================================================================
一句话：把真实 AASIST 权重丢进 `external/aasist/exp_result/<实验>/weights/best.pth`，
把各域数据按 ASVspoof-LA 风格布局放好，跑本脚本就得到跨域泛化矩阵
（`evaluation/cross_domain_matrix.md`）。

设计要点：
  - 所有「测试域」都复用 ASVspoof-LA 风格目录约定
    （`ASVspoof2019_LA_cm_protocols/*.trl.txt` + `ASVspoof2019_LA_<split>/flac/`），
    因此 `scripts/convert_fmfcc_protocol.py` 产出的 FMFCC-A 根可直接当作一个域接入。
  - 权重自动挑选：best.pth > swa.pth > epoch_{N}_{EER}.pth > epoch_{N}.pth
    （复用 attack_breakdown_eval 的 resolve_ckpt 逻辑）。
  - 任一域缺数据 / 缺权重时，该格标 TODO，不产生伪数字——诚实优先。

支持的域（行=训练域固定为 2019LA 现模型；列=各测试域）：
  2019LA-eval(英) | FMFCC-A(中) | CFAD(中) | 红队干净 | 红队电话 | In-the-Wild

无权重/无数据时本脚本仍会生成「骨架矩阵 + 填充流程」，保证 repo 始终有一份
可答辩引用的矩阵文档（已知格：2019LA eval 3.49%）。

用法：
    python evaluation/cross_domain_eval.py                # 自动发现域 + 计算可算的格
    python evaluation/cross_domain_eval.py --exp <dir> --ckpt <pth>
    python evaluation/cross_domain_eval.py --list-domains # 只报告各域可用性
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from paths import ASVSPOOF_LA, AASIST_DIR, EXP_RESULT_DIR  # noqa: E402
import attack_breakdown_eval as abe  # noqa: E402  (纯 Python，无 torch 顶层导入)
from metrics import compute_eer, far_frr_at  # noqa: E402


# ------------------------------------------------------------------ #
# 域定义：每个测试域 = 一个 ASVspoof-LA 风格数据根 + 使用的 split
# ------------------------------------------------------------------ #
def _domain_roots() -> dict[str, Path]:
    """返回 {域名: 数据根}，数据根不存在则 Path 指向预期位置（用于 TODO 提示）。"""
    roots = {}
    roots["2019LA-eval"] = ASVSPOOF_LA
    # FMFCC-A / CFAD：convert_fmfcc_protocol.py 的 --out 默认位置（可用 env 覆盖）
    roots["FMFCC-A"] = Path(os.environ.get(
        "VERICALL_FMFCCA_ROOT",
        str(_ROOT / "data" / "raw" / "FMFCC-A_asvspoof"))).expanduser()
    roots["CFAD"] = Path(os.environ.get(
        "VERICALL_CFAD_ROOT",
        str(_ROOT / "data" / "raw" / "CFAD_asvspoof"))).expanduser()
    # 红队音频（P2-1 真实克隆）：REDTEAM_DIR 下出现 *.wav 即视为可用
    from paths import REDTEAM_DIR
    roots["redteam-clean"] = REDTEAM_DIR
    roots["redteam-phone"] = REDTEAM_DIR
    roots["in-the-wild"] = Path(os.environ.get(
        "VERICALL_WILD_ROOT", str(_ROOT / "data" / "raw" / "wild"))).expanduser()
    return roots


def _domain_available(name: str, root: Path) -> tuple[bool, str]:
    """判断某域是否有可评测数据。返回 (可用, 说明)。"""
    if name.startswith("2019LA"):
        proto = root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
        return (proto.is_file(), "eval 协议" if proto.is_file() else "缺 eval 协议")
    if name in ("FMFCC-A", "CFAD"):
        proto = root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
        flac = root / "ASVspoof2019_LA_dev" / "flac"
        ok = proto.is_file() and flac.is_dir()
        return (ok, "dev 协议+flac" if ok else "缺 dev 协议/flac（先跑 convert_fmfcc_protocol.py）")
    if name.startswith("redteam"):
        wavs = list(root.rglob("*.wav")) if root.is_dir() else []
        return (len(wavs) > 0, f"{len(wavs)} 条 wav" if wavs else "缺红队音频（P2-1）")
    if name == "in-the-wild":
        return (root.is_dir() and any(root.rglob("*.wav")), "缺 In-the-Wild 数据")
    return (False, "未知域")


# ------------------------------------------------------------------ #
# 单域推理（仅在有真实权重 + 数据时调用；torch 在函数内惰性导入）
# ------------------------------------------------------------------ #
def compute_domain_eer(data_root: Path, split: str, exp_dir: str, ckpt: str,
                       device: str, limit: int | None = None) -> dict:
    """对单个 ASVspoof-LA 风格域跑 AASIST 推理并算 EER。

    复用 attack_breakdown_eval 的协议解析与推理骨架，但数据根可参数化，
    从而同一套代码覆盖 LA / FMFCC-A / CFAD 等所有「风格对齐」的域。
    """
    import torch
    from torch.utils.data import DataLoader

    if str(AASIST_DIR) not in sys.path:
        sys.path.insert(0, str(AASIST_DIR))
    from data_utils import Dataset_ASVspoof2019_devNeval  # noqa: E402

    proto_name = ("ASVspoof2019.LA.cm.eval.trl.txt" if split == "eval"
                  else "ASVspoof2019.LA.cm.dev.trl.txt")
    proto_path = data_root / "ASVspoof2019_LA_cm_protocols" / proto_name
    if not proto_path.is_file():
        raise FileNotFoundError(f"缺协议: {proto_path}")
    rows: list[tuple[str, str, str]] = []
    for line in proto_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(" ")
        if len(parts) < 5:
            continue
        rows.append((parts[1], parts[3], parts[4]))

    cfg_path = Path(exp_dir) / "config.conf"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"缺实验配置: {cfg_path}")
    mcfg = json.loads(cfg_path.read_text(encoding="utf-8"))["model_config"]
    arch = mcfg["architecture"]
    module = __import__("models." + arch, fromlist=["Model"])
    model = getattr(module, "Model")(mcfg).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    base = data_root / f"ASVspoof2019_LA_{split}"
    cut = int(mcfg.get("nb_samp", 64600))
    keys = [u for u, _, _ in rows]
    if limit:
        keys = keys[:limit]
        rows = rows[:limit]
    ds = Dataset_ASVspoof2019_devNeval(list_IDs=keys, base_dir=base, cut=cut)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    scores: dict[str, float] = {}
    with torch.no_grad():
        for batch_x, utt_ids in loader:
            batch_x = batch_x.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, out = model(batch_x)
            prob = torch.softmax(out.float(), dim=-1)[:, 0].cpu().numpy()
            for s, u in zip(prob, utt_ids):
                scores[u] = float(s)

    bon, spo = [], []
    meta = {u: lab for u, _, lab in rows}
    for u, s in scores.items():
        (bon if meta.get(u) == "bonafide" else spo).append(s)
    eer, thr = compute_eer(bon, spo)
    far, frr = far_frr_at(bon, spo, thr)
    return {"n_bonafide": len(bon), "n_spoof": len(spo),
            "eer_pct": round(eer, 4), "threshold": round(thr, 6),
            "far_pct": round(far, 4), "frr_pct": round(frr, 4)}


# ------------------------------------------------------------------ #
# 矩阵组装
# ------------------------------------------------------------------ #
KNOWN_CELLS = {
    ("2019LA（现模型）", "2019LA-eval"): "3.49%",   # 来自实训产出（tag v0.2-repro）
}


def build_matrix(avail: dict[str, tuple[bool, str]],
                 computed: dict[str, dict]) -> str:
    cols = ["2019LA-eval", "FMFCC-A", "CFAD", "redteam-clean",
            "redteam-phone", "in-the-wild"]
    col_cn = {
        "2019LA-eval": "2019LA eval（英）", "FMFCC-A": "FMFCC-A（中）",
        "CFAD": "CFAD（中）", "redteam-clean": "红队干净",
        "redteam-phone": "红队电话", "in-the-wild": "In-the-Wild",
    }
    lines = [
        "# 谛听 VeriCall · 跨域评测矩阵（P2-2）",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 训练域：2019LA（现模型，AASIST，dev EER 0.745% / eval EER 3.49%）",
        "",
        "> 本矩阵回答评委对「跨语言/跨信道泛化」的核心疑问。每个测试格 = "
        "英文训练模型在该域的 EER；泛化鸿沟 = 该格 EER − 英文 eval EER。",
        "> 缺权重或数据时不编造数字，标 TODO 并附填充流程。",
        "",
        "## 矩阵（行=训练域，列=测试域）",
        "",
        "| 训练域 \\ 测试域 | " + " | ".join(col_cn[c] for c in cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    row_label = "2019LA（现模型）"
    cells = []
    for c in cols:
        if (row_label, c) in KNOWN_CELLS:
            cells.append(KNOWN_CELLS[(row_label, c)])
        elif c in computed:
            r = computed[c]
            cells.append(f"{r['eer_pct']:.3f}%")
        elif avail.get(c, (False, ""))[0]:
            cells.append("计算中…")
        else:
            cells.append("TODO")
    lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    lines.append("")

    # 可用性 / 计算明细
    lines += ["## 各域状态与填充流程", ""]
    lines.append("| 测试域 | 数据可用 | 说明 | 结果 |")
    lines.append("|---|---|---|---|")
    for c in cols:
        ok, note = avail.get(c, (False, "未登记"))
        res = computed[c]["eer_pct"] if c in computed else ("已知 3.49%" if (row_label, c) in KNOWN_CELLS else "TODO")
        lines.append(f"| {col_cn[c]} | {'✅' if ok else '❌'} | {note} | {res} |")

    lines += [
        "",
        "## 即插即用填充步骤（真实数据到位后）",
        "",
        "1. **放权重**：把训练好的 `best.pth` 放进 "
        "`external/aasist/exp_result/<实验>/weights/`（含 `config.conf`）。",
        "2. **放英文域**：`VERICALL_ASVSPOOF_LA` 指向 ASVspoof2019 LA（已有）。",
        "3. **放中文域**：",
        "   ```",
        "   python scripts/convert_fmfcc_protocol.py --root <FMFCC-A> \\",
        "           --out data/raw/FMFCC-A_asvspoof",
        "   # CFAD 同理 -> data/raw/CFAD_asvspoof",
        "   ```",
        "4. **放红队音频**：真实方言克隆音频（P2-1 GPT-SoVITS 合成）放入 `data/redteam/`。",
        "5. **跑本脚本**：`python evaluation/cross_domain_eval.py` → 自动计算可算的格并刷新本矩阵。",
        "",
        "## 泛化鸿沟（待填充）",
        "",
        "泛化鸿沟 = 各中文/红队域 EER − 英文 eval 3.49%。**英文训练→多域测试的劣化规律**"
        "本身就是挑战杯/大创的研究叙事（方案书阶段二：把劣化变成研究点）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="P2-2 跨域评测矩阵即插即用封装")
    ap.add_argument("--exp", default=None, help="实验目录，默认取最新")
    ap.add_argument("--ckpt", default=None, help="指定权重，优先于自动挑选")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None, help="每个域只测前 N 条（冒烟）")
    ap.add_argument("--list-domains", action="store_true", help="只报告各域可用性")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                                          "cross_domain_matrix.md"))
    args = ap.parse_args()

    roots = _domain_roots()
    avail = {name: _domain_available(name, root) for name, root in roots.items()}

    if args.list_domains:
        print("各域可用性：")
        for name, (ok, note) in avail.items():
            print(f"  {'✅' if ok else '❌'} {name:16s} {note}")
        return

    # 权重可用性
    has_weights = False
    exp_dir = ckpt = None
    try:
        exp_dir = args.exp or abe._latest_exp()
        ckpt = abe.resolve_ckpt(exp_dir, args.ckpt)
        has_weights = Path(ckpt).is_file()
    except Exception as e:
        print(f"[权重] 不可用：{e}")

    computed: dict[str, dict] = {}
    if has_weights:
        device = args.device
        if device == "cuda":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        split_for = {"2019LA-eval": "eval", "FMFCC-A": "dev", "CFAD": "dev"}
        for name, (ok, _) in avail.items():
            if not ok or name not in split_for:
                continue
            try:
                r = compute_domain_eer(roots[name], split_for[name],
                                       exp_dir, ckpt, device, args.limit)
                computed[name] = r
                print(f"  [{name}] EER={r['eer_pct']:.3f}%  "
                      f"(bon={r['n_bonafide']} spo={r['n_spoof']})")
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] 计算失败：{e}")
    else:
        print("[跳过推理] 无真实权重（external/aasist/exp_result/ 缺 best.pth），"
              "仅生成骨架矩阵 + 已知格。")

    md = build_matrix(avail, computed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\n跨域矩阵 -> {out}")


if __name__ == "__main__":
    main()
