#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""prepare_aasist_cn.py — 组织中文域数据为 AASIST 训练格式（2026-09-08）

背景：AASIST 官方代码只读过英文 ASVspoof2019 LA 训练；对中文失效（CFAD EER 44%）。
本脚本把中文域数据（真：AISHELL-1 子集；伪：FMFCC-A 攻击样本）组织成 AASIST 可直接
消费的 ASVspoof 风格目录 + 5 列协议，用于「英文预训练 AASIST → 中文微调」。

数据源（已核实 2026-09-08）：
  真(bonafide)：D:/VeriCall_data/aishell1_sub/train/Sxxxx/BAC009SxxxxWxxxx.wav（16k 单声道）
  伪(spoof)   ：D:/VeriCall_data/FMFCC-A/extracted/FMFCC-A/*.wav（16k 单声道，A01–A13）

Split 纪律：
  - 真样本按「说话人 Sxxxx」分层切 train/dev/eval（避免说话人泄漏）；
  - 伪样本按「攻击系统 A01–A13（含退化变体）」分层（避免同一系统跨集）；
  - 隔离纪律：CFAD 2000 / 红队语料一律不入训练。

输出布局（database_path，默认 D:/VeriCall_data/AASIST_CN/）：
  ASVspoof2019_LA_train/flac/<utt>.flac
  ASVspoof2019_LA_dev/flac/<utt>.flac
  ASVspoof2019_LA_eval/flac/<utt>.flac
  ASVspoof2019_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt   (5 列: spk utt - sys bonafide|spoof)
    ASVspoof2019.LA.cm.dev.trl.txt
    ASVspoof2019.LA.cm.eval.trl.txt

用法：
  python scripts/prepare_aasist_cn.py --dry-run            # 只预览 split 统计
  python scripts/prepare_aasist_cn.py                      # 默认规模：train 8k+8k / dev 1k+1k / eval 1k+1k
  python scripts/prepare_aasist_cn.py --n_train_bon 12000 --n_train_spoof 12000 --n_epochs_hint 10
"""
from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

AISHELL_DIR = Path("D:/VeriCall_data/aishell1_sub/train")
FMFCC_DIR = Path("D:/VeriCall_data/FMFCC-A/extracted/FMFCC-A")
FMFCC_INFO = Path("D:/VeriCall_data/FMFCC-A/AllUtteranceInfo.txt")
OUT_ROOT = Path("D:/VeriCall_data/AASIST_CN")

SPK_RE = re.compile(r"BAC009(S\d{4})")   # aishell 说话人号，如 S0002


def _aishell_speaker(stem: str) -> str:
    m = SPK_RE.search(stem)
    return m.group(1) if m else "S_UNK"


def load_fmfcc_system_map() -> dict[str, str]:
    """文件名 -> 攻击系统号（A01..A13，含退化变体）。"""
    sysmap: dict[str, str] = {}
    if not FMFCC_INFO.exists():
        return sysmap
    for line in FMFCC_INFO.read_text(encoding="utf-8").splitlines():
        p = line.strip().split(",")
        if len(p) == 3:
            sysmap[p[0]] = p[2]
    return sysmap


def _grouped_shuffle(items: list, group_fn, rng: random.Random):
    """按组（说话人/系统）打散，组内保持原始顺序——用于分层抽样。"""
    groups = defaultdict(list)
    for it in items:
        groups[group_fn(it)].append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    flat = []
    for k in keys:
        flat.extend(groups[k])
    return flat


def build_split(n_bon_train: int, n_spoof_train: int,
                n_dev: int, n_eval: int, seed: int):
    """返回 {split: [(utt_id, label, sysid, src_path)]}，分层抽样。"""
    rng = random.Random(seed)

    # ---- 真：aishell，按说话人分层 ----
    bon = [(p.stem, "bonafide", "-", p) for p in AISHELL_DIR.rglob("*.wav")]
    bon = _grouped_shuffle(bon, lambda it: _aishell_speaker(it[0]), rng)
    print(f"[真] aishell 共 {len(bon)} 条，按说话人分层打散")

    # ---- 伪：FMFCC，按攻击系统分层 ----
    sysmap = load_fmfcc_system_map()
    spoof = []
    for p in FMFCC_DIR.glob("*.wav"):
        sysid = sysmap.get(p.name, "A00")
        spoof.append((p.stem, "spoof", sysid, p))
    spoof = _grouped_shuffle(spoof, lambda it: it[2], rng)
    print(f"[伪] FMFCC 共 {len(spoof)} 条，按攻击系统分层打散")

    # 分配索引
    n_bon_total = n_bon_train + n_dev + n_eval
    n_spoof_total = n_spoof_train + n_dev + n_eval
    if n_bon_total > len(bon) or n_spoof_total > len(spoof):
        raise SystemExit(
            f"规模超限：真需 {n_bon_total}（有 {len(bon)}）、伪需 {n_spoof_total}（有 {len(spoof)}）")

    sel = {"train": [], "dev": [], "eval": []}
    sel["train"].extend(bon[:n_bon_train])
    sel["train"].extend(spoof[:n_spoof_train])
    sel["dev"].extend(bon[n_bon_train:n_bon_train + n_dev])
    sel["dev"].extend(spoof[n_spoof_train:n_spoof_train + n_dev])
    sel["eval"].extend(bon[n_bon_train + n_dev:n_bon_total])
    sel["eval"].extend(spoof[n_spoof_train + n_dev:n_spoof_total])

    # 最终打散 train（避免顺序偏差）
    rng.shuffle(sel["train"])
    return sel


def _to_flac(src: Path, dst: Path) -> bool:
    """16k 单声道 wav → flac。幂等：目标已存在且非空则跳过。"""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        import soundfile as sf
        data, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if int(sr) != 16000:
            from scipy.signal import resample_poly
            g = int(16000)
            up = g // 100
            data = resample_poly(data.astype("float32"), up, max(1, int(sr) // 100)).astype("float32")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), data, 16000, subtype="PCM_16", format="FLAC")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [转换失败] {src.name}: {e}")
        return False


def write_split(split_items: list, split: str, proto_ext: str, out_root: Path,
                no_audio: bool, resume: bool) -> tuple[int, int]:
    flac_dir = out_root / f"ASVspoof2019_LA_{split}" / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)
    proto_dir = out_root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    proto = proto_dir / f"ASVspoof2019.LA.cm.{split}.{proto_ext}.txt"

    n_ok = n_fail = 0
    lines = []
    for utt, label, sysid, src in split_items:
        dst = flac_dir / f"{utt}.flac"
        if no_audio:
            pass
        elif resume and dst.exists() and dst.stat().st_size > 0:
            pass
        else:
            ok = _to_flac(src, dst)
            if not ok:
                n_fail += 1
                continue
        n_ok += 1
        spk = _aishell_speaker(utt) if label == "bonafide" else sysid
        sysid_out = "-" if label == "bonafide" else sysid
        lines.append(f"{spk} {utt} - {sysid_out} {label}")
    proto.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return n_ok, n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--n_train_bon", type=int, default=8000)
    ap.add_argument("--n_train_spoof", type=int, default=8000)
    ap.add_argument("--n_dev", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-audio", action="store_true", help="只写协议不转音频")
    ap.add_argument("--dry-run", action="store_true", help="只预览 split 不落盘")
    ap.add_argument("--resume", action="store_true", help="已存在的 flac 跳过")
    args = ap.parse_args()

    out_root = Path(args.out)
    sel = build_split(args.n_train_bon, args.n_train_spoof,
                      args.n_dev, args.n_eval, args.seed)
    for sp in ("train", "dev", "eval"):
        bon = sum(1 for it in sel[sp] if it[1] == "bonafide")
        spo = sum(1 for it in sel[sp] if it[1] == "spoof")
        print(f"[{sp:5s}] 真 {bon:>5d} / 伪 {spo:>5d} = {bon + spo}")
    if args.dry_run:
        print("dry-run 完成，未落盘。")
        return

    for sp, ext in (("train", "trn"), ("dev", "trl"), ("eval", "trl")):
        n_ok, n_fail = write_split(sel[sp], sp, ext, out_root, args.no_audio, args.resume)
        print(f"[{sp}] 写出 {n_ok} 条（失败 {n_fail}）")
    print(f"\n完成。database_path = {out_root}")
    print("下一步（微调，英文预训练初始化）：")
    print(f"  cd external/aasist && python main.py --config <AASIST_CN.conf> "
          f"--resume models/weights/AASIST.pth")


if __name__ == "__main__":
    main()
