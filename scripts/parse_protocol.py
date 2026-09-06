# -*- coding: utf-8 -*-
"""
解析 ASVspoof 2019 LA 协议文件，生成统一的数据索引 CSV。

索引列:
    utt_id, subset(train/dev/eval), label(bonafide/spoof), attack_id, filepath

用法:
    python parse_protocol.py --root ../data/raw/ASVspoof2019_LA
"""
import argparse
import csv
from pathlib import Path

# 协议文件固定命名（官方发布格式）
PROTO_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ASVspoof2019_LA 目录")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    args = ap.parse_args()

    root = Path(args.root)
    flac_dir = root / "ASVspoof2019_LA_flac"
    proto_dir = None
    for d in root.rglob("ASVspoof2019_LA_cm_protocols"):
        proto_dir = d
        break
    if proto_dir is None:
        raise SystemExit("未找到协议目录 ASVspoof2019_LA_cm_protocols，请检查 --root")

    rows = []
    for subset, fname in PROTO_FILES.items():
        fpath = next(proto_dir.rglob(fname), None)
        if fpath is None:
            raise SystemExit(f"未找到协议文件 {fname}")
        for line in fpath.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            # 格式: speaker utt_id attack_id label  （LA 协议 5 列）
            _, utt_id, attack_id, label = parts[1], parts[1], parts[2], parts[-1]
            wav = flac_dir / f"{utt_id}.flac"
            rows.append({
                "utt_id": utt_id,
                "subset": subset,
                "label": "bonafide" if label == "bonafide" else "spoof",
                "attack_id": attack_id,
                "filepath": str(wav),
            })

    out = Path(args.out) if args.out else root / "index.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "subset", "label", "attack_id", "filepath"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter((r["subset"], r["label"]) for r in rows)
    print(f"写出 {len(rows)} 条 -> {out}")
    for (subset, label), n in sorted(c.items()):
        print(f"  {subset:6s} {label:9s} {n:7d}")
    # 官方参考值: train 2548真/22296假, dev 2484真/24444假, eval 7355真/63882假


if __name__ == "__main__":
    main()
