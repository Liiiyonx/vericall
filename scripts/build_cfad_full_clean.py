#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 CFAD clean test 全量（62,999）从 parquet 物化为 AASIST 风格布局。

背景：本地 asvspoof_layout 只有 2000 条子样（协议也只 2000 行），而论文需要
clean 全量以收窄 CI（消掉「62999 为什么只测 2000」这一审稿质疑）。

产出（默认 D:/VeriCall_data/CFAD_full）：
    asvspoof_layout/ASVspoof2019_LA_dev/flac/<utt_id>.flac
    asvspoof_layout/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
协议列约定与现有 2000 版一致：
    <source> <utt_id> - <system_id> <bonafide|spoof>
真音 source=corpus 名, system='-'; 伪音 source=fake_clean 目录名, system=其大写。

用法：<vericall python> scripts/build_cfad_full_clean.py [--out D:/...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

SRC = Path("D:/VeriCall_data/CFAD/parquet/data")
SUB_SET_PROTO = Path("D:/VeriCall_data/CFAD/asvspoof_layout/"
                     "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")


def load_subset_convention() -> dict:
    """从已有 2000 版协议里学 utt_id -> (col0, system_id) 的取值约定。"""
    m = {}
    if SUB_SET_PROTO.is_file():
        for ln in SUB_SET_PROTO.read_text(encoding="utf-8").splitlines():
            p = ln.strip().split(" ")
            if len(p) >= 5:
                m[p[1]] = (p[0], p[3])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/VeriCall_data/CFAD_full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shards", default=None, help="逗号分隔的 shard 序号，默认全部")
    args = ap.parse_args()

    out = Path(args.out)
    flac_dir = out / "asvspoof_layout" / "ASVspoof2019_LA_dev" / "flac"
    proto_dir = out / "asvspoof_layout" / "ASVspoof2019_LA_cm_protocols"
    flac_dir.mkdir(parents=True, exist_ok=True)
    proto_dir.mkdir(parents=True, exist_ok=True)

    conv = load_subset_convention()
    print(f"已加载 2000 版约定 {len(conv)} 条（用于逐字节对齐 name 列）")

    shards = sorted(SRC.glob("test-*-of-00014.parquet"))
    if args.shards:
        want = {int(x) for x in args.shards.split(",")}
        shards = [s for s in shards if int(s.stem.split("-")[1]) in want]
    print(f"shards: {[s.name for s in shards]}")

    proto_lines = []
    n = 0
    for si, sh in enumerate(shards, 1):
        t = pq.ParquetFile(sh).read(columns=["path", "audio", "label", "notes"])
        paths = t["path"].to_pylist()
        audios = t["audio"].to_pylist()
        labels = t["label"].to_pylist()
        notes = t["notes"].to_pylist()
        for p, au, lb, nt in zip(paths, audios, labels, notes):
            uid = json.loads(nt).get("utterance_id")
            seg = p.split("/")
            family = seg[2] if len(seg) > 2 else "unknown"
            data = au.get("bytes")
            if data is None:
                continue
            (flac_dir / f"{uid}.flac").write_bytes(data)
            if uid in conv:
                c0, sysid = conv[uid]
            else:
                c0 = family.upper() if int(lb) == 1 else family
                sysid = c0 if int(lb) == 1 else "-"
            lab = "spoof" if int(lb) == 1 else "bonafide"
            proto_lines.append(f"{c0} {uid} - {sysid} {lab}")
            n += 1
            if args.limit and n >= args.limit:
                break
        print(f"  [{si}/{len(shards)}] {sh.name} -> 累计 {n}", flush=True)
        if args.limit and n >= args.limit:
            break

    (proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt").write_text(
        "\n".join(proto_lines) + "\n", encoding="utf-8")
    print(f"完成：{n} 条 -> {out}")


if __name__ == "__main__":
    main()
