#!/usr/bin/env python
"""merge_factory_meta.py — 红队工厂 meta 分片收口合并（2026-09-05 跟进会话新增）

8 个 meta_shard_*.csv 经多次断点续跑 append，可能含重复 path；
本脚本与既有 meta.csv 一起按 path 去重合并，输出干净 meta.csv。

用法：python scripts/redteam_factory/merge_factory_meta.py
产出：data/redteam/factory/meta.csv（覆盖）
      data/redteam/factory/meta.csv.bak（合并前备份）
"""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "redteam" / "factory" / "meta.csv"
SHARD_DIR = OUT.parent
META_FIELDS = ["path", "label", "attack_id", "channel", "source", "license",
               "engine", "dialect", "script_id", "speaker_ref", "duration_s"]


def read_rows(p: Path):
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {p.name}: {type(e).__name__}")
        return []


def main():
    sources = [OUT] + sorted(SHARD_DIR.glob("meta_shard_*.csv"))
    rows = []
    for p in sources:
        got = read_rows(p)
        print(f"{p.name}: {len(got)} 行")
        rows += got
    print(f"\n合并前总行数: {len(rows)}")

    # 按 path 去重（保留首次出现）
    seen, uniq = set(), []
    for r in rows:
        k = r.get("path", "")
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f"去重后: {len(uniq)} 行（去重 {len(rows) - len(uniq)}）")

    # 检查物理文件存在性
    missing = [r for r in uniq if not (ROOT / r["path"]).exists()]
    print(f"path 指向文件缺失: {len(missing)}（记录来自已删隔离/超长母本? 保留在 meta 由 quality_gate 处理）")

    # 统计
    cc = Counter(r["channel"] for r in uniq)
    ec = Counter(r["engine"] for r in uniq)
    dc = Counter(r["dialect"] for r in uniq)
    print("\n信道分布:", dict(cc))
    print("引擎分布:", dict(ec))
    print("方言分布:", dict(dc))

    # 备份并写出
    if OUT.exists():
        bak = OUT.with_suffix(".csv.bak")
        shutil.copy2(OUT, bak)
        print(f"\n备份: {bak.name}")
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for r in uniq:
            w.writerow({k: r.get(k, "") for k in META_FIELDS})
    print(f"写入: {OUT}（{len(uniq)} 行）")


if __name__ == "__main__":
    main()
