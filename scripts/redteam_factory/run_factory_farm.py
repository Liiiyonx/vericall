#!/usr/bin/env python
"""run_factory_farm.py — 红队工厂分片驱动（2026-09-05 产能改造）

把 run_factory 的网格 round-robin 切成 N 片并行子进程（默认 8），
每片独立 meta 文件，全部完成后合并为 meta.csv。
配套：
  - 纯网络/CPU 引擎（edgetts / sovits_api 服务端自管）可满并发；
  - sovits_api 依赖已启动的 api_v2 服务（GPU 常驻，见 run_factory.py 配置 note）；
  - GPU 纪律：sovits_api 与其他 GPU 引擎一次只启一个（-e 限定），
    edgetts/信道退化等 CPU 活可与 GPU 引擎并发。

用法：
  python run_factory_farm.py --workers 8 [-e edgetts] [--per-cell 200] [--channels clean]
  python run_factory_farm.py --workers 8 -e sovits_api --per-cell 200 --channels clean
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "redteam_factory" / "run_factory.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("-e", "--engine", help="传给 run_factory 的引擎")
    ap.add_argument("--dialect", help="只跑指定方言")
    ap.add_argument("--per-cell", type=int, default=0)
    ap.add_argument("--channels", help="信道列表")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n = max(1, args.workers)
    base = [sys.executable, str(RUNNER)]
    if args.engine:
        base += ["--engine", args.engine]
    if args.dialect:
        base += ["--dialect", args.dialect]
    if args.per_cell:
        base += ["--per-cell", str(args.per_cell)]
    if args.channels:
        base += ["--channels", args.channels]

    out_root = ROOT / "data" / "redteam" / "factory"
    out_root.mkdir(parents=True, exist_ok=True)
    meta_shards = []
    if args.dry_run:
        # 单次 dry-run 看总计划（分片不拆分显示）
        subprocess.run(base + ["--dry-run"], check=False)
        return

    procs = []
    t0 = time.time()
    # 防再犯：历史分片若残留（上次 merge 遗漏），会被 append 式合并重复入库。
    # 启动前先清理旧分片，保证合并段只含本次新产行。
    for old in out_root.glob("meta_shard_*.csv"):
        try:
            old.unlink()
        except OSError as e:
            print(f"[warn] 旧分片删除失败 {old.name}: {e}")
    for i in range(n):
        meta_shard = out_root / f"meta_shard_{i}.csv"
        meta_shards.append(meta_shard)
        cmd = base + ["--shard-idx", str(i), "--shard-total", str(n),
                      "--meta-file", str(meta_shard)]
        log = ROOT / "reports" / f"factory_shard_{i}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as lf:
            procs.append((i, subprocess.Popen(cmd, cwd=ROOT,
                                              stdout=lf, stderr=subprocess.STDOUT)))

    failed = 0
    for i, p in procs:
        rc = p.wait()
        print(f"分片 {i}: exit={rc} (见 reports/factory_shard_{i}.log)")
        if rc != 0:
            failed += 1

    # 合并 meta（按分片序号，去表头）
    final = out_root / "meta.csv"
    wrote_header = not final.exists()
    with open(final, "a", encoding="utf-8", newline="") as mf:
        # 防重复：合并前加载 final 已有行集合，只补缺失行
        existing = set()
        if final.exists():
            for ln in final.read_text(encoding="utf-8-sig").splitlines():
                if ln and not ln.startswith("path,"):
                    existing.add(ln)
        for shard in meta_shards:
            if not shard.exists():
                continue
            lines = shard.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            if lines[0].startswith("path,"):  # 表头
                lines = lines[1:]
            for ln in lines:
                if ln and ln not in existing:
                    mf.write(ln + "\n")
                    existing.add(ln)
            mf.flush()
    print(f"\n全部完成：{n} 片，失败分片 {failed}，耗时 {time.time()-t0:.0f}s")
    print(f"meta -> {final}（分片文件可删：meta_shard_*.csv）")


if __name__ == "__main__":
    main()
