# -*- coding: utf-8 -*-
"""
构造场景集（任务书 P1-5 第 1 步）
================================
从 ASVspoof2019 LA dev 协议抽样 N=20 位「家人」说话人 ×（2 条 bonafide + 2 条 spoof）
+ 20 位陌生人真声 = 100 条带期望标签的测试用例：
    家人真声   = allow
    家人克隆   = block
    陌生人真声 = block|caution
输出 evaluation/scenario_manifest.csv（供跑分脚本消费）。协议/数据缺失时生成带说明的骨架。
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import DEV_FLAC, ASVSPOOF_LA

OUT = ROOT / "evaluation" / "scenario_manifest.csv"
N_FAMILY = 20
N_STRANGER = 20


def _find_protocol() -> Path | None:
    for cand in (ASVSPOOF_LA.parent if ASVSPOOF_LA.is_absolute() else ASVSPOOF_LA).rglob(
            "ASVspoof2019_LA_dev_protocol*.txt"):
        return cand
    return None


def _parse_protocol(proto: Path) -> dict[str, str]:
    """返回 {filename(无扩展): 'bonafide'|'spoof'}。"""
    out: dict[str, str] = {}
    for line in proto.read_text(encoding="utf-8", errors="ignore").splitlines():
        cols = line.split()
        if len(cols) < 5:
            continue
        fname, label = cols[1], cols[4]
        out[fname] = label  # label: bonafide / spoof
    return out


def main():
    random.seed(42)
    rows: list[dict] = []
    proto = _find_protocol()
    labels = _parse_protocol(proto) if proto else {}

    if DEV_FLAC.is_dir():
        flacs = sorted(DEV_FLAC.glob("LA_D_*.flac"))
        # 按说话人归组
        by_spk: dict[str, list[Path]] = {}
        for f in flacs:
            spk = f.stem.split("_")[2][:4] if len(f.stem.split("_")) > 2 else "x"
            by_spk.setdefault(spk, []).append(f)

        family_spks = random.sample(list(by_spk), min(N_FAMILY, len(by_spk)))
        for spk in family_spks:
            files = by_spk[spk]
            bona = [f for f in files if labels.get(f.stem, "bonafide") == "bonafide"]
            spoof = [f for f in files if labels.get(f.stem, "spoof") == "spoof"]
            for f in (bona[:2] + spoof[:2]):
                if not f:
                    continue
                kind = "spoof" if labels.get(f.stem) == "spoof" else "bonafide"
                rows.append({"id": f.stem, "speaker": spk, "kind": kind,
                             "expected": "block" if kind == "spoof" else "allow",
                             "audio": str(f)})
        # 陌生人真声
        stranger_spks = [s for s in by_spk if s not in family_spks]
        for spk in random.sample(stranger_spks, min(N_STRANGER, len(stranger_spks))):
            bona = [f for f in by_spk[spk] if labels.get(f.stem, "bonafide") == "bonafide"]
            for f in bona[:1]:
                rows.append({"id": f.stem, "speaker": spk, "kind": "stranger",
                             "expected": "block|caution", "audio": str(f)})
    else:
        print(f"[warn] DEV_FLAC 不存在（{DEV_FLAC}），生成骨架 manifest（需补数据后重跑）")
        for i in range(N_FAMILY):
            rows.append({"id": f"FAM{i:03d}_bona", "speaker": f"FAM{i:03d}",
                         "kind": "bonafide", "expected": "allow", "audio": "TODO"})
            rows.append({"id": f"FAM{i:03d}_spoof", "speaker": f"FAM{i:03d}",
                         "kind": "spoof", "expected": "block", "audio": "TODO"})
        for i in range(N_STRANGER):
            rows.append({"id": f"STR{i:03d}_bona", "speaker": f"STR{i:03d}",
                         "kind": "stranger", "expected": "block|caution", "audio": "TODO"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "speaker", "kind", "expected", "audio"])
        w.writeheader()
        w.writerows(rows)
    print(f"场景集: {len(rows)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
